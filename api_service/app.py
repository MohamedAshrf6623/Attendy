from __future__ import annotations

import argparse
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import anyio
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ai_core.detection import FaceDetector
from ai_core.embedding import FaceEmbedder
from ai_core.model_loader import load_facenet_model


LOGGER = logging.getLogger("api_service")


def _processing_time_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _error_response(
    start_time: float,
    error_code: str,
    message: str,
    status_code: int,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "error_code": error_code,
            "message": message,
            "processing_time_ms": _processing_time_ms(start_time),
        },
    )


def _decode_base64_image(image_base64: str) -> np.ndarray:
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("image_base64 must be a non-empty string.")

    if "," in image_base64 and image_base64.startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise ValueError("The provided base64 string is invalid.") from exc

    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(
            "The provided base64 string could not be decoded into a valid image matrix."
        )

    return image_bgr


def _decode_raw_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Uploaded image content is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Uploaded binary payload is not a valid image.")
    return image_bgr


def _process_frame(
    detector: FaceDetector,
    embedder: FaceEmbedder,
    frame: np.ndarray,
    require_alignment: bool,
) -> list[dict[str, Any]]:
    detections = detector.detect_faces(
        frame,
        require_alignment=require_alignment,
    )

    result_data = []
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox
        if require_alignment:
            embedding = embedder.embed_face(detection.face_bgr).astype(np.float32).tolist()
        else:
            embedding = []

        result_data.append(
            {
                "face_index": idx,
                "confidence": round(float(detection.confidence), 6),
                "bounding_box": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "embedding": embedding,
            }
        )

    return result_data


def create_app(
    model_path: str,
    detector_method: str = "haar",
    dnn_prototxt: str | None = None,
    dnn_weights: str | None = None,
    dnn_confidence: float = 0.6,
    max_inference_concurrency: int = 2,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Model and detector are loaded once per process on startup.
        if not model_path:
            raise RuntimeError(
                "MODEL_PATH is not configured. Set MODEL_PATH env var or run with --model-path."
            )

        app.state.detector = FaceDetector(
            method=detector_method,
            dnn_prototxt=dnn_prototxt,
            dnn_weights=dnn_weights,
            confidence_threshold=dnn_confidence,
        )
        model = load_facenet_model(model_path)
        app.state.embedder = FaceEmbedder(model)
        app.state.inference_semaphore = anyio.Semaphore(max_inference_concurrency)
        yield

    app = FastAPI(title="AI Face Extractor Service", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "ai-face-extractor"}

    @app.post("/api/v1/extract-faces")
    async def extract_faces(
        request: Request,
        image: UploadFile | None = File(default=None),
        camera_id_form: str | None = Form(default=None, alias="camera_id"),
        timestamp_form: str | None = Form(default=None, alias="timestamp"),
        require_alignment_form: bool | None = Form(default=None, alias="require_alignment"),
    ) -> JSONResponse:
        started = time.perf_counter()

        detector: FaceDetector = request.app.state.detector
        embedder: FaceEmbedder = request.app.state.embedder

        content_type = (request.headers.get("content-type") or "").lower()
        camera_id = ""
        frame_timestamp: Any = None
        require_alignment = True

        try:
            if "multipart/form-data" in content_type:
                if image is None:
                    return _error_response(
                        started,
                        error_code="MISSING_IMAGE",
                        message="Multipart request must include file field 'image'.",
                        status_code=400,
                    )

                image_bytes = await image.read()
                frame = await run_in_threadpool(_decode_raw_image, image_bytes)
                camera_id = camera_id_form or ""
                frame_timestamp = timestamp_form
                require_alignment = (
                    True if require_alignment_form is None else bool(require_alignment_form)
                )
            else:
                payload: Dict[str, Any] | None = await request.json()
                if payload is None:
                    return _error_response(
                        started,
                        error_code="INVALID_JSON",
                        message="Request body must be valid JSON.",
                        status_code=400,
                    )

                image_base64 = payload.get("image_base64")
                if image_base64 is None:
                    return _error_response(
                        started,
                        error_code="MISSING_IMAGE",
                        message="Field 'image_base64' is required for JSON requests.",
                        status_code=400,
                    )

                frame = await run_in_threadpool(_decode_base64_image, image_base64)
                camera_id = str(payload.get("camera_id", ""))
                frame_timestamp = payload.get("timestamp")
                require_alignment = bool(payload.get("require_alignment", True))
        except ValueError as err:
            return _error_response(
                started,
                error_code="IMAGE_UNREADABLE",
                message=str(err),
                status_code=400,
            )
        except Exception:
            return _error_response(
                started,
                error_code="INVALID_JSON",
                message="Request body must be valid JSON or multipart/form-data.",
                status_code=400,
            )

        try:
            semaphore = request.app.state.inference_semaphore
            with anyio.move_on_after(0.05) as scope:
                await semaphore.acquire()
            if scope.cancel_called:
                return _error_response(
                    started,
                    error_code="SERVICE_BUSY",
                    message="Inference queue is full, retry later.",
                    status_code=429,
                )

            try:
                result_data = await run_in_threadpool(
                    _process_frame,
                    detector,
                    embedder,
                    frame,
                    require_alignment,
                )
            finally:
                semaphore.release()

            LOGGER.debug(
                "extract-faces camera_id=%s timestamp=%s faces=%s",
                camera_id,
                frame_timestamp,
                len(result_data),
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "faces_count": len(result_data),
                    "processing_time_ms": _processing_time_ms(started),
                    "data": result_data,
                },
            )
        except Exception as err:
            return _error_response(
                started,
                error_code="INTERNAL_ERROR",
                message=f"Unexpected error while processing frame: {err}",
                status_code=500,
            )

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI face extraction FastAPI service.")
    parser.add_argument("--model-path", required=True, help="Path to FaceNet .h5 or SavedModel")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--detector", default="haar", choices=["haar", "dnn"])
    parser.add_argument("--dnn-prototxt", default=None)
    parser.add_argument("--dnn-weights", default=None)
    parser.add_argument("--dnn-conf", type=float, default=0.6)
    parser.add_argument("--max-inference-concurrency", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["DETECTOR_METHOD"] = args.detector
    os.environ["DNN_PROTOTXT"] = args.dnn_prototxt or ""
    os.environ["DNN_WEIGHTS"] = args.dnn_weights or ""
    os.environ["DNN_CONF"] = str(args.dnn_conf)
    os.environ["MAX_INFERENCE_CONCURRENCY"] = str(args.max_inference_concurrency)

    uvicorn.run(
        "api_service.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


app = create_app(
    model_path=os.getenv("MODEL_PATH", ""),
    detector_method=os.getenv("DETECTOR_METHOD", "haar"),
    dnn_prototxt=os.getenv("DNN_PROTOTXT") or None,
    dnn_weights=os.getenv("DNN_WEIGHTS") or None,
    dnn_confidence=float(os.getenv("DNN_CONF", "0.6")),
    max_inference_concurrency=int(os.getenv("MAX_INFERENCE_CONCURRENCY", "2")),
)


if __name__ == "__main__":
    main()
