from __future__ import annotations

import argparse
import base64
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request

from ai_core.detection import FaceDetector
from ai_core.embedding import FaceEmbedder
from ai_core.model_loader import load_facenet_model


def _processing_time_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _error_response(
    start_time: float,
    error_code: str,
    message: str,
    status_code: int,
):
    return (
        jsonify(
            {
                "status": "error",
                "error_code": error_code,
                "message": message,
                "processing_time_ms": _processing_time_ms(start_time),
            }
        ),
        status_code,
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


def create_app(
    model_path: str,
    detector_method: str = "haar",
    dnn_prototxt: str | None = None,
    dnn_weights: str | None = None,
    dnn_confidence: float = 0.6,
) -> Flask:
    app = Flask(__name__)

    detector = FaceDetector(
        method=detector_method,
        dnn_prototxt=dnn_prototxt,
        dnn_weights=dnn_weights,
        confidence_threshold=dnn_confidence,
    )
    model = load_facenet_model(model_path)
    embedder = FaceEmbedder(model)

    @app.get("/health")
    def health() -> Tuple[Any, int]:
        return jsonify({"status": "ok", "service": "ai-face-extractor"}), 200

    @app.post("/api/v1/extract-faces")
    def extract_faces() -> Tuple[Any, int]:
        started = time.perf_counter()

        payload: Dict[str, Any] | None = request.get_json(silent=True)
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
                message="Field 'image_base64' is required.",
                status_code=400,
            )

        camera_id = str(payload.get("camera_id", ""))
        frame_timestamp = payload.get("timestamp")
        require_alignment = bool(payload.get("require_alignment", True))

        try:
            frame = _decode_base64_image(image_base64)
        except ValueError as err:
            return _error_response(
                started,
                error_code="IMAGE_UNREADABLE",
                message=str(err),
                status_code=400,
            )

        try:
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

            app.logger.debug(
                "extract-faces camera_id=%s timestamp=%s faces=%s",
                camera_id,
                frame_timestamp,
                len(result_data),
            )

            return (
                jsonify(
                    {
                        "status": "success",
                        "faces_count": len(result_data),
                        "processing_time_ms": _processing_time_ms(started),
                        "data": result_data,
                    }
                ),
                200,
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
    parser = argparse.ArgumentParser(description="AI face extraction API service.")
    parser.add_argument("--model-path", required=True, help="Path to FaceNet .h5 or SavedModel")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--detector", default="haar", choices=["haar", "dnn"])
    parser.add_argument("--dnn-prototxt", default=None)
    parser.add_argument("--dnn-weights", default=None)
    parser.add_argument("--dnn-conf", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = create_app(
        model_path=args.model_path,
        detector_method=args.detector,
        dnn_prototxt=args.dnn_prototxt,
        dnn_weights=args.dnn_weights,
        dnn_confidence=args.dnn_conf,
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
