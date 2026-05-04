from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ai_core.detection import FaceDetector
from ai_core.embedding import FaceEmbedder
from ai_core.model_loader import load_facenet_model


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Uploaded payload is not a valid image.")
    return image_bgr


def _extract_faces(
    detector: FaceDetector,
    embedder: FaceEmbedder,
    frame: np.ndarray,
) -> list[dict[str, object]]:
    detections = detector.detect_faces(frame, require_alignment=True)
    faces: list[dict[str, object]] = []

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        embedding = embedder.embed_face(detection.face_bgr).astype(np.float32).tolist()

        faces.append(
            {
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "embedding": embedding,
            }
        )

    return faces


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.detector = FaceDetector()
        app.state.embedder = FaceEmbedder(load_facenet_model())
        yield

    app = FastAPI(title="AI Face Embedding Service", lifespan=lifespan)

    @app.post("/api/v1/extract-faces")
    async def extract_faces(
        request: Request,
        image: UploadFile = File(...),
    ) -> dict[str, list[dict[str, object]]]:
        detector: FaceDetector = request.app.state.detector
        embedder: FaceEmbedder = request.app.state.embedder

        try:
            image_bytes = await image.read()
            frame = await run_in_threadpool(_decode_image, image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            faces = await run_in_threadpool(_extract_faces, detector, embedder, frame)
            return {"faces": faces}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to extract faces: {exc}") from exc

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI face extraction FastAPI service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run(
        "api_service.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


app = create_app()


if __name__ == "__main__":
    main()
