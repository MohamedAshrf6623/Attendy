from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

import api_service.app as app_module


class _DummyDetector:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def detect_faces(self, image_bgr: np.ndarray, require_alignment: bool = True) -> list:
        return []


class _DummyEmbedder:
    def __init__(self, model) -> None:
        self.model = model


def _build_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(app_module, "FaceDetector", _DummyDetector)
    monkeypatch.setattr(app_module, "load_facenet_model", lambda: object())
    monkeypatch.setattr(app_module, "FaceEmbedder", _DummyEmbedder)

    app = app_module.create_app()
    return TestClient(app)


def _jpeg_bytes() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_ok(monkeypatch) -> None:
    with _build_client(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_extract_faces_json_missing_image(monkeypatch) -> None:
    with _build_client(monkeypatch) as client:
        response = client.post("/api/v1/extract-faces", json={"camera_id": "cam-1"})

    body = response.json()
    assert response.status_code == 400
    assert body["error_code"] == "MISSING_IMAGE"


def test_extract_faces_json_base64_success(monkeypatch) -> None:
    image_base64 = base64.b64encode(_jpeg_bytes()).decode("ascii")
    payload = {"image_base64": image_base64, "require_alignment": True}

    with _build_client(monkeypatch) as client:
        response = client.post("/api/v1/extract-faces", json=payload)

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["faces_count"] == 0
    assert isinstance(body["processing_time_ms"], int)


def test_extract_faces_multipart_success(monkeypatch) -> None:
    files = {"image": ("frame.jpg", _jpeg_bytes(), "image/jpeg")}
    data = {"camera_id": "cam-2", "require_alignment": "true"}

    with _build_client(monkeypatch) as client:
        response = client.post("/api/v1/extract-faces", files=files, data=data)

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["faces_count"] == 0
