from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    face_bgr: np.ndarray


class FaceDetector:
    """OpenCV face detector supporting Haar cascade or DNN models."""

    def __init__(
        self,
        method: str = "haar",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (40, 40),
        confidence_threshold: float = 0.6,
        dnn_prototxt: Optional[str] = None,
        dnn_weights: Optional[str] = None,
        crop_margin: float = 0.15,
    ) -> None:
        self.method = method.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        self.crop_margin = crop_margin

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        self.dnn_net = None
        if self.method == "dnn":
            if not dnn_prototxt or not dnn_weights:
                raise ValueError(
                    "DNN mode requires both dnn_prototxt and dnn_weights paths."
                )
            self.dnn_net = cv2.dnn.readNetFromCaffe(dnn_prototxt, dnn_weights)

        if self.method not in {"haar", "dnn"}:
            raise ValueError("method must be 'haar' or 'dnn'.")

    def detect_faces(
        self, image_bgr: np.ndarray, require_alignment: bool = True
    ) -> List[FaceDetection]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        if self.method == "haar":
            return self._detect_haar(image_bgr, require_alignment=require_alignment)
        return self._detect_dnn(image_bgr, require_alignment=require_alignment)

    def _detect_haar(
        self, image_bgr: np.ndarray, require_alignment: bool = True
    ) -> List[FaceDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
        )

        detections: List[FaceDetection] = []
        for (x, y, w, h) in faces:
            x1, y1, x2, y2 = self._expand_box(
                x, y, x + w, y + h, image_bgr.shape[1], image_bgr.shape[0]
            )
            face_crop = image_bgr[y1:y2, x1:x2].copy()
            aligned_face = self._align_face(face_crop) if require_alignment else face_crop
            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=1.0,
                    face_bgr=aligned_face,
                )
            )
        return detections

    def _detect_dnn(
        self, image_bgr: np.ndarray, require_alignment: bool = True
    ) -> List[FaceDetection]:
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        self.dnn_net.setInput(blob)
        detections_raw = self.dnn_net.forward()

        detections: List[FaceDetection] = []
        for i in range(detections_raw.shape[2]):
            confidence = float(detections_raw[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1, x2, y2 = self._expand_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = image_bgr[y1:y2, x1:x2].copy()
            aligned_face = self._align_face(face_crop) if require_alignment else face_crop
            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    face_bgr=aligned_face,
                )
            )
        return detections

    def _align_face(self, face_bgr: np.ndarray) -> np.ndarray:
        """Rotate face crop to reduce tilt using eye locations when available."""
        if face_bgr is None or face_bgr.size == 0:
            return face_bgr

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(12, 12),
        )

        if len(eyes) < 2:
            return face_bgr

        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        eye_centers = []
        for (ex, ey, ew, eh) in eyes_sorted:
            eye_centers.append((ex + ew // 2, ey + eh // 2))

        eye_centers = sorted(eye_centers, key=lambda p: p[0])
        left_eye, right_eye = eye_centers[0], eye_centers[1]
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        if dx == 0:
            return face_bgr

        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            face_bgr,
            rot_mat,
            (face_bgr.shape[1], face_bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def _expand_box(
        self, x1: int, y1: int, x2: int, y2: int, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        box_w = x2 - x1
        box_h = y2 - y1
        margin_x = int(box_w * self.crop_margin)
        margin_y = int(box_h * self.crop_margin)

        x1e = max(0, x1 - margin_x)
        y1e = max(0, y1 - margin_y)
        x2e = min(width, x2 + margin_x)
        y2e = min(height, y2 + margin_y)
        return x1e, y1e, x2e, y2e
