from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def normalize_lighting(image_rgb: np.ndarray) -> np.ndarray:
    """Apply CLAHE on luminance channel to reduce lighting inconsistencies."""
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_equalized = clahe.apply(y_channel)
    merged = cv2.merge([y_equalized, cr_channel, cb_channel])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def resize_face(image_rgb: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    return cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)


def scale_pixels(image_rgb: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [-1, 1]."""
    image = image_rgb.astype(np.float32)
    return (image / 127.5) - 1.0


def preprocess_face(
    face_bgr: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
    apply_lighting_norm: bool = True,
) -> np.ndarray:
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("face_bgr must be a non-empty image.")

    face_rgb = bgr_to_rgb(face_bgr)
    if apply_lighting_norm:
        face_rgb = normalize_lighting(face_rgb)

    resized = resize_face(face_rgb, target_size=target_size)
    scaled = scale_pixels(resized)
    return scaled
