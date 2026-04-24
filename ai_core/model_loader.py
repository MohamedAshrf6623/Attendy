from __future__ import annotations

import os

import tensorflow as tf


def load_facenet_model(model_path: str, compile_model: bool = False) -> tf.keras.Model:
    """Load a pre-trained FaceNet Keras model from .h5 or SavedModel path."""
    if not model_path:
        raise ValueError("model_path is required.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FaceNet model not found at: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=compile_model)
    _validate_model_input(model)
    return model


def _validate_model_input(model: tf.keras.Model) -> None:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    if not shape or len(shape) != 4:
        raise ValueError(f"Unexpected model input shape: {shape}")

    h = shape[1]
    w = shape[2]
    c = shape[3]

    if (h is not None and h != 160) or (w is not None and w != 160) or (c is not None and c != 3):
        raise ValueError(
            "FaceNet model input must be compatible with (160, 160, 3). "
            f"Got {shape}."
        )
