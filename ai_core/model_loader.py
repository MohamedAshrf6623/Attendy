from __future__ import annotations

import tensorflow as tf


def load_facenet_model() -> tf.keras.Model:
    from keras_facenet import FaceNet

    facenet = FaceNet()
    model = facenet.model

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