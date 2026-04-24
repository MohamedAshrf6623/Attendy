from __future__ import annotations

from typing import Iterable, List

import numpy as np
import tensorflow as tf

from .preprocessing import preprocess_face


class FaceEmbedder:
    """Generate L2-normalized face embeddings using a FaceNet model."""

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    def embed_face(self, face_bgr: np.ndarray) -> np.ndarray:
        preprocessed = preprocess_face(face_bgr)
        return self.embed_preprocessed(preprocessed)

    def embed_preprocessed(self, preprocessed_face: np.ndarray) -> np.ndarray:
        if preprocessed_face.ndim != 3:
            raise ValueError("preprocessed_face must have shape (160, 160, 3).")

        batch = np.expand_dims(preprocessed_face, axis=0)
        embedding = self.model.predict(batch, verbose=0)[0]
        return self._l2_normalize(embedding)

    def embed_faces(self, faces_bgr: Iterable[np.ndarray]) -> List[np.ndarray]:
        embeddings = []
        for face in faces_bgr:
            embeddings.append(self.embed_face(face))
        return embeddings

    @staticmethod
    def _l2_normalize(vector: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / (norm + epsilon)
