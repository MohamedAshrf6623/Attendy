from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RecognitionResult:
    identity: str
    score: float
    distance: float
    is_match: bool


class EmbeddingStore:
    """Local embedding storage using a dictionary persisted as a .pkl file."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.embeddings: Dict[str, List[np.ndarray]] = {}

    def load(self) -> None:
        if not os.path.exists(self.db_path):
            self.embeddings = {}
            return

        with open(self.db_path, "rb") as file:
            raw_data = pickle.load(file)

        self.embeddings = {
            name: [np.asarray(vec, dtype=np.float32) for vec in vectors]
            for name, vectors in raw_data.items()
        }

    def save(self) -> None:
        serializable = {
            name: [vec.astype(np.float32) for vec in vectors]
            for name, vectors in self.embeddings.items()
        }
        with open(self.db_path, "wb") as file:
            pickle.dump(serializable, file)

    def add_identity(self, name: str, vectors: List[np.ndarray]) -> None:
        if name not in self.embeddings:
            self.embeddings[name] = []
        self.embeddings[name].extend([np.asarray(v, dtype=np.float32) for v in vectors])

    def identities(self) -> List[str]:
        return list(self.embeddings.keys())


class FaceRecognizer:
    """Recognize identities by embedding similarity."""

    def __init__(
        self,
        store: EmbeddingStore,
        metric: str = "cosine",
        threshold: float = 0.6,
    ) -> None:
        metric = metric.lower()
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be either 'cosine' or 'euclidean'.")

        self.store = store
        self.metric = metric
        self.threshold = threshold

    def recognize(self, query_embedding: np.ndarray) -> RecognitionResult:
        if not self.store.embeddings:
            return RecognitionResult("Unknown", score=0.0, distance=float("inf"), is_match=False)

        best_identity = "Unknown"
        best_score = -1.0
        best_distance = float("inf")

        for identity, vectors in self.store.embeddings.items():
            for reference in vectors:
                score, distance = self._compare(query_embedding, reference)
                if self.metric == "cosine":
                    if score > best_score:
                        best_score = score
                        best_distance = distance
                        best_identity = identity
                else:
                    if distance < best_distance:
                        best_score = score
                        best_distance = distance
                        best_identity = identity

        is_match = self._is_match(best_score, best_distance)
        if not is_match:
            best_identity = "Unknown"

        return RecognitionResult(
            identity=best_identity,
            score=float(best_score),
            distance=float(best_distance),
            is_match=is_match,
        )

    def _compare(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if self.metric == "cosine":
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            similarity = float(np.dot(a, b) / denom)
            distance = 1.0 - similarity
            return similarity, distance

        distance = float(np.linalg.norm(a - b))
        score = 1.0 / (1.0 + distance)
        return score, distance

    def _is_match(self, score: float, distance: float) -> bool:
        if self.metric == "cosine":
            return score >= self.threshold
        return distance <= self.threshold
