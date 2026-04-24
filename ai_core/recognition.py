from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


@dataclass
class RecognitionResult:
    identity: str
    score: float
    distance: float
    is_match: bool


class EmbeddingStore:
    """SQLite-backed embedding storage with in-memory cache for fast reads."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.embeddings: dict[str, list[np.ndarray]] = {}
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        parent = Path(self.db_path).parent
        if str(parent) not in {"", "."}:
            parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dim INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_identity ON embeddings(identity)"
            )

    def load(self) -> None:
        with self._lock:
            loaded: dict[str, list[np.ndarray]] = {}
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT identity, vector, dim FROM embeddings ORDER BY id"
                ).fetchall()

            for identity, vector_blob, dim in rows:
                vector = np.frombuffer(vector_blob, dtype=np.float32, count=dim).copy()
                loaded.setdefault(identity, []).append(vector)

            self.embeddings = loaded

    def save(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM embeddings")
                for name, vectors in self.embeddings.items():
                    for vec in vectors:
                        vec32 = np.asarray(vec, dtype=np.float32)
                        conn.execute(
                            "INSERT INTO embeddings(identity, vector, dim) VALUES (?, ?, ?)",
                            (name, vec32.tobytes(), int(vec32.size)),
                        )
                conn.commit()

    def add_identity(self, name: str, vectors: list[np.ndarray]) -> None:
        normalized_vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        if not normalized_vectors:
            return

        with self._lock:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                for vec in normalized_vectors:
                    conn.execute(
                        "INSERT INTO embeddings(identity, vector, dim) VALUES (?, ?, ?)",
                        (name, vec.tobytes(), int(vec.size)),
                    )
                conn.commit()

            if name not in self.embeddings:
                self.embeddings[name] = []
            self.embeddings[name].extend(normalized_vectors)

    def identities(self) -> list[str]:
        with self._lock:
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

    def _compare(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
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
