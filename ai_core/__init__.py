"""AI core package for face detection, embeddings, recognition, and attendance."""

from .attendance import AttendanceLogger
from .detection import FaceDetection, FaceDetector
from .embedding import FaceEmbedder
from .model_loader import load_facenet_model
from .recognition import EmbeddingStore, FaceRecognizer, RecognitionResult

__all__ = [
    "AttendanceLogger",
    "FaceDetection",
    "FaceDetector",
    "FaceEmbedder",
    "load_facenet_model",
    "EmbeddingStore",
    "FaceRecognizer",
    "RecognitionResult",
]
