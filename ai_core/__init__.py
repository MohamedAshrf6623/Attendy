"""AI core package for face detection and embedding extraction."""

from .detection import FaceDetection, FaceDetector
from .embedding import FaceEmbedder
from .model_loader import load_facenet_model

__all__ = [
    "FaceDetection",
    "FaceDetector",
    "FaceEmbedder",
    "load_facenet_model",
]
