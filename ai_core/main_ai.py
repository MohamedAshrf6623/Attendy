from __future__ import annotations

import argparse

import cv2
import numpy as np

from .attendance import AttendanceLogger, RedisAttendanceState
from .detection import FaceDetection, FaceDetector
from .embedding import FaceEmbedder
from .model_loader import load_facenet_model
from .recognition import EmbeddingStore, FaceRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI module for real-time face recognition attendance."
    )

    parser.add_argument("--db-path", default="embeddings.db", help="Path to embeddings SQLite DB.")
    parser.add_argument(
        "--source",
        default="0",
        help="Input source: webcam index (e.g., 0) or image file path.",
    )

    parser.add_argument("--detector", default="haar", choices=["haar", "dnn"])
    parser.add_argument("--dnn-prototxt", default=None, help="DNN detector prototxt path.")
    parser.add_argument("--dnn-weights", default=None, help="DNN detector caffemodel path.")
    parser.add_argument("--dnn-conf", type=float, default=0.6, help="DNN confidence threshold.")

    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold.")
    parser.add_argument(
        "--duplicate-window",
        type=int,
        default=60,
        help="Duplicate attendance suppression window in seconds.",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL for shared attendance debouncing.",
    )
    parser.add_argument(
        "--redis-key-prefix",
        default="attendance:last",
        help="Redis key prefix used for distributed attendance deduplication.",
    )

    parser.add_argument(
        "--add-name",
        default=None,
        help="If provided, run enrollment and add this identity to the local embedding store.",
    )
    parser.add_argument(
        "--enroll-samples",
        type=int,
        default=10,
        help="Number of face samples to capture for enrollment.",
    )

    return parser.parse_args()


def build_detector(args: argparse.Namespace) -> FaceDetector:
    return FaceDetector(
        method=args.detector,
        confidence_threshold=args.dnn_conf,
        dnn_prototxt=args.dnn_prototxt,
        dnn_weights=args.dnn_weights,
    )


def open_source(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return source


def draw_result(frame: np.ndarray, detection: FaceDetection, label: str, score: float) -> None:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(
        frame,
        f"{label} | score={score:.3f}",
        (x1, max(y1 - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def enroll_from_image(
    image_path: str,
    name: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    store: EmbeddingStore,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = detector.detect_faces(image)
    if not detections:
        raise RuntimeError("No face detected in enrollment image.")

    vectors = [embedder.embed_face(d.face_bgr) for d in detections]
    store.add_identity(name, vectors)
    print(f"Enrolled '{name}' with {len(vectors)} embedding(s).")


def enroll_from_webcam(
    cap: cv2.VideoCapture,
    name: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    store: EmbeddingStore,
    samples: int,
) -> None:
    collected: list[np.ndarray] = []

    while len(collected) < samples:
        ok, frame = cap.read()
        if not ok:
            continue

        detections = detector.detect_faces(frame)
        if detections:
            # Use the largest detected face for enrollment.
            detections = sorted(
                detections,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                reverse=True,
            )
            vector = embedder.embed_face(detections[0].face_bgr)
            collected.append(vector)
            draw_result(frame, detections[0], f"Enrolling {name}", score=1.0)

        cv2.putText(
            frame,
            f"Captured: {len(collected)}/{samples} | Press q to cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Enrollment")

    if not collected:
        raise RuntimeError("Enrollment cancelled or no face samples collected.")

    store.add_identity(name, collected)
    print(f"Enrolled '{name}' with {len(collected)} embedding(s).")


def run_recognition_webcam(
    cap: cv2.VideoCapture,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    recognizer: FaceRecognizer,
    attendance: AttendanceLogger,
) -> None:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        detections = detector.detect_faces(frame)
        for detection in detections:
            query_embedding = embedder.embed_face(detection.face_bgr)
            result = recognizer.recognize(query_embedding)
            draw_result(frame, detection, result.identity, result.score)

            print(
                f"Prediction: {result.identity} | score={result.score:.3f} | "
                f"distance={result.distance:.3f}"
            )

            if result.is_match and attendance.mark_present(result.identity):
                print(attendance.summary())

        cv2.imshow("Real-Time Face Recognition Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def run_recognition_image(
    image_path: str,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    recognizer: FaceRecognizer,
    attendance: AttendanceLogger,
) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = detector.detect_faces(frame)
    if not detections:
        print("No faces detected.")
        return

    for detection in detections:
        query_embedding = embedder.embed_face(detection.face_bgr)
        result = recognizer.recognize(query_embedding)
        draw_result(frame, detection, result.identity, result.score)
        print(
            f"Prediction: {result.identity} | score={result.score:.3f} | "
            f"distance={result.distance:.3f}"
        )

        if result.is_match and attendance.mark_present(result.identity):
            print(attendance.summary())

    cv2.imshow("Image Recognition", frame)
    cv2.waitKey(0)


def main() -> None:
    args = parse_args()

    detector = build_detector(args)
    model = load_facenet_model()
    embedder = FaceEmbedder(model)

    store = EmbeddingStore(args.db_path)
    store.load()

    recognizer = FaceRecognizer(store=store, metric=args.metric, threshold=args.threshold)
    attendance_state = None
    if args.redis_url:
        from redis import Redis

        attendance_state = RedisAttendanceState(
            redis_client=Redis.from_url(args.redis_url),
            key_prefix=args.redis_key_prefix,
        )

    attendance = AttendanceLogger(
        duplicate_window_seconds=args.duplicate_window,
        state_store=attendance_state,
    )

    source = open_source(args.source)

    if args.add_name:
        if isinstance(source, str):
            enroll_from_image(source, args.add_name, detector, embedder, store)
        else:
            enroll_from_webcam(
                source,
                args.add_name,
                detector,
                embedder,
                store,
                samples=args.enroll_samples,
            )
            source.release()
        return

    if isinstance(source, str):
        run_recognition_image(source, detector, embedder, recognizer, attendance)
    else:
        run_recognition_webcam(source, detector, embedder, recognizer, attendance)
        source.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
