# Real-Time Face Recognition Attendance (AI Service + Core)

A modular, production-oriented face recognition system split into:
- `ai_core`: pure AI pipeline logic
- `api_service`: FastAPI service exposing face extraction endpoints

## Scope

This repository contains:
- Face detection and alignment
- Face preprocessing
- Face embedding extraction (FaceNet)
- Similarity-based face recognition
- In-memory attendance logging
- Stateless AI extraction API

It intentionally excludes:
- Frontend/UI frameworks
- Database integrations

## Key Features

- Real-time webcam processing
- Image-file inference support
- Face detection via OpenCV Haar Cascade or OpenCV DNN
- Face alignment using eye landmarks (Haar eye detector)
- FaceNet-compatible preprocessing (160x160, RGB, normalized to [-1, 1])
- Embedding storage in SQLite (`embeddings.db` by default)
- Configurable matching metric (`cosine` or `euclidean`) and threshold
- Attendance deduplication within a configurable time window
- Optional distributed attendance deduplication with Redis

## Project Structure

- `ai_core/detection.py`: OpenCV face detection, crop expansion, optional alignment
- `ai_core/preprocessing.py`: RGB conversion, lighting normalization, resize and scaling
- `ai_core/model_loader.py`: FaceNet model loading and input-shape validation
- `ai_core/embedding.py`: Embedding extraction and L2 normalization
- `ai_core/recognition.py`: Embedding store + face matching logic
- `ai_core/attendance.py`: AI-side attendance tracker with duplicate suppression
- `ai_core/main_ai.py`: End-to-end CLI pipeline (enrollment + recognition)
- `api_service/app.py`: FastAPI API with `/api/v1/extract-faces`

## Requirements

- Python 3.10+
- A pre-trained FaceNet model (`.h5` or SavedModel format)
- Webcam (for real-time mode)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Enroll an identity from webcam

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source 0 --add-name Alice --enroll-samples 10
```

### 2) Enroll an identity from image

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source path/to/alice.jpg --add-name Alice
```

### 3) Run real-time recognition (webcam)

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source 0 --metric cosine --threshold 0.6
```

### 4) Run recognition on a single image

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source path/to/test.jpg
```

Press `q` to quit webcam mode.

## Detector Options

Use Haar cascade (default):

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source 0 --detector haar
```

Use OpenCV DNN detector:

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --source 0 --detector dnn --dnn-prototxt deploy.prototxt --dnn-weights res10_300x300_ssd_iter_140000.caffemodel
```

## Embedding Database

Embeddings are saved in SQLite (`embeddings.db`) by default.

You can override the file path:

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --db-path data/my_embeddings.db --source 0
```

Use Redis for shared duplicate-suppression across multiple replicas:

```bash
python -m ai_core.main_ai --model-path PATH_TO_FACENET_MODEL --redis-url redis://localhost:6379/0
```

## API Service

Run the FastAPI AI extraction API:

```bash
python -m api_service.app --model-path PATH_TO_FACENET_MODEL --host 0.0.0.0 --port 5000 --workers 2
```

Production example:

```bash
MODEL_PATH=PATH_TO_FACENET_MODEL uvicorn api_service.app:app --host 0.0.0.0 --port 5000 --workers 2
```

Gunicorn + Uvicorn workers:

```bash
MODEL_PATH=PATH_TO_FACENET_MODEL gunicorn api_service.app:app -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:5000
```

Note: for GPU inference, prefer `-w 1` per GPU device to avoid overcommitting VRAM.

Contract endpoint:
- `POST /api/v1/extract-faces`

Request formats:
- Preferred: `multipart/form-data` with file field `image` (raw binary upload)
- Backward compatible: JSON body with `image_base64`

Burst protection:
- API applies bounded in-process inference concurrency.
- If overloaded, endpoint returns `429 SERVICE_BUSY`.

## Notes for Integration

This module is designed to be imported and wrapped later by any API/UI/database layer.

Suggested integration points:
- Use `FaceDetector` + `FaceEmbedder` + `FaceRecognizer` in your service layer
- Replace `EmbeddingStore` persistence backend as needed
- Forward `AttendanceLogger` events to external systems if required

## Limitations

- Accuracy depends heavily on the quality and compatibility of your FaceNet model
- Haar detection is lightweight but less robust than advanced detectors
- This repository does not include training or fine-tuning scripts

## License

This project is released under the MIT License. See `LICENSE`.
