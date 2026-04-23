# Real-Time Face Recognition Attendance (AI Module)

A modular, production-oriented AI pipeline for real-time face recognition attendance using OpenCV and a CNN-based FaceNet model with TensorFlow/Keras.

## Scope

This repository contains **AI logic only**:
- Face detection and alignment
- Face preprocessing
- Face embedding extraction (FaceNet)
- Similarity-based face recognition
- In-memory attendance logging

It intentionally excludes:
- Frontend/UI frameworks
- Backend APIs
- Database integrations

## Key Features

- Real-time webcam processing
- Image-file inference support
- Face detection via OpenCV Haar Cascade or OpenCV DNN
- Face alignment using eye landmarks (Haar eye detector)
- FaceNet-compatible preprocessing (160x160, RGB, normalized to [-1, 1])
- Embedding storage in local `.pkl` file
- Configurable matching metric (`cosine` or `euclidean`) and threshold
- Attendance deduplication within a configurable time window

## Project Structure

- `detection.py`: OpenCV face detection, face crop expansion, basic alignment
- `preprocessing.py`: RGB conversion, lighting normalization, resize and scaling
- `model_loader.py`: FaceNet model loading and input-shape validation
- `embedding.py`: Embedding extraction and L2 normalization
- `recognition.py`: Embedding store + face matching logic
- `attendance.py`: AI-side attendance tracker with duplicate suppression
- `main_ai.py`: End-to-end execution pipeline (enrollment + recognition)

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
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source 0 --add-name Alice --enroll-samples 10
```

### 2) Enroll an identity from image

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source path/to/alice.jpg --add-name Alice
```

### 3) Run real-time recognition (webcam)

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source 0 --metric cosine --threshold 0.6
```

### 4) Run recognition on a single image

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source path/to/test.jpg
```

Press `q` to quit webcam mode.

## Detector Options

Use Haar cascade (default):

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source 0 --detector haar
```

Use OpenCV DNN detector:

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --source 0 --detector dnn --dnn-prototxt deploy.prototxt --dnn-weights res10_300x300_ssd_iter_140000.caffemodel
```

## Embedding Database

Embeddings are saved locally to `embeddings.pkl` by default.

You can override the file path:

```bash
python main_ai.py --model-path PATH_TO_FACENET_MODEL --db-path data/my_embeddings.pkl --source 0
```

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
