FROM python:3.10-slim

# the MODEL_PATH is included here because it was needed in 
# a previous version of the app, although it is not needed now,
# but it still exists in the source code.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/dummy_path

WORKDIR /app

RUN apt-get update && apt-get install -y \
libgl1 \
libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir .

RUN groupadd -r nonroot && useradd -m -r -g nonroot nonroot

# this line is used to download and cache the model in the docker image
# instead of downloading it after container startup, which prevent the cold start problem
RUN python -c "from keras_facenet import FaceNet; FaceNet()"

COPY --chown=nonroot:nonroot ./ai_core ./ai_core
COPY --chown=nonroot:nonroot ./api_service ./api_service

USER nonroot

EXPOSE 5000

CMD ["gunicorn", "api_service.app:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:5000"]
