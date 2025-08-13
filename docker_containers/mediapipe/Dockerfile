FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mediapipe~=0.10" \
    "opencv-python-headless~=4.12" \
    "numpy>1,<3"

WORKDIR /app