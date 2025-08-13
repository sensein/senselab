# MediaPipe

This image provides a minimal Python 3.11 environment with MediaPipe (\~0.10), OpenCV (headless, \~4.10), and NumPy (\<3) pre-installed.
It is intended for running MediaPipe inference without installing it into your host environment (for example inside Pydra tasks and workflows or other containerized pipelines).

## Usage:

```bash
# Log in to Docker Hub
docker login

# Build locally
docker build -t fabiocat93/mediapipe-deps:latest .

# Push to Docker Hub
docker push fabiocat93/mediapipe-deps:latest

# Run and print MediaPipe version
docker run --rm fabiocat93/mediapipe-deps:latest \
    python -c "import mediapipe as mp; print(mp.__version__)"
```

You can find it already built [here](https://hub.docker.com/r/fabiocat93/mediapipe-deps).