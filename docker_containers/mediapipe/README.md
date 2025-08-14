# MediaPipe

This image provides a minimal Python 3.11 environment with MediaPipe (\~0.10), OpenCV (headless, \~4.10), and NumPy (\<3) pre-installed.
It is intended for running MediaPipe inference without installing it into your host environment (for example inside Pydra tasks and workflows or other containerized pipelines).

## Usage:

```bash
# Log in to Docker Hub
docker login
```

- One-time setup (if needed, ensure emulators for cross-builds are installed)
```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

- Create & use a Buildx builder that supports multi-arch
```bash
docker buildx create --name multi --use
docker buildx inspect --bootstrap
```

- Build & push a multi-arch image (from the folder with the Dockerfile)
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t fabiocat93/mediapipe-deps:latest \
  --push .
```
You can find the multi-arch image already built and pushed [here](https://hub.docker.com/r/fabiocat93/mediapipe-deps).


- Verify both architectures are present (You should see entries for linux/amd64 and linux/arm64)
```bash
docker buildx imagetools inspect fabiocat93/mediapipe-deps:latest
```

- Run and print MediaPipe version
```bash
docker run --rm fabiocat93/mediapipe-deps:latest \
    --platform=linux/amd64 \
    python -c "import mediapipe as mp; print(mp.__version__)"
```
and
```bash
docker run --rm fabiocat93/mediapipe-deps:latest \
    --platform=linux/arm64 \
    python -c "import mediapipe as mp; print(mp.__version__)"
```
