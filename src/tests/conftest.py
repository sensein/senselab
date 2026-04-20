"""Top-level test configuration.

Verifies all dependencies (core + extras) are importable before any
tests run.  CI installs with `uv sync --all-extras --group dev`, so
every Python dependency should be available.  If anything fails to
import, the test session aborts immediately with a clear message.

The only things NOT checked here are:
- Subprocess-venv packages (coqui-tts, ppgs, SPARC) — installed on demand
- Docker — runtime service, not a pip package
"""

import importlib

import pytest

# All dependencies that must be importable in the test environment.
# Grouped by pyproject.toml section for clarity.
REQUIRED_DEPS = {
    # Core dependencies
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchcodec": "torchcodec",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "speechbrain": "speechbrain",
    "pyannote-audio": "pyannote.audio",
    "opensmile": "opensmile",
    "praat-parselmouth": "parselmouth",
    "audiomentations": "audiomentations",
    "torch-audiomentations": "torch_audiomentations",
    "vocos": "vocos",
    "pydantic": "pydantic",
    "datasets": "datasets",
    "scikit-learn": "sklearn",
    "soundfile": "soundfile",
    # Extra: text
    "sentence-transformers": "sentence_transformers",
    "pylangacq": "pylangacq",
    # Extra: nlp
    "jiwer": "jiwer",
    "nltk": "nltk",
    # Extra: video
    "av": "av",
    "opencv": "cv2",
    "ultralytics": "ultralytics",
    # Extra: articulatory
    "speech-articulatory-coding": "articulatory_coding",
}


def pytest_configure(config: pytest.Config) -> None:
    """Verify all dependencies are importable at session start."""
    missing = []
    for name, module in REQUIRED_DEPS.items():
        try:
            importlib.import_module(module)
        except (ImportError, RuntimeError) as e:
            missing.append(f"  {name} ({module}): {e}")

    if missing:
        lines = [
            "\n\nDependencies failed to import — test environment is broken.\n",
            "Failed imports:\n" + "\n".join(missing) + "\n",
            "\nHow to fix:",
            "  1. Install Python packages:  uv sync --all-extras --group dev",
            "  2. System dependencies:",
            "     - FFmpeg <= 7 shared libs (required by torchcodec).",
            "       The CI workflow compiles FFmpeg from source; ensure build",
            "       tools are available (gcc, cmake, nasm — see EC2_GPU_RUNNER.md).",
            "       macOS: brew install ffmpeg@7",
            "     - libsndfile (required by soundfile, usually bundled)",
            "",
        ]
        pytest.exit("\n".join(lines), returncode=1)
