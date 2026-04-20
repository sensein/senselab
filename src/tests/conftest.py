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
import warnings

import pytest

# All dependencies that must be importable in the test environment.
# Grouped by pyproject.toml section for clarity.
REQUIRED_DEPS = {
    # Core dependencies
    "torch": "torch",
    "torchaudio": "torchaudio",
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

# torchcodec needs FFmpeg shared libs AND libpython as a shared library.
# uv's python-build-standalone doesn't include libpython.so (static build),
# so torchcodec's native extension fails to load on those environments.
# We warn instead of aborting — torchaudio provides a fallback for all
# audio I/O operations.  TODO: resolve once python-build-standalone ships
# shared libs or torchcodec removes the libpython dependency.
SOFT_DEPS = {
    "torchcodec": "torchcodec",
}


def pytest_configure(config: pytest.Config) -> None:
    """Verify all dependencies are importable at session start."""
    missing = []
    for name, module in REQUIRED_DEPS.items():
        try:
            importlib.import_module(module)
        except (ImportError, RuntimeError) as e:
            missing.append(f"  {name} ({module}): {e}")

    soft_missing = []
    for name, module in SOFT_DEPS.items():
        try:
            importlib.import_module(module)
        except (ImportError, RuntimeError) as e:
            soft_missing.append(f"  {name} ({module}): {e}")

    if soft_missing:
        warnings.warn(
            "Optional system deps unavailable (torchaudio fallback will be used):\n"
            + "\n".join(soft_missing)
            + "\n\nTo fix: install FFmpeg <= 7 shared libs AND ensure Python was"
            " built with --enable-shared (uv's python-build-standalone is static).",
            stacklevel=1,
        )

    if missing:
        lines = [
            "\n\nDependencies failed to import — test environment is broken.\n",
            "Failed imports:\n" + "\n".join(missing) + "\n",
            "\nHow to fix:",
            "  1. Install Python packages:  uv sync --all-extras --group dev",
            "  2. System dependencies:",
            "     - FFmpeg <= 7 shared libs (required by torchcodec).",
            "       CI installs via: conda install 'ffmpeg<8' (miniforge/conda-forge)",
            "       macOS: brew install ffmpeg@7",
            "     - libpython shared library (torchcodec needs libpython3.XX.so)",
            "       Note: uv's python-build-standalone is static — torchcodec uses",
            "       torchaudio as fallback when libpython.so is unavailable.",
            "",
        ]
        pytest.exit("\n".join(lines), returncode=1)
