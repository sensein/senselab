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

# torchcodec needs FFmpeg shared libs installed system-wide.
# This is a hard requirement for a complete test environment, but
# since AL2023 doesn't ship FFmpeg in default repos, we warn for now
# and fall back to torchaudio.  TODO: bake FFmpeg into the AMI and
# promote this to REQUIRED_DEPS.
SYSTEM_DEPS = {
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

    sys_missing = []
    for name, module in SYSTEM_DEPS.items():
        try:
            importlib.import_module(module)
        except (ImportError, RuntimeError) as e:
            sys_missing.append(f"  {name} ({module}): {e}")

    if sys_missing:
        warnings.warn(
            "System dependencies unavailable (torchaudio fallback will be used):\n"
            + "\n".join(sys_missing)
            + "\n\nTo fix: install FFmpeg <= 7 shared libs system-wide.\n"
            "  Linux: sudo dnf install ffmpeg-free ffmpeg-free-devel (via rpmfusion)\n"
            "  macOS: brew install ffmpeg@7",
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
            "       Linux: sudo dnf install ffmpeg-free ffmpeg-free-devel (via rpmfusion)",
            "       macOS: brew install ffmpeg@7",
            "     - libsndfile (required by soundfile, usually bundled)",
            "",
        ]
        pytest.exit("\n".join(lines), returncode=1)
