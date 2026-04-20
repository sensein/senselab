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
from typing import List

import pytest
import torch

from senselab.utils.data_structures import DeviceType

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
    # Note: coqui-tts, ppgs, and sparc run in isolated subprocess venvs
    # and are not checked here.
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


# ---------------------------------------------------------------------------
# Device parameterization fixtures (available to ALL tests)
# ---------------------------------------------------------------------------
def _available_devices() -> List[DeviceType]:
    """Detect available compute devices."""
    devices = [DeviceType.CPU]
    if torch.cuda.is_available():
        devices.append(DeviceType.CUDA)
    if torch.backends.mps.is_available():
        devices.append(DeviceType.MPS)
    return devices


def _gpu_devices() -> List[DeviceType]:
    """Detect available GPU devices (MPS or CUDA)."""
    return [d for d in _available_devices() if d != DeviceType.CPU]


AVAILABLE_DEVICES = _available_devices()
GPU_DEVICES = _gpu_devices()


@pytest.fixture(params=AVAILABLE_DEVICES, ids=lambda d: f"device={d.value}")
def any_device(request: pytest.FixtureRequest) -> DeviceType:
    """Parameterize over all available devices (cpu, mps, cuda).

    Use for Tier 1 and Tier 2 models that can run on any device.
    """
    return request.param


@pytest.fixture(
    params=GPU_DEVICES if GPU_DEVICES else [pytest.param("skip", marks=pytest.mark.skip("No GPU available"))],
    ids=lambda d: f"device={d.value}" if isinstance(d, DeviceType) else "no-gpu",
)
def gpu_device(request: pytest.FixtureRequest) -> DeviceType:
    """Parameterize over GPU-only devices (mps, cuda).

    Use for Tier 3 models that are too slow on CPU.
    Skips automatically if no GPU is available.
    """
    return request.param
