"""Feature/dependency compatibility matrix and runtime checks.

Maps each senselab public API function to its required dependencies,
supported Python/torch versions, and isolation requirements. Used for:
1. Runtime: graceful error messages when deps are missing
2. Documentation: auto-generated compatibility matrix
"""

import importlib
import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompatibilityEntry:
    """Compatibility metadata for a single function."""

    required_deps: list[str] = field(default_factory=list)
    python_versions: str = ">=3.11"
    torch_versions: str = ">=2.8"
    gpu_required: bool = False
    isolated: bool = False
    venv_name: Optional[str] = None
    venv_requirements: list[str] = field(default_factory=list)
    venv_python: Optional[str] = None
    install_hint: str = ""


# ── Compatibility Matrix ──────────────────────────────────────────────
# Populated incrementally as functions are analyzed.
# Key: "module.function_name" (e.g., "audio.tasks.voice_cloning.clone_voices")

COMPATIBILITY_MATRIX: dict[str, CompatibilityEntry] = {
    # ── Audio: Speech to Text ──
    "audio.tasks.speech_to_text.transcribe_audios": CompatibilityEntry(
        required_deps=["transformers", "torchaudio"],
        gpu_required=False,
        install_hint="pip install senselab",
    ),
    # ── Audio: Speaker Diarization ──
    "audio.tasks.speaker_diarization.diarize_audios": CompatibilityEntry(
        required_deps=["pyannote-audio", "torchaudio"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Speaker Embeddings ──
    "audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios": CompatibilityEntry(
        required_deps=["speechbrain", "torchaudio"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Speech Enhancement ──
    "audio.tasks.speech_enhancement.enhance_audios": CompatibilityEntry(
        required_deps=["speechbrain", "torchaudio"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Voice Cloning (ISOLATED) ──
    "audio.tasks.voice_cloning.clone_voices": CompatibilityEntry(
        required_deps=["coqui-tts"],
        gpu_required=True,
        isolated=True,
        venv_name="coqui",
        venv_requirements=["coqui-tts~=0.27", "torch~=2.8"],
        venv_python="3.11",
        install_hint="Automatically provisioned in isolated environment",
    ),
    # ── Audio: Text to Speech ──
    "audio.tasks.text_to_speech.synthesize_texts": CompatibilityEntry(
        required_deps=["transformers"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Classification ──
    "audio.tasks.classification.classify_audios": CompatibilityEntry(
        required_deps=["transformers"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Forced Alignment ──
    "audio.tasks.forced_alignment.align_transcriptions": CompatibilityEntry(
        required_deps=["transformers", "torchaudio"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Features Extraction (PPGs - ISOLATED) ──
    "audio.tasks.features_extraction.extract_ppg_from_audios": CompatibilityEntry(
        required_deps=["ppgs", "espnet"],
        gpu_required=True,
        isolated=True,
        venv_name="ppgs",
        venv_requirements=["ppgs>=0.0.9,<0.0.10", "espnet", "snorkel>=0.10.0,<0.11.0", "lightning~=2.4"],
        venv_python="3.11",
        install_hint="Automatically provisioned in isolated environment",
    ),
    # ── Text: Embeddings ──
    "text.tasks.embeddings_extraction.extract_embeddings_from_text": CompatibilityEntry(
        required_deps=["transformers"],
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Video: Pose Estimation ──
    "video.tasks.pose_estimation.estimate_pose": CompatibilityEntry(
        required_deps=["ultralytics", "opencv-python-headless"],
        gpu_required=False,
        install_hint="pip install 'senselab[video]'",
    ),
}


def check_compatibility(function_key: str) -> bool:
    """Check if the dependencies for a function are available.

    Args:
        function_key: Key in COMPATIBILITY_MATRIX (e.g., "audio.tasks.voice_cloning.clone_voices").

    Returns:
        True if all dependencies are available.

    Raises:
        ImportError: With a clear message naming the missing package and install command.
    """
    entry = COMPATIBILITY_MATRIX.get(function_key)
    if entry is None:
        return True  # Unknown function — no restrictions

    # Check Python version
    # Simple check: extract minimum from ">=X.Y"
    if entry.python_versions.startswith(">="):
        min_ver = entry.python_versions[2:].split(",")[0]
        parts = min_ver.split(".")
        if sys.version_info < tuple(int(p) for p in parts):
            raise ImportError(
                f"Function '{function_key}' requires Python {entry.python_versions}, "
                f"but you are running Python {sys.version_info.major}.{sys.version_info.minor}."
            )

    # Isolated backends don't need deps in the host environment
    if entry.isolated:
        return True

    # Check each required dependency
    for dep in entry.required_deps:
        module_name = dep.replace("-", "_").replace(".", "_")
        # Map common package names to their import names
        import_map = {
            "pyannote_audio": "pyannote.audio",
            "coqui_tts": "TTS",
            "opencv_python_headless": "cv2",
            "scikit_learn": "sklearn",
            "sentence_transformers": "sentence_transformers",
        }
        import_name = import_map.get(module_name, module_name)
        try:
            importlib.import_module(import_name)
        except (ImportError, RuntimeError):
            raise ImportError(
                f"Package '{dep}' is required for '{function_key}' but is not installed.\n"
                f"Install with: {entry.install_hint}"
            ) from None

    return True


def get_matrix() -> dict[str, CompatibilityEntry]:
    """Return the full compatibility matrix."""
    return COMPATIBILITY_MATRIX


def generate_matrix_markdown() -> str:
    """Generate a markdown table from the compatibility matrix."""
    lines = [
        "# Senselab Compatibility Matrix",
        "",
        "| Function | Required Deps | GPU | Isolated | Python | Torch |",
        "|----------|--------------|-----|----------|--------|-------|",
    ]
    for key, entry in sorted(COMPATIBILITY_MATRIX.items()):
        deps = ", ".join(entry.required_deps) if entry.required_deps else "core"
        gpu = "Yes" if entry.gpu_required else "No"
        isolated = f"Yes ({entry.venv_name})" if entry.isolated else "No"
        lines.append(f"| `{key}` | {deps} | {gpu} | {isolated} | {entry.python_versions} | {entry.torch_versions} |")

    return "\n".join(lines) + "\n"
