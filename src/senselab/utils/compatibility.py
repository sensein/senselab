"""Feature/dependency compatibility matrix and runtime checks.

Maps each senselab public API function to its required dependencies,
supported version ranges (lower AND upper bounds), and isolation
requirements. Used for:

1. **Runtime**: graceful error messages when deps are missing or wrong version
2. **Documentation**: auto-generated compatibility matrix
3. **CI test matrix**: generate pytest parametrize configs for version testing
"""

import importlib
import sys
from dataclasses import dataclass, field
from typing import Optional

from packaging.specifiers import SpecifierSet
from packaging.version import Version


@dataclass
class VersionRange:
    """A version range with explicit lower and upper bounds.

    Uses PEP 440 specifiers (e.g., ">=2.8,<2.11").
    Upper bounds should be set conservatively — many packages don't
    declare breaking changes properly.
    """

    spec: str  # PEP 440 specifier set, e.g., ">=2.8,<2.11"

    def contains(self, version: str) -> bool:
        """Check if a version string is within this range."""
        return Version(version) in SpecifierSet(self.spec)

    def __str__(self) -> str:
        """Return the PEP 440 specifier string."""
        return self.spec


@dataclass
class CompatibilityEntry:
    """Compatibility metadata for a single function."""

    required_deps: list[str] = field(default_factory=list)
    python_versions: VersionRange = field(default_factory=lambda: VersionRange(">=3.11,<3.13"))
    torch_versions: VersionRange = field(default_factory=lambda: VersionRange(">=2.8,<3.0"))
    gpu_required: bool = False
    isolated: bool = False
    venv_name: Optional[str] = None
    venv_requirements: list[str] = field(default_factory=list)
    venv_python: Optional[str] = None
    install_hint: str = ""
    # Known-good version ranges for each dep (upper bounds from testing)
    dep_versions: dict[str, str] = field(default_factory=dict)


# ── Compatibility Matrix ──────────────────────────────────────────────

COMPATIBILITY_MATRIX: dict[str, CompatibilityEntry] = {
    # ── Audio: Speech to Text ──
    "audio.tasks.speech_to_text.transcribe_audios": CompatibilityEntry(
        required_deps=["transformers", "torchaudio"],
        dep_versions={"transformers": ">=4.40,<5.0", "torchaudio": ">=2.8,<3.0"},
        install_hint="pip install senselab",
    ),
    # ── Audio: Speaker Diarization ──
    "audio.tasks.speaker_diarization.diarize_audios": CompatibilityEntry(
        required_deps=["pyannote-audio", "torchaudio"],
        dep_versions={"pyannote-audio": ">=3.0,<5.0", "torchaudio": ">=2.8,<3.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Speaker Embeddings ──
    "audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios": CompatibilityEntry(
        required_deps=["speechbrain", "torchaudio"],
        dep_versions={"speechbrain": ">=1.0,<2.0", "torchaudio": ">=2.8,<3.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Speech Enhancement ──
    "audio.tasks.speech_enhancement.enhance_audios": CompatibilityEntry(
        required_deps=["speechbrain", "torchaudio"],
        dep_versions={"speechbrain": ">=1.0,<2.0", "torchaudio": ">=2.8,<3.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Voice Cloning (ISOLATED) ──
    "audio.tasks.voice_cloning.clone_voices": CompatibilityEntry(
        required_deps=["coqui-tts"],
        dep_versions={"coqui-tts": ">=0.27,<1.0", "torch": ">=2.4,<2.9"},
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
        dep_versions={"transformers": ">=4.40,<5.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Classification ──
    "audio.tasks.classification.classify_audios": CompatibilityEntry(
        required_deps=["transformers"],
        dep_versions={"transformers": ">=4.40,<5.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Forced Alignment ──
    "audio.tasks.forced_alignment.align_transcriptions": CompatibilityEntry(
        required_deps=["transformers", "torchaudio"],
        dep_versions={"transformers": ">=4.40,<5.0", "torchaudio": ">=2.8,<3.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Audio: Features Extraction (PPGs - ISOLATED) ──
    "audio.tasks.features_extraction.extract_ppg_from_audios": CompatibilityEntry(
        required_deps=["ppgs", "espnet"],
        dep_versions={"ppgs": ">=0.0.9,<0.0.10", "espnet": ">=202205", "torch": ">=2.0,<2.9"},
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
        dep_versions={"transformers": ">=4.40,<5.0"},
        gpu_required=True,
        install_hint="pip install senselab",
    ),
    # ── Video: Pose Estimation ──
    "video.tasks.pose_estimation.estimate_pose": CompatibilityEntry(
        required_deps=["ultralytics", "opencv-python-headless"],
        dep_versions={"ultralytics": ">=8.0,<9.0", "opencv-python-headless": ">=4.8,<5.0"},
        install_hint="pip install 'senselab[video]'",
    ),
}


# ── Runtime checks ────────────────────────────────────────────────────

# Map pip package names to their importable module names
_IMPORT_MAP: dict[str, str] = {
    "pyannote-audio": "pyannote.audio",
    "coqui-tts": "TTS",
    "opencv-python-headless": "cv2",
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "torch-audiomentations": "torch_audiomentations",
    "praat-parselmouth": "parselmouth",
    "huggingface-hub": "huggingface_hub",
}


def _get_installed_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package, or None if not installed."""
    import_name = _IMPORT_MAP.get(package_name, package_name.replace("-", "_"))
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", None)
    except (ImportError, RuntimeError):
        return None


def check_compatibility(function_key: str) -> bool:
    """Check if the dependencies for a function are available and compatible.

    Verifies both presence and version ranges.

    Args:
        function_key: Key in COMPATIBILITY_MATRIX.

    Returns:
        True if all dependencies are available and within version bounds.

    Raises:
        ImportError: Missing package with install hint.
        RuntimeError: Package installed but wrong version.
    """
    entry = COMPATIBILITY_MATRIX.get(function_key)
    if entry is None:
        return True

    # Check Python version range
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if not entry.python_versions.contains(py_ver):
        raise RuntimeError(
            f"Function '{function_key}' requires Python {entry.python_versions}, "
            f"but you are running Python {py_ver}."
        )

    # Isolated backends don't need deps in the host environment
    if entry.isolated:
        return True

    # Check each required dependency — presence AND version
    for dep in entry.required_deps:
        version = _get_installed_version(dep)
        if version is None:
            raise ImportError(
                f"Package '{dep}' is required for '{function_key}' but is not installed.\n"
                f"Install with: {entry.install_hint}"
            )

        # Check version range if specified
        version_spec = entry.dep_versions.get(dep)
        if version_spec and not VersionRange(version_spec).contains(version):
            raise RuntimeError(
                f"Package '{dep}' version {version} is outside the tested range "
                f"{version_spec} for '{function_key}'.\n"
                f"This may work but is not guaranteed. "
                f"Install a compatible version: pip install '{dep}{version_spec}'"
            )

    return True


# ── Test matrix generation ────────────────────────────────────────────


def generate_test_matrix() -> list[dict[str, str]]:
    """Generate a pytest-compatible test matrix from the compatibility matrix.

    Returns a list of test configurations, each with:
    - function_key: the function being tested
    - python_version: Python version to test with
    - torch_version: torch version to test with
    - deps: comma-separated dep specs

    This can be used with ``@pytest.mark.parametrize`` or to generate
    CI matrix entries.
    """
    matrix: list[dict[str, str]] = []

    # Extract unique Python and torch version ranges
    python_versions_to_test = ["3.11", "3.12"]
    torch_versions_to_test = ["2.8", "2.10"]  # min tested, max tested

    for func_key, entry in COMPATIBILITY_MATRIX.items():
        if entry.isolated:
            # Isolated backends test in their own venv — separate config
            matrix.append({
                "function_key": func_key,
                "python_version": entry.venv_python or "3.11",
                "torch_version": "venv-managed",
                "deps": ",".join(entry.venv_requirements),
                "isolated": "true",
            })
            continue

        for py_ver in python_versions_to_test:
            if not entry.python_versions.contains(f"{py_ver}.0"):
                continue
            for torch_ver in torch_versions_to_test:
                if not entry.torch_versions.contains(f"{torch_ver}.0"):
                    continue
                matrix.append({
                    "function_key": func_key,
                    "python_version": py_ver,
                    "torch_version": torch_ver,
                    "deps": ",".join(f"{d}{entry.dep_versions.get(d, '')}" for d in entry.required_deps),
                    "isolated": "false",
                })

    return matrix


# ── Documentation generation ──────────────────────────────────────────


def get_matrix() -> dict[str, CompatibilityEntry]:
    """Return the full compatibility matrix."""
    return COMPATIBILITY_MATRIX


def generate_matrix_markdown() -> str:
    """Generate a markdown table from the compatibility matrix."""
    lines = [
        "# Senselab Compatibility Matrix",
        "",
        "| Function | Required Deps | Dep Versions | GPU | Isolated | Python | Torch |",
        "|----------|--------------|-------------|-----|----------|--------|-------|",
    ]
    for key, entry in sorted(COMPATIBILITY_MATRIX.items()):
        deps = ", ".join(entry.required_deps) if entry.required_deps else "core"
        dep_vers = ", ".join(f"{k}{v}" for k, v in entry.dep_versions.items()) if entry.dep_versions else "—"
        gpu = "Yes" if entry.gpu_required else "No"
        isolated = f"Yes ({entry.venv_name})" if entry.isolated else "No"
        lines.append(
            f"| `{key}` | {deps} | {dep_vers} | {gpu} | {isolated} "
            f"| {entry.python_versions} | {entry.torch_versions} |"
        )

    lines.extend([
        "",
        "## Test Matrix",
        "",
        "| Function | Python | Torch | Deps | Isolated |",
        "|----------|--------|-------|------|----------|",
    ])
    for row in generate_test_matrix():
        lines.append(
            f"| `{row['function_key']}` | {row['python_version']} | {row['torch_version']} "
            f"| {row['deps']} | {row['isolated']} |"
        )

    return "\n".join(lines) + "\n"
