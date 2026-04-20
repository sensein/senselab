#!/usr/bin/env python3
"""Profile model inference to determine CPU/GPU tier classification.

Runs each model used in senselab tests through a standardized benchmark:
  - Wall-clock time (cold start + warm inference)
  - Peak host RAM delta
  - Peak GPU memory (if CUDA/MPS)

Can run standalone or as a pytest plugin (--profile-tiers flag).

Usage:
    # Standalone: profile all models on all available devices
    uv run python scripts/profile_model_tiers.py

    # Standalone: specific tier(s) or device
    uv run python scripts/profile_model_tiers.py --tiers 1 2
    uv run python scripts/profile_model_tiers.py --device cuda

    # Via pytest: profile tests and collect timing/memory data
    uv run pytest src/tests/ --profile-tiers --profile-output=artifacts/profile-report.json

    # With scalene (deep CPU/GPU/memory profiling on a single model):
    uv run scalene scripts/profile_model_tiers.py --models whisper-tiny --device cpu

    # Update the audit document with results
    uv run python scripts/profile_model_tiers.py --update-audit

    # JSON output only (for CI consumption)
    uv run python scripts/profile_model_tiers.py --json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import torch

# ---------------------------------------------------------------------------
# Tier thresholds (seconds on CPU).  Models exceeding a threshold move to the
# next tier.  Adjust these after initial profiling if the defaults are wrong.
# ---------------------------------------------------------------------------
TIER_THRESHOLDS = {
    # tier 1 → tier 2 boundary (seconds, CPU wall-clock)
    "tier1_max_cpu_seconds": 30,
    # tier 2 → tier 3 boundary
    "tier2_max_cpu_seconds": 120,
    # Peak host RAM delta (GB) — flag if a model uses more than this
    "ram_warning_gb": 4.0,
    # Peak GPU memory (GB) — flag if exceeds this on a T4 (16GB)
    "gpu_warning_gb": 12.0,
}


@dataclass
class ProfileResult:
    """Result of profiling a single model on a single device."""

    model_id: str
    task: str
    device: str
    wall_seconds: float = 0.0
    warm_seconds: float = 0.0  # second run (model cached)
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    error: Optional[str] = None
    suggested_tier: Optional[int] = None


@dataclass
class ModelSpec:
    """Specification for a model to profile."""

    model_id: str
    task: str
    current_tier: int
    setup: Callable[..., Any]  # returns (func, args, kwargs) to call for inference
    cleanup: Optional[Callable[[], None]] = None


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def get_ram_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_gpu_mb(device: str) -> float:
    """Current GPU memory allocated in MB."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_size() / (1024 * 1024)
        except Exception:
            return 0.0
    return 0.0


def reset_gpu(device: str) -> None:
    """Free GPU caches."""
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def peak_gpu_mb(device: str) -> float:
    """Peak GPU memory since last reset, in MB."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    # MPS: no peak tracking; return current as approximation
    return get_gpu_mb(device)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def _load_mono_16k():
    """Load the standard mono test audio resampled to 16kHz."""
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import resample_audios

    mono_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "tests"
        / "data_for_testing"
        / "audio_48khz_mono_16bits.wav"
    )
    audio = Audio(filepath=str(mono_path))
    return resample_audios([audio], 16000)[0]


# ---------------------------------------------------------------------------
# Model setup functions
# Each returns (callable, args, kwargs) for a single inference call.
# ---------------------------------------------------------------------------


def setup_text_embeddings_minilm(device: str):
    from senselab.text.tasks.embeddings_extraction import extract_embeddings_from_text
    from senselab.utils.data_structures import DeviceType, SentenceTransformersModel

    model = SentenceTransformersModel(path_or_uri="sentence-transformers/all-MiniLM-L6-v2", revision="main")
    dev = DeviceType(device)
    texts = ["Hello, world!", "Testing embeddings extraction.", "A third sentence for good measure."]
    return extract_embeddings_from_text, (texts, model), {"device": dev}


def setup_whisper_tiny(device: str):
    from senselab.audio.tasks.speech_to_text import transcribe_audios
    from senselab.utils.data_structures import DeviceType, HFModel

    model = HFModel(path_or_uri="openai/whisper-tiny", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return transcribe_audios, ([audio],), {"model": model, "device": dev}


def setup_mms_tts_eng(device: str):
    from senselab.audio.tasks.text_to_speech import synthesize_texts
    from senselab.utils.data_structures import DeviceType, HFModel

    model = HFModel(path_or_uri="facebook/mms-tts-eng", revision="main")
    dev = DeviceType(device)
    return synthesize_texts, (["Hello world"],), {"model": model, "device": dev}


def setup_metricgan(device: str):
    from senselab.audio.tasks.speech_enhancement import enhance_audios
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/metricgan-plus-voicebank", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return enhance_audios, ([audio],), {"model": model, "device": dev}


def setup_ecapa(device: str):
    from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return extract_speaker_embeddings_from_audios, ([audio],), {"model": model, "device": dev}


def setup_xvector(device: str):
    from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-xvect-voxceleb", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return extract_speaker_embeddings_from_audios, ([audio],), {"model": model, "device": dev}


def setup_resnet_speaker(device: str):
    from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-resnet-voxceleb", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return extract_speaker_embeddings_from_audios, ([audio],), {"model": model, "device": dev}


def setup_sepformer(device: str):
    from senselab.audio.tasks.speech_enhancement import enhance_audios
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return enhance_audios, ([audio],), {"model": model, "device": dev}


def setup_wav2vec2_base(device: str):
    from senselab.audio.tasks.forced_alignment.forced_alignment import align_transcriptions
    from senselab.utils.data_structures import Language, ScriptLine

    audio = _load_mono_16k()
    transcript = ScriptLine(text="this is a test of forced alignment")
    lang = Language(language_code="en")
    return align_transcriptions, ([(audio, transcript, lang)],), {}


def setup_pyannote_diarization(device: str):
    from senselab.audio.tasks.speaker_diarization import diarize_audios
    from senselab.utils.data_structures import DeviceType, PyannoteAudioModel

    model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return diarize_audios, ([audio],), {"model": model, "device": dev}


def setup_wav2vec2_emotion_dim(device: str):
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.classification.speech_emotion_recognition import classify_emotions_from_speech
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.utils.data_structures import DeviceType, HFModel

    mono_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "tests"
        / "data_for_testing"
        / "audio_48khz_mono_16bits.wav"
    )
    audio = Audio(filepath=str(mono_path))
    resampled = resample_audios([audio], 16000)

    model = HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    dev = DeviceType(device)
    return classify_emotions_from_speech, (resampled, model), {"device": dev}


def setup_seamless_m4t(device: str):
    from senselab.audio.tasks.speech_to_text import transcribe_audios
    from senselab.utils.data_structures import DeviceType, HFModel

    model = HFModel(path_or_uri="facebook/seamless-m4t-unity-small")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return transcribe_audios, ([audio],), {"model": model, "device": dev}


def setup_wav2vec2_xlsr_emotion(device: str):
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.classification.speech_emotion_recognition import classify_emotions_from_speech
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.utils.data_structures import DeviceType, HFModel

    mono_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "tests"
        / "data_for_testing"
        / "audio_48khz_mono_16bits.wav"
    )
    audio = Audio(filepath=str(mono_path))
    resampled = resample_audios([audio], 16000)

    model = HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    dev = DeviceType(device)
    return classify_emotions_from_speech, (resampled, model), {"device": dev}


def setup_bark_small(device: str):
    from senselab.audio.tasks.text_to_speech import synthesize_texts
    from senselab.utils.data_structures import DeviceType, HFModel

    model = HFModel(path_or_uri="suno/bark-small", revision="main")
    dev = DeviceType(device)
    return synthesize_texts, (["Hello"],), {"model": model, "device": dev}


def setup_pyannote_vad(device: str):
    from senselab.audio.tasks.voice_activity_detection import detect_human_voice_activity_in_audios
    from senselab.utils.data_structures import DeviceType, PyannoteAudioModel

    model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return detect_human_voice_activity_in_audios, ([audio],), {"model": model, "device": dev}


def setup_speaker_verification(device: str):
    from senselab.audio.tasks.speaker_verification import verify_speaker
    from senselab.utils.data_structures import DeviceType, SpeechBrainModel

    model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
    dev = DeviceType(device)
    audio = _load_mono_16k()
    return verify_speaker, (audio, audio), {"model": model, "device": dev}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_SPECS: List[ModelSpec] = [
    # Tier 1: CPU-fast (<300MB, expected <30s)
    ModelSpec("sentence-transformers/all-MiniLM-L6-v2", "text-embeddings", 1, setup_text_embeddings_minilm),
    ModelSpec("openai/whisper-tiny", "speech-to-text", 1, setup_whisper_tiny),
    ModelSpec("facebook/mms-tts-eng", "text-to-speech", 1, setup_mms_tts_eng),
    ModelSpec("speechbrain/metricgan-plus-voicebank", "speech-enhancement", 1, setup_metricgan),
    ModelSpec("speechbrain/spkrec-ecapa-voxceleb", "speaker-embeddings", 1, setup_ecapa),
    ModelSpec("speechbrain/spkrec-xvect-voxceleb", "speaker-embeddings", 1, setup_xvector),
    ModelSpec("speechbrain/spkrec-resnet-voxceleb", "speaker-embeddings", 1, setup_resnet_speaker),
    ModelSpec("speechbrain/sepformer-wham16k-enhancement", "speech-enhancement", 1, setup_sepformer),
    ModelSpec("facebook/wav2vec2-base-960h", "forced-alignment", 1, setup_wav2vec2_base),
    ModelSpec("speechbrain/spkrec-ecapa-voxceleb", "speaker-verification", 1, setup_speaker_verification),
    # Tier 2: CPU-feasible (300MB-1GB, expected <120s)
    ModelSpec("pyannote/speaker-diarization-community-1", "speaker-diarization", 2, setup_pyannote_diarization),
    ModelSpec("pyannote/speaker-diarization-community-1", "voice-activity-detection", 2, setup_pyannote_vad),
    ModelSpec(
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "emotion-continuous", 2, setup_wav2vec2_emotion_dim
    ),
    ModelSpec("facebook/seamless-m4t-unity-small", "speech-to-text", 2, setup_seamless_m4t),
    # Tier 3: GPU-preferred (>1GB, expected >120s on CPU)
    ModelSpec(
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "emotion-discrete",
        3,
        setup_wav2vec2_xlsr_emotion,
    ),
    ModelSpec("suno/bark-small", "text-to-speech", 3, setup_bark_small),
]


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


def profile_one(spec: ModelSpec, device: str, warmup: bool = True) -> ProfileResult:
    """Profile a single model on a single device."""
    result = ProfileResult(model_id=spec.model_id, task=spec.task, device=device)

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        result.error = "CUDA not available"
        return result
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        result.error = "MPS not available"
        return result

    reset_gpu(device)
    gc.collect()

    try:
        # Setup (includes model construction / validation)
        func, args, kwargs = spec.setup(device)

        # Cold run (includes model download + load)
        ram_before = get_ram_mb()
        gpu_before = get_gpu_mb(device)
        t0 = time.perf_counter()

        func(*args, **kwargs)

        t1 = time.perf_counter()
        result.wall_seconds = round(t1 - t0, 2)
        result.peak_ram_mb = round(get_ram_mb() - ram_before, 1)
        result.peak_gpu_mb = round(peak_gpu_mb(device) - gpu_before, 1)

        # Warm run (model already in memory)
        if warmup:
            reset_gpu(device)
            t2 = time.perf_counter()
            func(*args, **kwargs)
            t3 = time.perf_counter()
            result.warm_seconds = round(t3 - t2, 2)

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    # Suggest tier based on CPU wall-clock
    if result.error is None and device == "cpu":
        if result.wall_seconds <= TIER_THRESHOLDS["tier1_max_cpu_seconds"]:
            result.suggested_tier = 1
        elif result.wall_seconds <= TIER_THRESHOLDS["tier2_max_cpu_seconds"]:
            result.suggested_tier = 2
        else:
            result.suggested_tier = 3

    return result


def get_available_devices() -> List[str]:
    """Return list of available devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def run_profiles(
    tiers: Optional[List[int]] = None,
    devices: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> List[ProfileResult]:
    """Run profiling across selected tiers and devices."""
    if devices is None:
        devices = get_available_devices()

    specs = MODEL_SPECS
    if tiers:
        specs = [s for s in specs if s.current_tier in tiers]
    if models:
        specs = [s for s in specs if any(m.lower() in s.model_id.lower() for m in models)]

    results: List[ProfileResult] = []
    total = len(specs) * len(devices)
    idx = 0

    for spec in specs:
        for device in devices:
            idx += 1
            print(f"\n[{idx}/{total}] {spec.model_id} ({spec.task}) on {device}...")
            result = profile_one(spec, device)
            results.append(result)

            status = "OK" if result.error is None else f"ERROR: {result.error}"
            if result.error is None:
                print(
                    f"  Cold: {result.wall_seconds}s | Warm: {result.warm_seconds}s | "
                    f"RAM: {result.peak_ram_mb}MB | GPU: {result.peak_gpu_mb}MB | {status}"
                )
                if result.suggested_tier and result.suggested_tier != spec.current_tier:
                    print(f"  ** TIER CHANGE: {spec.current_tier} -> {result.suggested_tier}")
            else:
                print(f"  {status}")

            # Force cleanup between models
            gc.collect()
            reset_gpu(device)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(results: List[ProfileResult]) -> Dict[str, Any]:
    """Generate a structured report from profiling results."""
    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "cpu_count": os.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        },
        "thresholds": TIER_THRESHOLDS,
        "results": [asdict(r) for r in results],
        "tier_changes": [],
        "warnings": [],
    }

    if torch.cuda.is_available():
        report["system"]["gpu_name"] = torch.cuda.get_device_name(0)
        report["system"]["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)

    # Detect tier changes and warnings
    for r in results:
        spec = next((s for s in MODEL_SPECS if s.model_id == r.model_id and s.task == r.task), None)
        if spec and r.suggested_tier and r.suggested_tier != spec.current_tier:
            report["tier_changes"].append(
                {
                    "model_id": r.model_id,
                    "task": r.task,
                    "current_tier": spec.current_tier,
                    "suggested_tier": r.suggested_tier,
                    "cpu_seconds": r.wall_seconds,
                }
            )
        if r.peak_ram_mb > TIER_THRESHOLDS["ram_warning_gb"] * 1024:
            report["warnings"].append(f"{r.model_id} ({r.device}): peak RAM {r.peak_ram_mb}MB exceeds threshold")
        if r.peak_gpu_mb > TIER_THRESHOLDS["gpu_warning_gb"] * 1024:
            report["warnings"].append(f"{r.model_id} ({r.device}): peak GPU {r.peak_gpu_mb}MB exceeds threshold")

    return report


def print_summary_table(results: List[ProfileResult]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 120)
    print(
        f"{'Model':<55} {'Task':<22} {'Device':<6} {'Cold(s)':<8} {'Warm(s)':<8} "
        f"{'RAM(MB)':<9} {'GPU(MB)':<9} {'Tier'}"
    )
    print("-" * 120)

    for r in results:
        if r.error:
            print(f"{r.model_id:<55} {r.task:<22} {r.device:<6} {'ERROR':>8}")
            continue

        tier_str = ""
        spec = next((s for s in MODEL_SPECS if s.model_id == r.model_id and s.task == r.task), None)
        if r.suggested_tier:
            if spec and r.suggested_tier != spec.current_tier:
                tier_str = f"{spec.current_tier}->{r.suggested_tier}"
            else:
                tier_str = str(r.suggested_tier)

        print(
            f"{r.model_id:<55} {r.task:<22} {r.device:<6} {r.wall_seconds:>7.1f} {r.warm_seconds:>7.1f} "
            f"{r.peak_ram_mb:>8.0f} {r.peak_gpu_mb:>8.0f} {tier_str:>5}"
        )

    print("=" * 120)


def update_audit_doc(report: Dict[str, Any]) -> None:
    """Update artifacts/test-device-audit.md with profiling results."""
    audit_path = Path(__file__).resolve().parent.parent / "artifacts" / "test-device-audit.md"
    if not audit_path.exists():
        print(f"Audit file not found: {audit_path}")
        return

    # Append profiling results section
    lines = [
        "\n\n## Profiling Results\n",
        f"\nGenerated: {report['timestamp']}\n",
        f"System: {report['system'].get('gpu_name', 'CPU only')}, "
        f"RAM: {report['system']['ram_total_gb']}GB, "
        f"torch {report['system']['torch']}\n",
        "\n| Model | Task | Device | Cold(s) | Warm(s) | RAM(MB) | GPU(MB) | Suggested Tier |\n",
        "|-------|------|--------|---------|---------|---------|---------|----------------|\n",
    ]

    for r in report["results"]:
        if r["error"]:
            lines.append(f"| `{r['model_id']}` | {r['task']} | {r['device']} | ERROR | | | | |\n")
        else:
            tier = r.get("suggested_tier", "")
            lines.append(
                f"| `{r['model_id']}` | {r['task']} | {r['device']} | "
                f"{r['wall_seconds']:.1f} | {r['warm_seconds']:.1f} | "
                f"{r['peak_ram_mb']:.0f} | {r['peak_gpu_mb']:.0f} | {tier} |\n"
            )

    if report["tier_changes"]:
        lines.append("\n### Suggested Tier Changes\n\n")
        for tc in report["tier_changes"]:
            lines.append(
                f"- **{tc['model_id']}** ({tc['task']}): "
                f"tier {tc['current_tier']} -> {tc['suggested_tier']} "
                f"(CPU: {tc['cpu_seconds']}s)\n"
            )

    if report["warnings"]:
        lines.append("\n### Warnings\n\n")
        for w in report["warnings"]:
            lines.append(f"- {w}\n")

    content = audit_path.read_text()
    # Remove old profiling section if present
    marker = "## Profiling Results"
    if marker in content:
        content = content[: content.index(marker)]

    content += "".join(lines)
    audit_path.write_text(content)
    print(f"\nUpdated: {audit_path}")


# ---------------------------------------------------------------------------
# pytest plugin (conftest integration)
# ---------------------------------------------------------------------------
# To use: add `conftest_plugins = ["scripts.profile_model_tiers"]` to conftest.py
# or run with: uv run pytest --profile-tiers


def pytest_addoption(parser):
    """Add --profile-tiers option to pytest."""
    parser.addoption(
        "--profile-tiers",
        action="store_true",
        default=False,
        help="Collect per-test timing and memory profiling data for tier classification.",
    )
    parser.addoption(
        "--profile-output",
        type=str,
        default="artifacts/profile-report.json",
        help="Output path for profiling JSON report (default: artifacts/profile-report.json).",
    )


def pytest_configure(config):
    """Register the profiling plugin if --profile-tiers is set."""
    if config.getoption("--profile-tiers", default=False):
        config.pluginmanager.register(ProfilePlugin(config), "profile_tiers_plugin")


class ProfilePlugin:
    """Pytest plugin that collects per-test timing and memory data."""

    def __init__(self, config):
        self.config = config
        self.results: List[Dict[str, Any]] = []

    def pytest_runtest_setup(self, item):
        """Record pre-test memory state."""
        gc.collect()
        item._profile_ram_before = get_ram_mb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        item._profile_gpu_before = get_gpu_mb("cuda") if torch.cuda.is_available() else 0.0
        item._profile_t0 = time.perf_counter()

    def pytest_runtest_teardown(self, item, nextitem):
        """Record post-test memory and timing."""
        t1 = time.perf_counter()
        ram_after = get_ram_mb()
        gpu_peak = peak_gpu_mb("cuda") if torch.cuda.is_available() else 0.0

        ram_before = getattr(item, "_profile_ram_before", ram_after)
        gpu_before = getattr(item, "_profile_gpu_before", 0.0)
        t0 = getattr(item, "_profile_t0", t1)

        self.results.append(
            {
                "test_id": item.nodeid,
                "wall_seconds": round(t1 - t0, 3),
                "ram_delta_mb": round(ram_after - ram_before, 1),
                "gpu_peak_mb": round(gpu_peak - gpu_before, 1),
            }
        )

    def pytest_sessionfinish(self, session, exitstatus):
        """Write profiling report at end of session."""
        output_path = self.config.getoption("--profile-output")
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "system": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cpu_count": os.cpu_count(),
                "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            },
            "tests": sorted(self.results, key=lambda r: -r["wall_seconds"]),
        }

        if torch.cuda.is_available():
            report["system"]["gpu_name"] = torch.cuda.get_device_name(0)
            report["system"]["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )

        # Summary stats
        times = [r["wall_seconds"] for r in self.results]
        report["summary"] = {
            "total_tests": len(self.results),
            "total_seconds": round(sum(times), 1),
            "slowest_test": self.results[0]["test_id"] if self.results else "",
            "slowest_seconds": times[0] if times else 0,
            "tests_over_30s": len([t for t in times if t > 30]),
            "tests_over_120s": len([t for t in times if t > 120]),
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(report, indent=2))
        print(f"\nProfile report written to: {output_path}")
        print(f"Total: {report['summary']['total_tests']} tests in {report['summary']['total_seconds']}s")
        print(f"Slowest: {report['summary']['slowest_test']} ({report['summary']['slowest_seconds']}s)")
        if report["summary"]["tests_over_30s"]:
            print(f"Tests >30s (tier 2+): {report['summary']['tests_over_30s']}")
        if report["summary"]["tests_over_120s"]:
            print(f"Tests >120s (tier 3): {report['summary']['tests_over_120s']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile senselab model inference for tier classification")
    parser.add_argument("--tiers", type=int, nargs="+", help="Only profile these tiers (1, 2, 3)")
    parser.add_argument("--device", type=str, help="Only profile on this device (cpu, cuda, mps)")
    parser.add_argument("--models", type=str, nargs="+", help="Filter by model name substring")
    parser.add_argument("--json", action="store_true", help="Output JSON only (no table)")
    parser.add_argument("--update-audit", action="store_true", help="Update artifacts/test-device-audit.md")
    parser.add_argument("--output", type=str, help="Write JSON report to this file")
    parser.add_argument(
        "--tier1-max",
        type=float,
        default=TIER_THRESHOLDS["tier1_max_cpu_seconds"],
        help=f"Tier 1 max CPU seconds (default: {TIER_THRESHOLDS['tier1_max_cpu_seconds']})",
    )
    parser.add_argument(
        "--tier2-max",
        type=float,
        default=TIER_THRESHOLDS["tier2_max_cpu_seconds"],
        help=f"Tier 2 max CPU seconds (default: {TIER_THRESHOLDS['tier2_max_cpu_seconds']})",
    )
    args = parser.parse_args()

    # Override thresholds
    TIER_THRESHOLDS["tier1_max_cpu_seconds"] = args.tier1_max
    TIER_THRESHOLDS["tier2_max_cpu_seconds"] = args.tier2_max

    devices = [args.device] if args.device else None

    print(f"Available devices: {get_available_devices()}")
    print(f"Profiling devices: {devices or get_available_devices()}")
    print(
        f"Thresholds: tier1 <= {TIER_THRESHOLDS['tier1_max_cpu_seconds']}s, "
        f"tier2 <= {TIER_THRESHOLDS['tier2_max_cpu_seconds']}s"
    )

    results = run_profiles(tiers=args.tiers, devices=devices, models=args.models)
    report = generate_report(results)

    if not args.json:
        print_summary_table(results)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nJSON report: {args.output}")
    elif args.json:
        print(json.dumps(report, indent=2))

    if args.update_audit:
        update_audit_doc(report)

    # Exit with error code if any models failed
    errors = [r for r in results if r.error and "not available" not in (r.error or "")]
    if errors:
        print(f"\n{len(errors)} model(s) failed profiling.")
        sys.exit(1)


if __name__ == "__main__":
    main()
