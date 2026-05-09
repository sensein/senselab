#!/usr/bin/env python3
r"""Analyze a single audio file with the full senselab task suite.

Resamples the input to 16 kHz mono, then runs each of:
    diarization, AST scene classification, YAMNet scene classification,
    multi-backend feature extraction (incl. torchaudio-squim quality
    metrics), ASR, and speaker embeddings.

Each task is run twice: once on the resampled-only audio, and once on
the same audio after speech enhancement. Tasks that have multiple
backends in ``model_registry.yaml`` (ASR, speaker embeddings,
diarization) accept a *list* of models and are run once per model.
Results are written as JSON (one file per variant per task per model)
under ``--output-dir``.

Available models per task (from ``src/senselab/model_registry.yaml``).
**Bold** entries are the defaults; the script runs every default (≥ 2 per
task where the registry offers more than one).

  diarization:
    **pyannote/speaker-diarization-community-1**   (PyannoteAudioModel)
    **nvidia/diar_sortformer_4spk-v1**             (HFModel, ≤ 4 speakers, NeMo)

  audio scene classification:
    **MIT/ast-finetuned-audioset-10-10-0.4593**    (AST, HF)
    **google/yamnet**                              (TF subprocess venv)

  speech_to_text (in defaults; mix of native-timestamp and post-aligned):
    **openai/whisper-large-v3-turbo**              (HFModel, 809M, multilingual; native timestamps)
    **ibm-granite/granite-speech-3.3-8b**          (~9B, EN + 7 translations; text-only, post-aligned by this script)
    **nvidia/canary-qwen-2.5b**                    (NeMo SALM subprocess venv, 2.5B; text-only, post-aligned)
    **Qwen/Qwen3-ASR-1.7B**                        (qwen-asr subprocess venv, 1.7B; native word timestamps via
                                                    Qwen3-ForcedAligner-0.6B companion)

  speech_to_text (additional, available via --asr-models):
    openai/whisper-large-v3                        (HFModel, 1.55B, multilingual; native timestamps)
    openai/whisper-small                           (HFModel, 244M; native timestamps)
    nvidia/stt_en_conformer_ctc_large              (NeMo subprocess venv, English-only; native CTC timestamps)

Auto-align stage: every ASR model in --asr-models that returns text-only
ScriptLines (no native timestamps and no chunks) is automatically passed
through the multilingual MMS forced-aligner (--aligner-model, default
facebook/mms-1b-all, 1100+ languages via per-language adapters) to add
per-segment timestamps. Pass --no-align-asr to skip this and emit a
single full-audio TextArea region for those models in the LS export.
The alignment cache is independent of the ASR cache (FR-024); changing
the aligner re-runs only alignment, not ASR.

  speaker_embeddings:
    **speechbrain/spkrec-ecapa-voxceleb**          (ECAPA-TDNN)
    **speechbrain/spkrec-resnet-voxceleb**         (ResNet-TDNN)
    speechbrain/spkrec-xvect-voxceleb              (X-Vector)

  speech_enhancement:
    **speechbrain/sepformer-wham16k-enhancement**  (16 kHz, default)
    speechbrain/sepformer-whamr-enhancement        (8 kHz, with reverb)

Scene-classification grid: AST and YAMNet each use their own native
temporal precision; the wrapper does *not* impose a common grid because
AST cannot operate on chunks much shorter than its 10.24 s native input
(its internal kaldi-fbank rejects them).

  AST    → ``--ast-win-length 10.24 --ast-hop-length 10.24`` (no overlap;
           matches AST's intrinsic 1024-frame mel-spectrogram input).
  YAMNet → ``--yamnet-win-length 0.96 --yamnet-hop-length 0.48`` (matches
           YAMNet's native log-mel frame and 50% overlap hop).

Each model's output JSON records its own ``window`` block, and the
hierarchical Label Studio export emits each model's regions on its own
timeline track at its own native rate. To force the two onto a shared
grid for direct comparison, pass matching ``--ast-*`` and ``--yamnet-*``
values; when the grids match, an extra ``scene_agreement.json`` is
written that pairs top-1 labels per window and reports an agreement rate.

Diarization and ASR timestamps come straight from each model and are
preserved at their native resolution (Pyannote ≈ 62.5 ms, NeMo Sortformer
≈ 80 ms, Whisper word-level ≈ 20 ms).

Cache + provenance: every per-task outcome is stored under
``--cache-dir`` (default ``artifacts/analyze_audio_cache/``) keyed by

    sha256(audio_signature || task || model_id || params ||
           wrapper_version_hash || senselab_version || cache_schema_version)

The audio signature is the sha256 of the post-resample, post-downmix
PCM samples + sampling rate, so two files with identical waveforms
share cache entries regardless of container or filename. On cache hit
the prior outcome is replayed verbatim and ``cache: "hit"`` is set in
that task's output JSON; on miss the task runs and ``cache: "miss"`` is
recorded along with a full ``provenance`` block (audio_source,
audio_signature, model_id, params, device, wrapper hash, senselab
version, timestamp). Pass ``--no-cache`` to disable both lookup and
store. Bump ``_CACHE_SCHEMA_VERSION`` in this script when output shape
changes in a way that should invalidate prior entries.

Install:
    uv sync --extra articulatory --extra text --extra video --group dev

Usage:
    uv run python scripts/analyze_audio.py path/to/audio.wav

    # Compare multiple ASR models on the same audio
    uv run python scripts/analyze_audio.py audio.wav \\
        --asr-models openai/whisper-large-v3-turbo openai/whisper-small

    # Run all three SpeechBrain speaker-embedding variants
    uv run python scripts/analyze_audio.py audio.wav \\
        --embeddings-models speechbrain/spkrec-ecapa-voxceleb \\
                            speechbrain/spkrec-resnet-voxceleb \\
                            speechbrain/spkrec-xvect-voxceleb
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification import classify_audios
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.forced_alignment import align_transcriptions
from senselab.audio.tasks.input_output import read_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, extract_segments, resample_audios
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    Language,
    PyannoteAudioModel,
    ScriptLine,
    SpeechBrainModel,
)

TARGET_SR = 16000
ALL_TASKS = ("diarization", "ast", "yamnet", "features", "asr", "embeddings", "alignment", "comparisons")
COMPARISON_AXES = ("raw_vs_enhanced", "within_stream", "cross_stream")
UNCERTAINTY_AGGREGATORS = ("min", "mean", "harmonic_mean", "disagreement_weighted")
DEFAULT_SPEECH_PRESENCE_LABELS = (
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Child speech, kid speaking",
    "Speech synthesizer",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to the input audio file (.wav, .flac, .mp3, ...)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analyze_audio"),
        help="Directory for JSON outputs (default: artifacts/analyze_audio/)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="auto",
        help="Compute device (default: auto-pick best available)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_TASKS,
        default=(),
        help="Tasks to skip (default: run all)",
    )
    parser.add_argument(
        "--no-enhancement",
        action="store_true",
        help="Skip the enhanced-audio pass; only run on the resampled original.",
    )
    parser.add_argument(
        "--diarization-models",
        nargs="+",
        default=[
            "pyannote/speaker-diarization-community-1",
            "nvidia/diar_sortformer_4spk-v1",
        ],
        help=(
            "Diarization models. Default runs both Pyannote and NeMo Sortformer. "
            "Pyannote ids → PyannoteAudioModel; NeMo ids (nvidia/diar_sortformer*) → HFModel."
        ),
    )
    parser.add_argument("--ast-model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--yamnet-model", default="google/yamnet")
    parser.add_argument(
        "--asr-models",
        nargs="+",
        default=[
            # Confirmed working through senselab today:
            "openai/whisper-large-v3-turbo",
            # Routes through the existing HF pipeline path with
            # return_timestamps=False (timestamp-less HF model known-list);
            # the script's auto-align stage adds per-segment timestamps via MMS:
            "ibm-granite/granite-speech-3.3-8b",
            # Both routed through dedicated subprocess venvs. Per-model
            # failures are captured in JSON without aborting the run.
            "nvidia/canary-qwen-2.5b",
            "Qwen/Qwen3-ASR-1.7B",
        ],
        help=(
            "ASR models. Defaults: Whisper Large v3 Turbo (native timestamps), "
            "IBM Granite Speech 3.3 8B (text-only via HF pipeline; auto-aligned "
            "by this script), NVIDIA Canary-Qwen 2.5B (text-only, auto-aligned), "
            "and Qwen3-ASR 1.7B (native word-level timestamps via the bundled "
            "forced-aligner companion). The script auto-aligns timestamp-less "
            "ASR output via the multilingual aligner; pass --no-align-asr to skip."
        ),
    )
    parser.add_argument(
        "--embeddings-models",
        nargs="+",
        default=[
            "speechbrain/spkrec-ecapa-voxceleb",
            "speechbrain/spkrec-resnet-voxceleb",
        ],
        help="SpeechBrain speaker-embedding models. Default runs ECAPA-TDNN + ResNet-TDNN.",
    )
    parser.add_argument(
        "--enhancement-model",
        default="speechbrain/sepformer-wham16k-enhancement",
        help="Speech-enhancement model. Default is the 16 kHz SepFormer variant.",
    )
    # Scene-classification windowing. AST and YAMNet each use their own native
    # frame to preserve their intended temporal precision in the output:
    #   - AST native input: 1024 mel frames at 10 ms hop = 10.24 s. AST's
    #     internal kaldi-fbank refuses chunks shorter than ~1 s of audio, so
    #     anything well below 10 s also degrades scientifically. Default to
    #     10.24 s with no overlap.
    #   - YAMNet native: 0.96 s log-mel frame, 0.48 s hop (50% overlap),
    #     per Google's YAMNet model card.
    # Override per model when you want to trade off resolution vs. cost; pass
    # matching --ast-* and --yamnet-* values to force a common grid (and
    # enable the optional scene_agreement.json comparison output).
    parser.add_argument(
        "--ast-win-length",
        type=float,
        default=10.24,
        help="AST sliding-window length, seconds (default: 10.24, AST's native input duration).",
    )
    parser.add_argument(
        "--ast-hop-length",
        type=float,
        default=10.24,
        help="AST sliding-window hop, seconds (default: 10.24, no overlap; equals win-length).",
    )
    parser.add_argument(
        "--yamnet-win-length",
        type=float,
        default=0.96,
        help="YAMNet sliding-window length, seconds (default: 0.96, matches YAMNet's native frame).",
    )
    parser.add_argument(
        "--yamnet-hop-length",
        type=float,
        default=0.48,
        help="YAMNet sliding-window hop, seconds (default: 0.48, matches YAMNet's native 50%% overlap hop).",
    )
    parser.add_argument(
        "--features-win-length",
        type=float,
        default=1.0,
        help=(
            "Sliding-window length for feature extraction, in seconds (default: 1.0). "
            "OpenSMILE/Parselmouth/torchaudio-squim are summary statistics by design — "
            "we re-run them per window so the resulting time series is comparable to "
            "the rest of the analysis (diarization, AST, YAMNet, ASR)."
        ),
    )
    parser.add_argument(
        "--features-hop-length",
        type=float,
        default=0.5,
        help="Hop between feature windows, in seconds (default: 0.5; 50%% overlap with the default 1.0s window).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/analyze_audio_cache"),
        help=(
            "Directory for the content-addressable result cache. Cache keys are "
            "sha256(audio_signature, task, model_id, params, wrapper_hash, "
            "senselab_version). Identical inputs replay prior outputs without "
            "re-running models. Default: artifacts/analyze_audio_cache/."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookup AND store. Every task runs fresh; nothing is written to the cache.",
    )
    # Auto-align controls. Auto-align is on by default: any ASR result without
    # native timestamps gets post-processed through senselab.audio.tasks.forced_alignment
    # before the LS export, so the LS bundle has region-level annotations on the
    # timeline regardless of whether the ASR produced timestamps natively.
    parser.add_argument(
        "--no-align-asr",
        action="store_true",
        help=(
            "Disable the auto-align stage for timestamp-less ASR. Outputs become "
            "text-only ScriptLines; the LS export emits a single full-audio TextArea "
            "region for each timestamp-less ASR model."
        ),
    )
    parser.add_argument(
        "--aligner-model",
        default="facebook/mms-1b-all",
        help=(
            "Multilingual forced-alignment model used by the auto-align stage "
            "(default: facebook/mms-1b-all, covers 1100+ languages via per-language "
            "adapters). Override to swap MMS for another senselab-supported aligner."
        ),
    )
    parser.add_argument(
        "--asr-language",
        default=None,
        help=(
            "Force a specific language for the auto-align stage (ISO 639-1 like 'en', "
            "'ja' or ISO 639-3 like 'eng', 'jpn'). When omitted, the script falls back "
            "to the ASR model's documented default language (typically English)."
        ),
    )
    parser.add_argument(
        "--qwen-asr-no-timestamps",
        action="store_true",
        help=(
            "Skip Qwen3-ASR's bundled forced-aligner companion model. The ASR returns "
            "text-only ScriptLines; the script's own auto-align stage then takes over "
            "(unless --no-align-asr is also set)."
        ),
    )
    # ── Comparison & uncertainty stage flags ───────────────────────────
    parser.add_argument(
        "--skip-comparisons",
        nargs="+",
        choices=COMPARISON_AXES,
        default=(),
        help="Skip individual comparison axes. Pass --skip comparisons to skip everything new.",
    )
    parser.add_argument(
        "--cross-stream-win-length",
        type=float,
        default=0.2,
        help="Window length (seconds) for cross-stream comparisons (ASR↔diar, scene↔diar, ASR↔PPG).",
    )
    parser.add_argument(
        "--cross-stream-hop-length",
        type=float,
        default=0.1,
        help="Hop between cross-stream comparison windows (default 0.1 s; must be <= win-length).",
    )
    parser.add_argument(
        "--uncertainty-aggregator",
        choices=UNCERTAINTY_AGGREGATORS,
        default="min",
        help="Aggregator that combines per-model confidences for the disagreements.json ranking.",
    )
    parser.add_argument(
        "--phoneme-disagreement-threshold",
        type=float,
        default=0.50,
        help="Phoneme-error-rate threshold for ASR↔PPG `phoneme_disagreement` flag (default 0.50).",
    )
    parser.add_argument(
        "--speech-presence-labels",
        type=str,
        default=",".join(DEFAULT_SPEECH_PRESENCE_LABELS),
        help=(
            "Comma-separated AudioSet labels that count as 'speech-present' for AST/YAMNet ↔ "
            "diarization comparison. Default covers the AudioSet 'Speech' subtree."
        ),
    )
    parser.add_argument(
        "--asr-reference-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Which ASR model is the soft reference for ASR-vs-ASR WER computation.",
    )
    parser.add_argument(
        "--diarization-boundary-shift-ms",
        type=float,
        default=50.0,
        help=(
            "Boundary-shift threshold (ms) for diarization disagreement detection. "
            "Per Constitution §VIII (No Hardcoded Parameters)."
        ),
    )
    parser.add_argument(
        "--disagreements-top-n",
        type=int,
        default=100,
        help="Top-N rows to emit in disagreements.json (default 100; 0 disables the index).",
    )
    args = parser.parse_args(argv)
    # Comparator flag validation (cli.md "Validation").
    if args.cross_stream_win_length <= 0:
        parser.error("--cross-stream-win-length must be positive")
    if args.cross_stream_hop_length <= 0 or args.cross_stream_hop_length > args.cross_stream_win_length:
        parser.error("--cross-stream-hop-length must be positive and ≤ --cross-stream-win-length")
    if not (0.0 <= args.phoneme_disagreement_threshold <= 1.0):
        parser.error("--phoneme-disagreement-threshold must be in [0, 1]")
    if args.diarization_boundary_shift_ms < 0:
        parser.error("--diarization-boundary-shift-ms must be non-negative")
    if args.disagreements_top_n < 0:
        parser.error("--disagreements-top-n must be non-negative")
    return args


def pick_dispatch_model(model_id: str, *, task: str) -> Any:  # noqa: ANN401
    """Wrap a model id in the right SenselabModel subclass for the given task.

    Routes diarization model ids to PyannoteAudioModel or HFModel based on the
    well-known prefix, matching ``diarize_audios``'s internal dispatch logic.
    """
    if task == "diarization":
        if model_id.startswith("nvidia/diar_sortformer"):
            return HFModel(path_or_uri=model_id)
        return PyannoteAudioModel(path_or_uri=model_id)
    if task == "asr":
        return HFModel(path_or_uri=model_id)
    if task == "embeddings":
        return SpeechBrainModel(path_or_uri=model_id)
    if task == "enhancement":
        return SpeechBrainModel(path_or_uri=model_id)
    raise ValueError(f"unknown task: {task}")


def pick_device(arg: str) -> DeviceType | None:
    """Resolve --device into a senselab DeviceType, or None for per-task auto.

    When the user passes ``--device auto`` we return ``None`` so each senselab
    task can pick its own compatible device (e.g., pyannote and AST reject MPS,
    so they need to fall back to CPU; Whisper and SepFormer can use MPS). When
    the user explicitly names a device we honor that and let the task error if
    it's incompatible (caller can ``--device cpu`` to be safe everywhere).
    """
    if arg == "cuda":
        return DeviceType.CUDA
    if arg == "mps":
        return DeviceType.MPS
    if arg == "cpu":
        return DeviceType.CPU
    return None


def prepare_audio(path: Path) -> Audio:
    """Read audio, downmix to mono, resample to 16 kHz."""
    audio = read_audios([str(path)])[0]
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != TARGET_SR:
        audio = resample_audios([audio], resample_rate=TARGET_SR)[0]
    return audio


# -- Cache + provenance ----------------------------------------------------

# Cache schema bumps when the cache key composition or output shape changes
# in a way that should invalidate prior cache entries even when the audio,
# task, model, and senselab version are unchanged. Bump on breaking changes.
#
# v2: introduces alignment as a separate cache entry, separate from the
#     ASR entry that produced its input transcript. The ASR cache key
#     covers the ASR call's parameters; the alignment cache key adds
#     transcript_sha and language. v1 cache entries are inert (won't be
#     served) and can be discarded.
_CACHE_SCHEMA_VERSION = 3


def audio_signature(audio: Audio) -> str:
    """Return a deterministic sha256 of the audio waveform PCM + sampling rate.

    Two identical-sounding files produce the same signature regardless of
    their on-disk format (e.g., WAV vs FLAC) — what matters is the post-
    resample, post-downmix waveform that each task actually sees. Extra
    metadata (file path, mtime, encoding) is intentionally excluded.
    """
    arr = audio.waveform.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(audio.sampling_rate).encode())
    h.update(b"|")
    h.update(str(arr.shape).encode())
    h.update(b"|")
    h.update(arr.tobytes())
    return h.hexdigest()


def wrapper_version_hash() -> str:
    """sha256 of this script's source. Captures wrapper-side behavior changes.

    When the cache hits, we replay outputs from a prior wrapper version. If
    the wrapper logic changed (e.g., we now post-process results differently)
    we want a cache miss. Hashing the script's bytes keys the cache to the
    exact wrapper that produced each entry.
    """
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unknown"


def senselab_version() -> str:
    """Return the installed senselab version, or 'unknown' if metadata is missing."""
    try:
        return importlib.metadata.version("senselab")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _canonical_params(params: dict[str, Any]) -> str:
    """Stable JSON encoding of params for cache keying. Sorted, no whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    *,
    audio_sig: str,
    task: str,
    model_id: str | None,
    params: dict[str, Any],
    wrapper_hash: str,
    senselab_ver: str,
) -> str:
    """Compute the deterministic cache key for one (audio, task, model, params) combo."""
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "audio_signature": audio_sig,
        "task": task,
        "model": model_id,
        "params": params,
        "wrapper_hash": wrapper_hash,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(_canonical_params(payload).encode()).hexdigest()


def cache_lookup(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return the cached result dict for ``key``, or None on miss."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_store(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Persist ``payload`` for ``key`` under the cache dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(serialize(payload), indent=2, default=str), encoding="utf-8")


def transcript_signature(text: str) -> str:
    """sha256 of an ASR transcript text — anchors an alignment outcome to its exact input.

    The alignment cache uses this as one of its keys: re-aligning the same
    transcript on the same audio with the same params returns the cached
    timestamps without re-loading the aligner model.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Comparison & uncertainty stage helpers ────────────────────────────


def comparison_cache_key(
    *,
    audio_sig: str,
    comparison_kind: str,
    task_or_pair: str,
    upstream_cache_keys: list[str],
    params: dict[str, Any],
    wrapper_hash: str,
    senselab_ver: str,
) -> str:
    """Cache key for one comparator output.

    Includes the sorted tuple of upstream-task cache_keys so that re-running
    an upstream task automatically invalidates downstream comparisons that
    depended on it (research.md §6).
    """
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "audio_signature": audio_sig,
        "comparison_kind": comparison_kind,
        "task_or_pair": task_or_pair,
        "upstream_cache_keys": sorted(upstream_cache_keys),
        "params": _canonical_params(params),
        "wrapper_hash": wrapper_hash,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(_canonical_params(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparisonGrid:
    """Shared time grid that comparators project onto before differencing."""

    win_length: float
    hop_length: float
    name: str  # "features" or "cross_stream"

    def iter_buckets(self, duration_s: float) -> Iterator[tuple[float, float, int]]:
        """Yield (start, end, idx) tuples covering [0, duration_s] at this grid."""
        eps = 1e-6
        t = 0.0
        idx = 0
        while t + self.win_length <= duration_s + eps:
            start = round(t, 4)
            end = round(min(t + self.win_length, duration_s), 4)
            yield start, end, idx
            t += self.hop_length
            idx += 1


# Common parquet columns shared across every comparison parquet (data-model.md).
COMPARISON_COMMON_COLS = (
    "start",
    "end",
    "comparison_kind",
    "task",
    "stream_pair",
    "model_a",
    "model_b",
    "pass_a",
    "pass_b",
    "agree",
    "mismatch_type",
    "comparison_status",
    "confidence_a",
    "confidence_b",
    "combined_uncertainty",
)
ASR_VS_ASR_EXTRA_COLS = ("wer", "a_text", "b_text", "reference_side")
CLASSIFICATION_VS_CLASSIFICATION_EXTRA_COLS = (
    "top1_a",
    "top1_b",
    "score_a",
    "score_b",
    "entropy_a",
    "entropy_b",
)
DIAR_VS_DIAR_EXTRA_COLS = ("speaks_a", "speaks_b")
ASR_VS_DIAR_EXTRA_COLS = ("asr_says_speech", "diar_says_speech")
CLASS_VS_DIAR_EXTRA_COLS = ("class_says_speech", "diar_says_speech", "top1_label")
ASR_VS_PPG_EXTRA_COLS = ("asr_phonemes", "ppg_phonemes", "phoneme_per", "phoneme_disagreement")


def write_comparison_parquet(
    rows: list[dict[str, Any]],
    dest_path: Path,
    provenance: dict[str, Any],
) -> None:
    """Write comparison rows to parquet with `comparator_provenance` metadata.

    No-op when ``rows`` is empty — comparators emit no parquet rather than
    an empty file (callers rely on absence to skip LS-track declaration).
    """
    if not rows:
        return
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = {b"comparator_provenance": json.dumps(provenance, sort_keys=True, default=str).encode("utf-8")}
    if table.schema.metadata:
        merged = dict(table.schema.metadata)
        merged.update(metadata)
        table = table.replace_schema_metadata(merged)
    else:
        table = table.replace_schema_metadata(metadata)
    pq.write_table(table, dest_path)


def _aggregate_uncertainty(
    confidences: list[float | None],
    aggregator: str,
    *,
    mismatch_severity: float = 1.0,
) -> float | None:
    """Combine per-model confidences into a single sortable score in [0, 1].

    Returns None when no confidences are available. Drops None entries
    rather than treating them as zero (FR-003, FR-007).
    """
    vals = [c for c in confidences if c is not None]
    if not vals:
        return None
    if aggregator == "min":
        agg = min(vals)
    elif aggregator == "mean":
        agg = sum(vals) / len(vals)
    elif aggregator == "harmonic_mean":
        # Harmonic mean is undefined when any value is 0; clip to small eps.
        clipped = [max(v, 1e-9) for v in vals]
        agg = len(clipped) / sum(1.0 / v for v in clipped)
    elif aggregator == "disagreement_weighted":
        mean = sum(vals) / len(vals)
        return float((1.0 - mean) * mismatch_severity)
    else:
        raise ValueError(f"unknown aggregator: {aggregator!r}")
    # combined_uncertainty = 1 - confidence (so high uncertainty = sort first).
    return float(1.0 - agg)


def _token_overlaps_window(result: Any, win_start: float, win_end: float) -> bool:  # noqa: ANN401
    """Return True if any transcript chunk's timestamp overlaps the window.

    Tolerant to both Pydantic ScriptLine objects (in-memory) and JSON-restored
    dicts (post-cache-hit). Walks ``result.chunks``; falls back to the
    top-level start/end when chunks are absent.
    """
    if not result:
        return False
    items = result if isinstance(result, list) else [result]
    for line in items:
        chunks = _seg_attr(line, "chunks") or []
        if chunks:
            for c in chunks:
                cs = _seg_attr(c, "start")
                ce = _seg_attr(c, "end")
                if cs is None or ce is None:
                    continue
                if float(cs) < win_end and float(ce) > win_start:
                    return True
        else:
            ls = _seg_attr(line, "start")
            le = _seg_attr(line, "end")
            if ls is not None and le is not None and float(ls) < win_end and float(le) > win_start:
                return True
    return False


def _disagreement_severity_rank(mismatch_type: str | None) -> int:
    """Tie-breaker priority for disagreements.json ordering (lower = higher priority)."""
    return {
        "phoneme_disagreement": 0,
        "text_edit": 1,
        "speech_presence_flip": 2,
        "label_flip": 3,
        "boundary_shift": 4,
    }.get(mismatch_type or "", 5)


def _build_disagreements_index(
    parquet_paths: list[Path],
    top_n: int,
    aggregator: str,
    *,
    run_dir: Path,
    config: dict[str, Any],
    incomparable_reasons: dict[str, str],
    missing_confidence_signals: list[str],
) -> dict[str, Any]:
    """Aggregate every comparison parquet into the top-level disagreements index."""
    import datetime
    import math

    import pandas as pd

    entries: list[dict[str, Any]] = []
    total_rows = 0
    rows_by_kind: dict[str, int] = {}
    disagreement_rows = 0
    if top_n <= 0:
        # Index disabled — still report totals at zero entries.
        return {
            "schema_version": 1,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "wrapper_hash": wrapper_version_hash(),
            "senselab_version": senselab_version(),
            "config": config,
            "missing_confidence_signals": {
                "models_without_native_signal": missing_confidence_signals,
                "note": "Aggregation drops models lacking a native confidence rather than treating as zero.",
            },
            "incomparable_reasons": incomparable_reasons,
            "totals": {
                "total_rows": 0,
                "rows_by_kind": {},
                "disagreement_rows": 0,
                "disagreement_rate": 0.0,
            },
            "entries": [],
        }
    for p in parquet_paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        kind = str(df["comparison_kind"].iloc[0]) if "comparison_kind" in df.columns else "unknown"
        total_rows += len(df)
        rows_by_kind[kind] = rows_by_kind.get(kind, 0) + len(df)
        if "agree" in df.columns:
            disagreement_rows += int((df["agree"] == False).sum())  # noqa: E712
        for idx, row in df.iterrows():
            if row.get("agree", True) is True:
                continue
            cu = row.get("combined_uncertainty")
            cu_val = float(cu) if cu is not None and not (isinstance(cu, float) and math.isnan(cu)) else float("nan")
            entries.append(
                {
                    "rank": 0,  # filled after sort
                    "region_id": _comparison_region_id(p, idx, row),
                    "pass": row.get("pass_a", row.get("pass_b", "")),
                    "comparison_kind": kind,
                    "task": row.get("task"),
                    "stream_pair": row.get("stream_pair"),
                    "model_a": row.get("model_a"),
                    "model_b": row.get("model_b"),
                    "parquet_path": str(p.relative_to(run_dir)) if p.is_relative_to(run_dir) else str(p),
                    "row_index": int(idx),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "mismatch_type": row.get("mismatch_type"),
                    "combined_uncertainty": cu_val,
                    "ls_track_name": _comparison_ls_track_name(row),
                    "summary": _comparison_row_summary(row),
                }
            )

    def _sort_key(e: dict[str, Any]) -> tuple[Any, ...]:
        cu = e["combined_uncertainty"]
        # NaN last → flip sign so high uncertainty comes first; NaN sorts to +inf.
        primary = -cu if cu == cu else float("inf")
        return (primary, _disagreement_severity_rank(e["mismatch_type"]), e["start"], e["region_id"])

    entries.sort(key=_sort_key)
    entries = entries[:top_n]
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "wrapper_hash": wrapper_version_hash(),
        "senselab_version": senselab_version(),
        "config": config,
        "missing_confidence_signals": {
            "models_without_native_signal": missing_confidence_signals,
            "note": "Aggregation drops models lacking a native confidence rather than treating as zero.",
        },
        "incomparable_reasons": incomparable_reasons,
        "totals": {
            "total_rows": total_rows,
            "rows_by_kind": rows_by_kind,
            "disagreement_rows": disagreement_rows,
            "disagreement_rate": (disagreement_rows / total_rows) if total_rows else 0.0,
        },
        "entries": entries,
    }


def _comparison_region_id(parquet_path: Path, row_idx: int, row: dict[str, Any] | Any) -> str:  # noqa: ANN401
    """Deterministic region id linking a comparison row to its LS region."""
    pass_label = row.get("pass_a") or row.get("pass_b") or ""
    pair = row.get("model_a") or row.get("stream_pair") or ""
    pair2 = row.get("model_b") or ""
    task = row.get("task") or row.get("stream_pair") or "compare"
    suffix = f"{_safe(str(pair))}_vs_{_safe(str(pair2))}" if pair2 else _safe(str(pair))
    return f"{pass_label}__compare__{_safe(str(task))}__{suffix}__{row_idx:04d}"


def _comparison_ls_track_name(row: dict[str, Any] | Any) -> str:  # noqa: ANN401
    """Track name for LS Labels region (spec FR-004)."""
    pass_label = row.get("pass_a") or row.get("pass_b") or "pass"
    task = row.get("task") or row.get("stream_pair") or "compare"
    a = row.get("model_a") or ""
    b = row.get("model_b") or ""
    if a and b:
        return f"{pass_label}__compare__{_safe(str(task))}__{_safe(str(a))}_vs_{_safe(str(b))}"
    return f"{pass_label}__compare__{_safe(str(task))}__{_safe(str(a or b))}"


def _comparison_row_summary(row: dict[str, Any] | Any) -> str:  # noqa: ANN401
    """One-line human summary for the disagreements index entry."""
    mt = row.get("mismatch_type") or "disagree"
    if "wer" in row and row.get("wer") is not None:
        a = (row.get("a_text") or "")[:30]
        b = (row.get("b_text") or "")[:30]
        return f"{mt} WER {float(row['wer']):.2f}: {a!r} vs {b!r}"
    if "phoneme_per" in row and row.get("phoneme_per") is not None:
        return f"{mt} PER {float(row['phoneme_per']):.2f}"
    if "top1_a" in row and row.get("top1_a") is not None:
        return f"{mt}: {row.get('top1_a')!r} vs {row.get('top1_b')!r}"
    if "speaks_a" in row:
        return f"{mt}: speaks_a={row.get('speaks_a')} vs speaks_b={row.get('speaks_b')}"
    return mt


def _comparison_ls_label_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    label: str,
) -> dict[str, Any]:
    """LS Labels region for a comparator track. Wraps `_ls_label_region` for symmetry."""
    return _ls_label_region(region_id=region_id, from_name=from_name, start=start, end=end, label=label)


def _comparison_ls_textarea_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    text: str,
) -> dict[str, Any]:
    """LS TextArea region for ASR-vs-ASR comparator pairs."""
    return _ls_textarea_region(region_id=region_id, from_name=from_name, start=start, end=end, text=text)


# ── Per-task differencers (US1: raw_vs_enhanced; reused by US2/US3) ────


def _diar_speaks_in_window(result: Any, win_start: float, win_end: float) -> bool:  # noqa: ANN401
    """Return True if any diarization segment overlaps the window.

    ``result`` may be ``List[List[ScriptLine]]`` or the same shape after a
    cache-restored dict round-trip. We unwrap the outer list and then test
    each segment's [start, end) against the window.
    """
    if not result:
        return False
    segments = result[0] if isinstance(result, list) and result else []
    for seg in segments:
        s = _seg_attr(seg, "start")
        e = _seg_attr(seg, "end")
        if s is None or e is None:
            continue
        if float(s) < win_end and float(e) > win_start:
            return True
    return False


def _classification_top1_in_window(result: Any, win_idx: int) -> tuple[str | None, float | None, float | None]:  # noqa: ANN401
    """Return (top1_label, top1_score, entropy) for the win_idx'th classification window.

    Classification results are List[List[List[{label, score}]]] — outer audio,
    middle window, inner top-K. We pick the requested window and compute the
    top-1 + Shannon entropy over the inner distribution.
    """
    import math

    if not result:
        return None, None, None
    windows = result[0] if isinstance(result, list) and result else []
    if win_idx >= len(windows):
        return None, None, None
    window = windows[win_idx]
    if not isinstance(window, list) or not window:
        return None, None, None
    top = max(window, key=lambda d: d.get("score", 0.0) if isinstance(d, dict) else 0.0)
    label = str(top.get("label", "?")) if isinstance(top, dict) else None
    score = float(top.get("score", 0.0)) if isinstance(top, dict) else None
    # Entropy over the full distribution (clip non-positive scores).
    probs = [max(float(d.get("score", 0.0)), 1e-12) for d in window if isinstance(d, dict)]
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs)
    return label, score, entropy


def _asr_text_in_window(result: Any, win_start: float, win_end: float) -> str:  # noqa: ANN401
    """Return concatenated transcript tokens overlapping the window.

    Tokens (chunks) are preferred when present; fall back to the full
    ScriptLine text when chunks are absent and the line's [start, end]
    overlaps the window.
    """
    if not result:
        return ""
    items = result if isinstance(result, list) else [result]
    pieces: list[str] = []
    for line in items:
        chunks = _seg_attr(line, "chunks") or []
        if chunks:
            for c in chunks:
                cs = _seg_attr(c, "start")
                ce = _seg_attr(c, "end")
                if cs is None or ce is None:
                    continue
                if float(cs) < win_end and float(ce) > win_start:
                    text = _seg_attr(c, "text") or ""
                    if text:
                        pieces.append(str(text).strip())
        else:
            ls = _seg_attr(line, "start")
            le = _seg_attr(line, "end")
            text = _seg_attr(line, "text") or ""
            if not text:
                continue
            if ls is None or le is None or (float(ls) < win_end and float(le) > win_start):
                pieces.append(str(text).strip())
    return " ".join(p for p in pieces if p).strip()


def _wer_per_window(text_a: str, text_b: str) -> float:
    """Word error rate (clipped to [0, 1]) between two token strings.

    Edge cases:
    - both empty → 0.0 (no signal, no disagreement)
    - one empty → 1.0 (insertion / deletion)
    - both populated → jiwer.wer; clipped at 1.0 for sortability.
    """
    a, b = (text_a or "").strip(), (text_b or "").strip()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    import jiwer  # local import — keeps cold-start fast

    return float(min(jiwer.wer(a, b), 1.0))


def _bool_pair_disagrees(a: bool | None, b: bool | None) -> bool:
    """True only when both sides are non-null and differ."""
    if a is None or b is None:
        return False
    return a != b


def _diff_diarization(
    result_a: Any,  # noqa: ANN401
    result_b: Any,  # noqa: ANN401
    grid: ComparisonGrid,
    duration_s: float,
    *,
    boundary_shift_ms: float,  # noqa: ARG001 — reserved for future per-segment boundary diff
    **base_row: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Compare two diarization outputs on ``grid``: speaks_a vs speaks_b per bucket."""
    rows: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        sa = _diar_speaks_in_window(result_a, start, end)
        sb = _diar_speaks_in_window(result_b, start, end)
        agree = sa == sb
        rows.append(
            {
                **base_row,
                "start": start,
                "end": end,
                "agree": agree,
                "mismatch_type": None if agree else "boundary_shift",
                "comparison_status": "ok",
                "speaks_a": sa,
                "speaks_b": sb,
            }
        )
    return rows


def _diff_classification(
    result_a: Any,  # noqa: ANN401
    result_b: Any,  # noqa: ANN401
    grid: ComparisonGrid,  # noqa: ARG001 — classification windowing follows result shape, not external grid
    duration_s: float,  # noqa: ARG001 — same reason
    *,
    win_length_a: float,  # noqa: ARG001 — reserved for grid-mismatch projection
    win_length_b: float,  # noqa: ARG001 — same
    **base_row: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Compare AST/YAMNet outputs window-by-window when both ran on the same grid.

    When the two outputs have different window counts (e.g. AST 10.24 s vs
    YAMNet 0.96 s), only the overlapping prefix is compared. A future
    refinement would project both onto a shared grid; for v1 we trust the
    per-model native grid.
    """
    rows: list[dict[str, Any]] = []
    windows_a = result_a[0] if isinstance(result_a, list) and result_a else []
    windows_b = result_b[0] if isinstance(result_b, list) and result_b else []
    n = min(len(windows_a), len(windows_b))
    for idx in range(n):
        ta, sa, ea = _classification_top1_in_window(result_a, idx)
        tb, sb, eb = _classification_top1_in_window(result_b, idx)
        # Use the smaller of the two native window lengths for the row span;
        # this is approximate but adequate for the timeline.
        # (We just use the AST grid for this row's start/end; callers needing
        # exact alignment should use cross-stream comparators.)
        # Sentinel: idx-based span using the side-A grid (caller's
        # responsibility to ensure both sides ran on the same grid for
        # within_stream and raw_vs_enhanced — single-model contracts).
        rows.append(
            {
                **base_row,
                "start": float(idx),  # placeholder — caller can override via post-processing
                "end": float(idx + 1),  # idem
                "agree": ta == tb,
                "mismatch_type": None if ta == tb else "label_flip",
                "comparison_status": "ok",
                "top1_a": ta,
                "top1_b": tb,
                "score_a": sa,
                "score_b": sb,
                "entropy_a": ea,
                "entropy_b": eb,
            }
        )
    return rows


def _diff_asr(
    result_a: Any,  # noqa: ANN401
    result_b: Any,  # noqa: ANN401
    grid: ComparisonGrid,
    duration_s: float,
    *,
    reference_side: str,  # "a" or "b"
    **base_row: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Per-window WER between two ASR outputs on ``grid``.

    Empty buckets (no speech in either result) are dropped — we only
    emit rows where at least one side had text overlap.
    """
    rows: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        a_text = _asr_text_in_window(result_a, start, end)
        b_text = _asr_text_in_window(result_b, start, end)
        if not a_text and not b_text:
            continue
        if reference_side == "b":
            wer = _wer_per_window(b_text, a_text)
        else:
            wer = _wer_per_window(a_text, b_text)
        agree = wer == 0.0
        rows.append(
            {
                **base_row,
                "start": start,
                "end": end,
                "agree": agree,
                "mismatch_type": None if agree else "text_edit",
                "comparison_status": "ok",
                "wer": wer,
                "a_text": a_text,
                "b_text": b_text,
                "reference_side": reference_side,
            }
        )
    return rows


def _comparator_params_for_run(args: argparse.Namespace) -> dict[str, Any]:
    """Snapshot of comparator-relevant CLI flags for cache-key + provenance."""
    return {
        "cross_stream_win_length": args.cross_stream_win_length,
        "cross_stream_hop_length": args.cross_stream_hop_length,
        "uncertainty_aggregator": args.uncertainty_aggregator,
        "phoneme_disagreement_threshold": args.phoneme_disagreement_threshold,
        "speech_presence_labels": [s.strip() for s in args.speech_presence_labels.split(",") if s.strip()],
        "asr_reference_model": args.asr_reference_model,
        "diarization_boundary_shift_ms": args.diarization_boundary_shift_ms,
    }


def _model_block_cache_key(model_block: dict[str, Any]) -> str:
    """Best-effort cache key for an upstream model block (falls back to '' when absent)."""
    return str(model_block.get("cache_key") or "")


def _result_or_none(model_block: dict[str, Any]) -> Any:  # noqa: ANN401
    """Return the model_block's `result` only when status is ok."""
    if not isinstance(model_block, dict):
        return None
    if model_block.get("status") != "ok":
        return None
    return model_block.get("result")


def _attach_comparator_regions_to_ls_tasks(
    ls_tasks: list[dict[str, Any]],
    parquet_paths: list[Path],
    top_n: int,
) -> list[dict[str, Any]]:
    """Append disagreement regions from each parquet onto the matching pass's LS task.

    Reads each parquet, picks the rows where ``agree`` is False (capped at
    top_n total across the entire run, sorted by combined_uncertainty
    descending with NaN-last), and attaches them as Labels (and optional
    TextArea) regions to the right pass's LS task.
    """
    import math

    import pandas as pd

    # Index ls_tasks by pass label for O(1) lookup.
    by_pass: dict[str, dict[str, Any]] = {t["data"]["pass"]: t for t in ls_tasks if "data" in t}
    # Aggregate disagreement rows from all parquets.
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for p in parquet_paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        for idx, row in df.iterrows():
            if row.get("agree", True) is True:
                continue
            cu = row.get("combined_uncertainty")
            cu_val = float(cu) if cu is not None and not (isinstance(cu, float) and math.isnan(cu)) else float("nan")
            row_dict = row.to_dict()
            track_name = _comparison_ls_track_name(row_dict)
            region_id = _comparison_region_id(p, idx, row_dict)
            label_value = (
                row_dict.get("comparison_status")
                if row_dict.get("comparison_status") in ("incomparable", "one_sided")
                else "disagree"
            )
            label_region = _comparison_ls_label_region(
                region_id=region_id,
                from_name=track_name,
                start=float(row_dict["start"]),
                end=float(row_dict["end"]),
                label=label_value,
            )
            extras: list[dict[str, Any]] = []
            if "wer" in row_dict and pd.notna(row_dict.get("wer")):
                wer = float(row_dict.get("wer", 0.0))
                a = (row_dict.get("a_text") or "")[:200]
                b = (row_dict.get("b_text") or "")[:200]
                extras.append(
                    _comparison_ls_textarea_region(
                        region_id=f"{region_id}__text",
                        from_name=f"{track_name}__text",
                        start=float(row_dict["start"]),
                        end=float(row_dict["end"]),
                        text=f"WER {wer:.2f}: A={a!r} | B={b!r}",
                    )
                )
            candidates.append((cu_val, label_region, {"extras": extras, "row": row_dict}))

    # Sort: NaN last (treat as +inf so they sort to the end after descending flip).
    def _sort_key(t: tuple[float, dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        cu = t[0]
        primary = -cu if cu == cu else float("inf")
        return (primary, _disagreement_severity_rank(t[2]["row"].get("mismatch_type")), t[1]["value"]["start"])

    candidates.sort(key=_sort_key)
    selected = candidates[:top_n]
    for _cu, label_region, meta in selected:
        # Find the pass — the from_name encodes it: "<pass>__compare__..."
        from_name = label_region["from_name"]
        pass_label = from_name.split("__compare__", 1)[0]
        target = by_pass.get(pass_label)
        if target is None and pass_label == "pass_pair":
            # raw_vs_enhanced regions go on the raw_16k task by convention.
            target = by_pass.get("raw_16k")
        if target is None:
            continue
        # First prediction block holds the analyzer regions; reuse it.
        if not target.get("predictions"):
            continue
        target["predictions"][0]["result"].append(label_region)
        for extra in meta["extras"]:
            target["predictions"][0]["result"].append(extra)
    return ls_tasks


def compare_raw_vs_enhanced(
    summaries: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    cache_dir: Path | None,
    audio_sig: str,
    duration_s: float,
    wrapper_hash: str,
    senselab_ver: str,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, str]]:
    """For each task × model present in BOTH passes, emit a raw_vs_enhanced parquet.

    Returns ``(parquet_paths, comparator_tracks, incomparable_reasons)`` for
    consumption by the disagreements index and the LS bundle.
    """
    parquets: list[Path] = []
    tracks: list[dict[str, Any]] = []
    incomparable: dict[str, str] = {}
    passes = summaries.get("passes", {})
    raw = passes.get("raw_16k") or {}
    enh = passes.get("enhanced_16k") or {}
    if not raw or not enh:
        return parquets, tracks, incomparable

    grid = ComparisonGrid(
        win_length=args.cross_stream_win_length,
        hop_length=args.cross_stream_hop_length,
        name="cross_stream",
    )
    base_dir = run_dir / "comparisons" / "raw_vs_enhanced"

    # Per-task by_model differencers.
    task_specs: list[tuple[str, str, Any]] = [
        ("diarization", "by_model", _diff_diarization),
        ("asr", "by_model", _diff_asr),
        ("alignment", "by_model", _diff_asr),  # alignment outputs are ScriptLines like ASR
    ]
    for task, _key, differencer in task_specs:
        raw_models = (raw.get(task) or {}).get("by_model") or {}
        enh_models = (enh.get(task) or {}).get("by_model") or {}
        for model_id in sorted(set(raw_models) & set(enh_models)):
            raw_block = raw_models[model_id]
            enh_block = enh_models[model_id]
            raw_res = _result_or_none(raw_block)
            enh_res = _result_or_none(enh_block)
            if raw_res is None or enh_res is None:
                incomparable[f"{task}/{_safe(model_id)}/raw_vs_enhanced"] = (
                    f"upstream {task}/{model_id} not ok in one of the passes"
                )
                continue
            base_row = {
                "comparison_kind": "raw_vs_enhanced",
                "task": task,
                "stream_pair": None,
                "model_a": model_id,
                "model_b": model_id,
                "pass_a": "raw_16k",
                "pass_b": "enhanced_16k",
                "confidence_a": None,
                "confidence_b": None,
                "combined_uncertainty": None,
            }
            kwargs: dict[str, Any] = {}
            if differencer is _diff_asr:
                kwargs["reference_side"] = "a"
            elif differencer is _diff_diarization:
                kwargs["boundary_shift_ms"] = args.diarization_boundary_shift_ms
            rows = differencer(raw_res, enh_res, grid, duration_s, **kwargs, **base_row)
            if not rows:
                continue
            dest = base_dir / task / f"{_safe(model_id)}.parquet"
            cache_key_str = comparison_cache_key(
                audio_sig=audio_sig,
                comparison_kind="raw_vs_enhanced",
                task_or_pair=f"{task}/{model_id}",
                upstream_cache_keys=[
                    _model_block_cache_key(raw_block),
                    _model_block_cache_key(enh_block),
                ],
                params=_comparator_params_for_run(args),
                wrapper_hash=wrapper_hash,
                senselab_ver=senselab_ver,
            )
            cached = cache_lookup(cache_dir, cache_key_str) if cache_dir else None
            if cached is None:
                provenance = {
                    "schema_version": 1,
                    "wrapper_hash": wrapper_hash,
                    "senselab_version": senselab_ver,
                    "grid": {"name": grid.name, "win_length": grid.win_length, "hop_length": grid.hop_length},
                    "upstream_cache_keys": sorted(
                        {_model_block_cache_key(raw_block), _model_block_cache_key(enh_block)}
                    ),
                    "comparator_params": _comparator_params_for_run(args),
                    "cache_key": cache_key_str,
                }
                write_comparison_parquet(rows, dest, provenance=provenance)
                if cache_dir is not None:
                    cache_store(cache_dir, cache_key_str, {"parquet_path": str(dest), "rows": len(rows)})
            parquets.append(dest)
            track_name = f"pass_pair__compare__{task}__{_safe(model_id)}"
            tracks.append({"name": track_name, "with_textarea": task == "asr"})

    return parquets, tracks, incomparable


def _flatten_feature_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested feature dict into a single-row dict suitable for a parquet column.

    Keys are joined with ``.``; tensors are coerced to floats (mean of
    last axis when 1-D) or skipped when high-dimensional (we don't want
    per-window MFCC tensors as parquet cells — caller can opt back in
    via the JSON sibling). Lists of scalars are kept as-is so pyarrow
    can store them as a list column.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_feature_dict(v, prefix=f"{key}."))
            continue
        if hasattr(v, "ndim") and hasattr(v, "tolist"):  # torch.Tensor / np.ndarray
            try:
                if v.ndim == 0:
                    out[key] = float(v.item())
                elif v.ndim == 1 and v.shape[0] <= 64:
                    out[key] = [float(x) for x in v.tolist()]
                else:
                    # multi-dim tensor (spectrogram, mfcc) — store mean as a scalar
                    # summary; full tensor stays in the JSON sibling for callers
                    # that want it.
                    out[f"{key}.mean"] = float(v.mean().item())
            except Exception:  # noqa: BLE001 — best effort
                pass
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, str):
            out[key] = v
        # silently drop anything else (callable, opaque object) to keep the row clean
    return out


def extract_temporal_features(
    audio: Audio,
    *,
    win_length: float,
    hop_length: float,
    device: DeviceType | None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract per-backend temporal features, preferring each backend's native time grid.

    - **opensmile**: uses ``LowLevelDescriptors`` (native ~10 ms frame
      grid). One row per opensmile frame.
    - **parselmouth**: aggregates over a sliding window since the
      senselab wrapper currently only exposes the summary form.
    - **torchaudio_squim**: STOI/PESQ/SI-SDR are inherently global
      quality scores — windowed externally so the resulting time series
      is comparable to the rest.

    Returns a dict ``{backend: [rows...]}`` so each backend can be
    written to its own parquet sidecar (different columns + time grids
    don't share a schema).
    """
    duration_s = float(audio.waveform.shape[1]) / float(audio.sampling_rate)
    out: dict[str, list[dict[str, Any]]] = {"opensmile": [], "parselmouth": [], "torchaudio_squim": []}

    # opensmile LLD — native windowing (DataFrame indexed by [start, end]).
    try:
        import opensmile as _os

        smile = _os.Smile(
            feature_set=_os.FeatureSet.eGeMAPSv02,
            feature_level=_os.FeatureLevel.LowLevelDescriptors,
        )
        df = smile.process_signal(audio.waveform.squeeze().numpy(), audio.sampling_rate)
        df = df.reset_index()
        df["start"] = df["start"].dt.total_seconds()
        df["end"] = df["end"].dt.total_seconds()
        out["opensmile"] = df.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        print(f"  [features.opensmile] warn: {exc!r}", file=sys.stderr)

    # External 1 s / 0.5 s loop for the summary-style backends.
    t = 0.0
    idx = 0
    while t + win_length <= duration_s + 1e-6:
        start = round(t, 4)
        end = round(min(t + win_length, duration_s), 4)
        clip = extract_segments([(audio, [(start, end)])])[0][0]
        try:
            pm = extract_features_from_audios(
                [clip], opensmile=False, parselmouth=True, torchaudio=False, torchaudio_squim=False, device=device
            )[0]
            row = _flatten_feature_dict(pm.get("praat_parselmouth", {}))
            row.update({"start": start, "end": end, "win_index": idx})
            out["parselmouth"].append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [features.parselmouth win {idx}] warn: {exc!r}", file=sys.stderr)
        try:
            sq = extract_features_from_audios(
                [clip], opensmile=False, parselmouth=False, torchaudio=False, torchaudio_squim=True, device=device
            )[0]
            row = _flatten_feature_dict(sq.get("torchaudio_squim", {}))
            row.update({"start": start, "end": end, "win_index": idx})
            out["torchaudio_squim"].append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [features.torchaudio_squim win {idx}] warn: {exc!r}", file=sys.stderr)
        t += hop_length
        idx += 1

    return out


def align_cache_key(
    *,
    audio_sig: str,
    transcript_sha: str,
    language: str | None,
    aligner_model_id: str,
    aligner_params: dict[str, Any],
    wrapper_hash: str,
    senselab_ver: str,
) -> str:
    """Cache key for one (audio, transcript, language, aligner) alignment call.

    Independent from the ASR cache: an alignment cache hit replays prior
    timestamps without invoking the aligner; an ASR-cache miss + alignment-cache
    hit (or vice versa) is supported by construction.
    """
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "audio_signature": audio_sig,
        "task": "alignment",
        "transcript_sha": transcript_sha,
        "language": language,
        "aligner_model": aligner_model_id,
        "aligner_params": aligner_params,
        "wrapper_hash": wrapper_hash,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(_canonical_params(payload).encode()).hexdigest()


def run_alignment_cached(
    name: str,
    fn: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    cache_dir: Path | None,
    cache_key_str: str,
    provenance: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run an alignment step with cache lookup → run → store, mirroring run_task_cached.

    Identical control flow to run_task_cached; the distinction is semantic — the
    provenance includes alignment-specific fields (transcript_sha, language,
    parent_asr_cache_key) and the cache key was built via align_cache_key.
    Failed alignments are NOT stored, so a future fix to the aligner / a
    senselab upgrade triggers a fresh attempt; the parent ASR cache is
    independent and unaffected.
    """
    if cache_dir is not None:
        hit = cache_lookup(cache_dir, cache_key_str)
        if hit is not None:
            print(f"  [{name}] alignment cache HIT ({cache_key_str[:12]}...)", flush=True)
            hit["cache"] = "hit"
            hit["cache_key"] = cache_key_str
            return hit
    outcome = run_task(name, fn, *args, **kwargs)
    outcome["provenance"] = provenance
    outcome["cache"] = "miss" if cache_dir is not None else "disabled"
    outcome["cache_key"] = cache_key_str
    if cache_dir is not None and outcome.get("status") == "ok":
        cache_store(cache_dir, cache_key_str, outcome)
    return outcome


def run_task_cached(
    name: str,
    fn: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    cache_dir: Path | None,
    cache_key_str: str,
    provenance: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run a task with cache lookup → run → cache store, attaching provenance.

    On cache hit, the stored outcome is returned with ``cache="hit"`` (and
    elapsed_s set to 0). On cache miss, the task runs, the outcome is stored,
    and ``cache="miss"`` is reported.
    """
    if cache_dir is not None:
        hit = cache_lookup(cache_dir, cache_key_str)
        if hit is not None:
            print(f"  [{name}] cache HIT ({cache_key_str[:12]}...)", flush=True)
            hit["cache"] = "hit"
            hit["cache_key"] = cache_key_str
            return hit
    outcome = run_task(name, fn, *args, **kwargs)
    outcome["provenance"] = provenance
    outcome["cache"] = "miss" if cache_dir is not None else "disabled"
    outcome["cache_key"] = cache_key_str
    if cache_dir is not None and outcome.get("status") == "ok":
        cache_store(cache_dir, cache_key_str, outcome)
    return outcome


def run_task(
    name: str,
    fn: Any,  # noqa: ANN401 — generic dispatcher
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run a task with timing + structured error capture."""
    print(f"  [{name}] running...", flush=True)
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — diagnostic capture by design
        elapsed = time.perf_counter() - started
        print(f"  [{name}] FAILED in {elapsed:.1f}s: {exc}", flush=True)
        return {
            "status": "failed",
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=5),
        }
    elapsed = time.perf_counter() - started
    print(f"  [{name}] ok in {elapsed:.1f}s", flush=True)
    return {"status": "ok", "elapsed_s": round(elapsed, 3), "result": result}


def _new_region_id(prefix: str, idx: int) -> str:
    """Stable per-region ID for Label Studio result entries."""
    return f"{prefix}_{idx:04d}"


def _ls_label_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    label: str,
    score: float | None = None,
) -> dict[str, Any]:
    """Build one Label Studio ``labels`` result entry on the audio timeline."""
    value: dict[str, Any] = {"start": float(start), "end": float(end), "labels": [label]}
    entry: dict[str, Any] = {
        "id": region_id,
        "from_name": from_name,
        "to_name": "audio",
        "type": "labels",
        "value": value,
    }
    if score is not None:
        entry["score"] = float(score)
    return entry


def _ls_textarea_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    text: str,
) -> dict[str, Any]:
    """Build one Label Studio ``textarea`` per-region transcription entry."""
    return {
        "id": region_id,
        "from_name": from_name,
        "to_name": "audio",
        "type": "textarea",
        "value": {"start": float(start), "end": float(end), "text": [text]},
    }


def _seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Return ``seg.name`` whether ``seg`` is a Pydantic model or a JSON dict.

    Cache reads deserialize ScriptLine into plain dicts; in-memory results are
    Pydantic objects. Both shapes flow through the LS helpers.
    """
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


def _diarization_to_ls(result: Any, prefix: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert diarize_audios output (List[List[ScriptLine]]) into LS regions."""
    out: list[dict[str, Any]] = []
    if not result:
        return out
    segments = result[0] if isinstance(result, list) and result else []
    for i, seg in enumerate(segments):
        start = _seg_attr(seg, "start")
        end = _seg_attr(seg, "end")
        speaker = _seg_attr(seg, "speaker") or "SPEAKER_UNKNOWN"
        if start is None or end is None:
            continue
        out.append(
            _ls_label_region(
                region_id=_new_region_id(f"{prefix}_dia", i),
                from_name=prefix,
                start=start,
                end=end,
                label=str(speaker),
            )
        )
    return out


def _classification_to_ls(
    result: Any,  # noqa: ANN401
    prefix: str,
    win_length: float,
    hop_length: float,
) -> list[dict[str, Any]]:
    """Convert classify_audios windowed output into LS regions (top-1 per window).

    Window centers advance by ``hop_length`` and span ``win_length`` seconds, so
    the LS regions reflect each model's own native frame stride.
    """
    out: list[dict[str, Any]] = []
    if not result:
        return out
    windows = result[0] if isinstance(result, list) and result else []
    for i, window in enumerate(windows):
        if not isinstance(window, list) or not window:
            continue
        top = max(window, key=lambda d: d.get("score", 0.0))
        label = str(top.get("label", "?"))
        score = float(top.get("score", 0.0))
        start = i * hop_length
        out.append(
            _ls_label_region(
                region_id=_new_region_id(f"{prefix}_cls", i),
                from_name=prefix,
                start=start,
                end=start + win_length,
                label=label,
                score=score,
            )
        )
    return out


def _scene_agreement(
    ast_result: Any,  # noqa: ANN401
    yamnet_result: Any,  # noqa: ANN401
    win_length: float,
    hop_length: float,
) -> dict[str, Any]:
    """Pair AST and YAMNet top-1 predictions per shared window for direct comparison.

    Both models share an AudioSet 521-class label space, so when they run on
    the same ``(win_length, hop_length)`` grid the per-window top-1 labels are
    directly comparable. Produces a list of ``{start, end, ast, yamnet,
    agree}`` dicts plus aggregate agreement statistics.
    """
    ast_windows = ast_result[0] if isinstance(ast_result, list) and ast_result else []
    yamnet_windows = yamnet_result[0] if isinstance(yamnet_result, list) and yamnet_result else []
    pairs: list[dict[str, Any]] = []
    n = min(len(ast_windows), len(yamnet_windows))
    agree_count = 0
    for i in range(n):
        a = _top1(ast_windows[i])
        y = _top1(yamnet_windows[i])
        same = bool(a and y and a["label"] == y["label"])
        agree_count += int(same)
        start = i * hop_length
        pairs.append(
            {
                "start": start,
                "end": start + win_length,
                "ast": a,
                "yamnet": y,
                "agree": same,
            }
        )
    return {
        "win_length": win_length,
        "hop_length": hop_length,
        "windows_compared": n,
        "ast_only_windows": max(0, len(ast_windows) - n),
        "yamnet_only_windows": max(0, len(yamnet_windows) - n),
        "agreement_rate": (agree_count / n) if n else 0.0,
        "agree_count": agree_count,
        "pairs": pairs,
    }


def _top1(window: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Return the highest-scoring entry of a classify_audios window, or None."""
    if not isinstance(window, list) or not window:
        return None
    top = max(window, key=lambda d: d.get("score", 0.0))
    return {"label": str(top.get("label", "?")), "score": float(top.get("score", 0.0))}


def _asr_has_timestamps(result: Any) -> bool:  # noqa: ANN401
    """Return True if any ScriptLine in ``result`` has a non-null start or non-empty chunks.

    Used by the LS export and by the auto-align stage to decide whether to skip
    forced alignment for an ASR result that already includes time information.
    Handles both ScriptLine objects (post-run, in-memory) and the dict shape that
    the JSON cache deserializes into (post-cache-hit).
    """
    if not result:
        return False
    items = result if isinstance(result, list) else [result]
    for line in items:
        if isinstance(line, dict):
            start = line.get("start")
            chunks = line.get("chunks") or []
        else:
            start = getattr(line, "start", None)
            chunks = getattr(line, "chunks", None) or []
        if start is not None or len(chunks) > 0:
            return True
    return False


def _extract_transcript_text(result: Any) -> str:  # noqa: ANN401
    """Concatenate the ``text`` field of every ScriptLine / dict in an ASR result."""
    if not result:
        return ""
    items = result if isinstance(result, list) else [result]
    parts: list[str] = []
    for line in items:
        text = line.get("text") if isinstance(line, dict) else getattr(line, "text", None)
        if text:
            parts.append(str(text))
    return " ".join(p.strip() for p in parts if p.strip())


def _asr_to_ls(result: Any, prefix: str, full_duration: float) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert transcribe_audios output into LS textarea regions, one per ScriptLine.

    Whisper sometimes returns one ScriptLine without timing for a short clip; in
    that case we pin the textarea to the full audio span.
    """
    out: list[dict[str, Any]] = []
    if not result:
        return out
    lines = result if isinstance(result, list) else [result]
    for i, line in enumerate(lines):
        text = _seg_attr(line, "text") or ""
        start = _seg_attr(line, "start")
        end = _seg_attr(line, "end")
        if start is None or end is None:
            start, end = 0.0, full_duration
        if not text:
            continue
        out.append(
            _ls_textarea_region(
                region_id=_new_region_id(f"{prefix}_asr", i),
                from_name=prefix,
                start=start,
                end=end,
                text=text,
            )
        )
    return out


def build_labelstudio_task(
    audio_uri: str,
    pass_label: str,
    duration_s: float,
    pass_summary: dict[str, Any],
    ast_win_length: float,
    ast_hop_length: float,
    yamnet_win_length: float,
    yamnet_hop_length: float,
) -> dict[str, Any]:
    """Build one Label Studio task with predictions for all analyzers in this pass.

    Each analyzer (diarization, ast, yamnet, asr) becomes its own
    ``from_name`` track on the audio timeline, so the importer sees parallel
    annotation rows. When a senselab task was run with multiple models, every
    model's output is exported as its own track (e.g., ``asr_whisper_turbo``,
    ``asr_whisper_small``) so they can be visually compared.
    """
    regions: list[dict[str, Any]] = []

    dia = pass_summary.get("diarization", {})
    for model_id, model_block in (dia.get("by_model") or {}).items():
        if model_block.get("status") == "ok":
            from_name = f"{pass_label}__diarization__{_safe(model_id)}"
            regions.extend(_diarization_to_ls(model_block.get("result"), from_name))

    ast_block = pass_summary.get("ast", {})
    if ast_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                ast_block.get("result"),
                f"{pass_label}__ast",
                win_length=ast_win_length,
                hop_length=ast_hop_length,
            )
        )

    yam_block = pass_summary.get("yamnet", {})
    if yam_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                yam_block.get("result"),
                f"{pass_label}__yamnet",
                win_length=yamnet_win_length,
                hop_length=yamnet_hop_length,
            )
        )

    asr = pass_summary.get("asr", {})
    alignment = pass_summary.get("alignment") or {}
    align_by_model = alignment.get("by_model") or {}
    for model_id, model_block in (asr.get("by_model") or {}).items():
        if model_block.get("status") != "ok":
            continue
        from_name = f"{pass_label}__asr__{_safe(model_id)}"
        # Three-case branch:
        # (a) ASR with native timestamps  -> use the ASR result for per-segment regions.
        # (b) ASR text-only + successful alignment -> use the alignment result.
        # (c) ASR text-only + alignment skipped or failed -> single full-audio TextArea.
        align_block = align_by_model.get(model_id) or {}
        asr_result = model_block.get("result")
        if _asr_has_timestamps(asr_result):
            regions.extend(_asr_to_ls(asr_result, from_name, duration_s))
        elif align_block.get("status") == "ok":
            # align_transcriptions returns List[List[ScriptLine | None]] —
            # one inner list per input audio. We always pass a single audio,
            # so unwrap to the inner segment list.
            ar = align_block.get("result")
            inner = ar[0] if isinstance(ar, list) and ar and isinstance(ar[0], list) else ar
            regions.extend(_asr_to_ls(inner, from_name, duration_s))
        else:
            regions.extend(_asr_to_ls(asr_result, from_name, duration_s))

    return {
        "data": {
            "audio": audio_uri,
            "pass": pass_label,
            "duration_s": duration_s,
        },
        "predictions": [
            {
                "model_version": f"senselab-analyze:{pass_label}",
                "score": 1.0,
                "result": regions,
            }
        ],
    }


def _safe(model_id: str) -> str:
    """Sanitize a model id for use inside an LS from_name (LS allows letters, digits, underscore)."""
    return "".join(c if c.isalnum() else "_" for c in model_id)


def build_labelstudio_config(
    summary: dict[str, Any],
    comparator_tracks: list[dict[str, Any]] | None = None,
) -> str:
    """Build a Label Studio labeling-config XML matching this run's tracks.

    Generates one ``<Labels>`` control per (pass, analyzer, model) and one
    ``<TextArea>`` control per (pass, asr_model). Speakers, scene labels,
    and transcripts each become a stacked timeline annotation row.

    When ``comparator_tracks`` is provided, additionally emits one fixed-enum
    ``<Labels>`` block per comparator track (values: ``agree`` / ``disagree``
    / ``incomparable`` / ``one_sided``) and a sibling ``<TextArea>`` block
    for ASR-vs-ASR pairs (FR-004, contracts/ls-bundle.md).
    """
    parts: list[str] = ["<View>", '  <Audio name="audio" value="$audio"/>']
    seen_label_sets: dict[str, list[str]] = {}

    for pass_label, pass_summary in summary.get("passes", {}).items():
        # Diarization tracks: one per model, with that model's discovered speaker labels
        dia_by_model = (pass_summary.get("diarization") or {}).get("by_model") or {}
        for model_id, model_block in dia_by_model.items():
            if model_block.get("status") != "ok":
                continue
            speakers = sorted({str(getattr(seg, "speaker", "?")) for seg in (model_block.get("result", [[]])[0] or [])})
            if not speakers:
                speakers = ["SPEAKER_00", "SPEAKER_01"]
            seen_label_sets[f"{pass_label}__diarization__{_safe(model_id)}"] = speakers

        # AST scene labels
        ast = pass_summary.get("ast") or {}
        if ast.get("status") == "ok":
            labels = _collect_classification_labels(ast.get("result"))
            if labels:
                seen_label_sets[f"{pass_label}__ast"] = sorted(labels)

        # YAMNet scene labels
        yam = pass_summary.get("yamnet") or {}
        if yam.get("status") == "ok":
            labels = _collect_classification_labels(yam.get("result"))
            if labels:
                seen_label_sets[f"{pass_label}__yamnet"] = sorted(labels)

        # ASR: each model gets its own TextArea
        asr_by_model = (pass_summary.get("asr") or {}).get("by_model") or {}
        for model_id, model_block in asr_by_model.items():
            if model_block.get("status") != "ok":
                continue
            from_name = f"{pass_label}__asr__{_safe(model_id)}"
            parts.append(
                f'  <TextArea name="{from_name}" toName="audio" perRegion="true" '
                f'editable="true" placeholder="ASR transcript ({model_id})"/>'
            )

    for from_name, label_values in sorted(seen_label_sets.items()):
        parts.append(f'  <Labels name="{from_name}" toName="audio">')
        for v in label_values:
            v_escaped = v.replace('"', "&quot;")
            parts.append(f'    <Label value="{v_escaped}"/>')
        parts.append("  </Labels>")

    # Comparator tracks (FR-004 / contracts/ls-bundle.md): fixed enumerated
    # label set per Labels block; optional sibling TextArea for ASR-vs-ASR.
    for track in sorted(comparator_tracks or [], key=lambda t: t["name"]):
        name = track["name"]
        parts.append(f'  <Labels name="{name}" toName="audio">')
        for v in ("agree", "disagree", "incomparable", "one_sided"):
            parts.append(f'    <Label value="{v}"/>')
        parts.append("  </Labels>")
        if track.get("with_textarea"):
            parts.append(
                f'  <TextArea name="{name}__text" toName="audio" perRegion="true" '
                f'editable="false" placeholder="WER and transcripts for this disagreement"/>'
            )
    parts.append("</View>")
    return "\n".join(parts) + "\n"


def _collect_classification_labels(result: Any) -> set[str]:  # noqa: ANN401
    """Extract the union of label strings observed in a classify_audios output."""
    labels: set[str] = set()
    if not result:
        return labels
    windows = result[0] if isinstance(result, list) and result else []
    for window in windows:
        if not isinstance(window, list):
            continue
        for entry in window:
            label = entry.get("label") if isinstance(entry, dict) else None
            if label:
                labels.add(str(label))
    return labels


def serialize(obj: Any) -> Any:  # noqa: ANN401 — recursive heterogeneous serializer
    """Convert senselab outputs (ScriptLine, tensor, etc.) to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return {
            "_tensor_shape": list(obj.shape),
            "_dtype": str(obj.dtype),
            "values": obj.detach().cpu().tolist(),
        }
    if hasattr(obj, "model_dump"):
        return serialize(obj.model_dump())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def write_json(path: Path, payload: Any) -> None:  # noqa: ANN401
    """Write a JSON file with senselab-aware serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialize(payload), fh, indent=2, default=str)


def run_pass(
    label: str,
    audio: Audio,
    args: argparse.Namespace,
    device: DeviceType | None,
    out_dir: Path,
    *,
    cache_dir: Path | None,
    wrapper_hash: str,
    senselab_ver: str,
) -> dict[str, Any]:
    """Run all six tasks against one audio variant and persist their outputs.

    Each task call is gated through the content-addressable cache: the cache
    key is ``sha256(audio_signature, task, model_id, params, wrapper_hash,
    senselab_ver)``. On cache hit the prior outcome is replayed verbatim and
    no model is loaded. On miss the task runs and the outcome is stored under
    the cache dir for future replay.
    """
    duration_s = audio.waveform.shape[1] / audio.sampling_rate
    audio_sig = audio_signature(audio)
    print(f"\n=== Pass: {label} ({duration_s:.2f}s @ {audio.sampling_rate}Hz, sig={audio_sig[:12]}...) ===")
    pass_dir = out_dir / label
    pass_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "label": label,
        "duration_s": duration_s,
        "audio_signature": audio_sig,
    }
    device_label_for_provenance = device.value if device is not None else "auto"

    def _provenance_for(task: str, model_id: str | None, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "task": task,
            "model_id": model_id,
            "params": params,
            "audio_signature": audio_sig,
            "audio_source": str(args.audio.resolve()),
            "pass": label,
            "device": device_label_for_provenance,
            "wrapper_version_hash": wrapper_hash,
            "senselab_version": senselab_ver,
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _key(task: str, model_id: str | None, params: dict[str, Any]) -> str:
        return cache_key(
            audio_sig=audio_sig,
            task=task,
            model_id=model_id,
            params=params,
            wrapper_hash=wrapper_hash,
            senselab_ver=senselab_ver,
        )

    if "diarization" not in args.skip:
        summary["diarization"] = {"by_model": {}}
        for model_id in args.diarization_models:
            params = {"device": device_label_for_provenance}
            outcome = run_task_cached(
                f"diarization[{model_id}]",
                diarize_audios,
                [audio],
                model=pick_dispatch_model(model_id, task="diarization"),
                device=device,
                cache_dir=cache_dir,
                cache_key_str=_key("diarization", model_id, params),
                provenance=_provenance_for("diarization", model_id, params),
            )
            summary["diarization"]["by_model"][model_id] = outcome
            write_json(pass_dir / "diarization" / f"{_safe(model_id)}.json", outcome)

    ast_win = args.ast_win_length
    ast_hop = args.ast_hop_length
    yam_win = args.yamnet_win_length
    yam_hop = args.yamnet_hop_length

    if "ast" not in args.skip:
        params = {"win_length": ast_win, "hop_length": ast_hop, "device": device_label_for_provenance}
        summary["ast"] = run_task_cached(
            "ast",
            classify_audios,
            [audio],
            model=HFModel(path_or_uri=args.ast_model),
            device=device,
            win_length=ast_win,
            hop_length=ast_hop,
            cache_dir=cache_dir,
            cache_key_str=_key("ast", args.ast_model, params),
            provenance=_provenance_for("ast", args.ast_model, params),
        )
        summary["ast"]["window"] = {"win_length": ast_win, "hop_length": ast_hop}
        write_json(pass_dir / "ast.json", summary["ast"])

    if "yamnet" not in args.skip:
        # YAMNet runs in senselab's TF subprocess venv (same pattern as NeMo
        # Sortformer). senselab.classify_audios's `_is_yamnet()` dispatcher
        # matches on the raw model-id *string*, not on a SenselabModel wrapper —
        # passing HFModel here would fail validation (yamnet isn't on HF).
        params = {"win_length": yam_win, "hop_length": yam_hop}
        summary["yamnet"] = run_task_cached(
            "yamnet",
            classify_audios,
            [audio],
            model=args.yamnet_model,
            win_length=yam_win,
            hop_length=yam_hop,
            cache_dir=cache_dir,
            cache_key_str=_key("yamnet", args.yamnet_model, params),
            provenance=_provenance_for("yamnet", args.yamnet_model, params),
        )
        summary["yamnet"]["window"] = {"win_length": yam_win, "hop_length": yam_hop}
        write_json(pass_dir / "yamnet.json", summary["yamnet"])

    # If both ran on the same grid, emit a side-by-side comparison.
    if (
        "ast" not in args.skip
        and "yamnet" not in args.skip
        and summary.get("ast", {}).get("status") == "ok"
        and summary.get("yamnet", {}).get("status") == "ok"
        and ast_win == yam_win
        and ast_hop == yam_hop
    ):
        agreement = _scene_agreement(
            ast_result=summary["ast"]["result"],
            yamnet_result=summary["yamnet"]["result"],
            win_length=ast_win,
            hop_length=ast_hop,
        )
        summary["scene_agreement"] = agreement
        write_json(pass_dir / "scene_agreement.json", agreement)

    if "features" not in args.skip:
        feat_params: dict[str, Any] = {
            "opensmile": "LowLevelDescriptors@native",
            "parselmouth": "windowed",
            "torchaudio_squim": "windowed",
            "device": device_label_for_provenance,
            "win_length": args.features_win_length,
            "hop_length": args.features_hop_length,
        }
        summary["features"] = run_task_cached(
            "features",
            extract_temporal_features,
            audio,
            win_length=args.features_win_length,
            hop_length=args.features_hop_length,
            device=device,
            cache_dir=cache_dir,
            cache_key_str=_key("features", None, feat_params),
            provenance=_provenance_for("features", None, feat_params),
        )
        # Each backend writes its own parquet sidecar — they have
        # different columns and different time grids (opensmile LLD is
        # native ~10 ms; parselmouth/torchaudio_squim follow the
        # ``--features-*-length`` window). Cache outcome metadata stays
        # in JSON for inspection parity with the other tasks.
        result = summary["features"].get("result") or {}
        if isinstance(result, dict):
            try:
                import pandas as pd

                feat_dir = pass_dir / "features"
                feat_dir.mkdir(parents=True, exist_ok=True)
                for backend, rows in result.items():
                    if not rows:
                        continue
                    pd.DataFrame(rows).to_parquet(feat_dir / f"{backend}.parquet", index=False)
            except Exception as exc:  # noqa: BLE001 — best-effort sidecar
                print(f"  [features] warn: parquet write failed: {exc!r}", file=sys.stderr)
        write_json(pass_dir / "features.json", {**summary["features"], "result": "see features/*.parquet"})

    if "asr" not in args.skip:
        summary["asr"] = {"by_model": {}}
        for model_id in args.asr_models:
            asr_params: dict[str, Any] = {"device": device_label_for_provenance}
            extra_kwargs: dict[str, Any] = {}
            # Qwen3-ASR ships its own forced-aligner companion model; allow
            # opt-out so the script's MMS auto-align stage can take over.
            if model_id.startswith("Qwen/Qwen3-ASR") and args.qwen_asr_no_timestamps:
                extra_kwargs["return_timestamps"] = False
                asr_params["return_timestamps"] = False
            outcome = run_task_cached(
                f"asr[{model_id}]",
                transcribe_audios,
                [audio],
                model=pick_dispatch_model(model_id, task="asr"),
                device=device,
                cache_dir=cache_dir,
                cache_key_str=_key("asr", model_id, asr_params),
                provenance=_provenance_for("asr", model_id, asr_params),
                **extra_kwargs,
            )
            summary["asr"]["by_model"][model_id] = outcome
            write_json(pass_dir / "asr" / f"{_safe(model_id)}.json", outcome)

    # Auto-align stage. Iterate every successful ASR ModelRun; when the ASR
    # result lacks native timestamps (Granite Speech, Canary-Qwen, etc.) and
    # the user hasn't disabled alignment, post-process the text through
    # senselab's forced_alignment to produce per-segment timestamps. The
    # alignment cache is independent from the ASR cache (FR-024); failed
    # alignments preserve the ASR text but fall back to a single full-audio
    # TextArea region in the LS export (FR-025).
    if "asr" not in args.skip and "alignment" not in args.skip and not args.no_align_asr and "asr" in summary:
        summary["alignment"] = {"by_model": {}}
        align_language = (
            Language(language_code=args.asr_language) if args.asr_language else Language(language_code="en")
        )
        for model_id, asr_outcome in summary["asr"]["by_model"].items():
            if asr_outcome.get("status") != "ok":
                continue
            asr_result = asr_outcome.get("result")
            if _asr_has_timestamps(asr_result):
                # Already has native timestamps — alignment would be a no-op.
                continue
            transcript_text = _extract_transcript_text(asr_result)
            if not transcript_text:
                continue
            transcript_sha = transcript_signature(transcript_text)
            aligner_params = {
                "language": align_language.language_code,
                "romanize": align_language.language_code in ("ja", "zh"),
            }
            align_provenance = {
                **_provenance_for("alignment", args.aligner_model, aligner_params),
                "transcript_sha": transcript_sha,
                "language": align_language.language_code,
                "parent_asr_cache_key": asr_outcome.get("cache_key"),
            }
            align_key = align_cache_key(
                audio_sig=audio_sig,
                transcript_sha=transcript_sha,
                language=align_language.language_code,
                aligner_model_id=args.aligner_model,
                aligner_params=aligner_params,
                wrapper_hash=wrapper_hash,
                senselab_ver=senselab_ver,
            )
            outcome = run_alignment_cached(
                f"alignment[{model_id}]",
                align_transcriptions,
                [(audio, ScriptLine(text=transcript_text), align_language)],
                aligner_model=args.aligner_model,
                cache_dir=cache_dir,
                cache_key_str=align_key,
                provenance=align_provenance,
            )
            summary["alignment"]["by_model"][model_id] = outcome
            write_json(pass_dir / "alignment" / f"{_safe(model_id)}.json", outcome)

    if "embeddings" not in args.skip:
        summary["embeddings"] = {"by_model": {}}
        for model_id in args.embeddings_models:
            params = {"device": device_label_for_provenance}
            outcome = run_task_cached(
                f"embeddings[{model_id}]",
                extract_speaker_embeddings_from_audios,
                [audio],
                model=pick_dispatch_model(model_id, task="embeddings"),
                device=device,
                cache_dir=cache_dir,
                cache_key_str=_key("embeddings", model_id, params),
                provenance=_provenance_for("embeddings", model_id, params),
            )
            summary["embeddings"]["by_model"][model_id] = outcome
            write_json(pass_dir / "embeddings" / f"{_safe(model_id)}.json", outcome)

    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the full analysis with and without enhancement."""
    args = parse_args(argv)

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    device_label = device.value if device is not None else "auto (per-task selection)"
    cache_dir: Path | None = None if args.no_cache else args.cache_dir.resolve()
    wrapper_hash = wrapper_version_hash()
    senselab_ver = senselab_version()
    print(f"Device: {device_label}")
    print(f"Input:  {args.audio}")
    if cache_dir is not None:
        print(f"Cache:  {cache_dir}")
        print(
            f"        key = sha256(audio | task | model | params | "
            f"wrapper={wrapper_hash[:8]} | senselab={senselab_ver})"
        )
    else:
        print("Cache:  disabled (--no-cache)")

    audio_16k = prepare_audio(args.audio)
    print(f"Resampled: {audio_16k.waveform.shape[1] / TARGET_SR:.2f}s @ {TARGET_SR}Hz mono")

    run_dir = args.output_dir / f"{args.audio.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    summaries: dict[str, Any] = {
        "input_audio": str(args.audio.resolve()),
        "device": device_label,
        "cache": {
            "enabled": cache_dir is not None,
            "dir": str(cache_dir) if cache_dir is not None else None,
            "schema_version": _CACHE_SCHEMA_VERSION,
        },
        "wrapper_version_hash": wrapper_hash,
        "senselab_version": senselab_ver,
        "target_sampling_rate": TARGET_SR,
        "scene_window": {
            "ast": {"win_length": args.ast_win_length, "hop_length": args.ast_hop_length},
            "yamnet": {"win_length": args.yamnet_win_length, "hop_length": args.yamnet_hop_length},
            "comparable": (
                args.ast_win_length == args.yamnet_win_length and args.ast_hop_length == args.yamnet_hop_length
            ),
        },
        "passes": {},
    }

    summaries["passes"]["raw_16k"] = run_pass(
        "raw_16k",
        audio_16k,
        args,
        device,
        run_dir,
        cache_dir=cache_dir,
        wrapper_hash=wrapper_hash,
        senselab_ver=senselab_ver,
    )

    if not args.no_enhancement:
        print("\n=== Enhancing audio (this loads the enhancement model)... ===")
        try:
            enhanced = enhance_audios(
                [audio_16k],
                model=pick_dispatch_model(args.enhancement_model, task="enhancement"),
                device=device,
            )[0]
            summaries["passes"]["enhanced_16k"] = run_pass(
                "enhanced_16k",
                enhanced,
                args,
                device,
                run_dir,
                cache_dir=cache_dir,
                wrapper_hash=wrapper_hash,
                senselab_ver=senselab_ver,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Enhancement failed: {exc!r}", file=sys.stderr)
            summaries["passes"]["enhanced_16k"] = {"status": "failed", "error": repr(exc)}

    write_json(run_dir / "summary.json", summaries)

    # ── Comparison & uncertainty stage ─────────────────────────────────
    # Runs only when the user did not opt out via --skip comparisons.
    comparator_parquets: list[Path] = []
    comparator_tracks: list[dict[str, Any]] = []
    incomparable_reasons: dict[str, str] = {}
    if "comparisons" not in args.skip:
        # Pick a duration to drive the cross-stream grid: prefer raw pass's,
        # fall back to enhanced if raw failed wholesale.
        comparison_duration = (
            (summaries.get("passes", {}).get("raw_16k") or {}).get("duration_s")
            or (summaries.get("passes", {}).get("enhanced_16k") or {}).get("duration_s")
            or 0.0
        )
        if comparison_duration > 0 and "raw_vs_enhanced" not in args.skip_comparisons:
            rve_parquets, rve_tracks, rve_incomparable = compare_raw_vs_enhanced(
                summaries=summaries,
                args=args,
                run_dir=run_dir,
                cache_dir=cache_dir,
                audio_sig=audio_signature(audio_16k),
                duration_s=comparison_duration,
                wrapper_hash=wrapper_hash,
                senselab_ver=senselab_ver,
            )
            comparator_parquets.extend(rve_parquets)
            comparator_tracks.extend(rve_tracks)
            incomparable_reasons.update(rve_incomparable)

    # Hierarchical Label Studio export — one LS task per audio variant, each
    # carrying parallel timeline tracks (one per analyzer × model). AST and
    # YAMNet contribute regions at their own native temporal resolution.
    audio_uri = str(args.audio.resolve())
    ls_tasks = [
        build_labelstudio_task(
            audio_uri=audio_uri,
            pass_label=pass_label,
            duration_s=pass_summary["duration_s"],
            pass_summary=pass_summary,
            ast_win_length=args.ast_win_length,
            ast_hop_length=args.ast_hop_length,
            yamnet_win_length=args.yamnet_win_length,
            yamnet_hop_length=args.yamnet_hop_length,
        )
        for pass_label, pass_summary in summaries["passes"].items()
        if isinstance(pass_summary, dict) and "duration_s" in pass_summary
    ]
    # Append comparator LS regions (top-N flagged disagreements) before write.
    if comparator_parquets and args.disagreements_top_n > 0:
        ls_tasks = _attach_comparator_regions_to_ls_tasks(ls_tasks, comparator_parquets, args.disagreements_top_n)
    write_json(run_dir / "labelstudio_tasks.json", ls_tasks)

    config_xml = build_labelstudio_config(summaries, comparator_tracks=comparator_tracks)
    (run_dir / "labelstudio_config.xml").write_text(config_xml, encoding="utf-8")

    # ── Disagreements index ─────────────────────────────────────────────
    if "comparisons" not in args.skip:
        index = _build_disagreements_index(
            parquet_paths=comparator_parquets,
            top_n=args.disagreements_top_n,
            aggregator=args.uncertainty_aggregator,
            run_dir=run_dir,
            config={
                "top_n": args.disagreements_top_n,
                "aggregator": args.uncertainty_aggregator,
                "phoneme_disagreement_threshold": args.phoneme_disagreement_threshold,
                "cross_stream_grid": {
                    "win_length": args.cross_stream_win_length,
                    "hop_length": args.cross_stream_hop_length,
                },
                "speech_presence_labels": [s.strip() for s in args.speech_presence_labels.split(",") if s.strip()],
            },
            incomparable_reasons=incomparable_reasons,
            missing_confidence_signals=[],
        )
        write_json(run_dir / "disagreements.json", index)

    print(f"\nDone. Summary: {run_dir / 'summary.json'}")
    print(f"Label Studio tasks:  {run_dir / 'labelstudio_tasks.json'}")
    print(f"Label Studio config: {run_dir / 'labelstudio_config.xml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
