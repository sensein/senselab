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
    **nyralabs/CrisperWhisper2.0_turbo**           (crisperwhisper CT2 subprocess venv; verbatim, native word
                                                    timestamps + per-word confidence)
    **nvidia/canary-qwen-2.5b**                    (NeMo SALM subprocess venv, 2.5B; text-only, post-aligned)
    **Qwen/Qwen3-ASR-1.7B**                        (qwen-asr subprocess venv, 1.7B; native word timestamps via
                                                    Qwen3-ForcedAligner-0.6B companion)

  speech_to_text (additional, available via --asr-models):
    openai/whisper-large-v3-turbo                  (HFModel, 809M, multilingual; native timestamps)
    ibm-granite/granite-speech-3.3-8b              (~9B, EN + 7 translations; text-only, post-aligned)
    openai/whisper-small                           (HFModel, 244M; native timestamps)
    nvidia/stt_en_conformer_ctc_large              (NeMo subprocess venv, English-only; native CTC timestamps)

Auto-align stage: every ASR model in --asr-models that returns text-only
ScriptLines (no native timestamps and no chunks) is automatically force-aligned
to add per-word timestamps. The aligner is selectable via --aligner: 'qwen'
(default) uses Qwen3-ForcedAligner-0.6B in the qwen-asr subprocess venv; 'mms'
uses the multilingual facebook/mms-1b-all path (--aligner-model, 1100+ languages
via per-language adapters). ASR models with native timestamps (CrisperWhisper,
Qwen3-ASR) are never re-aligned. Pass --no-align-asr to skip this and emit a
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
           stage_version || senselab_version || cache_schema_version)

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
from senselab.audio.tasks.features_extraction.temporal import extract_temporal_features
from senselab.audio.tasks.forced_alignment import align_transcriptions
from senselab.audio.tasks.input_output import read_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, extract_segments, resample_audios
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.audio.tasks.speech_to_text.qwen import QwenASR
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_window_top1 as _classification_window_top1,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_windows as _classification_windows,
)
from senselab.audio.workflows.audio_analysis.stage_context import STAGE_VERSIONS, PassPlan, StageContext
from senselab.audio.workflows.audio_analysis.stages import (
    _asr_has_timestamps,
    run_pass,
)
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    Language,
    ScriptLine,
    model_for_task,
    safe_model_id,
)
from senselab.utils.tasks.cached_inference import (
    CACHE_SCHEMA_VERSION as _CACHE_SCHEMA_VERSION,
)
from senselab.utils.tasks.cached_inference import (
    align_cache_key,
    audio_signature,
    cache_key,
    cache_lookup,
    cache_store,
    run_alignment_cached,
    run_task,
    run_task_cached,
    senselab_version,
    serialize,
    transcript_signature,
    write_json,
)
from senselab.utils.tasks.cached_inference import (
    canonical_params as _canonical_params,
)
from senselab.utils.tasks.cached_inference import (
    sync_cache_with_schema_version as _sync_cache_with_schema_version,
)

TARGET_SR = 16000
ALL_TASKS = ("diarization", "ast", "yamnet", "features", "asr", "alignment", "comparisons")
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
        help="Skip the enhanced-audio pass; only run on the resampled original. Alias for --enhancement never.",
    )
    # ── Adaptive loop (T040; contracts/cli.md) ────────────────────────
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help=(
            "Total adaptive rounds including the baseline. 1 = baseline only: no interventions and no "
            "rounds/>=2, though final/ is still emitted from the round-1 belief. Use 1 for "
            "golden-compat runs."
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Adaptive policy YAML, deep-merged over the packaged default. CLI overrides below win over it.",
    )
    parser.add_argument(
        "--budget-medium",
        type=int,
        default=None,
        help="Per-run budget for medium-cost interventions (default from the policy file).",
    )
    parser.add_argument(
        "--budget-heavy",
        type=int,
        default=None,
        help="Per-run budget for heavy-cost interventions (default from the policy file). 0 disables them.",
    )
    parser.add_argument(
        "--max-region-rounds",
        type=int,
        default=None,
        help="Cap on how many rounds may touch the same region (default from the policy file).",
    )
    parser.add_argument(
        "--region-top-n",
        type=int,
        default=None,
        help="How many high-uncertainty regions to admit per round (default from the policy file).",
    )
    parser.add_argument(
        "--reserve-asr-models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Reserve ASR pool for U2 escalation, in order (default from the policy file).",
    )
    parser.add_argument(
        "--enable-overlap-separation",
        action="store_true",
        help=(
            "Force the overlap-detection rule on, overriding a policy file that disabled it. NOTE: "
            "contracts/cli.md calls this a 'v2 U4 rule (off by default)'; the shipped rule is "
            "I4_overlap_detection and the packaged policy already enables it, so this flag only "
            "matters against a policy that turns it off."
        ),
    )
    parser.add_argument(
        "--no-adaptive-outputs",
        action="store_true",
        help="Suppress the adaptive rounds/ and final/ artifacts (debug / regression aid).",
    )
    parser.add_argument(
        "--enhancement",
        choices=("auto", "always", "never"),
        default="always",
        help=(
            "Enhanced-pass policy (spec 20260723-225523 FR-003). 'always' preserves the historical "
            "unconditional two-pass behavior; 'auto' runs a triage round 0 (segmentation-3.0 frame "
            "posteriors + Brouhaha/DSP SNR) and runs the enhanced pass only when degraded speech is "
            "found — and skips diarization/ASR/alignment/PPG entirely when no speech is found (FR-004); "
            "'never' ≡ --no-enhancement."
        ),
    )
    parser.add_argument(
        "--triage-speech-threshold",
        type=float,
        default=0.5,
        help="Triage: per-window P(speech) at/above which a ~100 ms window counts as speech.",
    )
    parser.add_argument(
        "--triage-min-speech-s",
        type=float,
        default=0.3,
        help="Triage: minimum total speech seconds for the run to proceed past round 0 (FR-004).",
    )
    parser.add_argument(
        "--triage-snr-floor-db",
        type=float,
        default=10.0,
        help="Triage: SNR (dB) below which a speech window counts as degraded.",
    )
    parser.add_argument(
        "--triage-low-snr-fraction",
        type=float,
        default=0.4,
        help="Triage: fraction of degraded speech windows at/above which the enhanced pass runs (FR-003).",
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
            # CrisperWhisper 2.0 turbo — verbatim, word-level timestamps + native
            # per-word confidence, via the crisperwhisper CT2 subprocess venv.
            "nyralabs/CrisperWhisper2.0_turbo",
            # Text-only NeMo SALM (subprocess venv); auto-aligned downstream.
            "nvidia/canary-qwen-2.5b",
            # Native word timestamps via the bundled Qwen3-ForcedAligner companion
            # (subprocess venv). Per-model failures are captured in JSON, non-fatal.
            "Qwen/Qwen3-ASR-1.7B",
        ],
        help=(
            "ASR models. Defaults: CrisperWhisper 2.0 turbo (verbatim, native word "
            "timestamps + confidence), NVIDIA Canary-Qwen 2.5B (text-only, "
            "auto-aligned), and Qwen3-ASR 1.7B (native word timestamps via the "
            "bundled Qwen3-ForcedAligner companion). Timestamp-less ASR output is "
            "auto-aligned by the script; pass --no-align-asr to skip."
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
        "--scene-top-k",
        type=int,
        default=50,
        help=(
            "Number of AudioSet classes to persist per AST/YAMNet window (default: 50). "
            "Feeds the presence-axis sound-source category masses (speech/people/machine/"
            "environment); 50 captures essentially all of the softmax mass. Raise toward the "
            "full label count (527 AST / 521 YAMNet) for the complete distribution at ~10x the "
            "cache size; the top-1 label (speech-presence / YAMNet-veto) is unaffected either way."
        ),
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
            "sha256(audio_signature, task, model_id, params, code_version, "
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
        "--ppg",
        action="store_true",
        help=(
            "Run the PPG (phonetic posteriorgram) backend on each pass and feed it into "
            "the comparator's utterance axis as a per-frame phoneme-disagreement signal "
            "(`phoneme_per_to_ppg` per ASR vote). Off by default — enabling pulls the "
            "ppgs subprocess venv (~1.4 GB)."
        ),
    )
    parser.add_argument(
        "--aligner",
        choices=("qwen", "mms"),
        default="qwen",
        help=(
            "Forced-aligner backend for text-only ASR output (no native timestamps). "
            "'qwen' (default) uses Qwen3-ForcedAligner-0.6B in the qwen-asr subprocess "
            "venv; 'mms' uses the multilingual facebook/mms-1b-all path. ASR models "
            "with native timestamps (CrisperWhisper, Qwen3-ASR) are never re-aligned."
        ),
    )
    parser.add_argument(
        "--aligner-model",
        default="facebook/mms-1b-all",
        help=(
            "MMS forced-alignment model used when --aligner mms (default: "
            "facebook/mms-1b-all, 1100+ languages via per-language adapters)."
        ),
    )
    parser.add_argument(
        "--qwen-aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        help="Qwen forced-aligner model id used when --aligner qwen (default: Qwen/Qwen3-ForcedAligner-0.6B).",
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
        default=0.25,
        help=(
            "Window length (seconds) for the identity axis / cross-stream / within-stream "
            "comparisons. Default 0.25 s overlapping — fine enough to localize sub-second "
            "speaker turns (multi-speaker clips routinely have 0.3-1 s turns). Presence has "
            "its own finer grid (``--presence-grid-*``) and utterance its own wider one "
            "(``--utterance-win-length``)."
        ),
    )
    parser.add_argument(
        "--cross-stream-hop-length",
        type=float,
        default=0.25,
        help="Hop between identity/cross-stream windows (default 0.25 s; must be <= win-length).",
    )
    parser.add_argument(
        "--utterance-win-length",
        type=float,
        default=1.0,
        help=(
            "Window length (seconds) for the utterance axis. Defaults to 1.0 s — wider "
            "than the presence/identity grid because most words don't fit inside a 0.5 s "
            "window. Combined with the 0.5 s hop default, every word lands fully inside "
            "at least one bucket."
        ),
    )
    parser.add_argument(
        "--utterance-hop-length",
        type=float,
        default=0.5,
        help=(
            "Hop between utterance windows (default 0.5 s, half the default win — "
            "windows overlap so words straddling a 0.5 s boundary still land inside "
            "at least one bucket). Must be <= --utterance-win-length."
        ),
    )
    parser.add_argument(
        "--no-scene-quality",
        action="store_true",
        help=(
            "Disable the scene-quality signals (Brouhaha SNR/C50 + DSP clipping/bandwidth). "
            "By default scene quality is REQUIRED: if Brouhaha cannot be loaded (e.g. gated "
            "access not granted) the run fails loudly rather than silently emitting null "
            "quality columns. Pass this flag to intentionally run without it."
        ),
    )
    parser.add_argument(
        "--no-sound-sources",
        action="store_true",
        help="Disable the background sound-source category masses (speech/people/machine/environment).",
    )
    parser.add_argument(
        "--utterance-scene-coupling-weights",
        type=float,
        nargs=2,
        metavar=("W_Q", "W_S"),
        default=(0.5, 0.25),
        help=(
            "Scene-to-utterance coupling weights (FR-019): reported utterance uncertainty is "
            "multiplied by 1 + W_Q * quality_snr + W_S * (src_machine + src_environment) over the "
            "bucket's span, clipped to 1.0. The multiplier is recorded in the "
            "scene_quality_coupling column and the pre-coupling value stays in "
            "raw_aggregated_uncertainty. Defaults to 0.5 0.25; pass '0 0' to disable coupling."
        ),
    )
    parser.add_argument(
        "--presence-grid-win-length",
        type=float,
        default=0.1,
        help=(
            "Window length (seconds) for the presence axis (default 0.1 s ≈ one phone). "
            "Presence uses continuous frame posteriors, so a fine grid localizes brief "
            "events (cough onset, inter-word breath) that a 0.5 s grid smears. Quality and "
            "source columns are broadcast onto this grid."
        ),
    )
    parser.add_argument(
        "--presence-grid-hop-length",
        type=float,
        default=0.02,
        help="Hop between presence windows (default 0.02 s ≈ frame hop). Must be <= --presence-grid-win-length.",
    )
    parser.add_argument(
        "--embedding-window-s",
        type=float,
        default=0.5,
        help=(
            "Window length (seconds) for fixed-grid speaker-embedding extraction. "
            "Defaults to 0.5 s: on conversational multi-speaker audio with short "
            "turns, a 0.5 s window recovers the correct speaker count and roughly "
            "doubles cluster-vs-truth agreement (ARI 0.70 vs 0.48 at 1.0 s on the "
            "4-speaker validation clip) because 1.0 s windows straddle turn "
            "boundaries and smear adjacent speakers together. The trade-off is that "
            "turns shorter than the window (< 0.5 s) may not resolve as their own "
            "cluster; raise toward 1.0 s for clean, long-form single/dual-speaker "
            "audio where per-embedding robustness matters more than turn resolution."
        ),
    )
    parser.add_argument(
        "--embedding-hop-s",
        type=float,
        default=0.25,
        help=(
            "Hop between embedding windows (default 0.25 s). A 0.25 s hop samples the "
            "0.5 s window densely so speaker-change boundaries localize to ~0.25 s; on "
            "the 4-speaker validation clip this flips the identity axis from inverted "
            "(uncertainty dipping at speaker changes) to correct (peaking within ~15 ms "
            "of the two clearest turn boundaries). Must be <= --embedding-window-s."
        ),
    )
    parser.add_argument(
        "--identity-same-speaker-floor",
        type=float,
        default=0.30,
        help=(
            "Cosine distance ≤ this is treated as confidently same-speaker for the "
            "identity axis (uncertainty 0 for same-claim, 1 for change-claim). "
            "Defaults to 0.30 — typical ECAPA same-speaker noise level on VoxCeleb."
        ),
    )
    parser.add_argument(
        "--identity-diff-speaker-floor",
        type=float,
        default=0.70,
        help=(
            "Cosine distance ≥ this is treated as confidently different-speaker for "
            "the identity axis. Defaults to 0.70. Distances between the two floors "
            "interpolate linearly. Must be > --identity-same-speaker-floor."
        ),
    )
    parser.add_argument(
        "--identity-cluster-cosine-threshold",
        type=float,
        default=0.5,
        help=(
            "Cosine similarity threshold for clustering (diar_model, raw_label) into "
            "pass-wide speaker IDs. Used to recognize that pyannote 'SPEAKER_00' and "
            "sortformer 'speaker_2' refer to the same person when their mean "
            "embeddings are close. Defaults to 0.5 (~ECAPA EER on VoxCeleb)."
        ),
    )
    parser.add_argument(
        "--clustering-algorithm",
        choices=["spectral", "kmeans"],
        default="spectral",
        help=(
            "Clustering algorithm for the windowed speaker-embedding step that "
            "estimates n_speakers per pass. 'spectral' (default) uses a precomputed "
            "cosine-similarity affinity, which handles non-convex speaker clusters "
            "better than k-means; 'kmeans' is the legacy choice. Spectral falls back "
            "to k-means automatically if a k fails."
        ),
    )
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        default=None,
        help=(
            "Scene-quality calibration profile JSON (US5, data-model §5): dB→[0,1] anchors for "
            "SNR/C50 plus per-axis aggregator temperatures. Default: the bundled profile "
            "(workflows/audio_analysis/data/scene_quality_calibration.json) when present, else the "
            "documented uncalibrated defaults. Fit one with scripts/calibrate_scene_quality.py."
        ),
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
        nargs="+",
        default=list(DEFAULT_SPEECH_PRESENCE_LABELS),
        metavar="LABEL",
        help=(
            "AudioSet labels (one per arg) that count as 'speech-present' for AST/YAMNet ↔ "
            "diarization comparison. AudioSet labels themselves contain commas "
            "(e.g. 'Narration, monologue'), so use space-separated quoted args rather than a "
            "single comma string. Default covers the AudioSet 'Speech' subtree."
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
    if args.utterance_win_length <= 0:
        parser.error("--utterance-win-length must be positive")
    if args.utterance_hop_length <= 0 or args.utterance_hop_length > args.utterance_win_length:
        parser.error("--utterance-hop-length must be positive and ≤ --utterance-win-length")
    if args.presence_grid_win_length <= 0:
        parser.error("--presence-grid-win-length must be positive")
    if args.presence_grid_hop_length <= 0 or args.presence_grid_hop_length > args.presence_grid_win_length:
        parser.error("--presence-grid-hop-length must be positive and ≤ --presence-grid-win-length")
    if not (0.0 <= args.phoneme_disagreement_threshold <= 1.0):
        parser.error("--phoneme-disagreement-threshold must be in [0, 1]")
    if args.diarization_boundary_shift_ms < 0:
        parser.error("--diarization-boundary-shift-ms must be non-negative")
    if args.disagreements_top_n < 0:
        parser.error("--disagreements-top-n must be non-negative")
    return args


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

# Cache schema. ``_sync_cache_with_schema_version`` keeps the on-disk marker
# (.schema_version inside the cache dir) in lockstep with this constant: any
# mismatch wipes the cache so we never serve a stale entry under a new schema.
# Bump (or reset) on any breaking change to cache key composition or to the
# shape of cached output. The current value is bundled into every cache key
# (see ``cache_key``) and stamped into parquet provenance via
# ``_CACHE_SCHEMA_VERSION`` references — never hardcode the literal anywhere
# else, otherwise the constant and the stamped value will drift.


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
    windows = _classification_windows(result)
    for i, window in enumerate(windows):
        label, score, _entropy = _classification_window_top1(window)
        if label is None:
            continue
        # Prefer per-window start/end when the canonical shape carries them.
        if isinstance(window, dict) and window.get("start") is not None and window.get("end") is not None:
            start = float(window["start"])
            end = float(window["end"])
        else:
            start = i * hop_length
            end = start + win_length
        out.append(
            _ls_label_region(
                region_id=_new_region_id(f"{prefix}_cls", i),
                from_name=prefix,
                start=start,
                end=end,
                label=label,
                score=score if score is not None else 0.0,
            )
        )
    return out


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
            from_name = f"{pass_label}__diarization__{safe_model_id(model_id)}"
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
        from_name = f"{pass_label}__asr__{safe_model_id(model_id)}"
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


def build_labelstudio_config(summary: dict[str, Any]) -> str:
    """Build a Label Studio labeling-config XML matching this run's per-task tracks.

    Generates one ``<Labels>`` control per (pass, analyzer, model) and one
    ``<TextArea>`` control per (pass, asr_model). Speakers, scene labels,
    and transcripts each become a stacked timeline annotation row.

    The three-axis uncertainty tracks are appended downstream by
    ``senselab.audio.workflows.audio_analysis.attach_uncertainty_tracks_to_ls``.
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
            seen_label_sets[f"{pass_label}__diarization__{safe_model_id(model_id)}"] = speakers

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
            from_name = f"{pass_label}__asr__{safe_model_id(model_id)}"
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

    parts.append("</View>")
    return "\n".join(parts) + "\n"


def _collect_classification_labels(result: Any) -> set[str]:  # noqa: ANN401
    """Extract the union of label strings observed in a classify_audios output."""
    labels: set[str] = set()
    for window in _classification_windows(result):
        if not isinstance(window, dict):
            continue
        for label in window.get("labels") or []:
            if label:
                labels.add(str(label))
    return labels


def _speech_presence_labels(args: argparse.Namespace) -> list[str]:
    """Resolve --speech-presence-labels into a clean list of AudioSet labels.

    Argparse ``nargs="+"`` always yields a list of strings; AudioSet labels themselves
    contain commas (e.g. ``"Narration, monologue"``) which is why the flag is space-
    separated rather than comma-joined.
    """
    return [str(s).strip() for s in args.speech_presence_labels if str(s).strip()]


_KNOWN_NULL_CONFIDENCE_MODEL_PREFIXES = (
    "pyannote/speaker-diarization",
    "nvidia/diar_sortformer",
    "ibm-granite/granite-speech",
    "nvidia/canary-qwen",
    "Qwen/Qwen3-ASR",
)


def _models_without_native_signal(summaries: dict[str, Any]) -> list[str]:
    """Return the documented set of models that do not expose a per-region confidence.

    Used by the disagreements.json builder to log which contributors fall back on
    cross-model entropy rather than a native scalar.
    """
    seen: set[str] = set()
    for pass_summary in (summaries.get("passes") or {}).values():
        if not isinstance(pass_summary, dict):
            continue
        for task in ("diarization", "asr"):
            block = (pass_summary.get(task) or {}).get("by_model") or {}
            for model_id in block:
                if any(model_id.startswith(prefix) for prefix in _KNOWN_NULL_CONFIDENCE_MODEL_PREFIXES):
                    seen.add(model_id)
    return sorted(seen)


def run_triage(audio: Audio, args: argparse.Namespace, device: DeviceType | None) -> dict[str, Any]:
    """Round 0 (spec US1): frame-posterior speech gate + SNR enhancement gate.

    Uses continuous segmentation-3.0 frame posteriors (never segmentized VAD —
    see SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md) and Brouhaha SNR with an ungated
    percentile-DSP fallback. Degrades conservatively: missing posteriors ⇒
    ``speech_present=True``; missing SNR ⇒ ``needs_enhancement=None`` (the
    caller treats unknown as "run the enhanced pass").
    """
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (
        extract_speech_frame_posteriors,
    )
    from senselab.audio.workflows.audio_analysis.adaptive.triage import dsp_snr_series, triage_decision

    t0 = time.time()
    posterior = extract_speech_frame_posteriors([audio], device=device)[0]

    snr_db: list[float] | None = None
    snr_hop_s: float | None = None
    snr_estimator: str | None = None
    try:
        from senselab.audio.tasks.scene_quality.brouhaha import extract_brouhaha_frames

        brouhaha = extract_brouhaha_frames([audio], device=device)[0]
        if brouhaha is not None:
            snr_db = [float(v) for v in brouhaha.snr_db]
            snr_hop_s = float(brouhaha.frame_hop_s)
            snr_estimator = "brouhaha"
    except Exception as exc:  # noqa: BLE001 — SNR is optional; DSP fallback below
        print(f"  [triage] brouhaha unavailable ({exc!r}); falling back to DSP SNR", file=sys.stderr)
    if snr_db is None:
        waveform = audio.waveform.squeeze().detach().cpu().numpy()
        snr_db, snr_hop_s = dsp_snr_series(
            waveform,
            audio.sampling_rate,
            p_speech=[float(p) for p in posterior.probs] if posterior is not None else None,
            p_hop_s=float(posterior.frame_hop_s) if posterior is not None else None,
        )
        snr_estimator = "dsp_posterior_masked" if posterior is not None else "dsp_percentile"

    if posterior is None:
        decision: dict[str, Any] = {
            "speech_present": True,
            "needs_enhancement": None,
            "inconclusive": True,
            "reason": "frame_posteriors_unavailable",
            "stats": {},
            "thresholds": {},
        }
    else:
        decision = triage_decision(
            p_speech=[float(p) for p in posterior.probs],
            frame_hop_s=float(posterior.frame_hop_s),
            snr_db=snr_db,
            snr_hop_s=snr_hop_s,
            speech_threshold=args.triage_speech_threshold,
            min_speech_s=args.triage_min_speech_s,
            snr_floor_db=args.triage_snr_floor_db,
            low_snr_fraction_threshold=args.triage_low_snr_fraction,
        )
    decision["snr_estimator"] = snr_estimator
    decision["elapsed_s"] = round(time.time() - t0, 3)
    return decision


def _pass_plan(args: argparse.Namespace) -> PassPlan:
    """Translate parsed CLI args into a library `PassPlan`.

    Called *after* triage, because `main` mutates `args.skip` / `args.ppg` on the
    no-speech path — a plan built before that would run diarization and ASR on
    silence. Absence-means-skip is expressed here so the library never sees a
    CLI-shaped skip set.
    """
    skip = set(args.skip)
    return PassPlan(
        diarization_models=() if "diarization" in skip else tuple(args.diarization_models),
        asr_models=() if "asr" in skip else tuple(args.asr_models),
        ast_model=None if "ast" in skip else args.ast_model,
        yamnet_model=None if "yamnet" in skip else args.yamnet_model,
        ast_win_length=args.ast_win_length,
        ast_hop_length=args.ast_hop_length,
        yamnet_win_length=args.yamnet_win_length,
        yamnet_hop_length=args.yamnet_hop_length,
        scene_top_k=args.scene_top_k,
        features="features" not in skip,
        features_win_length=args.features_win_length,
        features_hop_length=args.features_hop_length,
        align_asr=not args.no_align_asr,
        aligner=args.aligner,
        qwen_aligner_model=args.qwen_aligner_model,
        mms_aligner_model=args.aligner_model,
        asr_language=args.asr_language,
        qwen_native_timestamps=not args.qwen_asr_no_timestamps,
        ppg=bool(args.ppg),
    )


def _policy_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build in-memory policy overrides from the adaptive CLI flags.

    `None` values are dropped by `load_policy`, so an unset flag leaves the policy
    file's value alone — the flags override, they don't reset to a CLI default.
    """
    overrides: dict[str, Any] = {
        "budget": {"medium_per_run": args.budget_medium, "heavy_per_run": args.budget_heavy},
        "regions": {"top_n_per_round": args.region_top_n, "max_region_rounds": args.max_region_rounds},
    }
    if args.reserve_asr_models is not None:
        overrides["reserve_asr_models"] = list(args.reserve_asr_models)
    if args.enable_overlap_separation:
        overrides["rules"] = {"I4_overlap_detection": {"enabled": True}}
    return overrides


def _stage_context(
    label: str,
    audio: Audio,
    args: argparse.Namespace,
    *,
    device: DeviceType | None,
    out_dir: Path,
    cache_dir: Path | None,
    senselab_ver: str,
) -> StageContext:
    """Build the per-pass `StageContext` from CLI args."""
    return StageContext(
        pass_label=label,
        audio_signature=audio_signature(audio),
        device=device,
        cache_dir=cache_dir,
        out_dir=out_dir / label,
        audio_source=str(args.audio.resolve()),
        senselab_ver=senselab_ver,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the full analysis with and without enhancement."""
    args = parse_args(argv)

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    device_label = device.value if device is not None else "auto (per-task selection)"
    cache_dir: Path | None = None if args.no_cache else args.cache_dir.resolve()
    if cache_dir is not None:
        _sync_cache_with_schema_version(cache_dir)
    senselab_ver = senselab_version()
    print(f"Device: {device_label}")
    print(f"Input:  {args.audio}")
    if cache_dir is not None:
        print(f"Cache:  {cache_dir}")
        print(f"        key = sha256(audio | task | model | params | stage_version | senselab={senselab_ver})")
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
        "stage_versions": dict(STAGE_VERSIONS),
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

    # ── Round 0: triage (spec US1; FR-002/003/004) ──────────────────────
    enhancement_mode = "never" if args.no_enhancement else args.enhancement
    triage: dict[str, Any] | None = None
    if enhancement_mode == "auto":
        print("\n=== Triage (round 0): frame posteriors + SNR ===")
        triage = run_triage(audio_16k, args, device)
        write_json(run_dir / "triage.json", triage)
        summaries["triage"] = triage
        stats = triage.get("stats") or {}
        print(
            f"  speech_present={triage['speech_present']} "
            f"(speech_s={stats.get('speech_s')}, fraction={stats.get('speech_fraction')})  "
            f"needs_enhancement={triage['needs_enhancement']} "
            f"(snr={triage.get('snr_estimator')}, median_snr={stats.get('median_snr_db_in_speech')} dB)"
        )
        if not triage["speech_present"]:
            summaries["run_state"] = "no_speech"
            args.skip = tuple(sorted(set(args.skip) | {"diarization", "asr", "alignment"}))
            args.ppg = False
            print("  no speech found — skipping diarization/ASR/alignment/PPG; presence outputs still emitted (FR-004)")
    run_enhanced_pass = enhancement_mode == "always" or (
        enhancement_mode == "auto"
        and triage is not None
        and triage["speech_present"]
        and triage["needs_enhancement"] is not False  # unknown SNR ⇒ conservative: run it
    )

    pass_audio: dict[str, Audio] = {"raw_16k": audio_16k}

    pass_plan = _pass_plan(args)
    summaries["passes"]["raw_16k"] = run_pass(
        audio_16k,
        _stage_context(
            "raw_16k",
            audio_16k,
            args,
            device=device,
            out_dir=run_dir,
            cache_dir=cache_dir,
            senselab_ver=senselab_ver,
        ),
        pass_plan,
    )

    if run_enhanced_pass:
        print("\n=== Enhancing audio (this loads the enhancement model)... ===")
        try:
            enhanced = enhance_audios(
                [audio_16k],
                model=model_for_task(args.enhancement_model, task="enhancement"),
                device=device,
            )[0]
            pass_audio["enhanced_16k"] = enhanced
            summaries["passes"]["enhanced_16k"] = run_pass(
                enhanced,
                _stage_context(
                    "enhanced_16k",
                    enhanced,
                    args,
                    device=device,
                    out_dir=run_dir,
                    cache_dir=cache_dir,
                    senselab_ver=senselab_ver,
                ),
                pass_plan,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Enhancement failed: {exc!r}", file=sys.stderr)
            summaries["passes"]["enhanced_16k"] = {"status": "failed", "error": repr(exc)}

    write_json(run_dir / "summary.json", summaries)

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
    config_xml = build_labelstudio_config(summaries)

    # ── Comparator: three-axis uncertainty workflow ─────────────────────
    if "comparisons" in args.skip and args.max_rounds > 1:
        print(
            "warn: --skip comparisons disables the belief store, so no adaptive rounds >= 2 and no "
            "final/ will be produced (contracts/cli.md). Drop --skip comparisons to enable them.",
            file=sys.stderr,
        )

    if "comparisons" not in args.skip:
        from senselab.audio.workflows.audio_analysis import (
            BucketGrid,
            attach_uncertainty_tracks_to_ls,
            build_aligned_timeline_plot,
            build_disagreements_index,
            compute_uncertainty_axes,
            write_axis_parquet,
        )

        grid = BucketGrid(
            win_length=args.cross_stream_win_length,
            hop_length=args.cross_stream_hop_length,
        )
        utterance_grid = BucketGrid(
            win_length=args.utterance_win_length,
            hop_length=args.utterance_hop_length,
        )
        presence_grid = BucketGrid(
            win_length=args.presence_grid_win_length,
            hop_length=args.presence_grid_hop_length,
        )
        comparator_params = {
            "win_length": grid.win_length,
            "hop_length": grid.hop_length,
            "utterance_win_length": utterance_grid.win_length,
            "utterance_hop_length": utterance_grid.hop_length,
            "presence_win_length": presence_grid.win_length,
            "presence_hop_length": presence_grid.hop_length,
            "aggregator": args.uncertainty_aggregator,
            "phoneme_disagreement_threshold": args.phoneme_disagreement_threshold,
            "speech_presence_labels": _speech_presence_labels(args),
            "asr_reference_model": args.asr_reference_model,
            "diarization_boundary_shift_ms": args.diarization_boundary_shift_ms,
            "clustering_algorithm": args.clustering_algorithm,
            "utterance_scene_coupling": {
                "w_q": float(args.utterance_scene_coupling_weights[0]),
                "w_s": float(args.utterance_scene_coupling_weights[1]),
            },
        }
        # US5 (T039): load + validate the calibration profile and thread its flat
        # runtime form into BOTH consumers — the harvest (quality dB→[0,1] anchors,
        # via the calibration kwarg below) and the aggregators (temperatures /
        # token-entropy reference, via comparator_params["calibration"]).
        from senselab.audio.workflows.audio_analysis.calibration import (
            load_calibration_profile,
            profile_to_runtime,
        )

        try:
            calibration_runtime = profile_to_runtime(load_calibration_profile(args.calibration_profile))
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: invalid calibration profile {args.calibration_profile}: {exc}", file=sys.stderr)
            sys.exit(2)
        comparator_params["calibration"] = calibration_runtime

        passes_for_compute = {
            pl: ps for pl, ps in summaries.get("passes", {}).items() if isinstance(ps, dict) and "duration_s" in ps
        }
        speaker_embedding_models = list(args.embeddings_models)
        per_window_embeddings_by_pass: dict[str, dict[str, Any]] = {}
        try:
            harvests_by_pass: dict[str, Any] = {}
            axis_results, incomparable_reasons, per_window_embeddings_by_pass = compute_uncertainty_axes(
                harvests_out=harvests_by_pass,
                passes=passes_for_compute,
                grid=grid,
                params=comparator_params,
                audio=pass_audio,
                speaker_embedding_models=speaker_embedding_models,
                aggregator=args.uncertainty_aggregator,
                speech_presence_labels=_speech_presence_labels(args),
                utterance_grid=utterance_grid,
                presence_grid=presence_grid,
                scene_quality=not args.no_scene_quality,
                sound_sources=not args.no_sound_sources,
                calibration=calibration_runtime,
                embedding_window_s=args.embedding_window_s,
                embedding_hop_s=args.embedding_hop_s,
                same_speaker_floor=args.identity_same_speaker_floor,
                diff_speaker_floor=args.identity_diff_speaker_floor,
                cluster_cosine_threshold=args.identity_cluster_cosine_threshold,
                clustering_algorithm=args.clustering_algorithm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: comparator workflow failed: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            axis_results, incomparable_reasons = ({}, {"workflow": f"failed: {exc!r}"})
            per_window_embeddings_by_pass = {}

        # Scene quality is a REQUIRED signal here unless explicitly disabled: the
        # library degrades gracefully to null (FR-023) for reuse, but this script
        # must not silently ship a run missing its quality columns. If the model
        # was requested but unavailable on any pass, fail loudly with guidance.
        if not args.no_scene_quality:
            unavailable_passes = [
                pl
                for (pl, axis), result in axis_results.items()
                if axis == "presence"
                and pl != "raw_vs_enhanced"
                and (result.provenance.get("scene_quality") or {}).get("model", {}).get("available") is False
            ]
            if unavailable_passes:
                print(
                    "ERROR: scene-quality model (pyannote/brouhaha) could not be loaded for "
                    f"pass(es) {sorted(unavailable_passes)}, so SNR/reverb quality columns would be null. "
                    "Ensure the model is accessible (request access at https://hf.co/pyannote/brouhaha, "
                    "set HF_TOKEN) and its backend is installed, or pass --no-scene-quality to run "
                    "without scene quality intentionally.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Persist 9 parquets (3 axes × 2 passes + 3 raw_vs_enhanced deltas).
        for (pass_label, axis), result in axis_results.items():
            if pass_label == "raw_vs_enhanced":
                dest = run_dir / "uncertainty" / "raw_vs_enhanced" / f"{axis}.parquet"
            else:
                dest = run_dir / pass_label / "uncertainty" / f"{axis}.parquet"
            write_axis_parquet(
                result,
                dest,
                provenance={
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "stage_versions": dict(STAGE_VERSIONS),
                    "senselab_version": senselab_ver,
                },
            )

        # PII detection per pass — scans each ASR transcript with regex layer
        # plus optional spaCy NER. Default-on; failures (e.g. spaCy not
        # installed) are surfaced via stderr + the report's failures dict.
        from senselab.audio.workflows.audio_analysis.global_summary import (
            compute_pass_global_summary,
        )
        from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result
        from senselab.audio.workflows.audio_analysis.pii import detect_pii_in_pass, report_to_dict

        pii_reports: dict[str, Any] = {}
        for pl, ps in passes_for_compute.items():
            align_by_model_pii = (ps.get("alignment") or {}).get("by_model") or {}
            asr_resolved_pii: dict[str, Any] = {}
            for m, b in ((ps.get("asr") or {}).get("by_model") or {}).items():
                if isinstance(b, dict) and b.get("status") == "ok":
                    asr_resolved_pii[m] = resolve_asr_result(b, align_by_model_pii.get(m))
            pii_reports[pl] = detect_pii_in_pass(
                pass_label=pl,
                asr_resolved=asr_resolved_pii,
            )
            write_json(run_dir / pl / "pii.json", report_to_dict(pii_reports[pl]))

        # Global per-pass summary: 4 claims (transcript / speaker / quality / PII)
        # → 1 scalar each + a max() combined. Persist to summary.json.
        global_pass_summaries: dict[str, Any] = {}
        for pl, ps in passes_for_compute.items():
            align_by_model_g = (ps.get("alignment") or {}).get("by_model") or {}
            asr_resolved_g: dict[str, Any] = {}
            for m, b in ((ps.get("asr") or {}).get("by_model") or {}).items():
                if isinstance(b, dict) and b.get("status") == "ok":
                    asr_resolved_g[m] = resolve_asr_result(b, align_by_model_g.get(m))
            global_pass_summaries[pl] = compute_pass_global_summary(
                pass_label=pl,
                pass_summary=ps,
                axis_results=axis_results,
                asr_resolved=asr_resolved_g,
                pii_report=pii_reports.get(pl),
                expects_speech=True,
            )
        # Top-level: pick the lower-uncertainty pass (best of raw vs enhanced)
        # so the bottom-line score reflects the cleaner interpretation.
        best_pass: str | None = None
        best_combined: float | None = None
        for pl, gs in global_pass_summaries.items():
            c = gs.get("combined_uncertainty")
            if c is None:
                continue
            if best_combined is None or c < best_combined:
                best_combined = c
                best_pass = pl
        summaries["global_uncertainty"] = {
            "combined_uncertainty": best_combined,
            "best_pass": best_pass,
            "by_pass": global_pass_summaries,
            "incomparable_reasons": incomparable_reasons,
        }
        # Re-persist summary.json — the original write at line 1782 happened
        # before the comparator stage so it does not contain
        # ``global_uncertainty``. Overwriting here keeps the on-disk summary
        # in sync with the in-memory dict.
        write_json(run_dir / "summary.json", summaries)

        # Persist per-pass windowed speaker embeddings — one JSON per (pass, model)
        # at ``<pass>/embeddings/<model>.json`` with the full window grid + vectors.
        for pass_label, by_model in per_window_embeddings_by_pass.items():
            if not by_model:
                continue
            for model_id, windows in by_model.items():
                payload = {
                    "status": "ok" if windows else "no_data",
                    "window_s": args.embedding_window_s,
                    "hop_s": args.embedding_hop_s,
                    "windows": [
                        {
                            "start_s": float(w.start_s),
                            "end_s": float(w.end_s),
                            "vector": [float(x) for x in w.vector.tolist()],
                        }
                        for w in windows
                    ],
                }
                write_json(run_dir / pass_label / "embeddings" / f"{safe_model_id(model_id)}.json", payload)

        # Attach per-axis Labels + utterance TextArea tracks to the LS bundle.
        if axis_results:
            ls_tasks, config_xml = attach_uncertainty_tracks_to_ls(
                ls_tasks=ls_tasks,
                ls_config=config_xml,
                axis_results=axis_results,
            )

        # Disagreements index — opt-out via --disagreements-top-n 0.
        if axis_results and args.disagreements_top_n > 0:
            index = build_disagreements_index(
                axis_results=axis_results,
                top_n=args.disagreements_top_n,
                run_dir=run_dir,
                config={
                    "top_n": args.disagreements_top_n,
                    "aggregator": args.uncertainty_aggregator,
                    "phoneme_disagreement_threshold": args.phoneme_disagreement_threshold,
                    "bucket_grid": {
                        "win_length": grid.win_length,
                        "hop_length": grid.hop_length,
                    },
                    "speech_presence_labels": _speech_presence_labels(args),
                    "stage_versions": dict(STAGE_VERSIONS),
                    "senselab_version": senselab_ver,
                },
                incomparable_reasons=incomparable_reasons,
                models_without_native_signal=_models_without_native_signal(summaries),
            )
            write_json(run_dir / "disagreements.json", index)

        # Timeline plot — best-effort sidecar.
        if axis_results:
            try:
                duration_s = float(next(iter(passes_for_compute.values())).get("duration_s", 0.0) or 0.0)
                # Build per-pass detail bundles for the plot's per-source rows.
                detail_by_pass: dict[str, dict[str, Any]] = {}
                for pass_label, pass_summary in passes_for_compute.items():
                    align_by_model = ((pass_summary.get("alignment") or {}).get("by_model")) or {}
                    diar_by_model: dict[str, list[Any]] = {}
                    for m, block in ((pass_summary.get("diarization") or {}).get("by_model") or {}).items():
                        if isinstance(block, dict) and block.get("status") == "ok":
                            res = block.get("result")
                            if isinstance(res, list) and res:
                                inner = res[0] if isinstance(res[0], list) else res
                                diar_by_model[m] = list(inner)
                    asr_by_model: dict[str, Any] = {}
                    for m, block in ((pass_summary.get("asr") or {}).get("by_model") or {}).items():
                        if not (isinstance(block, dict) and block.get("status") == "ok"):
                            continue
                        from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

                        asr_by_model[m] = resolve_asr_result(block, align_by_model.get(m))
                    ppg_block = pass_summary.get("ppgs") or {}
                    ppg_per_frame: list[str] = []
                    ppg_frame_hop = 0.0
                    if isinstance(ppg_block, dict) and ppg_block.get("status") == "ok":
                        from senselab.audio.workflows.audio_analysis.harvesters import ppg_argmax_per_frame

                        ppg_per_frame, ppg_frame_hop = ppg_argmax_per_frame(
                            ppg_block.get("result"),
                            ppg_block.get("phoneme_labels"),
                            float(pass_summary.get("duration_s", 0.0) or 0.0),
                        )
                    detail_by_pass[pass_label] = {
                        "diar_by_model": diar_by_model,
                        "asr_by_model": asr_by_model,
                        "per_window_embeddings": per_window_embeddings_by_pass.get(pass_label, {}),
                        "ppg": {
                            "per_frame_phonemes": ppg_per_frame,
                            "frame_hop": ppg_frame_hop,
                        },
                    }
                raw_pass_audio = pass_audio.get("raw_16k")
                raw_waveform = (
                    raw_pass_audio.waveform.detach().cpu().numpy().squeeze() if raw_pass_audio is not None else None
                )
                raw_sr = int(raw_pass_audio.sampling_rate) if raw_pass_audio is not None else 16000
                timeline_path = build_aligned_timeline_plot(
                    run_dir=run_dir,
                    axis_results=axis_results,
                    duration_s=duration_s,
                    grid_hop=grid.hop_length,
                    utterance_grid_hop=utterance_grid.hop_length,
                    detail_by_pass=detail_by_pass,
                    title=f"Aggregate uncertainty · {args.audio.name}",
                    audio_waveform=raw_waveform,
                    audio_sr=raw_sr,
                )
                if timeline_path is not None:
                    print(f"Timeline plot: {timeline_path}")
            except Exception as exc:  # noqa: BLE001 — best-effort sidecar
                print(f"warn: timeline plot failed: {exc!r}", file=sys.stderr)

        # ── Adaptive loop, in-process (T040) ──────────────────────────
        # Runs on the harvests the parquets were just built from, so the belief
        # store needs no parquet round-trip. Gated on --no-adaptive-outputs; a
        # --max-rounds 1 run still emits final/ from the round-1 belief.
        if not args.no_adaptive_outputs and harvests_by_pass:
            from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop

            try:
                adaptive_log = run_adaptive_loop(
                    run_dir,
                    cache_dir=cache_dir,
                    policy_path=args.policy,
                    out_dir=run_dir,
                    max_rounds=args.max_rounds,
                    aggregator=args.uncertainty_aggregator,
                    harvests=harvests_by_pass,
                    summary=summaries,
                    policy_overrides=_policy_overrides(args),
                )
                summaries["adaptive"] = {
                    "enabled": True,
                    "max_rounds": args.max_rounds,
                    "policy": str(args.policy) if args.policy else "packaged default",
                    "policy_hash": adaptive_log.get("policy_hash"),
                    "rounds": adaptive_log.get("rounds"),
                    "run_state": adaptive_log.get("run_state"),
                    "n_interventions_fired": adaptive_log.get("n_interventions_fired"),
                    "n_words_fused": adaptive_log.get("n_words_fused"),
                    "parity_check": (adaptive_log.get("parity_check") or {}).get("status", "checked"),
                    "ingest": "in_process_harvests",
                }
                print(f"Adaptive: {run_dir / 'final'} ({summaries['adaptive'].get('run_state')})")
            except Exception as exc:  # noqa: BLE001 — additive artifacts must not fail the run
                print(f"warn: adaptive loop failed: {exc!r}", file=sys.stderr)
                summaries["adaptive"] = {"enabled": True, "status": "failed", "error": repr(exc)}
            write_json(run_dir / "summary.json", summaries)
        elif args.no_adaptive_outputs:
            summaries["adaptive"] = {"enabled": False, "reason": "--no-adaptive-outputs"}
            write_json(run_dir / "summary.json", summaries)

    write_json(run_dir / "labelstudio_tasks.json", ls_tasks)
    (run_dir / "labelstudio_config.xml").write_text(config_xml, encoding="utf-8")

    print(f"\nDone. Summary: {run_dir / 'summary.json'}")
    print(f"Label Studio tasks:  {run_dir / 'labelstudio_tasks.json'}")
    print(f"Label Studio config: {run_dir / 'labelstudio_config.xml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
