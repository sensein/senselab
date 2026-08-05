#!/usr/bin/env python3
r"""Analyze one audio file with the full senselab task suite.

    uv run python scripts/analyze_audio.py <audio> [--out <dir>]

Two arguments, and one optional third (``--config``). Everything else — model ids, the bucket grid,
window and hop, the aggregator, the task type, the triage and enhancement gates, the ASR set, the
aligner backend, which stages run — lives in one versioned file with its derivation written beside
each value:

    src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml

There were seventy flags. They are gone deliberately, not for tidiness: the run recipes in this
repo's own docs differed from one another only in flags whose right value a reader had no basis to
choose, and the *shipped defaults* of the four grid flags put the four uncertainty axes on four
different spacings — 242 / 242 / 19 / 8 rows sharing zero bucket keys — so every cross-axis coupling
in the pipeline ran and did nothing. A knob nobody can choose between settings for is an unmeasured
decision with a public interface. To change a value, write a YAML with just that key and pass
``--config``; it deep-merges over the packaged one and the merged hash is stamped on every artifact.

What the run does, per pass: resample to 16 kHz mono, then diarization, AST and YAMNet scene
classification, multi-backend feature extraction (incl. torchaudio-squim), ASR with auto-alignment
for text-only backends, speaker embeddings, the background mask, and the three-axis uncertainty
comparator. There are two passes — the recording as-is and the same audio after speech enhancement —
because they are the same recording under a transform and therefore a *perturbation sample*: every
signal's fusion weight is measured from how far its answer moves between them.

Layout under ``--out/<stem>_<utc-timestamp>/``:

    L1/<pass>/signals/<signal>.parquet   per-signal measurements, native units, no axis anywhere
    L1/stability/<signal>.parquet        cross-pass |delta| per bucket — what the two passes bought
    L2/round<N>/uncertainty/<axis>.parquet   the fused axes, one fold per axis across every pass
    L2/round0/votes/<axis>.parquet       the linked evidence at vote level
    final/                               the deliverables: transcript, diarization, speakers,
                                         disagreements_resolved, timeline, LS bundle

Cache + provenance: every per-task outcome is stored under the config's ``cache.dir`` keyed by

    sha256(audio_signature || task || model_id || params ||
           stage_version || senselab_version || cache_schema_version)

The audio signature is the sha256 of the post-resample, post-downmix PCM samples plus sampling rate,
so two files with identical waveforms share cache entries regardless of container or filename. On a
hit the prior outcome is replayed verbatim and ``cache: "hit"`` is recorded; on a miss the task runs
and a full provenance block is written. Set ``cache.enabled: false`` to disable both. Bump
``_CACHE_SCHEMA_VERSION`` in this script when an output shape changes in a way that should invalidate
prior entries.

Install:
    uv sync --extra text --extra video --extra senselab-ai --extra nlp --extra pii --group dev
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
from senselab.audio.workflows.audio_analysis.labelstudio import (
    build_labelstudio_config,
    build_labelstudio_task,
)
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES, HARVESTED_AXES
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.layout import (
    belief_dir,
    derivatives_dir,
    final_dir,
    last_round,
    perturbation_dir,
    signals_dir,
)
from senselab.audio.workflows.audio_analysis.perturbations import (
    Perturbation,
)
from senselab.audio.workflows.audio_analysis.perturbations import (
    apply as apply_perturbation,
)
from senselab.audio.workflows.audio_analysis.perturbations import (
    identity as identity_perturbation,
)
from senselab.audio.workflows.audio_analysis.perturbations import (
    speech_enhancement as speech_enhancement_perturbation,
)
from senselab.audio.workflows.audio_analysis.perturbations import (
    write_register as write_perturbation_register,
)
from senselab.audio.workflows.audio_analysis.run_config import (
    DEFAULT_CONFIG_PATH,
    RunConfig,
    load_run_config,
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
from senselab.utils.data_structures.logging import logger
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI: an audio file, where results go, and optionally which config to run under.

    Three arguments, and the third is a whole file rather than a knob. Every value that used to be a
    flag is in ``data/run_config/default.yaml`` with its derivation, and ``--config`` deep-merges one
    YAML over it — so an override is a named, hashable object that travels into every artifact's
    provenance, instead of a shell line nobody kept.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every other value lives in the run config:\n"
            f"  {DEFAULT_CONFIG_PATH}\n"
            "Override with a YAML holding only the keys you are changing:\n"
            "  uv run python scripts/analyze_audio.py audio.wav --config my.yaml"
        ),
    )
    parser.add_argument("audio", type=Path, help="Input audio file (.wav, .flac, .mp3, ...)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for the run tree (default: the config's output_dir, artifacts/analyze_audio/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Run config YAML, deep-merged over the packaged default. Name only the keys you are "
            "changing; the merged mapping is hashed and its identity is recorded on every artifact."
        ),
    )
    return parser.parse_args(argv)



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


def _distinct_speaker_count(outcome: object) -> int | None:
    """Distinct speaker labels in one diarization outcome, or ``None`` if it did not run."""
    if not isinstance(outcome, dict) or outcome.get("status") != "ok":
        return None
    result = outcome.get("result")
    while isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list):
        return None
    labels = set()
    for seg in result:
        speaker = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        if speaker is not None:
            labels.add(str(speaker))
    return len(labels) or None


def _diarize_counts_for_probe(cfg: RunConfig) -> Callable[[Any, int], dict[str, int]]:
    """Return a callable that diarizes a waveform and reports each model's speaker count.

    Used only by ``stages.invariance_probe``. Built here rather than inline so the probe re-runs
    the *same* models the pass used, with the same settings — a probe against different
    settings would measure the settings rather than the model's invariance.
    """

    def run(waveform: np.ndarray, sampling_rate: int) -> dict[str, int]:
        from senselab.audio.workflows.audio_analysis.stages import model_for_task

        audio = Audio(
            waveform=torch.tensor(np.asarray(waveform, dtype=np.float32)).unsqueeze(0),
            sampling_rate=int(sampling_rate),
        )
        counts: dict[str, int] = {}
        for model_id in cfg.diarization_models:
            try:
                result = diarize_audios(
                    audios=[audio],
                    model=model_for_task(model_id, task="diarization"),
                    device=pick_device(cfg.device),
                )
            except Exception:  # noqa: BLE001 — a model that cannot run yields no evidence
                continue
            n = _distinct_speaker_count({"status": "ok", "result": result})
            if n is not None:
                counts[model_id] = n
        return counts

    return run


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


def run_triage(audio: Audio, cfg: RunConfig, device: DeviceType | None) -> dict[str, Any]:
    """Round 0 (spec US1): frame-posterior speech gate + SNR enhancement gate.

    Uses **continuous** frame posteriors from Brouhaha's VAD head — never segmentized VAD, see
    SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md — plus Brouhaha SNR with an ungated percentile-DSP
    fallback. Degrades conservatively: missing posteriors ⇒ ``speech_present=True``; missing SNR ⇒
    ``needs_enhancement=None`` (the caller treats unknown as "run the enhanced pass").

    Brouhaha rather than ``segmentation-3.0``: the gate needs a continuous speech probability, which
    both provide, and Brouhaha is already loaded here for SNR — so this is one model where there
    were two. What ``segmentation-3.0`` uniquely offered was per-speaker channels, which a speech
    gate does not use.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
    from senselab.audio.workflows.audio_analysis.adaptive.triage import dsp_snr_series, triage_decision

    t0 = time.time()
    snr_db: list[float] | None = None
    snr_hop_s: float | None = None
    snr_estimator: str | None = None
    posterior = None
    try:
        from senselab.audio.tasks.scene_quality.brouhaha import extract_brouhaha_frames

        brouhaha = extract_brouhaha_frames([audio], device=device)[0]
        if brouhaha is not None:
            snr_db = [float(v) for v in brouhaha.snr_db]
            snr_hop_s = float(brouhaha.frame_hop_s)
            snr_estimator = "brouhaha"
            # A VAD head is genuinely one channel, so ``single`` is a declaration, not a collapse.
            posterior = FramePosterior(
                activations=np.asarray(brouhaha.vad, dtype=np.float64)[:, None],
                frame_hop_s=float(brouhaha.frame_hop_s),
                channel_format="single",
            )
    except Exception as exc:  # noqa: BLE001 — SNR is optional; DSP fallback below
        print(f"  [triage] brouhaha unavailable ({exc!r}); falling back to DSP SNR", file=sys.stderr)
    if snr_db is None:
        waveform = audio.waveform.squeeze().detach().cpu().numpy()
        snr_db, snr_hop_s = dsp_snr_series(
            waveform,
            audio.sampling_rate,
            p_speech=[float(p) for p in posterior.speech_prob()] if posterior is not None else None,
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
            p_speech=[float(p) for p in posterior.speech_prob()],
            frame_hop_s=float(posterior.frame_hop_s),
            snr_db=snr_db,
            snr_hop_s=snr_hop_s,
            speech_threshold=cfg.triage_speech_threshold,
            min_speech_s=cfg.triage_min_speech_s,
            snr_floor_db=cfg.triage_snr_floor_db,
            low_snr_fraction_threshold=cfg.triage_low_snr_fraction,
        )
    decision["snr_estimator"] = snr_estimator
    decision["elapsed_s"] = round(time.time() - t0, 3)
    return decision


def _pass_plan(cfg: RunConfig) -> PassPlan:
    """Translate the run config into a library `PassPlan`.

    Called *after* triage, with the config triage returned: the no-speech path widens the skip set,
    and a plan built from the pre-triage config would run diarization and ASR on silence. It reads
    ``cfg.skipped_stages`` rather than a mutated field, so "what was configured" and "what the audio
    turned out to justify" stay distinguishable. Absence-means-skip is expressed here so the library
    never sees a CLI-shaped skip set.
    """
    skip = set(cfg.skipped_stages)
    return PassPlan(
        diarization_models=() if "diarization" in skip else tuple(cfg.diarization_models),
        asr_models=() if "asr" in skip else tuple(cfg.asr_models),
        ast_model=None if "ast" in skip else cfg.ast_model,
        yamnet_model=None if "yamnet" in skip else cfg.yamnet_model,
        ast_win_length=cfg.ast_win_length,
        ast_hop_length=cfg.ast_hop_length,
        yamnet_win_length=cfg.yamnet_win_length,
        yamnet_hop_length=cfg.yamnet_hop_length,
        scene_top_k=cfg.scene_top_k,
        background_mask=cfg.background_mask,
        task_type=cfg.task_type,
        mask_guard_interval_s=cfg.mask_guard_interval_s,
        # The run's one grid (D-24), the same object the comparator harvests every axis on. It used
        # to be rebuilt here from two CLI values that also fed a separate presence grid — two
        # constructions of one pair, which cannot disagree only as long as nobody edits one of them.
        mask_grid=_bucket_grid(cfg),
        features="features" not in skip,
        features_win_length=cfg.features_win_length,
        features_hop_length=cfg.features_hop_length,
        align_asr=cfg.align_asr,
        aligner=cfg.aligner,
        qwen_aligner_model=cfg.qwen_aligner_model,
        mms_aligner_model=cfg.mms_aligner_model,
        asr_language=cfg.asr_language,
        qwen_native_timestamps=cfg.qwen_native_timestamps,
    )


def _bucket_grid(cfg: RunConfig) -> BucketGrid:
    """The run's one bucket grid, constructed in one place.

    Every axis is harvested on it and the background mask is cut on it, so there is nothing to
    reconcile between them. Six separate constructions of three different grids preceded this, and
    the consequence was not a style problem: the axes shared no bucket keys, so ``project_axis_onto``
    found nothing to project and every round came out byte-identical to the last.
    """
    return BucketGrid(win_length=cfg.grid_win_length, hop_length=cfg.grid_hop_length)


def _write_run_summary(run_dir: Path, summaries: dict[str, Any]) -> None:
    """Write ``final/summary.json`` — the run provenance, and nothing L1 already holds.

    ``summaries["passes"]`` is 4.8 MB of per-perturbation model output that already exists on
    disk under ``L1/``. Inlining it made ``final/`` the copy the pipeline read back — the
    adaptive loop and the timeline both reconstructed a finished run from it — which is a
    deliverable being used as an intermediate. With two copies of the same bytes nothing enforced
    the boundary: a consumer reaching into ``final/`` got exactly what one reading ``L1/`` would.

    The small index every later stage actually needs (duration, audio signature, source path)
    lives in ``L1/perturbations.json``, beside the declaration of what each perturbation *is*.
    """
    write_json(
        final_dir(run_dir) / "summary.json",
        {k: v for k, v in summaries.items() if k != "passes"},
    )


def _stage_context(
    perturbation: Perturbation,
    audio: Audio,
    audio_path: Path,
    *,
    device: DeviceType | None,
    out_dir: Path,
    cache_dir: Path | None,
    senselab_ver: str,
) -> StageContext:
    """Build the per-perturbation `StageContext`.

    The variant is the perturbation's **declared** transform. It used to be
    ``"speech_enhanced" if label.startswith("enhanced")`` — inferring what had been done to the
    audio from how the directory happened to be spelled, so a perturbation named
    ``enhanced_lowpass`` would have claimed to be plain enhancement and one named ``sepformer``
    would have claimed to be unmodified. Stages that are only meaningful on unmodified audio (the
    background mask, most importantly) gate on this, so a wrong answer here defeats the gate with
    every unit test still passing.
    """
    return StageContext(
        perturbation=perturbation.name,
        audio_signature=audio_signature(audio),
        variant=perturbation.transform,
        variant_gain_db=perturbation.gain_db,
        device=device,
        cache_dir=cache_dir,
        out_dir=perturbation_dir(out_dir, perturbation.name),
        run_dir=out_dir,
        audio_source=str(audio_path.resolve()),
        senselab_ver=senselab_ver,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the full analysis with and without enhancement."""
    args = parse_args(argv)

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    try:
        cfg = load_run_config(args.config)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: invalid run config {args.config or DEFAULT_CONFIG_PATH}: {exc}", file=sys.stderr)
        return 2

    device = pick_device(cfg.device)
    device_label = device.value if device is not None else "auto (per-task selection)"
    cache_dir: Path | None = cfg.cache_dir.resolve() if cfg.cache_enabled else None
    if cache_dir is not None:
        _sync_cache_with_schema_version(cache_dir)
    senselab_ver = senselab_version()
    print(f"Config: {cfg.identity.name} v{cfg.identity.version} ({cfg.identity.config_hash[:12]})")
    for source in cfg.identity.sources:
        print(f"        {source}")
    print(f"Device: {device_label}")
    print(f"Input:  {args.audio}")
    print(f"Grid:   {cfg.grid_win_length} s window / {cfg.grid_hop_length} s hop, every axis")
    if cache_dir is not None:
        print(f"Cache:  {cache_dir}")
        print(f"        key = sha256(audio | task | model | params | stage_version | senselab={senselab_ver})")
    else:
        print("Cache:  disabled (cache.enabled: false)")

    audio_16k = prepare_audio(args.audio)
    print(f"Resampled: {audio_16k.waveform.shape[1] / TARGET_SR:.2f}s @ {TARGET_SR}Hz mono")

    out_root = args.out if args.out is not None else cfg.output_dir
    run_dir = out_root / f"{args.audio.stem}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
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
        # What the run was configured by, named so it can be reproduced. Hashed over the *merged*
        # mapping, so a --config override cannot inherit the packaged file's identity.
        "run_config": cfg.identity.to_json(),
        "target_sampling_rate": TARGET_SR,
        "scene_window": {
            "ast": {"win_length": cfg.ast_win_length, "hop_length": cfg.ast_hop_length},
            "yamnet": {"win_length": cfg.yamnet_win_length, "hop_length": cfg.yamnet_hop_length},
            "comparable": (
                cfg.ast_win_length == cfg.yamnet_win_length and cfg.ast_hop_length == cfg.yamnet_hop_length
            ),
        },
        "passes": {},
    }

    # ── Round 0: triage (spec US1; FR-002/003/004) ──────────────────────
    enhancement_mode = cfg.enhancement_mode
    triage: dict[str, Any] | None = None
    if enhancement_mode == "auto":
        print("\n=== Triage (round 0): frame posteriors + SNR ===")
        triage = run_triage(audio_16k, cfg, device)
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
            # A new config, not a mutated field: what the audio justified is a different fact from
            # what the run was configured to do, and both have readers downstream.
            cfg = cfg.with_skipped({"diarization", "asr", "alignment"})
            print(
                "  no speech found — skipping diarization/ASR/alignment; speech_presence outputs still emitted (FR-004)"
            )
    run_enhanced_pass = enhancement_mode == "always" or (
        enhancement_mode == "auto"
        and triage is not None
        and triage["speech_present"]
        and triage["needs_enhancement"] is not False  # unknown SNR ⇒ conservative: run it
    )

    # The perturbation set, declared once and then only iterated. Adding a third — a second
    # enhancement model, a band-limited variant — is one more entry here and no edit anywhere
    # downstream: the register below tells every later stage what the set was, the loop applies
    # each in turn, and nothing counts them.
    perturbations: list[Perturbation] = [identity_perturbation()]
    if run_enhanced_pass:
        perturbations.append(speech_enhancement_perturbation(cfg.enhancement_model))

    pass_audio: dict[str, Audio] = {}
    pass_plan = _pass_plan(cfg)
    for perturbation in perturbations:
        if not perturbation.is_identity:
            print(f"\n=== Applying perturbation {perturbation.name!r} ({perturbation.transform})... ===")
        try:
            audio = apply_perturbation(perturbation, audio_16k, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"Perturbation {perturbation.name!r} failed: {exc!r}", file=sys.stderr)
            summaries["passes"][perturbation.name] = {"status": "failed", "error": repr(exc)}
            continue
        pass_audio[perturbation.name] = audio
        summaries["passes"][perturbation.name] = run_pass(
            audio,
            _stage_context(
                perturbation,
                audio,
                args.audio,
                device=device,
                out_dir=run_dir,
                cache_dir=cache_dir,
                senselab_ver=senselab_ver,
            ),
            pass_plan,
        )

    # The register, written once, by L1, after the set has actually been measured: it carries
    # both the declaration (name, transform, parameters) and what running each one produced.
    write_perturbation_register(
        run_dir,
        perturbations,
        source_audio=str(args.audio.resolve()),
        measured={
            name: {k: block.get(k) for k in ("duration_s", "audio_signature", "status") if k in block}
            for name, block in summaries["passes"].items()
            if isinstance(block, dict)
        },
    )
    _write_run_summary(run_dir, summaries)

    # Hierarchical Label Studio export — one LS task per audio variant, each
    # carrying parallel timeline tracks (one per analyzer × model). AST and
    # YAMNet contribute regions at their own native temporal resolution.
    audio_uri = str(args.audio.resolve())
    ls_tasks = [
        build_labelstudio_task(
            audio_uri=audio_uri,
            perturbation=perturbation,
            duration_s=pass_summary["duration_s"],
            pass_summary=pass_summary,
            ast_win_length=cfg.ast_win_length,
            ast_hop_length=cfg.ast_hop_length,
            yamnet_win_length=cfg.yamnet_win_length,
            yamnet_hop_length=cfg.yamnet_hop_length,
        )
        for perturbation, pass_summary in summaries["passes"].items()
        if isinstance(pass_summary, dict) and "duration_s" in pass_summary
    ]
    config_xml = build_labelstudio_config(summaries)

    # ── Comparator: three-axis uncertainty workflow ─────────────────────
    if "comparisons" in cfg.skipped_stages and cfg.max_rounds > 1:
        print(
            "warn: stages.comparisons is false, which disables the belief store, so no adaptive "
            "rounds >= 2 and no final/ will be produced (contracts/cli.md).",
            file=sys.stderr,
        )

    # Defined before the guard: the per-speaker and adaptive blocks below both read it,
    # and with comparisons disabled there is simply nothing harvested to read.
    harvests_by_pass: dict[str, Any] = {}
    reliability_by_axis: dict[str, Any] = {}

    if "comparisons" not in cfg.skipped_stages:
        from senselab.audio.workflows.audio_analysis import (
            attach_uncertainty_tracks_to_ls,
            build_aligned_timeline_plot,
            build_disagreements_index,
            compute_uncertainty_axes,
            write_linked_votes,
            write_signal_parquet,
            write_signal_stability,
        )

        # One grid, every axis. Three were built here — a 0.25 s speaker grid, a 1.0/0.5 asr grid and
        # a 0.1/0.02 presence grid — and the four axes that came out of them shared zero bucket keys.
        grid = _bucket_grid(cfg)
        comparator_params = {
            "win_length": grid.win_length,
            "hop_length": grid.hop_length,
            "aggregator": cfg.aggregator,
            "speech_presence_labels": list(cfg.speech_presence_labels),
            "clustering_algorithm": cfg.clustering_algorithm,
            "run_config": cfg.identity.to_json(),
            "asr_scene_coupling": {
                "w_q": cfg.asr_scene_coupling_w_q,
                "w_s": cfg.asr_scene_coupling_w_s,
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
            calibration_runtime = profile_to_runtime(load_calibration_profile(cfg.calibration_profile))
        except (OSError, ValueError, KeyError) as exc:
            print(f"ERROR: invalid calibration profile {cfg.calibration_profile}: {exc}", file=sys.stderr)
            sys.exit(2)
        comparator_params["calibration"] = calibration_runtime

        passes_for_compute = {
            pl: ps for pl, ps in summaries.get("passes", {}).items() if isinstance(ps, dict) and "duration_s" in ps
        }
        speaker_embedding_models = list(cfg.embeddings_models)
        per_window_embeddings_by_pass: dict[str, dict[str, Any]] = {}
        stability_evidence: dict[str, Any] = {}
        linked_by_pass: dict[str, Any] = {}
        try:
            (
                signal_results_by_pass,
                fused_axes,
                incomparable_reasons,
                per_window_embeddings_by_pass,
            ) = compute_uncertainty_axes(
                harvests_out=harvests_by_pass,
                weights_out=reliability_by_axis,
                stability_out=stability_evidence if cfg.stability else None,
                linked_out=linked_by_pass,
                passes=passes_for_compute,
                grid=grid,
                params=comparator_params,
                audio=pass_audio,
                speaker_embedding_models=speaker_embedding_models,
                aggregator=cfg.aggregator,
                speech_presence_labels=list(cfg.speech_presence_labels),
                scene_quality=cfg.scene_quality,
                sound_sources=cfg.sound_sources,
                calibration=calibration_runtime,
                embedding_window_s=cfg.embedding_window_s,
                embedding_hop_s=cfg.embedding_hop_s,
                same_speaker_floor=cfg.speaker_same_floor,
                diff_speaker_floor=cfg.speaker_diff_floor,
                cluster_cosine_threshold=cfg.speaker_cluster_cosine_threshold,
                clustering_algorithm=cfg.clustering_algorithm,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: comparator workflow failed: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            signal_results_by_pass = {}
            fused_axes = {}
            incomparable_reasons = {"workflow": f"failed: {exc!r}"}
            per_window_embeddings_by_pass = {}

        # Scene quality is a REQUIRED signal here unless explicitly disabled: the
        # library degrades gracefully to null (FR-023) for reuse, but this script
        # must not silently ship a run missing its quality columns. If the model
        # was requested but unavailable on any pass, fail loudly with guidance.
        if cfg.scene_quality:
            # Straight off the harvest's provenance: whether the model loaded is a fact about the
            # pass, and reaching for an axis result to find it needed a pseudo-pass guard.
            unavailable_passes = [
                pl
                for pl, h in harvests_by_pass.items()
                if ((getattr(h, "provenance_extras", {}) or {}).get("scene_quality") or {})
                .get("model", {})
                .get("available")
                is False
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

        # L1 evidence: one parquet per signal, accumulating across raw and every perturbation,
        # in native units. Nothing under L1/ is named for an axis — an axis is a fold across
        # signals and perturbations — and nothing under L1/signals/ is keyed by *where* it sat:
        # the perturbation is a column, so a reader asking for one has to say so.
        run_provenance = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "stage_versions": dict(STAGE_VERSIONS),
            "senselab_version": senselab_ver,
        }
        results_by_signal: dict[str, list[Any]] = {}
        for _perturbation, by_signal in signal_results_by_pass.items():
            for signal, result in by_signal.items():
                results_by_signal.setdefault(signal, []).append(result)
        for signal, results in results_by_signal.items():
            write_signal_parquet(
                results,
                signals_dir(run_dir) / f"{safe_model_id(signal)}.parquet",
                provenance=run_provenance,
            )

        # Perturbation stability, keyed by signal — the property it is a property of — and a
        # *round derivative*, not evidence: relating two perturbations is a fold over an input
        # dimension, which is L2's by the same argument that makes an axis L2's. The run-level
        # mean has no file: it is already on every fused row as weight_basis[signal]["stability"].
        per_bucket_stability: dict[str, list[dict[str, Any]]] = {}
        if stability_evidence:
            for _axis, by_signal_rows in (stability_evidence.get("per_bucket") or {}).items():
                for signal, rows in by_signal_rows.items():
                    per_bucket_stability.setdefault(signal, []).extend(rows)
            for signal, rows in per_bucket_stability.items():
                write_signal_stability(
                    sorted(rows, key=lambda r: (r["start"], r["end"])),
                    derivatives_dir(run_dir, 0) / "stability" / f"{safe_model_id(signal)}.parquet",
                    provenance=run_provenance,
                )

        # The linked evidence, at the vote level — where (axis, bucket, source, pass, scope) is a
        # legitimate key. This is what the artifact-driven adaptive path ingests, so it sees the
        # same evidence the in-process path does instead of a per-pass axis fold.
        for axis_name in HARVESTED_AXES:
            write_linked_votes(
                {label: linked.buckets_by_axis.get(axis_name, []) for label, linked in linked_by_pass.items()},
                axis_name,
                derivatives_dir(run_dir, 0) / "votes" / f"{axis_name}.parquet",
                provenance={
                    **run_provenance,
                    "speech_presence_policy": next(
                        (linked.provenance.get("speech_presence_policy") for linked in linked_by_pass.values()), None
                    ),
                },
            )

        # PII detection per pass — scans each ASR transcript with regex layer
        # plus optional spaCy NER. Default-on; failures (e.g. spaCy not
        # installed) are surfaced via stderr + the report's failures dict.
        from senselab.audio.workflows.audio_analysis.global_summary import (
            compute_run_global_summary,
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
                perturbation=pl,
                asr_resolved=asr_resolved_pii,
            )
            write_json(perturbation_dir(run_dir, pl) / "pii.json", report_to_dict(pii_reports[pl]))

        # Global run summary: 4 claims (transcript / speaker / quality / PII) → 1 scalar each
        # + a max() combined. One block for the run, not one per pass: the axes it reads are
        # already folded across passes, and the genuinely per-pass evidence stays visible under
        # ``by_pass`` rather than being reduced to a winner.
        asr_resolved_by_pass: dict[str, dict[str, Any]] = {}
        for pl, ps in passes_for_compute.items():
            align_by_model_g = (ps.get("alignment") or {}).get("by_model") or {}
            resolved_g: dict[str, Any] = {}
            for m, b in ((ps.get("asr") or {}).get("by_model") or {}).items():
                if isinstance(b, dict) and b.get("status") == "ok":
                    resolved_g[m] = resolve_asr_result(b, align_by_model_g.get(m))
            asr_resolved_by_pass[pl] = resolved_g
        summaries["global_uncertainty"] = {
            **compute_run_global_summary(
                fused_axes=fused_axes,
                passes=passes_for_compute,
                asr_resolved_by_pass=asr_resolved_by_pass,
                pii_reports=pii_reports,
                expects_speech=True,
            ),
            # What enhancement bought, as evidence about each signal rather than as a choice
            # between two answers.
            "stability": {
                signal: round(float(value), 6)
                for axis_map in (stability_evidence.get("instability") or {}).values()
                for signal, value in axis_map.items()
            },
            "incomparable_reasons": incomparable_reasons,
        }
        # Re-persist summary.json — the original write at line 1782 happened
        # before the comparator stage so it does not contain
        # ``global_uncertainty``. Overwriting here keeps the on-disk summary
        # in sync with the in-memory dict.
        _write_run_summary(run_dir, summaries)

        # Persist per-pass windowed speaker embeddings — one JSON per (pass, model)
        # at ``<pass>/embeddings/<model>.json`` with the full window grid + vectors.
        for pert_name, by_model in per_window_embeddings_by_pass.items():
            if not by_model:
                continue
            for model_id, windows in by_model.items():
                payload = {
                    "status": "ok" if windows else "no_data",
                    "window_s": cfg.embedding_window_s,
                    "hop_s": cfg.embedding_hop_s,
                    "windows": [
                        {
                            "start_s": float(w.start_s),
                            "end_s": float(w.end_s),
                            "vector": [float(x) for x in w.vector.tolist()],
                        }
                        for w in windows
                    ],
                }
                write_json(
                    perturbation_dir(run_dir, pert_name) / "embeddings" / f"{safe_model_id(model_id)}.json", payload
                )

        # ── Level 2: the fused uncertainty maps ───────────────────────
        # L1 holds per-signal measurements in native units; these are the answer — one fold per
        # axis, across every signal and every pass, each signal weighted by its measured
        # stability and support. Runs *before* the LS bundle, the disagreements index and the
        # timeline, because all three read an axis and an axis exists only here. The old order
        # was possible only while L1 was producing one.
        if harvests_by_pass:
            try:
                from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty

                # The reserved __basis__ key carries the per-factor breakdown; it is not an
                # axis, so it is split off rather than fused.
                basis = reliability_by_axis.get("__basis__") or {}
                axis_weights = {k: v for k, v in reliability_by_axis.items() if k != "__basis__"}
                # The mask and the speaker claims are what make the rounds able to *do* anything:
                # without them the loop has no regional trust to withdraw and no fourth axis to
                # emit, so it folds once and stops. They were previously left at their defaults,
                # which silently reduced every run to a single round.
                from senselab.audio.workflows.audio_analysis.fuse import (
                    mask_regions_from_rows,
                    speaker_claims_from_votes,
                )

                fusion_mask_rows: list[dict[str, Any]] = []
                mask_path = belief_dir(run_dir) / "background_mask.parquet"
                if mask_path.exists():
                    import pandas as _pd_mask

                    fusion_mask_rows = _pd_mask.read_parquet(mask_path).to_dict("records")
                # The mask governs fusion from the *raw* pass: whether a region is target-free is a
                # fact about the recording, not about the enhancement transform.
                mask_regions = mask_regions_from_rows(fusion_mask_rows)
                reference = harvests_by_pass.get("raw") or next(iter(harvests_by_pass.values()))
                speaker_claims = speaker_claims_from_votes(getattr(reference, "speaker_votes", []) or [])
                from senselab.audio.workflows.audio_analysis.speech_presence_link import (
                    policy_from_params as _policy_from_params,
                )

                final_maps = write_final_uncertainty(
                    run_dir,
                    harvests=harvests_by_pass,
                    weights_by_axis=axis_weights,
                    aggregator=cfg.aggregator,
                    weight_basis_by_axis=basis,
                    mask_regions=mask_regions,
                    speaker_claims=speaker_claims,
                    max_rounds=cfg.max_rounds,
                    # The same policy the round-0 link used. Left at the packaged default it read
                    # the presence measurements under different thresholds than
                    # ``compute_uncertainty_axes`` did, so the two folds were not comparable.
                    speech_presence_policy=_policy_from_params(comparator_params),
                    # The scene measurements round 0 attached, so the coupling is applied to every
                    # round these calls write rather than to in-memory rows this loop then
                    # overwrites (the copy-back below re-reads triage_score and coupled_from).
                    scene_rows=(fused_axes["speech_presence"].rows if "speech_presence" in fused_axes else ()),
                    comparator_params=comparator_params,
                )
                summaries["final_uncertainty"] = final_maps

                # The fourth axis's votes are written by the loop over ``HARVESTED_AXES`` above,
                # from the same per-bucket harvest every other axis's come from. A second write
                # used to happen here from ``mask_axis_votes(mask_regions)`` — one vote per mask
                # *region*, keyed under a fabricated perturbation called "mask" — and it clobbered
                # the per-bucket file. Both ingest paths then saw a single bucket where L2 had
                # folded 1070, which is an axis with nowhere to be uncertain. ``mask_regions``
                # itself stays: it is what ``write_final_uncertainty`` withdraws regional trust
                # with, and a region is the right unit for that.

                # The per-round timelines are drawn by ``write_final_uncertainty`` itself now. They
                # were drawn here, so a caller of the workflow API got rounds with no view of
                # themselves — a third of what a round owes. What is still read back is the last
                # round's rows, for the in-memory copy-back below.
                import pandas as _pd_round

                by_round: dict[int, dict[str, list[dict[str, Any]]]] = {}
                for key, path in final_maps.items():
                    if "@round" not in key or key.startswith("summary@"):
                        continue
                    axis_name, round_token = key.split("@round", 1)
                    by_round.setdefault(int(round_token), {})[axis_name] = _pd_round.read_parquet(path).to_dict(
                        "records"
                    )
                # Name the directory the writer actually used. The maps live under
                # L2/round<N>/uncertainty/, and pointing at final/uncertainty/ sent a reader to an
                # empty path; the count also included @round aliases, so it overstated the axes.
                #
                # Filtered against the *declaration* rather than by excluding the non-axis keys one
                # at a time. The denylist ("not round_logs, not summary@…") went stale the moment
                # ``final_rows`` was added and the line reported "5 axis map(s)" for four axes — an
                # allowlist cannot drift that way, which is the whole argument for ``axes.AXES``.
                axis_names = sorted({k.split("@")[0] for k in final_maps} & set(AXIS_NAMES))
                print(
                    f"  [final uncertainty] {len(axis_names)} axis map(s) under L2/round<N>/uncertainty/: {', '.join(axis_names)}"
                )
                # Advance the in-memory axes to the last round the loop actually ran, so the LS
                # bundle, the disagreements index and the timeline all show the same numbers the
                # parquets do. The per-bucket scene measurements harvested at round 0 ride along
                # unchanged — they are measurements, and no round re-measures them.
                if by_round:
                    last_round = max(by_round)
                    for axis_name, axis_result in fused_axes.items():
                        final_rows = {
                            (round(float(r["start"]), 6), round(float(r["end"]), 6)): r
                            for r in by_round[last_round].get(axis_name, [])
                        }
                        for row in axis_result.rows:
                            fresh = final_rows.get((round(float(row["start"]), 6), round(float(row["end"]), 6)))
                            if fresh is None:
                                continue
                            for key in (
                                "uncertainty",
                                "epistemic_uncertainty",
                                "confidence",
                                "variability",
                                "triage_score",
                                "round",
                                "coupled_from",
                                "scene_quality_coupling",
                                "triage_score_pre_coupling",
                            ):
                                if key in fresh:
                                    row[key] = fresh[key]
            except Exception as exc:  # noqa: BLE001 — a derived artifact must not fail the run
                logger.warning("final uncertainty maps could not be written: %s", exc)
                summaries["final_uncertainty"] = {"status": "failed", "error": repr(exc)}

        # One Labels track per fused axis + per-pass, per-signal evidence tracks.
        if fused_axes:
            ls_tasks, config_xml = attach_uncertainty_tracks_to_ls(
                ls_tasks=ls_tasks,
                ls_config=config_xml,
                fused_axes=fused_axes,
                signal_results_by_pass=signal_results_by_pass,
            )

        # Disagreements index — opt-out via --disagreements-top-n 0.
        if fused_axes and cfg.disagreements_top_n > 0:
            # The index is **pre-adaptive by design**: the adaptive stage *consumes* it to propose
            # regions, so it runs after this point and the index cannot rank that stage's output.
            # ``final/estimates/`` does not exist yet here, and a block that reached for it was
            # simply dead. What identifies the fold an entry describes is its own ``round`` field —
            # ``fused_axes`` has already been advanced in place to the last round
            # ``write_final_uncertainty`` wrote for each axis (the copy-back above), so an entry's
            # ``round`` and its ``parquet`` pointer name the file its ``triage_score`` came from.
            index = build_disagreements_index(
                fused_axes=fused_axes,
                signal_results_by_pass=signal_results_by_pass,
                top_n=cfg.disagreements_top_n,
                run_dir=run_dir,
                config={
                    "top_n": cfg.disagreements_top_n,
                    "aggregator": cfg.aggregator,
                    "run_config": cfg.identity.to_json(),
                    "bucket_grid": {
                        "win_length": grid.win_length,
                        "hop_length": grid.hop_length,
                    },
                    "speech_presence_labels": list(cfg.speech_presence_labels),
                    "stage_versions": dict(STAGE_VERSIONS),
                    "senselab_version": senselab_ver,
                },
                incomparable_reasons=incomparable_reasons,
                models_without_native_signal=_models_without_native_signal(summaries),
            )
            # L2: a ranked index of where the fold was least sure is a statement of belief, and
            # the adaptive stage consumes it. ``final/disagreements_resolved.json`` — the same
            # index annotated with what the loop did about each entry — is the deliverable.
            write_json(belief_dir(run_dir) / "disagreements.json", index)

        # Timeline plot — best-effort sidecar.
        if fused_axes:
            try:
                duration_s = float(next(iter(passes_for_compute.values())).get("duration_s", 0.0) or 0.0)
                # Build per-pass detail bundles for the plot's per-source rows.
                detail_by_pass: dict[str, dict[str, Any]] = {}
                for perturbation, pass_summary in passes_for_compute.items():
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
                    detail_by_pass[perturbation] = {
                        "diar_by_model": diar_by_model,
                        "asr_by_model": asr_by_model,
                        "per_window_embeddings": per_window_embeddings_by_pass.get(perturbation, {}),
                    }
                raw_pass_audio = pass_audio.get("raw")
                raw_waveform = (
                    raw_pass_audio.waveform.detach().cpu().numpy().squeeze() if raw_pass_audio is not None else None
                )
                raw_sr = int(raw_pass_audio.sampling_rate) if raw_pass_audio is not None else 16000
                timeline_path = build_aligned_timeline_plot(
                    run_dir=run_dir,
                    fused_axes=fused_axes,
                    stability_by_signal=per_bucket_stability,
                    duration_s=duration_s,
                    grid_hop=grid.hop_length,
                    detail_by_pass=detail_by_pass,
                    title=f"Aggregate uncertainty · {args.audio.name}",
                    audio_waveform=raw_waveform,
                    audio_sr=raw_sr,
                )
                if timeline_path is not None:
                    print(f"Timeline plot: {timeline_path}")
            except Exception as exc:  # noqa: BLE001 — best-effort sidecar
                print(f"warn: timeline plot failed: {exc!r}", file=sys.stderr)

        # ── Per-speaker speaker (US1) ────────────────────────────────
        # Derived from the completed passes rather than from a fresh inference: the raw
        # and enhanced passes are the same recording under a transform, so each diarizer's
        # two answers are already a stability sample. A diarizer whose speaker count
        # changes under enhancement is telling us its answer is not robust here, and that
        # governs how far it moves the posterior.
        if cfg.per_speaker_identity:
            try:
                from senselab.audio.workflows.audio_analysis.adaptive.fusion import write_speaker_outputs
                from senselab.audio.workflows.audio_analysis.speaker_identity import (
                    build_speech_presence_tracks,
                    build_speaker_identity,
                )

                # Per-speaker structure reads the speaker harvest of the unmodified pass.
                # Enhancement is a perturbation used to test how stable each diarizer's
                # answer is; where a speaker was active is a fact about the recording, so
                # the tracks come from the audio as recorded.
                speaker_harvest = next(
                    (
                        getattr(harvests_by_pass.get(label), "speaker_votes", None)
                        for label in ("raw", *sorted(harvests_by_pass))
                        if getattr(harvests_by_pass.get(label), "speaker_votes", None)
                    ),
                    None,
                )
                # Measured support, not a declared source kind: a source is attenuated for
                # claiming speakers where no voice detector reports speech, never for its
                # name. Measured on the unmodified pass — whether the audio carries a claim
                # is a fact about the recording, not about the transform.
                from senselab.audio.workflows.audio_analysis.support import (
                    evidence_signal_names,
                    informative_evidence,
                    signal_support,
                )

                from senselab.audio.workflows.audio_analysis.speech_presence_link import (
                    votes_for_harvest,
                )

                # Support is measured over verdicts; the harvest holds L1 measurements, so link
                # first. Prefer the unmodified pass, falling back to whichever pass measured
                # anything at all.
                speech_presence_harvest = next(
                    (
                        linked
                        for h in (
                            harvests_by_pass.get("raw"),
                            *harvests_by_pass.values(),
                        )
                        if h is not None and (linked := votes_for_harvest(h))
                    ),
                    [],
                )
                # Invariance folds in as a third measured factor when asked for. Unlike the
                # enhanced-pass comparison, a failure here cannot be explained away as the
                # audio having changed — these perturbations leave the answer well-defined.
                invariance: dict[str, float] = {}
                if cfg.invariance_probe:
                    try:
                        from senselab.audio.workflows.audio_analysis.invariance import (
                            probe_diarization_invariance,
                        )

                        raw_audio = pass_audio.get("raw")
                        reference = {
                            model: n
                            for model, n in (
                                (m, _distinct_speaker_count(o))
                                for m, o in (
                                    (summaries["passes"].get("raw", {}).get("diarization") or {}).get("by_model") or {}
                                ).items()
                            )
                            if n is not None
                        }
                        if raw_audio is not None and reference:
                            invariance = probe_diarization_invariance(
                                raw_audio.waveform.squeeze().numpy(),
                                int(raw_audio.sampling_rate),
                                reference_counts=reference,
                                run_diarization=_diarize_counts_for_probe(cfg),
                            )
                            print(f"  [invariance] probed {len(invariance)} model(s): {invariance}")
                    except Exception as exc:  # noqa: BLE001 — an opt-in probe must not fail a run
                        logger.warning("invariance probe failed: %s", exc)

                measured_support = signal_support(
                    speech_presence_harvest,
                    evidence_signals=sorted(
                        informative_evidence(
                            speech_presence_harvest, sorted(evidence_signal_names(speech_presence_harvest))
                        )
                    ),
                )
                # Support x invariance: a source is discounted for claiming speakers the
                # audio does not carry, and for changing its answer when nothing about the
                # question changed. Absent either measurement, that factor stays 1.0.
                combined = {
                    name: float(measured_support.get(name, 1.0)) * float(invariance.get(name, 1.0))
                    for name in set(measured_support) | set(invariance)
                }
                posterior, hypotheses, correspondence = build_speaker_identity(
                    summaries["passes"],
                    speaker_votes=speaker_harvest,
                    support=combined,
                )
                tracks = build_speech_presence_tracks(speaker_harvest or [])
                write_speaker_outputs(
                    run_dir,
                    posterior=posterior,
                    hypotheses=hypotheses,
                    correspondence=correspondence,
                    tracks=tracks,
                    profile_version=str(cfg.detection_margin_profile or "detection-margin/default"),
                    influence_profile=str(cfg.influence_profile or "influence/default"),
                )
                summaries["speaker_identity"] = posterior.to_json()
                print(
                    f"  [speaker speaker] count posterior "
                    f"{ {k: round(v, 2) for k, v in posterior.to_json()['probabilities'].items()} }"
                    f"{'  MULTI-MODAL' if posterior.is_multimodal else ''}"
                    f"  |  {len(hypotheses)} speaker(s), {len(tracks)} speech_presence row(s)"
                )
            except Exception as exc:  # noqa: BLE001 — never fail a run over a derived summary
                logger.warning("per-speaker speaker could not be derived: %s", exc)
                summaries["speaker_identity"] = {"status": "failed", "error": repr(exc)}

    # Scene-context tracks complete the L2 annotation bundle: they read artifacts written by the
    # per-speaker and mask stages above, and both are questions a reviewer cannot answer from the
    # uncertainty tracks alone — which intervals the machine trusted, and which speaker each
    # contested claim was about.
    try:
        from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

        mask_rows: list[dict[str, Any]] = []
        mask_parquet = belief_dir(run_dir) / "background_mask.parquet"
        if mask_parquet.exists():
            import pandas as _pd

            mask_rows = _pd.read_parquet(mask_parquet).to_dict("records")
        speaker_rows: list[dict[str, Any]] = []
        speech_presence_parquet = final_dir(run_dir) / "per_speaker_presence.parquet"
        if speech_presence_parquet.exists():
            import pandas as _pd

            speaker_rows = _pd.read_parquet(speech_presence_parquet).to_dict("records")
        if mask_rows or speaker_rows:
            ls_tasks, config_xml = attach_scene_context_tracks_to_ls(
                ls_tasks=ls_tasks,
                ls_config=config_xml,
                mask_rows=mask_rows,
                speaker_rows=speaker_rows,
            )
    except Exception as exc:  # noqa: BLE001 — an annotation sidecar must not fail the run
        logger.warning("scene-context LS tracks could not be attached: %s", exc)

    # The annotation bundle lands under L2 — it is the belief rendered for a human: per-pass
    # uncertainty tracks, the mask, the per-speaker presence. Written here, before the adaptive
    # loop, because the loop's final stage appends its consensus tracks to *this* bundle and
    # writes the result to final/. Writing it after the loop is what made that stage a no-op.
    write_json(belief_dir(run_dir) / "labelstudio_tasks.json", ls_tasks)
    (belief_dir(run_dir) / "labelstudio_config.xml").write_text(config_xml, encoding="utf-8")

    # ── Adaptive loop, in-process (T040) ──────────────────────────
    # Runs on the harvests the parquets were just built from, so the belief store needs no parquet
    # round-trip. Gated on ``stages.adaptive_outputs``; a ``rounds.max_rounds: 1`` config still emits
    # final/ from the round-1 belief.
    if cfg.adaptive_outputs and harvests_by_pass:
        from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop

        try:
            adaptive_log = run_adaptive_loop(
                run_dir,
                cache_dir=cache_dir,
                config_path=args.config,
                out_dir=run_dir,
                max_rounds=cfg.max_rounds,
                aggregator=cfg.aggregator,
                harvests=harvests_by_pass,
                summary=summaries,
            )
            summaries["adaptive"] = {
                "enabled": True,
                # Whether the loop wrote the final bundle, from what the loop *returned*. The
                # driver used to probe ``final/labelstudio_tasks.json`` with ``.exists()`` to
                # decide what to print, which is a stage branching on a deliverable — and a
                # probe is a read, so it was the one surviving read of final/ in the pipeline.
                "final_bundle": bool((adaptive_log.get("labelstudio") or {}).get("ls_tracks_added")),
                "max_rounds": cfg.max_rounds,
                # The config that supplied the policy, and the policy's own hash. Two identities,
                # because a model change and a policy change are not the same event.
                "run_config": cfg.identity.to_json(),
                "policy_hash": adaptive_log.get("policy_hash"),
                "rounds": adaptive_log.get("rounds"),
                "run_state": adaptive_log.get("run_state"),
                # `run_state` is the loop's own reason for stopping; `termination_reason` is
                # that reason after non-convergence detection has had its say, so a run that
                # ran out of moves while two interpretations traded places does not read as
                # agreement (FR-011e).
                "termination_reason": adaptive_log.get("termination_reason"),
                "converged": adaptive_log.get("converged"),
                "n_interventions_fired": adaptive_log.get("n_interventions_fired"),
                "n_words_fused": adaptive_log.get("n_words_fused"),
                "replay_check": (adaptive_log.get("replay_check") or {}).get("status", "checked"),
                "ingest": "in_process_harvests",
                "timeline": adaptive_log.get("timeline"),
            }
            print(f"Adaptive: {run_dir / 'final'} ({summaries['adaptive'].get('termination_reason')})")
            if summaries["adaptive"].get("timeline"):
                print(f"Adaptive timeline: {summaries['adaptive']['timeline']}")
        except Exception as exc:  # noqa: BLE001 — additive artifacts must not fail the run
            print(f"warn: adaptive loop failed: {exc!r}", file=sys.stderr)
            summaries["adaptive"] = {"enabled": True, "status": "failed", "error": repr(exc)}
        _write_run_summary(run_dir, summaries)
    elif not cfg.adaptive_outputs:
        summaries["adaptive"] = {"enabled": False, "reason": "--no-adaptive-outputs"}
        _write_run_summary(run_dir, summaries)

    # L1 evidence plot: every signal that reported, plus the level track. No uncertainty rows
    # — those are level-2 conclusions drawn from this evidence, and mixing them in invites
    # reading a conclusion as another observation.
    if harvests_by_pass:
        try:
            from senselab.audio.workflows.audio_analysis.l1_plot import build_l1_signal_plot, classify_signal

            reference = harvests_by_pass.get("raw") or next(iter(harvests_by_pass.values()))
            raw_summary = summaries["passes"].get("raw", {}) or {}

            # Continuous voters get a trace of their own confidence; binary ones get spans.
            # Rendering a frame posterior as on/off discards everything it measured.
            l1_signals: dict[str, list[tuple[float, float]]] = {}
            l1_series: dict[str, tuple[list[float], list[float]]] = {}
            from senselab.audio.workflows.audio_analysis.speech_presence_link import votes_for_harvest

            # The diagnostic plots what each signal *concluded*, so it reads the linked votes.
            # Plotting the raw measurements instead would need a per-signal axis scale — tracked as
            # follow-up in the L1 post-processing register.
            for bucket in votes_for_harvest(reference):
                centre = (float(bucket["start"]) + float(bucket["end"])) / 2.0
                for name, entry in (bucket.get("votes") or {}).items():
                    if str(name).startswith("__") or not isinstance(entry, dict):
                        continue
                    confidence = entry.get("native_confidence")
                    if isinstance(confidence, (int, float)) and classify_signal(str(name)) in (
                        "frame",
                        "acoustic",
                    ):
                        times, values = l1_series.setdefault(str(name), ([], []))
                        times.append(centre)
                        values.append(float(confidence))
                        continue
                    l1_signals.setdefault(str(name), [])
                    if entry.get("speaks"):
                        l1_signals[str(name)].append((float(bucket["start"]), float(bucket["end"])))

            # Which cluster each diarizer placed per bucket, so its row can colour by speaker.
            # A flat row makes a two-speaker conversation look identical to a one-speaker one,
            # which is precisely what the speaker axis is arguing about.
            l1_speakers: dict[str, list[tuple[float, float, str]]] = {}
            for bucket in getattr(reference, "speaker_votes", []) or []:
                for name, entry in (bucket.get("votes") or {}).items():
                    if "::" in str(name) or str(name).startswith("__") or not isinstance(entry, dict):
                        continue
                    cluster = entry.get("cluster_id")
                    if cluster and str(cluster) != "SIL":
                        l1_speakers.setdefault(str(name), []).append(
                            (float(bucket["start"]), float(bucket["end"]), str(cluster))
                        )

            # Words attributed to the model that produced them, taken from whichever result
            # actually carries usable timings. Reading only the alignment block was wrong:
            # CrisperWhisper and Qwen3-ASR carry *native* word timings in their own chunks and
            # so are correctly skipped by the aligner, which meant the plot showed words for
            # the one text-only model and none for the two that had them all along.
            from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

            asr_by_model = (raw_summary.get("asr") or {}).get("by_model") or {}
            align_by_model = (raw_summary.get("alignment") or {}).get("by_model") or {}
            l1_words: dict[str, list[dict[str, Any]]] = {}
            for model, asr_outcome in asr_by_model.items():
                if not isinstance(asr_outcome, dict) or asr_outcome.get("status") != "ok":
                    continue
                timed = resolve_asr_result(asr_outcome, align_by_model.get(model))
                lines = timed if isinstance(timed, list) else [timed]
                while isinstance(lines, list) and lines and isinstance(lines[0], list):
                    lines = lines[0]
                for line in lines or []:
                    if not isinstance(line, dict):
                        continue
                    for word in line.get("chunks") or line.get("words") or []:
                        if isinstance(word, dict) and word.get("start") is not None:
                            l1_words.setdefault(str(model), []).append(
                                {
                                    "start": float(word["start"]),
                                    "end": float(word.get("end") or word["start"]),
                                    "text": str(word.get("text") or word.get("word") or ""),
                                }
                            )

            # A signal that ran and failed keeps its row, marked: omitting it makes a failure
            # indistinguishable from a signal that was never configured.
            l1_scene: dict[str, list[dict[str, Any]]] = {}
            l1_failed: list[str] = []
            for classifier in ("ast", "yamnet"):
                block = raw_summary.get(classifier)
                if not isinstance(block, dict):
                    continue
                if block.get("status") == "ok":
                    l1_scene[classifier] = list(_classification_windows(block.get("result")) or [])
                else:
                    l1_failed.append(classifier)

            raw_audio = pass_audio.get("raw")
            l1_path = build_l1_signal_plot(
                run_dir,
                signals=l1_signals,
                duration_s=float(summaries["passes"].get("raw", {}).get("duration_s") or 0.0),
                waveform=None if raw_audio is None else raw_audio.waveform.squeeze().numpy(),
                sampling_rate=int(getattr(raw_audio, "sampling_rate", 16000) or 16000),
                series=l1_series,
                words_by_model=l1_words,
                speakers_by_model=l1_speakers,
                scene_by_classifier=l1_scene,
                failed=l1_failed,
                title=f"L1 signals — {args.audio.name}",
            )
            print(f"L1 signals plot: {l1_path}")
        except Exception as exc:  # noqa: BLE001 — a figure must not fail a completed run
            logger.warning("L1 signal plot failed: %s", exc)

    # The run headline, readable without a parquet reader. Built from the L2 maps rather than
    # recomputed, so it cannot disagree with them.
    try:
        import pandas as _pd

        from senselab.audio.workflows.audio_analysis.summary import build_run_summary, render_run_summary

        axis_rows: dict[str, list[dict[str, Any]]] = {}
        for axis in AXIS_NAMES:
            path = (summaries.get("final_uncertainty") or {}).get(axis)
            if path and Path(path).exists():
                frame = _pd.read_parquet(path)
                axis_rows[axis] = frame.to_dict("records")
        speakers_doc: dict[str, Any] = {}
        speakers_path = final_dir(run_dir) / "speakers.json"
        if speakers_path.exists():
            speakers_doc = json.loads(speakers_path.read_text())
        # From what the fold returned, not from a file. The per-round log now lives in each
        # round's own summary.json, and reassembling it by reading five documents back would be
        # this stage re-deriving what it was already handed.
        rounds_doc: dict[str, Any] = (summaries.get("final_uncertainty") or {}).get("round_logs") or {}
        headline = build_run_summary(axis_rows=axis_rows, speakers=speakers_doc, rounds=rounds_doc)
        write_json(final_dir(run_dir) / "run_summary.json", headline)
        (final_dir(run_dir) / "summary.md").write_text(render_run_summary(headline), encoding="utf-8")
        print(f"Summary: {final_dir(run_dir) / 'summary.md'}")
    except Exception as exc:  # noqa: BLE001 — a headline must not fail a completed run
        logger.warning("run summary could not be written: %s", exc)

    # The bundle is written to L2 before the adaptive loop, which appends its consensus tracks and
    # writes final/. Report whichever the *loop said* it produced: the loop is opt-out, and on a
    # run without it L2 is the whole bundle. This used to be an ``.exists()`` probe of
    # ``final/labelstudio_tasks.json`` — a stage branching on a deliverable, which is treating it
    # as state, and the one surviving read of final/ in the pipeline.
    ls_home = final_dir(run_dir) if (summaries.get("adaptive") or {}).get("final_bundle") else belief_dir(run_dir)
    print(f"\nDone. Summary: {final_dir(run_dir) / 'summary.json'}")
    print(f"Label Studio tasks:  {ls_home / 'labelstudio_tasks.json'}")
    print(f"Label Studio config: {ls_home / 'labelstudio_config.xml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
