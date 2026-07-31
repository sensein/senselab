"""Per-task analysis stages for one audio pass (T051).

Moved out of ``scripts/analyze_audio.py`` so the pipeline is importable rather
than CLI-only — the precondition for running it in-process from the adaptive loop
(T040).

Each ``stage_*`` function takes the audio, a frozen :class:`StageContext`, and its
own explicit keyword knobs, and **returns** the fragment it contributes to the
pass summary. It does not mutate a shared dict. The returned keys are a published
contract: ``presence.py``, ``compute.py``, ``identity.py``, ``utterance.py``,
``global_summary.py`` and the adaptive interventions all read
``pass_summary["asr"]["by_model"]``, ``["ast"]``, ``["yamnet"]``,
``["features"]["result"]`` and ``["ppgs"]`` — so the fragments stay plain dicts
keyed exactly as before.

Two contracts worth stating because breaking them fails *silently*:

- ``stage_features`` returns the **live** row dict in ``outcome["result"]``. The
  JSON sidecar deliberately substitutes a ``"see features/*.parquet"``
  placeholder, since the real payload goes to parquet. Returning the sidecar
  shape instead would leave every loudness/quality column ``None`` rather than
  raising.
- ``stage_ppg`` returns the key ``"ppgs"`` (plural). Consumers accept both
  spellings, so a rename degrades to null signals instead of failing.
"""

from __future__ import annotations

import sys
from typing import Any, Literal, Mapping, Sequence

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification import classify_audios
from senselab.audio.tasks.features_extraction.temporal import extract_temporal_features
from senselab.audio.tasks.forced_alignment import align_transcriptions
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.audio.tasks.speech_to_text.qwen import QwenASR
from senselab.audio.workflows.audio_analysis.harvesters import (
    asr_has_timestamps as _asr_has_timestamps,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_window_top1 as _classification_window_top1,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_windows as _classification_windows,
)
from senselab.audio.workflows.audio_analysis.sound_sources import AUDIOSET_SCORE_FUNCTION
from senselab.audio.workflows.audio_analysis.stage_context import PassPlan, StageContext
from senselab.utils.data_structures import HFModel, Language, ScriptLine, model_for_task, safe_model_id
from senselab.utils.tasks.cached_inference import (
    run_alignment_cached,
    run_task_cached,
    transcript_signature,
)

__all__ = [
    "run_pass",
    "stage_alignment",
    "stage_asr",
    "stage_diarization",
    "stage_features",
    "stage_ppg",
    "stage_scene",
]


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
    ast_windows = _classification_windows(ast_result)
    yamnet_windows = _classification_windows(yamnet_result)
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
    label, score, _entropy = _classification_window_top1(window)
    if label is None:
        return None
    return {"label": label, "score": score if score is not None else 0.0}


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


# ── Stages ────────────────────────────────────────────────────────────


def stage_diarization(audio: Audio, ctx: StageContext, *, models: Sequence[str]) -> dict[str, Any]:
    """Run each diarization model.

    Args:
        audio: The pass audio.
        ctx: Run environment.
        models: Diarization model ids; empty means the stage contributes nothing.

    Returns:
        ``{"diarization": {"by_model": {model_id: outcome}}}``.
    """
    by_model: dict[str, Any] = {}
    for model_id in models:
        params = {"device": ctx.device_label}
        outcome = run_task_cached(
            f"diarization[{model_id}]",
            diarize_audios,
            [audio],
            model=model_for_task(model_id, task="diarization"),
            device=ctx.device,
            cache_dir=ctx.cache_dir,
            cache_key_str=ctx.cache_key_for("diarization", model_id, params),
            provenance=ctx.provenance_for("diarization", model_id, params),
        )
        by_model[model_id] = outcome
        ctx.write_sidecar(f"diarization/{safe_model_id(model_id)}.json", outcome)
    return {"diarization": {"by_model": by_model}}


def stage_scene(
    audio: Audio,
    ctx: StageContext,
    *,
    ast_model: str | None,
    yamnet_model: str | None,
    ast_win_length: float,
    ast_hop_length: float,
    yamnet_win_length: float,
    yamnet_hop_length: float,
    top_k: int,
) -> dict[str, Any]:
    """Run AST and/or YAMNet windowed classification, plus a same-grid agreement sidecar.

    Args:
        audio: The pass audio.
        ctx: Run environment.
        ast_model: AST model id, or ``None`` to skip AST.
        yamnet_model: YAMNet model id, or ``None`` to skip YAMNet.
        ast_win_length: AST window length, seconds.
        ast_hop_length: AST hop length, seconds.
        yamnet_win_length: YAMNet window length, seconds.
        yamnet_hop_length: YAMNet hop length, seconds.
        top_k: Classes retained per window.

    Returns:
        Only the keys that ran: ``"ast"``, ``"yamnet"``, and ``"scene_agreement"``
        (the last only when both ran ok on an identical grid).
    """
    fragment: dict[str, Any] = {}

    if ast_model is not None:
        # `function_to_apply` participates in the cache key: it changes the stored
        # scores, so a cached softmax result must not be replayed for a sigmoid request.
        params = {
            "win_length": ast_win_length,
            "hop_length": ast_hop_length,
            "top_k": top_k,
            "device": ctx.device_label,
            "function_to_apply": AUDIOSET_SCORE_FUNCTION,
        }
        ast_outcome = run_task_cached(
            "ast",
            classify_audios,
            [audio],
            model=HFModel(path_or_uri=ast_model),
            device=ctx.device,
            win_length=ast_win_length,
            hop_length=ast_hop_length,
            top_k=top_k,
            function_to_apply=AUDIOSET_SCORE_FUNCTION,
            cache_dir=ctx.cache_dir,
            cache_key_str=ctx.cache_key_for("ast", ast_model, params),
            provenance=ctx.provenance_for("ast", ast_model, params),
        )
        ast_outcome["window"] = {"win_length": ast_win_length, "hop_length": ast_hop_length}
        fragment["ast"] = ast_outcome
        ctx.write_sidecar("ast.json", ast_outcome)

    if yamnet_model is not None:
        # YAMNet runs in senselab's TF subprocess venv (same pattern as NeMo
        # Sortformer). senselab.classify_audios's `_is_yamnet()` dispatcher
        # matches on the raw model-id *string*, not on a SenselabModel wrapper —
        # passing HFModel here would fail validation (yamnet isn't on HF).
        params = {"win_length": yamnet_win_length, "hop_length": yamnet_hop_length, "top_k": top_k}
        yam_outcome = run_task_cached(
            "yamnet",
            classify_audios,
            [audio],
            model=yamnet_model,
            win_length=yamnet_win_length,
            hop_length=yamnet_hop_length,
            top_k=top_k,
            cache_dir=ctx.cache_dir,
            cache_key_str=ctx.cache_key_for("yamnet", yamnet_model, params),
            provenance=ctx.provenance_for("yamnet", yamnet_model, params),
        )
        yam_outcome["window"] = {"win_length": yamnet_win_length, "hop_length": yamnet_hop_length}
        fragment["yamnet"] = yam_outcome
        ctx.write_sidecar("yamnet.json", yam_outcome)

    # If both ran on the same grid, emit a side-by-side comparison.
    if (
        fragment.get("ast", {}).get("status") == "ok"
        and fragment.get("yamnet", {}).get("status") == "ok"
        and ast_win_length == yamnet_win_length
        and ast_hop_length == yamnet_hop_length
    ):
        agreement = _scene_agreement(
            ast_result=fragment["ast"]["result"],
            yamnet_result=fragment["yamnet"]["result"],
            win_length=ast_win_length,
            hop_length=ast_hop_length,
        )
        fragment["scene_agreement"] = agreement
        ctx.write_sidecar("scene_agreement.json", agreement)

    return fragment


def stage_features(audio: Audio, ctx: StageContext, *, win_length: float, hop_length: float) -> dict[str, Any]:
    """Extract temporal features and write one parquet sidecar per backend.

    Args:
        audio: The pass audio.
        ctx: Run environment.
        win_length: Window length for the summary-style backends, seconds.
        hop_length: Hop length for those backends, seconds.

    Returns:
        ``{"features": outcome}`` where ``outcome["result"]`` is the **live**
        ``{backend: rows}`` dict. The JSON sidecar gets a placeholder instead,
        because the rows themselves go to parquet — see the module docstring.
    """
    feat_params: dict[str, Any] = {
        "opensmile": "LowLevelDescriptors@native",
        "parselmouth": "windowed",
        "torchaudio_squim": "windowed",
        "device": ctx.device_label,
        "win_length": win_length,
        "hop_length": hop_length,
    }
    outcome = run_task_cached(
        "features",
        extract_temporal_features,
        audio,
        win_length=win_length,
        hop_length=hop_length,
        device=ctx.device,
        cache_dir=ctx.cache_dir,
        cache_key_str=ctx.cache_key_for("features", None, feat_params),
        provenance=ctx.provenance_for("features", None, feat_params),
    )
    # Each backend writes its own parquet sidecar — they have different columns
    # and different time grids (opensmile LLD is native ~10 ms;
    # parselmouth/torchaudio_squim follow the window args).
    result = outcome.get("result") or {}
    if isinstance(result, dict) and ctx.out_dir is not None:
        try:
            import pandas as pd

            feat_dir = ctx.out_dir / "features"
            feat_dir.mkdir(parents=True, exist_ok=True)
            for backend, rows in result.items():
                if not rows:
                    continue
                pd.DataFrame(rows).to_parquet(feat_dir / f"{backend}.parquet", index=False)
        except Exception as exc:  # noqa: BLE001 — best-effort sidecar
            print(f"  [features] warn: parquet write failed: {exc!r}", file=sys.stderr)
    ctx.write_sidecar("features.json", {**outcome, "result": "see features/*.parquet"})
    return {"features": outcome}


def stage_asr(
    audio: Audio, ctx: StageContext, *, models: Sequence[str], qwen_native_timestamps: bool = True
) -> dict[str, Any]:
    """Run each ASR model.

    Args:
        audio: The pass audio.
        ctx: Run environment.
        models: ASR model ids; empty means the stage contributes nothing.
        qwen_native_timestamps: When ``False``, Qwen3-ASR's bundled aligner is
            disabled so the MMS auto-align stage can take over instead.

    Returns:
        ``{"asr": {"by_model": {model_id: outcome}}}``.
    """
    by_model: dict[str, Any] = {}
    for model_id in models:
        asr_params: dict[str, Any] = {"device": ctx.device_label}
        extra_kwargs: dict[str, Any] = {}
        # Qwen3-ASR ships its own forced-aligner companion model; allow opt-out
        # so the MMS auto-align stage can take over.
        if model_id.startswith("Qwen/Qwen3-ASR") and not qwen_native_timestamps:
            extra_kwargs["return_timestamps"] = False
            asr_params["return_timestamps"] = False
        outcome = run_task_cached(
            f"asr[{model_id}]",
            transcribe_audios,
            [audio],
            model=model_for_task(model_id, task="asr"),
            device=ctx.device,
            cache_dir=ctx.cache_dir,
            cache_key_str=ctx.cache_key_for("asr", model_id, asr_params),
            provenance=ctx.provenance_for("asr", model_id, asr_params),
            **extra_kwargs,
        )
        by_model[model_id] = outcome
        ctx.write_sidecar(f"asr/{safe_model_id(model_id)}.json", outcome)
    return {"asr": {"by_model": by_model}}


def stage_alignment(
    audio: Audio,
    ctx: StageContext,
    *,
    asr_by_model: dict[str, Any],
    aligner: Literal["qwen", "mms"] = "qwen",
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    mms_aligner_model: str = "facebook/mms-1b-all",
    language: str = "en",
) -> dict[str, Any]:
    """Align text-only ASR outputs so they gain per-word timestamps.

    ``asr_by_model`` is an explicit parameter rather than a read out of a shared
    summary dict. That turns what was an invisible run-after-``stage_asr``
    ordering dependency into a checked signature, and lets a caller align a
    *cached* ASR block it did not produce — which is what the adaptive loop's
    escalation path does.

    The alignment cache is independent from the ASR cache (FR-024); a failed
    alignment preserves the ASR text and falls back to a single full-audio
    TextArea region in the LS export (FR-025).

    Args:
        audio: The pass audio.
        ctx: Run environment.
        asr_by_model: The ``{model_id: outcome}`` mapping from :func:`stage_asr`.
        aligner: Which aligner backend to use.
        qwen_aligner_model: Qwen forced-aligner model id.
        mms_aligner_model: MMS aligner model id.
        language: ISO language code for the aligner.

    Returns:
        ``{"alignment": {"by_model": {model_id: outcome}}}``, containing only the
        models that actually needed alignment.
    """
    by_model: dict[str, Any] = {}
    align_language = Language(language_code=language or "en")
    if aligner == "qwen":
        aligner_fn: Any = QwenASR.align_with_qwen
        aligner_model_id = qwen_aligner_model
    else:
        aligner_fn = align_transcriptions
        aligner_model_id = mms_aligner_model

    for model_id, asr_outcome in asr_by_model.items():
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
            # Levels-to-keep is part of the cache key — bumping its value
            # invalidates earlier entries that were stored with the all-False
            # default (which produced empty chunks).
            "levels_to_keep": "utterance+word",
        }
        align_provenance = {
            **ctx.provenance_for("alignment", aligner_model_id, aligner_params),
            "transcript_sha": transcript_sha,
            "language": align_language.language_code,
            "aligner_backend": aligner,
            "parent_asr_cache_key": asr_outcome.get("cache_key"),
        }
        outcome = run_alignment_cached(
            f"alignment[{model_id}]",
            aligner_fn,
            [(audio, ScriptLine(text=transcript_text), align_language)],
            # Keep word-level chunks (and the utterance wrapper) so the comparator
            # can read per-token timestamps. Default is all-False which filters
            # everything out and leaves a meaningless punctuation-only ScriptLine.
            levels_to_keep={"utterance": True, "word": True, "char": False},
            aligner_model=aligner_model_id,
            cache_dir=ctx.cache_dir,
            cache_key_str=ctx.align_key_for(
                transcript_sha=transcript_sha,
                language=align_language.language_code,
                aligner_model_id=aligner_model_id,
                aligner_params=aligner_params,
            ),
            provenance=align_provenance,
        )
        by_model[model_id] = outcome
        ctx.write_sidecar(f"alignment/{safe_model_id(model_id)}.json", outcome)
    return {"alignment": {"by_model": by_model}}


def stage_ppg(audio: Audio, ctx: StageContext) -> dict[str, Any]:
    """Extract phonetic posteriorgrams and write an argmax-per-frame sidecar.

    Args:
        audio: The pass audio.
        ctx: Run environment.

    Returns:
        ``{"ppgs": outcome}`` — note the plural key, which consumers read.
        ``outcome["phoneme_labels"]`` carries the inventory so the harvester can
        decode argmax indices without importing the ppgs library.
    """
    from senselab.audio.tasks.features_extraction.ppg import (
        _PHONEME_LABELS as _PPG_PHONEME_LABELS,
    )
    from senselab.audio.tasks.features_extraction.ppg import (
        extract_ppgs_from_audios,
    )
    from senselab.audio.workflows.audio_analysis.harvesters import ppg_argmax_per_frame

    params = {"device": ctx.device_label}
    outcome = run_task_cached(
        "ppgs",
        extract_ppgs_from_audios,
        [audio],
        device=ctx.device,
        cache_dir=ctx.cache_dir,
        cache_key_str=ctx.cache_key_for("ppgs", "ppgs/0.0.9", params),
        provenance=ctx.provenance_for("ppgs", "ppgs/0.0.9", params),
    )
    outcome["phoneme_labels"] = list(_PPG_PHONEME_LABELS)

    # The full (40 × N_frames) tensor is too large to dump; the argmax-per-frame
    # sequence + frame_hop is what the comparator actually consumes. Write it so
    # reviewers can inspect the phoneme timeline without rerunning the model.
    argmax_payload: dict[str, Any] = {
        "phoneme_labels": list(_PPG_PHONEME_LABELS),
        "per_frame_phonemes": [],
        "frame_hop_s": 0.0,
    }
    if outcome.get("status") == "ok":
        try:
            pf, fh = ppg_argmax_per_frame(
                outcome.get("result"),
                list(_PPG_PHONEME_LABELS),
                audio.waveform.shape[-1] / audio.sampling_rate,
            )
            argmax_payload["per_frame_phonemes"] = pf
            argmax_payload["frame_hop_s"] = float(fh)
        except Exception as exc:  # noqa: BLE001
            argmax_payload["argmax_error"] = repr(exc)
    ctx.write_sidecar(
        "ppgs.json",
        {
            **{k: v for k, v in outcome.items() if k != "result"},
            "result_summary": "argmax-per-frame sequence in 'argmax' field; full tensor in process memory only",
            "argmax": argmax_payload,
        },
    )
    return {"ppgs": outcome}


def stage_background_mask(
    ctx: StageContext,
    *,
    pass_summary: dict[str, Any],
    duration_s: float,
    task_type: str | None,
    grid: Any = None,  # noqa: ANN401 — BucketGrid; defaulted to avoid an import at call sites
    profile: Mapping[str, Any] | None = None,
    guard_interval_s: float | None = None,
    long_window_s: float = 10.24,
) -> dict[str, Any]:
    """Build the background mask for this pass and write its sidecars (T038, FR-031).

    Runs *after* diarization and scene classification because it consumes both: speech
    targets are evidenced by diarization, non-speech targets (breath, cough) by classifier
    labels. Ordering it earlier would leave a breath task with no evidence source and a
    mask that silently reported "never active" (FR-033a).

    Args:
        ctx: Stage context.
        pass_summary: The summary built so far — diarization and scene blocks must be in it.
        duration_s: Recording duration.
        task_type: Task name from metadata, or ``None`` for the conservative fallback.
        grid: Bucket grid; a default 0.5 s grid is used when omitted.
        profile: Detection-margin profile; the bundled default is loaded when omitted.
        guard_interval_s: Override for the profile's guard interval.
        long_window_s: Long-window classifier window, for the FR-045 support flag.

    Returns:
        A ``{"background_mask": {...}}`` fragment carrying the mask document.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import (
        build_mask,
        nontarget_confidence_by_bucket,
        target_confidence_by_bucket,
        target_event_types_for,
    )
    from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.io import write_background_mask

    resolved = dict(profile or load_detection_margin_profile())
    if guard_interval_s is not None:
        mask_cfg = dict(resolved.get("mask") or {})
        mask_cfg["guard_interval_s"] = float(guard_interval_s)
        resolved["mask"] = mask_cfg

    bucket_grid = grid or BucketGrid()
    buckets = [(b_start, b_end) for b_start, b_end, _idx in bucket_grid.iter_buckets(duration_s)]
    event_types, _provenance = target_event_types_for(task_type, resolved)
    active_threshold = float((resolved.get("mask") or {}).get("target_active_confidence", 0.6))
    rows = target_confidence_by_bucket(pass_summary, buckets, event_types, active_threshold=active_threshold)
    # Second quantity: is there *other* content where the target is absent. Without it the
    # mask cannot distinguish a silent pause from one carrying room tone or machine noise, and
    # only the second is worth introspecting — which is why a 21 s conversation previously
    # produced one uninformative region.
    scene_mass = _scene_source_mass(pass_summary, buckets)
    if scene_mass:
        rows = nontarget_confidence_by_bucket(rows, scene_mass)
    mask = build_mask(rows, task_type, profile=resolved, long_window_s=long_window_s)

    doc = mask.to_json()
    if ctx.out_dir is not None:
        write_background_mask(mask, ctx.out_dir)
    return {
        "background_mask": {
            "status": "ok",
            "result": doc,
            "provenance": ctx.provenance_for(
                "background_mask", None, {"task_type": task_type, "guard_interval_s": mask.guard_interval_s}
            ),
        }
    }


def run_pass(audio: Audio, ctx: StageContext, plan: PassPlan) -> dict[str, Any]:
    """Run the planned stages for one pass and return its summary.

    Stage order is load-bearing in exactly one place: alignment consumes
    :func:`stage_asr`'s output, which is why it is passed explicitly rather than
    read back out of the summary.

    Args:
        audio: The pass audio (already resampled / downmixed).
        ctx: Run environment.
        plan: Which stages to run and with what knobs.

    Returns:
        The pass summary — ``label``, ``duration_s``, ``audio_signature`` plus each
        stage's fragment.
    """
    duration_s = audio.waveform.shape[1] / audio.sampling_rate
    print(
        f"\n=== Pass: {ctx.pass_label} ({duration_s:.2f}s @ {audio.sampling_rate}Hz, "
        f"sig={ctx.audio_signature[:12]}...) ==="
    )
    summary: dict[str, Any] = {
        "label": ctx.pass_label,
        "duration_s": duration_s,
        "audio_signature": ctx.audio_signature,
    }

    if plan.diarization_models:
        summary.update(stage_diarization(audio, ctx, models=plan.diarization_models))

    summary.update(
        stage_scene(
            audio,
            ctx,
            ast_model=plan.ast_model,
            yamnet_model=plan.yamnet_model,
            ast_win_length=plan.ast_win_length,
            ast_hop_length=plan.ast_hop_length,
            yamnet_win_length=plan.yamnet_win_length,
            yamnet_hop_length=plan.yamnet_hop_length,
            top_k=plan.scene_top_k,
        )
    )

    if plan.features:
        summary.update(
            stage_features(audio, ctx, win_length=plan.features_win_length, hop_length=plan.features_hop_length)
        )

    if plan.asr_models:
        summary.update(
            stage_asr(audio, ctx, models=plan.asr_models, qwen_native_timestamps=plan.qwen_native_timestamps)
        )

    if plan.asr_models and plan.align_asr:
        summary.update(
            stage_alignment(
                audio,
                ctx,
                asr_by_model=summary["asr"]["by_model"],
                aligner=plan.aligner,
                qwen_aligner_model=plan.qwen_aligner_model,
                mms_aligner_model=plan.mms_aligner_model,
                language=plan.asr_language,
            )
        )

    if plan.ppg:
        summary.update(stage_ppg(audio, ctx))

    # Only on the unmodified variant. Measured on a real recording: the enhanced pass
    # masked 50% of the file against the unmodified pass's 17.9%, because speech
    # enhancement removes the non-speech evidence the mask reads target activity from.
    # A mask built there is misleadingly generous -- it reports "safe for background
    # claims" precisely where the background was destroyed.
    if plan.background_mask and ctx.variant == "unmodified":
        summary.update(
            stage_background_mask(
                ctx,
                pass_summary=summary,
                duration_s=duration_s,
                task_type=plan.task_type,
                guard_interval_s=plan.mask_guard_interval_s,
                long_window_s=plan.ast_win_length,
            )
        )
        if plan.background_sources:
            summary.update(stage_background_sources(audio, ctx, pass_summary=summary, duration_s=duration_s))
    elif plan.background_mask:
        summary["background_mask"] = {
            "status": "skipped",
            "reason": (
                f"variant={ctx.variant!r}: the mask is only meaningful on unmodified audio. "
                "Enhancement removes the non-speech evidence target activity is read from, so a "
                "mask built here would report more of the recording as safe for background claims "
                "exactly where the background was removed."
            ),
        }

    return summary


def stage_background_sources(
    audio: Any,  # noqa: ANN401 — senselab Audio
    ctx: StageContext,
    *,
    pass_summary: dict[str, Any],
    duration_s: float,
    profile: Mapping[str, Any] | None = None,
    suppression: Any = None,  # noqa: ANN401 — ForegroundSuppression
) -> dict[str, Any]:
    """Estimate the noise floor, screen candidates, and write the background outputs.

    Runs after the scene and mask stages because it consumes both: candidates come from the
    classifiers, and the mask says where a finding can be trusted without relying on
    suppression depth.

    Args:
        audio: The pass audio.
        ctx: Stage context, carrying the variant and gain every finding is attributed to.
        pass_summary: Summary built so far.
        duration_s: Recording duration.
        profile: Detection-margin profile; the bundled default is loaded when omitted.
        suppression: Foreground-suppression record, when the suppressed variant was built.

    Returns:
        A ``{"background_sources": {...}}`` fragment. Zero findings is a valid — and on
        noise-floor input, the *expected* — result.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile
    from senselab.audio.workflows.audio_analysis.io import (
        write_background_sources,
        write_noise_floor,
        write_suppression_json,
    )
    from senselab.audio.workflows.audio_analysis.noise_floor import (
        detect_stationary_sources,
        estimate_noise_floor,
        estimate_recorder_floor_db,
    )

    resolved = dict(profile or load_detection_margin_profile())
    wav = np.asarray(audio.waveform.squeeze().numpy(), dtype=np.float64)

    floors = estimate_noise_floor(wav, audio.sampling_rate, profile=resolved)
    recorder = estimate_recorder_floor_db(floors)
    floors = [
        type(f)(**{**f.__dict__, "recorder_floor_db": recorder}) if f.recorder_floor_db is None else f for f in floors
    ]
    # The unsubtracted pass: a source running through the whole recording is absorbed into
    # its own band floor, so it has to be found by comparing bands rather than by excess.
    stationary = detect_stationary_sources(floors, profile=resolved)

    if ctx.out_dir is not None:
        write_noise_floor(floors, ctx.out_dir)
        write_background_sources([], ctx.out_dir)
        if suppression is not None:
            write_suppression_json(suppression, ctx.out_dir)

    return {
        "background_sources": {
            "status": "ok",
            "result": {
                "bands": len(floors),
                "recorder_floor_db": recorder,
                "stationary_sources": stationary,
                "findings": [],
                "variant": ctx.variant,
                "gain_db": ctx.variant_gain_db,
            },
            "provenance": ctx.provenance_for("background_sources", None, {"bands": len(floors)}),
        }
    }


def _scene_source_mass(
    pass_summary: dict[str, Any],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], dict[str, float]]:
    """Per-bucket scene category mass from the AST / YAMNet blocks already in the summary.

    Reuses the classifier output the pass has: the evidence for "something other than the
    target is happening here" was being computed and then not shown to the mask.
    """
    from senselab.audio.workflows.audio_analysis.harvesters import classification_windows
    from senselab.audio.workflows.audio_analysis.sound_sources import (
        _category_for,
        load_source_category_map,
    )

    try:
        doc = load_source_category_map()
    except (OSError, ValueError):
        return {}
    mapping, default = dict(doc.get("map") or {}), str(doc.get("default") or "environment")

    per_bucket: dict[tuple[float, float], dict[str, float]] = {}
    for classifier in ("ast", "yamnet"):
        block = pass_summary.get(classifier)
        if not (isinstance(block, dict) and block.get("status") == "ok"):
            continue
        for window in classification_windows(block.get("result")) or []:
            if not isinstance(window, dict):
                continue
            w_start = float(window.get("start", 0.0) or 0.0)
            w_end = float(window.get("end", 0.0) or 0.0)
            overlapping = [
                (round(b_start, 6), round(b_end, 6))
                for b_start, b_end in buckets
                if not (b_end <= w_start or b_start >= w_end)
            ]
            if not overlapping:
                continue
            for label, score in zip(window.get("labels") or [], window.get("scores") or []):
                field = f"src_{_category_for(str(label), mapping, default)}"
                for key in overlapping:
                    slot = per_bucket.setdefault(key, {})
                    slot[field] = max(slot.get(field, 0.0), float(score))
    return per_bucket
