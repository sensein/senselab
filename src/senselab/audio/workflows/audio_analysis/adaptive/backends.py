"""Guarded gateways from the adaptive loop to senselab's task APIs.

Post-T047 (architecture-review.md F2) this module no longer *implements* model
capabilities — every function delegates to the owning task/workflow API and
exists only to provide the loop's failure envelope: lazy imports, a
``(result, reason)`` return instead of exceptions, crop/offset bookkeeping, and
an explicit degraded-environment fallback where one is justified.

- U1 re-ASR → ``senselab.audio.tasks.speech_to_text.transcribe_audios``
  (word-level timestamps via ``return_timestamps="word"``); a bare HF-pipeline
  fallback remains for environments where the senselab task stack cannot import.
- U3 consensus alignment → ``senselab.audio.tasks.forced_alignment.mms_fa``.
- I1/I2 fine-hop embeddings → the workflow's own
  ``audio_analysis.embeddings.extract_per_window_embeddings``.
- P2 fine speech posteriors → Brouhaha's VAD head (same 16.9 ms hop as segmentation-3.0).
- I4 overlap → cross-diarizer spans (FR-016), not one model's per-class channels.

Nothing here is file- or model-id-specific: models, windows, and hops come from
the policy.
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.audio_io import TARGET_SR

_ASR_PIPELINE_CACHE: dict[str, Any] = {}


def _to_audio(wav_crop: Any) -> Any:  # noqa: ANN401 — returns senselab Audio
    """Wrap a 1-D float32 numpy crop as a 16 kHz mono senselab ``Audio``."""
    import torch  # noqa: PLC0415

    from senselab.audio.data_structures import Audio  # noqa: PLC0415

    return Audio(waveform=torch.from_numpy(wav_crop).unsqueeze(0), sampling_rate=TARGET_SR)


# ── U1: region re-ASR ────────────────────────────────────────────────────


def transcribe_crop(
    wav_crop: Any,  # noqa: ANN401 — np.ndarray
    *,
    model_id: str,
    offset_s: float,
    language: str | None = None,
    meta: dict[str, Any] | None = None,
    backend: str = "auto",
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Word-timestamped transcription of one crop, in FILE time (offset applied).

    ``backend`` (policy ``u1_backend``): ``"auto"`` (default) tries the senselab
    speech-to-text task API first (arbitrary ``HFModel`` ids,
    ``return_timestamps="word"``) and falls back to a bare HF pipeline;
    ``"senselab"`` / ``"pipeline"`` pin one path. The backend actually used is
    recorded in ``meta["backend"]`` (never silent).
    """
    reason: str | None = None
    if backend in ("auto", "senselab"):
        words, reason = _transcribe_crop_senselab(wav_crop, model_id=model_id, offset_s=offset_s)
        if words is not None:
            if meta is not None:
                meta["backend"] = "senselab.speech_to_text"
            return words, None
        if backend == "senselab":
            return None, reason
    fallback_words, fb_reason = _transcribe_crop_pipeline(
        wav_crop, model_id=model_id, offset_s=offset_s, language=language
    )
    if fallback_words is not None:
        if meta is not None:
            meta["backend"] = "hf_pipeline" if backend == "pipeline" else "hf_pipeline_fallback"
            if reason is not None:
                meta["senselab_path_reason"] = reason
        return fallback_words, None
    return None, f"{reason}; fallback: {fb_reason}" if reason else fb_reason


def _transcribe_crop_senselab(
    wav_crop: Any,  # noqa: ANN401
    *,
    model_id: str,
    offset_s: float,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        from senselab.audio.tasks.speech_to_text import transcribe_audios  # noqa: PLC0415
        from senselab.utils.data_structures import HFModel  # noqa: PLC0415
    except ImportError as exc:
        return None, f"senselab_asr_unavailable ({getattr(exc, 'name', exc)})"
    try:
        lines = transcribe_audios([_to_audio(wav_crop)], model=HFModel(path_or_uri=model_id), return_timestamps="word")
    except Exception as exc:  # noqa: BLE001
        return None, f"senselab_asr_failed ({exc!r})"
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import iter_word_leaves  # noqa: PLC0415

    serialized = [line.model_dump() if hasattr(line, "model_dump") else line for line in lines or []]
    words = [
        {"text": w["text"], "start": round(w["start"] + offset_s, 4), "end": round(w["end"] + offset_s, 4)}
        for w in iter_word_leaves(serialized)
    ]
    return words, None


def _transcribe_crop_pipeline(
    wav_crop: Any,  # noqa: ANN401
    *,
    model_id: str,
    offset_s: float,
    language: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as exc:
        return None, f"asr_backend_unavailable ({getattr(exc, 'name', exc)})"
    try:
        if model_id not in _ASR_PIPELINE_CACHE:
            _ASR_PIPELINE_CACHE[model_id] = pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        asr = _ASR_PIPELINE_CACHE[model_id]
        kwargs: dict[str, Any] = {"return_timestamps": "word"}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        out = asr({"raw": wav_crop, "sampling_rate": TARGET_SR}, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return None, f"asr_failed ({exc!r})"
    words: list[dict[str, Any]] = []
    for ch in out.get("chunks") or []:
        ts = ch.get("timestamp") or (None, None)
        text = (ch.get("text") or "").strip()
        if not text or ts[0] is None:
            continue
        end = ts[1] if ts[1] is not None else ts[0]
        words.append({"text": text, "start": round(float(ts[0]) + offset_s, 4), "end": round(float(end) + offset_s, 4)})
    return words, None


# ── I1/I2: fine-hop speaker embeddings ───────────────────────────────────


def embed_windows(
    wav: Any,  # noqa: ANN401
    *,
    model_id: str,
    span: tuple[float, float],
    win_s: float,
    hop_s: float,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fine-hop speaker embeddings over ``span`` → ``[{start_s, end_s, vector}]`` (file time).

    Delegates to the workflow's uniform-grid extractor (which itself routes
    through ``tasks/speaker_embeddings``) on the cropped waveform.
    """
    try:
        from senselab.audio.workflows.audio_analysis.embeddings import (  # noqa: PLC0415
            extract_per_window_embeddings,
        )
    except ImportError as exc:
        return None, f"embedding_backend_unavailable ({getattr(exc, 'name', exc)})"
    try:
        lo, hi = int(round(span[0] * TARGET_SR)), int(round(span[1] * TARGET_SR))
        failures: dict[str, str] = {}
        per_model = extract_per_window_embeddings(
            audio=_to_audio(wav[lo:hi]), models=[model_id], window_s=win_s, hop_s=hop_s, failures=failures
        )
        windows = per_model.get(model_id) or []
        if not windows:
            return None, f"embedding_failed ({failures.get(model_id, 'no windows produced')})"
        return [
            {
                "start_s": round(float(w.start_s) + span[0], 4),
                "end_s": round(float(w.end_s) + span[0], 4),
                "vector": [float(x) for x in w.vector.tolist()],
            }
            for w in windows
        ], None
    except Exception as exc:  # noqa: BLE001
        return None, f"embedding_failed ({exc!r})"


# ── I4: overlap posteriors ───────────────────────────────────────────────


def speech_posteriors(
    wav: Any,  # noqa: ANN401
    *,
    span: tuple[float, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Continuous per-frame speech probability over ``span``, from Brouhaha's VAD head.

    P2's purpose is **localisation**: it fires when a region's votes are dominated by coarse voters,
    each casting one identical vote across every bucket it spans, so agreement among them is an
    artifact of window size rather than evidence about the bucket. Re-measuring at frame resolution
    on the crop is the answer.

    Brouhaha rather than ``segmentation-3.0``, whose per-speaker channels nothing uses any more (D-19):
    its VAD head runs at **the same 16.9 ms hop**, so nothing is lost at the one thing P2 exists for.

    **What is lost, stated because it is a real reduction.** This is the same model that already voted
    in round 0, so P2 now buys locality — the same estimator on a crop, which is a genuine
    re-measurement because a model given a short span sees different context — but not a second
    independent opinion. Under ``segmentation-3.0`` it bought both.
    """
    import os  # noqa: PLC0415

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return None, "posteriors_unavailable (HF token required for pyannote/brouhaha)"
    try:
        from senselab.audio.tasks.scene_quality import extract_brouhaha_frames  # noqa: PLC0415
    except ImportError as exc:
        return None, f"posteriors_unavailable ({getattr(exc, 'name', exc)})"
    try:
        lo, hi = int(round(span[0] * TARGET_SR)), int(round(span[1] * TARGET_SR))
        frames = extract_brouhaha_frames([_to_audio(wav[lo:hi])])[0]
    except Exception as exc:  # noqa: BLE001
        return None, f"posteriors_failed ({exc!r})"
    if frames is None:
        return None, "posteriors_unavailable (model load/access failed — see logs)"
    return {
        "frame_hop": float(frames.frame_hop_s),
        "speech": [float(x) for x in frames.vad],
        "model_id": "pyannote/brouhaha",
    }, None


def overlap_track_from_spans(
    by_model: Any,  # noqa: ANN401 — pass summary's diarization.by_model
    *,
    span: tuple[float, float],
    hop: float = 0.016875,
) -> tuple[dict[str, Any] | None, str | None]:
    """Per-frame overlap over ``span``, from **cross-diarizer** spans rather than one model's channels.

    I4 asks whether two people were talking at once. That is derivable from diarization output —
    ``segmentation-3.0`` was the local segmentation model *inside* ``community-1``, so the pipeline
    already computes it — and it comes from more than one tool, which is the evidence this design
    prefers everywhere else.

    Two things this depends on, both established rather than assumed:

    - The diarizers must emit the **overlapping** view. senselab asked pyannote for
      ``exclusive_speaker_diarization``, a partition where concurrent speech is resolved away; on a
      constructed clip that view did not merely drop the overlap, it **lost the second speaker
      entirely**. ``exclusive=False`` is now wired.
    - Overlap here is a **decision, not a posterior**: 1.0 where two or more distinct speakers cover
      the instant, 0.0 where fewer do. A soft probability would need a model that reports one, and
      pretending to have one from hard spans would manufacture confidence. The count it comes from is
      censored per tool (D-19).

    Returns ``None`` when no diarizer contributed spans — different from an overlap track of zeros.
    """
    from senselab.audio.workflows.audio_analysis.occupancy import count_at, spans_from_diarization  # noqa: PLC0415

    spans_by_tool = spans_from_diarization(by_model or {})
    if not spans_by_tool:
        return None, "overlap_unavailable (no diarizer produced spans for this span)"
    start, end = float(span[0]), float(span[1])
    n = max(1, int(round((end - start) / hop)))
    track = []
    for i in range(n):
        t = start + (i + 0.5) * hop
        track.append(1.0 if max(count_at(s, t) for s in spans_by_tool.values()) > 1 else 0.0)
    return {
        "frame_hop": hop,
        "overlap": track,
        "contributing_models": sorted(spans_by_tool),
        "is_decision": True,
    }, None


# ── U3: consensus word re-alignment ──────────────────────────────────────


def consensus_align(
    wav: Any,  # noqa: ANN401
    words: list[dict[str, Any]],
    *,
    timeout_s: float = 600.0,
    backend: str = "qwen",
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """U3 (C8): time the consensus word sequence against the audio.

    Needed because the consensus is a sequence **no single model produced**: its per-word timings
    would otherwise be a vote over member timings for a word order none of them emitted, which can
    come out non-monotonic. ``fusion.consensus_alignment: off`` keeps the member-vote timings.

    **Default backend is the Qwen forced aligner, the same one the pre-fusion path uses.** This was
    hard-coded to torchaudio MMS_FA with no way to choose, which left the pipeline running two
    aligners — Qwen3-ForcedAligner before fusion, MMS after — and D-1 moved Canary off MMS precisely
    so that word-boundary differences would "reflect the models, not two different aligners". A third
    aligner appearing after fusion reintroduced what that decision removed.

    The trade is real and worth knowing rather than discovering: Qwen's aligner already times
    Qwen3-ASR (bundled) and Canary (externally), so a Qwen-timed consensus shares its source with
    most members and the published boundary sits closer to theirs by construction. MMS is
    independent of every member but is a third opinion nobody asked for. Consistency won because the
    per-edge confidences measure spread *among members*, which either choice leaves untouched — what
    changes is only whether the published value is drawn from inside or outside that set.

    Returns ``(spans, None)``, or ``(None, reason)`` when the backend is unavailable, the aligner
    returns a count that does not match the words given, or the timeout fires. Never raises for
    those: the caller keeps member timings and records the reason.
    """
    texts = [str(w["text"]) for w in words]
    if not texts:
        return None, "no_words_to_align"

    if str(backend).lower() == "mms":
        try:
            from senselab.audio.tasks.forced_alignment.mms_fa import align_words_mms_fa  # noqa: PLC0415
        except ImportError as exc:
            return None, f"aligner_backend_unavailable ({getattr(exc, 'name', exc)})"
        return align_words_mms_fa(wav, texts, timeout_s=timeout_s)

    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from senselab.audio.data_structures import Audio  # noqa: PLC0415
        from senselab.audio.tasks.speech_to_text.qwen import QwenASR  # noqa: PLC0415
        from senselab.utils.data_structures import Language, ScriptLine  # noqa: PLC0415
    except ImportError as exc:
        return None, f"aligner_backend_unavailable ({getattr(exc, 'name', exc)})"

    try:
        waveform = torch.as_tensor(np.asarray(wav, dtype=np.float32)).reshape(1, -1)
        audio = Audio(waveform=waveform, sampling_rate=16000)
        aligned = QwenASR.align_with_qwen(
            [(audio, ScriptLine(text=" ".join(texts)), Language(language_code="en"))],
            aligner_model=aligner_model,
        )
    except Exception as exc:  # noqa: BLE001 — a failed realignment must not fail the run
        return None, f"qwen_alignment_failed ({type(exc).__name__})"

    leaves = [leaf for line in (aligned[0] if aligned else []) for leaf in _word_leaves(line)]
    spans = [{"start": float(t[0]), "end": float(t[1])} for t in leaves]
    if len(spans) != len(texts):
        # A count mismatch means the aligner and the caller disagree about which word is which, so
        # every span after the divergence would be attached to the wrong word. Refuse rather than
        # publish a plausible-looking misalignment.
        return None, f"qwen_alignment_count_mismatch ({len(spans)} != {len(texts)})"
    return spans, None


def _word_leaves(line: Any) -> list[tuple[float, float]]:  # noqa: ANN401 — ScriptLine tree
    """Deepest timed nodes of a ScriptLine, in order — the words the aligner placed."""
    chunks = getattr(line, "chunks", None)
    if chunks:
        return [span for child in chunks for span in _word_leaves(child)]
    start, end = getattr(line, "start", None), getattr(line, "end", None)
    return [(float(start), float(end))] if start is not None and end is not None else []


def senselab_transcribe_available() -> bool:
    """True when the full senselab ASR stack is importable (U1 primary path)."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("senselab.audio.tasks.speech_to_text") is not None
        )
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
