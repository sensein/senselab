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
- I4 overlap posteriors → ``voice_activity_detection.frame_posteriors``
  (``include_per_class=True`` + ``FramePosterior.overlap_probs`` — FR-016).

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


def overlap_posteriors(
    wav: Any,  # noqa: ANN401
    *,
    span: tuple[float, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Per-class segmentation posteriors over ``span`` → speech + overlap tracks (FR-016).

    Delegates to ``extract_speech_frame_posteriors(include_per_class=True)`` and
    ``FramePosterior.overlap_probs()``; frames are span-local.
    """
    import os  # noqa: PLC0415

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return None, "posteriors_unavailable (HF token required for pyannote/segmentation-3.0)"
    try:
        from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (  # noqa: PLC0415
            extract_speech_frame_posteriors,
        )
    except ImportError as exc:
        return None, f"posteriors_unavailable ({getattr(exc, 'name', exc)})"
    try:
        lo, hi = int(round(span[0] * TARGET_SR)), int(round(span[1] * TARGET_SR))
        fp = extract_speech_frame_posteriors([_to_audio(wav[lo:hi])], include_per_class=True)[0]
    except Exception as exc:  # noqa: BLE001
        return None, f"posteriors_failed ({exc!r})"
    if fp is None:
        return None, "posteriors_unavailable (model load/access failed — see logs)"
    overlap = fp.overlap_probs()
    if overlap is None:
        return None, "posteriors_unexpected_shape (per-class posteriors missing)"
    return {
        "frame_hop": float(fp.frame_hop_s),
        "overlap": [float(x) for x in overlap],
        "speech": [float(x) for x in fp.probs],
        "n_classes": int(fp.per_class.shape[1]) if fp.per_class is not None else None,
    }, None


# ── U3: consensus word re-alignment ──────────────────────────────────────


def consensus_align(
    wav: Any,  # noqa: ANN401
    words: list[dict[str, Any]],
    *,
    timeout_s: float = 600.0,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """U3 (C8): align the consensus word sequence via the forced-alignment task's MMS_FA backend."""
    try:
        from senselab.audio.tasks.forced_alignment.mms_fa import align_words_mms_fa  # noqa: PLC0415
    except ImportError as exc:
        return None, f"aligner_backend_unavailable ({getattr(exc, 'name', exc)})"
    return align_words_mms_fa(wav, [w["text"] for w in words], timeout_s=timeout_s)


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
