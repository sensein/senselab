"""Live model backends for interventions — all lazily imported and guarded.

Each function returns ``(result, None)`` on success or ``(None, reason)`` when
the backend, weights, or gated access are unavailable, so the policy engine
can turn missing capability into ``blocked_guard`` + ``next_actions`` instead
of a crash. Nothing here is file-specific: model ids, windows, and hops all
come from the policy.
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.audio_io import TARGET_SR

_ASR_CACHE: dict[str, Any] = {}
_EMB_CACHE: dict[str, Any] = {}


def transcribe_crop(
    wav_crop: Any,  # noqa: ANN401 — np.ndarray
    *,
    model_id: str,
    offset_s: float,
    language: str | None = None,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Word-timestamped transcription of one crop via the HF whisper pipeline.

    Returns word leaves ``[{text, start, end}]`` in FILE time (offset applied).
    Used by U1 when the full senselab ASR backends are not importable; in the
    full environment U1 prefers ``senselab.audio.tasks.speech_to_text``.
    """
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as exc:
        return None, f"asr_backend_unavailable ({getattr(exc, 'name', exc)})"
    try:
        if model_id not in _ASR_CACHE:
            _ASR_CACHE[model_id] = pipeline("automatic-speech-recognition", model=model_id, device="cpu")
        asr = _ASR_CACHE[model_id]
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


def embed_windows(
    wav: Any,  # noqa: ANN401
    *,
    model_id: str,
    span: tuple[float, float],
    win_s: float,
    hop_s: float,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fine-hop speaker embeddings over ``span`` → ``[{start_s, end_s, vector}]``.

    The live upgrade path for I1/I2; the stored-artifact path (per-window
    embeddings persisted by analyze_audio) is preferred when its hop is
    already fine enough (policy ``identity.max_stored_hop_s``).
    """
    try:
        import torch  # noqa: PLC0415
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415
    except ImportError as exc:
        return None, f"embedding_backend_unavailable ({getattr(exc, 'name', exc)})"
    try:
        if model_id not in _EMB_CACHE:
            _EMB_CACHE[model_id] = EncoderClassifier.from_hparams(source=model_id, run_opts={"device": "cpu"})
        enc = _EMB_CACHE[model_id]
        out: list[dict[str, Any]] = []
        t = span[0]
        while t + win_s <= span[1] + 1e-9:
            lo, hi = int(round(t * TARGET_SR)), int(round((t + win_s) * TARGET_SR))
            seg = torch.from_numpy(wav[lo:hi]).unsqueeze(0)
            with torch.no_grad():
                vec = enc.encode_batch(seg).squeeze().cpu().numpy()
            out.append({"start_s": round(t, 4), "end_s": round(t + win_s, 4), "vector": vec.tolist()})
            t += hop_s
        return out, None
    except Exception as exc:  # noqa: BLE001
        return None, f"embedding_failed ({exc!r})"


def overlap_posteriors(
    wav: Any,  # noqa: ANN401
    *,
    span: tuple[float, float],
) -> tuple[dict[str, Any] | None, str | None]:
    """Per-class segmentation-3.0 posteriors over ``span`` → overlap posterior (FR-016).

    Gated model: requires HF token access to ``pyannote/segmentation-3.0``.
    Returns ``{"frame_hop": h, "overlap": [p...], "speech": [p...]}`` in span-local frames.
    """
    import os  # noqa: PLC0415

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return None, "posteriors_unavailable (HF token required for pyannote/segmentation-3.0)"
    try:
        import torch  # noqa: PLC0415
        from pyannote.audio import Inference, Model  # noqa: PLC0415
    except ImportError as exc:
        return None, f"posteriors_unavailable ({getattr(exc, 'name', exc)})"
    try:
        import numpy as np  # noqa: PLC0415

        span_len = span[1] - span[0]
        chunk_s, step_s = min(10.0, span_len), 2.0
        model = Model.from_pretrained("pyannote/segmentation-3.0")
        if model is None:
            return None, "posteriors_unavailable (segmentation-3.0 load returned None)"
        inference = Inference(model, duration=chunk_s, step=step_s)
        lo, hi = int(round(span[0] * TARGET_SR)), int(round(span[1] * TARGET_SR))
        waveform = torch.from_numpy(wav[lo:hi]).unsqueeze(0)
        scores = inference({"waveform": waveform, "sample_rate": TARGET_SR})
        arr = np.asarray(scores.data, dtype=float)  # type: ignore[union-attr]

        # pyannote 3.x: (frames, 7) powerset probs. pyannote 4.x Inference on a
        # sliding window: (chunks, frames_per_chunk, K) — K=3 per-speaker
        # activations (powerset already decoded to multilabel). Stitch chunks
        # by overlap-averaging into an absolute frame grid.
        if arr.ndim == 2:
            chunks = arr[None]
        elif arr.ndim == 3:
            chunks = arr
        else:
            return None, f"posteriors_unexpected_shape ({arr.shape})"
        n_chunks, frames_per_chunk, n_classes = chunks.shape
        hop = chunk_s / max(1, frames_per_chunk)
        total_frames = int(round(((n_chunks - 1) * step_s + chunk_s) / hop))
        acc = np.zeros((total_frames, n_classes))
        cnt = np.zeros((total_frames, 1))
        for i in range(n_chunks):
            start_f = int(round(i * step_s / hop))
            end_f = min(total_frames, start_f + frames_per_chunk)
            acc[start_f:end_f] += chunks[i, : end_f - start_f]
            cnt[start_f:end_f] += 1.0
        probs = np.clip(acc / np.maximum(cnt, 1.0), 0.0, 1.0)

        row_sums = probs.sum(axis=1)
        if n_classes >= 7 and float(np.nanmean(np.abs(row_sums - 1.0))) < 0.1:
            # Powerset [∅, s1, s2, s3, s1s2, s1s3, s2s3].
            speech = 1.0 - probs[:, 0]
            overlap = probs[:, 4:].sum(axis=1)
        else:
            # Multilabel per-speaker activations: speech = any speaker active;
            # overlap = second-highest activation (P of a 2nd concurrent speaker).
            sorted_desc = np.sort(probs, axis=1)[:, ::-1]
            speech = sorted_desc[:, 0]
            overlap = sorted_desc[:, 1] if n_classes >= 2 else np.zeros(total_frames)
        return {
            "frame_hop": hop,
            "overlap": overlap.tolist(),
            "speech": speech.tolist(),
            "n_classes": int(n_classes),
        }, None
    except Exception as exc:  # noqa: BLE001
        return None, f"posteriors_failed ({exc!r})"


def consensus_align(
    wav: Any,  # noqa: ANN401
    words: list[dict[str, Any]],
    *,
    timeout_s: float = 600.0,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """U3 (C8): force-align the consensus word sequence via torchaudio's MMS_FA bundle.

    Returns a list of ``{start, end}`` (file time) matching ``words`` 1:1, or
    ``(None, reason)``. Guarded by a SIGALRM timeout so a cold-cache bundle
    download (~1.2 GB) cannot wedge the loop — on timeout the fusion keeps its
    weighted member timestamps (the documented fallback).
    """
    import re
    import signal

    try:
        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415
        from torchaudio.pipelines import MMS_FA as bundle  # noqa: PLC0415, N811
    except ImportError as exc:
        return None, f"aligner_backend_unavailable ({getattr(exc, 'name', exc)})"

    norm = [re.sub(r"[^a-z']", "", w["text"].lower()) for w in words]
    if any(not t for t in norm):
        return None, "unalignable_tokens (non-romanizable consensus words)"

    def _raise_timeout(signum: int, frame: Any) -> None:  # noqa: ANN401, ARG001
        raise TimeoutError(f"mms_fa_timeout ({timeout_s}s)")

    old_handler = None
    timer_armed = False
    try:
        old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.1, timeout_s))
        timer_armed = True
    except ValueError:  # not in main thread — proceed unguarded
        old_handler = None
    try:
        model = bundle.get_model()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()
        with torch.no_grad():
            emission, _ = model(torch.from_numpy(wav).unsqueeze(0))
            spans = aligner(emission[0], tokenizer(norm))
        ratio = len(wav) / emission.shape[1]
        out = []
        for span in spans:
            start = span[0].start * ratio / TARGET_SR
            end = span[-1].end * ratio / TARGET_SR
            out.append({"start": round(float(start), 4), "end": round(float(end), 4)})
        if len(out) != len(words):
            return None, f"alignment_count_mismatch ({len(out)} != {len(words)})"
        return out, None
    except TimeoutError as exc:
        return None, f"aligner_timeout ({exc})"
    except Exception as exc:  # noqa: BLE001
        return None, f"alignment_failed ({exc!r})"
    finally:
        if timer_armed:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


def senselab_transcribe_available() -> bool:
    """True when the full senselab ASR stack is importable (full-env U1 path)."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("senselab.audio.tasks.speech_to_text") is not None
        )
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
