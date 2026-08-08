"""Minimal audio access for live interventions (no senselab data-structure deps).

Loads the run's input audio as 16 kHz mono float32, crops regions, and
regenerates the enhanced stream on demand (SepFormer) when a live backend is
available. Kept dependency-light so the loop degrades gracefully: every entry
point returns ``(payload, None)`` or ``(None, reason)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.perturbations import IDENTITY_NAME, IDENTITY_TRANSFORM

TARGET_SR = 16000


def load_wav_16k_mono(
    path: Path, *, backend: str = "auto", backend_out: dict[str, Any] | None = None
) -> tuple[Any | None, str | None]:  # noqa: ANN401 — np.ndarray
    """Read ``path`` → (float32 mono 16 kHz ndarray, None) or (None, reason).

    ``backend`` (policy ``audio_io_backend``):

    - ``"auto"`` (default): the senselab preprocessing path first — replicating
      analyze_audio's ``prepare_audio`` exactly (``read_audios →
      downmix_audios_to_mono → resample_audios``) so crops share
      ``audio_signature``/cache entries with pipeline-produced crops
      (architecture-review.md F3/T048) — then the soundfile+polyphase DSP
      fallback for artifact-driven runs outside a senselab install (different
      resampler ⇒ different signatures; results valid, not cache-shareable).
    - ``"senselab"``: strict — no fallback; fail loudly when the preprocessing
      stack is unavailable (recommended in production).
    - ``"dsp"``: pin the fallback loader (skips the senselab attempt entirely —
      useful where that stack is known-broken or too slow to probe).

    Never silent: the loader used lands in ``backend_out["loader"]`` and is
    surfaced in run provenance (``convergence.json → audio_backend``).
    """
    reason: str | None = None
    if backend in ("auto", "senselab"):
        wav, reason = _load_senselab(path)
        if wav is not None:
            if backend_out is not None:
                backend_out["loader"] = "senselab.preprocessing"
            return wav, None
        if backend == "senselab":
            return None, f"audio_fallback_forbidden ({reason})"
    wav, fb_reason = _load_fallback(path, senselab_reason=reason)
    if wav is not None and backend_out is not None:
        backend_out["loader"] = "dsp_fallback"
        if reason is not None:
            backend_out["senselab_path_reason"] = reason
    return wav, fb_reason


def _load_senselab(path: Path) -> tuple[Any | None, str | None]:  # noqa: ANN401
    try:
        from senselab.audio.tasks.input_output import read_audios  # noqa: PLC0415
        from senselab.audio.tasks.preprocessing import (  # noqa: PLC0415
            downmix_audios_to_mono,
            resample_audios,
        )
    except ImportError as exc:
        return None, f"senselab_audio_unavailable ({getattr(exc, 'name', exc)})"
    try:
        import numpy as np  # noqa: PLC0415

        audio = read_audios([str(path)])[0]
        audio = downmix_audios_to_mono([audio])[0]
        if audio.sampling_rate != TARGET_SR:
            audio = resample_audios([audio], resample_rate=TARGET_SR)[0]
        wav = audio.waveform.detach().cpu().numpy().squeeze()
        return np.ascontiguousarray(wav, dtype="float32"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"senselab_audio_failed ({exc!r})"


def _load_fallback(path: Path, *, senselab_reason: str | None) -> tuple[Any | None, str | None]:  # noqa: ANN401
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        return None, f"audio_io_unavailable ({exc.name}; senselab path: {senselab_reason})"
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        return None, f"audio_read_failed ({exc!r})"
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        try:
            from scipy.signal import resample_poly
        except ImportError:
            return None, f"audio_io_unavailable (scipy; senselab path: {senselab_reason})"
        from math import gcd

        g = gcd(int(sr), TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, int(sr) // g).astype("float32")
    return np.ascontiguousarray(mono, dtype="float32"), None


def crop(wav: Any, start_s: float, end_s: float) -> Any:  # noqa: ANN401
    """Slice ``[start_s, end_s)`` (clipped); timestamps of outputs must add ``start_s`` back."""
    lo = max(0, int(round(start_s * TARGET_SR)))
    hi = min(len(wav), int(round(end_s * TARGET_SR)))
    return wav[lo:hi]


def get_stream_wav(ctx: dict[str, Any], stream: str) -> tuple[Any | None, str | None]:  # noqa: ANN401
    """Waveform for one perturbation: the identity loads the input file, any other is regenerated.

    Dispatch is on the **declared transform** from ``L1/perturbations.json`` (``ctx["perturbations"]``),
    not on the perturbation's name. It used to be a two-armed comparison against the two pass
    names of the day (``perturbations.py`` records which), so a third perturbation was an edit
    here — in a module that has no business knowing any perturbation's name.

    Results are cached in ``ctx["_wav_cache"]``. When the backend for a transform is unavailable the
    caller decides whether the identity is an acceptable fallback (recorded as ``stream_fallback``
    in the intervention log).
    """
    cache = ctx.setdefault("_wav_cache", {})
    if stream in cache:
        return cache[stream]
    input_audio = ctx.get("input_audio")
    io_backend = str((ctx.get("policy") or {}).get("audio_io_backend", "auto"))
    backend_out: dict[str, Any] = {}
    result: tuple[Any | None, str | None]
    transform = _declared_transform(ctx, stream)
    if not input_audio or not Path(input_audio).exists():
        result = (None, "input_audio_missing")
    elif transform is None:
        result = (None, f"unknown_stream ({stream})")
    elif transform == IDENTITY_TRANSFORM:
        result = load_wav_16k_mono(Path(input_audio), backend=io_backend, backend_out=backend_out)
        if backend_out:
            ctx.setdefault("audio_backend", {})[stream] = backend_out
    elif transform == "speech_enhanced":
        raw, reason = get_stream_wav(ctx, IDENTITY_NAME)
        result = (None, reason) if raw is None else _enhance(raw, ctx)
    else:
        result = (None, f"transform_not_regenerable ({transform})")
    cache[stream] = result
    return result


def _declared_transform(ctx: dict[str, Any], stream: str) -> str | None:
    """The transform ``stream`` declared in the run's register, or ``None`` if it declared none.

    ``ctx["perturbations"]`` is the register as loaded by the loop. The identity name resolves
    even with no register at all: a live-audio caller that never ran L1 still has the recording,
    and refusing it would make the loop dependent on an artifact it does not need.
    """
    for entry in ctx.get("perturbations") or []:
        if isinstance(entry, dict) and str(entry.get("name")) == stream:
            return str(entry.get("transform") or "") or None
    return IDENTITY_TRANSFORM if stream == IDENTITY_NAME else None


def _enhance(wav: Any, ctx: dict[str, Any]) -> tuple[Any | None, str | None]:  # noqa: ANN401
    """Whole-file enhancement, routed through ``tasks/speech_enhancement`` (T048).

    Mirrors analyze_audio's enhanced pass (same default model, same task API) so
    the regenerated stream matches pipeline semantics; the direct speechbrain
    call remains only as the degraded-environment fallback.
    """
    model_id = ctx["policy"].get("enhancement_model", "speechbrain/sepformer-wham16k-enhancement")
    try:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        from senselab.audio.data_structures import Audio  # noqa: PLC0415
        from senselab.audio.tasks.speech_enhancement import enhance_audios  # noqa: PLC0415
        from senselab.utils.data_structures import SpeechBrainModel  # noqa: PLC0415

        audio = Audio(waveform=torch.from_numpy(wav).unsqueeze(0), sampling_rate=TARGET_SR)
        enhanced = enhance_audios([audio], model=SpeechBrainModel(path_or_uri=model_id))[0]
        return np.ascontiguousarray(enhanced.waveform.detach().cpu().numpy().squeeze(), dtype="float32"), None
    except ImportError as exc:
        senselab_reason = f"senselab_enhancement_unavailable ({getattr(exc, 'name', exc)})"
    except Exception as exc:  # noqa: BLE001
        senselab_reason = f"senselab_enhancement_failed ({exc!r})"
    try:
        import torch
        from speechbrain.inference.separation import SepformerSeparation

        from senselab.utils.dependencies import resolve_model
    except ImportError as exc:
        return None, f"enhancement_backend_unavailable ({getattr(exc, 'name', exc)}; {senselab_reason})"
    try:
        # Stage once (download-once via the heartbeat lock) + load from the local
        # snapshot dir so SpeechBrain makes no per-file Hub HEAD (429 source under
        # batch). SpeechBrain has no revision arg -> pin via the snapshot path.
        _, _snapshot = resolve_model(str(model_id), "main")
        model = SepformerSeparation.from_hparams(source=str(_snapshot), run_opts={"device": "cpu"})
        with torch.no_grad():
            est = model.separate_batch(torch.from_numpy(wav).unsqueeze(0))
        out = est[0, :, 0].cpu().numpy().astype("float32")
        peak = max(1e-9, float(abs(out).max()))
        if peak > 1.0:
            out = out / peak
        return out, None
    except Exception as exc:  # noqa: BLE001 — model/download failures degrade to a reason
        return None, f"enhancement_failed ({exc!r}; {senselab_reason})"
