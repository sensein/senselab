"""Minimal audio access for live interventions (no senselab data-structure deps).

Loads the run's input audio as 16 kHz mono float32, crops regions, and
regenerates the enhanced stream on demand (SepFormer) when a live backend is
available. Kept dependency-light so the loop degrades gracefully: every entry
point returns ``(payload, None)`` or ``(None, reason)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TARGET_SR = 16000


def load_wav_16k_mono(path: Path) -> tuple[Any | None, str | None]:  # noqa: ANN401 — np.ndarray
    """Read ``path`` → (float32 mono 16 kHz ndarray, None) or (None, reason)."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        return None, f"audio_io_unavailable ({exc.name})"
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        return None, f"audio_read_failed ({exc!r})"
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        try:
            from scipy.signal import resample_poly
        except ImportError:
            return None, "audio_io_unavailable (scipy)"
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
    """Waveform for a pass: raw loads the input file; enhanced is regenerated on demand.

    Results are cached in ``ctx["_wav_cache"]``. The enhanced stream mirrors
    analyze_audio's whole-file SepFormer pass (research.md D4); when the
    backend is unavailable the caller decides whether raw is an acceptable
    fallback (recorded as ``stream_fallback`` in the intervention log).
    """
    cache = ctx.setdefault("_wav_cache", {})
    if stream in cache:
        return cache[stream]
    input_audio = ctx.get("input_audio")
    result: tuple[Any | None, str | None]
    if not input_audio or not Path(input_audio).exists():
        result = (None, "input_audio_missing")
    elif stream == "raw_16k":
        result = load_wav_16k_mono(Path(input_audio))
    elif stream == "enhanced_16k":
        raw, reason = get_stream_wav(ctx, "raw_16k")
        result = (None, reason) if raw is None else _enhance(raw, ctx)
    else:
        result = (None, f"unknown_stream ({stream})")
    cache[stream] = result
    return result


def _enhance(wav: Any, ctx: dict[str, Any]) -> tuple[Any | None, str | None]:  # noqa: ANN401
    """Whole-file SepFormer enhancement (same default model as analyze_audio)."""
    model_id = ctx["policy"].get("enhancement_model", "speechbrain/sepformer-wham16k-enhancement")
    try:
        import torch
        from speechbrain.inference.separation import SepformerSeparation
    except ImportError as exc:
        return None, f"enhancement_backend_unavailable ({getattr(exc, 'name', exc)})"
    try:
        model = SepformerSeparation.from_hparams(source=model_id, run_opts={"device": "cpu"})
        with torch.no_grad():
            est = model.separate_batch(torch.from_numpy(wav).unsqueeze(0))
        out = est[0, :, 0].cpu().numpy().astype("float32")
        peak = max(1e-9, float(abs(out).max()))
        if peak > 1.0:
            out = out / peak
        return out, None
    except Exception as exc:  # noqa: BLE001 — model/download failures degrade to a reason
        return None, f"enhancement_failed ({exc!r})"
