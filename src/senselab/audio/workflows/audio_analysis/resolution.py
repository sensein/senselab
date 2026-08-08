"""Per-signal temporal resolution, declared at L1 and converted at L2.

Forcing every signal onto one bucket grid loses information in both directions, and the losses
were both measured on real runs:

- A frame posterior at ~17 ms collapsed onto 250 ms buckets **saturates**. The VAD trace came
  out flat at 1.0 across a conversation with four clear pauses, because a bucket containing one
  speech frame was reported as fully active.
- An AST decision spanning 10.24 s spread across those same buckets **claims precision it does
  not have**, which is why its scene composition row was nearly constant: three real decisions
  stretched over eighty-odd buckets.

So L1 declares its resolution and L2 converts. The declaration travels with the signal because
a resolution inferred at fusion time is a guess about what the harvester did — and the two
failures above are exactly what that guess gets wrong.

Conversion direction matters:

- **Coarser → finer is a hold.** A 10 s decision applies across its whole window; interpolating
  between windows would invent detail the model never produced.
- **Finer → coarser is an integral.** Point-sampling a 17 ms posterior at 250 ms discards
  fourteen of every fifteen measurements and which one survives is arbitrary; averaging keeps
  what they collectively said, and is what stops the saturation above.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "NATIVE_RESOLUTION_S",
    "declared_resolution_s",
    "resample_series",
]

NATIVE_RESOLUTION_S: dict[str, float] = {
    # pyannote receptive-field step, shared by segmentation-3.0 and brouhaha.
    "frame_segmentation": 0.017,
    "frame_brouhaha_vad": 0.017,
    # AST's fixed input length. Overridable per run: a windowed AST with a shorter hop earns
    # the finer figure, which is the point of running it windowed.
    "ast": 10.24,
    # YAMNet's own frame rate.
    "yamnet": 0.48,
}
"""Native resolution per signal family, keyed by the prefix the harvester assigns.

Kept beside the extractors that produce them rather than scattered across call sites: a
resolution repeated at three call sites is three places to get it wrong."""


def declared_resolution_s(
    signal: str,
    *,
    hop_s: float | None = None,
    grid_s: float = 0.25,
) -> float:
    """The resolution at which a signal actually decides.

    Args:
        signal: Signal name.
        hop_s: Actual hop, when the signal was run windowed. Wins over the table — a windowed
            AST genuinely decides more often than its input length suggests.
        grid_s: Fallback for a signal with no declaration. The bucket grid rather than the
            finest available: an unknown signal must not silently claim frame precision.

    Returns:
        Resolution in seconds.
    """
    if hop_s is not None:
        return float(hop_s)
    name = str(signal)
    if name in NATIVE_RESOLUTION_S:
        return NATIVE_RESOLUTION_S[name]
    for prefix, value in NATIVE_RESOLUTION_S.items():
        if name.startswith(prefix):
            return value
    return float(grid_s)


def resample_series(
    times: Sequence[float],
    values: Sequence[float],
    *,
    target_hop_s: float,
    duration_s: float,
    kind: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a signal onto a target resolution.

    Args:
        times: Source sample times, ascending.
        values: Source values, aligned with ``times``.
        target_hop_s: Target spacing.
        duration_s: Extent to cover.
        kind: ``"mean"`` to integrate when going coarser, ``"hold"`` to hold when going finer.

    Returns:
        ``(target_times, target_values)``. A target bucket with no source sample is ``NaN``,
        never ``0.0`` — a gap must stay a gap, and zero would assert the signal reported
        absence there.
    """
    src_t = np.asarray(times, dtype=np.float64)
    src_v = np.asarray(values, dtype=np.float64)
    n = max(1, int(math.ceil(max(duration_s, target_hop_s) / target_hop_s)))
    out_t = np.arange(n) * float(target_hop_s)
    out_v = np.full(n, np.nan)
    if src_t.size == 0:
        return out_t, out_v

    for i, start in enumerate(out_t):
        end = start + float(target_hop_s)
        inside = (src_t >= start) & (src_t < end)
        if inside.any():
            out_v[i] = float(np.mean(src_v[inside])) if kind == "mean" else float(src_v[inside][0])
            continue
        if kind == "hold":
            # No sample in this bucket: hold the most recent earlier one, which is what a
            # window-scoped decision means. Nothing earlier leaves it unmeasured.
            earlier = np.nonzero(src_t <= start)[0]
            if earlier.size:
                out_v[i] = float(src_v[earlier[-1]])
    return out_t, out_v
