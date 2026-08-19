"""The DSP envelope span-bounder, and its check against the two verified cough spans.

The envelope hop is 1 ms so that a +-5 ms verdict is not limited by the grid: quantisation is a
fifth of the claimed tolerance, which means a measured error of 40 ms cannot be blamed on it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as sps

FRAME_MS = 4.0
HOP_MS = 1.0
HIGHPASS_HZ = 80.0
LOCAL_FLOOR_S = 1.5
FLOOR_PERCENTILE = 10.0


def envelope(wave: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Short-time RMS in dB on a 1 ms grid, after an 80 Hz high-pass.

    Returns:
        ``(times, env_db)`` — frame centre times in seconds and the envelope in dB.
    """
    sos = sps.butter(4, HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    filtered = sps.sosfiltfilt(sos, wave)
    frame = max(2, int(round(FRAME_MS * 1e-3 * sr)))
    hop = max(1, int(round(HOP_MS * 1e-3 * sr)))
    window = np.hanning(frame)
    n_frames = 1 + max(0, (len(filtered) - frame) // hop)
    starts = np.arange(n_frames) * hop
    view = np.lib.stride_tricks.as_strided(
        filtered, shape=(n_frames, frame), strides=(filtered.strides[0] * hop, filtered.strides[0])
    )
    rms = np.sqrt(np.mean((view * window) ** 2, axis=1) + 1e-20)
    times = (starts + frame / 2.0) / sr
    return times, 20.0 * np.log10(rms)


def local_floor_db(times: np.ndarray, env_db: np.ndarray, centre: float) -> float:
    """The 10th-percentile envelope level within +-1.5 s of ``centre``.

    Local rather than global: a bounder that reads the whole file's floor would be handed the
    verified-empty stretches, which a blind run does not have.
    """
    mask = np.abs(times - centre) <= LOCAL_FLOOR_S
    return float(np.percentile(env_db[mask], FLOOR_PERCENTILE))


def bound_span(
    wave: np.ndarray,
    sr: int,
    proposal: Tuple[float, float],
    delta_db: float,
    precomputed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, float]:
    """Refine ``proposal`` to the span where the envelope exceeds the local floor by ``delta_db``.

    The peak is sought inside the proposal, then the edges walk outward to the first frame below
    threshold. Edges are allowed to leave the proposal — a proposer window is a neighbourhood, not
    a bound — but not to cross the midpoint of a neighbouring proposal.

    Args:
        wave: Mono samples.
        sr: Sample rate.
        proposal: ``(start, end)`` seconds from the proposer.
        delta_db: Threshold height above the local floor.
        precomputed: An ``envelope()`` result to reuse.

    Returns:
        ``onset``, ``offset``, ``peak_time``, ``floor_db``, ``threshold_db``, ``peak_db``.
    """
    times, env_db = precomputed if precomputed is not None else envelope(wave, sr)
    inside = (times >= proposal[0]) & (times <= proposal[1])
    if not inside.any():
        raise ValueError(f"proposal {proposal} contains no envelope frame")
    local = np.flatnonzero(inside)
    peak_i = int(local[int(np.argmax(env_db[inside]))])
    floor = local_floor_db(times, env_db, float(times[peak_i]))
    threshold = floor + delta_db

    left = peak_i
    while left > 0 and env_db[left - 1] >= threshold:
        left -= 1
    right = peak_i
    last = len(env_db) - 1
    while right < last and env_db[right + 1] >= threshold:
        right += 1

    return {
        "onset": float(times[left]),
        "offset": float(times[right]),
        "peak_time": float(times[peak_i]),
        "floor_db": floor,
        "threshold_db": threshold,
        "peak_db": float(env_db[peak_i]),
    }


DELTAS_DB: List[float] = [3.0, 6.0, 10.0, 12.0, 15.0, 20.0]
