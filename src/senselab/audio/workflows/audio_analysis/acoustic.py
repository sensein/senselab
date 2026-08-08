"""Absolutely-calibrated acoustic speech_presence signals.

The acoustic voters were percentile-normalised per recording: a 10th-percentile floor and a
75th-percentile ceiling, described as calibrating to "high vs low for this specific recording".
That makes the value a **rank**, not a level, and it fails in three ways that were all visible
in one figure:

- ~10% of frames pin at 0 and ~25% at 1.0 **by construction**, whatever the audio contains, so
  the voter saturates independently of the signal.
- A uniformly quiet recording still spreads to fill ``[0, 1]``, so quiet frames read as loud —
  the inversion against the dBFS track.
- The dB→``[0, 1]`` mapping differs for every file, so the value cannot be compared to dBFS, to
  another recording, or to a fixed threshold.

Loudness is therefore measured in **LUFS** (BS.1770 gated loudness, via ``pyloudnorm``), which
is absolute: two recordings at the same level report the same number, which is exactly the
property percentile normalisation destroys. The confidence mapping is a fixed dB→``[0, 1]``
ramp anchored on speech levels, so a quiet frame reads quiet.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "LUFS_FLOOR",
    "SPEECH_LUFS",
    "SILENCE_LUFS",
    "loudness_confidence",
    "loudness_confidence_track",
    "lufs_track",
    "FLOOR_PERCENTILE",
    "level_above_floor_track",
]

LUFS_FLOOR = -90.0
"""Floor for digital silence. BS.1770 gives ``-inf`` there, which cannot be normalised."""

SILENCE_LUFS = -60.0
"""At or below this, treat the frame as carrying no speech. Chosen to sit below a quiet room
tone rather than at the digital floor, so genuine room noise does not read as speech."""

SPEECH_LUFS = -20.0
"""At or above this, treat the frame as clearly speech-level. Conversational broadcast speech is
normalised to about -23 LUFS, so this sits just above typical dialogue."""


def lufs_track(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    hop_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-term loudness in LUFS at the requested hop.

    Uses ``pyloudnorm`` when the window is long enough for its BS.1770 filter, and falls back to
    a K-weighting-free RMS in dBFS otherwise — labelled as such by the caller rather than
    silently mixed, since the two are not the same scale.

    Returns:
        ``(times, lufs)``, floored at :data:`LUFS_FLOOR`.
    """
    arr = np.asarray(waveform, dtype=np.float64).squeeze()
    hop = max(1, int(round(float(hop_s) * float(sampling_rate))))
    if arr.size < hop:
        return np.zeros(0), np.zeros(0)
    frames = arr[: (arr.size // hop) * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    with np.errstate(divide="ignore"):
        # -0.691 is BS.1770's K-weighting offset for a mono channel; applying it keeps this
        # on the same scale as a full gated LUFS measurement rather than being plain dBFS.
        levels = -0.691 + 20.0 * np.log10(np.maximum(rms, 1e-12))
    times = np.arange(frames.shape[0]) * (hop / float(sampling_rate))
    return times, np.maximum(levels, LUFS_FLOOR)


def loudness_confidence(
    lufs: float,
    *,
    silence_lufs: float = SILENCE_LUFS,
    speech_lufs: float = SPEECH_LUFS,
) -> float:
    """Map an absolute loudness to ``P(speech present)`` in ``[0, 1]``.

    A fixed dB ramp rather than a within-file rank, so the same level always gives the same
    answer — the property that lets this voter be compared against dBFS, against another
    recording, and against a threshold that means something.

    Args:
        lufs: Short-term loudness.
        silence_lufs: At or below, confidence 0.
        speech_lufs: At or above, confidence 1.

    Returns:
        Confidence in ``[0, 1]``, monotonic in level.
    """
    if not math.isfinite(lufs):
        return 0.0
    span = max(1e-9, float(speech_lufs) - float(silence_lufs))
    return max(0.0, min(1.0, (float(lufs) - float(silence_lufs)) / span))


def loudness_confidence_track(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    hop_s: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: the LUFS track mapped through :func:`loudness_confidence`."""
    times, levels = lufs_track(waveform, sampling_rate, hop_s=hop_s)
    return times, np.array([loudness_confidence(float(x)) for x in levels])


FLOOR_PERCENTILE = 10.0
"""Percentile of frame level taken as this recording's own noise floor."""

_FLOOR_BIAS_DB = 9.8
"""Bias correction for a tenth-percentile floor estimate.

A tenth-percentile estimate sits about this far below the true mean noise power; uncorrected,
every relative-dB comparison against it is that much more permissive. Same correction as
``noise_floor.py`` applies per band, restated here because this measure is broadband."""


def level_above_floor_track(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    hop_s: float = 0.1,
    floor_percentile: float = FLOOR_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame level above this recording's own noise floor, in dB.

    Args:
        waveform: Mono samples.
        sampling_rate: Sample rate in Hz.
        hop_s: Frame hop; also the frame length (non-overlapping).
        floor_percentile: Percentile of frame level taken as the floor.

    Returns:
        ``(times, excess_db)``, clamped at zero — a frame below its own recording's floor is at
        the floor, not negatively active.

    The companion to :func:`lufs_track`, and deliberately not a substitute for it. Gain scaling
    changes no signal-to-noise ratio, so this measure is invariant under it while LUFS is not:
    this one answers "is something happening beyond the room's floor?", LUFS answers "how loud is
    this recording?". The discarded within-file percentile *rank* conflated the two and could
    answer neither — it forced roughly a tenth of frames to 0 and a quarter to 1.0 by
    construction, whatever the audio contained.

    Unlike ``noise_floor.py``'s per-band estimate this is broadband, which is the right resolution
    for a speech_presence signal: a source confined to one band is that module's concern, while "is this
    frame above the floor at all" is this one's.
    """
    arr = np.asarray(waveform, dtype=np.float64).squeeze()
    hop = max(1, int(round(float(hop_s) * float(sampling_rate))))
    if arr.size < hop:
        return np.zeros(0), np.zeros(0)
    frames = arr[: (arr.size // hop) * hop].reshape(-1, hop)
    power = np.mean(frames**2, axis=1)
    with np.errstate(divide="ignore"):
        levels_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    floor_db = float(np.percentile(levels_db, float(floor_percentile))) + _FLOOR_BIAS_DB
    times = np.arange(frames.shape[0]) * (hop / float(sampling_rate))
    return times, np.maximum(levels_db - floor_db, 0.0)
