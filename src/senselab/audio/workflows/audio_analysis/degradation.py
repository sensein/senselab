"""L2 scene-quality degradation: measurements to ``[0, 1]`` scores against calibrated anchors.

The anchors here are *calibration* — claims about what counts as clean for a task — which is why
they live at L2 and why a fitted profile may replace them. Holding them at L1 destroyed the
underlying measurements: ``clip((25 − snr_db)/20, 0, 1)`` returned ``0.0`` in every bucket of every
recording measured, because clean speech sits at 60–70 dB SNR against a 25 dB anchor. See
``specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md``.

Two rules the functions here obey.

**A missing measurement stays missing.** ``None`` in gives ``None`` out, never ``0.0``. A degraded
score of zero is the confident claim "this audio is clean"; producing it from an estimator that
failed would manufacture confidence from an absence.

**Saturation is visible, not silent.** Every score reports whether it hit an anchor, so a column
pinned at an extreme can be recognised as anchor-limited rather than read as a measurement. That
distinction took a figure and six defects to notice the first time.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = [
    "DEFAULT_ANCHORS",
    "bandwidth_degradation",
    "clip_degradation",
    "reverb_degradation",
    "scene_degradation",
    "snr_degradation",
]

DEFAULT_ANCHORS: dict[str, float] = {
    # Speech-appropriate SNR anchors: clean conversational or studio speech sits at ~25 dB+, a
    # noisy recording at 10-15 dB reads mid-scale, and heavy noise at <=5 dB saturates. Studio TTS
    # measures far above the clean anchor and correctly reads 0.0 — that is the anchor working, and
    # is only a defect if the underlying dB value is not also recorded (it is, at L1).
    "snr_clean_db": 25.0,
    "snr_floor_db": 5.0,
    # C50 above ~30 dB is a dry room; -5 dB is heavily reverberant.
    "c50_clean_db": 30.0,
    "c50_floor_db": -5.0,
}
"""Uncalibrated but documented defaults, replaced by a fitted calibration profile when supplied."""


def _ramp(
    value: Optional[float],
    clean: float,
    floor: float,
) -> Optional[float]:
    """Linear ``[0, 1]`` degradation between two dB anchors, ``0`` at ``clean``.

    Returns ``None`` for a missing value or a non-positive span — an inverted anchor pair is a
    configuration error, and mapping it to ``0.0`` would report the audio as clean on the strength
    of a broken profile.
    """
    if value is None:
        return None
    span = clean - floor
    if span <= 0:
        return None
    return max(0.0, min(1.0, (clean - float(value)) / span))


def snr_degradation(
    snr_db: Optional[float],
    *,
    clean_db: float = DEFAULT_ANCHORS["snr_clean_db"],
    floor_db: float = DEFAULT_ANCHORS["snr_floor_db"],
) -> Optional[float]:
    """Noise degradation from an SNR measurement in dB. ``0`` = clean, ``1`` = fully degraded."""
    return _ramp(snr_db, clean_db, floor_db)


def reverb_degradation(
    c50_db: Optional[float],
    *,
    clean_db: float = DEFAULT_ANCHORS["c50_clean_db"],
    floor_db: float = DEFAULT_ANCHORS["c50_floor_db"],
) -> Optional[float]:
    """Reverberation degradation from a C50 clarity measurement in dB."""
    return _ramp(c50_db, clean_db, floor_db)


def bandwidth_degradation(rolloff_hz: Optional[float], *, sampling_rate: int) -> Optional[float]:
    """Band-limiting degradation: how far the spectral roll-off sits below Nyquist.

    Args:
        rolloff_hz: Frequency below which most spectral energy sits, from L1.
        sampling_rate: Sample rate, which fixes Nyquist.

    Returns:
        ``0`` when the signal rolls off at Nyquist (full-band), approaching ``1`` as content is
        truncated — telephone-band speech at ~3.4 kHz against an 8 kHz Nyquist reads ~0.58. ``None``
        when the roll-off was not measured.

    The comparison against Nyquist is the reason this belongs at L2: "band-limited is bad" is a
    task-dependent judgement, and L1 reports the roll-off as a frequency so a task that does not
    care can ignore it.
    """
    if rolloff_hz is None:
        return None
    nyquist = float(sampling_rate) / 2.0
    if nyquist <= 0:
        return None
    return max(0.0, min(1.0, 1.0 - float(rolloff_hz) / nyquist))


def clip_degradation(proportion_clipped: Optional[float]) -> Optional[float]:
    """Clipping degradation. Already a proportion, so this only bounds and preserves ``None``."""
    if proportion_clipped is None:
        return None
    return max(0.0, min(1.0, float(proportion_clipped)))


SNR_PREFERENCE: tuple[str, ...] = ("snr_brouhaha_db", "snr_spectral_gating_db", "snr_peak_db")
"""Which SNR estimator to score, most-preferred first.

An explicit, *recorded* preference order rather than an average. The three estimators use different
noise-floor definitions, so their mean is not an estimate of any one quantity — which is also why
their standard deviation, once reported as ``quality_uncertainty``, measured definitional
disagreement and pinned at 1.0 structurally. Selection is legitimately a fusion decision and so
belongs here; what was wrong before was making it silently at L1 and reporting one anonymous
number. :func:`scene_degradation` returns ``snr_source`` naming the estimator actually used.
"""


def scene_degradation(
    measurements: Mapping[str, Any],
    *,
    sampling_rate: int,
    calibration: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Score one bucket's L1 scene-quality measurements.

    Args:
        measurements: An L1 row from :func:`quality.harvest_quality_measurements`.
        sampling_rate: Sample rate, for the Nyquist comparison.
        calibration: Optional fitted anchors overriding :data:`DEFAULT_ANCHORS`.

    Returns:
        ``{"quality_snr", "quality_reverb", "quality_bandwidth", "quality_clip"}`` each in
        ``[0, 1]`` or ``None`` where its measurement was unavailable, plus ``snr_source`` naming
        which estimator produced ``quality_snr`` (``None`` when none was available).

        ``snr_source`` exists so that a change in the reported score can be attributed: a bucket
        scored from Brouhaha and its neighbour scored from the spectral-gating fallback are not
        measuring the same thing, and without the attribution that discontinuity looks like a
        change in the audio.
    """
    anchors = {**DEFAULT_ANCHORS, **(calibration or {})}
    snr_source = next((name for name in SNR_PREFERENCE if measurements.get(name) is not None), None)
    return {
        "snr_source": snr_source,
        "quality_snr": snr_degradation(
            measurements.get(snr_source) if snr_source else None,
            clean_db=float(anchors["snr_clean_db"]),
            floor_db=float(anchors["snr_floor_db"]),
        ),
        "quality_reverb": reverb_degradation(
            measurements.get("c50_brouhaha_db"),
            clean_db=float(anchors["c50_clean_db"]),
            floor_db=float(anchors["c50_floor_db"]),
        ),
        "quality_bandwidth": bandwidth_degradation(
            measurements.get("rolloff_95_hz"),
            sampling_rate=sampling_rate,
        ),
        "quality_clip": clip_degradation(measurements.get("proportion_clipped")),
    }
