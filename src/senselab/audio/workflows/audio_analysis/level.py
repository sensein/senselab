"""Audio-variant level measurement, gain policy, and provenance (T008, data-model.md §3).

Every result the scene stages produce must be attributable to the **audio variant** it was
computed from and the **gain** applied to that variant (FR-012, SC-006). This module owns
that record and the measurement behind it.

Why the gain policy is shaped the way it is
-------------------------------------------

Measurement established three things that constrain it:

1. **Neither scene classifier normalizes input level.** Both are amplitude-sensitive, so
   the level a classifier sees changes what it reports on unchanged audio.
2. **Amplification changes no signal-to-noise ratio.** Attenuate-then-reamplify is
   bit-exact in floating point (:func:`apply_gain_db` is tested for this), so gain never
   *recovers* content — it only keeps a classifier's absolute floor from destroying it.
   Detection therefore lives in ``noise_floor``/``sources``, not here.
3. **Above roughly +10 dB the classifiers respond to clipping distortion**, so the cap is
   a correctness boundary rather than a preference. Exceeding it raises
   :class:`GainCapExceededError` instead of clamping: a silently clamped gain makes the
   recorded provenance wrong, which is worse than a failed run.

Loudness uses ITU-R BS.1770 gated integrated loudness via ``pyloudnorm``. The
broadcast-style ``-23 LUFS`` target is chosen for **headroom**, not convention — it leaves
about 23 dB below full scale, enough to apply the capped gain without clipping, where a
streaming-style ``-14 LUFS`` target would leave only 14 dB.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from senselab.utils.data_structures.logging import logger

__all__ = [
    "AudioVariant",
    "GainCapExceededError",
    "apply_gain_db",
    "clipped_fraction",
    "integrated_lufs",
    "loudness_range_lu",
    "measure_variant",
    "normalization_gain_db",
    "peak_limited_gain_db",
    "true_peak_dbtp",
    "write_level_json",
]

VARIANT_NAMES = ("unmodified", "speech_enhanced", "foreground_suppressed")

_TRUE_PEAK_OVERSAMPLE = 4
"""BS.1770 Annex 2 specifies at least 4x oversampling for true-peak metering."""

_LRA_LOW_PCT, _LRA_HIGH_PCT = 10.0, 95.0
"""EBU Tech 3342 loudness-range percentiles.

P10 stops a fade-out dominating the lower edge; P95 stops one unusually loud event
dominating the upper edge.
"""

_LRA_ABS_GATE_LUFS = -70.0
_LRA_REL_GATE_LU = -20.0
"""EBU Tech 3342 gating.

The relative gate exists to separate the weakest *real* signal from the noise floor —
close in spirit to this feature's detection margin, though applied to programme loudness
rather than per-band excess.
"""


class GainCapExceededError(ValueError):
    """A requested gain exceeds the policy cap.

    Raised rather than clamping, because a clamped gain would be recorded as the requested
    one and every downstream provenance claim would be wrong.
    """


@dataclass(frozen=True)
class AudioVariant:
    """One named version of a recording, with the gain applied and its measurements."""

    name: str
    gain_db: float
    measured_lufs: float
    lra_lu: float
    true_peak_dbtp: float
    clipped_fraction: float
    requantized: bool = False
    target_lufs: float | None = None
    per_segment_gain_db: list[dict[str, float]] = field(default_factory=list)
    segment_rms_dbfs: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize, encoding non-finite loudness as ``None`` so the JSON stays valid."""
        return {
            "name": self.name,
            "gain_db": self.gain_db,
            "measured_lufs": _finite_or_none(self.measured_lufs),
            "lra_lu": _finite_or_none(self.lra_lu),
            "true_peak_dbtp": _finite_or_none(self.true_peak_dbtp),
            "clipped_fraction": self.clipped_fraction,
            "requantized": self.requantized,
            "target_lufs": self.target_lufs,
            "per_segment_gain_db": list(self.per_segment_gain_db),
            "segment_rms_dbfs": dict(self.segment_rms_dbfs),
        }


def _finite_or_none(value: float) -> float | None:
    """JSON has no ``-Infinity``; silence legitimately measures ``-inf``."""
    return float(value) if math.isfinite(value) else None


def _as_mono_f64(waveform: np.ndarray) -> np.ndarray:
    """Collapse to 1-D float64, which is what the loudness meter expects."""
    arr = np.asarray(waveform, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    return arr


def apply_gain_db(waveform: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale ``waveform`` by ``gain_db``, in float — no clipping, no requantization.

    Exact in both directions: applying ``-g`` then ``+g`` returns the input bit-for-bit.
    That property is the reason gain cannot be a detection mechanism.
    """
    return np.asarray(waveform, dtype=np.float64) * (10.0 ** (float(gain_db) / 20.0))


def integrated_lufs(waveform: np.ndarray, sampling_rate: int) -> float:
    """ITU-R BS.1770 gated integrated loudness, in LUFS.

    Returns ``-inf`` for digital silence or for material entirely below the standard's
    absolute gate — honest, where a large finite negative would be invented.
    """
    arr = _as_mono_f64(waveform)
    if arr.size == 0 or not np.any(arr):
        return -math.inf
    import pyloudnorm

    meter = pyloudnorm.Meter(sampling_rate)
    # The meter needs at least one full gating block (400 ms).
    min_len = int(0.4 * sampling_rate) + 1
    if arr.size < min_len:
        arr = np.pad(arr, (0, min_len - arr.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(meter.integrated_loudness(arr))
    return value if math.isfinite(value) else -math.inf


def loudness_range_lu(waveform: np.ndarray, sampling_rate: int) -> float:
    """EBU Tech 3342 loudness range (P95 − P10 of doubly-gated short-term loudness), in LU.

    Reported for both variants: a large raw-versus-residual change is a cheap,
    standards-grounded signal that suppression altered the dynamic structure.
    """
    arr = _as_mono_f64(waveform)
    win = int(3.0 * sampling_rate)
    if arr.size < win or not np.any(arr):
        return 0.0
    import pyloudnorm

    meter = pyloudnorm.Meter(sampling_rate)
    hop = max(1, int(0.1 * sampling_rate))  # >= 10 Hz, per Tech 3342
    short_term: list[float] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for start in range(0, arr.size - win + 1, hop):
            block = arr[start : start + win]
            if not np.any(block):
                continue
            value = float(meter.integrated_loudness(block))
            if math.isfinite(value) and value > _LRA_ABS_GATE_LUFS:
                short_term.append(value)
    if len(short_term) < 2:
        return 0.0
    values = np.asarray(short_term, dtype=np.float64)
    gated = values[values > (float(np.mean(values)) + _LRA_REL_GATE_LU)]
    if gated.size < 2:
        return 0.0
    return float(np.percentile(gated, _LRA_HIGH_PCT) - np.percentile(gated, _LRA_LOW_PCT))


def true_peak_dbtp(waveform: np.ndarray, sampling_rate: int) -> float:
    """Oversampled true peak, in dBTP (BS.1770 Annex 2).

    Oversampling matters here: a signal whose per-sample peak sits below full scale can
    still overshoot between samples, and that overshoot is what a classifier's front end
    would see as clipping.
    """
    arr = _as_mono_f64(waveform)
    if arr.size == 0 or not np.any(arr):
        return -math.inf
    # Zero-stuffed band-limited interpolation via FFT resampling.
    from scipy import signal as sp_signal

    up = sp_signal.resample_poly(arr, _TRUE_PEAK_OVERSAMPLE, 1)
    peak = float(np.max(np.abs(up)))
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def clipped_fraction(waveform: np.ndarray, *, threshold: float = 0.999_9) -> float:
    """Fraction of samples pinned at or beyond full scale (FR-017d).

    Reported rather than corrected: silently clamping would hand the classifier distorted
    audio while the provenance claimed clean audio.
    """
    arr = _as_mono_f64(waveform)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) >= threshold) / arr.size)


def normalization_gain_db(waveform: np.ndarray, sampling_rate: int, *, target_lufs: float) -> float:
    """Gain needed to bring ``waveform`` to ``target_lufs``.

    Returns ``0.0`` for silence — there is no gain that normalizes silence toward a
    target, and returning an infinite one would poison every downstream record.

    The caller is expected to apply the **same** returned scalar to every variant of the
    recording (FR-019c). Renormalizing each variant independently would corrupt the
    cross-variant comparison this feature exists to make.
    """
    measured = integrated_lufs(waveform, sampling_rate)
    if not math.isfinite(measured):
        logger.debug("normalization_gain_db: non-finite loudness (silence?); returning 0 dB")
        return 0.0
    return float(target_lufs) - measured


def peak_limited_gain_db(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    target_lufs: float,
    true_peak_ceiling_dbtp: float,
    gain_cap_db: float,
) -> tuple[float, str]:
    """Gain toward ``target_lufs``, reduced so the true peak stays under the ceiling.

    Three limits bind, and the smallest wins:

    1. the loudness target,
    2. the headroom between the current true peak and the ceiling,
    3. the policy gain cap.

    The headroom limit is not hypothetical. A recording with a high crest factor — loud
    transients over a quiet median, which is the normal shape for close-microphone
    capture — can sit far below a loudness target while its peak is already at full scale.
    Normalizing on loudness alone then clips, and the classifiers respond to the resulting
    distortion rather than to content. Reducing the gain is the honest fix: a limiter would
    keep the target at the cost of distorting exactly the transients under analysis.

    Args:
        waveform: Samples.
        sampling_rate: Sample rate in Hz.
        target_lufs: Loudness target.
        true_peak_ceiling_dbtp: Maximum permitted true peak.
        gain_cap_db: Policy cap.

    Returns:
        ``(gain_db, binding_limit)`` where ``binding_limit`` is ``"target"``,
        ``"true_peak"``, ``"gain_cap"``, or ``"unmeasurable"`` (loudness below BS.1770's
        absolute gate) — recorded so a variant that never reached its target is not
        mistaken for one that did.
    """
    measured = integrated_lufs(waveform, sampling_rate)
    peak = true_peak_dbtp(waveform, sampling_rate)
    headroom = (float(true_peak_ceiling_dbtp) - peak) if math.isfinite(peak) else float("inf")
    if not math.isfinite(measured):
        # Below BS.1770's absolute gate the loudness is not measurable, so there is no
        # target to normalize toward. Reporting "target" here would claim the material
        # was already at the target when in fact nothing could be measured — a variant
        # that was never normalized must not look like one that was.
        return min(0.0, headroom), "unmeasurable"
    limits = {
        "target": float(target_lufs) - measured,
        "true_peak": headroom,
        "gain_cap": float(gain_cap_db),
    }
    binding = min(limits, key=lambda k: limits[k])
    return limits[binding], binding


def measure_variant(
    name: str,
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    gain_db: float,
    gain_cap_db: float,
    requantized: bool = False,
    target_lufs: float | None = None,
    per_segment_gain_db: Sequence[dict[str, float]] | None = None,
    segment_rms_dbfs: dict[str, float] | None = None,
) -> AudioVariant:
    """Measure one variant and record its provenance.

    Args:
        name: Variant name; one of :data:`VARIANT_NAMES`.
        waveform: The variant's samples, **after** ``gain_db`` has been applied.
        sampling_rate: Sample rate in Hz.
        gain_db: Gain already applied to ``waveform``.
        gain_cap_db: Policy cap. Binds on ``gain_db`` and on every per-segment gain.
        requantized: Whether a lossy serialization occurred in this variant's path.
        target_lufs: Normalization target, when one was applied.
        per_segment_gain_db: Per-segment gains, each ``{start, end, gain_db}`` (FR-019a).
        segment_rms_dbfs: Per-segment pre-gain levels, for the FR-020a reject.

    Returns:
        The populated :class:`AudioVariant`.

    Raises:
        GainCapExceededError: If ``gain_db`` or any per-segment gain exceeds ``gain_cap_db``.
        ValueError: If ``name`` is not a recognized variant name.
    """
    if name not in VARIANT_NAMES:
        raise ValueError(f"unknown audio variant {name!r}; expected one of {VARIANT_NAMES}")
    segments = [dict(s) for s in (per_segment_gain_db or [])]
    over = [float(g) for g in [gain_db, *[s.get("gain_db", 0.0) for s in segments]] if float(g) > float(gain_cap_db)]
    if over:
        raise GainCapExceededError(
            f"variant {name!r}: requested gain {max(over)} dB exceeds gain_cap_db {gain_cap_db} dB. "
            "Refusing rather than clamping — a clamped gain would be recorded as the requested one."
        )
    return AudioVariant(
        name=name,
        gain_db=float(gain_db),
        measured_lufs=integrated_lufs(waveform, sampling_rate),
        lra_lu=loudness_range_lu(waveform, sampling_rate),
        true_peak_dbtp=true_peak_dbtp(waveform, sampling_rate),
        clipped_fraction=clipped_fraction(waveform),
        requantized=requantized,
        target_lufs=target_lufs,
        per_segment_gain_db=segments,
        segment_rms_dbfs=dict(segment_rms_dbfs or {}),
    )


def write_level_json(
    out_dir: Path | str,
    *,
    target_lufs: float,
    gain_cap_db: float,
    variants: Sequence[AudioVariant],
) -> Path:
    """Write ``<out_dir>/level.json`` per ``contracts/level-verdicts.md``.

    Args:
        out_dir: Run directory.
        target_lufs: The normalization target in force.
        gain_cap_db: The gain cap in force.
        variants: One entry per audio variant; names must be unique.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If ``variants`` is empty or contains duplicate names — a duplicate
            makes every downstream variant reference ambiguous.
    """
    if not variants:
        raise ValueError("write_level_json requires at least one variant")
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate variant names in level.json: {dupes}")
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "level.json"
    out.write_text(
        json.dumps(
            {
                "target_lufs": float(target_lufs),
                "gain_cap_db": float(gain_cap_db),
                "variants": [v.to_json() for v in variants],
            },
            indent=2,
        )
        + "\n"
    )
    return out
