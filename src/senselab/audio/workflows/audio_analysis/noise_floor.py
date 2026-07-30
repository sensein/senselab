"""Per-band noise-floor estimation (T056-T059, FR-021a to FR-021i).

Detection works by subtracting a locally estimated per-band floor and applying one margin
threshold to what remains — not by amplification. That is what makes the different-distances
problem tractable: after subtraction a near source and a far source are each judged against
their own band floor, so a single threshold holds at every distance. The approach is taken
from established detection practice (ecoacoustics, marine and terrestrial bioacoustics),
where the design goal is stated directly: after per-bin floor subtraction, power fluctuates
around 0 dB during silence and is considerably higher during an event, so one absolute
threshold suffices.

Four properties, each ruling out a simpler estimator:

**Percentile, not mean or minimum.** A low percentile tolerates high event occupancy — a
tenth percentile survives up to 90% of a band's frames being event — where a mean absorbs
events by construction and a raw minimum carries a large downward bias.

**Bias-corrected.** A ``q``-quantile of exponentially distributed noise power sits a
calculable factor below the mean: about 9.8 dB for a tenth percentile. Uncorrected, every
relative-dB gate is that much more permissive, and the failure looks like generosity rather
than a bug. The correction is validated against synthetic noise in the tests.

**Patch-aggregated, never per-bin.** A single time-frequency bin's log-power has a spread of
about 5.6 dB, so 3 sigma is ~17 dB and a few-dB threshold on one bin is meaningless. Over a
~1 s patch the spread falls below a few tenths of a dB, making the same threshold many sigma.

**Conditioned on target activity.** Every published estimator assumes the floor is
independent of the events. A suppression residual violates that — artifact level correlates
with the removed talker's level — so one unconditioned floor over-gates quiet stretches and
under-gates busy ones. Estimating per activity stratum is the mitigation, and it has **no
published precedent**: validate before relying on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.calibration import quantile_bias_correction_db

__all__ = [
    "NoiseFloorEstimate",
    "band_excess_db",
    "binding_floor",
    "estimate_band_floor_db",
    "estimate_noise_floor",
    "third_octave_bands",
]

_THIRD_OCTAVE_RATIO = 2.0 ** (1.0 / 3.0)
"""Band edge ratio. Third-octave by definition, not a fitted value."""

_BASE_CENTER_HZ = 1000.0
_LOWEST_CENTER_HZ = 25.0

TargetActivity = Literal["active", "quiet", "all"]


def third_octave_bands(sampling_rate: int, *, lowest_center_hz: float = _LOWEST_CENTER_HZ) -> list[tuple[float, float]]:
    """Third-octave band edges from ``lowest_center_hz`` up to Nyquist.

    Per-band rather than broadband because environmental and microphone noise are heavily
    low-frequency weighted: a broadband floor is set by the low bands and leaves mid- and
    high-band events ungated. Room-acoustics and ecoacoustics practice agree on this
    unanimously.
    """
    nyquist = sampling_rate / 2.0
    half = _THIRD_OCTAVE_RATIO**0.5
    bands: list[tuple[float, float]] = []
    n = int(round(3.0 * math.log2(lowest_center_hz / _BASE_CENTER_HZ)))
    while True:
        center = _BASE_CENTER_HZ * (2.0 ** (n / 3.0))
        lo, hi = center / half, center * half
        if lo >= nyquist:
            break
        if hi > nyquist:
            break
        bands.append((lo, hi))
        n += 1
    return bands


@dataclass(frozen=True)
class NoiseFloorEstimate:
    """One band's floor for one activity stratum, with the provenance to reproduce it."""

    band_hz: tuple[float, float]
    floor_db: float | None
    quantile: float
    bias_correction_db: float
    window_s: float
    iterations: int
    target_activity: TargetActivity
    frames: int
    recorder_floor_db: float | None = None
    binding: str = "perceptual"

    def to_row(self) -> dict[str, Any]:
        """Row for ``noise_floor.parquet`` per ``contracts/background-sources.md``."""
        return {
            "band_low_hz": self.band_hz[0],
            "band_high_hz": self.band_hz[1],
            "target_activity": self.target_activity,
            "floor_db": self.floor_db,
            "quantile": self.quantile,
            "bias_correction_db": self.bias_correction_db,
            "window_s": self.window_s,
            "iterations": self.iterations,
            "frames": self.frames,
            "recorder_floor_db": self.recorder_floor_db,
            "binding": self.binding,
        }


def estimate_band_floor_db(
    frame_power: np.ndarray,
    *,
    quantile: float = 0.10,
    max_iterations: int = 3,
    event_exclusion_db: float = 6.0,
    apply_bias_correction: bool = True,
) -> tuple[float | None, int]:
    """Estimate one band's noise floor in dB from its per-frame power.

    Iterates: take the ``quantile`` of surviving frames, exclude frames exceeding it by
    ``event_exclusion_db``, re-estimate. The exclusion threshold follows published
    noise-tracking practice, where the same 6 dB-ish figure appears independently in several
    estimators.

    Args:
        frame_power: Per-frame **linear** band power.
        quantile: Floor quantile, in ``(0, 0.5]``.
        max_iterations: Cap on event-exclusion passes.
        event_exclusion_db: Excess above the running floor that marks a frame as event.
        apply_bias_correction: Whether to correct the quantile's downward bias. Off only
            for tests that need to observe the uncorrected value.

    Returns:
        ``(floor_db, iterations)``; ``floor_db`` is ``None`` when there are no frames.
    """
    power = np.asarray(frame_power, dtype=np.float64)
    power = power[np.isfinite(power) & (power > 0.0)]
    if power.size == 0:
        return None, 0

    # The correction is always applied *inside* the loop, because the exclusion reference
    # must be an estimate of the noise mean; using the raw quantile there re-introduces the
    # runaway described below. `apply_bias_correction` only affects the returned value, so a
    # caller comparing the two observes the correction alone rather than the correction plus
    # a different iteration trajectory.
    correction_db = quantile_bias_correction_db(quantile)
    correction_lin = 10.0 ** (correction_db / 10.0)

    surviving = power
    floor_lin = float(np.quantile(surviving, quantile)) * correction_lin
    iterations = 1
    for _ in range(max(0, max_iterations - 1)):
        # Compare against the *corrected* floor -- an estimate of the noise mean -- not the
        # raw quantile. Excluding relative to the raw quantile removes most of the noise
        # distribution itself (a tenth-percentile-plus-6 dB cut discards roughly two thirds
        # of exponentially distributed noise), and re-taking a low quantile of the truncated
        # remainder drives the estimate down every pass. That runaway reads as a very quiet
        # floor, which makes every margin permissive.
        limit = floor_lin * (10.0 ** (event_exclusion_db / 10.0))
        kept = surviving[surviving <= limit]
        if kept.size < max(8, int(0.05 * power.size)):
            # Too few frames left to estimate from; stop rather than chase the floor into a
            # handful of samples.
            break
        new_floor = float(np.quantile(kept, quantile)) * correction_lin
        iterations += 1
        converged = abs(10.0 * math.log10(new_floor / floor_lin)) < 0.1
        floor_lin, surviving = new_floor, kept
        if converged:
            break

    floor_db = 10.0 * math.log10(floor_lin)
    return (floor_db if apply_bias_correction else floor_db - correction_db), iterations


def _band_frame_power(
    waveform: np.ndarray,
    sampling_rate: int,
    bands: Sequence[tuple[float, float]],
    *,
    frame_s: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(band_power[n_bands, n_frames], frame_start_sample)``."""
    n_fft = 1 << int(math.ceil(math.log2(max(32, int(frame_s * sampling_rate)))))
    hop = max(1, int(hop_s * sampling_rate))
    window = np.hanning(n_fft)
    starts = np.arange(0, max(1, len(waveform) - n_fft + 1), hop)
    if starts.size == 0:
        starts = np.array([0])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)

    spec = np.empty((len(starts), n_fft // 2 + 1))
    for i, s in enumerate(starts):
        seg = waveform[s : s + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        spec[i] = np.abs(np.fft.rfft(seg * window)) ** 2

    out = np.zeros((len(bands), len(starts)))
    for b, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        out[b] = spec[:, mask].mean(axis=1) if mask.any() else 0.0
    return out, starts


def estimate_noise_floor(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    target_active: np.ndarray | None = None,
    profile: Mapping[str, Any] | None = None,
    recorder_floor_db: float | None = None,
) -> list[NoiseFloorEstimate]:
    """Estimate the per-band noise floor, optionally split by target activity.

    Args:
        waveform: Mono samples.
        sampling_rate: Sample rate in Hz.
        target_active: Per-sample boolean target-activity mask. When given, floors are
            estimated separately for active and quiet strata (FR-021h) — the mitigation for
            a residual floor that correlates with the removed talker's level, which has no
            published precedent and should be validated rather than trusted.
        profile: Detection-margin profile; the bundled default is used when omitted.
        recorder_floor_db: Estimated capture-chain self-noise, when known.

    Returns:
        One :class:`NoiseFloorEstimate` per band per stratum.
    """
    from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile

    cfg = dict((profile or load_detection_margin_profile()).get("noise_floor") or {})
    quantile = float(cfg.get("quantile", 0.10))
    max_iterations = int(cfg.get("max_iterations", 3))
    exclusion = float(cfg.get("event_exclusion_db", 6.0))
    window_s = float(cfg.get("window_s", 20.0))
    margin = float(cfg.get("recorder_margin_db", 3.0))

    wav = np.asarray(waveform, dtype=np.float64).squeeze()
    bands = third_octave_bands(sampling_rate)
    power, starts = _band_frame_power(wav, sampling_rate, bands, frame_s=0.025, hop_s=0.010)

    if target_active is None:
        strata: list[tuple[TargetActivity, np.ndarray]] = [("all", np.ones(power.shape[1], dtype=bool))]
    else:
        active = np.asarray(target_active, dtype=bool)
        frame_active = np.array([bool(active[s : s + 1].any()) if s < active.size else False for s in starts])
        strata = [("quiet", ~frame_active), ("active", frame_active)]

    rows: list[NoiseFloorEstimate] = []
    correction = quantile_bias_correction_db(quantile)
    for stratum, mask in strata:
        if target_active is not None and not mask.any():
            continue
        for b, band in enumerate(bands):
            floor_db, iters = estimate_band_floor_db(
                power[b, mask],
                quantile=quantile,
                max_iterations=max_iterations,
                event_exclusion_db=exclusion,
            )
            rows.append(
                NoiseFloorEstimate(
                    band_hz=band,
                    floor_db=floor_db,
                    quantile=quantile,
                    bias_correction_db=correction,
                    window_s=window_s,
                    iterations=iters,
                    target_activity=stratum,
                    frames=int(mask.sum()),
                    recorder_floor_db=recorder_floor_db,
                    binding=binding_floor(
                        band_floor_db=floor_db, recorder_floor_db=recorder_floor_db, margin_db=margin
                    ),
                )
            )
    return rows


def band_excess_db(band_power: float, *, floor_db: float) -> float:
    """Excess of ``band_power`` (linear) over ``floor_db``, in dB.

    Negative values are returned rather than clamped: a sub-floor measurement is
    information, and clamping would hide a floor estimate that ran high.
    """
    if band_power <= 0.0:
        return -math.inf
    return 10.0 * math.log10(band_power) - float(floor_db)


def binding_floor(
    *,
    band_floor_db: float | None,
    recorder_floor_db: float | None,
    margin_db: float,
) -> str:
    """Which limit binds for this band: ``"recorder"`` or ``"perceptual"`` (FR-022a).

    For consumer-grade capture the microphone's own self-noise frequently sits above the
    level of a quiet room, so content near human threshold was never recorded at all. Where
    that is the case the honest statement is "your microphone could not hear it" — both more
    defensible and more useful than a perceptual claim the recording cannot support.
    """
    if band_floor_db is None or recorder_floor_db is None:
        return "perceptual"
    return "recorder" if (band_floor_db - float(recorder_floor_db)) < float(margin_db) else "perceptual"
