"""Per-band noise-floor estimation (T039-T045, FR-021a to FR-021i).

Detection works by per-band floor subtraction rather than amplification, so this estimator
is where the margin ladder's meaning comes from. Get the floor wrong by 10 dB and every
tier is wrong by 10 dB, silently and in the permissive direction.

The first group of tests validates the **derivation itself** on synthetic noise before
anything is built on it. Those statistics were flagged as unpublished synthesis
(research.md risk 2): straightforward chi-squared results, but not found stated in that
form in the noise-estimation literature. Validating them here converts a provisional
assumption into a regression guard.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.calibration import quantile_bias_correction_db
from senselab.audio.workflows.audio_analysis.noise_floor import (
    NoiseFloorEstimate,
    band_excess_db,
    binding_floor,
    estimate_band_floor_db,
    estimate_noise_floor,
    third_octave_bands,
)

SR = 16000


# ── the derivation, validated on synthetic noise (T039, research risk 2) ──


@pytest.mark.parametrize("q", [0.05, 0.10, 0.20, 0.50])
def test_bias_correction_recovers_the_mean_of_exponential_power(q: float) -> None:
    """A q-quantile of noise power sits ``-ln(1-q)`` below the mean — measured, not assumed.

    This is the load-bearing statistic: an uncorrected tenth-percentile floor is ~9.8 dB
    low, which makes every relative-dB gate that much more permissive. The failure would
    look like generosity rather than a bug.
    """
    rng = np.random.default_rng(0)
    power = rng.exponential(scale=1.0, size=2_000_000)
    measured_db = 10.0 * math.log10(float(power.mean()) / float(np.quantile(power, q)))
    assert quantile_bias_correction_db(q) == pytest.approx(measured_db, abs=0.1)


def test_per_bin_log_power_sigma_is_about_five_and_a_half_db() -> None:
    """Why a single time-frequency bin cannot carry a few-dB threshold.

    At this spread, 3 sigma is roughly 17 dB — so a 3 dB excess on one bin is noise.
    """
    rng = np.random.default_rng(1)
    s = 10.0 * np.log10(rng.exponential(1.0, 1_000_000))
    assert float(s.std()) == pytest.approx(5.57, abs=0.1)


def test_log_of_mean_exceeds_mean_of_log_by_about_two_and_a_half_db() -> None:
    """The log-domain bias that makes averaging in dB and in power differ."""
    rng = np.random.default_rng(2)
    s = 10.0 * np.log10(rng.exponential(1.0, 1_000_000))
    assert float(s.mean()) == pytest.approx(-2.51, abs=0.1)


def test_patch_aggregation_collapses_the_variance(patch_size: int = 768) -> None:
    """Aggregating over a patch is what makes a 3 dB threshold meaningful (FR-021e).

    A ~0.96 s patch over a handful of bins is ~768 samples, and the spread of its log-mean
    falls to a fraction of a dB — so the same 3 dB that is noise on one bin is many sigma
    on a patch.
    """
    rng = np.random.default_rng(3)
    means = [10.0 * np.log10(rng.exponential(1.0, patch_size).mean()) for _ in range(5000)]
    assert float(np.std(means)) < 0.3


# ── band decomposition ────────────────────────────────────────────────


def test_third_octave_bands_cover_the_audible_range() -> None:
    """Bands span from the low end up to Nyquist without exceeding it."""
    bands = third_octave_bands(SR)
    assert bands[0][0] > 0.0
    assert bands[-1][1] <= SR / 2 + 1e-6


def test_third_octave_bands_are_contiguous_and_ascending() -> None:
    """No gaps or reversals in the band ladder."""
    bands = third_octave_bands(SR)
    for (lo_a, hi_a), (lo_b, _hi_b) in zip(bands, bands[1:]):
        assert hi_a > lo_a
        assert lo_b >= lo_a


def test_band_ratio_is_the_third_octave_constant() -> None:
    """Each band spans a factor of 2**(1/3) — the definition, not a fitted value."""
    bands = third_octave_bands(SR)
    lo, hi = bands[len(bands) // 2]
    assert hi / lo == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)


# ── floor estimation and event exclusion (T041) ────────────────────────


def _noise_frames(n: int, level_db: float = -60.0, seed: int = 0) -> np.ndarray:
    """Exponentially distributed band power at a given mean level."""
    rng = np.random.default_rng(seed)
    return rng.exponential(10.0 ** (level_db / 10.0), size=n)


def test_floor_of_pure_noise_recovers_its_level() -> None:
    """The whole point: on noise-only input the estimate is the noise level."""
    floor_db, _iters = estimate_band_floor_db(_noise_frames(4000, -60.0), quantile=0.10)
    assert floor_db == pytest.approx(-60.0, abs=1.0)


def test_sustained_source_is_not_absorbed_into_the_floor() -> None:
    """A source occupying half the band's frames must not become the floor (FR-021f).

    This is the property that rules out mean- and minimum-based estimators.
    """
    frames = _noise_frames(4000, -60.0)
    frames[:2000] += 10.0 ** (-30.0 / 10.0)  # 30 dB louder source, 50% occupancy
    floor_db, _ = estimate_band_floor_db(frames, quantile=0.10)
    assert floor_db == pytest.approx(-60.0, abs=2.0)


def test_event_exclusion_iterates_to_stability() -> None:
    """Iteration terminates rather than running to the cap every time."""
    frames = _noise_frames(3000, -55.0)
    frames[:900] += 10.0 ** (-25.0 / 10.0)
    _floor, iterations = estimate_band_floor_db(frames, quantile=0.10, max_iterations=5)
    assert 1 <= iterations <= 5


def test_high_occupancy_source_defeats_a_tenth_percentile_floor() -> None:
    """An honest limit, asserted rather than hidden.

    A tenth-percentile floor tolerates up to 90% event occupancy. Past that the source
    *is* the tenth percentile and the floor rises to meet it. Recording the limit here so
    a surprising result upstream is diagnosable.
    """
    frames = _noise_frames(2000, -60.0) + 10.0 ** (-30.0 / 10.0)  # 100% occupancy
    floor_db, _ = estimate_band_floor_db(frames, quantile=0.10)
    assert floor_db > -45.0


def test_bias_correction_is_applied_to_the_estimate() -> None:
    """Without it the estimate sits ~9.8 dB below the true level (FR-021d)."""
    frames = _noise_frames(4000, -60.0)
    corrected, _ = estimate_band_floor_db(frames, quantile=0.10)
    uncorrected, _ = estimate_band_floor_db(frames, quantile=0.10, apply_bias_correction=False)
    assert corrected - uncorrected == pytest.approx(quantile_bias_correction_db(0.10), abs=0.2)


def test_empty_frames_yield_no_estimate() -> None:
    """No frames is no estimate, not a zero."""
    floor_db, _ = estimate_band_floor_db(np.array([]), quantile=0.10)
    assert floor_db is None


# ── excess over floor ─────────────────────────────────────────────────


def test_excess_is_zero_at_the_floor() -> None:
    """A band exactly at its floor has no excess."""
    assert band_excess_db(10.0 ** (-6.0), floor_db=-60.0) == pytest.approx(0.0, abs=1e-6)


def test_excess_tracks_level_above_the_floor() -> None:
    """Excess is the dB difference, directly."""
    assert band_excess_db(10.0 ** (-5.0), floor_db=-60.0) == pytest.approx(10.0, abs=1e-6)


def test_excess_below_the_floor_is_negative() -> None:
    """Reported rather than clamped: a sub-floor measurement is information."""
    assert band_excess_db(10.0 ** (-7.0), floor_db=-60.0) < 0.0


# ── activity conditioning (T043, FR-021h — unpublished synthesis) ──────


def test_conditioned_floors_differ_when_the_residual_tracks_activity() -> None:
    """The mitigation for a problem no published estimator addresses.

    Every published floor estimator assumes the floor is independent of the events. A
    suppression residual violates that: artifact level correlates with the removed
    talker's level. One unconditioned floor then over-gates quiet stretches and
    under-gates busy ones, so the floor is estimated per activity stratum instead.
    """
    rng = np.random.default_rng(4)
    n = 2000
    # busy half carries a 12 dB louder artifact floor than the quiet half
    quiet = rng.exponential(10.0 ** (-6.0), n)
    busy = rng.exponential(10.0 ** (-4.8), n)
    frames = np.concatenate([quiet, busy])
    active = np.concatenate([np.zeros(n, dtype=bool), np.ones(n, dtype=bool)])

    q_floor, _ = estimate_band_floor_db(frames[~active], quantile=0.10)
    b_floor, _ = estimate_band_floor_db(frames[active], quantile=0.10)
    pooled, _ = estimate_band_floor_db(frames, quantile=0.10)
    assert b_floor - q_floor == pytest.approx(12.0, abs=2.0)
    assert q_floor < pooled < b_floor, "a pooled floor sits between, mis-gating both strata"


def test_estimate_noise_floor_emits_both_strata_when_activity_varies() -> None:
    """Active and quiet strata are estimated separately (FR-021h)."""
    rng = np.random.default_rng(5)
    wav = (rng.standard_normal(SR * 4) * 1e-3).astype(np.float64)
    active = np.zeros(SR * 4, dtype=bool)
    active[SR * 2 :] = True
    rows = estimate_noise_floor(wav, SR, target_active=active)
    strata = {r.target_activity for r in rows}
    assert strata == {"active", "quiet"}


def test_estimate_noise_floor_without_activity_emits_one_stratum() -> None:
    """Without an activity mask there is nothing to stratify by."""
    rng = np.random.default_rng(6)
    wav = (rng.standard_normal(SR * 2) * 1e-3).astype(np.float64)
    rows = estimate_noise_floor(wav, SR)
    assert {r.target_activity for r in rows} == {"all"}


def test_estimate_returns_one_row_per_band_per_stratum() -> None:
    """Every band gets an estimate, even quiet ones."""
    rng = np.random.default_rng(7)
    wav = (rng.standard_normal(SR * 2) * 1e-3).astype(np.float64)
    rows = estimate_noise_floor(wav, SR)
    assert len(rows) == len(third_octave_bands(SR))
    assert all(isinstance(r, NoiseFloorEstimate) for r in rows)


def test_estimate_records_its_own_parameters() -> None:
    """A floor without its provenance cannot be reproduced or argued with."""
    rng = np.random.default_rng(8)
    rows = estimate_noise_floor((rng.standard_normal(SR) * 1e-3).astype(np.float64), SR)
    row = rows[len(rows) // 2]
    assert row.quantile > 0.0
    assert row.bias_correction_db > 0.0
    assert row.band_hz[1] > row.band_hz[0]


# ── recorder floor and the binding limit (T045, FR-021b / FR-022a) ─────


def test_recorder_floor_binds_when_the_band_floor_is_close_to_it() -> None:
    """For consumer capture the microphone, not human hearing, is often the limit.

    Saying so is both more defensible and more useful than a perceptual claim the
    recording cannot support.
    """
    assert binding_floor(band_floor_db=-70.0, recorder_floor_db=-71.0, margin_db=3.0) == "recorder"


def test_perceptual_limit_binds_when_the_band_floor_is_well_above_the_recorder() -> None:
    """A band well clear of the microphone floor is limited by perception."""
    assert binding_floor(band_floor_db=-50.0, recorder_floor_db=-90.0, margin_db=3.0) == "perceptual"


def test_absent_recorder_estimate_leaves_the_perceptual_limit() -> None:
    """Without a recorder estimate no recorder claim can be made."""
    assert binding_floor(band_floor_db=-50.0, recorder_floor_db=None, margin_db=3.0) == "perceptual"


def test_margin_widens_the_recorder_verdict() -> None:
    """The margin is what decides how close counts as close."""
    assert binding_floor(band_floor_db=-70.0, recorder_floor_db=-78.0, margin_db=3.0) == "perceptual"
    assert binding_floor(band_floor_db=-70.0, recorder_floor_db=-78.0, margin_db=10.0) == "recorder"
