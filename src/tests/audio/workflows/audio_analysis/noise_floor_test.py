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


def _measured_floor(frames: np.ndarray, **kwargs: float | bool | int) -> float:
    """Unwrap a measured band floor, failing loudly if the estimator declined to measure.

    The estimator returns ``None`` when there is no energy to measure, which is a different
    outcome from a low floor. Comparing that ``None`` directly would raise a TypeError deep
    in an assertion rather than saying what actually happened.
    """
    value, _iterations = estimate_band_floor_db(frames, **kwargs)
    assert value is not None, "expected a measurable floor; the estimator found no energy"
    return value


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
    floor_db = _measured_floor(_noise_frames(4000, -60.0), quantile=0.10)
    assert floor_db == pytest.approx(-60.0, abs=1.0)


def test_sustained_source_is_not_absorbed_into_the_floor() -> None:
    """A source occupying half the band's frames must not become the floor (FR-021f).

    This is the property that rules out mean- and minimum-based estimators.
    """
    frames = _noise_frames(4000, -60.0)
    frames[:2000] += 10.0 ** (-30.0 / 10.0)  # 30 dB louder source, 50% occupancy
    floor_db = _measured_floor(frames, quantile=0.10)
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
    floor_db = _measured_floor(frames, quantile=0.10)
    assert floor_db > -45.0


def test_bias_correction_is_applied_to_the_estimate() -> None:
    """Without it the estimate sits ~9.8 dB below the true level (FR-021d)."""
    frames = _noise_frames(4000, -60.0)
    corrected = _measured_floor(frames, quantile=0.10)
    uncorrected = _measured_floor(frames, quantile=0.10, apply_bias_correction=False)
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

    q_floor = _measured_floor(frames[~active], quantile=0.10)
    b_floor = _measured_floor(frames[active], quantile=0.10)
    pooled = _measured_floor(frames, quantile=0.10)
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


# ── stationary sources present throughout (T068, FR-021i) ──────────────
#
# The case the per-band floor actively breaks on. A source in ~100% of frames *is* the
# tenth percentile of its own band, so it is absorbed into the floor and its excess over
# that floor reads ~0 dB. Air conditioning, ventilation, mains hum, a music bed running
# under the whole recording -- and those are among the sources a background characterizer
# most wants to name, so failing silently on them is not acceptable.


def test_a_source_running_throughout_reads_below_its_own_floor() -> None:
    """States the failure plainly, and it is worse than "invisible".

    At 100% occupancy the band is dominated by a near-constant level rather than by
    exponentially distributed noise. The quantile lands on the source, and then the bias
    correction — which is calibrated for noise, where a low quantile genuinely sits below
    the mean — adds its full offset on top. So the floor ends up *above* the source: excess
    goes negative and a real steady source reads as sub-floor, not merely as zero-excess.

    This is why the stationary pass compares against neighbouring bands instead.
    """
    frames = _noise_frames(2000, -60.0) + 10.0 ** (-30.0 / 10.0)  # 100% occupancy
    floor_db = _measured_floor(frames, quantile=0.10)
    mean_db = 10.0 * math.log10(float(frames.mean()))
    assert floor_db > mean_db, "expected the floor to overshoot a near-constant source"
    assert floor_db - mean_db == pytest.approx(quantile_bias_correction_db(0.10), abs=0.5)
    assert band_excess_db(float(frames.mean()), floor_db=floor_db) < 0.0


def test_prominence_finds_a_narrowband_source_the_same_band_floor_cannot() -> None:
    """Comparing against neighbours instead of against the band's own history.

    A steady tone is prominent relative to adjacent third-octave bands however many frames
    it occupies, which is exactly the property a same-band comparison lacks.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import prominence_ratio_db

    levels = [-60.0, -60.0, -45.0, -60.0, -60.0]  # 15 dB bump in band 2
    assert prominence_ratio_db(levels, 2) == pytest.approx(15.0, abs=0.1)


def test_a_flat_floor_has_no_prominence() -> None:
    """A genuinely flat noise floor contains no stationary source to report."""
    from senselab.audio.workflows.audio_analysis.noise_floor import prominence_ratio_db

    assert prominence_ratio_db([-60.0] * 5, 2) == pytest.approx(0.0, abs=0.1)


def test_prominence_needs_two_neighbours() -> None:
    """Edge bands have no two-sided reference, so no prominence is claimed for them."""
    from senselab.audio.workflows.audio_analysis.noise_floor import prominence_ratio_db

    assert prominence_ratio_db([-60.0, -45.0, -60.0], 0) is None
    assert prominence_ratio_db([-60.0, -45.0, -60.0], 2) is None


def test_neighbour_reference_is_computed_in_the_power_domain() -> None:
    """Averaging neighbours in dB would overstate prominence beside one loud neighbour."""
    from senselab.audio.workflows.audio_analysis.noise_floor import prominence_ratio_db

    # neighbours 20 dB apart: the power mean sits near the louder one, the dB mean midway
    value = prominence_ratio_db([-40.0, -30.0, -60.0], 1)
    assert value is not None
    assert value < -30.0 - (-50.0), "a dB-domain average would have reported more prominence"


def test_stationary_detection_reports_a_hum_like_band() -> None:
    """End-to-end: a prominent band is reported with its prominence and threshold."""
    from senselab.audio.workflows.audio_analysis.noise_floor import detect_stationary_sources

    rows = [
        NoiseFloorEstimate(
            band_hz=(lo, lo * 1.26),
            floor_db=level,
            quantile=0.10,
            bias_correction_db=9.77,
            window_s=20.0,
            iterations=1,
            target_activity="all",
            frames=100,
        )
        for lo, level in ((100.0, -60.0), (126.0, -60.0), (159.0, -40.0), (200.0, -60.0), (252.0, -60.0))
    ]
    found = detect_stationary_sources(rows)
    assert len(found) == 1
    assert found[0]["band_low_hz"] == pytest.approx(159.0)
    assert found[0]["prominence_db"] > 9.0
    assert found[0]["stationary_pass"] is True


def test_stationary_detection_reports_nothing_on_a_flat_floor() -> None:
    """No fabrication: a flat floor yields no stationary source."""
    from senselab.audio.workflows.audio_analysis.noise_floor import detect_stationary_sources

    rows = [
        NoiseFloorEstimate(
            band_hz=(lo, lo * 1.26),
            floor_db=-60.0,
            quantile=0.10,
            bias_correction_db=9.77,
            window_s=20.0,
            iterations=1,
            target_activity="all",
            frames=100,
        )
        for lo in (100.0, 126.0, 159.0, 200.0, 252.0)
    ]
    assert detect_stationary_sources(rows) == []


def test_stationary_threshold_matches_the_published_tone_criterion() -> None:
    """9 dB is ECMA-74 / ISO 7779's prominent-discrete-tone figure, not a fitted value."""
    from senselab.audio.workflows.audio_analysis.calibration import DEFAULT_DETECTION_MARGIN

    assert DEFAULT_DETECTION_MARGIN["guards"]["prominence_ratio_db"] == pytest.approx(9.0)


def test_recorder_floor_proxy_is_the_quietest_band() -> None:
    """A lower bound on the equipment floor, not a measurement of it.

    If the room contributes across the whole spectrum -- which is what ventilation does --
    the quietest band is still room plus microphone, so this over-estimates how quiet the
    equipment is. It exists so ``binding_floor`` has something to work with absent an
    operator-supplied value.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import estimate_recorder_floor_db

    rows = [
        NoiseFloorEstimate(
            band_hz=(lo, lo * 1.26),
            floor_db=level,
            quantile=0.10,
            bias_correction_db=9.77,
            window_s=20.0,
            iterations=1,
            target_activity="all",
            frames=100,
        )
        for lo, level in ((100.0, -20.0), (200.0, -35.0), (400.0, -53.0), (800.0, -48.0))
    ]
    assert estimate_recorder_floor_db(rows) == pytest.approx(-53.0)


def test_recorder_floor_proxy_is_none_without_estimates() -> None:
    """No bands, no proxy — not a fabricated default."""
    from senselab.audio.workflows.audio_analysis.noise_floor import estimate_recorder_floor_db

    assert estimate_recorder_floor_db([]) is None


def test_broadband_stationary_source_is_not_detected_by_prominence() -> None:
    """The documented limitation, asserted so nobody assumes coverage it does not have.

    A broadband stationary source — ventilation hiss, room rumble, a dense music bed —
    raises every band together, so prominence sees nothing: all the neighbours are raised
    too. Separating it from the microphone's own floor needs a reference the recording does
    not contain.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import detect_stationary_sources

    # every band lifted 20 dB: a broadband source, invisible to a neighbour comparison
    rows = [
        NoiseFloorEstimate(
            band_hz=(lo, lo * 1.26),
            floor_db=-40.0,
            quantile=0.10,
            bias_correction_db=9.77,
            window_s=20.0,
            iterations=1,
            target_activity="all",
            frames=100,
        )
        for lo in (100.0, 126.0, 159.0, 200.0, 252.0)
    ]
    assert detect_stationary_sources(rows) == [], "prominence cannot see a uniform lift"


# ── foreground-referenced SNR ──────────────────────────────────────────
#
# The operative quantity for near-field capture, and available regardless of what the
# background contains: structure and content do not obstruct measuring a level. Only the
# *attribution* of a smooth broadband floor to equipment versus room needs an external
# reference — which is a narrower claim than "structured background is unmeasurable".


def _fg_bg(sr: int = SR, fg_db: float = -20.0, bg_db: float = -55.0, seconds: float = 4.0) -> tuple:
    """Half background-only, half background plus a louder structured foreground."""
    rng = np.random.default_rng(11)
    n = int(sr * seconds)
    bg = rng.standard_normal(n) * (10.0 ** (bg_db / 20.0))
    t = np.arange(n) / sr
    fg = np.sin(2 * math.pi * 440 * t) * (10.0 ** (fg_db / 20.0))
    active = np.zeros(n, dtype=bool)
    active[n // 2 :] = True
    return bg + fg * active, active


def test_foreground_ratio_recovers_a_known_separation() -> None:
    """A 35 dB foreground-to-background separation reads back as a large positive ratio."""
    from senselab.audio.workflows.audio_analysis.noise_floor import foreground_background_ratio_db

    wav, active = _fg_bg(fg_db=-20.0, bg_db=-55.0)
    rows = foreground_background_ratio_db(wav, SR, target_active=active)
    assert rows
    in_band = [r for r in rows if r["band_low_hz"] <= 440 <= r["band_high_hz"]]
    assert in_band and in_band[0]["ratio_db"] > 20.0


def test_foreground_reference_is_a_high_percentile_not_the_floor() -> None:
    """Using the active stratum's floor measures the quiet part *within* active frames.

    On a real recording that understated the ratio to a median of +5.6 dB where the true
    broadband separation was 15-35 dB. The reference has to be a high percentile.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import foreground_background_ratio_db

    wav, active = _fg_bg()
    high = foreground_background_ratio_db(wav, SR, target_active=active, foreground_percentile=90.0)
    low = foreground_background_ratio_db(wav, SR, target_active=active, foreground_percentile=10.0)
    band = lambda rows: next(r for r in rows if r["band_low_hz"] <= 440 <= r["band_high_hz"])  # noqa: E731
    assert band(high)["ratio_db"] > band(low)["ratio_db"]


def test_ratio_is_measurable_when_the_background_has_structure() -> None:
    """The point of the correction: a structured background does not prevent an SNR.

    Background here is a tonal complex rather than white noise — content and structure —
    and the foreground ratio is still recovered.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import foreground_background_ratio_db

    rng = np.random.default_rng(12)
    n = SR * 4
    t = np.arange(n) / SR
    structured_bg = sum(np.sin(2 * math.pi * f * t) for f in (120.0, 240.0, 360.0)) * (10.0 ** (-55.0 / 20.0))
    structured_bg = structured_bg + rng.standard_normal(n) * 1e-5
    fg = np.sin(2 * math.pi * 1000 * t) * (10.0 ** (-20.0 / 20.0))
    active = np.zeros(n, dtype=bool)
    active[n // 2 :] = True
    rows = foreground_background_ratio_db(structured_bg + fg * active, SR, target_active=active)
    in_band = [r for r in rows if r["band_low_hz"] <= 1000 <= r["band_high_hz"]]
    assert in_band and in_band[0]["ratio_db"] > 20.0


def test_blank_recording_yields_no_ratio() -> None:
    """No foreground means no reference; inventing one would be worse than abstaining."""
    from senselab.audio.workflows.audio_analysis.noise_floor import foreground_background_ratio_db

    rng = np.random.default_rng(13)
    wav = rng.standard_normal(SR * 2) * 1e-3
    assert foreground_background_ratio_db(wav, SR, target_active=np.zeros(SR * 2, dtype=bool)) == []


# ── cross-recording baseline ───────────────────────────────────────────


def _floors(levels: dict[float, float]) -> list[NoiseFloorEstimate]:
    return [
        NoiseFloorEstimate(
            band_hz=(lo, lo * 1.26),
            floor_db=db,
            quantile=0.10,
            bias_correction_db=9.77,
            window_s=20.0,
            iterations=1,
            target_activity="all",
            frames=100,
        )
        for lo, db in levels.items()
    ]


def test_cohort_separates_common_equipment_floor_from_room_excess() -> None:
    """The reference a single recording lacks, supplied by more than one recording.

    Across recordings from one rig the equipment contribution is common while the room
    contribution varies — which is what makes a broadband stationary source nameable.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import cross_recording_baseline

    quiet_room = _floors({100.0: -60.0, 200.0: -58.0})
    noisy_room = _floors({100.0: -48.0, 200.0: -58.0})
    out = cross_recording_baseline({"a": quiet_room, "b": noisy_room})
    band100 = next(b for b in out["bands"] if b["band_low_hz"] == 100.0)
    assert band100["common_floor_db"] == pytest.approx(-60.0)
    assert band100["spread_db"] == pytest.approx(12.0)
    excess = {e["band_low_hz"]: e["room_excess_db"] for e in out["recordings"]["b"]}
    assert excess[100.0] == pytest.approx(12.0)
    assert excess[200.0] == pytest.approx(0.0)


def test_a_single_recording_cannot_separate_common_from_particular() -> None:
    """Reported as a stated requirement rather than a silently degenerate answer."""
    from senselab.audio.workflows.audio_analysis.noise_floor import cross_recording_baseline

    out = cross_recording_baseline({"only": _floors({100.0: -60.0})})
    assert out["bands"] == []
    assert "two recordings" in out["note"]


# ── band resolvability and absent-floor reasons ────────────────────────


def test_bands_narrower_than_the_bin_spacing_are_dropped() -> None:
    """Third-octave bands narrow with frequency; FFT spacing does not.

    At 16 kHz with a 25 ms frame the spacing is 31.25 Hz while the 22-28 Hz band spans
    5.7 Hz, so it contains no bins at all. Emitting it produces a NaN that looks like a
    measurement which happened to be missing rather than one that could never exist.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import resolvable_bands

    bands = third_octave_bands(SR)
    assert len(resolvable_bands(bands, SR, frame_s=0.025)) < len(bands)


def test_a_longer_frame_resolves_more_low_bands() -> None:
    """The floor is a long-term percentile, so it can buy frequency resolution with time.

    This matters concretely: a 25 ms frame cannot resolve below ~140 Hz, which is where
    mains hum and ventilation fundamentals live — the stationary sources most worth finding.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import resolvable_bands

    bands = third_octave_bands(SR)
    short = resolvable_bands(bands, SR, frame_s=0.025)
    long = resolvable_bands(bands, SR, frame_s=0.100)
    assert len(long) > len(short)
    assert long[0][0] < short[0][0], "the longer frame reaches lower"


def test_resolvability_counts_actual_bins_not_band_width() -> None:
    """Width is only a proxy: a wide-enough band can fall between two bin centres.

    Three NaN floors survived a width-based test for exactly that reason.
    """
    from senselab.audio.workflows.audio_analysis.noise_floor import resolvable_bands

    for lo, hi in resolvable_bands(third_octave_bands(SR), SR, frame_s=0.100):
        freqs = np.fft.rfftfreq(2048, d=1.0 / SR)
        assert int(((freqs >= lo) & (freqs < hi)).sum()) >= 1


def test_a_band_with_no_energy_is_reported_as_such_not_as_unmeasurable() -> None:
    """An unmeasurable band and an empty band are different facts.

    A high-passed recording genuinely has no energy at the bottom of the range. Collapsing
    that into the same NaN as an unresolvable band makes both unreadable.
    """
    rng = np.random.default_rng(20)
    # high-passed: no low-frequency content at all
    wav = np.asarray(np.convolve(rng.standard_normal(SR * 2), [1.0, -0.98], mode="same"), dtype=np.float64)
    rows = estimate_noise_floor(wav, SR)
    assert all(r.status in ("ok", "no_energy") for r in rows)
    assert all((r.floor_db is None) == (r.status == "no_energy") for r in rows)
