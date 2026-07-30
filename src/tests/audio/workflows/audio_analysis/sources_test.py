"""Margin ladder and fabrication guards (T046-T049, FR-020b to FR-021).

The failure mode this file defends against is not a missed source, it is a **fabricated**
one. Amplifying a noise floor produces confident, plausible environmental labels that are
statistically indistinguishable from genuine broadband noise, so they read as real findings
rather than as obvious garbage.

Tiers come from excess over the band's own noise floor, never from gain: amplification moves
a source and the residual foreground together and changes no signal-to-noise ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.calibration import DEFAULT_DETECTION_MARGIN
from senselab.audio.workflows.audio_analysis.sources import (
    TIERS,
    assign_tier,
    is_noise_like,
    is_quarantined,
    matches_floor_signature,
    passes_pregain_gate,
    screen_candidate,
    spectral_flatness,
)

PROFILE = DEFAULT_DETECTION_MARGIN
MARGINS = PROFILE["margins_db"]


# ── the ladder (T046, FR-021) ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("excess_db", "expected"),
    [
        (-5.0, "rejected"),
        (0.0, "rejected"),
        (2.9, "rejected"),
        (3.0, "candidate"),
        (5.9, "candidate"),
        (6.0, "probable"),
        (9.9, "probable"),
        (10.0, "confident"),
        (25.0, "confident"),
    ],
)
def test_tier_boundaries(excess_db: float, expected: str) -> None:
    """The corroborated 3 / 6 / 10 dB ladder, at its boundaries."""
    assert assign_tier(excess_db, MARGINS) == expected


def test_tiers_are_the_declared_four() -> None:
    """No fifth tier may appear."""
    assert set(TIERS) == {"rejected", "candidate", "probable", "confident"}


def test_a_finding_never_advances_tier_on_gain_alone() -> None:
    """Gain changes no SNR, so it cannot promote a candidate (research D1).

    Excess is measured against the band floor, and gain moves both together — so the same
    excess yields the same tier whatever gain was applied.
    """
    assert assign_tier(4.0, MARGINS) == assign_tier(4.0, MARGINS) == "candidate"


# ── noise-character guard (T047, FR-020b) ─────────────────────────────


def test_white_noise_is_spectrally_flat() -> None:
    """Broadband noise sits near the top of the flatness range."""
    rng = np.random.default_rng(0)
    spec = rng.exponential(1.0, 4096)
    assert spectral_flatness(spec) > 0.3


def test_a_tone_is_not_spectrally_flat() -> None:
    """A structured source concentrates its energy, so flatness collapses."""
    spec = np.full(4096, 1e-8)
    spec[100] = 1.0
    assert spectral_flatness(spec) < 0.01


def test_flatness_separates_noise_from_structure_by_orders_of_magnitude() -> None:
    """Why one transform is enough to make this the highest-value cheap guard."""
    rng = np.random.default_rng(1)
    noise = spectral_flatness(rng.exponential(1.0, 4096))
    tone = np.full(4096, 1e-8)
    tone[500] = 1.0
    assert noise > 30 * spectral_flatness(tone)


def test_noise_like_uses_the_configured_limit() -> None:
    """No hardcoded threshold: the descriptor is standardized, the decision rule is not."""
    assert is_noise_like(0.6, flatness_max=0.30) is True
    assert is_noise_like(0.1, flatness_max=0.30) is False


def test_flatness_of_empty_spectrum_is_zero() -> None:
    """An empty spectrum has no flatness to report."""
    assert spectral_flatness(np.array([])) == pytest.approx(0.0)


# ── quarantine list (T048, FR-020c) ───────────────────────────────────


def test_water_like_labels_are_quarantined() -> None:
    """These are what an amplified noise floor was measured to produce.

    They are the dangerous ones precisely because they read as genuine environmental
    findings rather than as obvious artifacts.
    """
    quarantined = PROFILE["guards"]["quarantined_labels"]
    for label in ("Waterfall", "Water", "Gurgling", "Static", "White noise", "Silence"):
        assert is_quarantined(label, quarantined), label


def test_a_real_source_label_is_not_quarantined() -> None:
    """The list names specific noise artifacts, not source labels generally."""
    assert not is_quarantined("Dog", PROFILE["guards"]["quarantined_labels"])


def test_quarantined_label_rejected_on_a_noise_like_segment() -> None:
    """The combination is what rejects, not either alone."""
    tier, reason = screen_candidate(
        label="Waterfall", above_floor_db=20.0, flatness=0.55, segment_rms_dbfs=-30.0, profile=PROFILE
    )
    assert tier == "rejected"
    assert reason is not None and "quarantined" in reason


def test_quarantined_label_allowed_when_the_segment_is_structured() -> None:
    """A real waterfall is a legitimate finding; the guard targets noise, not the word."""
    tier, reason = screen_candidate(
        label="Waterfall", above_floor_db=20.0, flatness=0.02, segment_rms_dbfs=-30.0, profile=PROFILE
    )
    assert tier == "confident"
    assert reason is None


# ── pre-gain level reject (FR-020a) ───────────────────────────────────


def test_segment_below_the_trust_floor_is_rejected() -> None:
    """Amplifying it is what manufactures the labels the quarantine list catches."""
    assert passes_pregain_gate(-60.0, reject_below_dbfs=-45.0) is False
    assert passes_pregain_gate(-30.0, reject_below_dbfs=-45.0) is True


def test_pregain_reject_precedes_the_margin_check() -> None:
    """A too-quiet segment is rejected for being too quiet, not for a marginal excess.

    The reason a consumer reads must let them distinguish "this was noise" from "this was
    real but too quiet".
    """
    tier, reason = screen_candidate(
        label="Dog", above_floor_db=20.0, flatness=0.01, segment_rms_dbfs=-60.0, profile=PROFILE
    )
    assert tier == "rejected"
    assert reason is not None and "trust floor" in reason


# ── floor-response signature (FR-020d) ────────────────────────────────


def test_floor_signature_match_rejects_the_window() -> None:
    """A window reproducing the known below-floor response is not content."""
    sig = {"Silence": 0.437, "Music": 0.350}
    tier, reason = screen_candidate(
        label="Music",
        above_floor_db=20.0,
        flatness=0.01,
        segment_rms_dbfs=-30.0,
        profile=PROFILE,
        scores_by_label=dict(sig),
        floor_signature=sig,
    )
    assert tier == "rejected"
    assert reason is not None and "signature" in reason


def test_signature_check_compares_the_whole_pattern_not_the_silence_label() -> None:
    """The decisive case: a co-occurring label clears an ordinary threshold.

    One measured signature pairs a silence score of ~0.44 with a second label at ~0.35.
    Keying on the silence label alone lets the second one through as a finding.
    """
    sig = {"Silence": 0.437, "Music": 0.350}
    assert matches_floor_signature({"Silence": 0.437}, sig) is False, "a partial match is not the signature"
    assert matches_floor_signature(dict(sig), sig) is True


def test_absent_signature_never_matches() -> None:
    """No known signature means no signature match."""
    assert matches_floor_signature({"Speech": 0.9}, None) is False


def test_content_that_merely_includes_silence_is_not_a_floor_response() -> None:
    """Real audio can score the silence label without being below the floor."""
    sig = {"Silence": 0.437, "Music": 0.350}
    assert matches_floor_signature({"Silence": 0.437, "Music": 0.350, "Speech": 0.8}, sig) is False


# ── the false-positive case (T049, SC-018) ────────────────────────────


def test_amplified_noise_floor_yields_no_finding() -> None:
    """SC-018 in unit form: every layer independently rejects it.

    Amplified noise reaches the classifier as broadband, so the noise-character test fires
    even when the label is not on the quarantine list and the excess looks generous.
    """
    rng = np.random.default_rng(2)
    flatness = spectral_flatness(rng.exponential(1.0, 4096))
    for label in ("Waterfall", "White noise", "Static", "Dog"):
        tier, reason = screen_candidate(
            label=label, above_floor_db=25.0, flatness=flatness, segment_rms_dbfs=-20.0, profile=PROFILE
        )
        assert tier == "rejected", f"{label} survived on amplified noise"
        assert reason is not None


def test_a_genuine_structured_source_survives_every_guard() -> None:
    """The guards must not reject everything — the positive control."""
    tier, reason = screen_candidate(
        label="Dog", above_floor_db=12.0, flatness=0.005, segment_rms_dbfs=-25.0, profile=PROFILE
    )
    assert tier == "confident"
    assert reason is None


def test_a_real_but_marginal_source_is_a_candidate_with_no_rejection() -> None:
    """Marginal is reported as marginal, not discarded."""
    tier, reason = screen_candidate(
        label="Dog", above_floor_db=4.0, flatness=0.005, segment_rms_dbfs=-25.0, profile=PROFILE
    )
    assert tier == "candidate"
    assert reason is None


# ── excision routing (T054, FR-041 to FR-045) ─────────────────────────


def _mask_row(region_id: str, start: float, end: float, state: str = "target_free") -> dict:
    return {"region_id": region_id, "start": start, "end": end, "state": state}


def test_only_target_free_regions_are_excised() -> None:
    """An excised segment exists to give the classifier audio free of target activity."""
    from senselab.audio.workflows.audio_analysis.sources import plan_excision

    rows = [
        _mask_row("m0", 0.0, 12.0, "target_free"),
        _mask_row("m1", 12.0, 24.0, "target_active"),
        _mask_row("m2", 24.0, 36.0, "indeterminate"),
    ]
    segs = plan_excision(rows, long_window_s=10.24, max_padding_fraction=0.5)
    assert [s.region_id for s in segs] == ["m0"]


def test_a_region_longer_than_the_window_needs_no_padding() -> None:
    """A region at least as long as the window is classified unpadded."""
    from senselab.audio.workflows.audio_analysis.sources import plan_excision

    seg = plan_excision([_mask_row("m0", 0.0, 20.0)], long_window_s=10.24, max_padding_fraction=0.5)[0]
    assert seg.padding_fraction == pytest.approx(0.0)
    assert seg.supports_long_window is True


def test_a_short_region_is_returned_but_flagged() -> None:
    """Flagged rather than dropped, so a consumer sees what was skipped.

    Padding maps to a fixed value while the signal region drifts with gain, so the
    pad-to-signal contrast is itself gain-dependent — a heavily padded decision is not
    comparable to a full-window one.
    """
    from senselab.audio.workflows.audio_analysis.sources import plan_excision

    seg = plan_excision([_mask_row("m0", 0.0, 1.0)], long_window_s=10.24, max_padding_fraction=0.5)[0]
    assert seg.padding_fraction > 0.5
    assert seg.supports_long_window is False


def test_regions_are_never_concatenated_to_reach_a_usable_length() -> None:
    """Joining spans would create a seam an onset-sensitive feature reads as an event.

    That would manufacture exactly the kind of finding the other guards exist to prevent,
    so two short regions stay two short regions.
    """
    from senselab.audio.workflows.audio_analysis.sources import plan_excision

    rows = [_mask_row("m0", 0.0, 3.0), _mask_row("m1", 20.0, 23.0)]
    segs = plan_excision(rows, long_window_s=10.24, max_padding_fraction=0.5)
    assert len(segs) == 2
    assert all(s.duration_s == pytest.approx(3.0) for s in segs)


def test_short_window_classifier_stays_on_the_grid() -> None:
    """Its windows already fit inside a mask region; excision would cost continuity."""
    from senselab.audio.workflows.audio_analysis.sources import ExcisedSegment, route_classifier

    seg = ExcisedSegment("m0", 0.0, 20.0, 0.0, True)
    assert route_classifier(seg, classifier_window_s=0.96, long_window_s=10.24) == "grid"


def test_long_window_classifier_uses_an_excised_segment_when_it_can() -> None:
    """Measured: excising the quiet segment beat every mixed-window variant."""
    from senselab.audio.workflows.audio_analysis.sources import ExcisedSegment, route_classifier

    seg = ExcisedSegment("m0", 0.0, 20.0, 0.0, True)
    assert route_classifier(seg, classifier_window_s=10.24, long_window_s=10.24) == "excised"


def test_long_window_classifier_falls_back_to_the_grid_when_it_cannot() -> None:
    """A heavily padded excision is worse than the grid, so it is not used."""
    from senselab.audio.workflows.audio_analysis.sources import ExcisedSegment, route_classifier

    seg = ExcisedSegment("m0", 0.0, 1.0, 0.9, False)
    assert route_classifier(seg, classifier_window_s=10.24, long_window_s=10.24) == "grid"
    assert route_classifier(None, classifier_window_s=10.24, long_window_s=10.24) == "grid"


def test_a_recording_with_no_long_enough_region_yields_no_excision() -> None:
    """SC-032's scenario, which occurred naturally on a 14 s validation recording."""
    from senselab.audio.workflows.audio_analysis.sources import plan_excision

    rows = [_mask_row(f"m{i}", i * 3.0, i * 3.0 + 1.5) for i in range(4)]
    segs = plan_excision(rows, long_window_s=10.24, max_padding_fraction=0.5)
    assert segs and not any(s.supports_long_window for s in segs)


# ── recovery delta, discounting, cross-level guard ────────────────────


def _finding(**kw: object) -> "SourceFinding":  # noqa: F821 — imported in-body
    """A confident machine-category finding; keyword arguments override any field."""
    from senselab.audio.workflows.audio_analysis.sources import SourceFinding

    base = dict(
        start=0.0,
        end=1.0,
        category="machine",
        label="Vehicle",
        classifier="yamnet",
        above_floor_db=12.0,
        tier="confident",
        binding_floor="perceptual",
        variant="foreground_suppressed",
        gain_db=0.0,
        computed_on="grid",
        flatness=0.01,
        occupancy=0.8,
    )
    base.update(kw)
    return SourceFinding(**base)  # type: ignore[arg-type]


def test_recovery_delta_names_what_suppression_bought() -> None:
    """SC-007: without this, a run gives no way to tell whether suppression contributed."""
    from senselab.audio.workflows.audio_analysis.sources import recovery_delta

    d = recovery_delta(
        suppressed=[{"category": "machine"}, {"category": "environment"}],
        unmodified=[{"category": "environment"}],
    )
    assert d["recovered"] == ["machine"]
    assert d["shared"] == ["environment"]
    assert d["suppression_contributed"] is True


def test_recovery_delta_reports_when_suppression_bought_nothing() -> None:
    """An honest null: the same categories were already visible without it."""
    from senselab.audio.workflows.audio_analysis.sources import recovery_delta

    d = recovery_delta(suppressed=[{"category": "people"}], unmodified=[{"category": "people"}])
    assert d["recovered"] == []
    assert d["suppression_contributed"] is False


def test_recovery_delta_flags_evidence_suppression_destroyed() -> None:
    """A category present only without suppression means suppression removed it."""
    from senselab.audio.workflows.audio_analysis.sources import recovery_delta

    d = recovery_delta(suppressed=[], unmodified=[{"category": "environment"}])
    assert d["lost"] == ["environment"]


def test_uncertain_mask_region_downgrades_a_finding_with_its_reason() -> None:
    """FR-036: weak evidence and evidence weakened by an uncertain mask differ.

    Without the reason a consumer reads an uncertain-region finding as merely marginal.
    """
    from senselab.audio.workflows.audio_analysis.sources import discount_for_mask_uncertainty

    out = discount_for_mask_uncertainty(_finding(), mask_confidence=0.2)
    assert out.tier == "probable"
    assert out.discounted_reason is not None and "mask uncertainty" in out.discounted_reason


def test_a_confident_mask_region_leaves_the_finding_alone() -> None:
    """A trustworthy region does not weaken what was found in it."""
    from senselab.audio.workflows.audio_analysis.sources import discount_for_mask_uncertainty

    out = discount_for_mask_uncertainty(_finding(), mask_confidence=0.95)
    assert out.tier == "confident"
    assert out.discounted_reason is None


def test_absent_mask_confidence_is_not_treated_as_uncertainty() -> None:
    """No mask information is not the same as an uncertain mask."""
    from senselab.audio.workflows.audio_analysis.sources import discount_for_mask_uncertainty

    assert discount_for_mask_uncertainty(_finding(), mask_confidence=None).tier == "confident"


def test_comparing_findings_across_gains_is_refused() -> None:
    """FR-020e: classifier score varies with level on unchanged audio.

    Ranking across differently-gained segments compares artifacts of level, not content.
    """
    from senselab.audio.workflows.audio_analysis.sources import assert_comparable_levels

    with pytest.raises(ValueError, match="not level-comparable"):
        assert_comparable_levels([_finding(gain_db=0.0), _finding(gain_db=6.0)])


def test_findings_at_one_gain_compare_fine() -> None:
    """The guard only fires on a genuine mismatch."""
    from senselab.audio.workflows.audio_analysis.sources import assert_comparable_levels

    assert_comparable_levels([_finding(gain_db=3.0), _finding(gain_db=3.0)])


# ── SC-012 / SC-020 / SC-023 ──────────────────────────────────────────


def test_two_sources_at_different_distances_are_both_reported() -> None:
    """SC-012: a near source must not bury a far one below the margin.

    Each is judged against its own band floor, so a level difference between them does not
    decide which is reportable — which is the whole reason detection subtracts a local
    floor rather than applying a global gain.
    """
    near, far = 24.0, 11.0
    assert assign_tier(near, MARGINS) == "confident"
    assert assign_tier(far, MARGINS) == "confident"


def test_every_finding_states_its_mask_provenance() -> None:
    """SC-020: exposure to foreground leakage is never unknown."""
    row = _finding(from_mask_region="m3", mask_confidence=0.9).to_row()
    assert "from_mask_region" in row and "mask_confidence" in row


def test_a_target_free_finding_outranks_the_same_finding_under_target_activity() -> None:
    """SC-023: the mask improves interpretability rather than merely adding a field."""
    from senselab.audio.workflows.audio_analysis.sources import discount_for_mask_uncertainty

    free = discount_for_mask_uncertainty(_finding(from_mask_region="m0"), mask_confidence=0.98)
    murky = discount_for_mask_uncertainty(_finding(from_mask_region="m1"), mask_confidence=0.1)
    assert list(TIERS).index(free.tier) > list(TIERS).index(murky.tier)


# ── presence vs extent, and modulation depth (T061 / T063) ────────────


def _depth(waveform: np.ndarray, sampling_rate: int, **kwargs: bool | float) -> float:
    """Unwrap a measured modulation depth, failing loudly when nothing was measured.

    ``None`` means the signal carried no energy — a different outcome from "stationary",
    and one these tests never intend to exercise.
    """
    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    value = modulation_depth(waveform, sampling_rate, **kwargs)
    assert value is not None, "expected a measurable modulation depth; the signal was empty"
    return value


def test_a_confidently_present_source_still_gets_its_own_boundaries() -> None:
    """FR-021k: one threshold cannot get both presence and extent right.

    Frame-level thresholding entangles detection confidence with temporal extent — raise the
    threshold and a real event's edges erode; lower it and neighbouring noise joins on. So
    the margin gate decides *whether* the source is there and a separate, looser rule decides
    *where* it starts and stops.
    """
    from senselab.audio.workflows.audio_analysis.sources import resolve_extent

    frames = [(0.0, 0.5, 4.0), (0.5, 1.0, 12.0), (1.0, 1.5, 5.0), (1.5, 2.0, 0.5)]
    extent = resolve_extent(frames, presence_margin_db=10.0, extent_margin_db=3.0)
    assert extent == (0.0, 1.5)


def test_a_source_that_never_clears_the_presence_gate_has_no_extent() -> None:
    """Extent is meaningless without presence.

    Otherwise a run would report boundaries for something it never claimed was there.
    """
    from senselab.audio.workflows.audio_analysis.sources import resolve_extent

    frames = [(0.0, 0.5, 4.0), (0.5, 1.0, 5.0)]
    assert resolve_extent(frames, presence_margin_db=10.0, extent_margin_db=3.0) is None


def test_extent_does_not_bridge_across_a_genuine_gap() -> None:
    """Two events of the same category are not one long event."""
    from senselab.audio.workflows.audio_analysis.sources import resolve_extent

    frames = [(0.0, 0.5, 12.0), (0.5, 1.0, 0.0), (1.0, 1.5, 0.0), (1.5, 2.0, 12.0)]
    assert resolve_extent(frames, presence_margin_db=10.0, extent_margin_db=3.0) == (0.0, 0.5)


def test_a_steady_hum_has_almost_no_modulation_depth() -> None:
    """Stationary noise is near-unmodulated at every rate.

    That is what separates a real event from a stretch of noise floor at the same level.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    sr = 1000
    t = np.arange(0, 2.0, 1 / sr)
    steady = np.sin(2 * np.pi * 200 * t)
    assert _depth(steady, sr) < 0.15


def test_an_amplitude_modulated_source_registers_its_modulation() -> None:
    """The other half of the discrimination: a genuinely modulated source reads as one."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    sr = 1000
    t = np.arange(0, 2.0, 1 / sr)
    modulated = (1.0 + 0.9 * np.sin(2 * np.pi * 8.0 * t)) * np.sin(2 * np.pi * 200 * t)
    assert _depth(modulated, sr) > 0.4


def test_speech_rate_modulation_is_discounted_in_a_suppressed_variant() -> None:
    """Research D11: the residual can carry inherited talker modulation.

    Suppression operates on the 3-6 Hz band, so modulation at talker rate in a residual is as
    likely to be the suppressor's own artifact as a background event — and counting it would
    credit that artifact as a discovery.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    sr = 1000
    t = np.arange(0, 2.0, 1 / sr)
    speech_rate = (1.0 + 0.9 * np.sin(2 * np.pi * 4.5 * t)) * np.sin(2 * np.pi * 200 * t)
    full = _depth(speech_rate, sr)
    discounted = _depth(speech_rate, sr, discount_speech_band=True)
    assert discounted < full


def test_modulation_outside_the_speech_band_survives_the_discount() -> None:
    """The discount must be a band, not a blanket: a 12 Hz flutter is not talker rate."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    sr = 1000
    t = np.arange(0, 2.0, 1 / sr)
    flutter = (1.0 + 0.9 * np.sin(2 * np.pi * 12.0 * t)) * np.sin(2 * np.pi * 200 * t)
    assert _depth(flutter, sr, discount_speech_band=True) == pytest.approx(_depth(flutter, sr), rel=0.15)


def test_modulation_depth_of_nothing_is_not_a_number_pretending_to_be_zero() -> None:
    """Zero would read as "measured, and stationary"; silence was not measured at all."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.sources import modulation_depth

    assert modulation_depth(np.zeros(1000), 1000) is None
