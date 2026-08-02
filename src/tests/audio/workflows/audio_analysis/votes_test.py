"""Unit tests for the pure link half of the harvest/link split (votes.py).

``link_pass`` produces two things and neither is an axis: the L1 per-signal rows, and the belief
buckets L2 fuses. The fold that used to happen here — once per pass, per axis — is a category
error and now lives only in ``fuse.fuse_axis``, which sees every pass at once.
"""

import pytest

from senselab.audio.workflows.audio_analysis.compute import _apply_scene_coupling
from senselab.audio.workflows.audio_analysis.degradation import DEFAULT_ANCHORS
from senselab.audio.workflows.audio_analysis.types import FusedAxis
from senselab.audio.workflows.audio_analysis.votes import (
    DEFAULT_UTTERANCE_SCENE_COUPLING,
    PassHarvest,
    intensity_mask,
    link_pass,
    mask_from_pvoice,
)


def _harvest() -> PassHarvest:
    return PassHarvest(
        pass_label="raw_16k",
        speech_presence_evidence=[
            {
                "start": 0.0,
                "end": 0.5,
                "evidence": {"m1": {"covered_fraction": 1.0}, "m2": {"covered_fraction": 1.0}},
            },
            {
                "start": 0.5,
                "end": 1.0,
                "evidence": {"m1": {"covered_fraction": 1.0}, "m2": {"covered_fraction": 0.0}},
                "frame_dispersion": 0.5,
            },
        ],
        speaker_votes=[{"start": 0.0, "end": 1.0, "votes": {"__cross_diar_label_disagreement__": {"value": 0.8}}}],
        asr_votes=[
            {
                "start": 0.0,
                "end": 1.0,
                "votes": {
                    "a": {"text": "hi", "avg_logprob": None},
                    "b": {"text": "hi", "avg_logprob": None},
                    "__pairwise_phoneme_distances__": {"pairs": {"a|b": 0.25}, "per_source_confidence": {}},
                },
            }
        ],
        # L1 measurements in dB; 23 dB against the 25/5 dB anchors scores 0.1 at L2.
        quality_by_bucket={(0.0, 0.5): {"snr_brouhaha_db": 23.0, "c50_brouhaha_db": 28.0}},
        source_by_bucket={(0.0, 0.5): {"src_speech": 0.9, "src_dominant": "speech", "_raw": {"speech": 0.9}}},
        grids={
            "speech_presence": {"win_length": 0.5, "hop_length": 0.5},
            "speaker": {"win_length": 1.0, "hop_length": 1.0},
            "asr": {"win_length": 1.0, "hop_length": 1.0},
        },
        provenance_extras={"scene_quality": {"enabled": True}},
    )


def test_link_pass_emits_per_signal_rows_in_native_units() -> None:
    """One row per (signal, bucket), carrying the tool's own measurement — no fold, no axis."""
    linked = link_pass(_harvest(), params={"p": 1})

    assert set(linked.signal_results) >= {"m1", "m2", "a", "b", "scene_quality", "frame_dispersion"}
    m1 = linked.signal_results["m1"]
    assert m1.pass_label == "raw_16k"
    assert [(r.start, r.end) for r in m1.rows] == [(0.0, 0.5), (0.5, 1.0)]
    assert m1.rows[0].measurement == {"covered_fraction": 1.0}

    # Native units, not a rescaled score.
    scene = linked.signal_results["scene_quality"].rows[0]
    assert scene.measurement["snr_brouhaha_db"] == 23.0
    assert "quality_snr" not in scene.measurement

    # Frame dispersion is persisted as a signal, so the artifact-driven path can read it.
    assert linked.signal_results["frame_dispersion"].rows[0].measurement["frame_dispersion"] == pytest.approx(0.5)


def test_link_pass_records_the_policy_that_read_the_measurements() -> None:
    """Every threshold that shaped a belief is named in the provenance the run records."""
    linked = link_pass(_harvest(), params={"p": 1})
    assert "speech_presence_policy" in linked.provenance
    assert linked.provenance["scene_quality"] == {"enabled": True}
    assert linked.provenance["grids"]["asr"] == {"win_length": 1.0, "hop_length": 1.0}


def test_link_pass_emits_belief_buckets_per_axis_but_no_axis_value() -> None:
    """The buckets L2 fuses come out; a per-pass axis number does not."""
    linked = link_pass(_harvest(), params={})
    assert set(linked.buckets_by_axis) == {"speech_presence", "speaker", "asr"}
    assert len(linked.buckets_by_axis["speech_presence"]) == 2
    for buckets in linked.buckets_by_axis.values():
        for bucket in buckets:
            assert "within_pass_uncertainty" not in bucket


def test_link_pass_derives_quality_scores_at_l2_not_l1() -> None:
    """The anchored [0,1] score is a fusion decision; L1 keeps only the dB reading."""
    linked = link_pass(_harvest(), params={})
    assert linked.quality_scores[(0.0, 0.5)]["quality_snr"] == pytest.approx(0.1)


def test_intensity_mask_ramp_and_overlap_average() -> None:
    """p_voice < 0.5 ramps linearly; overlapping intervals average."""
    assert mask_from_pvoice(0.8) == 1.0
    assert mask_from_pvoice(0.25) == pytest.approx(0.5)
    assert intensity_mask(0.0, 1.0, [(0.0, 0.5, 1.0), (0.5, 1.0, 0.0)]) == pytest.approx(0.5)
    assert intensity_mask(0.0, 1.0, []) == 1.0  # no speech_presence evidence → no masking


# ── Scene→asr coupling, now applied at L2 on the fused axis (FR-019) ───────────


def _fused(
    *,
    quality_snr: float | None = None,
    src_machine: float | None = None,
    src_environment: float | None = None,
    triage: float = 0.4,
) -> dict[str, FusedAxis]:
    """One 1 s fused asr bucket over two 0.5 s fused presence buckets carrying scene columns."""
    presence_rows = []
    for start, end in ((0.0, 0.5), (0.5, 1.0)):
        row: dict[str, object] = {"start": start, "end": end, "uncertainty": 0.0}
        if quality_snr is not None:
            row["quality_snr"] = quality_snr
        if src_machine is not None:
            row["src_machine"] = src_machine
        if src_environment is not None:
            row["src_environment"] = src_environment
        presence_rows.append(row)
    return {
        "speech_presence": FusedAxis(axis="speech_presence", rows=presence_rows),
        "asr": FusedAxis(axis="asr", rows=[{"start": 0.0, "end": 1.0, "triage_score": triage}]),
    }


def test_poor_quality_raises_the_asr_triage_score() -> None:
    """FR-019: degraded audio must push asr triage up, visibly."""
    clean = _fused(quality_snr=0.0)
    noisy = _fused(quality_snr=1.0)
    _apply_scene_coupling(clean, {})
    _apply_scene_coupling(noisy, {})
    assert noisy["asr"].rows[0]["triage_score"] > clean["asr"].rows[0]["triage_score"]


def test_coupling_multiplier_and_pre_coupling_value_are_recorded() -> None:
    """The multiplier and the un-coupled number land on the row, so the adjustment is auditable."""
    axes = _fused(quality_snr=1.0)
    _apply_scene_coupling(axes, {})
    row = axes["asr"].rows[0]
    expected = 1.0 + DEFAULT_UTTERANCE_SCENE_COUPLING["w_q"]
    assert row["scene_quality_coupling"] == pytest.approx(expected)
    assert row["triage_score_pre_coupling"] == pytest.approx(0.4)
    assert row["coupled_from"] == ["scene_quality"]
    assert axes["asr"].provenance["asr_scene_coupling"]["applies_to"] == "triage_score"


def test_entropy_uncertainty_is_not_coupled() -> None:
    """Coupling is a policy fold; the entropy measure has no policy in it and must not move."""
    axes = _fused(quality_snr=1.0)
    axes["asr"].rows[0]["uncertainty"] = 0.3
    _apply_scene_coupling(axes, {})
    assert axes["asr"].rows[0]["uncertainty"] == pytest.approx(0.3)


def test_clean_scene_leaves_triage_untouched() -> None:
    """Zero degradation and no competing source → coupling exactly 1.0."""
    axes = _fused(quality_snr=0.0, src_machine=0.0, src_environment=0.0)
    _apply_scene_coupling(axes, {})
    assert axes["asr"].rows[0]["scene_quality_coupling"] == pytest.approx(1.0)
    assert axes["asr"].rows[0]["triage_score"] == pytest.approx(0.4)


def test_absent_scene_columns_are_a_no_op() -> None:
    """Scene features disabled → identical values to the pre-feature behavior (SC-008)."""
    axes = _fused()
    _apply_scene_coupling(axes, {})
    assert axes["asr"].rows[0]["scene_quality_coupling"] == pytest.approx(1.0)
    assert axes["asr"].rows[0]["triage_score"] == pytest.approx(0.4)


def test_competing_non_speech_source_raises_triage() -> None:
    """A machine / environment source competing with speech raises the triage score."""
    quiet = _fused(src_machine=0.0, src_environment=0.0)
    noisy = _fused(src_machine=0.6, src_environment=0.4)
    _apply_scene_coupling(quiet, {})
    _apply_scene_coupling(noisy, {})
    assert noisy["asr"].rows[0]["triage_score"] > quiet["asr"].rows[0]["triage_score"]


def test_coupled_triage_clamped_to_one() -> None:
    """The multiplier can't push the reported value out of [0, 1]."""
    axes = _fused(quality_snr=1.0, src_machine=1.0, src_environment=1.0, triage=0.95)
    _apply_scene_coupling(axes, {})
    assert axes["asr"].rows[0]["triage_score"] == pytest.approx(1.0)


def test_coupling_weights_configurable_via_params() -> None:
    """Operators can retune (or disable) the coupling without a code change."""
    off = _fused(quality_snr=1.0)
    _apply_scene_coupling(off, {"asr_scene_coupling": {"w_q": 0.0, "w_s": 0.0}})
    assert off["asr"].rows[0]["scene_quality_coupling"] == pytest.approx(1.0)
    assert off["asr"].rows[0]["triage_score"] == pytest.approx(0.4)

    strong = _fused(quality_snr=1.0)
    _apply_scene_coupling(strong, {"asr_scene_coupling": {"w_q": 2.0, "w_s": 0.0}})
    assert strong["asr"].rows[0]["scene_quality_coupling"] == pytest.approx(3.0)


def test_default_anchors_are_the_l2_calibration_not_an_l1_constant() -> None:
    """The dB→[0,1] anchors belong to the scoring step, so they are named where scoring happens."""
    assert DEFAULT_ANCHORS["snr_clean_db"] > DEFAULT_ANCHORS["snr_floor_db"]
