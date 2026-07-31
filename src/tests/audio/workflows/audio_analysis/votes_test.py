"""Unit tests for the pure aggregate half of the harvest/aggregate split (votes.py)."""

import pytest

from senselab.audio.workflows.audio_analysis.aggregate import aggregate_presence, aggregate_utterance
from senselab.audio.workflows.audio_analysis.types import UncertaintyRow
from senselab.audio.workflows.audio_analysis.votes import (
    DEFAULT_UTTERANCE_SCENE_COUPLING,
    PassHarvest,
    aggregate_pass,
    compute_pass_deltas,
    intensity_mask,
    mask_from_pvoice,
)


def _harvest() -> PassHarvest:
    return PassHarvest(
        pass_label="raw_16k",
        presence_votes=[
            {  # confident speech → low uncertainty, p_voice 1.0
                "start": 0.0,
                "end": 0.5,
                "votes": {"m1": {"speaks": True, "native_confidence": None}, "m2": {"speaks": True}},
            },
            {  # split vote + instability → OR formula on presence_uncertainty column
                "start": 0.5,
                "end": 1.0,
                "votes": {"m1": {"speaks": True}, "m2": {"speaks": False}},
                "frame_instability": 0.5,
            },
        ],
        identity_votes=[{"start": 0.0, "end": 1.0, "votes": {"__cross_diar_label_disagreement__": {"value": 0.8}}}],
        utterance_votes=[
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
        quality_by_bucket={(0.0, 0.5): {"quality_snr": 0.1, "_raw": {"snr_db": 22.0}}},
        source_by_bucket={(0.0, 0.5): {"src_speech": 0.9, "src_dominant": "speech", "_raw": {}}},
        grids={
            "presence": {"win_length": 0.5, "hop_length": 0.5},
            "identity": {"win_length": 1.0, "hop_length": 1.0},
            "utterance": {"win_length": 1.0, "hop_length": 1.0},
        },
        provenance_extras={"scene_quality": {"enabled": True}},
    )


def test_aggregate_pass_matches_pure_aggregators_and_columns() -> None:
    """Row values must equal the pure per-axis aggregators; additive columns attach."""
    results = aggregate_pass(_harvest(), aggregator="min", params={"p": 1})
    pres = results["presence"].rows
    assert len(pres) == 2
    assert pres[0].within_pass_uncertainty == pytest.approx(
        aggregate_presence({"m1": {"speaks": True}, "m2": {"speaks": True}})
    )
    assert pres[0].quality_snr == 0.1 and pres[0].src_dominant == "speech"
    assert "__quality__" in pres[0].model_votes and "__quality__" not in pres[0].contributing_models
    # OR formula: u=1.0 (50/50 split), instability 0.5 → presence_uncertainty = 1-(1-1)(1-0.5) = 1.0
    assert pres[1].within_pass_uncertainty == pytest.approx(1.0)
    assert pres[1].presence_uncertainty == pytest.approx(1.0)
    assert pres[1].raw_within_pass_uncertainty == pres[1].within_pass_uncertainty  # OR never leaks into primary

    ident = results["identity"].rows[0]
    assert ident.within_pass_uncertainty == pytest.approx(0.8)
    # intensity mask = mean over the two presence buckets: p=1.0 → 1.0; p=0.5 → 1.0 (>=0.5)
    assert ident.intensity_weight == pytest.approx(1.0)
    assert ident.raw_within_pass_uncertainty == pytest.approx(0.8)  # mask kept out of primary

    utt = results["utterance"].rows[0]
    # The fixture's presence bucket carries quality_snr=0.1, so FR-019 coupling
    # applies: reported = raw × (1 + w_q·0.1) = raw × 1.05. The pre-coupling number
    # stays on raw_within_pass_uncertainty.
    utt_raw = aggregate_utterance(_harvest().utterance_votes[0]["votes"], aggregator="min")
    assert utt_raw is not None
    assert utt.raw_within_pass_uncertainty == pytest.approx(utt_raw)
    expected_coupling = 1.0 + DEFAULT_UTTERANCE_SCENE_COUPLING["w_q"] * 0.1
    assert utt.scene_quality_coupling == pytest.approx(expected_coupling)
    assert utt.within_pass_uncertainty == pytest.approx(utt_raw * expected_coupling)
    assert results["presence"].provenance["scene_quality"] == {"enabled": True}
    assert results["utterance"].provenance["grid"] == {"win_length": 1.0, "hop_length": 1.0}


def test_intensity_mask_ramp_and_overlap_average() -> None:
    """p_voice < 0.5 ramps linearly; overlapping intervals average."""
    assert mask_from_pvoice(0.8) == 1.0
    assert mask_from_pvoice(0.25) == pytest.approx(0.5)
    assert intensity_mask(0.0, 1.0, [(0.0, 0.5, 1.0), (0.5, 1.0, 0.0)]) == pytest.approx(0.5)
    assert intensity_mask(0.0, 1.0, []) == 1.0  # no presence evidence → no masking


def _row(start: float, end: float, u: float | None, axis: str = "utterance") -> UncertaintyRow:
    return UncertaintyRow(
        start=start,
        end=end,
        axis=axis,  # type: ignore[arg-type]
        within_pass_uncertainty=u,
        contributing_models=["m"],
        model_votes={"m": {"text": "x"}},
        comparison_status="ok" if u is not None else "incomparable",
        raw_within_pass_uncertainty=u,
        intensity_weight=1.0,
    )


def test_compute_pass_deltas_pairing() -> None:
    """|raw − enh| on shared buckets; one_sided rows for unpaired; incomparable on None."""
    raw = [_row(0.0, 1.0, 0.2), _row(1.0, 2.0, None), _row(2.0, 3.0, 0.5)]
    enh = [_row(0.0, 1.0, 0.7), _row(1.0, 2.0, 0.4)]
    deltas = compute_pass_deltas(raw, enh, "utterance", "min")
    assert [d.comparison_status for d in deltas] == ["ok", "incomparable", "one_sided"]
    assert deltas[0].within_pass_uncertainty == pytest.approx(0.5)
    assert "raw_16k::m" in deltas[0].model_votes and "enhanced_16k::m" in deltas[0].model_votes
    assert deltas[2].within_pass_uncertainty is None


# ── Scene→utterance coupling (T033, FR-019) ───────────────────────────


def _coupling_harvest(
    *,
    quality_snr: float | None = None,
    src_machine: float | None = None,
    src_environment: float | None = None,
) -> PassHarvest:
    """One 1 s utterance bucket over two 0.5 s presence buckets carrying scene columns."""
    quality: dict[tuple[float, float], dict[str, object]] = {}
    sources: dict[tuple[float, float], dict[str, object]] = {}
    for bucket in ((0.0, 0.5), (0.5, 1.0)):
        if quality_snr is not None:
            quality[bucket] = {"quality_snr": quality_snr, "_raw": {}}
        if src_machine is not None or src_environment is not None:
            sources[bucket] = {
                "src_machine": src_machine,
                "src_environment": src_environment,
                "_raw": {},
            }
    return PassHarvest(
        pass_label="raw_16k",
        presence_votes=[
            {"start": 0.0, "end": 0.5, "votes": {"m1": {"speaks": True}}},
            {"start": 0.5, "end": 1.0, "votes": {"m1": {"speaks": True}}},
        ],
        utterance_votes=[
            {
                "start": 0.0,
                "end": 1.0,
                "votes": {
                    "a": {"text": "hi"},
                    "b": {"text": "hi"},
                    "__pairwise_phoneme_distances__": {"pairs": {"a|b": 0.4}, "per_source_confidence": {}},
                },
            }
        ],
        quality_by_bucket=quality,
        source_by_bucket=sources,
        grids={"utterance": {"win_length": 1.0, "hop_length": 1.0}},
    )


def test_poor_quality_raises_reported_utterance_uncertainty() -> None:
    """FR-019: degraded audio must push utterance uncertainty up, visibly."""
    clean = aggregate_pass(_coupling_harvest(quality_snr=0.0), aggregator="min", params={})
    noisy = aggregate_pass(_coupling_harvest(quality_snr=1.0), aggregator="min", params={})
    clean_row = clean["utterance"].rows[0]
    noisy_row = noisy["utterance"].rows[0]
    assert noisy_row.within_pass_uncertainty is not None and clean_row.within_pass_uncertainty is not None
    assert noisy_row.within_pass_uncertainty > clean_row.within_pass_uncertainty


def test_coupling_multiplier_is_recorded_not_hidden() -> None:
    """The multiplier lands on its own column so the adjustment is auditable."""
    row = aggregate_pass(_coupling_harvest(quality_snr=1.0), aggregator="min", params={})["utterance"].rows[0]
    assert row.scene_quality_coupling is not None
    assert row.scene_quality_coupling > 1.0


def test_pre_coupling_value_preserved() -> None:
    """raw_within_pass_uncertainty and model_votes keep the un-coupled number."""
    harvest = _coupling_harvest(quality_snr=1.0)
    expected = aggregate_utterance(harvest.utterance_votes[0]["votes"], aggregator="min")
    row = aggregate_pass(harvest, aggregator="min", params={})["utterance"].rows[0]
    assert row.raw_within_pass_uncertainty == pytest.approx(expected)
    assert row.model_votes["__utterance_pre_coupling__"]["value"] == pytest.approx(expected)
    assert row.within_pass_uncertainty != pytest.approx(expected)


def test_clean_scene_leaves_uncertainty_untouched() -> None:
    """Zero degradation and no competing source → coupling exactly 1.0."""
    harvest = _coupling_harvest(quality_snr=0.0, src_machine=0.0, src_environment=0.0)
    expected = aggregate_utterance(harvest.utterance_votes[0]["votes"], aggregator="min")
    row = aggregate_pass(harvest, aggregator="min", params={})["utterance"].rows[0]
    assert row.scene_quality_coupling == pytest.approx(1.0)
    assert row.within_pass_uncertainty == pytest.approx(expected)


def test_absent_scene_columns_are_a_no_op() -> None:
    """Scene features disabled → identical values to the pre-feature behavior (SC-008)."""
    harvest = _coupling_harvest()  # no quality, no sources
    expected = aggregate_utterance(harvest.utterance_votes[0]["votes"], aggregator="min")
    row = aggregate_pass(harvest, aggregator="min", params={})["utterance"].rows[0]
    assert row.within_pass_uncertainty == pytest.approx(expected)
    assert row.scene_quality_coupling == pytest.approx(1.0)


def test_competing_non_speech_source_raises_uncertainty() -> None:
    """A machine / environment source competing with speech raises uncertainty."""
    quiet = aggregate_pass(_coupling_harvest(src_machine=0.0, src_environment=0.0), aggregator="min", params={})[
        "utterance"
    ].rows[0]
    noisy = aggregate_pass(_coupling_harvest(src_machine=0.6, src_environment=0.4), aggregator="min", params={})[
        "utterance"
    ].rows[0]
    assert noisy.within_pass_uncertainty is not None and quiet.within_pass_uncertainty is not None
    assert noisy.within_pass_uncertainty > quiet.within_pass_uncertainty


def test_coupled_uncertainty_clamped_to_one() -> None:
    """The multiplier can't push the reported value out of [0, 1]."""
    harvest = _coupling_harvest(quality_snr=1.0, src_machine=1.0, src_environment=1.0)
    harvest.utterance_votes[0]["votes"]["__pairwise_phoneme_distances__"]["pairs"]["a|b"] = 0.95
    row = aggregate_pass(harvest, aggregator="min", params={})["utterance"].rows[0]
    assert row.within_pass_uncertainty == pytest.approx(1.0)


def test_coupling_weights_configurable_via_params() -> None:
    """Operators can retune (or disable) the coupling without a code change."""
    harvest = _coupling_harvest(quality_snr=1.0)
    expected = aggregate_utterance(harvest.utterance_votes[0]["votes"], aggregator="min")
    off = aggregate_pass(harvest, aggregator="min", params={"utterance_scene_coupling": {"w_q": 0.0, "w_s": 0.0}})[
        "utterance"
    ].rows[0]
    assert off.scene_quality_coupling == pytest.approx(1.0)
    assert off.within_pass_uncertainty == pytest.approx(expected)

    strong = aggregate_pass(harvest, aggregator="min", params={"utterance_scene_coupling": {"w_q": 2.0, "w_s": 0.0}})[
        "utterance"
    ].rows[0]
    assert strong.scene_quality_coupling == pytest.approx(3.0)
