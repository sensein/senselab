"""Unit tests for the pure aggregate half of the harvest/aggregate split (votes.py)."""

import pytest

from senselab.audio.workflows.audio_analysis.aggregate import aggregate_presence, aggregate_utterance
from senselab.audio.workflows.audio_analysis.types import UncertaintyRow
from senselab.audio.workflows.audio_analysis.votes import (
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
    assert pres[0].aggregated_uncertainty == pytest.approx(
        aggregate_presence({"m1": {"speaks": True}, "m2": {"speaks": True}})
    )
    assert pres[0].quality_snr == 0.1 and pres[0].src_dominant == "speech"
    assert "__quality__" in pres[0].model_votes and "__quality__" not in pres[0].contributing_models
    # OR formula: u=1.0 (50/50 split), instability 0.5 → presence_uncertainty = 1-(1-1)(1-0.5) = 1.0
    assert pres[1].aggregated_uncertainty == pytest.approx(1.0)
    assert pres[1].presence_uncertainty == pytest.approx(1.0)
    assert pres[1].raw_aggregated_uncertainty == pres[1].aggregated_uncertainty  # OR never leaks into primary

    ident = results["identity"].rows[0]
    assert ident.aggregated_uncertainty == pytest.approx(0.8)
    # intensity mask = mean over the two presence buckets: p=1.0 → 1.0; p=0.5 → 1.0 (>=0.5)
    assert ident.intensity_weight == pytest.approx(1.0)
    assert ident.raw_aggregated_uncertainty == pytest.approx(0.8)  # mask kept out of primary

    utt = results["utterance"].rows[0]
    assert utt.aggregated_uncertainty == pytest.approx(
        aggregate_utterance(_harvest().utterance_votes[0]["votes"], aggregator="min")
    )
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
        aggregated_uncertainty=u,
        contributing_models=["m"],
        model_votes={"m": {"text": "x"}},
        comparison_status="ok" if u is not None else "incomparable",
        raw_aggregated_uncertainty=u,
        intensity_weight=1.0,
    )


def test_compute_pass_deltas_pairing() -> None:
    """|raw − enh| on shared buckets; one_sided rows for unpaired; incomparable on None."""
    raw = [_row(0.0, 1.0, 0.2), _row(1.0, 2.0, None), _row(2.0, 3.0, 0.5)]
    enh = [_row(0.0, 1.0, 0.7), _row(1.0, 2.0, 0.4)]
    deltas = compute_pass_deltas(raw, enh, "utterance", "min")
    assert [d.comparison_status for d in deltas] == ["ok", "incomparable", "one_sided"]
    assert deltas[0].aggregated_uncertainty == pytest.approx(0.5)
    assert "raw_16k::m" in deltas[0].model_votes and "enhanced_16k::m" in deltas[0].model_votes
    assert deltas[2].aggregated_uncertainty is None
