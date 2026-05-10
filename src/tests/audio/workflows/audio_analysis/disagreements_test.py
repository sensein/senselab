"""Disagreements index ranking + axis-priority tiebreak (T034)."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.disagreements import build_disagreements_index
from senselab.audio.workflows.audio_analysis.types import AxisResult, UncertaintyRow


def _row(start: float, axis: str, u: float | None) -> UncertaintyRow:
    return UncertaintyRow(
        start=start,
        end=start + 0.5,
        axis=axis,  # type: ignore[arg-type]
        aggregated_uncertainty=u,
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
    )


def test_disagreements_index_ranks_by_uncertainty_desc(tmp_path: Path) -> None:
    """Disagreements index ranks by uncertainty desc."""
    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[_row(0.0, "presence", 0.2), _row(1.0, "presence", 0.9)],
        ),
        ("raw_16k", "utterance"): AxisResult(
            pass_label="raw_16k",
            axis="utterance",
            rows=[_row(0.5, "utterance", 0.5)],
        ),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert [e["aggregated_uncertainty"] for e in idx["entries"]] == [0.9, 0.5, 0.2]
    assert idx["entries"][0]["rank"] == 1
    assert idx["entries"][2]["rank"] == 3


def test_disagreements_axis_priority_tiebreak(tmp_path: Path) -> None:
    """Same uncertainty → utterance > identity > presence."""
    axis_results = {
        ("raw_16k", "presence"): AxisResult(pass_label="raw_16k", axis="presence", rows=[_row(0.0, "presence", 0.5)]),
        ("raw_16k", "identity"): AxisResult(pass_label="raw_16k", axis="identity", rows=[_row(0.0, "identity", 0.5)]),
        ("raw_16k", "utterance"): AxisResult(
            pass_label="raw_16k", axis="utterance", rows=[_row(0.0, "utterance", 0.5)]
        ),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    axes = [e["axis"] for e in idx["entries"]]
    assert axes == ["utterance", "identity", "presence"]


def test_disagreements_top_n_truncates(tmp_path: Path) -> None:
    """Disagreements top n truncates."""
    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[_row(float(i), "presence", 1.0 - i * 0.1) for i in range(5)],
        ),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=2,
        run_dir=tmp_path,
        config={"top_n": 2, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert len(idx["entries"]) == 2
    assert idx["totals"]["total_rows"] == 5


def test_disagreements_top_n_zero_returns_empty_entries(tmp_path: Path) -> None:
    """top_n=0 → no entries listed; totals still populated."""
    axis_results = {
        ("raw_16k", "presence"): AxisResult(pass_label="raw_16k", axis="presence", rows=[_row(0.0, "presence", 0.9)]),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=0,
        run_dir=tmp_path,
        config={"top_n": 0, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert idx["entries"] == []
    assert idx["totals"]["total_rows"] == 1


def test_disagreements_nan_uncertainty_sorts_last(tmp_path: Path) -> None:
    """Disagreements nan uncertainty sorts last."""
    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[_row(0.0, "presence", None), _row(1.0, "presence", 0.5)],
        ),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    # Non-null first.
    assert idx["entries"][0]["aggregated_uncertainty"] == 0.5
    assert idx["entries"][1]["aggregated_uncertainty"] is None
