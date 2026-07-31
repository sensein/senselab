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
        within_pass_uncertainty=u,
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
    )


def test_disagreements_index_ranks_by_uncertainty_desc(tmp_path: Path) -> None:
    """Disagreements index ranks by uncertainty desc."""
    axis_results = {
        ("raw_16k", "speech_presence"): AxisResult(
            pass_label="raw_16k",
            axis="speech_presence",
            rows=[_row(0.0, "speech_presence", 0.2), _row(1.0, "speech_presence", 0.9)],
        ),
        ("raw_16k", "asr"): AxisResult(
            pass_label="raw_16k",
            axis="asr",
            rows=[_row(0.5, "asr", 0.5)],
        ),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert [e["within_pass_uncertainty"] for e in idx["entries"]] == [0.9, 0.5, 0.2]
    assert idx["entries"][0]["rank"] == 1
    assert idx["entries"][2]["rank"] == 3


def test_disagreements_axis_priority_tiebreak(tmp_path: Path) -> None:
    """Same uncertainty → asr > speaker > speech_presence."""
    axis_results = {
        ("raw_16k", "speech_presence"): AxisResult(
            pass_label="raw_16k", axis="speech_presence", rows=[_row(0.0, "speech_presence", 0.5)]
        ),
        ("raw_16k", "speaker"): AxisResult(pass_label="raw_16k", axis="speaker", rows=[_row(0.0, "speaker", 0.5)]),
        ("raw_16k", "asr"): AxisResult(pass_label="raw_16k", axis="asr", rows=[_row(0.0, "asr", 0.5)]),
    }
    idx = build_disagreements_index(
        axis_results=axis_results,
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    axes = [e["axis"] for e in idx["entries"]]
    assert axes == ["asr", "speaker", "speech_presence"]


def test_disagreements_top_n_truncates(tmp_path: Path) -> None:
    """Disagreements top n truncates."""
    axis_results = {
        ("raw_16k", "speech_presence"): AxisResult(
            pass_label="raw_16k",
            axis="speech_presence",
            rows=[_row(float(i), "speech_presence", 1.0 - i * 0.1) for i in range(5)],
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
        ("raw_16k", "speech_presence"): AxisResult(
            pass_label="raw_16k", axis="speech_presence", rows=[_row(0.0, "speech_presence", 0.9)]
        ),
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
        ("raw_16k", "speech_presence"): AxisResult(
            pass_label="raw_16k",
            axis="speech_presence",
            rows=[_row(0.0, "speech_presence", None), _row(1.0, "speech_presence", 0.5)],
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
    assert idx["entries"][0]["within_pass_uncertainty"] == 0.5
    assert idx["entries"][1]["within_pass_uncertainty"] is None


# ── FR-024 (T042): speech_presence sub-signals in the index ─────────────────────


def test_speech_presence_entries_carry_and_tiebreak_on_sub_signals(tmp_path: Path) -> None:
    """Presence entries expose scene sub-signals; equal-aggregated ties rank by speech_presence_uncertainty."""
    low_instability = UncertaintyRow(
        start=0.0,
        end=0.5,
        axis="speech_presence",
        within_pass_uncertainty=0.7,
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
        speech_presence_uncertainty=0.7,
        snr_brouhaha_db=12.0,
        src_dominant="speech",
    )
    high_instability = UncertaintyRow(
        start=1.0,
        end=1.5,
        axis="speech_presence",
        within_pass_uncertainty=0.7,  # same primary — tiebreak must use speech_presence_uncertainty
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
        speech_presence_uncertainty=0.95,
    )
    index = build_disagreements_index(
        axis_results={
            ("raw_16k", "speech_presence"): AxisResult(
                pass_label="raw_16k", axis="speech_presence", rows=[low_instability, high_instability]
            )
        },
        top_n=10,
        run_dir=tmp_path,
        config={},
        incomparable_reasons={},
    )
    entries = index["entries"]
    assert entries[0]["start"] == 1.0 and entries[0]["speech_presence_uncertainty"] == 0.95
    assert entries[1]["speech_presence_uncertainty"] == 0.7
    assert entries[1]["snr_brouhaha_db"] == 12.0 and entries[1]["src_dominant"] == "speech"
    assert "speech_presence_unc=" in entries[0]["summary"]
    # Null-safe: absent sub-signals are omitted, not null-filled.
    assert "snr_brouhaha_db" not in entries[0]
