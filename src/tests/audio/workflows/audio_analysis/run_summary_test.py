"""``final/summary.md`` — the run's headline, readable without opening a parquet.

``summary.json`` is the machine record and is already large. A person opening a run needs to
know four things quickly: how many speakers, how uncertain each axis was, where the worst
regions are, and whether the loop converged or ran out of rounds. Those are scattered across
L2 parquets and JSON today, which means the answer to "how did this run go" requires knowing
the layout.
"""

from __future__ import annotations

import json

import pytest

from senselab.audio.workflows.audio_analysis.summary import build_run_summary


def _axis_rows(values: list[float | None]) -> list[dict]:
    return [
        {"start": i * 0.5, "end": i * 0.5 + 0.5, "uncertainty": v, "epistemic_uncertainty": v}
        for i, v in enumerate(values)
    ]


def test_the_summary_reports_a_per_axis_headline() -> None:
    """Mean and worst per axis, because a mean alone hides a single bad region."""
    doc = build_run_summary(
        axis_rows={"identity": _axis_rows([0.1, 0.9, 0.2])},
        speakers={"count_posterior": {"probabilities": {"2": 1.0}, "modal_count": 2}},
        rounds={"identity": [{"round": 0, "converged": True}]},
    )
    axis = doc["axes"]["identity"]
    assert axis["mean_uncertainty"] == pytest.approx(0.4)
    assert axis["max_uncertainty"] == pytest.approx(0.9)


def test_unmeasured_buckets_are_counted_not_averaged_in() -> None:
    """Treating "not measured" as zero would report a run as more certain than it was."""
    doc = build_run_summary(axis_rows={"identity": _axis_rows([0.8, None, None])}, speakers={}, rounds={})
    axis = doc["axes"]["identity"]
    assert axis["mean_uncertainty"] == pytest.approx(0.8)
    assert axis["unmeasured_buckets"] == 2


def test_the_reducible_share_is_reported() -> None:
    """Whether more measurement could help is the actionable part of an uncertainty figure."""
    doc = build_run_summary(axis_rows={"identity": _axis_rows([0.6, 0.6])}, speakers={}, rounds={})
    assert doc["axes"]["identity"]["mean_epistemic_uncertainty"] == pytest.approx(0.6)


def test_the_worst_regions_are_named_with_their_times() -> None:
    """ "Uncertainty was 0.4" is not actionable; "0.9 at 0.5-1.0 s" is."""
    doc = build_run_summary(axis_rows={"identity": _axis_rows([0.1, 0.9, 0.2])}, speakers={}, rounds={}, top_n=1)
    worst = doc["axes"]["identity"]["worst_regions"][0]
    assert worst["start"] == pytest.approx(0.5) and worst["uncertainty"] == pytest.approx(0.9)


def test_the_speaker_count_is_reported_with_its_competing_reading() -> None:
    """A modal count alone hides a contested posterior, which is the case worth seeing."""
    doc = build_run_summary(
        axis_rows={},
        speakers={"count_posterior": {"probabilities": {"1": 0.6, "5": 0.4}, "modal_count": 1, "is_multimodal": True}},
        rounds={},
    )
    assert doc["speakers"]["modal_count"] == 1
    assert doc["speakers"]["is_multimodal"] is True
    assert doc["speakers"]["probabilities"] == {"1": 0.6, "5": 0.4}


def test_convergence_distinguishes_settling_from_running_out() -> None:
    """They call for different follow-up: one is done, the other needs more budget."""
    settled = build_run_summary(axis_rows={}, speakers={}, rounds={"identity": [{"round": 1, "converged": True}]})
    exhausted = build_run_summary(axis_rows={}, speakers={}, rounds={"identity": [{"round": 1, "converged": False}]})
    assert settled["convergence"]["identity"] == "converged"
    assert exhausted["convergence"]["identity"] == "rounds_exhausted"


def test_the_summary_renders_as_markdown() -> None:
    """Readable without a parquet reader — the whole point of it existing."""
    from senselab.audio.workflows.audio_analysis.summary import render_run_summary

    text = render_run_summary(
        build_run_summary(
            axis_rows={"identity": _axis_rows([0.1, 0.9])},
            speakers={"count_posterior": {"probabilities": {"2": 1.0}, "modal_count": 2}},
            rounds={"identity": [{"round": 0, "converged": True}]},
        )
    )
    assert "identity" in text
    assert "0.9" in text


def test_an_empty_run_summarises_without_inventing_numbers() -> None:
    """A run where nothing was measured must say so rather than report zeros."""
    doc = build_run_summary(axis_rows={"identity": []}, speakers={}, rounds={})
    assert doc["axes"]["identity"]["mean_uncertainty"] is None
    json.dumps(doc)
