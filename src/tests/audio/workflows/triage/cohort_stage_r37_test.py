"""COHORT's r37 amendments: speaker pools restricted to acoustic activity (B2); whether COHORT reads the participant (C2)."""

from __future__ import annotations

from dataclasses import replace

from senselab.audio.workflows.triage.cohort_fold import participant_matched
from senselab.audio.workflows.triage.cohort_stage import (
    RecordingFacts,
    activity_of,
    non_task_spans,
    target_spans,
)
from senselab.utils.prov_store import ProvStore


def _facts(stem: str = "sub-a_ses-1_task-word-color-stroop", **fields: object) -> RecordingFacts:
    base = RecordingFacts(
        stem=stem,
        session="sub-a_ses-1",
        family="word-color-stroop",
        lexical_task=True,
        duration_s=60.0,
        task_extent=(0.0, 60.0),
        lexical=((10.0, 10.4), (20.0, 20.4), (30.0, 30.4)),
        target=((10.0, 10.4), (20.0, 20.4), (30.0, 30.4)),
    )
    return replace(base, **fields)  # type: ignore[arg-type]


def test_a_target_pool_is_restricted_to_acoustic_activity() -> None:
    """017b7678: words merged across their gaps pool silence; activity leaves only the sound."""
    facts = _facts(target=((10.0, 10.4), (10.6, 11.0), (11.2, 11.6)), activity=((10.0, 10.5), (11.2, 11.4)))
    assert target_spans(facts, lexical_gap_s=0.5) == [(10.0, 10.5), (11.2, 11.4)]
    assert target_spans(replace(facts, activity=None), lexical_gap_s=0.5) == [(10.0, 11.6)]


def test_no_contrast_leaves_nothing_to_pool() -> None:
    """PREPROCESS found no activity at all: an empty pool, not the silence between words."""
    assert target_spans(_facts(activity=()), lexical_gap_s=0.5) == []


def test_the_non_task_pool_is_restricted_too() -> None:
    """The speech outside the task is pooled only where there is sound."""
    facts = _facts(
        family="respiration-and-cough-breath",
        lexical_task=False,
        target=(),
        non_task=((40.0, 40.5), (40.7, 41.2)),
        activity=((40.0, 40.4),),
    )
    assert non_task_spans(facts, lexical_gap_s=0.5) == [(40.0, 40.4)]


def test_activity_is_the_amplitude_spans() -> None:
    """Every live amplitude span on any signal, merged; none and no absence recorded is unknown."""
    store = ProvStore(run_id="activity")
    assert activity_of(store) is None
    for extent, measure in (((1.0, 2.0), "amplitude"), ((1.5, 3.0), "amplitude"), ((5.0, 6.0), "continuity")):
        store.entity(prov_type="span", extent=extent, attributes={"measure": measure, "signal": "preemphasised"})
    assert activity_of(store) == ((1.0, 3.0),)


def test_participant_matched_needs_a_match_and_no_non_match() -> None:
    """A matching target with a non-matching non-task pool is not the participant alone."""
    match = {"status": "measured", "match": True, "spans": [[0.0, 3.0]]}
    miss = {"status": "measured", "match": False, "spans": [[5.0, 8.0]]}
    assert participant_matched({"other_speaker": {"status": "measured", "target": match}})
    assert not participant_matched({"other_speaker": {"status": "measured", "target": match, "non_task": miss}})
    assert not participant_matched({"other_speaker": {"status": "unavailable", "target": match}})
    assert not participant_matched(None)
