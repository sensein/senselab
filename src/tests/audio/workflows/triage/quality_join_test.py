"""Tests for QUALITY's join of the background against the task spans (``quality_join.py``) and how the fold reads it."""

from __future__ import annotations

from typing import Any

import numpy as np

from senselab.audio.workflows.triage.background_model import BandFrames, Floor
from senselab.audio.workflows.triage.quality_join import (
    drop_levels,
    join_record,
    quality_join_parameters,
    separate_gates,
    split_by_task,
    stream_agreement,
    touches,
)
from senselab.audio.workflows.triage.task_events import GenericView
from senselab.audio.workflows.triage.vocabulary import (
    ACOUSTICALLY_EMPTY,
    KEY_STREAMS_DISAGREE,
    TaskEvidence,
    nothing_captured,
    task_event_heard,
)


def _view(level_db: np.ndarray, floor_db: float = -60.0) -> GenericView:
    """A one-band view whose broadband floor is ``floor_db``."""
    times = np.arange(len(level_db)) * 0.01
    frames = BandFrames(times, level_db[:, None], level_db)
    floor = Floor(np.asarray([floor_db]), "quiet_frames", 1.0, np.asarray([floor_db]), None)
    return GenericView(frames, floor, (), (), floor_db, 0.01)


def test_a_finding_touches_a_span_it_overlaps_or_abuts() -> None:
    """Overlap and a gap within the tolerance touch; a gap beyond it does not."""
    spans = [(1.0, 2.0)]
    assert touches((1.5, 1.6), spans, 0.1)
    assert touches((2.05, 2.5), spans, 0.1)
    assert not touches((2.3, 2.5), spans, 0.1)


def test_a_short_finding_inside_the_task_is_outside_it() -> None:
    """A fault shorter than its minimum touching the task is an annotation, not a review."""
    split = split_by_task([(1.2, 1.205), (1.3, 1.5)], [(1.0, 2.0)], 0.1, min_s=0.05)
    assert split["in"] == [[1.3, 1.5]]
    assert split["out"] == [[1.2, 1.205]]


def test_a_drop_from_a_sounding_signal_is_a_cut_and_one_from_a_decayed_signal_is_a_gate() -> None:
    """f47eeda7's phonation was still sounding when the recorder died; a cough's tail had decayed into a gate."""
    level = np.full(800, -60.0)
    level[100:300] = -25.0  # phonation sounding until the drop at 3.0 s
    level[400:420] = -25.0  # a cough at 4.0-4.2 s, decayed to the floor before the drop at 4.6 s
    view = _view(level)
    drops = drop_levels([[3.0, 3.5], [4.6, 5.2]], view, quality_join_parameters())
    cut, gated = separate_gates(drops, quality_join_parameters())
    assert cut == [(3.0, 3.5)]
    assert gated == [(4.6, 5.2)]
    assert separate_gates([[1.0, 2.0, None]], quality_join_parameters()) == ([(1.0, 2.0)], [])


def test_a_shutoff_soon_after_the_task_cut_it() -> None:
    """A shutoff within its own abut tolerance of the task's end is in the task."""
    record = join_record(
        task_spans=[(1.0, 6.0)],
        event_kind="phonation",
        faults={"shutoff": [[6.3, 8.0]], "dropout": [[9.0, 9.2]]},
        other_voice=[],
        streams=None,
        plain_active_s=5.0,
        enhanced_active_s=5.0,
        level_rel_db=0.0,
    )
    assert record["faults_in_task"] == ["shutoff"]
    assert record["faults"]["dropout"]["out"] == [[9.0, 9.2]]


def test_a_shutoff_decides_only_for_a_held_task() -> None:
    """A recorder gating after a cough is not a cut: the shutoff reviews only where phonation is held."""
    common: dict[str, Any] = {
        "task_spans": [(1.0, 6.0)],
        "faults": {"shutoff": [[5.9, 7.5]]},
        "other_voice": [],
        "streams": None,
        "plain_active_s": 5.0,
        "enhanced_active_s": 5.0,
        "level_rel_db": 0.0,
    }
    assert join_record(**common, event_kind="phonation")["faults_in_task"] == ["shutoff"]
    assert join_record(**common, event_kind="cough")["faults_in_task"] == []


def test_another_voice_inside_the_task_decides_and_one_outside_does_not() -> None:
    """Interference reviews only where it touches a task span (owner, 2026-10-07)."""
    inside = join_record(
        task_spans=[(2.0, 10.0)],
        event_kind="cough",
        faults={},
        other_voice=[(8.0, 9.0)],
        streams=None,
        plain_active_s=4.0,
        enhanced_active_s=4.0,
        level_rel_db=0.0,
    )
    outside = join_record(
        task_spans=[(2.0, 5.0)],
        event_kind="cough",
        faults={},
        other_voice=[(8.0, 9.0)],
        streams=None,
        plain_active_s=4.0,
        enhanced_active_s=4.0,
        level_rel_db=0.0,
    )
    assert inside["interference_in_task"] == ["other_voice"]
    assert outside["interference_in_task"] == []
    assert outside["interference"]["other_voice"]["out"] == [[8.0, 9.0]]


def test_events_the_enhanced_stream_loses_disagree() -> None:
    """Two raw events standing over the floor, one gone on the enhanced stream: half lost."""
    raw = np.full(400, -60.0)
    raw[50:80] = -30.0
    raw[200:230] = -30.0
    enhanced = np.full(400, -60.0)
    enhanced[50:80] = -30.0
    agreement = stream_agreement([(0.5, 0.8), (2.0, 2.3)], _view(raw), _view(enhanced), quality_join_parameters())
    assert agreement["standing_n"] == 2
    assert agreement["lost_n"] == 1
    assert agreement["disagree"] is True
    assert stream_agreement([(0.5, 0.8)], _view(raw), None, quality_join_parameters())["disagree"] is None


def test_nothing_captured_reads_the_activity_and_the_session_level() -> None:
    """No activity on either stream, or a recording far under its session, captured nothing."""
    base: dict[str, Any] = {"task_spans": [], "event_kind": None, "faults": {}, "other_voice": None, "streams": None}
    silent = join_record(**base, plain_active_s=0.0, enhanced_active_s=0.0, level_rel_db=None)
    room = join_record(**base, plain_active_s=3.0, enhanced_active_s=3.0, level_rel_db=-51.0)
    normal = join_record(**base, plain_active_s=3.0, enhanced_active_s=3.0, level_rel_db=-5.0)
    enhanced_only = join_record(**base, plain_active_s=0.0, enhanced_active_s=1.0, level_rel_db=None)
    assert nothing_captured(silent) and nothing_captured(room)
    assert not nothing_captured(normal) and not nothing_captured(enhanced_only)
    assert not nothing_captured({})


def test_a_task_event_heard_keeps_a_quiet_recording() -> None:
    """A word, a breath, a cough or phonation is a task event; the level rule needs none."""
    assert task_event_heard(TaskEvidence(), 3)
    assert task_event_heard(TaskEvidence(voice_found=True), None)
    assert not task_event_heard(TaskEvidence(), 0)


def test_the_fold_discards_a_recording_that_captured_nothing() -> None:
    """Routed, with no task event and the join reading no activity: the nothing-captured discard."""
    from senselab.audio.workflows.triage.vocabulary import (  # noqa: PLC0415
        NodeVerdict,
        Outcome,
        Triage,
        fold_file_verdict,
    )

    quality = join_record(
        task_spans=[],
        event_kind=None,
        faults={},
        other_voice=None,
        streams=None,
        plain_active_s=0.0,
        enhanced_active_s=0.0,
        level_rel_db=None,
    )
    folded = fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "admitted")],
        branch_decisions={},
        ran={},
        hint_claims={},
        route_state="unexplained",
        task=TaskEvidence(quality=quality),
    )
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == ACOUSTICALLY_EMPTY
    assert any(entry.name == "nothing_captured" and entry.decisive for entry in folded.evidence)


def test_streams_disagreeing_on_an_event_kind_that_decides_reviews() -> None:
    """Disagreement reviews only for the event kinds the parameters name."""
    params = {**quality_join_parameters(), "streams": {**quality_join_parameters()["streams"], "decides": ["cough"]}}
    base: dict[str, Any] = {
        "task_spans": [(1.0, 3.0)],
        "faults": {},
        "other_voice": [],
        "streams": {"standing_n": 2, "lost_n": 2, "lost_fraction": 1.0, "disagree": True},
        "plain_active_s": 2.0,
        "enhanced_active_s": 0.0,
        "level_rel_db": 0.0,
    }
    assert join_record(**base, event_kind="cough", p=params)["streams_disagree"] is True
    assert join_record(**base, event_kind="breath", p=params)["streams_disagree"] is False
    assert KEY_STREAMS_DISAGREE == "streams_disagree"
