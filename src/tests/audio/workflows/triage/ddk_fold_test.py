"""How the fold decides a syllable-repetition task on SPEECH's task-layer reading."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.triage.decision import reasons_of
from senselab.audio.workflows.triage.vocabulary import (
    DECLINED,
    DISCARD_GROUNDS,
    GROUND_KEYS,
    KEY_DDK_REVIEW_IDENTITY,
    KEY_DDK_REVIEW_WEAK_EVENTS,
    KEY_NO_LEXICAL_ITEM,
    KEY_OWNING_BRANCH_INPUT_ABSENT,
    KEY_TASK_MISMATCH,
    NO_SYLLABLE_TRAIN_CAPTURED,
    ROUTED,
    SYLLABLE_TRAIN_NOT_TARGET,
    TASK,
    UNDETERMINED,
    BranchDecision,
    BranchReport,
    FileVerdict,
    NodeVerdict,
    Outcome,
    RedactionEvidence,
    RunState,
    RunStatus,
    TaskEvidence,
    Triage,
    fold_file_verdict,
)

_ADMIT_OK = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]


def _reading(
    decision: str | None,
    *,
    why: str = "clear",
    standing: int = 6,
    clear: int = 6,
    identity: float | None = 0.7,
    events: int = 6,
    required: int | None = None,
) -> dict[str, Any]:
    """The task layer's record as VERDICT reads it."""
    return {
        "mode": "cycle",
        "decision": decision,
        "why": why,
        "unit": "cycle",
        "events_n": events,
        "identity": identity,
        "inputs": {
            "floor_db": 40.0 if standing else 4.0,
            "standing_n": standing,
            "clear_n": clear,
            "clear_free_n": clear,
            "events_min": 2,
            "identity": identity,
            "identity_min": 0.3,
            "identity_absent_max": 0.15,
            "snr_low_db": 10.0,
            "snr_high_db": 16.0,
        },
        "annotations": {
            "events_n": events,
            "required_count": required,
            "syllable_rate_hz": 5.0,
            "cycle_rate_hz": 1.6,
            "period_cv": 0.1,
            "period_trend_s_per_step": 0.0,
            "train_fraction": 0.8,
            "realised_mass": [0.8] * 6,
        },
    }


def _fold(
    decision: str | None,
    *,
    absent: tuple[str, ...] = (),
    speech_route: str = ROUTED,
    lexical: int = 3,
    **reading: Any,  # noqa: ANN401
) -> FileVerdict:
    """Fold one declared pataka recording on a reading."""
    return fold_file_verdict(
        _ADMIT_OK,
        branch_reports=[BranchReport(node="SPEECH", kind="speech", conformance=UNDETERMINED, conformance_of=TASK)],
        spans_by_node={"SPEECH": 1},
        branch_decisions={
            "SPEECH": BranchDecision(
                branch="SPEECH",
                will_run=True,
                route_state=speech_route,
                forced_by_declaration=speech_route != ROUTED,
                declared=True,
            ),
            "AIRWAY": BranchDecision(
                branch="AIRWAY", will_run=False, route_state=DECLINED, forced_by_declaration=False
            ),
        },
        ran={"SPEECH": RunState.COMPLETED},
        hint_claims={"SPEECH": True},
        route_state="routed",
        declared_family="diadochokinesis-pataka",
        redaction=RedactionEvidence(lexical_words_n=lexical, scanned=True),
        task=TaskEvidence(
            owning_branches=("SPEECH",),
            duration_s=8.0,
            minimum_duration_s=0.5,
            owner_absent_inputs=absent,
            required_event="syllable",
            ddk_mode="cycle",
            ddk_decision=decision,
            ddk_reading=_reading(decision, **reading) if decision is not None else {},
        ),
    )


def test_a_present_train_passes_and_its_items_decide() -> None:
    """Clear events and the target's identity pass; the task layer's items are the decisive ones."""
    folded = _fold("present")
    assert folded.triage is Triage.PASS
    decisive = {item.name for item in folded.evidence if item.decisive}
    assert {"ddk_event_db_over_floor", "ddk_identity", "ddk_clear_events"} <= decisive


def test_no_train_over_the_floor_discards_as_no_task_captured() -> None:
    """Nothing standing: discarded on its own ground, read as no task captured."""
    folded = _fold("absent", why="no syllable train over the floor", standing=0, clear=0, identity=None, events=0)
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == NO_SYLLABLE_TRAIN_CAPTURED
    assert folded.reason == "no_task_captured"
    assert [item.name for item in folded.evidence if item.decisive] == ["ddk_event_db_over_floor"]


def test_a_train_of_another_activity_discards_as_task_not_found() -> None:
    """A train stands but its identity reads another activity."""
    folded = _fold("absent", why="not the target", identity=0.05)
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == SYLLABLE_TRAIN_NOT_TARGET
    assert folded.reason == "task_not_found"
    assert [item.name for item in folded.evidence if item.decisive] == ["ddk_identity"]


def test_an_ambiguous_identity_reviews_as_task_not_conforming() -> None:
    """The train is present but which syllables it holds is ambiguous."""
    folded = _fold("review", why="identity", identity=0.22)
    assert folded.triage is Triage.REVIEW
    assert KEY_DDK_REVIEW_IDENTITY in folded.ground_keys
    assert folded.reason == "task_not_conforming"


def test_weak_events_review_as_weak_events() -> None:
    """Too few events clear their local background."""
    folded = _fold("review", why="weak", clear=1)
    assert folded.triage is Triage.REVIEW
    assert KEY_DDK_REVIEW_WEAK_EVENTS in folded.ground_keys
    assert folded.reason == "weak_events"


def test_a_short_count_annotates_and_never_decides() -> None:
    """Four cycles of the ten asked pass, annotated as a task mismatch."""
    folded = _fold("present", events=4, required=10)
    assert folded.triage is Triage.PASS
    assert KEY_TASK_MISMATCH in folded.annotation_keys
    [item] = [item for item in folded.evidence if item.name == "ddk_events_against_instructed"]
    assert not item.decisive


def test_asr_finding_no_word_is_an_annotation() -> None:
    """The recogniser is an annotation on a syllable task: no lexical word does not review it."""
    folded = _fold("present", lexical=0)
    assert folded.triage is Triage.PASS
    assert KEY_NO_LEXICAL_ITEM in folded.annotation_keys
    assert KEY_NO_LEXICAL_ITEM not in folded.ground_keys


def test_routing_that_declined_speech_does_not_overrule_the_reading() -> None:
    """A routing mismatch is an annotation where the task layer decides."""
    folded = _fold("present", speech_route=DECLINED)
    assert folded.triage is Triage.PASS
    assert "route_mismatch:SPEECH" in folded.annotation_keys


def test_an_absent_input_is_not_measured() -> None:
    """No background view: the reading could not be taken, so the recording is owed a rerun."""
    folded = _fold(None, absent=("SPEECH:background_model",))
    assert folded.triage is Triage.REVIEW and folded.run_status is RunStatus.INCOMPLETE
    assert KEY_OWNING_BRANCH_INPUT_ABSENT in folded.ground_keys


def test_the_new_grounds_are_declared_and_mapped() -> None:
    """Every ground the fold writes is in the vocabulary and maps to a reason."""
    for key in (NO_SYLLABLE_TRAIN_CAPTURED, SYLLABLE_TRAIN_NOT_TARGET):
        assert key in DISCARD_GROUNDS
    for key in (
        NO_SYLLABLE_TRAIN_CAPTURED,
        SYLLABLE_TRAIN_NOT_TARGET,
        KEY_DDK_REVIEW_IDENTITY,
        KEY_DDK_REVIEW_WEAK_EVENTS,
    ):
        assert key in GROUND_KEYS
        assert reasons_of([key], None)
