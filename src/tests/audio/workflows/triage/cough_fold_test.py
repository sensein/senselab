"""Tests for how the fold decides a cough task on the cough-onset measure (owner labels, 2026-10-06)."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.triage.vocabulary import (
    COUGH_COUNTED,
    COUGH_PERFORMED,
    DECLINED,
    DISCARD_GROUNDS,
    GROUND_KEYS,
    KEY_COUGH_REVIEW_LOW_CONFIDENCE,
    KEY_NO_COUGH_CAPTURED,
    KEY_OWNING_BRANCH_INPUT_ABSENT,
    KEY_ROUTE_UNEXPLAINED,
    KEY_TASK_MISMATCH,
    NO_COUGH_CAPTURED,
    OPERATIONAL_GROUND_KEYS,
    ROUTED,
    TASK,
    UNDETERMINED,
    BranchDecision,
    BranchReport,
    FileVerdict,
    NodeVerdict,
    Outcome,
    RunState,
    TaskEvidence,
    Triage,
    fold_file_verdict,
)


def _report(node: str, kind: str, *, conformance: Any) -> BranchReport:  # noqa: ANN401
    return BranchReport(node=node, kind=kind, conformance=conformance, conformance_of=TASK)


def _decisions(forced: tuple[str, ...] = (), **routes: str) -> dict[str, BranchDecision]:
    return {
        branch: BranchDecision(
            branch=branch,
            will_run=state == ROUTED or branch in forced,
            route_state=state,
            forced_by_declaration=branch in forced,
            declared=branch in forced,
        )
        for branch, state in routes.items()
    }


_ADMIT_OK = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]
_FAMILIES = {
    COUGH_COUNTED: ("respiration-and-cough-cough", 5),
    COUGH_PERFORMED: ("respiration-and-cough-v2-hardcough", None),
}


def _fold(
    *,
    mode: str,
    onsets: int | None,
    conformance: Any = False,  # noqa: ANN401
    absent: tuple[str, ...] = (),
    review: bool = False,
    airway_route: str = ROUTED,
    route_state: str = "routed",
    family: str | None = None,
    instructed: int | None = None,
) -> FileVerdict:
    declared, count = _FAMILIES[mode]
    return fold_file_verdict(
        _ADMIT_OK,
        branch_reports=[_report("AIRWAY", "airway", conformance=conformance)],
        spans_by_node={"AIRWAY": 1},
        branch_decisions=_decisions(forced=("AIRWAY",), AIRWAY=airway_route, SPEECH=DECLINED, VOICE=DECLINED),
        ran={"AIRWAY": RunState.COMPLETED},
        hint_claims={"AIRWAY": True},
        route_state=route_state,
        declared_family=family or declared,
        task=TaskEvidence(
            owning_branches=("AIRWAY",),
            duration_s=6.0,
            minimum_duration_s=0.5,
            owner_absent_inputs=absent,
            required_event="cough",
            events_found_n=0,
            event_kind="cough",
            instructed_count=instructed if instructed is not None else count,
            cough_mode=mode,
            cough_onsets_n=onsets,
            cough_review=review,
            cough_reading={"onsets_n": onsets, "onsets_strict_n": onsets, "onsets_lenient_n": onsets}
            if onsets is not None
            else {},
        ),
    )


def test_the_instructed_count_passes_whatever_airway_found() -> None:
    """53dddba6 / c60d8bb8: five onsets for five asked pass, though AIRWAY's detector found none."""
    folded = _fold(mode=COUGH_COUNTED, onsets=5)
    assert folded.triage is Triage.PASS
    assert "conformance:AIRWAY" not in folded.ground_keys


def test_one_onset_performs_a_hard_cough() -> None:
    """b91cd93f / 889afc08: a single coherent onset performs v2-hardcough."""
    folded = _fold(mode=COUGH_PERFORMED, onsets=1)
    assert folded.triage is Triage.PASS


def test_too_few_coughs_flag_a_task_mismatch() -> None:
    """22c5f400: one voluntary cough act where three were asked flags task_mismatch."""
    folded = _fold(mode=COUGH_COUNTED, onsets=1, family="voluntary-cough", instructed=3)
    assert folded.triage is Triage.FLAG
    [reason] = [reason for reason in folded.reasons if reason.key == KEY_TASK_MISMATCH]
    assert "detected 1 coughs where 3 were instructed" in reason.why


def test_no_onset_discards_as_no_cough_captured() -> None:
    """cdfa7e4e / a28e5022: nothing there, or a hum, discards rather than flags."""
    folded = _fold(mode=COUGH_PERFORMED, onsets=0, conformance=True)
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == NO_COUGH_CAPTURED


def test_an_absent_spectrogram_reruns() -> None:
    """No stored spectrogram: the measure could not look, so the recording is owed a rerun."""
    folded = _fold(mode=COUGH_COUNTED, onsets=None, absent=("spectrogram_narrowband",), conformance=UNDETERMINED)
    assert folded.triage is Triage.RERUN
    assert KEY_OWNING_BRANCH_INPUT_ABSENT in folded.ground_keys


def test_a_found_cough_is_no_route_mismatch() -> None:
    """4164d528: routing declined AIRWAY and the measure found coughs; the found cough does not flag."""
    folded = _fold(mode=COUGH_PERFORMED, onsets=2, airway_route=DECLINED)
    assert folded.triage is Triage.PASS
    assert not any(key.startswith("route_mismatch") for key in folded.ground_keys)


def test_a_found_cough_explains_the_route() -> None:
    """45f1c7ec: no branch routed, but the declared task's coughs explain the content, so no rerun."""
    folded = _fold(mode=COUGH_COUNTED, onsets=5, airway_route=DECLINED, route_state="unexplained")
    assert folded.triage is Triage.PASS
    assert KEY_ROUTE_UNEXPLAINED not in folded.ground_keys


def test_a_low_confidence_count_is_flagged_for_review() -> None:
    """A count whose decision differs inside the review band flags for review, never discards."""
    folded = _fold(mode=COUGH_COUNTED, onsets=5, review=True)
    assert folded.triage is Triage.FLAG
    assert KEY_COUGH_REVIEW_LOW_CONFIDENCE in folded.ground_keys


def test_the_cough_grounds_are_named() -> None:
    """The new grounds are keys, the discard is a discard ground, and neither is operational."""
    assert KEY_NO_COUGH_CAPTURED in GROUND_KEYS and KEY_COUGH_REVIEW_LOW_CONFIDENCE in GROUND_KEYS
    assert NO_COUGH_CAPTURED in DISCARD_GROUNDS
    assert not {KEY_NO_COUGH_CAPTURED, KEY_COUGH_REVIEW_LOW_CONFIDENCE} & OPERATIONAL_GROUND_KEYS
