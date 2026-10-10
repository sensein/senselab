"""Tests for how the fold decides a cough task on the cough-onset measure (owner labels, 2026-10-06)."""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.triage.decision import reasons_of
from senselab.audio.workflows.triage.quality_join import join_record
from senselab.audio.workflows.triage.vocabulary import (
    COUGH_COUNTED,
    COUGH_PERFORMED,
    DECLINED,
    DISCARD_GROUNDS,
    GROUND_KEYS,
    KEY_COUGH_REVIEW_LOW_CONFIDENCE,
    KEY_DISCARD_CONTESTED,
    KEY_NO_BRANCH_MEASURED,
    KEY_NO_COUGH_CAPTURED,
    KEY_OTHER_VOICE_IN_TASK,
    KEY_OWNING_BRANCH_INPUT_ABSENT,
    KEY_RESIDUAL_SPEECH_UNATTRIBUTED,
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
    Release,
    RunState,
    RunStatus,
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
    background: dict[str, Any] | None = None,
    detected: int = 0,
    contest_min: int | None = None,
    cohort: dict[str, Any] | None = None,
    llm_redaction: dict[str, Any] | None = None,
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
            events_found_n=detected,
            event_kind="cough",
            instructed_count=instructed if instructed is not None else count,
            cough_mode=mode,
            cough_onsets_n=onsets,
            cough_review=review,
            cough_reading={"onsets_n": onsets, "onsets_strict_n": onsets, "onsets_lenient_n": onsets}
            if onsets is not None
            else {},
            quality=background or {},
            contest_events_min=contest_min,
        ),
        cohort=cohort,
        llm_redaction=llm_redaction,
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


def test_too_few_coughs_annotate_a_task_mismatch() -> None:
    """22c5f400: one voluntary cough act where three were asked passes, annotated task_mismatch."""
    folded = _fold(mode=COUGH_COUNTED, onsets=1, family="voluntary-cough", instructed=3)
    assert folded.triage is Triage.PASS
    assert KEY_TASK_MISMATCH in folded.annotation_keys
    assert KEY_TASK_MISMATCH not in folded.ground_keys
    [annotation] = [each for each in folded.annotations if each.key == KEY_TASK_MISMATCH]
    assert "detected 1 coughs where 3 were instructed" in annotation.why


def test_no_onset_discards_as_no_cough_captured() -> None:
    """cdfa7e4e / a28e5022: nothing there, or a hum, discards rather than flags."""
    folded = _fold(mode=COUGH_PERFORMED, onsets=0, conformance=True)
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == NO_COUGH_CAPTURED


def test_the_detector_finding_the_instructed_count_contests_a_no_cough_discard() -> None:
    """The measure found no onset; AIRWAY's own detector found the five asked: flag for review."""
    folded = _fold(mode=COUGH_COUNTED, onsets=0, conformance=True, detected=5, contest_min=5)
    assert folded.triage is Triage.REVIEW
    assert KEY_DISCARD_CONTESTED in folded.ground_keys
    assert folded.discard_ground is None


def test_a_detector_short_of_the_threshold_leaves_the_discard() -> None:
    """Fewer detector events than the threshold contest nothing."""
    folded = _fold(mode=COUGH_COUNTED, onsets=0, conformance=True, detected=4, contest_min=5)
    assert folded.triage is Triage.DISCARD
    assert folded.discard_ground == NO_COUGH_CAPTURED


def test_a_discard_releases_nothing() -> None:
    """A discarded recording releases ``discarded``, whatever redaction would say."""
    folded = _fold(mode=COUGH_PERFORMED, onsets=0, conformance=True)
    assert folded.release is None
    assert folded.record()["release_ground_key"] == "discarded"


def test_an_absent_spectrogram_reruns() -> None:
    """No stored spectrogram: the measure could not look, so the recording is owed a rerun."""
    folded = _fold(mode=COUGH_COUNTED, onsets=None, absent=("spectrogram_narrowband",), conformance=UNDETERMINED)
    assert folded.triage is Triage.REVIEW and folded.run_status is RunStatus.INCOMPLETE
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
    assert KEY_NO_BRANCH_MEASURED not in folded.ground_keys


def test_a_low_confidence_count_is_flagged_for_review() -> None:
    """A count whose decision differs inside the review band flags for review, never discards."""
    folded = _fold(mode=COUGH_COUNTED, onsets=5, review=True)
    assert folded.triage is Triage.REVIEW
    assert KEY_COUGH_REVIEW_LOW_CONFIDENCE in folded.ground_keys


def test_the_cough_grounds_are_named() -> None:
    """The new grounds are keys, the discard is a discard ground, and neither is operational."""
    assert KEY_NO_COUGH_CAPTURED in GROUND_KEYS and KEY_COUGH_REVIEW_LOW_CONFIDENCE in GROUND_KEYS
    assert NO_COUGH_CAPTURED in DISCARD_GROUNDS
    assert not {KEY_NO_COUGH_CAPTURED, KEY_COUGH_REVIEW_LOW_CONFIDENCE} & OPERATIONAL_GROUND_KEYS


def _intercom(**kwargs: Any) -> FileVerdict:  # noqa: ANN401
    """6ca9935e: a voice the enhancer removed from inside the task, folded with whatever attributes it."""
    return _fold(
        mode=COUGH_COUNTED,
        onsets=9,
        family="voluntary-cough",
        instructed=3,
        background=join_record(
            task_spans=[(2.0, 20.0)],
            event_kind="cough",
            faults={},
            other_voice=[(16.0, 18.9)],
            streams=None,
            plain_active_s=12.0,
            enhanced_active_s=12.0,
            level_rel_db=0.0,
        ),
        **kwargs,
    )


def test_background_speech_nothing_attributes_is_an_annotation() -> None:
    """Residual speech over the task that neither COHORT nor a reader attributes is no reason."""
    folded = _intercom()
    assert folded.triage is Triage.PASS
    assert KEY_RESIDUAL_SPEECH_UNATTRIBUTED in {verdict.key for verdict in folded.annotations}
    assert KEY_OTHER_VOICE_IN_TASK not in folded.ground_keys
    assert not any(key.startswith("interference_in_task") for key in folded.ground_keys)
    matched = _intercom(cohort={"other_speaker": {"status": "measured", "nonmatch_spans": []}})
    assert matched.triage is Triage.PASS and KEY_OTHER_VOICE_IN_TASK not in matched.ground_keys


def test_background_speech_cohort_attributes_flags_another_speaker() -> None:
    """A non-matching enrollment span over the voice makes it another speaker's: review, never discard."""
    folded = _intercom(cohort={"other_speaker": {"status": "measured", "nonmatch_spans": [[16.5, 17.0]]}})
    assert folded.triage is Triage.REVIEW
    assert KEY_OTHER_VOICE_IN_TASK in folded.ground_keys
    assert reasons_of(folded.ground_keys)[0] == "other_speaker"


def test_background_speech_a_reader_attributes_flags_another_speaker() -> None:
    """The reviewer reading an assistant speak attributes the residual's voice."""
    folded = _intercom(llm_redaction={"status": "clean", "other_speaker": "assistant"})
    assert folded.triage is Triage.REVIEW
    assert KEY_OTHER_VOICE_IN_TASK in folded.ground_keys
