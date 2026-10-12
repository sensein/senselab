"""The r37 fold amendments: the speaker gate's downgrade, the second opinion on the participant's own speech,
the narrative off-task aside, the identifier test, the corroborated "not free" hold and speech before the task.

Design: ``specs/20261012-r37-words-speaker/design.md``.
"""

from __future__ import annotations

from typing import Any, Mapping

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.decision import ANNOTATION, REVIEW
from senselab.audio.workflows.triage.vocabulary import (
    DECLINED,
    DOMINANT_SPEAKER_GATE,
    KEY_OFF_TASK_ASIDE,
    KEY_REVIEWER_OFF_TASK,
    KEY_SPEAKER_GATE_RECORDING_CONDITION,
    KEY_SPEECH_BEFORE_TASK,
    ROUTED,
    SPEECH_BEFORE_TASK_ITEM,
    BranchDecision,
    FileVerdict,
    FoldPolicy,
    NodeVerdict,
    Outcome,
    RedactionEvidence,
    Release,
    TaskEvidence,
    Triage,
    fold_file_verdict,
    second_voice_signals,
)

_READ = {"status": "clean", "original": "clean", "speakers": "one", "proposal": []}
_FIRED = {
    "gate": DOMINANT_SPEAKER_GATE,
    "passed": False,
    "ground": "another speaker holds part of the task extent",
    "reading": "extent_dominant_exclusive_share",
    "value": 0.7,
    "bound": 0.9,
    "op": "at_least",
}
_HOT = {"rms_dbfs_median": -13.8, "clipped_fraction": 0.87}
_COOL = {"rms_dbfs_median": -28.0, "clipped_fraction": 0.0}


def _cohort(*, target: bool | None = True, condition: Mapping[str, Any] | None = None) -> dict[str, Any]:
    pool: dict[str, Any] = {"status": "not_applicable"}
    if target is not None:
        pool = {"status": "measured", "spans": [[1.0, 6.0]], "pooled_s": 5.0, "cosine": 0.8, "match": target}
        if not target:
            pool["cosine"] = 0.1
    return {
        "other_speaker": {"status": "measured", "target": pool, "non_task": {"status": "not_applicable"}},
        "condition": {"participant": dict(condition)} if condition else {},
    }


def _fold(
    *,
    annotation: Mapping[str, Any] | None = None,
    opinion: Mapping[str, Any] | None = None,
    cohort: Mapping[str, Any] | None = None,
    task: TaskEvidence | None = None,
    redaction: RedactionEvidence | None = None,
    flag_gates: list[dict[str, Any]] | None = None,
    family: str = "free-speech",
    redacted: bool = False,
) -> FileVerdict:
    decisions = {
        name: BranchDecision(name, False, route, False)
        for name, route in (("AIRWAY", DECLINED), ("SPEECH", ROUTED), ("VOICE", DECLINED))
    }
    verdicts = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]
    if redacted:
        verdicts.append(NodeVerdict("REDACT", Outcome.PASS, None, "every finding redacted"))
    return fold_file_verdict(
        verdicts,
        branch_decisions=decisions,
        ran={},
        hint_claims={},
        route_state=ROUTED,
        declared_family=family,
        redaction=redaction or RedactionEvidence(lexical_words_n=40, scanned=True),
        llm_redaction=dict(annotation if annotation is not None else _READ),
        second_opinion=dict(opinion) if opinion is not None else None,
        cohort=cohort,
        task=task or TaskEvidence(owning_branches=("SPEECH",)),
        flag_gates=flag_gates,
        policy=FoldPolicy.from_config(load_triage_config()),
    )


def _gate_items(folded: FileVerdict) -> list[Any]:
    return [entry for entry in folded.evidence if entry.name == f"gate:{DOMINANT_SPEAKER_GATE}"]


class TestTheSpeakerGateDowngradesOnlyUnderAHotParticipant:
    """C1: an uncorroborated exclusive-time fire stays in review; the hot or clipped participant annotates it."""

    def test_an_uncorroborated_fire_on_a_hot_participant_is_an_annotation(self) -> None:
        """SUBJ08: hot and clipped across the participant, nothing else reads a second voice."""
        folded = _fold(flag_gates=[dict(_FIRED)], cohort=_cohort(condition=_HOT))
        assert f"gate:{DOMINANT_SPEAKER_GATE}" not in folded.ground_keys
        assert KEY_SPEAKER_GATE_RECORDING_CONDITION in folded.annotation_keys
        assert folded.triage is Triage.PASS
        assert [entry.effect for entry in _gate_items(folded)] == [ANNOTATION]

    def test_an_uncorroborated_fire_elsewhere_stays_in_review(self) -> None:
        """168c5e68: a real second voice nothing corroborates is not lost."""
        for condition in (_COOL, {}):
            folded = _fold(flag_gates=[dict(_FIRED)], cohort=_cohort(condition=condition))
            assert f"gate:{DOMINANT_SPEAKER_GATE}" in folded.ground_keys and folded.triage is Triage.REVIEW
            assert folded.reason == "other_speaker"
            assert [entry.effect for entry in _gate_items(folded)] == [REVIEW]

    def test_a_corroborated_fire_on_a_hot_participant_stays_in_review(self) -> None:
        """Another reading of a second voice keeps the gate a flag whatever the condition."""
        corroborations: list[dict[str, Any]] = [
            {"cohort": _cohort(target=False, condition=_HOT)},
            {"cohort": _cohort(condition=_HOT), "task": TaskEvidence(owning_branches=("SPEECH",), separated_speakers_n=2)},
            {"cohort": _cohort(condition=_HOT), "annotation": {**_READ, "speakers": "more_than_one"}},
            {
                "cohort": _cohort(target=None, condition=_HOT),
                "opinion": {"status": "ok", "probabilities": {"other_voice": 0.95}},
            },
        ]
        for extra in corroborations:
            folded = _fold(flag_gates=[dict(_FIRED)], **extra)
            assert KEY_SPEAKER_GATE_RECORDING_CONDITION not in folded.annotation_keys, extra
            assert folded.triage is Triage.REVIEW

    def test_the_signals_are_named(self) -> None:
        """Each reading that places a second voice is named once, in order."""
        assert second_voice_signals(
            reviewer=True, second_opinion=True, cohort=_cohort(target=False), separated_speakers_n=2
        ) == ["reviewer", "second_opinion", "cohort", "separation"]
        assert second_voice_signals(reviewer=False, second_opinion=False, cohort=None, separated_speakers_n=1) == []


class TestTheSecondOpinionOnTheParticipantsOwnSpeech:
    """C2 (07929b7b productive vocabulary): COHORT reads the participant, so the opinion's other voice is off task."""

    _OPINION = {"status": "ok", "probabilities": {"other_voice": 0.84, "policy_identifier_present": 0.5}}

    def test_a_matched_participant_turns_other_voice_into_off_task_speech(self) -> None:
        """The reason is off_task_speech; nothing names another speaker."""
        folded = _fold(opinion=self._OPINION, cohort=_cohort(target=True), family="productive-vocabulary")
        assert "second_opinion:other_voice_participant" in folded.ground_keys
        assert "second_opinion:other_voice" not in folded.ground_keys
        assert folded.reason == "off_task_speech" and "other_speaker" not in folded.reason_keys

    def test_without_a_match_it_stays_another_speaker(self) -> None:
        """No pool, or a pool that does not match: the opinion's other voice is other_speaker."""
        for cohort in (None, _cohort(target=None), _cohort(target=False)):
            folded = _fold(opinion=self._OPINION, cohort=cohort, family="productive-vocabulary")
            assert "second_opinion:other_voice" in folded.ground_keys
            assert folded.reason == "other_speaker"


class TestTheNarrativeOffTaskAside:
    """G1 (07929b7b cinderella): in a free-response family an off-task aside annotates."""

    def test_some_in_a_narrative_family_is_an_annotation(self) -> None:
        """One quote of "some" off-task speech: annotation, no flag."""
        task = TaskEvidence(owning_branches=("SPEECH",), narrative=True)
        folded = _fold(annotation={**_READ, "off_task_speech": "some", "off_task_quotes": ["x"]}, task=task)
        assert KEY_OFF_TASK_ASIDE in folded.annotation_keys and KEY_REVIEWER_OFF_TASK not in folded.ground_keys
        assert folded.triage is Triage.PASS

    def test_extensive_or_an_ordered_task_still_flags(self) -> None:
        """Masking and the flag stay for extensive off-task speech and for item-set and ordered tasks."""
        narrative = TaskEvidence(owning_branches=("SPEECH",), narrative=True)
        extensive = _fold(annotation={**_READ, "off_task_speech": "extensive"}, task=narrative)
        assert KEY_REVIEWER_OFF_TASK in extensive.ground_keys
        ordered = _fold(annotation={**_READ, "off_task_speech": "some"}, family="harvard-sentences-list")
        assert KEY_REVIEWER_OFF_TASK in ordered.ground_keys

    def test_a_non_task_speech_mask_is_not_an_identifier(self) -> None:
        """Only a mask over identifying content is the reviewer's "identifier present"."""
        opinion = {"status": "ok", "probabilities": {"policy_identifier_present": 0.07}}
        speech_only = RedactionEvidence(lexical_words_n=40, scanned=True, masks_n=1, masks_final_n=1)
        folded = _fold(opinion=opinion, redaction=speech_only)
        assert "second_opinion:policy_identifier_present" not in folded.ground_keys
        identifier = RedactionEvidence(
            lexical_words_n=40, scanned=True, findings_n=1, masks_n=1, masks_final_n=1, pii_masks_final_n=1
        )
        assert "second_opinion:policy_identifier_present" in _fold(opinion=opinion, redaction=identifier).ground_keys


class TestTheSecondOpinionsNotFreeHoldsOnlyWhenCorroborated:
    """G2 (fac74f45): the second opinion alone reviews and keeps the release; corroborated, it withholds."""

    _OPINION = {"status": "ok", "probabilities": {"masked_text_free_of_identifiers": 0.15}}

    def test_alone_it_reviews_and_keeps_the_release(self) -> None:
        """No reviewer proposal and no identifier mask: review, release as is."""
        folded = _fold(opinion=self._OPINION)
        assert "second_opinion:masked_text_free_of_identifiers" in folded.ground_keys
        assert folded.triage is Triage.REVIEW and folded.release is Release.AS_IS

    def test_corroborated_it_withholds(self) -> None:
        """A reviewer proposal or a standing identifier mask corroborates it."""
        for evidence in (
            RedactionEvidence(lexical_words_n=40, scanned=True, reviewer_requested_n=1),
            RedactionEvidence(
                lexical_words_n=40, scanned=True, findings_n=1, masks_n=1, masks_final_n=1, pii_masks_final_n=1
            ),
        ):
            folded = _fold(opinion=self._OPINION, redaction=evidence, redacted=True)
            assert folded.release is Release.WITHHELD, evidence


class TestSpeechBeforeTheTask:
    """G3 (168c5e68): lexical speech before a lexical task's extent is an annotation carrying its duration."""

    def test_it_annotates_with_its_duration(self) -> None:
        """Two short utterances before the item hull: an annotation and an evidence item, no flag."""
        before = {"words_n": 8, "start_s": 0.1, "end_s": 1.84, "seconds": 1.74}
        folded = _fold(task=TaskEvidence(owning_branches=("SPEECH",), speech_before_task=before))
        assert KEY_SPEECH_BEFORE_TASK in folded.annotation_keys and folded.triage is Triage.PASS
        items = [entry for entry in folded.evidence if entry.name == SPEECH_BEFORE_TASK_ITEM]
        assert [(entry.value, entry.effect) for entry in items] == [(1.74, ANNOTATION)]

    def test_nothing_before_the_task_writes_nothing(self) -> None:
        """No words before the extent: no annotation, no item."""
        folded = _fold()
        assert KEY_SPEECH_BEFORE_TASK not in folded.annotation_keys
        assert not [entry for entry in folded.evidence if entry.name == SPEECH_BEFORE_TASK_ITEM]


def _item(folded: FileVerdict, name: str) -> Any:  # noqa: ANN401 -- an evidence item
    [found] = [entry for entry in folded.evidence if entry.name == name]
    return found


class TestEveryOtherSpeakerItemSaysWhoReadItAndWhere:
    """C3: an other_speaker evidence item names its source, model, spans and value against its bound."""

    def test_the_speaker_gate_names_the_diarizer_and_the_exclusive_spans(self) -> None:
        """017b7678: "perhaps need to know where other speaker was found or how?"."""
        fired = {**_FIRED, "model": "pyannote/speaker-diarization-3.1", "spans": [[4.0, 6.0]]}
        [gate] = _gate_items(_fold(flag_gates=[fired]))
        assert (gate.source, gate.model, gate.spans) == ("diarizer", "pyannote/speaker-diarization-3.1", ((4.0, 6.0),))
        assert (gate.value, gate.comparison, gate.threshold) == (0.7, ">=", 0.9)

    def test_the_second_opinion_names_its_probability_against_its_bound(self) -> None:
        """A bare flag becomes p against confident_yes, with the model that answered."""
        opinion = {"status": "ok", "model_id": "ollama:clef:27b", "probabilities": {"other_voice": 0.84}}
        entry = _item(_fold(opinion=opinion), "second_opinion:other_voice")
        assert (entry.source, entry.model, entry.value, entry.comparison, entry.threshold) == (
            "second_opinion",
            "ollama:clef:27b",
            0.84,
            ">=",
            0.8,
        )

    def test_the_reviewer_and_cohort_name_their_spans(self) -> None:
        """The reviewer's placed quotes, and COHORT's non-matching pool."""
        task = TaskEvidence(owning_branches=("SPEECH",), reviewer_other_spans=((1.0, 2.5),))
        folded = _fold(annotation={**_READ, "speakers": "more_than_one", "model_id": "gemma"}, task=task)
        entry = _item(folded, "reviewer_second_speaker")
        assert (entry.source, entry.model, entry.spans) == ("reviewer", "gemma", ((1.0, 2.5),))
        cohort = _cohort(target=False)
        cohort["other_speaker"]["model_id"] = "ecapa"
        cohort["other_speaker"]["target"]["cosine_min"] = 0.398
        folded = _fold(cohort=cohort)
        pool = _item(folded, "cohort.target_cosine")
        assert (pool.source, pool.model, pool.spans, pool.value, pool.threshold) == (
            "cohort",
            "ecapa",
            ((1.0, 6.0),),
            0.1,
            0.398,
        )
        session = _item(folded, "other_speaker_in_session")
        assert session.source == "cohort" and session.spans == ((1.0, 6.0),)

    def test_the_record_carries_them(self) -> None:
        """The verdict record, and so the tables, carry who read the item and where."""
        fired = {**_FIRED, "model": "pyannote", "spans": [[4.0, 6.0]]}
        [row] = [r for r in _fold(flag_gates=[fired]).record()["evidence"] if r["name"].startswith("gate:")]
        assert (row["source"], row["model"], row["spans"]) == ("diarizer", "pyannote", [[4.0, 6.0]])
