"""Speech outside the task (task_speech.py): what is read, what the fold reviews and what the mask plan hides."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.common import consensus_words, software_agent, write_measurement
from senselab.audio.workflows.triage.nodes.redact import NON_TASK_SPEECH, mask_plan
from senselab.audio.workflows.triage.task_content import task_content_ids
from senselab.audio.workflows.triage.task_speech import (
    TASK_SPEECH_READING,
    phrase_share_bound,
    task_speech_of,
)
from senselab.audio.workflows.triage.vocabulary import (
    DECLINED,
    NON_TASK_SPEECH_EXTENSIVE,
    NON_TASK_SPEECH_MASKED,
    NON_TASK_SPEECH_UNTIMED,
    ROUTED,
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
)
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes
from tests.audio.workflows.triage.nodes.redact_test import _seed_redact_store, _word_extent

BREATH_STEM = "sub-x_ses-y_task-respiration-and-cough-v2-threebreathsnose"
VOWEL_STEM = "sub-x_ses-y_task-prolonged-vowel"
ITEMS_STEM = "sub-x_ses-y_task-random-item-generation"
QWEN = {"asr_crisperwhisper": "Hey", "asr_qwen": "Hay"}


def _breath_reading(store: ProvStore, events: Sequence[tuple[float, float]]) -> None:
    activity = store.activity(node="AIRWAY", step="align", parameters={})
    evidence = {"decision": "present", "events": [{"start_s": a, "end_s": b} for a, b in events]}
    write_measurement(
        store,
        activity,
        software_agent(store),
        name="airway_breath_reading",
        signal="plain",
        attributes={"decision": "present", "reading": {"evidence": evidence}},
    )


def _write(store: ProvStore, family: str, **kwargs: object) -> None:
    reading = task_speech_of(store, family, **kwargs)  # type: ignore[arg-type]
    activity = store.activity(node="AIRWAY", step="speech", parameters={"family": family})
    write_measurement(
        store, activity, software_agent(store), name=TASK_SPEECH_READING, signal="plain", attributes=reading.record()
    )


class TestSpeechInANonLexicalTask:
    """Owner, 2026-10-10: any lexical word in a non-lexical recording that is not task content is speech."""

    def test_a_talker_outside_a_wrong_extent_is_read_over_the_whole_file(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Words far from the breaths, which the extent never reached, are all speech outside the task."""
        _seed_redact_store(
            store, tmp_path, words=["[breath]", "[breath]", "so", "where", "did", "you"], recording_stem=BREATH_STEM
        )
        _breath_reading(store, [_word_extent(0), _word_extent(1)])
        read = task_speech_of(store, "respiration-and-cough-v2-threebreathsnose")
        assert read.words_n == 4 and read.lexical_n == 4
        assert len(read.runs) == 1 and read.runs[0][2] == 4

    def test_a_misreading_alone_on_a_breath_is_task_content_and_an_agreed_run_on_it_is_not(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The recognisers' disagreement on one word on one event is the task misread; agreed words are speech."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["Hey", "[breath]", "what", "time", "is"],
            readings={0: QWEN},
            recording_stem=BREATH_STEM,
        )
        _breath_reading(store, [_word_extent(0), (_word_extent(2)[0], _word_extent(4)[1])])
        read = task_speech_of(store, "respiration-and-cough-v2-threebreathsnose")
        assert len(read.task_content_ids) == 1 and read.words_n == 3

    def test_the_prolonged_vowel_count_in_is_task_text(self, store: ProvStore, tmp_path: Path) -> None:
        """The count-in before the vowel is the voice family's declared text; anything else is speech."""
        _seed_redact_store(store, tmp_path, words=["1", "two", "three", "aaaah", "okay"], recording_stem=VOWEL_STEM)
        read = task_speech_of(store, "prolonged-vowel")
        assert len(read.task_text_ids) == 3 and read.words_n == 1

    def test_a_word_with_no_usable_timing_is_untimed(self, store: ProvStore) -> None:
        """A consensus word at (0, 0) cannot be masked."""
        store.entity(
            prov_type="stream",
            extent=(0.0, 10.0),
            attributes={"name": "recording", "path": f"streams/{BREATH_STEM}.wav"},
        )
        store.entity(
            prov_type="word",
            extent=(0.0, 0.0),
            attributes=word_attributes("hello", (0.0, 0.0), index=0, timings={"asr_qwen": (0.0, 0.0)}),
        )
        read = task_speech_of(store, "respiration-and-cough-v2-threebreathsnose")
        assert read.words_n == 1 and len(read.untimed_ids) == 1 and read.runs == ()

    def test_a_lexical_family_reads_nothing(self, store: ProvStore, tmp_path: Path) -> None:
        """A free-speech recording's words are its task."""
        _seed_redact_store(store, tmp_path, words=["hello", "there"], recording_stem="sub-x_ses-y_task-free-speech")
        assert task_speech_of(store, "free-speech").words_n == 0


class TestTaskContentIsAMisreadingOnly:
    """Owner, 2026-10-10: task content is a recogniser's misreading of the task's sound, never agreed words."""

    def test_a_syllable_family_keeps_any_word_on_its_events(self, store: ProvStore, tmp_path: Path) -> None:
        """A word alone on a syllable train is the train misread, as before."""
        _seed_redact_store(store, tmp_path, words=["pat", "cake"], recording_stem="plain")
        words = consensus_words(store)
        events = [(_word_extent(0)[0], _word_extent(1)[1])]
        assert len(task_content_ids(words, events, family="diadochokinesis-pa")) == 0
        assert len(task_content_ids(words[:1], events, family="diadochokinesis-pa")) == 1

    def test_two_words_on_one_breath_are_not_one_to_one(self, store: ProvStore, tmp_path: Path) -> None:
        """Two disputed words on one event are speech, not one misreading."""
        _seed_redact_store(store, tmp_path, words=["Hey", "Hey"], readings={0: QWEN, 1: QWEN}, recording_stem="plain")
        words = consensus_words(store)
        one_event = [(_word_extent(0)[0], _word_extent(1)[1])]
        two_events = [_word_extent(0), _word_extent(1)]
        family = "respiration-and-cough-v2-threebreathsnose"
        assert task_content_ids(words, one_event, family=family) == set()
        assert len(task_content_ids(words, two_events, family=family)) == 2


class TestSpeechOutsideTheTaskIsMasked:
    """The fold's mask plan hides each run of speech outside the task, whatever the content-word trim says."""

    def test_each_run_is_one_mask_and_no_word_is_trimmed(self, store: ProvStore, tmp_path: Path) -> None:
        """Function words of a talker's sentence stay masked; the breath tokens between runs stay audible."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["[breath]", "is", "it", "on", "[breath]", "yes"],
            recording_stem=BREATH_STEM,
            scanned=False,
        )
        _write(store, "respiration-and-cough-v2-threebreathsnose")
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50)
        speech = [mask for mask in plan.masks if mask.source == NON_TASK_SPEECH]
        assert [len(mask.words) for mask in speech] == [3, 1]
        assert all(word.state == "masked" for mask in speech for word in mask.words)
        assert plan.non_task_speech_masked_n == 4


def _fold(task: TaskEvidence, redaction: RedactionEvidence, *, airway: str = ROUTED) -> FileVerdict:
    decisions = {
        name: BranchDecision(name, False, route, False)
        for name, route in (("AIRWAY", airway), ("SPEECH", DECLINED), ("VOICE", DECLINED))
    }
    return fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
        branch_decisions=decisions,
        ran={},
        hint_claims={},
        route_state=ROUTED,
        redaction=redaction,
        task=task,
    )


class TestTheFoldReviewsAndReleases:
    """VERDICT: speech outside the task reviews; its masks release a redacted copy; an untimed word withholds."""

    _TASK = TaskEvidence(
        owning_branches=("AIRWAY",),
        task_speech={"words_n": 2, "runs_n": 1, "untimed_n": 0},
        task_speech_words_min=1,
    )

    def test_speech_reviews_and_its_masks_release_a_redacted_copy(self) -> None:
        """K = 1: two words are review; the masks make the release redacted, itself review in an airway task."""
        folded = _fold(self._TASK, RedactionEvidence(non_task_speech_masked_n=2))
        assert folded.triage is Triage.REVIEW
        assert folded.release is Release.REDACTED and folded.release_ground == NON_TASK_SPEECH_MASKED
        assert "speech_in_task" in folded.ground_keys and folded.reason == "off_task_speech"

    def test_an_untimed_word_withholds(self) -> None:
        """A word no mask can place holds the recording."""
        folded = _fold(self._TASK, RedactionEvidence(non_task_speech_masked_n=1, non_task_speech_untimed_n=1))
        assert folded.release is Release.WITHHELD and folded.release_ground == NON_TASK_SPEECH_UNTIMED

    def test_an_extensive_off_task_share_withholds(self) -> None:
        """An item list whose phrases reach their bound is held."""
        folded = _fold(self._TASK, RedactionEvidence(non_task_speech_masked_n=4, non_task_speech_extensive=True))
        assert folded.release is Release.WITHHELD and folded.release_ground == NON_TASK_SPEECH_EXTENSIVE

    def test_a_non_lexical_redacted_release_is_reviewed_with_a_reason(self) -> None:
        """A redacted airway recording is review even with no speech reading."""
        task = TaskEvidence(owning_branches=("AIRWAY",))
        folded = _fold(task, RedactionEvidence(non_task_speech_masked_n=1))
        assert folded.triage is Triage.REVIEW
        assert "release_in_nonlexical_task" in folded.ground_keys

    def test_no_speech_passes(self) -> None:
        """The control: no word outside the task, nothing masked, the original released."""
        task = TaskEvidence(owning_branches=("AIRWAY",), task_speech={"words_n": 0}, task_speech_words_min=1)
        folded = _fold(task, RedactionEvidence())
        assert folded.triage is Triage.PASS and folded.release is Release.AS_IS


def _timed_words(store: ProvStore, stem: str, words: Sequence[tuple[str, float]]) -> None:
    """A recording stream naming the stem, and one agreed word per ``(text, start)``, 0.4 s long."""
    store.entity(
        prov_type="stream", extent=(0.0, 60.0), attributes={"name": "recording", "path": f"streams/{stem}.wav"}
    )
    for index, (text, start) in enumerate(words):
        extent = (start, start + 0.4)
        store.entity(prov_type="word", extent=extent, attributes=word_attributes(text, extent, index=index))


def _said(store: ProvStore, utterances: Sequence[str], *, pause_s: float = 0.8) -> None:
    """One utterance per string, its words 0.4 s long and 0.1 s apart, the utterances ``pause_s`` apart."""
    words: list[tuple[str, float]] = []
    at = 0.0
    for utterance in utterances:
        for text in utterance.split():
            words.append((text, at))
            at += 0.5
        at += pause_s - 0.1
    _timed_words(store, ITEMS_STEM, words)


def _item_fold(
    store: ProvStore,
    *,
    annotation: dict | None = None,
    opinion: dict | None = None,
    cohort: dict | None = None,
) -> FileVerdict:
    """Fold an item list's reading, with the reviewer, the second opinion and COHORT as given."""
    family = "random-item-generation"
    reading = task_speech_of(store, family)
    decisions = {
        name: BranchDecision(name, False, route, False)
        for name, route in (("AIRWAY", DECLINED), ("SPEECH", ROUTED), ("VOICE", DECLINED))
    }
    return fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
        branch_decisions=decisions,
        ran={},
        hint_claims={},
        route_state=ROUTED,
        redaction=RedactionEvidence(lexical_words_n=reading.lexical_n, scanned=True),
        llm_redaction=annotation,
        second_opinion=opinion,
        cohort=cohort,
        task=TaskEvidence(
            owning_branches=("SPEECH",),
            task_speech=reading.record(),
            item_set=True,
            phrase_share_review=phrase_share_bound(family),
        ),
        policy=FoldPolicy.from_config(load_triage_config()),
    )


_ITEMS = ["lion", "tiger", "bear", "zebra", "giraffe", "monkey", "elephant", "horse"]


class TestPhrasesInsteadOfItems:
    """Owner, 2026-10-10: in an item-set task the question is whether phrases are said instead of items."""

    def test_utterances_split_at_pauses_and_a_long_clause_is_a_phrase(self, store: ProvStore) -> None:
        """Items apart are each an utterance; four words holding a closed-class word is a phrase, three is not."""
        _said(store, ["lion", "polar bear", "is that enough", "i don't know what else to say"])
        read = task_speech_of(store, "animal-fluency")
        assert [(n, phrase) for _a, _b, n, phrase in read.utterances] == [
            (1, False),
            (2, False),
            (3, False),
            (7, True),
        ]
        assert read.phrase_share == 0.25 and read.phrase_word_share == round(7 / 13, 3)
        assert read.words_n == 7 and len(read.task_content_ids) == 6

    def test_four_items_said_together_are_not_a_phrase(self, store: ProvStore) -> None:
        """A run of content words with no closed-class word is a list, however long; a letter is not a function word."""
        _said(store, ["one two three four five", "a b c d e"])
        read = task_speech_of(store, "random-item-generation")
        assert read.phrase_share == 0.0 and read.words_n == 0

    def test_a_list_with_asides_is_not_reviewed(self, store: ProvStore) -> None:
        """Fillers and short asides between items: no phrase, and the reviewer's occasional is an annotation."""
        _said(store, [*_ITEMS[:4], "um", "let me think", *_ITEMS[4:], "is that enough"])
        assert task_speech_of(store, "random-item-generation").phrase_share == 0.0
        annotation = {
            "status": "clean",
            "original": "clean",
            "proposal": [],
            "phrases_instead_of_items": "occasional",
            "phrase_quotes": ["let me think"],
            "off_task_speech": "some",
        }
        opinion = {"status": "ok", "choices": {"phrases_instead_of_items": "occasional"}, "probabilities": {}}
        folded = _item_fold(store, annotation=annotation, opinion=opinion)
        assert folded.triage is Triage.PASS and folded.release is Release.AS_IS
        assert "phrases_occasional" in {note.key for note in folded.annotations}
        assert not {"reviewer_off_task_speech", "phrases_instead_of_items"} & set(folded.ground_keys)

    def test_a_list_turned_into_narration_is_reviewed(self, store: ProvStore) -> None:
        """One item, then clauses: the phrase share passes its bound and the recording is reviewed."""
        _said(
            store,
            [
                "lion",
                "my dog's name is buddy",
                "he likes to run in the park",
                "and we go there every day",
                "i don't know what else to say",
                "it was a long time ago",
            ],
        )
        read = task_speech_of(store, "random-item-generation")
        bound = phrase_share_bound("random-item-generation")
        assert bound is not None and read.phrase_share is not None and read.phrase_share >= bound
        folded = _item_fold(store)
        assert folded.triage is Triage.REVIEW and "phrases_instead_of_items" in folded.ground_keys
        assert folded.reason == "off_task_speech"

    def test_predominant_from_either_reader_reviews(self, store: ProvStore) -> None:
        """Below the bound, the reviewer's or the second opinion's predominant still reviews."""
        _said(store, _ITEMS)
        read = {"status": "clean", "original": "clean", "proposal": []}
        reviewer = _item_fold(store, annotation={**read, "phrases_instead_of_items": "predominant"})
        assert "reviewer_phrases_instead_of_items" in reviewer.ground_keys and reviewer.triage is Triage.REVIEW
        opinion = {"status": "ok", "choices": {"phrases_instead_of_items": "predominant"}, "probabilities": {}}
        second = _item_fold(store, annotation=read, opinion=opinion)
        assert "second_opinion:phrases_instead_of_items" in second.ground_keys and second.triage is Triage.REVIEW

    def test_a_list_with_a_background_speaker_is_reviewed_through_cohort(self, store: ProvStore) -> None:
        """No phrase at all, but a run that does not match the session enrollment: another speaker, review."""
        _said(store, _ITEMS)
        cohort = {
            "other_speaker": {
                "status": "measured",
                "nonmatch_n": 1,
                "runs_n": 3,
                "nonmatch_s": 2.1,
                "cut": 0.3,
                "nonmatch_spans": [[0.0, 2.1]],
            }
        }
        folded = _item_fold(store, cohort=cohort)
        assert folded.triage is Triage.REVIEW and "other_speaker_in_session" in folded.ground_keys
        assert folded.reason == "other_speaker"


def test_the_packaged_setting_reviews_every_release_but_the_original() -> None:
    """Owner, 2026-10-10: masks are not yet validated, so redacted and withheld releases are both reviewed."""
    from senselab.audio.workflows.triage.decision import reviewed_releases

    assert reviewed_releases() == frozenset({"redacted", "withheld"})


class TestTheReviewerAndTheSecondOpinionDecide:
    """Owner, 2026-10-10: the reviewer's task reading and the second opinion on the shipped text reach the verdict."""

    _READ = {"status": "clean", "original": "clean", "proposal": []}

    def _fold(
        self,
        *,
        annotation: dict | None = None,
        opinion: dict | None = None,
        cohort: dict | None = None,
        task: TaskEvidence | None = None,
    ) -> FileVerdict:
        decisions = {
            name: BranchDecision(name, False, route, False)
            for name, route in (("AIRWAY", DECLINED), ("SPEECH", ROUTED), ("VOICE", DECLINED))
        }
        return fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=decisions,
            ran={},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(lexical_words_n=5, scanned=True),
            llm_redaction=annotation,
            second_opinion=opinion,
            cohort=cohort,
            task=task or TaskEvidence(owning_branches=("SPEECH",)),
            policy=FoldPolicy.from_config(load_triage_config()),
        )

    def test_the_reviewer_s_off_task_and_other_speaker_readings_review_under_their_subjects(self) -> None:
        """Off-task speech is the participant's; an assistant is another speaker."""
        folded = self._fold(annotation={**self._READ, "off_task_speech": "some", "other_speaker": "assistant"})
        assert {"reviewer_off_task_speech", "reviewer_other_speaker"} <= set(folded.ground_keys)
        assert folded.reason_keys[:2] == ["other_speaker", "off_task_speech"]

    def test_a_confident_not_free_on_the_shipped_text_holds_the_release(self) -> None:
        """The second opinion reads identifiers in what would ship: review, and the release waits for a person."""
        opinion = {"status": "ok", "probabilities": {"masked_text_free_of_identifiers": 0.05}}
        folded = self._fold(annotation=self._READ, opinion=opinion)
        assert "second_opinion:masked_text_free_of_identifiers" in folded.ground_keys
        assert folded.release is Release.WITHHELD and folded.reason == "identifying_content"

    def test_speech_matching_no_enrollment_is_another_speaker(self) -> None:
        """The same speech outside the task is other_speaker where COHORT's comparison does not match it."""
        task = TaskEvidence(
            owning_branches=("AIRWAY",),
            task_speech={"words_n": 3, "runs_n": 1, "runs": [[100.0, 104.0, 3]]},
            task_speech_words_min=1,
        )
        cohort = {"other_speaker": {"status": "measured", "nonmatch_spans": [[99.0, 110.0]]}}
        folded = self._fold(cohort=cohort, task=task)
        assert "other_speaker_in_task" in folded.ground_keys and "speech_in_task" not in folded.ground_keys
        matched = self._fold(cohort={"other_speaker": {"status": "measured", "nonmatch_spans": []}}, task=task)
        assert "speech_in_task" in matched.ground_keys


class TestTheReviewerQuotesBecomeMasks:
    """The reviewer's off-task quotes are masked; its task-content quotes release a finding."""

    _READ = {"status": "clean", "original": "clean", "proposal": []}

    def _annotate(self, store: ProvStore, **attributes: object) -> None:
        activity = store.activity(node="REVIEW", step="read", parameters={})
        write_measurement(
            store,
            activity,
            software_agent(store),
            name="redaction_llm_annotation",
            signal="plain",
            attributes={**self._READ, **attributes},
        )

    def test_an_item_list_masks_phrase_quotes_only_when_predominant_and_never_an_item(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Occasional phrases mask nothing; predominant ones mask the phrase's words and never an item's."""
        from tests.audio.workflows.triage.nodes import redact_test

        words = ["lion", "tiger", "so", "my", "doctor", "said", "that"]
        monkeypatch.setattr(
            redact_test,
            "_word_extent",
            lambda i: (float(i), i + 0.5) if i < 2 else (2.0 + (i - 2) * 0.6, 2.5 + (i - 2) * 0.6),
        )
        joined: dict[int, dict[str, tuple[float, float]]] = {}
        _seed_redact_store(store, tmp_path, words=words, timings=joined, recording_stem=ITEMS_STEM, scanned=False)
        _write(store, "random-item-generation")
        self._annotate(store, phrases_instead_of_items="occasional", phrase_quotes=["tiger so my doctor said that"])
        assert mask_plan(store, reviewer_applies=True, padding_ms=50).non_task_speech_masked_n == 0
        other = ProvStore(run_id="predominant")
        _seed_redact_store(other, tmp_path, words=words, timings=joined, recording_stem=ITEMS_STEM, scanned=False)
        _write(other, "random-item-generation")
        self._annotate(other, phrases_instead_of_items="predominant", phrase_quotes=["tiger so my doctor said that"])
        plan = mask_plan(other, reviewer_applies=True, padding_ms=50)
        masked = [word.text for mask in plan.masks if mask.source == NON_TASK_SPEECH for word in mask.words]
        assert masked == words[2:] and plan.non_task_speech_extensive

    def test_an_off_task_quote_masks_its_words_and_a_task_content_quote_releases_them(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Three quoted off-task words stay masked; a finding the reviewer called task content is not."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["lion", "tiger", "my", "doctor", "said", "Paris"],
            findings=[("LOCATION", _word_extent(5))],
            recording_stem="sub-x_ses-y_task-cinderella-story",
        )
        activity = store.activity(node="REVIEW", step="read", parameters={})
        write_measurement(
            store,
            activity,
            software_agent(store),
            name="redaction_llm_annotation",
            signal="plain",
            attributes={
                "status": "clean",
                "original": "clean",
                "proposal": [],
                "off_task_speech": "some",
                "off_task_quotes": ["my doctor said"],
                "task_content_quotes": ["Paris"],
            },
        )
        plan = mask_plan(store, reviewer_applies=True, padding_ms=50)
        assert plan.non_task_speech_masked_n == 3
        assert all(word.text != "Paris" or word.state != "masked" for mask in plan.masks for word in mask.words)


def test_a_lexical_task_s_own_words_are_its_own_sound_and_an_airway_task_has_none(
    store: ProvStore, tmp_path: Path
) -> None:
    """QUALITY sets a spoken task's words aside as the participant's; a breath task's words stay candidates."""
    from senselab.audio.workflows.triage.nodes.quality import spoken_task_words

    _seed_redact_store(store, tmp_path, words=["once", "upon", "a"], recording_stem="sub-x_ses-y_task-cinderella-story")
    assert len(spoken_task_words(store)) == 3
    other = ProvStore(run_id="airway")
    _seed_redact_store(other, tmp_path, words=["hello"], recording_stem=BREATH_STEM)
    assert spoken_task_words(other) == []
