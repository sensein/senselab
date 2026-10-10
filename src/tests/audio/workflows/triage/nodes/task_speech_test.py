"""Speech outside the task (task_speech.py): what is read, what the fold reviews and what the mask plan hides."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from senselab.audio.workflows.triage.nodes.common import consensus_words, software_agent, write_measurement
from senselab.audio.workflows.triage.nodes.redact import NON_TASK_SPEECH, mask_plan
from senselab.audio.workflows.triage.task_content import task_content_ids
from senselab.audio.workflows.triage.task_speech import (
    TASK_SPEECH_READING,
    item_runs,
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
        assert "speech_in_task" in folded.ground_keys and folded.reason == "speech_in_task"

    def test_an_untimed_word_withholds(self) -> None:
        """A word no mask can place holds the recording."""
        folded = _fold(self._TASK, RedactionEvidence(non_task_speech_masked_n=1, non_task_speech_untimed_n=1))
        assert folded.release is Release.WITHHELD and folded.release_ground == NON_TASK_SPEECH_UNTIMED

    def test_an_extensive_off_task_share_withholds(self) -> None:
        """An item list mostly spoken outside its item runs is held."""
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


class TestItemRuns:
    """An item list's task runs: split at long pauses, enough category members; the rest is outside the task."""

    def test_a_background_speaker_before_the_list_is_outside_it(self, store: ProvStore) -> None:
        """A sentence at the start, a pause, then the items: only the items are the task, and half is off it."""
        _timed_words(
            store,
            ITEMS_STEM,
            [("are", 0.0), ("you", 0.5), ("recording", 1.0), ("cat", 6.0), ("dog", 8.0), ("horse", 10.0)],
        )
        words = consensus_words(store)
        members = {words[i].id for i in (3, 4, 5)}
        runs = item_runs(words, members, gap_s=3.0, member_share_min=0.5)
        assert [w.id for run in runs for w in run] == [w.id for w in words[3:]]
        read = task_speech_of(store, "random-item-generation", member_ids=members)
        assert read.words_n == 3 and read.off_task_fraction == 0.5
        assert read.item_extent == (6.0, 10.4)

    def test_with_no_category_every_run_is_the_task(self, store: ProvStore) -> None:
        """Nothing is read as outside the list where no word names a member."""
        _timed_words(store, ITEMS_STEM, [("one", 0.0), ("two", 6.0)])
        assert task_speech_of(store, "random-item-generation").words_n == 0


def test_the_packaged_setting_reviews_every_release_but_the_original() -> None:
    """Owner, 2026-10-10: masks are not yet validated, so redacted and withheld releases are both reviewed."""
    from senselab.audio.workflows.triage.decision import reviewed_releases

    assert reviewed_releases() == frozenset({"redacted", "withheld"})
