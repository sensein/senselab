"""COHORT's word gate on a real store: which words of a non-lexical task qualify a run for the speaker comparison."""

from __future__ import annotations

from pathlib import Path

from senselab.audio.workflows.triage.cohort_stage import facts_of, speech_outside_task, spells_task_syllables
from senselab.audio.workflows.triage.nodes.common import software_agent, write_measurement
from senselab.audio.workflows.triage.task_speech import TASK_SPEECH_READING
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.redact_test import _seed_redact_store, _word_extent

BREATH_STEM = "sub-x_ses-y_task-respiration-and-cough-v2-threebreathsnose"
DDK_STEM = "sub-x_ses-y_task-diadochokinesis-pa"
SPEECH_STEM = "sub-x_ses-y_task-rainbow-passage"


def _reading(store: ProvStore, name: str, events: list[tuple[float, float]]) -> None:
    activity = store.activity(node="TASK", step="align", parameters={"name": name})
    evidence = {"decision": "present", "events": [{"start_s": a, "end_s": b} for a, b in events]}
    write_measurement(
        store,
        activity,
        software_agent(store),
        name=name,
        signal="plain",
        attributes={"decision": "present", "reading": {"evidence": evidence}},
    )


def test_a_breath_task_qualifies_its_talker_words_and_not_its_bracketed_ones(store: ProvStore, tmp_path: Path) -> None:
    """With no stored speech reading, the lexical words off the breath events are the talker's."""
    _seed_redact_store(store, tmp_path, words=["[breath]", "so", "where", "did"], recording_stem=BREATH_STEM)
    _reading(store, "airway_breath_reading", [_word_extent(0)])
    family = "respiration-and-cough-v2-threebreathsnose"
    assert len(speech_outside_task(store, family)) == 3
    facts = facts_of(store, None)
    assert facts.words_gated and len(facts.lexical) == 3


def test_the_stored_speech_reading_names_the_qualifying_words(store: ProvStore, tmp_path: Path) -> None:
    """The owning branch's reading decides which words are speech outside the task."""
    _seed_redact_store(store, tmp_path, words=["so", "where", "did"], recording_stem=BREATH_STEM)
    words = speech_outside_task(store, "respiration-and-cough-v2-threebreathsnose")
    activity = store.activity(node="AIRWAY", step="speech", parameters={})
    write_measurement(
        store,
        activity,
        software_agent(store),
        name=TASK_SPEECH_READING,
        signal="plain",
        attributes={"word_ids": [words[0].id]},
    )
    assert [w.id for w in speech_outside_task(store, "respiration-and-cough-v2-threebreathsnose")] == [words[0].id]


def test_a_ddk_word_on_a_syllable_event_does_not_qualify(store: ProvStore, tmp_path: Path) -> None:
    """In a syllable-repetition task any word on its events is the task misread; one off them is speech."""
    _seed_redact_store(store, tmp_path, words=["pa", "pa", "hello"], recording_stem=DDK_STEM)
    _reading(store, "ddk_task_reading", [_word_extent(0), _word_extent(1)])
    facts = facts_of(store, None)
    assert facts.words_gated and facts.lexical == (_word_extent(2),)


def test_agreed_syllable_words_off_the_events_are_task_text_in_a_ddk_task(store: ProvStore, tmp_path: Path) -> None:
    """Recognisers agreeing on the syllables is still the task's text; a dictionary word beside them is speech."""
    _seed_redact_store(store, tmp_path, words=["Pa-ta-ka", "pataka", "okay"], recording_stem=DDK_STEM)
    assert [w.attributes["text"] for w in speech_outside_task(store, "diadochokinesis-pataka")] == ["okay"]


def test_task_syllables_spell_only_their_own_family() -> None:
    """A syllable spelling is task text in a syllable-repetition family and nowhere else."""
    assert spells_task_syllables("Buttercup", "diadochokinesis-buttercup")
    assert spells_task_syllables("puhtuhkuh", "diadochokinesis-v2-puhtuhkuh")
    assert not spells_task_syllables("pattern", "diadochokinesis-pa")
    assert not spells_task_syllables("pa", "prolonged-vowel")


def test_a_speech_task_keeps_every_lexical_word(store: ProvStore, tmp_path: Path) -> None:
    """A lexical speech task is not gated: its lexical words are all read."""
    _seed_redact_store(store, tmp_path, words=["the", "rainbow", "[noise]"], recording_stem=SPEECH_STEM)
    facts = facts_of(store, None)
    assert not facts.words_gated and len(facts.lexical) == 2
