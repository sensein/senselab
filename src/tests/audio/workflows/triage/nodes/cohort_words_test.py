"""COHORT's pools on a real store: which words a recording's target and non-task speaker comparisons hold."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.triage.cohort_fold import (
    MEASURED,
    NOT_APPLICABLE,
    NOT_MEASURED,
    POOL_NON_TASK,
    POOL_TARGET,
)
from senselab.audio.workflows.triage.cohort_stage import (
    TOO_SHORT_FOR_SPEAKER_ID,
    RecordingFacts,
    cohort_attributes,
    compare_speaker,
    enrollment_candidates,
    facts_of,
    lexical_task,
    speech_outside_task,
    spells_task_syllables,
    target_rule,
)
from senselab.audio.workflows.triage.nodes.common import software_agent, write_measurement
from senselab.audio.workflows.triage.task_speech import TASK_SPEECH_READING, task_speech_of
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.cohort_stage_test import (
    CUT,
    MIN_S,
    PARTICIPANT,
    TALKER,
    _embed,
    _enroll,
    _loader,
    _session,
)
from tests.audio.workflows.triage.nodes.redact_test import _seed_redact_store, _word_extent

BREATH_STEM = "sub-x_ses-y_task-respiration-and-cough-v2-threebreathsnose"
DDK_STEM = "sub-x_ses-y_task-diadochokinesis-pa"
SPEECH_STEM = "sub-x_ses-y_task-rainbow-passage"
LOUDNESS_STEM = "sub-x_ses-y_task-loudness"


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
    assert len(facts.non_task) == 3 and facts.target == ()


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
    assert facts.non_task == (_word_extent(2),)


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


def test_a_loudness_target_is_its_hey(store: ProvStore, tmp_path: Path) -> None:
    """Owner, 2026-10-11: loudness is compared on its "hey", the target speaker's task text."""
    _seed_redact_store(store, tmp_path, words=["hey", "Hey!", "HEY-HEY"], recording_stem=LOUDNESS_STEM)
    facts = facts_of(store, None)
    assert facts.family == "loudness" and not facts.lexical_task
    assert target_rule("loudness") == target_rule("loudness-v2") == "task_text"
    assert facts.target == (_word_extent(0), _word_extent(1), _word_extent(2)) and facts.non_task == ()


def test_a_loudness_word_other_than_hey_is_masked_and_pooled_apart(store: ProvStore, tmp_path: Path) -> None:
    """Any other word is speech outside the task: flagged and masked by SPEECH, and in the non-task pool only."""
    _seed_redact_store(store, tmp_path, words=["hey", "okay", "hey"], recording_stem=LOUDNESS_STEM)
    facts = facts_of(store, None)
    assert facts.target == (_word_extent(0), _word_extent(2)) and facts.non_task == (_word_extent(1),)
    assert task_speech_of(store, "loudness").words_n == 1


def _loudness_facts(store: ProvStore, tmp_path: Path, words: list[str]) -> RecordingFacts:
    _seed_redact_store(store, tmp_path, words=words, recording_stem=LOUDNESS_STEM)
    return dataclasses.replace(facts_of(store, None), session="sub-a_ses-1")


def _compare(facts: RecordingFacts, level: float) -> dict[str, Any]:
    speech, _, _ = _session()
    audio = _loader([(0.0, 10.0, level)], 20.0)
    return compare_speaker(facts, audio, _enroll(speech), _embed, cut=CUT, min_s=MIN_S, lexical_gap_s=0.6)


@pytest.mark.parametrize("level,flagged", [(PARTICIPANT, False), (TALKER, True)])
def test_a_hey_pool_is_other_speaker_only_where_it_misses_the_enrollment(
    store: ProvStore, tmp_path: Path, level: float, flagged: bool
) -> None:
    """A pooled "hey" matching the enrollment raises nothing; one under the cut is other_speaker."""
    from senselab.audio.workflows.triage.vocabulary import fold_file_verdict  # noqa: PLC0415

    facts = _loudness_facts(store, tmp_path, ["hey", "hey", "HEY!"])
    block = _compare(facts, level)
    assert block[POOL_TARGET]["status"] == MEASURED and block[POOL_TARGET]["pooled_s"] == 2.5
    assert (not block[POOL_TARGET]["match"]) is flagged and block[POOL_NON_TASK]["status"] == NOT_APPLICABLE
    attributes = cohort_attributes(facts, other_speaker=block, checks={}, quantiles=None, key="k")
    common: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}
    folded = fold_file_verdict([], cohort=attributes, **common)
    assert ("other_speaker" in folded.reason_keys) is flagged


def test_a_hey_pool_under_two_seconds_is_not_measured(store: ProvStore, tmp_path: Path) -> None:
    """Two short "hey"s pool to 1.5 s: too short for speaker identification, whoever speaks."""
    facts = _loudness_facts(store, tmp_path, ["hey", "hey"])
    block = _compare(facts, TALKER)
    assert block[POOL_TARGET]["status"] == NOT_MEASURED and block[POOL_TARGET]["reason"] == TOO_SHORT_FOR_SPEAKER_ID
    assert block["nonmatch"] == []


def test_a_loudness_recording_without_hey_has_no_target(store: ProvStore, tmp_path: Path) -> None:
    """Words other than "hey" never enter the target pool; they are masked and flagged as speech outside the task."""
    facts = _loudness_facts(store, tmp_path, ["okay", "so", "well"])
    assert facts.target == ()
    block = _compare(facts, TALKER)
    assert block[POOL_TARGET]["status"] == NOT_MEASURED and block[POOL_TARGET]["pooled_s"] == 0.0
    assert block[POOL_NON_TASK]["status"] == MEASURED and not block[POOL_NON_TASK]["match"]
    assert task_speech_of(store, "loudness").words_n == 3


def test_other_families_keep_their_rule() -> None:
    """Voice, airway and DDK families have no target pool; lexical speech pools its extent."""
    for family in ("prolonged-vowel", "glides-high-to-low", "diadochokinesis-pa", "respiration-and-cough-breath"):
        assert target_rule(family) == "none"
    assert target_rule("rainbow-passage") == "speech"


def test_loudness_never_enrolls_and_prolonged_vowel_is_unchanged() -> None:
    """Loudness leaves the enrollment as a voice family does; the vowel stays non-lexical."""
    assert not lexical_task("loudness") and not lexical_task("prolonged-vowel") and lexical_task("rainbow-passage")
    span = [(0.0, 3.0, "A")]
    members = [
        (
            RecordingFacts(
                stem=f"sub-a_ses-1_task-{family}",
                session="sub-a_ses-1",
                family=family,
                lexical_task=lexical_task(family),
                duration_s=3.0,
                task_extent=None,
                diarization=tuple(span),
            ),
            lambda: None,
        )
        for family in ("loudness", "loudness-v2", "rainbow-passage")
    ]
    enrolled = enrollment_candidates(members, min_s=MIN_S, lexical_gap_s=0.5)
    assert [candidate.facts.family for candidate in enrolled] == ["rainbow-passage"]


def test_a_speech_task_pools_its_extent_and_the_words_outside_it(store: ProvStore, tmp_path: Path) -> None:
    """A lexical speech task's target holds its words; with no task extent, nothing is outside the task."""
    _seed_redact_store(store, tmp_path, words=["the", "rainbow", "[noise]"], recording_stem=SPEECH_STEM)
    facts = facts_of(store, None)
    assert len(facts.lexical) == 2 and facts.target == facts.lexical and facts.non_task == ()
    block = _compare(dataclasses.replace(facts, session="sub-a_ses-1"), PARTICIPANT)
    assert block[POOL_NON_TASK]["status"] == NOT_APPLICABLE
