"""COHORT's word gate on a real store: which words of a non-lexical task qualify a run for the speaker comparison."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.triage.cohort_stage import (
    RecordingFacts,
    cohort_attributes,
    compare_on,
    enrollment_candidates,
    facts_of,
    lexical_task,
    match_runs,
    speech_outside_task,
    spells_task_syllables,
    word_gated,
)
from senselab.audio.workflows.triage.nodes.common import software_agent, write_measurement
from senselab.audio.workflows.triage.task_speech import TASK_SPEECH_READING, task_speech_of
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.cohort_stage_test import CUT, PARTICIPANT, TALKER, _embed, _enroll, _loader, _session
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


def test_a_loudness_run_is_compared_on_its_hey(store: ProvStore, tmp_path: Path) -> None:
    """Owner, 2026-10-11: loudness is compared on its "hey", the target speaker's task text."""
    _seed_redact_store(store, tmp_path, words=["hey", "Hey!", "HEY-HEY"], recording_stem=LOUDNESS_STEM)
    facts = facts_of(store, None)
    assert facts.family == "loudness" and facts.words_gated and not facts.lexical_task
    assert compare_on("loudness") == compare_on("loudness-v2") == "task_text"
    assert facts.lexical == (_word_extent(0), _word_extent(1), _word_extent(2))


def test_a_loudness_word_other_than_hey_is_masked_and_not_compared(store: ProvStore, tmp_path: Path) -> None:
    """Any other word is speech outside the task, flagged and masked by SPEECH, and qualifies no run."""
    _seed_redact_store(store, tmp_path, words=["hey", "okay", "hey"], recording_stem=LOUDNESS_STEM)
    facts = facts_of(store, None)
    assert facts.lexical == (_word_extent(0), _word_extent(2))
    assert task_speech_of(store, "loudness").words_n == 1


def _loudness_facts(store: ProvStore, tmp_path: Path, words: list[str]) -> RecordingFacts:
    _seed_redact_store(store, tmp_path, words=words, recording_stem=LOUDNESS_STEM)
    span = (0.0, len(words) * 1.0)
    return dataclasses.replace(
        facts_of(store, None), session="sub-a_ses-1", diarization=((*span, "A"),), speech=(span,)
    )


@pytest.mark.parametrize("level,flagged", [(PARTICIPANT, False), (TALKER, True)])
def test_a_hey_run_is_other_speaker_only_where_it_misses_the_enrollment(
    store: ProvStore, tmp_path: Path, level: float, flagged: bool
) -> None:
    """A "hey" run matching the enrollment raises nothing; one under the cut is other_speaker."""
    from senselab.audio.workflows.triage.vocabulary import fold_file_verdict  # noqa: PLC0415

    facts = _loudness_facts(store, tmp_path, ["hey", "hey", "HEY!"])
    speech, _, _ = _session()
    block = match_runs(
        facts, _loader([(0.0, 3.0, level)], 20.0), _enroll(speech), _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.6
    )
    assert block["runs_n"] > 0 and (block["nonmatch_n"] > 0) is flagged
    attributes = cohort_attributes(facts, other_speaker=block, checks={}, quantiles=None, key="k")
    common: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}
    folded = fold_file_verdict([], cohort=attributes, **common)
    assert ("other_speaker" in folded.reason_keys) is flagged


def test_a_loudness_run_without_hey_is_not_compared(store: ProvStore, tmp_path: Path) -> None:
    """A diarized speech run holding only words other than "hey" is not compared, whoever speaks."""
    facts = _loudness_facts(store, tmp_path, ["okay", "so", "well"])
    assert facts.lexical == ()
    speech, _, _ = _session()
    block = match_runs(
        facts, _loader([(0.0, 3.0, TALKER)], 20.0), _enroll(speech), _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.6
    )
    assert block["runs_n"] == 0 and block["nonmatch_n"] == 0
    assert task_speech_of(store, "loudness").words_n == 3


def test_other_families_keep_their_rule() -> None:
    """Voice, airway and DDK families compare on speech outside the task; lexical speech on speech regions."""
    for family in ("prolonged-vowel", "glides-high-to-low", "diadochokinesis-pa", "respiration-and-cough-breath"):
        assert compare_on(family) == "non_task_words" and word_gated(family)
    assert compare_on("rainbow-passage") == "speech" and not word_gated("rainbow-passage")


def test_loudness_never_enrolls_and_prolonged_vowel_is_unchanged() -> None:
    """Loudness leaves the enrollment as a voice family does; the vowel stays gated and non-lexical."""
    assert word_gated("loudness") and word_gated("loudness-v2") and word_gated("prolonged-vowel")
    assert not lexical_task("loudness") and not lexical_task("prolonged-vowel") and lexical_task("rainbow-passage")
    assert not word_gated("rainbow-passage")
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
                lexical=(),
                speech=((0.0, 3.0),),
                removed_speech=(),
                task_events=(),
            ),
            lambda: None,
        )
        for family in ("loudness", "loudness-v2", "rainbow-passage")
    ]
    enrolled = enrollment_candidates(members, min_s=1.0, lexical_gap_s=0.5)
    assert [candidate.facts.family for candidate in enrolled] == ["rainbow-passage"]


def test_a_speech_task_keeps_every_lexical_word(store: ProvStore, tmp_path: Path) -> None:
    """A lexical speech task is not gated: its lexical words are all read."""
    _seed_redact_store(store, tmp_path, words=["the", "rainbow", "[noise]"], recording_stem=SPEECH_STEM)
    facts = facts_of(store, None)
    assert not facts.words_gated and len(facts.lexical) == 2
