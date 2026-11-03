"""Tests for the task lexicon: a task's own words, matched one way for the residue, REDACT and the fold."""

from __future__ import annotations

import pytest
import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.dataset_profile import data_path
from senselab.audio.workflows.triage.residue import FREE, residue_rule, task_residue
from senselab.audio.workflows.triage.task_catalogue import item_category_rule, task_list
from senselab.audio.workflows.triage.task_lexicon import (
    SECTIONS,
    TaskLexicon,
    category_members,
    category_phrases,
    declared_names_lexicon,
    packaged_phrases,
    task_lexicon,
)


def test_the_packaged_cinderella_lexicon_loads_as_normalised_phrases() -> None:
    """Every section reads; phrases stay whole; no function word stands alone."""
    phrases = packaged_phrases("cinderella-story")
    assert ("cinderella",) in phrases and ("glass", "slipper") in phrases and ("one", "day") in phrases
    assert ("happily", "ever", "after") in phrases
    assert not {("the",), ("and",), ("she",), ("a",)} & set(phrases)
    assert set(SECTIONS) == {"characters", "objects", "events", "time"}


def test_a_family_without_a_file_has_no_packaged_phrases() -> None:
    """Nothing is invented for a family the study gives no list for."""
    assert packaged_phrases("free-speech") == ()


def test_the_packaged_config_turns_the_cinderella_lexicon_on() -> None:
    """Owner, 2026-09-28; the declared cast rides along either way."""
    config = load_triage_config()
    on = task_lexicon(config, "cinderella-story")
    assert ("pumpkin",) in on.phrases and ("prince",) in on.phrases
    assert task_lexicon(config, "free-speech").phrases == ()
    assert declared_names_lexicon("cinderella-story").phrases and not declared_names_lexicon(None).phrases


@pytest.mark.parametrize(
    "texts, expected",
    [
        (["Cinderella's", "coach"], {0, 1}),
        (["the", "stepsisters", "laughed"], {1}),
        (["one", "day", "she", "left"], {0, 1}),
        (["one", "cat"], set()),
        (["Sinderella", "ran"], {0}),
        (["my", "sister", "Jane"], {1}),
    ],
    ids=["possessive", "plural", "phrase", "half-a-phrase", "asr-spelling", "a-name-outside-the-story"],
)
def test_words_match_as_whole_runs(texts: list[str], expected: set[int]) -> None:
    """Possessives and plurals fold; a phrase matches whole; near-match only for longer tokens."""
    lexicon = task_lexicon(load_triage_config(), "cinderella-story")
    assert lexicon.positions(texts) == expected


def test_the_residue_sets_a_free_response_familys_task_words_aside() -> None:
    """A free response keeps every word except the ones the task itself is made of."""
    config = load_triage_config()
    texts = ["cinderella", "met", "the", "prince", "at", "midnight", "near", "Boston"]
    lexicon = task_lexicon(config, "cinderella-story")
    residue = task_residue(texts, "cinderella-story", [], residue_rule(config), lexicon=lexicon)
    assert residue.method == FREE
    assert [texts[p] for p in residue.positions] == ["met", "the", "at", "near", "Boston"]
    assert residue.task_n == 3


def test_an_empty_lexicon_changes_nothing() -> None:
    """No phrases, no subtraction."""
    config = load_triage_config()
    texts = ["cinderella", "met", "Boston"]
    plain = task_residue(texts, "cinderella-story", [], residue_rule(config))
    empty = task_residue(texts, "cinderella-story", [], residue_rule(config), lexicon=TaskLexicon(None, ()))
    assert plain == empty


def test_every_listed_category_names_a_words_file_of_its_own_category() -> None:
    """Each ``list`` category of the task list loads, under the name its file declares."""
    listed = {
        category.words: category
        for entry in task_list().values()
        for category in entry.categories.values()
        if category.words is not None
    }
    assert listed
    for words, category in listed.items():
        assert str(yaml.safe_load(data_path(words).read_text())["category"]) == category.name
        assert category_phrases(category), words
    packaged = {f"task_lexicon/categories/{path.name}" for path in data_path("task_lexicon/categories").glob("*.yaml")}
    assert packaged == set(listed)


def test_an_animal_fluency_list_is_task_content() -> None:
    """The family declares its category, so the animals it lists never reach a detector."""
    lexicon = task_lexicon(load_triage_config(), "animal-fluency")
    assert lexicon.positions(["Cat,", "dogs", "and", "Marisol"]) == {0, 1}


def test_a_random_item_list_takes_its_category_from_the_hint() -> None:
    """The category ``First names`` makes the listed first names task content; nothing else is."""
    hint = AudioHints(item_category="First names")
    config = load_triage_config()
    assert task_lexicon(config, "random-item-generation", hint).positions(["Susan", "John", "table"]) == {0, 1}
    assert task_lexicon(config, "random-item-generation").positions(["Susan"]) == set()


def test_a_rule_category_admits_numbers_without_a_list() -> None:
    """A ``Numbers`` list read as digits is the task, not a phone number."""
    hint = AudioHints(item_category="Numbers")
    lexicon = task_lexicon(load_triage_config(), "random-item-generation-v2", hint)
    assert lexicon.positions(["4", "seven", "hello"]) == {0, 1}
    assert category_members(item_category_rule("random-item-generation", "Letters"), ["b", "bee"]) == {0}


def test_a_category_names_its_rule_and_nothing_is_parsed_from_its_name() -> None:
    """The letter, the number words and the repetition rule are the entry's; a name the entry lacks has no rule."""
    letter = item_category_rule("random-item-generation", "English words starting with 't'")
    assert letter is not None and (letter.rule, letter.letter) == ("initial_letter", "t")
    assert category_members(letter, ["table", "time", "apple"]) == {0, 1}
    numbers = item_category_rule("random-item-generation", "Numbers")
    assert numbers is not None and numbers.repetition_allowed and "twelve" in numbers.number_words
    assert item_category_rule("random-item-generation", "words starting with 'b'") is None
    assert category_members(None, ["b"]) == set()
