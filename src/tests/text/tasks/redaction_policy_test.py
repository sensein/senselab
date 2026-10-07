"""Redaction policy v7: the words it always masks and the ones it releases by kind, read off the words alone."""

from __future__ import annotations

import pytest

from senselab.text.tasks.pii_detection import redaction_policy as policy


def _hits(sentence: str, finder) -> list[str]:  # noqa: ANN001 -- one of the module's position finders
    words = sentence.split()
    return [words[position] for position in sorted(finder(words))]


@pytest.mark.parametrize(
    ("sentence", "masked"),
    [
        ("I had COVID in 2021 and", ["2021"]),
        ("back in nineteen ninety eight we", ["nineteen", "ninety", "eight"]),
        ("in two thousand twenty one", ["two", "thousand", "twenty", "one"]),
        ("since the '98 season", ["'98"]),
        ("walked two thousand steps", []),
        ("every summer we go to Halloween parties in October", ["Halloween", "October"]),
        ("en invierno y en verano", []),
        ("the 14th of March", ["14th", "March"]),
        ("el 3 de marzo", ["3", "marzo"]),
        ("this spring and in the fall", []),
        ("I had a fall and may spring up", []),
        ("on Monday last year two-three weeks ago", []),
        ("I am seventy-three years old and", ["seventy-three", "years", "old"]),
        ("I'm 73. My wife", ["73."]),
        ("a 73-year-old man", ["73-year-old"]),
        ("tengo 35 años y mi mamá", ["35", "años"]),
        ("in my early sixties", ["early", "sixties"]),
        ("my 50th birthday", ["50th", "birthday"]),
        ("I'm 5 minutes late", []),
    ],
)
def test_date_positions_mask_every_absolute_date_element_and_every_age(sentence: str, masked: list[str]) -> None:
    """2021, October and Halloween are redacted; a season, a duration or a weekday is not (v9, 2026-10-04)."""
    assert _hits(sentence, policy.date_positions) == masked


def test_a_state_is_a_place_written_as_a_proper_noun() -> None:
    """New York and Florida are states; "maine" written lower-case is not read as one."""
    assert _hits("I live in New York and grew up in Florida near Mexico", policy.state_positions) == [
        "New",
        "York",
        "Florida",
    ]
    assert _hits("the maine thing", policy.state_positions) == []


def test_a_country_is_its_own_run_and_never_part_of_a_state() -> None:
    """Mexico is a country; New Mexico is a state; "US" written upper-case is the country, "us" is not."""
    runs = policy.country_runs("from Mexico to the United States and to New Mexico".split())
    words = "from Mexico to the United States and to New Mexico".split()
    assert [" ".join(words[a:b]) for a, b in runs] == ["Mexico", "United States"]
    assert [b - a for a, b in policy.country_runs("us and the US".split())] == [1]


def test_kinship_words_in_english_and_spanish() -> None:
    """Relationship words are never masked: an in-law hyphenated is one word."""
    assert _hits("my mother-in-law and my brother and mi abuela met Ana", policy.kinship_positions) == [
        "mother-in-law",
        "brother",
        "abuela",
    ]


def test_the_ledger_records_the_packaged_policy_version() -> None:
    """The version on every ledger is the policy file's own, not a literal that can fall behind it."""
    import yaml

    from senselab.audio.workflows.triage.nodes import redact

    packaged = yaml.safe_load(policy.POLICY_PATH.read_text())["version"]
    assert policy.version() == packaged == redact.POLICY_VERSION
