"""Redaction policy v7: the words it always masks and the ones it releases by kind, read off the words alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.text.tasks.pii_detection import redaction_policy as policy

LEXICON = """
version: 1
months: [january, february, march, april, may, june, july, august, september, october, november, december]
months_es: [enero, febrero, marzo, abril, mayo]
seasons: [summer, winter, autumn, verano, invierno]
seasons_in_context: [spring, fall]
season_context_before: [the, this, last, next, in]
season_context_after: [break, semester]
holidays: [christmas, thanksgiving, halloween, easter]
holiday_phrases:
- [new, year]
- [fourth, of, july]
age_after: [old, olds, yo]
age_units: [year, years, ano, anos]
age_before: [age, aged, edad, tengo]
age_decades: [teens, twenties, thirties, forties, fifties, sixties, seventies]
age_decade_before: [my, his, her]
birthday: [birthday, cumpleanos]
first_person_be: [i'm, im, i am]
number_words: [one, two, three, five, seven, eight, ten, twenty, thirty, fifty, seventy, ninety, hundred, thousand,
  nineteen, y]
ordinal_words: [first, second, third, fiftieth]
year_centuries: [seventeen, eighteen, nineteen]
year_twenty_followers: [ten, eleven, twelve, twenty]
year_thousand_heads: [[two, thousand], [dos, mil]]
year_lead: [in, since, until, from, en, desde]
kinship: [mother, motherinlaw, brother, wife, abuela, mama]
countries: [mexico, canada, france]
country_phrases:
- [united, states]
country_abbreviations: [US, USA, UK]
states: [florida, maine, texas]
state_phrases:
- [new, york]
- [new, mexico]
state_abbreviations: [NYC]
"""
"""A small word list of the test's own: the functions take the caller's lists (plan step 6b), not a dataset's."""


@pytest.fixture(scope="module")
def lexicon(tmp_path_factory: pytest.TempPathFactory) -> policy.Lexicon:
    """The test's word list, loaded as a caller loads its own."""
    path = Path(tmp_path_factory.mktemp("lexicon")) / "redaction_policy.yaml"
    path.write_text(LEXICON)
    return policy.load_lexicon(path)


def _hits(sentence: str, finder, lexicon: policy.Lexicon) -> list[str]:  # noqa: ANN001 -- a position finder
    words = sentence.split()
    return [words[position] for position in sorted(finder(words, lexicon=lexicon))]


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
def test_date_positions_mask_every_absolute_date_element_and_every_age(
    sentence: str, masked: list[str], lexicon: policy.Lexicon
) -> None:
    """2021, October and Halloween are redacted; a season, a duration or a weekday is not (v9, 2026-10-04)."""
    assert _hits(sentence, policy.date_positions, lexicon) == masked


def test_a_state_is_a_place_written_as_a_proper_noun(lexicon: policy.Lexicon) -> None:
    """New York and Florida are states; "maine" written lower-case is not read as one."""
    assert _hits("I live in New York and grew up in Florida near Mexico", policy.state_positions, lexicon) == [
        "New",
        "York",
        "Florida",
    ]
    assert _hits("the maine thing", policy.state_positions, lexicon) == []


def test_a_country_is_its_own_run_and_never_part_of_a_state(lexicon: policy.Lexicon) -> None:
    """Mexico is a country; New Mexico is a state; "US" written upper-case is the country, "us" is not."""
    runs = policy.country_runs("from Mexico to the United States and to New Mexico".split(), lexicon=lexicon)
    words = "from Mexico to the United States and to New Mexico".split()
    assert [" ".join(words[a:b]) for a, b in runs] == ["Mexico", "United States"]
    assert [b - a for a, b in policy.country_runs("us and the US".split(), lexicon=lexicon)] == [1]


def test_kinship_words_in_english_and_spanish(lexicon: policy.Lexicon) -> None:
    """Relationship words are never masked: an in-law hyphenated is one word."""
    assert _hits("my mother-in-law and my brother and mi abuela met Ana", policy.kinship_positions, lexicon) == [
        "mother-in-law",
        "brother",
        "abuela",
    ]


def test_the_version_is_the_lexicon_files_own(lexicon: policy.Lexicon) -> None:
    """The version on every ledger is the word list's own."""
    assert policy.version(lexicon) == 1
