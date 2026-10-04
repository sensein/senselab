"""Redaction policy v7: which transcript words are masked or released by kind, read off the words alone.

The policy's lexicons are packaged in ``data/redaction_policy.yaml``. Every function here takes one
recording's words as the recognizer wrote them, in order, and returns positions into that list, so
the triage fold and the reviewer's own answer check read the same rule. The rules and their
derivation are in ``specs/20261003-redaction-policy-v7/design.md``.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import yaml

POLICY_PATH = Path(__file__).parent / "data" / "redaction_policy.yaml"
"""The packaged lexicons of the redaction policy."""

_EDGE = "\"'`.,;:!?()[]{}<>‘’“”…¿¡"
_SENTENCE_END = (".", "?", "!", "…")
_YEAR_DIGITS = re.compile(r"^(18|19|20)\d\d(s)?$")
_SHORT_YEAR = re.compile(r"^['’]\d\d(s)?$")
_ORDINAL_DIGITS = re.compile(r"^\d{1,2}(st|nd|rd|th)$")
_DAY_DIGITS = re.compile(r"^\d{1,2}$")
_AGE_DIGITS = re.compile(r"^\d{1,3}$")
_HYPHEN_AGE = re.compile(r"^(\d{1,3}|[a-z]+(-[a-z]+)?)-years?-olds?$")


@lru_cache(maxsize=1)
def policy() -> dict[str, Any]:
    """The packaged policy, its token lists as sets and its phrases as token tuples.

    Returns:
        ``{key: frozenset | tuple of tuples | value}``.
    """
    raw = dict(yaml.safe_load(POLICY_PATH.read_text()) or {})
    table: dict[str, Any] = {}
    for key, value in raw.items():
        if key.endswith("_phrases") or key == "year_thousand_heads":
            table[key] = tuple(tuple(fold(str(word)) for word in phrase) for phrase in value or ())
        elif key.endswith("abbreviations"):
            table[key] = frozenset(str(word) for word in value or ())
        elif isinstance(value, list):
            table[key] = frozenset(fold(str(word)) for word in value)
        else:
            table[key] = value
    return table


def fold(text: str) -> str:
    """One token as the policy compares it.

    Args:
        text: A word's surface.

    Returns:
        Lower-cased, accents folded, edge punctuation and a possessive ``'s`` dropped, internal
        apostrophes dropped. Hyphens are kept.
    """
    token = str(text).replace("’", "'").strip().strip(_EDGE).lower()
    if token.endswith("'s"):
        token = token[:-2]
    token = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode()
    return token.replace("'", "")


def _raw(text: str) -> str:
    return str(text).strip().strip("\"'`,;:!?()[]{}<>‘’“”…¿¡")


def _capitalised(text: str) -> bool:
    stripped = _raw(text)
    return bool(stripped[:1]) and stripped[:1].isupper()


def _phrase_runs(keys: Sequence[str], phrases: Sequence[tuple[str, ...]]) -> list[tuple[int, int]]:
    """Every ``(start, end)`` run of keys that spells one of the phrases, longest first, non-overlapping."""
    runs: list[tuple[int, int]] = []
    taken: set[int] = set()
    for phrase in sorted(phrases, key=len, reverse=True):
        width = len(phrase)
        for start in range(len(keys) - width + 1):
            if tuple(keys[start : start + width]) == phrase and not taken & set(range(start, start + width)):
                runs.append((start, start + width))
                taken.update(range(start, start + width))
    return sorted(runs)


def is_number(key: str) -> bool:
    """Whether a folded token is a number: digits, a number word, or a hyphenated pair of them.

    Args:
        key: A folded token (:func:`fold`).

    Returns:
        True for ``73``, ``seventy``, ``seventy-three``, ``treinta``.
    """
    words = policy()["number_words"] - {"y"}
    if key.isdigit():
        return True
    parts = key.split("-")
    return bool(parts) and all(part in words for part in parts)


def _number_run(keys: Sequence[str], start: int) -> int:
    """The end of the run of number tokens beginning at ``start`` ("seventy three", "treinta y cinco")."""
    end = start
    while end < len(keys) and (
        is_number(keys[end]) or (keys[end] == "y" and end > start and end + 1 < len(keys) and is_number(keys[end + 1]))
    ):
        end += 1
    return end


def year_positions(texts: Sequence[str]) -> set[int]:
    """Positions that write a year: four digits, an apostrophe and two digits, or a spelled year.

    Args:
        texts: The words, in order.

    Returns:
        The positions.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    raws = [str(text).strip().rstrip(".,;:!?…") for text in texts]
    found: set[int] = set()
    for position, (key, raw) in enumerate(zip(keys, raws)):
        if _YEAR_DIGITS.match(key) or _SHORT_YEAR.match(raw):
            found.add(position)
    for position, key in enumerate(keys):
        nxt = keys[position + 1] if position + 1 < len(keys) else ""
        head = nxt.split("-")[0]
        if key in table["year_centuries"] and (head in table["number_words"] and head != "y" or nxt == "hundred"):
            found.update(range(position, _number_run(keys, position + 1)))
            found.add(position)
        elif key == "twenty" and (nxt in table["year_twenty_followers"] or nxt.startswith("twenty-")):
            found.update(range(position, _number_run(keys, position + 1)))
            found.add(position)
    for start, end in _phrase_runs(keys, table["year_thousand_heads"]):
        rest = end + 1 if end < len(keys) and keys[end] in ("and", "y") else end
        tail = _number_run(keys, rest)
        before = keys[start - 1] if start else ""
        if tail > rest or before in table["year_lead"]:
            found.update(range(start, max(end, tail)))
    return found


def month_positions(texts: Sequence[str]) -> set[int]:
    """Positions that name a month, and a day number or ordinal written beside one.

    An English month counts written with a capital, a Spanish one in any case.

    Args:
        texts: The words, in order.

    Returns:
        The positions.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    months = {
        position
        for position, (key, text) in enumerate(zip(keys, texts))
        if (key in table["months"] and _capitalised(text)) or key in table["months_es"]
    }
    found = set(months)
    for position in months:
        for offset in (-2, -1, 1, 2):
            other = position + offset
            if not 0 <= other < len(keys):
                continue
            key = keys[other]
            near = abs(offset) == 1 or keys[position + offset // 2] in ("of", "de", "the", "el")
            if near and (_ORDINAL_DIGITS.match(key) or _DAY_DIGITS.match(key) or key in table["ordinal_words"]):
                found.add(other)
    return found


def season_positions(texts: Sequence[str]) -> set[int]:
    """Positions that name a season.

    Args:
        texts: The words, in order.

    Returns:
        The positions. "spring" and "fall" count only beside a season context word.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    found: set[int] = set()
    for position, key in enumerate(keys):
        if key in table["seasons"]:
            found.add(position)
        elif key in table["seasons_in_context"]:
            before = keys[position - 1] if position else ""
            after = keys[position + 1] if position + 1 < len(keys) else ""
            if before in table["season_context_before"] or after in table["season_context_after"]:
                found.add(position)
    return found


def holiday_positions(texts: Sequence[str]) -> set[int]:
    """Positions that name a holiday, one word or a phrase of several.

    Args:
        texts: The words, in order.

    Returns:
        The positions.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    found = {position for position, key in enumerate(keys) if key in table["holidays"]}
    for start, end in _phrase_runs(keys, table["holiday_phrases"]):
        found.update(range(start, end))
    return found


def age_positions(texts: Sequence[str]) -> set[int]:
    """Positions that state an age.

    A number followed by a year unit and "old" ("seventy-three years old"), a hyphenated age
    ("73-year-old"), a number after "age", "aged" or "turned", a number with "años" after "tengo", a
    decade after a possessive ("in my sixties"), "I'm" or "I am" before a number that ends the clause,
    and an ordinal before "birthday".

    Args:
        texts: The words, in order.

    Returns:
        The positions: the number, its unit and "old" where written; never the cue word itself.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    raws = [str(text) for text in texts]
    found: set[int] = set()
    for position, key in enumerate(keys):
        if _HYPHEN_AGE.match(key):
            found.add(position)
            continue
        if is_number(key) and (not key.isdigit() or _AGE_DIGITS.match(key)):
            end = _number_run(keys, position)
            if position and is_number(keys[position - 1]):
                continue
            unit = keys[end] if end < len(keys) else ""
            after = keys[end + 1] if end + 1 < len(keys) else ""
            before = keys[position - 1] if position else ""
            before2 = keys[position - 2] if position >= 2 else ""
            span = set(range(position, end))
            if unit in table["age_units"] and (after in table["age_after"] or after == "de"):
                found |= span | {end, end + 1}
                if after == "de" and end + 2 < len(keys) and keys[end + 2] == "edad":
                    found.add(end + 2)
            elif unit in table["age_after"]:
                found |= span | {end}
            elif before in table["age_before"] and (before != "tengo" or unit in table["age_units"]):
                found |= span | ({end} if unit in table["age_units"] else set())
            elif f"{before2} {before}" in table["first_person_be"] or before in table["first_person_be"]:
                ends_clause = end >= len(keys) or raws[end - 1].rstrip().endswith((",", ".", "!", "?", ";"))
                if ends_clause or unit in ("and", "now", "y") or unit in table["age_units"]:
                    found |= span | ({end} if unit in table["age_units"] else set())
        if (
            key in table["age_decades"]
            and position
            and (
                keys[position - 1] in table["age_decade_before"]
                or (
                    keys[position - 1] in ("early", "late", "mid")
                    and position >= 2
                    and keys[position - 2] in table["age_decade_before"]
                )
            )
        ):
            found.add(position)
            if keys[position - 1] in ("early", "late", "mid"):
                found.add(position - 1)
        if key in table["birthday"] and position:
            previous = keys[position - 1]
            if _ORDINAL_DIGITS.match(previous) or previous in table["ordinal_words"] or is_number(previous):
                found |= {position - 1, position}
    return {position for position in found if position < len(keys)}


def date_positions(texts: Sequence[str]) -> set[int]:
    """Every position the policy always masks as a date element or an age.

    Args:
        texts: The words, in order.

    Returns:
        :func:`year_positions`, :func:`month_positions`, :func:`season_positions`,
        :func:`holiday_positions` and :func:`age_positions`, together.
    """
    return (
        year_positions(texts)
        | month_positions(texts)
        | season_positions(texts)
        | holiday_positions(texts)
        | age_positions(texts)
    )


def state_positions(texts: Sequence[str]) -> set[int]:
    """Positions that name a state, a province or their equivalent, written as a proper noun.

    Args:
        texts: The words, in order.

    Returns:
        The positions.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    found: set[int] = set()
    for start, end in _phrase_runs(keys, table["state_phrases"]):
        if _capitalised(texts[start]):
            found.update(range(start, end))
    for position, (key, text) in enumerate(zip(keys, texts)):
        if position in found:
            continue
        if (key in table["states"] and _capitalised(text)) or _raw(text) in table["state_abbreviations"]:
            found.add(position)
    return found


def is_state_word(text: str) -> bool:
    """Whether one word, in any case, is a one-word state or province name.

    Args:
        text: A word's surface.

    Returns:
        True for ``wisconsin`` or ``Florida``.
    """
    key = fold(text)
    return bool(key) and key in policy()["states"]


def country_runs(texts: Sequence[str]) -> list[tuple[int, int]]:
    """Every run of words that names a country, written as a proper noun, and not part of a state's name.

    Args:
        texts: The words, in order.

    Returns:
        ``(start, end)`` runs, in order.
    """
    table = policy()
    keys = [fold(text) for text in texts]
    states = state_positions(texts)
    runs = [
        (start, end)
        for start, end in _phrase_runs(keys, table["country_phrases"])
        if _capitalised(texts[start]) and not states & set(range(start, end))
    ]
    taken = {position for start, end in runs for position in range(start, end)}
    for position, (key, text) in enumerate(zip(keys, texts)):
        if position in taken or position in states:
            continue
        if (key in table["countries"] and _capitalised(text)) or _raw(text) in table["country_abbreviations"]:
            runs.append((position, position + 1))
    return sorted(runs)


def kinship_positions(texts: Sequence[str]) -> set[int]:
    """Positions that are a relationship word ("mom", "my brother", "abuela").

    Args:
        texts: The words, in order.

    Returns:
        The positions; a hyphenated in-law ("mother-in-law") is one word.
    """
    table = policy()
    return {position for position, text in enumerate(texts) if fold(text).replace("-", "") in table["kinship"]}


def is_sentence_initial(texts: Sequence[str], position: int) -> bool:
    """Whether a word opens the transcript or follows a sentence end.

    Args:
        texts: The words, in order.
        position: The word's index.

    Returns:
        True where nothing precedes it or the word before ends with a sentence terminator.
    """
    return position == 0 or str(texts[position - 1]).strip().endswith(_SENTENCE_END)
