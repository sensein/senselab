"""Where a detector finding sits among the transcript's words.

A detector reads one scanned text and returns each finding's own text. :func:`place` matches that
text against the scanned tokens -- every occurrence, as a run of whole tokens compared after
:func:`match_key`, so ``Alan`` places on ``Alan's`` and ``ninety-three`` on ``ninety three`` -- and
never extends a run past its match. A run is then cut to the words that carry the finding's kind
(:func:`cut`): a date, time or age finding keeps its temporal words, and a name, place or
organisation finding longer than ``pii.name_words_max`` keeps its proper nouns. A finding whose
text matches no run is unplaced and covers no word. The rule and its measurement are in
``specs/20260927-mask-placement-and-second-speaker/design.md``, section 9.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml

from senselab.audio.workflows.triage.nodes.redact import category_family
from senselab.audio.workflows.triage.residue import is_content_word, is_proper_form

TEMPORAL_PATH = Path(__file__).parent / "data" / "temporal_words.yaml"

TEMPORAL_FAMILIES = frozenset({"DATE_TIME"})
"""The families whose findings are cut to their temporal words."""

NAME_FAMILIES = frozenset({"PERSON", "LOCATION", "ORGANIZATION"})
"""The families whose over-long findings are cut to their proper nouns."""

_EDGE = "\"'`.,;:!?()[]{}<>«»“”‘’…-–—"
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_POSSESSIVE = re.compile(r"(?:'s|s')$")


def match_key(token: str) -> str:
    """A token as a finding's text and a transcript word are compared.

    Args:
        token: A scanned token, or one whitespace-separated piece of a finding's text.

    Returns:
        Casefolded, apostrophes straight, edge punctuation, a trailing possessive and every internal
        hyphen and apostrophe removed. Empty where nothing but punctuation was there.
    """
    key = token.translate(_APOSTROPHES).casefold().strip(_EDGE)
    key = _POSSESSIVE.sub("", key)
    return key.replace("-", "").replace("'", "").replace("‐", "")


def locate(text: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    """Every place a finding's text matches a run of whole tokens.

    The finding's keys, joined, must equal the joined keys of a contiguous run of tokens, so a
    split or a joined spelling places and a part of a word does not.

    Args:
        text: The detector's own text for the finding.
        tokens: The scanned text's tokens, in order.

    Returns:
        ``[(first, last), ...]`` into ``tokens``, non-overlapping and in order; empty where it matches
        nowhere.
    """
    target = "".join(match_key(piece) for piece in text.split())
    keys = [match_key(token) for token in tokens]
    runs: list[tuple[int, int]] = []
    if not target:
        return runs
    start = 0
    while start < len(keys):
        if not keys[start] or not target.startswith(keys[start]):
            start += 1
            continue
        joined = ""
        end = start
        while end < len(keys) and len(joined) < len(target):
            joined += keys[end]
            end += 1
        if joined == target:
            last = end - 1
            while last > start and not keys[last]:
                last -= 1
            runs.append((start, last))
            start = end
        else:
            start += 1
    return runs


@lru_cache(maxsize=1)
def _temporal() -> tuple[frozenset[str], tuple[str, ...]]:
    raw = yaml.safe_load(TEMPORAL_PATH.read_text()) or {}
    numbers = tuple(sorted({str(word) for word in raw.get("number_words") or ()}, key=len, reverse=True))
    words = {str(word) for key in ("units", "calendar", "relative") for word in raw.get(key) or ()}
    return frozenset(words | set(numbers)), numbers


def _number_compound(key: str, numbers: Sequence[str]) -> bool:
    if not key:
        return False
    if key in numbers:
        return True
    return any(key.startswith(word) and _number_compound(key[len(word) :], numbers) for word in numbers)


def is_temporal(token: str) -> bool:
    """Whether a token is a date, time or age word.

    Args:
        token: A transcript token.

    Returns:
        True for a token holding a digit, a number word or a compound of them (``ninetythree``), or a
        word of ``data/temporal_words.yaml``.
    """
    key = match_key(token)
    if not key:
        return False
    words, numbers = _temporal()
    return any(ch.isdigit() for ch in key) or key in words or _number_compound(key, numbers)


def _runs(kept: Sequence[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for index in kept:
        if out and index == out[-1][1] + 1:
            out[-1] = (out[-1][0], index)
        else:
            out.append((index, index))
    return out


def cut(
    run: tuple[int, int], tokens: Sequence[str], category: str, name_words_max: int
) -> tuple[list[tuple[int, int]], bool]:
    """The parts of a placed run that carry the finding's kind.

    A date, time or age finding keeps its temporal tokens and any other token lying between two of
    them; one with no temporal token keeps its run. A name, place or organisation finding longer than
    ``name_words_max`` tokens keeps its proper nouns (:func:`~senselab.audio.workflows.triage.residue.is_proper_form`);
    one with none keeps its run. Any other finding keeps its run.

    Args:
        run: ``(first, last)`` into ``tokens``.
        tokens: The scanned tokens.
        category: The finding's category.
        name_words_max: ``pii.name_words_max``.

    Returns:
        ``(runs, cut)``: the contiguous runs kept, and whether anything was cut.
    """
    first, last = run
    span = list(range(first, last + 1))
    family = category_family(category)
    if family in TEMPORAL_FAMILIES:
        marked = [index for index in span if is_temporal(tokens[index])]
        if not marked:
            return [run], False
        kept = [index for index in span if marked[0] <= index <= marked[-1]]
        kept = [
            index
            for index in kept
            if is_temporal(tokens[index])
            or not is_content_word(tokens[index])
            and any(m < index for m in marked)
            and any(m > index for m in marked)
        ]
    elif family in NAME_FAMILIES and len(span) > name_words_max:
        kept = [index for index in span if is_proper_form(tokens[index], tokens[index - 1] if index > 0 else None)]
        if not kept:
            return [run], False
    else:
        return [run], False
    runs = _runs(kept)
    return runs, runs != [run]
