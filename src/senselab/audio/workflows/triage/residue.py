"""The lexical residue: the words a recording carries that are neither task content nor non-lexical.

One reading per recording, and the only text the PII pathway sees: SPEECH scans it, REDACT verifies
against it and REVIEW reads it. Every other word of the transcript is one of two things. It is
non-lexical: a syllable train, a vocalisation, a filler, a truncated fragment or a bracketed marker.
Or it is lexical content the task asked for: the stimulus, aligned as a sequence, together with
its recognition variants and the task's declared vocabulary. The design and the replay behind each
rule are in ``specs/20260925-lexical-only-pii-pathway/design.md``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from senselab.audio.workflows.audio_analysis.harmonize import align_pair, normalise_token
from senselab.audio.workflows.triage.nodes.branches import (
    AIRWAY_EXPECTATIONS,
    SPEECH_EXPECTATIONS,
    VOICE_EXPECTATIONS,
    Expectation,
    expected_names,
)
from senselab.audio.workflows.triage.nodes.gates import Pattern
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match

if TYPE_CHECKING:  # pragma: no cover
    from senselab.audio.workflows.triage.config import TriageConfig

__all__ = [
    "ALIGNED",
    "FREE",
    "FUNCTION_WORDS",
    "SYLLABLE",
    "VOCAL",
    "Residue",
    "ResidueRule",
    "is_non_lexical",
    "residue_method",
    "residue_rule",
    "task_residue",
]

_SECTION = "stimulus.residue"

SYLLABLE = "syllable_train"
"""A syllable-repetition task: a syllable train in any spelling or script is the task."""

VOCAL = "vocal_task"
"""A task whose production is a sound, not words: its declared words and any vocalisation are the task."""

ALIGNED = "stimulus_alignment"
"""A task with declared words: the transcript is aligned to them as a sequence."""

FREE = "free_response"
"""A task that asks the participant to speak freely: every lexical word is residue."""

_VOCALISATION = re.compile(
    r"(?:u+h+|u+m+|h*m{2,}|hm+|m+h+m+|e+r+m*|e+h+|a+h+|a{2,}|o+h+|o{2,}h*|e{2,}|i{2,}|u{2,}|h+u+h+)"
)
"""A vocalisation or filler, whole-token: ``uh``, ``umm``, ``hmm``, ``ahh``, ``aaaah``, ``oh``, ``ooh``."""

_INTERJECTIONS = frozenset("啊哦嗯呃哈呼诶哎嘿咦耶唉哼噢喔呀嘻呵")
"""Single-character interjections a recognizer writes for a vocalisation in Chinese script."""

_BRACKETED = re.compile(r"[\[(<][^\])>]*[\])>]")

_CONSONANTS_OF_A_VOCAL_TASK = frozenset("hmwy")
"""Letters a vocalisation may carry beside its vowels in a task whose production is a sound."""

FUNCTION_WORDS = frozenset(
    """
    a an the this that these those it its it's i me my mine we us our you your he him his she her they them
    their there here and or but nor so if then than as at by for from in into of off on onto out over to up
    with without about after before under upon is am are was were be been being do does did done have has
    had having will would shall should can could may might must not no yes yeah okay ok oh well just also
    too very what which who whom whose when where why how all any some each every both
    let's that's there's what's i'm i'll i've i'd you're you'll he's she's we're they're don't doesn't
    didn't can't won't isn't aren't wasn't weren't
    el la los las un una unos unas y o de del al en con por para que se lo le es no si mi tu su
    """.split()
)
"""Closed-class words: a residue made of nothing else carries no content a detector could find."""

_NUMERALS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
}
"""A digit string the stimulus spells as a word, for comparison only."""

_CLITICS = ("n't", "'s", "'re", "'ll", "'ve", "'d", "'m", "s'", "'")

_SYLLABLE_LETTERS = {
    "p": "pb",
    "b": "bp",
    "t": "tdr",
    "d": "dtr",
    "k": "kcgqx",
    "g": "gkc",
    "er": "r",
}
"""The letters a recognizer writes for each ARPAbet consonant of a syllable template."""

_VOWEL_LETTERS = frozenset("aeiouyw")

_SKELETON_CLASSES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


@dataclass(frozen=True)
class ResidueRule:
    """How close a transcript word must be to a stimulus word to be a reading of it.

    Attributes:
        near: The stimulus near-match bound every other stimulus test uses.
        variant_similarity_min: The least ``1 - edit distance / longer length`` at which an aligned
            substitution, or an inserted word, is a recognition variant of a stimulus word.
        insertion_window: How many stimulus words either side of an insertion it may be a
            repetition, restart, split or variant of.
        train_repetitions_min: How many times one token must recur in a syllable task's transcript
            to be read as the train whatever its spelling or script.
    """

    near: NearMatch
    variant_similarity_min: float
    insertion_window: int
    train_repetitions_min: int


@dataclass(frozen=True)
class Residue:
    """Which of a recording's lexical words the PII pathway reads.

    Attributes:
        positions: Indices into the word sequence the residue was computed over, in order.
        method: :data:`SYLLABLE`, :data:`VOCAL`, :data:`ALIGNED` or :data:`FREE`.
        non_lexical_n: How many words were set aside as non-lexical.
        task_n: How many words were set aside as task content.
        content: Whether any residue word is outside :data:`FUNCTION_WORDS`; a residue without one
            is not scanned.
    """

    positions: tuple[int, ...]
    method: str
    non_lexical_n: int
    task_n: int
    content: bool


def residue_rule(config: "TriageConfig") -> ResidueRule:
    """The run's residue rule, read from its config.

    Args:
        config: The triage configuration.

    Returns:
        The rule.

    Raises:
        ValueError: If any key is unmeasured.
    """
    return ResidueRule(
        near=near_match(config),
        variant_similarity_min=float(config.require(f"{_SECTION}.variant_similarity_min")),
        insertion_window=int(config.require(f"{_SECTION}.insertion_window")),
        train_repetitions_min=int(config.require(f"{_SECTION}.train_repetitions_min")),
    )


def _expectation(task_family: str | None) -> Expectation | None:
    family = str(task_family)
    for table in (SPEECH_EXPECTATIONS, VOICE_EXPECTATIONS, AIRWAY_EXPECTATIONS):
        if family in table:
            return table[family]
    return None


def residue_method(task_family: str | None) -> str:
    """Which residue method a declared family is read with.

    Args:
        task_family: The declared family, or None.

    Returns:
        :data:`SYLLABLE` for a syllable-repetition family; :data:`VOCAL` for a VOICE or AIRWAY
        family and for a SPEECH family whose words are literal tokens rather than a stimulus text;
        :data:`FREE` for a free response, an item list or an undeclared family; :data:`ALIGNED`
        otherwise. :func:`task_residue` reads an undeclared family that declares prompt text as
        :data:`ALIGNED`.
    """
    expectation = _expectation(task_family)
    if expectation is None:
        return FREE
    if expectation.pattern in (Pattern.SYLLABLE_TRAIN, Pattern.SYLLABLE_SEQUENCE):
        return SYLLABLE
    if expectation.pattern in (Pattern.FREE_RESPONSE, Pattern.ITEM_LIST):
        return FREE
    if str(task_family) not in SPEECH_EXPECTATIONS or (expectation.tokens and expectation.token_source is None):
        return VOCAL
    return ALIGNED


def _pieces(text: str) -> list[str]:
    return [key for key in (normalise_token(piece) for piece in re.split(r"[-‐-―\s]+", text)) if key]


def _stripped(text: str) -> str:
    return str(text).strip().strip('.,;:!?"“”‘’()¿¡…')


def is_non_lexical(text: str, *, vocal_task: bool = False) -> bool:
    """Whether a transcript token is a vocalisation, a filler, a fragment or a bracketed marker.

    Args:
        text: The token as the recognizer wrote it.
        vocal_task: Whether the task's production is a sound, in which case any token whose letters
            are vowels and ``h``, ``m``, ``w`` or ``y`` alone is a vocalisation.

    Returns:
        True when the token is a bracketed marker; a single interjection character; a fragment of at
        most two letters cut off by a trailing hyphen; or a token every hyphen-separated piece of
        which is a vocalisation.
    """
    raw = str(text).strip()
    if _BRACKETED.fullmatch(raw.rstrip(".,;:!?")):
        return True
    stripped = _stripped(raw)
    if stripped and all(ch in _INTERJECTIONS for ch in stripped):
        return True
    pieces = _pieces(stripped)
    if not pieces:
        return False
    if stripped.endswith("-") and len(pieces) == 1 and len(pieces[0]) <= 2:
        return True
    if all(_VOCALISATION.fullmatch(piece) for piece in pieces):
        return True
    return vocal_task and all(
        piece.isalpha() and all(ch in _VOWEL_LETTERS or ch in _CONSONANTS_OF_A_VOCAL_TASK for ch in piece)
        for piece in pieces
    )


def _syllable_letters(sequence: Sequence[str]) -> frozenset[str]:
    return frozenset(letter for phoneme in sequence for letter in _SYLLABLE_LETTERS.get(phoneme, ""))


def _is_syllable_piece(piece: str, consonants: frozenset[str]) -> bool:
    if not piece.isalpha():
        return False
    if not piece.isascii():
        return len(piece) <= 4
    if not any(ch in _VOWEL_LETTERS for ch in piece):
        return len(piece) == 1 and piece in consonants
    return all(
        ch in _VOWEL_LETTERS or ch in consonants or (ch == "h" and position > 0) for position, ch in enumerate(piece)
    )


def _is_periodic(text: str, repetitions_min: int) -> bool:
    """Whether a unit of at most four characters recurs ``repetitions_min`` times and covers most of it."""
    joined = "".join(_pieces(text))
    for width in range(1, 5):
        for unit in {joined[start : start + width] for start in range(len(joined) - width + 1)}:
            found = joined.count(unit)
            if found >= repetitions_min and 2 * found * width > len(joined):
                return True
    return False


def _is_syllable_token(text: str, consonants: frozenset[str], repetitions_min: int) -> bool:
    pieces = _pieces(text)
    if not pieces:
        return False
    if _is_periodic(text, repetitions_min):
        return True
    return all(_is_syllable_piece(piece, consonants) for piece in pieces)


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    longer = max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return 1.0 - previous[-1] / longer


def _skeleton(key: str) -> str:
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", key) if ch.isascii() and ch.isalpha())
    classes = [_SKELETON_CLASSES.get(ch, "") for ch in folded]
    out: list[str] = []
    for mark in classes:
        if mark and (not out or out[-1] != mark):
            out.append(mark)
    return "".join(out)


def _key(text: str) -> str:
    key = normalise_token(text)
    return _NUMERALS.get(key, key)


def _base(key: str) -> str:
    for clitic in _CLITICS:
        if key.endswith(clitic) and len(key) > len(clitic):
            return key[: -len(clitic)]
    return key


def _variant(word: str, expected: str, rule: ResidueRule) -> bool:
    if rule.near.matches(word, expected) or _similarity(word, expected) >= rule.variant_similarity_min:
        return True
    if len(word) >= 2 and expected.startswith(word):
        return True
    skeleton = _skeleton(word)
    return len(skeleton) >= 2 and skeleton == _skeleton(expected)


def _declared_sequence(task_family: str | None, stimulus: Sequence[str]) -> list[str]:
    keys = [_key(token) for token in stimulus]
    expectation = _expectation(task_family)
    if not any(keys) and expectation is not None and expectation.tokens:
        keys = [_key(token) for token in expectation.tokens]
    return [key for key in keys if key]


def _declared_vocabulary(task_family: str | None) -> frozenset[str]:
    expectation = _expectation(task_family)
    vocabulary = [*(expectation.vocabulary or () if expectation is not None else ()), *expected_names(task_family)]
    if expectation is not None and expectation.tokens:
        vocabulary.extend(expectation.tokens)
    return frozenset(key for key in (_key(word) for word in vocabulary) if key)


def _accounted(word: str, window: Sequence[str], before: str, after: str, rule: ResidueRule) -> bool:
    if any(_variant(word, candidate, rule) for candidate in window):
        return True
    base = _base(word)
    if base != word and any(_variant(base, candidate, rule) for candidate in window):
        return True
    joined = [window[k] + window[k + 1] for k in range(len(window) - 1)]
    if word in joined:
        return True
    if (before and before + word in window) or (after and word + after in window):
        return True
    return any(
        _variant(pair, candidate, rule) for pair in (before + word, word + after) if pair for candidate in joined
    )


def _aligned_residue(keys: Sequence[str], expected: Sequence[str], rule: ResidueRule) -> set[int]:
    """The positions of ``keys`` the aligned stimulus does not account for."""
    if not expected:
        return set(range(len(keys)))
    residue: set[int] = set()
    anchor = -1
    declared = set(expected)
    for i, j in align_pair(list(expected), list(keys)):
        if i is not None:
            anchor = i
        if j is None:
            continue
        word = keys[j]
        if (i is not None and word == expected[i]) or word in declared:
            continue
        low = max(0, anchor - rule.insertion_window + (1 if i is None else 0))
        window = expected[low : min(len(expected), anchor + rule.insertion_window + 1)]
        before = keys[j - 1] if j > 0 else ""
        after = keys[j + 1] if j + 1 < len(keys) else ""
        if not _accounted(word, window, before, after, rule):
            residue.add(j)
    return residue


def task_residue(texts: Sequence[str], task_family: str | None, stimulus: Sequence[str], rule: ResidueRule) -> Residue:
    """Which transcript words are neither non-lexical nor what the task asked for.

    Args:
        texts: The words' texts, in stream order.
        task_family: The declared family, or None.
        stimulus: The recording's declared prompt tokens, verbatim, in declared order.
        rule: The residue rule.

    Returns:
        The residue over ``texts``.
    """
    method = residue_method(task_family)
    if method == FREE and _expectation(task_family) is None and any(_key(token) for token in stimulus):
        method = ALIGNED
    vocal = method == VOCAL
    lexical = [
        position for position, text in enumerate(texts) if _key(text) and not is_non_lexical(text, vocal_task=vocal)
    ]
    non_lexical_n = len(texts) - len(lexical)
    if method == SYLLABLE:
        expectation = _expectation(task_family)
        consonants = _syllable_letters(expectation.sequence or () if expectation is not None else ())
        recurring = {
            key
            for key in {_key(texts[position]) for position in lexical}
            if sum(1 for position in lexical if _key(texts[position]) == key) >= rule.train_repetitions_min
        }
        remaining = [
            position
            for position in lexical
            if _key(texts[position]) not in recurring
            and not _is_syllable_token(texts[position], consonants, rule.train_repetitions_min)
        ]
        non_lexical_n += len(lexical) - len(remaining)
        lexical = remaining
    task: set[int] = set()
    if method != FREE:
        vocabulary = _declared_vocabulary(task_family)
        keys = [_key(texts[position]) for position in lexical]
        in_vocabulary = {
            index for index, key in enumerate(keys) if any(rule.near.matches(key, word) for word in vocabulary)
        }
        expected = _declared_sequence(task_family, stimulus)
        flagged = _aligned_residue(keys, expected, rule) if expected else set(range(len(keys)))
        task = {index for index in range(len(keys)) if index not in flagged or index in in_vocabulary}
    kept = tuple(position for index, position in enumerate(lexical) if index not in task)
    content = any(_base(_key(texts[position])) not in FUNCTION_WORDS for position in kept)
    return Residue(kept, method, non_lexical_n, len(task), content)
