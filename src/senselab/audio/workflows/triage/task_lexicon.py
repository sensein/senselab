"""A task's own vocabulary: the words and phrases a faithful performance is made of.

One definition read by three consumers, so they cannot drift: the residue (task words never reach the
PII detectors), REDACT's stimulus exemption, and the fold's mask plan (no mask covers a task word). A
family's lexicon is the names its expectation row declares (``expected_names``) plus the file its task list entry
names (``lexicon``) and, for a word-generation
task, the words of the category the recording's hint declares, by the rule the entry gives that category
(:class:`~senselab.audio.workflows.triage.task_catalogue.ItemCategory`). senselab names no file and no category: both
are the dataset profile's. The derivation of each packaged list is in the file itself and in
``specs/20261022-triage-dataset-profile/plan.md`` (step 4) and
``specs/20260928-reviewer-contradictions-and-task-lexicon/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.audio_analysis.harmonize import normalise_token
from senselab.audio.workflows.triage.dataset_profile import data_path, profile_cache
from senselab.audio.workflows.triage.nodes.branches import expected_names, item_category
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match
from senselab.audio.workflows.triage.task_catalogue import (
    INITIAL_LETTER_RULE,
    LETTERS_RULE,
    NUMBERS_RULE,
    ItemCategory,
    has_own_words,
    item_category_rule,
    task_lexicon_file,
)

if TYPE_CHECKING:
    from senselab.audio.workflows.triage.config import TriageConfig

SECTIONS = ("characters", "objects", "events", "time")
"""The lists a lexicon file carries; every entry of each is a word or a phrase."""

_POSSESSIVE = ("'s", "s'", "'")
_PLURAL = ("es", "s")


def _key(text: str) -> str:
    return normalise_token(text).replace("’", "'")


@profile_cache
def packaged_phrases(family: str) -> tuple[tuple[str, ...], ...]:
    """The phrases of the lexicon file a family's task list entry names, each as its normalised tokens.

    Args:
        family: The declared task family.

    Returns:
        One token tuple per entry, in file order; empty where the entry names no file.

    Raises:
        ValueError: If the file names a different family.
    """
    relative = task_lexicon_file(family)
    if relative is None:
        return ()
    path = data_path(relative)
    document = yaml.safe_load(path.read_text()) or {}
    if str(document.get("family")) != family:
        raise ValueError(f"{path} declares family {document.get('family')!r}, not {family!r}")
    phrases: list[tuple[str, ...]] = []
    for section in SECTIONS:
        for entry in document.get(section) or ():
            tokens = tuple(key for key in (_key(token) for token in str(entry).split()) if key)
            if tokens and tokens not in phrases:
                phrases.append(tokens)
    return tuple(phrases)


@profile_cache
def category_phrases(category: ItemCategory | None) -> tuple[tuple[str, ...], ...]:
    """The phrases of a ``list`` category's words file, each as its normalised tokens.

    Args:
        category: The category, as the task's entry gives it, or None.

    Returns:
        One token tuple per item; empty for None or a category that is a rule rather than a list.

    Raises:
        ValueError: If the file names a different category.
    """
    if category is None or category.words is None:
        return ()
    path = data_path(category.words)
    document = yaml.safe_load(path.read_text()) or {}
    if str(document.get("category")) != category.name:
        raise ValueError(f"{path} declares category {document.get('category')!r}, not {category.name!r}")
    phrases: list[tuple[str, ...]] = []
    for entry in document.get("items") or ():
        tokens = tuple(key for key in (_key(token) for token in str(entry).split()) if key)
        if tokens and tokens not in phrases:
            phrases.append(tokens)
    return tuple(phrases)


def rule_member(category: ItemCategory | None, key: str) -> bool:
    """Whether one normalised word belongs to a category that is a rule rather than a list.

    Args:
        category: The category, as the task's entry gives it, or None.
        key: The word, normalised.

    Returns:
        True for a single letter under a ``letters`` rule, a digit string or one of the category's number words
        under a ``numbers`` rule, and a word with the category's letter under an ``initial_letter`` rule.
    """
    if category is None or not key:
        return False
    if category.rule == LETTERS_RULE:
        return len(key) == 1 and key.isalpha()
    if category.rule == NUMBERS_RULE:
        return key.isdigit() or key in category.number_words
    return category.rule == INITIAL_LETTER_RULE and category.letter is not None and key.startswith(category.letter)


def category_members(category: ItemCategory | None, texts: Sequence[str]) -> set[int]:
    """Which listed words are members of the recording's category.

    Args:
        category: The category, as the task's entry gives it, or None.
        texts: The words' surfaces, in stream order.

    Returns:
        The indices of every word a rule admits or a whole-run match of a listed item covers; empty
        where there is no category.
    """
    if category is None:
        return set()
    return TaskLexicon(None, category_phrases(category), rule=category).positions(texts)


@dataclass(frozen=True)
class TaskLexicon:
    """A family's task vocabulary, and how a transcript word is compared against it.

    Attributes:
        family: The declared task family, or None.
        phrases: One normalised token tuple per word or phrase.
        near: The stimulus near-match applied to a single token; None compares exactly.
        rule: An item category whose members are admitted by :func:`rule_member`, or None.
    """

    family: str | None
    phrases: tuple[tuple[str, ...], ...]
    near: NearMatch | None = None
    rule: ItemCategory | None = None

    def _token(self, key: str, expected: str) -> bool:
        if not key:
            return False
        if key == expected:
            return True
        for suffix in (*_POSSESSIVE, *_PLURAL):
            if key.endswith(suffix) and key[: -len(suffix)] == expected:
                return True
        return self.near is not None and self.near.matches(key, expected)

    def positions(self, texts: Sequence[str]) -> set[int]:
        """Which words are the task's own vocabulary, as whole-run matches of any phrase.

        Args:
            texts: The words' surfaces, in stream order.

        Returns:
            The indices of every word inside a matched run.
        """
        keys = [_key(text) for text in texts]
        hit: set[int] = {index for index, key in enumerate(keys) if rule_member(self.rule, key)}
        for start in range(len(keys)):
            for phrase in self.phrases:
                end = start + len(phrase)
                if end <= len(keys) and all(self._token(keys[start + i], token) for i, token in enumerate(phrase)):
                    hit.update(range(start, end))
        return hit

    def texts(self) -> list[str]:
        """The phrases as plain strings, for the reviewer's context."""
        rule = [f"any item of the category {self.rule.name}"] if self.rule and not self.phrases else []
        return [*(" ".join(phrase) for phrase in self.phrases), *rule]


def recording_category(family: str | None, hint: AudioHints | None) -> ItemCategory | None:
    """The rule of the category a recording of a word-generation task asks for.

    Args:
        family: The declared task family, or None.
        hint: What the recording was declared to contain.

    Returns:
        The task entry's category of the name :func:`~senselab.audio.workflows.triage.nodes.branches.item_category`
        reads; None where there is none, or the entry does not list it.
    """
    return item_category_rule(family, item_category(family, hint))


def declared_names_lexicon(family: str | None) -> TaskLexicon:
    """The lexicon of a family's declared names alone, compared exactly.

    Args:
        family: The declared task family, or None.

    Returns:
        The names as single-phrase entries; empty where the family declares none.
    """
    phrases = tuple(
        tokens
        for tokens in (tuple(key for key in (_key(t) for t in name.split()) if key) for name in expected_names(family))
        if tokens
    )
    return TaskLexicon(family, phrases)


def task_lexicon(config: "TriageConfig", family: str | None, hint: AudioHints | None = None) -> TaskLexicon:
    """The run's task lexicon for one family.

    Args:
        config: The triage configuration, for the near-match.
        family: The declared task family, or None.
        hint: What the recording was declared to contain, for a category named per recording.

    Returns:
        The declared names plus, where the task's entry names its own words (a lexicon file or categories), the
        lexicon file's phrases and the item category's, with the run's near-match.
    """
    names = declared_names_lexicon(family).phrases
    on = has_own_words(family)
    packaged = packaged_phrases(str(family)) if on else ()
    category = recording_category(family, hint) if on else None
    items = category_phrases(category)
    phrases = tuple(dict.fromkeys((*names, *packaged, *items)))
    return TaskLexicon(family, phrases, near_match(config), rule=category)
