"""A task's own vocabulary: the words and phrases a faithful performance is made of.

One definition read by three consumers, so they cannot drift: the residue (task words never reach the
PII detectors), REDACT's stimulus exemption, and the fold's mask plan (no mask covers a task word). A
family's lexicon is the names its expectation row declares (``expected_names``) plus, where the run's
``stimulus.task_lexicons`` enables it, the packaged ``data/task_lexicon/<family>.yaml``. The
derivation of each packaged list is in the file itself and in
``specs/20260928-reviewer-contradictions-and-task-lexicon/design.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.audio_analysis.harmonize import normalise_token
from senselab.audio.workflows.triage.nodes.branches import expected_names, item_category
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match

if TYPE_CHECKING:
    from senselab.audio.workflows.triage.config import TriageConfig

LEXICON_DIR = Path(__file__).parent / "data" / "task_lexicon"
"""Where the packaged lexicons live, one ``<family>.yaml`` each."""

SECTIONS = ("characters", "objects", "events", "time")
"""The lists a lexicon file carries; every entry of each is a word or a phrase."""

CATEGORY_DIR = LEXICON_DIR / "categories"
"""Where the packaged item-category lexicons live, one ``<slug>.yaml`` each, carrying ``category`` and ``items``."""

LETTERS = "Letters"
NUMBERS = "Numbers"
INITIAL_LETTER = re.compile(r"words starting with '(\w)'", re.IGNORECASE)
"""A category whose members are the words that begin with one letter."""

NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million".split()
)
"""The spelled numbers a ``Numbers`` list is made of, beside the digits."""

_CONFIG_KEY = "stimulus.task_lexicons"
_POSSESSIVE = ("'s", "s'", "'")
_PLURAL = ("es", "s")


def _key(text: str) -> str:
    return normalise_token(text).replace("’", "'")


@lru_cache(maxsize=None)
def packaged_phrases(family: str) -> tuple[tuple[str, ...], ...]:
    """The phrases of a family's packaged lexicon file, each as its normalised tokens.

    Args:
        family: The declared task family.

    Returns:
        One token tuple per entry, in file order; empty where no file exists for the family.

    Raises:
        ValueError: If the file names a different family.
    """
    path = LEXICON_DIR / f"{family}.yaml"
    if not path.is_file():
        return ()
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


def category_slug(category: str) -> str:
    """The file stem a category's lexicon is packaged under: ``Country names`` is ``country-names``."""
    return re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")


@lru_cache(maxsize=None)
def category_phrases(category: str) -> tuple[tuple[str, ...], ...]:
    """The phrases of a category's packaged lexicon, each as its normalised tokens.

    Args:
        category: The category, as the instructions name it.

    Returns:
        One token tuple per item; empty where no file is packaged for the category.

    Raises:
        ValueError: If the file names a different category.
    """
    path = CATEGORY_DIR / f"{category_slug(category)}.yaml"
    if not path.is_file():
        return ()
    document = yaml.safe_load(path.read_text()) or {}
    if str(document.get("category")) != category:
        raise ValueError(f"{path} declares category {document.get('category')!r}, not {category!r}")
    phrases: list[tuple[str, ...]] = []
    for entry in document.get("items") or ():
        tokens = tuple(key for key in (_key(token) for token in str(entry).split()) if key)
        if tokens and tokens not in phrases:
            phrases.append(tokens)
    return tuple(phrases)


def rule_member(category: str | None, key: str) -> bool:
    """Whether one normalised word belongs to a category that is a rule rather than a list.

    Args:
        category: The category, or None.
        key: The word, normalised.

    Returns:
        True for a single letter under ``Letters``, a digit string or spelled number under ``Numbers``,
        and a word with the named initial under ``words starting with '<letter>'``.
    """
    if not category or not key:
        return False
    if category == LETTERS:
        return len(key) == 1 and key.isalpha()
    if category == NUMBERS:
        return key.isdigit() or key in NUMBER_WORDS
    initial = INITIAL_LETTER.search(category)
    return initial is not None and key.startswith(initial.group(1).lower())


def category_members(category: str | None, texts: Sequence[str]) -> set[int]:
    """Which listed words are members of the recording's category.

    Args:
        category: The category, or None.
        texts: The words' surfaces, in stream order.

    Returns:
        The indices of every word a rule admits or a whole-run match of a packaged item covers; empty
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
    rule: str | None = None

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
        rule = [f"any item of the category {self.rule}"] if self.rule and not self.phrases else []
        return [*(" ".join(phrase) for phrase in self.phrases), *rule]


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
        config: The triage configuration; ``stimulus.task_lexicons`` lists the families whose packaged
            file, and whose item category's packaged file or rule, is read.
        family: The declared task family, or None.
        hint: What the recording was declared to contain, for a category named per recording.

    Returns:
        The declared names plus, where enabled, the packaged file's phrases and the item category's, with
        the run's near-match.
    """
    names = declared_names_lexicon(family).phrases
    enabled = {str(name) for name in (config.get(_CONFIG_KEY) or ())}
    on = bool(family) and str(family) in enabled
    packaged = packaged_phrases(str(family)) if on else ()
    category = item_category(family, hint) if on else None
    items = category_phrases(category) if category else ()
    phrases = tuple(dict.fromkeys((*names, *packaged, *items)))
    return TaskLexicon(family, phrases, near_match(config), rule=category)
