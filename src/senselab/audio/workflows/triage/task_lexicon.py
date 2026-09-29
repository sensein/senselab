"""A task's own vocabulary: the words and phrases a faithful performance is made of.

One definition read by three consumers, so they cannot drift: the residue (task words never reach the
PII detectors), REDACT's stimulus exemption, and the fold's mask plan (no mask covers a task word). A
family's lexicon is the names its expectation row declares (``expected_names``) plus, where the run's
``stimulus.task_lexicons`` enables it, the packaged ``data/task_lexicon/<family>.yaml``. The
derivation of each packaged list is in the file itself and in
``specs/20260928-reviewer-contradictions-and-task-lexicon/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import yaml

from senselab.audio.workflows.audio_analysis.harmonize import normalise_token
from senselab.audio.workflows.triage.nodes.branches import expected_names
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match

if TYPE_CHECKING:
    from senselab.audio.workflows.triage.config import TriageConfig

LEXICON_DIR = Path(__file__).parent / "data" / "task_lexicon"
"""Where the packaged lexicons live, one ``<family>.yaml`` each."""

SECTIONS = ("characters", "objects", "events", "time")
"""The lists a lexicon file carries; every entry of each is a word or a phrase."""

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


@dataclass(frozen=True)
class TaskLexicon:
    """A family's task vocabulary, and how a transcript word is compared against it.

    Attributes:
        family: The declared task family, or None.
        phrases: One normalised token tuple per word or phrase.
        near: The stimulus near-match applied to a single token; None compares exactly.
    """

    family: str | None
    phrases: tuple[tuple[str, ...], ...]
    near: NearMatch | None = None

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
        hit: set[int] = set()
        for start in range(len(keys)):
            for phrase in self.phrases:
                end = start + len(phrase)
                if end <= len(keys) and all(self._token(keys[start + i], token) for i, token in enumerate(phrase)):
                    hit.update(range(start, end))
        return hit

    def texts(self) -> list[str]:
        """The phrases as plain strings, for the reviewer's context."""
        return [" ".join(phrase) for phrase in self.phrases]


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


def task_lexicon(config: "TriageConfig", family: str | None) -> TaskLexicon:
    """The run's task lexicon for one family.

    Args:
        config: The triage configuration; ``stimulus.task_lexicons`` lists the families whose packaged
            file is read.
        family: The declared task family, or None.

    Returns:
        The declared names plus, where enabled, the packaged file's phrases, with the run's near-match.
    """
    names = declared_names_lexicon(family).phrases
    enabled = {str(name) for name in (config.get(_CONFIG_KEY) or ())}
    packaged = packaged_phrases(str(family)) if family and str(family) in enabled else ()
    phrases = tuple(dict.fromkeys((*names, *packaged)))
    return TaskLexicon(family, phrases, near_match(config))
