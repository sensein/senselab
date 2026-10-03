"""The health conditions a study recruits for, read from a packaged profile.

A reviewer ``redact`` entry in a human-review category (``verdict.llm_human_review_categories``) is
a **cohort** condition where its text matches one of the profile's patterns, and an **other**
condition where it does not. The profile is named by ``verdict.cohort_conditions`` and lives in
``data/cohort_conditions/<name>.yaml``; its derivation is in
``specs/20260927-pii-span-ledger/design.md``, section 10.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROFILE_DIR = Path(__file__).parent / "data" / "cohort_conditions"

COHORT = "cohort"
"""A condition the study recruits for."""

OTHER = "other"
"""A condition outside the study's cohorts."""

CONDITION_KINDS = (COHORT, OTHER)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def _normalise(text: str) -> str:
    """A proposal's text as the patterns are searched in: lower-cased, straight apostrophes, one space."""
    return " ".join(str(text).translate(_APOSTROPHES).lower().split())


@dataclass(frozen=True)
class CohortProfile:
    """One study's cohort conditions.

    Attributes:
        name: The profile's file stem.
        release: The data release the list was read from.
        conditions: Diagnosis name to its compiled patterns, in file order.
    """

    name: str
    release: str
    conditions: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]

    def diagnosis(self, text: str) -> str | None:
        """The first cohort diagnosis a proposal's text matches.

        Args:
            text: The reviewer's quote.

        Returns:
            The diagnosis name, or None where no pattern matches.
        """
        normalised = _normalise(text)
        for name, patterns in self.conditions:
            if any(pattern.search(normalised) for pattern in patterns):
                return name
        return None

    def kind(self, text: str) -> str:
        """Whether a proposal's text names a cohort condition.

        Args:
            text: The reviewer's quote.

        Returns:
            :data:`COHORT` or :data:`OTHER`.
        """
        return COHORT if self.diagnosis(text) is not None else OTHER


@lru_cache(maxsize=8)
def load_cohort_profile(name: str) -> CohortProfile:
    """The packaged profile of this name.

    Args:
        name: The file stem under :data:`PROFILE_DIR`.

    Returns:
        The profile.

    Raises:
        FileNotFoundError: If no such profile is packaged.
        ValueError: If the file does not carry a ``conditions`` mapping of pattern lists.
    """
    path = PROFILE_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no cohort profile named {name!r} under {PROFILE_DIR}")
    data = yaml.safe_load(path.read_text()) or {}
    conditions = data.get("conditions")
    if not isinstance(conditions, dict) or not all(isinstance(v, list) for v in conditions.values()):
        raise ValueError(f"cohort profile {name!r} carries no conditions mapping of pattern lists")
    return CohortProfile(
        name=name,
        release=str(data.get("release") or ""),
        conditions=tuple(
            (str(diagnosis), tuple(re.compile(str(pattern), re.IGNORECASE) for pattern in patterns))
            for diagnosis, patterns in conditions.items()
        ),
    )
