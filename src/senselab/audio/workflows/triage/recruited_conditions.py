"""The health conditions a study recruits for, read from the dataset profile.

A health condition the reviewer lists is a **recruited** condition where its text matches one of the list's
patterns, and an **other** condition where it does not; under redaction policy v7 the kind is recorded on the ledger
and moves neither axis. The list is named by ``verdict.recruited_conditions`` and lives in the dataset profile's
``recruited_conditions/<name>.yaml``; its derivation is in ``specs/20260927-pii-span-ledger/design.md``, section 10.

Until plan step 9 these were called the study's "cohort" conditions, which also names the COHORT stage. A store
written before then records the kind ``cohort`` and the keys ``cohort_diagnosis``, ``cohort_diagnoses`` and
``cohort_condition_n``; the readers here accept both (:func:`condition_kind_of`, :func:`ledger_value`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from senselab.audio.workflows.triage.dataset_profile import data_path, profile_cache

RECRUITED = "recruited"
"""A condition the study recruits for."""

OTHER = "other"
"""A condition outside the study's recruited conditions."""

CONDITION_KINDS = (RECRUITED, OTHER)

RECRUITED_CONDITIONS_DIR = "recruited_conditions"
"""The dataset-profile folder the lists live in, one ``<name>.yaml`` each."""

LEGACY_RECRUITED = "cohort"
"""The kind a store written before plan step 9 records for :data:`RECRUITED`."""

DIAGNOSIS_KEY = "recruited_diagnosis"
DIAGNOSES_KEY = "recruited_diagnoses"
"""The ledger keys of one condition's recruited diagnosis and of the recording's distinct ones."""

_LEGACY_KEYS = {
    DIAGNOSIS_KEY: "cohort_diagnosis",
    DIAGNOSES_KEY: "cohort_diagnoses",
    f"{RECRUITED}_condition_n": f"{LEGACY_RECRUITED}_condition_n",
}


def condition_kind_of(kind: object) -> str:
    """A recorded condition kind under its current name: a store from before plan step 9 says ``cohort``.

    Args:
        kind: The ledger's ``condition_kind``.

    Returns:
        :data:`RECRUITED`, :data:`OTHER`, or the value as given.
    """
    text = str(kind or "")
    return RECRUITED if text == LEGACY_RECRUITED else text


def ledger_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:  # noqa: ANN401 -- a ledger value
    """A ledger value under its current key, or under the key a store from before plan step 9 wrote it under.

    Args:
        record: The ledger, its ``counts``, or one of its conditions.
        key: :data:`DIAGNOSIS_KEY`, :data:`DIAGNOSES_KEY` or ``recruited_condition_n``.
        default: Returned where neither key is present.

    Returns:
        The value.
    """
    if key in record:
        return record[key]
    return record.get(_LEGACY_KEYS.get(key, key), default)


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def _normalise(text: str) -> str:
    """A proposal's text as the patterns are searched in: lower-cased, straight apostrophes, one space."""
    return " ".join(str(text).translate(_APOSTROPHES).lower().split())


@dataclass(frozen=True)
class RecruitedConditions:
    """One study's recruited conditions.

    Attributes:
        name: The profile's file stem.
        release: The data release the list was read from.
        conditions: Diagnosis name to its compiled patterns, in file order.
    """

    name: str
    release: str
    conditions: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]

    def diagnosis(self, text: str) -> str | None:
        """The first recruited diagnosis a proposal's text matches.

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
        """Whether a proposal's text names a recruited condition.

        Args:
            text: The reviewer's quote.

        Returns:
            :data:`RECRUITED` or :data:`OTHER`.
        """
        return RECRUITED if self.diagnosis(text) is not None else OTHER


@profile_cache
def load_recruited_conditions(name: str) -> RecruitedConditions:
    """The dataset profile's recruited-conditions list of this name.

    Args:
        name: The file stem under the dataset profile's ``recruited_conditions/``.

    Returns:
        The list.

    Raises:
        FileNotFoundError: If the profile holds no such list.
        ValueError: If the file does not carry a ``conditions`` mapping of pattern lists.
    """
    path = data_path(RECRUITED_CONDITIONS_DIR) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no recruited-conditions list named {name!r} under {data_path(RECRUITED_CONDITIONS_DIR)}"
        )
    data = yaml.safe_load(path.read_text()) or {}
    conditions = data.get("conditions")
    if not isinstance(conditions, dict) or not all(isinstance(v, list) for v in conditions.values()):
        raise ValueError(f"recruited-conditions list {name!r} carries no conditions mapping of pattern lists")
    return RecruitedConditions(
        name=name,
        release=str(data.get("release") or ""),
        conditions=tuple(
            (str(diagnosis), tuple(re.compile(str(pattern), re.IGNORECASE) for pattern in patterns))
            for diagnosis, patterns in conditions.items()
        ),
    )
