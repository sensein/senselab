"""The decision table's vocabulary: one reason per recording, and the evidence items behind a decision.

Reasons come from ``data/decision_reasons.yaml``, which maps every ground key the fold writes to one
short reason and orders them; the evidence items are what the fold read at each decision point, each
with its value, comparison, threshold and effect.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PASS = "pass"
REVIEW = "review"
DISCARD = "discard"
WITHHOLD = "withhold"
ANNOTATION = "annotation"
EFFECTS = (PASS, REVIEW, DISCARD, WITHHOLD, ANNOTATION)
"""What an evidence item does to the decision. ``withhold`` acts on the release axis only."""

NUMERIC_COMPARISONS = (">=", "<=", ">", "<", "==")
"""How a number is compared with a numeric bound."""
BOOLEAN_COMPARISONS = ("==",)
"""How a flag is compared with a boolean bound."""
CATEGORY_COMPARISONS = ("in", "not in")
"""How a category is compared with the list of categories it is read against."""


@cache
def reason_vocabulary() -> dict[str, Any]:
    """The reason vocabulary.

    Returns:
        ``data/decision_reasons.yaml``, parsed.
    """
    text = resources.files("senselab.audio.workflows.triage.data").joinpath("decision_reasons.yaml").read_text()
    return dict(yaml.safe_load(text))


def reviewed_releases() -> frozenset[str]:
    """The release values that hold a recording for review whatever else the fold read.

    Returns:
        ``reviewed_releases`` of ``data/decision_reasons.yaml``: release axis values (``redacted``,
        ``withheld``).
    """
    return frozenset(str(value) for value in reason_vocabulary().get("reviewed_releases") or ())


def reason_of(ground_key: str, *, release: bool = False) -> str | None:
    """The reason a ground key maps to.

    Args:
        ground_key: A verdict ground key, or a release ground key where ``release`` is True.
        release: Whether the key is a release ground key.

    Returns:
        The reason, or None where the vocabulary does not map the key.
    """
    vocabulary = reason_vocabulary()
    exact = vocabulary["release_keys"] if release else vocabulary["keys"]
    if ground_key in exact:
        return str(exact[ground_key])
    if release:
        return None
    prefix, sep, _ = ground_key.partition(":")
    if sep and prefix in vocabulary["prefixes"]:
        return str(vocabulary["prefixes"][prefix])
    return None


def reasons_of(ground_keys: Iterable[str], release_ground_key: str | None = None) -> list[str]:
    """Every reason behind a decision, in the vocabulary's precedence.

    Args:
        ground_keys: The fold's ground keys.
        release_ground_key: The release ground key where the release axis holds or could not be assessed,
            else None.

    Returns:
        The distinct reasons, highest precedence first. A key the vocabulary does not map is logged and
        left out.
    """
    found: set[str] = set()
    for key in ground_keys:
        reason = reason_of(key)
        if reason is None:
            logger.warning("ground key %s maps to no decision reason", key)
            continue
        found.add(reason)
    if release_ground_key is not None:
        reason = reason_of(release_ground_key, release=True)
        if reason is None:
            logger.warning("release ground key %s maps to no decision reason", release_ground_key)
        else:
            found.add(reason)
    order = list(reason_vocabulary()["precedence"])
    return sorted(found, key=order.index)


@dataclass(frozen=True)
class EvidenceItem:
    """One reading the fold weighed for a recording.

    Attributes:
        name: The item's name, one of ``data/decision_evidence.yaml``.
        value: What was read: one number, count, flag or category, or None where nothing was read.
        unit: The value's unit, or None.
        comparison: How the value was compared, by its type (see :func:`row_problems`); None where it
            was compared against nothing.
        threshold: What it was compared against, from ``data/``; None where it was compared against
            nothing.
        effect: What the item does: ``pass``, ``review``, ``discard``, ``withhold`` or ``annotation``.
        decisive: Whether the item produced this recording's decision.
    """

    name: str
    value: Any
    unit: str | None
    comparison: str | None
    threshold: Any
    effect: str
    decisive: bool = False

    def record(self) -> dict[str, Any]:
        """The item, JSON-ready.

        Returns:
            Its fields, keyed by name.
        """
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "effect": self.effect,
            "decisive": self.decisive,
        }


def item(
    name: str,
    value: Any,  # noqa: ANN401 -- a reading is any scalar
    effect: str,
    *,
    unit: str | None = None,
    comparison: str | None = None,
    threshold: Any = None,  # noqa: ANN401 -- a bound is any scalar
    decisive: bool = False,
) -> EvidenceItem:
    """An evidence item.

    Args:
        name: Its name.
        value: What was read.
        effect: What it does.
        unit: The value's unit.
        comparison: How it was compared.
        threshold: What it was compared against.
        decisive: Whether it produced the decision.

    Returns:
        The item.

    Raises:
        ValueError: On an effect outside :data:`EFFECTS`.
    """
    if effect not in EFFECTS:
        raise ValueError(f"effect {effect!r} is not one of {EFFECTS}")
    return EvidenceItem(name, value, unit, comparison if threshold is not None else None, threshold, effect, decisive)


def _is_number(value: Any) -> bool:  # noqa: ANN401 -- any reading
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:  # noqa: ANN401 -- any reading
    return value is None or isinstance(value, (bool, int, float, str))


def row_problems(entry: EvidenceItem) -> list[str]:
    """What makes an evidence item something other than one scalar row with a comparison of its type.

    A row's value is one scalar or None. A number is compared with :data:`NUMERIC_COMPARISONS` against
    a number; a flag with :data:`BOOLEAN_COMPARISONS` against a flag; a category with
    :data:`CATEGORY_COMPARISONS` against a list of categories. A row compared against nothing has
    neither comparison nor threshold.

    Args:
        entry: The item.

    Returns:
        One sentence per problem; empty where the row is well formed.
    """
    problems: list[str] = []
    value, comparison, threshold = entry.value, entry.comparison, entry.threshold
    if not _is_scalar(value):
        problems.append(f"{entry.name}: value {value!r} is not a scalar")
    if comparison is None or threshold is None:
        if comparison is not None or threshold is not None:
            problems.append(f"{entry.name}: comparison {comparison!r} without a threshold, or the reverse")
        return problems
    if isinstance(threshold, bool):
        if comparison not in BOOLEAN_COMPARISONS or not (value is None or isinstance(value, bool)):
            problems.append(f"{entry.name}: {value!r} {comparison} {threshold!r} is not a flag comparison")
    elif _is_number(threshold):
        if comparison not in NUMERIC_COMPARISONS or not (value is None or _is_number(value)):
            problems.append(f"{entry.name}: {value!r} {comparison} {threshold!r} is not a numeric comparison")
    elif isinstance(threshold, (list, tuple)) and all(isinstance(v, str) for v in threshold):
        if comparison not in CATEGORY_COMPARISONS or not (value is None or isinstance(value, str)):
            problems.append(f"{entry.name}: {value!r} {comparison} {threshold!r} is not a category comparison")
    else:
        problems.append(f"{entry.name}: threshold {threshold!r} is not a number, a flag or a list of categories")
    return problems


def decisive_items(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """The items that produced the decision.

    Args:
        items: Every item read.

    Returns:
        Those marked decisive, in order.
    """
    return [entry for entry in items if entry.decisive]


def records(items: Sequence[EvidenceItem]) -> list[dict[str, Any]]:
    """The items, JSON-ready.

    Args:
        items: The items.

    Returns:
        Each item's record.
    """
    return [entry.record() for entry in items]


def from_records(rows: Sequence[Mapping[str, Any]]) -> list[EvidenceItem]:
    """Items from their records.

    Args:
        rows: Records as :meth:`EvidenceItem.record` writes them.

    Returns:
        The items.
    """
    return [
        EvidenceItem(
            str(row["name"]),
            row.get("value"),
            row.get("unit"),
            row.get("comparison"),
            row.get("threshold"),
            str(row["effect"]),
            bool(row.get("decisive")),
        )
        for row in rows
    ]
