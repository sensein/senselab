"""Which labels one classifier window carries: the top-K by score that also clear a floor.

One rule, read from ``windows.<classifier>`` in ``data/config/default.yaml``, so PREPROCESS stamps
the membership the routing analysis re-derives rather than the two drifting apart. The measurements
behind the pair are in ``specs/20260817-triage-workflow-dag/family-taxonomy-ruleset.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from senselab.audio.workflows.triage.config import TriageConfig

TOP_K_KEY = "label_top_k"
FLOOR_KEY = "default_threshold"
OVERRIDES_KEY = "label_thresholds"
"""The three ``windows.<classifier>`` keys one membership rule is read from."""


@dataclass(frozen=True)
class LabelMembership:
    """The rule deciding which of a window's labels the window carries.

    Attributes:
        top_k: How many labels, ranked by score, are eligible at all.
        floor: The score an eligible label needs, where no override names it.
        label_floors: Per-label overrides of ``floor``.
    """

    top_k: int
    floor: float
    label_floors: Mapping[str, float]

    def members(self, scores: Mapping[str, float]) -> dict[str, float]:
        """The labels one score mapping carries, each with the score behind it.

        Args:
            scores: ``{label: score}``, the classifier's whole output for the window.

        Returns:
            ``{label: score}`` in descending score order over the labels that are both in the top
            :attr:`top_k` and at or above their own floor. Empty is a window nothing cleared, which
            is a different fact from a window that was never classified. Ties in score are ranked
            by label so the cut is deterministic.
        """
        ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))[: self.top_k]
        return {
            str(label): float(score)
            for label, score in ranked
            if float(score) >= float(self.label_floors.get(str(label), self.floor))
        }


def load_label_membership(config: TriageConfig, classifier: str) -> LabelMembership:
    """One classifier's membership rule, with every value required to have been measured.

    Args:
        config: The resolved configuration.
        classifier: ``yamnet``, ``ast`` or ``hear``.

    Returns:
        The rule.

    Raises:
        ValueError: When any of the three keys is null. A caller reading through here records the
            derivative absent rather than inventing a value for it.
    """
    return LabelMembership(
        top_k=int(config.require(f"windows.{classifier}.{TOP_K_KEY}")),
        floor=float(config.require(f"windows.{classifier}.{FLOOR_KEY}")),
        label_floors={
            str(label): float(value) for label, value in config.require(f"windows.{classifier}.{OVERRIDES_KEY}").items()
        },
    )


def optional_label_membership(config: TriageConfig, classifier: str) -> LabelMembership | None:
    """One classifier's membership rule, or None while its floor is unmeasured.

    Unlike :func:`load_label_membership` a null ``label_thresholds`` is read as no per-label
    override rather than as an unmeasured value: the overrides refine a floor that is already
    declared, so their absence is a complete rule and not a missing one.

    Args:
        config: The resolved configuration.
        classifier: ``yamnet``, ``ast`` or ``hear``.

    Returns:
        The rule, or None when ``windows.<classifier>.default_threshold`` is null. A caller that
        gets None writes the raw scores and no membership, rather than inventing a floor.
    """
    if config.get(f"windows.{classifier}.{FLOOR_KEY}") is None:
        return None
    return LabelMembership(
        top_k=int(config.require(f"windows.{classifier}.{TOP_K_KEY}")),
        floor=float(config.require(f"windows.{classifier}.{FLOOR_KEY}")),
        label_floors={
            str(label): float(value)
            for label, value in (config.get(f"windows.{classifier}.{OVERRIDES_KEY}") or {}).items()
        },
    )
