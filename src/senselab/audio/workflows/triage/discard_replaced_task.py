"""Discard where the task is absent or weak and talk outside it takes its place.

Three rules, each over readings the fold already holds, with the bounds in
``data/discard_replaced_task.yaml``:

- **non-lexical** (airway, voice, and the families read as a voice task): the talk outside the task runs at
  ``talk_fraction_min`` of the corpus free-speech rate or more, and the task is weak: a ``task_mismatch``
  annotation, or an extent covering under ``extent_fraction_max`` of the recording;
- **lexical** (lexical speech, less those): no task extent, or none of the expected tokens matched, and the
  reviewer reads off-task speech or more than one speaker;
- **short response** (the free-response families): the response is under ``fraction_max`` of the family's
  median, and the reviewer reads off-task speech or an examiner speaking the instructions.

The rule is switched by ``enabled``. The design and the measurements behind the bounds are
``specs/20261007-task-events-in-background/design.md``, "Discard when talk replaces the task".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

PARAMETERS_PATH = Path(__file__).parent / "data" / "discard_replaced_task.yaml"

TASK_REPLACED_BY_TALK = "task_replaced_by_talk"
"""The discard ground: the task is absent or weak and talk outside it takes its place."""

RULE_NON_LEXICAL = "non_lexical"
RULE_LEXICAL = "lexical"
RULE_SHORT_RESPONSE = "short_response"
RULES = (RULE_NON_LEXICAL, RULE_LEXICAL, RULE_SHORT_RESPONSE)


@cache
def parameters() -> dict[str, Any]:
    """The packaged ``data/discard_replaced_task.yaml``.

    Returns:
        The parsed file.
    """
    return dict(yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8")))


def enabled(params: Mapping[str, Any] | None = None) -> bool:
    """Whether the rule discards.

    Args:
        params: The parameters; None reads the packaged ones.

    Returns:
        ``enabled``.
    """
    return bool((params if params is not None else parameters()).get("enabled"))


@dataclass(frozen=True)
class ReplacedTaskInputs:
    """What the rule reads about one recording.

    Attributes:
        family: The declared task family.
        duration_s: The recording's duration.
        talk_words_n: The speech words outside a non-lexical task (``speech_in_task``), or None.
        task_mismatch: Whether the fold annotated ``task_mismatch``.
        extent_fraction: The task extent's share of the recording, 0.0 with no extent, None where unknown.
        expected_tokens_matched: The ``expected_tokens_matched_min`` gate's reading, or None.
        response_s: The ``response_min_s`` gate's reading, or None.
        reviewer_off_task: Whether the reviewer read off-task speech (some or extensive).
        reviewer_more_than_one: Whether the reviewer read more than one speaker.
        instructions_by_other: Whether the reviewer read the instructions spoken by an assistant or background voice.
    """

    family: str | None
    duration_s: float | None = None
    talk_words_n: float | None = None
    task_mismatch: bool = False
    extent_fraction: float | None = None
    expected_tokens_matched: float | None = None
    response_s: float | None = None
    reviewer_off_task: bool = False
    reviewer_more_than_one: bool = False
    instructions_by_other: bool = False


def _in_kinds(family: str, block: Mapping[str, Any]) -> bool:
    from senselab.audio.workflows.triage.routing_analysis.families import DECLARED_KIND  # noqa: PLC0415

    if family in set(block.get("families") or ()):
        return True
    if family in set(block.get("exclude") or ()):
        return False
    return any(family in DECLARED_KIND[str(kind)] for kind in block.get("kinds") or ())


def rule_family(family: str | None, params: Mapping[str, Any] | None = None) -> str | None:
    """Which rule reads a family: :data:`RULE_NON_LEXICAL`, :data:`RULE_LEXICAL`, or None.

    Args:
        family: The declared task family.
        params: The parameters; None reads the packaged ones.

    Returns:
        The rule, or None for a family no rule reads (syllable repetition, an undeclared one).
    """
    p = params if params is not None else parameters()
    if not family:
        return None
    if _in_kinds(family, p[RULE_NON_LEXICAL]):
        return RULE_NON_LEXICAL
    if _in_kinds(family, p[RULE_LEXICAL]):
        return RULE_LEXICAL
    return None


def replaced_task(inputs: ReplacedTaskInputs, params: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Whether the task is absent or weak and talk outside it takes its place.

    Args:
        inputs: The recording's readings.
        params: The parameters; None reads the packaged ones.

    Returns:
        ``rule``, ``why``, ``name``, ``value``, ``comparison`` and ``threshold`` of the first rule that holds,
        or None where none does. Computed whether or not the rule is enabled.
    """
    p = params if params is not None else parameters()
    kind = rule_family(inputs.family, p)
    reviewer = inputs.reviewer_off_task or inputs.reviewer_more_than_one
    if kind == RULE_NON_LEXICAL:
        block = p[RULE_NON_LEXICAL]
        if not inputs.duration_s or inputs.talk_words_n is None:
            return None
        talk = float(inputs.talk_words_n) / float(inputs.duration_s) / float(block["free_speech_words_per_s"])
        weak = inputs.task_mismatch or (
            inputs.extent_fraction is not None and inputs.extent_fraction < float(block["extent_fraction_max"])
        )
        if talk >= float(block["talk_fraction_min"]) and weak:
            weakness = (
                "task_mismatch" if inputs.task_mismatch else f"extent {inputs.extent_fraction:.2f} of the recording"
            )
            return {
                "rule": RULE_NON_LEXICAL,
                "why": f"talk outside the task at {talk:.2f} of the free-speech rate over a weak task ({weakness})",
                "name": "talk_fraction_of_free_speech",
                "value": round(talk, 4),
                "comparison": "<",
                "threshold": float(block["talk_fraction_min"]),
            }
        return None
    if kind == RULE_LEXICAL:
        absent = inputs.extent_fraction == 0.0 or inputs.expected_tokens_matched == 0
        if absent and reviewer:
            return {
                "rule": RULE_LEXICAL,
                "why": "no task extent or no expected token matched, and the reviewer read "
                + ("off-task speech" if inputs.reviewer_off_task else "more than one speaker"),
                "name": "expected_tokens_matched",
                "value": inputs.expected_tokens_matched,
                "comparison": ">",
                "threshold": 0,
            }
        block = p[RULE_SHORT_RESPONSE]
        median = (block.get("family_median_s") or {}).get(inputs.family)
        if median and inputs.response_s is not None and (inputs.reviewer_off_task or inputs.instructions_by_other):
            fraction = float(inputs.response_s) / float(median)
            if fraction < float(block["fraction_max"]):
                return {
                    "rule": RULE_SHORT_RESPONSE,
                    "why": (
                        f"the response is {fraction:.2f} of the family median {float(median):.1f} s, and the "
                        "reviewer read "
                        + ("off-task speech" if inputs.reviewer_off_task else "the instructions spoken by another")
                    ),
                    "name": "response_fraction_of_family_median",
                    "value": round(fraction, 4),
                    "comparison": ">=",
                    "threshold": float(block["fraction_max"]),
                }
    return None
