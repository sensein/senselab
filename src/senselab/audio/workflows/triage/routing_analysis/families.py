"""Grouping a BIDS stem's ``task-`` id into the elicitation family it belongs to.

Only trailing numeric segments are collapsed, which folds a repeat index and the Harvard list
index and nothing else. A ``v2`` marker is not numeric and therefore survives, so
``respiration-and-cough`` v1 and v2 stay different families.
"""

from __future__ import annotations

import re

_TASK = re.compile(r"_task-(?P<task>[^_]+)$")
_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")

SYLLABLE_REPETITION: frozenset[str] = frozenset(
    {
        "diadochokinesis-buttercup",
        "diadochokinesis-ka",
        "diadochokinesis-pa",
        "diadochokinesis-pataka",
        "diadochokinesis-ta",
        "diadochokinesis-v2-buttercup",
        "diadochokinesis-v2-kuh",
        "diadochokinesis-v2-puh",
        "diadochokinesis-v2-puhtuhkuh",
        "diadochokinesis-v2-tuh",
    }
)
"""Diadochokinesis: speech production whose target is a syllable, not a word."""

LEXICAL_SPEECH: frozenset[str] = frozenset(
    {
        "animal-fluency",
        "cape-v-sentences",
        "cape-v-sentences-v2",
        "caterpillar-passage",
        "cinderella-story",
        "free-speech",
        "free-speech-v2",
        "harvard-sentences-list",
        "loudness",
        "loudness-v2",
        "open-response-questions",
        "picture-description",
        "picture-description-option1",
        "picture-description-option2",
        "productive-vocabulary",
        "rainbow-passage",
        "random-item-generation",
        "random-item-generation-v2",
        "story-recall",
        "story-recall-v2",
        "word-color-stroop",
    }
)
"""Families whose instructions ask the participant to say words. A declaration, not a label."""

SPEECH_ELICITING: frozenset[str] = LEXICAL_SPEECH | SYLLABLE_REPETITION
"""Every family whose instructions ask the participant to produce speech."""

AIRWAY_ELICITING: frozenset[str] = frozenset(
    {
        "breath-sounds",
        "respiration-and-cough-breath",
        "respiration-and-cough-cough",
        "respiration-and-cough-fivebreaths",
        "respiration-and-cough-threequickbreaths",
        "respiration-and-cough-v2-breath",
        "respiration-and-cough-v2-hardcough",
        "respiration-and-cough-v2-threebreaths",
        "respiration-and-cough-v2-threebreathsmouth",
        "respiration-and-cough-v2-threebreathsnose",
        "voluntary-cough",
    }
)
"""Families whose instructions ask for a breath, a cough or a throat manoeuvre."""

VOICE_ELICITING: frozenset[str] = frozenset(
    {
        "glides-high-to-low",
        "glides-low-to-high",
        "high-to-low",
        "maximum-phonation-time",
        "maximum-phonation-time-v2",
        "prolonged-vowel",
    }
)
"""Families whose instructions ask for sustained or glided phonation without words."""

DECLARED_KIND: dict[str, frozenset[str]] = {
    "speech": SPEECH_ELICITING,
    "lexical_speech": LEXICAL_SPEECH,
    "airway": AIRWAY_ELICITING,
    "voice": VOICE_ELICITING,
}
"""Each kind's declared families, the proxy reference standard for airway and voice."""


def task_id_of(stem: str) -> str:
    """The sanitized task id in a BIDS stem.

    Args:
        stem: A stem such as ``sub-a_ses-b_task-harvard-sentences-list-10-3``.

    Returns:
        The lowercased task id, or ``"unknown"`` when the stem carries no ``task-`` entity.
    """
    match = _TASK.search(stem)
    return match.group("task").lower() if match else "unknown"


def task_family(task_id: str) -> str:
    """The family a task id collapses into.

    Args:
        task_id: A sanitized task id.

    Returns:
        The id with every trailing numeric segment removed.
    """
    return _TRAILING_INDEX.sub("", task_id)


def declared_kinds(family: str) -> frozenset[str]:
    """Which entries of :data:`DECLARED_KIND` a family belongs to.

    Args:
        family: A task family.

    Returns:
        Every key whose family set contains it; empty when the family is in none. A lexical
        speech family is in both ``speech`` and ``lexical_speech``.
    """
    return frozenset(kind for kind, families in DECLARED_KIND.items() if family in families)
