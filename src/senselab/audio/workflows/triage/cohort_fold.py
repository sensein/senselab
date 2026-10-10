"""What VERDICT reads off COHORT's ``cohort_reading``: its grounds, its evidence items, its spans.

COHORT runs after QUALITY over the stored outputs of a whole session or corpus
(:mod:`senselab.audio.workflows.triage.cohort_stage`) and writes one ``cohort_reading`` into each
recording's store. This module is the reader's half and imports nothing from the graph, so the fold,
REDACT and the tables can all read it. The design is ``specs/20261007-task-events-in-background/design.md``,
section "Cohort stage".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from senselab.audio.workflows.triage.decision import ANNOTATION, PASS, REVIEW, EvidenceItem, item

COHORT_NODE = "COHORT"
COHORT_READING = "cohort_reading"
"""The measurement COHORT writes into every store it reads."""

MEASURED = "measured"
UNAVAILABLE = "unavailable"
"""A check, or the whole reading, that had no cohort to read against."""

NOT_APPLICABLE = "not_applicable"
"""A check that does not read this recording's declared kind."""

OTHER_SPEAKER = "other_speaker"
"""The reading's block for the session enrollment and the per-run matches against it."""

CHECKS = "checks"
"""The reading's block of distribution checks, one per check name."""

KEY_OTHER_SPEAKER_IN_SESSION = "other_speaker_in_session"
"""A speech run in the recording does not match the participant's session enrollment."""

KEY_RECORDING_DURATION_OUTLIER = "recording_duration_outlier"
"""The recording is far longer than its family's and the task covers little of it."""

COHORT_GROUND_KEYS = (KEY_OTHER_SPEAKER_IN_SESSION, KEY_RECORDING_DURATION_OUTLIER)
"""Every ground key a cohort reading can raise."""

ITEM_PREFIX = "cohort"


def _block(reading: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    value = (reading or {}).get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def other_speaker_block(reading: Mapping[str, Any] | None) -> dict[str, Any]:
    """The enrollment-match block of a cohort reading.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        The block, empty where the reading carries none.
    """
    return _block(reading, OTHER_SPEAKER)


def check_blocks(reading: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The distribution checks of a cohort reading, by name.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        Each check's block, in the reading's order.
    """
    return {str(name): dict(block) for name, block in _block(reading, CHECKS).items() if isinstance(block, Mapping)}


def nonmatch_spans(reading: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    """The spans whose speech did not match the participant's session enrollment, merged.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        ``(start, end)`` in seconds, earliest first; empty where nothing was compared or all matched.
    """
    block = other_speaker_block(reading)
    if block.get("status") != MEASURED:
        return []
    return [(float(start), float(end)) for start, end in block.get("nonmatch_spans") or ()]


def non_task_speech_spans(reading: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    """The non-matching spans a redaction treats as non-task speech: those of a non-lexical task.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        :func:`nonmatch_spans` where the recording's declared task is not a lexical speech task, else
        empty.
    """
    if (reading or {}).get("lexical_task") is not False:
        return []
    return nonmatch_spans(reading)


def cohort_grounds(reading: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """The flag grounds a cohort reading raises.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        ``(why, ground key)`` per ground, in :data:`COHORT_GROUND_KEYS` order. An unavailable check
        raises nothing.
    """
    grounds: list[tuple[str, str]] = []
    block = other_speaker_block(reading)
    if block.get("status") == MEASURED and int(block.get("nonmatch_n") or 0) > 0:
        grounds.append(
            (
                f"{KEY_OTHER_SPEAKER_IN_SESSION}: {block.get('nonmatch_n')} of {block.get('runs_n')} speech run(s), "
                f"{block.get('nonmatch_s')} s, below cosine {block.get('cut')} against the session enrollment",
                KEY_OTHER_SPEAKER_IN_SESSION,
            )
        )
    for name, check in check_blocks(reading).items():
        if check.get("status") == MEASURED and check.get("outcome") == REVIEW and check.get("ground"):
            grounds.append((f"{check.get('ground')}: {check.get('why') or name}", str(check.get("ground"))))
    return grounds


def _named(name: str) -> str:
    return f"{ITEM_PREFIX}.{name}"


def cohort_evidence(reading: Mapping[str, Any] | None) -> list[EvidenceItem]:
    """The cohort reading's evidence items: the enrollment match and every check's comparisons.

    A check the reading marks unavailable is one item with no value, an annotation: not measured, and
    moving nothing.

    Args:
        reading: The ``cohort_reading`` attributes, or None where no cohort reading is in the store.

    Returns:
        The items, in the reading's order; empty where there is no reading.
    """
    if not reading:
        return []
    items: list[EvidenceItem] = []
    block = other_speaker_block(reading)
    if block.get("status") == MEASURED:
        nonmatch = int(block.get("nonmatch_n") or 0)
        effect = REVIEW if nonmatch else PASS
        items.append(item(_named("other_speaker_runs"), nonmatch, effect, unit="runs", comparison="<=", threshold=0))
        lowest = block.get("lowest_cosine")
        if lowest is not None:
            items.append(
                item(
                    _named("lowest_run_cosine"), lowest, effect, comparison=">=", threshold=float(block.get("cut") or 0)
                )
            )
    elif block:
        items.append(item(_named("other_speaker_runs"), None, ANNOTATION, unit="runs"))
    for name, check in check_blocks(reading).items():
        if check.get("status") == UNAVAILABLE:
            items.append(item(_named(name), None, ANNOTATION))
        if check.get("status") != MEASURED:
            continue
        effect = REVIEW if check.get("outcome") == REVIEW else PASS
        for comparison in check.get("comparisons") or ():
            items.append(
                item(
                    _named(str(comparison.get("name"))),
                    comparison.get("value"),
                    effect,
                    unit=comparison.get("unit"),
                    comparison=comparison.get("comparison"),
                    threshold=comparison.get("threshold"),
                )
            )
    return items
