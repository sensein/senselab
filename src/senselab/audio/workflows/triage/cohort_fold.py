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
"""A check that does not read this recording's declared kind, or a speaker pool with nothing in it."""

NOT_MEASURED = "not_measured"
"""A speaker pool too short to identify a speaker by: an annotation, never a flag."""

OTHER_SPEAKER = "other_speaker"
"""The reading's block for the session enrollment and the pooled comparisons against it."""

POOL_TARGET = "target"
"""The pool of the task's own speech: what the participant was asked to say."""

POOL_NON_TASK = "non_task"
"""The pool of the speech outside the task."""

POOLS = (POOL_TARGET, POOL_NON_TASK)

POOL_ITEMS = {POOL_TARGET: "target_cosine", POOL_NON_TASK: "non_task_cosine"}
"""Each pool's evidence item name, under :data:`ITEM_PREFIX`."""

POOL_WHAT = {POOL_TARGET: "the task's own speech", POOL_NON_TASK: "the speech outside the task"}

CHECKS = "checks"
"""The reading's block of distribution checks, one per check name."""

KEY_OTHER_SPEAKER_IN_SESSION = "other_speaker_in_session"
"""A pooled speech comparison does not match the participant's enrollment: the session's own, or the other sessions'."""

KEY_RECORDING_DURATION_OUTLIER = "recording_duration_outlier"
"""The recording is far longer than its family's and the task covers little of it."""

KEY_SESSION_SPEAKER_INCONSISTENT = "session_speaker_inconsistent"
"""The session's enrollment is far from the participant's other sessions': another person, or a mislabelled session."""

COHORT_GROUND_KEYS = (KEY_OTHER_SPEAKER_IN_SESSION, KEY_RECORDING_DURATION_OUTLIER, KEY_SESSION_SPEAKER_INCONSISTENT)
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


def pool_block(reading: Mapping[str, Any] | None, pool: str) -> dict[str, Any]:
    """One pooled comparison of a cohort reading's ``other_speaker`` block.

    Args:
        reading: The ``cohort_reading`` attributes, or None.
        pool: :data:`POOL_TARGET` or :data:`POOL_NON_TASK`.

    Returns:
        The pool's record, empty where the block carries none.
    """
    value = other_speaker_block(reading).get(pool)
    return dict(value) if isinstance(value, Mapping) else {}


def _nonmatching(reading: Mapping[str, Any] | None) -> list[str]:
    if other_speaker_block(reading).get("status") != MEASURED:
        return []
    return [
        pool
        for pool in POOLS
        if pool_block(reading, pool).get("status") == MEASURED and pool_block(reading, pool).get("match") is False
    ]


def check_blocks(reading: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """The distribution checks of a cohort reading, by name.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        Each check's block, in the reading's order.
    """
    return {str(name): dict(block) for name, block in _block(reading, CHECKS).items() if isinstance(block, Mapping)}


def nonmatch_spans(reading: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    """The spans of every pool that did not match the participant's enrollment.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        ``(start, end)`` in seconds, earliest first; empty where nothing was compared or every pool matched.
    """
    return sorted(
        (float(start), float(end))
        for pool in _nonmatching(reading)
        for start, end in pool_block(reading, pool).get("spans") or ()
    )


def non_task_speech_spans(reading: Mapping[str, Any] | None) -> list[tuple[float, float]]:
    """The spans a redaction treats as non-task speech: a non-matching non-task pool of a non-lexical task.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        The non-task pool's spans where it did not match and the declared task is not a lexical speech task,
        else empty.
    """
    if (reading or {}).get("lexical_task") is not False or POOL_NON_TASK not in _nonmatching(reading):
        return []
    return [(float(start), float(end)) for start, end in pool_block(reading, POOL_NON_TASK).get("spans") or ()]


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
    nonmatch = _nonmatching(reading)
    if nonmatch:
        read = "; ".join(
            f"{POOL_WHAT[pool]}, {pool_block(reading, pool).get('pooled_s')} s pooled, cosine "
            f"{pool_block(reading, pool).get('cosine')}"
            for pool in nonmatch
        )
        grounds.append(
            (
                f"{KEY_OTHER_SPEAKER_IN_SESSION}: {read}, below {block.get('cut')} against the "
                f"{block.get('enrollment_source') or 'session'} enrollment",
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
    """The cohort reading's evidence items: each pooled comparison and every check's comparisons.

    A check the reading marks unavailable, a speaker comparison with no enrollment, and a pool too short to
    identify a speaker by are each one item with no value, an annotation: not measured, and moving nothing.
    A pool with nothing in it writes no item.

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
        cut = float(block.get("cut") or 0)
        for pool in POOLS:
            record = pool_block(reading, pool)
            if record.get("status") == MEASURED:
                effect = PASS if record.get("match") else REVIEW
                items.append(
                    item(_named(POOL_ITEMS[pool]), record.get("cosine"), effect, comparison=">=", threshold=cut)
                )
            elif record.get("status") in (NOT_MEASURED, UNAVAILABLE):
                items.append(item(_named(POOL_ITEMS[pool]), None, ANNOTATION))
    elif block:
        items.append(item(_named("speaker_comparison"), None, ANNOTATION))
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
