"""What VERDICT reads off COHORT's ``cohort_reading``: its grounds, its evidence items, its spans.

COHORT runs after QUALITY over the stored outputs of a whole session or corpus
(:mod:`senselab.audio.workflows.triage.cohort_stage`) and writes one ``cohort_reading`` into each
recording's store. This module is the reader's half and imports nothing from the graph, so the fold,
REDACT and the tables can all read it. The design is ``specs/20261007-task-events-in-background/design.md``,
section "Cohort stage".
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from senselab.audio.workflows.triage.decision import ANNOTATION, PASS, REVIEW, EvidenceItem, item

PARAMETERS_PATH = Path(__file__).parent / "data" / "cohort_stage.yaml"

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

KEY_SESSION_CONDITION_SHIFT = "session_condition_shift"
"""The session's enrollment is far from the other sessions', and its recording condition differs from theirs."""

KEY_RECURRING_NON_TARGET_VOICE = "recurring_non_target_voice"
"""The non-matching voice outside the task recurs across the participant's recordings: likely the examiner."""

KEY_NON_TASK_SELF_TALK = "non_task_self_talk"
"""The non-matching voice outside the task matches the participant's own speech in their other recordings."""

KEY_SPEAKER_NONMATCH_SHARED = "speaker_nonmatch_shared"
"""A speaker non-match shared across the participant's recordings that nothing else corroborates."""

COHORT_ANNOTATION_KEYS = (
    KEY_SESSION_CONDITION_SHIFT,
    KEY_RECURRING_NON_TARGET_VOICE,
    KEY_NON_TASK_SELF_TALK,
    KEY_SPEAKER_NONMATCH_SHARED,
)
"""Every annotation key a cohort reading can raise."""

CONDITION = "condition"
"""The reading's block of condition summaries: the recording, its session, the participant, the other sessions."""

NON_TARGET_VOICE = "non_target_voice"
"""The reading's block placing its non-matching voice outside the task in the participant's clusters."""

PARTICIPANT_NONMATCH = "participant_nonmatch"
"""The reading's block counting the participant's recordings that hold a speaker non-match."""

LABEL_SELF_TALK = "self_talk"
LABEL_RECURRING = KEY_RECURRING_NON_TARGET_VOICE
LABEL_SINGLE = "single"
"""``non_target_voice.label``: the participant's own voice, a voice recurring across recordings, or one alone."""

CONSISTENCY_CHECK_NAME = "session_speaker_consistency"

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


@cache
def _parameters() -> dict[str, Any]:
    return dict(yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8")))


def non_target_voice_block(reading: Mapping[str, Any] | None) -> dict[str, Any]:
    """The non-target voice block of a cohort reading.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        The block, empty where the reading carries none.
    """
    return _block(reading, NON_TARGET_VOICE)


def self_talk(reading: Mapping[str, Any] | None) -> bool:
    """Whether the non-matching voice outside the task is the participant's own.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        True where :data:`NON_TARGET_VOICE` labels it :data:`LABEL_SELF_TALK`.
    """
    return non_target_voice_block(reading).get("label") == LABEL_SELF_TALK


def _nonmatching(reading: Mapping[str, Any] | None) -> list[str]:
    if other_speaker_block(reading).get("status") != MEASURED:
        return []
    return [
        pool
        for pool in POOLS
        if pool_block(reading, pool).get("status") == MEASURED
        and pool_block(reading, pool).get("match") is False
        and not (pool == POOL_NON_TASK and self_talk(reading))
    ]


def _hot_or_clipped(summary: Any) -> bool | None:  # noqa: ANN401 -- a stored block or None
    if not isinstance(summary, Mapping):
        return None
    bounds = _parameters()["condition"]
    rms, clipped = summary.get("rms_dbfs_median"), summary.get("clipped_fraction")
    if rms is None and clipped is None:
        return None
    return bool(
        (rms is not None and float(rms) >= float(bounds["hot_rms_dbfs_min"]))
        or (clipped is not None and float(clipped) >= float(bounds["clipped_fraction_min"]))
    )


def hot_or_clipped(reading: Mapping[str, Any] | None, scope: str = "participant") -> bool | None:
    """Whether a participant's (or a session's) recordings are hot or clipped.

    The bounds are ``condition.hot_rms_dbfs_min`` and ``condition.clipped_fraction_min`` of
    ``data/cohort_stage.yaml``: a median RMS at or over the first, or a share of clipped recordings at or over
    the second.

    Args:
        reading: The ``cohort_reading`` attributes, or None.
        scope: ``participant`` or ``session``.

    Returns:
        The flag, or None where the reading carries no condition for that scope.
    """
    return _hot_or_clipped(_block(reading, CONDITION).get(scope))


def condition_shift(reading: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """How far the session's recording condition is from the participant's other sessions'.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        ``level_db`` and ``snr_db`` (session median less the other sessions'), ``hot_or_clipped_differs`` and
        ``explained``: a level or SNR difference at or over ``condition.shift_level_db_min`` /
        ``condition.shift_snr_db_min``, or a hot-or-clipped flag the other sessions do not share. None where
        the participant has no other session.
    """
    block = _block(reading, CONDITION)
    own, other = block.get("session"), block.get("other_sessions")
    if not isinstance(own, Mapping) or not isinstance(other, Mapping):
        return None
    bounds = _parameters()["condition"]

    def difference(key: str) -> float | None:
        a, b = own.get(key), other.get(key)
        return None if a is None or b is None else round(float(a) - float(b), 2)

    level, snr = difference("rms_dbfs_median"), difference("snr_db_median")
    flags = (_hot_or_clipped(own), _hot_or_clipped(other))
    differs = None not in flags and flags[0] != flags[1]
    explained = (
        (level is not None and abs(level) >= float(bounds["shift_level_db_min"]))
        or (snr is not None and abs(snr) >= float(bounds["shift_snr_db_min"]))
        or differs
    )
    return {"level_db": level, "snr_db": snr, "hot_or_clipped_differs": differs, "explained": bool(explained)}


def shared_nonmatch(reading: Mapping[str, Any] | None) -> bool:
    """Whether the participant's speaker non-matches are shared across their recordings.

    Args:
        reading: The ``cohort_reading`` attributes, or None.

    Returns:
        True where at least ``shared_nonmatch.recordings_min`` of the participant's compared recordings hold a
        non-match and they are at least ``shared_nonmatch.fraction_min`` of them.
    """
    block = _block(reading, PARTICIPANT_NONMATCH)
    bounds = _parameters()["shared_nonmatch"]
    count, fraction = block.get("recordings_nonmatch_n"), block.get("nonmatch_fraction")
    if count is None or fraction is None:
        return False
    return int(count) >= int(bounds["recordings_min"]) and float(fraction) >= float(bounds["fraction_min"])


def _suppressed(reading: Mapping[str, Any] | None, corroborated: bool) -> bool:
    """A non-match shared across the participant, uncorroborated, and not a recurring other voice."""
    return (
        shared_nonmatch(reading)
        and not corroborated
        and non_target_voice_block(reading).get("label") != LABEL_RECURRING
    )


def _condition_explains(reading: Mapping[str, Any] | None, name: str, check: Mapping[str, Any]) -> bool:
    if name != CONSISTENCY_CHECK_NAME or check.get("outcome") != REVIEW:
        return False
    shift = condition_shift(reading)
    return bool(shift and shift["explained"])


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


def cohort_grounds(reading: Mapping[str, Any] | None, *, corroborated: bool = False) -> list[tuple[str, str]]:
    """The flag grounds a cohort reading raises.

    A non-match the participant shares across recordings (:func:`shared_nonmatch`) that nothing corroborates and
    that is not a recurring other voice raises none; nor does a session inconsistency its condition shift
    explains (:func:`condition_shift`). :func:`cohort_annotations` names both.

    Args:
        reading: The ``cohort_reading`` attributes, or None.
        corroborated: Whether another reading (a reader, the diarizer) places another speaker in the recording.

    Returns:
        ``(why, ground key)`` per ground, in :data:`COHORT_GROUND_KEYS` order. An unavailable check
        raises nothing.
    """
    grounds: list[tuple[str, str]] = []
    block = other_speaker_block(reading)
    nonmatch = [] if _suppressed(reading, corroborated) else _nonmatching(reading)
    if nonmatch:
        read = "; ".join(
            f"{POOL_WHAT[pool]}, {pool_block(reading, pool).get('pooled_s')} s pooled, cosine "
            f"{pool_block(reading, pool).get('cosine')} under {pool_block(reading, pool).get('cosine_min')}"
            for pool in nonmatch
        )
        grounds.append(
            (
                f"{KEY_OTHER_SPEAKER_IN_SESSION}: {read}, against the "
                f"{block.get('enrollment_source') or 'session'} enrollment",
                KEY_OTHER_SPEAKER_IN_SESSION,
            )
        )
    for name, check in check_blocks(reading).items():
        if _condition_explains(reading, name, check):
            continue
        if check.get("status") == MEASURED and check.get("outcome") == REVIEW and check.get("ground"):
            grounds.append((f"{check.get('ground')}: {check.get('why') or name}", str(check.get("ground"))))
    return grounds


def cohort_annotations(reading: Mapping[str, Any] | None, *, corroborated: bool = False) -> list[tuple[str, str]]:
    """The annotations a cohort reading raises: what it measured across recordings that moves no verdict.

    Args:
        reading: The ``cohort_reading`` attributes, or None.
        corroborated: As :func:`cohort_grounds` takes it.

    Returns:
        ``(why, annotation key)`` per annotation, in :data:`COHORT_ANNOTATION_KEYS` order.
    """
    out: list[tuple[str, str]] = []
    for name, check in check_blocks(reading).items():
        if _condition_explains(reading, name, check):
            shift = condition_shift(reading) or {}
            out.append(
                (
                    f"{KEY_SESSION_CONDITION_SHIFT}: {check.get('why') or name}; the session's level differs by "
                    f"{shift.get('level_db')} dB and its SNR by {shift.get('snr_db')} dB from the other sessions'",
                    KEY_SESSION_CONDITION_SHIFT,
                )
            )
    voice = non_target_voice_block(reading)
    if voice.get("status") == MEASURED and voice.get("label") == LABEL_RECURRING:
        out.append(
            (
                f"{KEY_RECURRING_NON_TARGET_VOICE}: cluster {voice.get('cluster')} spans "
                f"{voice.get('cluster_recordings_n')} recording(s) in {voice.get('cluster_sessions_n')} session(s)",
                KEY_RECURRING_NON_TARGET_VOICE,
            )
        )
    if voice.get("status") == MEASURED and voice.get("label") == LABEL_SELF_TALK:
        out.append(
            (
                f"{KEY_NON_TASK_SELF_TALK}: cosine {voice.get('self_cosine')} to the participant's speech in "
                f"{voice.get('self_recordings_n')} other recording(s)",
                KEY_NON_TASK_SELF_TALK,
            )
        )
    if _suppressed(reading, corroborated) and _nonmatching(reading):
        shared = _block(reading, PARTICIPANT_NONMATCH)
        out.append(
            (
                f"{KEY_SPEAKER_NONMATCH_SHARED}: {shared.get('recordings_nonmatch_n')} of the participant's "
                f"{shared.get('recordings_compared_n')} compared recordings do not match, and nothing corroborates",
                KEY_SPEAKER_NONMATCH_SHARED,
            )
        )
    return out


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
        for pool in POOLS:
            record = pool_block(reading, pool)
            if record.get("status") == MEASURED and record.get("match") is None:
                items.append(item(_named(POOL_ITEMS[pool]), record.get("cosine"), ANNOTATION))
            elif record.get("status") == MEASURED:
                effect = PASS if record.get("match") else REVIEW
                bound = float(record.get("cosine_min") or 0)
                items.append(
                    item(_named(POOL_ITEMS[pool]), record.get("cosine"), effect, comparison=">=", threshold=bound)
                )
            elif record.get("status") in (NOT_MEASURED, UNAVAILABLE):
                items.append(item(_named(POOL_ITEMS[pool]), None, ANNOTATION))
    elif block:
        items.append(item(_named("speaker_comparison"), None, ANNOTATION))
    flagged = hot_or_clipped(reading)
    if flagged is not None:
        items.append(item(_named("participant_hot_or_clipped"), flagged, ANNOTATION))
    for name, check in check_blocks(reading).items():
        if check.get("status") == UNAVAILABLE:
            items.append(item(_named(name), None, ANNOTATION))
        if check.get("status") != MEASURED:
            continue
        explained = _condition_explains(reading, name, check)
        effect = REVIEW if check.get("outcome") == REVIEW and not explained else PASS
        shift = condition_shift(reading) if name == CONSISTENCY_CHECK_NAME else None
        if shift is not None:
            items.append(item(_named("session_level_shift_db"), shift["level_db"], ANNOTATION, unit="dB"))
            items.append(item(_named("session_snr_shift_db"), shift["snr_db"], ANNOTATION, unit="dB"))
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
