"""The ``background_mask`` axis's vote harvest: VAD, ASR words and speaker spans (D-22).

The mask's uncertainty was ``1 - confidence`` of a single derived judgement. That read as a property of
the mask when it was a property of there being **one producer** — and it kept the axis out of
``HARVESTED_AXES``, which is why ``disagreements.json`` never listed it while ``estimates/`` did.

Three sources bear on whether the target was active in a bucket, so the axis's doubt becomes
cross-source disagreement like every other axis's:

===========  ==================================================================
``speech``   a continuous speech probability (Brouhaha's VAD head)
``words``    seconds of ASR word coverage
``speakers`` diarizer occupancy — how much of the bucket a speaker covers
===========  ==================================================================

**What each one *means* depends on the task**, which is why these are votes and not a formula. In a
speech task, all three indicate target *activity*. In a breathing task the target is the breath, speech
detection is silent through it, and a speech vote therefore indicates target **absence** — and since
AudioSet maps ``Breathing`` to ``people``, a mask built from voice activity alone reported the collected
signal as a background source. That is the failure this module's task gate exists to prevent.

Emitted on the **speech-presence grid**, which the mask shares (D-24 correction): the mask is derived
from the presence axis, so on one grid the derivation is exact and needs no projection.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Optional

from senselab.audio.workflows.audio_analysis.grid import BucketGrid

__all__ = ["MASK_SOURCES", "TARGET_POLARITY", "harvest_background_mask_evidence"]

MASK_SOURCES: Final[tuple[str, ...]] = ("speech", "words", "speakers")
"""The sources that vote on target activity, in provenance order."""

TARGET_POLARITY: Final[Mapping[str, Mapping[str, bool]]] = {
    # task type -> {source -> does a positive reading mean the TARGET was active?}
    "speech": {"speech": True, "words": True, "speakers": True},
    # The breath is the target. Speech detection is silent through it, so speech present means
    # something other than the target was happening. Words and speakers likewise.
    "breathing": {"speech": False, "words": False, "speakers": False},
    # A sustained vowel or /a/ phonation: voiced, so VAD fires on the target, but an ASR transcribing
    # words is hearing something else.
    "phonation": {"speech": True, "words": False, "speakers": True},
}
"""Whether a positive reading from each source indicates **target** activity, per task type.

The mapping is the whole reason these are votes: the same measurement means opposite things about the
target depending on what was asked for. A default would make the breathing case silently wrong, which is
how a collected breath came to be reported as background.
"""

DEFAULT_TASK_TYPE: Final[str] = "speech"
"""What the pipeline assumes when the caller declares nothing. Recorded on every row, not assumed."""

_SPEECH_THRESHOLD: Final[float] = 0.5
_WORD_COVERAGE_THRESHOLD: Final[float] = 0.0
_OCCUPANCY_THRESHOLD: Final[float] = 0.0


def harvest_background_mask_evidence(
    *,
    duration_s: float,
    grid: BucketGrid,
    task_type: Optional[str] = None,
    speech_by_bucket: Optional[Mapping[tuple[float, float], float]] = None,
    word_coverage_by_bucket: Optional[Mapping[tuple[float, float], float]] = None,
    speaker_occupancy_by_bucket: Optional[Mapping[tuple[float, float], float]] = None,
) -> list[dict[str, Any]]:
    """Per-bucket votes on whether the **target** was active, one per source that measured.

    Args:
        duration_s: Recording length.
        grid: The reporting grid — the speech-presence grid, which the mask shares.
        task_type: What was asked for. ``None`` means :data:`DEFAULT_TASK_TYPE`, recorded on the row.
        speech_by_bucket: ``{bucket → speech probability}``.
        word_coverage_by_bucket: ``{bucket → seconds of ASR word overlap}``.
        speaker_occupancy_by_bucket: ``{bucket → fraction of the bucket a speaker covers}``.

    Returns:
        ``[{"start", "end", "task_type", "votes"}, …]`` in time order, where ``votes`` maps a source to
        ``{"target_active", "reading", "same_label_uncertainty"}``.

        **A bucket no source measured yields no row**, and a source that measured nothing casts no vote:
        absence of evidence is not evidence of a target-free region, which is the claim this axis exists
        to make carefully.

    Raises:
        ValueError: For a task type with no declared polarity. Guessing would make the breathing case
            silently wrong in the direction that reports the collected signal as background.
    """
    task = task_type or DEFAULT_TASK_TYPE
    polarity = TARGET_POLARITY.get(task)
    if polarity is None:
        raise ValueError(
            f"no target polarity declared for task_type {task!r}; add it to TARGET_POLARITY — "
            f"known: {sorted(TARGET_POLARITY)}"
        )

    readings: dict[str, Mapping[tuple[float, float], float]] = {
        "speech": speech_by_bucket or {},
        "words": word_coverage_by_bucket or {},
        "speakers": speaker_occupancy_by_bucket or {},
    }
    thresholds = {
        "speech": _SPEECH_THRESHOLD,
        "words": _WORD_COVERAGE_THRESHOLD,
        "speakers": _OCCUPANCY_THRESHOLD,
    }

    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        key = (round(start, 6), round(end, 6))
        votes: dict[str, dict[str, Any]] = {}
        for source in MASK_SOURCES:
            value = readings[source].get(key)
            if value is None:
                continue  # this source said nothing here; silence is not a vote
            positive = float(value) > thresholds[source]
            votes[source] = {
                # Does this reading say the *target* was active? The polarity flip is the task gate.
                "target_active": positive if polarity[source] else not positive,
                # The measurement itself, so the interpretation can be redone without re-measuring.
                "reading": float(value),
                "same_label_uncertainty": _uncertainty(source, float(value)),
            }
        if votes:
            out.append({"start": start, "end": end, "task_type": task, "votes": votes})
    return out


def _uncertainty(source: str, value: float) -> float:
    """How unsure this reading is, in ``[0, 1]``, independent of which way it points.

    For a probability, the least informative reading is 0.5 and both extremes are informative — so
    uncertainty is distance from the extremes, not the complement of the value. A vote that pointed
    confidently at *target-absent* would otherwise be recorded as maximally uncertain.

    For a coverage measurement, any positive reading is a definite observation; only exactly zero is
    ambiguous, because a bucket can be free of words for reasons other than being free of the target.
    """
    if source == "speech":
        return max(0.0, min(1.0, 1.0 - 2.0 * abs(value - 0.5)))
    return 0.5 if value <= 0.0 else 0.0
