"""Which transcript words are the task's own content because they are a recogniser's misreading of its events.

A non-lexical family (a syllable train, a breath, a cough, a held vowel) has its events read by the task
layer. A transcript word lying on those events can be a recogniser's reading of the task's own sound, and
is then never PII and never speech outside the task: in a syllable-repetition family any such word; in an
airway or voice family only a word the recognisers disagree on, one spelled as a sound, or one of the
task's lexicon, that lies alone on one event. A run of words every recogniser agrees on is speech, never
task content. The families, the readings and the rules are in ``data/task_content.yaml``; the design is
``specs/20261007-task-events-in-background/design.md`` ("Task content in the transcript", "Speech in a
non-lexical task").
"""

from __future__ import annotations

import functools
import re
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from senselab.audio.workflows.audio_analysis.harmonize import normalise_token
from senselab.audio.workflows.triage.nodes.common import find_measurement, word_hull
from senselab.audio.workflows.triage.residue import is_non_lexical
from senselab.audio.workflows.triage.routing_analysis import families as family_sets
from senselab.utils.prov_store import Entity, ProvStore

TASK_CONTENT_PATH = Path(__file__).parent / "data" / "task_content.yaml"

Span = tuple[float, float]


@functools.cache
def task_content_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_content.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(TASK_CONTENT_PATH.read_text()) or {})


def readings_for(family: str | None) -> tuple[str, ...]:
    """The task readings whose events are a family's own content.

    Args:
        family: The declared family, or None.

    Returns:
        The measurement names, in the data file's order; empty for a lexical, undeclared or unknown family.
    """
    if not family:
        return ()
    for set_name, names in (task_content_parameters().get("families") or {}).items():
        if family in getattr(family_sets, str(set_name), frozenset()):
            return tuple(str(name) for name in names or ())
    return ()


def reading_events(attributes: Mapping[str, Any], decisions: Sequence[str]) -> list[Span]:
    """The events one stored task reading read the task as.

    Args:
        attributes: The reading measurement's attributes.
        decisions: The evidence decisions that read the task.

    Returns:
        A phonation reading's holds where it found one; otherwise the evidence's events where its
        decision is one of ``decisions``; empty otherwise.
    """
    if "holds" in attributes:
        if not attributes.get("found"):
            return []
        return [(float(start), float(end)) for start, end in attributes.get("holds") or ()]
    evidence = (attributes.get("reading") or {}).get("evidence") or {}
    if evidence.get("decision") not in decisions:
        return []
    return [(float(event["start_s"]), float(event["end_s"])) for event in evidence.get("events") or ()]


def task_events(store: ProvStore, family: str | None) -> list[Span]:
    """The events the task layer read of the declared family's own task.

    Args:
        store: The provenance store.
        family: The declared family.

    Returns:
        Every event of the family's readings, in time order; empty for a lexical family or where no
        reading read the task.
    """
    decisions = tuple(str(d) for d in task_content_parameters().get("decisions") or ())
    events: list[Span] = []
    for name in readings_for(family):
        measurement = find_measurement(store, name)
        if measurement is not None:
            events.extend(reading_events(measurement.attributes, decisions))
    return sorted(events)


def _merged(spans: Sequence[Span], pad_s: float) -> list[Span]:
    merged: list[list[float]] = []
    for start, end in sorted((start - pad_s, end + pad_s) for start, end in spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def share_inside(hull: Span, spans: Sequence[Span], pad_s: float) -> float:
    """How much of a word's timing hull lies inside the events, each widened by ``pad_s``.

    Args:
        hull: The word's timing hull.
        spans: The events.
        pad_s: The widening either side.

    Returns:
        The covered fraction of the hull; for a zero-length hull, 1.0 where it lies inside an event and
        0.0 otherwise.
    """
    widened = _merged(spans, pad_s)
    low, high = hull
    if high <= low:
        return 1.0 if any(start <= low <= end for start, end in widened) else 0.0
    covered = sum(max(0.0, min(high, end) - max(low, start)) for start, end in widened)
    return covered / (high - low)


_TRIPLED = re.compile(r"(.)\1\1")
_VOWELS = frozenset("aeiouy")


def sound_spelling(text: str) -> bool:
    """Whether a token is spelled as a sound rather than as a dictionary word.

    Args:
        text: The token as the recognizer wrote it.

    Returns:
        True where its letters hold no vowel (``hm``, ``pff``, ``shh``) or one letter three times running
        (``ahhh``, ``hmmm``); False for a token with no letter.
    """
    letters = "".join(ch for ch in str(text).lower() if ch.isalpha())
    return bool(letters) and (not _VOWELS & set(letters) or bool(_TRIPLED.search(letters)))


def readings_disagree(word: Entity) -> bool:
    """Whether the recognisers read one word differently.

    Args:
        word: A consensus word.

    Returns:
        True where its outcome is not ``agreement``, or its readings normalise to more than one token.
    """
    readings = (word.attributes.get("readings") or {}).values()
    keys = {normalise_token(str(reading)) for reading in readings} - {""}
    return word.attributes.get("outcome") != "agreement" or len(keys) > 1


def misreading(word: Entity, lexicon_ids: Collection[str]) -> bool:
    """Whether a word may be a recogniser's misreading of the task's own sound.

    Args:
        word: A consensus word.
        lexicon_ids: The words of the task's own lexicon.

    Returns:
        True where the recognisers disagree on it, it is spelled as a sound, or it is a lexicon word.
    """
    text = str(word.attributes.get("text") or "")
    return readings_disagree(word) or sound_spelling(text) or word.id in lexicon_ids


def agreed_runs(words: Sequence[Entity], lexicon_ids: Collection[str], run_min: int) -> set[str]:
    """The words in runs of agreed dictionary words, which are speech and never task content.

    Args:
        words: The consensus words, in stream order.
        lexicon_ids: The words of the task's own lexicon.
        run_min: The fewest adjacent words that make a run.

    Returns:
        The ids of the lexical words none of which may be a misreading (:func:`misreading`), in runs of at
        least ``run_min`` adjacent words.
    """
    runs: list[list[str]] = [[]]
    for word in words:
        text = str(word.attributes.get("text") or "")
        if word.attributes.get("bracketed") or is_non_lexical(text, vocal_task=True) or misreading(word, lexicon_ids):
            runs.append([])
        else:
            runs[-1].append(word.id)
    return {word_id for run in runs if len(run) >= run_min for word_id in run}


def task_content_ids(
    words: Sequence[Entity],
    events: Sequence[Span],
    *,
    family: str | None = None,
    lexicon_ids: Collection[str] = frozenset(),
) -> set[str]:
    """The words that are the task's own content: recognisers' misreadings lying on its events.

    Args:
        words: The consensus words, in stream order.
        events: The task events (:func:`task_events`).
        family: The declared family.
        lexicon_ids: The words of the task's own lexicon.

    Returns:
        The ids of the timed words whose hull's share inside the widened events is at least
        ``overlap.min_share`` and which are in no run of agreed dictionary words (:func:`agreed_runs`); in a
        family outside ``any_word_families``, only those that are a misreading (:func:`misreading`) and lie
        alone on one event. Empty where there are no events.
    """
    if not events:
        return set()
    parameters = task_content_parameters()
    overlap = parameters["overlap"]
    pad_s, min_share = float(overlap["pad_s"]), float(overlap["min_share"])
    spoken = agreed_runs(words, lexicon_ids, int(parameters["agreed_run_min"]))
    on = [
        word
        for word in words
        if word.extent is not None
        and word.id not in spoken
        and share_inside(word_hull(word), events, pad_s) >= min_share
    ]
    any_word = any(
        family in getattr(family_sets, str(name), frozenset()) for name in parameters.get("any_word_families") or ()
    )
    if any_word:
        return {word.id for word in on}
    widened = _merged(events, pad_s)
    home: dict[str, int] = {}
    for word in on:
        start, end = word_hull(word)
        shares = [max(0.0, min(end, b) - max(start, a)) for a, b in widened]
        home[word.id] = max(range(len(widened)), key=lambda i: shares[i])
    held: dict[int, int] = {}
    for index in home.values():
        held[index] = held.get(index, 0) + 1
    return {word.id for word in on if misreading(word, lexicon_ids) and held[home[word.id]] == 1}
