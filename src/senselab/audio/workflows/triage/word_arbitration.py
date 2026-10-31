"""Word timing arbitration: where the recognizers place a word far apart, it takes the time that falls on sound.

Text and timing are separate choices. The consensus keeps the word's text; this step moves only its time, to the
one recognizer's span whose midpoint lies on PREPROCESS's acoustic activity (and, in a voice family, off the held
phonation), and marks a word no recognizer places on sound ``unlocated``. PREPROCESS runs it after its spans;
:func:`arbitrate_words` also brings a finished store to it. Parameters: ``data/word_timing.yaml``
``arbitration``. Design: ``specs/20261012-r37-words-speaker/design.md`` ("B1").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from senselab_core.runtime.provenance import ProvStore, revised

from senselab.audio.workflows.triage.consensus import (
    ConsensusWord,
    arbitrate_time,
    arbitration_parameters,
    located_span,
    render_transcript,
    word_attributes,
    word_from_attributes,
)
from senselab.audio.workflows.triage.nodes.branches import VOICE_EXPECTATIONS, derivative_arrays, merge
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    live_entities,
    recording_duration,
    software_agent,
    write_measurement,
)

PREPROCESS_NODE = "PREPROCESS"
CONSENSUS_TRANSCRIPT = "consensus_transcript"
WORD_ARBITRATION = "word_arbitration"
"""The measurement recording what the arbitration moved, and the step it runs under."""
PHONATION_TRACKS = "phonation_tracks"
ACTIVITY_MEASURE = "amplitude"
WORD_SUPERSEDED = "word_superseded"
_WORD_REASON = "the recognizers placed the word far apart; its time was arbitrated on sound"
_TRANSCRIPT_REASON = "rewritten with the arbitrated words"

Interval = tuple[float, float]


def activity_spans(store: ProvStore) -> list[Interval]:
    """PREPROCESS's acoustic activity: every live ``amplitude`` span, on any signal, merged.

    Args:
        store: The provenance store.

    Returns:
        The merged spans, earliest first.
    """
    return merge(
        [
            (float(span.extent[0]), float(span.extent[1]))
            for span in live_entities(store, "span")
            if span.attributes.get("measure") == ACTIVITY_MEASURE and span.extent is not None
        ]
    )


def held_voicing(times_s: np.ndarray, voiced: np.ndarray, *, min_s: float, bridge_s: float) -> list[Interval]:
    """The voiced runs long enough to be a held phonation rather than a word.

    Args:
        times_s: Frame times.
        voiced: Whether each frame is voiced.
        min_s: The shortest held run.
        bridge_s: The longest unvoiced gap bridged inside a run.

    Returns:
        The held runs, earliest first.
    """
    if times_s.size == 0:
        return []
    hop = float(np.median(np.diff(times_s))) if times_s.size > 1 else 0.0
    runs: list[list[float]] = []
    for time, on in zip(times_s.tolist(), voiced.tolist()):
        if not on:
            continue
        if runs and time - runs[-1][1] <= bridge_s + hop:
            runs[-1][1] = time
        else:
            runs.append([time, time])
    return [(start, end + hop) for start, end in runs if end + hop - start >= min_s]


def phonation_holds(store: ProvStore, run_dir: Path | None, family: str | None) -> list[Interval]:
    """A voice family's held phonation, off PREPROCESS's phonation tracks; empty for any other family.

    Args:
        store: The provenance store.
        run_dir: The run directory the tracks' sidecar is relative to, or None.
        family: The declared family.

    Returns:
        The held runs (:func:`held_voicing`).
    """
    if run_dir is None or family not in VOICE_EXPECTATIONS:
        return []
    arrays = derivative_arrays(store, run_dir, PHONATION_TRACKS)
    if arrays is None or "times_s" not in arrays or "strength" not in arrays:
        return []
    p = arbitration_parameters()
    f0 = arrays.get("f0_hz")
    voiced = np.asarray(arrays["strength"], dtype=np.float64) >= p["voiced_strength_min"]
    if f0 is not None:
        voiced &= np.isfinite(np.asarray(f0, dtype=np.float64)) & (np.asarray(f0, dtype=np.float64) > 0)
    return held_voicing(
        np.asarray(arrays["times_s"], dtype=np.float64),
        voiced,
        min_s=p["held_voicing_min_s"],
        bridge_s=p["held_voicing_bridge_s"],
    )


def _subtract(span: Interval, holes: Sequence[Interval]) -> list[Interval]:
    """What is left of a span once every hole is cut out of it, earliest first."""
    pieces = [span]
    for lo, hi in holes:
        pieces = [
            piece
            for start, end in pieces
            for piece in ((start, min(end, lo)), (max(start, hi), end))
            if piece[1] > piece[0]
        ]
    return pieces


def with_mask_gaps(
    words: Sequence[ConsensusWord], *, duration_s: float | None, off_limits: Sequence[Interval] = ()
) -> list[ConsensusWord]:
    """Each unlocated word given the place a mask over it covers: the gap between its timed neighbours.

    A neighbour is the nearest word on either side that is not unlocated and carries a non-empty extent. With
    none before, the gap opens at 0; with none after, it closes at ``duration_s`` (else the latest word end).
    Where the neighbours touch or overlap, the gap is the hull of both neighbours. No word lies on the held
    phonation (``off_limits``), so the gap less the held phonation is kept where a recogniser's placement
    midpoint falls in what is left (the hull of those pieces); where none does, the whole gap.

    Args:
        words: The positions, in stream order, as arbitrated.
        duration_s: The recording's duration, or None.
        off_limits: The held phonation, in a voice family.

    Returns:
        The positions, every unlocated one with ``mask_gap`` set.
    """
    timed = [i for i, word in enumerate(words) if not word.unlocated and word.extent[1] > word.extent[0]]
    end_of_stream = (
        float(duration_s) if duration_s is not None else max((word.extent[1] for word in words), default=0.0)
    )
    out = list(words)
    for i, word in enumerate(words):
        if not word.unlocated:
            continue
        before = [j for j in timed if j < i]
        after = [j for j in timed if j > i]
        prev = words[before[-1]].extent if before else None
        nxt = words[after[0]].extent if after else None
        start = prev[1] if prev is not None else 0.0
        end = nxt[0] if nxt is not None else end_of_stream
        if end <= start:
            start = prev[0] if prev is not None else 0.0
            end = nxt[1] if nxt is not None else end_of_stream
        gap = (float(start), float(max(start, end)))
        midpoints = [
            (span[0] + span[1]) / 2.0
            for a, b in word.timings.values()
            if (span := located_span(a, b, duration_s=duration_s)) is not None
        ]
        held = [piece for piece in _subtract(gap, off_limits) if any(piece[0] <= m <= piece[1] for m in midpoints)]
        if held:
            gap = (min(piece[0] for piece in held), max(piece[1] for piece in held))
        out[i] = replace(word, mask_gap=(round(gap[0], 3), round(gap[1], 3)))
    return out


@dataclass(frozen=True)
class Arbitrated:
    """What the arbitration made of one transcript.

    Attributes:
        words: Every position, re-timed where it was arbitrated.
        moved: The positions (indices into ``words``) whose time or flags changed.
        record: The ``word_arbitration`` measurement's fields.
    """

    words: tuple[ConsensusWord, ...]
    moved: tuple[int, ...]
    record: dict[str, Any]


def arbitrate(
    words: Sequence[ConsensusWord],
    *,
    on_sound: Sequence[Interval],
    off_limits: Sequence[Interval] = (),
    duration_s: float | None = None,
) -> Arbitrated:
    """Arbitrate every position's time (:func:`~senselab.audio.workflows.triage.consensus.arbitrate_time`).

    Args:
        words: The positions, in stream order.
        on_sound: The acoustic activity.
        off_limits: The held phonation, in a voice family.
        duration_s: The duration the spans were timed against.

    Returns:
        The words, the moved positions and the counts: ``disagreed_n`` (positions the recognizers placed
        further apart than the bound), ``retimed_n`` and ``unlocated_n``, and per source how many it won.
    """
    bound = arbitration_parameters()["disagreement_s"]
    out: list[ConsensusWord] = []
    moved: list[int] = []
    won: dict[str, int] = {}
    disagreed = 0
    for position, word in enumerate(words):
        result = arbitrate_time(
            word, on_sound=on_sound, off_limits=off_limits, disagreement_s=bound, duration_s=duration_s
        )
        starts = [
            span[0]
            for start, end in word.timings.values()
            if (span := located_span(start, end, duration_s=duration_s)) is not None
        ]
        if len(starts) >= 2 and max(starts) - min(starts) > bound:
            disagreed += 1
        if result != word:
            moved.append(position)
            if result.arbitrated_source is not None:
                won[result.arbitrated_source] = won.get(result.arbitrated_source, 0) + 1
        out.append(result)
    out = with_mask_gaps(out, duration_s=duration_s, off_limits=off_limits)
    return Arbitrated(
        words=tuple(out),
        moved=tuple(moved),
        record={
            "disagreement_s": bound,
            "disagreed_n": disagreed,
            "retimed_n": sum(won.values()),
            "unlocated_n": sum(1 for word in out if word.unlocated),
            "unlocated_mask_gap_s": round(sum(w.mask_gap[1] - w.mask_gap[0] for w in out if w.mask_gap), 3),
            "won_by_source": dict(sorted(won.items())),
            "held_phonation": [[round(a, 3), round(b, 3)] for a, b in off_limits],
            "activity_s": round(sum(b - a for a, b in on_sound), 3),
        },
    )


def arbitrate_words(store: ProvStore, *, run_dir: Path | None, family: str | None) -> str | None:
    """Arbitrate a store's consensus words, superseding each moved word and the transcript listing them.

    A word that does not move keeps its id. Nothing is written where no word moves; the ``word_arbitration``
    measurement is written either way once a transcript stands, so a store records that it was read.

    Args:
        store: The provenance store.
        run_dir: The run directory, for the phonation tracks of a voice family; None reads none.
        family: The declared family.

    Returns:
        The ``word_arbitration`` measurement's id, or None where the store carries no consensus transcript.
    """
    consensus = find_measurement(store, CONSENSUS_TRANSCRIPT)
    if consensus is None:
        return None
    stored = [store.get_entity(str(word_id)) for word_id in consensus.attributes.get("word_ids") or ()]
    stored = [entity for entity in stored if not store.is_invalidated(entity.id) and entity.extent is not None]
    words = [word_from_attributes(entity.attributes, entity.extent) for entity in stored]  # type: ignore[arg-type]
    holds = phonation_holds(store, run_dir, family)
    result = arbitrate(words, on_sound=activity_spans(store), off_limits=holds, duration_s=recording_duration(store))
    software = software_agent(store)
    activity = store.activity(
        node=PREPROCESS_NODE,
        step=WORD_ARBITRATION,
        parameters={**arbitration_parameters(), "family": family},
    )
    store.was_associated_with(activity, software)
    store.used(activity, consensus.id)
    transcript_id = consensus.id
    if result.moved:
        moved = set(result.moved)
        word_ids: list[str] = []
        for position, (entity, word) in enumerate(zip(stored, result.words)):
            if position not in moved:
                word_ids.append(entity.id)
                continue
            word_id = store.entity(prov_type="word", extent=word.extent, attributes=word_attributes(word))
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, software)
            store.was_derived_from(word_id, entity.id)
            _supersede(store, entity.id, WORD_SUPERSEDED, _WORD_REASON, software)
            word_ids.append(word_id)
        transcript_id = write_measurement(
            store,
            activity,
            software,
            name=CONSENSUS_TRANSCRIPT,
            signal=str(consensus.attributes["signal"]),
            attributes=revised(
                {k: v for k, v in consensus.attributes.items() if k not in ("name", "signal")},
                word_ids=word_ids,
                text=render_transcript(result.words, strong=("", "")),
                arbitrated_n=result.record["retimed_n"],
                arbitration_unlocated_n=result.record["unlocated_n"],
            ),
            derived_from=(consensus.id,),
            extent=consensus.extent,
        )
        _supersede(store, consensus.id, f"{CONSENSUS_TRANSCRIPT}_superseded", _TRANSCRIPT_REASON, software)
    return write_measurement(
        store,
        activity,
        software,
        name=WORD_ARBITRATION,
        signal=str(consensus.attributes["signal"]),
        attributes={**result.record, "moved_n": len(result.moved), "consensus_transcript": transcript_id},
        derived_from=(transcript_id,),
    )


def _supersede(store: ProvStore, entity_id: str, step: str, reason: str, software: str) -> None:
    retire = store.activity(node=PREPROCESS_NODE, step=step, parameters={"superseded": entity_id, "reason": reason})
    store.was_associated_with(retire, software)
    store.used(retire, entity_id)
    store.was_invalidated_by(entity_id, retire)
