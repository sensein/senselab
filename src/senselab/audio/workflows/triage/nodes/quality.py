"""QUALITY — the terminal node, reading the recording's own evidence for internal contradiction.

Its first check is clip consistency. A clip span asserts that the signal reached its ceiling over
that extent; a sample outside every clip span, louder than that ceiling, contradicts the assertion.
QUALITY records each contradiction as an ``assertion`` derived from the span it contests and names
the count in its verdict. PREPROCESS's spans are never invalidated here: the store is append-only
and the span is PREPROCESS's reading, not QUALITY's to withdraw. The design is in
``specs/20260912-quality-clip-consistency/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    live_entities,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "QUALITY"
KIND: str | None = None
CLIP_FAMILY = "clip"
CONTRADICTED_CLIP = "clip_above_unclipped_sample"
"""The vocabulary token for a clip span an unclipped sample is louder than."""


@dataclass(frozen=True)
class _Contradiction:
    """One clip span an unclipped sample contradicts.

    Attributes:
        span_id: The clip span's entity id.
        extent: The span's extent, in seconds.
        clip_level: The peak absolute amplitude inside the span — the level it calls the ceiling.
        louder_amplitude: The loudest unclipped sample's absolute amplitude.
        louder_time_s: Where that sample sits, in seconds.
        louder_samples_n: How many unclipped samples exceed ``clip_level`` by more than the margin.
    """

    span_id: str
    extent: tuple[float, float]
    clip_level: float
    louder_amplitude: float
    louder_time_s: float
    louder_samples_n: int

    def as_detail(self) -> dict[str, Any]:
        """The row this contradiction contributes to the verdict and to its assertion.

        Returns:
            The fields, without the span id, which the assertion carries as a derivation edge.
        """
        return {
            "extent": list(self.extent),
            "clip_level": self.clip_level,
            "louder_amplitude": self.louder_amplitude,
            "louder_time_s": self.louder_time_s,
            "louder_samples_n": self.louder_samples_n,
        }


def _clip_spans(store: ProvStore, signal: str) -> list[Entity]:
    """Every live clip span PREPROCESS proposed over this signal, in time order.

    Args:
        store: The provenance store.
        signal: The stream name the spans were detected on.

    Returns:
        The ``span`` entities whose ``family`` is ``clip`` and whose ``signal`` is ``signal``,
        sorted by extent.
    """
    spans = [
        entity
        for entity in live_entities(store, "span")
        if entity.attributes.get("family") == CLIP_FAMILY
        and entity.attributes.get("signal") == signal
        and entity.extent is not None
    ]
    return sorted(spans, key=lambda entity: entity.extent or (0.0, 0.0))


def _sample_range(extent: tuple[float, float], sampling_rate: int, samples_n: int) -> tuple[int, int]:
    """The half-open sample range an extent names, clamped to the decoded signal.

    Args:
        extent: ``(start, end)`` in seconds.
        sampling_rate: The signal's sampling rate.
        samples_n: How many samples the signal decoded to.

    Returns:
        ``(first, stop)`` with ``stop`` exclusive; ``stop <= first`` when the extent names no
        sample of this signal.
    """
    first = int(round(extent[0] * sampling_rate))
    stop = int(round(extent[1] * sampling_rate))
    return max(0, min(first, samples_n)), max(0, min(stop, samples_n))


def _unclipped_mask(samples_n: int, ranges: list[tuple[int, int]], guard: int) -> np.ndarray:
    """Which samples are outside every clip span and outside every span's guard band.

    Args:
        samples_n: How many samples the signal decoded to.
        ranges: The half-open sample range of each clip span.
        guard: How many samples each side of a span are excluded with it.

    Returns:
        A boolean mask, true where a sample is unclipped evidence.
    """
    excluded = np.zeros(samples_n, dtype=bool)
    for first, stop in ranges:
        excluded[max(0, first - guard) : min(samples_n, stop + guard)] = True
    return ~excluded


def quality(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Read PREPROCESS's clip spans against the recording, and contest the ones a louder sample denies.

    A clip span's level is the peak absolute amplitude of the samples it covers, on the same
    channel-averaged signal ``detect_clip_events`` read. An unclipped sample contradicts that span
    when its own absolute amplitude exceeds the level by more than
    ``quality.clip_contradiction_margin`` of the level; samples within
    ``quality.clip_edge_guard_samples`` of any span edge are not unclipped evidence.

    Args:
        store: The provenance store, holding ADMIT's recording stream and PREPROCESS's clip spans.
        source: The store-held stream the clip spans were detected on, ``"recording"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read. No declaration can make a recording
            internally consistent.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The verdict, the view over the assertions written, and the verdict entity id.

    Raises:
        ValueError: If either key read at entry is null — raised before the store is written to.
        LookupError: If the ``source`` stream is absent.
    """
    margin = float(config.require("quality.clip_contradiction_margin"))
    guard = int(config.require("quality.clip_edge_guard_samples"))
    stream_id, recording = resolve_stream(store, run_dir, source)
    software = software_agent(store)

    spans = _clip_spans(store, source)
    activity = store.activity(
        node=NODE,
        step="clip_consistency",
        parameters={
            "signal": source,
            "clip_contradiction_margin": margin,
            "clip_edge_guard_samples": guard,
            "clip_spans_n": len(spans),
        },
    )
    store.was_associated_with(activity, software)
    store.used(activity, stream_id)
    for span in spans:
        store.used(activity, span.id)

    magnitude = np.abs(recording.waveform.mean(dim=0).numpy().astype(np.float64))
    sampling_rate = int(recording.sampling_rate)
    samples_n = int(magnitude.shape[0])
    ranges = [_sample_range(span.extent or (0.0, 0.0), sampling_rate, samples_n) for span in spans]
    measured = [(span, first, stop) for span, (first, stop) in zip(spans, ranges) if stop > first]

    unclipped = np.flatnonzero(_unclipped_mask(samples_n, ranges, guard))
    levels = np.sort(magnitude[unclipped]) if unclipped.size else np.empty(0)
    peak = float(levels[-1]) if levels.size else None
    peak_time_s = float(unclipped[int(np.argmax(magnitude[unclipped]))] / sampling_rate) if unclipped.size else None

    contradictions: list[_Contradiction] = []
    for span, first, stop in measured:
        clip_level = float(magnitude[first:stop].max())
        louder_samples_n = int(levels.size - np.searchsorted(levels, clip_level * (1.0 + margin), side="right"))
        if louder_samples_n == 0 or peak is None or peak_time_s is None:
            continue
        start_s, end_s = span.extent or (0.0, 0.0)
        contradictions.append(
            _Contradiction(
                span_id=span.id,
                extent=(float(start_s), float(end_s)),
                clip_level=clip_level,
                louder_amplitude=peak,
                louder_time_s=peak_time_s,
                louder_samples_n=louder_samples_n,
            )
        )

    assertion_ids: list[str] = []
    for contradiction in contradictions:
        assertion_id = store.entity(
            prov_type="assertion",
            extent=contradiction.extent,
            attributes={
                "verb": "contest",
                "claim": CLIP_FAMILY,
                "reason": CONTRADICTED_CLIP,
                "signal": source,
                "margin": margin,
                **contradiction.as_detail(),
            },
        )
        store.was_generated_by(assertion_id, activity)
        store.was_attributed_to(assertion_id, software)
        store.was_derived_from(assertion_id, contradiction.span_id)
        assertion_ids.append(assertion_id)

    flags: list[str] = []
    if contradictions:
        loudest = max(contradictions, key=lambda found: found.louder_amplitude)
        flags.append(
            f"{CONTRADICTED_CLIP}: {len(contradictions)} of {len(measured)} clip spans sit below the "
            f"unclipped sample of amplitude {loudest.louder_amplitude:.4f} at {loudest.louder_time_s:.3f}s"
        )

    if flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    elif not measured:
        outcome, why = Outcome.PASS, "no clip span over the recording; nothing to contradict"
    else:
        outcome, why = Outcome.PASS, "no clip span sits below an unclipped sample"

    verdict_id, verdict = write_verdict(
        store,
        activity,
        software,
        node=NODE,
        outcome=outcome,
        kind=KIND,
        why=why,
        detail={
            "signal": source,
            "clip_spans_n": len(spans),
            "checked_n": len(measured),
            "unmeasurable_n": len(spans) - len(measured),
            "contradicted_n": len(contradictions),
            "unclipped_samples_n": int(unclipped.size),
            "unclipped_peak": peak,
            "unclipped_peak_time_s": peak_time_s,
            "clip_contradiction_margin": margin,
            "clip_edge_guard_samples": guard,
            "contradictions": [contradiction.as_detail() for contradiction in contradictions],
            "flags": flags,
        },
    )
    return NodeResult(verdict=verdict, view=(*assertion_ids, verdict_id), verdict_entity_id=verdict_id)
