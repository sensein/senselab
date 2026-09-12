"""QUALITY — the terminal node, reading the store's own records for internal contradiction.

**When it runs.** After every branch, on every path PREPROCESS completed — whatever routing
selected, and whether or not routing itself raised. It is the last node before REDACT and VERDICT,
so every branch has already written whatever it was going to write. ``run._drive_branches`` places
the call; ``run_test`` pins the position.

**What it may read.** Stored outputs only: entities and their attributes. QUALITY decodes no audio,
opens no sidecar and re-derives nothing — every amplitude it compares was measured by the node that
held the signal. The clip-consistency check reads PREPROCESS's ``clip`` spans and the
:data:`CLIP_AMPLITUDE_MEASUREMENT` measurement written beside them.

**What it refuses.** A dependency that is absent is an operational fact, not a finding: clip spans
with no clip-amplitude measurement beside them raise, and the runner records the node ``ERRORED``.
A contradiction QUALITY *can* measure is always a finding and never a raise.

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

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    live_entities,
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

CLIP_AMPLITUDE_MEASUREMENT = "clip_amplitude"
"""PREPROCESS's whole-file amplitude reading of the signal its clip spans were detected on."""

CLIP_LEVEL = "clip_level"
"""The clip span attribute carrying the peak absolute amplitude inside the span."""

UNCLIPPED_LOUDER_N = "unclipped_louder_n"
"""The clip span attribute carrying how many unclipped samples exceed that span's own level."""


@dataclass(frozen=True)
class _Contradiction:
    """One clip span an unclipped sample contradicts.

    Attributes:
        span_id: The clip span's entity id.
        extent: The span's extent, in seconds.
        clip_level: The peak absolute amplitude inside the span — the level it calls the ceiling.
        louder_amplitude: The loudest unclipped sample's absolute amplitude.
        louder_time_s: Where that sample sits, in seconds.
        louder_samples_n: How many unclipped samples exceed ``clip_level``.
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


def _stream_id(store: ProvStore, name: str) -> str:
    """The live stream entity's id, by name, without decoding it.

    Reads by the store's shared rule — invalidated entities are never returned, latest write wins —
    which is ``resolve_stream``'s rule with the load left out: QUALITY names the stream its findings
    are about and never opens it.

    Args:
        store: The provenance store.
        name: The stream entity's ``name`` attribute.

    Returns:
        The stream entity's id.

    Raises:
        LookupError: If no live stream entity carries that name.
    """
    found = [entity for entity in live_entities(store, "stream") if entity.attributes.get("name") == name]
    if not found:
        raise LookupError(f"no stream named {name!r} in the store; the node that writes it has not run")
    return found[-1].id


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


def _clip_amplitudes(store: ProvStore, signal: str) -> Entity:
    """PREPROCESS's clip-amplitude measurement over this signal.

    Args:
        store: The provenance store.
        signal: The stream name the clip spans were detected on.

    Returns:
        The live :data:`CLIP_AMPLITUDE_MEASUREMENT` measurement entity.

    Raises:
        LookupError: If nothing live carries that name over ``signal``.
    """
    found = find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT)
    if found is None or found.attributes.get("signal") != signal:
        raise LookupError(
            f"no live {CLIP_AMPLITUDE_MEASUREMENT!r} measurement over {signal!r}, but the store carries "
            "clip spans over it; PREPROCESS writes the two together and QUALITY reads no audio of its own"
        )
    return found


def quality(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Read PREPROCESS's clip spans against its own amplitude reading, and contest the denied ones.

    A clip span's level is the peak absolute amplitude of the samples it covers, which PREPROCESS
    measured while it held the signal and stored on the span. An unclipped sample contradicts that
    span when its own absolute amplitude exceeds the level by more than
    ``quality.clip_contradiction_margin`` of the level. What counts as unclipped — the edge guard
    included — was decided by PREPROCESS, and the guard it used is recorded on the measurement.

    Args:
        store: The provenance store, holding ADMIT's recording stream, PREPROCESS's clip spans and
            the clip-amplitude measurement beside them.
        source: The store-held stream the clip spans were detected on, ``"recording"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read. No declaration can make a recording
            internally consistent.
        run_dir: Accepted for the shared node shape; not read. QUALITY writes no sidecar and opens
            none.

    Returns:
        The verdict, the view over the assertions written, and the verdict entity id.

    Raises:
        ValueError: If the key read at entry is null — raised before the store is written to.
        LookupError: If the ``source`` stream is absent, or if clip spans over it carry no
            clip-amplitude measurement to read them against.
    """
    del hint, run_dir
    margin = float(config.require("quality.clip_contradiction_margin"))
    stream_id = _stream_id(store, source)
    software = software_agent(store)

    spans = _clip_spans(store, source)
    amplitudes = _clip_amplitudes(store, source).attributes if spans else {}
    guard = int(amplitudes.get("edge_guard_samples", 0))
    peak = amplitudes.get("unclipped_peak")
    peak_time_s = amplitudes.get("unclipped_peak_time_s")
    unclipped_samples_n = int(amplitudes.get("unclipped_samples_n", 0))

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

    measured = [
        (span, float(span.attributes[CLIP_LEVEL])) for span in spans if span.attributes.get(CLIP_LEVEL) is not None
    ]
    contradictions: list[_Contradiction] = []
    for span, clip_level in measured:
        if peak is None or peak_time_s is None or float(peak) <= clip_level * (1.0 + margin):
            continue
        start_s, end_s = span.extent or (0.0, 0.0)
        contradictions.append(
            _Contradiction(
                span_id=span.id,
                extent=(float(start_s), float(end_s)),
                clip_level=clip_level,
                louder_amplitude=float(peak),
                louder_time_s=float(peak_time_s),
                louder_samples_n=int(span.attributes.get(UNCLIPPED_LOUDER_N, 0)),
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
            "unclipped_samples_n": unclipped_samples_n,
            "unclipped_peak": None if peak is None else float(peak),
            "unclipped_peak_time_s": None if peak_time_s is None else float(peak_time_s),
            "clip_contradiction_margin": margin,
            "clip_edge_guard_samples": guard,
            "contradictions": [contradiction.as_detail() for contradiction in contradictions],
            "flags": flags,
        },
    )
    return NodeResult(verdict=verdict, view=(*assertion_ids, verdict_id), verdict_entity_id=verdict_id)
