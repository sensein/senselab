"""QUALITY — the join of the recording-level background against the task spans, on every recording.

Runs after every branch, on every path PREPROCESS completed, and before REDACT and VERDICT;
``run._drive_branches`` places the call. The recording-level background is not QUALITY's: it is
read before the branches (:mod:`~senselab.audio.workflows.triage.nodes.background`). QUALITY joins
it against the branches' task spans (:func:`write_join`, :mod:`~senselab.audio.workflows.triage.quality_join`)
and writes the clip-consistency audit below, a self-check of the store's own records.

The audit is clip consistency. A clip span asserts that the signal reached its ceiling over its
extent; a sample outside every clip span, louder than that ceiling, contradicts the assertion.
QUALITY reads PREPROCESS's ``clip`` spans for their extents and its
:data:`CLIP_AMPLITUDE_MEASUREMENT` for the amplitudes, and records each contradiction as an
``assertion`` derived from the span it contests. It withdraws no span: the store is append-only.
Clip spans with no clip-amplitude measurement beside them raise, and the runner records the node
``ERRORED``; a contradiction QUALITY can measure is always a finding and never a raise.

``preceded_by`` lists the nodes whose records were live in the store when QUALITY read it, which on
a fresh run is routing and the branches and on the ``scripts/extend_quality.py`` pass is whatever
that run actually ran.

The design is in ``specs/20260912-quality-clip-consistency/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.background_model import BACKGROUND_MODEL, _named_stream
from senselab.audio.workflows.triage.background_speech import background_speech_of
from senselab.audio.workflows.triage.config import TriageConfig, UnmeasuredConfigKey
from senselab.audio.workflows.triage.nodes.airway_task import BREATH_READING, COUGH_READING
from senselab.audio.workflows.triage.nodes.background import SESSION_FLOOR
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    consensus_words,
    find_measurement,
    live_entities,
    software_agent,
    write_measurement,
    write_report,
)
from senselab.audio.workflows.triage.nodes.voice import PHONATION_READING
from senselab.audio.workflows.triage.quality_join import (
    QUALITY_JOIN,
    Span,
    join_record,
    quality_join_parameters,
    stream_agreement,
)
from senselab.audio.workflows.triage.task_events import generic_view, generic_view_of
from senselab.audio.workflows.triage.vocabulary import (
    STORE_ASSERTIONS,
    UNDETERMINED,
    Conformance,
    standing_task_extents,
)
from senselab.utils.prov_store import Entity, ProvStore

ENHANCED_STREAM = "enhanced"
EVENT_READINGS = ((BREATH_READING, "breath"), (COUGH_READING, "cough"), (PHONATION_READING, "phonation"))
"""The task readings whose events the join re-reads, with the event kind each holds."""

NODE = "QUALITY"
KIND: str | None = None
CLIP_FAMILY = "clip"
CONTRADICTED_CLIP = "clip_above_unclipped_sample"
"""The vocabulary token for a clip span an unclipped sample is louder than."""

CONTEST_VERB = "contest"
"""The assertion verb for a clip span this node answers but does not withdraw."""

CLIP_AMPLITUDE_MEASUREMENT = "clip_amplitude"
"""PREPROCESS's amplitude reading of the signal its clip spans were detected on.

It carries the whole-file values and, keyed by span id, the per-span ones.
"""

CLIP_LEVELS = "clip_levels"
"""The measurement key mapping each clip span's id to the peak absolute amplitude inside it."""

UNCLIPPED_LOUDER_N = "unclipped_louder_n"
"""The measurement key mapping each clip span's id to how many unclipped samples exceed its level."""


@dataclass(frozen=True)
class _Contradiction:
    """One clip span an unclipped sample contradicts.

    Attributes:
        span_id: The clip span's entity id.
        extent: The span's extent, in seconds.
        clip_level: The peak absolute amplitude inside the span.
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
        """The row this contradiction contributes to the report and to its assertion.

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


def preceded_by(store: ProvStore) -> list[str]:
    """The nodes that had concluded, in this store, by the time QUALITY read it.

    Args:
        store: The provenance store.

    Returns:
        The ``node`` of every live verdict and every live branch report other than QUALITY's own,
        sorted and deduplicated.
    """
    return sorted(
        {
            str(entity.attributes["node"])
            for entity in (*live_entities(store, "verdict"), *live_entities(store, "branch_report"))
            if entity.attributes.get("node") != NODE
        }
    )


def _stream_id(store: ProvStore, name: str) -> str:
    """The live stream entity's id, by name, without decoding it.

    Invalidated entities are never returned and the latest write wins.

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


def clip_spans(store: ProvStore, signal: str) -> list[Entity]:
    """Every live clip span PREPROCESS proposed over this signal, in time order.

    Public: the extend pass that appends a missing :data:`CLIP_AMPLITUDE_MEASUREMENT` to a finished
    run selects the same spans.

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
            "clip spans over it; QUALITY reads no audio of its own. A fresh run writes the two together; "
            "a finished run gains the measurement from scripts/extend_clip_amplitudes.py"
        )
    return found


def task_spans_of(store: ProvStore) -> list[Span]:
    """The task spans that stand: the branches' task extents, or the measure's where it superseded them.

    Args:
        store: The provenance store.

    Returns:
        Each standing ``task_extent`` span's extent, in time order.
    """
    spans = standing_task_extents(live_entities(store, "span"))
    return sorted((float(s.extent[0]), float(s.extent[1])) for s in spans if s.extent is not None)


def task_events_of(store: ProvStore) -> tuple[str | None, list[Span]]:
    """The task events the branches read, and their kind.

    Args:
        store: The provenance store, read for the breath, cough and phonation readings.

    Returns:
        ``(kind, events)``: the first reading of :data:`EVENT_READINGS` that holds events, with its
        breath or cough events or its phonation holds; ``(None, [])`` where none does.
    """
    for name, kind in EVENT_READINGS:
        found = find_measurement(store, name)
        if found is None:
            continue
        attributes = found.attributes
        if kind == "phonation":
            events = [(float(a), float(b)) for a, b in attributes.get("holds") or ()]
        else:
            evidence = (attributes.get("reading") or {}).get("evidence") or {}
            events = [(float(e["start_s"]), float(e["end_s"])) for e in evidence.get("events") or ()]
        if events:
            return kind, events
    return None, []


def measure_join(store: ProvStore, run_dir: Path) -> dict[str, Any]:
    """The join of the recording-level background against the branches' task spans.

    Args:
        store: The provenance store, read for the task spans and events, BACKGROUND's reading, the
            session floor, the consensus words and the plain, enhanced and residual streams.
        run_dir: The run directory the streams' and sidecars' paths are relative to.

    Returns:
        :func:`~senselab.audio.workflows.triage.quality_join.join_record`, with ``missing`` naming
        what was not stored. Without BACKGROUND's reading the record carries ``missing`` alone.
    """
    background = find_measurement(store, BACKGROUND_MODEL)
    if background is None or background.attributes.get("missing"):
        return {"missing": [BACKGROUND_MODEL]}
    p = quality_join_parameters()
    spans = task_spans_of(store)
    kind, events = task_events_of(store)
    missing: list[str] = []
    raw = generic_view_of(store, run_dir)
    enhanced_signal = _named_stream(store, run_dir, ENHANCED_STREAM)
    enhanced = generic_view(enhanced_signal, None) if enhanced_signal is not None else None
    if enhanced is None:
        missing.append(ENHANCED_STREAM)
    streams = stream_agreement(events, raw, enhanced, p) if raw is not None and events else None
    other_voice: list[Span] | None = None
    if spans:
        words = [(float(w.extent[0]), float(w.extent[1])) for w in consensus_words(store) if w.extent is not None]
        hull = (spans[0][0], max(b for _, b in spans))
        heard = background_speech_of(store, run_dir, hull, [*events, *words])
        if heard is None:
            missing.append("background_speech")
        else:
            other_voice = [(float(w[0]), float(w[1])) for w in (*heard.speech_windows, *heard.voice_runs)]
    session = find_measurement(store, SESSION_FLOOR)
    level = session.attributes.get("level_rel_db") if session is not None else None
    record = join_record(
        task_spans=spans,
        event_kind=kind,
        faults=dict(background.attributes.get("faults") or {}),
        other_voice=other_voice,
        streams=streams,
        plain_active_s=background.attributes.get("active_s"),
        enhanced_active_s=None if enhanced is None else float(sum(r.end_s - r.start_s for r in enhanced.regions)),
        level_rel_db=None if level is None else float(level),
        p=p,
    )
    return {**record, "missing": missing}


def write_join(store: ProvStore, activity: str, software: str, run_dir: Path) -> str:
    """Write the join's measurement (:func:`measure_join`).

    Args:
        store: The provenance store.
        activity: QUALITY's activity.
        software: The software agent.
        run_dir: The run directory.

    Returns:
        The measurement's id.
    """
    return write_measurement(
        store, activity, software, name=QUALITY_JOIN, signal="plain", attributes=measure_join(store, run_dir)
    )


def quality(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> BranchResult:
    """Join the background against the task spans, and audit PREPROCESS's clip spans against their amplitudes.

    The join (:func:`write_join`) is one measurement every recording gets. A clip span's level is the
    peak absolute amplitude of the samples it covers, which PREPROCESS stored in the clip-amplitude
    measurement under the span's id. An unclipped sample contradicts
    that span when its own absolute amplitude exceeds the level by more than
    ``quality.clip_contradiction_margin`` of the level. What counts as unclipped, the edge guard
    included, was decided by PREPROCESS and is recorded on the measurement.

    Args:
        store: The provenance store, holding ADMIT's recording stream, PREPROCESS's clip spans and
            the clip-amplitude measurement they are read against.
        source: The store-held stream the clip spans were detected on, ``"recording"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: The run directory the join reads streams and sidecars under.

    Returns:
        The branch report, the view over the assertions and the join written, and the ``branch_report``
        entity's id.

    Raises:
        UnknownConfigKey: If ``quality.clip_contradiction_margin`` is not a packaged key. A key
            that is packaged but null is reported instead, in ``unmeasured``.
        LookupError: If the ``source`` stream is absent, or if clip spans over it carry no
            clip-amplitude measurement to read them against.
    """
    del hint
    # An unmeasured margin is reported, not raised on; a misspelled key still raises.
    margin: float | None
    try:
        margin = float(config.require("quality.clip_contradiction_margin"))
    except UnmeasuredConfigKey:
        margin = None
    stream_id = _stream_id(store, source)
    preceded = preceded_by(store)
    software = software_agent(store)

    spans = clip_spans(store, source)
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

    levels = amplitudes.get(CLIP_LEVELS) or {}
    louder_counts = amplitudes.get(UNCLIPPED_LOUDER_N) or {}
    # The comparison IS the margin, so an unmeasured margin leaves the checked set empty.
    measured = (
        [] if margin is None else [(span, float(levels[span.id])) for span in spans if levels.get(span.id) is not None]
    )
    contradictions: list[_Contradiction] = []
    for span, clip_level in measured:
        if peak is None or peak_time_s is None or margin is None or float(peak) <= clip_level * (1.0 + margin):
            continue
        start_s, end_s = span.extent or (0.0, 0.0)
        contradictions.append(
            _Contradiction(
                span_id=span.id,
                extent=(float(start_s), float(end_s)),
                clip_level=clip_level,
                louder_amplitude=float(peak),
                louder_time_s=float(peak_time_s),
                louder_samples_n=int(louder_counts.get(span.id, 0)),
            )
        )

    assertion_ids: list[str] = []
    for contradiction in contradictions:
        assertion_id = store.entity(
            prov_type="assertion",
            extent=contradiction.extent,
            attributes={
                "verb": CONTEST_VERB,
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

    notes: list[str] = []
    if contradictions:
        loudest = max(contradictions, key=lambda found: found.louder_amplitude)
        notes.append(
            f"{CONTRADICTED_CLIP}: {len(contradictions)} of {len(measured)} clip spans sit below the "
            f"unclipped sample of amplitude {loudest.louder_amplitude:.4f} at {loudest.louder_time_s:.3f}s"
        )

    # The referent is STORE_ASSERTIONS, not TASK: QUALITY has no route and no declared task.
    conformance: Conformance = UNDETERMINED if not measured else not contradictions

    join_activity = store.activity(
        node=NODE, step=QUALITY_JOIN, parameters={"version": quality_join_parameters()["version"]}
    )
    store.was_associated_with(join_activity, software)
    join_id = write_join(store, join_activity, software, run_dir)

    report_id, report = write_report(
        store,
        activity,
        software,
        node=NODE,
        kind=KIND,
        conformance=conformance,
        conformance_of=STORE_ASSERTIONS,
        deviations=(CONTRADICTED_CLIP,) if contradictions else (),
        unmeasured=() if margin is not None else ("quality.clip_contradiction_margin",),
        detail={
            "signal": source,
            "preceded_by": preceded,
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
            "notes": notes,
        },
    )
    return BranchResult(report=report, view=(*assertion_ids, join_id, report_id), report_entity_id=report_id)
