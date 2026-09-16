"""AIRWAY — breath and cough events, in the two modes the shared foundation dispatches between.

``align_airway`` evaluates a declared airway task against what its instruction asked for.
``detect_airway`` finds breath and cough wherever they occur and evaluates no task. Both read
PREPROCESS's own derivatives: the energy envelope segments candidate events and the stored
classifier scores type them, which inverts reading a count off window labels.

Neither mode reads the ``airway.cough`` routing gate. The design, the measurements behind that and
what the restructuring cost are in ``specs/20260817-triage-workflow-dag/branch-airway.md``.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
    AIRWAY_EXPECTATIONS,
    NOT_SEPARABLE_BY_THIS_DESIGN,
    UNDETERMINED,
    BranchParams,
    EnvelopeTrack,
    Expectation,
    Finding,
    Pattern,
    Proposal,
    Result,
    SpectrogramBlock,
    acquisition_covariates,
    airway_span,
    amplitude_spans,
    branch_params,
    contest,
    count,
    declared_duration_count,
    deviation,
    deviation_names,
    dispatch,
    duration,
    events_in_span,
    hull,
    measured,
    merge,
    mode_of,
    off_task_findings,
    overlaps,
    peak_over_floor_db,
    propose_spans,
    read_envelope_track,
    read_spectrogram_block,
    sounds_like,
    spectral_balance_db,
    stream_extent,
    touches_edge,
    unviable_findings,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.common import (
    BranchResult,
    find_measurement,
    find_measurements,
    lexical_words,
    live_entities,
    software_agent,
    write_report,
)
from senselab.audio.workflows.triage.routing_analysis.families import task_id_of
from senselab.audio.workflows.triage.vocabulary import TASK
from senselab.utils.prov_store import Entity, ProvStore

NODE = "AIRWAY"
"""The branch's name, in the vocabulary's own spelling."""

KIND = "airway"
"""The kind this branch screens, which is also its span family."""

BREATH = "breath"
"""The breath label set's name in ``branch.label_sets``."""

EVENT_SOURCES = ("energy_envelope", "span_hear", "span_yamnet")
"""The derivatives an event proposal names as its evidence, beside its carrier span."""

ENVELOPE_BOUNDARIES = "envelope_event"
"""``boundaries`` on a span whose start and end the envelope walk resolved."""

CARRIER_BOUNDARIES = "carrier_span"
"""``boundaries`` on a span that took its carrier's extent because the walk resolved none."""

WINDOW_RUN_BOUNDARIES = "hear_window_run"
"""``boundaries`` on a span whose extent is a merged run of scoring classifier windows."""

TASK_EXTENT = "task_extent"
"""The role of the one span covering the part of the recording that serves the task."""

ROUTE_INDEX_KEY = "airway.route_by_task_index"
"""The config key mapping a task's trailing index to the route its instruction prescribes."""

INSTRUMENT_ABSENT = "event_instrument"
"""The measurement name recording that the derivative a mode needs never reached the store."""


class Event(NamedTuple):
    """One breath or cough event, and where its boundaries came from.

    Attributes:
        start: Extent start, in seconds.
        end: Extent end, in seconds.
        span_id: The carrier span it was found inside, which the proposal must name.
        boundaries: :data:`ENVELOPE_BOUNDARIES` or :data:`CARRIER_BOUNDARIES`.
    """

    start: float
    end: float
    span_id: str
    boundaries: str


# --------------------------------------------------------------------- reading the store


def candidate_spans(store: ProvStore) -> list[Entity]:
    """The spans this branch may read: PREPROCESS's own, plus anything already in its own family.

    Args:
        store: The provenance store.

    Returns:
        The live spans carrying no family or this branch's family and an extent, earliest first.
    """
    found = [
        span
        for span in live_entities(store, "span")
        if span.attributes.get("family") in (None, KIND) and span.extent is not None
    ]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


def classifier_windows(store: ProvStore) -> list[Entity]:
    """PREPROCESS's per-span classifier windows, both classifiers together.

    Args:
        store: The provenance store.

    Returns:
        Every ``span_hear`` and ``span_yamnet`` measurement, which is what :func:`sounds_like` reads.
    """
    return [*find_measurements(store, "span_hear"), *find_measurements(store, "span_yamnet")]


def evidence_ids(store: ProvStore, *names: str) -> tuple[str, ...]:
    """The ids of the named measurements that reached the store.

    Args:
        store: The provenance store.
        *names: The measurement names.

    Returns:
        One id per name the store holds, in the order given. A name it does not hold contributes
        nothing rather than a placeholder, so a derivation never names an entity that is not there.
    """
    found: list[str] = []
    for name in names:
        measurement = find_measurement(store, name)
        if measurement is not None:
            found.append(measurement.id)
    return tuple(found)


def stream_entity(store: ProvStore, name: str) -> Entity | None:
    """One live stream entity by name.

    Args:
        store: The provenance store.
        name: The stream's name, e.g. ``"plain"``.

    Returns:
        The newest live stream of that name, or None.
    """
    found = [entity for entity in live_entities(store, "stream") if entity.attributes.get("name") == name]
    return found[-1] if found else None


def sampling_rate_of(store: ProvStore, name: str) -> float | None:
    """A stream's sampling rate, which the spectrogram derivative does not record.

    Args:
        store: The provenance store.
        name: The stream's name.

    Returns:
        The rate in Hz, or None when no such stream carries one.
    """
    entity = stream_entity(store, name)
    rate = None if entity is None else entity.attributes.get("sampling_rate")
    return None if rate is None else float(rate)


def silence_windows(store: ProvStore) -> list[dict[str, Any]] | None:
    """PREPROCESS's silence-graded windows.

    Args:
        store: The provenance store.

    Returns:
        The windows, or None when nothing graded any.
    """
    measurement = find_measurement(store, "silence")
    if measurement is None:
        return None
    windows = measurement.attributes.get("windows")
    return None if windows is None else [dict(window) for window in windows]


def inside_certified_silence(extent: tuple[float, float], windows: list[dict[str, Any]] | None) -> bool | None:
    """Whether every silence-graded window overlapping this extent was certified silent.

    Args:
        extent: The extent being described.
        windows: PREPROCESS's graded windows, or None when it graded none.

    Returns:
        True or False when at least one graded window overlaps, and None when the question has no
        answer here — an unavailable grading is an absence, never a negative.
    """
    if windows is None:
        return None
    overlapping = [
        window for window in windows if float(window["start"]) < extent[1] and float(window["end"]) > extent[0]
    ]
    if not overlapping:
        return None
    return all(bool(window["is_silence"]) for window in overlapping)


def overlaps_transcript(store: ProvStore, extent: tuple[float, float]) -> bool:
    """Whether a lexical consensus word overlaps this extent.

    A covariate, not a filter: a cough inside a sentence reading is a cough, and a bracketed
    word — ``[COUGH]`` — is what this branch looks for rather than a transcript.

    Args:
        store: The provenance store.
        extent: The extent.

    Returns:
        True when at least one live lexical ``word`` entity overlaps.
    """
    return any(word.extent is not None and overlaps(word.extent, extent) for word in lexical_words(store))


def hear_score_windows(store: ProvStore, run_dir: Path) -> list[tuple[tuple[float, float], dict[str, float]]] | None:
    """The raw HeAR windows, on the model's own 2 s grid, from the ``hear_scores`` sidecar.

    Not ``hear_windows``: that derivative reads ``windows.hear.label_thresholds``, which the
    packaged config leaves null, so it is written on no run.

    Args:
        store: The provenance store.
        run_dir: The run directory the sidecar path is relative to.

    Returns:
        ``[(extent, {label: score}), ...]``, or None when the measurement, its path or the file is
        absent.
    """
    measurement = find_measurement(store, "hear_scores")
    if measurement is None:
        return None
    relative = measurement.attributes.get("path")
    if not relative:
        return None
    sidecar = run_dir / str(relative)
    if not sidecar.is_file():
        return None
    out: list[tuple[tuple[float, float], dict[str, float]]] = []
    for window in json.loads(sidecar.read_text()):
        scores = {label: score for pair in label_scores(window) for label, score in pair.items()}
        out.append(((float(window["start"]), float(window["end"])), scores))
    return out


def content_band_hz(store: ProvStore) -> float | None:
    """The upper edge of the band the recording actually carries content in.

    Args:
        store: The provenance store.

    Returns:
        ``band_profile``'s roll-off, or None — which is every run today, the derivative not
        existing. Carried as absent rather than omitted so the route negative is attributable.
    """
    measurement = find_measurement(store, "band_profile")
    rolloff = None if measurement is None else measurement.attributes.get("rolloff_hz")
    return None if rolloff is None else float(rolloff)


def rounded(value: float | None, digits: int = 2) -> float | None:
    """A measurement rounded for storage, or None when it is not finite.

    Args:
        value: The value, or None.
        digits: Decimal places.

    Returns:
        The rounded value, or None. A non-finite reading is an absent measurement, and None is how
        every other absence in this branch is written.
    """
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits)


# --------------------------------------------------------------------- the event source


def airway_events(
    label_set: str,
    store: ProvStore,
    params: BranchParams,
    *,
    spans: Sequence[Entity],
    envelope: EnvelopeTrack,
    windows: Sequence[Entity],
) -> list[Event]:
    """Every event of one kind: segmented from the envelope, typed by the stored scores.

    A carrier that scored the label and inside which the envelope walk resolves no boundary
    contributes one event at the carrier's own extent, marked :data:`CARRIER_BOUNDARIES`.

    Args:
        label_set: Which entry of ``branch.label_sets`` names this kind.
        store: The provenance store.
        params: The operating points.
        spans: The carrier spans to look inside.
        envelope: The energy envelope.
        windows: The per-span classifier windows.

    Returns:
        The events, earliest first.
    """
    labels = (params.point("label_sets") or {}).get(label_set)
    minimum = params.point("score_min")
    if labels is None or minimum is None:
        return []
    events: list[Event] = []
    for span in spans:
        extent = span.extent
        if extent is None or not sounds_like(span, windows, labels, minimum):
            continue
        resolved = events_in_span(envelope, span, params)
        if resolved:
            events.extend(Event(start, end, span.id, ENVELOPE_BOUNDARIES) for start, end in resolved)
        elif extent[1] > extent[0]:
            events.append(Event(extent[0], extent[1], span.id, CARRIER_BOUNDARIES))
    return sorted(events)


def decided_label_sets(span: Entity, windows: Sequence[Entity], label_sets: dict[str, tuple[str, ...]]) -> list[str]:
    """Which label sets a stored window over this span decided a label from.

    The decision is ``labels``, which PREPROCESS writes only where a membership rule exists; the
    measurement is ``raw_scores``, which it always writes. A span carrying the decision and no raw
    score over the minimum is what :func:`detect_airway` contests.

    Args:
        span: The span.
        windows: The per-span classifier windows.
        label_sets: Label-set name to the classifier labels that are that sound.

    Returns:
        The label-set names, sorted.
    """
    decided: set[str] = set()
    for window in windows:
        if window.attributes.get("span_id") != span.id:
            continue
        for label in window.attributes.get("labels") or []:
            for name, members in label_sets.items():
                if str(label) in members:
                    decided.add(name)
    return sorted(decided)


def event_proposals(
    events: Sequence[Event],
    label_set: str,
    *,
    store: ProvStore,
    evidence: Sequence[str],
    graded: list[dict[str, Any]] | None,
    extra: dict[str, Any] | None = None,
) -> list[Proposal]:
    """One proposed span per event, each naming its carrier and the derivatives behind it.

    Args:
        events: The events.
        label_set: The kind they are, which becomes each span's ``label``.
        store: The provenance store.
        evidence: The derivative ids every event proposal names.
        graded: PREPROCESS's silence-graded windows, for the covariate.
        extra: Attributes added to every proposal.

    Returns:
        The proposals, in the order the events were given.
    """
    out: list[Proposal] = []
    for index, event in enumerate(events):
        extent = (event.start, event.end)
        out.append(
            airway_span(
                f"{label_set}_event",
                extent,
                event.span_id,
                *evidence,
                label=label_set,
                index=index,
                boundaries=event.boundaries,
                in_certified_silence=inside_certified_silence(extent, graded),
                overlaps_transcript=overlaps_transcript(store, extent),
                **(extra or {}),
            )
        )
    return out


def event_measurements(
    events: Sequence[Event],
    label_set: str,
    *,
    store: ProvStore,
    envelope: EnvelopeTrack,
    block: SpectrogramBlock | None,
    params: BranchParams,
) -> list[Finding]:
    """One acoustic descriptor per event, with the covariates its own extent must be read against.

    The spectral balance is taken only where ``spectrogram_wideband`` reached the store, so
    ``branch.effort_split_hz`` is read only when the instrument it configures exists.

    Args:
        events: The events.
        label_set: The kind they are.
        store: The provenance store.
        envelope: The energy envelope.
        block: The wideband spectrogram, or None.
        params: The operating points.

    Returns:
        One ``<kind>_peak_over_floor_db`` measurement per event.
    """
    findings: list[Finding] = []
    for index, event in enumerate(events):
        extent = (event.start, event.end)
        split_hz = params.point("effort_split_hz")
        balance = None if block is None or split_hz is None else rounded(spectral_balance_db(block, extent, split_hz))
        findings.append(
            measured(
                f"{label_set}_peak_over_floor_db",
                event.start,
                event.end,
                rounded(peak_over_floor_db(envelope, extent)),
                index=index,
                boundaries=event.boundaries,
                spectral_balance_db=balance,
                **acquisition_covariates(store, extent),
            )
        )
    return findings


def lexical_intrusions(store: ProvStore) -> list[Finding]:
    """One ``off_task_extent`` deviation per lexical word inside an airway task.

    AIRWAY owns this deviation, and it is the one correct ``off_task_extent``: it keys on
    positively-identified off-task content that has its own extent, not on the absence of the
    target. Reached only from the in-family mode, so a cough inside a sentence reading carries no
    penalty for the words around it.

    Args:
        store: The provenance store.

    Returns:
        The deviations, in the consensus's own index order. Each names its word by id and carries
        the agreement behind it; the word's text never enters the store here.
    """
    out: list[Finding] = []
    for word in lexical_words(store):
        if word.extent is None:
            continue
        out.append(
            deviation(
                "off_task_extent",
                word.extent[0],
                word.extent[1],
                reading="lexical_intrusion",
                word_id=word.id,
                agreement=word.attributes.get("agreement"),
            )
        )
    return out


def declared_task_ids(store: ProvStore, hint: AudioHints | None) -> list[str]:
    """Every task id the store and the hint carry, best carrier first, trailing index intact.

    The foundation's ``declared_task_family`` collapses the trailing index, which is what carries
    the ``fivebreaths`` route, so the id is read again here rather than recovered from the family.

    Args:
        store: The provenance store.
        hint: The recording's hints, or None.

    Returns:
        The candidate task ids, lowercased.
    """
    found: list[str] = []
    token = (hint.metadata or {}).get("task_token") if hint is not None else None
    if isinstance(token, str) and token.strip():
        found.append(token.strip().lower())
    recording = stream_entity(store, "recording")
    path = None if recording is None else recording.attributes.get("path")
    if path:
        found.append(task_id_of(Path(str(path)).stem))
    return found


def declared_route(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams
) -> str | None:
    """The route the instruction prescribes, from the row or from the task's trailing index.

    Args:
        expectation: The row.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points, whose config carries the index map.

    Returns:
        The declared route, or None when neither carrier names one.
    """
    if not expectation.route_from_index:
        return expectation.declared_route
    routes = {str(index): str(route) for index, route in (params.config.require(ROUTE_INDEX_KEY) or {}).items()}
    for task_id in declared_task_ids(store, hint):
        segment = task_id.rsplit("-", 1)[-1]
        if segment in routes:
            return routes[segment]
    return None


def route_findings(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams
) -> tuple[str | None, list[Finding]]:
    """What the instruction declared about the route, beside the measurement nothing here can take.

    Args:
        expectation: The row.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.

    Returns:
        The declared route and the findings — nothing at all for a row that declares no route.
    """
    route = declared_route(expectation, store, hint, params)
    if route is None and not expectation.route_from_index:
        return None, []
    return route, [
        count("declared_route", None, route),
        measured("measured_route", None, None, NOT_SEPARABLE_BY_THIS_DESIGN, content_band_hz=content_band_hz(store)),
    ]


# --------------------------------------------------------------------- the three in-family patterns


def instrument_absent(name: str) -> Result:
    """A result for a mode whose only instrument never reached the store.

    Args:
        name: The absent derivative.

    Returns:
        ``done = UNDETERMINED``, no components, and one measurement recording the absence. An
        unavailable measurement is an absence, never a negative.
    """
    return Result(UNDETERMINED, [], [measured(INSTRUMENT_ABSENT, None, None, None, absent=name)])


def _airway_event_series(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams, run_dir: Path
) -> Result:
    """One span per event, plus ``task_extent`` over their hull.

    One per event is the point: the branch used to increment once per (span, label) pair, so a 4 s
    span holding three coughs counted 1. The count compared against the instruction's
    ``expected_event_count`` is the number of these spans.

    Args:
        expectation: The row.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        Whether any event was found, the spans, and the findings.
    """
    assert expectation.label_set is not None
    envelope = read_envelope_track(store, run_dir)
    if envelope is None:
        return instrument_absent("energy_envelope")
    kind = expectation.label_set
    spans = candidate_spans(store)
    windows = classifier_windows(store)
    events = airway_events(kind, store, params, spans=amplitude_spans(spans), envelope=envelope, windows=windows)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)
    rate = sampling_rate_of(store, "plain")
    block = None if rate is None else read_spectrogram_block(store, run_dir, "spectrogram_wideband", rate)

    components = event_proposals(events, kind, store=store, evidence=evidence, graded=graded)
    findings: list[Finding] = [
        count("expected_event_count", len(events), expectation.expected_event_count),
        count(
            "events_with_carrier_boundaries",
            sum(1 for event in events if event.boundaries == CARRIER_BOUNDARIES),
            None,
        ),
    ]

    onsets = [event.start for event in events]
    intervals = [round(later - earlier, 3) for earlier, later in zip(onsets, onsets[1:])]
    if expectation.timed_intervals:
        findings.append(count("inter_onset_interval_s", intervals, None))
        interval_max_s = params.point("interval_max_s")
        if interval_max_s is not None:
            findings.append(
                count(
                    "intervals_over_p_interval_max_s",
                    sum(1 for value in intervals if value > interval_max_s),
                    0,
                )
            )
    findings.extend(event_measurements(events, kind, store=store, envelope=envelope, block=block, params=params))

    route, route_reported = route_findings(expectation, store, hint, params)
    findings.extend(route_reported)

    task = hull([(event.start, event.end) for event in events])
    whole = stream_extent(store)
    if task is not None:
        components.append(
            airway_span(
                TASK_EXTENT,
                task,
                *sorted({event.span_id for event in events}),
                *evidence,
                label=kind,
                events_n=len(events),
                declared_event_count=expectation.expected_event_count,
                declared_route=route,
            )
        )
        if whole is not None and touches_edge(task, whole):
            findings.append(deviation("truncation", task[0], task[1]))
        if expectation.relax_s is not None and duration(whole) >= expectation.relax_s + duration(task):
            findings.append(deviation("off_task_extent", 0.0, expectation.relax_s, reading="declared_relax_period"))

    findings.extend(lexical_intrusions(store))
    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task_findings(components, spans, params))
    return Result(len(events) > 0, components, findings)


def _airway_alternation(expectation: Expectation, store: ProvStore, params: BranchParams, run_dir: Path) -> Result:
    """One span per cough, one per breath, plus ``task_extent``.

    The expected pattern is an alternation, so material between coughs is matched as breath and
    never scored off task: a cough detector alone is insufficient here.

    Args:
        expectation: The row.
        store: The provenance store.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        Whether any cough was found, the spans, and the findings.
    """
    assert expectation.label_set is not None
    envelope = read_envelope_track(store, run_dir)
    if envelope is None:
        return instrument_absent("energy_envelope")
    kind = expectation.label_set
    spans = candidate_spans(store)
    windows = classifier_windows(store)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)

    coughs = airway_events(kind, store, params, spans=spans, envelope=envelope, windows=windows)
    breath_labels = (params.point("label_sets") or {}).get(BREATH) or ()
    breath_minimum = params.point("score_min")
    carriers = (
        []
        if breath_minimum is None
        else [span for span in spans if sounds_like(span, windows, breath_labels, breath_minimum)]
    )
    breaths = merge([span.extent for span in carriers if span.extent is not None])

    components = event_proposals(coughs, kind, store=store, evidence=evidence, graded=graded)
    for index, extent in enumerate(breaths):
        covered = sorted({span.id for span in carriers if span.extent is not None and overlaps(span.extent, extent)})
        components.append(
            airway_span(
                f"{BREATH}_event",
                extent,
                *covered,
                *evidence,
                label=BREATH,
                index=index,
                boundaries=CARRIER_BOUNDARIES,
                in_certified_silence=inside_certified_silence(extent, graded),
                overlaps_transcript=overlaps_transcript(store, extent),
            )
        )

    cycles = sum(1 for cough in coughs if any(breath[0] >= cough.end for breath in breaths))
    findings: list[Finding] = [
        count("expected_event_count", len(coughs), expectation.expected_event_count),
        count("cough_then_breathe_cycles", cycles, expectation.expected_event_count),
    ]
    task = hull([(component.start, component.end) for component in components])
    if task is not None:
        components.append(
            airway_span(
                TASK_EXTENT,
                task,
                *sorted({source for component in components for source in component.derived_from}),
                label=kind,
                coughs_n=len(coughs),
                breaths_n=len(breaths),
            )
        )
    findings.extend(lexical_intrusions(store))
    findings.extend(unviable_findings(expectation))
    findings.extend(off_task_findings(components, spans, params))
    return Result(len(coughs) > 0, components, findings)


def _airway_coverage(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams, run_dir: Path
) -> Result:
    """One span per merged run of breath-scoring HeAR windows, plus ``task_extent``.

    ``residual.energy_fraction`` is not read here. The residual is ``plain`` minus the enhanced
    stream and FRCRN is a speech enhancer, so a high fraction means *this is not speech* — which a
    cough, a glide, room noise and a near-silent file all satisfy. As a routing gate that may be
    adequate; as this row's presence measurement it is not.

    Args:
        expectation: The row.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        Whether the covered fraction reached the minimum, the spans, and the findings.
    """
    assert expectation.label_set is not None
    scored = hear_score_windows(store, run_dir)
    if scored is None:
        return instrument_absent("hear_scores")
    kind = expectation.label_set
    labels = (params.point("label_sets") or {}).get(kind) or ()
    minimum = params.point("score_min")
    covered = (
        []
        if minimum is None
        else merge(
            [extent for extent, scores in scored if any(float(scores.get(label, 0.0)) >= minimum for label in labels)]
        )
    )
    total = sum(duration(extent) for extent in covered)
    whole = duration(stream_extent(store))
    coverage = total / whole if whole > 0.0 else 0.0
    evidence = evidence_ids(store, "hear_scores")
    graded = silence_windows(store)

    components: list[Proposal] = [
        airway_span(
            f"{kind}_run",
            extent,
            *evidence,
            label=kind,
            index=index,
            boundaries=WINDOW_RUN_BOUNDARIES,
            in_certified_silence=inside_certified_silence(extent, graded),
            overlaps_transcript=overlaps_transcript(store, extent),
        )
        for index, extent in enumerate(covered)
    ]
    findings: list[Finding] = [
        measured("breath_coverage_fraction", None, None, rounded(coverage, 3), covered_s=rounded(total))
    ]
    route, route_reported = route_findings(expectation, store, hint, params)
    findings.extend(route_reported)
    task = hull(covered)
    if task is not None:
        components.append(
            airway_span(
                TASK_EXTENT,
                task,
                *evidence,
                label=kind,
                runs_n=len(covered),
                covered_s=rounded(total),
                declared_route=route,
            )
        )
    findings.extend(lexical_intrusions(store))
    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task_findings(components, candidate_spans(store), params))
    coverage_min = params.point("breath_coverage_min")
    return Result(UNDETERMINED if coverage_min is None else coverage >= coverage_min, components, findings)


# --------------------------------------------------------------------- the two modes


def align_airway(
    task_family: str,
    store: ProvStore,
    hint: AudioHints | None,
    params: BranchParams,
    *,
    run_dir: Path | None = None,
) -> Result:
    """Evaluate a declared airway task against what its instruction asked for.

    Args:
        task_family: The declared family, which is a key of ``AIRWAY_EXPECTATIONS``.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory the persisted derivatives are relative to. Keyword-only and
            defaulted so the positional signature is the foundation's ``AlignMode``; the node binds
            it, the contract carrying no other way to reach a sidecar.

    Returns:
        Whether the expected patterns were found, the spans proposed, and the findings.

    Raises:
        KeyError: If ``task_family`` is not an AIRWAY family, which the caller owes ``detect``.
        ValueError: If ``run_dir`` is None, or if the row's pattern has no matcher here.
    """
    if run_dir is None:
        raise ValueError("align_airway reads persisted derivatives and needs the run directory")
    expectation = AIRWAY_EXPECTATIONS[task_family]
    if expectation.pattern is Pattern.EVENT_SERIES:
        return _airway_event_series(expectation, store, hint, params, run_dir)
    if expectation.pattern is Pattern.EVENT_ALTERNATION:
        return _airway_alternation(expectation, store, params, run_dir)
    if expectation.pattern is Pattern.SOUND_COVERAGE:
        return _airway_coverage(expectation, store, hint, params, run_dir)
    raise ValueError(f"{task_family} carries {expectation.pattern}, which align_airway has no matcher for")


def detect_airway(store: ProvStore, params: BranchParams, *, run_dir: Path | None = None) -> Result:
    """Find breath and cough wherever they occur, and evaluate no task.

    Task-agnostic, on the same machinery the in-family mode uses. It reads no routing gate: the
    shipped ``airway.cough`` cut was selected under a scoped reference and reads J -0.1398 against
    ``declared_airway`` over 61,721 recordings, so over an arbitrary recording it is a loudness
    detector. A breath during passage reading is not a deviation — it is how SPEECH measures breath
    groups — and no lexical word is off task in someone else's task, so neither is emitted here.

    Args:
        store: The provenance store.
        params: The operating points.
        run_dir: The run directory the persisted derivatives are relative to. Keyword-only, as on
            :func:`align_airway`.

    Returns:
        A result whose ``done`` is :data:`UNDETERMINED`, the spans proposed, and the findings.

    Raises:
        ValueError: If ``run_dir`` is None.
    """
    if run_dir is None:
        raise ValueError("detect_airway reads persisted derivatives and needs the run directory")
    envelope = read_envelope_track(store, run_dir)
    if envelope is None:
        return instrument_absent("energy_envelope")
    spans = candidate_spans(store)
    windows = classifier_windows(store)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)
    label_sets = params.point("label_sets") or {}

    components: list[Proposal] = []
    findings: list[Finding] = []
    marked: list[tuple[float, float]] = []
    for name in sorted(label_sets):
        events = airway_events(name, store, params, spans=spans, envelope=envelope, windows=windows)
        components.extend(
            event_proposals(
                events, name, store=store, evidence=evidence, graded=graded, extra={"evaluates_no_task": True}
            )
        )
        marked.extend((event.start, event.end) for event in events)
        findings.append(count(f"{name}_events", len(events), None))

    for span in spans:
        extent = span.extent
        if extent is None or any(overlaps(extent, each) for each in marked):
            continue
        for name in decided_label_sets(span, windows, label_sets):
            findings.append(contest(span.id, extent, name, "no_raw_score_over_p_score_min"))
    findings.append(count("airway_events", len(components), None))
    return Result(UNDETERMINED, components, findings)


# --------------------------------------------------------------------- the node


def airway(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> BranchResult:
    """Propose this branch's breath and cough spans, write its findings, and report.

    The declaration picks the mode and never supplies the answer: a declared airway family takes
    :func:`align_airway`; anything else — another branch's kind, an unreadable stem, no declaration
    at all — takes :func:`detect_airway`.

    There is no FLAG path. ``lexical_contamination`` was the only flag this branch could raise and
    it is now an ``off_task_extent`` deviation, located and conditioned on the declaration, so the
    outcome is PASS or FAIL.

    Args:
        store: The provenance store, holding PREPROCESS's spans, derivatives and classifications.
        source: The store-held stream the findings are taken over, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory the derivative sidecars are relative to.

    Returns:
        The verdict, the view over what was written, and the verdict entity's id.
    """
    params = branch_params(config)
    software = software_agent(store)
    mode, family = mode_of(NODE, store, hint)
    activity = store.activity(node=NODE, step=mode, parameters={"task_family": family, "source": source})
    store.was_associated_with(activity, software)
    stream = stream_entity(store, source)
    if stream is not None:
        store.used(activity, stream.id)
    spans = candidate_spans(store)
    for span in spans:
        store.used(activity, span.id)
    for entity_id in evidence_ids(store, *EVENT_SOURCES, "hear_scores", "silence", "spectrogram_wideband"):
        store.used(activity, entity_id)

    result = dispatch(
        NODE,
        store,
        params,
        hint,
        align=partial(align_airway, run_dir=run_dir),
        detect=partial(detect_airway, run_dir=run_dir),
    )
    findings = [*result.deviations, *params.record()]
    span_ids = propose_spans(store, activity, software, result.components)
    finding_ids = write_findings(store, activity, software, findings, signal=source)

    report_id, report = write_report(
        store,
        activity,
        software,
        node=NODE,
        kind=KIND,
        conformance=result.done,
        conformance_of=TASK,
        deviations=deviation_names(findings),
        unmeasured=tuple(params.missing),
        detail={"mode": mode, "task_family": family, **_detail(result, spans, params)},
    )
    view = [*(span.id for span in spans), *span_ids, *finding_ids, report_id]
    return BranchResult(report=report, view=tuple(view), report_entity_id=report_id)


def _detail(result: Result, spans: Sequence[Entity], params: BranchParams) -> dict[str, Any]:
    """The report's design-named observation fields, read off what the mode returned.

    ``labelled_n`` counts the events this branch proposed rather than the spans PREPROCESS merged
    them out of, which is the repair: a 4 s span holding three coughs used to count one.

    Args:
        result: What the selected mode returned.
        spans: The candidate spans, for the merge rate.
        params: The operating points, read for what was asked for and could not be measured.

    Returns:
        ``labelled_n``, ``by_label``, ``contested_n``, ``merged_n``, ``spans_n`` and ``notes``.
        ``notes`` is what this branch could not measure, in controlled vocabulary; nothing folds it.
    """
    events = [proposal for proposal in result.components if proposal.role != TASK_EXTENT]
    by_label: dict[str, int] = {}
    for proposal in events:
        label = str(proposal.attributes.get("label") or KIND)
        by_label[label] = by_label.get(label, 0) + 1
    carriers = {source for proposal in events for source in proposal.derived_from}
    absent = [
        finding for finding in result.deviations if finding.kind == "measure" and finding.name == INSTRUMENT_ABSENT
    ]
    notes = [f"the {finding.evidence.get('absent')} derivative is absent" for finding in absent]
    if params.missing:
        notes.append(f"branch.* unmeasured: {', '.join(params.missing)}")
    return {
        "labelled_n": len(events),
        "by_label": by_label,
        "contested_n": sum(1 for finding in result.deviations if finding.kind == "contest"),
        "merged_n": sum(int(span.attributes.get("merged_proposals", 1)) for span in spans if span.id in carriers),
        "spans_n": len(result.components),
        "notes": notes,
    }
