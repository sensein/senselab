"""AIRWAY — breath and cough events, in the two modes the shared foundation dispatches between.

``align_airway`` evaluates a declared airway task against what its instruction asked for.
``detect_airway`` finds breath and cough wherever they occur and evaluates no task. Both read
PREPROCESS's own derivatives: the energy envelope segments candidate events and the stored
classifier scores type them. Neither mode reads the ``airway.cough`` routing gate.

The design is in ``specs/20260817-triage-workflow-dag/branch-airway.md``, the grounds for what this
branch does and does not report in ``airway-flag-grounds.md`` beside it.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.classifier_ontology import PROFILE_PATH_KEY, corroboration_sets
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
    AIRWAY_EXPECTATIONS,
    NOT_SEPARABLE_BY_THIS_DESIGN,
    UNDETERMINED,
    BranchParams,
    Done,
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
    stream_ids,
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
"""The measurement name recording that a derivative a mode needs never reached the store."""

COVERAGE_FRACTION = "breath_coverage_fraction"
"""How much of the extent the instruction asked for carries breath evidence. No gate reads it."""

DECLARED_EXTENT = "declared_duration_s"
"""``asked_from`` on a coverage fraction whose denominator is the instruction's own duration."""

STREAM_EXTENT = "stream_extent"
"""``asked_from`` on one whose denominator is the recording, the row declaring no duration."""

CLASSIFIER_WINDOWS = ("span_hear", "span_yamnet")
"""The two per-span window measurements :func:`classifier_windows` collects. PREPROCESS writes one
per (span, classifier) pair."""


HEAR = "hear"
"""The ``classifier`` attribute PREPROCESS stamps on a ``span_hear`` window."""

YAMNET = "yamnet"
"""The ``classifier`` attribute it stamps on a ``span_yamnet`` window."""


class LabelSet(NamedTuple):
    """One kind of sound, in each classifier's own vocabulary.

    Attributes:
        hear: The HeAR head names, which are what ``branch.label_sets`` declares.
        yamnet: The AudioSet display names, the union of those heads' corroboration sets.
    """

    hear: tuple[str, ...]
    yamnet: tuple[str, ...]

    def by_classifier(self) -> dict[str, tuple[str, ...]]:
        """The mapping :func:`~...nodes.branches.sounds_like` reads.

        Returns:
            Classifier name to its spellings.
        """
        return {HEAR: self.hear, YAMNET: self.yamnet}


def label_sets_by_classifier(params: BranchParams) -> dict[str, LabelSet]:
    """``branch.label_sets`` resolved into each classifier's own vocabulary.

    The configured value names HeAR heads; the AudioSet side is the union of those heads'
    corroboration sets from the packaged ontology profile.

    Args:
        params: The operating points, whose config names the profile.

    Returns:
        Kind to its two vocabularies, empty when the operating point is unmeasured.
    """
    configured = params.point("label_sets")
    if configured is None:
        return {}
    corroborating = corroboration_sets(params.config.get(PROFILE_PATH_KEY))
    resolved: dict[str, LabelSet] = {}
    for kind, heads in configured.items():
        audioset = {name for head in heads for name in corroborating.get(head, frozenset())}
        resolved[kind] = LabelSet(tuple(heads), tuple(sorted(audioset)))
    return resolved


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
        One id per name the store holds, in the order given; a name it does not hold contributes
        nothing.
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
        True or False when at least one graded window overlaps, and None when none does or none was
        graded.
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

    A covariate recorded on the proposal, not a filter over it.

    Args:
        store: The provenance store.
        extent: The extent.

    Returns:
        True when at least one live lexical ``word`` entity overlaps.
    """
    return any(word.extent is not None and overlaps(word.extent, extent) for word in lexical_words(store))


def hear_score_windows(store: ProvStore, run_dir: Path) -> list[tuple[tuple[float, float], dict[str, float]]] | None:
    """The raw HeAR windows, on the model's own 2 s grid, from the ``hear_scores`` sidecar.

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
        ``band_profile``'s roll-off, or None when PREPROCESS recorded that block absent.
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
        The rounded value, or None when it is absent or non-finite.
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
    resolved_set = label_sets_by_classifier(params).get(label_set)
    minimum = params.point("score_min")
    if resolved_set is None or minimum is None:
        return []
    wanted = resolved_set.by_classifier()
    events: list[Event] = []
    for span in spans:
        extent = span.extent
        if extent is None or not sounds_like(span, windows, wanted, minimum):
            continue
        resolved = events_in_span(envelope, span, params)
        if resolved:
            events.extend(Event(start, end, span.id, ENVELOPE_BOUNDARIES) for start, end in resolved)
        elif extent[1] > extent[0]:
            events.append(Event(extent[0], extent[1], span.id, CARRIER_BOUNDARIES))
    return sorted(events)


def decided_label_sets(span: Entity, windows: Sequence[Entity], label_sets: Mapping[str, LabelSet]) -> list[str]:
    """Which label sets a stored window over this span decided a label from.

    Reads ``labels``, PREPROCESS's own decision, matching each window against its own classifier's
    spellings.

    Args:
        span: The span.
        windows: The per-span classifier windows.
        label_sets: Kind to its two vocabularies.

    Returns:
        The label-set names, sorted.
    """
    decided: set[str] = set()
    for window in windows:
        if window.attributes.get("span_id") != span.id:
            continue
        classifier = str(window.attributes.get("classifier"))
        for label in window.attributes.get("labels") or []:
            for name, resolved in label_sets.items():
                if str(label) in resolved.by_classifier().get(classifier, ()):
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
    evidence: Sequence[str] = (),
) -> list[Finding]:
    """One acoustic descriptor per event, with the covariates its own extent must be read against.

    The spectral balance is taken only where ``spectrogram_wideband`` reached the store.

    Args:
        events: The events.
        label_set: The kind they are.
        store: The provenance store.
        envelope: The energy envelope.
        block: The wideband spectrogram, or None.
        params: The operating points.
        evidence: The derivative ids every event was read off, beside its carrier span.

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
                event.span_id,
                *evidence,
                index=index,
                boundaries=event.boundaries,
                spectral_balance_db=balance,
                **acquisition_covariates(store, extent),
            )
        )
    return findings


def lexical_intrusions(store: ProvStore) -> list[Finding]:
    """One ``off_task_extent`` deviation per lexical word inside an airway task.

    Reached only from the in-family mode.

    Args:
        store: The provenance store.

    Returns:
        The deviations, in the consensus's own index order. Each derives from its word and carries
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
                word.id,
                reading="lexical_intrusion",
                agreement=word.attributes.get("agreement"),
            )
        )
    return out


def declared_task_ids(store: ProvStore, hint: AudioHints | None) -> list[str]:
    """Every task id the store and the hint carry, best carrier first, trailing index intact.

    The foundation's ``declared_task_family`` collapses the trailing index, so the id is read again
    here rather than recovered from the family.

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


def _events_reading(events: Sequence[Event], params: BranchParams) -> Done:
    """Whether the instruction's own pattern was found, or no answer where the cut is unmeasured.

    Args:
        events: The events the walk reported.
        params: The operating points, read for whether the label cut was measurable.

    Returns:
        True with an event in hand; :data:`UNDETERMINED` where ``branch.score_min`` is unmeasured;
        False otherwise.
    """
    if events:
        return True
    return UNDETERMINED if params.point("score_min") is None else False


def instrument_absent(*names: str) -> Result:
    """A result for a mode one of whose instruments never reached the store.

    Args:
        *names: The absent derivatives, in the order the mode reads them.

    Returns:
        ``done = UNDETERMINED``, no components, and one measurement naming every absence.

    Raises:
        ValueError: If no name is given.
    """
    if not names:
        raise ValueError("instrument_absent names the derivatives that are absent")
    return Result(UNDETERMINED, [], [measured(INSTRUMENT_ABSENT, None, None, None, absent=list(names))])


def coverage_denominator(expectation: Expectation, store: ProvStore) -> tuple[float, str]:
    """How long the extent the instruction asked for is, and where that length came from.

    Args:
        expectation: The row.
        store: The provenance store.

    Returns:
        The length in seconds and :data:`DECLARED_EXTENT` or :data:`STREAM_EXTENT`.
    """
    if expectation.declared_duration_s is not None:
        return float(expectation.declared_duration_s), DECLARED_EXTENT
    return duration(stream_extent(store)), STREAM_EXTENT


def absence_note(names: Sequence[str]) -> str:
    """The report's note for one :data:`INSTRUMENT_ABSENT` measurement.

    Args:
        names: The absent derivatives.

    Returns:
        The note, in the report's controlled vocabulary.
    """
    joined = ", ".join(names)
    return f"the {joined} derivative is absent" if len(names) == 1 else f"the {joined} derivatives are absent"


def absent_instruments(envelope: EnvelopeTrack | None, windows: Sequence[Entity]) -> tuple[str, ...]:
    """Which of the label search's two instruments are absent.

    Args:
        envelope: The energy envelope, or None.
        windows: The per-span classifier windows, empty when neither classifier wrote one.

    Returns:
        The absent derivative names, empty when both instruments are in hand.
    """
    missing: list[str] = []
    if envelope is None:
        missing.append("energy_envelope")
    if not windows:
        missing.extend(CLASSIFIER_WINDOWS)
    return tuple(missing)


def _airway_event_series(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams, run_dir: Path
) -> Result:
    """One span per event, plus ``task_extent`` over their hull.

    The count read against the instruction's ``expected_event_count`` is the number of these spans.

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
    windows = classifier_windows(store)
    if envelope is None or not windows:
        return instrument_absent(*absent_instruments(envelope, windows))
    kind = expectation.label_set
    spans = candidate_spans(store)
    events = airway_events(kind, store, params, spans=amplitude_spans(spans), envelope=envelope, windows=windows)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)
    rate = sampling_rate_of(store, "plain")
    block = None if rate is None else read_spectrogram_block(store, run_dir, "spectrogram_wideband", rate)

    components = event_proposals(events, kind, store=store, evidence=evidence, graded=graded)
    carriers = sorted({event.span_id for event in events})
    findings: list[Finding] = [
        count("expected_event_count", len(events), expectation.expected_event_count, *carriers),
        count(
            "events_with_carrier_boundaries",
            sum(1 for event in events if event.boundaries == CARRIER_BOUNDARIES),
            None,
            *carriers,
        ),
    ]

    onsets = [event.start for event in events]
    intervals = [round(later - earlier, 3) for earlier, later in zip(onsets, onsets[1:])]
    if expectation.timed_intervals:
        findings.append(count("inter_onset_interval_s", intervals, None, *carriers))
        interval_max_s = params.point("interval_max_s")
        if interval_max_s is not None:
            findings.append(
                count(
                    "intervals_over_p_interval_max_s",
                    sum(1 for value in intervals if value > interval_max_s),
                    0,
                    *carriers,
                )
            )
    findings.extend(
        event_measurements(events, kind, store=store, envelope=envelope, block=block, params=params, evidence=evidence)
    )

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
            findings.append(deviation("truncation", task[0], task[1], *carriers, *evidence))
        if expectation.relax_s is not None and duration(whole) >= expectation.relax_s + duration(task):
            findings.append(
                deviation(
                    "off_task_extent", 0.0, expectation.relax_s, *stream_ids(store), reading="declared_relax_period"
                )
            )

    findings.extend(lexical_intrusions(store))
    findings.extend(unviable_findings(expectation))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task_findings(components, spans, params))
    return Result(_events_reading(events, params), components, findings)


def _airway_alternation(expectation: Expectation, store: ProvStore, params: BranchParams, run_dir: Path) -> Result:
    """One span per cough, one per breath, plus ``task_extent``.

    The expected pattern is an alternation, so material between coughs is matched as breath rather
    than scored off task.

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
    windows = classifier_windows(store)
    if envelope is None or not windows:
        return instrument_absent(*absent_instruments(envelope, windows))
    kind = expectation.label_set
    spans = candidate_spans(store)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)

    coughs = airway_events(kind, store, params, spans=spans, envelope=envelope, windows=windows)
    breath_labels = label_sets_by_classifier(params).get(BREATH, LabelSet((), ())).by_classifier()
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
    cough_carriers = sorted({cough.span_id for cough in coughs})
    breath_carriers = sorted({span.id for span in carriers})
    findings: list[Finding] = [
        count("expected_event_count", len(coughs), expectation.expected_event_count, *cough_carriers),
        count(
            "cough_then_breathe_cycles",
            cycles,
            expectation.expected_event_count,
            *cough_carriers,
            *breath_carriers,
        ),
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
    return Result(_events_reading(coughs, params), components, findings)


def _airway_coverage(
    expectation: Expectation, store: ProvStore, hint: AudioHints | None, params: BranchParams, run_dir: Path
) -> Result:
    """One span per merged run of breath-scoring HeAR windows, plus ``task_extent``.

    Reports the covered fraction and answers :data:`UNDETERMINED`; no bound is read against it. See
    ``specs/20260817-triage-workflow-dag/airway-flag-grounds.md``.

    Args:
        expectation: The row.
        store: The provenance store.
        hint: What the recording was declared to contain.
        params: The operating points.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        :data:`UNDETERMINED`, the spans, and the findings.
    """
    assert expectation.label_set is not None
    scored = hear_score_windows(store, run_dir)
    if scored is None:
        return instrument_absent("hear_scores")
    kind = expectation.label_set
    labels = label_sets_by_classifier(params).get(kind, LabelSet((), ())).hear
    minimum = params.point("score_min")
    covered = (
        []
        if minimum is None
        else merge(
            [extent for extent, scores in scored if any(float(scores.get(label, 0.0)) >= minimum for label in labels)]
        )
    )
    total = sum(duration(extent) for extent in covered)
    asked_s, asked_from = coverage_denominator(expectation, store)
    coverage = total / asked_s if asked_s > 0.0 else None
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
        measured(
            COVERAGE_FRACTION,
            None,
            None,
            rounded(coverage, 3),
            *evidence,
            covered_s=rounded(total),
            asked_s=rounded(asked_s),
            asked_from=asked_from,
            runs_n=len(covered),
            windows_n=len(scored),
        )
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
    return Result(UNDETERMINED, components, findings)


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
            defaulted so the positional signature is the foundation's ``AlignMode``; the node
            binds it.

    Returns:
        Whether the expected patterns were found, the spans proposed, and the findings.

    Raises:
        KeyError: If ``task_family`` is not an AIRWAY family.
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

    Task-agnostic, on the same machinery the in-family mode uses. It reads no routing gate and
    emits neither a breath deviation nor a lexical one.

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
    windows = classifier_windows(store)
    if envelope is None or not windows:
        return instrument_absent(*absent_instruments(envelope, windows))
    spans = candidate_spans(store)
    evidence = evidence_ids(store, *EVENT_SOURCES)
    graded = silence_windows(store)
    label_sets = label_sets_by_classifier(params)

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
        findings.append(count(f"{name}_events", len(events), None, *sorted({each.span_id for each in events})))

    for span in spans:
        extent = span.extent
        if extent is None or any(overlaps(extent, each) for each in marked):
            continue
        for name in decided_label_sets(span, windows, label_sets):
            findings.append(contest(span.id, extent, name, "no_raw_score_over_p_score_min"))
    findings.append(
        count(
            "airway_events",
            len(components),
            None,
            *sorted({source for component in components for source in component.derived_from}),
        )
    )
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

    The declaration picks the mode: a declared airway family takes :func:`align_airway`, anything
    else takes :func:`detect_airway`. The branch reports; VERDICT decides.

    Args:
        store: The provenance store, holding PREPROCESS's spans, derivatives and classifications.
        source: The store-held stream the findings are taken over, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory the derivative sidecars are relative to.

    Returns:
        The branch report, the view over what was written, and the ``branch_report`` entity's id.
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
        in_family=mode == "align",
        detail={"mode": mode, "task_family": family, **_detail(result, spans, params)},
    )
    view = [*(span.id for span in spans), *span_ids, *finding_ids, report_id]
    return BranchResult(report=report, view=tuple(view), report_entity_id=report_id)


def _detail(result: Result, spans: Sequence[Entity], params: BranchParams) -> dict[str, Any]:
    """The report's design-named observation fields, read off what the mode returned.

    ``labelled_n`` counts the events this branch proposed, not the spans PREPROCESS merged them out
    of.

    Args:
        result: What the selected mode returned.
        spans: The candidate spans, for the merge rate.
        params: The operating points, read for what was asked for and could not be measured.

    Returns:
        ``labelled_n``, ``by_label``, ``contested_n``, ``merged_n``, ``spans_n``, ``notes`` — what
        this branch could not measure, in controlled vocabulary — and the three coverage fields,
        which are None on every mode that takes no coverage.
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
    notes = [absence_note(finding.evidence.get("absent") or ()) for finding in absent]
    if params.missing:
        notes.append(f"branch.* unmeasured: {', '.join(params.missing)}")
    coverage = next(
        (finding for finding in result.deviations if finding.kind == "measure" and finding.name == COVERAGE_FRACTION),
        None,
    )
    return {
        "labelled_n": len(events),
        "by_label": by_label,
        "contested_n": sum(1 for finding in result.deviations if finding.kind == "contest"),
        "merged_n": sum(int(span.attributes.get("merged_proposals", 1)) for span in spans if span.id in carriers),
        "spans_n": len(result.components),
        "notes": notes,
        COVERAGE_FRACTION: None if coverage is None else coverage.evidence.get("value"),
        "coverage_asked_s": None if coverage is None else coverage.evidence.get("asked_s"),
        "coverage_asked_from": None if coverage is None else coverage.evidence.get("asked_from"),
    }
