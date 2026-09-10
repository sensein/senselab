"""Per-recording routing evidence, stream-parsed out of one triage run's provenance stores.

The store is read one line at a time and never held whole. Only the entity records a routing
detector could read are decoded; the label summaries are reduced to the peaks in
:data:`~senselab.audio.workflows.triage.routing_analysis.labels.TRACKED_LABELS` and everything else
in them is dropped.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from senselab.audio.workflows.triage.routing_analysis.labels import (
    CLASSIFIERS,
    TRACKED_LABELS,
    peak_key,
)

SPAN_MEASURES: tuple[str, ...] = ("amplitude", "continuity", "gap")
"""The span measures PREPROCESS proposes, each counted and timed separately."""

SQUIM_METRICS: tuple[str, ...] = ("pesq", "si_sdr", "stoi")
"""The three objective heads a ``squim`` assertion carries."""

SQUIM_POPULATIONS: tuple[str, ...] = ("all", "amplitude", "gap")
"""Which spans a SQUIM summary is taken over: every span, or one measure's spans."""

LEVEL_KEYS: tuple[str, ...] = ("peak_dbfs", "rms_dbfs", "lufs")
"""The whole-file ``level`` measurement's scalars."""

DISRUPTION_KEYS: tuple[str, ...] = (
    "clipped_runs",
    "clipped_s",
    "dc_offset",
    "discontinuities",
    "dropout_runs",
    "dropout_s",
    "zero_crossing_rate",
)
"""The whole-file ``disruptions_file`` measurement's scalars, over the ORIGINAL recording."""

RESIDUAL_BANDS: tuple[str, ...] = ("0_200", "200_1000", "1000_4000", "4000_8000")
"""The residual measurement's band energy fractions, folded in as ``band_<edges>``."""

WORD_OUTCOMES: tuple[str, ...] = ("agreement", "variant", "insertion")
"""The consensus outcomes a ``word`` entity can carry."""

_ENTITY_PREFIX = '{"attributes"'
_RELATION_MARKER = '"relation"'
_INVALIDATED = "wasInvalidatedBy"
_SUMMARY_SUFFIX = "_label_summary"
_SUMMARY_ALL_SUFFIX = "_summary_all"
_SKIP_NAMES = ('"name": "span_yamnet"',)

TRANSCRIPT_CAP = 300
"""How much of the consensus transcript is kept, so a disagreement can be read, not just counted."""


@dataclass
class RecordingFeatures:
    """Everything a candidate routing detector could read about one recording.

    Attributes:
        stem: The recording's BIDS stem.
        run_root: The run directory the summary named.
        task_id: The sanitized task id parsed from the stem.
        family: The task family the id collapses into.
        duration_s: The recording's extent, from the ``recording`` stream entity.
        words: How many live consensus words carry each outcome, plus ``total``, ``bracketed``
            and ``lexical`` (non-bracketed, any outcome).
        consensus_present: Whether a ``consensus_transcript`` measurement was written at all.
        transcript: The consensus transcript's first :data:`TRANSCRIPT_CAP` characters.
        residual: The ``residual`` measurement's scalar attributes, or empty when it is absent.
        span_count: Live spans per measure.
        span_longest_s: The longest live span per measure, in seconds.
        span_total_s: Total live span seconds per measure.
        span_stats: ``{"<measure>.<statistic>": value}`` over the live spans of each measure —
            the duration distribution, the ``peak_over_floor_db`` distribution, how many carry a
            ``corroborated_by`` entry, and the duty fraction against the recording's extent.
        squim: ``{"<population>.<metric>.<statistic>": value}`` over the per-span SQUIM
            assertions, plus ``<population>.n`` and ``<population>.unmeasured``.
        level: The whole-file ``level`` measurement's scalars.
        disruptions: The whole-file ``disruptions_file`` measurement's scalars.
        silence: ``threshold``, ``n_windows``, ``n_silence`` and ``fraction`` from the YAMNet
            ``Silence`` projection.
        peaks: ``{peak_key: score}`` for every tracked label on every stream and classifier.
        classifier_streams: Which ``<stream>|<classifier>`` summaries were present at all.
        kind_state: TAXONOMY's own state per kind, so its current behaviour can be measured too.
        verdicts: Each node's recorded outcome.
        n_entities: How many entity records the store held, as a parse sanity check.
    """

    stem: str
    run_root: str
    task_id: str
    family: str
    duration_s: float | None = None
    words: dict[str, int] = field(default_factory=dict)
    consensus_present: bool = False
    transcript: str = ""
    residual: dict[str, float] = field(default_factory=dict)
    span_count: dict[str, int] = field(default_factory=dict)
    span_longest_s: dict[str, float] = field(default_factory=dict)
    span_total_s: dict[str, float] = field(default_factory=dict)
    span_stats: dict[str, float] = field(default_factory=dict)
    squim: dict[str, float] = field(default_factory=dict)
    level: dict[str, float] = field(default_factory=dict)
    disruptions: dict[str, float] = field(default_factory=dict)
    silence: dict[str, float] = field(default_factory=dict)
    peaks: dict[str, float] = field(default_factory=dict)
    classifier_streams: list[str] = field(default_factory=list)
    kind_state: dict[str, str] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)
    n_entities: int = 0

    def as_json(self) -> dict[str, Any]:
        """This record as a plain mapping, for one line of the features shard.

        Returns:
            The dataclass fields, unchanged.
        """
        return asdict(self)


def read_store(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the store's decoded entity records, and its invalidation relations, one at a time.

    Args:
        path: The ``run/store.jsonl`` to read.

    Yields:
        Each decoded record. Activity, agent, environment and non-invalidation relation records are
        skipped without being decoded, and so are the per-span classifier score dumps, which no
        routing detector reads and which dominate the file's bytes.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(_ENTITY_PREFIX):
                if any(marker in line for marker in _SKIP_NAMES):
                    yield {"record": "entity", "prov_type": "measurement", "attributes": {"name": "span_skipped"}}
                    continue
                yield json.loads(line)
            elif _RELATION_MARKER in line and _INVALIDATED in line:
                yield json.loads(line)


def _peak_of(entry: Any) -> float | None:  # noqa: ANN401
    """One label's peak from a summary entry, whichever key that summary spells it with.

    Args:
        entry: The summary's per-label mapping.

    Returns:
        ``peak`` or ``max_score``, or None when the entry carries neither.
    """
    if not isinstance(entry, dict):
        return None
    for key in ("peak", "max_score"):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return None


def _summary_stream_classifier(name: str) -> tuple[str, str] | None:
    """The stream and classifier a whole-file label summary belongs to.

    Args:
        name: The measurement's ``name`` attribute.

    Returns:
        ``(stream, classifier)`` for ``<c>_label_summary`` (plain) and
        ``<stream>_<c>_summary_all``, else None.
    """
    for classifier in CLASSIFIERS:
        if name == f"{classifier}{_SUMMARY_SUFFIX}":
            return "plain", classifier
        for stream in ("enhanced", "residual"):
            if name == f"{stream}_{classifier}{_SUMMARY_ALL_SUFFIX}":
                return stream, classifier
    return None


def _absorb_measurement(features: RecordingFeatures, attributes: dict[str, Any]) -> None:
    """Fold one ``measurement`` entity into the record.

    Args:
        features: The record being built.
        attributes: The measurement's attributes.
    """
    name = str(attributes.get("name") or "")
    if name == "consensus_transcript":
        features.consensus_present = True
        features.transcript = str(attributes.get("text") or "")[:TRANSCRIPT_CAP]
        return
    if name == "residual":
        features.residual = {
            key: float(value)
            for key, value in attributes.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        features.residual["speech_present"] = float(bool(attributes.get("speech_present")))
        bands = attributes.get("bands") or {}
        for band in RESIDUAL_BANDS:
            if band in bands:
                features.residual[f"band_{band}"] = float(bands[band])
        return
    if name == "level":
        features.level = {key: float(attributes[key]) for key in LEVEL_KEYS if key in attributes}
        return
    if name == "disruptions_file":
        features.disruptions = {key: float(attributes[key]) for key in DISRUPTION_KEYS if key in attributes}
        return
    if name == "silence":
        windows = attributes.get("windows") or []
        silent = sum(1 for window in windows if window.get("is_silence"))
        features.silence = {
            "threshold": float(attributes.get("threshold") or 0.0),
            "n_windows": float(len(windows)),
            "n_silence": float(silent),
            "fraction": silent / len(windows) if windows else 0.0,
            "score_mean": statistics.fmean(float(window.get("score") or 0.0) for window in windows) if windows else 0.0,
        }
        return
    if name == "span_hear":
        tracked_hear = TRACKED_LABELS["hear"]
        for label, score in (attributes.get("raw_scores") or {}).items():
            if label in tracked_hear:
                key = peak_key("span", "hear", str(label))
                features.peaks[key] = max(features.peaks.get(key, 0.0), float(score))
        return
    if name == "consensus_taxonomy":
        for row in attributes.get("labels") or []:
            label = str(row.get("label"))
            for classifier, peak in (row.get("peak_by_classifier") or {}).items():
                if label in TRACKED_LABELS.get(str(classifier), frozenset()):
                    features.peaks[peak_key("consensus", str(classifier), label)] = float(peak)
        return
    pair = _summary_stream_classifier(name)
    if pair is None:
        return
    stream, classifier = pair
    features.classifier_streams.append(f"{stream}|{classifier}")
    tracked = TRACKED_LABELS[classifier]
    for label, entry in (attributes.get("labels") or {}).items():
        if label in tracked:
            peak = _peak_of(entry)
            if peak is not None:
                features.peaks[peak_key(stream, classifier, str(label))] = peak


def _stats(values: Sequence[float]) -> dict[str, float]:
    """The distribution summary every span- and SQUIM-derived quantity is reduced to.

    Args:
        values: The sample.

    Returns:
        ``n``, ``min``, ``median``, ``mean``, ``max`` and ``iqr``; empty when the sample is.
    """
    if not values:
        return {}
    ordered = sorted(values)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        spread = quartiles[2] - quartiles[0]
    else:
        spread = ordered[-1] - ordered[0]
    return {
        "n": float(len(ordered)),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
        "iqr": spread,
    }


def _extent_key(extent: Sequence[float]) -> tuple[float, float]:
    """The rounded extent a SQUIM assertion is joined to its span by.

    Args:
        extent: The entity's ``[start, end]``.

    Returns:
        Both bounds rounded to the microsecond, which is finer than any span PREPROCESS proposes.
    """
    return (round(float(extent[0]), 6), round(float(extent[1]), 6))


def _span_statistics(spans: Sequence[dict[str, Any]], duration_s: float | None) -> dict[str, float]:
    """Reduce one recording's live spans to the numbers a routing detector could read.

    Args:
        spans: The live spans, each carrying ``measure``, ``duration``, ``peak_db`` and
            ``corroborated``.
        duration_s: The recording's extent, for the duty fraction, or None.

    Returns:
        ``{"<measure>.<statistic>": value}`` per measure, plus ``all.*`` over every measure.
    """
    out: dict[str, float] = {}
    for measure in (*SPAN_MEASURES, "all"):
        selected = [span for span in spans if measure == "all" or span["measure"] == measure]
        durations = [float(span["duration"]) for span in selected]
        for key, value in _stats(durations).items():
            out[f"{measure}.duration_{key}"] = value
        peaks = [
            float(span["peak_db"])
            for span in selected
            if span["peak_db"] is not None and math.isfinite(float(span["peak_db"]))
        ]
        for key, value in _stats(peaks).items():
            out[f"{measure}.peak_over_floor_db_{key}"] = value
        out[f"{measure}.corroborated_n"] = float(sum(1 for span in selected if span["corroborated"]))
        out[f"{measure}.corroborated_total"] = float(sum(int(span["corroborated"]) for span in selected))
        if duration_s:
            out[f"{measure}.duty_fraction"] = float(sum(durations)) / float(duration_s)
            out[f"{measure}.rate_per_s"] = len(selected) / float(duration_s)
    return out


def _squim_statistics(
    assertions: Sequence[tuple[tuple[float, float], dict[str, Any]]],
    measure_by_extent: dict[tuple[float, float], str],
) -> dict[str, float]:
    """Reduce the per-span SQUIM assertions to per-population distribution summaries.

    Args:
        assertions: ``(extent_key, attributes)`` for every ``squim`` assertion in the store.
        measure_by_extent: Which measure each live span's extent belongs to.

    Returns:
        ``{"<population>.<metric>.<statistic>": value}`` plus ``<population>.n`` and
        ``<population>.unmeasured``.
    """
    joined = [
        (measure_by_extent[extent], attributes) for extent, attributes in assertions if extent in measure_by_extent
    ]
    out: dict[str, float] = {}
    for population in SQUIM_POPULATIONS:
        selected = [attributes for measure, attributes in joined if population in ("all", measure)]
        out[f"{population}.n"] = float(len(selected))
        out[f"{population}.unmeasured"] = float(sum(1 for row in selected if "unmeasured" in row))
        for metric in SQUIM_METRICS:
            values = [float(row[metric]) for row in selected if metric in row]
            for key, value in _stats(values).items():
                out[f"{population}.{metric}.{key}"] = value
    return out


def extract_features(store_path: Path, stem: str, run_root: str, task_id: str, family: str) -> RecordingFeatures:
    """Stream one store and reduce it to the routing evidence.

    Invalidated entities are dropped, matching the store's shared read rule in
    :func:`senselab.audio.workflows.triage.nodes.common.live_entities`.

    Args:
        store_path: The ``run/store.jsonl`` to read.
        stem: The recording's BIDS stem.
        run_root: The run directory the summary named.
        task_id: The sanitized task id.
        family: The task family.

    Returns:
        The record.
    """
    features = RecordingFeatures(stem=stem, run_root=run_root, task_id=task_id, family=family)
    invalidated: set[str] = set()
    words: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    squim: list[tuple[tuple[float, float], dict[str, Any]]] = []
    kinds: dict[str, str] = {}

    for record in read_store(store_path):
        if record.get("record") == "relation":
            if record.get("relation") == _INVALIDATED:
                invalidated.add(str(record.get("source")))
            continue
        prov_type = record.get("prov_type")
        attributes = record.get("attributes") or {}
        if prov_type is None:
            continue
        features.n_entities += 1
        entity_id = str(record.get("id"))
        if prov_type == "measurement":
            _absorb_measurement(features, attributes)
        elif prov_type == "word":
            words.append({"id": entity_id, **attributes})
        elif prov_type == "span":
            extent = record.get("extent")
            if extent is not None:
                spans.append(
                    {
                        "id": entity_id,
                        "measure": str(attributes.get("measure")),
                        "extent": _extent_key(extent),  # type: ignore[arg-type]
                        "duration": float(extent[1]) - float(extent[0]),  # type: ignore[index]
                        "peak_db": attributes.get("peak_over_floor_db"),
                        "corroborated": len(attributes.get("corroborated_by") or []),
                    }
                )
        elif prov_type == "assertion" and attributes.get("name") == "squim":
            extent = record.get("extent")
            if extent is not None:
                squim.append((_extent_key(extent), attributes))  # type: ignore[arg-type]
        elif prov_type == "kind":
            kinds[entity_id] = str(attributes.get("kind"))
            features.kind_state[str(attributes.get("kind"))] = str(attributes.get("state"))
        elif prov_type == "verdict":
            features.verdicts[str(attributes.get("node"))] = str(attributes.get("outcome"))
        elif prov_type == "stream" and attributes.get("name") == "recording":
            extent = record.get("extent")
            if extent is not None:
                features.duration_s = float(extent[1]) - float(extent[0])

    live_words = [word for word in words if word["id"] not in invalidated]
    counts = {outcome: 0 for outcome in WORD_OUTCOMES}
    for word in live_words:
        outcome = str(word.get("outcome"))
        if outcome in counts:
            counts[outcome] += 1
    counts["total"] = len(live_words)
    counts["bracketed"] = sum(1 for word in live_words if word.get("bracketed"))
    counts["lexical"] = counts["total"] - counts["bracketed"]
    counts["agreement_lexical"] = sum(
        1 for word in live_words if word.get("outcome") == "agreement" and not word.get("bracketed")
    )
    features.words = counts

    live_spans = [span for span in spans if span["id"] not in invalidated]
    for measure in SPAN_MEASURES:
        durations = [float(span["duration"]) for span in live_spans if span["measure"] == measure]
        features.span_count[measure] = len(durations)
        features.span_longest_s[measure] = max(durations) if durations else 0.0
        features.span_total_s[measure] = float(sum(durations))
    features.span_stats = _span_statistics(live_spans, features.duration_s)
    features.squim = _squim_statistics(squim, {span["extent"]: span["measure"] for span in live_spans})
    features.classifier_streams = sorted(set(features.classifier_streams))
    for kind_name in ("speech", "airway", "voice"):
        features.kind_state.setdefault(kind_name, "missing")
    return features
