"""Per-recording routing evidence, stream-parsed out of one triage run's provenance stores.

The store is read one line at a time and never held whole. Only the entity records a routing
detector could read are decoded; the label summaries are reduced to the peaks in
:data:`~senselab.audio.workflows.triage.routing_analysis.labels.TRACKED_LABELS` and everything else
in them is dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from senselab.audio.workflows.triage.routing_analysis.labels import (
    CLASSIFIERS,
    TRACKED_LABELS,
    peak_key,
)

SPAN_MEASURES: tuple[str, ...] = ("amplitude", "continuity", "gap")
"""The span measures PREPROCESS proposes, each counted and timed separately."""

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
    spans: list[tuple[str, str, float]] = []
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
                spans.append((entity_id, str(attributes.get("measure")), float(extent[1]) - float(extent[0])))  # type: ignore[arg-type]
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

    for measure in SPAN_MEASURES:
        durations = [
            duration for entity_id, kind, duration in spans if kind == measure and entity_id not in invalidated
        ]
        features.span_count[measure] = len(durations)
        features.span_longest_s[measure] = max(durations) if durations else 0.0
        features.span_total_s[measure] = float(sum(durations))
    features.classifier_streams = sorted(set(features.classifier_streams))
    for kind_name in ("speech", "airway", "voice"):
        features.kind_state.setdefault(kind_name, "missing")
    return features
