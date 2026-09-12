"""Per-recording routing evidence, stream-parsed out of one triage run's provenance stores.

The store is read one line at a time and never held whole. Only the entity records a routing
detector could read are decoded; the label summaries are reduced to the peaks in
:data:`~senselab.audio.workflows.triage.routing_analysis.labels.TRACKED_LABELS` and everything else
in them is dropped. A span carries every label its classifier put in the span's top K and at or
above the floor, both read from ``windows.<classifier>`` through
:class:`~senselab.audio.workflows.triage.label_membership.LabelMembership`, so a span may carry
several labels and counts toward each. A label outside ``TRACKED_LABELS`` is dropped.

The ``praat_features`` scalars are carried as the store keys them. The ``ppg_posteriorgram``
sidecar the store references is opened, reduced to the summary in :data:`PPG_SUMMARY_KEYS` and
closed; the array itself never enters a record.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.label_membership import LabelMembership, optional_label_membership
from senselab.audio.workflows.triage.routing_analysis.labels import (
    CLASSIFIERS,
    LABEL_SETS,
    TRACKED_LABELS,
    peak_key,
)

SPAN_MEASURES: tuple[str, ...] = ("amplitude", "continuity", "gap")
"""The span measures PREPROCESS proposes, each counted and timed separately."""

SQUIM_METRICS: tuple[str, ...] = ("pesq", "si_sdr", "stoi")
"""The three objective heads a ``squim`` assertion carries."""

SQUIM_POPULATIONS: tuple[str, ...] = ("all", "amplitude", "gap")
"""Which spans a SQUIM summary is taken over: every span, or one measure's spans."""

SPAN_CLASSIFIERS: dict[str, str] = {"span_yamnet": "yamnet", "span_hear": "hear"}
"""The per-span classifier measurements PREPROCESS writes, and the classifier each one belongs to."""

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

PPG_ENTITY_KEYS: tuple[str, ...] = ("frames", "n_phonemes", "seconds_per_frame")
"""The ``ppg_posteriorgram`` scalars the measurement itself carries, readable without the sidecar."""

PPG_SUMMARY_KEYS: tuple[str, ...] = (
    *PPG_ENTITY_KEYS,
    "segment_count",
    "segment_rate_per_s",
    "segment_duration_n",
    "segment_duration_min",
    "segment_duration_median",
    "segment_duration_mean",
    "segment_duration_max",
    "segment_duration_iqr",
    "segment_duration_p75",
    "segment_duration_p90",
    "distinct_phonemes",
    "silent_fraction",
    "repetition_peak",
    "repetition_lag_segments",
    "repetition_mean",
    "repetition_prominence",
)
"""Every key a posteriorgram is reduced to. The keys past the entity ones need the sidecar."""

SILENT_PHONEME = "<silent>"
"""The ppgs inventory's silence label, as its own ``PHONEME_LABELS`` spells it."""

_ENTITY_PREFIX = '{"attributes"'
_RELATION_MARKER = '"relation"'
_INVALIDATED = "wasInvalidatedBy"
_SUMMARY_SUFFIX = "_label_summary"
_SUMMARY_ALL_SUFFIX = "_summary_all"

TRANSCRIPT_CAP = 300
"""How much of the consensus transcript is kept, so a disagreement can be read, not just counted."""

_BRACKET_TYPE_DROP = re.compile(r"[^a-z0-9]+")


def bracket_type(text: str) -> str | None:
    """The type a bracketed consensus token names, normalised.

    Args:
        text: A consensus word's ``text`` attribute, as the recognizer or the onomatopoeic
            vocabulary spelled it.

    Returns:
        The bracket's content, casefolded with every non-alphanumeric character removed, so
        ``"[Throat Clearing]"`` and ``"[THROATCLEARING]"`` are one type; None when the token is not
        bracketed or carries nothing between the brackets.
    """
    stripped = text.strip()
    if len(stripped) < 2 or not stripped.startswith("[") or not stripped.endswith("]"):
        return None
    return _BRACKET_TYPE_DROP.sub("", stripped[1:-1].casefold()) or None


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
        bracketed_types: How many live consensus words carry each :func:`bracket_type`, taken off
            the word entities rather than off the capped transcript. Only types the recording
            carries are keyed.
        consensus_present: Whether a ``consensus_transcript`` measurement was written at all.
        transcript: The consensus transcript's first :data:`TRANSCRIPT_CAP` characters.
        residual: The ``residual`` measurement's scalar attributes, or empty when it is absent.
        span_count: Live spans per measure.
        span_longest_s: The longest live span per measure, in seconds.
        span_total_s: Total live span seconds per measure.
        span_stats: ``{"<measure>.<statistic>": value}`` over the live spans of each measure —
            the duration distribution, the ``peak_over_floor_db`` distribution, how many carry a
            ``corroborated_by`` entry, and the duty fraction against the recording's extent.
        span_label_stats: ``{"<classifier>.<label>.span_count": n}`` and
            ``{"<classifier>.<label>.peak_over_floor_db_<statistic>": value}`` over the live spans
            carrying that per-span classifier label. A span carries every label in its own top K
            that also clears the floor, so it contributes to each of them. Only labels in
            :data:`~senselab.audio.workflows.triage.routing_analysis.labels.TRACKED_LABELS` that at
            least one live span carries appear at all.
        span_label_set_stats: The same two families of keys over the named unions in
            :data:`~senselab.audio.workflows.triage.routing_analysis.labels.LABEL_SETS`, keyed
            ``"<classifier>.<set>.…"``. A span counts toward a set when any member of that set is
            one of the labels it carries, and it counts once however many members qualify.
        squim: ``{"<population>.<metric>.<statistic>": value}`` over the per-span SQUIM
            assertions, plus ``<population>.n`` and ``<population>.unmeasured``.
        level: The whole-file ``level`` measurement's scalars.
        disruptions: The whole-file ``disruptions_file`` measurement's scalars.
        silence: ``threshold``, ``n_windows``, ``n_silence`` and ``fraction`` from the YAMNet
            ``Silence`` projection.
        praat: The ``praat_features`` scalars, keyed as the measurement keys them. A scalar the
            store recorded as null is not keyed at all, so an unmeasured quantity stays distinct
            from a measured zero, and the whole mapping is empty when the measurement is absent.
        ppg: The ``ppg_posteriorgram`` sidecar reduced to :data:`PPG_SUMMARY_KEYS`. The keys in
            :data:`PPG_ENTITY_KEYS` come off the measurement; the rest need the sidecar and are
            absent when it cannot be read.
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
    bracketed_types: dict[str, int] = field(default_factory=dict)
    consensus_present: bool = False
    transcript: str = ""
    residual: dict[str, float] = field(default_factory=dict)
    span_count: dict[str, int] = field(default_factory=dict)
    span_longest_s: dict[str, float] = field(default_factory=dict)
    span_total_s: dict[str, float] = field(default_factory=dict)
    span_stats: dict[str, float] = field(default_factory=dict)
    span_label_stats: dict[str, float] = field(default_factory=dict)
    span_label_set_stats: dict[str, float] = field(default_factory=dict)
    squim: dict[str, float] = field(default_factory=dict)
    level: dict[str, float] = field(default_factory=dict)
    disruptions: dict[str, float] = field(default_factory=dict)
    silence: dict[str, float] = field(default_factory=dict)
    praat: dict[str, float] = field(default_factory=dict)
    ppg: dict[str, float] = field(default_factory=dict)
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
        skipped without being decoded.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(_ENTITY_PREFIX):
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


def _absorb_span_window(
    span_scores: dict[str, dict[str, dict[str, float]]], classifier: str, attributes: dict[str, Any]
) -> None:
    """Pool one per-span classifier window into the per-label maxima of the span it names.

    The maximum is taken before the membership rule rather than after, so a span whose classifier
    placed several windows over it is ranked on one score per label. Taking the top K of each
    window and unioning them instead would rank a label on whichever window it happened to survive.

    Args:
        span_scores: ``{classifier: {span_id: {label: max score}}}``, updated in place.
        classifier: The classifier the window belongs to.
        attributes: The ``span_<classifier>`` measurement's attributes.
    """
    span_id = attributes.get("span_id")
    scores = attributes.get("raw_scores") or {}
    if span_id is None or not scores:
        return
    pooled = span_scores.setdefault(classifier, {}).setdefault(str(span_id), {})
    for label, score in scores.items():
        key = str(label)
        value = float(score)
        if value > pooled.get(key, -math.inf):
            pooled[key] = value


def _finite_scalars(table: Mapping[str, Any]) -> dict[str, float]:
    """The finite numbers in a mapping, under the keys the mapping already uses.

    Args:
        table: The mapping, whose values may be of any type.

    Returns:
        Every value that is a finite real number, keyed unchanged. A null, a bool, a string and a
        non-finite number are all left out rather than coerced.
    """
    kept: dict[str, float] = {}
    for key, value in table.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            kept[str(key)] = number
    return kept


def _ppg_segments(path: Path) -> tuple[list[dict[str, Any]], float, int]:
    """The posteriorgram sidecar's contiguous argmax-phoneme segments.

    Args:
        path: The ``ppg_posteriorgram.npz`` the measurement's ``path`` attribute names.

    Returns:
        The segments :func:`~senselab.audio.tasks.features_extraction.ppg.extract_ppg_segments`
        found, the analysed duration in seconds and the frame count. The segments are empty when
        the array holds no frame.

    Raises:
        OSError: When the sidecar cannot be opened.
        ValueError: When it holds no readable posteriorgram.
    """
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.features_extraction.ppg import extract_ppg_segments, to_frame_major_posteriorgram

    with np.load(path) as archive:
        posteriorgram = torch.from_numpy(archive["posteriorgram"].astype(np.float32))
        duration_s = float(archive["duration_s"])
        sampling_rate = int(archive["sampling_rate"])
    frame_major = to_frame_major_posteriorgram(posteriorgram)
    frames = int(frame_major.shape[0])
    if frames == 0 or duration_s <= 0.0:
        return [], duration_s, frames
    clock = Audio(waveform=torch.zeros(1, round(duration_s * sampling_rate)), sampling_rate=sampling_rate)
    return extract_ppg_segments(clock, frame_major), duration_s, frames


def _repetition(labels: Sequence[int]) -> dict[str, float]:
    """How strongly the argmax phoneme sequence repeats itself.

    Args:
        labels: Each segment's dominant phoneme index, in order.

    Returns:
        ``repetition_peak``, the largest fraction of positions agreeing with the sequence shifted
        by one lag, over every lag up to half the sequence; ``repetition_lag_segments``, the lag it
        was reached at; ``repetition_mean`` over every lag, which is this sequence's own agreement
        by chance; and ``repetition_prominence``, the peak over that mean. Empty when the sequence
        is too short to carry two lags.
    """
    length = len(labels)
    if length // 2 < 2:
        return {}
    sequence = np.asarray(labels, dtype=np.int64)
    matches = np.array(
        [np.count_nonzero(sequence[:-lag] == sequence[lag:]) / (length - lag) for lag in range(1, length // 2 + 1)]
    )
    peak = int(np.argmax(matches))
    mean = float(matches.mean())
    return {
        "repetition_peak": float(matches[peak]),
        "repetition_lag_segments": float(peak + 1),
        "repetition_mean": mean,
        "repetition_prominence": float(matches[peak]) - mean,
    }


def _ppg_summary(attributes: Mapping[str, Any], run_dir: Path) -> dict[str, float]:
    """One ``ppg_posteriorgram`` measurement reduced to :data:`PPG_SUMMARY_KEYS`.

    Args:
        attributes: The measurement's attributes, carrying the sidecar's relative path.
        run_dir: The run directory that path is relative to.

    Returns:
        The summary. Only the :data:`PPG_ENTITY_KEYS` are keyed when the sidecar names no path,
        cannot be opened or holds no readable array; a summary key is absent rather than zero
        whenever the quantity it names was not computed.
    """
    summary = _finite_scalars({key: attributes.get(key) for key in PPG_ENTITY_KEYS})
    relative = attributes.get("path")
    if not relative:
        return summary
    try:
        segments, duration_s, frames = _ppg_segments(run_dir / str(relative))
    except (OSError, ValueError, KeyError):
        return summary
    if not segments:
        return summary
    durations = [float(segment["duration_seconds"]) for segment in segments]
    summary["segment_count"] = float(len(segments))
    if duration_s > 0.0:
        summary["segment_rate_per_s"] = len(segments) / duration_s
    for key, value in _stats(durations).items():
        summary[f"segment_duration_{key}"] = value
    labels = [int(segment["phoneme_index"]) for segment in segments]
    summary["distinct_phonemes"] = float(len(set(labels)))
    if frames > 0:
        silent = sum(int(segment["frame_count"]) for segment in segments if segment["phoneme"] == SILENT_PHONEME)
        summary["silent_fraction"] = silent / frames
    summary.update(_repetition(labels))
    return summary


def _absorb_measurement(
    features: RecordingFeatures,
    attributes: dict[str, Any],
    span_scores: dict[str, dict[str, dict[str, float]]],
    run_dir: Path,
) -> None:
    """Fold one ``measurement`` entity into the record.

    Args:
        features: The record being built.
        attributes: The measurement's attributes.
        span_scores: The per-span per-label maxima table, updated in place.
        run_dir: The run directory a sidecar-bearing measurement's path is relative to.
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
    if name == "praat_features":
        features.praat = _finite_scalars(attributes.get("features") or {})
        return
    if name == "ppg_posteriorgram":
        features.ppg = _ppg_summary(attributes, run_dir)
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
    span_classifier = SPAN_CLASSIFIERS.get(name)
    if span_classifier is not None:
        _absorb_span_window(span_scores, span_classifier, attributes)
        if span_classifier == "hear":
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
        ``n``, ``min``, ``median``, ``mean``, ``max``, ``iqr``, ``p75`` and ``p90``; empty when the
        sample is. A sample too small to interpolate a percentile reports its largest value for it.
    """
    if not values:
        return {}
    ordered = sorted(values)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        spread = quartiles[2] - quartiles[0]
    else:
        spread = ordered[-1] - ordered[0]
    if len(ordered) >= 2:
        upper = statistics.quantiles(ordered, n=4, method="inclusive")[2]
        ninth = statistics.quantiles(ordered, n=10, method="inclusive")[8]
    else:
        upper = ninth = ordered[-1]
    return {
        "n": float(len(ordered)),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
        "iqr": spread,
        "p75": upper,
        "p90": ninth,
    }


def _extent_key(extent: Sequence[float]) -> tuple[float, float]:
    """The rounded extent a SQUIM assertion is joined to its span by.

    Args:
        extent: The entity's ``[start, end]``.

    Returns:
        Both bounds rounded to the microsecond, which is finer than any span PREPROCESS proposes.
    """
    return (round(float(extent[0]), 6), round(float(extent[1]), 6))


def _finite_peaks(spans: Sequence[dict[str, Any]]) -> list[float]:
    """The ``peak_over_floor_db`` of every span that carries a finite one.

    Args:
        spans: The spans.

    Returns:
        The sample, in span order.
    """
    return [
        float(span["peak_db"])
        for span in spans
        if span["peak_db"] is not None and math.isfinite(float(span["peak_db"]))
    ]


def _span_labels(
    spans: Sequence[dict[str, Any]],
    span_scores: dict[str, dict[str, dict[str, float]]],
    classifier: str,
    membership: LabelMembership,
) -> dict[str, tuple[str, ...]]:
    """Which labels each live span carries, under one classifier's membership rule.

    Args:
        spans: The live spans, each carrying ``id``.
        span_scores: ``{classifier: {span_id: {label: max score}}}``.
        classifier: The classifier whose spans are read.
        membership: The rule read from ``windows.<classifier>``.

    Returns:
        ``{span_id: labels}`` over the spans carrying at least one label, whether tracked or not.
    """
    pooled = span_scores.get(classifier) or {}
    carried: dict[str, tuple[str, ...]] = {}
    for span in spans:
        span_id = str(span["id"])
        scores = pooled.get(span_id)
        if not scores:
            continue
        members = tuple(membership.members(scores))
        if members:
            carried[span_id] = members
    return carried


def _distribution(prefix: str, selected: Sequence[dict[str, Any]]) -> dict[str, float]:
    """One group of spans reduced to the keys a span-label detector reads.

    Args:
        prefix: ``"<classifier>.<label>"`` or ``"<classifier>.<set>"``.
        selected: The spans in the group.

    Returns:
        ``{"<prefix>.span_count": n}`` and ``{"<prefix>.peak_over_floor_db_<statistic>": value}``.
    """
    out = {f"{prefix}.span_count": float(len(selected))}
    for key, value in _stats(_finite_peaks(selected)).items():
        out[f"{prefix}.peak_over_floor_db_{key}"] = value
    return out


def _label_span_statistics(
    spans: Sequence[dict[str, Any]],
    span_scores: dict[str, dict[str, dict[str, float]]],
    memberships: Mapping[str, LabelMembership],
) -> tuple[dict[str, float], dict[str, float]]:
    """Reduce the live spans to one distribution per label, and one per named label set.

    Labels outside
    :data:`~senselab.audio.workflows.triage.routing_analysis.labels.TRACKED_LABELS` are dropped, so
    a recording whose spans carry none of them contributes nothing. A set's distribution is over the
    spans carrying any of its members, counted once each, and is not the union of its members'
    distributions.

    Args:
        spans: The live spans, each carrying ``id`` and ``peak_db``.
        span_scores: ``{classifier: {span_id: {label: max score}}}``.
        memberships: The membership rule per classifier.

    Returns:
        The per-label statistics and the per-set statistics, in that order.
    """
    per_label: dict[str, float] = {}
    per_set: dict[str, float] = {}
    for classifier in SPAN_CLASSIFIERS.values():
        carried = _span_labels(spans, span_scores, classifier, memberships[classifier])
        tracked = TRACKED_LABELS[classifier]
        grouped: dict[str, list[dict[str, Any]]] = {}
        by_set: dict[str, list[dict[str, Any]]] = {}
        for span in spans:
            labels = carried.get(str(span["id"]), ())
            for label in labels:
                if label in tracked:
                    grouped.setdefault(label, []).append(span)
            for set_name, per_classifier in LABEL_SETS.items():
                members = per_classifier[classifier]
                if members and any(label in members for label in labels):
                    by_set.setdefault(set_name, []).append(span)
        for label, selected in sorted(grouped.items()):
            per_label.update(_distribution(f"{classifier}.{label}", selected))
        for set_name, selected in sorted(by_set.items()):
            per_set.update(_distribution(f"{classifier}.{set_name}", selected))
    return per_label, per_set


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
        for key, value in _stats(_finite_peaks(selected)).items():
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


def span_label_memberships(config: TriageConfig) -> dict[str, LabelMembership]:
    """The membership rule for each classifier that writes per-span windows.

    Args:
        config: The resolved triage configuration.

    Returns:
        The rule per classifier in :data:`SPAN_CLASSIFIERS`, read from the same
        ``windows.<classifier>`` keys PREPROCESS stamps its own windows with.

    Raises:
        ValueError: When a classifier's floor or top-K is null. Attribution needs both, and a
            default chosen here would be an unfitted threshold in code.
    """
    memberships: dict[str, LabelMembership] = {}
    for classifier in SPAN_CLASSIFIERS.values():
        rule = optional_label_membership(config, classifier)
        if rule is None:
            raise ValueError(
                f"windows.{classifier}.default_threshold has no value; span label attribution "
                "cannot run without the floor it and PREPROCESS share"
            )
        memberships[classifier] = rule
    return memberships


def extract_features(
    store_path: Path,
    stem: str,
    run_root: str,
    task_id: str,
    family: str,
    memberships: Mapping[str, LabelMembership],
) -> RecordingFeatures:
    """Stream one store and reduce it to the routing evidence.

    Invalidated entities are dropped, matching the store's shared read rule in
    :func:`senselab.audio.workflows.triage.nodes.common.live_entities`.

    Args:
        store_path: The ``run/store.jsonl`` to read. A sidecar a measurement names by relative path
            is resolved against its directory.
        stem: The recording's BIDS stem.
        run_root: The run directory the summary named.
        task_id: The sanitized task id.
        family: The task family.
        memberships: Which labels a span carries, per classifier, from
            :func:`span_label_memberships`.

    Returns:
        The record.
    """
    features = RecordingFeatures(stem=stem, run_root=run_root, task_id=task_id, family=family)
    invalidated: set[str] = set()
    words: list[dict[str, Any]] = []
    spans: list[dict[str, Any]] = []
    squim: list[tuple[tuple[float, float], dict[str, Any]]] = []
    span_scores: dict[str, dict[str, dict[str, float]]] = {}
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
            _absorb_measurement(features, attributes, span_scores, store_path.parent)
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
    typed: Counter[str] = Counter()
    for word in live_words:
        if word.get("bracketed"):
            named = bracket_type(str(word.get("text") or ""))
            if named is not None:
                typed[named] += 1
    features.bracketed_types = dict(sorted(typed.items()))

    live_spans = [span for span in spans if span["id"] not in invalidated]
    for measure in SPAN_MEASURES:
        durations = [float(span["duration"]) for span in live_spans if span["measure"] == measure]
        features.span_count[measure] = len(durations)
        features.span_longest_s[measure] = max(durations) if durations else 0.0
        features.span_total_s[measure] = float(sum(durations))
    features.span_stats = _span_statistics(live_spans, features.duration_s)
    features.span_label_stats, features.span_label_set_stats = _label_span_statistics(
        live_spans, span_scores, memberships
    )
    features.squim = _squim_statistics(squim, {span["extent"]: span["measure"] for span in live_spans})
    features.classifier_streams = sorted(set(features.classifier_streams))
    for kind_name in ("speech", "airway", "voice"):
        features.kind_state.setdefault(kind_name, "missing")
    return features
