"""The candidate routing detectors and the thresholds each is swept over.

A detector reads one number out of a :class:`~senselab.audio.workflows.triage.routing_analysis.
features.RecordingFeatures` and fires when that number is at or above a threshold. No threshold is
preferred here: every one in a detector's grid is scored, and
``taxonomy.consolidation_floor`` (0.2) is marked where it falls rather than adopted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.labels import FAMILIES, LABEL_SETS, peak_key

GATE_CLOSED = -1.0
"""The value a gated detector reads when its corroborator did not fire: below every threshold."""

GATE_CLOSED_BELOW = math.inf
"""The same, for a ``below``-polarity detector, where a small value is what fires."""

CONSOLIDATION_FLOOR = 0.2
"""``taxonomy.consolidation_floor`` in ``data/config/default.yaml``, marked in every score grid."""

SCORE_GRID: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
)
"""Classifier-score thresholds. Contains :data:`CONSOLIDATION_FLOOR` exactly."""

COUNT_GRID: tuple[float, ...] = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)
"""Word-count thresholds."""

DURATION_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
"""Span-duration thresholds, in seconds."""

FRACTION_GRID: tuple[float, ...] = (
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    3e-3,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    0.9,
)
"""Residual energy-fraction thresholds."""

DB_OVER_FLOOR_GRID: tuple[float, ...] = (
    3.0,
    6.0,
    9.0,
    12.0,
    15.0,
    18.0,
    21.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    55.0,
    60.0,
    70.0,
    80.0,
)
"""``peak_over_floor_db`` thresholds, in dB above the span's own floor."""

DBFS_GRID: tuple[float, ...] = (-60.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -6.0, -3.0)
"""Whole-file level thresholds, in dBFS."""

PESQ_GRID: tuple[float, ...] = (1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0)
"""SQUIM PESQ thresholds; the head's range is 1.0-4.5."""

SI_SDR_GRID: tuple[float, ...] = (-20.0, -15.0, -12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 5.0, 10.0, 15.0)
"""SQUIM SI-SDR thresholds, in dB."""

STOI_GRID: tuple[float, ...] = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
"""SQUIM STOI thresholds; the head's range is 0-1."""

SPREAD_GRID: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 5.0)
"""Thresholds for an interquartile spread, whose unit is that of the quantity it spreads."""

ZCR_GRID: tuple[float, ...] = (100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0)
"""Zero-crossing-rate thresholds, in crossings per second over the original recording."""

RATE_GRID: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
"""Span-rate thresholds, in spans per second of recording."""

DB_SPREAD_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0)
"""Thresholds for an interquartile spread whose unit is dB."""

PROPORTION_GRID: tuple[float, ...] = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
"""Thresholds for a quantity that is a proportion of a whole, spanning the unit interval."""

F0_SPREAD_HZ_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
"""Thresholds for the standard deviation of F0 over a recording, in hertz."""

RELATIVE_SPREAD_GRID: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
"""Thresholds for a spread divided by its own mean, which is dimensionless."""

VOICE_QUALITY_DB_GRID: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0)
"""Thresholds for a Praat voice-quality level — harmonics-to-noise, cepstral peak — in dB."""

SYLLABLE_RATE_GRID: tuple[float, ...] = (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0)
"""Thresholds for a Praat syllable rate, in syllables per second."""

PAUSE_RATE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
"""Thresholds for a Praat pause rate, in pauses per second."""

SEGMENT_RATE_GRID: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
"""Thresholds for the posteriorgram's argmax-segment rate, in segments per second."""

SEGMENT_DURATION_GRID: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
"""Thresholds for one posteriorgram argmax segment's duration, in seconds."""

SIGNED_SCORE_GRID: tuple[float, ...] = (
    -0.9,
    -0.7,
    -0.5,
    -0.3,
    -0.2,
    -0.1,
    -0.05,
    -0.01,
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    0.9,
)
"""Thresholds for a difference of two classifier scores, which is signed."""


@dataclass(frozen=True)
class Detector:
    """One candidate rule TAXONOMY could route on.

    Attributes:
        name: The detector's id, unique across the catalogue.
        kind: The branch it would route to. ``speech``, ``airway`` and ``voice`` are the branches
            TAXONOMY carries a state for; ``cough``, ``glide`` and ``ddk`` are analysis-only.
        reader: What to read, as ``(source, *arguments)``; see :func:`detector_value`. A
            ``gated`` reader nests two of these.
        unit: The unit of the number it reads.
        thresholds: Every threshold it is scored at.
        polarity: ``above`` fires at or over the threshold, ``below`` fires at or under it.
    """

    name: str
    kind: str
    reader: tuple[Any, ...]
    unit: str
    thresholds: tuple[float, ...]
    polarity: str = "above"


def sweep_points(detector: Detector) -> tuple[float, ...]:
    """The thresholds a detector is scored at.

    Args:
        detector: The detector.

    Returns:
        Its threshold grid, ascending.
    """
    return tuple(sorted(detector.thresholds))


def detector_value(features: RecordingFeatures, detector: Detector) -> float | None:
    """The number a detector reads out of one recording.

    Args:
        features: The recording's extracted evidence.
        detector: The detector.

    Returns:
        The number, or None when the evidence the detector reads is not in the store, which
        excludes the recording from that detector's scoring rather than counting as a zero.

    Raises:
        ValueError: When the detector names a source this function does not implement.
    """
    source, *arguments = detector.reader
    if source == "words":
        return float(features.words.get(arguments[0], 0))
    if source == "bracketed_set":
        if not features.consensus_present:
            return None
        return float(sum(features.bracketed_types.get(str(name), 0) for name in arguments))
    if source == "stream_peak_max":
        stream, classifier = arguments
        if f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        prefix = peak_key(stream, classifier, "")
        return max((score for key, score in features.peaks.items() if key.startswith(prefix)), default=0.0)
    if source == "residual":
        if not features.residual:
            return None
        value = features.residual.get(arguments[0])
        return None if value is None else float(value)
    if source == "span_longest":
        return float(features.span_longest_s.get(arguments[0], 0.0))
    if source == "span_total":
        return float(features.span_total_s.get(arguments[0], 0.0))
    if source == "span_count":
        return float(features.span_count.get(arguments[0], 0))
    if source == "peak":
        stream, classifier, kind = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        labels = FAMILIES[kind][classifier]
        if not labels:
            return None
        return max((features.peaks.get(peak_key(stream, classifier, label), 0.0) for label in labels), default=0.0)
    if source == "gated":
        primary, gate, gate_threshold, polarity = arguments
        gate_value = detector_value(features, Detector(detector.name, detector.kind, gate, "score", ()))
        if gate_value is None:
            return None
        passes = gate_value >= float(gate_threshold) if polarity == "above" else gate_value < float(gate_threshold)
        if not passes:
            return GATE_CLOSED if detector.polarity == "above" else GATE_CLOSED_BELOW
        return detector_value(
            features, Detector(detector.name, detector.kind, primary, detector.unit, (), detector.polarity)
        )
    if source == "peak_label":
        stream, classifier, label = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        return float(features.peaks.get(peak_key(stream, classifier, label), 0.0))
    if source == "peak_set":
        stream, classifier, set_name = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        labels = LABEL_SETS[set_name][classifier]
        if not labels:
            return None
        return max((features.peaks.get(peak_key(stream, classifier, label), 0.0) for label in labels), default=0.0)
    if source == "span_stat":
        return _optional(features.span_stats, arguments[0])
    if source == "span_label_stat":
        return _optional(features.span_label_stats, arguments[0])
    if source == "span_label_set_stat":
        return _optional(features.span_label_set_stats, arguments[0])
    if source == "squim":
        return _optional(features.squim, arguments[0])
    if source == "level":
        return _optional(features.level, arguments[0])
    if source == "disruptions":
        return _optional(features.disruptions, arguments[0])
    if source == "silence":
        return _optional(features.silence, arguments[0])
    if source == "praat":
        return _optional(features.praat, arguments[0])
    if source == "ppg":
        return _optional(features.ppg, arguments[0])
    if source == "ratio":
        numerator = detector_value(features, Detector(detector.name, detector.kind, arguments[0], detector.unit, ()))
        denominator = detector_value(features, Detector(detector.name, detector.kind, arguments[1], detector.unit, ()))
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return numerator / denominator
    if source == "difference":
        left = detector_value(features, Detector(detector.name, detector.kind, arguments[0], detector.unit, ()))
        right = detector_value(features, Detector(detector.name, detector.kind, arguments[1], detector.unit, ()))
        return None if left is None or right is None else left - right
    raise ValueError(f"unknown detector source {source!r}")


def _optional(table: dict[str, float], key: str) -> float | None:
    """One value out of a feature table, absent rather than zero when the derivative is missing.

    Args:
        table: The feature table.
        key: The key.

    Returns:
        The value, or None when the key is absent or not finite.
    """
    value = table.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _peak_detectors() -> list[Detector]:
    """One detector per kind, classifier and stream, reading that family's best score.

    Returns:
        The classifier-score detectors.
    """
    built: list[Detector] = []
    for kind, per_classifier in FAMILIES.items():
        for classifier, labels in per_classifier.items():
            if not labels:
                continue
            for stream in ("plain", "enhanced", "residual"):
                built.append(
                    Detector(
                        name=f"{kind}.{classifier}_peak.{stream}",
                        kind=kind,
                        reader=("peak", stream, classifier, kind),
                        unit="score",
                        thresholds=SCORE_GRID,
                    )
                )
            if classifier == "hear":
                built.append(
                    Detector(
                        name=f"{kind}.hear_peak.span",
                        kind=kind,
                        reader=("peak", "span", "hear", kind),
                        unit="score",
                        thresholds=SCORE_GRID,
                    )
                )
            built.append(
                Detector(
                    name=f"{kind}.{classifier}_peak.consensus",
                    kind=kind,
                    reader=("peak", "consensus", classifier, kind),
                    unit="score",
                    thresholds=SCORE_GRID,
                )
            )
    return built


def _singing_detectors() -> list[Detector]:
    """The AudioSet singing-subtree union, on each stream and both AudioSet classifiers.

    Returns:
        The union detectors, for the ``voice`` and ``glide`` kinds.
    """
    built: list[Detector] = []
    for kind in ("voice", "glide"):
        for classifier in ("yamnet", "ast"):
            for stream in ("plain", "enhanced"):
                built.append(
                    Detector(
                        name=f"{kind}.{classifier}_singing_union.{stream}",
                        kind=kind,
                        reader=("peak_set", stream, classifier, "singing"),
                        unit="score",
                        thresholds=SCORE_GRID,
                    )
                )
    return built


_COUGH_VS_BREATH: tuple[Detector, ...] = (
    Detector("cough.longest_amplitude_span", "cough", ("span_longest", "amplitude"), "seconds", DURATION_GRID, "below"),
    Detector(
        "cough.amplitude_duration_median",
        "cough",
        ("span_stat", "amplitude.duration_median"),
        "seconds",
        DURATION_GRID,
        "below",
    ),
    Detector(
        "cough.amplitude_duration_max", "cough", ("span_stat", "amplitude.duration_max"), "seconds", DURATION_GRID
    ),
    Detector("cough.amplitude_span_count", "cough", ("span_count", "amplitude"), "spans", COUNT_GRID),
    Detector("cough.amplitude_rate_per_s", "cough", ("span_stat", "amplitude.rate_per_s"), "spans/s", RATE_GRID),
    Detector(
        "cough.amplitude_duty_fraction",
        "cough",
        ("span_stat", "amplitude.duty_fraction"),
        "fraction",
        FRACTION_GRID,
        "below",
    ),
    Detector(
        "cough.amplitude_peak_over_floor_db_max",
        "cough",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector(
        "cough.amplitude_peak_over_floor_db_median",
        "cough",
        ("span_stat", "amplitude.peak_over_floor_db_median"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector("cough.squim_stoi_median", "cough", ("squim", "all.stoi.median"), "stoi", STOI_GRID, "below"),
    Detector("cough.squim_stoi_max", "cough", ("squim", "all.stoi.max"), "stoi", STOI_GRID),
    Detector("cough.squim_stoi_iqr", "cough", ("squim", "all.stoi.iqr"), "stoi", SPREAD_GRID),
    Detector("cough.squim_pesq_max", "cough", ("squim", "all.pesq.max"), "pesq", PESQ_GRID),
    Detector("cough.squim_pesq_iqr", "cough", ("squim", "all.pesq.iqr"), "pesq", SPREAD_GRID),
    Detector("cough.squim_si_sdr_median", "cough", ("squim", "all.si_sdr.median"), "dB", SI_SDR_GRID, "below"),
    Detector("cough.squim_si_sdr_max", "cough", ("squim", "all.si_sdr.max"), "dB", SI_SDR_GRID),
    Detector("cough.squim_si_sdr_iqr", "cough", ("squim", "all.si_sdr.iqr"), "dB", DB_SPREAD_GRID),
    Detector(
        "cough.squim_amplitude_si_sdr_max", "cough", ("squim", "amplitude.si_sdr.max"), "dB", SI_SDR_GRID, "below"
    ),
    Detector(
        "cough.yamnet_cough_labels.plain", "cough", ("peak_set", "plain", "yamnet", "cough_labels"), "score", SCORE_GRID
    ),
    Detector(
        "cough.hear_cough_labels.plain", "cough", ("peak_set", "plain", "hear", "cough_labels"), "score", SCORE_GRID
    ),
    Detector(
        "cough.yamnet_cough_minus_breath.plain",
        "cough",
        (
            "difference",
            ("peak_set", "plain", "yamnet", "cough_labels"),
            ("peak_set", "plain", "yamnet", "breath_labels"),
        ),
        "score",
        SIGNED_SCORE_GRID,
    ),
    Detector(
        "cough.hear_cough_minus_breath.plain",
        "cough",
        ("difference", ("peak_set", "plain", "hear", "cough_labels"), ("peak_set", "plain", "hear", "breath_labels")),
        "score",
        SIGNED_SCORE_GRID,
    ),
    Detector(
        "cough.yamnet_cough_label.plain", "cough", ("peak_label", "plain", "yamnet", "Cough"), "score", SCORE_GRID
    ),
    Detector(
        "cough.yamnet_breathing_label.plain",
        "cough",
        ("peak_label", "plain", "yamnet", "Breathing"),
        "score",
        SCORE_GRID,
        "below",
    ),
    Detector("cough.zero_crossing_rate", "cough", ("disruptions", "zero_crossing_rate"), "crossings/s", ZCR_GRID),
    Detector("cough.level_peak_dbfs", "cough", ("level", "peak_dbfs"), "dBFS", DBFS_GRID),
    Detector(
        "cough.level_crest_db",
        "cough",
        ("difference", ("level", "peak_dbfs"), ("level", "rms_dbfs")),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector("cough.silence_fraction", "cough", ("silence", "fraction"), "fraction", FRACTION_GRID),
    Detector(
        "cough.residual_energy_fraction", "cough", ("residual", "energy_fraction"), "fraction", FRACTION_GRID, "below"
    ),
    Detector("cough.residual_band_0_200", "cough", ("residual", "band_0_200"), "fraction", FRACTION_GRID, "below"),
    Detector("cough.residual_band_1000_4000", "cough", ("residual", "band_1000_4000"), "fraction", FRACTION_GRID),
    Detector(
        "cough.yamnet_cough_labels.plain+short_span",
        "cough",
        ("gated", ("peak_set", "plain", "yamnet", "cough_labels"), ("span_longest", "amplitude"), 1.0, "below"),
        "score",
        SCORE_GRID,
    ),
    Detector(
        "cough.level_crest_db+hear_cough>=0.5",
        "cough",
        (
            "gated",
            ("difference", ("level", "peak_dbfs"), ("level", "rms_dbfs")),
            ("peak_set", "plain", "hear", "cough_labels"),
            0.5,
            "above",
        ),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
)
"""Candidate discriminators between a declared cough family and a declared breath family."""

_GLIDES: tuple[Detector, ...] = (
    Detector("glide.yamnet_whistle.plain", "glide", ("peak_set", "plain", "yamnet", "whistle"), "score", SCORE_GRID),
    Detector(
        "glide.yamnet_humming_label.plain", "glide", ("peak_label", "plain", "yamnet", "Humming"), "score", SCORE_GRID
    ),
    Detector(
        "glide.yamnet_chant_label.plain", "glide", ("peak_label", "plain", "yamnet", "Chant"), "score", SCORE_GRID
    ),
    Detector("glide.yamnet_peak.plain", "glide", ("peak", "plain", "yamnet", "voice"), "score", SCORE_GRID),
    Detector("glide.longest_amplitude_span", "glide", ("span_longest", "amplitude"), "seconds", DURATION_GRID),
    Detector(
        "glide.amplitude_duration_iqr", "glide", ("span_stat", "amplitude.duration_iqr"), "seconds", DURATION_GRID
    ),
    Detector(
        "glide.amplitude_duty_fraction", "glide", ("span_stat", "amplitude.duty_fraction"), "fraction", FRACTION_GRID
    ),
    Detector("glide.amplitude_span_count", "glide", ("span_count", "amplitude"), "spans", COUNT_GRID, "below"),
    Detector(
        "glide.amplitude_peak_over_floor_db_max",
        "glide",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector("glide.squim_stoi_max", "glide", ("squim", "all.stoi.max"), "stoi", STOI_GRID),
    Detector("glide.squim_pesq_max", "glide", ("squim", "all.pesq.max"), "pesq", PESQ_GRID),
    Detector("glide.squim_si_sdr_max", "glide", ("squim", "all.si_sdr.max"), "dB", SI_SDR_GRID),
    Detector("glide.squim_si_sdr_iqr", "glide", ("squim", "all.si_sdr.iqr"), "dB", DB_SPREAD_GRID),
    Detector("glide.squim_amplitude_stoi_median", "glide", ("squim", "amplitude.stoi.median"), "stoi", STOI_GRID),
    Detector("glide.silence_fraction", "glide", ("silence", "fraction"), "fraction", FRACTION_GRID, "below"),
    Detector("glide.level_lufs", "glide", ("level", "lufs"), "dBFS", DBFS_GRID),
    Detector("glide.zero_crossing_rate", "glide", ("disruptions", "zero_crossing_rate"), "crossings/s", ZCR_GRID),
    Detector(
        "glide.residual_energy_fraction", "glide", ("residual", "energy_fraction"), "fraction", FRACTION_GRID, "below"
    ),
    Detector("glide.words_lexical", "glide", ("words", "lexical"), "words", COUNT_GRID, "below"),
    Detector(
        "glide.yamnet_singing_union.plain+no_agreed_word",
        "glide",
        ("gated", ("peak_set", "plain", "yamnet", "singing"), ("words", "agreement"), 1, "below"),
        "score",
        SCORE_GRID,
    ),
    Detector(
        "glide.yamnet_singing_union.plain+amplitude>=2s",
        "glide",
        ("gated", ("peak_set", "plain", "yamnet", "singing"), ("span_longest", "amplitude"), 2.0, "above"),
        "score",
        SCORE_GRID,
    ),
    Detector(
        "glide.longest_amplitude_span+singing>=0.05",
        "glide",
        ("gated", ("span_longest", "amplitude"), ("peak_set", "plain", "yamnet", "singing"), 0.05, "above"),
        "seconds",
        DURATION_GRID,
    ),
    Detector(
        "glide.squim_stoi_max+no_agreed_word",
        "glide",
        ("gated", ("squim", "all.stoi.max"), ("words", "agreement"), 1, "below"),
        "stoi",
        STOI_GRID,
    ),
)
"""Candidate detectors for the glide families, the worst-served of the voice families."""

_NEW_DERIVATIVES: tuple[Detector, ...] = (
    Detector(
        "airway.hear_cough_labels.plain", "airway", ("peak_set", "plain", "hear", "cough_labels"), "score", SCORE_GRID
    ),
    Detector(
        "airway.amplitude_peak_over_floor_db_max",
        "airway",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector("airway.amplitude_rate_per_s", "airway", ("span_stat", "amplitude.rate_per_s"), "spans/s", RATE_GRID),
    Detector("airway.zero_crossing_rate", "airway", ("disruptions", "zero_crossing_rate"), "crossings/s", ZCR_GRID),
    Detector("airway.squim_stoi_median", "airway", ("squim", "all.stoi.median"), "stoi", STOI_GRID, "below"),
    Detector("airway.squim_si_sdr_iqr", "airway", ("squim", "all.si_sdr.iqr"), "dB", DB_SPREAD_GRID),
    Detector("airway.silence_fraction", "airway", ("silence", "fraction"), "fraction", FRACTION_GRID),
    Detector("airway.residual_band_0_200", "airway", ("residual", "band_0_200"), "fraction", FRACTION_GRID),
    Detector("airway.residual_band_4000_8000", "airway", ("residual", "band_4000_8000"), "fraction", FRACTION_GRID),
    Detector(
        "airway.residual_correlation_residual", "airway", ("residual", "correlation_residual"), "r", FRACTION_GRID
    ),
    Detector("speech.squim_stoi_median", "speech", ("squim", "all.stoi.median"), "stoi", STOI_GRID),
    Detector("speech.squim_pesq_median", "speech", ("squim", "all.pesq.median"), "pesq", PESQ_GRID),
    Detector("speech.squim_si_sdr_median", "speech", ("squim", "all.si_sdr.median"), "dB", SI_SDR_GRID),
    Detector("speech.silence_fraction", "speech", ("silence", "fraction"), "fraction", FRACTION_GRID, "below"),
    Detector("speech.amplitude_rate_per_s", "speech", ("span_stat", "amplitude.rate_per_s"), "spans/s", RATE_GRID),
    Detector("speech.level_lufs", "speech", ("level", "lufs"), "dBFS", DBFS_GRID),
    Detector("voice.squim_stoi_max", "voice", ("squim", "all.stoi.max"), "stoi", STOI_GRID),
    Detector("voice.squim_amplitude_pesq_max", "voice", ("squim", "amplitude.pesq.max"), "pesq", PESQ_GRID),
    Detector(
        "voice.amplitude_duty_fraction", "voice", ("span_stat", "amplitude.duty_fraction"), "fraction", FRACTION_GRID
    ),
    Detector("voice.silence_fraction", "voice", ("silence", "fraction"), "fraction", FRACTION_GRID, "below"),
    Detector("voice.level_lufs", "voice", ("level", "lufs"), "dBFS", DBFS_GRID),
)
"""Detectors reading a derivative the first sweep ignored, on the three kinds it already covered."""

_PITCH_SWEEP: tuple[Detector, ...] = (
    Detector("glide.praat_std_f0_hertz", "glide", ("praat", "std_f0_hertz"), "Hz", F0_SPREAD_HZ_GRID),
    Detector(
        "glide.praat_f0_relative_spread",
        "glide",
        ("ratio", ("praat", "std_f0_hertz"), ("praat", "mean_f0_hertz")),
        "ratio",
        RELATIVE_SPREAD_GRID,
    ),
    Detector(
        "glide.praat_std_f0_hertz+no_agreed_word",
        "glide",
        ("gated", ("praat", "std_f0_hertz"), ("words", "agreement"), 1, "below"),
        "Hz",
        F0_SPREAD_HZ_GRID,
    ),
    Detector("glide.praat_phonation_ratio", "glide", ("praat", "phonation_ratio"), "fraction", PROPORTION_GRID),
    Detector("voice.praat_phonation_ratio", "voice", ("praat", "phonation_ratio"), "fraction", PROPORTION_GRID),
    Detector("voice.praat_mean_hnr_db", "voice", ("praat", "mean_hnr_db"), "dB", VOICE_QUALITY_DB_GRID),
    Detector(
        "voice.praat_cepstral_peak_prominence_mean",
        "voice",
        ("praat", "cepstral_peak_prominence_mean"),
        "dB",
        VOICE_QUALITY_DB_GRID,
    ),
)
"""Pitch spread and phonation read off Praat, neither of which needs a span to reach 3 s."""

_SYLLABLE_REPETITION: tuple[Detector, ...] = (
    Detector("ddk.praat_articulation_rate", "ddk", ("praat", "articulation_rate"), "syllables/s", SYLLABLE_RATE_GRID),
    Detector("ddk.praat_speaking_rate", "ddk", ("praat", "speaking_rate"), "syllables/s", SYLLABLE_RATE_GRID),
    Detector("ddk.ppg_segment_rate_per_s", "ddk", ("ppg", "segment_rate_per_s"), "segments/s", SEGMENT_RATE_GRID),
    Detector("ddk.ppg_repetition_peak", "ddk", ("ppg", "repetition_peak"), "fraction", PROPORTION_GRID),
    Detector("ddk.ppg_repetition_prominence", "ddk", ("ppg", "repetition_prominence"), "fraction", PROPORTION_GRID),
    Detector(
        "ddk.ppg_repetition_lag_segments", "ddk", ("ppg", "repetition_lag_segments"), "segments", COUNT_GRID, "below"
    ),
    Detector(
        "ddk.ppg_segment_duration_median",
        "ddk",
        ("ppg", "segment_duration_median"),
        "seconds",
        SEGMENT_DURATION_GRID,
        "below",
    ),
    Detector("ddk.ppg_distinct_phonemes", "ddk", ("ppg", "distinct_phonemes"), "phonemes", COUNT_GRID, "below"),
)
"""Candidate detectors for diadochokinesis, which is a rate and a repetition rather than a word."""

_UNVOICED_AIRWAY: tuple[Detector, ...] = (
    Detector(
        "airway.praat_phonation_ratio", "airway", ("praat", "phonation_ratio"), "fraction", PROPORTION_GRID, "below"
    ),
    Detector("airway.praat_pause_rate", "airway", ("praat", "pause_rate"), "pauses/s", PAUSE_RATE_GRID),
    Detector("airway.praat_mean_pause_duration", "airway", ("praat", "mean_pause_duration"), "seconds", DURATION_GRID),
    Detector("airway.praat_mean_hnr_db", "airway", ("praat", "mean_hnr_db"), "dB", VOICE_QUALITY_DB_GRID, "below"),
    Detector("airway.ppg_silent_fraction", "airway", ("ppg", "silent_fraction"), "fraction", PROPORTION_GRID),
)
"""Candidate detectors for a breath, which is unvoiced and need not accumulate energy to be read."""

_BRACKETED_TOKENS: tuple[Detector, ...] = (
    Detector("airway.bracketed_breath", "airway", ("bracketed_set", "breath"), "tokens", COUNT_GRID),
    Detector("airway.bracketed_cough", "airway", ("bracketed_set", "cough"), "tokens", COUNT_GRID),
    Detector("airway.bracketed_throatclearing", "airway", ("bracketed_set", "throatclearing"), "tokens", COUNT_GRID),
    Detector("airway.bracketed_sniff", "airway", ("bracketed_set", "sniff"), "tokens", COUNT_GRID),
    Detector("airway.bracketed_laughter", "airway", ("bracketed_set", "laughter"), "tokens", COUNT_GRID),
    Detector(
        "airway.bracketed_airway_union",
        "airway",
        ("bracketed_set", "breath", "cough", "throatclearing", "sniff"),
        "tokens",
        COUNT_GRID,
    ),
    Detector("speech.bracketed_uh", "speech", ("bracketed_set", "uh"), "tokens", COUNT_GRID),
    Detector("speech.bracketed_um", "speech", ("bracketed_set", "um"), "tokens", COUNT_GRID),
    Detector("speech.bracketed_filler_union", "speech", ("bracketed_set", "uh", "um"), "tokens", COUNT_GRID),
)
"""Typed bracketed consensus tokens, so the AIRWAY bracket gate can be swept rather than assumed."""

_STREAM_PEAKS: tuple[Detector, ...] = (
    Detector(
        "speech.enhanced_yamnet_peak_max", "speech", ("stream_peak_max", "enhanced", "yamnet"), "score", SCORE_GRID
    ),
    Detector(
        "speech.residual_yamnet_peak_max", "speech", ("stream_peak_max", "residual", "yamnet"), "score", SCORE_GRID
    ),
)
"""The highest tracked-label score a whole stream carries, which is what an empty recording lacks."""

_LABEL_CONDITIONED_SPANS: tuple[Detector, ...] = (
    Detector(
        "cough.yamnet_cough_span_peak_over_floor_db_max",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_max"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector(
        "cough.yamnet_cough_span_peak_over_floor_db_p75",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_p75"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
    Detector(
        "cough.yamnet_cough_span_peak_over_floor_db_p90",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_p90"),
        "dB",
        DB_OVER_FLOOR_GRID,
    ),
)
"""Span amplitude read only over the spans YAMNet itself labelled ``Cough``."""

_COUGH_SET = "cough_labels"
"""The :data:`~senselab.audio.workflows.triage.routing_analysis.labels.LABEL_SETS` union read below."""

_LABEL_SET_CONDITIONED_SPANS: tuple[Detector, ...] = tuple(
    Detector(
        f"cough.yamnet_cough_set_span_peak_over_floor_db_{statistic}",
        "cough",
        ("span_label_set_stat", f"yamnet.{_COUGH_SET}.peak_over_floor_db_{statistic}"),
        "dB",
        DB_OVER_FLOOR_GRID,
    )
    for statistic in ("max", "p75", "p90")
)
"""Span amplitude over the spans carrying any cough-set label, not the single ``Cough`` string."""


DETECTORS: tuple[Detector, ...] = tuple(
    [
        Detector("speech.words_agreement", "speech", ("words", "agreement"), "words", COUNT_GRID),
        Detector("speech.words_agreement_lexical", "speech", ("words", "agreement_lexical"), "words", COUNT_GRID),
        Detector("speech.words_lexical", "speech", ("words", "lexical"), "words", COUNT_GRID),
        Detector("speech.words_total", "speech", ("words", "total"), "words", COUNT_GRID),
        Detector(
            "speech.residual_speech_coverage",
            "speech",
            ("residual", "speech_coverage_fraction"),
            "fraction",
            FRACTION_GRID,
        ),
        Detector(
            "airway.residual_energy_fraction", "airway", ("residual", "energy_fraction"), "fraction", FRACTION_GRID
        ),
        Detector(
            "airway.residual_enhanced_energy_fraction",
            "airway",
            ("residual", "enhanced_energy_fraction"),
            "fraction",
            FRACTION_GRID,
        ),
        Detector("voice.longest_amplitude_span", "voice", ("span_longest", "amplitude"), "seconds", DURATION_GRID),
        Detector("voice.longest_continuity_span", "voice", ("span_longest", "continuity"), "seconds", DURATION_GRID),
        Detector("voice.total_amplitude_span", "voice", ("span_total", "amplitude"), "seconds", DURATION_GRID),
        Detector(
            "airway.residual_energy_fraction+hear>=0.2",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("peak", "plain", "hear", "airway"), 0.2, "above"),
            "fraction",
            FRACTION_GRID,
        ),
        Detector(
            "airway.residual_energy_fraction+hear>=0.5",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("peak", "plain", "hear", "airway"), 0.5, "above"),
            "fraction",
            FRACTION_GRID,
        ),
        Detector(
            "airway.yamnet_peak.plain+hear>=0.2",
            "airway",
            ("gated", ("peak", "plain", "yamnet", "airway"), ("peak", "plain", "hear", "airway"), 0.2, "above"),
            "score",
            SCORE_GRID,
        ),
        Detector(
            "airway.residual_energy_fraction+no_agreed_word",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("words", "agreement"), 1, "below"),
            "fraction",
            FRACTION_GRID,
        ),
        Detector(
            "voice.longest_amplitude_span+no_agreed_word",
            "voice",
            ("gated", ("span_longest", "amplitude"), ("words", "agreement"), 1, "below"),
            "seconds",
            DURATION_GRID,
        ),
        Detector(
            "voice.yamnet_peak.plain+no_agreed_word",
            "voice",
            ("gated", ("peak", "plain", "yamnet", "voice"), ("words", "agreement"), 1, "below"),
            "score",
            SCORE_GRID,
        ),
        Detector(
            "voice.yamnet_peak.plain+amplitude>=3s",
            "voice",
            ("gated", ("peak", "plain", "yamnet", "voice"), ("span_longest", "amplitude"), 3.0, "above"),
            "score",
            SCORE_GRID,
        ),
        Detector(
            "speech.words_agreement+ast>=0.5",
            "speech",
            ("gated", ("words", "agreement"), ("peak", "plain", "ast", "speech"), 0.5, "above"),
            "words",
            COUNT_GRID,
        ),
        Detector(
            "voice.yamnet_chant_peak.plain",
            "voice",
            ("peak_label", "plain", "yamnet", "Chant"),
            "score",
            SCORE_GRID,
        ),
        Detector(
            "voice.yamnet_mantra_peak.plain",
            "voice",
            ("peak_label", "plain", "yamnet", "Mantra"),
            "score",
            SCORE_GRID,
        ),
    ]
    + _peak_detectors()
    + _singing_detectors()
    + list(_COUGH_VS_BREATH)
    + list(_GLIDES)
    + list(_NEW_DERIVATIVES)
    + list(_LABEL_CONDITIONED_SPANS)
    + list(_LABEL_SET_CONDITIONED_SPANS)
    + list(_PITCH_SWEEP)
    + list(_SYLLABLE_REPETITION)
    + list(_UNVOICED_AIRWAY)
    + list(_BRACKETED_TOKENS)
    + list(_STREAM_PEAKS)
)
"""Every candidate detector, scored at every threshold in its own grid."""
