"""The candidate routing detectors and the thresholds each is swept over.

A detector reads one number out of a :class:`~senselab.audio.workflows.triage.routing_analysis.
features.RecordingFeatures` and fires when that number is at or above a threshold. No threshold is
preferred here: every one in a detector's grid is scored, and
``taxonomy.consolidation_floor`` (0.2) is marked where it falls rather than adopted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.labels import FAMILIES, peak_key

GATE_CLOSED = -1.0
"""The value a gated detector reads when its corroborator did not fire: below every threshold."""

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


@dataclass(frozen=True)
class Detector:
    """One candidate rule TAXONOMY could route on.

    Attributes:
        name: The detector's id, unique across the catalogue.
        kind: ``speech``, ``airway`` or ``voice`` — the branch it would route to.
        reader: What to read, as ``(source, *arguments)``; see :func:`detector_value`. A
            ``gated`` reader nests two of these.
        unit: The unit of the number it reads.
        thresholds: Every threshold it is scored at.
    """

    name: str
    kind: str
    reader: tuple[Any, ...]
    unit: str
    thresholds: tuple[float, ...]


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
            return GATE_CLOSED
        return detector_value(features, Detector(detector.name, detector.kind, primary, detector.unit, ()))
    if source == "peak_label":
        stream, classifier, label = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        return float(features.peaks.get(peak_key(stream, classifier, label), 0.0))
    raise ValueError(f"unknown detector source {source!r}")


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
)
"""Every candidate detector, scored at every threshold in its own grid."""
