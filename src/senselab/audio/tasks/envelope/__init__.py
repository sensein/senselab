"""Amplitude envelope, global floor, and dynamic-range normalization built on both."""

from senselab.audio.tasks.envelope.api import (
    AnalyticEnvelope,
    ButterworthSmoothing,
    EnvelopeSmoothing,
    MedianSmoothing,
    PercentileSmoothing,
    analytic_envelope,
    analytic_magnitude,
    dynamic_range_normalize,
    envelope_dbfs,
    global_floor_dbfs,
    hilbert_envelope_dbfs,
)

__all__ = [
    "AnalyticEnvelope",
    "ButterworthSmoothing",
    "EnvelopeSmoothing",
    "MedianSmoothing",
    "PercentileSmoothing",
    "analytic_envelope",
    "analytic_magnitude",
    "dynamic_range_normalize",
    "envelope_dbfs",
    "global_floor_dbfs",
    "hilbert_envelope_dbfs",
]
