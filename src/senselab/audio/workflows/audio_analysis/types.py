"""Typed dataclasses for the audio_analysis workflow.

These match the public schema in ``contracts/uncertainty-row.parquet.md``. They live as
plain dataclasses (not Pydantic) because they are workflow-internal — the parquet writer
serializes them via pyarrow, not via Pydantic JSON, and we want zero overhead for the
hot per-bucket aggregation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UncertaintyAxis = Literal["presence", "identity", "utterance"]
"""Three uncertainty axes — see FR-001 / FR-002."""

PassLabel = Literal["raw_16k", "enhanced_16k", "raw_vs_enhanced"]
"""Pass identifier used for parquet pathing and the disagreements.json `pass` field."""

ComparisonStatus = Literal["ok", "incomparable", "unavailable", "one_sided"]
"""Per-row status. ``one_sided`` only appears on raw_vs_enhanced parquets."""


@dataclass(slots=True)
class UncertaintyRow:
    """One bucket on one (pass, axis) uncertainty parquet.

    See ``contracts/uncertainty-row.parquet.md`` for the column-level schema.
    """

    start: float
    end: float
    axis: UncertaintyAxis
    aggregated_uncertainty: float | None
    contributing_models: list[str]
    model_votes: dict[str, dict[str, Any]]
    comparison_status: ComparisonStatus = "ok"
    # Audio-intensity weight in [0, 1]. Derived from per-bucket loudness
    # (per-pass percentile-normalized openSMILE Loudness_sma3). Used to
    # downweight uncertainty contributions from silent / background buckets
    # so they don't artificially inflate the time-aggregated mean. The raw
    # uncertainty is also stored multiplied by this weight in
    # ``aggregated_uncertainty`` (see compute.py); ``intensity_weight`` here
    # carries the unmasked weight for downstream re-weighting if desired.
    intensity_weight: float | None = None
    raw_aggregated_uncertainty: float | None = None  # pre-mask value


@dataclass(slots=True)
class AxisResult:
    """All rows for one (pass, axis) plus the provenance recorded on the parquet.

    Held in memory by ``compute_uncertainty_axes``; serialized to disk by
    ``write_axis_parquet``.
    """

    pass_label: PassLabel
    axis: UncertaintyAxis
    rows: list[UncertaintyRow] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PerSegmentEmbedding:
    """One speaker-embedding vector for one diarization segment.

    Used by the identity axis's across-time sub-signal: per-bucket cosine distance is
    computed against the embedding of the most recent prior bucket on the same speaker
    track.
    """

    seg_start: float
    seg_end: float
    speaker_label: str
    model_id: str
    vector: list[float]
