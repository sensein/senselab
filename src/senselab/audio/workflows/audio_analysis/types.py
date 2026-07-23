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

    # ── Scene-aware presence extensions (feature 20260722-175022) ────────────
    # All default None and are populated only on the axis they belong to
    # (presence for the confidence/quality/source columns; utterance for the
    # token-entropy/coupling columns). They are additive: existing readers that
    # project a fixed column set are unaffected, and ``aggregated_uncertainty``
    # keeps its original meaning. See
    # ``specs/20260722-175022-scene-quality-utterance/data-model.md``.
    #
    # Presence confidence/uncertainty split (FR-013):
    presence_confidence: float | None = None  # calibrated mean P(speech) in [0,1]
    presence_uncertainty: float | None = None  # decisiveness uncertainty 1-|2p-1| in [0,1]
    # Audio-quality degradation scores, 0 = clean, 1 = fully degraded (FR-001):
    quality_snr: float | None = None
    quality_clip: float | None = None
    quality_reverb: float | None = None
    quality_bandwidth: float | None = None
    quality_uncertainty: float | None = None  # spread among SNR estimators (FR-005)
    # Background sound-source category masses, sum ~1 when present (FR-007):
    src_speech: float | None = None
    src_people: float | None = None
    src_machine: float | None = None
    src_environment: float | None = None
    src_dominant: str | None = None  # argmax category name
    # Utterance extensions (FR-017, FR-019):
    token_entropy: float | None = None  # mean per-token ASR entropy over the bucket
    scene_quality_coupling: float | None = None  # recorded coupling multiplier (>=1.0)


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
