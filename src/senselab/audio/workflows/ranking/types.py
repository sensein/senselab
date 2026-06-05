"""Typed dataclasses for the ranking workflow.

These match the data model in
``specs/20260604-173646-iterative-metric-ranking/data-model.md``.

Following the audio_analysis / speaker_profile convention, these are plain
``@dataclass(slots=True)`` classes (not Pydantic) because they are
workflow-internal — the parquet/JSON writers in ``io.py`` serialize them
explicitly, not via Pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Enumerations / Literals ────────────────────────────────────────────────

RankUnit = Literal["file", "segment"]
"""Ranking unit — chosen per run; file-level and segment-level are distinct spaces."""

QualityLabel = Literal["good", "acceptable", "poor"]
"""Default fixed ordinal annotation scale."""

ScoreStatus = Literal["scored", "unscorable"]
Band = Literal["top", "middle", "bottom"]
SignalTransform = Literal["identity", "zscore", "minmax", "rank", "clip", "threshold"]
Direction = Literal["higher_is_better", "lower_is_better"]
MetricOrigin = Literal["initial", "manual", "recalibrated"]
RecalStatus = Literal["proposed", "refused", "warned"]
ItemDelta = Literal["moved", "unchanged", "added", "removed", "became_unscorable"]
CutKind = Literal["rank", "percentile"]


# ── Metric definition ──────────────────────────────────────────────────────


@dataclass(slots=True)
class SignalTerm:
    """One signal's contribution to a metric (see ``metric-definition.schema.md``)."""

    signal: str
    weight: float
    transform: SignalTransform = "identity"
    transform_params: dict[str, Any] = field(default_factory=dict)
    missing: str = "unscorable"  # "unscorable" | "neutral" | "fill:<float>"


@dataclass(slots=True)
class MetricDefinition:
    """Declarative, serializable combination of signals into one comparable score."""

    name: str
    terms: list[SignalTerm]
    direction: Direction = "higher_is_better"
    combine: Literal["weighted_sum"] = "weighted_sum"
    notes: str = ""


@dataclass(slots=True)
class RecalibrationResult:
    """Outcome of an assisted-recalibration attempt (advisory; never auto-adopted)."""

    status: RecalStatus
    proposed_definition: MetricDefinition | None
    n_annotations_used: int
    n_pairs: int
    n_distinct_levels: int
    agreement_before: float | None
    agreement_after: float | None
    message: str = ""


@dataclass(slots=True)
class MetricVersion:
    """Immutable snapshot of a metric definition plus provenance."""

    version_id: str
    definition: MetricDefinition
    origin: MetricOrigin
    parent_version_id: str | None
    created_at: str
    recal: RecalibrationResult | None = None


# ── Signal table (input) ───────────────────────────────────────────────────


@dataclass(slots=True)
class SignalTable:
    """Per-item signal table (see ``signal-table.parquet.md``).

    ``columns`` maps a signal name to a 1-D float array aligned with ``item_ids``;
    ``NaN`` marks a missing signal. ``locators`` (segment unit only) maps
    ``item_id`` → ``(source_audio, start, end)``.
    """

    unit: RankUnit
    item_ids: list[str]
    columns: dict[str, Any]  # signal -> numpy float array (NaN = missing)
    signal_columns: list[str]
    locators: dict[str, tuple[str, float, float]] = field(default_factory=dict)


# ── Ranking ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class RankingItem:
    """One row of a ranking."""

    item_id: str
    score: float | None
    rank: int | None
    percentile: float | None
    band: Band | None
    status: ScoreStatus
    reason: str | None = None


@dataclass(slots=True)
class Ranking:
    """Ordered result for one metric version (see ``ranking.parquet.md``)."""

    version_id: str
    unit: RankUnit
    band_fraction: float
    items: list[RankingItem]
    n_scored: int
    n_unscorable: int
    provenance: dict[str, Any] = field(default_factory=dict)


# ── Annotation ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Annotation:
    """A reviewer's ground-truth quality judgment for one item."""

    item_id: str
    label: QualityLabel | None
    score: float | None
    unit: RankUnit
    reviewed_under_version: str | None = None
    reviewer: str | None = None
    created_at: str = ""
    note: str = ""
    resolution: Literal["active", "superseded"] = "active"


# ── Evaluation / triage ────────────────────────────────────────────────────


@dataclass(slots=True)
class SeparationResult:
    """Outcome of the ranking-quality check against available annotations."""

    version_id: str
    rank_agreement_spearman: float | None
    rank_agreement_kendall_tau_b: float | None
    band_pairwise_agreement: float | None
    band_quality_margin: float | None
    n_annotated: int
    n_annotated_top: int
    n_annotated_bottom: int
    evaluable: bool
    reason: str | None = None
    meets_separation_target: bool | None = None


@dataclass(slots=True)
class TriageThreshold:
    """A release-vs-review cut point with its annotation readout.

    Unscorable items are auto-fail: never auto-accepted, always routed to human
    review (counted in ``n_unscorable_routed`` and ``n_human_review``). See FR-010b.
    """

    version_id: str
    cut: float
    cut_kind: CutKind
    n_auto_accept: int
    n_human_review: int
    n_unscorable_routed: int
    above_counts: dict[str, int]
    below_counts: dict[str, int]
    auto_accept_poor_rate: float | None = None


# ── Movement ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MovementEntry:
    """Per-item movement between two rankings."""

    item_id: str
    from_rank: int | None
    to_rank: int | None
    position_delta: int | None
    percentile_delta: float | None
    from_band: Band | None
    to_band: Band | None
    delta_kind: ItemDelta
    annotated: bool = False
    annotation_label: QualityLabel | None = None


@dataclass(slots=True)
class MovementReport:
    """Comparison of two rankings over the same corpus + unit."""

    from_version: str
    to_version: str
    unit: RankUnit
    band_fraction: float
    entries: list[MovementEntry]
    band_summary: dict[str, int]
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    became_unscorable: list[str] = field(default_factory=list)
