"""Typed dataclasses for the audio_analysis workflow.

These match the public schema in ``contracts/uncertainty-row.parquet.md``. They live as
plain dataclasses (not Pydantic) because they are workflow-internal — the parquet writer
serializes them via pyarrow, not via Pydantic JSON, and we want zero overhead for the
hot per-bucket aggregation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UncertaintyAxis = Literal["speech_presence", "speaker", "asr"]
"""The axes with a *vote harvest* — see FR-001 / FR-002.

Narrower than the axis set L2 fuses, deliberately. ``background_mask`` is a fourth axis at fusion
(its votes come from the mask's own per-region confidence, not from an ensemble) and ``task`` is a
punted fifth, but neither is harvested, so neither belongs in the type that describes what harvest
produces. Widening it here would promise `compute.py` inputs that no harvester emits.
"""

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
    within_pass_uncertainty: float | None
    """What this pass alone would conclude, before anything is measured about the signals.

    A **level-1 diagnostic**, not the answer. The answer is level 2's
    ``final/uncertainty/<axis>.parquet``, which fuses across signals and passes with measured
    weights. Named so it cannot be mistaken for a verdict — under its previous name,
    ``within_pass_uncertainty``, every consumer read it as one, which is how a fold computed
    before any weighting came to be reported as the run's belief."""

    contributing_models: list[str]
    model_votes: dict[str, dict[str, Any]]
    comparison_status: ComparisonStatus = "ok"
    signal_uncertainty: dict[str, float] = field(default_factory=dict)
    """Each signal's *own* uncertainty in this bucket — the level-1 emission.

    First-class rather than recoverable only by re-parsing ``model_votes``: level 2 has to
    weight the signals, and a value already folded cannot be re-weighted. A signal that said
    nothing is absent rather than zero-filled, since zero is a confident claim."""

    # Audio-intensity weight in [0, 1]. Derived from per-bucket loudness
    # (per-pass percentile-normalized openSMILE Loudness_sma3). Used to
    # downweight uncertainty contributions from silent / background buckets
    # so they don't artificially inflate the time-aggregated mean. The raw
    # uncertainty is also stored multiplied by this weight in
    # ``within_pass_uncertainty`` (see compute.py); ``intensity_weight`` here
    # carries the unmasked weight for downstream re-weighting if desired.
    intensity_weight: float | None = None
    raw_within_pass_uncertainty: float | None = None  # pre-mask value

    # ── Scene-aware speech_presence extensions (feature 20260722-175022) ────────────
    # All default None and are populated only on the axis they belong to
    # (speech_presence for the confidence/quality/source columns; asr for the
    # token-entropy/coupling columns). They are additive: existing readers that
    # project a fixed column set are unaffected, and ``within_pass_uncertainty``
    # keeps its original meaning. See
    # ``specs/20260722-175022-scene-quality-asr/data-model.md``.
    #
    # Presence confidence/uncertainty split (FR-013):
    speech_presence_confidence: float | None = None  # calibrated mean P(speech) in [0,1]
    speech_presence_uncertainty: float | None = None  # decisiveness uncertainty 1-|2p-1| in [0,1]
    # L1 scene-quality measurements, in native units. Recorded alongside the derived scores so a
    # consumer can always see what was measured, not only how it was scored — the previous version
    # kept only the scores, which is why a column pinned at 0.0 by its anchor was
    # indistinguishable from one measured at 0.0.
    snr_brouhaha_db: float | None = None
    c50_brouhaha_db: float | None = None
    snr_spectral_gating_db: float | None = None
    snr_peak_db: float | None = None
    rolloff_95_hz: float | None = None
    proportion_clipped: float | None = None
    # L2 degradation scores derived from the above against calibrated anchors, 0 = clean,
    # 1 = fully degraded (FR-001). See ``degradation.scene_degradation``.
    quality_snr: float | None = None
    quality_clip: float | None = None
    quality_reverb: float | None = None
    quality_bandwidth: float | None = None
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

    Used by the speaker axis's across-time sub-signal: per-bucket cosine distance is
    computed against the embedding of the most recent prior bucket on the same speaker
    track.
    """

    seg_start: float
    seg_end: float
    speaker_label: str
    model_id: str
    vector: list[float]
