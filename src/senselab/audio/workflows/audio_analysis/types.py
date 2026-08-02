"""Typed dataclasses for the audio_analysis workflow.

They live as plain dataclasses (not Pydantic) because they are workflow-internal — the parquet
writer serializes them via pyarrow, not via Pydantic JSON, and we want zero overhead for the hot
per-bucket loop.

**An uncertainty axis is an aggregator.** It aggregates across signals *and* across passes, so
there is no such thing as a per-pass axis: a pass is an input dimension to the fold, never an index
on its output. That is why there is no ``(pass, axis)`` product type here — L1 emits
:class:`SignalResult` (per pass, per signal, no axis) and L2 emits :class:`FusedAxis` (per axis, no
pass). The pass dimension appears on a fused row only as the ``contributing_passes`` column.
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

PassLabel = Literal["raw_16k", "enhanced_16k"]
"""A pass is the same recording under a transform — as recorded, or after speech enhancement.

``raw_vs_enhanced`` used to be a member. It is not a pass; it was a perturbation-stability
*measurement* wearing a pass label so that it could be an index on an axis. Stability is now keyed
by signal (``L1/stability/<signal>.parquet``), which is what it is a property of.
"""

ComparisonStatus = Literal["ok", "incomparable", "unavailable"]
"""Per-row status: did this signal produce a comparable measurement in this bucket?"""


@dataclass(slots=True)
class SignalRow:
    """One signal's measurement in one bucket of one pass — the level-1 emission.

    No axis and no fold. ``measurement`` holds what the tool reported, in the tool's own units,
    exactly as harvested; ``units``/``model_id``/``revision``/``native_window_s``/``resolution_s``
    are the provenance a different lab would need to reproduce the number from the audio alone.

    A signal that said nothing in a bucket has no row here rather than a zero-filled one: zero is a
    confident claim, and imputing it would manufacture confidence nobody expressed.
    """

    start: float
    end: float
    signal: str
    measurement: dict[str, Any] = field(default_factory=dict)
    units: str | None = None
    native_window_s: float | None = None
    resolution_s: float | None = None
    model_id: str | None = None
    revision: str | None = None
    status: ComparisonStatus = "ok"


@dataclass(slots=True)
class SignalResult:
    """All L1 rows for one ``(pass, signal)`` plus the provenance recorded on the parquet.

    Held in memory by ``compute_uncertainty_axes``; serialized by ``io.write_signal_parquet`` to
    ``L1/<pass>/signals/<signal>.parquet``.
    """

    pass_label: PassLabel
    signal: str
    rows: list[SignalRow] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusedAxis:
    """One axis, fused across every signal and every pass — the level-2 product.

    Deliberately has no ``pass_label``. Each row is a plain dict as emitted by
    :func:`~senselab.audio.workflows.audio_analysis.fuse.fuse_axis`: ``uncertainty``,
    ``epistemic_uncertainty``, ``confidence``, ``variability``, ``triage_score``,
    ``contributing_signals``, ``contributing_passes``, ``signal_weights``, ``weight_basis``,
    ``round``.
    """

    axis: UncertaintyAxis
    rows: list[dict[str, Any]] = field(default_factory=list)
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
