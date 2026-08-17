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

__all__ = ["ComparisonStatus", "FusedAxis", "PerSegmentEmbedding", "SignalResult", "SignalRow", "UncertaintyAxis"]

UncertaintyAxis = str
"""An axis is a plain ``str``, not a ``Literal``.

The set is **open**: ``task`` is declared-but-punted, a fifth may follow, and a type that
enumerates the members is a promise the pipeline is not allowed to keep. This alias *was* a
three-member ``Literal``, justified as "narrower than the set L2 fuses", and that narrowing is
precisely what made ``background_mask`` unrepresentable in every consumer that needed to act on it.
What an axis *is* lives in ``axes.AXES``, where the properties travel with the name; a caller that
wants only the harvested ones asks ``axes.HARVESTED_AXES`` rather than narrowing a type.

Declared here rather than re-exported from ``axes``, because ``str`` is the whole content of the
alias and importing it bought nothing but an edge: this module is reachable from the extraction
layer, which must not depend on the axis vocabulary that consumes its output. That the two
declarations cannot drift is checked at the source level in ``axes_test.py`` — an equality
assertion could not fail, since both sides are ``str``.
"""

# A perturbation is a plain ``str``, not a Literal. The set is **open** — raw is the identity,
# enhancement is one more, and a future L2 round may propose another — so anything that enumerates
# it in a type is a promise the pipeline is not allowed to keep. What each name means is declared
# in ``L1/perturbations.json`` (see ``perturbations.Perturbation``), where the transform and its
# parameters travel with it instead of being inferred from the spelling.

ComparisonStatus = Literal["ok", "incomparable", "unavailable"]
"""Per-row status: did this signal produce a comparable measurement in this bucket?"""


@dataclass(slots=True)
class SignalRow:
    """One signal's measurement in one bucket of one pass — the level-1 emission.

    No axis and no fold. ``measurement`` holds what the tool reported, in the tool's own units,
    exactly as harvested; ``units``/``model_id``/``native_window_s``/``resolution_s``
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
    status: ComparisonStatus = "ok"


@dataclass(slots=True)
class SignalResult:
    """All L1 rows for one ``(perturbation, signal)`` plus the provenance recorded on the parquet.

    Held in memory by ``compute_uncertainty_axes``; serialized by ``io.write_signal_parquet``,
    which writes every perturbation's rows for one signal into a single
    ``L1/signals/<signal>.parquet`` with ``perturbation`` as a column. The perturbation is a
    dimension of the measurement, not of its location.
    """

    perturbation: str
    signal: str
    rows: list[SignalRow] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FusedAxis:
    """One axis, fused across every signal and every pass — the level-2 product.

    Deliberately has no ``perturbation``. Each row is a plain dict as emitted by
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
