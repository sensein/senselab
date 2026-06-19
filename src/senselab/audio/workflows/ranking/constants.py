"""Documented constants and thresholds for the ranking workflow.

Surfaced here (not buried as magic numbers) so the operating points are
auditable and adjustable. See research.md D2/D4/D5 and the spec Clarifications.
"""

from __future__ import annotations

SCHEMA_VERSION: int = 1
"""Bumped on any breaking change to a persisted artifact shape."""

DEFAULT_BAND_FRACTION: float = 0.20
"""Top/bottom band fraction — a coarse lens, configurable (spec Clarifications 2026-06-04)."""

ORDINAL_QUALITY_MAP: dict[str, float] = {"good": 2.0, "acceptable": 1.0, "poor": 0.0}
"""Ordinal label → numeric value for rank-correlation and band separation (research D1)."""

QUALITY_LABELS: tuple[str, ...] = ("good", "acceptable", "poor")

DEFAULT_SEPARATION_TARGET: float = 0.80
"""SC-001 target for top-vs-bottom-band pairwise agreement."""

MIN_BAND_ITEMS: int = 1
"""Fewer scored items than this per band ⇒ separation not evaluable."""

MIN_ANNOTATED_PER_BAND: int = 2
"""Fewer annotated items than this in either band ⇒ band separation not evaluable."""

MIN_ANNOTATIONS_RECAL: int = 10
"""Fewer active annotations than this ⇒ assisted recalibration refuses (FR-017)."""

MIN_QUALITY_LEVELS_RECAL: int = 2
"""Fewer distinct quality levels than this ⇒ no orderable pairs ⇒ recalibration refuses."""

LOW_PAIR_WARN: int = 30
"""At or below this many training pairs, recalibration proceeds but warns (overfit risk)."""

TIE_BREAK: str = "score_desc,item_id_asc"
"""Deterministic order: by score (per direction), ties broken by item_id ascending (SC-003)."""
