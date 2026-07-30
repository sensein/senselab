"""Background source detection: margin ladder and fabrication guards (T060-T062).

A candidate becomes a reported background source only by clearing a **margin above its own
band's noise floor**. Never by being amplified: gain moves a source and the residual
foreground together and changes no signal-to-noise ratio, so it cannot promote a tier.

The 3 / 6 / 10 dB ladder is corroborated from three independent directions — human masked
threshold and audibility criteria, a dozen unrelated detection traditions in bioacoustics
and noise standards, and the classifiers' own measured reliable-detection floors. That
convergence is the reason to trust the values; none of them was fitted here.

The guards exist because the failure mode is not a missed source, it is a **fabricated**
one. Amplifying a noise floor produces confident, plausible environmental labels —
waterfall, water, gurgling, static — that are statistically indistinguishable from genuine
broadband noise and read as real findings. Three layers stop that:

- a **pre-gain level reject**, because a segment below the classifiers' measured trust floor
  should not be amplified and interpreted at all;
- a **noise-character test**, because broadband noise separates cleanly from structured
  sources on spectral flatness for the cost of one transform;
- a **quarantine list** for the labels amplified noise actually produces, which may only be
  reported when the noise-character test passes.

Plus a **floor-response signature** check: a classifier below its floor can emit a fixed
label pattern, and one measured signature pairs a silence label at 0.44 with a co-occurring
label at 0.35 — so keying on the silence label alone would let the second one through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

__all__ = [
    "TIERS",
    "ExcisedSegment",
    "plan_excision",
    "route_classifier",
    "SourceFinding",
    "assign_tier",
    "is_noise_like",
    "passes_pregain_gate",
    "spectral_flatness",
]

Tier = Literal["rejected", "candidate", "probable", "confident"]

TIERS: tuple[Tier, ...] = ("rejected", "candidate", "probable", "confident")


@dataclass(frozen=True)
class SourceFinding:
    """One reported background source occurrence."""

    start: float
    end: float
    category: str
    label: str
    classifier: str
    above_floor_db: float
    tier: Tier
    binding_floor: str
    variant: str
    gain_db: float
    computed_on: Literal["grid", "excised"]
    flatness: float
    occupancy: float
    padding_fraction: float | None = None
    from_mask_region: str | None = None
    mask_confidence: float | None = None
    leakage_margin_db: float | None = None
    suppression_depth_db: float | None = None
    modulation_depth: float | None = None
    stationary_pass: bool = False
    discounted_reason: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Row for ``background_sources.parquet``."""
        return {
            "start": self.start,
            "end": self.end,
            "category": self.category,
            "label": self.label,
            "classifier": self.classifier,
            "above_floor_db": self.above_floor_db,
            "tier": self.tier,
            "binding_floor": self.binding_floor,
            "variant": self.variant,
            "gain_db": self.gain_db,
            "computed_on": self.computed_on,
            "padding_fraction": self.padding_fraction,
            "from_mask_region": self.from_mask_region,
            "mask_confidence": self.mask_confidence,
            "leakage_margin_db": self.leakage_margin_db,
            "suppression_depth_db": self.suppression_depth_db,
            "flatness": self.flatness,
            "modulation_depth": self.modulation_depth,
            "occupancy": self.occupancy,
            "stationary_pass": self.stationary_pass,
            "discounted_reason": self.discounted_reason,
        }


def assign_tier(above_floor_db: float, margins: Mapping[str, float]) -> Tier:
    """Assign a tier from a candidate's excess over its band floor.

    Args:
        above_floor_db: Excess over the band's noise floor, in dB.
        margins: The profile's ``margins_db`` block.

    Returns:
        The tier. ``"rejected"`` below the reject threshold — such a candidate is
        indistinguishable from the noise floor and is not a finding at any confidence.
    """
    excess = float(above_floor_db)
    if excess >= float(margins.get("confident", 10.0)):
        return "confident"
    if excess >= float(margins.get("probable", 6.0)):
        return "probable"
    if excess >= float(margins.get("candidate", 3.0)):
        return "candidate"
    return "rejected"


def spectral_flatness(power_spectrum: np.ndarray) -> float:
    """Geometric-to-arithmetic mean ratio of a power spectrum, in ``[0, 1]``.

    Near 1 for broadband noise, near 0 for tonal or structured content. Measured separation
    between noise floors and structured sources on this statistic is large — around 0.55
    versus below 0.01 — which makes it the cheapest high-value guard available.
    """
    p = np.asarray(power_spectrum, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size == 0:
        return 0.0
    geo = math.exp(float(np.mean(np.log(p))))
    arith = float(np.mean(p))
    return 0.0 if arith <= 0.0 else min(1.0, geo / arith)


def is_noise_like(flatness: float, *, flatness_max: float) -> bool:
    """Whether a segment looks like broadband noise rather than a structured source.

    Compared against a configured limit rather than a hardcoded one: the descriptor is
    standardized but no standard fixes a decision threshold for it, so the value is a
    policy entry that can be re-tuned on evidence.
    """
    return float(flatness) > float(flatness_max)


def passes_pregain_gate(segment_rms_dbfs: float, *, reject_below_dbfs: float) -> bool:
    """Whether a segment is loud enough to be worth amplifying and interpreting.

    Below the classifiers' measured trust floor the result is not interpretable at any gain,
    so the segment is rejected rather than amplified — amplifying it is what manufactures
    the water-like labels the quarantine list exists to catch.
    """
    return float(segment_rms_dbfs) >= float(reject_below_dbfs)


def matches_floor_signature(
    scores_by_label: Mapping[str, float],
    signature: Mapping[str, float] | None,
    *,
    tolerance: float = 0.05,
) -> bool:
    """Whether a window reproduces a classifier's known below-floor response (FR-020d).

    Compares the **whole** pattern, not the silence label alone. One measured signature pairs
    a silence score of ~0.44 with a co-occurring label at ~0.35: a silence-only check passes
    it straight through while the second label clears an ordinary threshold.
    """
    if not signature:
        return False
    if set(scores_by_label) != set(signature):
        return False
    return all(abs(float(scores_by_label[k]) - float(signature[k])) <= tolerance for k in signature)


def is_quarantined(label: str, quarantined: Sequence[str]) -> bool:
    """Whether ``label`` is one an amplified noise floor is known to produce."""
    return str(label) in set(quarantined)


def screen_candidate(
    *,
    label: str,
    above_floor_db: float,
    flatness: float,
    segment_rms_dbfs: float,
    profile: Mapping[str, Any],
    scores_by_label: Mapping[str, float] | None = None,
    floor_signature: Mapping[str, float] | None = None,
) -> tuple[Tier, str | None]:
    """Apply every guard and return ``(tier, rejection_reason)``.

    Order matters: the cheap structural checks run before the margin, so a fabricated
    candidate is rejected for the *right* reason rather than for a marginal excess. A
    consumer reading ``rejection_reason`` should be able to tell "this was noise" from
    "this was real but too quiet".

    Args:
        label: Classifier label.
        above_floor_db: Excess over the band floor.
        flatness: Spectral flatness of the segment.
        segment_rms_dbfs: Pre-gain segment level.
        profile: Detection-margin profile.
        scores_by_label: The window's full label→score map, for the signature check.
        floor_signature: The classifier's known below-floor response, if any.

    Returns:
        ``(tier, reason)``; ``reason`` is ``None`` when the candidate survives.
    """
    guards = dict(profile.get("guards") or {})
    level = dict(profile.get("level") or {})
    margins = dict(profile.get("margins_db") or {})

    if matches_floor_signature(scores_by_label or {}, floor_signature):
        return "rejected", "classifier floor-response signature"
    if not passes_pregain_gate(
        segment_rms_dbfs, reject_below_dbfs=float(level.get("reject_below_pregain_dbfs", -45.0))
    ):
        return "rejected", "segment below the classifier trust floor before gain"
    noise_like = is_noise_like(flatness, flatness_max=float(guards.get("flatness_max", 0.30)))
    if noise_like and is_quarantined(label, guards.get("quarantined_labels") or []):
        return "rejected", "quarantined label on a noise-like segment"
    if noise_like:
        return "rejected", "segment is broadband noise, not a structured source"

    tier = assign_tier(above_floor_db, margins)
    if tier == "rejected":
        return tier, "below the reject margin above the band noise floor"
    return tier, None


# ── excision routing (T067, FR-041 to FR-045) ──────────────────────────
#
# The long-window classifier runs on *excised* mask segments rather than on the full
# timeline. Measurement drove it: with a loud-then-quiet test signal, excising the
# quiet segment and classifying it alone beat every mixed-window variant (0.705 vs a
# best of 0.548), because one 10.24 s window spanning both halves couples them and the
# loud half dominates the decision. The short-window classifier needs no excision --
# its ~1 s windows already sit entirely inside one half or the other.
#
# The cost is real and is reported rather than absorbed: a mask region shorter than the
# classifier's window is zero-padded, and padding maps to a fixed value while the signal
# region drifts with gain, so the pad-to-signal contrast is itself gain-dependent.


@dataclass(frozen=True)
class ExcisedSegment:
    """One mask region prepared for the long-window classifier."""

    region_id: str
    start: float
    end: float
    padding_fraction: float
    supports_long_window: bool

    @property
    def duration_s(self) -> float:
        """Segment length in seconds."""
        return max(0.0, self.end - self.start)


def plan_excision(
    mask_rows: Sequence[Mapping[str, Any]],
    *,
    long_window_s: float,
    max_padding_fraction: float,
    min_region_s: float = 0.0,
) -> list[ExcisedSegment]:
    """Choose which target-free regions to excise for the long-window classifier.

    Only ``target_free`` regions are candidates: an excised segment exists to give the
    classifier audio uncontaminated by target activity, which is the whole point of cutting
    it out.

    Regions are **not** concatenated to reach a usable length. Joining two spans would
    create a discontinuity at the seam that an onset-sensitive feature reads as an event —
    manufacturing exactly the kind of finding the guards elsewhere work to prevent.

    Args:
        mask_rows: Rows from the background mask.
        long_window_s: The long-window classifier's analysis window.
        max_padding_fraction: Padding share above which a decision is not trustworthy.
        min_region_s: Shortest region worth excising at all.

    Returns:
        One :class:`ExcisedSegment` per target-free region, each carrying whether it can
        host an unpadded decision. Regions that cannot are still returned, flagged, so a
        consumer sees what was skipped instead of inferring it from absence.
    """
    out: list[ExcisedSegment] = []
    for row in mask_rows:
        if str(row.get("state")) != "target_free":
            continue
        start, end = float(row["start"]), float(row["end"])
        duration = max(0.0, end - start)
        if duration < min_region_s:
            continue
        padding = max(0.0, (long_window_s - duration) / long_window_s) if long_window_s > 0 else 0.0
        out.append(
            ExcisedSegment(
                region_id=str(row.get("region_id", "")),
                start=start,
                end=end,
                padding_fraction=padding,
                supports_long_window=padding <= max_padding_fraction,
            )
        )
    return out


def route_classifier(segment: ExcisedSegment | None, *, classifier_window_s: float, long_window_s: float) -> str:
    """Return ``"grid"`` or ``"excised"`` for one classifier on one segment.

    The short-window classifier always stays on the regular grid: its windows already fit
    inside a single mask region, so excision would buy nothing and would cost the
    continuity its overlapping hop provides.
    """
    if classifier_window_s < long_window_s / 2.0:
        return "grid"
    return "excised" if segment is not None and segment.supports_long_window else "grid"
