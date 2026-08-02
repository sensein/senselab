"""Amplitude-invariance probe for scene classifiers (T017, FR-013 to FR-017b).

Answers one research question: **does either scene classifier normalize input signal level
as part of its own inference?** The question matters because if a classifier
self-normalizes, a quiet background is already scaled up before classification and the
reason faint sources go unreported is masking rather than level; if it does not, faint
sources sit near an absolute floor and explicit amplification is required to lift them.

Measurement answered it: **neither classifier self-normalizes.** Both are
amplitude-sensitive, and the expected asymmetry did not hold — the long-window model,
which is the one with an explicit normalization step, turned out to be the more
gain-brittle of the two, because that step divides by *fixed dataset-level constants* and
so cannot cancel a per-recording level offset.

Consequences that shape the rest of the feature:

- Detection cannot be based on gain. Attenuate-then-reamplify is bit-exact in floating
  point, so amplification never recovers content; it only keeps a classifier's floor from
  destroying it. Detection lives in ``noise_floor``/``sources`` instead.
- Each classifier has an absolute floor beneath which it reports nothing, and the more
  restrictive of the two binds the detection margin.
- Label speaker migrates with level on unchanged audio, so scores are not comparable
  across segments recorded or gained differently.

Everything in this module is pure and numpy-free: it derives verdicts from per-gain
classification results that a caller obtained however it liked. The model-touching sweep
lives in ``scripts/probe_classifier_levels.py`` so this module stays cheap to import and
testable without a checkpoint.

Note on units:
    The sweep applies *relative gain* to one recording, so :attr:`
    AmplitudeInvarianceVerdict.low_level_floor_db` is a gain offset in dB, **not** an
    absolute dBFS level. Converting to dBFS requires the source recording's own level, and
    conflating the two would invent an absolute claim the probe cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from senselab.audio.tasks.classification.label_scores import label_scores

__all__ = [
    "GAIN_RANGE_MIN_DB",
    "AmplitudeInvarianceVerdict",
    "detect_floor_signature",
    "label_stability",
    "max_score_delta",
    "top_k_labels",
    "validate_gain_range",
    "verdict_from_sweep",
]

GAIN_RANGE_MIN_DB = 30.0
"""Minimum gain span a verdict must be measured over (SC-005).

A narrower sweep cannot see a classifier's low-level floor, so it cannot distinguish
"invariant" from "not yet pushed far enough".
"""

_SCORE_EQ_TOL = 1e-6
"""Below this, two scores are the same number rather than a measured difference."""

_FLOOR_SIGNATURE_MIN_WINDOWS = 2
"""One window repeating nothing is not evidence of a saturated response."""

Verdict = Literal["self_normalizing", "level_sensitive"]


def top_k_labels(window: Any, k: int) -> tuple[str, ...]:  # noqa: ANN401 — senselab window dicts
    """Return the top-``k`` labels of one classification window, in rank order.

    ``classify_audios`` emits labels and scores pre-sorted descending, so index order is
    rank order. A malformed or empty window contributes nothing rather than raising —
    a probe should not fall over on one bad window.
    """
    if not isinstance(window, dict):
        return ()
    labels = [next(iter(d)) for d in label_scores(window)]
    return tuple(str(label) for label in labels[:k])


def _scores_by_label(window: Any) -> dict[str, float]:  # noqa: ANN401
    """Map label → score for one window, tolerating malformed input."""
    if not isinstance(window, dict):
        return {}
    pairs = label_scores(window)
    labels = [next(iter(d)) for d in pairs]
    scores = [next(iter(d.values())) for d in pairs]
    return {str(label): float(score) for label, score in zip(labels, scores)}


def label_stability(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    *,
    k: int = 5,
) -> float | None:
    """Fraction of windows whose top-``k`` label list is *identical* to the reference's.

    Order-sensitive on purpose. A reordered top-k means the classifier's ranking moved
    with level, which is exactly the instability being measured — a set-overlap measure
    would score a reversal as perfectly stable.

    Compares only the windows both runs produced: window counts can differ at the tail,
    and that difference is a windowing artifact rather than a level effect.

    Args:
        reference: Per-window results at unity gain.
        candidate: Per-window results at the probed gain.
        k: Rank depth to compare.

    Returns:
        A fraction in ``[0, 1]``, or ``None`` when there is nothing to compare — no
        windows means no measurement, which is not the same as zero stability.
    """
    n = min(len(reference), len(candidate))
    if n == 0:
        return None
    matches = sum(1 for i in range(n) if top_k_labels(reference[i], k) == top_k_labels(candidate[i], k))
    return matches / n


def max_score_delta(reference: Sequence[Any], candidate: Sequence[Any]) -> float:
    """Largest absolute per-label score change between two runs.

    Compared per label rather than per rank, so a reordering does not masquerade as a
    small change. A label that vanishes from the candidate's top-k counts as a change of
    its whole reference score — dropping out is a large change, not a zero one.
    """
    worst = 0.0
    for i in range(min(len(reference), len(candidate))):
        ref_scores = _scores_by_label(reference[i])
        cand_scores = _scores_by_label(candidate[i])
        for label in set(ref_scores) | set(cand_scores):
            delta = abs(ref_scores.get(label, 0.0) - cand_scores.get(label, 0.0))
            worst = max(worst, delta)
    return worst


def detect_floor_signature(windows: Sequence[Any]) -> dict[str, float] | None:
    """Return the fixed label→score response a classifier saturates to, if it has one.

    Below its floor a classifier can emit the *same* label pattern regardless of content.
    Detecting that pattern matters because thresholding on a silence label alone will not
    catch it: one measured signature pairs a silence score of ~0.44 with a co-occurring
    label at ~0.35, and the second one clears most practical thresholds while the first
    does not (FR-020d).

    Args:
        windows: Per-window results, typically from a digital-silence probe.

    Returns:
        The repeated label→score mapping, or ``None`` when the response varies (which
        means the classifier was responding to content, not saturating).
    """
    if len(windows) < _FLOOR_SIGNATURE_MIN_WINDOWS:
        return None
    first = _scores_by_label(windows[0])
    if not first:
        return None
    for window in windows[1:]:
        other = _scores_by_label(window)
        if set(other) != set(first):
            return None
        if any(abs(other[label] - first[label]) > _SCORE_EQ_TOL for label in first):
            return None
    return dict(first)


def validate_gain_range(gains_db: Sequence[float]) -> tuple[float, float]:
    """Validate a probe's gain points and return ``(min, max)``.

    Args:
        gains_db: The gains probed, in dB.

    Returns:
        The inclusive range covered.

    Raises:
        ValueError: If fewer than two points were probed, if unity gain is missing
            (stability is measured *against* unity, so it must be there), or if the span
            is under :data:`GAIN_RANGE_MIN_DB`.
    """
    if len(gains_db) < 2:
        raise ValueError(f"a gain sweep needs at least 2 points; got {len(gains_db)}")
    if not any(abs(float(g)) <= _SCORE_EQ_TOL for g in gains_db):
        raise ValueError("gain sweep must include unity gain (0 dB) — stability is measured against it")
    lo, hi = min(float(g) for g in gains_db), max(float(g) for g in gains_db)
    if (hi - lo) < GAIN_RANGE_MIN_DB - _SCORE_EQ_TOL:
        raise ValueError(
            f"gain sweep spans {hi - lo:.1f} dB; at least {GAIN_RANGE_MIN_DB:.0f} dB is required to see a "
            "classifier's low-level floor (SC-005)"
        )
    return lo, hi


@dataclass(frozen=True)
class AmplitudeInvarianceVerdict:
    """One classifier's level-sensitivity verdict and the evidence behind it.

    Attributes:
        classifier: Model identifier.
        window_length_s: The classifier's analysis window. Carried so a verdict is never
            generalized across classifiers with different windows (FR-015).
        verdict: ``"self_normalizing"`` or ``"level_sensitive"``.
        gain_range_db: The span the verdict was measured over.
        label_stability: gain → fraction of windows with an identical top-k list.
        score_delta_max: gain → largest absolute per-label score change.
        low_level_floor_db: Relative gain at which the classifier stopped reporting
            content, or ``None`` if it never did within the probed range. A **gain
            offset**, not an absolute dBFS level.
        floor_signature: Fixed label→score response below the floor, if any.
        floor_mechanism: Prose description of the mechanism.
        mechanism_source: Code location corroborating the empirical verdict (FR-016).
    """

    classifier: str
    window_length_s: float
    verdict: Verdict
    gain_range_db: tuple[float, float]
    label_stability: Mapping[float, float | None]
    score_delta_max: Mapping[float, float]
    low_level_floor_db: float | None = None
    floor_signature: Mapping[str, float] | None = None
    floor_mechanism: str = ""
    mechanism_source: str = ""
    notes: str = ""
    _unused: tuple[()] = field(default=(), repr=False, compare=False)

    def require_corroboration(self) -> None:
        """Raise unless the verdict cites the code that explains it.

        An empirical verdict without a mechanism is a measurement nobody can check
        (FR-016), and this feature's thresholds depend on these floors.

        Raises:
            ValueError: If :attr:`mechanism_source` is empty.
        """
        if not self.mechanism_source.strip():
            raise ValueError(
                f"verdict for {self.classifier!r} has no mechanism_source — an empirical verdict must be "
                "corroborated against the code that produces it (FR-016)"
            )

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/level-verdicts.md``."""
        return {
            "classifier": self.classifier,
            "window_length_s": self.window_length_s,
            "verdict": self.verdict,
            "gain_range_db": list(self.gain_range_db),
            "label_stability": {str(g): v for g, v in self.label_stability.items()},
            "score_delta_max": {str(g): v for g, v in self.score_delta_max.items()},
            "low_level_floor_db": self.low_level_floor_db,
            "floor_signature": dict(self.floor_signature) if self.floor_signature else None,
            "floor_mechanism": self.floor_mechanism,
            "mechanism_source": self.mechanism_source,
            "notes": self.notes,
        }


def verdict_from_sweep(
    classifier: str,
    *,
    window_length_s: float,
    per_gain: Mapping[float, Sequence[Any]],
    silence_windows: Sequence[Any] | None = None,
    floor_mechanism: str = "",
    mechanism_source: str = "",
    notes: str = "",
    k: int = 5,
) -> AmplitudeInvarianceVerdict:
    """Derive a verdict from per-gain classification results.

    ``self_normalizing`` requires that **nothing** moved — every probed gain reproduced
    the unity-gain label list exactly, with no score change beyond numerical noise. The
    bar is deliberately high: that verdict would overturn a measured finding, so it should
    not be reachable by a near-miss.

    Args:
        classifier: Model identifier.
        window_length_s: The classifier's analysis window length.
        per_gain: gain (dB) → per-window results. Must include unity gain.
        silence_windows: Optional digital-silence results, for floor-signature detection.
        floor_mechanism: Prose description of the mechanism, if known.
        mechanism_source: Code location corroborating the verdict.
        notes: Free-form observations.
        k: Rank depth for label comparison.

    Returns:
        The assembled verdict.

    Raises:
        ValueError: If the sweep is too narrow, has too few points, or omits unity gain.
    """
    lo, hi = validate_gain_range(list(per_gain))
    reference = next(results for gain, results in per_gain.items() if abs(float(gain)) <= _SCORE_EQ_TOL)

    stability: dict[float, float | None] = {}
    deltas: dict[float, float] = {}
    for gain, results in sorted(per_gain.items()):
        stability[float(gain)] = label_stability(reference, results, k=k)
        deltas[float(gain)] = max_score_delta(reference, results)

    moved = any(
        (value is not None and value < 1.0) or deltas[gain] > _SCORE_EQ_TOL for gain, value in stability.items()
    )
    verdict: Verdict = "level_sensitive" if moved else "self_normalizing"

    # The floor is where the top-1 label stopped agreeing with unity at reduced gain —
    # a collapse, not the wobble that `label_stability` also captures.
    ref_top1 = top_k_labels(reference[0], 1) if reference else ()
    collapsed = [
        float(gain)
        for gain, results in per_gain.items()
        if float(gain) < 0.0 and results and top_k_labels(results[0], 1) != ref_top1
    ]
    floor = max(collapsed) if collapsed else None

    return AmplitudeInvarianceVerdict(
        classifier=classifier,
        window_length_s=float(window_length_s),
        verdict=verdict,
        gain_range_db=(lo, hi),
        label_stability=stability,
        score_delta_max=deltas,
        low_level_floor_db=floor,
        floor_signature=detect_floor_signature(silence_windows or []),
        floor_mechanism=floor_mechanism,
        mechanism_source=mechanism_source,
        notes=notes,
    )
