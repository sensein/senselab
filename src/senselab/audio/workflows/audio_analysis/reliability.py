"""Per-signal reliability measured by perturbation, for use as aggregation weight.

A sub-signal's own uncertainty is evidence about how far its vote should carry. Without it,
aggregation treats every signal as equally trustworthy, and under max-doubt a single
unreliable signal decides the axis outright — which is exactly how a saturated embedding
check came to outvote unanimous diarizer agreement on a real recording.

The reliability is **derived rather than assigned**, on the same argument the speaker-count
posterior already uses: the raw and enhanced passes are the same recording under a
transform, so each signal's two answers already constitute a stability sample. A signal that
contradicts itself between them has not earned its weight; one that answers identically has.

Two properties are deliberate:

**One pass yields no claim.** A single observation is not a stability sample. Reporting
perfect reliability there would assert something never measured, so a signal with no
perturbation evidence simply keeps its full weight by default.

**Reliability never reaches zero.** With two perturbation points the measure is coarse, so a
hard zero would erase a dissenting claim rather than down-weight it — the same reasoning as
the influence gate's floor in ``adaptive/influence.py``.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "signal_names",
    "derivation_weights",
    "combined_weights",
    "MIN_RELIABILITY",
    "reliability_from_stability",
    "signal_stability",
]

MIN_RELIABILITY = 0.05
"""Floor on a signal's weight, so a maximally-unstable signal is attenuated, not silenced."""

_AXIS_VOTES = {"presence": "presence_votes", "identity": "identity_votes", "utterance": "utterance_votes"}

# Sub-signal fields whose value is an uncertainty in [0, 1] and therefore comparable across
# passes. Fields carrying raw measurements (cosine distances, transcripts) are excluded —
# comparing those across passes measures the perturbation's effect on the audio rather than
# the signal's stability of judgement.
_COMPARABLE_FIELDS = (
    "same_label_uncertainty",
    "change_inconsistency_uncertainty",
    "value",
)


def _bucket_values(votes: Any) -> dict[tuple[float, float], dict[str, float]]:  # noqa: ANN401
    """``{(start, end) → {signal → uncertainty}}`` for one pass's harvested votes."""
    out: dict[tuple[float, float], dict[str, float]] = {}
    for bucket in votes or []:
        if not isinstance(bucket, Mapping):
            continue
        key = (round(float(bucket.get("start", 0.0)), 6), round(float(bucket.get("end", 0.0)), 6))
        per_signal: dict[str, float] = {}
        for name, entry in (bucket.get("votes") or {}).items():
            if not isinstance(entry, Mapping):
                continue
            for field in _COMPARABLE_FIELDS:
                v = entry.get(field)
                if isinstance(v, (int, float)):
                    per_signal[str(name)] = float(v)
                    break
        if per_signal:
            out[key] = per_signal
    return out


def signal_stability(harvests: Mapping[str, Any], *, axis: str) -> dict[str, float]:
    """Mean absolute disagreement of each signal with itself across passes.

    Args:
        harvests: ``{pass_label → PassHarvest}``. At least two passes are required for any
            signal to be scored.
        axis: ``"presence"``, ``"identity"``, or ``"utterance"``.

    Returns:
        ``{signal → instability in [0, 1]}``, empty when fewer than two passes are present or
        no signal appears in more than one of them. Only buckets a signal reported in *both*
        passes are compared — a signal that dropped out of one pass is silent there, which is
        different from disagreeing.
    """
    field = _AXIS_VOTES.get(str(axis))
    if field is None:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXIS_VOTES)}")
    per_pass = {label: _bucket_values(getattr(h, field, None)) for label, h in sorted(harvests.items())}
    labels = [lab for lab, v in per_pass.items() if v]
    if len(labels) < 2:
        return {}

    deltas: dict[str, list[float]] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            shared = set(per_pass[a]) & set(per_pass[b])
            for bucket in shared:
                left, right = per_pass[a][bucket], per_pass[b][bucket]
                for signal in set(left) & set(right):
                    deltas.setdefault(signal, []).append(abs(left[signal] - right[signal]))
    return {s: sum(d) / len(d) for s, d in sorted(deltas.items()) if d}


def reliability_from_stability(
    instability: Mapping[str, float],
    *,
    min_reliability: float = MIN_RELIABILITY,
) -> dict[str, float]:
    """Convert measured instability into an aggregation weight in ``(0, 1]``.

    Args:
        instability: ``{signal → mean absolute cross-pass disagreement}``.
        min_reliability: Floor, so a maximally-unstable signal is attenuated rather than
            erased.

    Returns:
        ``{signal → reliability}``.
    """
    return {s: max(float(min_reliability), 1.0 - max(0.0, min(1.0, float(v)))) for s, v in sorted(instability.items())}


def derivation_weights(
    signals: Any,  # noqa: ANN401 — any iterable of signal names
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Per-signal derivation gate (FR-011c), applied where the axis is aggregated.

    The gate already governs the speaker-count posterior; the axis aggregator was bypassing
    it. Observed consequence on a single-speaker recording: a clustering-derived
    pseudo-diarizer split one speaker into five and therefore claimed a speaker change at
    nearly every bucket. Both embedding models correctly refuted every change — and each
    refutation was recorded as maximal identity uncertainty, pinning the axis at 1.0 while
    two independent diarizers and both embeddings agreed nothing had changed.

    A derived claim contradicted by the audio is evidence against the claimant, not evidence
    that identity is in doubt. Gating it is what lets the corroborated answer show through.

    Identity sub-signals are keyed ``<diar_model>::<embedding_model>``; the kind belongs to
    the diar model that made the claim, so the suffix is stripped before resolution.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    gates = dict(((policy or {}).get("influence") or {}).get("derivation_gate") or {})
    independent = float(gates.get("independent", 1.0))
    derived = float(gates.get("derived", 0.4))
    out: dict[str, float] = {}
    for name in signals:
        claimant = str(name).split("::", 1)[0]
        kind = source_kind_for(claimant, policy or _default_policy())
        out[str(name)] = independent if kind == "independent" else derived
    return out


def _default_policy() -> Mapping[str, Any]:
    """The packaged policy, so the gate is in force without every caller wiring it.

    A guard that only works when explicitly enabled is a guard that was off on the run that
    needed it.
    """
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

        return load_policy(None)
    except Exception:  # noqa: BLE001 — a missing policy must not break aggregation
        return {"influence": {"source_kinds": {"embedding_silhouette": "derived"}}}


def combined_weights(
    instability: Mapping[str, float],
    signals: Any,  # noqa: ANN401 — any iterable of signal names
    *,
    policy: Mapping[str, Any] | None = None,
    min_reliability: float = MIN_RELIABILITY,
) -> dict[str, float]:
    """Reliability × derivation gate, the weight a signal's doubt actually carries.

    The two factors say different things and neither subsumes the other: perturbation
    stability asks whether a signal agrees with itself, the derivation gate asks whether its
    agreement with another signal is independent evidence at all. This mirrors
    ``adaptive/influence.effective_weight``, which composes the same two gates.
    """
    reliability = reliability_from_stability(instability, min_reliability=min_reliability)
    gates = derivation_weights(signals, policy=policy)
    return {name: reliability.get(name, 1.0) * gates.get(name, 1.0) for name in sorted(gates)}


def signal_names(harvests: Mapping[str, Any], *, axis: str) -> list[str]:
    """Every signal name that voted on this axis, across all passes.

    The derivation gate must reach signals with no perturbation evidence too — a signal that
    appeared in only one pass is unmeasured for stability but still derived or not.
    """
    field = _AXIS_VOTES.get(str(axis))
    if field is None:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXIS_VOTES)}")
    names: set[str] = set()
    for harvest in harvests.values():
        for bucket in getattr(harvest, field, None) or []:
            if isinstance(bucket, Mapping):
                names.update(str(k) for k in (bucket.get("votes") or {}))
    return sorted(names)
