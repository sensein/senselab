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
the influence gate's floor in ``influence.py``.
"""

from __future__ import annotations

from typing import Any, Mapping

from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT

__all__ = [
    "signal_names",
    "measured_weights",
    "MIN_RELIABILITY",
    "reliability_from_stability",
    "signal_stability",
    "stability_rows",
]

MIN_RELIABILITY = MIN_EVIDENCE_WEIGHT
"""Floor on a signal's weight, so a maximally-unstable signal is attenuated, not silenced. The
number and its derivation live in
:data:`~senselab.audio.workflows.audio_analysis.floors.MIN_EVIDENCE_WEIGHT`."""

_AXIS_SIGNALS: dict[str, tuple[str, str]] = {
    # (PassHarvest field, key holding the per-signal mapping). The speech-presence axis stores L1
    # *measurements* under "evidence"; the other two still store votes. Used for enumerating
    # signal names, which needs no verdict.
    "speech_presence": ("speech_presence_evidence", "evidence"),
    "speaker": ("speaker_votes", "votes"),
    "asr": ("asr_votes", "votes"),
}


def _bucket_beliefs(buckets: Any) -> dict[tuple[float, float], dict[str, float]]:  # noqa: ANN401
    """``{(start, end) → {signal → linked belief in [0, 1]}}`` for one pass's linked buckets.

    Measured on the *linked belief* — ``fuse.per_signal_uncertainty``, the exact quantity
    ``fuse_axis`` consumes — so a signal's weight and its value can no longer be derived from
    different things. The previous version matched a fixed tuple of vote field names, none of
    which the presence harvest emits, so presence stability silently returned ``{}`` on every real
    run and every presence signal kept weight 1.0: unmeasured, hence floored, hence never applied.
    """
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty

    out: dict[tuple[float, float], dict[str, float]] = {}
    for bucket in buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        key = (round(float(bucket.get("start", 0.0)), 6), round(float(bucket.get("end", 0.0)), 6))
        per_signal = dict(per_signal_uncertainty(bucket))
        # A voter that states only a direction — ``speaks: True`` with no confidence — has no
        # per-signal *doubt* to compare, so `per_signal_uncertainty` rightly omits it. But its
        # answer is what flipped, and a flip is exactly what stability is asking about. Each
        # signal is only ever compared against itself across passes, so the scale need only be
        # consistent per signal; omitting these voters is what left them permanently unmeasured.
        for name, entry in (bucket.get("votes") or {}).items():
            if str(name) in per_signal or not isinstance(entry, Mapping):
                continue
            if isinstance(entry.get("speaks"), bool):
                per_signal[str(name)] = 1.0 if entry["speaks"] else 0.0
        if per_signal:
            out[key] = per_signal
    return out


def signal_stability(
    harvests: Mapping[str, Any],
    *,
    axis: str,
    buckets_by_pass: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Mean absolute disagreement of each signal with itself across passes.

    Args:
        harvests: ``{pass_label → PassHarvest}``. At least two passes are required for any
            signal to be scored. Ignored when ``buckets_by_pass`` is given.
        axis: ``"speech_presence"``, ``"speaker"``, or ``"asr"``.
        buckets_by_pass: ``{pass_label → linked buckets}``. Preferred: presence measurements have
            to be *linked* before they can be compared as beliefs, and the caller has already done
            that under the run's policy.

    Returns:
        ``{signal → instability in [0, 1]}``, empty when fewer than two passes are present or
        no signal appears in more than one of them. Only buckets a signal reported in *both*
        passes are compared — a signal that dropped out of one pass is silent there, which is
        different from disagreeing.

    Raises:
        ValueError: On an unknown ``axis``.
    """
    resolved = _AXIS_SIGNALS.get(str(axis))
    if resolved is None:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXIS_SIGNALS)}")
    if buckets_by_pass is None:
        from senselab.audio.workflows.audio_analysis.speech_presence_link import votes_for_harvest

        field, _ = resolved
        buckets_by_pass = {
            label: (votes_for_harvest(h) if axis == "speech_presence" else (getattr(h, field, []) or []))
            for label, h in harvests.items()
        }
    per_pass = {label: _bucket_beliefs(buckets) for label, buckets in sorted(buckets_by_pass.items())}
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


def stability_rows(
    buckets_by_pass: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Per-bucket cross-pass ``|Δ|`` per signal — the evidence behind ``signal_stability``.

    Returns ``{signal → [{start, end, signal, pass_a, pass_b, abs_delta, n_passes_present}]}``.
    Written to ``L1/stability/<signal>.parquet`` so the number that set a signal's weight is
    inspectable per bucket rather than only as a run-level mean.
    """
    per_pass = {label: _bucket_beliefs(buckets) for label, buckets in sorted(buckets_by_pass.items())}
    labels = sorted(lab for lab, v in per_pass.items() if v)
    out: dict[str, list[dict[str, Any]]] = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            for bucket in sorted(set(per_pass[a]) & set(per_pass[b])):
                left, right = per_pass[a][bucket], per_pass[b][bucket]
                for signal in sorted(set(left) & set(right)):
                    out.setdefault(signal, []).append(
                        {
                            "start": bucket[0],
                            "end": bucket[1],
                            "signal": signal,
                            "pass_a": a,
                            "pass_b": b,
                            "abs_delta": abs(left[signal] - right[signal]),
                            "n_passes_present": 2,
                        }
                    )
    return out


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


def signal_names(harvests: Mapping[str, Any], *, axis: str) -> list[str]:
    """Every signal name that voted on this axis, across all passes.

    The derivation gate must reach signals with no perturbation evidence too — a signal that
    appeared in only one pass is unmeasured for stability but still derived or not.
    """
    resolved = _AXIS_SIGNALS.get(str(axis))
    if resolved is None:
        raise ValueError(f"unknown axis {axis!r}; expected one of {sorted(_AXIS_SIGNALS)}")
    field, signal_key = resolved
    names: set[str] = set()
    for harvest in harvests.values():
        for bucket in getattr(harvest, field, None) or []:
            if isinstance(bucket, Mapping):
                names.update(str(k) for k in (bucket.get(signal_key) or {}))
    return sorted(names)


def measured_weights(
    instability: Mapping[str, float],
    support: Mapping[str, float],
    signals: Any,  # noqa: ANN401 — any iterable of signal names
    *,
    min_reliability: float = MIN_RELIABILITY,
) -> dict[str, float]:
    """The weight a signal's doubt carries: perturbation stability x physical support.

    Both factors are *measured*. The gate they replace was declared — a source kind written
    into policy — and a declared gate encodes a judgement from whichever recording motivated
    it. That judgement was wrong about this very model on a second recording: on a 4.9 s group
    introduction the down-weighted clusterer recovered five named speakers in the right places
    while both "independent" diarizers merged them into one.

    The two factors answer different questions and neither subsumes the other. Stability asks
    whether a signal agrees with itself when the audio is transformed. Support asks whether
    the audio carries what the signal claimed — a speaker asserted where independent speech_presence
    evidence reports silence is a claim the recording does not back.

    Identity sub-signals are keyed ``<diar_model>::<embedding_model>``; the claim about where a
    speaker is belongs to the diar model, so support resolves through the prefix.

    A signal absent from either mapping was not measured on that factor and keeps full weight
    there — a factor never gathered must not act as a discount.
    """
    reliability = reliability_from_stability(instability, min_reliability=min_reliability)
    out: dict[str, float] = {}
    for name in sorted({str(s) for s in signals}):
        claimant = name.split("::", 1)[0]
        stability = reliability.get(name, reliability.get(claimant, 1.0))
        backing = float(support.get(name, support.get(claimant, 1.0)))
        out[name] = stability * max(0.0, min(1.0, backing))
    return out
