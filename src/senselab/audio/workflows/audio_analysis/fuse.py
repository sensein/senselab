"""Level 2: fuse per-signal uncertainties across signals and passes into the final maps.

The pipeline has two levels and they answer different questions.

**Level 1** (``harvest_*``) computes signals and each signal's *own* uncertainty. It must not
decide the answer. The per-pass fold it used to perform is a within-pass diagnostic — "what
did this pass think" — and folding early is precisely how one saturated sub-signal came to pin
an axis at 1.0 while two independent diarizers, both embedding models and the per-speaker
speech_presence track all agreed nothing had changed: the fold ran before anything had been measured
about the signals, so there was no weight to apply.

**Level 2** (this module) aggregates across every signal and every pass, weighting each signal
by what was measured about it — perturbation stability and physical support — and iterates. Its
maps are the answer a consumer should read.

Once a fold has happened the weights can no longer be applied, which is why the ordering
matters rather than being a matter of taste.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator
from senselab.audio.workflows.audio_analysis.statistics import epistemic_uncertainty, variability

__all__ = [
    "fuse_rounds",
    "write_final_uncertainty",
    "fuse_axis",
    "per_signal_uncertainty",
]

# Fields carrying a signal's own uncertainty in [0, 1]. A signal may expose more than one
# (a same-label claim and a change claim); each counts separately, because they are distinct
# assertions that can disagree.
_UNCERTAINTY_FIELDS = (
    "same_label_uncertainty",
    "change_inconsistency_uncertainty",
    "value",
)

# Presence and asr signals report a *confidence*, not an uncertainty. Converting rather
# than ignoring them matters: on a real run the speaker axis fused 85 buckets while speech_presence
# fused 0 of 1070 and asr 0 of 41, because only speaker happens to name its field
# "uncertainty". A level that silently covers one axis of three is worse than one that covers
# none, since the gap is invisible in the output.
_CONFIDENCE_FIELDS = ("native_confidence", "p_speech", "p_voice", "argmax_confidence")

# Utterance signals expose neither an uncertainty nor a [0, 1] confidence — they report
# ``avg_logprob``, a mean token log-probability. exp() takes it back to a probability, which is
# the model's own confidence in the transcript it produced. Left unhandled, the asr axis
# fused 0 of 41 buckets while the other two fused fully.
_LOGPROB_FIELDS = ("avg_logprob",)


def per_signal_uncertainty(bucket: Mapping[str, Any]) -> dict[str, float]:
    """Each signal's own uncertainty in one bucket — the level-1 emission.

    Reported per signal rather than folded, so level 2 can weight them. A fold cannot be
    re-weighted after the fact, which is the whole reason for the split.

    A signal that said nothing is absent from the result rather than zero-filled: zero is a
    confident claim, and imputing it would manufacture confidence nobody expressed (FR-007).
    """
    votes = bucket.get("votes") or {}
    if not isinstance(votes, Mapping):
        return {}
    out: dict[str, float] = {}
    # Pairwise evidence first, then overridden below by anything a signal states directly.
    out.update(_pairwise_per_signal(votes))
    for name, entry in votes.items():
        if not isinstance(entry, Mapping):
            continue
        for field in _UNCERTAINTY_FIELDS:
            value = entry.get(field)
            if isinstance(value, (int, float)):
                out[str(name)] = max(0.0, min(1.0, float(value)))
                break
        else:
            if str(name) in out:
                continue
            for field in _LOGPROB_FIELDS:
                value = entry.get(field)
                if isinstance(value, (int, float)):
                    out[str(name)] = max(0.0, min(1.0, 1.0 - math.exp(min(0.0, float(value)))))
                    break
            for field in _CONFIDENCE_FIELDS:
                if str(name) in out:
                    break
                value = entry.get(field)
                if isinstance(value, (int, float)):
                    # A confidence of c leaves 1 - c of doubt. Distance from 0.5 would be the
                    # wrong transform: it maps a confident "no" to zero uncertainty about
                    # *speech_presence*, which is right, but this field is a confidence in the claim
                    # the signal made, so its complement is the doubt in that claim.
                    out[str(name)] = max(0.0, min(1.0, 1.0 - float(value)))
                    break
    return out


def _pairwise_per_signal(votes: Mapping[str, Any]) -> dict[str, float]:
    """Per-signal uncertainty from pairwise disagreement, for axes that only report pairs.

    The asr axis carries ``{model_a|model_b: phoneme_distance}`` rather than a per-model
    confidence, and on a real recording all three ASR backends were text-only so every
    ``avg_logprob`` was ``None`` — leaving the axis with no per-signal quantity at all and L2
    fusing 0 of 41 buckets.

    A distance belongs to both models in the pair, so each model's uncertainty is the mean of
    the distances it participates in: a transcript that differs from everyone else's is the
    doubtful one, and that is recoverable from pairs even when no model reports its own
    confidence. Attributing to one side only would blame whichever name sorted first.
    """
    block = votes.get("__pairwise_phoneme_distances__")
    if not isinstance(block, Mapping):
        return {}
    per_model: dict[str, list[float]] = {}
    for key, distance in (block.get("pairs") or {}).items():
        if not isinstance(distance, (int, float)):
            continue
        parts = str(key).split("|", 1)
        if len(parts) != 2:
            continue
        for model in parts:
            per_model.setdefault(model, []).append(max(0.0, min(1.0, float(distance))))
    return {model: sum(v) / len(v) for model, v in sorted(per_model.items()) if v}


def fuse_axis(
    buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float],
    aggregator: str = "mean",
    weight_basis: Mapping[str, Mapping[str, float]] | None = None,
    round_index: int = 0,
) -> list[dict[str, Any]]:
    """Fuse one axis's per-signal uncertainties across signals and passes.

    Args:
        buckets_by_pass: ``{pass_label → per-bucket harvested votes}``.
        weights: ``{signal → measured weight}``. A signal absent from the mapping carries
            full weight: a factor never measured must not act as a discount.
        aggregator: How to combine the weighted per-signal uncertainties into
            ``triage_score``. It does not affect ``uncertainty``, which is the entropy
            measure and has no policy choice in it.
        weight_basis: ``{signal → {factor → value}}``, so a weight of 0.05 says whether the
            signal was unstable, unsupported, or both. Without it a discounted signal is
            indistinguishable from a differently discounted one.
        round_index: Which iteration produced these values. Later rounds refine earlier ones,
            so a value without its round cannot be compared against another.

    Returns:
        One row per bucket, in time order. Each carries four distinct quantities —
        ``uncertainty`` (normalised entropy), ``epistemic_uncertainty`` (its reducible part),
        ``confidence`` (a probability), ``variability`` (a dispersion) — plus ``triage_score``
        (the policy fold), the signals and passes that contributed, and the weight each signal
        carried with the factors behind it.
        Ordering is by time then pass so the maps stay byte-identical across runs (SC-004).

        Every quantity is ``None`` where no signal spoke. That is not the same as ``0.0``,
        which would assert confidence nobody expressed.
    """
    # (start, end) → signal → [uncertainty, ...]; a signal appearing in both passes
    # contributes both readings, because disagreement between passes is evidence too.
    collected: dict[tuple[float, float], dict[str, list[float]]] = {}
    passes_seen: dict[tuple[float, float], set[str]] = {}

    for pass_label in sorted(buckets_by_pass):
        for bucket in buckets_by_pass[pass_label] or []:
            if not isinstance(bucket, Mapping):
                continue
            key = (round(float(bucket.get("start", 0.0)), 6), round(float(bucket.get("end", 0.0)), 6))
            slot = collected.setdefault(key, {})
            passes_seen.setdefault(key, set()).add(str(pass_label))
            for signal, value in per_signal_uncertainty(bucket).items():
                slot.setdefault(signal, []).append(value)

    rows: list[dict[str, Any]] = []
    for start, end in sorted(collected):
        readings = collected[(start, end)]
        signals = sorted(readings)
        # One reading per signal: the mean across passes. Averaging here rather than treating
        # each pass as a separate voter keeps a signal's weight from scaling with how many
        # passes it happened to appear in.
        values = [sum(readings[name]) / len(readings[name]) for name in signals]
        applied = [float(weights.get(s, 1.0)) for s in signals]
        candidates: list[float | None] = list(values)
        fused = apply_aggregator(candidates, aggregator, weights=applied) if signals else None
        # Three quantities with three estimators, deliberately not collapsed:
        #   confidence  — P(the axis is settled here), the weighted mean of per-signal
        #                 certainties; a probability, so it is calibratable.
        #   variability — dispersion across signals, in the units of the quantity. ``None``
        #                 for a lone signal, which cannot disagree with anyone.
        #   uncertainty — normalised entropy over the {settled, unsettled} outcome space,
        #                 which decomposes into a reducible part.
        certainties = [1.0 - v for v in values]
        weighted_confidence = (
            sum(c * w for c, w in zip(certainties, applied)) / sum(applied) if sum(applied) > 0 else None
        )
        spread = variability(values)
        total, epistemic = epistemic_uncertainty([{"unsettled": v, "settled": 1.0 - v} for v in values])
        rows.append(
            {
                "start": start,
                "end": end,
                # ``uncertainty`` is the entropy measure, not the aggregator fold. Having two
                # columns whose names both say "uncertainty" but whose maths differ is the
                # conflation this module exists to remove.
                "uncertainty": total,
                "epistemic_uncertainty": epistemic,
                "confidence": weighted_confidence,
                "variability": spread,
                # The policy-driven fold, kept under a name that says what it is for: ranking
                # where to spend the adaptive loop's budget. Max-doubt is the right operator
                # for "is any signal unsure", and the wrong one for "what do we believe".
                "triage_score": fused,
                "contributing_signals": signals,
                "contributing_passes": sorted(passes_seen[(start, end)]),
                "signal_weights": {s: w for s, w in zip(signals, applied)},
                "weight_basis": {s: dict((weight_basis or {}).get(s, {})) for s in signals},
                "round": int(round_index),
            }
        )
    return rows


def fuse_rounds(
    buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float],
    aggregator: str = "mean",
    weight_basis: Mapping[str, Mapping[str, float]] | None = None,
    mask_regions: Sequence[Mapping[str, Any]] = (),
    speaker_claims: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    max_rounds: int = 1,
    tolerance: float = 1e-3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Iterate the fusion until it stops moving, or ``max_rounds`` is reached.

    Round 0 fuses with one weight per signal. Later rounds apply *regional* trust: where the
    mask contradicts a signal's speaker claim, that signal is discounted in that region only.
    A global discount for a local failure is what suppressed the source that was right about
    the five named speakers, so the withdrawal has to stay local to be safe to apply at all.

    Args:
        buckets_by_pass: ``{pass_label → L1 buckets}``.
        weights: Round-0 per-signal weights.
        aggregator: Aggregator for ``triage_score``.
        weight_basis: Per-signal factor breakdown.
        mask_regions: Mask regions with ``state`` and ``confidence``, used for regional trust.
        speaker_claims: ``{signal → spans}`` where the signal asserted a speaker.
        max_rounds: Cap on iterations.
        tolerance: Per-bucket change below which a round counts as no change.

    Returns:
        ``(rows, log)`` — the final rows, and one log entry per round recording whether it
        changed anything. The log is what distinguishes "converged" from "ran out of rounds",
        which a bare result cannot say.
    """
    from senselab.audio.workflows.audio_analysis.rounds import regional_weights, round_converged

    log: list[dict[str, Any]] = []
    rows = fuse_axis(
        buckets_by_pass,
        weights=weights,
        aggregator=aggregator,
        weight_basis=weight_basis,
        round_index=0,
    )
    log.append({"round": 0, "buckets": len(rows), "converged": False, "regional_trust_applied": False})

    if not mask_regions or not speaker_claims:
        # Nothing to localise trust against: further rounds would recompute the same numbers,
        # and reporting them as convergence would overstate what was checked.
        log[-1]["converged"] = True
        log[-1]["reason"] = "no mask regions or speaker claims to localise trust against"
        return rows, log

    per_region = regional_weights(base_weights=dict(weights), regions=mask_regions, claims=speaker_claims)
    for round_index in range(1, max(1, int(max_rounds))):
        # Apply the tightest regional weight covering each signal, so a signal contradicted
        # anywhere it spoke is attenuated for the fold rather than silently rescued by a
        # region where it stayed quiet.
        tightened = {
            signal: min((w.get(signal, 1.0) for w in per_region.values()), default=weights.get(signal, 1.0))
            for signal in weights
        }
        candidate = fuse_axis(
            buckets_by_pass,
            weights=tightened,
            aggregator=aggregator,
            weight_basis=weight_basis,
            round_index=round_index,
        )
        converged = round_converged(rows, candidate, tolerance=tolerance)
        rows = candidate
        log.append(
            {
                "round": round_index,
                "buckets": len(rows),
                "converged": converged,
                "regional_trust_applied": True,
            }
        )
        if converged:
            break
    return rows, log


def write_final_uncertainty(
    out_dir: Any,  # noqa: ANN401 — Path
    *,
    harvests: Mapping[str, Any],
    weights_by_axis: Mapping[str, Mapping[str, float]],
    aggregator: str = "mean",
    weight_basis_by_axis: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    mask_regions: Sequence[Mapping[str, Any]] = (),
    speaker_claims: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    max_rounds: int = 1,
) -> dict[str, Any]:
    """Write ``L2/round<N>/uncertainty/{speech_presence,speaker,asr}.parquet``.

    One directory per round, because a reader needs to see what each iteration changed and not
    only where it ended up — a single map cannot distinguish "settled immediately" from
    "moved a long way and then settled".

    These are the maps a consumer should read. The per-pass parquets under ``L1/<pass>/`` are
    level-1 diagnostics: what each signal said, and what one pass alone would have concluded
    before anything was measured about the signals' reliability.

    Args:
        out_dir: Run directory.
        harvests: ``{pass_label → PassHarvest}``.
        weights_by_axis: ``{axis → {signal → measured weight}}``.
        aggregator: Aggregator for the ``triage_score`` fold.
        weight_basis_by_axis: ``{axis → {signal → {factor → value}}}``, so a discounted signal
            records which factor discounted it.
        mask_regions: Mask regions, enabling regional trust in rounds after the first.
        speaker_claims: ``{signal → spans}`` where each signal asserted a speaker.
        max_rounds: Cap on L2 iterations.

    Returns:
        ``{axis → written path}`` as strings, for the run summary.
    """
    from pathlib import Path

    import pandas as pd

    axis_field = {"speech_presence": "speech_presence_votes", "speaker": "speaker_votes", "asr": "asr_votes"}
    level2 = Path(out_dir) / "L2"
    level2.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    logs: dict[str, Any] = {}
    for axis, field in axis_field.items():
        by_pass = {label: getattr(h, field, []) or [] for label, h in harvests.items()}
        rows, round_log = fuse_rounds(
            by_pass,
            weights=weights_by_axis.get(axis, {}),
            aggregator=aggregator,
            weight_basis=(weight_basis_by_axis or {}).get(axis),
            mask_regions=mask_regions,
            speaker_claims=speaker_claims,
            max_rounds=max_rounds,
        )
        logs[axis] = round_log
        frame = pd.DataFrame(
            [
                {
                    "start": r["start"],
                    "end": r["end"],
                    "axis": axis,
                    "uncertainty": r["uncertainty"],
                    "epistemic_uncertainty": r["epistemic_uncertainty"],
                    "confidence": r["confidence"],
                    "variability": r["variability"],
                    "triage_score": r["triage_score"],
                    "round": r["round"],
                    "contributing_signals": r["contributing_signals"],
                    "contributing_passes": r["contributing_passes"],
                    "signal_weights": json.dumps(r["signal_weights"], sort_keys=True),
                    "weight_basis": json.dumps(r["weight_basis"], sort_keys=True),
                }
                for r in rows
            ],
            columns=[
                "start",
                "end",
                "axis",
                "uncertainty",
                "epistemic_uncertainty",
                "confidence",
                "variability",
                "triage_score",
                "round",
                "contributing_signals",
                "contributing_passes",
                "signal_weights",
                "weight_basis",
            ],
        )
        # One directory per round, so a reader can see what each iteration changed rather
        # than only where it ended up. The last round is also the axis's headline path.
        for round_index in sorted({int(r["round"]) for r in rows} or {0}):
            per_round = frame[frame["round"] == round_index] if len(frame) else frame
            round_dir = level2 / f"round{round_index}" / "uncertainty"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_path = round_dir / f"{axis}.parquet"
            per_round.to_parquet(round_path, index=False)
            written[f"{axis}@round{round_index}"] = str(round_path)
            written[axis] = str(round_path)

    # The round log distinguishes "converged" from "ran out of rounds", which the maps alone
    # cannot say and which call for different follow-up.
    (level2 / "rounds.json").write_text(json.dumps(logs, indent=2, sort_keys=True) + "\n")
    written["rounds"] = str(level2 / "rounds.json")
    return written
