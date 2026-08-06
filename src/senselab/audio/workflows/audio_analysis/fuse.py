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

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator
from senselab.audio.workflows.audio_analysis.estimates import estimate_frame
from senselab.audio.workflows.audio_analysis.influence import effective_weight
from senselab.audio.workflows.audio_analysis.perturbations import IDENTITY_NAME
from senselab.audio.workflows.audio_analysis.statistics import epistemic_uncertainty, variability

__all__ = [
    "mask_axis_votes",
    "fold_run_axes",
    "Derivatives",
    "mask_regions_from_rows",
    "speaker_claims_from_votes",
    "measure_axis_overlap",
    "project_axis_onto",
    "derive_mask_from_axes",
    "fuse_axes",
    "fuse_rounds",
    "write_final_uncertainty",
    "fuse_axis",
    "is_direction_only_claim",
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

# Some signals expose neither an uncertainty nor a [0, 1] confidence — they report
# ``avg_logprob``, a mean token log-probability. exp() takes it back to a probability, which is
# the model's own confidence in the transcript it produced. Read on the *presence* axis, where an
# ASR backend's per-chunk logprob is one of the voters (``speech_presence_link``); the asr axis has
# a single voter and it reports ``value`` directly.
_LOGPROB_FIELDS = ("avg_logprob",)

# The direction a vote cast, when it scored nothing at all. See :func:`is_direction_only_claim`.
_DIRECTION_FIELD = "speaks"

_SCORED_FIELDS = (*_UNCERTAINTY_FIELDS, *_LOGPROB_FIELDS, *_CONFIDENCE_FIELDS)


def is_direction_only_claim(entry: Any) -> bool:  # noqa: ANN401 — vote entries are duck-typed
    """True when a vote asserts a direction and scores nothing.

    Three real voters are shaped this way, and none of them is defective:

    - a **diarizer**, because a segment boundary is asserted rather than scored
      (``speech_presence_link._link_diar`` deliberately reports no ``native_confidence``);
    - an **ASR backend without token logits** placing words in the bucket — CrisperWhisper 2.0
      turbo, Canary-Qwen and Qwen3-ASR all expose ``avg_logprob``/``no_speech_prob`` as ``None``,
      so word coverage is the whole of what they said;
    - the adaptive loop's **missed-speech adjudicator**, whose claim is that two model families
      agree words are here.

    Every other reader of a presence vote already takes such a vote at full strength — see
    ``aggregate.per_source_voice`` and ``support.presence_probability``, both of which map it to
    ``p = 1.0``/``0.0``. :func:`per_signal_uncertainty` did not, and dropped it instead. The cost
    was measured on a real run: with ``--asr-models openai/whisper-*`` the presence axis fused 12
    signals, and on the *shipped* default ASR set only 8 — all three ASR models and both
    diarizers had silently left the axis, because Whisper is the only backend whose per-segment
    ``avg_logprob`` gave the fold a number to read. ``reliability._bucket_beliefs`` had already
    had to reintroduce these voters by hand to measure their stability, so a weight was being
    computed for signals the fold could never use.

    Args:
        entry: One signal's vote payload.

    Returns:
        ``True`` only when ``speaks`` is a bool *and* no field carrying a scored quantity holds a
        number. A vote that scores anything is not direction-only, and its score is authoritative.
    """
    if not isinstance(entry, Mapping):
        return False
    if not isinstance(entry.get(_DIRECTION_FIELD), bool):
        return False
    for field in _SCORED_FIELDS:
        value = entry.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return False
    return True


def per_signal_uncertainty(bucket: Mapping[str, Any]) -> dict[str, float]:
    """Each signal's own uncertainty in one bucket — the level-1 emission.

    Reported per signal rather than folded, so level 2 can weight them. A fold cannot be
    re-weighted after the fact, which is the whole reason for the split.

    A signal that said nothing is absent from the result rather than zero-filled: zero is a
    confident claim, and imputing it would manufacture confidence nobody expressed (FR-007). A
    signal that asserted a *direction* without scoring it did say something, and is read at the
    full strength every other consumer of a presence vote already reads it at — see
    :func:`is_direction_only_claim` for why that is a different case, and for what dropping it cost.
    """
    votes = bucket.get("votes") or {}
    if not isinstance(votes, Mapping):
        return {}
    out: dict[str, float] = {}
    for name, entry in votes.items():
        if not isinstance(entry, Mapping):
            continue
        if is_direction_only_claim(entry):
            # A full-strength claim leaves no doubt of its own. Plain assignment: this was
            # ``setdefault`` so that a pairwise distance measured *about* this signal could not be
            # overwritten by a vote that scored nothing, and the asr axis's pairwise block was the
            # only thing that ever pre-filled ``out``. With one voter per signal there is nothing to
            # defer to, and a ``setdefault`` with no other writer reads as if there were.
            out[str(name)] = 0.0
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


@dataclass(frozen=True)
class SnrGate:
    """Per-bucket admission for perturbations that only count where the audio is degraded.

    A speech-enhancement pass is a *repair*. Above the SNR floor there is nothing for it to
    repair, so a downstream answer that changes there reports the transform, not the recording.
    This gate is what keeps such a pass out of the fold in exactly those buckets, while leaving
    it free to contribute the cross-pass ``|delta|`` that ``reliability.signal_stability`` turns
    into every signal's weight. See :data:`perturbations.SNR_GATED_TRANSFORMS` for why the gate
    is on SNR and not on whether the raw sources happened to disagree.

    Attributes:
        floor_db: SNR at or above which a gated perturbation is not admitted. From
            ``triage.snr_floor_db`` in the run config, so the fold and the run-level
            ``enhancement.mode: auto`` decision are gated on one number rather than two.
        snr_db_by_bucket: ``{(start, end) → SNR in dB}`` on the axis grid. A bucket may map to
            ``None``: SNR was not measurable there.
        gated_passes: Names of the perturbations this applies to — those whose
            ``admission_requires_low_snr`` is true. Every other pass is admitted everywhere.
    """

    floor_db: float
    snr_db_by_bucket: Mapping[tuple[float, float], float | None]
    gated_passes: frozenset[str]

    @classmethod
    def build(
        cls,
        harvests: Mapping[str, Any],
        *,
        floor_db: float,
        gated_passes: frozenset[str],
    ) -> "SnrGate | None":
        """The gate for a run, from its harvests. ``None`` when there is nothing to gate.

        **One constructor for every fold in the run.** ``compute.compute_uncertainty_axes`` and
        ``write_final_uncertainty`` both fuse the same harvests, and if they built the gate from
        different SNR fields or different floors they would publish two differently-gated numbers
        under one axis name — the class of defect this codebase keeps hitting, where two producers
        of one quantity disagree and nothing says so.

        SNR is read from the **identity** perturbation. How degraded a recording is is a fact about
        the recording; measuring it on the enhanced audio would ask the repair to certify its own
        necessity, and it would pass every time.
        """
        if not gated_passes:
            return None
        quality: Mapping[tuple[float, float], Mapping[str, Any]] = {}
        for label, harvest in harvests.items():
            if label == IDENTITY_NAME:
                quality = getattr(harvest, "quality_by_bucket", None) or {}
                break
        return cls(
            floor_db=float(floor_db),
            snr_db_by_bucket={bucket: q.get("snr_brouhaha_db") for bucket, q in quality.items()},
            gated_passes=frozenset(gated_passes),
        )

    def admits(self, perturbation: str, bucket: tuple[float, float]) -> bool:
        """May ``perturbation``'s readings for ``bucket`` enter the fold?

        An unmeasured SNR does **not** admit. The gate is the primary condition, and "we could
        not measure the degradation" is not evidence that the recording is degraded — the same
        rule the rest of this module follows in refusing to read ``None`` as ``0.0``. It is
        visible rather than silent: the bucket's ``snr_gated_passes`` column names what was held
        out, so a reader can tell a pass that was excluded from one that never ran.
        """
        if perturbation not in self.gated_passes:
            return True
        snr = self.snr_db_by_bucket.get(bucket)
        if snr is None:
            return False
        return float(snr) < float(self.floor_db)


def fuse_axis(
    buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float],
    aggregator: str = "mean",
    weight_basis: Mapping[str, Mapping[str, float]] | None = None,
    round_index: int = 0,
    snr_gate: SnrGate | None,
) -> list[dict[str, Any]]:
    """Fuse one axis's per-signal uncertainties across signals and passes.

    Args:
        buckets_by_pass: ``{perturbation → per-bucket harvested votes}``.
        snr_gate: Which perturbations may contribute in which buckets, or ``None`` for no
            gating. **Required rather than defaulted**: a caller that silently folds a
            speech-enhancement pass into buckets at 70 dB SNR publishes the transform's answer
            as the recording's, and a default would let that happen by omission. Pass ``None``
            explicitly to say the run has nothing to gate.
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
    # (start, end) → signal → [uncertainty, ...]. A signal appearing under more than one
    # perturbation contributes each reading, subject to ``snr_gate``: a repair transform is
    # evidence only where there is something to repair. The comment here used to claim that
    # "disagreement between passes is evidence too" — which the mean two loops below then threw
    # away, so the disagreement was neither reported as reducible nor kept out. It is kept out
    # now, where the gate says it does not apply, and recorded per bucket when it is.
    collected: dict[tuple[float, float], dict[str, list[float]]] = {}
    passes_seen: dict[tuple[float, float], set[str]] = {}
    gated_out: dict[tuple[float, float], set[str]] = {}

    for perturbation in sorted(buckets_by_pass):
        for bucket in buckets_by_pass[perturbation] or []:
            if not isinstance(bucket, Mapping):
                continue
            key = (round(float(bucket.get("start", 0.0)), 6), round(float(bucket.get("end", 0.0)), 6))
            if snr_gate is not None and not snr_gate.admits(str(perturbation), key):
                # Recorded, not dropped in silence: a bucket showing ``contributing_passes:
                # ['raw']`` is otherwise indistinguishable from a run that never computed the
                # perturbation at all.
                gated_out.setdefault(key, set()).add(str(perturbation))
                continue
            slot = collected.setdefault(key, {})
            passes_seen.setdefault(key, set()).add(str(perturbation))
            for signal, value in per_signal_uncertainty(bucket).items():
                slot.setdefault(signal, []).append(value)

    # A bucket every pass was gated out of still owes a row: it has no fused value, and that is a
    # measurement ("nothing was admitted here"), not an absence to be skipped over.
    for key in gated_out:
        collected.setdefault(key, {})
        passes_seen.setdefault(key, set())

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
                # What the SNR gate held out of *this* bucket. Empty is the common case and says
                # "nothing was withheld"; a name here says the pass ran and was not admitted,
                # which a shrunken ``contributing_passes`` alone cannot distinguish from a run
                # that never computed the perturbation.
                "snr_gated_passes": sorted(gated_out.get((start, end), ())),
                "signal_weights": {s: w for s, w in zip(signals, applied)},
                "weight_basis": {s: dict((weight_basis or {}).get(s, {})) for s in signals},
                "round": int(round_index),
            }
        )
    return rows


def _round_record(
    round_index: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    untried_actions: int | None,
    assignment: Mapping[str, str] | None = None,
    overwrote_values: bool = False,
) -> Any:  # noqa: ANN401
    """Summarise a fused round in the terms convergence is judged on.

    ``assignment`` comes from ``joint.per_speaker_presence`` (J4) and ``untried_actions`` from the
    intervention catalogue. Both are ``None`` when this path has neither, and ``None`` **blocks**
    the corresponding criterion rather than satisfying it — a criterion nobody measured must not
    read as one that passed.

    ``overwrote_values`` says this round *replaced* a value rather than observing it again — which
    a round that re-derived the shared structure does. C1 then refuses to credit any fall that
    followed, so the loop cannot buy itself more rounds by revising its own derivatives.
    """
    from senselab.audio.workflows.audio_analysis.rounds import RoundRecord

    # `epistemic_uncertainty`, the column `fuse_axis` actually emits. This read a per-pass axis
    # column that fuse_axis never produced, so every record carried `epistemic=None`
    # and `measured_buckets=0`, C1 had nothing to compare, and every round digested to the same
    # signature, which the shared detector then correctly reported as a repeating state. A real
    # run stopped with `oscillation` on all four axes for that reason alone: not four dynamics
    # agreeing, one name resolving to nothing on all of them.
    values = [r.get("epistemic_uncertainty") for r in rows]
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    epistemic = sum(numeric) / len(numeric) if numeric else None
    # A bucket counts as measured when the axis has a value there, which is what C3's
    # unmeasured -> measured progress check is about.
    measured = sum(1 for r in rows if isinstance(r.get("uncertainty"), (int, float)))
    digest = ";".join(f"{r.get('start')}:{r.get('uncertainty')}:{r.get('epistemic_uncertainty')}" for r in rows)
    return RoundRecord(
        round_index=round_index,
        epistemic=epistemic,
        assignment=assignment,
        measured_buckets=measured,
        untried_actions=None if untried_actions is None else int(untried_actions),
        overwrote_values=bool(overwrote_values),
        signature=hashlib.sha1(digest.encode()).hexdigest(),
    )


@dataclass(frozen=True)
class Derivatives:
    """Round outputs that are not axes — and the channel through which the axes reach each other.

    A round produces two kinds of thing. The axes are the answer; the derivatives are the shared
    structure every axis is estimated *against*: which regions the mask calls target-free, where
    each signal claimed a speaker, and (as they land) the speaker allocation, the ASR consensus and
    the scene components.

    They are round outputs rather than fixed inputs because they are themselves estimates. Computed
    once from round 0 and then held constant, every later round withdraws trust on the strength of a
    judgement the loop had already improved on — the same staleness the per-axis driver had for the
    axes themselves.

    **This is why the coupling needs no gate.** A speaker ambiguity reaches the presence axis by
    changing the speaker allocation both axes are conditioned on, not by presence averaging in a
    number the speaker axis reported. That distinction is not cosmetic: a shared latent is shared
    structure, while another axis's *value* is a second copy of the evidence that produced it, and
    counting the copy would double-count whatever signal both axes read. An earlier draft routed the
    coupling through the vote fold and needed a fixed discount to keep the extra voter from moving
    the mean — a band-aid on the wrong channel, since a discount bounds double-counting without
    removing it, and convergence cannot detect it either (a biased fixed point is still a fixed
    point).

    Attributes:
        mask_regions: Regions with ``state`` and ``confidence``.
        speaker_claims: ``{signal → spans}`` where the signal asserted a speaker.
    """

    mask_regions: tuple[Mapping[str, Any], ...] = ()
    speaker_claims: Mapping[str, Sequence[tuple[float, float]]] | None = None


def derive_mask_from_axes(
    rows_by_axis: Mapping[str, Sequence[Mapping[str, Any]]],
    current: Derivatives,
    *,
    settled_below: float = 0.35,
) -> Optional[Derivatives]:
    """Re-derive the mask from the previous round's presence and background-mask axes.

    The default coupling path. A region is called target-free where the presence axis has settled
    *and* the background-mask axis agrees it is settled there, with the region's confidence taken
    from how settled the two are. That mask then discounts, region by region, any signal claiming a
    speaker where the evidence says nobody spoke — so a presence result reaches the speaker axis
    through the structure both are conditioned on rather than as a vote.

    Only regions the previous round actually settled are emitted. A bucket where presence measured
    nothing yields no region: absence of a claim is not a claim of absence, and a mask that filled
    those in would withdraw trust on the strength of a gap.

    Args:
        rows_by_axis: Every axis's rows from the previous round.
        current: The derivatives in force, returned unchanged when nothing can be re-derived.
        settled_below: Uncertainty at or below which an axis counts as settled in a bucket. Named
            rather than inlined because it decides which regions may withdraw trust at all.

    Returns:
        Updated derivatives, or ``None`` when the previous round gave no grounds to change them —
        which is different from deriving an empty mask, and is reported as such.
    """
    presence = rows_by_axis.get("speech_presence") or []
    if not presence:
        return None
    mask_by_key = {
        (round(float(r.get("start", 0.0)), 6), round(float(r.get("end", 0.0)), 6)): r
        for r in (rows_by_axis.get("background_mask") or [])
    }
    regions: list[Mapping[str, Any]] = []
    for row in presence:
        value = row.get("uncertainty")
        if value is None:
            continue
        key = (round(float(row.get("start", 0.0)), 6), round(float(row.get("end", 0.0)), 6))
        agreeing = mask_by_key.get(key, {}).get("uncertainty")
        settled = [float(value)] + ([float(agreeing)] if agreeing is not None else [])
        if max(settled) > float(settled_below):
            continue
        regions.append(
            {
                "start": key[0],
                "end": key[1],
                "state": "target_free",
                # How settled the agreeing axes are, so a tentative mask withdraws proportionally
                # less trust — the mask's confidence already gates how far it may act.
                "confidence": max(0.0, min(1.0, 1.0 - (sum(settled) / len(settled)))),
            }
        )
    if not regions:
        return None
    return Derivatives(mask_regions=tuple(regions), speaker_claims=current.speaker_claims)


CROSS_AXIS_PASS = "__axes__"
"""Synthetic pass label carrying the previous round's axes, kept distinct from any real pass so a
value the loop computed can never be mistaken for something a microphone recorded."""


def project_axis_onto(
    source_rows: Sequence[Mapping[str, Any]],
    target_spans: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float]:
    """Project one axis's values onto another axis's buckets (H1's common lattice).

    Cross-axis input previously matched on exact ``(start, end)`` keys, which on real audio means
    never matching: the four axes carried 85 / 41 / 1070 / 1 buckets on four different grids and
    shared *zero* keys, so coupling did nothing and every round came out byte-identical to the
    last. Unit tests missed it because their fixtures put every axis on one synthetic grid — the
    one thing real data never does.

    Each target bucket takes the **overlap-weighted mean** of the source buckets it intersects.
    Weighting by overlap rather than taking the nearest bucket matters when the grids are coarse
    relative to each other: a source bucket that barely touches the target would otherwise decide
    its whole value.

    Args:
        source_rows: The contributing axis's rows, each with ``start``, ``end`` and ``uncertainty``.
        target_spans: The receiving axis's bucket spans.

    Returns:
        ``{span → value}``, omitting spans no source bucket covers. Source buckets whose
        ``uncertainty`` is ``None`` contribute nothing: that is the absence of a claim, and
        averaging it in as zero would manufacture confidence nobody expressed.
    """
    measured = [
        (float(r.get("start", 0.0)), float(r.get("end", 0.0)), float(r["uncertainty"]))
        for r in source_rows or ()
        if r.get("uncertainty") is not None
    ]
    out: dict[tuple[float, float], float] = {}
    for span in target_spans:
        lo, hi = float(span[0]), float(span[1])
        total = 0.0
        weighted = 0.0
        for s_lo, s_hi, value in measured:
            overlap = min(hi, s_hi) - max(lo, s_lo)
            if overlap > 0:
                total += overlap
                weighted += overlap * value
        if total > 0:
            out[(lo, hi)] = weighted / total
    return out


def cross_axis_inputs(
    axis: str,
    rows_by_axis: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    own_keys: Optional[set[tuple[float, float]]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The other axes' previous-round values, as inputs to estimating ``axis``.

    Every round takes all three kinds of thing the loop holds — the signals, the derivatives, and
    the previous round's axes — and re-estimates each axis from them.

    **No assigned discount.** An earlier draft multiplied these by a fixed 0.4 to keep them from
    dominating the fold. That contradicts the premise this whole module rests on: weights are
    *measured* — perturbation stability, physical support — and a factor never measured must not act
    as a discount. A hand-set constant is exactly such a factor. A cross-axis input therefore carries
    full weight like any other signal absent from the weights mapping, and is attenuated only by
    something actually measured about it.

    The residual concern the constant was standing in for is real but different: an axis's value is
    built from signals, so if two axes read the same signal, its evidence appears twice. That is a
    correlation to *measure* — the same kind of thing perturbation stability already measures — not
    a number to assume. Until it is measured, the derivatives carry the coupling that can be
    justified structurally, and this path carries the rest at face value.

    An axis never contributes to itself: its previous value is the thing being updated, not evidence
    about it. An axis that measured nothing contributes nothing, because ``None`` is the absence of
    a claim rather than a low uncertainty.

    **Coupling informs an axis's grid; it never extends it.** Restricted to ``own_keys``, because
    on a real run the mask contributed one whole-clip region and the fused ``background_mask`` axis
    emitted 1197 buckets — every one of them sourced from the other axes' finer grid. An axis
    holding one datum has nothing to contribute back, so it can only echo, and in the output an
    echo is indistinguishable from a measurement. The evidence-overlap gate does not catch this
    either: overlap is measured on signal *names*, and a lone axis's name collides with nothing, so
    a fully-derived axis reads as fully independent.

    Args:
        axis: The axis being estimated.
        rows_by_axis: Every axis's rows from the previous round.
        own_keys: Bucket spans this axis measured itself. Buckets outside it are dropped; ``None``
            keeps every bucket, for callers that have no grid of their own to respect.

    Returns:
        ``(buckets, contributing_axes)``, both sorted, so the fixed point cannot depend on the order
        the axes happen to be visited (FR-011f).
    """
    by_key: dict[tuple[float, float], dict[str, dict[str, float]]] = {}
    contributors: set[str] = set()
    # The receiver's own grid is the lattice both axes meet on: coupling informs it and never
    # extends it, so an axis holding one datum cannot acquire another axis's 1197 buckets.
    targets = sorted(own_keys) if own_keys is not None else []
    for other in sorted(rows_by_axis):
        if other == axis:
            continue
        projected = project_axis_onto(rows_by_axis[other] or [], targets)
        for key, value in projected.items():
            by_key.setdefault(key, {})[f"axis::{other}"] = {"same_label_uncertainty": float(value)}
            contributors.add(other)
    buckets = [{"start": s, "end": e, "votes": by_key[(s, e)]} for s, e in sorted(by_key) if by_key[(s, e)]]
    return buckets, sorted(contributors)


def measure_axis_overlap(
    target_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    """Fraction of one axis's evidence the other already holds.

    The measured successor to a deleted constant. A cross-axis input is only partly new
    information: an axis's value is built from signals, so where two axes read the same signal its
    evidence enters twice. How much is *measurable* — it is the overlap of the two axes'
    contributing signals — and measuring it is the difference between a gate this codebase accepts
    and one it has already been burned by.

    ``speaker_identity`` learned this first. Its gate is ``claim.support``, a measured quantity,
    adopted after a *declared* source-kind gate down-weighted the one source that matched the
    speaker names actually spoken on a recording. A hand-set multiplier here was that same
    construct in a different module.

    Signals contributed by a previous round's coupling are excluded from the source's evidence.
    Counting them would let the measure feed on its own output and drift with every round.

    Args:
        target_rows: Rows of the axis being estimated.
        source_rows: Rows of the axis contributing.

    Returns:
        Overlap in ``[0, 1]``, or ``None`` when the source contributed no evidence of its own —
        which applies **no** discount, because a factor never measured must not act as one.
    """
    target = {s for r in target_rows or () for s in (r.get("contributing_signals") or ())}
    source = {
        s for r in source_rows or () for s in (r.get("contributing_signals") or ()) if not str(s).startswith("axis::")
    }
    if not source:
        return None
    return len(target & source) / len(source)


def fuse_axes(
    buckets_by_axis: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    weights_by_axis: Mapping[str, Mapping[str, float]],
    aggregator: str = "mean",
    weight_basis_by_axis: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    mask_regions: Sequence[Mapping[str, Any]] = (),
    speaker_claims: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    max_rounds: int = 1,
    tolerance: float = 1e-3,
    speaker_assignment_by_axis: Mapping[str, Mapping[str, str] | None] | None = None,
    untried_actions: int | None = None,
    derive: Any = derive_mask_from_axes,  # noqa: ANN401 — (rows_by_axis, Derivatives) -> Derivatives | None
    couple_axes: bool = True,
    remeasure: Any = None,  # noqa: ANN401 — (axis, regions, rows_by_axis) -> {pass: buckets} | None
    unsettled_above: float = 0.6,
    return_history: bool = False,
    snr_gate: SnrGate | None,
) -> Any:  # noqa: ANN401 — 2-tuple, or 3-tuple when return_history
    """Iterate every axis together, so each round can read the others (D-11).

    Each round asks whether the axes should change, given everything the loop knows: the signals,
    and the previous round's axes and derivatives. Concretely, a round re-derives the derivatives
    from the previous round's axes, then re-estimates every axis from its own signals conditioned on
    those derivatives. Reading the *previous* round throughout means the result cannot depend on the
    order the axes happen to be visited.

    **The derivatives are how the axes reach each other.** A speaker ambiguity changes the presence
    answer by changing the shared structure — the mask, the speaker allocation — that both axes are
    estimated against, never by one axis averaging in a number another axis reported. Routing it
    through the vote fold instead needs an arbitrary discount to stop the extra voter moving the
    mean, and still double-counts whatever signal both axes read; conditioning on a shared latent
    has neither problem.

    Two things the per-axis driver could not do, and the reason this exists:

    - **The categories are coupled.** A speaker ambiguity is frequently a presence ambiguity, and
      D-7 makes speaker and presence explicitly joint. Running one axis to completion before
      starting the next means a region settled on one can never reach another, so the coupling was
      structurally unable to act however it was configured.
    - **Regional trust stays local.** Where the mask contradicts a signal's speaker claim, that
      signal is discounted *in that region only*. A global discount for a local failure is what
      suppressed the source that was right about the five named speakers.

    The driver takes any number of axes. The design names four — ``speech_presence``, ``speaker``,
    ``asr``, ``background_mask`` — with ``task`` punted, and an axis count baked into the loop is a
    fact that would need re-finding every time the set changes.

    Args:
        buckets_by_axis: ``{axis → {perturbation → L1 buckets}}``.
        weights_by_axis: ``{axis → {signal → measured weight}}``.
        aggregator: Aggregator for ``triage_score``.
        weight_basis_by_axis: ``{axis → {signal → {factor → value}}}``.
        mask_regions: Mask regions with ``state`` and ``confidence``, for regional trust.
        speaker_claims: ``{signal → spans}`` where the signal asserted a speaker.
        max_rounds: Cap on iterations.
        tolerance: Per-bucket change below which a round counts as no change, and the credited
            epistemic change below which C1 holds.
        speaker_assignment_by_axis: ``{axis → binding}`` from ``joint.per_speaker_presence``, for
            C2. An axis without one leaves C2 unmeasured, which blocks convergence for that axis
            rather than passing it.
        untried_actions: Remaining unattempted actions, for C4, from a caller whose inventory is
            wider than this loop's — the adaptive catalogue can re-run models over a region, which
            is invisible here. Omit and the loop counts its own action set instead.
        derive: ``(rows_by_axis, current) → Derivatives | None``, called before each round after
            the first. Defaults to :func:`derive_mask_from_axes`; pass ``None`` to freeze the
            derivatives at what round 0 was handed. A hook returning ``None`` leaves them in force,
            and each round's log records ``derivatives_refreshed`` so a stale judgement cannot look
            like a current one.
        couple_axes: Whether the previous round's *other* axes are inputs. Setting both this to
            ``False`` and ``derive`` to ``None`` runs several axes fully isolated, which has to stay
            reachable: a coupling that cannot be turned off cannot be evaluated against anything.
        remeasure: ``(axis, regions, rows_by_axis) → {perturbation: buckets} | None`` (D-10). Called
            once per axis per round with the regions that axis has left unsettled, so a round may
            *re-measure* rather than only re-weight — re-weighting can only redistribute evidence
            already gathered, and a region no re-weighting resolves is exactly the one that needs a
            finer look. Buckets it returns join that axis's inputs from the next fold on. A region
            is offered once: the same finer look repeated is not new evidence, and C4 would never
            reach zero. The hook never learns what a model is; it is offered a region and its
            answer is spent.
        unsettled_above: Uncertainty above which a bucket is offered for re-measurement. Named
            because it decides what gets looked at again, and defaulting it silently would make the
            loop's spending invisible.

    Convergence is reached when no axis changes, or when the loop enters a periodic one. Both are
    judged per axis by ``rounds.assess_convergence``: C1 and C3 cover "nothing moved" — the values
    holding still *and* no bucket going unmeasured → measured — while a repeating state is caught by
    the shared non-convergence detector and reported as ``oscillation`` rather than agreement. A
    round whose derivatives were re-derived records ``overwrote_values``, so C1 declines to credit
    a fall the loop produced by revising its own shared structure.

    **C4's inventory is this loop's own, and the log says so.** Working from a fixed harvest, this
    loop can withdraw regional trust and re-examine an axis against the others; both are spent the
    moment a later round runs, so from round 1 the count is a *measured* zero rather than a
    defaulted one. That distinction is the whole of C4 — never having looked must not read as
    having checked — but a measured zero here is still the narrow claim "this loop ran out of
    moves", not the wide one "no further measurement would help". ``action_scope`` names which
    inventory was counted so the two cannot be conflated.

        return_history: Also return ``{axis → {round → rows}}``. The per-round maps under
            ``L2/round<N>/`` are supposed to show what each iteration changed; without this the
            caller only ever had the final rows, every one carrying the final round index, so the
            writer emitted a single directory and the trail the layout promises did not exist.

        snr_gate: Passed to every :func:`fuse_axis` fold this driver performs, including the
            coupled re-folds of later rounds, so a perturbation held out of round 0 stays held out
            of round 3. ``None`` for no gating.

    Returns:
        ``({axis → rows}, {axis → log})``, plus ``{axis → {round → rows}}`` when
        ``return_history``.
    """
    from senselab.audio.workflows.audio_analysis.rounds import (
        assess_convergence,
        regional_weights,
        round_converged,
    )

    axes = sorted(buckets_by_axis)
    assignments = dict(speaker_assignment_by_axis or {})
    basis_by_axis = dict(weight_basis_by_axis or {})

    derived = Derivatives(mask_regions=tuple(mask_regions or ()), speaker_claims=speaker_claims)

    per_region: dict[tuple[float, float], dict[str, float]] = {}
    regional_by_axis: dict[str, dict[str, float]] = {}

    def _apply_derivatives(state: Derivatives) -> None:
        # Recomputed whenever the derivatives change, because regional trust is a function *of*
        # them: holding the weights while the mask moved would keep withdrawing trust on the
        # strength of a judgement a later round had already improved on.
        per_region.clear()
        regional_by_axis.clear()
        if not state.mask_regions or not state.speaker_claims:
            return
        for axis in axes:
            weights = weights_by_axis.get(axis, {})
            computed = regional_weights(
                base_weights=dict(weights), regions=state.mask_regions, claims=state.speaker_claims
            )
            per_region.update(computed)
            # The tightest regional weight covering each signal, so a signal contradicted anywhere
            # it spoke is attenuated for the fold rather than silently rescued by a region where it
            # stayed quiet.
            regional_by_axis[axis] = {
                signal: min((w.get(signal, 1.0) for w in computed.values()), default=weights.get(signal, 1.0))
                for signal in weights
            }

    _apply_derivatives(derived)

    # This loop's own action inventory (C4): withdrawing regional trust where it would actually
    # change a weight, plus re-examining each axis against the others. A region that contradicts
    # nothing offers no action, so counting it would manufacture work the loop cannot do.
    regional_actions = sum(
        1
        for axis in axes
        for region_weights in ([per_region[k] for k in per_region] if regional_by_axis.get(axis) else [])
        if any(region_weights.get(s, w) < w for s, w in weights_by_axis.get(axis, {}).items())
    )
    # Two coupling actions: re-deriving the shared structure, and reading the other axes.
    use_axes = bool(couple_axes) and len(axes) > 1
    can_rederive = derive is not None and len(axes) > 1
    can_remeasure = remeasure is not None
    # Regions already offered to the hook, per axis. Offering one twice would let C4 count an
    # action that has in fact been spent, so "converged" could never be reached.
    offered: dict[str, set[tuple[float, float]]] = {axis: set() for axis in axes}

    def _pending(axis: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows or ():
            value = row.get("uncertainty")
            if value is None or float(value) <= float(unsettled_above):
                continue
            key = (round(float(row.get("start", 0.0)), 6), round(float(row.get("end", 0.0)), 6))
            if key in offered[axis]:
                continue
            out.append({"start": key[0], "end": key[1], "uncertainty": float(value)})
        return out

    caller_supplied = untried_actions is not None
    if caller_supplied:
        action_scope = "caller_supplied"
    elif can_remeasure:
        action_scope = "remeasure"
    elif can_rederive or use_axes:
        action_scope = "cross_axis"
    else:
        action_scope = "regional_trust"

    def _untried(applied: bool) -> Optional[int]:
        # The caller's wider inventory wins where given; otherwise the count is this loop's own,
        # and its actions are spent the moment a later round runs.
        if caller_supplied:
            return untried_actions
        if applied:
            return 0
        pending = sum(len(_pending(axis, rows_by_axis.get(axis) or [])) for axis in axes) if can_remeasure else 0
        if applied:
            return pending
        return regional_actions + (1 if can_rederive else 0) + (1 if use_axes else 0) + pending

    # Working copy: re-measurement adds inputs for later rounds without mutating the caller's.
    inputs: dict[str, dict[str, Sequence[Mapping[str, Any]]]] = {a: dict(buckets_by_axis[a]) for a in axes}
    rows_by_axis: dict[str, list[dict[str, Any]]] = {}
    logs: dict[str, list[dict[str, Any]]] = {axis: [] for axis in axes}
    history: dict[str, list[Any]] = {}
    stopped: dict[str, bool] = {axis: False for axis in axes}

    per_round: dict[str, dict[int, list[dict[str, Any]]]] = {axis: {} for axis in axes}
    for axis in axes:
        rows_by_axis[axis] = fuse_axis(
            buckets_by_axis[axis],
            weights=weights_by_axis.get(axis, {}),
            aggregator=aggregator,
            weight_basis=basis_by_axis.get(axis),
            round_index=0,
            snr_gate=snr_gate,
        )
        for row in rows_by_axis[axis]:
            row["coupled_from"] = []
        per_round[axis][0] = [dict(r) for r in rows_by_axis[axis]]
        history[axis] = [
            _round_record(
                0, rows_by_axis[axis], untried_actions=_untried(applied=False), assignment=assignments.get(axis)
            )
        ]
        logs[axis].append(
            {
                "round": 0,
                "buckets": len(rows_by_axis[axis]),
                "converged": False,
                "regional_trust_applied": False,
                "action_scope": action_scope,
            }
        )

    can_iterate = bool(regional_by_axis) or can_rederive or use_axes or can_remeasure
    if not can_iterate:
        for axis in axes:
            # The loop stops because it *cannot iterate*, which is not the four-criteria verdict
            # this field carries everywhere else. Flagged rather than left to look like one.
            logs[axis][-1]["converged"] = True
            logs[axis][-1]["criteria_evaluated"] = False
            logs[axis][-1]["reason"] = "nothing to re-derive and nothing to localise trust against"
        if return_history:
            return rows_by_axis, logs, per_round
        return rows_by_axis, logs

    for round_index in range(1, max(1, int(max_rounds))):
        # Read from the previous round for every axis, so the result cannot depend on the order the
        # axes are visited within this one.
        previous = {axis: list(rows_by_axis[axis]) for axis in axes}
        # Derivatives are round *outputs*, so a round that can re-derive them does so before using
        # them. Without this the mask and the speaker claims stay frozen at whatever round 0
        # produced, and every later round withdraws trust on the strength of a stale judgement.
        derivatives_refreshed = False
        if derive is not None:
            refreshed = derive(previous, derived)
            if refreshed is not None and refreshed != derived:
                derived = refreshed
                _apply_derivatives(derived)
                derivatives_refreshed = True
        for axis in axes:
            if stopped[axis]:
                continue
            # D-10: offer this axis's unsettled regions a finer look before re-folding it.
            remeasured = False
            if can_remeasure:
                regions = _pending(axis, previous[axis])
                if regions:
                    for region in regions:
                        offered[axis].add((region["start"], region["end"]))
                    refined = remeasure(axis, regions, previous)
                    if refined:
                        inputs[axis] = {**inputs[axis], **{str(k): v for k, v in refined.items()}}
                        remeasured = True
            weights = dict(regional_by_axis.get(axis) or weights_by_axis.get(axis, {}))
            folded = dict(inputs[axis])
            # All three inputs the loop holds: this axis's signals, the derivatives (through the
            # regional weights above), and the previous round's axes.
            own_keys = {
                (round(float(r.get("start", 0.0)), 6), round(float(r.get("end", 0.0)), 6)) for r in previous[axis]
            }
            extra, from_axes = cross_axis_inputs(axis, previous, own_keys=own_keys) if use_axes else ([], [])
            basis = dict(basis_by_axis.get(axis) or {})
            if extra:
                folded[CROSS_AXIS_PASS] = extra
                for other in from_axes:
                    overlap = measure_axis_overlap(previous[axis], previous[other])
                    if overlap is None:
                        continue
                    # The uncertainty gate is deliberately left open: the quantity a cross-axis
                    # input carries *is* the other axis's uncertainty, so discounting it for being
                    # uncertain would suppress exactly the informative case.
                    weights[f"axis::{other}"] = effective_weight(
                        1.0, uncertainty=0.0, derivation_gate=1.0 - float(overlap)
                    )
                    basis[f"axis::{other}"] = {"evidence_overlap": float(overlap)}
            candidate = fuse_axis(
                folded,
                weights=weights,
                aggregator=aggregator,
                weight_basis=basis,
                round_index=round_index,
                snr_gate=snr_gate,
            )
            # Which axes reached this one: directly as inputs, and — when the shared structure was
            # re-derived this round — through the derivatives every axis contributed to. Recorded
            # per row so a coupled value is distinguishable from one reached on this axis's own
            # evidence.
            coupled = sorted({*from_axes, *(a for a in axes if a != axis)} if derivatives_refreshed else from_axes)
            for row in candidate:
                row["coupled_from"] = list(coupled) if row.get("contributing_signals") else []
            numbers_settled = round_converged(rows_by_axis[axis], candidate, tolerance=tolerance)
            rows_by_axis[axis] = candidate
            # Snapshot before the next round overwrites it: the copy is what makes the per-round
            # maps comparable rather than N references to one mutated list.
            per_round[axis][round_index] = [dict(r) for r in candidate]
            history[axis].append(
                _round_record(
                    round_index,
                    candidate,
                    untried_actions=_untried(applied=True),
                    assignment=assignments.get(axis),
                    # A coupled round *replaced* a value rather than observing it again, so the
                    # self-confirmation guard must refuse to credit any fall that followed.
                    overwrote_values=bool(derivatives_refreshed or from_axes or remeasured),
                )
            )
            verdict = assess_convergence(history[axis], tolerance=tolerance, max_rounds=max(1, int(max_rounds)))
            logs[axis].append(
                {
                    "round": round_index,
                    "buckets": len(candidate),
                    # Kept separate on purpose: the numbers holding still is one of four criteria,
                    # and reporting it as convergence is what let a round stop while the assignment
                    # was still flipping or a region still had an untried action.
                    "numbers_settled": numbers_settled,
                    "converged": verdict["converged"],
                    "blocking": verdict["blocking"],
                    "credited_epistemic_change": verdict["credited_epistemic_change"],
                    "diverged": verdict["diverged"],
                    "stop_reason": verdict["stop_reason"],
                    # Which round states traded places, when they did. A bare "oscillation" says the
                    # loop failed to settle without saying between what, which is the part an
                    # operator needs to know whether the disagreement is resolvable.
                    "repeating_states": verdict["repeating_states"],
                    "regional_trust_applied": bool(regional_by_axis.get(axis)),
                    "coupled_from": list(coupled),
                    # Whether this round re-derived the mask and speaker claims, or reused round
                    # 0's. A reader cannot otherwise tell a refreshed judgement from a stale one.
                    "derivatives_refreshed": derivatives_refreshed,
                    # Whether this round took a finer look rather than only re-weighting (D-10).
                    "remeasured": remeasured,
                    # Which action inventory C4 was answered against. A measured zero here says this
                    # loop ran out of moves, not that no further measurement would help.
                    "action_scope": action_scope,
                }
            )
            if verdict["stop"]:
                stopped[axis] = True
        if all(stopped.values()):
            break
    if return_history:
        return rows_by_axis, logs, per_round
    return rows_by_axis, logs


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
    speaker_assignment: Mapping[str, str] | None = None,
    untried_actions: int | None = None,
    snr_gate: SnrGate | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One axis on its own — :func:`fuse_axes` with a single axis and no re-derivation.

    A wrapper rather than a second implementation: two round loops over the same criteria could
    reach opposite verdicts on identical history, which is the duplication that already had to be
    undone once for non-convergence detection.

    Args:
        buckets_by_pass: ``{perturbation → L1 buckets}``.
        weights: Round-0 per-signal weights.
        aggregator: Aggregator for ``triage_score``.
        weight_basis: Per-signal factor breakdown.
        mask_regions: Mask regions with ``state`` and ``confidence``, used for regional trust.
        speaker_claims: ``{signal → spans}`` where the signal asserted a speaker.
        max_rounds: Cap on iterations.
        tolerance: Per-bucket change below which a round counts as no change.
        speaker_assignment: The ``S_k`` → channel binding, for C2.
        untried_actions: Remaining unattempted actions, for C4.

        snr_gate: Which perturbations may contribute in which buckets; see :class:`SnrGate`.
            ``None`` for no gating.

    Returns:
        ``(rows, log)`` for the single axis.
    """
    axis = "_"
    rows_by_axis, logs = fuse_axes(
        {axis: buckets_by_pass},
        weights_by_axis={axis: weights},
        aggregator=aggregator,
        weight_basis_by_axis={axis: weight_basis} if weight_basis is not None else None,
        mask_regions=mask_regions,
        speaker_claims=speaker_claims,
        max_rounds=max_rounds,
        tolerance=tolerance,
        speaker_assignment_by_axis={axis: speaker_assignment},
        untried_actions=untried_actions,
        derive=None,
        snr_gate=snr_gate,
    )
    return rows_by_axis[axis], logs[axis]


def _speaker_assignment(harvests: Mapping[str, Any]) -> Optional[dict[str, str]]:
    """The speaker → tool-label binding for the reference pass, or ``None`` when unmeasurable.

    Measured on the unmodified pass where available: whether a speaker occupies a channel is a fact
    about the recording, not about the enhancement transform, the same reasoning physical support
    already uses.

    Returns ``None`` rather than an empty mapping when the inputs are missing, because an empty
    binding and an unmeasured one mean different things to C2 — two empty mappings compare equal
    and would read as a stable assignment nobody checked.
    """
    from senselab.audio.workflows.audio_analysis.identity_binding import bind_labels
    from senselab.audio.workflows.audio_analysis.joint import speaker_spans_from_votes
    from senselab.audio.workflows.audio_analysis.occupancy import spans_from_diarization

    harvest = harvests.get("raw") or next(iter(harvests.values()), None)
    if harvest is None:
        return None
    spans = speaker_spans_from_votes(getattr(harvest, "speaker_votes", None) or [])
    if not spans:
        return None
    diar = spans_from_diarization(getattr(harvest, "diarization_by_model", None) or {})
    if not diar:
        return None
    # One entry per (tool, speaker), because the binding is now per tool: each diarizer has its own
    # label namespace, so there is no single channel index to bind to. C2 therefore judges a *set*
    # of bindings for stability, which is strictly more information than the channel version had —
    # and strictly more ways for a round to differ, so C2 blocks convergence more readily than
    # before. That is the honest direction to err in: the old single binding could hold still while
    # two diarizers disagreed about who was whom, and report that as stability.
    out: dict[str, str] = {}
    for tool in sorted(diar):
        binding = bind_labels(spans, diar[tool])
        if binding is None:
            continue
        for speaker, label in sorted(binding["assignment"].items()):
            if label is not None:
                out[f"{tool}:{speaker}"] = label
    return out or None


def _draw_round_timeline(
    out_dir: Any,  # noqa: ANN401 — Path
    round_index: int,
    axis_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    duration_s: float,
) -> None:
    """Draw ``L2/round/<n>/timeline.png``. Best-effort: a plot must not fail a fold."""
    import sys

    from senselab.audio.workflows.audio_analysis.l2_plot import build_round_timeline

    try:
        build_round_timeline(out_dir, round_index=round_index, axis_rows=axis_rows, duration_s=duration_s)
    except Exception as exc:  # noqa: BLE001 — sidecar
        print(f"warn: round {round_index} timeline plot failed: {exc!r}", file=sys.stderr)


def mask_axis_votes(mask_regions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The mask's own per-**region** confidence, restated as votes in the shared units.

    **Not the ``background_mask`` axis's vote source.** That axis is harvested per bucket like the
    other three (``mask_harvest.harvest_background_mask_evidence`` →
    ``PassHarvest.background_mask_evidence``, read through ``votes.buckets_for_axis``), and using
    this function for it is what gave a run 1070 mask buckets at round 0 and one by round 4: a
    region is as coarse as the mask happens to be, so on a recording with a single region the axis
    had nowhere to be uncertain.

    What survives is the region-scoped reading, for a consumer whose unit genuinely *is* a region.
    ``rounds.regional_weights`` is that consumer, and it takes its regions from
    :func:`mask_regions_from_rows` rather than from here — so as of this change nothing in the
    pipeline calls this function, and it is kept as the one place that states how a region's
    confidence converts into the axes' units.

    An ``indeterminate`` region is skipped rather than voted at maximum uncertainty. "I cannot tell"
    is the absence of a claim, and the other axes already treat an absent claim as absent rather
    than as a confident maximum — imputing one here would let an unresolved region outvote the
    regions that were actually resolved.

    Args:
        mask_regions: Regions with ``start``, ``end``, ``state`` and ``confidence``.

    Returns:
        Bucket dicts in the shape :func:`fuse_axis` consumes, in time order.
    """
    votes: list[dict[str, Any]] = []
    for region in mask_regions or ():
        state = str(region.get("state") or "indeterminate")
        confidence = region.get("confidence")
        if state == "indeterminate" or not isinstance(confidence, (int, float)):
            continue
        votes.append(
            {
                "start": float(region.get("start", 0.0)),
                "end": float(region.get("end", 0.0)),
                "votes": {"mask": {"same_label_uncertainty": max(0.0, min(1.0, 1.0 - float(confidence)))}},
            }
        )
    return sorted(votes, key=lambda v: (v["start"], v["end"]))


def mask_regions_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise ``background_mask.parquet`` rows into the shape regional trust expects.

    The mask reports ``uncertainty``; ``rounds.regional_weights`` and :func:`mask_axis_votes`
    both read ``confidence``. Passing the rows through unconverted is not a cosmetic mismatch —
    ``region.get("confidence", 1.0)`` would default to *fully confident*, so a mask that was
    unsure would withdraw the maximum trust it is capable of withdrawing. That is the
    absent-reads-as-fine failure this design has hit repeatedly, and it is worth one conversion
    function to close.

    A row with no measured uncertainty is dropped rather than assumed certain, for the same
    reason: a mask that never said how sure it was has not earned the right to act.

    Args:
        rows: Mask rows with ``start``, ``end``, ``state`` and ``uncertainty``.

    Returns:
        Regions carrying ``confidence``, in time order.
    """
    out: list[dict[str, Any]] = []
    for row in rows or ():
        uncertainty = row.get("uncertainty")
        if not isinstance(uncertainty, (int, float)):
            continue
        out.append(
            {
                "start": float(row.get("start", 0.0)),
                "end": float(row.get("end", 0.0)),
                "state": str(row.get("state") or "indeterminate"),
                "confidence": max(0.0, min(1.0, 1.0 - float(uncertainty))),
            }
        )
    return sorted(out, key=lambda r: (r["start"], r["end"]))


def speaker_claims_from_votes(
    speaker_votes: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    """``{model → spans}`` where that model actually named a speaker.

    Regional trust discounts a signal *where it made a claim the mask contradicts*, so the claim
    has to be a real one. A model reporting silence has claimed nothing, and treating that as a
    claim would let the mask discount a model for agreeing with it.

    Args:
        speaker_votes: Per-bucket dicts with ``start``, ``end`` and ``votes``.

    Returns:
        Spans per model, in time order, with models that named nobody omitted entirely.
    """
    from senselab.audio.workflows.audio_analysis.harvesters import SILENCE_LABEL

    claims: dict[str, list[tuple[float, float]]] = {}
    for bucket in speaker_votes or ():
        votes = bucket.get("votes")
        if not isinstance(votes, Mapping):
            continue
        span = (float(bucket.get("start", 0.0)), float(bucket.get("end", 0.0)))
        for model, entry in votes.items():
            if not isinstance(entry, Mapping):
                continue
            label = entry.get("speaker_label")
            if not label or str(label) == SILENCE_LABEL:
                continue
            claims.setdefault(str(model), []).append(span)
    return {m: sorted(spans) for m, spans in sorted(claims.items())}


def fold_run_axes(
    buckets_by_axis: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    speaker_assignment: Optional[Mapping[str, str]] = None,
    weights_by_axis: Mapping[str, Mapping[str, float]],
    aggregator: str = "mean",
    weight_basis_by_axis: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
    mask_regions: Sequence[Mapping[str, Any]] = (),
    speaker_claims: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    max_rounds: int = 1,
    scene_rows: Sequence[Mapping[str, Any]] = (),
    comparator_params: Mapping[str, Any] | None = None,
    snr_gate: SnrGate | None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[int, list[dict[str, Any]]]],
]:
    """The run's **one** fold: aggregate every axis, then apply the scene→asr coupling.

    Extracted from ``write_final_uncertainty``, which used to do both — and that is why two artifacts
    of one run could disagree about the same row. ``fuse_axes`` had two callers, each producing a
    user-visible artifact: this fold fed ``estimates/*.parquet``, and a *separate* fold inside
    ``fuse_rounds`` fed the ``fused_axes`` that ``disagreements.json`` ranks. Two computations wearing
    one name, so any divergence in their arguments surfaced as the parquet and the index reporting
    different numbers for the same bucket — with the parquet being the one a reader keeps.

    The coupling belongs here rather than in the writer for the same reason: it *mutates* the folded
    rows, so a writer that applied it produced rows no other consumer had seen.

    Returns:
        ``(rows_by_axis, logs, per_round)`` — the final rows, the per-axis convergence log (keyed by
        axis, each a list of per-round entries), and the per-round history.

        Hand the same triple to the writer *and* to the disagreements index. A consumer that re-folds
        is reintroducing the second lineage this function exists to remove.
    """
    rows_by_axis, logs, per_round = fuse_axes(
        buckets_by_axis,
        # C2 is a claim about the speaker axis specifically; handing the same binding to the other
        # axes would report a criterion as measured on axes it says nothing about.
        speaker_assignment_by_axis={"speaker": speaker_assignment} if speaker_assignment else None,
        weights_by_axis=weights_by_axis,
        aggregator=aggregator,
        weight_basis_by_axis=weight_basis_by_axis,
        mask_regions=mask_regions,
        speaker_claims=speaker_claims,
        max_rounds=max_rounds,
        return_history=True,
        snr_gate=snr_gate,
    )
    if scene_rows and "asr" in per_round:
        from senselab.audio.workflows.audio_analysis.votes import apply_scene_coupling

        for rows in per_round["asr"].values():
            apply_scene_coupling(rows, scene_rows, comparator_params or {})
    return rows_by_axis, logs, per_round


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
    speech_presence_policy: Any = None,  # noqa: ANN401 — SpeechPresencePolicy, imported lazily
    scene_rows: Sequence[Mapping[str, Any]] = (),
    comparator_params: Mapping[str, Any] | None = None,
    snr_floor_db: float,
    snr_gated_passes: frozenset[str],
) -> dict[str, Any]:
    """Write ``L2/round/<n>/estimates/<axis>.parquet`` for every round the fold ran.

    One directory per round, because a reader needs to see what each iteration changed and not
    only where it ended up — a single map cannot distinguish "settled immediately" from
    "moved a long way and then settled".

    ``estimates`` rather than ``uncertainty``: each row carries uncertainty, epistemic
    uncertainty, confidence and variability, so the old name named one column rather than the
    thing itself.

    These are the maps a consumer should read. ``L1/signals/`` is level-1 evidence: what each
    signal said, in its own units, before anything was measured about its reliability.

    Args:
        out_dir: Run directory.
        harvests: ``{perturbation → PassHarvest}``.
        weights_by_axis: ``{axis → {signal → measured weight}}``.
        aggregator: Aggregator for the ``triage_score`` fold.
        weight_basis_by_axis: ``{axis → {signal → {factor → value}}}``, so a discounted signal
            records which factor discounted it.
        mask_regions: Mask regions, enabling regional trust in rounds after the first.
        speaker_claims: ``{signal → spans}`` where each signal asserted a speaker.
        max_rounds: Cap on L2 iterations.
        speech_presence_policy: Policy used to read L1 speech-presence measurements as beliefs.
            Defaults to the documented anchors. Fusion needs verdicts, and the harvest stores
            measurements, so the link has to happen here rather than being assumed done.
        scene_rows: Presence rows carrying the scene measurements, for the scene→asr coupling
            (FR-019). Empty disables it. It has to be applied *here*, per round: these rounds
            re-fold every axis from the harvests, so a coupling applied by the caller beforehand
            was overwritten before any row reached disk — leaving ``scene_quality_coupling`` on a
            ``triage_score`` that did not contain it.
        comparator_params: Comparator params, for the coupling weights.

        snr_floor_db: SNR below which a repair perturbation is admitted to the fold, from
            ``triage.snr_floor_db``.
        snr_gated_passes: Names of the perturbations whose readings only count where the recording
            is degraded. Built into a gate via :meth:`SnrGate.build`, the same constructor
            ``compute_uncertainty_axes`` uses, so both folds of these harvests are gated alike.

    Returns:
        ``{axis → written path}`` as strings, for the run summary.
    """
    from pathlib import Path

    import pandas as pd

    from senselab.audio.workflows.audio_analysis.axes import HARVEST_SOURCES
    from senselab.audio.workflows.audio_analysis.speech_presence_link import DEFAULT_POLICY
    from senselab.audio.workflows.audio_analysis.votes import buckets_for_axis

    policy = speech_presence_policy if speech_presence_policy is not None else DEFAULT_POLICY
    # C2 asks whether the speaker-to-channel binding is stable, so it has to be measured before the
    # rounds that judge it. Unavailable on a pass with no per-speaker channels or no harmonised
    # clusters, in which case it stays None and C2 blocks rather than passing.
    speaker_assignment = _speaker_assignment(harvests)
    # Every harvested axis, read where its own declaration says its evidence lives (D-17), through
    # the one reader all three consumers share. The fourth axis used to be bolted on after a
    # hand-written map of three from ``mask_axis_votes(mask_regions)`` — one vote per *region*, so on
    # a run that found a single region the whole recording was one bucket, the axis had nowhere to be
    # uncertain, and it reported 0.000 across the board. On the presence grid it shares (D-24) it is
    # one row per bucket like everything else. ``mask_axis_votes`` keeps its region-scoped callers —
    # ``rounds.regional_weights`` withdrawing trust regionally, and the per-region mask export —
    # because a *region* is the right unit for both. What was wrong was using it as a vote source.
    buckets_by_axis: dict[str, Mapping[str, Sequence[Mapping[str, Any]]]] = {
        axis: {label: buckets_for_axis(h, axis, policy=policy) for label, h in harvests.items()}
        for axis in HARVEST_SOURCES
    }
    from senselab.audio.workflows.audio_analysis.io import merge_json
    from senselab.audio.workflows.audio_analysis.layout import estimates_dir, round_dir

    rows_by_axis, logs, per_round = fold_run_axes(
        buckets_by_axis,
        snr_gate=SnrGate.build(harvests, floor_db=snr_floor_db, gated_passes=snr_gated_passes),
        speaker_assignment=speaker_assignment,
        weights_by_axis=weights_by_axis,
        aggregator=aggregator,
        weight_basis_by_axis=weight_basis_by_axis,
        mask_regions=mask_regions,
        speaker_claims=speaker_claims,
        max_rounds=max_rounds,
        scene_rows=scene_rows,
        comparator_params=comparator_params,
    )

    written: dict[str, Any] = {}

    def _frame(axis: str, rows: Sequence[Mapping[str, Any]], *, round_index: int) -> Any:  # noqa: ANN401 — DataFrame
        # Through ``estimates.estimate_frame``, the declaration the adaptive loop's rounds are
        # also written through: this directory holds one trajectory whose early rounds this
        # function writes and whose later ones the belief store does, so a column list local to
        # either producer is two shapes under one artifact name.
        #
        # ``round`` is the directory's, stamped by the declaration; the fold's own index rides on
        # ``last_refolded_round``, which is what tells a reader whether this round recomputed the
        # value or inherited it.
        return estimate_frame(
            axis,
            [
                {
                    "start": r["start"],
                    "end": r["end"],
                    "uncertainty": r["uncertainty"],
                    "epistemic_uncertainty": r["epistemic_uncertainty"],
                    "confidence": r["confidence"],
                    "variability": r["variability"],
                    "triage_score": r["triage_score"],
                    "last_refolded_round": r["round"],
                    "contributing_signals": r["contributing_signals"],
                    "contributing_passes": r["contributing_passes"],
                    "signal_weights": json.dumps(r["signal_weights"], sort_keys=True),
                    "weight_basis": json.dumps(r["weight_basis"], sort_keys=True),
                    # Which other axes moved this value (D-11). Without it a coupled row is
                    # indistinguishable from one this axis reached on its own evidence.
                    "coupled_from": r.get("coupled_from") or [],
                    # The scene→asr coupling and the value it multiplied, so the adjustment can be
                    # undone or disagreed with from the parquet alone.
                    "scene_quality_coupling": r.get("scene_quality_coupling"),
                    "triage_score_pre_coupling": r.get("triage_score_pre_coupling"),
                }
                for r in rows
            ],
            round_index=round_index,
        )

    # One directory per round, from the per-round history rather than the final rows. Writing the
    # final rows N times would have produced N identical directories claiming to be a trajectory;
    # deriving the round set from the rows' own ``round`` field produced exactly one, because every
    # final row carries the final index. Neither shows what an iteration changed.
    #
    # The rounds **the run** took, not the rounds each axis took. Axes stop independently — a
    # converged one is skipped by every later round — so a directory set derived per axis left
    # round 2 of a three-round fold holding three axes out of four, and "the speaker axis was never
    # asked" and "the speaker axis settled in round 1 and its estimate still stands" were written
    # the same way: not at all. The second is what happened, so the last fold is *carried forward*
    # and stamped with the directory's round, while ``last_refolded_round`` keeps saying which round
    # produced the numbers. An empty file would have said the opposite of both — a round does not
    # stop believing an axis because it stopped re-folding it.
    run_rounds = sorted({index for rounds_for_axis in per_round.values() for index in rounds_for_axis})
    # Resolved once, for the parquets *and* the figures: a round whose file has four axes and whose
    # picture has three is the same gap moved somewhere a reader is less likely to check.
    # An axis that folded no round at all resolves to the empty table rather than to no file —
    # absent is the one thing this may not produce, because absent is read as "never asked".
    believed: dict[int, dict[str, Sequence[Mapping[str, Any]]]] = {index: {} for index in run_rounds}
    for axis, rounds_for_axis in sorted(per_round.items()):
        carried: Sequence[Mapping[str, Any]] = []
        for round_index in run_rounds:
            carried = rounds_for_axis.get(round_index, carried)
            believed[round_index][axis] = carried

    for axis in sorted(per_round):
        for round_index in run_rounds:
            dest = estimates_dir(out_dir, round_index)
            dest.mkdir(parents=True, exist_ok=True)
            round_path = dest / f"{axis}.parquet"
            _frame(axis, believed[round_index][axis], round_index=round_index).to_parquet(round_path, index=False)
            written[f"{axis}@round{round_index}"] = str(round_path)
            # The headline path is the last round the axis actually ran — the fold, not the last
            # directory that carries it, because that is the round whose numbers these are.
            if round_index in per_round[axis]:
                written[axis] = str(round_path)

    # One figure per round, drawn here from the rows this function already holds. The driver used
    # to do it, by reading every parquet it had just been handed the paths of — so a caller of the
    # workflow API got rounds with no view of themselves, which is a third of what a round owes.
    # ``duration_s`` from the rows rather than from a caller: the figure spans what was measured,
    # and a length nobody measured is not something this function should invent.
    for round_index in run_rounds:
        rows_for_round = believed[round_index]
        span = max(
            (float(row["end"]) for rows in rows_for_round.values() for row in rows),
            default=0.0,
        )
        _draw_round_timeline(out_dir, round_index, rows_for_round, duration_s=span)

    # The round log distinguishes "converged" from "ran out of rounds", which the maps alone
    # cannot say and which call for different follow-up. It is per round, so it goes in each
    # round's own ``summary.json`` rather than in one ``L2/rounds.json`` at the belief root —
    # where it had no round to belong to, and where it left rounds 0 and 1 of every run with no
    # account of what they did at all while the adaptive loop wrote summaries for its own.
    # Merged rather than written, because the loop has its own block to add for the round it
    # adopts as a baseline, and a second write would erase this one.
    for round_index in sorted({int(entry["round"]) for entries in logs.values() for entry in entries}):
        merge_json(
            round_dir(out_dir, round_index) / "summary.json",
            {"fusion": {axis: e for axis, entries in logs.items() for e in entries if int(e["round"]) == round_index}},
        )
        written[f"summary@round{round_index}"] = str(round_dir(out_dir, round_index) / "summary.json")
    # Handed back rather than left for the driver to read out of the tree: the headline summary
    # wants this log, and the only other way to get it was to open the file that has just been
    # split across the rounds it belongs to.
    written["round_logs"] = logs
    # The final rows, so a consumer can rank *what was written* rather than re-deriving something
    # close to it. The disagreements index used to rank the round-0 fold that feeds this function's
    # ``scene_rows``, so the index and the parquet described different rounds of the same run under
    # one name — and the round each described was nowhere stated.
    written["final_rows"] = rows_by_axis
    return written
