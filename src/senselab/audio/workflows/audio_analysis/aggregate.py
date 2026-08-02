"""Per-axis aggregators — fold a bucket's vote dict into one ``[0, 1]`` uncertainty."""

from __future__ import annotations

import math
import sys
from itertools import combinations  # used by aggregate_asr for pairwise WER
from typing import Any, Mapping

# Surface-level differences (case + punctuation + repeated whitespace) are
# stripped before pairwise WER so the asr axis reflects *semantic*
# disagreement rather than surface noise. The canonical normalizer moved to the
# task layer (architecture-review T049) so task- and workflow-level WER share
# one definition; re-exported under the historical name for existing importers.
from senselab.audio.tasks.speech_to_text_evaluation.utils import (
    normalize_transcript_for_wer as _normalize_transcript_for_wer,
)
from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator

__all__ = [
    "_normalize_transcript_for_wer",
    "aggregate_speaker",
    "aggregate_speech_presence",
    "aggregate_asr",
    "mean_token_entropy",
    "speech_presence_p_voice",
]


# ── speech_presence ──────────────────────────────────────────────────────────


def _evidence_factor(weights: Mapping[str, float] | None, source: str) -> float:
    """Per-source evidence weight from ``weights``, or 1.0 when this source was never measured.

    Absent means *unmeasured*, so it must not act as a discount: mapping a missing entry to
    anything below 1.0 would let a factor nobody gathered decide the fold.
    """
    if not weights:
        return 1.0
    raw = weights.get(source)
    if raw is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


def _weighted_p_voice(votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None) -> float | None:
    """Weighted mean per-voter probability of voice for one bucket, or ``None``.

    Each voter maps ``(speaks, native_confidence)`` to a per-voter voice
    probability, then contributes with its optional ``weight`` (default 1.0):

    - ``native_confidence`` ``c`` with ``speaks=True`` → ``p = c``;
      with ``speaks=False`` → ``p = 1 - c``.
    - No ``native_confidence`` → ``p = 1.0`` if ``speaks`` else ``0.0``.
    - ``hallucinated`` → ``p = 0.1`` (vote against voice).

    ``weight`` lets a caller demote coarse voters (whole-window scene tags,
    per-segment no-speech probability, sentence-level ASR) on fine reporting
    grids without dropping them (FR-014). When every weight is 1.0 (the
    default) this is the plain mean, so existing outputs are unchanged.

    ``weights`` is the *second*, independent factor: how far this source's claim was corroborated
    by evidence measured about it (``belief.VoteStore.evidence_weights``). The two multiply and
    stay separately recoverable — the payload keeps what the link layer decided about the voter's
    coarseness, the map keeps what a later round measured about its corroboration. A source absent
    from ``weights`` was not measured and keeps its payload weight untouched.
    """
    num = 0.0
    den = 0.0
    for source, v in votes.items():
        if not isinstance(v, dict) or "speaks" not in v:
            continue
        speak_val = v.get("speaks")
        if speak_val is None:
            continue
        try:
            weight = float(v.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        # The payload weight may legitimately be zero — that is policy declaring a voter
        # inapplicable on this grid. Attenuation cannot reach zero (its floor is > 0), so this
        # guard never erases a corroboration-weighted vote.
        if weight <= 0:
            continue
        weight *= _evidence_factor(weights, str(source))
        raw_nc = v.get("native_confidence")
        nc: float | None
        if raw_nc is None:
            nc = None
        else:
            try:
                nc = max(0.0, min(1.0, float(raw_nc)))
            except (TypeError, ValueError):
                nc = None
        if v.get("hallucinated"):
            p_voter = 0.1
        elif nc is None:
            p_voter = 1.0 if speak_val else 0.0
        else:
            p_voter = nc if speak_val else (1.0 - nc)
        num += weight * p_voter
        den += weight
    if den <= 0:
        return None
    return num / den


def aggregate_speech_presence(
    votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None
) -> float | None:
    """Calibrated "is voice present?" uncertainty in ``[0, 1]``.

    The speech_presence question is binary, but the goal is *not* to measure
    disagreement among voters — it's how decisively the evidence supports a
    conclusion. Uncertainty = ``1 − |2 · p_voice − 1|``: 0 when all evidence
    agrees (either way), 1 at a perfect 50/50 split. Whether voice is more
    likely present or absent is recoverable from ``p_voice`` itself
    (``speech_presence_p_voice``); this metric only grades decisiveness. See
    ``_weighted_p_voice`` for the per-voter math (weights default to 1.0, so
    this matches the historical unweighted behavior).
    """
    p_voice = _weighted_p_voice(votes, weights=weights)
    if p_voice is None:
        return None
    return max(0.0, min(1.0, 1.0 - abs(2.0 * p_voice - 1.0)))


def speech_presence_p_voice(
    votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None
) -> float | None:
    """Return the calibrated probability of voice ``p_voice`` for one bucket.

    Same per-voter math as ``aggregate_speech_presence`` but returns the raw
    probability rather than the symmetric uncertainty. Used both as the
    speech_presence-axis ``speech_presence_confidence`` column and to MASK speaker /
    asr buckets where we are confident there is no speech.
    """
    return _weighted_p_voice(votes, weights=weights)


# ── speaker ──────────────────────────────────────────────────────────


def aggregate_speaker(
    votes: dict[str, dict[str, Any]],
    *,
    raw_vs_enh: bool | None,
    aggregator: str,
    reliability: Mapping[str, float] | None = None,
    evidence_weights: Mapping[str, float] | None = None,
) -> float | None:
    """Combine speaker sub-signals into a single uncertainty in ``[0, 1]``.

    Three sub-signal families are folded via ``--uncertainty-aggregator``:

    1. ``same_label_uncertainty`` (one per ``(diar, emb)`` pair): calibrated
       uncertainty about a "same speaker as prior bucket-on-this-track" claim.
    2. ``change_inconsistency_uncertainty`` (one per ``(diar, emb)`` pair when
       the diar model claimed a change): calibrated uncertainty that the audio
       supports the change.
    3. ``__cross_diar_label_disagreement__.value``: fraction of diar-model pairs
       that disagree on this bucket's cluster (after embedding-based clustering).

    Plus ``raw_vs_enh`` (only on raw_vs_enhanced parquets): 0.0 / 1.0 for
    cross-pass label disagreement.

    Pairs / signals that are ``None`` (no prior to validate, both sides silent,
    same window dedup, etc.) drop out of the aggregator per FR-007 — never
    zero-imputed.

    Two independent per-signal factors multiply into the aggregator weight: ``reliability``
    (measured by perturbation across passes) and ``evidence_weights`` (measured by corroboration
    in this bucket). They answer different questions — "does this signal agree with itself?" and
    "does anything else support what it claims here?" — so they compose rather than replace one
    another, and both floors keep the product visible.
    """
    sub_signals: list[float | None] = []
    weights: list[float] = []
    rel = dict(reliability or {})
    ev = dict(evidence_weights or {})

    def _add(value: float, signal: str) -> None:
        sub_signals.append(float(value))
        weights.append(float(rel.get(signal, 1.0)) * _evidence_factor(ev, signal))

    for name, v in votes.items():
        if not isinstance(v, dict):
            continue
        same = v.get("same_label_uncertainty")
        if same is not None:
            _add(same, str(name))
        chg = v.get("change_inconsistency_uncertainty")
        if chg is not None:
            _add(chg, str(name))

    cross = votes.get("__cross_diar_label_disagreement__")
    if isinstance(cross, dict):
        cross_val = cross.get("value")
        if cross_val is not None:
            _add(cross_val, "__cross_diar_label_disagreement__")

    if raw_vs_enh is not None:
        # A cross-pass label flip is the perturbation result itself, not a signal whose
        # stability could be measured — it carries full weight by construction.
        _add(1.0 if raw_vs_enh else 0.0, "__raw_vs_enhanced__")

    # Either map alone must reach the aggregator. Gating on `rel` only (as this did) silently
    # discarded evidence weights on every run without a reliability measurement.
    return apply_aggregator(sub_signals, aggregator, weights=weights if (rel or ev) else None)


# ── asr ─────────────────────────────────────────────────────────


_DEFAULT_TOKEN_ENTROPY_REFERENCE_NATS = 3.0
"""Entropy (nats) treated as fully uncertain when normalizing to ``[0, 1]``.

Deliberately *not* ``log(vocab)``: Whisper's vocabulary is ~51.9k tokens, so
``log(vocab) ≈ 10.9`` nats, while real per-token entropies run ~0.05–0.5 nats when
the decoder is confident and ~2–4 nats when it is guessing. Normalizing by
``log(vocab)`` would compress every observed value into the bottom tenth of the
range, making the sub-signal invisible under ``mean`` aggregation. 3.0 nats puts
"confident" near 0 and "guessing" near 1. Override per-deployment via
``calibration["token_entropy_reference_nats"]``.
"""


def _axis_temperature(calibration: dict[str, Any] | None, axis: str) -> float:
    """Temperature for ``axis`` from a calibration profile; 1.0 (speaker) by default.

    ``temperature`` may be a per-axis mapping (``{"asr": 1.5}``) or a bare
    scalar applied to every axis. A non-positive or unparsable value falls back to
    1.0 rather than silently inverting the mapping.
    """
    if not calibration:
        return 1.0
    raw = calibration.get("temperature")
    if isinstance(raw, dict):
        raw = raw.get(axis)
    try:
        temperature = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    return temperature if temperature > 0 else 1.0


def mean_token_entropy(votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None) -> float | None:
    """Mean per-model token entropy in nats, collapsing per-token lists to their mean.

    ``weights`` supplies an optional per-source evidence weight (see
    :func:`_weighted_p_voice`); sources absent from it contribute unweighted.
    """
    num = 0.0
    den = 0.0
    for key, vote in votes.items():
        if key.startswith("__") or not isinstance(vote, dict):
            continue
        raw = vote.get("token_entropy")
        if raw is None:
            continue
        value: float | None = None
        if isinstance(raw, (list, tuple)):
            values = []
            for item in raw:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    continue
            if values:
                value = sum(values) / len(values)
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None
        if value is None:
            continue
        w = _evidence_factor(weights, str(key))
        num += w * value
        den += w
    if den <= 0:
        return None
    return num / den


def aggregate_asr(
    votes: dict[str, dict[str, Any]],
    *,
    aggregator: str,
    calibration: dict[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
) -> float | None:
    """Combine asr sub-signals into a single uncertainty.

    Per-*sub-signal* reliability weighting is deliberately not applied here, and that argument
    still holds: the sub-signals below are already model-fused when they reach this function — a
    pairwise mean over ASR sources, a mean log-probability across backends — so there is no
    sub-signal left whose perturbation stability could be weighed. The speaker axis, where
    sub-signals stay per-model and where a single saturated signal was demonstrably overriding
    unanimous agreement, is weighted that way (see ``aggregate_speaker``).

    ``weights`` is a different quantity and enters at a different place: a **per-source** evidence
    weight, applied *inside* each per-model fold rather than to the fused result. It reaches the
    pairwise family (via the pair weight), the log-probability mean and the token-entropy mean, so
    a source whose claim independent evidence does not corroborate is attenuated everywhere it
    speaks rather than in two places out of three. Sources absent from ``weights`` were never
    measured and contribute unweighted.

    A weighted mean over a *single* source is that source's own value, so attenuating the only
    ASR in a bucket does not move this axis. That is the honest answer: with one witness there is
    no disagreement to reweigh, and manufacturing one would be the unmeasured constant this
    module exists to avoid.

    Three sub-signal families (the third added by FR-017):

    1. **Pairwise phoneme edit-distance rate** across all available phoneme
       sources in this bucket — the 4 ASR transcripts (post-g2p_en, with
       phoneme-midpoint distribution across MMS-aligned word timestamps) and
       the PPG argmax sequence. With 4 ASRs + PPG that's up to ``C(5, 2)=10``
       pairwise comparisons per bucket, each normalized to ``[0, 1]``.
       Sources with no phonemes in this bucket are dropped (don't contribute
       spurious 1.0 distances). The aggregator collapses the surviving pairs
       per ``--uncertainty-aggregator`` (default ``min`` — worst-case wins).
    2. ``1 − exp(avg_logprob / T)`` averaged across ASRs that expose ``avg_logprob``
       (Whisper today). Reflects the model's self-confidence. Independent of
       the pairwise comparisons. ``T`` is the calibration temperature (FR-018),
       1.0 by default, which reproduces the historical mapping exactly.
    3. **Normalized token entropy** (FR-017) — mean per-token softmax entropy over
       the contributing models, divided by
       ``calibration["token_entropy_reference_nats"]`` (default
       ``_DEFAULT_TOKEN_ENTROPY_REFERENCE_NATS``) and clipped to ``[0, 1]``. Unlike
       the pairwise family this fires when a *single* model is internally unsure,
       so it catches doubt that transcript agreement hides. Absent (``None``) for
       every backend that doesn't report token logits, in which case the fold
       degrades to families 1 and 2 exactly.

    Args:
        votes: The bucket's vote dict from ``harvest_asr_votes``.
        aggregator: One of ``AGGREGATORS`` — how the sub-signals are collapsed.
        calibration: Optional calibration profile supplying ``temperature`` and
            ``token_entropy_reference_nats``. ``None`` ⇒ documented defaults, which
            preserve the pre-calibration numbers bit-for-bit.
        weights: Optional per-source evidence weights (see the note above). ``None`` or an empty
            map reproduces the unweighted fold exactly.

    Returns:
        The bucket's asr uncertainty in ``[0, 1]``, or ``None`` when no
        sub-signal was available.
    """
    sub_signals: list[float | None] = []
    temperature = _axis_temperature(calibration, "asr")

    # Pairwise phoneme distances (the dominant asr signal). Each pair
    # is weighted by the joint confidence of its two sources — high-confidence
    # ASR/PPG pairs dominate, while pairs involving an uncertain transcript
    # contribute proportionally less. The weighted mean is folded as a single
    # sub-signal (the "uncertainty over what was said" headline number); the
    # individual sub-signals below capture orthogonal aspects.
    pair_block = votes.get("__pairwise_phoneme_distances__")
    if isinstance(pair_block, dict):
        pairs = pair_block.get("pairs") or {}
        per_source_conf = pair_block.get("per_source_confidence") or {}

        def _conf(src: str) -> float:
            c = per_source_conf.get(src)
            if c is None:
                # Neutral full trust when source has no confidence info —
                # using 0.5 here would systematically downweight pairs
                # involving text-only ASRs that don't expose logprobs (3 of 4
                # ASR backends), letting Whisper-pairs dominate the weighted
                # mean. 1.0 keeps the weighting equitable when sources lack
                # confidence; only sources that actively report low confidence
                # get downweighted.
                return 1.0
            try:
                return max(0.0, min(1.0, float(c)))
            except (TypeError, ValueError):
                return 1.0

        weighted_sum = 0.0
        weight_total = 0.0
        for pair_key, dist in pairs.items():
            if dist is None:
                continue
            try:
                d = float(dist)
            except (TypeError, ValueError):
                continue
            # pair_key is "<source_a>|<source_b>".
            try:
                src_a, src_b = pair_key.split("|", 1)
            except ValueError:
                continue
            w = _conf(src_a) * _conf(src_b) * _evidence_factor(weights, src_a) * _evidence_factor(weights, src_b)
            weighted_sum += w * d
            weight_total += w
        if weight_total > 0:
            sub_signals.append(weighted_sum / weight_total)

    # Whisper self-confidence (separate sub-signal class).
    alp_num = 0.0
    alp_den = 0.0
    for src, v in votes.items():
        if not isinstance(v, dict) or v.get("avg_logprob") is None:
            continue
        w = _evidence_factor(weights, str(src))
        alp_num += w * float(v["avg_logprob"])
        alp_den += w
    if alp_den > 0:
        try:
            mean_alp = alp_num / alp_den
            # Temperature-scaled so backends with differently-sharp logprob
            # distributions land on a common [0,1] scale (FR-018). T=1 is the
            # historical mapping.
            confidence = max(0.0, min(1.0, math.exp(mean_alp / temperature)))
            sub_signals.append(1.0 - confidence)
        except (ValueError, OverflowError, ZeroDivisionError):
            pass

    # MMS-CTC alignment scores are recorded on the parquet for diagnostic
    # inspection (see ``alignment_ctc_score`` in each ASR vote) but are NOT
    # aggregated as a sub-signal: the aligner's path posterior given a
    # (possibly hallucinated) transcript doesn't reflect transcript
    # correctness — it reflects path quality conditional on the transcript.
    # Using it as asr uncertainty would mask hallucinated transcripts
    # rather than expose them.

    # PPG argmax confidence — per-bucket model confidence in its top-1
    # phoneme decode. Uncertainty = 1 − mean argmax probability.
    ppg_conf = votes.get("__ppg_argmax_confidence__")
    if isinstance(ppg_conf, dict) and ppg_conf.get("value") is not None:
        try:
            sub_signals.append(max(0.0, min(1.0, 1.0 - float(ppg_conf["value"]))))
        except (ValueError, TypeError):
            pass

    # Token-level entropy (FR-017) — a single model's private doubt, which
    # transcript agreement cannot reveal.
    mean_entropy = mean_token_entropy(votes, weights=weights)
    if mean_entropy is not None:
        reference = _DEFAULT_TOKEN_ENTROPY_REFERENCE_NATS
        if calibration:
            try:
                candidate = float(calibration.get("token_entropy_reference_nats", reference))
                if candidate > 0:
                    reference = candidate
            except (TypeError, ValueError):
                pass
        sub_signals.append(max(0.0, min(1.0, mean_entropy / reference)))

    return apply_aggregator(sub_signals, aggregator)
