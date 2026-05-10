"""Per-axis aggregators — fold a bucket's vote dict into one ``[0, 1]`` uncertainty."""

from __future__ import annotations

import math
import re
import sys
from itertools import combinations  # used by aggregate_utterance for pairwise WER
from typing import Any

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator

# Surface-level differences (case + punctuation + repeated whitespace) that we
# strip before pairwise WER so the utterance axis reflects *semantic* disagreement
# rather than surface noise. ``"first."`` and ``"first!"`` both normalize to
# ``"first"``; ``"I"`` and ``"i"`` both normalize to ``"i"``.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_transcript_for_wer(text: str) -> str:
    """Lowercase, strip non-word punctuation, collapse whitespace."""
    if not text:
        return ""
    cleaned = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    return _WHITESPACE_PATTERN.sub(" ", cleaned).strip()


# ── presence ──────────────────────────────────────────────────────────


def aggregate_presence(votes: dict[str, dict[str, Any]]) -> float | None:
    """Calibrated "is voice present?" uncertainty in ``[0, 1]``.

    The presence question is binary, but the goal is *not* to measure
    disagreement among voters (Shannon entropy did that) — it's to tell the
    user whether voice is present and how strongly the evidence supports
    that conclusion. A 6-of-8 split should read as "moderate evidence for
    voice", not "high uncertainty".

    Algorithm:

    1. For each voter, derive a per-voter probability of voice:

       - With ``native_confidence`` ``c`` and ``speaks=True``: ``p = c``
         (the model says "voice with confidence c").
       - With ``native_confidence`` ``c`` and ``speaks=False``: ``p = 1 - c``
         (the model says "silence with confidence c", so voice prob is ``1-c``).
       - Without ``native_confidence``: ``p = 1.0`` if ``speaks=True``, else
         ``0.0`` (binary vote — the model is fully committed either way).

    2. ``p_voice = mean(per-voter probabilities)`` — equal weight per voter.
       Models with a ``native_confidence`` that aren't sure pull ``p_voice``
       toward 0.5 even when their boolean ``speaks`` flips one way; binary-only
       voters get equal weight to confidence-bearing ones.

    3. Uncertainty = ``1 − |2 · p_voice − 1|``: 0 when all evidence agrees
       (either way), 1 at a perfect 50/50 split. Whether voice is more likely
       present or absent is recoverable from ``p_voice`` itself; this metric
       only grades how decisive the evidence is.
    """
    p_voice_per_voter: list[float] = []

    for v in votes.values():
        if not isinstance(v, dict):
            continue
        if "speaks" not in v:
            continue
        speak_val = v.get("speaks")
        if speak_val is None:
            continue
        raw_nc = v.get("native_confidence")
        nc: float | None
        if raw_nc is None:
            nc = None
        else:
            try:
                nc = max(0.0, min(1.0, float(raw_nc)))
            except (TypeError, ValueError):
                nc = None
        # ASR hallucination flag: the voter said "speaks" but the model's own
        # silence head says otherwise — treat as a vote AGAINST voice instead
        # of for it (reflecting that the transcript came from a hallucination,
        # not actual audio content).
        if v.get("hallucinated"):
            p_voter = 0.1  # near-zero with a hint of caution rather than full 0
        elif nc is None:
            p_voter = 1.0 if speak_val else 0.0
        else:
            p_voter = nc if speak_val else (1.0 - nc)
        p_voice_per_voter.append(p_voter)

    if not p_voice_per_voter:
        return None
    p_voice = sum(p_voice_per_voter) / len(p_voice_per_voter)
    return max(0.0, min(1.0, 1.0 - abs(2.0 * p_voice - 1.0)))


def presence_p_voice(votes: dict[str, dict[str, Any]]) -> float | None:
    """Return the calibrated probability of voice ``p_voice`` for one bucket.

    Same per-voter math as ``aggregate_presence`` (each voter's
    ``(speaks, native_confidence)`` mapped to a ``p_voice`` contribution then
    averaged), but returns the raw probability instead of the symmetric
    uncertainty derived from it. Used to MASK identity / utterance buckets
    where we are confident there is no speech: when ``p_voice`` is near zero,
    the speaker / transcript axes are not meaningfully measurable for that
    bucket and their per-bucket uncertainty is downweighted.
    """
    p_voice_per_voter: list[float] = []
    for v in votes.values():
        if not isinstance(v, dict):
            continue
        if "speaks" not in v:
            continue
        speak_val = v.get("speaks")
        if speak_val is None:
            continue
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
        p_voice_per_voter.append(p_voter)
    if not p_voice_per_voter:
        return None
    return sum(p_voice_per_voter) / len(p_voice_per_voter)


# ── identity ──────────────────────────────────────────────────────────


def aggregate_identity(
    votes: dict[str, dict[str, Any]],
    *,
    raw_vs_enh: bool | None,
    aggregator: str,
) -> float | None:
    """Combine identity sub-signals into a single uncertainty in ``[0, 1]``.

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
    """
    sub_signals: list[float | None] = []

    for v in votes.values():
        if not isinstance(v, dict):
            continue
        same = v.get("same_label_uncertainty")
        if same is not None:
            sub_signals.append(float(same))
        chg = v.get("change_inconsistency_uncertainty")
        if chg is not None:
            sub_signals.append(float(chg))

    cross = votes.get("__cross_diar_label_disagreement__")
    if isinstance(cross, dict):
        cross_val = cross.get("value")
        if cross_val is not None:
            sub_signals.append(float(cross_val))

    if raw_vs_enh is not None:
        sub_signals.append(1.0 if raw_vs_enh else 0.0)

    return apply_aggregator(sub_signals, aggregator)


# ── utterance ─────────────────────────────────────────────────────────


def aggregate_utterance(votes: dict[str, dict[str, Any]], *, aggregator: str) -> float | None:
    """Combine utterance sub-signals into a single uncertainty.

    Two sub-signal families:

    1. **Pairwise phoneme edit-distance rate** across all available phoneme
       sources in this bucket — the 4 ASR transcripts (post-g2p_en, with
       phoneme-midpoint distribution across MMS-aligned word timestamps) and
       the PPG argmax sequence. With 4 ASRs + PPG that's up to ``C(5, 2)=10``
       pairwise comparisons per bucket, each normalized to ``[0, 1]``.
       Sources with no phonemes in this bucket are dropped (don't contribute
       spurious 1.0 distances). The aggregator collapses the surviving pairs
       per ``--uncertainty-aggregator`` (default ``min`` — worst-case wins).
    2. ``1 − exp(avg_logprob)`` averaged across ASRs that expose ``avg_logprob``
       (Whisper today). Reflects the model's self-confidence. Independent of
       the pairwise comparisons.
    """
    sub_signals: list[float | None] = []

    # Pairwise phoneme distances (the dominant utterance signal). Each pair
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
            w = _conf(src_a) * _conf(src_b)
            weighted_sum += w * d
            weight_total += w
        if weight_total > 0:
            sub_signals.append(weighted_sum / weight_total)

    # Whisper self-confidence (separate sub-signal class).
    avg_logprobs = [
        v["avg_logprob"] for v in votes.values() if isinstance(v, dict) and v.get("avg_logprob") is not None
    ]
    if avg_logprobs:
        try:
            mean_alp = sum(avg_logprobs) / len(avg_logprobs)
            confidence = max(0.0, min(1.0, math.exp(mean_alp)))
            sub_signals.append(1.0 - confidence)
        except (ValueError, OverflowError):
            pass

    # MMS-CTC alignment scores are recorded on the parquet for diagnostic
    # inspection (see ``alignment_ctc_score`` in each ASR vote) but are NOT
    # aggregated as a sub-signal: the aligner's path posterior given a
    # (possibly hallucinated) transcript doesn't reflect transcript
    # correctness — it reflects path quality conditional on the transcript.
    # Using it as utterance uncertainty would mask hallucinated transcripts
    # rather than expose them.

    # PPG argmax confidence — per-bucket model confidence in its top-1
    # phoneme decode. Uncertainty = 1 − mean argmax probability.
    ppg_conf = votes.get("__ppg_argmax_confidence__")
    if isinstance(ppg_conf, dict) and ppg_conf.get("value") is not None:
        try:
            sub_signals.append(max(0.0, min(1.0, 1.0 - float(ppg_conf["value"]))))
        except (ValueError, TypeError):
            pass

    return apply_aggregator(sub_signals, aggregator)
