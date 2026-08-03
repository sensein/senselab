"""Utterance axis vote harvesters — "what was said?".

Per FR-002, asr uncertainty integrates two sub-signals per bucket:

1. **ASR pairwise mean WER** — among contributing ASR transcripts on the bucket.
2. **Whisper native** — ``1 − exp(avg_logprob)`` averaged over Whisper chunks.

A third, **ASR-vs-PPG PER**, is gone with the ``ppgs`` signal: the ASR outputs are good enough that
a phoneme posteriorgram was not earning its cost. What survives is the *cross-ASR* pairwise phoneme
distance, which never involved PPG — it compares two recognizers' g2p sequences with each other.

This harvester emits per-ASR votes; aggregation downstream computes the three
sub-signals and folds them via ``--uncertainty-aggregator``.
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    _levenshtein,
    asr_alignment_score_in_window,
    asr_phoneme_sequence_in_window,
    asr_text_in_window,
    mean_token_entropy_in_window,
    resolve_asr_result,
    whisper_bucket_avg_logprob,
)


def harvest_asr_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    alignment_by_model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the asr axis.

    ``votes`` is a dict ``{asr_model_id → {"text": str, "avg_logprob": float | None,
    "phoneme_sequence": list[str], ...}}``. ``avg_logprob`` is shipped as the raw
    scalar (negative); the aggregator converts it to ``1 − exp(...)`` for the
    uncertainty sub-signal so reviewers can read the original from the parquet.

    Two asr-specific rules:

    - **ASR text per bucket uses fully-contained chunks only** (``fully_contained=True``).
      Words straddling a bucket boundary contribute to NEITHER side — partial words
      were inflating the WER on every boundary. Pair this with a wider+overlapping
      asr grid (recommended: 1.0 s window with 0.5 s hop) so most words still
      land inside at least one bucket.
    - **Phoneme sequences are English-gated.** ``g2p_en`` maps English text to ARPAbet and
      nothing else, so for non-English transcripts the pairwise distances would be noise.
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    asr_blocks = (pass_summary.get("asr") or {}).get("by_model") or {}
    asr_ok = {m: b for m, b in asr_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    asr_resolved = {m: resolve_asr_result(b, alignment_by_model.get(m)) for m, b in asr_ok.items()}

    # Detect transcript language to gate the phoneme sub-signal. ``g2p_en`` maps English text to
    # ARPAbet and nothing else, so for non-English output the pairwise distances are meaningless
    # (mostly 1.0). The gate outlived the PPG signal it was written for, because the constraint was
    # never PPG's inventory — it was g2p_en's. Whisper exposes ``language`` on each segment when
    # detection ran; we honour any model's explicit ``language``, and default to English otherwise.
    transcript_languages: set[str] = set()
    for resolved in asr_resolved.values():
        items = resolved if isinstance(resolved, list) else [resolved]
        for line in items:
            lang = getattr(line, "language", None)
            if lang is None and isinstance(line, dict):
                lang = line.get("language")
            if lang:
                transcript_languages.add(str(lang).lower()[:2])
    phoneme_signal_enabled = not transcript_languages or "en" in transcript_languages
    if not phoneme_signal_enabled:
        import sys as _sys

        print(
            f"warn: asr phoneme sub-signal skipped — transcript language(s) "
            f"{sorted(transcript_languages)} are non-English and g2p_en maps English only; "
            "the pairwise distances would be meaningless edit distances",
            file=_sys.stderr,
        )

    from itertools import combinations

    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, dict[str, Any]] = {}
        # Per-source phoneme sequences: g2p_en distributed across each model's word timestamps,
        # with phoneme-midpoint inclusion. One entry per ASR that produced any.
        per_source_phoneme_seq: dict[str, list[str]] = {}

        for m, resolved in asr_resolved.items():
            text = asr_text_in_window(resolved, start, end, fully_contained=True)
            avg_logprob = _avg_logprob_in_window(resolved, start, end)
            # MMS-CTC alignment posterior — mean per-character score for the
            # alignment leaves overlapping this bucket. Reflects how confident
            # the Wav2Vec2-CTC aligner was in its character-level path through
            # the trellis. Only present for text-only ASRs that went through
            # MMS alignment (Granite, Canary, Qwen3); Whisper natively
            # timestamps and skips alignment so this returns None for it.
            ctc_score = asr_alignment_score_in_window(resolved, start, end)
            asr_phon_seq: list[str] = []
            if phoneme_signal_enabled:
                asr_phon_seq = asr_phoneme_sequence_in_window(resolved, start, end, fully_contained=False)
            if asr_phon_seq:
                per_source_phoneme_seq[m] = asr_phon_seq
            votes[m] = {
                "text": text,
                "phoneme_sequence": asr_phon_seq,
                "avg_logprob": avg_logprob,
                "alignment_ctc_score": ctc_score,
                # Per-token softmax entropy (FR-017) — the model's private doubt,
                # which transcript agreement cannot reveal. None for every backend
                # that doesn't expose token logits.
                "token_entropy": mean_token_entropy_in_window(resolved, start, end),
            }

        # Pairwise phoneme edit-distance rate across the ASRs that produced a sequence
        # (4 ASRs → up to C(4,2)=6 distances per bucket). Each
        # distance is normalized by the longer sequence length, clipped to
        # [0,1]. Sources with no phonemes in this bucket are excluded from
        # the pairwise grid (they'd contribute spurious 1.0 distances against
        # everything else, drowning out real disagreement).
        sources = sorted(per_source_phoneme_seq.keys())
        pair_distances: dict[str, float] = {}
        for a, b in combinations(sources, 2):
            seq_a = per_source_phoneme_seq[a]
            seq_b = per_source_phoneme_seq[b]
            distance = _levenshtein(seq_a, seq_b)
            denom = max(len(seq_a), len(seq_b))
            if denom > 0:
                pair_distances[f"{a}|{b}"] = min(1.0, distance / denom)
        # Per-source confidences for weighting the pairwise distances.
        # Only TRUE per-source confidences participate in pairwise weighting:
        # Whisper avg_logprob → exp(). The MMS-CTC
        # alignment_ctc_score is NOT used as a confidence proxy because it
        # measures the aligner's path posterior given a (possibly hallucinated)
        # transcript, not the model's confidence in the transcript itself —
        # using it as a confidence weight would systematically reward
        # confident hallucinations. It is recorded on the parquet for
        # diagnostic inspection but doesn't drive aggregation.
        # Sources without a real confidence default to weight 1.0 (neutral
        # full trust) — this keeps Whisper from dominating the weighted mean
        # when 3 of 4 ASRs have no logprob signal.
        import math as _math

        per_source_conf: dict[str, float] = {}
        for m, v in votes.items():
            if not isinstance(v, dict):
                continue
            alp = v.get("avg_logprob")
            if alp is not None:
                try:
                    per_source_conf[m] = max(0.0, min(1.0, _math.exp(float(alp))))
                except (ValueError, OverflowError):
                    pass
        votes["__pairwise_phoneme_distances__"] = {
            "pairs": pair_distances,
            "n_sources": len(sources),
            "sources": sources,
            "per_source_confidence": per_source_conf,
        }
        out.append({"start": start, "end": end, "votes": votes})
    return out


def _avg_logprob_in_window(result: Any, win_start: float, win_end: float) -> float | None:  # noqa: ANN401
    """Mean per-chunk avg_logprob over chunks overlapping the window.

    Returns the raw avg_logprob (negative) so the parquet preserves the native
    scalar. The aggregator computes ``1 − exp(avg_logprob)`` to obtain a confidence
    in [0, 1]. Uses ``whisper_bucket_avg_logprob`` directly (averaging logprobs)
    rather than round-tripping through ``log(mean(exp(x)))``, which would bias the
    result high (Jensen's inequality).
    """
    return whisper_bucket_avg_logprob(result, win_start, win_end)
