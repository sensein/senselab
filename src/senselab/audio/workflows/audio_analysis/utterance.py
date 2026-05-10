"""Utterance axis vote harvesters — "what was said?".

Per FR-002, utterance uncertainty integrates three sub-signals per bucket:

1. **ASR pairwise mean WER** — among contributing ASR transcripts on the bucket.
2. **Whisper native** — ``1 − exp(avg_logprob)`` averaged over Whisper chunks.
3. **ASR-vs-PPG PER** — when PPG is provisioned. Per-frame phoneme-disagreement rate
   between the PPG argmax sequence and the ASR-implied phoneme timeline (no
   argmax-deduping over the bucket — every PPG frame contributes one comparison).

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
    ppg_argmax_confidence_per_frame,
    ppg_argmax_per_frame,
    ppg_argmax_runs_in_window,
    ppg_mean_confidence_in_window,
    resolve_asr_result,
    whisper_bucket_avg_logprob,
)


def harvest_utterance_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    ppg_block: dict[str, Any],
    alignment_by_model: dict[str, Any],
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the utterance axis.

    ``votes`` is a dict ``{asr_model_id → {"text": str, "avg_logprob": float | None,
    "phoneme_per_to_ppg": float | None}}``. ``avg_logprob`` is shipped as the raw
    scalar (negative); the aggregator converts it to ``1 − exp(...)`` for the
    uncertainty sub-signal so reviewers can read the original from the parquet.

    Two utterance-specific rules:

    - **ASR text per bucket uses fully-contained chunks only** (``fully_contained=True``).
      Words straddling a bucket boundary contribute to NEITHER side — partial words
      were inflating the WER on every boundary. Pair this with a wider+overlapping
      utterance grid (recommended: 1.0 s window with 0.5 s hop) so most words still
      land inside at least one bucket.
    - **PPG temporal-frame comparison** is unchanged — per-frame argmax disagreement
      against the ASR-implied phoneme timeline.
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    asr_blocks = (pass_summary.get("asr") or {}).get("by_model") or {}
    asr_ok = {m: b for m, b in asr_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    asr_resolved = {m: resolve_asr_result(b, alignment_by_model.get(m)) for m, b in asr_ok.items()}

    # Detect transcript language to gate the PPG-vs-ASR PER signal. The ``ppgs``
    # library's argmax inventory is English-ARPAbet; ``g2p_en`` only maps
    # English text to ARPAbet. For non-English ASR output the resulting PER
    # is meaningless garbage (mostly 1.0). Whisper exposes ``language`` on
    # each segment when language detection ran. We honour any model's
    # explicit ``language`` field if present; lacking that, we default to
    # English (the dominant case for the workflow's English-PPG pairing).
    transcript_languages: set[str] = set()
    for resolved in asr_resolved.values():
        items = resolved if isinstance(resolved, list) else [resolved]
        for line in items:
            lang = getattr(line, "language", None)
            if lang is None and isinstance(line, dict):
                lang = line.get("language")
            if lang:
                transcript_languages.add(str(lang).lower()[:2])
    ppg_per_signal_enabled = not transcript_languages or "en" in transcript_languages
    if not ppg_per_signal_enabled:
        import sys as _sys

        print(
            f"warn: utterance.PPG-vs-ASR PER skipped — transcript language(s) "
            f"{sorted(transcript_languages)} are non-English and the PPG inventory "
            "is English-ARPAbet; the cross-check would produce meaningless edit "
            "distances",
            file=_sys.stderr,
        )

    # Pre-compute the PPG argmax-per-frame sequence once for the whole audio. Each
    # bucket then samples the slice covering its time range — no per-bucket
    # re-derivation, no argmax-deduping.
    ppg_per_frame: list[str] = []
    ppg_frame_hop: float = 0.0
    ppg_argmax_conf_per_frame: list[float] = []
    if isinstance(ppg_block, dict) and ppg_block.get("status") == "ok":
        ppg_per_frame, ppg_frame_hop = ppg_argmax_per_frame(
            ppg_block.get("result"),
            ppg_block.get("phoneme_labels"),
            duration_s,
        )
        ppg_argmax_conf_per_frame, _ = ppg_argmax_confidence_per_frame(
            ppg_block.get("result"),
            ppg_block.get("phoneme_labels"),
            duration_s,
        )

    from itertools import combinations

    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, dict[str, Any]] = {}
        # Per-source phoneme sequences (overlap rule for both ASR and PPG so
        # both sides see the same time span). PPG sequence is silent-stripped
        # argmax runs; ASR sequence is g2p_en distributed across word
        # timestamps with phoneme-midpoint inclusion.
        per_source_phoneme_seq: dict[str, list[str]] = {}
        ppg_seq: list[str] = []
        if ppg_per_frame and ppg_frame_hop > 0:
            runs = ppg_argmax_runs_in_window(ppg_per_frame, ppg_frame_hop, start, end)
            ppg_seq = [p for _, _, p in runs if p != "<silent>"]
            if ppg_seq:
                per_source_phoneme_seq["__ppg__"] = ppg_seq

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
            if ppg_per_signal_enabled:
                asr_phon_seq = asr_phoneme_sequence_in_window(resolved, start, end, fully_contained=False)
            if asr_phon_seq:
                per_source_phoneme_seq[m] = asr_phon_seq
            votes[m] = {
                "text": text,
                "phoneme_sequence": asr_phon_seq,
                "avg_logprob": avg_logprob,
                "alignment_ctc_score": ctc_score,
            }

        # Pairwise phoneme edit-distance rate across all available sources
        # (4 ASRs + PPG → up to C(5,2)=10 distances per bucket). Each
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
        # Whisper avg_logprob → exp() and PPG argmax confidence. The MMS-CTC
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
        # PPG argmax confidence — mean over frames inside the bucket.
        ppg_conf = ppg_mean_confidence_in_window(ppg_argmax_conf_per_frame, ppg_frame_hop, start, end)
        if ppg_conf is not None:
            per_source_conf["__ppg__"] = float(ppg_conf)
            votes["__ppg_argmax_confidence__"] = {"value": float(ppg_conf)}
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
