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

from typing import Any, Mapping, Sequence

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


def _aligned_columns(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[list[dict[str, Any]]] | None:
    """Group the recognizers' words by **sequence alignment**, one list per aligned position.

    Time-overlap grouping cannot represent an insertion, and the consequence is not a lost filler
    but a corrupted transcript. Measured on three recognizers reading "I uh think" where only
    CrisperWhisper emits the filler: the filler overlaps *"think"* in the other two, joins that
    group, loses the vote 2-to-1 and is dropped — and CrisperWhisper's own "think" cannot rejoin the
    group its model already occupies, so it forms a second group and "think" is emitted twice.

    Capturing disfluencies is the point, not a side effect: a filler one model heard and others
    discarded is evidence about the speaker, and the axis should record that the recognizers
    disagreed about it rather than quietly picking the majority reading of a different word.

    ``harmonize_transcripts`` already does this alignment — star-shaped against the median-length
    transcript, with insertions keyed between reference positions — and had no caller outside its
    own tests. Returns ``None`` when the alignment cannot be formed, so the caller falls back to
    time-overlap grouping rather than losing every word.
    """
    from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts

    by_model = {
        model: [(float(w["start"]), float(w["end"]), str(w["text"])) for w in words]
        for model, words in streams.items()
        if words
    }
    if len(by_model) < 2:
        return None
    try:
        harmonised = harmonize_transcripts(by_model)
    except Exception:  # noqa: BLE001 — a fold must not fail on an alignment edge case
        return None
    if not harmonised.slots:
        return None

    # Rebuild full word dicts per column: the alignment carries surface forms and each model's own
    # span, and the fold needs the rest of the word (confidence, corroboration, timing provenance)
    # to weigh and to measure boundary agreement. Matched back by (model, start) since a model
    # cannot place two words at one onset.
    index = {(model, round(float(w["start"]), 6)): w for model, words in streams.items() for w in words}
    columns: list[list[dict[str, Any]]] = []
    for slot in harmonised.slots:
        members: list[dict[str, Any]] = []
        for model, span in (slot.times or {}).items():
            if span is None:
                continue
            original = index.get((model, round(float(span[0]), 6)))
            if original is None:
                continue
            members.append({**dict(original), "model": model})
        if members:
            columns.append(members)
    return columns or None


def phoneme_similarity(a: str, b: str) -> float:
    """How close two words sound, in ``[0, 1]`` — 1.0 identical, 0.0 sharing no phoneme.

    Supplied to the ensemble so word accuracy grades its disagreements instead of counting exact
    matches. The task API stays stdlib-only and receives this as a callable, the same way it
    receives ``calibrator`` and ``speaker_at``: g2p is a workflow dependency and does not belong
    inside a model-independent voting routine.

    ARPAbet with stress markers stripped, so ``AH0`` and ``AH1`` are one phoneme — stress is not a
    lexical difference and counting it would penalise two recognizers that agree on the word.

    Falls back to **exact match** when g2p is unavailable or produces nothing for either side, not
    to grapheme overlap: letters are not sounds, and substituting one measure for the other would
    change the number's meaning invisibly. A homophone pair therefore scores 1.0 where g2p works
    and 0.0 where it does not, which is a real limitation and better than an unrecorded proxy.
    """
    if a == b:
        return 1.0
    try:
        from senselab.audio.workflows.audio_analysis.harvesters import g2p_phonemes, normalize_arpabet

        seq_a = [normalize_arpabet(p) for p in g2p_phonemes(a) if str(p).strip()]
        seq_b = [normalize_arpabet(p) for p in g2p_phonemes(b) if str(p).strip()]
    except Exception:  # noqa: BLE001 — a missing g2p must not fail a fold
        return 0.0
    if not seq_a or not seq_b:
        return 0.0
    denominator = max(len(seq_a), len(seq_b))
    return max(0.0, 1.0 - _levenshtein(seq_a, seq_b) / denominator)


def _as_plain(node: Any) -> Any:  # noqa: ANN401 — ScriptLine tree or its JSON form
    """Normalise a ScriptLine tree to the dict/list form the word walker understands.

    ``resolve_asr_result`` hands back ``ScriptLine`` *objects* from a live backend and dicts from
    the cache, and ``iter_word_leaves`` walks dicts only. Without this the fold silently found no
    words and the asr axis came out with zero contributing signals on a real run — the failure
    every unit test here missed, because they all construct dicts.
    """
    if isinstance(node, list):
        return [_as_plain(item) for item in node]
    dump = getattr(node, "model_dump", None)
    return dump() if callable(dump) else node


def _consensus_word_doubt(
    asr_resolved: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
) -> tuple[dict[tuple[float, float], float | None], dict[str, Any]]:
    """Fold the recognizers' words once, then resample the result onto the grid (D-27).

    Returns ``(doubt_by_bucket, provenance)``. The provenance travels onto every row because the
    fold's parameters *are* its policy: a derivative whose choices are not in the artifact is the
    default argument this design keeps removing (D-21 rule 4).

    The slot parameters are the task API's own defaults, named here rather than inherited silently
    so the recorded value and the used value cannot drift.
    """
    from senselab.audio.tasks.speech_to_text_ensemble import fuse_word_streams, iter_word_leaves

    slot_overlap, slot_mid_tol_s = 0.3, 0.15
    streams: dict[str, list[dict[str, Any]]] = {}
    for model_id, resolved in asr_resolved.items():
        words = iter_word_leaves(_as_plain(resolved))
        if words:
            streams[str(model_id)] = words
    if not streams:
        return ({b: None for b in buckets}, {})

    fused = fuse_word_streams(
        streams,
        slot_overlap=slot_overlap,
        slot_mid_tol_s=slot_mid_tol_s,
        text_similarity=phoneme_similarity,
        columns=_aligned_columns(streams),
    )
    counts = sorted({int(w["timing_sources"]) for w in fused if w.get("timing_sources") is not None})
    provenance = {
        "operator": "consensus_words/resample",
        "sources": sorted(streams),
        "n_words": len(fused),
        "slot_overlap": slot_overlap,
        "slot_mid_tol_s": slot_mid_tol_s,
        # How many *independent* timing opinions the words had — an int when every word had the
        # same number, the sorted set otherwise. Two recognizers sharing an aligner count as one,
        # so this is routinely lower than ``len(sources)`` and the row has to say so rather than
        # leaving a reader to infer independence from the model count.
        "timing_sources": (counts[0] if len(counts) == 1 else counts) if counts else None,
    }
    return resample_word_doubt(fused, buckets), provenance


def resample_word_doubt(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Project **word-sequence accuracy** onto a time grid — the asr axis (D-27, revised).

    One question, one axis: how much do the recognizers disagree about *what words were said*. Each
    word contributes doubt ``1 - existence_confidence`` over its own span, and nothing else enters.

    **Temporal agreement is deliberately excluded** — from the mass and from the reach. It is not
    that localisation does not matter; it is that a single number cannot carry both and stay
    readable, and two attempts proved it. Bucketed pairwise WER made timing jitter *look* like
    textual disagreement, reporting an axis mean of 0.4266 on a pair of word-identical transcripts.
    Replacing that with a joint of accuracy × localisation fixed the conflation but not the
    legibility: 0.788 could mean either half, and a reader had no way to tell which. Localisation
    now lives on the word, split per edge (``onset_confidence`` / ``offset_confidence``), where the
    figure can show *which* boundary is in doubt instead of averaging it into a score.

    The cost, stated because it is real: on a run where every recognizer agrees, this axis is near
    zero — and that is the honest answer, since there is no disagreement about what was said. A
    poorly localised word no longer registers here at all, which is why the word-level fields and
    the figure's onset/offset marks are the place that question is answered.

    Args:
        words: Fused words carrying ``start``, ``end`` and ``existence_confidence``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → doubt}``, the coverage-weighted mean of the doubt reaching it. ``None`` where no
        word reaches at all: a bucket nothing was said in is unmeasured, and reporting ``0.0``
        there would assert that we are certain nothing was said — a claim of a different kind.
    """
    contributions: dict[tuple[float, float], list[tuple[float, float]]] = {b: [] for b in buckets}
    for word in words:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        accuracy = word.get("existence_confidence")
        doubt = 1.0 - float(accuracy) if isinstance(accuracy, (int, float)) else 1.0
        # Accuracy only, and the word's own span only. Temporal agreement is deliberately **not**
        # here — not in the mass and not in the reach — so this axis answers one question: how much
        # do the recognizers disagree about the word sequence. Mixing localisation in is what made
        # the axis unreadable twice, first as bucketed-WER disagreement that was really timing jitter
        # and then as a joint whose two halves could not be told apart in the number. The temporal
        # halves live on the word (``onset_confidence`` / ``offset_confidence``) where a reader can
        # see which edge is in doubt.
        lo, hi = start, end
        for bucket in buckets:
            overlap = min(hi, bucket[1]) - max(lo, bucket[0])
            if overlap > 0:
                contributions[bucket].append((overlap, max(0.0, min(1.0, doubt))))

    out: dict[tuple[float, float], float | None] = {}
    for bucket, reaching in contributions.items():
        if not reaching:
            out[bucket] = None
            continue
        total = sum(w for w, _ in reaching)
        out[bucket] = sum(w * d for w, d in reaching) / total if total > 0 else None
    return out


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

    # One fold per pass, then resampled per bucket — not recomputed per bucket, and deliberately
    # not "which words fell fully inside this bucket". Reach is set by how well each word is
    # localised, so a word straddling a grid line no longer reads as two models disagreeing.
    buckets = [(round(float(s), 6), round(float(e), 6)) for s, e, _ in grid.iter_buckets(duration_s)]
    word_doubt, word_doubt_provenance = _consensus_word_doubt(asr_resolved, buckets)

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
        # Recorded, not voted (D-27). The pairwise phoneme distance is computed over the same
        # transcripts the consensus derivative folds, so its source closure is a subset of that
        # derivative's — counting both is one body of evidence twice (D-21 rule 6). It stays on the
        # row because it is the readable form of *which pair* diverged, which the fold cannot say.
        votes["__pairwise_phoneme_distances__"] = {
            "pairs": pair_distances,
            "n_sources": len(sources),
            "sources": sources,
            "per_source_confidence": per_source_conf,
            "scored": False,
        }
        # The axis's one voter: the two-part word confidence, resampled onto this bucket. Absent
        # where no word reaches, which is not the same as a bucket nobody doubts.
        doubt = word_doubt.get((round(float(start), 6), round(float(end), 6)))
        if doubt is not None:
            votes["consensus_words"] = {"value": doubt, **word_doubt_provenance}
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
