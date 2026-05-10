# Phase 0 Research — Comparison & Uncertainty Stage

**Date**: 2026-05-09

This document captures the design decisions that flow into the implementation plan. Each
decision is paired with the rationale and a brief note on alternatives considered.

---

## Decision 1 — Per-axis uncertainty math

**Decision**: Each axis uses a per-axis rule for raw vote → bucket scalar; sub-signals within
the axis collapse via the shared `--uncertainty-aggregator` flag (`min` / `mean` /
`harmonic_mean` / `disagreement_weighted`, default `min` over confidences ≡ `max` over
uncertainties).

- **presence**: Shannon entropy `H = -Σ p_i log p_i` over binary "speech-present" votes from
  the contributing models, normalized to `[0, 1]` by dividing by `log(n_contributing_models)`
  so a 50/50 vote saturates at `1.0` independent of how many models voted. Edge cases:
  exactly one contributing model → entropy is 0 (no disagreement signal possible — the row
  is still emitted with `aggregated_uncertainty = 1 − model_native_confidence` if the model
  exposes one, else `null`); zero contributing models → no row.
- **identity**: per FR-017, three sub-signals:
  - Cross-model speaker-label disagreement = `1 − (n_agreeing_pairs / n_pairs)` over the
    speaker labels assigned by every diar model that has a segment overlapping the bucket.
    `n_pairs = C(k, 2)` where k = number of diar models with overlap. Edge case: k = 1 →
    sub-signal contributes `null`.
  - Same-model raw-vs-enhanced speaker-label = bool: did the same diar model assign different
    speaker labels to overlapping segments on the two passes? Mapped to `0.0` (agree) /
    `1.0` (disagree). Only emitted on the raw_vs_enhanced/identity parquet.
  - Across-time speaker-change = `1 − cos_similarity(emb_this_bucket, emb_prev_bucket_same_track)`.
    Embeddings come from `senselab.audio.tasks.speaker_embeddings.extract_speaker_embeddings`
    invoked once per *diarization segment*; a bucket inherits the embedding of the
    diarization segment that overlaps it. The "previous bucket on the same track" is the
    most recent prior bucket whose dominant diar speaker label matches; if none exists,
    sub-signal contributes `null`.
- **utterance**: per FR-016, three sub-signals:
  - ASR pairwise mean WER = `mean(jiwer.wer(t_i, t_j) for all i < j)` clipped to `[0, 1]`,
    where each `t_i` is the per-bucket text from contributing ASR model i (resolved via
    alignment block per FR-013 for text-only models). Edge case: only one ASR has text →
    sub-signal contributes `null`.
  - Whisper native = `1 − exp(avg_logprob)` clipped to `[0, 1]`, averaged across all Whisper
    chunks overlapping the bucket. Edge case: no avg_logprob in the chunks → sub-signal
    contributes `null`.
  - PPG-ASR PER = the existing `_diff_asr_vs_ppg` phoneme-error-rate output, mean across
    contributing ASR models for the bucket. Edge case: PPG not provisioned → sub-signal
    contributes `null`.

**Rationale**: presence is a binary classification problem (Bernoulli votes → entropy is the
canonical disagreement scalar). Identity and utterance have heterogeneous sub-signals that
can't naively be summed, so the per-axis aggregator gives reviewers a single knob that
controls how the sub-signals combine.

**Alternatives considered**:
- Single uniform aggregator across all axes — rejected: forcing one rule for the binary,
  categorical, and textual axes throws away signal; the 2026-05-09 clarify explicitly chose
  per-axis rules (Q3 option A).
- Disagreement-weighted formula `(1 − mean_confidence) × disagreement_severity` as the only
  rule — rejected: still useful as one of the four `--uncertainty-aggregator` options, but
  not a strong default for utterance (where `1 − mean_confidence` of Whisper alone often
  dominates the WER signal).

---

## Decision 2 — Per-bucket model contribution policy

**Decision**: Maximally inclusive — every model whose output naturally encodes the axis votes.

| Axis | Models per default run | Vote shape |
|---|---|---|
| presence | pyannote, Sortformer, whisper-turbo, granite, canary-qwen, qwen3-asr, AST, YAMNet | bool (8 votes max) |
| identity (cross-model sub-signal) | pyannote, Sortformer | speaker_label string (2 votes max) |
| identity (across-time sub-signal) | ECAPA, ResNet | float cosine distance (2 votes max) |
| utterance | whisper-turbo, granite, canary-qwen, qwen3-asr, PPG (when present) | str (transcript) + optional float (avg_logprob, PER) |

A model that emits an unusable / null signal in a bucket is dropped from that bucket's vote
(no zero-imputation), per FR-007.

**Rationale**: the 2026-05-09 clarify chose option A — "Maximally inclusive: every model whose
output naturally encodes the axis contributes." Excluding contributors discards real
evidence; the aggregator already handles the heterogeneous-confidence-magnitudes question.

**Alternatives considered**:
- Diarization-canonical for presence/identity, ASR-canonical for utterance — rejected: too
  rigid; on enhanced audio where pyannote loses speech but whisper still picks up text,
  diar-canonical would mark presence as "no" and reviewers would miss the disagreement.
- User-configurable `--{presence,identity,utterance}-models` flags — deferred: the default A
  is the right baseline and the flag adds CLI surface area for a niche case.

---

## Decision 3 — Speaker-embedding source for identity across-time sub-signal

**Decision**: Reuse `extract_speaker_embeddings` per *diarization segment* (not per bucket).
A bucket inherits the embedding of the diarization segment that overlaps it; cosine distance
is computed against the previous bucket's embedding only when they sit on the same
diarization speaker track.

Implementation outline:

1. After diarization runs (existing pipeline stage), iterate the segments of the *first*
   successful diar model (deterministic preference: pyannote first, Sortformer second).
2. For each segment, slice the audio waveform to `[seg.start, seg.end]` and call
   `extract_speaker_embeddings(audio_slices, model=ECAPA)` — and once more for ResNet —
   batched per pass to amortize model load cost.
3. Cache the resulting per-segment embeddings under the existing analyze_audio cache, keyed
   by `(audio_signature, "speaker_embeddings_per_segment", seg_signature, model_id,
   wrapper_hash, senselab_version)`.
4. At per-bucket aggregation time, look up the embedding of the segment overlapping each
   bucket; compute cosine to the previous bucket on the same speaker track; emit the
   `1 − cos_sim` value as the across-time sub-signal.

**Rationale**: per-bucket embedding extraction (slicing the audio to every 0.5 s window and
running ECAPA on each) would add ≈ 120 inferences per minute per pass per embedding model =
480 calls/min — costly and noisy (0.5 s is below ECAPA's recommended ≈ 2 s minimum input).
Per-segment extraction matches the granularity at which the embedding is actually meaningful
(one speaker turn), is cheap (≈ 10 segments / minute), and reuses the existing senselab API
verbatim. The across-time signal is preserved because diarization speaker-track transitions
between segments are exactly the points where we want to detect a speaker change.

**Alternatives considered**:
- Per-bucket extraction — rejected: ≈ 50× more compute, sub-second windows below the model's
  reliable input length.
- Use only the *single* full-audio embedding the existing speaker_embeddings task already
  produces — rejected: that gives one vector for the whole pass, no temporal information at
  all.

---

## Decision 4 — Plot row layout

**Decision**: 5-row figure — presence (raw solid + enhanced dashed), identity (raw + enhanced
overlay), utterance (raw + enhanced overlay), raw-vs-enhanced delta strip with one band per
axis, reference context row (raw diar speakers + raw ASR token spans). All rows share the
x-axis; uncertainty rows share a y-axis range of `[0, 1]`.

**Rationale**: the 2026-05-09 clarify chose option A (Q4). Three rows would lose the
head-to-head raw-vs-enhanced overlay; six would scatter the comparison across the figure;
the chosen design preserves overlay AND the "did enhancement help?" delta in a single
readable layout.

---

## Decision 5 — Output layout and naming

**Decision**: Per FR-018 —

```text
<run_dir>/
├── <pass>/
│   └── uncertainty/
│       ├── presence.parquet
│       ├── identity.parquet
│       └── utterance.parquet
├── uncertainty/
│   └── raw_vs_enhanced/
│       ├── presence.parquet
│       ├── identity.parquet
│       └── utterance.parquet
├── disagreements.json
└── timeline.png
```

9 parquets total per default two-pass run.

**Rationale**: filesystem layout matches what reviewers ask for ("show me the three
uncertainty time series"); the per-pass / raw_vs_enhanced split mirrors the existing
analyze_audio per-task / raw_vs_enhanced split for consistency.

---

## Decision 6 — ASR text resolution for text-only models

**Decision**: For text-only ASR backends (Granite Speech 3.3, Canary-Qwen) the comparator
consults the post-MMS alignment block when the raw ASR result lacks per-token timestamps
(FR-013). Without alignment, text without a time anchor is treated as
`asr_says_speech = false` for that window — it produces no token overlap.

**Rationale**: broadcasting a text-only ASR's full transcript across every comparison window
is the wrong default; it inflates speech-presence votes uniformly across the audio and
destroys the temporal precision that the rest of the comparator depends on.

---

## Decision 7 — Cross-stream grid default

**Decision**: `--cross-stream-win-length 0.5` and `--cross-stream-hop-length 0.5` (i.e.
non-overlapping). AST / YAMNet windows are projected onto this grid using floor-based
index lookup (`win_idx = floor(start / native_hop)`).

**Rationale**: finer grids over-resolve every signal in the system (Whisper word-level ≈
20 ms; pyannote frames ≈ 62.5 ms; AST window 10.24 s) and overlap double-counts buckets.
0.5 s non-overlapping is coarse enough that each bucket is independently meaningful and
fine enough to localize disagreements within an utterance.

---

## Decision 8 — Aggregator default

**Decision**: Default `--uncertainty-aggregator min` (over confidences) ≡ `max` over
uncertainties. The most-doubtful contributing signal sets the bucket's uncertainty.

**Rationale**: reviewers reading the disagreements.json index want to find the cases where
*any* signal raises a flag; `min` of confidences surfaces those efficiently. `mean` /
`harmonic_mean` / `disagreement_weighted` remain available for specific workflows.

---

## Decision 9 — Phoneme-disagreement threshold

**Decision**: Two-tier — emit a continuous `phoneme_per` (phoneme error rate) on every
ASR-vs-PPG row, and set a boolean `phoneme_disagreement = true` only when
`phoneme_per >= --phoneme-disagreement-threshold` (default `0.50`). LS regions are produced
only for rows where `phoneme_disagreement` is true.

**Rationale**: the continuous PER feeds the utterance axis aggregator; the boolean keeps the
LS bundle from drowning in low-PER segments where ASR and PPG disagree by one phoneme.

---

## Open follow-ups (not in scope)

- Configurable LS bin thresholds (`--ls-low-threshold`, `--ls-high-threshold`).
- Per-axis `--{presence,identity,utterance}-models` overrides for advanced workflows.
- Ensemble / Monte-Carlo dropout uncertainty for individual models.
- Cross-language phoneme inventory unification beyond English ARPAbet + uroman.
