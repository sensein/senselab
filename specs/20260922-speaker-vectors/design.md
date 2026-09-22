# One speaker vector per subject, over task extent — what was decided and what was measured

Continuation of the note held in `specs/20260922-compact-recording-vectors/design.md`
("Speaker embeddings — held"), which deferred this until task extent stopped moving.

What a task extent is, and how much of the corpus has one: `coverage.md`. The parquet a reader
decodes: `schema.md`. This file is the decisions and the measurements behind them.

All measurements below are on the finished design corpus,
`/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out`, at senselab `f3da4081`, with
`speechbrain/spkrec-ecapa-voxceleb` at commit `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.

---

## D-1. The model: ECAPA, one of them, at a resolved commit

`speechbrain/spkrec-ecapa-voxceleb`, 192-D, which is already senselab's
`DEFAULT_SPEAKER_EMBEDDING_MODEL`.

**One model, not two.** `audio_analysis`'s speaker-identity path scores ECAPA and ResNet per
window because it is comparing voters; this derivative has a single consumer and one vector per
speaker, so a second backend is cost with no reader. `model_id` on every row records which
produced it, so adding a second later is a new row set, not a schema change.

**Pinned by SHA, never by ref.** SpeechBrain's `from_hparams` has no `revision` argument, so the
established senselab pattern applies: `resolve_model` resolves the ref to a 40-hex commit and
returns the immutable snapshot directory, and `source=` is pointed at that directory. The row
carries `model_commit_sha`, and `unresolved_reason` is non-null exactly when the sha is null —
recording a ref in the sha column would be provenance that is confidently wrong.

## D-2. The stream: `plain`

Every PREPROCESS stream shares the source recording's time axis and duration (verified on a run
store: `recording`, `plain`, `preemphasised`, `normalized`, `enhanced`, `residual` and `redacted`
all carry `extent [0.0, 6.2694375]` at 16 kHz mono), so an extent's seconds index any of them.
`plain` is chosen over the two plausible alternatives:

- **not `enhanced`** — enhancement alters voice timbre, which is precisely the signal being
  embedded. The related note in the corpus memory, that enhancement hides intruders, applies with
  more force here: the vector *is* the timbre.
- **not `redacted`** — redaction blanks PII spans, which would punch silence into the middle of an
  extent and into the windows covering it. A speaker vector carries no text, so redaction buys
  nothing here and costs coverage.

## D-3. Embed each extent, then pool — never concatenate first

Rejected: concatenating a subject's extents into one signal and embedding that.

- A concatenation puts a splice inside a 2.0 s window wherever two extents meet. Those windows
  are then neither extent.
- It destroys the per-extent accounting. `cos_extent_pairwise_q50`,
  `leave_one_extent_out_cos_min` and `auc_same_extent_vs_diff_extent` are the only evidence on the
  row that the pooled vector describes *one* speaker; all three are defined per extent and none
  survives a concatenation.
- It makes the refusal floor unenforceable: a short extent spliced between two long ones has no
  boundary left to refuse it at.

## D-4. The window grid: 2.0 s on a 1.0 s hop, spherical mean

Taken unchanged from `estimate_speaker_embedding_from_audios`, whose defaults were fitted for
exactly this job — a profile centroid — and measured at cross-file centroid stability 0.890 and
cross-subject separation 0.168, against 0.331 for a 0.5/0.25 grid carrying four times the
windows. Three separately measured grids exist in this repo for three different jobs (0.5/0.25
for `audio_analysis` detection, 1.0/0.5 as `windowing.py`'s fallback, 2.0/1.0 for enrollment);
this is the enrollment job and it takes the enrollment grid.

Pooling over the window grid is by spherical mean. How the *extents* are then combined was left
open here and settled by measurement in D-7: **extent-equal**, not window-weighted.

## D-5. Batching must be duration-matched, and here it is by construction

**This is a correctness gate, not an optimisation.**
`SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios` zero-pads every member
of a batch to the longest member and passes `wav_lens`. The short members' vectors come back
wrong, and nothing raises.

Measured directly (job `23470961`, 30 spans per row, real task-extent audio): the same span
embedded alone, in a batch of eight spans of its own length, and in a batch of two with a 14 s
neighbour.

| span length | cos(alone, duration-matched batch) | cos(alone, mixed batch) |
| ---: | ---: | ---: |
| 0.05 s | **1.0000** [1.0000, 1.0000] | 0.5596 [0.1687, 0.7992] |
| 0.10 s | **1.0000** | 0.7867 [0.4620, 0.8986] |
| 0.20 s | **1.0000** | 0.8958 [0.6312, 0.9657] |
| 0.30 s | **1.0000** | 0.9443 [0.7606, 0.9827] |
| 0.50 s | **1.0000** | 0.9754 [0.8849, 0.9931] |
| 0.75 s | **1.0000** | 0.9901 [0.9469, 0.9977] |
| 1.00 s | **1.0000** | 0.9955 [0.9524, 0.9987] |
| 1.50 s | **1.0000** | 0.9979 [0.9779, 0.9993] |
| 2.00 s | **1.0000** | 0.9989 [0.9866, 0.9998] |
| 3.00 s | **1.0000** | 0.9994 [0.9822, 0.9999] |
| 5.00 s | **1.0000** | 0.9998 [0.9990, 1.0000] |

Median, with [min, max] across the 30 spans.

Two things this settles:

1. **A duration-matched batch is exact, not merely close.** Every one of the 330 comparisons
   returned cosine 1.0000 against the alone vector — the batch is numerically identical, not
   approximately so. So duration-matched batching is free: there is no accuracy cost to pay for
   the throughput.
2. **A mixed batch is wrong in exactly the regime that matters here.** At 0.2 s the *worst* case
   is 0.63 — a vector that is not the span's. Even at 1.0 s the worst case is 0.95, and 0.95 is
   already inside the within-speaker score range (D-7), so a mixed batch could turn a
   same-speaker pair into a near-miss.

This reproduces and extends the same effect recorded in
`specs/20260915-preprocess-diarization/design.md` (0.52 at 0.05 s, 0.88 at 0.2–0.3 s, 0.98 at
0.5 s), which was measured against a different neighbour on different audio. The agreement across
two independent measurements is what makes it a property of the backend rather than of either
measurement.

**How the pipeline satisfies it.** `extract_per_window_embeddings` slices one extent into windows
and passes those windows as one batch. `window_starts` anchors the final window to
`duration - window_s`, so **every window of one extent is exactly `WINDOW_S` long** — the batch is
homogeneous by construction, at a 1.00× spread rather than the 1.25× the diarization measurement
allowed itself. An extent shorter than `WINDOW_S` produces a single window, i.e. a batch of one.
Extents are never batched with each other, because `estimate_speaker_embedding_from_audios` loops
per file.

So the guarantee is structural, and the table above is why it must not be relaxed for throughput.

## D-6. The refusal floor: 1.0 s, fitted

An extent below the floor is excluded from the estimate and counted in the shard report. The
floor is in `data/speaker_vector_profile/2026-09-22.yaml`, not in the code.

**What was measured** (job `23470962`): 120 subjects, each with a task extent of at least 25 s.
The last 12 s of that extent is the subject's reference vector. Truncations are taken from the
*head* of the same extent, so a truncation and its reference share no samples. Each truncation is
scored against its own subject's reference (same-speaker) and against 12 other subjects'
references (between-speaker). Every embedding is a batch of one, so D-5's artefact is excluded by
construction.

| truncation | same median | same p05 | between median | between p95 | AUC | d′ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 s | 0.1829 | 0.0421 | 0.0319 | 0.1616 | 0.8887 | 1.63 |
| 0.50 s | 0.3548 | 0.0983 | 0.0469 | 0.1929 | 0.9563 | 2.58 |
| 0.75 s | 0.4480 | 0.1759 | 0.0459 | 0.1959 | 0.9821 | 3.30 |
| **1.00 s** | 0.5227 | **0.2493** | 0.0513 | **0.2125** | 0.9868 | 3.78 |
| 1.25 s | 0.5773 | 0.3266 | 0.0601 | 0.2165 | 0.9883 | 4.25 |
| 1.50 s | 0.6082 | 0.3709 | 0.0589 | 0.2201 | 0.9894 | 4.64 |
| 2.00 s | 0.6638 | 0.3798 | 0.0588 | 0.2320 | 0.9883 | 4.82 |
| 2.50 s | 0.6862 | 0.3967 | 0.0662 | 0.2453 | 0.9928 | 5.17 |
| 3.00 s | 0.7257 | 0.4752 | 0.0703 | 0.2390 | 0.9924 | 5.48 |
| 5.00 s | 0.7783 | 0.5471 | 0.0803 | 0.2698 | 0.9919 | 5.74 |
| 10.00 s | 0.8454 | 0.6523 | 0.0741 | 0.2488 | 0.9923 | 6.75 |

**The criterion, and why it is the tail and not the mean.** AUC plateaus early — 99.0% of its
10 s ceiling by 0.75 s — so an AUC criterion would have licensed 0.75 s or even 0.5 s. But AUC is
an average over pairs, and a refusal floor exists to protect the *worst* vectors, not the median
one. The criterion used is therefore: **the shortest grid point at which the 5th percentile of
the same-speaker distribution clears the 95th percentile of the between-speaker distribution.**

- At 0.75 s: 0.1759 against 0.1959 — the distributions **still overlap** at the 5/95 tails.
- At 1.00 s: 0.2493 against 0.2125 — separated, by 0.037.
- At every longer grid point: separated, and widening monotonically.

So **1.0 s**, and the crossing is between two adjacent grid points rather than inside a flat
region, which is what makes it a fitted boundary rather than a preference.

**What it costs.** From the corpus histogram in `coverage.md`, extents under 1.0 s are
329 + 364 + 530 = **1,223 of 54,923, i.e. 2.23%**. Subjects lost entirely: at most a handful —
1,523 subjects carry an extent and 1,518 carry two or more.

**d′ keeps climbing to the end of the grid** (1.63 → 6.75). The floor is a refusal boundary, not
an optimum: longer is monotonically better, and a reader weighting rows should use
`extent_seconds`, which is on every row for that purpose.

### The gap the floor does not close

Between the 1.0 s floor and the 2.0 s window, an extent yields exactly one window, and
`describe_embedding_distribution` refuses fewer than two vectors. In a normal subject that is
invisible — the subject's other extents supply the rest of the pool. A subject whose entire supply
is one sub-2.0 s extent produces no row, and the report names the exception per subject rather
than counting it, so that case is distinguishable from a decode error. Lowering the window to
admit those subjects was rejected: it would replace a measured grid (D-4) with an unmeasured one
to rescue a handful of speakers whose vector would rest on a single window anyway.

## D-7. Within- versus between-speaker separation, and the pooling it settles

**The protocol.** 1,513 subjects — essentially the whole corpus — each contributing up to 12 task
extents, each extent embedded separately on the production window grid (jobs `23471343`, 24
shards, and `23471698`, 16 shards). A subject's extents are split at random into two disjoint
halves; each half is pooled into a vector. A **within**-speaker score is one subject's half-A
against its own half-B. A **between**-speaker score is half-A against another subject's half-B —
2,287,656 pairs. No recording contributes to both sides of a within pair, so this measures
cross-recording, cross-session speaker identity rather than within-recording consistency.

| pooling | within mean | within p05 | between mean | between p95 | AUC | EER | d′ | rank-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window-weighted (one-stage) | 0.747 | 0.343 | 0.260 | 0.632 | 0.9500 | 0.1196 | 2.64 | 0.722 |
| **extent-equal (two-stage)** | 0.781 | **0.553** | 0.281 | 0.594 | **0.9847** | **0.0588** | 3.54 | **0.818** |
| extent-equal, corpus-centred | 0.672 | 0.319 | 0.009 | 0.301 | 0.9870 | 0.0476 | 3.84 | 0.804 |

**Extent-equal wins, and by a margin worth the code.** It halves the equal error rate (5.9%
against 12.0%) and lifts rank-1 identification from 0.722 to 0.818 against 1,512 impostors. The
mechanism is visible in the within p05: 0.553 against 0.343. One-stage pooling weights an extent
by its window count, and the corpus spans 1.0 s to 330 s, so a single long free-speech extent can
outvote a subject's entire remaining supply — and if that one extent is atypical, the speaker's
vector is that extent. The production pooling was changed to two-stage on this measurement.

**Centring was measured and not adopted.** The corpus mean vector has norm **0.3697**. The mean
of *n* independent unit directions has expected norm `1/sqrt(n)`, about **0.0074** for the
~18,000 extent vectors here — so 0.37 is fifty times the null: every extent in the corpus points
substantially in one shared direction, which is channel and protocol rather than identity.
Subtracting it does what you would expect — the between-speaker mean falls from 0.281 to 0.009,
essentially to the orthogonality null — and buys a further AUC 0.9870 and EER 4.8%. It is **not**
applied, for two reasons: it *lowers* rank-1 (0.804 against 0.818), which is the operation a
reader of this file most plausibly wants; and it is a corpus-level statistic, so baking it in
would make one row uninterpretable without the other 1,512. The raw vector is stored, the corpus
mean is recoverable from the parquet in one pass, and a reader who wants verification rather than
identification should centre and gets the numbers above.

**Honest limits of this result.** The two distributions still **overlap at the 5/95 tails even at
the best pooling**: within p05 0.553 against between p95 0.594. An EER of 5.9% means about one in
seventeen decisions is wrong at the equal-error operating point. This is a usable speaker
embedding, not a strong one, and no threshold is shipped with it.

**181 of 2,287,656 between-speaker pairs score above 0.9** under extent-equal pooling (2 under
centring). Two `sub-` ids that are one person re-enrolled would look exactly like this, and so
would a genuine failure. Not investigated; named because a reader deduplicating on this vector
will meet it.

### How it scales with supply

Per-extent count, on the thinner half (centred pooling, where the buckets are populated):

| extents in the thinner half | n | within mean | within p05 | rank-1 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 5 | 0.358 | 0.145 | 0.200 |
| 4 | 77 | 0.607 | 0.251 | 0.727 |
| 5 | 609 | 0.660 | 0.271 | 0.793 |
| 6 | 819 | 0.690 | 0.359 | **0.825** |

Monotone in the **number** of extents, which is the number this module should be read on.

Seconds, by contrast, is **not** monotone: the [10,20) s bucket reaches rank-1 0.942 while
[40,80) s reaches 0.759. Total duration is confounded with task mix — the longest suppliers are
free-speech-heavy, the 10–20 s ones spread over many short prompted tasks — so seconds is a worse
predictor of a good speaker vector than extent count is, and `n_extents` is the column to weight
by. Both are on every row.

At the small-sample end the fast replicate (123 subjects, at most 3 extents per half) gives
AUC 0.899 and rank-1 0.667 under extent-equal pooling — the same ordering between poolings, at
materially worse absolute numbers. Three extents is not enough; six is workable.

## What was not measured

- **That one `sub-` id is one human.** Not verifiable from the corpus; stated in `coverage.md`.
- **That a task extent contains only the participant.** The protocol is single-speaker prompted
  tasks, and a task extent biases towards the participant, but
  `specs/20260915-preprocess-diarization/design.md` documents a recording where a second speaker's
  labels fell inside the participant's own narration. The per-extent agreement columns are what a
  reader uses to notice this; nothing here reaches a verdict about it.
- **Why `random-item-generation` mints an extent on none of its 472 recordings.** A coverage fact
  this module inherits; see `coverage.md`. It is 6.2% of all missing extents — the rest is
  per-recording, not structural.
- **Whether the replay moves any of this.** Every number above is from the *design* corpus. Task
  extent is one of the things the replay may move, which is why the production run targets the
  replayed tree. The replayed smoke set (16 recordings, 16 distinct subjects) confirms the shape
  and the reading path but carries no two recordings of one speaker, so it can validate the
  interface and not the separation.
- **A second embedding model as a cross-check.** One model by decision (D-1); a disagreement
  between two backends would be a different and larger question.
