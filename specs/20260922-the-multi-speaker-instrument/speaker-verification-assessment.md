# Could verification tell the participant from someone else inside a task extent?

**An assessment. Nothing here was built, and on the evidence below nothing here should be built
yet.** The question was asked because `tasks/speaker_verification/verify_speaker()` exists, has
**zero references anywhere under `workflows/triage/`**, and looks on its face like the missing
half of the multi-speaker instrument: the instrument says *a second voice is at 2.4-2.6 s*, and
verification would say *and it is not the participant*.

Short answer: **the embedding is too weak at the durations that matter, the enrollment it would be
verified against is contaminated by the thing it is meant to detect, and the only method measured
to work on this corpus needs impostors from other subjects, which a per-recording node cannot
reach.** Each of those is separately disqualifying. The cost is not the obstacle.

## What exists today

**`verify_speaker`** (`src/senselab/audio/tasks/speaker_verification/speaker_verification.py:19`)
takes pairs of `Audio`, embeds both with `speechbrain/spkrec-ecapa-voxceleb`, and returns
`(cosine, cosine > threshold)` per pair. Three properties matter before anything else:

- **Its threshold is `0.25`, a bare default in the signature (`:23`).** No derivation, no fit, no
  reference in `specs/`. That is precisely the shape of literal this repository forbids.
- **It loads through `revision="main"` (`:43`),** not a resolved commit, so a verification's
  provenance would name weights that may have moved.
- **It embeds each pair as a batch of two** (`:59-61`), and
  `speaker_embeddings/speechbrain.py:131-137` zero-pads a batch to its longest member. This is the
  corruption D-5 of `specs/20260922-speaker-vectors/design.md:70-112` measured: a 0.2 s span
  batched with a 14 s neighbour returns cosine 0.8958 median, 0.6312 worst, against its own
  alone-vector; at 0.05 s the median is 0.5596. **A short probe verified against a long enrollment
  clip is therefore embedded wrong, silently, by exactly this function.** Any use would have to
  bypass it.

**The enrollment path** (`nodes/speech.py:2143-2227`) is the place the answer would go, and it has
never run: `enrollment` was `None` across the entire corpus, and both `speech.enrollment_model` and
`speech.target_match_cosine` ship null. `benchmarks/open.md:90,125` records the attempt to fit the
threshold and its failure — leave-one-out medians of 0.669/0.705/0.693 on three subjects, and *no
single threshold separates speech from non-speech on all three*; the two-subject intersection is a
0.035-wide band. The b2ai-v2 run guessed 0.5 and attributed the target on 53% of recordings.

**The per-subject embedding** (`speaker_vectors.py`, `specs/20260922-speaker-vectors/`) is the
newest and strongest piece: ECAPA 192-D over `plain`, 2 s windows on a 1 s hop, pooled
window → extent → recording → subject, written offline to parquet and **read by nothing in the
pipeline**. On the production configuration (speech-family only) it separates subjects at
**EER 0.0122, AUC 0.99868, rank-1 0.973** (D-10, `design.md:396-422`); the 0.0588 quoted in the
brief is the all-families figure (D-7, `:194`), which the shipped configuration does not use.

## Why the strong EER does not transfer to a task extent

**It is the wrong comparison, measured on the wrong unit.** EER 0.0122 is a *subject-level,
cross-recording* number: each subject's extents were split into two disjoint halves and each half
pooled, so both sides of every comparison are many seconds of pooled speech from several
recordings (`design.md:183-194`). The question here is whether **one turn of 0.6 s inside one
extent** belongs to the participant. Those are not the same measurement and the second is far
harder.

The repository already measured how much harder. D-6 (`design.md:124-169`, job 23470962, 120
subjects, no shared samples, every embedding a batch of one) truncates a probe and scores it
against the same speaker's 12 s reference and twelve impostors:

| probe duration | same-speaker p05 | impostor p95 | do the tails separate? |
| --- | --- | --- | --- |
| 0.25 s | 0.0421 | 0.1616 | **no** |
| 0.50 s | 0.0983 | 0.1929 | **no** |
| 0.75 s | 0.1759 | 0.1959 | **no** |
| 1.00 s | 0.2493 | 0.2125 | barely — by 0.037 |
| 2.00 s | 0.3798 | 0.2320 | yes |
| 5.00 s | 0.5471 | 0.2698 | yes |

Below 1 s the same-speaker p05 sits *below* the impostor p95: one decision in twenty is on the
wrong side of any threshold you pick, in each direction. That is why the shipped profile carries
`min_extent_s: 1.0` (`data/speaker_vector_profile/2026-09-22.yaml`) and nothing shorter is
embedded at all.

**And the turns to adjudicate are shorter than that.** No corpus-wide distribution of secondary-turn
durations has been taken — `secondary_s` is computed per extent but the trigger scan reported only
the share — so what exists is two bounds, both pointing the same way. From the share distribution
(`design.md`, § the bound): of the 2,045 recordings whose task extent holds a second speaker, 1,432
give that speaker under 10% of the attributed seconds, and 555 sit between 0.99 and 1.00, which is
a fraction of a second. And the one recording anyone has looked at closely
(`specs/20260915-preprocess-diarization/design.md:317-362`) had 17 second-speaker fragments: one at
0.017 s, the embeddable 16 spanning 0.12-1.35 s, mean 0.6 s, **only four reaching 1.0 s and none
reaching 2.0 s**. Verification would be asked to adjudicate mostly below its own measured floor.

**Duration, not identity, is what moves the cosine at these lengths.** The same investigation found
`SPEAKER_00`'s own segments scoring 0.74-0.91 when they were 3 s or longer and 0.21/0.38/0.46 when
they were 0.52/0.61/1.11 s — inside the range of the putative intruder. A raw threshold on
segment-level cosine "looks like separation and is mostly a duration contrast" (`:317-323`).

## The enrollment would be contaminated by what it is meant to detect

Even granting a long enough probe, the reference has a circularity. `speaker_vectors.py` embeds
spans with `role == "task_extent"` (`:47`), speech-family only (`:50`) — **the very spans the
instrument suspects of holding a second voice.** A subject whose extents contain an examiner's
prompts has an enrollment vector pulled toward the examiner, and the pull is largest for exactly the
subjects the instrument fires on. The speaker-vectors spec lists this among its own
non-measurements in as many words: it has not established "that a task extent contains only the
participant" (`design.md:330-347`), and it cites the diarization spec's recording where a second
speaker's labels fell inside the participant's narration.

Building an uncontaminated enrollment is possible — restrict to extents the instrument found
single-source, or to the `solo_extent` spans the instrument already mints — but that is a corpus
pass that depends on the instrument's output, so it cannot be a step inside the branch that produces
it.

## What would have to be true

1. **A probe of at least 1 s, and preferably 2 s.** Below that the measured tails overlap and a
   single-extent decision is not supported by anything in this repository. That alone removes most
   of the turns in question, on the only evidence available about their durations.
2. **An enrollment built from regions the instrument found single-source**, so the reference is not
   pulled toward the intruder. That makes verification a second pass over a completed corpus, not a
   node in the branch.
3. **A duration-matched impostor control per probe, not a remembered threshold.** This is the one
   method measured to work here: `specs/20260915-preprocess-diarization/design.md:291-315` settled
   a real case by scoring each fragment against 24 same-speaker and 24 impostor windows *of exactly
   that fragment's length*, getting AUC 0.889 where a fixed threshold would have been meaningless.
   Its own author notes the comparison is rescued not by the vectors being good but by the impostor
   control being computed from the same bad vectors — and that had the two populations overlapped,
   "the honest answer here would have been 'not settled by this evidence'". **Impostor windows come
   from other subjects, which a per-recording node cannot reach.** This is the structural
   obstacle, and it is independent of the duration one.
4. **Embedding one clip at a time.** `verify_speaker` batches the pair and pads to the longest
   member, which corrupts the short side. It would have to be bypassed, or fixed.
5. **Labels to fit against.** The same absence that leaves `dominant_speaker_share_min` unfitted
   leaves this unfittable: no recording in this corpus carries a label saying whether a second
   person was in the room, so neither a cosine threshold nor an impostor-control operating point
   can be validated. Every number above would be an instrument reading, never a verified verdict.

## What it would cost

**Not much, and that is not the point.** ECAPA is small and the probes are short; the arithmetic is
negligible beside the 132 CPU-hours separation itself costs. The real cost is structural, in two
places. First, requirement 3 turns this from a node into a second corpus pass with an impostor pool
— a pool that must be duration-stratified, which means embedding tens of short windows per probe
rather than one. Second, requirement 2 makes that pass depend on the instrument's own output, so
the ordering is: run the corpus, build clean enrollments from the single-source regions, then
verify. No figure is put on either here, because none has been measured and a guess would be the
same kind of unfitted number this assessment is arguing against.

## Recommendation

**Do not wire verification into the branch.** Leave `verify_speaker` unreferenced; as written it is
unsafe for this use on three counts (unfitted threshold, ref-not-commit load, padded pair batching)
and none of them is the reason to avoid it.

What is worth doing, in order and independently of this instrument:

- **Measure the secondary-turn duration distribution over the corpus.** `secondary_s` is already
  computed per extent, and the instrument now writes `secondary_spans` per run. One aggregation
  pass would replace this assessment's two bounds with the actual distribution, and it is the single
  measurement that would most change the conclusion: if the turns are mostly over 2 s, requirements
  1 and 3 both soften.
- **Fix `verify_speaker`'s three defects** on their own merits — a derived threshold in `data/`, a
  commit-pinned load, and one clip per batch — because it is public API that a tutorial already
  calls.
- **Revisit only after adjudicated labels exist.** Without them, a verification result is a second
  unvalidated reading laid over an unvalidated separation, and the instrument gains confidence it
  has not earned.
