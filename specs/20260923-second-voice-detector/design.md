# A second-voice detector for recordings the incumbent diarizer calls single-speaker

Status: measurement in progress. Numbers below are recorded as they are measured; anything
not yet measured says so.

## The case

Stem `sub-00053adb…_ses-33F6D051…_task-story-recall`, 63.60 s, 16 kHz mono.

The shipped store says one speaker: 10 diarization segments on the `enhanced` stream, every
one `SPEAKER_00`, `max_concurrent_speakers: 1`, so `speaker_count: 1`,
`separation: "not_needed"`, and the multi-speaker instrument never fired.

The owner has listened and reports two voices, with roughly a third of the recording being
conversation away from the declared task.

### Ground truth, in seconds

Recovered from the consensus word timings already in the store (`prov_type: "word"`), read
by hand. No transcript text is recorded here — only the boundary it establishes.

| span | who | what it is |
|---|---|---|
| 0.51 – 15.64 s | speaker B | task administration: instructions about the test, how long to read, and what to recall |
| 15.77 – 18.00 s | speaker A | the participant asking whether to begin |
| 18.65 – 61.70 s | speaker A | the story recall itself |

So the second voice occupies **0.5 – 15.6 s, 15.1 s of 63.6 s = 23.8 % of the recording and
29.3 % of the 51.7 s of detected speech**. The owner's "about a third, far from the declared
task" is this span. The boundary is a single clean turn change at ≈ 15.7 s, not scattered
interjections — which matters, because it is the easiest possible case for any detector and a
candidate that misses it will miss everything harder.

## What the incumbent actually gets wrong

Pyannote `speaker-diarization-community-1`, revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`,
on the `plain` stream, mono 16 kHz, `exclusive=False`, no hints — the PREPROCESS call:

* free: **1 speaker**, 9 segments, 51.71 s of speech.
* `num_speakers=2`: **2 speakers**, the *same* 9 segments, split
  `SPEAKER_00` = 0.031 – 15.657 s (12.22 s) and `SPEAKER_01` = 15.725 – 61.777 s (39.49 s).

That second partition is the ground-truth boundary, to within 0.1 s, recovered with no change
of model, no change of stream and no extra inference beyond a second clustering pass.

**The failure is speaker counting, not segmentation and not embedding.** Pyannote's own
segmentation already puts a boundary exactly where the turn change is, and its own embeddings
already separate the two sides well enough to assign them correctly once it is told there are
two. What it will not do unprompted is decide that two is the answer.

This is the single most important finding here, because it reframes the question: the useful
candidate is not necessarily *a different diarizer* but *a different decision rule over the
one already running*.

The same holds on the stream the pipeline actually diarized. The stored sidecars read:

* `enhanced_diarization.npz` — 10 segments, 1 speaker, and a segment boundary at
  15.657 / 15.725 s, again exactly the turn change.
* `residual_diarization.npz` — 2 segments, 1 speaker, 7.9 s of speech total.

So the already-computed second stream does not catch it either, and the boundary the
diarizer needed was in its own output all along.

### Consequence for the `speech.second_diarizer` slot

The archived design for that slot runs a second diarizer **only when pyannote's count is not
1**. On this recording pyannote's count *is* 1, so that design would never have fired.
Corroboration-on-disagreement is the wrong shape for this failure mode; whatever fills the
slot has to run when pyannote says one, or not help here at all.

## Candidate families

1. **Retune the incumbent.** The shipped `config.yaml` carries
   `params.clustering.threshold: 0.6` for VBx clustering. Lowering it makes the pipeline
   readier to split. Same checkpoint, same single inference pass, no new dependency — the
   cheapest possible change if the false-positive rate holds.
2. **A different diarizer.** The five non-pyannote backends behind `diarize_audios`.
3. **Cluster the recording's own voice embeddings**, independent of any diarizer.

## The embedding route, measured

ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) over a fixed window grid across the recording's
speech, every window exactly the same length — so the duration contrast that dominates
variable-length spans cannot express itself as a split, and the mixed-batch padding defect
cannot fire either.

### The information is there; the shipped clusterer does not find it

Grid 2.0 s / 1.0 s, 53 speech windows, 13 before the true boundary and 40 after. ARI is
against the ground-truth two-way labelling.

| clusterer at k=2 | sizes | ARI |
|---|---|---|
| AHC average linkage (what `select_dominant_vectors` uses) | 3 / 50 | **−0.063** |
| AHC complete linkage | 13 / 40 | **0.697** |
| AHC Ward | 17 / 36 | **0.708** |
| k-means | 16 / 37 | 0.639 |
| spectral (cosine affinity) | 14 / 39 | 0.633 |

Average linkage peels three outliers off the edge of one speaker's cloud and calls that the
split. Complete linkage and Ward recover the true partition almost exactly. This is a
property of the linkage, not of the embeddings.

Run through `select_dominant_vectors` unchanged — its merge-gap rule with the fitted
significance margin of 5.0 — the recording returns **one cluster**, and
`cluster_pass_speakers` (spectral, silhouette sweep) returns **`n_speakers: 1`,
best silhouette 0.187**. Both shipped entry points refuse the split.

### Why a cosine threshold alone cannot decide it

At the true labelling: centroid cosine between the two speakers **0.7285**, mean within-A
cosine **0.7604**, mean within-B cosine **0.7133**. The between-speaker cosine is *higher*
than the within-speaker cosine of the larger group. Cosine silhouette of the true labels is
**0.1817**.

So at recording level the separation is real but weak in absolute terms, and no fixed cosine
can distinguish "two speakers" from "one speaker with range". Whatever statistic is used has
to be calibrated against recordings known to hold one voice — which is what the sample below
is for.

### Window grid matters, and 2.0 s is the best of the three tried

| grid | speech windows | oracle cos(A,B) | best ARI (method) | silhouette of true labels |
|---|---|---|---|---|
| 0.5 s / 0.25 s | 207 | — | — (forced split scattered) | — |
| 1.0 s / 0.5 s | 103 | 0.7338 | 0.644 (spectral) | 0.0891 |
| **2.0 s / 1.0 s** | 53 | 0.7285 | **0.708 (Ward)** | **0.1817** |
| 3.0 s / 1.5 s | 36 | 0.7329 | 0.673 (spectral) | 0.2568 |

Consistent with the duration curve in
`specs/20260922-the-multi-speaker-instrument/speaker-verification-assessment.md`: below about
1 s the embedding does not separate. 2.0 s / 1.0 s is also the grid the shipped speaker-vector
profile already uses, so choosing it adds no new unmeasured constant.

## Calibration sample

Two arms, both drawn with seed 17 from the per-recording counts the earlier
`extend_diarization` campaign already computed, and resolved against
`triage_rerun_20260923/out`:

* **`multi`, 24 recordings** — pyannote already calls them two-speaker. Pool: 2,616.
  These are the positives a candidate must keep finding.
* **`single`, 24 recordings** — sustained-vowel, maximum-phonation, diadochokinesis and
  glide tasks that pyannote calls one speaker. Pool: 13,335. A second voice in one of these
  is a recording fault, not a protocol feature, so these are the negatives.

The false-positive rate on the `single` arm is what decides whether a candidate can be
trusted on 43,000 recordings. A detector that finds two speakers everywhere is worse than
the incumbent, which is at least reliably conservative.

## Measurements pending

* The threshold sweep on the incumbent's VBx clustering.
* Per-arm silhouette distributions for every candidate clusterer.
* The five alternative backends on the known case and on both arms (GPU job, subprocess
  venvs building first).
* Per-recording runtime on an idle GPU node, and the corpus-pass cost that follows from it.

## Defects found on the way

* `_second_diarizer_model` (`nodes/speech.py`) always builds a plain `HFModel`, and
  `diarize_audios` dispatches to pyannote only on `isinstance(model, PyannoteAudioModel)` —
  a subclass of `HFModel`, not the reverse. So configuring `speech.second_diarizer` to any
  pyannote checkpoint falls through every prefix branch and raises `NotImplementedError`.
  Only the five HF-prefixed backends work through that path. Tests miss it because they
  monkeypatch `_second_diarizer_model`.
* `max_concurrent_speakers` is computed by `speaker_activity()` and written onto every
  `<stream>_diarization` measurement, and nothing in the tree reads it back. It is the
  overlap statistic — the one signal that distinguishes two people talking over each other
  from one person — and it has never been consulted.
* `BUT-FIT/diarizen-wavlm-large-s80-md` weights are **CC BY-NC 4.0, non-commercial only**.
  That is a licensing question for the owner regardless of how it scores.
