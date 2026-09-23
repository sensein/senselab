# Results

Everything below was measured on ORCD, pinned checkout `0f7997bc` at
`/orcd/scratch/bcs/002/satra/senselab-secondvoice`, work directory
`/orcd/scratch/bcs/002/satra/secondvoice_20260923`. Timings are from an idle H100 (node2803)
or A100 (node3805) allocation; the laptop figures taken earlier are discarded because the
machine was under load 17–25 and they are upper bounds at best.

Every model loaded at a resolved 40-hex commit. Pyannote
`speaker-diarization-community-1` = `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.

## 1. The known case, per backend

Jobs 23558004 (`sv-backends`). `plain` stream, mono 16 kHz, 63.60 s. Ground truth: one
second voice holding 0.5 – 15.6 s.

Scoring: the **minority label** is each backend's claim about the second voice, since the
participant holds the bulk of every recording in this corpus. Recall is of the 15.13 s
ground-truth span; precision is the share of that label's own seconds falling inside it.

| backend | speakers | segs | minority label | recall | precision | wall s | needs |
|---|---|---|---|---|---|---|---|
| pyannote community-1 (incumbent) | **1** | 9 | — | **0** | — | 1.67 | in-process |
| pyannote community-1, `num_speakers=2` | 2 | 9 | 12.22 s (0.03–15.66) | **0.775** | **0.959** | ~1.7 | in-process; the only backend honouring the hint |
| NVIDIA Sortformer `diar_sortformer_4spk-v1` | **1** | 25 | — | **0** | — | 54.8 | subprocess venv (NeMo); structural cap 4 |
| VibeVoice-ASR-HF | **1** | 4 | — | **0** | — | 104.0 | in-process, `transformers>=5.3` |
| **MOSS-Transcribe-Diarize** | **2** | 12 | 9.37 s (3.68–15.68) | **0.617** | **0.996** | 93.1 | subprocess venv, `transformers>=5.6` |
| DiariZen `diarizen-wavlm-large-s80-md` | 2 | 25 | 1.12 s (0.51–2.49) | **0.074** | 1.000 | 44.9 | subprocess venv; **weights CC BY-NC 4.0** |
| USC-SAIL child-adult | 2 | 28 | CHILD 0.18 s | **0.012** | 1.000 | 105.5 | subprocess venv, CUDA only, role labels, cap 2 |

**MOSS-Transcribe-Diarize is the only backend that found the second voice unprompted.** It
placed 9.33 of its 9.37 minority seconds inside the true span and recovered the turn change
at 15.68 / 15.69 s to within 0.01 s. It missed only the administrator's first utterance
(0.39–2.55 s), which it gave to the participant.

DiariZen and child-adult both report two, but their second label holds **1.12 s** and
**0.18 s** — the same shape as the micro-splits that dominate the corpus's existing
two-speaker population, not a finding about the 15 s turn. Counting either as a hit would be
reading a true count off a false reason. Child-adult's labels are roles, not identities, so
even a correct count would not say "a second person".

One stream observation, recorded and not pursued because the owner ruled the stream
comparison out: on the `enhanced` stream VibeVoice returns **2** speakers where on `plain`
it returns 1, while pyannote and Sortformer return 1 on both.

## 2. Retuning the incumbent: the documented knob is inert

`config.yaml` ships `params.clustering.threshold: 0.6` for `VBxClustering`. Swept across
**0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95** on the known case:

* every value returns **1 speaker, 51.705 s**, byte-identical.
* the parameter is demonstrably applied — the instrumentation prints
  `after={'threshold': 0.05, …} obj_threshold=0.05 obj=VBxClustering` at each step.

Swept across the calibration sample too: 24 recordings at 0.6 / 0.5 / 0.45 / 0.4 / 0.35 /
0.3, then 72 recordings at 0.6 / 0.4 / 0.2. **Zero of the 96 recordings changed their
output at any value** — not the count, not the per-speaker seconds.

**So this pipeline's speaker count is not reachable through the threshold parameter its own
config exposes.** Retuning the incumbent is not a candidate, at least not through that knob.

## 3. Clustering on voice embeddings: measured, and it does not separate

ECAPA `speechbrain/spkrec-ecapa-voxceleb`, 2.0 s windows on a 1.0 s hop over pyannote's own
speech spans — every window exactly 2.0 s, so duration is controlled by construction.

**Positives** (`multi`): 24 recordings the incumbent calls two-speaker, ≥ 20 s of speech.
**Negatives** (`speech_single`): 40 connected-speech recordings of comparable length that
the incumbent calls one-speaker (caterpillar, harvard sentences, picture description, story
recall, free speech). A third arm of 8 sustained-vowel / DDK recordings is reported but is
too short to window (median 6.2 s, only 3 usable) and is not the operating comparison.

Mann-Whitney AUC of positives over negatives, best statistics first by distance from chance:

| statistic | AUC | pos median | neg median | known case | negatives admitted to catch it |
|---|---|---|---|---|---|
| AHC-Ward minority share | 0.680 | 0.405 | 0.283 | 0.321 | 0.42 |
| AHC-complete centroid cosine | 0.371 | 0.683 | 0.780 | 0.712 | **0.33** |
| AHC-average centroid cosine | 0.381 | 0.475 | 0.575 | 0.664 | 0.80 |
| AHC-Ward separation margin | 0.379 | −0.095 | −0.071 | −0.033 | 0.70 |
| k-means minority share | 0.603 | 0.405 | 0.351 | 0.302 | 0.70 |
| AHC-Ward silhouette | 0.407 | 0.129 | 0.144 | 0.178 | 0.78 |
| hinted-run minority seconds | 0.584 | 16.83 | 12.88 | 12.22 | 0.60 |
| AHC-complete silhouette | 0.560 | 0.179 | 0.159 | 0.188 | 0.45 |
| hinted-run minority share | 0.457 | 0.399 | 0.403 | 0.236 | 0.90 |
| AHC-Ward centroid cosine | 0.500 | 0.810 | 0.818 | 0.750 | 0.85 |

Twenty-six statistics were scored; none reaches 0.70 AUC and several are inverted. **The best
of them catches the known case only by also calling a third of genuine single-speaker
connected-speech recordings two-speaker.** On 43,000 recordings that is roughly 14,000 false
flags to win one true one.

Two caveats that cut in opposite directions and are both worth stating:

* the positive arm is weak. In the `multi` arm the incumbent's second speaker holds a median
  of **0.30 s** out of 20–90 s of speech. These are micro-splits, not conversations, so the
  AUC is partly measuring "can the embedding tell a micro-split from a clean recording",
  which is not the question asked.
* the `neg ≥ target` column does not depend on the positive arm at all. It compares the
  known case directly against 40 recordings that really do hold one voice, and 0.33 is the
  best any of the 26 statistics achieves. That number stands on its own.

### Why it fails, concretely

At the ground-truth labelling of the known case: centroid cosine between the two speakers
**0.7285**, mean within-speaker cosine **0.7604** (the 15 s administrator) and **0.7133**
(the 45 s participant). The between-speaker distance is *smaller* than the within-speaker
spread of the larger talker. Cosine silhouette of the true labels is **0.1817** — the median
for a genuine single-speaker recording forced into two groups is **0.159** (AHC-complete).

The partition is nonetheless recoverable: AHC with **complete** linkage gives ARI **0.697**
and Ward **0.708** against ground truth, while **average** linkage — the linkage
`select_dominant_vectors` uses — gives **−0.063**, peeling three outliers off one speaker's
cloud instead. So the embedding knows *where* the boundary is and cannot tell you *whether*
there is one.

### What the shipped entry points say on this recording

* `select_dominant_vectors` (AHC average, fitted merge-gap margin 5.0): **one cluster**.
* `cluster_pass_speakers` (spectral, silhouette sweep): **`n_speakers: 1`**, best silhouette
  0.187.

Both refuse it, and given the AUCs above, both are right to be conservative.

## 4. What the incumbent currently counts, corpus-wide

Every `enhanced_diarization.npz` in `triage_rerun_20260923/out` read back (job 23559165):
**62,518 recordings, all readable.**

| count | recordings |
|---|---|
| 0 speakers | 12,254 |
| 1 speaker | 47,500 |
| 2 speakers | 2,762 |
| 3 speakers | 2 |
| **≥ 2** | **2,764 = 4.42 %** |

Overlap greater than zero: 2,723 = 4.36 %.

Restricted to the 8,062 recordings with at least 20 s of speech, 12.88 % get two.

Two-speaker rate by task, tasks with ≥ 200 recordings:

| task | rate |
|---|---|
| cinderella-story | 26.4 % |
| caterpillar-passage | 19.6 % |
| story-recall-v2 | 18.3 % |
| random-item-generation | 14.3 % |
| story-recall | 13.5 % |
| word-color-stroop | 11.9 % |
| free-speech | 10.5 % |
| picture-description | 10.5 % |
| **maximum-phonation-time-v2-1** | **10.1 %** |

The last row is the tell. **A sustained-vowel task — one person holding a single vowel —
gets a second speaker on one recording in ten.** Together with the median second-speaker
duration of 0.30 s in the sampled positives, this says the existing `speaker_count ≥ 2`
population is substantially diarizer noise, and any consumer treating it as "two people were
present" is reading it wrong.

## 5. Cost

Amortized over 73 recordings on an idle A100, models resident, median audio 39.9 s:

| step | median wall s | × realtime |
|---|---|---|
| pyannote diarization, free | 0.32 | 125 |
| pyannote diarization, `num_speakers=2` | 0.32 | 125 |
| ECAPA over the 2.0 s / 1.0 s window grid | 0.04 | 1,080 |

Cold-start figures for the subprocess-venv backends on one 63.6 s recording (first call,
venv already built, weights already cached): Sortformer 54.8 s, VibeVoice 104.0 s, MOSS
93.1 s, DiariZen 44.9 s, child-adult 105.5 s. These are dominated by process start and model
load, so a corpus pass amortizes them — the MOSS sample run is what prices that honestly.

Corpus scale: **43,335 recordings run SPEECH**, mean 34.7 s, median 24.7 s (corpus census),
so about **418 hours of audio**.

Amortized rates measured over the sample runs, models resident:

| candidate | rate | corpus pass over 418 h |
|---|---|---|
| pyannote, one pass | 151 × realtime | 2.8 GPU-hours |
| second pyannote pass with `num_speakers=2` | 125 × realtime | **3.3 GPU-hours** |
| ECAPA window grid | 1,080 × realtime | **0.4 GPU-hours** |
| NVIDIA Sortformer | 2.9 × realtime | 147 GPU-hours |
| **MOSS-Transcribe-Diarize** | **2.7–3.7 × realtime** | **113–155 GPU-hours** |
| DiariZen | 4.3–5.3 × realtime | 79–97 GPU-hours |
| VibeVoice-ASR-HF | 4.2 × realtime | 99 GPU-hours |
| USC-SAIL child-adult | 14.4 × realtime | 29 GPU-hours (2 of 7 failed) |

For the pyannote and ECAPA candidates cost is not the obstacle; the false-positive rate is.
MOSS costs two orders of magnitude more than a second pyannote pass, but 113–155 GPU-hours
for a one-off corpus pass is comparable to the 132 CPU-hours the separation instrument
already spends each pass.

## 6. False positives on the matched negative arm — interim

Jobs **23559301** (`sv-mossneg`, negatives first) and **23558701** (`sv-moss`, whole matched
sample) are still running; **23558004** (`sv-backends`, all six backends over the first
sample) likewise. What has landed so far, on `speech_single` — long single-speaker
connected speech, the arm that decides this:

| backend | reported ≥ 2 | with minority ≥ 3 s |
|---|---|---|
| MOSS-Transcribe-Diarize | **0 / 10** | **0 / 10** |
| DiariZen | 1 / 10 | 0 / 10 |

DiariZen's one call gives its second label under 0.5 s — the micro-split signature again.

And on the `multi` arm — recordings the incumbent calls two-speaker, which the census above
suggests are mostly micro-splits:

| backend | reported ≥ 2 | with minority ≥ 3 s | n so far |
|---|---|---|---|
| pyannote (incumbent) | 7 / 8 | **1 / 8** | 8 |
| VibeVoice-ASR-HF | 4 / 7 | 4 / 7 | 7 |
| USC-SAIL child-adult | 4 / 5 | 0 / 5 | 5 (+2 failed) |
| MOSS-Transcribe-Diarize | 0 / 7 | 0 / 7 | 7 |
| DiariZen | 0 / 7 | 0 / 7 | 7 |
| NVIDIA Sortformer | 0 / 7 | 0 / 7 | 7 |

Read that second table carefully: MOSS, DiariZen and Sortformer *disagree* with the
incumbent on almost every one of its existing two-speaker calls. Given that the incumbent's
second speaker there holds a median 0.30 s, and that it calls 10.1 % of sustained-vowel
recordings two-speaker, the disagreement is most likely the alternatives being right.
Recall against that population is therefore the wrong thing to ask of a candidate, and the
only trustworthy positive in this whole exercise is the one recording with ground truth.
