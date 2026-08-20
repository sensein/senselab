# SPEECH branch — quality over speech, one speaker or many, a transcript with spans

Decided 2026-08-20. This file governs and replaces the earlier node-graph design outright.

## What it decides

Whether the recording's speech is measurable, how many speakers it holds, and what was said where.
It does not decide what the speech *means*, and it does not detect airway events — that is the airway
branch, whose product it reads.

## Signature

```
speech(derivatives, airway_spans?, hint?) -> fail(reason) | flag(reason, partial) | pass(product)
```

Every input is a PREPROCESS derivative or another branch's product. This branch runs pyannote, an
optional second diarizer, an optional speaker-embedding comparison, and optional separation. **It runs
no ASR**: PREPROCESS does, and this branch compares what it is given.

| input | from | used for |
| --- | --- | --- |
| `silence`, `energy_envelope` | PREPROCESS | coarse speech regions; the floor every span rule references |
| `squim` | PREPROCESS, plain signal | file-level quality — **necessary but not sufficient**, see below |
| `asr_crisperwhisper`, `asr_qwen`, `alignment` | PREPROCESS, plain signal | the transcript, its word edges, and the agreement between two recognizers |
| `spectrogram_wb`, `gammatone` | PREPROCESS | the figure only |
| `airway_spans` | AIRWAY | withdrawing pyannote segments that are coughs, not turns |
| `hint` | caller | conditions decisions, never measurements |

## What it does, in five steps

### 1. Coarse speech regions — YAMNet `Speech`, and only as a presence gate

`Speech` is the cleanest gate measured anywhere in this design. On the labelled recording its scores
across 29 windows are 0.993, 0.987, 0.980, 0.970, 0.919, then **0.145** — a **0.77-wide empty gap**, so a
0.5 threshold is not fitted to anything.

**Its boundaries are not speech edges, and the error is large and one-directional.** The region where
`Speech ≥ 0.5` runs 10.08–13.44 s against a label of 11.62–13.20:

| | region | label | error |
| --- | --- | --- | --- |
| onset | 10.08 s | 11.62 s | **−1.54 s** |
| offset | 13.44 s | 13.20 s | +0.24 s |

A 0.96 s window with a 0.48 s hop cannot do better than its own hop, and the −1.54 s onset is worse
than that: the window at 10.08–11.04 s scores `Speech` **0.9194** over a stretch the labels call
nothing. So this step establishes *where to look*, and **edges come from step 4**. This is the same
division the airway branch draws between HeAR and the envelope, for the same reason.

**One unresolved observation, recorded rather than explained.** That 10.08–11.04 s window is not empty:
the envelope carries real energy through roughly 10.3–11.2 s, and an early span rule of the airway
branch proposed a span there. Either the recording holds speech the labels do not mark, or YAMNet is
confidently wrong over a full second. Nothing here settles it, and **the branch must not treat the
label file as exhaustive** — a coarse region with no label is not evidence of a false positive.

### 2. Quality, over speech regions and never over the file

SQUIM is a *speech*-quality estimator, so what it is given decides whether its answer means anything.
Measured on the labelled recording:

| region | duration | STOI | PESQ | SI-SDR |
| --- | --- | --- | --- | --- |
| whole file | 14.03 s | 0.864 | 1.34 | **−12.92** |
| **the speech label alone** | 1.58 s | **0.950** | **1.69** | **+12.40** |
| cough 1 | 0.57 s | 0.443 | 1.12 | −13.68 |
| exhalation 1 | 1.22 s | 0.388 | 1.10 | −15.84 |
| silence | 1.00 s | 0.397 | 1.23 | −13.51 |

**SI-SDR flips sign — a 25 dB swing between the file and its speech.** A gate reading the file-level
number would reject a recording whose speech is in fact clean, and on this file the speech is 11% of the
duration, so the file-level figure is mostly a measurement of coughs and silence.

Two consequences. **The gate reads SQUIM over the coarse regions from step 1**, not `squim` from
PREPROCESS — which means PREPROCESS's file-level `squim` is *not* this branch's quality input, and one
of the two must change: either PREPROCESS learns to emit per-region SQUIM given regions, or this branch
computes it and PREPROCESS's row loses this consumer. **That decision is open and is flagged rather
than silently resolved.** And **a low SQUIM score on non-speech is not a quality finding at all** — the
0.397 on pure silence shows the estimator returns a confident number for input it cannot interpret, so
scoring anything but speech produces noise that looks like data.

Nothing here is fitted. There is no threshold on STOI or PESQ in this file, because no threshold has
been derived from labelled verdicts; the gate's constants are a config artifact with a written
derivation, and until that exists this step reports the numbers and does not dismiss on them.

### 3. Speaker count — pyannote, and the coughs it counts as turns

`pyannote/speaker-diarization-community-1` on the labelled recording returns **three segments, all
`SPEAKER_00`**:

| segment | what is actually there |
| --- | --- |
| 7.95–9.01 s | **cough 1** |
| 9.51–10.19 s | **cough 2** |
| 11.62–13.06 s | the speech |

**The count is right and two thirds of the evidence is wrong.** 1.74 s of the 3.18 s pyannote calls
speech is cough. So a speaker count taken from pyannote is trustworthy on this file while *anything
derived from its spans* — speech duration, turn count, speaking rate — is not, unless the airway spans
are subtracted first. That is why `airway_spans` is an input: a pyannote segment overlapping an
airway-labelled span is **withdrawn**, not relabelled.

Its speech segment is worth its own note: onset **11.62 s against a label of 11.62 s** — exact — and
offset 13.06 against 13.20, **−140 ms early**. The same onset-sharp/offset-early asymmetry that holds
across every instrument in this design.

**The count's codomain is `{1, 2, ≥3}`.** One speaker is the case this project cares about; two is a
different problem; three or more is a recording nobody intended. Escalation to a second diarizer
happens only when the count is not 1, and **whether any alternative is better is not established** —
the figures available compare diarizers at different speaker counts (VibeVoice 95% at k=2, DiariZen
75–90% at k=2–3, pyannote 85%), which is not a comparison. Escalation is therefore specified as
"consult a second diarizer and report disagreement", not "replace pyannote with a better one".

### 4. Transcript and edges — two recognizers, compared not merged

PREPROCESS supplies `asr_crisperwhisper` and `asr_qwen` with word edges, and `alignment` over the
transcript they agree on. This branch's work is the comparison:

- **word-level agreement** between the two hypotheses becomes the per-word confidence;
- **edges** come from the recognizer with measured edge accuracy on a verified span (CrisperWhisper:
  onset −13 ms, offset −27 ms, coverage 98.3%), not from an average of the two;
- **a word slot over no energy and no periodicity** is a fabrication candidate, tested against
  `energy_envelope` and its silence-derived floor.

Agreement is not accuracy. Two recognizers sharing a training distribution can agree and both be
wrong, so agreement bounds confidence from above and is reported as agreement, never as correctness.

### 5. Separation — available, and not yet justifiable here

`MossFormer2_SS_16K` produces a clean, interpretable two-way split of this recording. The streams are
orthogonal (`corr(src0, src1) = +0.0003`) and each concentrates a different kind:

| region | src0 − mixture | src1 − mixture |
| --- | --- | --- |
| speech | **+9.17 dB** | −16.57 dB |
| cough 1 | −10.53 dB | −1.95 dB |
| cough 2 | −8.05 dB | −0.95 dB |
| exhalation 1 | −0.59 dB | −47.31 dB |

So **src0 is speech plus exhalations and src1 is the coughs** — a partition that matches the branch
split this design already draws, from a model that knows nothing about it.

**And it cannot be scored for benefit on this recording.** SQUIM over the speech region moves 0.950 →
0.960, which is nothing, because the speech was already clean and **the speech and the airway events
never overlap in time**. A +9.17 dB gain against sources that were not concurrent buys no intelligibility.
Separation earns its place only where speech and airway events coincide, and **this file cannot
demonstrate that either way** — so separation stays optional, off by default, and its justification
waits for a recording with overlap.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | no coarse speech region at all, or quality over the speech regions is too poor to measure by a derived threshold that does not yet exist |
| `flag` | speaker count is not 1; the two recognizers disagree beyond threshold; fabrication candidates survive; a hint asserts speech content the branch did not find; or a target was given and no speaker matches it |
| `pass` | a transcript with per-word confidence, spans attributed to speakers, and quality measured over the speech |

**`fail` is currently unreachable by the quality route**, because the threshold it needs has not been
derived. That is stated rather than papered over with a literal: an unfitted constant here would
dismiss recordings on a number nobody measured.

## The product

```
transcript:    [ { word, start, end, confidence, speaker } ]
speaker_spans: [ { start, end, speaker, withdrawn: bool, withdrawn_because } ]
speaker_count: 1 | 2 | ">=3"
quality:       { per_region: [ {start, end, stoi, pesq, si_sdr} ], regions_measured_s }
target_match:  { speaker, similarity } | absent
figure:        one aligned figure per recording
```

`withdrawn` travels with the span rather than the span being deleted, because a pyannote segment that
was withdrawn as an airway event is exactly what a reader needs to see when the count and the spans
disagree.

`target_match` exists only when a target embedding was supplied. Absent a target, the branch names
speakers `SPEAKER_*` and claims no identity.

**The figure** carries the envelope, YAMNet `Speech` and `Silence`, pyannote's segments with the
withdrawn ones distinguished, per-region SQUIM, the separated envelopes when separation ran, and the
wideband spectrogram — one aligned time axis. Generating script `spplot.py` beside this file.

## What this branch does not do

No ASR — PREPROCESS runs it. No airway detection — it reads `airway_spans`. No speaker identity without
a supplied target. No emotion, no language identification, no diarizer ranking (see step 3). And no
quality dismissal until a threshold exists with a derivation behind it.

## Limits on every number here

One recording, one healthy adult, one speaker, 1.58 s of speech, close mic. Every figure above
justifies the *shape* of a rule and none of them a constant. Three in particular are single
observations: pyannote's exact onset, CrisperWhisper's −13 ms, and the 0.77 gap in YAMNet `Speech`.
