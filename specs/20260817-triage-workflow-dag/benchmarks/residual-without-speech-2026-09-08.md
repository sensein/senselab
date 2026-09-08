# FRCRN's residual without speech — eight cough/breath recordings, two speech comparators

The PREPROCESS step under development computes a background stream as
`residual = original - g*FRCRN_SE_16K(original)`. That formula is validated on speech: on
`sub-17578482-…_task-Story-recall.wav` (the standing reference,
`~/Downloads/buzz_separation_20260906/00_original.wav`) it raises `Buzz` 0.066 → 0.372, leaves
1.57% of the input's energy, and fits a gain of 1.003 at zero lag. `FRCRN_SE_16K` is a
**speech**-enhancement model; the b2ai `Respiration-and-cough-*` tasks contain no speech at all, so
its behaviour there is outside its training target, and whether the residual formula still means
"the noise that was removed" was unmeasured. This measures it.

**Answer, ahead of the numbers**: it depends on structure, not on subject or nominal task name.
Recordings holding a **brief, isolated burst against an otherwise-quiet stretch** — three of four
`Cough` recordings, including a single hard cough with the shortest duration of the whole set —
reproduce the speech reference almost exactly: gain within 0.2 dB of unity, residual 0.15-1.5% of
input energy, and the burst's own labels (`Cough`, `Sneeze`, `Breathing`) stay in the enhanced
output while the residual carries none of them. Recordings where the target sound **fills most or
all of the duration** — all four `Breathe` recordings, plus the one `Cough` recording that runs
11.5 s of repeated coughing rather than a few seconds of isolated bursts — do not: gain runs 8-36 dB
from unity, the residual retains 57-94% of the input's energy, and either the enhanced output goes
to near-silence while the residual carries the recording's actual content, or the same
label (`Breathing`, `Cough`) shows up at nearly equal strength in *both* streams. Neither failure
mode looks like "background"; both are covered in detail below.

## Method

1. Load each recording as a senselab `Audio`.
2. Run `alibabasglab/FRCRN_SE_16K` through `senselab.audio.tasks.speech_enhancement.enhance_audios`
   (the ClearerVoice subprocess venv) — all ten recordings in **one call** (209.35 s of audio total),
   so the ~22 s/recording cost from the venv startup is paid once, not ten times.
3. For each recording, run the existing, already-validated
   `~/Downloads/buzz_separation_20260906/residual/subtract.py` on `(original, enhanced)`: it
   cross-correlates to find the lag, fits `g = <original, enhanced> / <enhanced, enhanced>` on the
   aligned overlap, and writes `residual_direct = original - g*enhanced` plus a JSON report (gain,
   lag, peak/RMS dBFS, energy fraction and dB, four-band energy split). This benchmark uses the
   *direct* (sample-domain) residual and gain only — `subtract.py`'s alternative spectral-subtraction
   residual is not used here.
4. One batched `classify_audios(model="yamnet", top_k=521)` call over all 30 files (10 original + 10
   enhanced + 10 residual). YAMNet returns per-0.96 s-window scores; each file's label score below is
   the **max over that file's windows** — the same aggregation `~/Downloads/buzz_separation_20260906/residual/yamnet_report.json`
   used for the speech reference, so the two are comparable.
5. Band energy fractions (0-200 / 200-1000 / 1000-4000 / 4000-8000 Hz) come straight out of the
   `subtract.py` report for original and residual.

Script: [`scripts/residual_without_speech.py`](scripts/residual_without_speech.py). It imports
`subtract.py` from its actual path rather than duplicating its logic, per the task instruction to
use it as-is.

**ClearerVoice's `ClearVoice` identifier is deliberate**, not a typo — the pip package for
ClearerVoice-Studio exports the class `ClearVoice`, and the backend already imports it correctly;
no changes were needed to run FRCRN. `uv sync --all-extras --group dev` was run before anything.

## Files used

Source tree `~/Downloads/b2ai_v31_bids_07_01_v3/`. All ten files are 16 kHz mono PCM_16, no
resampling needed. Eight non-speech recordings across three subjects, plus two `Story-recall`
speech comparators (one of which is the standing buzz-separation reference recording, subject
`sub-17578482`, already measured against FRCRN elsewhere — reused here rather than re-run, and
consistent with that prior measurement to within noise, see the table below):

| key | subject | task | instructions | duration |
| --- | --- | --- | --- | --- |
| `s1_cough1` | sub-17578482 | Respiration-and-cough-Cough-1 | "cough 5 times" | 6.11 s |
| `s1_cough2` | sub-17578482 | Respiration-and-cough-Cough-2 | "cough 5 times" | 6.76 s |
| `s1_fivebreaths1` | sub-17578482 | Respiration-and-cough-FiveBreaths-1 | "5 big breaths in and out through your nose" | 20.57 s |
| `s2_cough1` | sub-17cee767 | Respiration-and-cough-Cough-1 | "cough 5 times" | 11.54 s |
| `s2_breath1` | sub-17cee767 | Respiration-and-cough-Breath-1 | "breathe comfortably for 30 seconds" | 30.02 s |
| `s2_threequickbreaths1` | sub-17cee767 | Respiration-and-cough-ThreeQuickBreaths-1 | "3 quick breaths, catching your breath" | 6.99 s |
| `s3_hardcough` | sub-1f4ea26f | Respiration-and-cough-(v2)-HardCough | "cough HARD as if something were stuck in your throat" | 2.46 s |
| `s3_breath` | sub-1f4ea26f | Respiration-and-cough-(v2)-Breath | "breathe comfortably through your mouth for 20 seconds" | 19.85 s |
| `s1_storyrecall` | sub-17578482 | Story-recall | speech (read-then-recall) | 25.54 s |
| `s3_storyrecall` | sub-1f4ea26f | Story-recall-(v2) | speech (frog-story recall) | 79.51 s |

Full relative paths are under each subject's `ses-*/audio/` directory, named
`<subject>_<session>_task-<Task>.wav`; the exact paths are recorded in
[`scripts/residual_without_speech.py`](scripts/residual_without_speech.py)'s `RECORDINGS` list and
in `subtract_reports.json` alongside the audio (`input` field), so every number below traces back to
one file.

Every subject's non-lexical recordings carry `"speech_type": "non-lexical"` and an empty
`stimulus_text` in their `recording-metadata.json` — confirmed by inspection, not assumed.

## Per-recording table

Gain, lag and energy from `subtract.py`; all recordings had **zero lag** (FRCRN does not delay its
output). Correlation is `corr(original, g*enhanced)` on the aligned region.

| key | gain (dB) | corr | residual energy (% of input) | residual (dB below input) |
| --- | --- | --- | --- | --- |
| `s1_cough1` | +0.20 | 0.992 | 1.51% | −18.20 |
| `s1_cough2` | +0.10 | 0.995 | 1.03% | −19.86 |
| `s3_hardcough` | +0.15 | 0.999 | **0.15%** | −28.11 |
| `s1_storyrecall` (speech) | +0.03 | 0.992 | 1.57% | −18.03 |
| `s3_storyrecall` (speech) | +0.12 | 0.993 | 1.39% | −18.56 |
| `s1_fivebreaths1` | **+15.45** | 0.252 | 93.64% | −0.29 |
| `s2_cough1` | **+18.26** | 0.358 | 87.17% | −0.60 |
| `s2_breath1` | **+35.70** | 0.583 | 67.09% | −1.73 |
| `s2_threequickbreaths1` | **+8.41** | 0.280 | 92.27% | −0.35 |
| `s3_breath` | **+8.05** | 0.654 | 57.24% | −2.42 |

The split is sharp and binary, not a gradient: five recordings sit within 0.2 dB of unity gain with
correlation ≥0.99 and residual ≤1.6% (matching both speech comparators to within noise); the other
five sit 8-36 dB from unity with correlation ≤0.65 and residual 57-94%. Nothing falls in between.

### What the enhanced output actually contains (the decisive measurement)

RMS of `FRCRN(original)` relative to the input, and the enhanced output's own YAMNet watch-label
scores (max over windows; blank means <0.01 for every watch label):

| key | enhanced RMS vs input | enhanced YAMNet (watch labels ≥0.01) |
| --- | --- | --- |
| `s1_cough1` | −0.26 dB | Cough 0.187, Breathing 0.350, Sneeze 0.531, Throat clearing 0.128, Speech 0.314 |
| `s1_cough2` | −0.14 dB | Cough 0.088, Breathing 0.418, Sneeze 0.341, Speech 0.567 |
| `s3_hardcough` | −0.15 dB | Breathing 0.110, Sneeze 0.031, Speech 0.602 |
| `s1_storyrecall` | −0.10 dB | Speech 1.000, Breathing 0.017, Sneeze 0.038 |
| `s3_storyrecall` | −0.18 dB | Speech 1.000, Breathing 0.054, Snoring 0.020, Buzz 0.010 |
| `s1_fivebreaths1` | **−27.42 dB** | Breathing 0.039, Sneeze 0.011, Snoring 0.011, Speech 0.417 |
| `s2_cough1` | **−27.18 dB** | Cough 0.752, Breathing 0.125, Sneeze 0.682, Throat clearing 0.371, Speech 0.925 |
| `s2_breath1` | **−40.52 dB** | *(none ≥0.01 — effectively silent)* |
| `s2_threequickbreaths1` | **−19.53 dB** | Breathing 0.993, Sneeze 0.024, Snoring 0.983, Speech 0.218 |
| `s3_breath` | **−11.74 dB** | Breathing 0.919, Sneeze 0.465, Snoring 0.828, Speech 0.546 |

`s2_breath1` is the clean case the "Why" section worried about: FRCRN drives a 30 s comfortable-breathing
recording to −79.8 dBFS RMS (from −39.3 dBFS input) — every watch label falls under 0.01, i.e. the
model produced something YAMNet cannot distinguish from silence. `s2_cough1` and `s3_breath` show a
different, equally disqualifying failure: the target label (`Cough` 0.752, `Breathing` 0.919) is
**not** discarded from the enhanced output — it is *duplicated*, showing up at nearly the same
strength in the residual (below).

### What the residual actually contains

Residual YAMNet watch labels (max over windows, ≥0.01 only) and the four-band energy split, original
vs. residual:

| key | residual YAMNet (watch labels ≥0.01) | band energy, original → residual (0-200/200-1k/1-4k/4-8k Hz, %) |
| --- | --- | --- |
| `s1_cough1` | Noise 0.017 | 45/17/24/14 → 17/31/15/**37** |
| `s1_cough2` | *(none)* | 22/17/41/20 → 5/30/16/**49** |
| `s3_hardcough` | *(none)* | 3/76/21/1 → 46/47/5/2 |
| `s1_storyrecall` | Buzz 0.953, Hum 0.368, Mains hum 0.337, Speech 0.148, Snoring 0.013 | 56/35/5/4 → 33/29/25/13 |
| `s3_storyrecall` | *(none)* | 9/82/9/0 → **98**/1/0/0 |
| `s1_fivebreaths1` | Speech 0.085, Buzz 0.113, Hum 0.277, Mains hum 0.144, Noise 0.225 | 78/21/1/0 → **78/21/1/0 (unchanged)** |
| `s2_cough1` | **Cough 0.704**, Speech 0.970, Sneeze 0.688, Throat clearing 0.458 | 14/35/44/7 → **12/34/46/7 (unchanged)** |
| `s2_breath1` | Speech 0.975, **Breathing 0.327**, Snoring 0.163, Cough 0.024 | 83/16/1/0 → 78/17/3/2 (nearly unchanged) |
| `s2_threequickbreaths1` | **Breathing 0.991**, Snoring 0.978, Speech 0.321, Buzz 0.031 | 17/50/30/3 → 12/54/31/3 (small shift, not the removed-content shape) |
| `s3_breath` | Speech 0.864, **Breathing 0.511**, Snoring 0.432 | 99/1/0/0 → 99/1/0/0 (unchanged) |

The five reference-like recordings' residual band split moves sharply away from the original's — by
17-89 percentage points in the largest-moving band of each, and not in one consistent direction
(toward 4-8 kHz for `s1_cough1`/`s1_cough2`/`s1_storyrecall`, toward the sub-200 Hz floor for
`s3_hardcough`/`s3_storyrecall`, i.e. each recording's own noise floor rather than a shared shape),
consistent with a noise floor rather than the removed content. The five outlier recordings'
residual band split stays close to the original's throughout — every band moves by at most 5.7
percentage points — which is consistent with "little of the input was actually subtracted," not with
a distinct background having been isolated. This is independent of, and agrees with, the YAMNet
finding above.

## Per-recording verdict

- **`s1_cough1`, `s1_cough2`, `s3_hardcough` — residual is the background, as intended.** Gain
  ≈unity, correlation ≥0.99, residual 0.15-1.5% of input energy — matching both speech comparators.
  The cough's own labels (`Cough`, `Sneeze`, `Breathing`) stay in the enhanced output; the residual
  carries none of them, just a noise floor with a spectral shape unlike the original's (shifted
  toward 4-8 kHz for the two cough bursts, toward the sub-200 Hz floor for the hard cough). `s3_hardcough`
  — a single hard cough, the shortest recording in the set at 2.46 s — has the *smallest* residual
  fraction measured anywhere, including the speech reference (0.15% vs. 1.57%).

- **`s1_fivebreaths1` — the residual is (almost exactly) the original, i.e. incoherent, not
  background.** FRCRN's output correlates with the input at only 0.252 and sits 27 dB below it in
  RMS; 93.64% of the input's energy survives subtraction, and the residual's band split is
  unchanged from the original's to within rounding. Note also that YAMNet does not read this
  original recording as breathing at all (`Wind` 0.804, `Hum` 0.391, `Buzz` 0.172, `Breathing`
  0.010) — this file's five nose-breaths clip near full scale (peak 0.99997), and the distortion
  reads as tonal/wind content to both YAMNet and, evidently, to FRCRN. The residual reproduces that
  same mislabeling.

- **`s2_cough1` — the model duplicated the cough into both streams; the residual is (also) the
  content.** This is the scenario the "Why" section named directly, just not in the form
  anticipated: FRCRN does not delete the cough from the enhanced output (`Cough` 0.752 there, up
  from 0.099 in the noisy/clipped original, because it also cleans up masking artefacts) — but the
  gain-fit subtraction leaves `Cough` at 0.704 in the residual too, alongside `Speech` 0.970 and
  `Sneeze` 0.688, all close to the enhanced output's own scores. Correlation is only 0.358; 87.17% of
  the input's energy remains. Whatever this recording's residual is, it is not "what FRCRN removed as
  noise" — the same content is legible in both halves of the split.

- **`s2_breath1` — the clean discard case: enhanced is silent, residual is the content.** FRCRN
  drives this 30 s comfortable-breathing recording to −79.8 dBFS RMS; every YAMNet watch label on the
  enhanced output falls under 0.01. The residual then necessarily carries almost the whole original
  (`Speech` 0.975, `Breathing` 0.327, `Snoring` 0.163, 67.09% of input energy, band split nearly
  unchanged from the original's). This is the exact mechanism the "Why" section described for a
  cough — realised here for a breath recording instead.

- **`s2_threequickbreaths1`, `s3_breath` — breathing duplicates into both streams, same as
  `s2_cough1`.** `Breathing`/`Snoring` sit at 0.99/0.98 in *both* the enhanced output and the residual
  for `s2_threequickbreaths1`; `s3_breath`'s residual (`Breathing` 0.511, `Snoring` 0.432, `Speech`
  0.864) is not attenuated relative to the original (`Breathing` 0.743, `Snoring` 0.494, `Speech`
  0.583) so much as re-weighted. Correlation 0.28 and 0.65, residual 92% and 57% of input energy.

## The general answer

**Whether `original - g*FRCRN(original)` means "the background" does not depend on whether the
recording contains speech or on which subject produced it — it depends on whether the target
non-speech sound is a brief, isolated event against an otherwise-quiet recording, or fills most/all
of the recording's duration.**

Three of the four `Cough` recordings and both `Story-recall` speech comparators share that "brief
event, quiet elsewhere" shape (a handful of short cough bursts, or continuous speech that FRCRN was
built to pass through) and produce a residual matching the validated speech case exactly: gain
within 0.2 dB of unity, correlation ≥0.99, residual 0.15-1.6% of input energy, target content absent
from the residual. All four `Breathe` recordings, and the one `Cough` recording that ran 11.5 s of
repeated coughing rather than a handful of isolated bursts, do not share that shape — the recording
is filled with the target sound throughout — and FRCRN either (a) drives its output to something
YAMNet cannot distinguish from silence, leaving the residual as essentially the untouched original
(`s2_breath1`, and by band-energy evidence also `s1_fivebreaths1`), or (b) reproduces the target
content in the enhanced output but at a level the least-squares gain cannot reconcile with the
input, leaving the *same* content, near-undiminished, in the residual as well (`s2_cough1`,
`s2_threequickbreaths1`, `s3_breath`). Neither failure mode is "background, weakly recovered" — both
put the recording's actual content, not noise, in the stream the design calls "background."

This is not a same-severity split of five-versus-five: the count that reproduces the validated
speech behaviour is 3-for-4 on `Cough` recordings (all but the one running 11.5 s of repeated
coughing) and 2-for-2 on the speech comparators, against 0-for-4 on `Breathe` recordings. That is a
real, structural boundary, not noise in eight measurements — but it is eight measurements on three
subjects, and the "brief-event-vs-filled-duration" account is this benchmark's explanation for the
boundary, not a property confirmed on a fourth or fifth subject.

**Does the method transfer to the speech-present comparators?** Yes, exactly. `s1_storyrecall`
(the standing buzz-separation reference) reproduces its previously-published numbers to within
rounding — gain 1.003 there vs. 1.0032 (+0.03 dB) here, residual 1.57% there vs. 1.57% here, and the
residual's content is the same background buzz family (`Buzz` 0.953, `Hum` 0.368, `Mains hum` 0.337)
while the enhanced output is clean speech (`Speech` 1.000, no buzz labels). `s3_storyrecall`, a
second subject with no salient background event, produces the same small-residual, no-content
pattern (1.39%, gain +0.12 dB, residual carries no watch label above 0.01) with no background to
recover — the "nothing there" control the method should show on a clean recording.

## What this does not license

- **Not a claim about which of the two failure modes a given non-speech recording will hit.**
  `s2_breath1` went to near-silence; `s2_cough1`, `s2_threequickbreaths1` and `s3_breath` duplicated
  their content into both streams instead. Both are disqualifying for the residual-as-background use,
  but they are different mechanisms and this benchmark does not explain what separates them beyond
  "recording filled with the target sound" — that account does not distinguish silence-collapse from
  duplication.
- **Not evidence about `MossFormerGAN_SE_16K` or `MossFormer2_SE_48K`.** Prior measurements
  (`specs/20260817-triage-workflow-dag/model-to-branch.md`,
  `specs/20260819-clearvoice-integration/design.md`) show those two checkpoints destroy breath and/or
  cough energy outright even on recordings FRCRN passes through cleanly; nothing here was re-run
  against them, and the residual formula's failure mode for those checkpoints is unmeasured.
- **Not a claim that FRCRN "detects speech absence" as a discrete decision.** The isolated-cough
  recordings and the speech recordings behave alike; the account offered here is about the
  *temporal density* of non-quiet content in the recording, which correlates with but is not the
  same variable as "contains speech."
- **Not generalizable past three subjects and eight recordings**, all from one BIDS release
  (`b2ai_v31_bids_07_01_v3`) and one microphone setup per subject. `s1_fivebreaths1` and `s2_cough1`
  both show near-clipped/loud original recordings; whether the failure mode is driven by that
  clipping, by the breathing/repeated-cough content itself, or both, is not separated here — a
  clean, unclipped, duration-filled breathing recording was not available to test that apart.
- **Not a licence to use FRCRN's residual as a background stream on `Respiration-and-cough-*`
  input.** Five of eight non-speech recordings failed outright; the three that passed cannot be told
  apart from the five that failed without already knowing (or measuring) whether the recording is
  event-sparse or duration-filled, which is exactly the information a precondition gate would need to
  check before trusting the residual — supporting a precondition on this step for non-speech input,
  which the task states is already being written into the implementation.

## Audio

Written to `~/Downloads/frcrn_residual_no_speech_20260907/` (16 kHz PCM_16, no clipping — two files
needed peak-safety scaling on write, both under 0.2 dB, reported in `subtract_reports.json`): for
each key above, `<key>__original.wav`, `<key>__frcrn.wav` (the enhanced output) and
`<key>__residual.wav`. Also `subtract_reports.json` (one `subtract.py` report per recording) and
`yamnet_by_file.json` (per-file top-5 labels and watch-label scores for all 30 files).

**Listen to `s2_breath1__original.wav` against `s2_breath1__frcrn.wav` first.** The original is 30 s
of clearly audible comfortable breathing; the enhanced output is close to dead silence (−79.8 dBFS
RMS, every YAMNet watch label under 0.01). `s2_breath1__residual.wav` then plays back as almost the
original recording. It is the single clearest audible demonstration that FRCRN can discard the
entire non-speech recording rather than isolate noise from it — the mechanism the "Why" section
described for a cough, realised here for a breath. `s2_cough1__frcrn.wav` against
`s2_cough1__residual.wav` is the second recording worth hearing: both contain an audible cough,
which is the duplication failure mode rather than the silence one.

## 2026-09-07 update — one implementation, the local/cluster contradiction, and what it settles

Two things happened after this benchmark first shipped: the gates it fed
(`residual.min_enhanced_energy_fraction`/`max_energy_fraction`/`min_energy_fraction`) were removed
from the config outright (the owner decided PREPROCESS should measure, not judge whether a residual
means "background" — see `config-derivations.md`'s `residual` section), and a 388-recording cluster
run of the same PREPROCESS block found the *opposite* of what this benchmark measured on breath/
long-cough recordings: 61 of 70 non-speech recordings had FRCRN's residual retain ~0.0000 of the
input's energy (hitting `min_energy_fraction`'s absent-below-minimum gate, back when that gate still
existed), where this benchmark measured 57–94% retained on exactly that recording shape. This section
records what settled that contradiction.

### The residual computation is now one implementation

Lag search, gain fit, subtraction, band split and correlation were extracted into
`senselab.audio.tasks.speech_enhancement.residual` (`find_lag`, `align`, `fit_gain`,
`band_energy_fractions`, `correlation`, `compute_residual`); PREPROCESS's `_residual` block and this
benchmark's own script (`scripts/residual_without_speech.py`) both call it now, and
`~/Downloads/buzz_separation_20260906/residual/subtract.py` — the owner's separate working
tool — was ported to call the same shared functions for the parts that overlap (lag/align/gain/bands/
correlation), keeping its own broader scope (multi-stream summing, the spectral-subtraction
alternative, its CLI) that the pipeline's residual block does not need. `subtract.py` is not
superseded; it still does things `compute_residual` does not.

**Rerunning `residual_without_speech.py` against the shared library reproduces every number in the
table above exactly** — gain (dB), correlation, residual energy fraction and its dB figure, for all
ten recordings, to the same rounding shown. The refactor changed no behaviour; this benchmark's
original numbers stand as measured.

### Diagnosing the contradiction: raw input vs. the pipeline's `plain` stream — ruled out

The leading candidate going in was that this benchmark feeds FRCRN the **raw** recording, while
PREPROCESS feeds it the **`plain`** stream — mono-downmixed, run through `resample_audios` (a
Butterworth low-pass at `target_hz/2 - 100` Hz plus a resample, applied even when the rate already
matches), then peak-scaled if needed. A low-pass filter removing high-frequency turbulent breath
energy before FRCRN ever sees it is a plausible reason the model's behaviour could differ.

Tested directly: for all eight non-speech recordings, both the raw file and a locally-reconstructed
`plain` stream (the exact construction `preprocess.py` performs) were run through FRCRN in one batched
call and each fed to `compute_residual` independently.

| key | feed | gain (dB) | enhanced energy frac | residual energy frac | corr(input, enhanced) | corr(input, residual) |
| --- | --- | --- | --- | --- | --- | --- |
| s1_cough1 | raw | 0.20 | 0.9412 | 0.0151 | 0.992 | 0.123 |
| s1_cough1 | plain | 0.20 | 0.9467 | 0.0098 | 0.995 | 0.099 |
| s1_cough2 | raw | 0.10 | 0.9677 | 0.0103 | 0.995 | 0.102 |
| s1_cough2 | plain | 0.09 | 0.9728 | 0.0057 | 0.997 | 0.075 |
| s1_fivebreaths1 | raw | 15.45 | 0.0018 | 0.9364 | 0.252 | 0.968 |
| s1_fivebreaths1 | plain | 16.91 | 0.0015 | 0.9255 | 0.273 | 0.962 |
| s2_cough1 | raw | 18.27 | 0.0019 | 0.8717 | 0.358 | 0.934 |
| s2_cough1 | plain | 18.16 | 0.0019 | 0.8739 | 0.355 | 0.935 |
| s2_breath1 | raw | 36.04 | 0.0001 | 0.6575 | 0.586 | 0.811 |
| s2_breath1 | plain | 36.17 | 0.0001 | 0.6550 | 0.588 | 0.810 |
| s2_threequickbreaths1 | raw | 8.41 | 0.0111 | 0.9227 | 0.280 | 0.962 |
| s2_threequickbreaths1 | plain | 8.34 | 0.0114 | 0.9223 | 0.280 | 0.961 |
| s3_hardcough | raw | 0.15 | 0.9654 | 0.0015 | 0.999 | 0.039 |
| s3_hardcough | plain | 0.16 | 0.9611 | 0.0018 | 0.999 | 0.043 |
| s3_breath | raw | 8.05 | 0.0671 | 0.5724 | 0.654 | 0.757 |
| s3_breath | plain | 8.07 | 0.0635 | 0.5928 | 0.638 | 0.770 |

**Raw and `plain` agree to within noise on every recording, every quantity.** The pass-through cases
(`s1_cough1`, `s1_cough2`, `s3_hardcough`) stay pass-through under both feeds; the null cases
(`s1_fivebreaths1`, `s2_cough1`, `s2_breath1`, `s2_threequickbreaths1`, `s3_breath`) stay null under
both, with `enhanced_energy_fraction` still under 2% and often under 0.2%. The low-pass filter
`resample_audios` applies is not the mechanism: it changes nothing about which side of FRCRN's
null/pass-through split a recording lands on. **This rules out "what is fed to FRCRN" as the cause of
the local/cluster contradiction.**

Script: `raw_vs_plain_diagnosis.py` (run ad hoc, not checked in — the comparison above is the
artifact that matters). Audio written to `~/Downloads/frcrn_raw_vs_plain_20260907/`: for each of the
16 (recording × feed) combinations, `<key>__<feed>__input.wav`, `__enhanced.wav`, `__residual.wav`
(16 kHz PCM_16, no clipping), plus `raw_vs_plain_report.json` carrying every number in the table
above.

### Cross-checking against the cluster's own files

`~/Downloads/frcrn_cluster_residuals_20260907/` holds `plain.wav`/`residual.wav` for the 98 (of 388)
cluster recordings whose residual passed both gates and was written — of the three subjects measured
locally, only `sub-17578482`'s `Respiration-and-cough-Cough-1`/`-2` runs are among them; every
breath/long-cough run for these three subjects on the cluster hit the `min_energy_fraction` gate and
so left no `residual.wav` (or `enhanced.wav` — not written at all until this session's Job 2) to
inspect directly.

Computing `energy(residual)/energy(plain)` straight from the cluster's own two files, no rerun
needed: `Cough-1` 0.98%, `Cough-2` 0.57% — matching this benchmark's local numbers for the same two
recordings (1.51%, 1.03%) to within the same noise band the speech comparators showed. **The cluster
and this machine agree closely on every recording where FRCRN passes the input through.** The
disagreement is confined to the recordings where FRCRN nulls the input locally — exactly the ones the
cluster has no retained artifact for, because the old gate discarded them before Job 2's fix.

### What this leaves as the explanation, and what it does not settle

Raw-vs-`plain` conditioning is ruled out directly, by measurement, above. What is left, by
elimination rather than by direct comparison (no GPU was available in this session to test on): FRCRN
runs on CPU in every local measurement in this file (`device=worker's choice` resolves to the
`clearvoice-cpu` venv here); the cluster run most likely ran on a GPU node (ORCD's own senselab
recipe targets GPU partitions). A speech-enhancement model given a recording outside its training
distribution — turbulent, broadband, non-speech breath/cough content the model was never asked to
separate anything from — is exactly the regime where CPU/GPU floating-point non-associativity,
cuDNN's algorithm selection, or a differing torch/ClearerVoice build can flip a qualitative outcome
that sits near a decision boundary, while leaving the *in-distribution* cases (clear speech, isolated
percussive coughs against silence) robust across environments — which is precisely the pattern
measured: cough and speech recordings agree closely between this machine and the cluster; only the
duration-filled breath recordings disagree, and only in which direction FRCRN's near-degenerate
output falls.

**This is the best-supported remaining account, not a confirmed mechanism.** Nothing in this session
directly ran the pipeline's FRCRN call on a GPU to reproduce the cluster's own pass-through numbers;
that comparison is the next step if the contradiction needs closing further, and it is a cheap one —
`raw_vs_plain_diagnosis.py`'s method, rerun once with a CUDA-visible `enhance_audios` call over the
same eight recordings, is sufficient.

### Does the 0.50 gate's derivation survive?

**No, independent of the gate now being removed from config.** `min_enhanced_energy_fraction: 0.50`
was fitted to an 87-point gap this benchmark measured between "FRCRN passes non-speech through"
(94–98% enhanced-energy retained) and "FRCRN nulls it" (0.01–6.7%) — a gap this session's local
measurements (both the original benchmark and this update's raw/plain rerun) continue to reproduce
exactly. But the cluster measured the opposite split on the same recording family (residual retaining
~0.0000, i.e. FRCRN keeping the input, not nulling it) for the majority of its non-speech recordings.
A threshold fitted to a gap that does not reproduce across the two environments the pipeline actually
runs in is not a derived value any more, whether or not a gate still reads it. The gates are gone from
config; if they are ever reintroduced, this measurement alone cannot be their derivation without also
explaining why the cluster's non-speech recordings land on the other side of the gap.
