# PREPROCESS parameters

## Working rate — 16 kHz

Every downstream model is 16 kHz native (YAMNet, HeAR, AST, CrisperWhisper, SQUIM), so the choice is one
named resample here or N inside N backends. Cost per labelled event, share of energy an 8 kHz Nyquist
discards:

| event | above 8 kHz | above 4 kHz | 95% of energy below |
| --- | --- | --- | --- |
| mouth non-speech sound | 1.610% | 13.918% | 7272 Hz |
| cough 1 | 3.661% | 16.124% | 6981 Hz |
| cough 2 | 0.741% | 3.189% | 2438 Hz |
| exhalation 1 | 0.225% | 4.754% | 3801 Hz |
| exhalation 2 | 0.088% | 4.374% | 3615 Hz |
| speech | 0.178% | 1.853% | **632 Hz** |

≥ 96.3% of every event's energy survives. The thin margin is not where one would guess: the two sharpest
events put 14–16% of their energy in 4–8 kHz with 95% points just under the ceiling, while speech is the
most band-limited thing in the file. **The airway branch sets the rate, not the speech branch.**

Resampling can overshoot full scale; on this recording it did not (0.9648 → 0.9593), so the guard is
worth having and not worth asserting as inevitable.

## Pre-emphasis — a = 0.97, switchable

A +6 dB/octave tilt, conventional in speech analysis rather than fitted here. Event-to-floor contrast in
the envelope:

| event | plain | pre-emphasised | gain |
| --- | --- | --- | --- |
| cough 1 | 45.84 dB | 56.79 dB | **+10.95** |
| exhalation 1 | 30.53 dB | 38.60 dB | +8.08 |
| mouth non-speech sound | 12.65 dB | 20.01 dB | **+7.36** |
| cough 2 | 51.93 dB | 55.66 dB | +3.73 |
| speech | 31.41 dB | 34.68 dB | +3.27 |
| exhalation 2 | 29.94 dB | 31.71 dB | +1.77 |

Every event gains; the largest gains land on the two hardest, both of which carry 14–16% of their energy
in 4–8 kHz.

**It does not consume its input.** `squim`, `level`, both ASRs and `alignment` read the plain signal.
For `squim` that is measured (see [`squim.md`](squim.md)). For `level`, pre-emphasis is not gain-neutral
— peak 0.9593 → 0.4199, RMS −6.2 dB — so peak/RMS would describe the filtered signal, clipping detection
would miss a clipped input, and LUFS (−30.75 → −31.83) is defined by a K-weighting a differently
pre-filtered signal does not have. **For the ASRs the argument is by analogy from SQUIM and is not
measured**; the missing measurement is word error rate and token-edge displacement, plain against
pre-emphasised.

## Envelope — Hilbert, 40 Hz, zero-phase

`|x + jH{x}|`, 4th-order Butterworth lowpass, `filtfilt`, dBFS.

**Zero-phase beats causal**, measured against the six labelled events: median 63.5 ms against 90.1 ms,
worst 137.9 against 147.4. This makes the envelope offline-only by construction.

**40 Hz is not an onset-precision choice.** Sweeping the cutoff:

| cutoff | rise time | median onset error | worst |
| --- | --- | --- | --- |
| 10 Hz | 100 ms | 75.4 ms | 130.5 ms |
| 20 Hz | 50 ms | 52.2 ms | 130.4 ms |
| **40 Hz** | 25 ms | 63.5 ms | 137.9 ms |
| 80 Hz | 12.5 ms | 129.1 ms | 142.9 ms |
| 320 Hz | 3.1 ms | 144.0 ms | 147.9 ms |

A wider band makes onsets **worse**, and every error is early: a wider-band envelope tracks pre-event
fluctuation, so a fixed floor-referenced rule fires on it sooner. The error belongs to the detection
rule, not the bandwidth. An earlier draft claimed the envelope delivers ±5 ms onsets; nothing supports
that. 40 Hz is the right modulation bandwidth for what the envelope is for.

## Two spectrograms

At a measured F0 of 88.1 Hz the glottal period is 11.4 ms:

| window | frequency resolution, Hann | harmonics resolve in frequency | pulses resolve in time |
| --- | --- | --- | --- |
| 5 ms | 300 Hz | no | yes, 0.44 of a period |
| 10 ms | 150 Hz | no | marginally, 0.88 of a period |
| 20 ms | 75 Hz | **yes** | no |

10 ms weakens both routes at once, which is why the node emits 5 ms and 20 ms rather than a compromise.

**Harmonic structure lives on both axes.** Measured on the same 400 ms: waveform autocorrelation gives
F0 = 87.75 Hz, and autocorrelation along the **time** axis of the 5 ms-window spectrogram gives
86.96 Hz. The wideband view is not F0-blind; each gives F0 by an independent route and their agreement is
a check.

**A rendering has two independent adequacy conditions** — the analysis window and the pixel density of
the span — and only the first appears in the parameters. A model was handed one wideband view and
concluded there was no sustained vowel on four seconds of 88 Hz voicing; F0 was present as striation
spacing, but at ten seconds across the figure an 11.4 ms period spans about two pixels. At 14 s across a
page the 5 ms and 20 ms views are visually indistinguishable, so an overview figure cannot verify
anything that depends on the analysis window.

## Spectral continuity — what it detects, and which spectrogram feeds it

`spectral_continuity` is cosine similarity between consecutive `log1p`-magnitude spectra. `_spans`
feeds it the **narrowband** magnitude and cuts the result at `spans.continuity_cut_percentile` by
rank. It previously fed the wideband magnitude and gated with an absolute `spans.continuity_margin`
of 0.03 above a percentile floor; [Which representation feeds continuity](#which-representation-feeds-continuity-and-why-the-gate-is-a-rank-cut)
below is the measurement that replaced both, and everything between here and there is the record of
how the earlier framing failed.

**What it detects.** On a glide, the voiced region reads 0.95–0.99 against 0.86–0.92 for the silence
flanking it — clean and well separated. On the breath tested it does **not** discriminate the breath
from the background around it: both read 0.85–0.90, with no separation tracking the breath's own
timing. Turbulent broadband noise does not carry the frame-to-frame bin-level coherence a harmonic
sound does. So this is a detector for sustained tonal/harmonic production — glides, phonation,
vowels — and not, on the evidence so far, for breath noise.

**Wideband against narrowband**, smoothing held at the production default
(`ButterworthSmoothing(cutoff_hz=40.0, order=4)`), three files:

| file | variant | IQR | floor | spans | standalone C | corrob C | step | step/IQR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prolonged-vowel | wideband | 0.0752 | 0.7396 | 12 | 0 | 12 | 0.0335 | 0.446 |
| Prolonged-vowel | narrowband | 0.0426 | 0.8610 | 12 | 0 | 9 | 0.0152 | 0.356 |
| Max-phonation-1 | wideband | 0.0814 | 0.7713 | 8 | 3 | 5 | 0.0275 | 0.338 |
| Max-phonation-1 | narrowband | 0.0294 | 0.8798 | 5 | 0 | 4 | 0.0231 | 0.787 |
| Respiration-Breath-1 | wideband | 0.0747 | 0.7104 | 30 | 0 | 30 | 0.0390 | 0.521 |
| Respiration-Breath-1 | narrowband | 0.0241 | 0.8533 | 39 | 9 | 28 | 0.0111 | 0.460 |

**Transition sharpness**, defined here as the median absolute step in the continuity trace across
onsets and offsets of primary-amplitude (envelope-owned) spans — which are computed upstream of
continuity and so are identical across both variants, making the comparison independent of the
measure being compared. Under wideband, `step/IQR` is **0.34–0.52: the trace's routine jitter is two
to three times larger than its response at a genuine acoustic transition.** This ratio had not been
quantified before, and it bears directly on whether tuning the smoother can help at all — every
smoothing candidate tried so far was working against a signal-to-jitter ratio below 1.

The mechanism is the one [Two spectrograms](#two-spectrograms) already states from the other
direction: at 88.1 Hz the glottal period is 11.4 ms, and a 5 ms window resolves *pulses in time*
(0.44 of a period) rather than harmonics in frequency. Consecutive wideband frames therefore land at
different phases of the glottal cycle, and the cosine similarity between them swings for that reason
alone, during steady phonation.

**Narrowband is not a drop-in replacement.** IQR falls consistently (−43%, −64%, −68%), so the
mechanism above is confirmed. But detection moves in *opposite directions* across files:
Max-phonation-1 loses all 3 standalone continuity spans while Respiration-Breath-1 gains 9, and
corroboration falls on all three. Narrowband also shrinks the step nearly as much as it shrinks the
noise (0.0335 → 0.0152, 0.0390 → 0.0111), so `step/IQR` improves on only one file of three.

**The comparison is confounded and settles nothing on its own.** `spans.continuity_margin` is an
absolute 0.03 on a [0, 1] scale, fitted against the wideband trace. Narrowband compresses the trace
into the top ~0.15 of the range, so the same literal gate is a very different sensitivity — 0.4 IQR
above floor under wideband against roughly 1.0 IQR under narrowband. The detection swings above may
measure a mistuned threshold rather than narrowband's real capability. Refitting
`continuity_margin` and `continuity_floor_margin` to the narrowband scale is required before that is
known; **that refit was not run.** It never was: the reframe below removed the absolute gate
entirely, so there was nothing left to refit.

**IQR was the wrong metric, and it is why the rounds above reached no verdict.** The IQR of the whole
trace conflates within-state jitter, which is noise, with between-state separation, which is signal.
A measure that perfectly separated stationary from transient regions would be strongly bimodal and
therefore have a *large* IQR. Minimising it rewards a flat, uninformative trace. That is why heavy
Butterworth smoothing "won" on IQR (0.024–0.037) while destroying detection — it was flattening the
trace, which is exactly what the metric asked for. Every number above that uses IQR as an objective,
including `step/IQR`, should be read as descriptive only. The sections below do not use it.

## Which representation feeds continuity, and why the gate is a rank cut

**The reframe.** Continuity is a spectral-flux/novelty function, so it is read for its **dips**, not
its plateaus. A high value only says nothing changed, which is equally true of steady silence and
steady phonation — no gate on the plateaus can separate those, which is what the rounds above kept
discovering without naming. Change points are therefore the lowest `spans.continuity_cut_percentile`
percent of trace samples **by rank**, and the runs between them are the spans.

**The expectation this is scored against**, owner-stated: continuity segments should run **long** over
sustained events (breathing, phonation) and stay **brief** within running speech. Prolonged-vowel is
the discriminating file because it contains both — spoken "One, two, three" followed by a sustained
vowel. The discriminator is median segment duration outside ASR-word regions divided by the same
inside them; ASR is used only to locate speech, never as boundary truth.

Three representations × two smoothings, cut percentile swept, `continuity_min_duration_ms` 300:

| cut | wideband/none | wideband/butter40 | narrowband/none | **narrowband/butter40** | gammatone/none | gammatone/butter40 |
| --- | --- | --- | --- | --- | --- | --- |
| p2 | 1.0× | 1.1× | 2.1× | **8.8×** | 1.1× | — |
| p5 | 1.0× | 1.3× | 4.0× | **7.6×** | 1.2× | 1.2× |
| p10 | 1.0× | 1.8× | 0.9× | **2.6×** | 1.3× | 1.0× |
| p15 | — | 1.3× | 1.1× | **4.2×** | — | 0.9× |
| p20 | — | 1.0× | 0.9× | **4.2×** | — | 1.1× |

Only narrowband/butter40 separates at every percentile. Everything else sits at ~1.0×, i.e. no
separation at all.

**The sustained-event test**, longest surviving segment at p5:

| | raw → kept | longest |
| --- | --- | --- |
| Prolonged-vowel, wideband/butter40 | 57 → 11 | 1.21 s |
| Prolonged-vowel, gammatone/butter40 | 49 → 14 | 1.17 s |
| **Prolonged-vowel, narrowband/butter40** | **61 → 5** | **6.07 s** |
| Max-phonation, wideband/butter40 | 76 → 17 | 3.22 s |
| Max-phonation, gammatone/butter40 | 73 → 12 | 2.94 s |
| **Max-phonation, narrowband/butter40** | **42 → 3** | **12.48 s** |

The vowel is ~6 s and the phonation ~17 s. Narrowband/butter40 returns each as one segment; every
other combination fragments them into 8–17 pieces.

**Gammatone fails, by the same mechanism as wideband.** Not saturation — its trace spans ~0.85–1.0
with clear sharp dips, a dynamic range comparable to narrowband's ~0.85–0.97. The failure is *where*
the dips fall: scattered throughout sustained phonation rather than concentrated at boundaries. Its
upper ERB channels are wide enough not to resolve individual harmonics, so those channel envelopes
pulse at F0 and the frame-to-frame similarity dips at pulse rate during steady voicing. That is the
same failure as the 5 ms STFT, from [Two spectrograms](#two-spectrograms): **any representation that
resolves the glottal pulse train in time makes continuity dip at F0 during steady phonation.**
Narrowband wins because its 20 ms window resolves the harmonics into stable bands in *frequency*
instead, so consecutive frames look alike right through a sustained vowel. One mechanism, not three
unrelated results.

**Smoothing helps and must be kept** — the opposite of what the reframe first predicted. If the dips
are the signal, blurring them looks obviously wrong; measured, unsmoothed traces give 125–1366 raw
segments of pure jitter and collapse *every* representation to ~1.0× discrimination. Butterworth-40
cuts raw counts 3–5× and is what lets long segments emerge at all. The dips it removes are
glottal-pulse jitter; the ones it keeps are real transitions. The prior smoothing work was
directionally right the whole time — only its IQR objective was wrong.

**Rank, not percentile value — a defect found before the run.** Thresholding against
`np.percentile`'s *value* breaks on a flat trace: the cut lands on the plateau, `>` admits no samples
and returns **zero** segments on a perfectly continuous signal, while `>=` fails the mirror case and
swallows a silent plateau. Ranking guarantees exactly that fraction of samples become change points
whatever the distribution's shape. `segments_between_change_points` cuts by rank; unit-tested on the
flat-trace, zero-cut, empty-input and minimum-duration edge cases.

**The chosen value: `continuity_cut_percentile` 5.0.** p2 reads 8.8× against p5's 7.6× — consistently
but not decisively better — and 5.0 is both the owner's own original proposal and the wider margin
against a file with fewer genuine events. Landed and verified end-to-end on Prolonged-vowel: three
brief segments over the counting (0.392 s, 0.472 s, 0.456 s) then one **6.074 s** segment spanning
3.679–9.753 s, reproducing the harness exactly.

**What these numbers do not license.** Three recordings, not a corpus. The discriminator is
unmeasurable on Respiration-Breath-1, which has **zero** ASR words — no ratio is reported for it
rather than a spurious one, and all three representations behave alike there (24–30 segments, median
0.4–0.85 s at p2–p5), so it neither supports nor contradicts the verdict. On Max-phonation the
sustained phonation is itself ASR-transcribed, so its "inside speech" partition is contaminated and
its ratio is not meaningful; the longest-segment column is what that file contributes. And a
percentile gate **cannot express "nothing here"** — it always marks exactly that fraction of frames as
change points, however continuous the recording. Dedup and corroboration counts under the new gate
were not measured.

## Clip spans — the repeat criterion, and what merging does

`clipping.min_duration_ms` and `clipping.merge_gap_ms` were `null` from the commit that introduced
them. Because `config.require` raises on a null by design, the whole `_clip_spans` block recorded
`absent` on every run for that entire period, `state["clip_span_extents"]` was never assigned, and
every span's `contains_clip` was `False` — not a measured "no clipping here" but a detector that
never executed.

`min_duration_ms` has since been **retired entirely**. It existed to suppress noise from a defect in
`detect_clip_events`: the docstring promised no event where the file's extreme never repeats, but
the code opened one at *any* sample equal to the global max or min, so a lone sample at a merely
relative peak always produced a short event. The repeat requirement now lives in the detector, and
the caller-side duration floor is gone.

### What opens an event, decided per extreme

**Below digital full scale** — the extreme must repeat across at least `MINIMUM_EXTREME_RUN` (2)
*consecutive* samples. Measured at 16 kHz over 3 s signals, the longest consecutive run at each
signal's own exact extreme:

| signal class | grid | longest consecutive run |
| --- | --- | --- |
| clean sine | 100, 120, 150, 200, 250, 300, 440, 500, 700, 1000, 1500, 2000 Hz x amplitude 0.15, 0.3, 0.5, 0.9, float32 and int16 | **1** (all 96 cells) |
| clean sine, non-integer period | 137.3, 211.7, 443.9 Hz | **1** |
| white noise | float32 and int16 | **1** |
| hard-limited sine at 0.6 | 2, 50, 100, 120, 200, 300, 440, 1000 Hz | **≥ 3** |
| hard-limited sine at 0.9 | 2, 50, 100, 120, 200, 300, 440 Hz | **≥ 3** |

Clean and clipped separate cleanly with no overlap, so 2 is the smallest threshold that works and it
is measured rather than conventional. The one non-separating cell is a 2 kHz sine limited at 0.9,
whose run is 1: at 8 samples per period the flat top spans under one sample interval, so no threshold
can resolve it. That is a sampling limit, not a tuning failure.

The repeat must be **consecutive**, not a count. A clean 200 Hz float32 tone hits its exact maximum
**600 times** in 3 s — the sampling grid revisits the same phase every period, giving bit-identical
values — while its longest consecutive run stays 1. A whole-file repeat count separates nothing.

**At digital full scale** — a single sample opens an event, no repeat required: reaching the
representable ceiling is itself evidence of saturation, and a single-sample clip is undetectable any
other way. The predicate is `|extreme| >= 1.0 - 1/32768`, one int16 step, **not** an equality test
against `1.0`. Measured through `Audio`: an int16 file's positive full scale decodes to
`0.9999695` (`32767/32768`) while its negative full scale is exactly `-1.0`, so an equality test
would catch negative saturation and silently miss positive. Float-sourced files give exactly ±1.0.
Sub-full-scale values (0.999, 0.99, 0.9) correctly fail the predicate.

### What the change fixes

Per-period clipping of voiced speech was previously invisible — plateaus around 1.5 ms against a
50 ms floor — and is now detected:

| signal | before (50 ms floor) | after (repeat criterion) |
| --- | --- | --- |
| clean half-scale 200 Hz sine | 0 spans | 0 spans |
| clean 0.3-scale 440 Hz sine | 0 spans | 0 spans |
| clean noise scaled below full scale | 0 spans | 0 spans |
| clean 0.9-amplitude 120 Hz sine | 0 spans | 0 spans |
| lone sub-full-scale 0.77 sample | 0 spans | 0 spans |
| lone sample at ±1.0 or 32767/32768 | 0 spans | **1 event** |
| 2 Hz sine hard-limited at 0.6 | 12 events, 1.777 s | 12 events, 1.778 s |
| 120 Hz hard-limited at 0.6 | **0 spans** | **720 events** |
| 200 Hz hard-limited at 0.6 | **0 spans** | **1200 events** |

A synthetic float signal whose samples exceed ±1.0 does open events. That is correct: real audio
cannot exceed the ceiling, and a float signal that does is out of range.

### `merge_gap_ms` 30 — coalescing, not suppression

Those 720 and 1200 events are one plateau per glottal cycle. Merging within 30 ms collapses each to
a single span covering the clipped region, which is the correct description of one clipped vowel;
without it a single clipped vowel would render as hundreds of adjacent spans:

| signal | raw events | merged at 30 ms | covered |
| --- | --- | --- | --- |
| 120 Hz hard-limited at 0.6 | 720 | **1 span** | 2.999 s |
| 200 Hz hard-limited at 0.6 | 1200 | **1 span** | 2.999 s |
| 2 Hz hard-limited at 0.6 | 12 | **12 spans** | 1.778 s |
| clean half-scale 200 Hz sine | 0 | 0 | 0.000 s |

30 ms comfortably exceeds one glottal period across the plausible F0 range while staying well below
the separation of distinct clipping episodes — the 2 Hz row, whose plateaus sit ~148 ms apart, stays
as 12 separate spans rather than being bridged into one. The value is conventional, not
corpus-fitted, and is still owed a corpus.

**What this licenses:** `contains_clip` reports both sustained saturation and per-period clipping of
voiced speech. What remains unfitted is `merge_gap_ms` itself, and the tolerance question — how much
clipped audio makes a span unusable — which is a separate gate nothing reads today. The three
campaign recordings (`Prolonged-vowel`, `Maximum-phonation-time-1`, `Respiration-and-cough-Breath-1`)
carry no clip spans and peak at 0.541, 0.256 and 0.153 — genuinely unclipped, so they exercise only
the negative case.

## YAMNet `Silence` as the floor source

An earlier design took the floor as a percentile of the whole file, which asserts its own answer: the
10th percentile assumes a tenth of the recording is silence, and on a 90%-speech file it returns quiet
speech with nothing in the number to reveal it.

`Silence` is bimodal — across 29 windows every score is ≤ 0.36 or ≥ 0.62, top-1 in 11 of them — so a 0.5
threshold sits in an empty gap. The statistic over those windows must be robust: inside silence the
envelope reaches **−225 dB**, exact zero samples, so a min or mean is meaningless.

On this file the two floors are 3.26 dB apart (−56.79 vs −53.53), which validates the percentile here and
licenses nothing further. The floor decides **existence**, not only extent: those 3.26 dB take the span
set from 7 to 5, dropping the mouth click, while improving cough 1's offset from 9.28 s to 8.51 s against
a labelled 8.494 s.

The current rule uses a **rolling local** floor rather than a file-global one — see
[`spans.md`](spans.md).
