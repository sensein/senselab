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
