# Disruptions: what stream they are measured on, and what a discontinuity is

Two defects, one measurement each.

## Clipping must be measured on the original recording

`detect_disruptions` ran on the `plain` stream — peak-normalised and resampled to 16 kHz. Across the
b2ai-28 campaign `clipped_runs` read **0 on every file**, including **four whose originals peak at
exactly 0.0 dBFS**. Peak normalisation rescales the plateau off the headroom and resampling
interpolates through what is left of it, so the flat runs clipping consists of do not survive into
the copy being measured. A resampled copy is the wrong instrument for a defect of the recording.

Every disruption reading — clipping, dropouts, discontinuities, DC offset, zero-crossing rate — is
now taken on the `recording` stream, still scoped to each speech span. The span extents are in
seconds, so they carry across the sample-rate change unchanged.

## An absolute jump threshold measures high-frequency energy, not defects

`disruptions.discontinuity_threshold` was **0.5**, an absolute sample-to-sample jump. On one
campaign recording of loud speech it reported **800 "discontinuities"**. Nothing was wrong with the
recording: a band-limited signal's neighbouring samples differ by more the louder and the
higher-frequency it is, so the rule is a loudness-and-bandwidth meter with a threshold on it.

The clean demonstration is a full-scale 3 kHz tone at 16 kHz. Its neighbouring samples differ by
1.18 at every zero crossing, and the absolute rule calls **11999 of one second's samples**
discontinuous. The tone is continuous everywhere.

### The local reference

A jump `x[i] -> x[i+1]` counts when

    |x[i+1] - x[i]|  >  factor * max(sd(window before i+1), sd(window after i+1))

Both windows are `discontinuity_window_ms` long and neither contains the jump, so the reference
cannot be inflated by the thing it is measuring. It is a standard deviation rather than an RMS so
that a constant offset — reported separately as `dc_offset` — is not read as variation; this is what
keeps a DC step detectable. Taking the larger of the two sides means a word onset, where a quiet
window abuts a loud one, is referenced to the loud side and does not fire.

### Choosing the factor

Ratio `|diff| / local variation` measured over **806142 samples** of peak-normalised clean speech,
three recordings at three sample rates — the plain stream's own regime, since that is where the
absolute rule misfired:

| recording | rate | samples | p99 | p99.9 | max | flagged at x10 | flagged by abs 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `audio_48khz_mono_16bits.wav` | 48 kHz | 236222 | 1.45 | 2.59 | 9.64 | 0 | 45 |
| `english_conversation_higgs_audio_v2.wav` | 24 kHz | 515520 | 2.26 | 3.90 | 8.72 | 0 | 329 |
| `had_that_curiosity.wav` | 16 kHz | 54400 | 2.31 | 3.90 | 6.05 | 0 | 2 |

`discontinuity_local_factor` **10.0** sits above every ratio any of this clean audio produced (largest
9.64) and flags none of the 376 samples the absolute rule flagged. `discontinuity_window_ms` **20.0**
is the conventional short-term analysis frame, the same length `spectrogram.narrowband_window_ms`
already uses; it is not separately fitted.

### What it catches, and what it is not

- A 0.9 step on a 0.1-amplitude tone: **1**. The same 0.9 step on a 0.9-amplitude tone: **0** — the
  criterion is contextual by construction, which is the whole point.
- A single-sample click injected into ordinary speech: **2** — its two edges.
- A dropout's edges: **0**. A zeroed run's edge jump is small against the loud window beside it. This
  is the right answer here because `dropout_runs` is the instrument for a dropout; the discontinuity
  count is not a second one.
- A splice made by excising samples mid-vowel: **0** on the case tried. This detector finds impulsive
  jumps, not edits. Nothing here licenses calling it a splice detector.

## Zero-crossing rate

Added as a plain per-span reading of the original recording, in crossings per second. A 200 Hz tone
reads 400; an 800 Hz tone reads four times that. Reported, never gated: no threshold on it has been
derived and the config carries none.

## Standing limits

Three recordings and one synthetic tone. The factor is fitted to keep *clean* audio clean and has
never been fitted against labelled defects, because no labelled-defect corpus exists here. What it
licenses is "10 does not fire on clean speech"; what it does not license is any claim about the rate
at which real defects are caught.
