# PREPROCESS — the derivatives every consumer shares

Decided 2026-08-20. This file governs.

## What it is for

Several nodes need the same views of the signal, and computing them independently means paying for
them repeatedly and — worse — computing them slightly differently in each place. PREPROCESS computes
them once, from the audio ADMIT passed, and shares them.

**It decides nothing.** No thresholds, no verdicts, no `flag`. Every output is a measurement, and every
threshold applied to one belongs to the consumer that applies it. That is what keeps this node from
becoming the place where decisions hide.

## Signature

```
preprocess(audio) -> derivatives
```

There is no `fail` and no `flag`. A derivative that cannot be computed — a model unavailable, a
dependency absent — is simply **not emitted**, and a consumer with no product does not run. One
missing derivative must not take the whole node down, because most consumers need only some of them.

## The derivatives, each with the consumer that needs it

| derivative | what it is | consumed by |
| --- | --- | --- |
| `energy_envelope` | broadband envelope, ~3 ms smoothing | airway onsets; the residual's RMS floor; voice branch energy level and modulation rate |
| `level` | peak dBFS, RMS dBFS, LUFS | voice branch — loud phonation is energy *relative to the rest of the recording*, so it needs a file-level reference; also clipping |
| `squim` | STOI, PESQ, SI-SDR from the objective head; MOS from the subjective head | speech branch quality gate |
| `spectrogram` | 20 ms window, 10 ms hop | span refinement; anything rendering the signal for a reader or a model |
| `gammatone` | auditory filterbank output | short-transient detection, where a cochleagram resolves what a linear-frequency window smears |

**Every row has a named consumer, and that is the admission rule.** A derivative with no consumer is a
guess about a node that does not exist yet, and this project has already paid for that once: ADMIT
emitted a level track, a band floor and a clip track that nothing read, and they were removed. When a
branch needs something new here, it declares the input and this node grows the output — in that order.

## Two things about the parameters worth stating

**A 20 ms window is narrowband by the convention the branch documents use.** Wideband means a short
analysis window, roughly 3-5 ms, which resolves glottal pulses as vertical striations and formants as
broad bands; narrowband means roughly 20-30 ms, which resolves individual harmonics as horizontal lines
and smears transients. At 20 ms the harmonics separate by about 40 Hz, so F0 is readable as their
spacing — and the striations are not visible. That is a reasonable default for spectral content, and it
is worth naming so nothing downstream claims to have read voicing pulses off it.

**The envelope and the spectrogram are complementary, not redundant.** A 10 ms hop gives 10 ms of time
resolution, which is coarser than the ±5 ms an envelope achieves on a cough onset. So time precision
comes from `energy_envelope` and frequency precision from `spectrogram`, and a consumer wanting both
takes both. Neither is a substitute for the other, which is why both are here.

## What exists and what does not

`squim` has both heads available already. The envelope and a 20/10 spectrogram are ordinary DSP.
**`level`'s LUFS and `gammatone` are new** — neither exists yet, LUFS needs a loudness meter and
gammatone needs a filterbank, and both are dependencies to add rather than code to write.

## Extensibility

The node is expected to grow. The rule for growing it is the rule above: a new derivative arrives with
the consumer that reads it, and its parameters are named values with a written derivation rather than
defaults nobody chose. A derivative whose window length or smoothing constant was picked to make one
recording look right is the failure this project has met repeatedly, most recently in a spectrogram
whose analysis window determined the conclusion a model reached from it.
