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
| `spectrogram_wb` | 5 ms window, 5 ms hop — wideband | onsets and transients; glottal pulses; anything reading voicing structure |
| `spectrogram_nb` | 20 ms window, 5 ms hop — narrowband | harmonics and F0 read off their spacing; span refinement; rendering for a reader or a model |
| `gammatone` | auditory filterbank output | short-transient detection, where a cochleagram resolves what a linear-frequency window smears |

**Every row has a named consumer, and that is the admission rule.** A derivative with no consumer is a
guess about a node that does not exist yet, and this project has already paid for that once: ADMIT
emitted a level track, a band floor and a clip track that nothing read, and they were removed. When a
branch needs something new here, it declares the input and this node grows the output — in that order.

## Two things about the parameters worth stating

**Two spectrograms, because one window cannot carry both cues.** The hop and the window are
independent: the hop sets how finely the view is sampled in time, the window sets what is resolvable at
all. So both share a 5 ms hop and differ only in window.

At a measured F0 of 88.1 Hz the glottal period is 11.4 ms, and the arithmetic decides:

| window | frequency resolution, Hann | harmonics resolve in *frequency* | pulses resolve in *time* |
| --- | --- | --- | --- |
| 5 ms | 300 Hz | no | yes, 0.44 of a period |
| 10 ms | 150 Hz | no | marginally, 0.88 of a period |
| 20 ms | 75 Hz | **yes** | no |

**Harmonic structure lives on both axes, and F0 is recoverable from either.** In frequency it is the
spacing of the harmonics, which needs resolution finer than F0. In time it is the spacing of the
glottal pulses, which needs a window shorter than the period. Measured on the same 400 ms of sustained
voicing, the two routes agree:

| route | F0 |
| --- | --- |
| waveform autocorrelation | 87.75 Hz |
| autocorrelation along the **time** axis of the 5 ms-window spectrogram | **86.96 Hz** |

So the wideband view is **not** F0-blind — it carries F0 as pulse spacing, to within 1.3% of the
waveform here. Each view gives F0 by an independent route, and their agreement is a check worth having
rather than a redundancy to remove.

A 10 ms window is still the least useful of the three, but for a narrower reason than "it shows
neither": it cannot separate 88 Hz harmonics in frequency, and at 0.88 of a period its pulses are
sampled with almost no margin, so the temporal route degrades too. Both routes are weakened at once,
which is why the node emits 5 ms and 20 ms rather than a compromise between them.

**And this corrects how an earlier failure here should be read.** A model was handed one wideband view
of a recording, reasoned that it saw no harmonic stacks, and concluded there was no sustained vowel — on
four seconds of 88 Hz voicing. It is tempting to say wideband hides F0. It does not: F0 was present in
that view as striation spacing. What was missing was **pixels** — at ten seconds across the figure a
11.4 ms period spans about two of them, so the information was in the signal and absent from the
image. A rendering has two independent adequacy conditions, the analysis window and the pixel density
of the span, and only the first is visible in the parameters.

**The envelope is still not redundant with either spectrogram.** A 5 ms hop gives 5 ms of time
resolution, which is at the edge of the ±5 ms an envelope achieves on a cough onset, and the envelope's
~3 ms smoothing sees the rise directly rather than through a windowed transform. So onset precision
comes from `energy_envelope`, and the spectrograms carry what frequency structure is present.

## What exists and what does not

`squim` has both heads available already. The envelope and both spectrograms are ordinary DSP.
**`level`'s LUFS and `gammatone` are new** — neither exists yet, LUFS needs a loudness meter and
gammatone needs a filterbank, and both are dependencies to add rather than code to write.

## Extensibility

The node is expected to grow. The rule for growing it is the rule above: a new derivative arrives with
the consumer that reads it, and its parameters are named values with a written derivation rather than
defaults nobody chose. A derivative whose window length or smoothing constant was picked to make one
recording look right is the failure this project has met repeatedly, most recently in a spectrogram
whose analysis window determined the conclusion a model reached from it.
