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
| `energy_envelope` | analytic-signal magnitude `\|x + jH{x}\|`, zero-phase 40 Hz lowpass, autoscaled to its own maximum | airway modulation rate; the residual's RMS floor; voice branch energy level |
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

**The envelope is a Hilbert modulation envelope, and it is not an onset detector.** It is the
magnitude of the analytic signal, `|x + jH{x}|`, lowpassed at 40 Hz and autoscaled by its own maximum
so it is invariant to input gain. Two parameter choices carry measurements.

*Zero-phase, not causal.* A 4th-order Butterworth applied forward-and-backward (`filtfilt`) has no
group delay; the same filter applied once does. Against the six labelled events, zero-phase is better
in both median and worst case — median 63.5 ms against 90.1 ms, worst 137.9 ms against 147.4 ms — so
the filter is zero-phase, which also means this envelope is offline-only by construction.

*40 Hz, and the reason is not onset precision.* Sweeping the cutoff against those same labels:

| cutoff | envelope rise time | median onset error | worst |
| --- | --- | --- | --- |
| 10 Hz | 100 ms | 75.4 ms | 130.5 ms |
| 20 Hz | 50 ms | 52.2 ms | 130.4 ms |
| **40 Hz** | **25 ms** | **63.5 ms** | 137.9 ms |
| 80 Hz | 12.5 ms | 129.1 ms | 142.9 ms |
| 320 Hz | 3.1 ms | 144.0 ms | 147.9 ms |

**A wider band makes onsets worse, not better** — 144 ms at 320 Hz against 63 ms at 40 Hz — and every
error is early. A wider-band envelope tracks pre-event fluctuation more faithfully, so a fixed
floor-plus-6 dB rule fires on it sooner. The onset error is therefore dominated by **the detection
rule**, not by the envelope's bandwidth, and no cutoff in that table buys the few-millisecond
precision an airway onset wants.

So this retracts a claim an earlier draft of this file made: that onset precision comes from the
envelope, at about ±5 ms. Nothing measured supports it. **This node emits the envelope and claims
nothing about onset accuracy**; a consumer that needs a span owns its detection rule and owns the
error that rule produces — which is the same admission rule as the table above, applied to precision
rather than to existence. 40 Hz is the right modulation bandwidth for the thing the envelope is
actually for: how amplitude varies over a syllable or a cough, not where the cough begins.

## Working sample rate: 16 kHz

Every model downstream of this node is 16 kHz native — YAMNet, HeAR, AST, CrisperWhisper and SQUIM all
resample internally — so the choice is not whether to resample but whether to do it **once, here, with
one named resampler**, or N times inside N backends with whatever each one ships. It happens once here.

The cost is measurable rather than assumed. On the labelled recording, per event, the share of energy
that an 8 kHz Nyquist discards:

| event | above 8 kHz | above 4 kHz | 95% of energy below |
| --- | --- | --- | --- |
| mouth non-speech sound | 1.610% | 13.918% | 7272 Hz |
| cough 1 | 3.661% | 16.124% | 6981 Hz |
| cough 2 | 0.741% | 3.189% | 2438 Hz |
| exhalation 1 | 0.225% | 4.754% | 3801 Hz |
| exhalation 2 | 0.088% | 4.374% | 3615 Hz |
| speech | 0.178% | 1.853% | 632 Hz |
| whole file | 1.448% | 6.576% | — |

**16 kHz keeps at least 96.3% of every labelled event's energy**, so it is adequate. But the table also
says where the margin is thin, and it is not where one would guess: the two sharpest events put 14-16%
of their energy in 4-8 kHz and their 95% points at 6981 and 7272 Hz — just **under** the 8 kHz ceiling.
Speech is the most band-limited thing in the file, at 632 Hz. So the rate is set by the airway branch,
not the speech branch, and the events that most need the top octave are the clicks and coughs.

Two consequences worth writing down. A narrowband input — a telephone or 8 kHz-sampled recording, with
a 4 kHz ceiling — would cut real airway content, not just headroom, so it is a genuine restriction on
what the airway branch can conclude rather than a formality. And resampling is not free of its own
artefact: it overshoots, so a signal at full scale can exceed ±1 afterwards, which is a known trap in
this repository's write path and applies here too.

## What exists and what does not

`squim` has both heads available already. Both spectrograms are ordinary DSP, and the Hilbert envelope
needs only `scipy.signal.hilbert` plus a Butterworth already available.
**`level`'s LUFS and `gammatone` are new** — neither exists yet, LUFS needs a loudness meter and
gammatone needs a filterbank, and both are dependencies to add rather than code to write.

## Extensibility

The node is expected to grow. The rule for growing it is the rule above: a new derivative arrives with
the consumer that reads it, and its parameters are named values with a written derivation rather than
defaults nobody chose. A derivative whose window length or smoothing constant was picked to make one
recording look right is the failure this project has met repeatedly, most recently in a spectrogram
whose analysis window determined the conclusion a model reached from it.
