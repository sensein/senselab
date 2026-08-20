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
preprocess(audio, preemphasis=True) -> derivatives
```

Internally two conditioning steps run before any derivative is computed:

```
                            +--> plain -------------> squim, level, ASR x2, alignment, silence
audio --> resample to 16 kHz +
                            +--> pre-emphasis ------> envelope --> spans
                                 (switchable)             spectrograms, gammatone
```

There is no `fail` and no `flag`. A derivative that cannot be computed — a model unavailable, a
dependency absent — is simply **not emitted**, and a consumer with no product does not run. One
missing derivative must not take the whole node down, because most consumers need only some of them.

## The derivatives, each with the consumer that needs it

| derivative | what it is | consumed by |
| --- | --- | --- |
| `energy_envelope` | analytic-signal magnitude `\|x + jH{x}\|`, zero-phase 40 Hz lowpass, autoscaled to its own maximum, with a floor from `silence` | airway modulation rate; the residual's RMS floor; voice branch energy level |
| `level` | peak dBFS, RMS dBFS, LUFS — **plain signal, never pre-emphasised** | voice branch — loud phonation is energy *relative to the rest of the recording*, so it needs a file-level reference; also clipping |
| `squim` | STOI, PESQ, SI-SDR from the objective head; MOS from the subjective head — **plain signal, never pre-emphasised** | speech branch quality gate |
| `spectrogram_wb` | 5 ms window, 5 ms hop — wideband | onsets and transients; glottal pulses; anything reading voicing structure |
| `spectrogram_nb` | 20 ms window, 5 ms hop — narrowband | harmonics and F0 read off their spacing; span refinement; rendering for a reader or a model |
| `gammatone` | auditory filterbank output | short-transient detection, where a cochleagram resolves what a linear-frequency window smears |
| `spans` | candidate event spans from the envelope: propose, onset, offset — the rules below | AIRWAY classifies them with HeAR; SPEECH interprets them with SQUIM and YAMNet. **Two consumers, which is why this lives here** |
| `silence` | YAMNet `Silence` per 0.96 s window at its native 0.48 s hop, and the floor derived from it — **plain signal** | the envelope's floor, which every span rule references; airway's `t_not_after_s` |
| `asr_crisperwhisper` | transcript with word and token edges — **plain signal** | speech branch transcript and speaker spans; airway branch's lexical-contamination check; voice branch's lexical exclusion |
| `asr_qwen` | transcript with word timings — **plain signal** | speech branch agreement confidence; a second opinion the speech branch compares against CrisperWhisper |
| `alignment` | forced alignment of the agreed transcript to the audio | word- and phone-level spans for every consumer that needs where a word was, not just that it was said |

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

## Conditioning: resample, then pre-emphasis

### Resample to 16 kHz

One resampler, here, named — the rate's justification is the next section. From 48 kHz it is an
integer decimation by 3, which is the common case and the cheap one. Resampling **can** overshoot full
scale, a trap this repository has already paid for in its write path; on the labelled recording it did
not (peak 0.9648 to 0.9593), so the guard is worth having and is not worth asserting as inevitable.

### Pre-emphasis, switchable

A first-order difference, `y[n] = x[n] - a*x[n-1]`, with `a = 0.97` — a +6 dB/octave tilt that offsets
the glottal source's roll-off and is conventional in speech analysis rather than fitted here. It is a
**switchable component**: on by default, and one flag turns it off for a consumer that wants the
signal as recorded.

It earns its place on the derivative it was least obviously for. Event-to-floor contrast in the
Hilbert envelope, plain against pre-emphasised:

| event | plain | pre-emphasised | gain |
| --- | --- | --- | --- |
| cough 1 | 45.84 dB | 56.79 dB | **+10.95** |
| exhalation 1 | 30.53 dB | 38.60 dB | +8.08 |
| mouth non-speech sound | 12.65 dB | 20.01 dB | **+7.36** |
| cough 2 | 51.93 dB | 55.66 dB | +3.73 |
| speech | 31.41 dB | 34.68 dB | +3.27 |
| exhalation 2 | 29.94 dB | 31.71 dB | +1.77 |

Every event gains, and **the largest gains land on the two hardest events** — the mouth click, which
one agent run missed entirely, and cough 1. That is the expected direction, since both carry 14-16% of
their energy in 4-8 kHz, which is exactly what the tilt boosts.

### Both signals stay available, and each derivative declares which one it reads

Pre-emphasis does not consume its input. The node holds the resampled signal **and** its pre-emphasised
form, and every row of the derivative table names which of the two it reads — so this is a property of
the graph, not a special case bolted onto it. Most derivatives read the pre-emphasised signal.
`squim`, `level`, both ASRs and `alignment` read the plain one, and for `squim` and `level` that is
not a preference.

**SQUIM goes off-distribution, and says so incoherently.** Pre-emphasised, it reports STOI rising
0.8635 to 0.9683 while SI-SDR falls -12.917 to -20.676 dB. One signal cannot be materially more
intelligible *and* far more distorted; the two heads disagree because neither is being asked about a
signal like its training data. A speech-branch quality gate reading STOI would be inflated by 0.10,
which is enough to flip a verdict on the strength of a filter nobody intended as a quality change.

**`level` stops measuring what it names.** Pre-emphasis is not gain-neutral — peak drops 0.9593 to
0.4199 and RMS falls 6.2 dB — so peak dBFS and RMS dBFS would describe the filtered signal while
being read as the recording's level, and clipping detection would miss a clipped input outright. LUFS
is worse than merely shifted (-30.75 to -31.83): it is defined by a standard K-weighting, and a signal
pre-filtered by something else does not have a LUFS.

So the switch governs the derivatives whose value pre-emphasis *changes*, and does not reach the two
whose **definition** it breaks. A consumer asking for a plain-signal derivative gets the same answer
whether the switch is on or off, which is the point: the switch is a knob on the analysis, not on what
the recording's level or its quality scores mean.

## Span detection lives here, because two branches need the same spans

Both AIRWAY and SPEECH begin from "where in this recording is something happening". They began by
deriving that separately, which is the condition this node exists to prevent — two producers of one
product, differing in ways nobody chose. So the envelope's span rules are a PREPROCESS derivative, and a
branch's job is to **interpret** a span, never to find one.

Three rules, each measured against the labelled recording's six events:

| rule | value | why not otherwise |
| --- | --- | --- |
| propose | peaks ≥ `floor + 18 dB`, separated by ≥ 150 ms | at `floor + 12 dB` a low-contrast peak's offset threshold sits near the floor, so its walk is effectively unbounded — two coughs 1.7 s apart merged into one 6 s span |
| onset | walk back from the peak to `peak − 15 dB` | peak-anchored puts 5 of 5 airway onsets inside the labels' declared windows; the same envelope with a floor-referenced threshold manages 2 of 6, every error early |
| offset | walk forward to `peak − 0.7 × (peak − floor)`, closing only after 120 ms continuously below | the events do not share a dynamic range — 20.0 dB for a mouth sound against 56.8 dB for a cough — so one fixed drop is at once too shallow for a cough and unreachable for the click. Median offset error 84.3 ms, against 573.9 ms for a fixed `peak − 10 dB` |

**The anchor, not the threshold, is what makes the onset work**, which is also why this node claims no
onset accuracy for `energy_envelope` itself: widening the envelope's bandwidth makes onsets *worse*
(median 144 ms at 320 Hz against 63 ms at 40 Hz), because a wider band tracks pre-event fluctuation that
a fixed threshold then fires on. The accuracy is a property of these rules.

On the labelled recording they yield **five spans**: 2.32–3.29, 5.32–6.22, 7.92–8.51, 9.61–9.96 and
11.75–13.16 s. Four are airway events and one is speech, and **the spans themselves carry no label** —
this node does not know which is which and must not guess.

**One parameter is per-consumer and cannot be settled here.** The 120 ms hangover must be shorter than
the shortest event a consumer intends to bound; at 250 ms it overshoots a 202 ms mouth click by 418 ms,
because the rule has to observe more silence than the event lasts before it will close. A consumer
needing a different hangover asks for it; the shipped value is not a universal.

**Limits.** One recording, one healthy adult, six events. These justify the *shape* of each rule and not
its constant, and the peak in every scored case was located inside a labelled span — running the same
rules unsupervised over a whole envelope is the harder problem, and it is what the figure exists to
expose.

## The envelope floor comes from YAMNet's `Silence`, not from a percentile

Every span rule in every branch is expressed relative to a floor — propose at `floor + 18 dB`, close the
offset at a fraction of `peak - floor`. So the floor is not a display parameter; it is the reference the
rules are written against, and how it is defined belongs here rather than in each branch.

**A percentile of the envelope assumes its own answer.** Taking the 10th percentile asserts that about a
tenth of the recording is silence. On a recording that is 90% speech the 10th percentile is quiet speech,
and nothing about the resulting number reveals that it is not a noise floor. YAMNet's `Silence` decides
which windows *are* silence, so the floor stays valid whatever the mix — which is the property a
percentile cannot have.

`Silence` is usable as a gate because its distribution on real audio is **bimodal**: across the 29
windows of the labelled recording every score is either ≤ 0.36 or ≥ 0.62, with nothing in between, and
`Silence` is the top-1 label in 11 of them. A 0.5 threshold therefore sits in an empty gap rather than
being fitted to a value.

**The statistic over those windows must be robust, and one number says why.** Inside YAMNet-silence the
envelope reaches **−225 dB** — exact zero samples, which real recordings contain. A minimum or a mean
over that region is meaningless; the floor is the **median** of the envelope across silence-labelled
windows.

**Measured against the percentile it replaces, on a file where both are computable:**

| floor | value |
| --- | --- |
| 10th percentile of the whole file | −56.79 dB |
| median over YAMNet `Silence` (56% of this file) | −53.53 dB |

3.26 dB apart, which is a validation of the percentile *for this recording* and licenses nothing beyond
it — the two agree here because this file genuinely is more than half silence.

**The floor decides which events exist, not only where they end.** Raising it by those 3.26 dB changes
the envelope's span set from 7 to 5: the mouth non-speech sound disappears, because its peak stands 20 dB
above the lower floor and a proposal needs 18 dB above the higher one. The same change *improves* cough
1, whose offset moves from 9.28 s to 8.51 s against a labelled end of 8.494 s. So the floor trades
sensitivity on low-contrast events against tightness on high-contrast ones, and that trade is a decision
this node makes once for every branch rather than one each branch makes silently.

**When YAMNet labels no window as silence, the floor has no producer.** That is the node's existing rule
and not a new exception: the derivative is absent, and a consumer whose rules reference a floor does not
run. A recording with no silence in it is exactly the case where an assumed floor would be most wrong,
so inventing one there is the failure this design is avoiding.

## ASR runs here, not in the speech branch

CrisperWhisper and Qwen3-ASR run in PREPROCESS, with forced alignment over the transcript they agree on.
This is the admission rule applied honestly rather than a convenience: **all three branches read word
timings**, so the alternative is either three copies of the same inference or a cross-branch dependency
on whichever branch happened to own the model.

| branch | what it reads the timings for |
| --- | --- |
| speech | the transcript itself, per-speaker spans, and the agreement between the two ASRs as a confidence |
| airway | word *presence* only — any word inside the interval its spans cover flags the file for review. Not word times: token edges land +20, +32, −26, −10 ms against four verified windows, so a second onset source is available, but it needs a disagreement rule no measurement supports yet |
| voice | lexical exclusion. Voice-no-words is defined as vocalic activity that is *not* speech, so it needs to know where words are in order to subtract them |

**The consequence for the speech branch is that it consumes transcripts instead of producing them.** Its
multi-ASR node becomes a comparison over inputs it is given, not a stage that runs the models. That is a
simplification of that branch and should be applied to it rather than left implied here.

**Both ASRs and the aligner read the plain signal**, alongside `squim` and `level`. The reason is the same
one the pre-emphasis section measures for SQUIM: an acoustic model trained on speech as recorded is
off-distribution on a +6 dB/octave tilt, and a transcript is not a quantity that degrades gracefully — a
word is either recognised or it is not. **This one is reasoned by analogy, not measured.** SQUIM's
incoherent shift is direct evidence for SQUIM; the corresponding measurement here is word error rate and
token-edge displacement for both ASRs on plain against pre-emphasised input, and it has not been run. It
is cheap, and it is the thing to run before anything rests on this choice.

Two constraints this places on the node. **ASR is by far the most expensive derivative here**, so the
"missing derivative simply is not emitted" rule matters most for these rows — a branch that needs no
transcript must not wait on two ASR models. And **`alignment` depends on the two ASR rows** rather than
on the audio alone, which makes it the first derivative in this node with a derivative as an input; if
the two transcripts disagree beyond the speech branch's threshold there is no agreed transcript to align,
and the row is absent rather than guessed.

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

One consequence worth writing down: a narrowband input — a telephone or 8 kHz-sampled recording, with
a 4 kHz ceiling — would cut real airway content, not just headroom, so it is a genuine restriction on
what the airway branch can conclude rather than a formality.

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
