# Branch 1's propose → confirm → bound chain, scored against the verified labels

2026-08-19. Runs the measurement `specs/20260817-triage-workflow-dag/branch-1-airway.md` names as
the one that would settle its proposer choice:

> So a proposer can be scored on recall over the six and **false positives per minute** over the
> empty stretches, on the original and on degraded copies with added noise and reverberation. That
> answers the robustness question the current draft can only assert.

Standalone. No `VoteStore`, no pass summaries, no `rounds.py`, no L1/L2 artifacts — the three stages
are called directly through `senselab.audio.tasks.classification` and
`senselab.audio.tasks.health_acoustics`, and everything is scored in this directory.

## Verdict, in one table

| claim under test | verdict | the number |
| --- | --- | --- |
| YAMNet as proposer recovers the events | **partly refuted** | 5/6 at every threshold to 0.70; event 1 is never seen — YAMNet calls it `Silence` at 0.72–1.00 |
| the proposer's FP/min is defensible | **not answerable on this file** | the whole verified-empty corpus supports 5 independent window decisions; one FP is worth 7.13 FP/min, and the exact 95% interval on the clean rate is [0.005, 0.716] |
| the proposer is robust to noise and reverberation | **refuted for the `Silence` route, partly upheld for the vocabulary route** | 1−P(`Silence`) fires on 5/5 verified-empty windows at every SNR down from 20 dB; the vocabulary route holds FP at 0/5 but loses recall 5/6 → 3/6 by 0 dB |
| HeAR as confirmer adds precision | **refuted** | 0 of 50 HeAR windows on this file overlap no event, so it has no negative available; it corroborates 4/7 planted verified-empty proposals at 0.90–1.00 |
| HeAR as confirmer costs little recall | **refuted** | at θ=0.5 it rejects the verified speech event (0.339) and the verified mouth sound (0.036), taking the chain from 5/6 to 4/6 |
| DSP envelope bounds a cough onset to ±5 ms | **upheld, and generalised from n=1 to n=2** | worst onset error over both coughs and Δ ∈ [3, 20] dB is 5 ms; at Δ=6 dB both are 0 ms |
| the same bounder gives a usable offset | **refuted** | cough offsets are −449 to +20 ms and move 183/182 ms across the same Δ range; cough 5 is never within 267 ms |
| the chain recovers the six events | **4 of 6, with 4 correct labels** | and its one good number, the cough onsets, comes entirely from the stage that uses neither model |

## Method

### The recording and the labels

`/Users/satra/Downloads/streaming-audio-2026-07-30T04-21-56-487Z.wav`, 14.027 s, 48 kHz mono, the
project's only human-verified audio (verified 2026-08-18). Six events; `ground_truth.py` holds them.
Events 4, 5 and 6 carry verified spans, events 1–3 verified onsets with an approximate duration, and
extents for 1–3 are onset + that duration.

Two corrections from the verification are treated as ground truth, not as candidates: **6.60–7.10 s
contains nothing**, and the 0.893 s event is a mouth non-speech sound rather than a handling click.

`13.79–14.04` is excluded from scoring in both directions — it is the one genuinely unlabelled
stretch, where Brouhaha's VAD rises and community-1 stays at zero and no human verdict exists. Any
detection reaching into it is neither credited nor penalised.

The prompt's table says "two have verified spans" but bolds three windows (4, 5 and 6). The ±5 ms
check is scoped to the two coughs as instructed; event 6's offset is reported alongside as
supplementary, flagged as such.

**Scorable-empty** is the seven as-given empty stretches minus every event extent and minus the
unlabelled tail. Three of the as-given stretches overlap an event extent by 5–95 ms, because those
extents come from approximate durations; subtracting keeps a detection that lands on a real event out
of the false-positive count.

```
 0.000- 0.780   1.095- 2.275   3.500- 5.300   6.300- 7.900
 8.500- 9.600  10.250-11.620  13.200-13.790          total 8.420 s = 0.1403 min
```

### Degradations

Eight degraded copies plus the original; `degrade.py`, seed 20260819, parameters recorded in
`raw/degradations.json`.

| variant | what it is |
| --- | --- |
| `white_snr{20,10,5,0}` | additive Gaussian white noise |
| `pink_snr{10,0}` | additive 1/f noise, shaped in the frequency domain from the same generator |
| `reverb_t60_{0.3,0.7}` | convolution with a synthetic RIR: a unit direct path plus exponentially-decaying Gaussian noise reaching −60 dB at T60, scaled to a direct-to-reverberant ratio of 0 dB, then re-levelled to the dry RMS |

**SNR is defined on full-file RMS**, and on this recording that is dominated by the events — 8.4 s of
its 14 s is verified-empty, so the "signal" in the ratio is essentially the six events. 0 dB is
therefore severe. No variant clipped: peaks ran 0.61–0.99 against full scale and the headroom gain
stayed at 1.000 for all nine, so no model saw distortion instead of content.

Written as 32-bit float WAV so neither backend's internal resample reads quantisation noise. The
files themselves are not committed — 24 MB, and exactly regenerable from the seed and the manifest.

### The three stages as called

- **propose** — `YAMNetClassifier.classify_with_yamnet(top_k=521)`, its own fixed 0.96 s window and
  0.48 s hop, 29 windows over this file. All 521 AudioSet posteriors are stored so the threshold
  sweep never re-runs the model.
- **confirm** — `detect_health_acoustic_events(hop_length=0.25)`, the 2 s window hard-fixed at 32000
  samples, 50 windows, all 8 labels kept.
- **bound** — `bounder.py`: 80 Hz high-pass, 4 ms Hann frames on a **1 ms hop**, 10th-percentile
  envelope level within ±1.5 s of the peak as the floor, edges walking outward from the peak to the
  first frame below floor + Δ. Δ is swept over {3, 6, 10, 12, 15, 20} dB. The floor is local rather
  than global because a bounder handed the file's verified-empty stretches would be reading the
  answer.

Everything is reproducible as:

```bash
UV_PROJECT_ENVIRONMENT=/path/to/senselab/.venv uv run --no-sync python degrade.py
UV_PROJECT_ENVIRONMENT=... PYTHONPATH=<worktree>/src uv run --no-sync python run_backends.py both
UV_PROJECT_ENVIRONMENT=... uv run --no-sync python score.py
```

`raw/report.txt` is the full output of the last of these; `results.json` is the same numbers as JSON.

### Two readings of "YAMNet proposes", both measured

`branch-1-airway.md` names YAMNet's vocabulary — `Silence`, plus `Breathing`, `Cough`, `Gasp`,
`Sigh`, `Throat clearing`, `Sneeze`, `Snoring` — and separately credits its speech detection. All
nine classes exist in the class map. That supports two proposers, and which one is meant decides
whether event 1 is reachable at all, so both are swept:

- **`vocab`** — max posterior over `Breathing`, `Cough`, `Gasp`, `Sigh`, `Throat clearing`, `Sneeze`,
  `Snoring`, `Speech`. The taxonomy-inside route.
- **`nonsilence`** — 1 − P(`Silence`). The "explicit `Silence` class" route, which can in principle
  propose an event the vocabulary has no word for.

## Probe 0 — the denominators, and what each probe could show

State the resolution before the result, because on this file it is the finding.

| quantity | value |
| --- | --- |
| events | 6 → recall quantises at 0.167 |
| scorable-empty | 8.420 s = 0.1403 min → **one false-positive detection is 7.13 FP/min** |
| YAMNet windows | 29 at 0.96 s / 0.48 s |
| windows wholly inside a scorable-empty region | **4** (indices 8, 9, 14, 22) |
| windows with zero overlap on any event | **5** (8, 9, 14, 18, 22) |
| HeAR windows | 50 at 2.0 s / 0.25 s |

So the FP-per-minute figure the design cannot currently defend **cannot be defended from this file
either**, and that is a property of the corpus, not of the models: 8.4 s of verified-empty audio at a
0.48 s hop supports five independent window decisions. A true FP rate below 1/5 is indistinguishable
from zero here. The measurement is reported anyway, with exact Clopper-Pearson intervals so the width
travels with the number, and with a continuous companion measure that does not quantise at one:

- **FP/min (window)** — one fired event-free window counted as one alarm. Does not collapse when
  adjacent detections merge.
- **FP/min (detection)** — merged detections with zero event overlap. Reported and then set aside; see
  the finding below.
- **duty_all** — the fraction of scorable-empty *seconds* covered by any detection. Continuous in
  time, and the honest picture of a 0.96 s proposer on a file whose gaps are 0.6–1.8 s.

## Probe 1 — the proposer, swept

Full sweep in `raw/report.txt`; the shape of both curves, thinned:

| τ | `vocab` recall | events | FPwin/5 | FP/min | duty_all | | `nonsilence` recall | events | FPwin/5 | FP/min | duty_all |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.753 | | **1.000** | 1–6 | 2 | 14.25 | 0.829 |
| 0.10 | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.712 | | **1.000** | 1–6 | 1 | 7.13 | 0.829 |
| 0.28 | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.655 | | **1.000** | 1–6 | 1 | 7.13 | 0.829 |
| 0.30 | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.655 | | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.753 |
| 0.50 | 0.833 | 2,3,4,5,6 | 1 | 7.13 | 0.598 | | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |
| 0.70 | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.500 | | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |
| 0.80 | 0.667 | 3,4,5,6 | 0 | 0.00 | 0.455 | | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |
| 0.90 | 0.333 | 5,6 | 0 | 0.00 | 0.249 | | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |

**Recall is 5 of 6 and event 1 is the miss, confirming the earlier table — but the mechanism is worse
than "missed".** The per-window trace (`raw/report.txt`) shows the three windows covering the 202 ms
mouth sound at 0.893 s scoring `Silence` 0.993, 0.715 and 1.000, with `vocab` at 0.003, 0.018 and
0.000. YAMNet does not weakly detect this event; it positively asserts silence over it.

**`nonsilence` reaches 6/6 only on a knife-edge.** The best window over event 1 gives
1 − P(`Silence`) = 0.285, so full recall exists at τ ≤ 0.28 and vanishes at 0.30. That is not an
operating point, it is a coincidence — and probe 2 shows what happens to that threshold under noise.

**The event-level FP/min is structurally zero and must not be quoted.** `FP/min (detection)` is 0.00
at every threshold for both proposers, and that is not precision. Merging consecutive fired 0.96 s
windows at a 0.48 s hop produces detections ≥ 1.44 s long, while the widest gap between events on
this file is 1.81 s, so essentially every detection touches some event and no detection is
event-free. What the design would read as "no false positives" is the proposer having almost no gaps
at all. `duty_all` says it plainly: at τ=0.50 the `vocab` proposer covers **60% of verified-empty
time**, and the `nonsilence` proposer 66%.

**There is one real, confident false positive.** Window 14 (6.72–7.68 s), wholly inside the
verified-empty 6.30–7.90 stretch, scores `Snoring` 0.645 against `Silence` 0.647 — a near-tie — and
survives every `vocab` threshold up to 0.60. This is the same stretch the human verification
specifically corrected for HeAR ("6.60–7.10 s contains nothing"). Both models fire in the same wrong
place, and for YAMNet it is not leakage: the nearest event starts at 7.926 s, outside the window.

At τ=0.50 the `vocab` FP window rate is 1/5 = 0.200, exact 95% CI **[0.005, 0.716]**. That interval
is the answer to "what is the FP/min", and it spans two orders of magnitude.

## Probe 2 — robustness

*Effect present if* recall falls or the FP window rate rises as SNR drops. Recall resolution is 1/6
and the FP denominator is 5 per variant, so a change of one event or one window is the smallest
visible step.

`vocab`, at τ=0.50:

| variant | recall | events | FPwin/5 | duty_all |
| --- | --- | --- | --- | --- |
| clean | 0.833 | 2,3,4,5,6 | 1 | 0.598 |
| white 20 dB | 0.833 | 2,3,4,5,6 | 0 | 0.500 |
| white 10 dB | 0.833 | 2,3,4,5,6 | 0 | 0.434 |
| white 5 dB | 0.667 | 2,4,5,6 | 0 | 0.364 |
| white 0 dB | **0.500** | 4,5,6 | 0 | 0.208 |
| pink 10 dB | 0.833 | 2,3,4,5,6 | 0 | 0.393 |
| pink 0 dB | **0.500** | 4,5,6 | 0 | 0.320 |
| reverb T60 0.3 | 0.667 | 2,4,5,6 | 0 | 0.297 |
| reverb T60 0.7 | 0.667 | 2,3,4,6 | 0 | 0.283 |

`nonsilence`, at τ=0.50:

| variant | recall | events | FPwin/5 | FP/min | duty_all |
| --- | --- | --- | --- | --- | --- |
| clean | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |
| white 20 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| white 10 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| white 5 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| white 0 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| pink 10 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| pink 0 dB | 1.000 | 1–6 | **5** | **35.63** | **1.000** |
| reverb T60 0.3 | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |
| reverb T60 0.7 | 0.833 | 2,3,4,5,6 | 0 | 0.00 | 0.655 |

**The `Silence` route does not survive additive noise at all.** At 20 dB SNR — a mild noise floor —
1 − P(`Silence`) exceeds 0.50 in every one of the 29 windows: FP rate 5/5, duty 100%, and the
recall of 1.000 is vacuous because the proposer is firing everywhere. The clean 6/6 at τ ≤ 0.28 was
resting on P(`Silence`) staying near 1.0 in the gaps, and a noise floor removes that. This is not a
threshold that needs retuning: no threshold separates events from gaps once the gaps are not silent.
The property `branch-1-airway.md` singles out as the reason to prefer YAMNet — "an explicit
`Silence` class that fires in the verified-empty gaps" — is the least robust thing about it.

**The vocabulary route degrades the other way: it keeps its specificity and loses recall.** FP stays
0/5 everywhere, while recall drops 5/6 → 4/6 at white 5 dB → 3/6 at 0 dB. Reverberation costs 1/6 at
both T60 values but changes *which* event is lost: at T60 0.3 s it drops event 3 and keeps 5, at
0.7 s it drops 5 and gets 3 back. Smearing moves events around the window grid rather than uniformly
weakening them, so a single recall figure hides which element is at risk.

One artefact worth flagging so it is not read as a result: `vocab` at τ=0.10 reaches 6/6 on
`white_snr20` only. Noise pushed a vocabulary class over 0.10 on the mouth sound. That is noise
being classified, not the event being found.

## Probe 3 — the confirmer

### 3a — HeAR has no negative available on this file

*Effect present if* some 2 s window overlaps no verified event. If none does, corroboration carries
no information about the proposal, because HeAR would be firing correctly no matter where it is
asked.

```
HeAR windows overlapping no event: 0 of 50
head gap (0 -> first onset)   : 0.893 s
tail gap (last offset -> end) : 0.827 s
gap  1.095 ->  2.275 : 1.180 s   < 2 s
gap  3.496 ->  5.308 : 1.812 s   < 2 s
gap  6.291 ->  7.926 : 1.635 s   < 2 s
gap  8.494 ->  9.610 : 1.116 s   < 2 s
gap 10.250 -> 11.620 : 1.370 s   < 2 s
widest event-free stretch: 1.812 s
```

Every gap in this recording, and both ends, are shorter than 2 s. **No placement of a 32000-sample
window on this file can avoid a verified event.** Combined with the established fact that 40 ms
inside the window suffices to fire the detector, HeAR's positive answer is guaranteed everywhere,
independent of the proposal it is asked about. As a confirmer on this file its discriminative power
is not low; it is zero by geometry.

This is the same box-car property `branch-1-airway.md` already uses to disqualify HeAR as a
*proposer* ("merges any two events closer than 2 s"). The measurement shows it disqualifies HeAR as a
*confirmer* on the same evidence — the doc drew half the conclusion.

### 3b — planted proposals in verified-empty audio

*Effect present if* HeAR scores a 480 ms proposal centred in each verified-empty region below the
threshold that keeps the true events. It does not.

| region | planted proposal | centred top | score | best-overlapping top | score |
| --- | --- | --- | --- | --- | --- |
| 0.000–0.780 | 0.150–0.630 | Breathe | 0.036 | Breathe | 0.612 |
| 1.095–2.275 | 1.445–1.925 | Breathe | **0.951** | Breathe | 0.989 |
| 3.500–5.300 | 4.160–4.640 | Breathe | **0.896** | Breathe | 0.998 |
| 6.300–7.900 | 6.860–7.340 | Cough | **1.000** | Cough | 1.000 |
| 8.500–9.600 | 8.810–9.290 | Cough | **0.994** | Cough | 1.000 |
| 10.250–11.620 | 10.695–11.175 | Speech | 0.336 | Cough | 0.999 |
| 13.200–13.790 | 13.255–13.735 | Snore | 0.087 | Speech | 0.339 |

At θ=0.5 the confirmer corroborates **4 of 7** planted verified-empty proposals (6 of 7 in the
generous best-overlapping mode), four of them above 0.89. Set that against the true events:

| ev | element | centred top | score |
| --- | --- | --- | --- |
| 1 | mouth non-speech sound | Breathe | **0.036** |
| 2 | exhalation (breath) | Breathe | 0.986 |
| 3 | exhalation (breath) | Breathe | 0.991 |
| 4 | cough | Cough | 1.000 |
| 5 | cough | Cough | 0.999 |
| 6 | speech | Speech | **0.339** |

**HeAR's ranking is inverted relative to ground truth on this file.** It scores the verified-empty
6.86–7.34 proposal at `Cough` 1.000 while scoring the verified speech event at 0.339 and the verified
mouth sound at 0.036. No threshold both keeps events 1 and 6 and rejects the empty regions — the
orders overlap. The mechanism is visible in the 6.86–7.34 row: the 2 s window nearest that centre
runs 6.10–8.10 and catches 174 ms of cough 4, which is four times what the detector needs.

### 3c — what confirmation does to the chain's numbers

`vocab` at τ=0.10, mode=centred; YAMNet alone gives recall 0.833, 5 detections, precision 1.000
(structurally — see probe 1):

| θ | kept | recall | events | precision | labels |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 5 | 0.833 | 2,3,4,5,6 | 1.000 | Breathe, Breathe, Cough, Cough, Speech |
| 0.20 | 5 | 0.833 | 2,3,4,5,6 | 1.000 | Breathe, Breathe, Cough, Cough, Speech |
| 0.30 | 4 | **0.667** | 2,3,4,5 | 1.000 | Breathe, Breathe, Cough, Cough |
| 0.50 | 4 | **0.667** | 2,3,4,5 | 1.000 | Breathe, Breathe, Cough, Cough |
| 0.90 | 4 | **0.667** | 2,3,4,5 | 1.000 | Breathe, Breathe, Cough, Cough |

**The confirmer costs recall and buys nothing.** Above θ=0.20 it deletes the speech event and takes
recall from 5/6 to 4/6. Precision is 1.000 before and after, so there is no gain to trade against —
and precision was 1.000 only because no detection is event-free, not because the proposer is precise.
The one confirmation the design would most want, rejecting window 14's confident `Snoring` in
verified-empty audio, is unavailable: that window is inside a merged detection that also covers
cough 4.

The `any_overlapping` mode is worse, not better. Taking the best of every overlapping window
relabels event 2, a breath, as `Cough` — the cough plateau leaks into the breath's neighbourhood, so
being generous to the confirmer costs label correctness.

## Probe 4 — the bounder

*Effect present if* |onset error| ≤ 5 ms is reachable at some Δ. The envelope grid is 1 ms, a fifth
of the claimed tolerance, so an error of tens of milliseconds is the bounder's and not the grid's.

Seeding the bounder from the verified span (`oracle`) and from the YAMNet detection (`chain`) gives
**identical** results at every Δ, for every event. The bounder's error is its own; the proposer's
seed does not enter it, because the peak inside the seed is the same peak either way.

| ev | Δ dB | onset | offset | onset err | offset err |
| --- | --- | --- | --- | --- | --- |
| 4 cough | 3 | 7.924 | 8.514 | **−2.0 ms** | +20.0 ms |
| 4 cough | 6 | 7.926 | 8.510 | **+0.0 ms** | +16.0 ms |
| 4 cough | 10 | 7.927 | 8.337 | **+1.0 ms** | −157.0 ms |
| 4 cough | 12 | 7.927 | 8.330 | **+1.0 ms** | −164.0 ms |
| 4 cough | 15 | 7.928 | 8.313 | **+2.0 ms** | −181.0 ms |
| 4 cough | 20 | 7.928 | 8.311 | **+2.0 ms** | −183.0 ms |
| 5 cough | 3 | 9.605 | 9.983 | **−5.0 ms** | −267.0 ms |
| 5 cough | 6 | 9.610 | 9.953 | **+0.0 ms** | −297.0 ms |
| 5 cough | 10 | 9.612 | 9.912 | **+2.0 ms** | −338.0 ms |
| 5 cough | 12 | 9.613 | 9.890 | **+3.0 ms** | −360.0 ms |
| 5 cough | 15 | 9.613 | 9.878 | **+3.0 ms** | −372.0 ms |
| 5 cough | 20 | 9.613 | 9.801 | **+3.0 ms** | −449.0 ms |

**The ±5 ms onset claim holds, and this is the first evidence for it beyond n=1.** Worst onset error
across both verified coughs and a 17 dB range of threshold choice is 5 ms; at Δ=6 dB both coughs land
at 0 ms. Onsets are also nearly threshold-free — the whole Δ ∈ [3, 20] dB range moves cough 4's onset
by 4 ms and cough 5's by 8 ms. That is the one claim in branch 1 that survives contact with the
labels intact, and the stage making it uses neither model.

**Offsets are not bounded, at any Δ.** Cough 4 ranges +20 ms to −183 ms and cough 5 −267 ms to
−449 ms across the same sweep — a 203 ms and 182 ms spread from a choice the design does not specify.
Cough 5 is never closer than 267 ms: this is not a threshold to be tuned but the bounder walking off
the decaying voiced tail of the cough, which the human included in the span and which never rises
5 dB above the local floor. `measurements-2026-08-17-span-probe.md` recorded the same failure for
breath offsets (2.03 s and 1.76 s of movement across floor+3 to floor+12 dB); it holds for coughs
too, at a tenth the magnitude but still 50× the onset error.

Supplementary, event 6 (speech, verified span 11.62–13.20, flagged above as outside the instructed
scope): onset +46 ms at Δ=3 but **+303 ms at Δ ≥ 10**, where the walk snaps past the low-energy
onset to a later syllable; offset −1064 ms to −1188 ms, where it stops at the first inter-word pause.
An amplitude-envelope bounder is structurally wrong for connected speech, not merely imprecise.

## Probe 5 — the chain, end to end

Best configuration found (`vocab` τ=0.30 or 0.50, HeAR θ=0.5 centred, bounder Δ=12 dB):

| ev | element | proposed | confirmed | label | onset err | offset err |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | mouth non-speech sound | **no** | — | — | — | — |
| 2 | exhalation (breath) | yes | yes | Breathe ✓ | +46 ms | n/a |
| 3 | exhalation (breath) | yes | yes | Breathe ✓ | +47 ms | n/a |
| 4 | cough | yes | yes | Cough ✓ | **+1 ms** | −164 ms |
| 5 | cough | yes | yes | Cough ✓ | **+3 ms** | −360 ms |
| 6 | speech | yes | **no** (0.268) | Speech | — | — |

**proposed 5/6 · confirmed 4/6 · correct label 4/6 · event-free detections 0.**

The `nonsilence` τ=0.10 configuration, which is the only one that proposes all six, ends up worse:
event 1 is proposed and then rejected by the confirmer at 0.036, and event 3 is mislabelled `Cough`
with a **+2619 ms** onset error, because at that threshold the proposer merges events 2 and 3 into one
detection and the bounder locks onto the wrong peak inside it. Recovery: proposed 6/6, confirmed 4/6,
correct label **3/6**.

So the number the design is claiming is **4 of 6 events with correct labels, two of them bounded to
±3 ms on the onset and none bounded on the offset** — and the ±3 ms comes from the DSP stage alone.

## What this settles, and what it refutes

1. **The FP/min measurement the draft defers to cannot be settled by this file.** Five independent
   event-free window decisions, one FP worth 7.13 FP/min, a clean-run 95% interval of [0.005, 0.716].
   Any FP/min number quoted from this recording is a placeholder. Settling it needs verified-empty
   audio measured in minutes, not 8.4 seconds — and a file whose gaps exceed 2 s, or HeAR cannot be
   scored on it at all.

2. **The event-level FP/min metric is the wrong metric for this proposer and should be replaced.** At
   0.96 s / 0.48 s the merged detections are wider than every gap in the recording, so the count is
   structurally zero and reads as perfect precision. `duty_all` — 50–66% of verified-empty time
   covered at usable thresholds — is the measure that shows what is actually happening. Whatever the
   graph reports about proposer specificity should be a duty cycle or a per-window rate, not a merged
   event count.

3. **"Explicit `Silence`" is the wrong reason to choose YAMNet as proposer.** It is the property that
   fails first: at 20 dB SNR the `nonsilence` route fires in 29 of 29 windows. The vocabulary route
   is the defensible one, and it costs event 1 permanently — YAMNet asserts `Silence` at 0.72–1.00
   over a verified mouth sound, so `transient_propose` has no vocabulary for that element and no
   threshold creates one.

4. **HeAR cannot serve as the confirmer for this branch, and the reason is already in the draft.**
   Zero of 50 windows on this file avoid a verified event, so it has no negative to give; it
   corroborates 4 of 7 planted verified-empty proposals above 0.89 while scoring the verified speech
   event at 0.339 and the verified mouth sound at 0.036; and adding it costs 1/6 recall for zero
   precision gain. The 2 s window that disqualifies HeAR as a proposer disqualifies it as a
   confirmer by exactly the same geometry, on any recording whose events are closer together than
   2 s — which is what respiratory bouts are.

5. **The ±5 ms cough-onset claim is upheld and now rests on n=2, threshold-insensitively.** Keep it.
   Δ=6 dB is the sweet spot on both coughs, but the claim does not depend on that choice.

6. **The same bounder must not be allowed to report an offset.** Cough offsets move 180–200 ms across
   the Δ sweep and cough 5 is never within 267 ms. `span_refine` should emit an onset and an
   explicit "offset unbounded", not a span that looks symmetric in precision and is not. For
   connected speech it should not run at all: 303 ms onset error and 1.1 s offset error at Δ=12 dB.

7. **A merged proposal destroys the bound even when the bounder is good.** Event 3's +2619 ms onset
   error at `nonsilence` τ=0.10 is the whole failure: the bounder is ±3 ms when the seed contains one
   event and useless when it contains two. `span_refine` reading "only inside a proposal" is
   therefore not the safety property the draft treats it as — it is a dependency on the proposer
   having already separated the events, which is the same thing `group_events` is recorded as unable
   to do.

## What this measurement does not show

- **n=1 recording, n=6 events.** Every recall figure moves in steps of 0.167 and every FP figure in
  steps of 0.2. Two of the six events are the same element (breath) and two more are the same element
  (cough), so the effective label diversity is four.
- **One speaker, one microphone, close-miked.** The bounder's onset performance in particular is
  measured on a 48 dB level step against a −67 dB floor. It has no claim on distant or noisy capture.
- **Degradations are synthetic and self-chosen.** White and pink noise and an exponential-decay RIR
  are not real rooms or real recording chains, and the SNR is defined on a file that is 60%
  verified-empty.
- **The bounder was not run on the degraded copies.** Its clean-audio onset accuracy is the only
  thing established; the probe that would matter next is whether ±5 ms survives 10 dB SNR.
- **No timing or memory figures are reported.** Other work was running on this host throughout; a
  contended cost number would be worse than none.

## Files

| file | what it is |
| --- | --- |
| `ground_truth.py` | the six events, the empty stretches, the scorable-empty derivation |
| `degrade.py` | writes the nine variants; seed 20260819 |
| `run_backends.py` | YAMNet (521 posteriors) and HeAR (8 labels) over all nine |
| `bounder.py` | the DSP envelope span-bounder |
| `score.py` | probes 0–5 |
| `inspect_labels.py` | checks the draft's AudioSet vocabulary exists |
| `raw/degradations.json` | every degradation parameter and the resulting peak/RMS |
| `raw/yamnet.json.gz`, `raw/hear.json.gz` | the stored posterior matrices |
| `raw/report.txt` | the full run output the tables above are drawn from |
| `results.json` | the same numbers as JSON |

The degraded WAVs are not committed (24 MB, exactly regenerable from the seed and the manifest).
