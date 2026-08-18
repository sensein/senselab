# Verified labels — `streaming-audio-2026-07-30T04-21-56-487Z.wav`

**Human-verified 2026-08-18.** The first labelled audio in this project. Everything prior to this was
model consensus, and gap G9 recorded that no labelled verdicts existed anywhere. These are usable as
ground truth.

| t (s) | element | verification |
| --- | --- | --- |
| 0.893 | **mouth non-speech sound** | corrected — had been inferred as a handling click |
| 2.275 | exhalation | confirmed |
| 5.308 | exhalation | confirmed |
| 6.60-7.10 | **nothing — no breath here** | corrected — HeAR scored Breathe 0.49 and was wrong |
| 7.924 | cough | confirmed; resolves YAMNet `Cough` 1.000 against AST `Throat clearing` 0.96 in YAMNet's favour |
| 9.609 | cough | confirmed, same |
| 11.62-13.20 | speech, "There's something going on." | confirmed |

Still unlabelled: **13.79-14.04 s**, where Brouhaha's VAD rises above 0.5 while community-1 stays at
zero. It was outside the clips presented for verification, so it remains the one open question on this
file.

## Three corrections this forces

**1. `mouth non-speech sound` is a taxonomy element and D11 omitted it.** A lip smack or tongue click
is vocal-tract produced, non-lexical, and very short — closer to an airway event in shape than to
anything in the speech branch. It is not an artifact to be filtered. The existing pipeline already had
it: `mask.target_event_types_by_task` lists `["speech", "breath", "mouth_noise"]` for speech tasks, so
the eighteen-element vocabulary in D11 dropped a category the current code already carries. Add it.

**2. HeAR produced a false positive and I credited it as a catch.** The earlier measurement note said
HeAR "found a quiet breath at 6.60-7.10 s that had been hand-labelled silence", framing the model as
having corrected a human. There is no breath there. Two things went wrong: HeAR scored 0.49, *below*
its own 0.5 threshold, and that was treated as a detection; and a model's disagreement with a human
label was read as the model being right. The original silence reading was correct.

**3. The discriminator has a blind spot wider than recorded.** With verified labels, per element:

| element | Brouhaha VAD | community-1 segmentation | discriminator verdict |
| --- | --- | --- | --- |
| speech | 0.689 | 0.790 | correct |
| cough | 0.005-0.009 | 0.574 / 0.906 | correct — voiced non-lexical |
| exhalation | 0.005 | **0.000** | invisible |
| mouth non-speech sound | ~0 | ~0 | invisible |
| silence | 0.007 | 0.000 | invisible |

D16 recorded that breath is invisible to both. Mouth noise is too, so the "neither responds" cell now
holds **three distinct things** — breath, mouth noise and silence — and cannot be read as "no voice".
The pair separates lexical speech from voiced non-lexical vocalization and nothing else. Every
unvoiced vocal element needs the DSP envelope and HeAR route, and HeAR's false positive here means
that route needs its own verification before it is trusted.

## Where the labels came from, and what that says about the fold

**Locations: DSP only.** Short-time RMS at 5 ms hop, a 1 ms envelope, and onset detection on spectral
flux at 5.33 ms hop produced every onset. No model.

**Identities: four independent sources.** YAMNet over 0.96 s windows; AST over the same; HeAR's bundled
event detector at 2 s / 0.25 s hop; and two Whisper models whose identical transcript fixed the speech
label and its boundaries.

**The tie was broken by a non-model measurement.** AST said `Throat clearing` 0.96 against YAMNet's
`Cough` 1.000. What settled it in YAMNet's favour was rise time and level step from the envelope —
9-17 ms and 45-49 dB for the coughs against 60-127 ms and 20-29 dB for the breaths — plus the
descending harmonic striations after each burst. Verification confirmed that call.

**The pattern in the two errors is the useful part.** Every label that proved correct had at least two
independent sources, **at least one of them not a classifier** — the DSP envelope, or an ASR transcript.
Both labels that proved wrong had exactly one source and it was a model score:

- the mouth sound was labelled from *absence* — no model classified it, so a level step with no
  harmonic structure became "handling click" by default;
- the phantom breath came from HeAR alone at **0.49, below its own threshold**, with no DSP support.

Two rules follow, and they are sharper than D6's "families with uncorrelated failure modes":

1. **The fold must include at least one non-classifier member.** Classifier agreement was not what made
   the correct labels correct; the envelope was, both as corroboration and as tiebreaker.
2. **A sub-threshold score from a single model must not become a label.** It may become a question.

## Corrections to this file's own conclusions, 2026-08-18 (second pass)

**HeAR was not the problem.** It scored 0.96-0.995 on the verified breaths and 0.995-0.9998 on the
verified coughs. Only the handling of a 0.49 — below its own threshold — was wrong, and that was the
caller's error, not the model's. The rule stands as "a sub-threshold single-model score is a question,
not a label"; the earlier framing of HeAR as unreliable does not.

**6.60-7.10 s is reopened as possibly an inhalation.** It was recorded above as "nothing". It may be an
inhalation, which is quieter, spectrally different and shorter-tailed than the two verified
exhalations. If so, HeAR's 0.49 was a weak true positive and the correct reading was neither "silence"
nor "breath" but *inhalation, a class nothing in the fold distinguishes*. D11's vocabulary separates
inhalation from exhalation as elements 1 and 2; none of the detectors used here does. **Status:
unresolved**, and it is the more informative outcome of the two.

**The stationary tones are probably music, not interference.** The tones at 85.0, 108.4, 164.1, 1564.5
and 1757.8 Hz were described as "mild, persistent, non-acoustic-looking interference". There is music in
the background of this recording, so those are more likely partials. Two consequences: the claim that
this file has no background content is wrong, and the tonal-interference reading would have sent a
downstream quality stage looking for an electrical fault.

**The rise-time and level-step figures must not become thresholds.** They are n=2 per class from one
healthy adult on a close mic: 9-17 ms rise with a 45-49 dB step for cough, 60-127 ms with 20-29 dB for
breath. They describe a *healthy adult voluntary* cough. Across the lifespan and across disorder they
are expected to fail, and to fail hardest where the signal matters most:

- reduced peak cough flow in neuromuscular disease, post-stroke, and sarcopenia — smaller step, slower
  rise, by the same mechanism that makes peak cough flow a clinical measure;
- absent glottic closure (vocal fold paralysis, tracheostomy) — no explosive phase at all;
- infant and child cough — shorter, higher, different spectral balance;
- COPD and asthma — prolonged expiration and wheeze outside any breath bound fitted here.

So the D6 rule holds in shape — the fold needs at least one non-classifier member — while these
particular numbers are a single-subject observation. Any use as a threshold requires a derivation over
a population spanning age and disorder, in `data/` per CLAUDE.md, and none exists.

## Verified cough windows, and what they reveal about CrisperWhisper — 2026-08-18

Human-verified spans, not just onsets:

| event | window | duration |
| --- | --- | --- |
| cough 1 | 7.926 - 8.494 s | 568 ms |
| cough 2 | 9.610 - 10.250 s | 640 ms |

`nyralabs/CrisperWhisper2.0_turbo` transcribed the file as
`[breath] [breath] [cough] [UH] [breath] There's something going on.` with token timings. Scored
against those windows:

| token | verdict |
| --- | --- |
| `[cough]` 7.90-8.48 | cough 1: onset −26 ms, offset −14 ms, duration 580 ms against 568 ms. Correct label, near-exact bounds. |
| `[UH]` 9.60-9.94 | cough 2: onset −10 ms, **offset −310 ms**, 340 ms of a 640 ms event. Wrong label. |
| `[breath]` 10.12-10.22 | **entirely inside cough 2**, 100 ms, 510 ms after its onset. Wrong label. |

**One physiological event became two mislabelled tokens.** The two tokens cover 440 of cough 2's
640 ms with a 180 ms gap between them. The mechanism is legible: cough 2 is the louder one — 9 ms
rise, 48.5 dB step — and carries a descending harmonic chirp from 9.65 to 10.00 s, its voiced phase.
A speech model maps that voiced phase to the nearest thing it knows, a filler vowel, and the aspirate
tail to a breath. It is a speech prior imposed on a non-speech event, not a random error.

### The two instrument classes are exact inverses

| | timing | label | event grouping |
| --- | --- | --- | --- |
| CrisperWhisper | **10-52 ms edges** | unreliable on non-speech | fragments one event into several |
| YAMNet / AST / HeAR | 480-960 ms smear, unfixable by hop | confident and correct at clip level | one window, one verdict |

So neither is a taxonomy instrument alone, and the split is not "models versus DSP" as recorded
earlier — it is **boundaries from CrisperWhisper, identity from the clip-level classifiers, and
grouping from neither**. Deciding that a burst, a voiced phase and an aspirate tail are one cough
rather than three events is a third problem, and nothing measured here solves it. It is the same
defect as D12's cough-bout question one level down: bout → cough → phase.

### And it corrects an earlier claim about offsets

The measurement note recorded that cough offsets carry 1.04-1.10 s of ambiguity, from moving an
envelope threshold between floor+12 dB and floor+3 dB. With verified windows the coughs are 568 and
640 ms, and CrisperWhisper bounded cough 1's offset to within 14 ms. So that ambiguity was an artifact
of the envelope method, not a property of the event. The breath-offset ambiguity (2.03 s) has not been
retested against verified windows and may be the same artifact.
