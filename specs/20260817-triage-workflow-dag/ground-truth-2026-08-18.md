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
