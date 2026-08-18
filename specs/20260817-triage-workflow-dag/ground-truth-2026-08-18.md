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
