# PII findings must be read against the stimulus

Measured 2026-09-19 over the 2026-09-08 corpus stores, 60,207 consensus transcripts, detectors
`gliner + presidio + rules`. The scan is a standalone census: the SPEECH branch never ran at corpus
scale, so these are the detectors' readings of the transcripts, not the pipeline's PII behaviour.
The Gemma-4 LLM re-read is a separate step and is in none of these numbers.

## What the corpus says

**25,853 of 60,207 consensus transcripts (42.9%) carry at least one PII span.** That number is not a
PII rate, and the corpus says so three independent ways.

By what the task asks the participant to produce:

| expectation | scanned | with a finding | rate | spans/recording |
|---|---:|---:|---:|---:|
| no words asked for | 27,119 | 11,445 | 42.2% | 0.83 |
| reads a fixed script | 20,765 | 6,864 | 33.1% | 0.54 |
| may disclose | 11,658 | 7,169 | 61.5% | 1.47 |
| produces items of a dictated class | 665 | 375 | 56.4% | 7.32 |

A task that asks for no words cannot contain participant PII; a task that reads a fixed script
contains the experimenter's words. Those two rows are 47,884 recordings and 18,309 findings.

Per family the same reading is plainer still: `caterpillar-passage` 98.7%, `diadochokinesis-buttercup`
91.1%, `diadochokinesis-pataka` 89.9%, `diadochokinesis-ta` 88.9% — against `breath-sounds` at 1.8%,
the one family in the column behaving as a PII detector should.

Directly, on the text: of 74,434 spans on transcripts that have a `stimulus_text`, **35,589 (47.8%)
fall inside the stimulus itself.**

By spread: 18,661 distinct surfaces, of which 14,202 appear in exactly one recording and 767 appear
in ten or more. Those 767 account for **100,292 of 152,722 spans (66%)**. A surface appearing across
hundreds of different speakers is corpus vocabulary, not personal information.

## What the branch does today

`nodes/speech.py` step 7 mints one `pii` entity per finding and marks every word it covers. It
consults nothing about the stimulus, although the branch computed a `StimulusAlignment` earlier in
the same run and that alignment already distinguishes an `ExpectedToken` — a word the stimulus asked
for — from an `UnexpectedWord`.

So the information needed to read a finding correctly is in the branch, in hand, unused.

## The change: annotate, never suppress

A `pii` entity keeps being minted for every finding. It gains an attribute saying whether the
surface it covers lies inside the aligned stimulus, and REDACT keeps redacting on the unfiltered
set.

Suppression is the wrong instrument here and the asymmetry is why: dropping a true positive exposes
a participant, while keeping a false one costs a redacted stimulus word. The graph already has the
right shape for this — a branch reports and VERDICT decides — and the reporting side is where the
correction belongs. VERDICT and REPORT can then separate "this recording discloses something" from
"the detector flagged the sentence the participant was asked to read", which is what makes a flag
legible to whoever reads the report.

Stimulus containment is a fact about the text and needs no threshold. A spread-based rule needs one,
and a threshold needs a derivation and a home in `data/` — that is the open half of this design, and
it waits on the census's own recommendation.

## Open

- The spread rule and its derivation.
- Whether REPORT should show the annotated count, the unannotated count, or both.
- Whether the LLM re-read should see the stimulus, so it is not re-deciding the same false positives.
