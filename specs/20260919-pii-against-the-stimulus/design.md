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

Stimulus containment is a fact about the text and needs no threshold.

## The spread rule is wrong, and that is measured

The 66%-of-spans figure above makes a spread cut look like the cheap filter, and it was the first
rule proposed here. The census tested it and it points the wrong way: among surviving surfaces,
those seen **only** in fixed-script recordings — definitionally false, every one — are *more*
concentrated in single-subject surfaces than free-response-only surfaces are, **92.2% against
85.6%**. Cutting by spread would therefore discard proportionally more of what is real than of what
is not. No spread threshold is derived, because none should be.

What the corpus supports instead is three rules, none of them statistical:

- **A — the task asks for no words.** 28,719 recordings; a family that elicits no lexical content
  cannot carry a participant's disclosure. A routing question, not a detector one.
- **B — the surface is an exact substring of the recording's own `stimulus_text`.** 84.6% of
  read-task findings are. A SPEECH question, answerable from the alignment already in hand.
- **C — the surface was condemned by A or B somewhere in the corpus.** A 5,363-surface list read off
  the corpus rather than chosen.

Together these take **42.9% down to 8.9%** — 5,327 of 60,207 recordings, 48,959 of 152,722 findings
surviving. Three whole false-positive classes lose every member: structured identifiers on
alphabetic surfaces, item-list place names, and bracketed event markers.

## The owner's ruling: the scan runs only on words the task did not ask for

Asked whether the scan should run at all on families that ask for no words, the owner ruled wider
than this document had proposed:

> pii should run only if there are words not in the task — and definitely on any open response tasks

So the gate is not a family list and not a post-hoc filter. It is a word-level test on the
recording's own transcript, taken before the detector is called:

- The scan runs when at least one lexical word is **not** in the recording's declared prompts. A
  recording declaring no prompt has every word outside the task, so it is always scanned — which is
  where disclosure lives.
- The scan runs unconditionally on a family whose expected pattern is a free response, whatever its
  transcript holds. Open response is the case that must never depend on a word test a sparse or
  prompt-echoing transcript could defeat.
- Otherwise the detector is never called, and a `pii_scan` measurement records `scanned: false` with
  the reason and the lexical word count, so a reader can tell "nothing was found" from "nothing was
  looked for".

This removes at the source the two largest false-positive classes the census measured: the 28,719
recordings whose task asks for no words, and the read tasks whose findings are the script.

## Why what still gets scanned annotates rather than suppresses

The asymmetry is the whole argument, and it applies hardest to C. A corpus-wide surface list
generalises B across recordings, so a name genuinely disclosed in one participant's free response is
dropped because the same string sits inside a different recording's script. The Rainbow Passage and
the Caterpillar Passage between them contain ordinary given names and place names; a participant who
shares one would be silently unredacted by rule C.

So none of A, B or C removes a `pii` entity. Each is recorded on the finding, REDACT keeps redacting
the unfiltered set, and VERDICT and REPORT read the annotations to decide what to *say*. The 8.9%
is then a number the reader is shown rather than a number the pipeline enforces, and no true
positive is ever lost to a rule derived from the corpus it is policing.

The gate settles rule A and most of rule B before the detector runs. What the annotation still
carries is the residual: a recording that *did* say something of its own, whose detector then fired
on the prompt anyway. That one is scanned, minted and marked — never dropped.

## Open

- Whether REPORT shows the annotated count, the unannotated count, or both.
- Whether the LLM re-read should see the stimulus, so it is not re-deciding the same false positives.
- The reference standard for the surviving 8.9%: nothing in this census was adjudicated.

## The gate was blind where no stimulus is declared, measured on the corpus run

Read from 8,212 live rows of the 2026-09-20 corpus run, which carries the gate:

| family stem | reached REDACT | declares a stimulus |
|---|---:|---:|
| harvard-sentences-list | 7.5% | 100% |
| cape-v-sentences | 7.8% | 100% |
| rainbow-passage | 4.8% | 100% |
| **diadochokinesis** | **83.5%** | **0%** |
| prolonged-vowel | 25.9% | 0% |
| picture-description | 34.7% | 0% |
| respiration-and-cough | 1.2% | 0% |

**45.8% of recordings declare a stimulus prompt at all.** Where one exists the gate does exactly
what it was built for: the read families fall to 4.8-7.8%, against the census's finding that 84.6%
of read-task findings were substrings of the script. Where none exists the gate cannot fire, and
diadochokinesis sits at 83.5% — reproducing the census's ~91% false-positive rate on carrier words.

A syllable task declares no `stimulus_text`, and its expectation holds the carrier as an **ARPAbet
sequence** (`("p", "aa")`), which no transcript can be matched against. But the family names it:
`diadochokinesis-buttercup` asks for "buttercup". So the carrier is taken from the family and joined
to the stimulus haystack, and the one existing word test then does the rest.

### What the first attempt got wrong

The first version skipped the scan outright for any task asking for no lexical content. That broke
a test named *the safety-critical invariant*:

> a disclosure spoken over a DDK take must not escape because the instruction asked for nonsense
> syllables

which is right, and the reason the carrier joins the haystack rather than switching the scan off.
Thirty repetitions of "pa" are not thirty disclosures; "alice" spoken during the same take still is.

`prolonged-vowel` and `maximum-phonation-time` remain uncovered: they ask for a sustained vowel, so
there is no carrier orthography to match and whatever ASR writes down is an artefact of a signal
with no words in it. Naming that case honestly is owed; suppressing it by family is not, for the
same reason the blanket skip was wrong.
