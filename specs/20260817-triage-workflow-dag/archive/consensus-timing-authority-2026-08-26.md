# Consensus Timing Authority

## Decision

The two-recognizer consensus is the authoritative transcript and word-timing product for triage.
PREPROCESS must not run a second forced-alignment pass over that transcript.  Each consensus word
retains the agreement and timing uncertainty emitted by `fuse_consensus_words`; downstream spans,
PII planning, and reports use those words directly.

This makes the evidence chain explicit: a word's text, extent, confidence, existence confidence,
temporal confidence, coverage, recognizers, and timing sources all come from the same ASR consensus
operation.  An additional aligner would produce a competing timing product that no branch consumes.

## Delivery Checklist

- [x] Confirm that triage branches already read consensus `word` entities rather than `alignment`.
- [x] Remove the unused forced-alignment block and its dependencies from PREPROCESS.
- [x] Record consensus ASR as the timing authority in provenance and retain all word uncertainty.
- [x] Update operational documentation and the code-derived flowchart.
- [x] Expose consensus word uncertainty in the machine-readable report.
- [x] Run focused tests and a prolonged-vowel evaluation on Engaging.

## Evaluation Record

Engaging job `21336145` completed the cold run on the B2AI prolonged-vowel file. It confirmed that
the store carries no `alignment` entity and that the consensus words expose the timing-confidence
fields, but it also exposed two newly required phonation settings missing from the campaign override.

Job `21338137` repeated the run with the two documented B2AI evaluation guesses (`250 Hz` maximum
unvoiced formant bandwidth and `0.8` minimum word-aligned evidence fraction). It completed in 88 s
with warm caches, produced voiced and unvoiced phonation spans, and preserved the same consensus
transcript authority. The file-level triage remains a `flag`: the reported VOICE flag is the known
period-doubling alias condition, not missing phonation evidence.
