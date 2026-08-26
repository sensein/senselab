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
- [ ] Remove the unused forced-alignment block and its dependencies from PREPROCESS.
- [ ] Record consensus ASR as the timing authority in provenance and retain all word uncertainty.
- [ ] Update operational documentation and the code-derived flowchart.
- [ ] Expose consensus word uncertainty in the machine-readable report.
- [ ] Run focused tests and a prolonged-vowel evaluation on Engaging.
