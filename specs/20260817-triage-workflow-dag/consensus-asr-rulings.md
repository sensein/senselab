# Consensus ASR redesign — owner rulings

Decisions taken by the owner on 2026-09-06, during the plan/review/verify cycle on
`consensus-asr-redesign.md`. The revised plan folds these in; they are recorded here because they
close questions the plan left open, and a later reader should not reopen them.

## R-1. A bracketed override and an insertion are two mechanisms, and stay separate

- **Override.** A bracketed token and a plain one that normalise to the same key are the *same*
  column. The bracketed surface wins the word's text. This is an agreement whose surface form is
  the more informative of the two — `[COUGH]` over `cough`.
- **Insertion.** A bracketed token no other source produced any token for is a column of its own,
  owned by one source. It is not an override of anything, because there is nothing to override.

CrisperWhisper's fillers (`[UM]`, `[UH]`) are almost always the second case: Qwen emits no token
there at all. Collapsing the two mechanisms would make a filler look like a contested reading of
its neighbour, which is the defect this redesign exists to remove.

## R-2. A failed recognizer block is a failure, not a degraded run

Fewer than two ASR hypotheses in the store is an error. The consensus does not proceed over a
single source and must not present it as agreement: bold means corroborated, and one source
corroborates nothing. The failure is raised and flagged rather than absorbed into a quieter
artifact that reads like a successful run.

This preserves the guarantee the retired `LookupError("both recognizers are needed")` gave, in
N-generic wording.

## R-3. Consensus is sequence alignment; time only sets a word's onset and offset

The recognizers ran on the same input, so their outputs are sequences to be aligned as sequences.
Time takes no part in deciding what matches what, what order words come in, or what is grouped
together. It is read once, after alignment, to set the resulting word's onset and offset.

This retires three things from the triage path:

- time-overlap slot grouping, and with it `slot_overlap` and `slot_mid_tol_s`;
- the re-sort of slots by averaged member time after grouping
  (`speech_to_text_ensemble/api.py:363`), which is what reorders a correctly aligned transcript;
- any notion that two words are "the same word" because they overlap in time.

**The order of the emitted transcript is the order of the alignment columns, verbatim.**

Measured, on `sub-1f4ea26f…task-Story-recall-(v2)`, 12.5–13.5 s. The aligner's columns are right:

```
col42 {cw: and}              col45 {cw: the}
col43 {cw: the}              col46 {cw: d-}
col44 {cw: and, qw: and}     col47 {cw: the, qw: the}
```

CrisperWhisper heard `and the and the d- the` — a real disfluency, correctly aligned against Qwen's
`and … the`. The emitted transcript reads `and and the the the d-`: same words, wrong order,
because col43 and col44 share a midpoint of 12.84 s and col47 (13.21 s) sorts ahead of col46
(13.36 s). Single-source insertions are shuffled past the agreed words they precede and identical
tokens end up adjacent, so a genuine repetition is displayed as a stutter the speaker did not
produce.

The repetitions are real and are preserved verbatim. Nothing collapses a repeated token.
