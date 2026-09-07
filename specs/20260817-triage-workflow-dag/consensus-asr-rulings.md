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
