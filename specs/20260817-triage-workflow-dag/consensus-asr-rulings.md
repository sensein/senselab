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

## R-4. Consensus emits words with time and stops; slots and spans are downstream consumers

Consensus does emit time — each word carries an onset and an offset, set from its sources after
alignment (R-3). What consensus does **not** do is produce slots or spans, and no slot or span
concern reaches back into it.

The boundary:

- **Consensus produces** an ordered list of words. Each carries its text, which sources produced
  it, each source's own reading and timing, and the onset/offset derived from those. Column order,
  verbatim. Nothing else.
- **Downstream consumes** that list. The `asr` span source in PREPROCESS's `_spans`, SPEECH's
  speech spans, the figure's lanes and the report's token panel each build whatever grouping they
  need *from* the words. None of them is built inside the consensus, and none of them influences
  how it matched or ordered anything.

The retired `fuse_word_streams` conflated these: its "slot" was simultaneously the grouping unit
for matching, the ordering unit for output, and the extent handed downstream. That is why a
downstream concern — the averaged member time a span consumer wanted — was able to reorder the
transcript itself.

A consequence worth stating: a span consumer that wants a different grouping is free to compute it,
and doing so cannot change the transcript.

## R-5. The stream is linear, its times are monotonic, and conflict is recorded as uncertainty

Consensus produces a **consensus text stream**: one linear sequence. Each position carries
word-level metadata — the text, which sources produced it, each source's own reading, each
source's own timing, and a derived onset/offset.

Four rules, in priority order:

1. **Sequence alignment is authoritative** for what matches what and for the order of the stream.
2. **Each source's own timing is recorded verbatim**, per source, on the word.
3. **The derived onset/offset is monotonically non-decreasing along the stream.** Speech is
   monotonic, and every source is internally monotonic — measured on
   `sub-1f4ea26f…task-Story-recall-(v2)`, zero backwards onsets within CrisperWhisper's 225 words
   or Qwen's 213.
4. **Where the sources disagree about a word's time, or where rule 3 has to move one, the word
   carries a correspondingly high temporal uncertainty.** The conflict is recorded. It is never
   resolved by discarding a source's reading, and never by rejecting the alignment.

**A timing conflict is not proof that the alignment is wrong.** The recognizer and its forced
aligner can each be wrong, and a wide disagreement is a statement about confidence in the time, not
about the sequence.

The case that settles it:

```
CW:   The@31.30   [UM]@31.52   the@35.30   little@35.36   boy@35.58
Qwen: the@31.20                            little@35.28   boy@35.60
```

Alignment pairs CW's `the`@35.30 with Qwen's `the`@31.20 — 4.1 s apart. It is tempting to call that
a mispairing, and it may be: Qwen has no token at all between 31.20 and 35.28, so its `the`@31.20
may correspond to CW's `The`@31.30, or to CW's `[UM]`@31.52, since a hesitation and a reduced
article are both schwa. But CW's `the`@35.30 is 60 ms long and butts exactly against `little`@35.36
— the signature of a forced aligner squeezing a token into a gap. The timestamp is at least as
suspect as the alignment.

So the two `the`s stay aligned in sequence, and the word carries a high temporal uncertainty. The
alignment is kept because the sequence evidence supports it; the 4.1 s spread is reported because
it is real and a downstream reader must be able to see it.
