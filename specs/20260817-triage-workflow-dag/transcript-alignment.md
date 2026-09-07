# Transcript alignment

The reasoning behind `_align_pair` and the slot lattice in
`src/senselab/audio/workflows/audio_analysis/harmonize.py`. The code states what the aligner is; this
file holds why, and the measurements.

`harmonize_transcripts` is the only caller of `_align_pair`; `aligned_columns` (`asr.py`) is the only
caller of `harmonize_transcripts` outside its tests, and it feeds `fuse_word_streams`, so this
alignment is what the triage `consensus_transcript` is fused over.

## Why an alignment path, not a distance

H3's purpose is *which* positions correspond. A distance says a model missed a word; an alignment
says which one and leaves the rest lined up. A time-based comparison cannot do this: a model that
inserts or drops one word shifts every timestamp after it, so one miss reads as a tail of
substitutions.

## Cost model: sclite weights, indel-preferring backtrace

Match 0, substitution 4, insertion or deletion 3 (the weights sclite uses for word alignment).
Backtrace preference at each cell, among candidates that reproduce the cell's cost: matching
diagonal, deletion from the reference, insertion from the model, mismatched diagonal.

### The defect this replaced

The aligner was unit-cost Levenshtein with a diagonal-first backtrace. Under unit costs two
substitutions (2) tie with delete + match + insert (2), and the diagonal won the tie, so a genuine
insertion beside any other difference was rendered as a run of substitutions. Measured on the
function before the change:

```
CW "I uh think"     Qwen "I think so"  ->  {I,I} {uh,think} {think,so}
CW "oh I uh think"  Qwen "I think so"  ->  {oh,-} {I,I} {uh,think} {think,so}
```

`uh` is an insertion by CrisperWhisper and `so` an insertion by Qwen, with `think` agreed; instead
the filler became a substitution and `think` was never recorded as agreed. CrisperWhisper is a
verbatim recognizer that emits fillers (`[UM]`, `[UH]`) and partial words (`a-`, `d-`, `ba-`) that
Qwen does not, so this is the common case in a CW/Qwen pair, not an edge case.

### Why these weights

With 4/3/3, `sub + sub` (8) is strictly worse than `del + match + ins` (6), so the DP itself no
longer produces the tie the backtrace was breaking wrongly, and a single substitution (4) stays
strictly cheaper than a deletion plus an insertion (6), so a one-for-one word difference is not
split into two single-source slots. Making the backtrace prefer indels over a mismatched diagonal
covers the ties that remain — a substitution that could sit at either of two positions, e.g.
CW "they are" against Qwen "they're" — and the rule the code follows is: take the diagonal only when
the tokens actually match; on a tie between a mismatched diagonal and an indel, take the indel.

The matching-diagonal-first order is what pins a repetition: "the the the" against "the" aligns the
**last** copy (`{the,-} {the,-} {the,the}`), and symmetrically in the other direction. That was the
behaviour before the change as well; it is now stated in the docstring and held by a test because
downstream repetition handling in the fused transcript depends on which copy carries two sources.

### Residual tie: which of two adjacent tokens is the substituted one

When one model inserts a token right beside a word the other model reads differently, the cost
model cannot say which of the two adjacent tokens is the substitution: for CW "I uh think" against
Qwen "I thing", `sub(uh, thing) + del(think)` and `del(uh) + sub(think, thing)` both cost 7. The
indel-first backtrace, walking from the end, resolves this toward the **earlier** token:
`{I,I} {uh,thing} {think,-}`. Held by `test_a_substitution_beside_an_insertion_lands_on_the_earlier_token`
so that a change to the rule shows up as a test change rather than a silent shift.

On the real pair this direction was right both times it fired — Qwen had merged two CW words
("they are" → "they're", "is, the" → "isthe") and pairing the merge with the first word is the
closer reading. It is the wrong direction for CrisperWhisper's filler-before-word pattern
(`[UM] the` against `a` would pair the filler with `a` and leave `the` alone), which did not occur
in this recording. Deciding it properly needs a graded substitution cost — how alike the two
tokens are — which `_normalise_token` does not provide and which has not been measured; until it
is, the earlier-token rule stands as documented behaviour, not as a claim of correctness.

Rejected: keeping unit costs and only changing the backtrace order. It gives the right answer on
the two cases above, but the DP still assigns the same cost to both readings, so the answer would
rest entirely on tie-breaking rather than on the cost model, and any later change to the backtrace
would silently move it. With the weights the DP and the backtrace agree.

### Measured effect on a real CW/Qwen pair

Story-recall recording `sub-1f4ea26f…_ses-D987B8B0…` (2026-09-04 run), 225 CrisperWhisper words
against 213 Qwen words, columns from `aligned_columns`:

| | columns | agreement | single-source | multi-key |
| --- | --- | --- | --- | --- |
| before | 226 | 209 | 14 | 3 |
| after | 226 | 209 | 14 | 3 |

The counts do not move on this recording because the bug needs both recognizers to insert around
one shared word, and here Qwen's only insertion ("But", column 164) sits between two agreed words.
Two columns change, both substitution-position ties that the indel-preferring backtrace now
resolves toward the earlier token: CW "they are" against Qwen "they're" was `{they} {are,they're}`
and is `{they,they're} {are}`; CW "is, the" against Qwen "isthe" was `{is,} {the,isthe}` and is
`{is,,isthe} {the}`.

The six adjacent duplicate pairs in the consensus text ("gets gets", "and and", "the the the"
counted twice, "The the", "he he") were already indels or agreements under the old aligner, with
the last copy of each CW repetition agreeing with Qwen's single word; the change leaves them where
they were:

| consensus | CW words | Qwen words | columns, before and after |
| --- | --- | --- | --- |
| gets gets | `gets a- gets` | `gets` | `{cw:gets} {cw:a-} {cw:gets,qw:gets}` |
| and and | `and the and` | `and` | `{cw:and} {cw:the} {cw:and,qw:and}` |
| the the the | `the … the d- the` | `the` | `{cw:the} {cw:d-} {cw:the,qw:the}` (first `the` single-source two columns earlier) |
| The the | `The [UM] the` | `the` | `{cw:The} {cw:[UM]} {cw:the,qw:the}` |
| he he | `he he` | `he he` | `{cw:he,qw:he} {cw:he,qw:he}` |

The consensus text orders them differently from the columns ("and the and the d- the" becomes
"and and the the the d-") because `fuse_word_streams` re-sorts slots by their averaged member times
after grouping; that is downstream of the aligner.

## The slot carries a word index, not an onset

`TranscriptSlot.indices` is `{model → index into that model's word list}`. A consumer rebuilding
richer word objects from the lattice cannot re-derive identity from `times`: forced aligners emit
words that share an onset and words of zero duration. Measured on the 5-speaker clip: a recognizer
placed "Josh" at `[2.72, 2.72]` and another placed two words at `2.72`. Matching by onset put one
word in two columns and dropped another, which is how "wanted to take" became "wanted take take".

## `times` is built from `members[m]`, not a loop variable

The `times` comprehension in `harmonize_transcripts` once read a bare `i` that resolved to whatever
the enclosing loop had left behind, so every model reported the last member's span. The lattice
still looked like a lattice while placing one word in two columns and losing another. The test
`test_each_model_reports_its_own_span_in_a_slot` holds this.
