# Consensus ASR redesign

Plan only. Nothing here is implemented. Line references are against `design/triage-workflow-dag`
at `bd71525e`, which already carries the aligner fix (`5aa4f607`, cost model in
[`transcript-alignment.md`](transcript-alignment.md)) and the owner's rulings R-1..R-5 in
[`consensus-asr-rulings.md`](consensus-asr-rulings.md). The rulings are binding; this plan folds
them in and does not reopen them.

Every worked example below was produced by running the code, not by reasoning: the real aligner
(`harmonize_transcripts`, `harmonize.py:485`) driven by a scratch prototype of the algorithm in
§3, over synthetic streams and over the reference store
`sub-1f4ea26f…_ses-D987B8B0…_task-Story-recall-(v2)_20260904-195726/run/store.jsonl`. The outputs
are pasted verbatim in §3.7 and §3.8.

**Revision after the second review round (2026-09-06).** Four substantive changes are folded in
below and marked where they land: the PII extent is the hull of the covered words' per-source
timings, never the derived extent (§3.6, §6.3); the time fit is a weighted isotonic **median**, not
least squares, and every measured number has been re-run under it (§3.6, §4.2); the figure's word
lane draws one fixed sub-band per source in each row (§5.2); and each source's own transcript is
scanned for PII alongside the consensus text (§6.3). The measurements in §3.6 and §3.8 are the
output of `senselab.audio.workflows.triage.consensus.align_sources` itself, over the same store.

## 1. What is wrong now, by line

| | where | what |
| --- | --- | --- |
| C-1 | `speech_to_text_ensemble/api.py:363` `slots.sort(key=lambda s: (s["start"], s["end"]))` — after the columns from the sequence alignment are grouped, the slots are **re-sorted by averaged member time**. | A correctly aligned transcript is reordered. Measured (§3.8): columns 42-47 are `and the and the d- the`; the emitted text is `and and the the the d-`, because col43 and col44 share a midpoint of 12.84 s and col47 (13.21 s) sorts ahead of col46 (13.36 s). Single-source insertions are shuffled past the agreed words they precede, so a real disfluency is displayed as a stutter the speaker did not produce. **The aligner is right; the duplicates are real.** |
| C-2 | `audio_analysis/asr.py:258` passes `columns=aligned_columns(streams)`; `:66-73` return `None` on `<2` streams, any exception, or an empty lattice; `api.py:322-362` then silently regroups by time overlap. `asr.py:265-266` record `slot_overlap`/`slot_mid_tol_s` on every run. | No artifact says which grouping ran. |
| C-3 | `preprocess.py:1264-1276` write one `word` per fused entry carrying `confidence`, `existence_confidence`, `temporal_confidence`, `coverage`, `recognizers`, `timing_sources` — and **not** `alternates`, `member_agreement`, `flags`, which `preprocess.py:1281` keeps only on the measurement's `words` list (`:1289`). | The disagreement *is* recorded (live proof, reference store: `'are'` [CW] with alternate `"they're"` share 0.5 [Qwen]; `'isthe'` [Qwen] with alternate `'the'` [CW]; `'a'` [Qwen] with alternate `'the'` [CW]) — but on the measurement, not the entity, so no reader of `word` sees it and the report renders a contested word exactly like an agreed one. |
| C-4 | `preprocess.py:1248-1263` routes any bracketed token to an `event` entity and `continue`s. No production code reads `event` (`grep '"event"' src/senselab` → `preprocess.py:1251` and the `PROV_TYPE` literal). Four test files seed or assert it: `preprocess_test.py:959,979`, `seed_preprocess_store_test.py:62,137`, `conftest.py:500-513`, `airway_test.py:186-190`. | Fillers vanish from the transcript; the events are write-only. |
| C-5 | `preprocess.py:1221-1232, 1233-1235, 1291` name `CRISPERWHISPER_ID`, `QWEN_ID` and `state["asr_*"]` explicitly. | A third source needs code in the consensus, not just a new block. |
| C-6 | `api.py:216-217` `winner_margin=0.66`, `alternate_min_share=0.15`; `asr.py:228-229` `0.3`/`0.15`; `api.py:418` breaks a two-way tie alphabetically. | Code-literal thresholds decide what is recorded. On the reference store no hypothesis word carries a score (0 of 438 have `score` other than `None`); `corroboration` is `None` on all 224 fused words and `member_corroboration` is a dict on every one of them whose values are all `None` (`{model: None, model: None}`), so the confidence-weighted machinery had nothing to weigh. |
| C-7 | `redact.py:223-232` `_consensus_words` sorts words **by extent**; `figure.py:469-483` and `report.py:196-209` read them in write/index order. | Even with C-1 fixed, the released transcript would be re-sorted by time on its own — the same defect, one node later. |

## 2. Design in one paragraph

The product is a **consensus text stream**: one linear sequence of positions, each carrying its
text, which sources produced it, each source's own reading and timing, and the onset/offset derived
from those (R-5). The sequence is the artifact; timing is metadata on its positions and never
orders it. PREPROCESS's consensus stops calling `fuse_consensus_words`. A new pure module,
`senselab.audio.workflows.triage.consensus`, discovers every ASR hypothesis in the store by role,
orders the sources by a stable rule that does not depend on write order, aligns them with
`harmonize_transcripts` (star alignment, already N-source), classifies each aligned column as
**agreement / variant / insertion**, and emits **one position per column, in column order,
verbatim** (R-3). Nothing collapses a repeated token; nothing re-sorts by time. Each source's own
timing is recorded verbatim on the position; the derived onset/offset is a monotonic fit over the
stream, and where the sources disagree about a word's time, or the fit had to move one, the
position carries a correspondingly high temporal uncertainty — the conflict is recorded, never
resolved by dropping a reading or rejecting the alignment (R-5, §3.6). Bracketed tokens are words: a bracketed reading that shares a key with a plain
one wins that column's surface (an override), and a bracketed token no other source produced is an
insertion owned by one source — two mechanisms, never conflated (R-1). Fewer than two hypotheses
raises (R-2). The consensus returns the stream and nothing else — no slots, no spans, no grouping
structure; every consumer that needs a grouping builds it from the positions, and doing so cannot
change the transcript (R-4). The `word` entities are how the stream's positions are stored; `index`
is the position and is authoritative for order. There is no `event` entity, no fallback grouping,
and no threshold anywhere in the path.

## 3. Data model and algorithm

### 3.1 `word` entity — one position of the stream (prov_type `word`, extent `(onset, offset)`)

One per aligned column; `index` is its position in the stream and **the only order any reader may
use**. `extent = (onset_s, offset_s)`, the derived monotone times of §3.6, is metadata on the
position, not a sort key. Every attribute is required.

| attribute | type | meaning |
| --- | --- | --- |
| `text` | `str` | The word's label surface (§3.4). Bracketed form when any member of the winning key is bracketed. For a variant, `variants[0].text`. |
| `bracketed` | `bool` | `text` starts with `[` and ends with `]`. |
| `outcome` | `"agreement" \| "variant" \| "insertion"` | §3.4. |
| `sources` | `list[str]` | The source names with a member in this column, in source order (§3.2). |
| `readings` | `dict[str, str]` | `source → surface as the recognizer produced it`. A source absent from the column is absent from the dict. For an onomatopoeic token the raw `khh` stays here while `text` is `[KHH]`. |
| `timings` | `dict[str, [float, float]]` | `source → [start, end]`, that source's own span, untouched (R-5 rule 2). Same keys as `readings`. |
| `onset_spread_s` | `float` | `max − min` of the members' own starts. `0.0` for one member. |
| `offset_spread_s` | `float` | `max − min` of the members' own ends. |
| `temporal_uncertainty_s` | `float` | The width of the smallest interval containing every member onset and the derived onset, or every member offset and the derived offset, whichever is wider (§3.6). `0.0` iff the sources gave identical times and the fit moved nothing. |
| `variants` | `list[{"text": str, "sources": list[str], "share": float}]` | Empty unless `outcome == "variant"`; then one entry per distinct normalised key, ordered by `(-len(sources), position of first source)`. `variants[0].text == text`. `share = len(sources) / n_sources`. |
| `agreement` | `float` | `|largest same-key group| / n_sources`. A count ratio, never thresholded. `1.0` iff `outcome == "agreement"`. |
| `index` | `int` | The position in the stream. Authoritative for order; `consensus_words` (§6.1) sorts by it and nothing else. |

A repetition is several adjacent `word`s. In the reference recording, CrisperWhisper's
`and the and the d- the` against Qwen's `and the` is six words: insertion, insertion, agreement,
insertion, insertion, agreement (§3.8). Nothing else marks it: the outcomes are the record.

A filler is a `word` with `bracketed = True`, `outcome = "insertion"`, one source. Nothing names
CrisperWhisper.

No per-position shift is stored. The fit's effect is recorded once, on the measurement
(`n_words_time_shifted`, `max_time_shift_s`, §4.2), and per word through `temporal_uncertainty_s`,
which includes the derived value; no consumer in §5 reads a signed shift, so the entity carries
none.

**Where the measurement's per-word disagreement content now lives.** The current
`consensus_transcript.words[*]` entries (keys measured on the reference store: `alternates`,
`confidence`, `corroboration`, `coverage`, `end`, `existence_confidence`, `flags`,
`member_agreement`, `member_confidence`, `member_corroboration`, `offset_confidence`,
`onset_confidence`, `sources`, `start`, `temporal_confidence`, `text`, `timing_sources`) are not a
duplicate of the entities — they carry disagreement the entities lack. Each field moves or retires
as follows; the measurement's `words` list is then removed because every field has a home:

| old `words[*]` key | new home |
| --- | --- |
| `text` | `word.text` |
| `sources` | `word.sources` |
| `start`, `end` | `word.extent` — **meaning changes** from the averaged member time to the monotonic fit of §3.6; the per-source spans stay verbatim in `word.timings` |
| `alternates[*].text`, `.models` | `word.variants[*].text`, `.sources` |
| `alternates[*].share`, `.share_uncorroborated` | `word.variants[*].share`, a count ratio. The two were equal on every observed word (0.5/0.5 on all three, above), because no member carried a confidence to weight |
| `flags: ["single_source"]` | `word.outcome == "insertion"`; N-generic: `len(word.sources) < n_sources` |
| `timing_sources` | `len(word.timings)` |
| `onset_confidence`, `offset_confidence` | replaced by `word.onset_spread_s`, `offset_spread_s` and `temporal_uncertainty_s` (§3.6), computed from `word.timings`, which stays recorded; the figure draws the disagreement itself (§5.2) |
| `confidence`, `existence_confidence`, `temporal_confidence`, `coverage` | retired: products of `share × coverage` and the code literals in C-6. `word.agreement` is the count ratio that remains |
| `member_agreement` | retired: a graded phoneme similarity from `text_similarity`; nothing in the triage path reads it, and on the reference run its grader recorded `grading_languages: []` |
| `member_confidence`, `member_corroboration`, `corroboration` | retired: `corroboration` is `None` on every observed word and `member_corroboration` is a dict on every word whose values are all `None` (no recognizer in the triage path emits a per-word score) |

Removed from `word`: `confidence`, `existence_confidence`, `temporal_confidence`, `coverage`,
`recognizers`, `timing_sources` (`preprocess.py:1268-1275`). Removed entirely: `event`
(`prov_store.py:20`, `preprocess.py:1250-1263`, `_as_non_word` at `:140-157`).

### 3.2 Source order

Sources are ordered **lexicographically by their `source` name** (`asr_crisperwhisper` before
`asr_qwen`). Not by store write order: `prov_store.py:5` promises merges are order-independent, and
`harmonize_transcripts` itself sorts its models by name (`harmonize.py:510`), so the alignment and
the record use one order. The rule is written into the measurement as
`source_order: "lexicographic_by_source_name"` (§4.2). Verified: the prototype fed the reference
store's two hypotheses in both orders and produced identical `(text, outcome, sources)` sequences
(§3.8, last line).

### 3.3 Algorithm — `consensus.align_sources(sources, *, onomatopoeic) -> Consensus`

Pure; takes `list[SourceHypothesis(name, words, timestamp_source, timestamp_model)]` in any order.

1. **R-2 guard.** `len(sources) < 2` → `LookupError("consensus needs at least two asr_hypothesis
   measurements; found {n}: {names}")`. Never proceeds over one source; one source corroborates
   nothing and must not read as agreement.
2. **Order** the sources per §3.2. `N = len(sources)`; `names` in that order.
3. **Members.** For each source token: `raw` = text as produced; `display` =
   `bracketed_form(raw, onomatopoeic)` when that returns a value (an onomatopoeic token becomes
   `[KHH]`; `_as_non_word`'s rule at `preprocess.py:150-157`, moved), else `raw`;
   `key = normalise_token(display)` (`harmonize.py:368-375`, promoted to public — brackets and
   punctuation drop, so `[UM]`, `um`, `Um.` share a key; measured: `[UM]`→`um`, `d-`→`d`,
   `is,`→`is`, `they're`→`they're`, `...`→`""`). A token with `key == ""` is dropped and counted in
   `empty_tokens_dropped[source]`. Every token that survives carries a span, because `_asr` drops
   untimed chunks before they reach the store (`preprocess.py:1190-1193`), so no column's
   `timings` is ever empty — the precondition the time fit in step 6 relies on.
4. **Alignment.** `harmonize_transcripts({name: [(start, end, display)]})` over sources with at
   least one member. Columns are its `slots` in order; each column's members come from
   `slot.indices` (word identity by index, `harmonize.py:396-405`, never by onset). A source with
   no members contributes no column and still counts in `N`. No source with members → no columns.
5. **Column → word** (§3.4). One word per column. **No collapse, no re-sort.**
6. **Time fit** (§3.6). Over the whole stream, in column order: the derived onsets are the
   isotonic median fit over every member onset, each reading stacked at its position; likewise the
   offsets. The spread and uncertainty fields are set from the members' own spans and the fit.
7. **Emission.** `index` = column position. `Consensus.words` and `Consensus.provenance` (§4.2
   minus what the node adds).

`align_sources` returns words only (R-4). It takes no parameter that serves a downstream grouping
preference: no overlap fraction, no midpoint tolerance, no span rule. `harmonize_transcripts`'s
`TranscriptSlot` objects do not escape the module.

### 3.4 Column classification

Group the column's members by `key`.

- one key, `len(members) == N` → **agreement**
- one key, `len(members) < N` → **insertion**
- two or more keys → **variant**

`surface(group)`: the first bracketed `display` in source order if any member of the group is
bracketed, else the first member's `display` in source order.

- one key: `text = surface(members)`. When the members mix a bracketed and a plain display this is
  the **bracket override** (R-1): `[COUGH]` over `cough`, one column, counted in
  `bracket_overrides_n`. The plain reading stays in `readings`.
- variant: groups ordered by `(-len(group), position of first source)`; `text = surface(groups[0])`;
  `variants = [{text: surface(g), sources, share: len(g)/N}]` for every group. No margin, no minimum
  share: every reading is recorded. A two-way tie resolves to the first source in source order,
  which is a label, not a verdict — `agreement = 0.5` and `variants` say so (open question 1).

`agreement = len(groups[0]) / N`. The members' own spans go to `timings` verbatim; `extent` comes
from the time fit over the whole stream (§3.6), not from this column alone.

R-1 in these terms: an override is an *agreement* (or, with `N > 2`, possibly an insertion of two
sources) whose surface is the more informative one; a bracketed *insertion* is a column with one
member. The first is bold in the display, the second is not (§5.1). They cannot be confused because
the outcome, not the brackets, decides the weight.

### 3.5 Rendering — `consensus.render_transcript(words, *, strong=("**", "**")) -> str`

Per word in `index` order: variant → `"/".join(v.text for v in variants)`; otherwise `text`. The
token is wrapped in `strong` iff `outcome == "agreement"`. With `strong=("", "")` this is the plain
text stored on the measurement — **a plain join of word texts**, which is what the PII scan reads
and what `_locate` matches back against (§6.3). The `a/b` form is display only and never enters the
measurement's `text`.

### 3.6 Time: verbatim per source, monotonic when derived, conflict recorded as uncertainty

R-5's four rules, in priority order: (1) the sequence alignment is authoritative for what matches
what and for the order of the stream; (2) each source's own timing is recorded verbatim, per
source, on the word — `timings`; (3) the derived `onset_s`/`offset_s` are monotonically
non-decreasing along the stream; (4) where the sources disagree about a word's time, or rule 3 had
to move one, the word carries a correspondingly high `temporal_uncertainty_s`. The conflict is
never resolved by discarding a source's reading and never by rejecting the alignment: a timing
conflict is a statement about confidence in the time, not about the sequence.

Rule 3 is grounded: each source is internally monotone. Measured on the reference store, zero
backwards onsets within CrisperWhisper's 225 words and zero within Qwen's 213. Taken across
sources the raw times are *not* — reading each position's onset as the **minimum** of its member
onsets, 7 of 226 positions have an earlier onset than the position before them (reading it as the
mean gives 3) — because adjacent positions take their times from different sources, which is
exactly the disagreement rule 4 records.

**The fit.** Over the stream in column order, for position `i` with member starts `S_i` and ends
`E_i` (never empty, §3.3 step 3):

- A position's **own** value is the median of its member readings. With an even count — `N = 2`
  is the norm — the median is any point between the two middle readings, and the rule adopted is
  the **midpoint** of that interval; for two sources this is the mean of the two readings. With
  one member it is that reading.
- `onset = isotonic_median(S)`, `offset = isotonic_median(E)` — the non-decreasing fit that
  minimises the summed absolute deviation over **every member reading, each stacked at its
  position** (so a position with two sources anchors twice as hard as one with one; this is what
  a weight of `|S_i|` per position means, said without a weight). It is computed by
  pool-adjacent-violators: adjacent positions whose own values decrease are pooled, and a pooled
  block's value is the median, under the same midpoint rule, of every reading in the block.
  Positions that do not violate the order are left exactly where they are, and the output is
  non-decreasing for any input — PAVA's invariant, held by construction, not by a check.
- **"The fit moved this position"** means the position is a member of a pooled block of size
  greater than one, for its onset or its offset. It does not mean a non-zero residual: a block of
  one position reproduces the position's own value exactly, so the definition by membership and
  the definition by residual agree, but only the first is stated.
- `onset_spread_s = max(S_i) − min(S_i)`, `offset_spread_s = max(E_i) − min(E_i)`;
- `temporal_uncertainty_s = max( max(S_i ∪ {onset_i}) − min(S_i ∪ {onset_i}),
  max(E_i ∪ {offset_i}) − min(E_i ∪ {offset_i}) )` — the width of every opinion about the onset,
  fitted one included, or about the offset, whichever is wider. It is `0.0` only when the sources
  gave identical times and nothing moved.

**Why the median and not the mean.** Under least squares one bad timestamp is spread across its
correct neighbours: with an adversarial single reading 15 s late at one two-source position, the
least-squares fit pooled nine consecutive positions into one extent and dragged eight correct
words with it (max shift 14.7 s). Under the median the same outlier costs linearly and pooling
stops at the offending position — measured on the reference store with Qwen's word 100 (stream
position 108, `yells`) moved 15 s late: positions 108-109 pool, 108's derived onset lands on 109's
own onset (0.325 s from its clean value), **no other position's derived time changes**, and the
measurement reports `n_words_time_shifted: 10` (8 clean + the pooled pair) and
`max_time_shift_s: 7.375` (the outlier position's own midpoint against the fit). A property check
over 20 000 random streams of 1-12 positions with 1-3 jittered readings each found zero cases of a
decreasing derived onset or offset and zero cases of `onset > offset`.

No threshold: the fit has no tolerance parameter, the uncertainty is a width in seconds, and
nothing is dropped. Measured with `align_sources` on the reference store (columns: each source's
own span; derived onset/offset; onset spread; uncertainty):

```
== gets case — CW gets@9.66, a-@9.88, gets@9.99 against Qwen gets@9.60 ==
 pos outcome   text       own timings                                     onset offset o_sprd  unc_s
  32 insertion 'gets'     crisperwhisper:[9.66,9.74]                         9.66   9.74   0.00   0.00
  33 insertion 'a-'       crisperwhisper:[9.88,9.99]                         9.88   9.99   0.00   0.00
  34 agreement 'gets'     crisperwhisper:[9.99,10.10]  qwen:[9.60,9.84]      9.88   9.99   0.39   0.39
  35 agreement 'out.'     crisperwhisper:[10.44,10.74]  qwen:[9.84,10.00]   10.14  10.37   0.60   0.74

== and the and the d- the ==
  42 insertion 'and'      crisperwhisper:[12.65,12.79]                      12.65  12.79   0.00   0.00
  43 insertion 'the'      crisperwhisper:[12.79,12.90]                      12.79  12.90   0.00   0.00
  44 agreement 'and'      crisperwhisper:[12.90,13.04]  qwen:[12.64,12.80]  12.79  12.92   0.26   0.26
  45 insertion 'the'      crisperwhisper:[13.20,13.29]                      13.20  13.29   0.00   0.00
  46 insertion 'd-'       crisperwhisper:[13.29,13.44]                      13.29  13.44   0.00   0.00
  47 agreement 'the'      crisperwhisper:[13.54,13.60]  qwen:[12.80,12.88]  13.29  13.44   0.74   0.74
  48 agreement 'boy'      crisperwhisper:[13.74,13.92]  qwen:[13.60,14.08]  13.67  14.00   0.14   0.16

== The [UM] the little boy — the case R-5 settles ==
  91 insertion 'The'      crisperwhisper:[31.30,31.52]                      31.30  31.52   0.00   0.00
  92 insertion '[UM]'     crisperwhisper:[31.52,31.70]                      31.52  31.70   0.00   0.00
  93 agreement 'the'      crisperwhisper:[35.30,35.36]  qwen:[31.20,31.84]  33.25  33.60   4.10   4.10
  94 agreement 'little'   crisperwhisper:[35.36,35.58]  qwen:[35.28,35.60]  35.32  35.59   0.08   0.08
  95 agreement 'boy'      crisperwhisper:[35.58,36.06]  qwen:[35.60,36.16]  35.59  36.11   0.02   0.10

derived onsets decreasing: 0; derived offsets decreasing: 0; offset<onset: 0
positions in a pooled block (onset or offset): 8 of 226 — onsets {33,34} {43,44} {46,47},
                                                         offsets {33,34} {46,47} {134,135}
positions whose fitted value differs from their own: 34 'gets' (onset +0.085, offset +0.020),
    44 'and' (onset +0.020), 47 'the' (onset +0.120, offset +0.200), 135 'loses' (offset +0.030)
max_time_shift_s = 0.200 (the offset of position 47)
highest temporal uncertainty: [(93, 'the', 4.1), (150, 'the', 1.28), (35, 'out.', 0.74), (47, 'the', 0.74),
                               (135, 'loses', 0.62), (219, 'he', 0.6), (136, 'balance', 0.42), (34, 'gets', 0.39)]
words with uncertainty 0.0 (identical times, no move): 17
```

Reading the three cases:

- **`gets`.** The agreed `gets` has member onsets 9.99 (CW) and 9.60 (Qwen), midpoint 9.795, which
  is earlier than the preceding `a-` at 9.88. The fit pools the two positions; the median of the
  three stacked readings `{9.60, 9.88, 9.99}` is 9.88, so `a-` keeps its own onset exactly and
  `gets` moves 85 ms later onto it. `gets` carries the 0.39 s spread of its sources. The order of
  the stream did not move.
- **`d-` / `the`.** Same mechanism: `the` (readings 13.54 and 12.80, midpoint 13.17) pools with
  `d-` at 13.29 and lands on `d-`'s own span, 13.29-13.44; `d-` is unmoved; `the` carries 0.74 s
  from Qwen having placed `and the` 0.7 s earlier than CrisperWhisper. This is the 0.200 s
  offset shift that `max_time_shift_s` reports.
- **`the`@35.30 / `the`@31.20.** The two `the`s stay aligned — the sequence evidence supports it and
  the 60 ms CW token butting against `little` is a forced aligner squeezing a token into a gap, so
  the timestamp is at least as suspect as the pairing. The midpoint 33.25 already sits between
  `[UM]` and `little`, so the fit moves nothing; the word carries `temporal_uncertainty_s = 4.10`,
  which is the fact a downstream reader must be able to see. Nothing was discarded and nothing
  rejected.

**The derived extent can sit where no source put the word.** It is a fit, not a reading: on the
reference store 7 of 226 positions have a derived extent that overlaps *neither* source's own
span — position 93 above (derived 33.25-33.60 against 31.20-31.84 and 35.30-35.36) and, less
dramatically, a word CW put at [44.34, 44.60] and Qwen at [44.96, 45.12], derived [44.65, 44.86].
Two consequences are accepted and one is not:

- The **PII/redaction extent** is never built from the derived extent. It is the **hull of the
  covered words' per-source `timings`** — the minimum member start to the maximum member end —
  so a name is cut wherever any recognizer heard it. Cutting on the derived extent would redact
  silence at 33.25 s while the spoken instance at 35.30 s (or 31.20 s) is released;
  over-redaction is the safe failure (§6.3).
- The **`asr` span source** in `_spans` reads the derived extents (§4.5), so such a word yields a
  span candidate that may sit in silence. `_novel` (`preprocess.py:640-656`) keeps a candidate
  with zero overlap against the other sources as a span of its own. This is accepted: the
  candidate is labelled `measure: asr`, the word it came from carries its `temporal_uncertainty_s`,
  and a span proposal is a proposal, not a verdict.
- The **figure** draws the sources' own spans and the derived extent side by side (§5.2), so the
  gap is visible rather than hidden.

Synthetic reproduction, `align_sources` output, for test 4b:

```
a: the[1.00,1.20] d-[1.20,1.35] boy[1.40,1.70]     b: the[1.00,1.18] boy[1.10,1.70]
 pos outcome   text   own timings                     onset offset o_sprd  unc_s
   0 agreement 'the'  a:[1.00,1.20]  b:[1.00,1.18]     1.00   1.19   0.00   0.02
   1 insertion 'd-'   a:[1.20,1.35]                    1.20   1.35   0.00   0.00
   2 agreement 'boy'  a:[1.40,1.70]  b:[1.10,1.70]     1.25   1.70   0.30   0.30
stream order: ['the', 'd-', 'boy']  derived onsets: [1.0, 1.2, 1.25]
```

**Tie-breaking, permitted and not adopted here.** Monotonicity may break ties among equal-cost
alignments — which copy of a repeated token matches — but may never reject an alignment. The
aligner's current tie-break is fixed and tested (matching-diagonal-first, so the **last** copy;
`transcript-alignment.md`, `5aa4f607`) and this change leaves it alone. The measured case for a
future, time-informed tie-break is `gets`: CW's first copy at 9.66 is nearer Qwen's 9.60 than the
last copy at 9.99, so a time-aware tie-break would emit `gets(agreed) a- gets(insertion)` and the
fit would then have nothing to move. That is a change to `_align_pair` with its own measurement,
not part of this plan (open question 6).

Order is by `index` (rule 1). Filtering or grouping words by time — which overlap a span, which
fall on a page — is legitimate; reading a time order back as the sequence is not, which is C-7.

### 3.7 Worked cases — prototype output, synthetic streams

Sources `a`, `b`(, `c`); tokens at 0.5 s spacing, 0.4 s long. The prototype was fed the sources in
**reverse** name order to exercise §3.2. Output verbatim (`agr` = `agreement`):

```
== repetition kept verbatim (finding): a='and the and the d- the' b='and the'
  [0] insertion text='and'      agr=0.50 sources=['a'] readings={'a': 'and'}
  [1] insertion text='the'      agr=0.50 sources=['a'] readings={'a': 'the'}
  [2] agreement text='and'      agr=1.00 sources=['a', 'b'] readings={'a': 'and', 'b': 'and'}
  [3] insertion text='the'      agr=0.50 sources=['a'] readings={'a': 'the'}
  [4] insertion text='d-'       agr=0.50 sources=['a'] readings={'a': 'd-'}
  [5] agreement text='the'      agr=1.00 sources=['a', 'b'] readings={'a': 'the', 'b': 'the'}
  render: and the **and** the d- **the**
  plain : and the and the d- the

== agreed repetition: a='that that' b='that that'
  [0] agreement text='that'  [1] agreement text='that'
  render: **that** **that**

== filler insertion: a='I [UM] think' b='I think'
  [0] agreement 'I'   [1] insertion '[UM]' sources=['a']   [2] agreement 'think'
  render: **I** [UM] **think**

== bracket override: a='[COUGH] hi' b='cough hi'
  [0] agreement text='[COUGH]' agr=1.00 sources=['a', 'b'] readings={'a': '[COUGH]', 'b': 'cough'}
  render: **[COUGH]** **hi**          prov bracket_overrides_n: 1

== partial word same key: a='a- cat' b='a cat'
  [0] agreement text='a-' agr=1.00 readings={'a': 'a-', 'b': 'a'}
  render: **a-** **cat**

== variant: a='hi Jon' b='hi John'
  [1] variant text='Jon' agr=0.50 readings={'a': 'Jon', 'b': 'John'} variants=[('Jon', ['a']), ('John', ['b'])]
  render: **hi** Jon/John       plain: hi Jon/John

== variant beside a repetition: a='the the' b='a'
  [0] variant text='the' variants=[('the', ['a']), ('a', ['b'])]   [1] insertion text='the' sources=['a']
  render: the/a the

== insertion / agreement / insertion: a='I uh think' b='I think so'
  render: **I** uh **think** so       outcomes {'agreement': 2, 'variant': 0, 'insertion': 2}

== empty key dropped: a='... hello' b='hello'
  [0] agreement 'hello'          prov empty_tokens_dropped: {'a': 1, 'b': 0}

== N=3 majority variant: a,b='cat' c='cot'
  [0] variant text='cat' agr=0.67 variants=[('cat', ['a', 'b']), ('cot', ['c'])]   render: cat/cot

== N=3 two-source insertion: a,b='x y' c='y'
  [0] insertion text='x' agr=0.67 sources=['a', 'b']   [1] agreement 'y' agr=1.00   render: x **y**

== N=3 three-way tie: a='cat' b='cot' c='cut'
  [0] variant text='cat' agr=0.33 variants=[('cat',['a']),('cot',['b']),('cut',['c'])]   render: cat/cot/cut

== onomatopoeic {'khh'}: a='hello khh world' b='hello khh world'
  [1] agreement text='[KHH]' readings={'a': 'khh', 'b': 'khh'}   render: **hello** **[KHH]** **world**
== onomatopoeic {'khh'}: a='hello khh world' b='hello world'
  [1] insertion text='[KHH]' sources=['a'] readings={'a': 'khh'}   render: **hello** [KHH] **world**

== one wordless source: a='hello' b=''
  [0] insertion 'hello' agr=0.50   prov n_sources: 2, reference_source: 'a'
== both wordless
  render: ''   prov n_words: 0, reference_source: None
== single source raises
  LookupError: consensus needs at least two asr_hypothesis measurements; found 1: ['a']
== zero sources raises
  LookupError: consensus needs at least two asr_hypothesis measurements; found 0: []
```

Note `plain: hi Jon/John` above is the *render* with empty marks, not the measurement text — the
measurement's `text` is the join of `word.text`, `hi Jon`. The prototype's `render(strong=("",""))`
kept the variant slash; the shipped `render_transcript(strong=("", ""))` must not (§3.5, test 15).

### 3.8 Worked cases — prototype output, reference store

225 CrisperWhisper words, 213 Qwen words, fed in reverse name order.

```
prov: {'algorithm': 'star_sequence_alignment', 'source_order': 'lexicographic_by_source_name',
       'sources': ['asr_crisperwhisper', 'asr_qwen'], 'n_sources': 2,
       'reference_source': 'asr_crisperwhisper', 'n_words': 226,
       'outcomes': {'agreement': 209, 'variant': 3, 'insertion': 14},
       'bracket_overrides_n': 0, 'empty_tokens_dropped': {'asr_crisperwhisper': 0, 'asr_qwen': 0}}

-- words 38..48 (12.0-14.0 s) --
  [38] agreement 'following' timings={'asr_crisperwhisper': [11.68, 11.94], 'asr_qwen': [11.68, 12.0]}
  [39] agreement 'morning'   timings={'asr_crisperwhisper': [12.04, 12.28], 'asr_qwen': [12.0, 12.32]}
  [40] agreement 'the'       timings={'asr_crisperwhisper': [12.28, 12.4],  'asr_qwen': [12.32, 12.4]}
  [41] agreement 'dog'       timings={'asr_crisperwhisper': [12.4, 12.65],  'asr_qwen': [12.4, 12.64]}
  [42] insertion 'and'       timings={'asr_crisperwhisper': [12.65, 12.79]}
  [43] insertion 'the'       timings={'asr_crisperwhisper': [12.79, 12.9]}
  [44] agreement 'and'       timings={'asr_crisperwhisper': [12.9, 13.04],  'asr_qwen': [12.64, 12.8]}
  [45] insertion 'the'       timings={'asr_crisperwhisper': [13.2, 13.29]}
  [46] insertion 'd-'        timings={'asr_crisperwhisper': [13.29, 13.44]}
  [47] agreement 'the'       timings={'asr_crisperwhisper': [13.54, 13.6],  'asr_qwen': [12.8, 12.88]}
  [48] agreement 'boy'       timings={'asr_crisperwhisper': [13.74, 13.92], 'asr_qwen': [13.6, 14.08]}

-- render 38..48 --
   **following** **morning** **the** **dog** and the **and** the d- **the** **boy**
-- plain 38..48 --
   following morning the dog and the and the d- the boy

-- every non-agreement word --
  [32]  insertion 'gets'   [33] insertion 'a-'
  [42]  insertion 'and'    [43] insertion 'the'   [45] insertion 'the'   [46] insertion 'd-'
  [91]  insertion 'The'    [92] insertion '[UM]'
  [133] insertion 'lost'   [134] insertion 'ba-'
  [149] insertion '[UM]'
  [164] insertion 'But'    readings={'asr_qwen': 'But'}
  [175] variant   'they'   readings={'asr_crisperwhisper': 'they', 'asr_qwen': "they're"}  variants=[('they', ['asr_crisperwhisper'], 0.5), ("they're", ['asr_qwen'], 0.5)]
  [176] insertion 'are'
  [208] variant   'is,'    readings={'asr_crisperwhisper': 'is,', 'asr_qwen': 'isthe'}
  [209] insertion 'the'
  [213] variant   'the'    readings={'asr_crisperwhisper': 'the', 'asr_qwen': 'a'}

-- words 89..93 --   **like** **that.** The [UM] **the**
-- words 173..177 -- **water,** **so** they/they're are **fine.**

order-independence: True
```

For contrast, the stored consensus (the current fold) over 12.0-14.0 s reads
`morning the dog and and the the the d- boy` with `'and' [12.77,12.92]` and `'the' [13.17,13.24]`
as averaged extents — the reordering C-1 describes. The three `alternates` the current
measurement carries (C-3) are exactly the three variants above, now on the entities.

## 4. Provenance

### 4.1 `measurement` `asr_<source>` — per-ASR

One per `_asr` block (`preprocess.py:1171-1217`). Attributes, all required:

| attribute | type | now |
| --- | --- | --- |
| `name` | `str` | unchanged (`write_measurement`, `common.py:110-134`) |
| `signal` | `"plain"` | unchanged |
| `role` | `"asr_hypothesis"` | **new** — what the consensus discovers by |
| `source` | `str` | **new** — `== name`; the key every consensus record uses and the sort key of §3.2 |
| `model_id` | `str` | rename of `recognizer` (`:1202`) |
| `commit_sha` | `str \| None` | **new** on the measurement (already on the agent) |
| `transcript` | `str` | unchanged (`:1203`) |
| `words` | `list[{text, start, end, score}]` | unchanged (`:1200`) |
| `n_words` | `int` | **new** |
| `untimed_chunks_n`, `out_of_bounds_chunks_n` | `int` | unchanged (`:1205-1206`) |
| `timestamp_source` | `"native" \| "bundled_aligner" \| "external_aligner"` | unchanged (`:1207`) |
| `timestamp_model` | `str \| None` | always present; `None` for native (`:1209-1210` writes it only when set) |
| `duration_s` | `float` | **new** — the bound `_bound_to_duration` applied |

Activity: step `name`, parameters `{model, **kwargs}` (`:1185-1187`), `used` the plain stream.
Agent: `model_id`, `commit_sha` (`:1184`). Unchanged.

### 4.2 `measurement` `consensus_transcript` — per-consensus

Flat; the nested `provenance` dict (`:1290`) goes. Attributes, all required:

| attribute | type |
| --- | --- |
| `name`, `signal` | as today |
| `role` | `"consensus"` |
| `algorithm` | `"star_sequence_alignment"` — the only value; there is no other path |
| `routine` | `"senselab.audio.workflows.triage.consensus.align_sources"` |
| `normalisation` | `"casefold; keep alphanumerics and apostrophe"` (`harmonize.normalise_token`) |
| `source_order` | `"lexicographic_by_source_name"` |
| `sources` | `list[{name, model_id, commit_sha, measurement_id, agent_id, n_words, timestamp_source, timestamp_model}]` in source order |
| `n_sources` | `int` — the denominator of every `agreement` and `share` |
| `reference_source` | `str \| None` — `TranscriptHarmonization.reference` (`harmonize.py:519`, median token count; for `N = 2` the longer stream). Which sequence anchored the star; it changes no word |
| `n_words` | `int` — `word` entities written; equals the column count |
| `outcomes` | `{"agreement": int, "variant": int, "insertion": int}` |
| `bracket_overrides_n` | `int` — columns where a bracketed and a plain reading shared a key |
| `empty_tokens_dropped` | `dict[str, int]` — per source, tokens normalising to `""` |
| `time_fit` | `"weighted_isotonic_median"` — the only value (§3.6) |
| `n_words_time_shifted` | `int` — positions that are members of a pooled block of size > 1, for their onset or their offset (8 of 226 on the reference store) |
| `max_time_shift_s` | `float` — the largest absolute difference between a position's fitted value and its own median, **over onsets and offsets alike** (0.200 s there: the offset of position 47, `the`) |
| `word_ids` | `list[str]` — in stream (`index`) order |
| `text` | `str` — `render_transcript(words, strong=("", ""))`, the plain join of `word.text` |

Removed: `words` (relocated per §3.1), `provenance.*` (`slot_overlap`, `slot_mid_tol_s`,
`grading_languages`, `operator`, `timing_sources`), `systems`, `timing_authority`, `event_ids`.

Activity: step `"consensus"`, parameters `{"routine": ..., "source_order": ..., "sources": [names]}`,
`used` every `asr_hypothesis` measurement it read. The measurement `was_derived_from` every source
measurement. **Each `word` `was_derived_from` only the source measurements named in its
`sources`** — an insertion by CrisperWhisper derives from `asr_crisperwhisper` alone, not from both.

### 4.3 Failure, per R-2 and the block loop

`preprocess.py:1519-1522` absorbs only `ValueError`/`LookupError` into `absent[name]`; any other
exception also lands in `hard_failures` and `:1527-1529` raises `RuntimeError`, failing the node.

- Fewer than two live `asr_hypothesis` measurements → `LookupError` (§3.3 step 1). This is a
  cascading absence — the recognizer block that did not write already recorded its own failure — so
  it lands in `absent["consensus_transcript"]` with `describe_exception`'s text, appears in the
  PREPROCESS verdict's `absent`, in the report's `preprocess_absences`, and SPEECH then raises its
  own `LookupError("no consensus_transcript in the store")` (`speech.py:463-464`). No consensus
  measurement and no `word` is written, so nothing presents one source as agreement. This is the
  guarantee the retired `LookupError("both recognizers are needed")` gave, in N-generic wording.
  When the missing recognizer failed *hard*, the node's `RuntimeError` fires regardless.

  **The run-level outcome, stated.** PREPROCESS's own verdict is unconditionally `PASS` with the
  absence listed (`preprocess.py:1535-1540`); the failure R-2 requires becomes visible one node
  later, when SPEECH raises (`speech.py:463-464`) and `run.py:185-186` records SPEECH as `ERRORED`
  with the exception's text. That *is* the visible failure: the run does not complete quietly, and
  no artifact reads like a successful two-source consensus.
- Any other exception from `align_sources` is a defect in the consensus, not an absence: it is a
  hard failure and the node raises. Test 20 picks `LookupError` and test 21 picks `RuntimeError`
  deliberately.
- Two sources both wordless: two hypotheses exist, so the guard passes; the measurement is written
  with `n_words: 0`, no `word`, `reference_source: None` (keeps
  `test_a_wordless_run_still_writes_the_consensus_with_a_filled_provenance`, adjusted). One source
  wordless: every word is an insertion by the other (§3.7).

### 4.4 Discovery in PREPROCESS

`_consensus` (`preprocess.py:1219-1303`) becomes:

1. `hypotheses = [m for m in live_entities(store, "measurement") if m.attributes.get("role") == "asr_hypothesis"]`;
   where one `source` appears more than once the latest write wins (the store's shared read rule,
   as `find_measurement`).
2. For each: `SourceHypothesis(name=attributes["source"], words=attributes["words"], ...)`; the
   provenance row takes `model_id`, `commit_sha` from the attributes, `measurement_id = m.id`,
   `agent_id` from `store.generated_by` → `store.associated_with`.
3. `align_sources(...)`; write one `word` per result (`was_generated_by` the consensus activity,
   `was_attributed_to` software, `was_derived_from` its own sources' measurements), then the
   measurement. `state["consensus"]` becomes the in-memory `Consensus.words` (for `_spans`, §4.5).

`state["asr_*"]` handoff (`:1216-1217`), the `LookupError("both recognizers are needed")`
(`:1221-1222`), and the `span is None: continue` at `:1244-1245` go — every source word was already
bounded in `_asr`, so a fused word cannot fall outside the decode. Adding a source is one more
`_asr(...)` entry in `blocks` (`:1494-1498`).

### 4.5 Downstream consumers, per R-4 and R-5

The consensus emits the stream. Everything that needs a grouping builds it from the positions, and
none of these can change the transcript:

- **`_spans`' `asr` source** (`preprocess.py:684-690`) reads the lexical words' extents and groups
  them with `group_extents_into_runs` into span candidates. Unchanged in kind: today it reads
  `state["consensus"]`, which already excludes the bracketed tokens (they were `continue`d before
  `kept.append`); it now reads the non-bracketed words of `Consensus.words`. It stays in `_spans`;
  it is not moved into the consensus.
- **SPEECH's speech spans** (`speech.py:512-513`) group lexical word extents the same way (§6.3).
- **The figure's word lane** (§5.2) and **the report's token panel** (§5.1) each place words on the
  time axis from `extent` and `timings`.

A consumer wanting a different grouping — a wider gap, a per-speaker run — computes it itself from
the words. The stream, its order and its texts do not depend on any consumer.

**Audit: every current consumer that sorts or filters words by time.** The order is `index` (R-5
rule 1). The derived times are monotone (rule 3), so no consumer has to tolerate a decrease — but
time is still metadata, and a consumer that sorts by it is reading metadata as sequence (C-7).
Filtering and grouping by time are legitimate.

| consumer | today | verdict | must do |
| --- | --- | --- | --- |
| `redact.py:223-232` `_consensus_words` | `sorted(..., key=extent)` | **sorts — wrong** | `common.consensus_words` (by `index`). Today this and `report.py:196-209` disagree; the index one is right |
| `redact.py:292-321` `_transcript` | iterates `words` as given; docstring says "in time order" | logic right, docstring wrong | docstring → "in stream order". A placeholder is emitted at the first overlapping position and later overlapping positions are dropped — correct under a non-monotone stream |
| `report.py:196-209` `_words` | `sorted(..., key=(index, extent))` | right; the extent tie-break is dead (`index` is unique) | `common.consensus_words` |
| `report.py:325-368` `_token_lane` | row = position mod 3, in entry order | right | unchanged |
| `plotting.py:1142-1200`, `_StaggeredTokenLane` (`:286`), `_token_label_slots` (`:116-143`) | slot fitting sorts its own copy of the centres (`:143`) and returns slots in the given order | right — layout, not order | unchanged |
| `figure.py:469-483` `_words` | `live_entities` write order, no sort | right by coincidence | `common.consensus_words` |
| `figure.py:1226-1274` `_asr_lane_panel` | filters to the page window by the derived extent (`:1250`); row = **page-local** index mod `asr_rows` (`enumerate(here)`, `:1251`), not stream position | filter right in kind, wrong in what it filters on; the row rule is page-local today | filter on the hull of `timings ∪ extent`, so a source rectangle on the page is drawn even when the derived extent is off-page; row by stream `index` mod `asr_rows` — **a change**, not a no-op (§5.2) |
| `speech.py:512-513` + `spans/api.py:223-248` `group_extents_into_runs` | sorts a private copy by start and returns **input positions** as members | right — grouping, and members map back to positions | unchanged |
| `speech.py:717-729` `word_speakers` | overlap of each word with speaker segments | right (filter) | unchanged |
| `speech.py:867-868` PII extent | `(words[first].extent[0], words[last].extent[1])` — the **derived** extent | **wrong** — the derived extent can overlap neither source's reading (§3.6), so a name is cut from silence while the spoken instance is released | the hull of the covered words' `timings`: min member start, max member end (§6.3) |
| `taxonomy.py:164`, `airway.py:63-79, 321-325` | overlap tests | right (filter) | read `lexical_words`; no ordering involved |
| `preprocess.py:684-690` `_spans` asr source | `group_extents_into_runs` | right | unchanged in kind |

## 5. Display

### 5.1 Report (`report.py`) and the tokens renderer (`plotting.py`)

- `transcript.text` — the measurement's `text`: the plain join (unchanged meaning).
- `transcript.marked_text` — **new**, `render_transcript(words)`: `**word**` for agreement, bare
  for insertion, `a/b` for a variant, brackets drawn as-is. Both text blocks on the summary page
  (`report.py:1633-1637`, `:1726-1728`) print `marked_text`; the monospaced page cannot bold, and
  `**` is the plain-text mark for it.
- `transcript.tokens_n` — rename of `words_n` (`:1237`): every consensus word, bracketed included
  (the page already says "transcript tokens", `:1614`). SPEECH's `words_n` is lexical (§6.3); the
  two names now differ because the counts do.
- `evidence.consensus_transcript_tokens[*]` (`:1179-1191`) carries: `entity_id`, `text`,
  `bracketed`, `outcome`, `sources`, `readings`, `timings`, `variants`, `agreement`, `timing`
  (the derived extent), `onset_spread_s`, `offset_spread_s`, `temporal_uncertainty_s`,
  `provenance`. Drops `confidence`, `existence_confidence`, `temporal_confidence`, `coverage`,
  `recognizers`, `timing_sources`, `timing_authority`.
- Token lane (`_token_lane`, `:325-368`): entry text = `a/b` for a variant, else `text`; fill =
  `_consensus_word_color(agreement)` (`:370-382`, the argument is now the count ratio); **new**
  token key `"bold": outcome == "agreement"`. The redacted lane keeps its shape; a placeholder is
  never bold.
- **Renderer.** `report.py` only emits the tokens-panel dict. The renderer is
  `src/senselab/audio/tasks/plotting/plotting.py:1142` (`elif ptype == "tokens":`), which builds
  each label at `:1168-1181` with `color="black"` and no `fontweight`. It gains
  `fontweight="bold" if token.get("bold") else "normal"` in the `_FittedTokenLabel(...)` call;
  `_FittedTokenLabel.__init__` forwards `**kwargs` to `matplotlib.text.Text` (`:198, :211`), so
  nothing else changes. Without this file the bold requirement cannot be met.
- `REPORT_SCHEMA_VERSION` (`report.py:37`) → `"triage-summary/v3"`.

Weight, in one table:

| word | text | weight | fill |
| --- | --- | --- | --- |
| agreement | `text` (bracketed if an override) | **bold** | `_consensus_word_color(1.0)` |
| insertion | `text` (bracketed if a filler) | normal | `_consensus_word_color(len(sources)/N)` |
| variant | `a/b` in the order of `variants` | normal | `_consensus_word_color(agreement)` |
| redacted placeholder | `[CATEGORY]` | normal | unchanged |

### 5.2 Temporal lane (figure, `_asr_lane_panel`, `figure.py:1226-1274`)

Per word, one staggered row (`asr_rows`, `asr_row_height`). Instead of one `word_fill` rectangle
over `extent`:

- **One fixed sub-band per source within each row.** A row of height `asr_row_height` is divided
  into `N` equal bands, one per source in the measurement's source order; source `k` always draws
  in band `k`, on every row, on every page. For each source `s` in the word's `timings`: a
  `Rectangle` at that source's own `[start, end]`, in `s`'s band, `facecolor =
  word_source_colours[k]`, `alpha = word_span_alpha`, no edge. Agreement reads as two bars stacked
  at the same time; disjoint readings — word 47 above, 0.7 s apart — read as two bars in two bands
  at two times, on one row. A single-source word draws one bar, in its source's band, and the
  other band stays empty. Nothing compounds: two rectangles never share a band, so alpha never
  adds up, and a far-flung source span cannot be mistaken for the reading of an unrelated word.
  (The rejected form — full-row fills that overlap and read darker on agreement — failed on the
  case it exists for: with rows shared by every fourth word, a far-flung span alpha-compounded
  with an unrelated word on the same row and looked identical to agreement.)
- one thin outline `Rectangle` over the derived `extent` (`onset_s`, `offset_s`), the full row
  height, no fill, `edgecolor = word_text_colour`, `linewidth = word_extent_linewidth`: the
  monotone time the consumers use, drawn over the raw readings it was fitted from.
- text at the derived `onset_s`: variant → `a/b`; else `text`; `fontweight = "bold"` iff
  `outcome == "agreement"`, else `"normal"`. Brackets drawn as-is.
- panel title: `"consensus ASR — " + ", ".join(f"{name}: {colour}")` in source order, so the page
  carries its own legend; no y-tick labels.
- **The page filter** keeps a word when the hull of its `timings` and its `extent` meets the
  window, so a source rectangle that falls on this page is drawn even when the derived extent is
  off-page (today `figure.py:1250` filters on the derived extent alone).
- **The row is the stream position**: `index mod asr_rows`. Today's row is the page-local
  enumeration (`enumerate(here)`, `figure.py:1251`), which restarts on every page; moving to the
  stream `index` is a change of behaviour, not a restatement — a word's row is then the same on
  every page it touches, and a page whose first word is at stream position 5 starts on row 1.
- The per-source fills are the sources' **verbatim** readings (R-5 rule 2), so they may sit
  anywhere the source put them: Qwen's `the`@31.20 is drawn 4.1 s left of CrisperWhisper's
  `the`@35.30 in the other band of the same row, and the outline between them is the fitted word
  with its `temporal_uncertainty_s = 4.10` visible as that gap. Each rectangle is placed from its
  own numbers and nothing is positioned relative to a neighbour.

`_words(store)` (`figure.py:469-483`) returns `index, extent, text, outcome, variants, readings,
timings, sources, bracketed` per word via `common.consensus_words`, plus the source order from the
`consensus_transcript` measurement. `FigureStyle`: remove `word_fill` (`:170`) and `asr_fontweight`
(`:176`); add `word_source_colours: tuple[str, ...]` (a cycle, at least 2 entries) and
`word_span_alpha: float`, plus `word_extent_linewidth: float`. All are drawing choices, not pipeline keys
(`test_the_style_shares_no_field_name_with_a_pipeline_key`, `figure_test.py:271`).

## 6. Readers

### 6.1 `common.py`

- `consensus_words(store) -> list[Entity]` — live `word` entities sorted by `index`. The one order.
- `lexical_words(store) -> list[Entity]` — those with `bracketed == False`.

### 6.2 TAXONOMY, AIRWAY, REDACT

- `taxonomy.py:137` (`n_words`, `element_ids`) and `:156` (`_transcribed_span_ids`) read
  `lexical_words`: a `[COUGH]` word is not lexical evidence and marks no span transcribed
  (`dag.md:299`'s rule, unchanged: *consensus words that are not all bracketed ⇒ speech present*).
- `airway.py:63-79` `_is_transcribed` and `:321-325` lexical contamination read `lexical_words`.
- `redact.py:223-232` `_consensus_words` delegates to `common.consensus_words` — **by index, not by
  extent** (C-7, R-5); `_transcript`'s docstring (`:301`) follows. Bracketed words pass into the
  released text unchanged; a variant contributes its `text`, never `a/b`.

### 6.3 SPEECH — lexical filter everywhere but the PII haystack

`speech.py` reads words at these sites. Every read uses the lexical subset **except the PII
haystack**, which must stay index-aligned with the measurement's `text` (the join over *all*
words) so a finding located in the scanned string maps back to the right entity.

| line | today | now |
| --- | --- | --- |
| `:3-4` | module docstring names `fuse_consensus_words` | names the consensus stream PREPROCESS writes with `triage.consensus.align_sources` |
| `:465-466` | `words` from `word_ids` | `words` (all, column order — the haystack) and `lexical = [w for w in words if not w.attributes["bracketed"]]`, with `lexical_index` mapping lexical positions to `words` positions |
| `:482` | `if not words:` → FAIL "no consensus word" | `if not lexical:` — a recording of only `[COUGH]` words has no subject |
| `:507` | single-recognizer flag over `words`, key `recognizers` | over `lexical`, `outcome == "insertion"` — N-generic; `len(sources) == 1` would miss a two-of-three insertion |
| `:512-513` | `word_extents` from `words` | from `lexical`; `grouped` members index `lexical` |
| `:717-729` | `word_speakers` over `words` | unchanged — over all words; a filler has a speaker and its `attribute` assertion is still written |
| `:817` | `owners = {word_speakers[i] for i in members}` | `word_speakers[lexical_index[i]]` |
| `:824` | span `words_n: len(members)` | unchanged (members are lexical) |
| `:844` | `scan_for_pii([transcript_text])` | **one call over several haystacks**: the consensus `text` (the plain join over all words) **and each source's own `transcript`** (already on its `asr_*` measurement, `preprocess.py:1203`), in source order |
| `:855` | `_locate(finding, words)` | per haystack: the consensus haystack locates against `word.text` over all words; a source's haystack locates against `readings[source]` over the words that source has a reading for, and maps the run back to stream positions. `_norm_token` (`:181-190`) strips `[]`, so `[UM]` normalises to `um` and **would** match a finding `um`; positions line up with the scanned string either way |
| `:867-868` | PII extent `(words[first].extent[0], words[last].extent[1])` — the derived extent | the **hull of the covered words' `timings`**: `min` of every member start, `max` of every member end, over `covered`. Over-redaction is the safe failure; the derived extent can sit in silence (§3.6) |
| `:871-872`, `:882` | `recognizers` on the `pii` entity | `sources`; plus `haystack` — `"consensus"` or the source name — so the record says which text the finding came from |
| `:896`, `:901` | marks by index into `words` | unchanged |
| `:1013` | verdict `words_n: len(words)` | `len(lexical)` |

So `[COUGH]`/`[UM]` do not clear the "no consensus word" guard, do not inflate `words_n`, do not
extend a speech span, and remain visible to the redaction scan.

**PII in a reading the consensus text does not show.** With two sources a variant's `text` is
always the lexicographically first source's reading, so a name only the other recognizer heard —
`John` where the consensus shows `Jon`, or a name Qwen alone recognised as an insertion — would
never enter the scanned string, and the audio would be released with it. That is a redaction hole,
not an open question, and it is closed by scanning each source's own transcript as a haystack of
its own (the `:844` row). A finding raised on a source's transcript is located against that
source's `readings` in stream order — a source's readings appear in the stream in the source's own
order, because the alignment preserves each sequence — and marks the words it lands on exactly as
a consensus finding does. Two haystacks can raise the same finding at the same positions; a
`(category, first, last)` already recorded is not recorded twice, so `pii.n` counts occurrences,
not haystacks. A finding located in no haystack keeps today's behaviour: `pii_unlocated` is
flagged and the whole speech extent is marked.

## 7. Configuration

No new key. No threshold exists in the path: alignment is exact on normalised keys, outcomes are
set relations, `agreement` and `share` are ratios. `words.onomatopoeic_tokens` (null) keeps its
meaning with one change of target: an onomatopoeic token becomes a **bracketed word** rather than
an event. `config-derivations.md:535-539` is reworded accordingly; nothing is added, so no
derivation is owed.

## 8. File-by-file change list

**New**

- `src/senselab/audio/workflows/triage/consensus.py` — `SourceHypothesis`, `ConsensusWord`,
  `Consensus` (dataclasses; words only, no slot or span type); `bracketed_form(text, onomatopoeic)
  -> str | None` (from `_as_non_word`); `is_bracketed(text) -> bool`; `align_sources(sources, *,
  onomatopoeic) -> Consensus`; `render_transcript(words, *, strong) -> str`. Stdlib + `harmonize`
  only.
- `src/tests/audio/workflows/triage/consensus_test.py`.

**Changed**

- `audio_analysis/harmonize.py` — `_normalise_token` → `normalise_token` (public; `:368`, the call
  site at `:515`, `__all__` at `:34`). Its docstring (`:369-373`) is rationale — why casing and
  punctuation are not transcription disputes — and moves to `transcript-alignment.md`; the
  docstring that stays says what the key is.
- `triage/nodes/preprocess.py` — `_asr` attributes per §4.1 (`:1201-1210`); `_consensus` per §4.4;
  `_spans`' asr source reads the lexical words of `Consensus.words` (`:684-690`, §4.5); delete
  `_as_non_word` (`:140-157`) and the `fuse_consensus_words` import (`:68`); keep `_norm_token`
  (`:135`) only if still read, else delete.
- `triage/nodes/common.py` — add `consensus_words`, `lexical_words` (§6.1).
- `triage/nodes/taxonomy.py` — `:137`, `:156` per §6.2.
- `triage/nodes/airway.py` — `:63-79`, `:321-325` per §6.2.
- `triage/nodes/speech.py` — every row of the §6.3 table, the module docstring included.
- `triage/nodes/redact.py` — `_consensus_words` (`:223-232`) per §6.2; `_transcript` docstring
  (`:301`) says stream order, not time order.
- `triage/nodes/report.py` — `_words` (`:196-209`) delegates; `_token_record` (`:1179-1191`),
  `_token_lane` (`:325-368`), `_consensus_word_color` (`:370-382`), `transcript` (`:1234-1238`),
  the two page text blocks (`:1633-1637`, `:1726-1728`), `REPORT_SCHEMA_VERSION` (`:37`) per §5.1.
- **`src/senselab/audio/tasks/plotting/plotting.py`** — the tokens-panel label at `:1168-1181`
  honours `token["bold"]` (§5.1).
- `triage/nodes/figure.py` — `_words` (`:469-483`), `_asr_lane_panel` (`:1226-1274`), `FigureStyle`
  fields (`:170`, `:176`; docstring `:109-110`, `:145-146`) per §5.2.
- `utils/prov_store.py` — remove `"event"` from `PROV_TYPE` (`:20`).

**Not in this change** — `audio_analysis/asr.py` C-2 (the other workflow's fallback provenance) is
separable and is not touched here; the triage path no longer calls it. Open question 3.

**Tests changed**

- `src/tests/audio/workflows/triage/nodes/conftest.py` — `seed_preprocess_store` (`:379-400`): drop
  the `events` argument and its block (`:500-513`); `words` accepts `str`, `(text, (start, end))`,
  and a full `{text, outcome, sources, readings, timings, variants, bracketed}` form; the seeded
  entities (`:519-529`) and measurement (`:535-556`) take the §3.1/§4.2 shapes.
- `preprocess_test.py` — `TestTheConsensusTranscript` (`:829-937`), `TestWordsAreBracketAware`
  (`:939-997`) rewritten per §9.
- `seed_preprocess_store_test.py:62,137` — `event` assertions go; `[COUGH]` seeds a bracketed word.
- `airway_test.py:186-190` — the `events` seeding becomes bracketed words; the tests at §9.26 hold
  the same behaviour through `lexical_words`.
- `speech_test.py` — `recognizers` → `sources` (`:1160-1169`); `test_an_event_is_not_a_word`
  (`:666`) becomes "a bracketed word is not a lexical word"; new tests §9.27.
- `report_test.py` — `TestTheWordsLaneFollowsTheConsensusStyle` (`:1756-1897`) gains the `bold`
  and `a/b` assertions; `TestConsensusWordColorIsRobust` (`:596`) takes `agreement`; schema v3.
- `figure_test.py` — the lane test at §9.30.
- `src/tests/audio/tasks/plotting_test.py` — §9.28.

**Specs updated in the same change**: `preprocess.md` (table rows `:72-74`; §"Consensus timing
authority" `:82-88`; §`consensus_transcript` `:172-182`; §"Words are bracket-aware" `:184-197`;
open derivations `:218`), `taxonomy.md:31,54`, `branch-airway.md:22,39-41,71`,
`branch-speech.md:20-21,66-77`, `dag.md:291-306` (the rule stays; the `:403` reference into the
ensemble tally goes), `config-derivations.md:535-539`, `report.md:46,49,89-91`,
`transcript-alignment.md` ("Measured effect" — the sentence that the fused transcript's repetition
handling depends on which copy carries two sources no longer describes the triage path; the
re-sort paragraph becomes past tense). `store.md` needs no change: its entity list at `:22` never
named `event`.

## 9. Test plan

`consensus_test.py` (pure, no store, sources named `a`, `b`, `c` — nothing names a model; each
case's expected output is the §3.7 run):

1. Agreement: two identical streams → every word `outcome == "agreement"`, `agreement == 1.0`,
   `readings` and `timings` carry each source's own values verbatim (different `start`/`end` per
   source survive unrounded; `extent` equals the shared times, and every spread and
   `temporal_uncertainty_s` is `0.0`).
2. Variant: `hi Jon` vs `hi John` → `Jon` is `variant`, `variants` has both with their sources and
   `share == 0.5`, `text == variants[0].text`, `agreement == 0.5`.
3. Insertion / agreement / insertion: `I uh think` vs `I think so` → four words, `uh` and `so`
   insertions of one source each, `think` agreement once.
4. **Regression, the finding.** (a) `a: and the and the d- the`, `b: and the` → **six** words,
   outcomes `[insertion, insertion, agreement, insertion, insertion, agreement]`,
   `render_transcript` → `and the **and** the d- **the**`, plain text `and the and the d- the`.
   (b) **Conflicting times, stream order kept, derived times monotone** (R-5): `a: the[1.00,1.20]
   d-[1.20,1.35] boy[1.40,1.70]`, `b: the[1.00,1.18] boy[1.10,1.70]` — `b` opens `boy` before `a`'s
   stumble. Assert the stream is `["the", "d-", "boy"]`, derived onsets `[1.0, 1.2, 1.25]`
   (non-decreasing), `boy.onset_spread_s == 0.30 == boy.temporal_uncertainty_s`, `n_words_time_shifted == 0`,
   and `timings` still hold `b`'s `[1.10, 1.70]` verbatim. The §3.6 run is the expected output.
   (c) **The `gets` case**: `a: gets[9.66,9.74] a-[9.88,9.99] gets[9.99,10.10]`,
   `b: gets[9.60,9.84]` → `a-` and the agreed `gets` are pooled; the block's median of
   `{9.60, 9.88, 9.99}` is `9.88`, so `a-.extent == (9.88, 9.99)` exactly (its own span) and
   `gets.extent == (9.88, 9.99)`; `gets.temporal_uncertainty_s ≈ 0.39`; the provenance counts
   `n_words_time_shifted == 2` and `max_time_shift_s ≈ 0.085`; the stream is `gets a- gets` with
   outcomes `[insertion, insertion, agreement]`, unchanged by the fit.
   (d) **The 4.1 s conflict**: `a: The[31.30] [UM][31.52] the[35.30,35.36] little[35.36] boy[35.58]`,
   `b: the[31.20,31.84] little[35.28] boy[35.60]` → `the` is one `agreement` word (not rejected),
   `onset_s == 33.25`, `n_words_time_shifted == 0`, `temporal_uncertainty_s == 4.10`; `[UM]` and
   `The` are insertions with uncertainty `0.0`.
   (e) Property, over every case above and a 200-word random pair with jittered times: derived
   onsets and offsets are non-decreasing along `index`, `onset_s ≤ offset_s` at every position,
   and `temporal_uncertainty_s ≥ max(onset_spread_s, offset_spread_s)`.
   (f) **The median tie rule.** Two sources reading `[1.00, 1.20]` and `[1.10, 1.30]` for one word
   with nothing to pool → `extent == (1.05, 1.25)`, the midpoint of each pair. Three sources with
   one pooled neighbour → the odd-count median is a member reading, not a mean:
   `isotonic_median_fit([[1.0, 3.0], [1.5]])` is `[1.5, 1.5]` and reports both positions pooled.
   (g) **One outlier costs one neighbour.** Twenty agreed words 0.5 s apart, one of them with one
   reading 15 s late → only that position and the next are pooled, the outlier position's derived
   onset equals its neighbour's own onset, and every other position's derived onset equals its own
   midpoint exactly.
5. Agreed repetition: `that that` in both → two agreement words.
6. Variant beside a repetition: `a: the the`, `b: a` → a variant then an insertion.
7. Source order: the same streams passed as `[b, a]` and `[a, b]` give identical words and
   `provenance.sources == ["a", "b"]` both times.
8. Bracket override (R-1): `[COUGH] hi` vs `cough hi` → one agreement word, `text == "[COUGH]"`,
   `bracketed`, `readings == {"a": "[COUGH]", "b": "cough"}`, `bracket_overrides_n == 1`; reversed
   source order gives the same result.
9. Bracketed insertion (R-1): `I [UM] think` vs `I think` → `[UM]` is an `insertion` of one source,
   `bracketed`, `bracket_overrides_n == 0`, rendered unmarked between two bold words. The override
   and the insertion tests assert different `outcome`s on the same bracketed surface.
10. Onomatopoeic: with `{"khh"}`, `khh` in both → agreement `text == "[KHH]"`, `readings` keep
    `khh`; in one → insertion `[KHH]`; with an empty vocabulary it stays a plain word.
11. Empty-key tokens (`...`) are dropped and counted per source.
12. **N = 3**: `a,b: cat`, `c: cot` → `variant`, `agreement == 2/3`, `variants[0].sources == ["a","b"]`;
    `a,b: x y`, `c: y` → `x` is `insertion` with `agreement == 2/3`; `cat/cot/cut` → three variants,
    `agreement == 1/3`.
13. R-2: one source → `LookupError`; zero sources → `LookupError`; the message names the count.
14. A source with zero words counts in `n_sources`, appears in `provenance.sources` with
    `n_words == 0`, and every word of the other source is an `insertion`; all sources wordless →
    no words, `n_words == 0`, `reference_source is None`.
15. `render_transcript`: every §3.7 case; `strong=("", "")` yields the plain join of `word.text`
    with **no** slash in a variant.
16. Provenance has exactly the §4.2 field set (minus what the node adds) — assert the key set;
    assert `slot_overlap`, `slot_mid_tol_s`, `winner_margin`, `alternate_min_share` are absent.
17. R-4: `Consensus` exposes `words` and `provenance` only; no attribute of it or of `ConsensusWord`
    is a slot or span; `align_sources`'s signature has no grouping parameter (assert
    `inspect.signature` parameters == `{sources, onomatopoeic}`).

`preprocess_test.py`:

18. The consensus reads only the store: a third `asr_hypothesis` measurement seeded directly (no
    block) appears in `provenance.sources` and `n_sources == 3`.
19. Each `asr_*` measurement carries every §4.1 field; `role == "asr_hypothesis"`, `source == name`,
    `model_id`, `commit_sha` match the agent.
20. One recognizer block raising → `absent["consensus_transcript"]` names the count; no
    `consensus_transcript` measurement and no `word` is written; the other blocks still run
    (monkeypatch the stub to raise `LookupError`, which is the absorbed type).
21. `align_sources` raising `RuntimeError` → `preprocess` raises `RuntimeError` naming the block
    (the non-absorbed type; `:1527-1529`).
22. `used` links the consensus activity to every hypothesis measurement; a word produced by one
    source `wasDerivedFrom` that source's measurement **only**; an agreement word derives from both.
23. No `event` entity is ever written; `hello [COUGH] world` from both yields three `word`s in that
    order, the middle one `bracketed` and `outcome == "agreement"`.
24. Wordless run still writes the measurement (adjusted from the existing test).
25. `_spans`' asr source proposes spans from the lexical words only, and the number and order of
    `word` entities is unaffected by whatever it proposes (R-4).

Readers:

26. `taxonomy` and `airway` — a bracketed word contributes nothing to `n_words`, marks no span
    transcribed, and does not contaminate.
27. `speech` — (a) a store of only bracketed words fails with "no consensus word"; (b) with words
    `my [UM] name is alice`, `words_n == 4`, the speech span's `words_n == 4`, and the finding
    `alice` is located at haystack index 4 and marks that entity; (c) **a name present only in a
    reading the consensus text does not show is still located and marked**: seed `hi` (agreement)
    and a variant `{text: "alyssa", readings: {a: "alyssa", b: "alice"}}` with the measurement
    `text == "hi alyssa"` and source `b`'s `asr_b` measurement carrying `transcript == "hi alice"`;
    stub the scanner to raise `("PERSON", "alice")` on the text that contains it and nothing on
    the other; assert one `pii` entity, `haystack == "b"`, and a `label/pii` assertion derived from
    the variant word; (d) `pii` entities carry `sources`, and the single-source flag counts
    lexical insertions only; (e) **the PII extent is the hull of the covered words' timings**: seed
    `alice` as an agreement word with `timings {a: [1.0, 1.2], b: [3.0, 3.2]}` and a derived
    extent of `(2.0, 2.2)`; the `pii` entity's extent is `(1.0, 3.2)`, not `(2.0, 2.2)`; (f) the
    same finding raised by two haystacks at the same positions yields one `pii` entity.
28. `plotting` — a tokens panel with `bold: True` on one token and absent on another renders the
    first label with `fontweight == "bold"` and the second `"normal"`.
29. `report` — `consensus_transcript_tokens` key set; `marked_text` bolds only agreement and
    renders `a/b`; `text` has no `**` and no `/`; the token lane sets `bold` iff agreement; the
    redacted placeholder is never bold; `schema_version == "triage-summary/v3"`; `tokens_n` counts
    bracketed words.
30. `figure` — drawing `_asr_lane_panel` on a bare axis: a two-source word draws two filled
    rectangles at two distinct y-bands in its row plus one outline, and a one-source word one fill
    plus the outline; an agreed word whose sources do not overlap in time draws its two fills at
    disjoint x-extents and **in different bands** (no two fills share a band on one row); a word
    whose derived extent is off-page but whose source span is on-page is drawn; the row is
    `index mod asr_rows`, so a word at stream position 5 with `asr_rows == 4` draws on row 1 even
    when it is the first word on the page; the title names each source with its colour; the agreed
    label is bold and the insertion's is not.
31. `redact` — words are released in `index` order; seeding extents that disagree with the index
    order (only possible in a seeded store, since PREPROCESS's are monotone) must not reorder the
    released text (C-7).

## 10. Open questions for the owner

1. **Variant `text` tie-break.** With two sources a variant is always a tie, and `text` goes to the
   lexicographically first source (`asr_crisperwhisper` today). It is a label — `agreement == 0.5`
   and `variants` are the record — but it is also what the PII scanner and REDACT's released text
   see. Keep, or name a preferred source in config with a derivation?
2. *Closed by the second review.* PII in a reading the consensus text does not show was a
   redaction hole, not a question; each source's own transcript is now scanned (§6.3).
3. Ship the `audio_analysis/asr.py` fallback-provenance fix (C-2) separately, as planned here?
4. `[COUGH]`-class bracketed words are now in the store with extents. Should AIRWAY read them as
   lexical corroboration of a cough span (currently: excluded, as the events were)?
5. `words.onomatopoeic_tokens` stays null and now yields bracketed words. Keep, or delete the key
   until a vocabulary exists?
6. **Time-informed tie-break in `_align_pair`.** R-5 permits monotonicity to break ties among
   equal-cost alignments. On `gets` (§3.6) it would pair CrisperWhisper's first copy (9.66, nearer
   Qwen's 9.60) instead of the last (9.99), and the fit would then move nothing. The aligner's
   last-copy rule is tested and documented; changing it is a measured change to `harmonize.py`.
   Take it up as a follow-up, or fold it into this change?
