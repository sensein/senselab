# DDK — the envelope mints no `task_extent`

The owner's ruling that removed the envelope fallback, the measurement that supports it, what died
with it, what survived, and what a reader sees change. Landed 2026-09-19. The code is
`src/senselab/audio/workflows/triage/nodes/ddk.py` (`task_extent_span`, `align_ddk`,
`ddk_carrier`). No operating point is introduced or removed.

This supersedes the surviving part of
[`ddk-task-extent-precedence.md`](ddk-task-extent-precedence.md); see
[*What the precedence document still holds*](#what-the-precedence-document-still-holds).

## The ruling

[`ddk-template-decode.md`](ddk-template-decode.md) made the decoded repetition span the
`task_extent` and kept the envelope carrier as a **fallback** where the decode had no reading. Its
own decision 6 named the reason — coverage — and flagged the fallback as the line to change if the
envelope side was meant to go entirely. The owner:

> "if there is no posteriorgram, there is likely no speech. i don't think any DDKs have no
> posteriorgram."

Measured over all **7,994 declared-DDK recordings**: **49 have no posteriorgram — 0.61%**. Every one
of those 49 carries **zero `word` entities**. Eleven of them also failed ADMIT.

So the premise holds in the direction that matters. The fallback protected 0.61% of recordings, none
of which carries recognised speech, and on those recordings it manufactured a claim about *where the
declared syllable task was performed* from an amplitude carrier — with no phonetic evidence that it
was performed at all. That is an assertion the graph cannot support, and losing it is not a coverage
regression but the removal of a false one.

## The rule

> **The decoded repetition span is the `task_extent`, and it is the only thing that can be one.**
> Where the decode reads no repetition, no `task_extent` is proposed and the branch reports the
> instrument absent.

One case replaces the two:

| | `task_extent` | `production` |
|---|---|---|
| the decode completed at least one repetition | first repetition's start … last repetition's end | `syllable_task_from_decode` |
| it did not | **none** | — |

`TASK_FROM_DECODE` is now the only production a `task_extent` on this family carries.

### The fallback fired wider than the 0.61%

The measurement above counts recordings with no posteriorgram *derivative*. The fallback's gate was
not that: `task_extent_span` fell through to the carrier whenever the decode produced no repetition
span, which is three distinct states, not one —

- the posteriorgram derivative is absent (the 49);
- the decode ran and completed **zero** repetitions;
- a class mapping is unmeasured, so the decode is not readable.

In the second of those, the envelope minted a `task_extent` over a recording where the instrument
that reads phonetic identity had explicitly read *nothing*. That is a worse case than the 49,
because there the instrument did run and did answer. Its corpus share is **not measured** — the
sweep above counted derivative presence, not decode outcome — and the removal does not depend on it.
It is verified as a property of the shipped code by
`test_a_decode_that_read_nothing_still_reads_as_found_nothing`, which fails against the unfixed tree
with a span present.

## What died and what survived

**Dead.** `TASK_FROM_ENVELOPE` (`"syllable_train"`) and `TASK_FROM_ENVELOPE_SEQUENCE`
(`"syllable_sequence"`), the only two productions the envelope could stamp on a span; the
`carrier_extent`, `carrier_ids` and `sequence` parameters of `task_extent_span`; and with them the
last path by which an amplitude reading could become a boundary claim. No reader outside this
module and its tests ever selected either production string — `report.py`'s span label renders
whatever `production` a span carries and needs no name of its own — so nothing is orphaned. The
`Pattern.SYLLABLE_TRAIN` / `Pattern.SYLLABLE_SEQUENCE` enum members hold the same two strings and
are a different namespace: they name what the *instruction asked for*, are read from
`SPEECH_EXPECTATIONS`, and are untouched.

**Surviving, deliberately.** The modulation-rate channel, whole:

- `ddk_carrier` still walks the amplitude spans and returns the longest that holds a readable
  modulation peak. Its extent is now used for one thing only — locating the rate reading.
- `train_rate_hz` still takes that rate, written as `RATE`
  (`ddk_syllable_rate_from_envelope_modulation_hz`) over the carrier's extent, with the ambiguous
  `cycles_or_syllables_per_s` unit on the sequential families and the decode's own two rates beside
  it as covariates.
- `read_envelope_track`, `EnvelopeTrack`, `amplitude_spans` and the `train_min_s` /
  `modulation_band_hz` / `rate_prominence_min` points are all still read.
- `_absent(ENVELOPE)` still records an absent envelope derivative as a valueless `RATE`.
- The envelope still answers conformance exactly as before: a carrier past the length guard is
  `done = True`, no carrier is `False`, an unmeasured guard is `UNDETERMINED`, and `_with_decode`
  folds the decode into that unchanged. **No conformance term is introduced or changed by this.**

The channel is kept because it reads periodicity *without segmenting*, and an independent
amplitude-domain rate is worth having precisely because the posteriorgram is imperfect. What it may
not do is say where the task was.

## What a reader sees change

On the 0.61%, and on any recording whose decode completes no repetition:

- **`final/spans`**: no span of role `task_extent` for the family. Its absence is the record.
- **SPEECH's report**: `trains_n` is `0` and `train_s` is `0.0`; `train_fraction` is absent, since
  `train_fraction_of_recording` keys on the span that ships. `modulation_peak_hz` and
  `modulation_unit` are unchanged and still present, so the recording is not silent — the rate
  channel reports and the extent does not.
- **The `truncation` deviation** keys on the surviving span and so is not raised. It never described
  the envelope's extent independently.

### Absent is still distinguishable from ran-and-found-nothing

The three states of
[`ddk-template-decode.md`](ddk-template-decode.md#the-instrument-did-not-run-versus-it-ran-and-found-nothing)
keep their separate records, and removing the fallback makes the span axis agree with them rather
than contradicting them — before, the first two states both shipped a span, minted by an instrument
that had read neither:

| state | record | span |
|---|---|---|
| the posteriorgram derivative is absent | `ddk_syllable_rate_from_ppg_decode_hz` with no value, carrying `unavailable="ppg_posteriorgram"`; **no** `ddk_repetition_count_from_ppg_decode` at all | none |
| a class mapping is unmeasured | no decode findings; the key in `params.missing`; conformance `UNDETERMINED` | none |
| the decode ran and completed zero repetitions | `ddk_repetition_count_from_ppg_decode` **valued 0**, the per-position mass beside it, and the rate carrying `reason="the decode completed no repetition of the declared template"` | none |

The distinction lives in the findings, where it always did, and not in span presence — which is why
collapsing the span arm does not weaken it. A reader separates the first row from the third by
`unavailable` against `reason`, and by whether a repetition count exists at all.

**A zero-length span is not reachable and was never the alternative.** `propose_span` refuses a span
of non-positive duration, so "propose a `task_extent` of length zero to mean found-nothing" fails at
the store boundary rather than shipping an ambiguous reading. The mutation that tries it is caught
by seven tests.

## What the precedence document still holds

[`ddk-task-extent-precedence.md`](ddk-task-extent-precedence.md) argued a **union** of the envelope
and CV readings from a corpus measurement: on 1,341 of 4,563 both-read recordings (**29.4%**) the
envelope extent reached past the CV hull on one side, so "CV wins outright" would have discarded
produced material on nearly a third of them.

That number is not wrong, and it is no longer load-bearing, for a reason internal to it: it was
measured against the **old per-syllable CV walk**, whose coverage the template decode substantially
exceeds. The same document records what the old walk's coverage was — a median envelope/CV duration
ratio of 0.408, and a third of all declared-DDK recordings where the envelope extent was under
*half* the CV hull, worst on the three-syllable families because a three-place cycle modulates the
envelope at the cycle rate. The decode reads the phoneme sequence directly and does not depend on
either instrument's peak resolution, so the 29.4% figure describes a gap between two instruments one
of which no longer exists. Re-measuring it against the decode would be a new sweep, and this change
does not need one: the envelope side is removed for what it *asserts*, not for how far it reaches.

**What is preserved, and is the part that mattered**: the `task_extent` is a **boundary, not a
mask**. It runs from the first completed repetition's start to the last one's end including every
intervening filler frame, and `ddk-cycle-counting.md`'s reading of the span — *"between these times,
the subject produced consonant-vowel syllables in response to a syllable-repetition instruction"*,
asserting nothing about whether every instant inside it holds speech — is unchanged. So is the
one-span invariant, now trivially: there is one minting instrument.

**What is superseded**: the union rule itself (already superseded by `ddk-template-decode.md`
decision 6), its three-case table, `merged_task_extent` and `cv_task_extent` as named functions, and
its first case — "the envelope instrument read a carrier and the CV instrument had no readable
reading → the envelope's extent" — which is the fallback this document removes. Its *reasoning* is
deliberately left in place rather than deleted: the asymmetry it identified, that an instrument
making no claim over a region cannot suppress another's claim over it, is still correct and is why
the surviving modulation rate is reported rather than suppressed where the decode disagrees with it.

## Verification

Every claim above is pinned by a test that fails against the unfixed tree. Reverting `ddk.py` alone
to `794334c3` and keeping the tests fails seven of them:

| test | file | what it pins |
|---|---|---|
| `test_a_carrier_the_decode_never_read_proposes_no_task_extent` | `ddk_test.py` | a readable carrier with no posteriorgram proposes no span |
| `test_no_task_extent_carries_a_production_the_envelope_minted` | `ddk_test.py` | both envelope productions are gone outright, not unreachable |
| `test_the_absent_decode_reads_as_an_absent_instrument_not_as_found_nothing` | `ddk_test.py` | no span, and the record says `unavailable` and carries no repetition count |
| `test_a_decode_that_read_nothing_still_reads_as_found_nothing` | `ddk_test.py` | no span, and the record says `reason` and counts 0 |
| `test_the_rate_is_read_over_the_carrier_and_mints_nothing_from_it` | `ddk_test.py` | the rate is still read over the carrier's own extent; no span comes off it |
| `test_a_declared_diadochokinesis_pa_gets_a_syllable_train_conformance` | `speech_test.py` | `trains_n` is 0 while the rate and conformance are unchanged |
| `test_a_diadochokinesis_buttercup_recording_takes_the_syllable_body` | `speech_test.py` | the same on a sequential family, unit still ambiguous |

Two tests are controls and pass on both trees, which is the point of them:
`test_the_train_is_a_speech_span_carrying_its_production` and
`test_the_train_fraction_is_taken_over_the_extent_that_ships`, both now seeded with a posteriorgram
— a recording the decode read is unaffected by the removal.

Four targeted mutations against the fixed tree, each restored after:

| mutation | caught by |
|---|---|
| a zero-length `task_extent` where the decode read no repetition | 7 tests, at `propose_span`'s positive-duration guard |
| the decoded span stamped `production="syllable_train"` | 3 tests, including SPEECH's own |
| the absent-posteriorgram `_absent(PPG, PPG_RATE)` record dropped | 2 tests |
| the surviving `RATE` measurement renamed | 6 tests, four of them the modulation channel's own |
