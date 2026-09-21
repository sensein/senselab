# Every gate in VERDICT, keyed by task group

Owner-directed 2026-09-21: *all gates should be in verdict not in branches and it should be task
group related. branches should just provide the info necessary for the gates. so instruments sit in
preprocess and branches.*

This completes the contract the graph already claims — *a branch reports and VERDICT decides* — which
today is true of outcomes and false of thresholds.

## What is wrong now

`branch:` is 31 flat operating points applied identically to every task, and the branches apply them
themselves. Two consequences, both measured on the 62,009-recording corpus run:

- **One threshold governs unlike tasks.** `f0_spread_max_semitones: 2.0` bounds how far pitch may
  move in a *held vowel* and is applied unchanged to a *glide*, whose whole purpose is that pitch
  moves. `production_min_s: 0.5` rejected 423 glide and MPT carriers at a 0.12 s median.
- **The branch decides.** `_speech_ordered` returns `bool(matched) and not omissions`; a single
  missed word in a 100-word passage reads exactly like silence. `caterpillar-passage` fails 48.6%
  and `rainbow-passage` 23.8% on that rule.

## The split: a gate decides, an instrument measures

An **instrument setting** says *how to take a reading* and stays with the instrument, in PREPROCESS
or in the branch. A **gate** says *what reading is good enough* and moves to VERDICT.

| gates — move to VERDICT, per task group | instrument settings — stay |
| --- | --- |
| `continuity_min`, `coverage_min`, `dominant_segment_min_fraction`, `echo_overlap_max`, `f0_spread_max_semitones`, `monotone_tolerance_semitones`, `production_min_s`, `rate_prominence_min`, `repeat_overlap_min`, `response_min_s`, `score_min`, `train_min_s`, `verbatim_overlap_max`, `voiced_fraction_min`, `interval_max_s`, `gap_off_task_min_s` | `burst_window_ms`, `echo_ngram_n`, `effort_split_hz`, `f0_spread_window_s`, `label_sets`, `modulation_band_hz`, `peak_prominence_db`, `phoneme_place_classes`, `phoneme_vowel_classes`, `smoothing_window_s`, `trough_return_db`, `event_min_s`, `voiced_strength_min`, `pause_min_s`, `run_gap_max_s` |

The right-hand column is definitional: `voiced_strength_min` defines what *counts* as a voiced
frame, `event_min_s` what the walk will *report*, `pause_min_s` what a gap *is*. Changing one changes
the reading, not the verdict on it. The left-hand column decides.

## Task groups

`Pattern` is already the grouping and is not invented for this: `SUSTAINED`, `GLIDE`,
`ORDERED_TOKENS`, `FREE_RESPONSE`, `ITEM_LIST`, `EVENT_SERIES`, `EVENT_ALTERNATION`,
`SOUND_COVERAGE`, `SYLLABLE_TRAIN`, `SYLLABLE_SEQUENCE`, `PER_SENTENCE`, `EFFORT`. A gate is
configured per group, and a group that names no value for a gate does not apply it.

```yaml
verdict:
  gates:
    SUSTAINED:       {production_min_s: 0.5, voiced_fraction_min: 0.5, f0_spread_max_semitones: 2.0}
    GLIDE:           {production_min_s: 0.5, voiced_fraction_min: 0.5, dominant_segment_min_fraction: 0.5}
    ORDERED_TOKENS:  {ordered_match_min: 0.75}
    FREE_RESPONSE:   {response_is_a_sentence: true}
```

`GLIDE` naming no `f0_spread_max_semitones` is the point: the gate that made no sense there simply
is not configured for it, rather than being set to a value that disables it.

### A group is not fine enough on its own

Owner-directed 2026-09-21: *there needs to be more specificity for task somewhere — for example the
different DDK tasks are different in requirements.* The DDK families show it exactly:

| families | pattern | what the instruction asks |
| --- | --- | --- |
| `diadochokinesis-pa`, `-ta`, `-ka` | `SYLLABLE_TRAIN` | a **count**: `expected_event_count=10` |
| `diadochokinesis-pataka`, `-buttercup` | `SYLLABLE_SEQUENCE` | a **count**: 30, being 10 of a 3-syllable carrier |
| `diadochokinesis-v2-puh`, `-tuh`, `-kuh` | `SYLLABLE_TRAIN` | a **duration**: `declared_duration_s=5.0`, no count |
| `diadochokinesis-v2-puhtuhkuh`, `-v2-buttercup` | `SYLLABLE_SEQUENCE` | a **duration**: 5 s |

So `SYLLABLE_TRAIN` holds both a count-based instruction and a timed one, and no single setting of a
rate or duration gate serves both. The same is true beyond DDK: `maximum-phonation-time` declares
`expect_inhale` and its v2 does not, while both are `SUSTAINED`.

**Gates therefore resolve most-specific-first: family, then group, then default.**

```yaml
verdict:
  gates:
    default:
      score_min: 0.2
    by_group:
      SYLLABLE_TRAIN:   {train_min_s: 1.0, rate_prominence_min: 2.0}
      SUSTAINED:        {production_min_s: 0.5, voiced_fraction_min: 0.5, f0_spread_max_semitones: 2.0}
      GLIDE:            {production_min_s: 0.5, voiced_fraction_min: 0.5, dominant_segment_min_fraction: 0.5}
      ORDERED_TOKENS:   {ordered_match_min: 0.75}
    by_family:
      diadochokinesis-v2-puh: {train_min_s: 4.0}
```

A family entry overrides its group key by key, not wholesale: a family naming one gate inherits the
group's others. The default layer carries only what is genuinely universal. **Ship the family layer
empty** — every value moves at its current setting, and an empty layer is the honest statement that
no per-family difference has been derived yet. What this buys now is that the distinctions above
become *expressible*; filling them in is the refitting work this change exists to enable.

The three layers must each be readable back from the verdict: a recording's record says which layer
supplied each gate it was judged by, so a reader can tell a family-specific bound from an inherited
one without consulting the config.

### A count the instruction gives, and a count nobody gave

Owner-directed 2026-09-21, restating an earlier ruling: *the instruction does not provide any such
number*, *there was a previous discussion on expected statistically vs required*, and *individuals
can vary*. One field, `expected_event_count`, currently holds two unlike things:

| family | count | where the number comes from |
| --- | --- | --- |
| `respiration-and-cough-fivebreaths` | 5 | **the instruction** — it is in the task's own name |
| `respiration-and-cough-threequickbreaths`, `-v2-threebreaths*` | 3 | the instruction |
| `voluntary-cough`, `breath-sounds` | 3 | the instruction |
| `loudness` / `-v2` | 3 / 2 | the instruction — its `tokens` enumerate them, `('hey','hey','hey')` |
| `diadochokinesis-pa`, `-ta`, `-ka` | 10 | **nobody.** The task says repeat, as fast as you can |
| `diadochokinesis-pataka`, `-buttercup` | 30 | nobody |

The docstring says "how many events the instruction asks for", which is true of the first group and
false of the second. A participant told to take five breaths and giving four has departed from the
instruction. A participant producing eight `/pa/` rather than ten has done the task correctly; ten
was never asked for, and **individuals vary** — rate is the measurement of interest, not compliance
with a number nobody spoke.

So the row must say which kind it carries, and the gates must treat them differently:

- **A required count may be gated**, with tolerance, because a departure from it is a departure from
  the instruction. What tolerance is a derivation, not a guess, and is not made here.
- **A statistically expected count may never be gated.** It is reported as a covariate beside the
  measurement it qualifies — the rate, the repetition count — and VERDICT applies no bound to it.
  This is the owner's standing ruling that an expected count is a heuristic and not a target.

Until each row declares its kind, no gate may read `expected_event_count` at all. **That is the
correct behaviour for this change**: the DDK families lose nothing, because a count nobody asked for
should never have decided anything, and the counted-breath families keep their reading as a
measurement until the kind is declared and a tolerance derived.

## The mechanism

1. **A branch reports readings, never a verdict.** Each gate's input becomes a `measurement` the
   branch already knows how to write — `write_findings` and `measured()` exist and are used. Branch
   conformance becomes `UNDETERMINED` in every arm.
2. **VERDICT reads them from the store** and applies its group's gates. It already reads
   `ruleset_routing` this way; measurements are live entities and need no new channel.
3. **A gate whose reading is absent yields `UNDETERMINED`, never `False`** — the rule this session
   established three times over, now enforced in one place instead of nine.
4. **Every gate applied is recorded** on the verdict: its name, the reading, the bound, and the
   group it came from, so a decision can be read backwards without rerunning anything.

## What this fixes by construction

- Thresholds become visible in one table instead of scattered across three branch modules.
- A gate that suits one task and not another is expressible; today it is not.
- `story-recall`'s 93.8% `False` and the ordered-token all-or-nothing become configuration
  questions rather than code changes.
- `carrier_rejected` stops being a branch-internal note and becomes the fold's own evidence.

## Out of scope here

Re-deriving any threshold. Every value moves at its current setting; this changes *where* a gate
lives and *which tasks* it applies to, not what it is set to. Refitting is separate work, and the
table this produces is what makes it possible.

## What was built

- [`implementation.md`](implementation.md) — what moved, where each gate is applied, the
  equivalence argument per reading, the four keys I would classify differently, the mutation
  results and the open items.
- [`corpus-replay.md`](corpus-replay.md) — conformance per family before and after, over 61,797
  recordings of the 2026-09-19 corpus run, replayed once per gate shape. **Nothing moved.**
- [`corpus-replay.py`](corpus-replay.py), [`corpus-replay-report.py`](corpus-replay-report.py) —
  the branch-only replay and its join, read-only over a finished run tree.
- [`mutations.py`](mutations.py) — one plausible wrong version of each behavioural line, and the
  tests that catch it.
