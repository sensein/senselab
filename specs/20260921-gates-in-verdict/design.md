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
