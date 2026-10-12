# r37 extents: complete in both directions (sections A, E2, E3)

2026-10-12. Owner-approved r37 plan v2 (`~/Downloads/b2ai-reviews/r37_plan.md`, sections A, E and H), with
the review against the owner's notes (`r37_plan_review.md`). Inputs read: the 62 extent labels
(`extent_check_9e550e2026a7.json` joined to `extent_check_listen.tsv`) and the 24 triage notes
(`triage_review_be5d2593d524.json`). Code: `task_extent.py`, `data/task_extent.yaml` (v1), and the owning
branches' readers. Every rule is general and **switched off** in `data/task_extent.yaml` until the corpus
measurement below is read; nothing here is fitted to the labelled files.

## What the labels say

61 of 62 extents contain the task; the failure is completeness, in both directions. Too narrow: breath 15 of
20, cough 1, the voice count-in 3, the stimulus edge words 3. Too wide: 43bc9a51 ("task starts after 1s
mark"), 119daa75 ("beginning has extra"). Breath extents are incomplete about half the time even when no
signal fires, so the rule that builds the extent changes; gating on a signal is not enough.

## A1. Breath and cough: bounded growth

Diagnosed on the stored r36 readings: the extent is the task cluster's hull, and the cluster splits at
`gap_min_s` (3 s) or 1.5 rhythm cycles. Slow breathing (8da471da, d8630e11: events 2.5-3 s apart), a pause
between instructed breaths (SUBJ21: 3.2 s), and coughs 5 s apart (23482e01) leave the rest of the task in
other clusters. 211cf773 keeps 1 event in its extent and 7 standing outside it.

**Rule** (`task_extent.grow`, called from `task_events.evidence_of` with `grow_kind`):

- start from the chosen task cluster (the decision is taken on it, before growth, and does not move);
- take the next standing event (over `decision.snr_low_db`) outward, one at a time, while its gap to the
  current edge is at most `k` x the cluster's median onset interval (`k` = 2.5: one missed cycle with 25%
  jitter; floor 0.25 s; a one-event cluster takes its family's clustering gap). The end-to-start gap was
  tried first: close inhale and exhale phases put it under a second (2aa113a2: limit 1.9 s);
- stop at the first **stop** between the edge and the event or on it: speech runs and segments, YAMNet speech
  windows, every lexical word (not only runs: growth is the conservative direction), and every diarized turn
  of at least 1 s (COHORT's `min_run_s`) on the enhanced stream, whoever's. YAMNet speech windows are not
  stops: on the labelled breath files they fell on breathing (211cf773 holds ten).

The growth record (`growth`: events added, seconds before/after, the limit, why each side stopped) is on the
reading's evidence; `extent_growth_s` is an evidence item. Speech left inside a grown span is the task
speech reading's, which is over the whole file, so it is already masked and the release review follows.

**Envelope-modulation detector** (`task_extent.modulation_events`; owner on 8da471da, "modulation should have
caught these events"): the background's mean band level over the stationary floor, read at 10 Hz,
band-passed to the breathing band (0.1-1.2 Hz, as `task_events.yaml` rhythm); one event per positive lobe
standing 3 dB proud, spanning the frames around its peak 6 dB over the floor, 0.3-4 s long. For breath the
type test is unchanged: at most half voiced, off speech. Its events join growth and completeness only, never
the cluster or the decision (`source: modulation`).

## A2. Over-inclusion trimmed to the task's acoustic onset and offset

- **Voice** (43bc9a51): the phonation extent counts a frame as phonation on level alone, so 0.3-1.0 s of
  unvoiced noise 10-20 dB over the floor opened the extent before the vowel at 1.15 s. The extent now starts
  at the **phonation onset**, the first voiced stretch of at least 0.2 s inside it (`trim.voice`); the
  inhale no longer attaches when trimming is on. The onset is recorded either way (`voiced_onset_s`).
- **DDK** (`task_extent.trim_train`): the train splits at runs of at least 2 agreed non-task words
  (`task_content.agreed_runs`, the family's task syllables from `cohort_stage.yaml`
  `non_task.task_syllables` excluded, so "buttercup" never cuts a buttercup train); the largest piece is kept;
  an edge event lasting over 2.5 x the kept events' median is dropped (119daa75: a 0.53 s opening before
  0.12 s syllables). 028943cf (agreed words over the last event) and 07929b7b (speech inserted at 12 s) end
  before the speech. Word spans are the consensus extent, not the hull of every recogniser's placement
  (028943cf has an agreed word whose hull runs 0.54-5.44 s).
- **Breath and cough**: by construction the extent is the events' hull (the breath inhale back-trace, the
  cough's 0.1 s padding); nothing further is added.

## A3. The voice count-in is task content

The declared count-in (the voice expectation's tokens, "one two three") is placed by `count_in_span`; where
it ends at most 2 s before the phonation's start it opens the task extent (`count_in`). It was already task
content for masking (`task_speech.task_text_ids`), so it is not masked and not off-task. The voice target
pool stays empty (`cohort_stage.yaml` target rule `none` for voice families), so it never enters the speaker
comparison; loudness's "hey" stays annotation-only. The words agent's held-vowel timing clause reads the
**phonation span** (`extent.onset_s..end_s`), not the task extent.

**An interrupted vowel** (e8044ea4): the extent took the longest run and merged neighbours only when they
were at least 0.8 s long, so the vowel from 3.35 s, broken by a word the recognisers disagreed on, was left
out. With `continuity` on, a neighbouring run within 2.5 s whose median F0 lies within 3 semitones of the
extent's, with at least 0.1 s of voicing and no **agreed** speech word between, continues the vowel whatever
its length.

## A4. Stimulus edges: diagnosed

Both recognisers had the dropped edge words. The stimulus extent ran from the first to the last **realised**
token; a token aligned under a different key is **substituted** and was left out. In c1807e86 (last token),
49a34a52 (first two) and 06b0a943 (last) the consensus kept the reading that differs from the stimulus while
the other recogniser read it exactly (or the reading is one or two edits away). 6535c3f1 is different: its
consensus word times are piled at the file end (aligner failure), which the words agent's timing
arbitration owns.

**Fix** (`nodes/speech._read_as_stimulus`, `stimulus_edges`): a substituted token counts as read when any
recogniser's reading normalises to its key, or the consensus reading is a near match under
`stimulus.near_match`; the extent spans every token read. No padding.

## A5. Columns and decision

`extent_complete` = complete / too_narrow / too_wide / both / no_task (`task_extent.ExtentCheck`), with
`task_events_outside_extent` and `extent_check` (the signals, `narrow:<name>` / `wide:<name>`):

| Family | Too narrow | Too wide |
|---|---|---|
| breath, cough | standing events (type test or modulation) outside the extent with no stop between | — |
| voice | the count-in outside; a reachable outside run with 0.8 s of voicing within 3 semitones | the extent starts over 0.3 s before the phonation onset with no count-in there |
| DDK | target events outside with no speech run between | a speech run inside; an over-long edge event |
| stimulus | words read as the stimulus outside the extent | — |

No extent is `no_task`. Not complete reviews (reason `extent_incomplete`, after `weak_events`) where
`completeness.reviews` is on, else it is an annotation. `no_task` reviews; a discard already standing stays
a discard. The right-task checks (DDK rate and period CV, glide travel, speech share, open/item coverage)
remain annotation values only.

## E2. Atypical attempt

An attempted cough or syllable task whose events fall outside its family's corpus range is the annotation
`atypical_attempt` (data, never a reason; distinct from `other_speaker`): the median cough event over the r36
p97.5 (1.877 s, n=2,643), or a syllable rate outside r36 p2.5-p97.5 (2.235-7.692 Hz, n=7,815). The bounds
are the corpus's central 95%, not fitted to a label. 0482bc77's cough (median 1.51 s) is under the bound;
its buttercup train (2.22 Hz) is over it. One label: annotation only.

## E3. A failed preprocessing block stops only its readers

PREPROCESS raised when any block failed unexpectedly, so a PPG timeout on 211cf773 (a breath recording AIRWAY
reads no PPG for) errored the run and left it not_measured. A failed block is now recorded on PREPROCESS's
verdict (`failed`, named in `why`) and is absent like any missing derivative; a branch that reads it reports
`owning_branch_input_absent`, every other branch runs. The ppgs worker's budget is at least
`ppg.timeout_min_s` (1800 s; `config-derivations.md`).

## Measured (r36 stores, read-only)

Every owning branch's reading taken in memory twice, all switches off and all on, over the 62,550 r36
stores (`manifest_r34.jsonl`), at f2a806af on a fresh clone; CPU array on ORCD; nothing written to a store.
Scratch: `/orcd/scratch/bcs/002/satra/r18d_20261010/extent_r37/` (`measure.py`, `analyse.py`,
`analysis_n.txt`). 50,158 recordings carry an extent this work owns; 0 errors. Switched off, every extent
equals the stored one except 140 coughs with no cough event, whose stored extent is AIRWAY's own fallback
span. No decision or decision input moved.

- **A1 growth.** Breath: 872 of 10,197 grow (median 8.2 s; p25 4.3, p75 12.9, p90 17.7, max 39 s); cough
  26 of 2,812 (median 13.8 s). Growth stops at the next event's gap on 711, at speech or a turn on 27. The
  modulation detector found events in 564 breath recordings, none in cough. No recording gains a word
  inside its grown extent (growth stops at every lexical word), so no new mask. f5c03456 (130 s, background
  talker) keeps 5.3-11.9 s: its tail stays out. Breath events reaching the instructed count: 5,210 → 5,342
  of 7,712.
- **A2 trims.** Voice: 2,715 of 8,227 start at the phonation onset, each more than 0.3 s later (MPT 1,133,
  glides 994, prolonged vowel 588). DDK: 417 of 7,881 trimmed (median 0.7 s removed; 144 at speech, 273 an
  over-long edge event); buttercup 29 (25 at speech, 4 edge). Breath and cough: none.
- **A3.** The count-in joins 1,119 prolonged-vowel extents; 107 count-ins lie over 2 s before the vowel and
  stay outside (too_narrow). Phonation continuity widens 194 voice extents.
- **A4.** 717 of 19,741 stimulus extents take in an edge word read as the stimulus (head 388, tail 350;
  Harvard 618, CAPE-V 74, stroop 13, passages 12).
- **A5** (off → on): breath complete 8,196 → 8,802, too_narrow 1,248 → 642, no_task 753; cough too_narrow
  71 → 45; voice too_wide 2,322 → 108, too_narrow 1,270 → 153; DDK too_wide 452 → 0, too_narrow 43 → 52,
  no_task 59 → 103 (trimming leaves no target event); stimulus too_narrow 681 → 6.
- **Labels** (52 of the 62 carry an extent this work reads; the 10 open and item-set files do not).
  Switched off, the check names the owner's direction on 46 of 52 (all 27 complete, 17 of 22 too narrow,
  both too wide; the no_task file reads complete). Switched on, the extent moves the owner's way on 14 of
  22 too narrow and 2 of 2 too wide, and moves none of the 27 complete. The 8 left: 108c14bf, 1b07b18b,
  f5c03456's opening (no event found there), 6535c3f1 (aligner timing), and 23482e01, 8da471da, bf78d13e,
  df9d06c5, which stay too_narrow and so review.
- **Participant halves** (hash of the participant id): off-check direction agreement 23/26 in each half;
  the 8 unmoved labels split 3 / 5. Corpus not-complete share on: 4.6% / 4.8% (too_narrow 434 / 464,
  no_task 589 / 514, too_wide 47 / 62).

Pass criteria for the listen set (H) are set before listening: at most 1 in 10 over-inclusions in the A1
strata; at most 1 in 10 trims that cut task sound in A2; none of the buttercup trims cutting the train.
