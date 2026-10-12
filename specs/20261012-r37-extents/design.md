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
  current edge is at most `k` x the cluster's median end-to-start gap (`k` = 2.5: one missed cycle with 25%
  jitter; floor 0.25 s; a one-event cluster takes its family's clustering gap);
- stop at the first **stop** between the edge and the event or on it: speech runs and segments, YAMNet speech
  windows, every lexical word (not only runs: growth is the conservative direction), and every diarized turn
  of at least 1 s (COHORT's `min_run_s`) on the enhanced stream, whoever's.

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

Filled from the corpus measurement below.
