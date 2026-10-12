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
between instructed breaths (2bb59c22: 3.2 s), and coughs 5 s apart (23482e01) leave the rest of the task in
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

## Round 2 (owner labels on the 114-recording listen set, 2026-10-12)

The owner labelled 106 of the 114 listen-set recordings (`r37_extents_47a340f876aa.json`). Ten no_task picks
were r36 discards, a sampling error. From here on, strata are drawn from recordings that are not discarded.
Extent changes are read only on recordings that are not discarded.

| Stratum (owner) | Change, `data/task_extent.yaml` v2 |
|---|---|
| A1 grown, 9 of 9 better | `growth` and `modulation` **on** |
| Voice onset trim, 10 of 10 better | `trim` **on** |
| Count-in, 10 of 10 better | `count_in` **on**; `gap_max_s` 3.0 s (r36 count-in-to-vowel gaps, p99 2.9 s, n=1,232); a count-in overlapping the vowel is taken in (282b529e: "three" timed 0.35 s past the vowel's start) |
| Stimulus edges, 10 of 10 better | `stimulus_edges` **on** |
| DDK trim, 7 of 10 better, "a bit narrow" | `trim` on with a margin: a trimmed edge keeps one of the train's own median onset intervals (`margin_intervals`), never into speech; a speech split stands only where the kept piece holds half the train (`kept_share_min`; 2ca44133 was cut to two cycles by agreed words read off the task) |
| Buttercup, 2 current better, "extends through most of recording" | `exclude_families`: buttercup families are never trimmed |
| Continuity, 7 better, 3 too wide | still **off**; a run continuity takes in ends `tail_s` (0.1 s) after its last voiced frame, and at most `runs_max` (1) run a side is taken in |
| A1 stopped at speech, 6 both wrong ("breathing spans the full recording") | `growth.cross_speech` (**off**): growth crosses speech between breaths, never an event on speech, and still stops at the gap |
| Phonation done in parts (805ef15c, 8d28ce2d) | `parts` (**off**): a further run with 1 s of voicing, voiced over half, within 6 s and no agreed word between, is part of the task |
| Weak breaths wider than their events (2b1c662a, 949e6c91, 49b2620c) | `edges` (**off**): the first and last breath run out to where the broadband level falls under 3 dB over the floor (dips of 0.1 s bridged, 3 s at most, never into a stop) |
| Background notes (837ad878, 430a8478, 893ee933, 30d1bba6, 16f9cd2d, 340f3db5, 92ad6bb2, a396e2c0; 49b2620c and 949e6c91 "can be passed") | every task reading records `background` (`task_events.background_reading`); `discard_replaced_task.yaml` `background` block, **off**, bound unfitted |

### Events against background

Each reading records:

- `event_db`: the median of the task events' peak levels;
- `background_db`: the median level of frames that no found event and no impulse holds;
- their difference, `event_over_background_db`;
- `background_over_floor_db`: the background over the stationary floor;
- `background_speech_share`: the share of the background inside speech or diarized turns.

On the labels, **no single ratio separates the owner's discard notes**:

- 837ad878 ("too much background relative weak breathing") reads 13.4 dB over a background at the floor.
- 49b2620c ("very weak events … no background … can be passed") reads 14.9 dB, also at the floor.
- In 837ad878 the floor itself is the noise, so a ratio against any in-file background cannot see it.

Two candidates, each tested against the labels and the r36 corpus:

| Candidate | Discard notes hit (half 0 / half 1) | Other labelled airway files hit (of 49) | r36 non-discarded airway files hit (of 11,682) |
|---|---|---|---|
| ratio under 6 dB | 1 / 1 | 0 | 255 |
| background over 20 dB above the floor | 2 / 1 | 2 | 413 |
| either | 3 / 1 | 2 | 563 |

Nine notes split across the participant halves do not support a bound. The two notes that are not breath or
cough (a396e2c0 DDK, 340f3db5 with no reading) fall outside the rule. The block stays off, with its bound null.

### Measured (r36 stores, read-only, 3bfe4e6f, fresh clone, 62,550 runs)

Scope: 50,158 recordings carry an owned extent, 0 errors. 47,612 of them are not discarded, and the figures
below cover only those. Each extent is compared with r36.

**Defaults on:**

- **Breath:** 856 of 9,036 grow.
- **Cough:** 26 of 2,646 grow.
- **Voice:** 3,471 of 8,225 change. The onset trim moves them later, and the count-in now sits inside 1,219
  extents (6 before).
- **DDK:** 316 of 7,882 trimmed, none widened. Buttercup: 0.
- **Stimulus:** 715 of 19,823 extended.
- **extent_complete, non-discarded:**
  - breath: 637 too_narrow;
  - voice: 98 too_narrow, 105 too_wide;
  - DDK: 109 too_wide, where a speech run under the half-train guard stays inside, and 45 too_narrow;
  - stimulus: 6 too_narrow.

**Switched-off rules, against the defaults:**

- **`cross_speech`:** changes 23 breath extents (median +6.0 s). 17 take in words, which are masked and so
  reviewed.
- **`edges`:** changes 8,459 of 9,036 breath extents (median +0.48 s). It widened all three labelled controls
  the owner read as right. It is too general as it stands.
- **`continuity`:** changes 206 voice extents. The three it widened that the owner read as too wide are
  still wide.
- **`parts`:** changes 93 voice extents.

**Discards:**

- No `too_short_for_task` discard would reach its family minimum under any setting (0 of 1,623). That ground
  reads the recording's duration, not the extent.
- f5c03456's 130 s recording keeps 5.3–11.9 s in every setting, `cross_speech` included, so its conversation
  tail stays out.
- 541e6b30's recordings are unchanged except by `edges` (+0.1 to +1.3 s).

**Agreement with the round-2 labels:**

| Stratum | Agreement |
|---|---|
| A1 grown | 8 of 9 (75c5b87a's owner direction is too wide) |
| A1 stopped at speech, with `cross_speech` | 6 of 7 that carry a direction |
| Voice onset | 9 of 10 |
| DDK | 8 of 9 that carry a direction |
| Count-in | 10 of 10 |
| Stimulus | 9 of 10 (e80daf66 wide at the start) |
| Continuity | 7 of 10 |
| A5 too_narrow, all switches | 7 of 7 that carry a direction |
| Controls | 7 of 10 (the 3 misses are `edges`) |
| Buttercup | 2 of 4 (the owner preferred one trim) |

**Round-1 labels (52 with an extent):** the 27 read complete stay put on 23. Of the 22 too narrow, 19 widen.
Both too-wide labels narrow.

## Round 3 (owner labels on the 70-recording listen set, 2026-10-12)

`r37_round3_extents_cc33de81b9e3.json` holds 15 labelled rows; the owner reads an unmarked row as "ok" (the
proposal is fine). Strata were drawn from the r2 measurement (`extent_r37/analysis_r2.txt`, "listen strata,
round 3"), non-discarded, from participants not labelled before.

| Stratum (10 each) | Owner | `data/task_extent.yaml` v3 |
|---|---|---|
| Continuity, end bounded (`tail_s`, `runs_max`) | 9 ok, 1 better | `continuity` **on** |
| Breath growth across speech | 7 ok, 3 better | `growth.cross_speech` **on** |
| Phonation in parts | 7 ok, 2 better, 1 current better | `parts` **on** |
| Count-in overlapping or 2-3 s before | 9 ok | `count_in` stays on (`gap_max_s` 3.0) |
| Weak-breath edges | 9 ok, 1 both wrong ("no task really") | `edges` **on** |
| DDK margin | 7 ok, 1 same, 2 current better (narrow on a difficult voice) | unchanged (`margin_intervals` 1.0); no corpus bound supports widening |
| Background rule candidates (lowest events over background) | 3 notes: "too much (background) noise", "should be withheld" | see "Deferred" |

### Deferred to after the release (owner, 2026-10-12: release first, no new research)

- **Background in airway tasks.** The owner's rule: background speech withholds the release (a privacy
  concern) and reviews; background noise alone reviews with the release kept, with the session background
  (`session_background.py`) as the reference where the file's own floor is the noise. Neither bound has been
  measured (`background_speech_share`, background over the session floor), so nothing is switched on:
  `discard_replaced_task.yaml` `background` stays off. Recordings with another voice in the background already
  reach `other_speaker` (the three noted files were `review/withheld (other_speaker)` at listen time).
- **E1 (`discard_replaced_task.yaml`) stays off.** On the owner's 20 round-3 labels it would discard 20; the
  owner said discard 9, review 9, pass 2: it fires on a weak task where it should fire on an absent one. The
  rework (require no task event anywhere in the file, read after the extent changes) and its corpus
  re-measurement are not done.
