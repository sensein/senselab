# A fragment is not the task: truncated captures and the declared task's own branch

2026-10-05. Owner, after listening to 16 empty-route recordings (gain-lifted copies in
`~/Downloads/empty_route_check_20261005/`):

> all of the ones brought in can be discarded. they are all too short relative to the task, and none
> of them contain the task. how is it that the task branch found a task match. and if the only output
> from ASR is a bracketed word, that doesn't match any task requirement for any speech task. also if
> something is intended for airway and speech is triggered for asr, then it can still be discarded for
> lack of airway.

and, on bracketed tokens: "unless the bracketed token is a [cough] in an airway cough task".

## How the task branch "found a task match"

`vocabulary._found` reads a branch as having found its kind when it proposed **any** live span into
its own family (`nodes/verdict._spans_by_node`), whatever the span's role. Over the 38 empty-route
recordings that stayed flagged at `73a9f66d` (re-folded read-only from the r9 stores):

| branch | spans that made it `present` | recordings |
|---|---|---|
| AIRWAY | `task_extent` from the activity-envelope fallback (`align`), no event | 19 |
| AIRWAY | the same extent plus one `breath_event` or one `cough_event` | 2 |
| SPEECH | `speaker_turn_0` (diarization, `identify`) and `speech_run_0` (a run built from one stray word, `corroborate`) | 15 |
| SPEECH | the same plus a `task_extent` (`expect`) | 2 |

The activity-envelope extent is the AIRWAY task-extent fix of `d95b0b2a`, written whenever there is
any activity. None of those spans is the task: in every one of the 38 the declared branch's task
conformance was `False` or `UNDETERMINED`, never `True`. The spans were 0.16-3.97 s long, the
transcripts a filler, a bracketed marker or one word (`[UM]` 46, `[laughter]` 20, `[UH]` 12, empty 12
over all 114).

Bracketed tokens were already non-lexical everywhere they are read as words:
`nodes/common.lexical_words`, `residue.is_non_lexical`, the PII scan's token list
(`nodes/speech.py`) and REDACT's re-scan (`nodes/redact.py`). The leak was only the span-based `found`.

## The rules

1. **The declared task's own branch decides whether the task was performed**
   (`vocabulary.declared_task_performed`): its task conformance is `True`. A span alone is never
   enough, and another branch's finding never counts. Each declared family has exactly one owning
   branch (`branches.EXPECTATIONS`, 48 families). The one exception is AIRWAY that could not decide
   its conformance (`UNDETERMINED`) yet found its kind, where the transcript carries a bracketed
   token naming the declared family's own event (`[cough]`, `[咳]` in a cough task; `[breath]` in a
   breath task, `data/airway_event_tokens.yaml`). Such a token corroborates; it is never a word.
2. **Too short for the task** (`too_short_for_task`): a recording shorter than its family's minimum
   in `data/task_minimum_duration.yaml` discards, whatever a branch reports. The default is 1.0 s and
   the two single-cough families are 0.5 s (the profile carries the derivation). Tried second, after
   ADMIT's own fail.
3. **Empty route** (`acoustically_empty`): the ruleset's empty state discards unless rule 1 says the
   task was performed. `route_mismatch:<branch>` is now a flag only for the owning branch's
   task-performed finding.
4. **Declared task absent** (`declared_task_absent`): the owning branch ran and found none of the
   task -- no span in its family, or for a SPEECH-owned task no lexical word in the consensus.
   Tried after the operational grounds: a recording owed a rerun (a missing derivative) is `rerun`,
   because the gap may be why the task was not found.

All four apply on a plain re-fold (`scripts/extend_refold.py`); the fold reads the recording's
duration from ADMIT's `recording` stream and the event tokens from the consensus words.

## Measured effect

In-memory re-fold of r9 stores, read-only (`/orcd/scratch/bcs/002/satra/tmp_trunc/`), old code
`73a9f66d` against this change, 0 errors.

All 114 empty-route recordings discard (76 already did). Of the 38 that were flagged, 36 go as
`too_short_for_task` and the two longer cough recordings (3.74 s, 3.97 s; AIRWAY conformance `False`)
as `acoustically_empty`; `route_mismatch` keys fall from 38 to 0.

A seeded random 2,000 of the other r12 recordings: 66 move to discard (40 from flag, 20 from rerun,
6 from pass), 9 stay `rerun`. 53 are `too_short_for_task`, every one below its family's minimum
(Harvard sentences 24, five breaths 9, breath 5, cough 4, three quick breaths 3, free speech 3, story
recall 2, and one each in five other families). 13 are `declared_task_absent`: SPEECH-owned tasks with no
lexical word (Harvard sentences 2, productive vocabulary 2, free-speech-v2 2, DDK pataka 1), VOICE tasks
whose branch proposed no span (glides 2, prolonged vowel 1, MPT-v2 1), and two breath recordings that
had passed (2.0 s and 19.7 s) in which AIRWAY found no activity at all. A DDK-ka recording owed a rerun
(`route_unexplained`) stays `rerun`. `route_mismatch` keys fall from 25 to 16.

## 2026-10-05: an owner that could not look is owed a rerun

The two breath recordings above were pulled and listened to (`~/Downloads/breath_absent_check_20261005/`).
The 1.98 s breath-2 recording is real audio at a normal level. AIRWAY found nothing because the
`hear_scores` derivative its event search needs never reached the store (`event_instrument` records
`absent: [hear_scores]`); it never looked. `declared_task_absent` read only that the owning branch's
finding was ABSENT, and so discarded a recording the pipeline still owed a measurement.

`declared_task_absent` and `acoustically_empty` now apply only when every owning branch had what it
needs to look. VERDICT collects each owner's absent inputs (`TaskEvidence.owner_absent_inputs`):
AIRWAY's `event_instrument` absences, ROUTING's critical absences for the owner, and the owning node's
gates left undetermined for an uncomputed reading (`absent_not_computed`, `instrument_absent`). With any
of them and the task not performed, the fold raises the operational ground `owning_branch_input_absent`
and the file is `rerun`. `too_short_for_task` is unchanged: a duration needs no instrument. The 19.7 s
v2-breath recording, near-silent with every input present, still discards as `declared_task_absent`.

Measured on the same sample, both codes re-folded over the same stores at the same moment (the v9
review was writing to them): of the 2,114 rows exactly one changes, the 1.98 s breath-2 recording,
from `discard`/`declared_task_absent` to `rerun` with `owning_branch_input_absent`. The 114
empty-route recordings are unchanged (all still discard; none had an owner lacking an input).

## 2026-10-05: a breath task is decided on detected breaths

The owner listened to ten breath recordings, with gain-lifted copies and HeAR/YAMNet readings
(`~/Downloads/contrast_check_20261005/index.csv`). Only one held a breath: `7c169ccc`
threequickbreaths-2, one long breath where three quick ones were asked. `9d16c147` v2-threebreathsmouth
may hold breaths too quiet to hear. The other eight held none, yet six of them passed. AIRWAY's
breath-event walk was the one reading that matched every listen (one event in 7c169ccc, none in the
rest). HeAR Breathe at `score_min` 0.2 scored three silent recordings above it (254e47be 0.47,
517381e9 0.25, 0b8dcad5 0.23), and `[breath]` was written on three (254e47be, 29be45ea, 0b8dcad5).

The two sustained-breath families (`respiration-and-cough-breath`, `respiration-and-cough-v2-breath`,
2,487 recordings) ran a HeAR coverage matcher with no conformance term and so passed by default; AIRWAY
never ran the breath-event walk on them (no breath event lane on any of the 2,485 in r12).

Owner: "yes do fix 1", "wrong pattern should be flagged as a task mismatch", "discard quiet ones".

- The SOUND_COVERAGE matcher (`nodes/airway.py:_airway_coverage`) runs the same walk as the counted
  families and writes `airway_events_found`; the group gains `events_min: 1`, so the task is decided on
  detected breaths. The HeAR coverage fraction is still written, as context. An absent envelope,
  classifier windows or `hear_scores` gives `event_instrument` and so a rerun, as before.
- A breath family listed in `data/airway_event_requirements.yaml` (every breath family; no cough family)
  whose AIRWAY branch, holding every input, detected no breath event discards as
  `no_breath_captured`, tried after the rerun grounds and before `declared_task_absent`. The detection
  floors are not lowered: a breath too quiet to detect is no breath captured.
- Breath events detected but fewer than the instruction asked (`events_min` passed,
  `instructed_count_min_fraction` failed) flag `task_mismatch`, naming the detected and instructed
  counts, in place of `conformance:AIRWAY`. Never a discard.
- `[breath]` tokens are removed from `data/airway_event_tokens.yaml`, and a family decided on detected
  events takes no event-token corroboration in the fold. `[cough]` stays.

The sustained families need AIRWAY to run again, not only a re-fold:
`triage_r9_20260929/fix_breath/replay.sbatch` (2,487 rows, eight slices, in place, REVIEW readings
carried forward), followed by the full re-fold.

Measured by replaying a sample at this code into a scratch out-root seeded with copies of the r9 stores
(r9 untouched): the ten listened recordings plus a seeded 200 sustained-breath recordings, 0 errors.

| listened | owner's ears | events | r12 | now |
|---|---|---|---|---|
| 7c169ccc threequickbreaths-2 | one long breath, three asked | 1 | flag | flag `task_mismatch` |
| 8c42135a breath-2 (0.38 s) | none | - | flag | discard `too_short_for_task` |
| 254e47be breath-1 | none | 0 | pass | discard `no_breath_captured` |
| 05bf77bd breath-2 | none | 0 | pass | discard `no_breath_captured` |
| 7869a54a breath-2 (0.35 s) | none | - | pass | discard `too_short_for_task` |
| 9d16c147 v2-threebreathsmouth | maybe very quiet (discard) | 0 | flag | discard `no_breath_captured` |
| 517381e9 v2-breath | none | 1 | pass | **pass** |
| 29be45ea breath-1 | none | 0 | pass | discard `no_breath_captured` |
| 0b8dcad5 breath-1 | none | 0 | pass | discard `no_breath_captured` |
| 531b6b20 breath-2 | none | 0 | pass | discard `no_breath_captured` |

Nine of ten match the owner's ears. `517381e9` still passes: the walk found one event inside a carrier
that scored breath (YAMNet hears speech there at 0.98). Over all 210, events detected: 0 in 63, 1 in 19,
2 in 18, 3 or more in 97, none reported in 13 (too short, or an input absent). Moves: pass to discard
66, flag to discard 9, pass to rerun 1, pass kept 131, flag kept 3. Discards: `no_breath_captured` 63
(57 of the 200 sampled, about 28%), `too_short_for_task` 12.
