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
