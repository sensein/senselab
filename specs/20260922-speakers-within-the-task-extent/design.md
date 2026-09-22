# Speakers inside the task extent — a branch-reported reading

## The gap

`speaker_count` is whole-file. It is
`len({speaker for _, speaker, _ in speaker_segments})` in `nodes/speech.py`, computed over every
diarized segment the recording carries, and its only product is a note, `speaker count N != 1`.

A task extent runs from the beginning to the end of the task. So the two cases that matter fall
on opposite sides of a boundary the whole-file count cannot see:

- **An examiner's prompt before the task.** Whole-file count 2, and the participant performed the
  task alone. Nothing about the recording is wrong.
- **An interjection in the middle of the task.** Whole-file count 2 as well — the same number, for
  a materially different recording.

The whole-file count cannot separate them, and no other reading in the graph tries. The one that
comes closest, `nontarget_speech_s`, depends on `target_speaker`, which comes from enrollment;
enrollment was `None` across the whole corpus, so that field is `None` on every recording. A
reading about whether a second voice is inside the task must therefore not depend on knowing
which voice is the participant — and it does not need to, because "does this extent contain more
than one voice" is answerable without an identity.

## What is read

PREPROCESS already writes `<stream>_diarization` with per-segment `(start, end, speaker)` and an
`n_speakers` attribute; SPEECH already reads it back through `_read_diarization` and bounds each
segment to the stream's decode. This adds **no inference**: it intersects segments the branch has
already read with the extent the branch has already minted.

Per task extent:

| reading | type | what it is |
| --- | --- | --- |
| `extent_speaker_count` | `int` | how many diarized speakers hold a segment intersecting the extent |
| `extent_dominant_speaker_share` | `float` or `null` | the largest contributor's seconds over **every attributed second inside the extent** |

Both carry the same covariates, so the share can be recomputed from the row rather than trusted:
`speaker_labels`, `speaker_seconds` (parallel, one per label), `attributed_s`, `dominant_s`,
`secondary_s`, `extent_s`, `diarizer`. Each names its evidence: the task-extent span, the
diarization measurement, and every `speaker` entity that intersected.

### Why the denominator is attributed seconds and not the extent

`dominant_s / extent_s` was rejected. A task extent legitimately contains silence — pauses inside
a narrative, the gap between prompted items — and under that denominator a single-speaker
recording with long pauses reads the same as one split between two voices. What is being asked is
"of the voice inside this task, how much is one voice", and that question's denominator is the
voice, not the clock. `extent_s` is on the row for a reader who wants the other ratio.

A consequence worth naming: because pyannote's view is overlapping rather than exclusive,
`attributed_s` can exceed `extent_s` where two speakers talk at once. That is honest under this
denominator and would be incoherent under the other.

### Why the share and not the count is the gateable number

`extent_speaker_count` is the more obvious reading and the worse gate. A diarizer that splits off
0.2 s of breath as a second label takes the count from 1 to 2, and a bound on the count would then
fail a clean recording on a segmentation artefact. The share degrades continuously: the same 0.2 s
artefact moves it from 1.00 to about 0.97, while a genuine eight-second examiner turn inside a
twenty-second task moves it to 0.60. Both are reported; the gate reads the share.

### The absences, which are absences and not zeroes

- **No diarization derivative.** No reading is written. VERDICT's rule applies unchanged: a gate
  whose reading is absent yields `UNDETERMINED`, never `False`.
- **No segment intersects the extent.** `extent_speaker_count` is `0` and
  `extent_dominant_speaker_share` is `null` — there is no dominant contributor among none. A null
  value is treated as absent by `gate_readings`, so this too is `UNDETERMINED`.
- **Detect mode, or no task extent minted.** The report carries
  `extent_speakers: "no_task_extent"` and no reading is written.

The branch report carries the whole reading under `extent_speakers`, so a reader who is not a
gate sees the per-speaker seconds without decoding the store.

## What it is scoped to, and what it is deliberately not taken on

**SPEECH only.** The owner's reasoning: another person breathing, coughing, sustaining a vowel,
gliding or performing diadochokinesis inside a participant's recording is very unlikely to
non-existent. The reading is worth taking where a second human voice is plausible, which is where
lexical content is.

There is a second, mechanical reason to scope it the same way. The derivative is a **speech**
diarizer; pyannote segments conversational speech and has no defined behaviour on a held vowel or
a cough, so a count taken over an AIRWAY or VOICE extent would be measuring the diarizer's
response to out-of-domain audio rather than the number of people in the room.

**The absence is recorded rather than silent, in two places.** SPEECH emits the reading for every
task extent it mints, across all of its task groups — reporting is not where the discrimination
happens. The discrimination is in the gate table, and
`specs/20260817-triage-workflow-dag/config-derivations.md` § verdict.gates names, group by group,
which groups carry `dominant_speaker_share_min` and which do not and why. A group that carries no
bound is a decision on the record, not an oversight.

AIRWAY and VOICE reports carry no `extent_speakers` key at all, which is the honest shape: the
reading was not taken, rather than taken and found empty.

## What this does not do

- **It does not identify anyone.** The labels are the diarizer's own (`SPEAKER_00`), in the first
  of the three id namespaces the graph keeps distinct. Nothing here says which label is the
  participant, and nothing here needs to.
- **It does not decide.** No threshold, no conformance, no outcome is computed in the branch. The
  bound lives in `verdict.gates` and is applied by VERDICT.
- **It does not validate the diarizer.** Every number is as good as `<stream>_diarization`, and no
  ground truth for who spoke when exists on this corpus. What a wrong segmentation does to the
  share is bounded by the share's continuity, which is the argument for the share and not a claim
  that the segmentation is right.
- **It is not measured against a corpus distribution.** The reading is new, so the distribution of
  `extent_dominant_speaker_share` over the corpus is unknown. That is exactly the measurement the
  gate's derivation names as the one that would fit its bound.
