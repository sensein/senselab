# The multi-speaker instrument, and the gate that reads it

An instrument inside the SPEECH branch that measures **where the speakers are**, and a VERDICT gate
that decides what to do about it. The two halves are deliberately separated: the branch reports and
VERDICT decides, and `Result` has no `done` field so the separation is structural rather than
conventional.

This extends `specs/20260922-speakers-within-the-task-extent/design.md`, which landed the
within-extent reading the gate reads. Nothing here replaces it.

## What the owner specified

> what the diarizer and separation should do is separate voices (SS does not work to separate
> cough/breath, just speech). so only spoken/vocalized content is relevant. this should only happen
> to files entering the speech branch. nothing else is relevant. and remember that branch is
> refining extent and generating new measures, so the multi-speaker instrument is just an
> instrument that fires within the speech branch to measure where the speakers are. it does not
> take decision. that's for verdict to use (if other speaker is within the task extent, flag, if
> not pass and combine with other decisions)

and, on the trigger:

> separation should only work on enhanced audios that pyannote detects more than 1 speaker in.

## The trigger, and the stream it reads

`diarization.streams` is `[enhanced]`. SPEECH's `_read_diarization` walks that list and returns the
first stream with a live measurement, so `read.signal` is `enhanced` and the count that triggers
separation already comes off the enhanced stream. That half needed no change.

The half that was wrong is what separation then decomposed. Step 5 called
`separate_audios([plain], ...)` and hung `used` and `wasDerivedFrom` off `plain_id` — it detected on
one signal and decomposed another. It now resolves `read.signal` and separates that, and every edge
names it. A run whose store carries the derivative but not the stream it was measured on records
`signal_stream_absent:<name>` rather than raising: that is a partial store, not a failure of the
instrument.

## What the instrument measures

Separation's two outputs, `separated_0` and `separated_1`, were written and read by nothing — one
docstring mention and no consumer. They now have one: a **localisation** step, `localise_speakers`.

Per task extent, over frames of `branch.smoothing_window_s`:

| reading | type | what it is |
| --- | --- | --- |
| `extent_source_active_s` | `float`, one per source | the seconds inside the extent on which that source is the loudest |
| `extent_secondary_source_s` | `float` | the seconds a source other than the largest one holds |

and one span:

| role | what it is |
| --- | --- |
| `solo_extent` | the longest run inside the task extent over which the dominant source holds every frame |

Each `extent_source_active_s` carries `active_spans` — where that source is, not just how much of
it there is — and `speaker`, the diarized label whose exclusive seconds that source is loudest
over. That last is what ties the separation's sources back to the diarizer's labels without
introducing a fourth id namespace: the source is not given a name of its own, it is reported as
being SPEAKER_00's.

### Why every step is argmax and nothing is a threshold

A frame belongs to whichever separated source reads loudest on it. That is a comparison between the
sources, not a comparison against a level, so it needs no operating point and cannot be
mis-calibrated. The same holds for the source-to-label match: the label whose exclusive slices a
source is loudest over, by `_exclusive_slices`, which is already arithmetic over the segments.

The one place a threshold would have crept in is silence. Argmax over two near-zero frames
attributes numerical noise. Rather than introduce a floor, the candidate frames are restricted to
those the diarizer already attributed to somebody — the same union of segments that is the
denominator of `extent_dominant_speaker_share`. So the two readings share a denominator and can be
compared, and the instrument adds no number nobody has measured. A frame on which every source
reads exactly zero is attributed to none.

### What refining the extent means here, and why it is additive

**A new span with its own role, not a supersession of the task extent.** `extend.py` has the
machinery to retire a span and spans carry a `role`, so either was available. Three reasons for the
additive one:

1. `speaker_vectors.py` selects on `role == "task_extent"`, and so does SPEECH's own
   `task_extents` list, which is what `extent_dominant_speaker_share` is taken over. Superseding
   the task extent would silently move the denominator of a reading already taken against it.
2. It would make two recordings incomparable for a reason that has nothing to do with either: the
   one where the backend was configured would carry a narrower extent than the one where it was
   not.
3. Supersession in `extend.py` is for corrections — a wrong word, a withdrawn clip. The task extent
   is not wrong. The solo extent is a *different* question asked of the same region, and the store
   stays readable precisely because both are in it and the refinement names which span it refines.

### What the instrument does not do

- **It takes no decision.** The `localise_speakers` activity writes measurements and one span. No
  assertion, no conformance, no outcome; a test asserts that.
- **It is SPEECH-only.** It runs inside `nodes/speech.py` and nowhere else. AIRWAY and VOICE get
  nothing, for the reason the owner gave: source separation separates voices, and a cough or a
  breath is not a voice it can pull apart.
- **It does not validate the separation.** Every number is as good as MossFormer2's decomposition,
  and no ground truth for who spoke when exists on this corpus.

## The gate

`verdict.gates.by_group.<group>.dominant_speaker_share_min`, `at_least`, reading
`extent_dominant_speaker_share`.

### Which reading, and why not the other two

The owner's rule is "if other speaker is **within the task extent**, flag". Three readings could
have served it:

- **`speaker_count`, whole-file.** Rejected: it cannot tell an examiner's prompt *before* the task
  from an interjection *inside* it. Both read 2. Over the finished corpus this is not a hypothetical
  — 11.3% of the recordings the trigger fires on have their second speaker wholly outside the task
  extent (2305 firings, 2045 with a second speaker inside). A gate on the whole-file count would
  flag those 260 recordings for something that did not happen during the task.
- **`extent_speaker_count`.** Rejected, for the reason the predecessor design gives: a diarizer
  that splits 0.2 s of breath off as a second label takes the count from 1 to 2, and a bound on the
  count fails a clean recording on a segmentation artefact.
- **`extent_dominant_speaker_share`.** Chosen. It is within-extent, so a speaker who only ever
  speaks outside the extent contributes no label and no seconds to it — `_speakers_within` sums
  segment∩extent overlap and nothing else, so the gate is structurally unable to be tripped by one.
  And it degrades continuously, so the artefact case moves it a little and a real turn moves it a
  lot.

The separation's own `extent_secondary_source_s` was *not* made the gate's input, although it is
the richer measurement. It exists only where separation ran, and `speech.separation_backend` ships
null, so gating on it would make the gate inapplicable on every recording today. The gate reads a
number every SPEECH recording with a diarization derivative carries; the separation sharpens the
picture for a reader without being load-bearing for the decision.

### Why it is a flag ground and not a conformance term

A second person speaking during the recording does not mean the participant failed to perform the
instruction. Conformance is about the task; this is about the room. So `dominant_speaker_share_min`
is in `gates.FLAG_GATES`, not in `CONFORMANCE_GATES`, and `gates.py` refuses at import a table that
puts a gate in both. VERDICT applies it beside the conformance gates, records it under
`gates.flagging`, and the fold turns a failure into its own reason — `EXTRA_SPEAKER_IN_EXTENT`,
with the reading and the bound appended. "Pass and combine with the other decisions" is then the
fold's ordinary behaviour: the flag joins every other ground and the triage axis folds them.

An `UNDETERMINED` answer — no reading, or a null bound — is never a flag, which is the fold's
standing rule for every gate.

### Scope, enforced by the keying

The five groups the gate is keyed under — `ORDERED_TOKENS`, `FREE_RESPONSE`, `ITEM_LIST`,
`SYLLABLE_TRAIN`, `SYLLABLE_SEQUENCE` — are exactly the five SPEECH owns, and AIRWAY and VOICE own
none of them. So the config keying is what scopes the gate to SPEECH; no code branch on the node
name is needed, and a VOICE task resolves a group that names no such gate and is gated by nothing.
`default` deliberately carries none: a default would claim a reading that AIRWAY and VOICE never
take.

## The bound, and the fact that it is not fitted

`0.9`. **Unfitted, and declared so in
`specs/20260817-triage-workflow-dag/config-derivations.md` § verdict.gates.**

What the corpus scan can say, over the 62,392 recordings of the replayed run at
`/orcd/scratch/bcs/002/satra/triage_replay_20260922/out/`:

| | |
| --- | --- |
| SPEECH ran | 43,335 (69.5% of all recordings) |
| with an `enhanced_diarization` | 43,334 |
| whole-file speakers > 1 — **the trigger** | 2,305, **5.32% of SPEECH**, 3.69% of all recordings |
| of those, exactly 2 (separable) | 2,303 |
| of those, 3 (backend refuses, reported) | 2 |
| with a within-extent share at all | 38,980 |

and the share's distribution over those 38,980:

| bound | recordings below it | |
| --- | --- | --- |
| 1.00 | 2,045 | 5.25% |
| 0.99 | 1,490 | 3.82% |
| 0.95 | 859 | 2.20% |
| **0.90** | **613** | **1.57%** |
| 0.80 | 358 | 0.92% |
| 0.70 | 210 | 0.54% |

p10, p25 and p50 are all exactly 1.00: on nine SPEECH recordings in ten, one voice holds every
attributed second of the task.

The 555 recordings between 0.99 and 1.00 are the population the continuity argument predicted — a
sliver of a second given to a second label. A bound at 0.99 would flag all of them; 0.9 does not.
That is what 0.9 buys, and it is the whole of what the scan establishes.

**What it does not establish, and what would.** No recording in this corpus carries a label saying
whether a second person was actually in the room, so the 613 recordings the bound flags cannot be
split into true and false. The measurement that would fit the bound is the same distribution
against adjudicated verdicts on that question, and it does not exist. Until it does the number is a
placeholder chosen for its distance from a known artefact mode, not a fitted operating point.

## The bill

22.22 hours of audio over 2,305 recordings — mean 34.7 s, median 24.7 s — every time the corpus is
processed with `speech.separation_backend` set. Per-recording CPU cost is measured by
`msinstr-sep` (job 23478224 on `mit_preemptable`), timing MossFormer2_SS_16K on a sample of the
corpus's own enhanced streams; multiply its median wall-seconds-per-audio-second by 22.22 h for the
corpus figure.

**A cheaper trigger is available and is not taken here.** Narrowing the trigger from "whole-file
speakers > 1" to "within-extent speakers > 1" would drop 260 of the 2,305 firings, an 11.3%
saving, and would skip exactly the recordings where the second speaker never touches the task —
which are also the recordings the gate will pass. The owner specified the whole-file trigger
explicitly, so it stands; the number is recorded here so the trade is visible rather than
rediscovered.

## Tests

| what it pins | where |
| --- | --- |
| separation reads the stream the speakers were counted on, and every edge names it | `speech_test.py::TestTheMultiSpeakerInstrument::test_separation_reads_the_stream_the_speakers_were_counted_on` |
| each separated source's seconds, spans and diarized label inside the extent | `…::test_the_separated_sources_are_localised_inside_the_task_extent` |
| the secondary seconds are a reading of their own, carrying no outcome | `…::test_the_seconds_another_source_holds_inside_the_task_are_a_reading_of_their_own` |
| the refined extent is additive; the task extent survives | `…::test_the_refined_extent_is_a_new_span_and_the_task_extent_survives` |
| the localisation writes measurements and spans only | `…::test_the_instrument_writes_no_decision` |
| a second voice inside the extent flags | `verdict_test.py::TestAnotherSpeakerInsideTheTaskExtentIsAFlag::test_a_second_voice_inside_the_task_extent_flags` |
| a speaker only outside the extent does not | `…::test_a_speaker_only_outside_the_task_extent_does_not_flag` |
| the gate reads the within-extent share, not the whole-file count | `…::test_the_gate_reads_the_within_extent_share_and_not_the_whole_file_count` |
| an absent reading is UNDETERMINED and never a flag | `…::test_an_absent_reading_is_undetermined_and_never_a_flag` |
| the gate is not a term in the task's conformance | `…::test_the_speaker_gate_is_not_a_term_in_the_tasks_conformance` |
| a VOICE task carries no speaker gate at all | `…::test_a_voice_task_carries_no_speaker_gate_at_all` |
