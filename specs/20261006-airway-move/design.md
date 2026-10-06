# The AIRWAY move: branches measure, VERDICT decides

Owner decision, 2026-10-06 ("yes"): move breath, cough and background-speech measurement out of
VERDICT and into the AIRWAY branch. Recorded in `specs/20260925-review-pass-resume/resume.md`
("Owner decisions not yet in code", items 2–4).

## What moved

Until this change VERDICT's `_task_evidence` called `breath_pattern_of`, `cough_pattern_of` and
`background_speech_of` itself and minted the breath or cough task extent (`_settle_breath_extent`).
That was a measurement taken inside the node that is supposed only to decide.

Now, for every declared airway family listed in `data/airway_event_requirements.yaml` (all eleven
AIRWAY families: eight breath, three cough), the AIRWAY branch in align mode
(`nodes/airway_task.py`, called from `nodes/airway.py:airway`):

1. reads the breathing pattern (breath families) or the cough onsets (cough families) and writes
   one measurement, `airway_breath_reading` or `airway_cough_reading`. It holds either `absent`
   (the derivatives the measure could not read) or the reading: `pattern`, `events_n`,
   `vetoed_by`, `train_breaths`, `review` for breath; `onsets_n`, `review` for cough; and the full
   `reading` record in both;
2. writes the measure's extent as the standing task-extent span (`supersedes` the branch spans),
   retiring any earlier superseding span, including the ones VERDICT minted before this change;
3. reads background speech over that extent with the measure's own events, and writes
   `airway_background_speech`.

VERDICT reads those three measurements and nothing else of the measures. The decision rules in
`vocabulary.fold_file_verdict` are unchanged by the move itself.

The review-band membership (`review`) is computed in AIRWAY, because it is a property of the
reading (strict and lenient re-reads on opposite sides of the instructed count, or the train's
confidence), not a decision; VERDICT turns it into the `*_review_low_confidence` flag.

## What AIRWAY's HeAR-gated detector still does

The event-series, alternation and coverage matchers still run in align mode. They feed the
`events_min`, `instructed_count_min_fraction` and coverage gates, whose readings are carried into
`recording_vectors` as `gate_*` columns, and nothing else: for a measure family the fold already
skipped AIRWAY's conformance and uncomputed-gate grounds (`measure_decides`), overrode AIRWAY's
found state with the measure's, and replaced the owner's absent inputs with the measure's. So the
detector no longer places the extent and no longer decides; it is a reported reading only. Its
hull span stays as the span the measure's extent supersedes.

## Inputs at AIRWAY time

The measures read PREPROCESS's derivatives (narrowband spectrogram, phonation tracks, streams,
YAMNet windows) and the consensus words, all present before AIRWAY runs. One input changed: the
fallback hull (`task_extent_bounds`) used by the breath veto, the breath extent fallback and the
background-speech extent, where the measure placed none. At VERDICT time it was the hull of every
branch's task-extent span; at AIRWAY time it is AIRWAY's own. `breath_pattern_of`'s docstring
already said "AIRWAY's own hull, as they were fitted", so this is the fitted behaviour. A recording
where SPEECH or VOICE also wrote a task-extent span can read differently; the airway replay will
show how many.

## An unmeasured store

A store AIRWAY reported on but whose reading is missing (every store written before this change)
names `AIRWAY:airway_breath_reading` or `AIRWAY:airway_cough_reading` as an owner absent input, so
a re-fold without the replay reruns those recordings rather than deciding them on nothing. Where
AIRWAY did not report at all, nothing of AIRWAY's is absent.
