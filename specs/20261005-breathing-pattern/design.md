# A breath task is decided on its breathing pattern

2026-10-05. Owner: "can the preprocessed spectrogram be used to check if there are alternating
breathing patterns observed alongside durations that make sense for breathing", then, after
listening, "implement the measure".

## Why

The breath-event rules (`specs/20261005-truncated-capture-discard/design.md`, the 2026-10-05 notes)
decided a breath task on AIRWAY's HeAR-scored event detector. On the owner's ten listens that
detector was the best single reading, but it still passed 517381e9, a v2-breath recording with no
breath in it: one burst inside speech counted as a breath event. HeAR Breathe at 0.2 passed three
silent recordings, ASR wrote `[breath]` on three, and Clef 27B on plain spectrograms called six of
ten silent recordings `alternating_breaths`.

## The measure

`breath_pattern.py` reads two PREPROCESS derivatives the store already holds:

- `spectrogram_narrowband` -- power over the pre-emphasised stream, 50 Hz bins, 5 ms hop;
- `phonation_tracks` -- per-frame F0 and pitch strength, for voicing.

A frame is active when the breath-band envelope (150-2500 Hz) is well above the recording's own
floor and the spectrum is noise-like (flatness). Active runs merge into events; an event of a
breath's length (0.3-4 s) voiced over at most half its length is a breath. Two or more breaths at a
breathing rhythm (autocorrelation peak at a 2-8 s period, or most onset intervals 0.4-8 s apart) read
`alternating_breaths`; one reads `single_breath`; none `no_breathing`; several without a rhythm
`irregular_events`. Every parameter and its derivation is in `data/breath_pattern.yaml`.

## The rules

For every breath family in `data/airway_event_requirements.yaml`, VERDICT reads the measure and it
decides the task in place of AIRWAY's detector:

| Family | Performed | Otherwise |
|---|---|---|
| sustained (`respiration-and-cough-breath`, `-v2-breath`) | `alternating_breaths` | discard `no_breath_captured` |
| counted (fivebreaths, threequickbreaths, v2-threebreaths*, breath-sounds) | at least the instructed number of breath events | 0 events: discard `no_breath_captured`; fewer than instructed: flag `task_mismatch` ("detected N breath events where M were instructed") |

Either derivative absent: rerun, `owning_branch_input_absent`. AIRWAY's own conformance, its
uncomputed readings and its HeAR coverage are not a flag on these families; its deviations still
are. The reading is carried on the verdict as `breath_pattern`.

## Agreement with the owner's ears

On the 23 recordings the owner listened to, the measure read every one as heard: alternating on the
ten confirmed breathing, single on 7c169ccc (one long breath where three were asked) and 517381e9
(the burst in speech), none on the silent ones and on the quiet 9d16c147.

In-memory re-fold (VERDICT only, r9 stores, nothing written; /orcd/scratch/bcs/002/satra/tmp_bpest/):

- Labelled 22: all as heard. 7c169ccc flags `task_mismatch`; the ten silent ones and 517381e9 discard
  (`no_breath_captured`, or `too_short_for_task` for the two sub-second fragments); nine of the ten
  confirmed breathing pass, and 39ba8784 (fivebreaths, 3 breaths measured where 5 were asked) flags
  `task_mismatch`.
- Random 300 breath-family recordings: pass→pass 180, pass→flag 34 (task_mismatch, counted families),
  pass→discard 30 (29 `no_breath_captured`, 21 of them respiration-and-cough-breath), flag→discard 33
  (27 too short, 6 no breath), flag→pass 10, flag→flag 9, rerun 4. Patterns: alternating 214, none 48,
  single 27, irregular 4, unread 7. `task_mismatch` on 46 rows overall.

## Effect on the run

The measure reads stored derivatives, so it applies on a plain re-fold. The breath-event replay
staged for the sustained families (`triage_r9_20260929/fix_breath/replay.sbatch`) only existed to make
AIRWAY run its detector on them; with the measure deciding, it is not needed.
