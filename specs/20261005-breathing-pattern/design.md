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

## Modulation cycles (2026-10-05)

Owner, asking for it: "perhaps also consider how detecting cycles over subband envelopes using a
modulation spectrum could help detect multiple breathing cycles". After listening to the six
`task_mismatch` recordings with the largest gap between modulation peaks and events
(~/Downloads/breath_modspec_check_20261005/): "seems like a rough doubling between instructed and
modulation peaks". Five-breath tasks gave 10, 11 and 11 peaks; three-breath tasks gave 7, 6 and 7. An
inhale and an exhale each make a burst, so `peaks_per_breath` is 2, and an odd count rounds half up
(the unpaired burst is still a breath; 39ba8784 has 9 peaks for five breaths).

The reading (`measure_modulation`) takes the same stored pre-emphasised `spectrogram_narrowband`:
seven subbands over 150-4000 Hz, log envelopes at 20 Hz, their modulation spectra, and the energy in
the breathing band (0.1-1.2 Hz) against the syllabic band (2-8 Hz) speech occupies. It counts peaks of
the breathing-band component inside the span where the broadband envelope stays 8 dB above its floor.
Prototype and its evaluation: /orcd/scratch/bcs/002/satra/breath_modspec/.

What it decides:

- Speech guard: below 3 dB breathing-over-syllabic the recording is not breathing, whatever the
  events say. On the labels 517381e9 (the burst in speech) reads +2.8 dB; the confirmed breathing
  recordings read +8.6 to +16.0 dB and the silent ones -4.3 to -1.4 dB.
- Counted tasks with fewer events than instructed pass when the reading shows breathing (at least
  7 dB over a non-zero active span) and its estimated breaths reach the instruction. Otherwise
  `task_mismatch` stays, naming both counts.
- Presence stays with the event measure, which handles single and quiet breaths. The cross-band
  coherence and surrogate tests of the prototype decide nothing: silent recordings passed them.

Effect, in memory on the 22 labelled and the 300-sample rows against 8f229ee0
(/orcd/scratch/bcs/002/satra/tmp_modest/): every labelled recording as heard, 39ba8784 now passing on
its cycles; `task_mismatch` 46 → 34 rows, 9 cleared on cycles (1ba3214d, 3c6a97e9, 670e8db6 among
them) and 3 discarded by the speech guard; 8 more recordings discard by the speech guard (7 of them
passed before, with 4-6 events at -1.2 to +2.7 dB). Those eight are unlabelled and want listening.

Open: 1ba3214d is five clean cycles the event measure merged into one event. The event measure's
0.25 s `merge_gap_s` or its +8 dB `rise_db` wants revisiting with labels; it is unchanged here.

## Breath evidence (2026-10-05), replacing the speech guard

The owner listened to the 11 recordings the 3 dB speech guard discarded
(`~/Downloads/speech_guard_check_20261005/`) and heard breathing in 5 of them: 01 3d889bf8, 02 dac345e2,
06 0bcffa40, 08 b16acf04, 09 b451fe70. The other 6 "either don't have it or masked by other sounds or very
soft". The breathing-over-syllabic ratio overlapped across the two (heard: −2.1 to +2.7 dB; not heard:
−1.0 to +2.9 dB), so it cannot be the guard, and it is removed.

On all 32 listened breath recordings (the 22 earlier labels and these 11, with 0b8dcad5 in both sets),
the stored classifier windows separate them:

| Feature | Heard (16): lowest | Not heard (16): highest |
|---|---|---|
| YAMNet `Breathing`, mean over the file's windows | 0.020 | 0.023 |
| HeAR `Breathe`, highest window | 0.214 | 0.728 |

Neither feature separates alone, but together they do. "Breath heard" means YAMNet's mean is at least
0.03, or HeAR's highest window is at least 0.8. That reproduces all 32 labels, and leave-one-out
refitting of both cut-offs agrees on 29 of 32.

- **0.8 on HeAR:** the heard recordings that rest on HeAR alone score 0.82–0.99; the not-heard recordings
  reach at most 0.73.
- **0.03 on YAMNet:** the heard recordings with HeAR below 0.8 (dac345e2 0.045, 0bcffa40 0.244, 39ba8784
  0.323) are above it; every not-heard recording is at or below 0.023.

A breath task now holds a breath only where the measure finds one and a classifier hears one. If neither
classifier's windows are stored, the recording reruns as `owning_branch_input_absent`.

The parameters are in `data/breath_pattern.yaml` (`evidence`). The modulation reading still records its
ratio and still counts cycles for counted tasks. Its cycles count where the ratio reaches 7 dB, and also
wherever a classifier hears breath over a non-zero active span, because the ratio misreads heard breathing
too: b16acf04 has 1 event, 5 peaks, 3 breaths and +1.8 dB, against 3 instructed.

## Breath vetoes (2026-10-05), replacing the breath evidence

The breath-evidence rule moved 537 recordings from pass to discard. The owner listened to 12 of them
(`~/Downloads/breath_evidence_check_20261005/`): "except for this [v2-breath 4d596bce] all the other
ones have respiratory events". 2337c1e6 (YAMNet Breathing 0.001, HeAR 0.026) and 324cd5d0 (0.000, 0.035)
breathe; 4d596bce (0.000, 0.014) does not. Classifier breath scores cannot decide presence, so they are
context only.

The measure decides presence (sustained: an alternating pattern; counted: at least one event), and a
breath it finds stands unless the recording shows positive evidence it is not breathing. On all 44
listened breath recordings (27 breath, 17 none), the measure alone agrees on 37; the seven it keeps
wrongly are each vetoed by one of four readings, and no breath recording is:

| Veto | Fires at | No-breath recordings it catches | Nearest breath recording |
|---|---|---|---|
| speech | consensus lexical words ≥ 5 | a03b5325 (10), 517381e9 (48) | ba1d1459 (3 words) |
| noise | mean highest of Vehicle/Car/Engine/Mechanical fan/White noise ≥ 0.4 | ae2a7223 (0.48) | 772aa876 (0.35) |
| little activity | active span / duration < 0.4 | 4d596bce (0.12), c7c405c7 (0.02), 430950c2 (0.02) | fac74f45 (0.71) |
| silence | YAMNet Silence mean ≥ 0.7 and breath-vs-syllabic < 3 dB | 5cc93330 (0.76, 1.2 dB), d5a327c7 (0.74, 1.8 dB) | 324cd5d0 (0.998, 3.7 dB) |

The rule reproduces all 44 labels; refitting the five cut-offs on 43 and predicting the held-out one
agrees on 44 of 44. No clear breathing is discarded. The thinnest margins are the noise cut (0.35
against 0.48) and the silence ratio (3.7 dB against 1.8 dB, with 324cd5d0 the only quiet breather above
0.7 silence); without the silence veto the rule still discards no breathing but keeps 5cc93330 and
d5a327c7, which the owner heard as "masked by other sounds or very soft". The parameters are in
`data/breath_pattern.yaml` (`veto`); a recording without YAMNet's windows reruns as
`owning_branch_input_absent`.
