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

## Breath vetoes over the task extent (2026-10-06), replacing the four vetoes

The four vetoes moved 181 recordings from pass to discard. The owner listened to 12
(`~/Downloads/breath_veto_check_20261006/`) and heard breathing in 11. Each reason the owner gave
became a structural rule:

- 6ca9935e, discarded as speech: "breath followed by speech. does not overlap in task extent". Speech
  is now counted only inside the task extent.
- c9b77a28 and 988c1609, discarded as speech on repeated 唉 / 啊 / 呜: these are a recognizer's
  rendering of a sigh. A speech word must not be bracketed, a vocalisation or an interjection
  (`residue.is_non_lexical`), and must be written in the declared language's script.
- 6f73b8d2, discarded as little activity at 0.382 of the file: "task extent is in the first half".
  The active fraction is now read over the task extent (0.841 there).
- b26208dd, discarded as silence: "silence is quiet but breathing is there". The four noise vetoes
  (0b9b68cf, both 01f52a78, 42442f80) also held breathing. Noise and silence no longer veto.
- 167ac3f5, discarded as little activity: "has little breath". Discarding it is acceptable.

The breathing pattern itself stays a whole-file reading. Twelve of the 56 labelled breath
recordings have a task extent of 4 s or less (the AIRWAY event hull). Read over the extent, the measure
turns 25351e19, 772aa876, 80e179b4, 42442f80 and dac345e2 from alternating to single or none, and
7c169ccc from single to none.

On the 56 listened recordings (38 breath, 18 none), the rule is: the measure finds the task's breath
(whole file); fewer than 5 speech words lie inside the task extent; and the active fraction is at least
0.36. The fraction is read over the task extent where that extent is long enough for a modulation
reading (`min_duration_s`, 4 s), and over the file otherwise. The rule agrees on 53 of 56. Refitting
the activity cut on 55 and predicting the held-out recording agrees on 51 of 56.

- **Real breathing discarded:** one, 0dc15213. It has no active span at all (0.0), so no activity
  rule keeps it.
- **No breath kept:** two, 5cc93330 and d5a327c7. The owner heard them as "masked by other sounds or
  very soft", which is the error the owner accepts.
- **The activity cut is narrow:** the nearest breathing recording is 81873ca0 at 0.39, the nearest
  vetoed no-breath recording is ae2a7223 at 0.325, and 0.36 is the midpoint of the zero-error range
  0.34–0.38.
- **Speech:** every breathing recording has 0 speech words inside its extent. The only no-breath
  recording with any is 517381e9, with 8.

YAMNet and HeAR scores over the extent are recorded on the veto for context and decide nothing. Their
absence no longer sends a breath recording to rerun. The parameters are in `data/breath_pattern.yaml`
(`veto`).

Estimated against r15, as an in-memory VERDICT re-fold with each recording's hint:
- **The 181 the four vetoes discarded:** 118 pass, and 63 stay discarded (44 for little activity, 19
  for speech).
- **The 537 the breath-evidence rule moved:** 471 pass, and 66 are discarded (16 of them pass in r15).
- **300 random:** 7 discard → pass, 2 discard → flag, 1 flag → discard.

## Breath-task extent from modulation (2026-10-06)

The owner: "wouldn't task extent for breathing be better from the modulation calculation?" The
breath-task extent was AIRWAY's event hull; on 14 of the 56 labelled breath recordings it was 4 s or
less and understated where the breathing was.

**How it is read** (`measure_breath_extent`, `tighten_to_events`, `breath_extent_fallback` in
`breath_pattern.py`; parameters in `data/breath_pattern.yaml`, `extent`):

1. Sliding 8 s windows, 2 s apart, over the same subband envelopes the modulation reading uses. A
   window breathes when its breathing-over-syllabic ratio is at least 5 dB and at least 15% of its
   frames are 8 dB above the recording's floor.
2. The longest run of breathing windows, bridging one non-breathing window, tightened to its active
   frames. It must hold at least two estimated breaths.
3. Narrowed to the hull of the breathing measure's events inside it, padded by 1 s. The windows place
   the edges coarsely, and without this the extent took in talk just before or after the breathing.
4. Fallbacks, in order: the measure's events, padded by 1 s (`measure_events`); then AIRWAY's own
   hull (`airway_events`). The source is recorded on the span.

**Where it lives.** VERDICT writes it, for breath families, as a `task_extent` span carrying
`supersedes`, which names the branches' live task-extent spans. Those stay live; the readers
(`task_audio.task_extent`, the recording vectors) take the superseding span in their place through
`vocabulary.standing_task_extents`. A re-fold that reads the same extent keeps the span; one that
reads none retires it, and the branches' spans stand again. It applies on a plain re-fold; no replay
is needed. Task audio and the parquet pick it up when they next run.

**The decision is unchanged.** The veto's readings (speech, the classifiers, the active fraction)
stay over AIRWAY's own hull, as they were fitted. Two attempts to read them over the new extent were
measured and dropped:

- **Activity over the new extent** lost three labels (fit 53 → 50 of 56): the extent is tightened to
  its active frames, so a fraction over it is always high, and the little-activity veto no longer
  caught 4d596bce, ae2a7223 or 167ac3f5.
- **Speech over the new extent** discarded 27 more unlabelled recordings than the hull does (29
  decisions moved; 2 rescued), from 5–28 words at the extent's edges. Stricter window thresholds
  (5/7/9 dB, 0.15/0.3 active) left about 25 of them speech-vetoed. Without labels on those, keeping
  breathing is the safer error.

**Measured** (in-memory re-fold of the r15 stores, 1,074 rows, 0 errors; `/orcd/scratch/bcs/002/satra/tmp_bxt/`):
the label fit is 53 of 56, the same three as before, with 0 decisions changed against `1a73d8d0`.

| Set | Source | Median extent, old → new |
|---|---|---|
| 54 labelled with an extent | modulation 26, measure events 21, AIRWAY 7 | 13.2 → 10.6 s |
| 284 random breath recordings with an extent | modulation 171, measure events 75, AIRWAY 38 | 10.6 → 11.9 s |

On the 14 labelled recordings whose hull was 4 s or less, the breathing ones widen (25351e19 2.0 →
30.0 s, 80e179b4 3.7 → 19.7 s, 475ff714 3.7 → 12.1 s, 42442f80 2.4 → 9.0 s, cb69d304 3.6 → 8.4 s,
7c169ccc 0.7 → 3.6 s); b16acf04 narrows (3.7 → 1.4 s, one measured event). 6ca9935e, breath then
speech, ends at 10.8 s instead of 14.0 s. The images are in
`~/Downloads/breath_extent_check_20261006/`.
