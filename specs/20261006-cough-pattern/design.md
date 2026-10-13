# Cough tasks decided on cough onsets

The three cough families (`respiration-and-cough-cough`, 5 instructed; `voluntary-cough`, 3;
`respiration-and-cough-v2-hardcough`, no count) are decided on `cough_pattern.py`. That module reads
cough onsets off PREPROCESS's stored `spectrogram_narrowband`, which is computed over the
pre-emphasised stream. AIRWAY's cough events, HeAR and YAMNet are recorded as context. They are not
gates.

## Why AIRWAY's cough events were not usable

`nodes/airway.py:airway_events` keeps only the carrier spans that `sounds_like` scores as the
family's label (HeAR/YAMNet over `score_min`). Inside a kept carrier it walks the activity envelope
and falls back to the carrier's extent. That causes two failures:

- **A cough the classifiers score low is not found at all.** AIRWAY found 0 events on 53dddba6 and
  c60d8bb8, which hold 5 coughs each, and 0 on b91cd93f (one cough).
- **A scored carrier's envelope walk merges or splits coughs.** It gave 1 event on 45f1c7ec (5
  coughs), 2 on a6de8e15 (5), 8 on 3b7ace99 (3 inhale+cough acts), 21 on 6ca9935e and 5 on
  bffe3a4f, where those events sat in background noise. 4c91e682's fourth event was misplaced.

## The measure

The spectrogram is summed into 9 subbands from 150 to 7800 Hz, with a 15 ms median per band. An
onset candidate must meet all of these:

- a broadband rise of at least 12 dB within 0.1 s;
- at least 60 % of the bands rising by 8 dB or more (coherence; a hum or buzz is narrowband and
  stationary);
- a peak at least 18 dB over the floor within 0.15 s;
- an event of at least 0.08 s;
- no speech word within 0.15 s;
- no rise out of the recording's leading digital silence.

A candidate becomes a new cough only if it meets one of these conditions:

- its pre-onset level (minimum over 0.3 s) is within 6 dB of the floor;
- its rise is at least 0.75 of the previous onset's rise (bunched coughs, 56baa32a).

Otherwise it is that cough's second phase or expiratory tail (45f1c7ec, 4c91e682). The same holds
for a weaker rise within 0.3 s of an onset (aad9e6c6). An onset more than 12 dB below the train's
median peak is not part of the train: 56baa32a's preparatory gasps, and 4c91e682's pre-cough click.

A cough event spans these parts:

1. A preparatory inhale ending within 0.5 s before the onset, held 15 dB over the floor
   (22c5f400, 3b7ace99).
2. The onset.
3. Its decay and any attached tail.

The extent runs from 0.1 s before the first event to 0.1 s after the last. It is cut at the nearest
speech word outside the coughs. VERDICT writes it as the standing task extent, superseding AIRWAY's.

The owner's words, as relayed for each recording:

- 4c91e682: "extent good. events generally good except the penultimate one"
- aad9e6c6: "coughs a little faster"
- 45f1c7ec: "coughs may have expiring breath"
- eab3b2e8: "irregular not captured well"
- 4164d528: "no consistent coughing pattern, but expected cough like patterns"
- b91cd93f: "extent off - single cough"
- 56baa32a: "lots of coughing but irregular patterns"
- 889afc08: "single cough"
- bffe3a4f: "noise at the beginning a few coughs low amplitude at the end"
- 22c5f400: "should have full extent, inhale, cough, a secondary cough/follow through"
- 28cc3eda: "3 coughs. last two muzzled"
- 3b7ace99: "extent and events incorrect (3 inhale + cough events)"
- cdfa7e4e, c7410c7c: "nothing"
- a28e5022: "nothing + background hum/buzz"
- 9c6e4508: "no cough"

## Fold

| Reading | Outcome |
|---|---|
| 0 onsets, input read | discard `no_cough_captured` |
| Spectrogram absent | rerun `owning_branch_input_absent` |
| Fewer than instructed | flag `task_mismatch` ("detected N coughs where M were instructed") |
| v2-hardcough, ≥ 1 onset | pass |
| Decision differs between strict and lenient thresholds (rise ±4 dB, coherence ±0.1) | flag `cough_review_low_confidence` |

When the cough measure decides, these AIRWAY flags are skipped:

- AIRWAY's own task conformance and uncomputed-reading flags;
- the routing mismatch on AIRWAY (4164d528);
- `route_unexplained`, when a cough was found (45f1c7ec). A rerun would re-evaluate the same
  ruleset and reach the same state, and the declared task's coughs explain the content.

A cough family whose measure was not read (VERDICT without a run directory) names no required
event. It falls back to the earlier behaviour and does not discard on AIRWAY's count.

## Label agreement (23 listened recordings, r9 stores)

Every one agrees:

- **Negatives (0 onsets, discard):**
  - the 4 earlier discards: 725c2db2, aa5114f6, f286ac55, f997d0e3;
  - cdfa7e4e, c7410c7c, a28e5022 (hum), 9c6e4508.
- **5 coughs:** 53dddba6, a6de8e15, c60d8bb8, 4c91e682, aad9e6c6 and 45f1c7ec.
- **eab3b2e8:** 5, irregular.
- **v2-hardcough, pass:**
  - 4164d528: 2;
  - b91cd93f: 1, extent 1.22–1.87 s;
  - 889afc08: 1, extent 0–1.57 s, the opening inhale and cough;
  - 56baa32a: 5;
  - bffe3a4f: 3, extent 7.40–8.70 s, late only.
- **22c5f400:** 1 act (inhale, cough, follow-through, no second sharp onset), extent 0–1.4 s.
  Instructed 3, so `task_mismatch`.
- **28cc3eda:** 3, the muffled two included.
- **3b7ace99:** 3 acts, each starting at its inhale.

On the enhanced stream, the same measure agrees with the plain stream on present or absent for all
24 picks. It is used only as a second reading, because enhancement often removes coughs: for
53dddba6, eab3b2e8, 4c91e682 and a6de8e15 the enhanced stream holds under 3 % of the energy.

## Background speech in the task

`background_speech.py` reads the `residual` stream, which is plain minus the gain-fitted enhanced
stream. Either of two readings flags `background_speech_in_task`. The flag never discards.

1. A residual YAMNet window that hears speech (≥ 0.5) and meets three further conditions:
   - the enhanced stream stands ≥ 6 dB over the residual there (the enhancer kept the foreground);
   - the residual stands ≥ 15 dB over its own floor;
   - the residual envelope does not track the enhanced one (correlation < 0.5).
   A residual that tracks the foreground is the participant's own cough leaking. That was 56baa32a
   at r = 0.96, and many r16 coughs at 0.8–0.99.
2. A harmonic run in the residual that meets three conditions:
   - autocorrelation peak ≥ 0.5 over 80–400 Hz;
   - ≥ 10 dB over the residual's frame floor;
   - ≥ 1 s long with ≥ 0.5 s harmonic, and away from the participant's events by 0.3 s.

The phonation track was tried first and rejected, because it is the plain stream's voicing. Breath
turbulence voices it, and it fired on 6 of 56 labelled breath recordings.

On 6ca9935e the intercom ("speech is not clear but it's clear someone is speaking") is the harmonic
residual between the last two cough phases. Residual YAMNet heard Speech 0.93 at 20.2 s, inside a
detected cough event, with r = 0.81. Rates measured on the labelled sets and the r16 sample are in
the job report.
