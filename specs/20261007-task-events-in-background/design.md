# Task events in background: one measurement shape for breath, cough and phonation

Status: design, 2026-10-07. No code yet. Code citations are at fix/policy-v8 `a8a95cea`.

## Why

From 2026-10-05 to 2026-10-07 the owner listened to about 140 recordings: breath, cough, voice,
contested discards and the r17 breath review band. Each listen produced a local fix, such as an
edge phase, a speech split, a floor, a veto or a band cut. The owner's direction on 2026-10-07:

> don't overoptimize on specifics. determine the more general components that apply across
> situations.

Nearly every failure was one of a few general mistakes:
- reading the background as the task, or the task as the background;
- rejecting real events by a single feature;
- requiring a rhythm the recording did not have;
- letting the extent run into the quiet tail;
- letting a secondary measure overrule the primary one.

This spec defines the shared components. Breath, cough and phonation each become a thin layer over
them.

## Failure classes found by listening

The 8-hex ids below are examples, not a test list. Every row of the label files is scored (see
Evaluation).

| Class | What happened | Examples |
|---|---|---|
| F1. The floor is the task | The floor is a percentile of the recording itself, so a task that fills the file sets its own floor | 685fb824, 68d62b1e, eaa742f4 (VOICE cause A) |
| F2. The background is the task | Phases were counted on stationary noise wobble, hum or clicks | 67c60a9f (knocks), 3bbc69ef (hum), 11ec42cc (click at the file start), 3d446af4 |
| F3. Task events dropped | Long, loud phases, slow breathing or high-band-only breathing were missed; events under speech-like ASR words were removed | 19c5c847, 75b49def, 5201d61d |
| F4. Rhythm treated as a requirement | Irregular but real breathing went to review; a single breath was overcounted | cbe8a315, 19c5c847, 1965766f |
| F5. The extent runs past the task | Low "phases" in the quiet tail stretched the extent; the preparatory inhale was missed | 2bb59c22, 1965766f; fa6befa4, e421c733 (inhale) |
| F6. A secondary measure overruled the primary | The older measure's `little_activity` veto discarded clear trains | 1f4ea26f, 793732ff, 2bb59c22 |
| F7. Swamped but present | Real breaths under heavy background; the right answer is review, not discard | fc8a8729, 6330763d, 16eded62 |
| F8. Tracker or stream artefacts read as absence | The pitch tracker found no voicing on low or hoarse voices; enhancement removed quiet task audio | c35284b1, 0581355f (cause B); 7cd07b02 (enhanced truncation) |

No single feature (rise, cycle CV or duration) separated present from absent across these. For
example, among the contested discards a 14.8 dB train was noise (3d446af4) while a 14.3 dB one was
a real breath (793732ff). The separation has to come from several pieces of evidence used together.

## The components

### C1. Background model (shared)

There are two parts:
- **A stationary floor:** per band, a slowly varying level, estimated *outside* candidate events.
  - The session group's floor is used where the session has enough recordings, falling back to the
    recording's own quiet frames.
  - Tonal lines (mains hum and its multiples, measured on the residual) are part of the floor, not
    events.
- **An impulse detector:** short broadband transients (clicks, knocks, handling) are marked as
  background events. An impulse has an attack of at most a few ms, a duration of tens of ms, and no
  sustained noise body.
  - Quick breaths (about 0.25 s with a noise body, 2bb59c22) are not impulses.
  - The test therefore uses attack slope and duration together, never duration alone.

The floor is estimated iteratively: first pass, detect events, re-estimate the floor without them.
That removes F1 by construction.

Current code, to be replaced by this one component:

| Measure | Current floor |
|---|---|
| breath | per-bin percentile over the whole recording (`breath_pattern.py:807-816`), plus a broadband percentile floor (`:950-951`) |
| cough | percentile of non-silent frames (`cough_pattern.py:270-271`) |
| voice | quietest 100 ms (`voice_phonation.py:144-164`) |
| background speech | percentile of residual windows (`background_speech.py:303-304`) |
| hum | `voice_phonation.py:293-327` (residual mains lines), already the right shape; it moves into C1 and serves all three measures |

### C2. Event detection (shared)

An event is a contiguous region where:
- the level exceeds the C1 floor by margin `m` in at least a fraction `q` of bands;
- it has a gradual onset and offset, not an impulse (C1);
- it is not inside a speech region.

Speech regions come from acoustic speech segments plus ASR words placed at their consensus extent.
A word whose recognisers disagree on timing does not spread (`breath_pattern.py:1074`, already
done).

Events carry evidence, not verdicts: SNR over the floor, bands covered, duration, onset slope,
harmonicity, and voiced fraction.

The current detectors are three variants of this one component:
- breath bursts: peak finding on a combined robust-z envelope with a duration window, flatness,
  coherence and template gates (`breath_pattern.py:911-1071`);
- cough onsets: a coherent rise across bands with tails attached (`cough_pattern.py:264-323`);
- phonation runs: level over floor or voiced, with bridged runs (`voice_phonation.py:368-424`).

What stays task-specific is the event *type* test layered on top:

| Task | Type test |
|---|---|
| breath | noisy (flat spectrum), unvoiced |
| cough | sharp coherent onset plus noisy tail |
| voice | harmonic, with f0 tracked over 50–1600 Hz and octave-checked |

### C3. Rhythm as a prior (breath, also glide and hold cadence)

Where the subband modulation spectrum has a breathing-band peak above prominence `r`, its period
does two things:
- it groups events into a train;
- it predicts phases a weaker detection should look for at about that period, across all bands,
  including high-band-only slow breathing (75b49def).

Where there is no peak, events stand as found:
- a single breath is one breath;
- irregular breathing is not penalised.

Rhythm strength is evidence for C5. Its absence never turns a clear event into review (F4).

The modulation reading exists (`breath_pattern.py:776`) but feeds only the old measure, not the
train. The train infers its cycle from burst gaps (`breath_pattern.py:1149-1155`), and
`in_review_band` reviews on cycle CV (`breath_pattern.py:420-435`). Both are replaced.

### C4. Extent (shared)

The extent runs from the first to the last event of the dominant cluster, plus the preparatory
inhale. The dominant cluster is the events within a rhythm-scaled gap, or a fixed gap without
rhythm.
- The inhale back-trace is the same rule everywhere: the level rising over the floor before the
  first event (`cough_pattern.py:326`, `voice_phonation.py:432-448`).
- No padding into a quiet tail.
- No extension to "following speech" unless the level between stays over the floor.

Today `breath_train` pads (`breath_pattern.py:1165`) and continues to following speech
(`:1171-1175`). `train_extent` also widens to the old measure's events (`:1241-1260`), which is
F5.

Broken or restarted holds merge, as already done for voice (`voice_phonation.py:411-420`); the same
rule serves breath.

### C5. Evidence to decision (shared; VERDICT reads it)

Each task measurement reports a small evidence vector:
- best-event SNR over the C1 floor, and median event SNR;
- event count and bands covered;
- rhythm strength (C3);
- raw-versus-enhanced agreement: the same events, or an extent within tolerance, on both streams;
- background burden: impulse rate and tonal lines inside the extent.

The decision is one rule for every airway and voice family:

| Condition | Decision |
|---|---|
| A clear event: SNR ≥ `s_hi`, on the raw stream or on both streams | present |
| Events exist, but SNR is in [`s_lo`, `s_hi`), or the streams disagree, or the background burden is high | review |
| Nothing exceeds `s_lo` on any stream | discard (no event captured) |

- The old measure's vetoes (`little_activity`, speech) become reported measurements.
  `breath_present` falling back to `_measure_found` (`vocabulary.py:377-398`) goes.
- `discard_contested` (`vocabulary.py:1754-1768`) is no longer needed as a separate rule. A discard
  can only happen when no stream shows an event above `s_lo`, so AIRWAY's detector contradicting it
  becomes a review condition inside C5, not a second-pass flag.
- The review band is now a band on evidence (F7), not on rhythm, and has one definition across
  families.

### C6. Instruction comparison (annotation only)

Count, pattern and direction are compared with the declared instruction and recorded as
`task_mismatch` annotations:
- breaths instructed against breaths found;
- coughs instructed against coughs found;
- glide direction, and held versus glided.

They never change a verdict. This is already the rule for airway (`8864ed00`) and voice
(`7075d730`); it moves into the shared reader.

## Shared versus task-specific

| Component | Shared | Task-specific layer |
|---|---|---|
| C1 floor, impulses, hum | all | none |
| C2 event detection | all | event type: breath noisy and unvoiced; cough onset plus tail; voice harmonic with f0 |
| C3 rhythm | breath trains; voice hold cadence | the breathing band (0.1–1.2 Hz) |
| C4 extent | all | inhale rule identical; voice adds holds and breaks (`voice_phonation.py:451`) |
| C5 decision | all, one rule | none, apart from which streams exist |
| C6 instruction | all | which instruction field is compared |

Background speech (`background_speech.py:274-331`) is a C2 event detector run on the residual with
a speech type test. It reports into C5's background burden, plus its own flag.

## QUALITY as the acquisition-quality branch

**What QUALITY does today** (r17, review 2026-10-07):
- **Its only check** audits PREPROCESS's clip spans for consistency (`nodes/quality.py:184-329`). It
  contested 0 recordings in r17; PREPROCESS had already withdrawn contradicted spans on 286.
- **Measures that decide nothing:** SQUIM (STOI/PESQ/SI-SDR), the whole-file disruptions, level, the
  band profile, and the parquet's floor and SNR.
- **The `quality:` config is part null, part mis-described.** `stoi_floor`, `pesq_floor` and both
  disruption maxima are null, with no reader. The section comment calls the whole section unread,
  which is wrong for `clip_contradiction_margin` and `clip_edge_guard_samples`, which are read.
- **The parquet's `q_raw_issues`** (−50 dBFS floor, 25 dB SNR, never derived) marks 20,672 r17 passes
  as unresolved; its SNR is structurally low for quiet airway tasks.
- **The two clipping readings disagree:** 9,692 recordings keep a clip span, but 595 have whole-file
  clipped seconds.
- **The quality problems the owner raised live elsewhere:** background speech in AIRWAY, shutoff and
  hum in VOICE, and session-relative level nowhere.

**Proposal.** QUALITY owns acquisition quality as measurements, and VERDICT decides:
- **Recording level:** the C1 background model (stationary floor, hum, impulses); shutoff;
  dropouts and clipping, as one reading; and level against the session group.
- **Per task span:** in-span interference (background speech, generalised from
  `background_speech.py` to every family) and the event SNR over the background, which is C5's
  evidence.
- **Moved in:** `background_speech_in_task`, shutoff detection (`voice_phonation.py`) and the residual
  hum guard. The branches read them from QUALITY.
- **Retired or fixed:** the null tolerances, the wrong comment, `q_raw_issues` (recomputed from C1 or
  dropped), and the dual clipping reading.
- **Kept:** the clip audit, as a self-check on the store rather than a judgement of the recording.
- **Fitted from labels before anything flags:** in-span interference (the cough and breath
  background listens), shutoff, and session level (the 2026-10-05 empty-route and contrast
  listens). SQUIM only if a labelled sample shows it separates usable recordings from unusable ones.

## The DAG

The owner's rule: every recording goes through QUALITY, and a recording a branch takes also gets
that branch's outputs alongside.

```
PREPROCESS → TAXONOMY → routing ─┬─ AIRWAY ─┐
                                 ├─ SPEECH ─┤
                                 └─ VOICE  ─┤
          QUALITY (recording level) ────────┤   every recording, no branch input
          QUALITY (per task span) ◀─────────┘   over whichever branch extents exist
                                   → REDACT → VERDICT
```

- **QUALITY runs on every recording, whatever routing decided.** It already runs unconditionally
  after the branches (`run.py:355`).
- **The recording-level part needs no branch output.** It also covers recordings no branch takes.
- **The per-span part runs after the branches**, over the task extents they produced. With no extent
  there is no per-span reading.
- **AIRWAY and VOICE** each write one reading carrying the C2–C4 results (`nodes/airway_task.py:157`).
  `settle_task_extent` (`nodes/airway_task.py:106`) is unchanged.
- **VERDICT reads QUALITY plus whatever branch readings exist, side by side**, and applies one rule:
  flag a quality problem only when it falls inside a task span, and discard only when nothing was
  captured. A recording with no branch reading is decided on QUALITY alone, which can discard it
  when nothing rises above the background.

## Parameters (fitted jointly; all in `data/`)

There are about eight, shared across families:
- floor margin `m` and band fraction `q` (C2);
- impulse attack slope and maximum duration (C1);
- rhythm prominence `r` (C3);
- `s_lo` and `s_hi` (C5);
- stream-agreement tolerance (C5).

The task-specific type tests keep their existing fitted values. Nothing is tuned per recording.

## Evaluation protocol

**Labels:**
- every row of `~/Downloads/triage_listening_labels_20261006.csv`, covering all listen sets
  (breath, cough, voice, discard-cause, contested, empty-route, contrast, veto, modspec);
- `~/Downloads/breath_review_check_20261006/index.csv`;
- each mapped to a three-way target: present, review-acceptable or absent. Owner phrases such as
  "OK if flagged or discarded" map to review-acceptable;
- only 8-hex prefixes are written into the repo.

**Split:**
- per listen set, stratified by target and grouped by subject, so that one subject's recordings
  fall on one side;
- 70% fit and 30% held out, `numpy.random.default_rng(20261007)`;
- the split file is written once and committed beside the evaluation script. Labels added later go
  to held-out first.

**Fit:**
- the C1–C5 parameters, jointly, on the fit side only;
- objective: agreement, where present→discard and absent→present cost 3 and anything→review costs 1.

**Report:**
- in-sample and held-out agreement per listen set and per target;
- the confusion matrix;
- the review-band rate on a random sample of 200 kept recordings per family (target ≤5%);
- the change in discard and flag counts against r17 on the full manifest.

**Acceptance:**
- held-out agreement is no worse than in-sample by more than one recording per set;
- no owner "present" label is discarded on either side;
- the review rate is within target for every family.

**Stores:** the stored derivatives on ORCD. The replay needs no PREPROCESS stage, as with
`voice_phonation`, which tracks pitch inside the branch.

## Dependencies

- Enhanced-stream cross-check: the analysis is pending, ORCD job 25116391, which runs the breath
  train on raw, enhanced and residual for the r17 breath-review sample. Whether enhanced recovers
  swamped breaths (F7) or loses quiet events (F8) decides how much weight raw-versus-enhanced
  agreement carries in C5. Until then C5 uses raw and treats disagreement as review.
- The session-group floor needs a session index over the corpus. The r17 parquet has
  session-level fields; the group floor is computed once per session and stored.

## Open questions for the owner

1. **Review-acceptable labels:** may "OK if flagged or discarded" and "leave as contested" be
   scored as review-acceptable? The evaluation counts either outcome as correct for them.
2. **Background burden:** should a high impulse or tonal burden alone send a clear event to review,
   or only when the events are weak?
3. **Session-group floor:** use it when available, or keep floors per recording for
   reproducibility of single-file runs?
4. **Review rate:** is ≤5% review per family the right target, or should it differ for open-ended
   breath tasks?
5. **Retiring the old breathing measure:** is it acceptable to retire it from decisions entirely
   (C5), keeping it as a reported measurement?
6. **In-span interference:** should it flag every family, speech and voice included, or start with
   airway and voice?
7. **Session-group level:** what defines a session group (the BIDS session)? And should "only the
   surroundings were recorded" discard, or go to review?
8. **SQUIM:** retire it, or keep it as a reported measure pending a labelled test?
9. **Dropout or clip inside a task span:** should it flag, and from what duration?
