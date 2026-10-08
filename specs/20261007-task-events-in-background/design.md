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

## Three layers: generic, task, join

Every detection falls into one of three layers. The components below (C1–C6) implement them; the
later sections refer here rather than repeat it.

**1. Generic layer: every recording, no task knowledge** (QUALITY recording level, plus what
PREPROCESS and TAXONOMY already compute). Its output is activity regions and their properties plus
faults and interference, labelled "something", never "a breath" or "speech".

| What | How it is detected | Exists today? |
|---|---|---|
| Stationary floor | per band, iteratively from frames outside active regions; session-group floor alongside (C1) | per recording only; fails when the task fills the file |
| Hum | stationary narrow lines at mains multiples (50/60 Hz) on the residual (C1) | VOICE only |
| Impulses (clicks, knocks) | attack slope and short duration together, across bands (C1) | no |
| Active regions | energy held above the floor across several bands, gradual onset and offset (C2) | partly: amplitude spans on a 5th-percentile floor |
| Acquisition faults | shutoff (all-band drop to a flat digital floor), dropouts, clipping, discontinuities | measured, unused; shutoff VOICE only |
| Level against the session | recording level relative to its BIDS session | no |
| Streams | raw, enhanced, residual; residual activity; harmonic residual runs (another voice) | yes |
| Labels and words | YAMNet per stream; ASR words and timings | yes |

**2. Task layer: inside each branch.** The branch tests active regions for its own event type and
outputs task events and the task extent (first to last task event plus the preparatory inhale, C4).

| Branch | Task event test | Task-specific extras (annotations, C6) |
|---|---|---|
| Breath | broadband noise body, smooth envelope, ≥ a few hundred ms | modulation rate as a rhythm prior to group phases and recover missed ones (C3); instructed count |
| Cough | sharp rise across most bands, then a tail; preparatory inhale attached | count against instruction |
| Voice | continuous phonation with harmonic structure; f0 on plain with an octave check | glide direction and range; holds and breaks |
| Speech | words aligned to the stimulus or prompt | stimulus conformance; PII |

**3. Join: QUALITY per task span, after the branches, by time overlap.**

| Situation | Rule |
|---|---|
| Interference (another voice, non-task sound, impulse) | flags only where it overlaps or abuts a task event or span; elsewhere an annotation |
| Fault (dropout, clip, shutoff) | inside the span flags; outside, annotation |
| Weak event | its SNR over the background in its own time window (C5); weak → review |
| Streams disagree | raw and enhanced give different task events → review |
| Nothing captured | no active region in any stream → discard; needs no branch, so it covers unrouted recordings |
| Activity, but not the task | active regions exist, none of the declared type → the branch reports task absent or mismatch |

VERDICT reads all three layers and decides with one rule (C5).

**New versus reused.** New: impulse detection, active regions against a proper floor, session level,
shutoff for every family, and the time-overlap join for interference. The task tests largely exist
(breath train, cough onsets, voice phonation), but each carries its own floor and noise handling;
those move into the generic layer.

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
- background entanglement: impulses or tonal lines overlapping or abutting an event. Background
  elsewhere in the file is reported but does not enter the event's decision.

The decision is one rule for every airway and voice family:

| Condition | Decision |
|---|---|
| A clear event: SNR ≥ `s_hi`, on the raw stream or on both streams | present |
| Events exist, but SNR is in [`s_lo`, `s_hi`), or the streams disagree, or a background impulse or tone overlaps or abuts the deciding event | review |
| Nothing exceeds `s_lo` on any stream | discard (no event captured) |

- The old `breathing_pattern` measure leaves decisions entirely: its modulation cycle count, its
  `alternating_breaths` pattern and its `little_activity` and speech vetoes, which `breath_present`
  reads through `_measure_found` (`vocabulary.py:377-398`) to split review from discard inside the
  breath-train review band (c521286e). Decisions use only task-event evidence: active spans, events,
  the modulation rate as a prior (C3), and event SNR over the background. The old measure stays a
  reported column at most.
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

By component; the per-branch event tests are in "Three layers".

| Component | Shared | Task-specific layer |
|---|---|---|
| C1 floor, impulses, hum | all | none |
| C2 event detection | all | event type: breath noisy and unvoiced; cough onset plus tail; voice harmonic with f0 |
| C3 rhythm | breath trains; voice hold cadence | the breathing band (0.1–1.2 Hz) |
| C4 extent | all | inhale rule identical; voice adds holds and breaks (`voice_phonation.py:451`) |
| C5 decision | all, one rule | none, apart from which streams exist |
| C6 instruction | all | which instruction field is compared |

Background speech (`background_speech.py:274-331`) is a C2 event detector run on the residual with
a speech type test. It reports into C5 as background entanglement where it overlaps or abuts an event, plus its own in-span flag.

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

**Proposal.** QUALITY owns acquisition quality as measurements, and VERDICT decides. Its recording
level is the generic layer and its per-span part is the join (see "Three layers"); in addition:
- dropouts and clipping become one reading;
- in-span interference generalises `background_speech.py` to every family.
- **Moved in:** `background_speech_in_task`, shutoff detection (`voice_phonation.py`) and the residual
  hum guard. The branches read them from QUALITY.
- **Retired or fixed:** the null tolerances, the wrong comment, `q_raw_issues` (recomputed from C1 or
  dropped), and the dual clipping reading.
- **Kept:** the clip audit, as a self-check on the store rather than a judgement of the recording.
- **Decision rules (owner, 2026-10-07):** in-span interference flags in every family; a dropout or
  clip inside a task span flags, and outside it is an annotation; with the BIDS session as the
  group, "only the surroundings recorded" discards when no task event rises above the background
  in any stream, otherwise review. SQUIM is reported only.
- **Fitted from labels before anything flags:** in-span interference (the cough and breath
  background listens), the in-span dropout/clip duration, shutoff, and session level (the
  2026-10-05 empty-route and contrast listens). SQUIM decides only if a labelled sample shows it
  separates usable recordings from unusable ones.

## The DAG

The owner's rule: every recording goes through QUALITY, and a recording a branch takes also gets
that branch's outputs alongside. The layers are in "Three layers"; this section places them.

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
- the review-band rate on a random sample of 200 kept recordings per family, reported as an outcome,
  not a target;
- the change in discard and flag counts against r17 on the full manifest.

**Acceptance:**
- held-out agreement is no worse than in-sample by more than one recording per set;
- no owner "present" label is discarded on either side.

There is no review-rate target; parameters are never tuned to move the review rate.

**Stores:** the stored derivatives on ORCD. The replay needs no PREPROCESS stage, as with
`voice_phonation`, which tracks pitch inside the branch.

## Dependencies

- Enhanced-stream cross-check (ORCD job 25116391, done): enhanced passes noise as readily as it
  recovers swamped breaths, so it never decides presence. C5 uses raw, treats disagreement as
  review, and attaches enhanced-derived spans as a reviewer hint.
- The session-group floor needs a session index over the corpus. The r17 parquet has
  session-level fields; the group floor is computed once per session and stored.

## Unit A: the generic layer (built)

Code: `background_model.py`, parameters `data/background_model.yaml`, written by QUALITY as the
`background_model` measurement (signal `plain`) on every recording; nothing decides on it yet.
QUALITY runs whatever routing selected (`run.py`, after the branch loop), so unrouted recordings get
it too. Where the `plain` stream is absent the measurement records `missing: ["plain"]`.

**Floor.** Per band (the breath train's nine subbands, 150–7500 Hz), a first pass takes the 10th
percentile of live frames; active frames (C2 test), widened by 0.1 s, are removed and the floor is
re-read as the median of what is left, up to four times. Two cases cannot be read off the
recording's own quiet frames:
- under 0.5 s of quiet frames;
- the "quiet" frames are the task itself (685fb824, 68d62b1e: a vowel filling the file). The
  residual tells: there the residual's 20th percentile sits ≥10 dB under the own floor in at least
  half the bands, because the residual (`plain − g·enhanced`, same scale) carries the background
  but not the voice.

In either case the floor is the residual's. Both floors are recorded with the source
(`quiet_frames`, `residual`, `digital`). The session-group floor is not built yet.

**Impulses.** On a 2 ms envelope of the pre-emphasised samples, read against its 200 ms running
median: a peak ≥15 dB over its background, rising within 5 ms, back within 3 dB of its background
within 80 ms, and with no other candidate within 30 ms. Attack and duration together keep a quick
breath (≈0.25 s noise body, slow onset; 2bb59c22) out. The isolation rule is what keeps a voiced
vowel out: a glottal pulse train is a run of sharp peaks a few ms apart, each of which passes the
attack and duration tests on its own (found on the synthetic sawtooth vowel).

**Activity.** A frame is active where at least 30% of bands stand 6 dB over their floor
(the C2 `q` and `m`, both unfitted); runs are bridged across 0.1 s, kept from 0.1 s, and an
impulse's own frames never make a region. Each region carries its peak over the broadband floor,
the largest band share, and its onset and offset times (to within 3 dB of its peak).

**Hum.** Residual mains lines, as VOICE's guard: ≥3 multiples of 50 or 60 Hz standing 10 dB over
their 2–8 Hz neighbourhood (31/36 hum, 0/30 clean, 2026-10-06). VOICE's f0-lock criterion stays in
VOICE: it needs a pitch track the generic layer does not compute.

**Faults.**
- Shutoff: VOICE's detector, moved here and read on every recording (VOICE now imports it).
- Dropouts and discontinuities: their extents on the original recording
  (`disruption_extents`, same parameters as `disruptions.*`), discontinuities closer than 10 ms
  joined.
- Clipping: one reading, PREPROCESS's clip spans that no unclipped sample contradicts. They carry
  extents (the join needs them) and are audited by the clip self-check; the whole-file
  `|x| ≥ 0.999` count has no extents and is retired from QUALITY's reading. Retiring the
  parquet's `raw_clipped_s` column follows with the parquet change in unit C.

## Unit B: the task layer (built)

Code: `task_events.py`, parameters `data/task_events.yaml`; the branches call it from
`breath_pattern.breath_evidence`, `cough_pattern.cough_pattern_of` and
`voice_phonation.phonation_reading_of`, and VERDICT reads the result through `airway_task` and
`vocabulary`.

**The generic view in the branches.** QUALITY runs after the branch loop (`run.py`), so a branch
cannot read the stored `background_model` measurement. It recomputes the same reading with the same
functions and parameters (`generic_view_of`: the plain and residual band frames, the floor, the
impulses and the regions). The numbers are identical. Running QUALITY's recording-level part before
the branches, so they read the measurement, belongs to unit C's DAG change.

**Events.** Each branch's own finder is its type test:
- breath: the breath train's bursts outside speech, from every run, not only the largest
  (`BreathTrain.candidates`);
- cough: the onsets with their inhale and tail;
- voice: the holds.

A candidate that an impulse explains, i.e. one with nothing 6 dB over the floor once impulse frames
are masked, is dropped. Every kept event records:
- `snr_db`: its peak over the stationary floor, impulse frames masked;
- `local_snr_db`: its peak over the background around it, i.e. the 20th percentile of the
  broadband level of the non-event, non-impulse frames within 2 s either side, never below the
  floor;
- `entangled`: an impulse overlaps it or ends or starts within 50 ms of it.

A cough's onset is itself impulsive, so for cough only an impulse outside the event entangles it.
Counting impulses inside sent 6 of 200 random kept coughs to review, all clear coughs (best SNR
59–77 dB).

**Rhythm (C3).** The modulation spectrum of the mean band-over-floor envelope. A peak in 0.1–1.2 Hz
standing ≥6 dB over the spectrum's median is a rhythm. It sets the breath cluster's gap to
max(3 s, 1.5 cycles), and recovers phases from regions of activity that no candidate overlaps, that
lie outside speech and that are at most half voiced. Recovered phases count toward the phases and
the extent, never toward the decision. In the grid, letting them decide cost one or two fit
agreements and gained none.

**Extent (C4).**
- Breath: the dominant cluster (most events ≥ s_lo within the gap), from the inhale back-trace (the
  broadband level ≥6 dB over the floor, at most 2 s back) to the last event's end. There is no
  padding and no extension to following speech; the source is `task_events`.
- Cough: keeps its padded, speech-bounded extent. The cluster extent dropped the intercom inside
  6ca9935e's task (`background_speech_in_task`, owner-labelled present-and-flagged) out of the extent.
- Voice: extent unchanged; its evidence is reported only.

**Decision (C5).** It is read over every found event, not only the cluster. The cluster bounds the
extent; it does not decide, because breathing slower than the gap splits a train, and 80e179b4 went
to review on the wrong half.
- absent: no non-recovered event ≥ `s_lo` = 10 dB over the floor;
- review: none of those ≥ `s_hi` = 16 dB over its local background ("weak"), or every one that is
  clear is entangled;
- present: otherwise.

VERDICT reads the result as follows:
- `breath_present` is decision ∈ {present, review};
- `breath_review` is decision = review;
- the counted-breath annotation reads the cluster's breaths.

The old `breathing_pattern` measure (pattern, events, the `little_activity` and speech vetoes) is
reported and decides nothing: `_measure_found` is gone. A cough's review is its strict/lenient band
or the evidence's review, and its count is the events that no impulse explains. `discard_contested`
is unchanged and folds into C5 in unit C.

**Fit (fit split only; the held-out side is reported after).**
- Stored inputs: the 167 labelled recordings, re-measured and re-folded in memory (ORCD
  `unitb_20261007`).
- Global-floor SNR alone (s_lo 3–8, s_hi 8–18, rhythm on or off) never beat r17: at best 103/118 fit.
  Noise reads 14–26 dB over the floor, as breath does.
- The background around the event separates them: 3d446af4 21 → 14 dB, ae2a7223 18 → 11 dB.
- Grid over s_lo ∈ {8, 9, 10, 11} and s_hi ∈ {14–18}: s_lo 10, s_hi 16 is best on fit.
- Entanglement on or off changed no labelled outcome.

**Result on the owner labels (fit / held-out; r17 baseline, same harness):**

| Group | r17 | unit B |
|---|---|---|
| breath | 42/47 / 18/19 | 41/47 / 17/19 |
| contested and review sets | 11/15 / 6/8 | 13/15 / 5/8 |
| cough | 18/18 / 7/7 | 18/18 / 7/7 |
| voice | 30/31 / 11/11 | 30/31 / 11/11 |
| all | 108/118 / 45/48 | 109/118 / 43/48 |

- Owner "absent" labels passed: 0 in both. Owner "present" labels discarded: 0dc15213 in both (no
  candidate over the floor; only recovered phases).
- Present labels left in review fall from 9 to 5: 1f4ea26f, 19c5c847, 2bb59c22, 1965766f, 81873ca0
  and fac74f45 now pass, while b451fe70 and 01f52a78 newly go to review.
- Review-acceptable labels now passed rise from 2 to 6: 5cc93330, 167ac3f5, 03ca6c69, 67c60a9f and
  16eded62, each with one event ≥16 dB over its local background.

**Outcome on a random 200 r17-kept recordings per group (reported, not a target):**

| Group | review band r17 | review band unit B | verdict moves |
|---|---|---|---|
| breath | 2.5% | 6.5% | 10 pass→flag, 3 flag→pass, 2 pass→discard |
| cough | 1.5% | 1.5% | none |
| voice | 2.0% | 2.0% | none |

The two new breath discards have their strongest event 7.5 and 9.99 dB over the floor, and a
breathing-band rhythm of 17 and 27 dB. They are not labelled. A rhythm is not allowed to decide
(C3), so they stand as discards until listened to.

**Not built in unit B.**
- The session-group floor.
- Raw-against-enhanced agreement and the enhanced reviewer hints.
- A high-band slow-breathing finder: the rhythm recovers from the generic regions only.
- The breath type test still detects on the train's own denoised spectrogram. Only the evidence,
  extent and decision moved onto the generic floor.

## Decided (owner, 2026-10-07)

1. **Review-acceptable labels.** "OK if flagged or discarded" and "leave as contested" score review
   or discard as correct.
2. **Background affects an event only where it is temporally entangled with it.** A click or tone
   overlapping or abutting an event makes that event ambiguous and sends it to review; background
   elsewhere in the file does not enter the decision. Owner: "does not make sense except if
   temporarily disambiguated."
3. **Session-group floor** where available, with the per-recording floor recorded alongside.
4. **No review-rate target.** The rate is an outcome to report, not something to tune to. Owner:
   "does not make sense."
5. **The old `breathing_pattern` measure leaves decisions** (modulation cycle count,
   `alternating_breaths`, the `little_activity` and speech vetoes, `_measure_found`). Decisions use
   active spans, events, the modulation rate and event SNR. Owner: "don't we now have active spans
   and events and modulation rates?"
6. **In-span interference flags every family**, speech and voice included.
7. **Session group = the BIDS session.** "Only the surroundings recorded" discards when no task event
   rises above the background in any stream; otherwise review.
8. **SQUIM is reported only**, deciding nothing until a labelled test shows it separates.
9. **A dropout or clip inside a task span flags**, with its minimum duration fitted from labels;
   outside the span it is an annotation.
10. **Decision vocabulary** (owner: "rename and use these in the decision table"). Verdict is
    `pass` · `review` · `discard`; `flag` is renamed `review` and `rerun` stops being a verdict.
    Release is `as_is` · `redacted` · `withheld` (redaction-policy holds only) and is empty on a
    discard. Pre-alpha: the old values are replaced outright, with no aliases.
11. **Evidence is row and task specific** (owner: "evidence should be row/task specific and can be
    different for different rows"). The decision table carries the items that produced each row's
    decision; a long table carries every item read.

## Decision table

The fold writes, beside the verdict, everything a reviewer needs to read the decision back.

**Run status.** A recording the pipeline still owes something (a missing derivative, a node that did
not finish, a configuration the fold cannot read, or a release it could not assess) has
`run_status = incomplete`, `missing` naming the operational grounds, verdict `review` and reason
`not_measured`. The replay manifest selects on `run_status`, not on a verdict value. At r17 the 460
`rerun` recordings and the 3 `not_assessed` releases are these.

**Reasons.** `data/decision_reasons.yaml` maps every ground key the fold writes to one of fourteen
reasons and orders them: the discard reasons (`unmeasurable`, `task_too_short`, `no_task_captured`,
`task_not_found`) before the review reasons (`not_measured`, `fault_in_task`,
`interference_in_task`, `other_speaker`, `weak_events`, `streams_disagree`, `task_not_conforming`,
`identifying_content`, `second_opinion_disagrees`), then the release axis (`redaction_hold`).
`reason` is the first of a recording's reasons; `reasons` is all of them. A release ground maps
through its own section, because `unplaced_finding_unread` names both a review ground and a
withholding release ground. A ground key the vocabulary does not map is logged at fold time and
fails `decision_test.py`.

**Evidence items.** The fold emits one item per reading it weighed, at the point it weighed it: name,
value, unit, comparison, threshold and effect (`pass`, `review`, `discard`, `withhold` on the
release axis, or `annotation`), and whether it was decisive. On a discard the item behind the discard
ground is decisive; on a review every item whose effect is review; on a pass the task and gate items
that passed; a withholding release item always. Instruction comparisons (breaths, coughs and glide
travel against the instruction) are annotations and never decide a pass. The breath items read the
task layer's own decision inputs (`task_events.decision_inputs`), so the table repeats the
comparison `decide` made rather than reconstructing it. `data/decision_evidence.yaml` names every
item by family group with the unit that produces it; an item a later unit produces (unit C's
stream agreement, in-span interference and faults, nothing captured, the session floor) is absent
until then rather than a null column. `withhold` extends the owner's effect list
(pass / review / discard / annotation) for release-axis items, which move no verdict.

**Tables.** `scripts/triage_recording_vectors.py` writes, beside each recording-vectors shard,
`triage_decisions.NNN.parquet` and `triage_evidence.NNN.parquet` from the same run directories, and
the merge writes `triage_decisions.parquet`, `triage_decisions.tsv` and `triage_evidence.parquet`.

| `triage_decisions` | |
| --- | --- |
| identity | `participant`, `session`, `task`, `declared_family`, `stem`, `run_dir` |
| decision | `verdict`, `release`, `reason`, `reasons`, `run_status`, `missing` |
| task | `extent_start_s`, `extent_end_s`, `extent_duration_s`, `annotations` |
| evidence | `evidence`: the decisive items, `list<struct{name, value, unit, comparison, threshold, effect}>`; value and threshold as JSON |
| review | `figure_path` (`summary/summary.pdf`), `audio_paths` (the task cuts, else the recording) |
| provenance | `commit` (the last replay or refold marker's), `config_hash` (VERDICT's), `schema_version` |

`triage_evidence` holds one row per (recording, item) with `group`, the item's fields and
`decisive`. In the TSV, list and struct columns are compact JSON.

## Unit C step 1: the session floor and the background before the branches (built)

**The DAG.** `GRAPH_ORDER` is ADMIT, PREPROCESS, SESSION, BACKGROUND, TAXONOMY, routing, the
branches, QUALITY, REDACT, REVIEW, VERDICT (`vocabulary.py`).

- **PREPROCESS** gains a `background_floor` block after `residual`
  (`nodes/background.py:write_own_floor`): the recording's own per-band floor (`floor_of` with no
  session), its source and quiet seconds, the residual's floor, and its active level (the 95th
  percentile of the broadband level over live frames).
- **SESSION** (`write_session_floor`) aggregates the own floors of one BIDS session
  (`sub-<label>_ses-<label>` off the stem). A single-file run sees no siblings and records
  `floor_source: recording`; the corpus pass is `scripts/extend_session_floor.py`, sharded by session.
- **BACKGROUND** (`write_background`) reads activity regions, impulses, hum and faults against the
  session-informed floor and writes `background_model` plus `derivatives/background_view.npz`, the
  frames, floor, impulses and regions at full precision. A store without a session floor or a plain
  stream gets `background_model` with `missing` only.
- **The branches** read that view (`task_events.generic_view_of`) instead of recomputing it. Breath
  and cough name `background_model` as an absent input where it is missing; VOICE, which decides on
  phonation alone, keeps its reading and loses only its task evidence. QUALITY no longer writes the
  background. Routing's emptiness rule is unchanged in this step: it runs after BACKGROUND, but still
  reads the YAMNet peaks; moving it onto the activity regions is a decision change for the join.
- **The replay** re-runs the graph from BACKGROUND (`extend.REPLAYED_NODES`). On the cluster:
  `extend_background_floor.py` over the corpus, then `extend_session_floor.py`, then
  `extend_replay_decisions.py`.

**The session-floor rule** (`data/background_model.yaml`, `session:`, version 2):

- **Members:** recordings whose own floor was read over quiet frames, with matching bands; at least
  `members_min` (3) of them, else the session has no floor. A task that fills its file reads the task
  itself as its floor; its own floor is one member among several, and the median keeps it out.
- **Statistic:** the per-band median of the members' own floors.
- **Use:** a recording takes the session floor where its own floor stands `gap_db` (25 dB, median
  over bands) over it, the same cut the residual check fitted on the labels (noise and breath
  recordings 5–20 dB, task-filling vowels 27–62 dB over the residual). Otherwise the residual check
  applies as before, then the own floor. Where the session floor is the recording's own, the reading
  is bit for bit the one with no session.
- **Recorded beside it:** `level_rel_db`, the recording's active level less its session's median, and
  `floor_rel_db`, its broadband own floor less the session floor. A recording that captured only the
  room stands far under its session's level; deciding on that is the join's, not this step's.
- `members_min` and `gap_db` are unfitted.

**A reading missing its decision inputs is not measured.** `nodes/verdict.py:DECISION_INPUTS` names
the attributes the fold decides each task reading on (breath `decision`, cough `onsets_n` and
`review`, phonation `found`). A stored reading lacking one becomes the owner's absent input
`<node>:<reading>.<field>`, so the recording is `review` / `not_measured` with an evidence item
naming the field, never a pass. The re-fold dry run of 100 r17 stores passed six pre-unit-B breath
readings, which had no `decision`, silently.

## Unit C step 2: QUALITY's join and the one rule (built)

Code: `quality_join.py`, `nodes/quality.py:measure_join`, parameters `data/quality_join.yaml`; VERDICT
reads the stored `quality_join` measurement through `vocabulary._join_reasons` and
`nothing_captured`.

**What the join decides.** QUALITY reads BACKGROUND's findings against the task spans by time
overlap. A finding is in the task where it overlaps a span or lies within `abut_s` (0.1 s) of one.
- **Faults.** BACKGROUND's dead stretches are split by the plain stream's level in the 0.1 s
  before each one: a cut where it stood `sounding_db` (10 dB) over the floor, otherwise a gate (the
  recorder's noise gate, reported only). Shutoffs, dropouts and clips in the task review, each above
  its fitted minimum; a shutoff reviews only for a held task (`event_kinds: [phonation]`). Every
  other fault is the annotation `fault_outside_task:<kind>`. Discontinuities are reported only.
- **Another voice.** Residual windows heard as speech and harmonic residual runs, outside the
  task's own events and words (`background_speech_of`). In the task they review every family
  (owner decision 6); elsewhere they are an annotation.
- **Streams.** The raw stream's task events at ≥10 dB over the floor, re-read on the enhanced
  stream. The share lost is reported; no kind decides.
- **Capture.** `no_activity`: no activity on either stream. `quiet_vs_session`: the recording stands
  ≥30 dB under its session's active level.

**The rule.** One fold over the join: anything in the task reviews (`fault_in_task:<kind>`,
`interference_in_task:<kind>`), anything outside annotates. A recording is empty, and discards as
`no_task_captured`, where routing found it empty or the join found nothing captured, unless a
branch heard a task event (a breath, a cough, phonation or a lexical word). Routing's emptiness now
reads BACKGROUND's `active_s` (`emptiness.active_s_max: 0`) rather than YAMNet peaks, on the live
and offline paths alike. A join that could not be read is `not_measured`. The old
`background_speech_in_task` and `capture_cut_*` grounds are gone; the join replaces them.

**Fitted bounds.** Fit split only. Each row re-folds the 167 labelled stores with one value
changed (`evaluate.py` variants over the replayed join).

| Bound | Value | On the labels |
|---|---|---|
| dropout `min_s` | 0.5 s | 0.1 s sends 3b06709b (a 0.42 s dropout at the start of a glide, heard present) to review; 0.02 s also bffe3a4f (cough, heard present). No label needs a dropout to decide. |
| clip `min_s` | 20 ms | 5 ms sends fac74f45 (a 16 ms clip in its breathing, heard present) to review; 50 ms and off change nothing. |
| shutoff `event_kinds` | phonation | Off, f47eeda7 (phonation cut at 13 s, owner present and flagged) passes. For cough the shutoffs in c60d8bb8 and bffe3a4f follow the coughs and both are heard present. |
| streams `decides` | none | Breath costs 1–4 agreements, cough 2, phonation 4, all three 9–12. |
| other voice | decides | Off: 6ca9935e (intercom in a cough task, owner present and flagged) passes and 1c139a67 passes correctly. That is net zero on the labels, so the owner's rule stands. |

Values with no labelled effect, which stay UNFITTED:
- `capture.level_rel_db_max` (−20 to −500 dB);
- `abut_s` (0 or 0.25 s);
- `shutoff.sounding_db` (5 or 15 dB, or off).

`streams.lost_fraction` is moot while no kind decides.

**Result on the owner labels.** Replayed from BACKGROUND at 4d896192 (ORCD `unitc2_20261007`,
`rows_c3`, 966 targets, no errors), scored by `scripts/evaluate_task_events.py` (fit / held-out):

| Group | r17 | unit B | unit C step 2 |
|---|---|---|---|
| breath | 42/47 / 18/19 | 41/47 / 17/19 | 41/47 / 17/19 |
| contested and review sets | 11/15 / 6/8 | 13/15 / 5/8 | 12/15 / 5/8 |
| cough | 18/18 / 7/7 | 18/18 / 7/7 | 18/18 / 7/7 |
| other | 7/7 / 3/3 | 7/7 / 3/3 | 7/7 / 3/3 |
| voice | 30/31 / 11/11 | 30/31 / 11/11 | 29/31 / 11/11 |
| all | 108/118 / 45/48 | 109/118 / 43/48 | 107/118 / 43/48 |

- Owner "absent" labels passed: 0. Owner "present" labels discarded: 0dc15213, as before.
- Two labelled outcomes move from unit B:
  - **1c139a67** (high-to-low glide, present): pass → review on `interference_in_task:other_voice`,
    one 0.96 s window at 0.96–1.92 s. That is the steep fall of the glide, whose harmonics the
    residual keeps.
  - **efadb6b3** (breath, owner: "a lot of background noise", review acceptable): review → pass. The
    breath reading is now present, with 12 standing events where unit B read one weak event. This
    holds with the join switched off (`join_off`), so it comes from step 1's session floor, not from
    this step.
- 10 labelled discards now carry `acoustically_empty`, where r17 discarded them on other grounds.
- `quiet_vs_session` holds on 1965766f (present, passes), because a breath was heard.

**Outcome on the random r17-kept sample (reported, not a target).** 200 per group (199 voice). The
moves against r17 (B) are in the table.

| Group | review r17 | review C | moves r17 → C | of which this step |
|---|---|---|---|---|
| breath | 8 | 15 (+1 discard) | 11 pass→review, 4 review→pass, 1 pass→discard | 1 other voice |
| cough | 7 | 9 | 2 pass→review | 1 other voice, 1 dropout |
| voice | 4 | 11 | 7 pass→review | 6 other voice, 1 dropout |
| speech | 12 | 18 | 10 pass→review, 4 review→pass | 10 other voice |

- The breath moves other than ffb59262 are unit B's (`breath_review_low_confidence`) or step 1's.
- Speech's review → pass are redaction holds, which now sit on the release axis (decision 10).
- Another voice is most of what this step adds. Switching it off returns 23 of the 53 sample reviews
  to pass. Five of the six voice cases are a single 0.96 s YAMNet window inside a glide or a
  maximum-phonation hold, the same shape as 1c139a67.
- Both dropouts are a dead stretch from 0 s that ends where the task span starts (31054d13 0–0.75 s,
  d00fd6df 0–0.84 s). They are listening candidates for a leading-silence exemption.

## Unit C step 4: the review page (built)

`scripts/triage_review_page.py` builds it in two phases: `extract` walks a run tree on the cluster and
writes one JSON line per recording; `render` writes `index.html` plus side files under `data/`. The
page code is `src/senselab/audio/workflows/triage/review_page/`.

- **One reader for the decision.** Each record's decision and evidence come from
  `decision_tables.decision_rows`, so the page shows exactly what `triage_decisions` and
  `triage_evidence` hold. A speech task's transcript, PII marks, release ground and LLM summary come
  from the free-speech page's own reader (`recording_record`, `paragraph`), and the page carries that
  page's mark rules (`MARK_STYLE`), so a mark looks and means the same on both.
- **Reuse, not a second plot.** Explore is the recording-vectors viewer's `CorpusView` over its axis
  catalogue; the page adds its columns through `SchemaAxes.register` (branch, reason, run status,
  reason and annotation sets, evidence item sets, and one `ev:<name>` column per evidence item, typed
  from its values) and re-reads the catalogue with `SchemaFacets.refresh`. Axes are offered per
  evidence group from `data/decision_evidence.yaml`. Brushes, facets and search narrow one
  selection, which the Review list shows.
- **Index versus side files.** The inlined index holds what filtering needs: dictionary-coded
  scalars, set columns, evidence values by item, and a speech task's plain transcript for search.
  Everything per recording (the quantised spectrogram, overlays, stream paths, every evidence item,
  the transcript view) is in side files of `shards.records` recordings each, loaded by a script tag
  when a recording in that block is opened, which works from `file://` where `fetch` does not.
- **Spectrogram.** 16 log-spaced bands × 64 time bins over the whole recording, four levels (2 bits a
  cell, 256 bytes, 344 characters of base64), levels relative to the recording's own 20th percentile
  and maximum; drawn on canvas with the extent, the task events, the background's active regions and
  its issues (impulses, faults, background speech). No audio is embedded. Served from the cluster over
  an ssh tunnel the page plays the stored streams and links the full figure.
- **The reviewer's export** is JSON whose entries carry the owner label table's columns
  (`listen_set` = `triage_review_<page id>`, `owner_label` = `reviewer_<verdict>`), so
  `scripts/evaluate_task_events.py` reads it as labels; `owner_label_map.yaml` maps
  `reviewer_pass`/`reviewer_review`/`reviewer_discard` to `present`/`present_flagged`/`absent`.
  Decisions are kept in the browser's storage under the page id and can be imported back.
- **Size, measured on a sample of eight whole r17 sessions (307 recordings) replayed at this
  branch:** `index.html` 252 kB, of which about 150 kB is the inlined code and styles, so about
  330 B per recording in the index; side files 1.47 MB, about 4.8 kB per recording (218 of the 307
  are speech tasks carrying a transcript view). Scaled to 62,550 recordings: an index of about
  21 MB and about 300 MB of side files in 126 files of about 2.4 MB, one loaded per block opened.
  Extract ran at about 7 recordings a second on four workers.

## DDK task layer (built)

Code: `ddk_task.py`, parameters `data/task_events.yaml` (`ddk:`), wired through `nodes/ddk.py:align_ddk`
(SPEECH), `nodes/verdict.py:_task_evidence` and `vocabulary._ddk_items`. The ten
`SYLLABLE_REPETITION` families stay SPEECH's; SPEECH writes one `ddk_task_reading` measurement that
VERDICT decides on, as AIRWAY's breath and cough readings and VOICE's phonation reading are decided.

### What the old instrument did (investigation at 8076fb5c)

- **The decision read one number.** `repetitions_min` ≥ 1, plus `instructed_count_min_fraction` 0.5
  for the five v1 families with an instructed count of 10. The five v2 families (5 s, no count)
  passed on "one repetition" alone. Rate, cycle period, dispersion, realised mass and train fraction
  were measured in `decode_evidence` and decided nothing.
- **The posteriorgram decode located the cycles.** It was a strict ordered cyclic Viterbi with no skip
  arcs, counting only runs that began at template position 0. That drops leading partial cycles and
  halves counts by borrowing phones across bursts. On sub-06d289b0's v2 buttercup (6 bursts about
  0.65 s apart, 0.015–3.655 s) it found 2 repetitions over 1.15–3.79 s.
- **In r17:** 3,301 DDK passes rested on "≥1 repetition" only; about 414 recordings were decided by
  non-performance grounds (route mismatch 158 and conformance 133 among the reviews, dominant-speaker
  share 99); no rate or regularity gate existed.

### Design

- **Events (generic regions, task test).** Syllable nuclei are read inside BACKGROUND's activity
  regions: peaks of the level over the floor, lowered on frames the pitch track calls unvoiced, standing
  `nucleus_prominence_db` over the troughs either side and `nucleus_spacing_s` apart. A syllable runs
  trough to trough; one outside `[syllable_min_s, syllable_max_s]` is not one (a held vowel is another
  activity). Single-syllable families (pa, ta, ka, puh, tuh, kuh) take the syllables as events.
  Sequence families (pataka, puhtuhkuh, buttercup) group them into cycles: each region's run is split
  into `round(duration / period)` cycles at the syllable boundaries nearest equal shares, so a pause
  always closes a cycle and a leading or trailing partial cycle stands as an event. The period is the
  cycle-band modulation peak (`cycle_band_hz`, the C3 rhythm) where it lies within `rhythm_agreement` of
  the template's syllable count times the median syllable interval, else that product. Impulses are
  never events (`impulse_explained`), and only an impulse outside every syllable run entangles one: a
  stop release is impulsive, so an impulse inside the run is the task's own.
- **Identity, not location.** The posteriorgram scores each candidate event against the template with a
  local alignment: a silence filler before and after, the positions in order, entry and exit at any
  position, and forward jumps that delete positions at `skip_log` each. Only silence is filler, so a
  phoneme outside the template inside the event is charged to a position (a substitution) and keeps
  its low mass. Per position it keeps the realised mass (mean class mass over the frames charged) and
  the peak; the event's identity is the mean of the peaks, a deleted position counting 0.
- **Extent.** The task events are the events standing over the floor whose identity reaches
  `identity_absent_max`, in their dominant cluster (`gap_s`). The extent is the first to the last, with
  nothing added. ASR words are an annotation only: the recogniser finding no lexical word, and routing
  declining SPEECH, are annotations on a syllable task.
- **Decision (C5, one rule).**
  - absent: no event `snr_low_db` over the floor (`no_syllable_train_captured`, reason
    `no_task_captured`), or the standing events' median identity under `identity_absent_max`
    (`syllable_train_not_target`, reason `task_not_found`);
  - review: identity in `[identity_absent_max, identity_min)` (`ddk_review_identity`,
    `task_not_conforming`); fewer than `events_min` events clear of their local background by
    `snr_high_db` (`weak`), or the clear ones all entangled (`ddk_review_weak_events`, `weak_events`).
    QUALITY's join reads the syllable events for stream agreement and in-span interference, as for
    every family (`streams.decides` is still empty);
  - present: otherwise.
- **Annotations, never deciding:** the count against the instruction (`task_mismatch` where short),
  syllable and cycle rate, the onset intervals' coefficient of variation and trend, the train fraction
  and the per-position realised mass. `repetitions_min` and `instructed_count_min_fraction` are retired
  as gates for the syllable groups (`CONFORMANCE_GATES` is empty for them, so their conformance answers
  `UNDETERMINED` and the reading decides). The cyclic decode, its measurements
  (`ddk_repetition_count_from_ppg_decode`, `ddk_syllable_rate_from_ppg_decode_hz`,
  `ddk_ppg_period_dispersion`, `ddk_repetitions_found`) and `branch.burst_window_ms` are gone.
- **Evidence items** (`data/decision_evidence.yaml`, group `ddk`): `ddk_event_db_over_floor`,
  `ddk_identity`, `ddk_clear_events` decide; `ddk_events_against_instructed`, the rates, regularity,
  train fraction and realised mass are annotations.

### Unfitted

There are no owner DDK labels yet. Initial values come from the current data: a read of the task
layer over 951 r17 DDK recordings (479 random passes, 120 random reviews, the 282 v2 passes with ≤2
decoded repetitions or a decoded rate under 3 Hz, the 69 discards and the buttercup case; ORCD
`ddk_20261007/explore_e2`).

| Parameter | Value | From |
|---|---|---|
| `identity_min` | 0.30 | 3 of 479 random passes fall under it (1st percentile 0.36) |
| `identity_absent_max` | 0.15 | no random pass falls under it; 17 of the 32 r17 discards that hold a standing event do |
| `events_min` | 2 cycles, 3 syllables | every random pass holds at least 4 clear events (1st percentile) |
| `gap_s` | 2.0 s | none |
| `skip_log` | −1 | none |
| `nucleus_prominence_db`, `unvoiced_db` | 6 dB | none |

Of the identity measures read, the mean realised mass separated random passes from r17 discards less
well (1st percentile of passes 0.20 against a discard median of 0.11) than the mean peak (0.36 against
0.14). Filler explaining every non-template phoneme let the alignment delete whole events of a
substituted syllable as silence; filler restricted to silence keeps the substitution visible.

### Validation (replayed from BACKGROUND at 1ed6b3c9)

ORCD `ddk_20261007`, `rows_1ed6b3c9`: 149 r17 DDK recordings, each copied and replayed from
BACKGROUND with its BIDS session's floor (own floors of 4,274 session members), no errors. The groups:
the buttercup case; 50 of the 282 r17 v2 passes with ≤2 decoded repetitions or a decoded rate under
3 Hz (seeded random); all 69 r17 DDK discards; 29 random r17 DDK passes.

| Group | n | r17 → new verdict |
|---|---|---|
| case | 1 | pass → pass |
| v2 passes, ≤2 reps or <3 Hz | 50 | 39 pass, 7 review (`ddk_review_identity`), 4 discard (`syllable_train_not_target`) |
| r17 discards | 69 | 55 stay discarded, 7 → review (identity), 7 → pass |
| random r17 passes | 29 | 29 pass |

| Group | r17 reps (median) | new events (median) | r17 decoded rate | new syllable rate | extent r17 → new (median s) |
|---|---|---|---|---|---|
| v2 low-rep / low-rate passes | 5 | 10 | 2.31 Hz | 4.0 Hz | 3.48 → 4.05 |
| r17 discards | 0 | 0 | 1.73 Hz (12 read) | 4.08 Hz (25 read) | 1.80 → 2.64 |
| random r17 passes | 10 | 12 | 4.28 Hz | 4.54 Hz | 4.14 → 4.25 |

- The v2 passes with ≤2 decoded repetitions fall from 13 of 50 to 3 of 50 with ≤2 task events; the
  decoded rates under 3 Hz were the decode's borrowing, not slow speech.
- The 55 discards that stay: 36 `no_task_captured` (no syllable over the floor, or nothing captured),
  11 `task_not_found` (a train that is not the target), 8 `task_too_short`.
- The 7 discards that now pass are v1 pa/ka trains whose decode read 1–4 repetitions against the ten
  asked and which r17 discarded as `declared_task_absent` after `conformance:SPEECH`; the task layer
  reads 6–16 events with identity 0.31–0.55.
- Every new review is the identity bound. Five of the seven among the v2 passes are v2-kuh: the velar
  class reads lower on the posteriorgram (random passes: v2-kuh median identity 0.48 against 0.68
  for pa, ta and pataka), so a per-family identity bound is the first thing labels should test.
- **The case, sub-06d289b0 v2-buttercup:** 12 syllables in 6 runs, 6 cycle events (two syllables each:
  "butter" is flapped and has no closure), 0.015–0.315, 0.505–0.965, 1.135–1.615, 1.835–2.325,
  2.535–2.975, 3.185–3.655 s; extent 0.015–3.655 s (r17's decode: 2 repetitions over 1.152–3.786 s);
  cycle rate 1.54 Hz, period CV 0.14; identity 0.307, the rhotic position never realised. Present.

## Unit C plan (owner-approved revisions, 2026-10-07)

1. **The recording-level background reading is a PREPROCESS output, not QUALITY.** QUALITY is only the
   per-span join after the branches. Branches read PREPROCESS's reading from the store and never
   recompute it.
2. **The session floor is computed before its first use.**
   - (a) PREPROCESS estimates the floor and level per band for each recording.
   - (b) A corpus-level SESSION step aggregates those per BIDS session.
   - (c) The rest of the background reading (activity regions, impulses, faults) is computed against
     the session-informed floor; routing's emptiness check depends on it.
   - (d) Everything downstream reads that same floor.

   PREPROCESS splits around SESSION, and the DAG declares the dependency. On the cluster, (a) runs
   corpus-wide, SESSION once per session, and the rest per recording. A single-file run with no
   siblings falls back to the per-recording floor and records that it did.
3. **Decision vocabulary.**
   - `verdict` ∈ {pass, review, discard}: `flag` is renamed `review`. `rerun` becomes a separate
     `run_status`; a recording still incomplete after recompute is `review` with reason `not_measured`.
   - `release` ∈ {as_is, redacted, withheld}, null when the verdict is discard; `withheld` is a
     redaction-policy hold only.
   - `reason` (primary) and `reasons` (all), drawn from a vocabulary of about ten keys in `data/`.
4. **Decision table.**
   - `triage_decisions`: one row per recording, with core fixed columns (identity, verdict, release,
     reason, reasons, extent, annotations, figure and audio paths, provenance) and an `evidence` list
     holding the items that decided *that* row: `{name, value, unit, comparison, threshold, effect}`.
     The contents differ by row and by task.
   - `triage_evidence`: a long table of every item read, with a `decisive` flag. Item names are
     defined per family in `data/`.
   - Items are emitted from VERDICT's decision path, never reconstructed afterwards.
5. **Review page.** A static HTML page over the **full set** (all 62,550 recordings). Only some
   recordings are ever reviewed; the page is how the reviewer narrows to them.
   - **Tabs over the same loaded data, with one shared selection and one shared set of decisions:**
     - **Explore:** a parallel-coordinate plot over the decision table's columns plus the evidence
       items. Evidence differs by family, so the axes are chosen per family or facet from the
       evidence names defined in `data/`. Brushing the axes narrows the selection. Reuses the
       recording-vectors viewer (`src/senselab/audio/workflows/triage/viewer/`:
       `recording_vectors_viewer.html`, `axes.js`), which already draws parallel coordinates; there
       is no second implementation.
     - **Review:** faceted filters in the pattern of the free-speech page
       (`scripts/free_speech_review_page.py`): verdict, release, reason, family/branch,
       `run_status`, annotations, presence of each evidence item, session/participant, plus text
       search over stem and transcript. The list shows whatever the brush and the facets select.
       Each recording shows its decision, reason and evidence items, and the spectrogram (or audio,
       below). **Speech tasks also show the ASR transcript, PII detections and redactions**
       (masked tokens, release form, reviewer and LLM proposals where present), as the free-speech
       page does, so the release decision is reviewable as well as the task verdict.
     - **Decisions:** the reviewer's entries so far (verdict pass/review/discard plus a note), with
       JSON export and import so a reviewing session can resume. Kept in the browser; no hosted
       state. The JSON keys match the owner label CSV's columns, so reviews feed the evaluation
       harness.
   - **Display modes:**
     - (a) **standalone:** each recording carries a compact quantised spectrogram, a coarse
       time × frequency grid with a few levels stored as a small integer array and drawn on canvas
       (not an image), with events, extent and issues drawn over it. Size estimate: 16 frequency
       bins × 64 time bins × 2 bits is 256 B per recording, about 342 B base64, so about 21 MB for
       62,550 recordings before the table and transcripts.
     - (b) **served from ORCD over an ssh tunnel** (an http server on the cluster plus `ssh -L`):
       additionally fetches, on demand, the raw, enhanced and (where it exists) released audio and
       the full per-recording figure.
   - No audio is ever embedded. If the single file would be too large, the per-recording data
     (spectrogram grids, transcripts, evidence) is sharded into side files (for example per family
     or session) loaded on demand.

## Evidence items are scalar rows (2026-10-08)

The owner found a `nothing_captured` item whose value was a dictionary (route state, both streams'
active seconds, the level against the session) compared `==` against `false`. A reviewer cannot read
which of four readings decided, and the page's per-item axis could not type the column. Every item is
now one scalar row (`decision.row_problems` states the rule; `src/tests/audio/workflows/triage/conftest.py`
applies it to every fold any triage test runs):

- `nothing_captured` → `capture.route_state` (`not in [empty]`), `capture.plain_active_s` and
  `capture.enhanced_active_s` (`> 0` s), `capture.level_rel_db` (`>` the join's
  `capture.level_rel_db_max`). The separate `level_rel_db` annotation is gone; `capture.level_rel_db`
  replaces it. On an `acoustically_empty` discard the capture rows that fail their comparison are the
  decisive ones; a stream silent while the other is active is an annotation.
- `owning_branch_input_absent` (a list) → one flag per absent input,
  `owning_branch_input_absent:<node>:<reading>.<field>`.
- `ddk_realised_mass` (a list) → one row per template position, `ddk_realised_mass:<index>.<phone>`.
- `admit_outcome` compares as a category (`in [pass]`).
- A row with no threshold carries no comparison (a `>=` against an unread instructed count used to be
  emitted), and a gate record with no recognised `op` is reported, not compared.

## Review page: transcripts name their source and show word agreement (2026-10-08)

The owner could not tell which transcript the page showed. A speech task's transcript is now labelled:
the **consensus** of the named ASR models (each by model id, with the pipeline's source name), or,
where the consensus holds no words, the **one model** the free-speech reader falls back to, named.
The consensus is the default. Each consensus word carries its stored `agreement` (largest same-key
group over the sources) and `outcome`, and is shaded in four bands (all agree, ≥2/3, ≥1/3, under 1/3);
hover or tap shows every model's reading at that word (`readings` on the `word` entity). PII and
redaction marks wrap the shaded words unchanged (`free_speech_review_page.paragraph` takes the word
renderer). Every model's own transcript (`asr_hypothesis.transcript`) is listed in a collapsed block.
Extract reads it with `review_page.records.transcripts`.

## Review page: raw recordings outside the corpus root (2026-10-08)

A replay copy's `recording` stream points at the raw file in place (`/orcd/data/...`), outside the
scan root, so the page built `audio_base + /orcd/...` and nothing played; earlier listening pages
needed hand-made symlinks. An absolute stream path now goes through a source route
(`--source-base`, default `/_source`): the page fetches `/_source/<absolute path>`, and
`scripts/triage_review_serve.py` (standard library only) serves the corpus root plus that route,
restricted to its `--source-root` directories.

## Review page: per-model ASR lanes, and the task span played in place (2026-10-08)

**What the owner saw.** On `sub-06d289b0…_task-diadochokinesis-v2-buttercup` (r18) the transcript read
"What are the time? What is it like? …" and nothing of Qwen's output. The store holds two hypotheses
over `plain`: `nyralabs/CrisperWhisper2.0_turbo`, 22 timed words ("What are the time? What is it like?"
repeated; its full transcript continues as a hallucinated loop, two chunks out of bounds), and
`Qwen/Qwen3-ASR-1.7B`, 6 words timed by `Qwen3-ForcedAligner-0.6B` ("Barakat, barakat, …" at
0.00–0.96, 0.96–1.60, 1.84–2.32, 2.56–2.96, 3.20–3.68, 3.68–3.68 s), the syllable train itself. Qwen
wrote no segments or non-lexical tokens; `asr_hypothesis` carries only words.

**Why it was lost.** Nothing was dropped. The star alignment put each Qwen word in a column with a
Whisper word (6 `variant` words, agreement 0.5) and left 16 Whisper-only `insertion` words. On a
1–1 tie `_column_word` takes the first group in source-name order, so every surface was Whisper's
(`reference_source: asr_crisperwhisper`); Qwen's readings survived only in each word's `readings`,
which the page showed in a hover. The extract kept each hypothesis's text but not its word times, so
no per-model timeline existed to show.

**What the page does now.** `transcripts` carries each hypothesis's own words as
`[start_s, end_s, text]` and each consensus word's extent and surface. Under the spectrogram, on its
time axis, the page draws a consensus lane (agreement-shaded) and one lane per model; spectrogram and
lanes share one window (zoom, pan, show-all, show-task-span; ctrl/⌘-wheel and drag), and a click
seeks the active player with a playhead across both. In the consensus text a word only one model gave
(`insertion`) is outlined and its tooltip names the model; a `variant` word keeps its surface and
shows each other reading beside it, outlined when one model alone gave it. The collapsed per-model
list shows each model's words with their times. Agreement shading and PII marks are unchanged.

**Audio.** `task_plain` and `task_enhanced` are cuts of the plain and enhanced streams over the
extent ±0.25 s, so on the page they repeated `recording` and `enhanced`. The page no longer lists any
`task_*` stream: it plays `recording` and `enhanced` (and, for a speech task, the full-length
`redacted` in place of `task_redacted`), each with a strip marking the task extent and a
"play task span" control that seeks to the extent and pauses at its end. The task-audio pipeline is
unchanged.
