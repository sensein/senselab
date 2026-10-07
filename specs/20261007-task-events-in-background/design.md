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
