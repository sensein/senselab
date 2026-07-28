# Feature Specification: Uncertainty-driven adaptive analysis workflow

**Feature Branch**: `20260723-225523-dynamic-uncertainty-workflow`
**Created**: 2026-07-23
**Status**: Draft
**Input**: User description: "analyze_audio should clean up the audio, detect environmental sounds, analyze quality, transcribe, and diarize — temporally; use individual temporal signals to improve other signals; use the signals to measure uncertainty about where speech exists, diarization, and word-aligned transcription; and run as a dynamic workflow that can iterate and come to a robust conclusion."

## Context

`scripts/analyze_audio.py` today is single-shot and exhaustive. `main()` runs every task on the raw
16 kHz audio (`run_pass`, `scripts/analyze_audio.py:1531`), unconditionally enhances the whole file and
repeats every task on the enhanced copy (`:1928-1949`), then calls the comparator once
(`compute_uncertainty_axes`, `compute.py:48`) to produce the three per-bucket uncertainty axes —
`presence`, `identity`, `utterance`. The axes are *reported* (9 parquets, `disagreements.json`, LS
tracks, `timeline.png`) but never *acted on*: the only consumer of high-uncertainty buckets is a human
reading `disagreements.json`. The couplings that do exist are one-directional and computed in a single
forward pass: embedding-cluster silhouette injected as a synthetic diarization voter
(`compute.py:211-236`), empirical cosine calibration floors flowing embeddings → identity
(`compute.py:441-456`), presence-derived `intensity_weight` masking identity/utterance rows
(`compute.py:405-418`), and the planned `scene_quality_coupling` multiplier (prior spec FR-019).

Three structural consequences follow. (1) **Compute is spent uniformly, not where it matters**: clean
audio still pays for a full enhanced pass; a file with no speech still pays for four ASR models, two
diarizers, and alignment. (2) **Uncertainty is terminal**: a bucket where ASR models disagree stays
uncertain even when one more targeted model call, or a switch to the cleaner stream, would resolve it.
(3) **Known cross-signal repairs are impossible in one pass** — hallucination adjudication, diarization
boundary snapping to embedding change-points, overlap detection explaining joint identity+utterance
uncertainty, per-region raw-vs-enhanced stream election — because the signals that would drive each
repair are only available after the pass ends.

This feature makes the three uncertainty axes the **control signal of a bounded, deterministic loop**:

```text
round 0            round 1                rounds 2..K (K = --max-rounds)
TRIAGE      →      BASELINE EVIDENCE  →   [ TARGET → INTERVENE → RE-AGGREGATE → CONVERGE? ]
cheap signals      conditional passes      region proposal   policy-ranked     cheap, pure
(quality, frame    (diarization, ASR,      from belief       actions on        re-aggregation
posteriors, scene, alignment, PPG,         state             cropped regions   over merged votes
openSMILE)         embeddings, comparator)
```

Every round appends evidence (votes with full provenance) to a persistent **belief store**; aggregation
is re-run cheaply after each round; regions either converge (uncertainty ≤ θ_low), are proven
**irreducible** (uncertainty does not respond to intervention — with a machine-readable reason such as
`overlapping_speech` or `snr_floor`), or exhaust their budget. The final round fuses the accumulated
evidence into a **robust conclusion**: a consensus word-level transcript with calibrated confidences and
speaker attribution, a refined diarization, a fused presence track, and a complete audit trail of every
decision the loop made. The loop is driven by a deterministic, declarative policy engine — no LLM in the
control path — so identical inputs and policy produce identical decisions, in keeping with the
provenance/caching philosophy of the existing pipeline.

## Signal-coupling matrix

The core of "use individual temporal signals to improve other signals". *Existing* couplings are kept
and formalized; *new* couplings are introduced by this feature. Round = where the coupling first fires.

| # | From (signal) | To (consumer) | Mechanism | Status | Round |
|---|---|---|---|---|---|
| C1 | Quality (SNR/C50/clip/bandwidth) + frame posteriors | Enhancement decision | `--enhancement auto`: run the enhanced pass only if some speech region is degraded | new | 0→1 |
| C2 | Frame posteriors (segmentation-3.0, Brouhaha) | Heavy-task gating | No confident speech anywhere ⇒ skip diarization/ASR/alignment/PPG, emit presence-only run | new | 0→1 |
| C3 | Quality per region | ASR vote weights | Per-region reliability prior on ASR votes (low SNR ⇒ down-weight fragile backends) | new | 2+ |
| C4 | Presence (p_voice) | Identity/utterance | `intensity_weight` mask (existing) + hard budget gate: no interventions on non-speech | existing+new | 1+ |
| C5 | Raw-vs-enhanced deltas + quality + presence | Stream election | Per-region choice of primary stream for re-processing; enhancement-artifact guard | new | 2+ |
| C6 | Embedding change-points (fine-hop) | Diarization boundaries | Snap disputed boundaries to calibrated cosine change-points; re-score identity | new | 2+ |
| C7 | Segmentation-3.0 per-class (powerset) posteriors | Identity + utterance | Overlap posterior explains joint uncertainty as aleatoric; routes overlap handling | new | 2+ |
| C8 | ASR consensus text | Forced alignment | Re-align consensus (not per-model) text ⇒ authoritative word timestamps | new | K |
| C9 | ASR+PPG+CTC agreement in "silent" buckets | Presence | Missed-speech correction vote where VAD said silence but phonetic evidence is strong | new | 2+ |
| C10 | Whisper no_speech/logprob + CTC + PPG + sources | Utterance + presence | Hallucination adjudication: purge hallucinated tokens from WER pairs and presence votes | new | 2+ |
| C11 | Diarization + overlap posterior | Utterance | Speaker-attributed re-ASR of overlap crops (v2, behind flag) | new (v2) | 3+ |
| C12 | Embedding clustering silhouette | Presence + identity | Synthetic diarization voter (`compute.py:211-236`) | existing | 1 |
| C13 | Per-pass empirical cosine calibration | Identity floors | `calibrate_cosine_uncertainty` band overrides CLI floors (`compute.py:441-456`) | existing | 1 |
| C14 | Scene sources (music/TV/machine masses) | Utterance + hallucination prior | `scene_quality_coupling` (prior spec FR-019) + raised hallucination prior in music regions | existing (planned) + new | 1, 2+ |

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cost-aware triage and conditional processing (Priority: P1)

A researcher runs the workflow on a batch of clinical recordings. Most are clean close-mic speech; a few
are silent (failed recording) or heavily degraded. On clean files the loop skips the enhanced pass
entirely; on silent files it stops after triage with a presence-only report; only degraded files pay for
enhancement and iteration.

**Why this priority**: Immediate, measurable compute savings on every run, independent of the iteration
machinery; it exercises round 0 and the gating couplings (C1, C2) end-to-end.

**Independent Test**: Run with `--enhancement auto` on (a) a clean clip, (b) a silent clip, (c) a
degraded clip. Verify (a) has no `enhanced_16k/` outputs, (b) stops after triage with
`run_state: "no_speech"` in `summary.json`, (c) runs both passes.

**Acceptance Scenarios**:

1. **Given** a clip whose triage SNR degradation stays below θ_enh in every speech region, **When** the
   workflow runs with `--enhancement auto`, **Then** no enhancement model is loaded, no `enhanced_16k`
   pass exists, and `iterations.json` records the decision with its trigger values.
2. **Given** a clip with frame-posterior P(speech) < 0.2 everywhere, **When** the workflow runs, **Then**
   diarization/ASR/alignment/PPG never run, presence parquet is still emitted, and the run exits 0 with
   `run_state: "no_speech"`.
3. **Given** `--enhancement always`, **Then** behavior matches today's unconditional two-pass run.

---

### User Story 2 - Targeted re-processing of high-uncertainty regions (Priority: P1)

After the baseline round, several 2–6 s regions show high utterance uncertainty (ASR models disagree).
The loop crops each region (with context padding), re-runs the ASR set on the region's elected stream,
escalates to a reserve model if disagreement persists, and re-aggregates. Regions where models now agree
converge; regions that don't respond are marked irreducible with a reason.

**Why this priority**: This is the core promise of the dynamic workflow — uncertainty goes down where it
can, and is explained where it can't.

**Independent Test**: Take a clean clip, inject localized noise into a 3 s span, run with
`--max-rounds 3`. Verify the injected span is proposed as a region, at least one intervention runs on
it, and either its utterance `aggregated_uncertainty` drops by ≥ ε or the region carries
`irreducible_reason`.

**Acceptance Scenarios**:

1. **Given** a bucket run of utterance uncertainty ≥ θ_high, **When** round 2 starts, **Then** a region
   covering it appears in `rounds/2/regions.json` with axis, uncertainty mass, and elected stream.
2. **Given** a proposed utterance region, **When** the policy engine fires `U1_region_reasr`, **Then**
   region-scoped ASR votes (scope `region:<id>`) shadow that model+stream's file-scoped votes in covered
   buckets, and re-aggregation changes only covered buckets.
3. **Given** an intervention whose re-aggregated uncertainty drop is < ε, **When** the round ends,
   **Then** the region's `interventions_remaining` decrements, and after `--max-region-rounds` the region
   is marked `irreducible` with the dominant explanation (e.g. `snr_floor`, `overlapping_speech`).
4. **Given** two identical runs (same audio, args, policy, warm cache), **Then** `iterations.json` is
   byte-identical.

---

### User Story 3 - Cross-signal repair (Priority: P2)

The loop uses one signal to fix another: ASR text in a region the VAD calls silent is adjudicated
(hallucination vs missed speech) using no-speech probability, alignment CTC score, PPG activity, and
sound-source masses; diarization boundaries disputed between pyannote and Sortformer are snapped to
fine-hop embedding change-points; joint identity+utterance uncertainty over a region with high overlap
posterior is explained as overlapping speech rather than model failure.

**Why this priority**: Cross-signal repair is what distinguishes iteration from merely "run more
models"; it reduces phantom disagreement (hallucinated tokens inflating WER pairs) and mislabeled
uncertainty (overlap read as model error).

**Independent Test**: On a clip with music-only span where one ASR emits text, verify adjudication marks
those votes `hallucinated`, utterance WER pairs exclude them, and presence votes drop them. On a
two-speaker clip with a disputed boundary, verify the boundary evidence vote lands within the crop and
identity disagreement decreases.

**Acceptance Scenarios**:

1. **Given** ASR tokens in buckets with p_voice < 0.2, **When** `P3_hallucination_adjudication` fires and
   ≥ 2 independent indicators support hallucination, **Then** affected votes are flagged, excluded from
   utterance pairing (C10), and the verdict + indicator values are logged.
2. **Given** phonetically supported speech (PPG active, CTC score high, ≥2 ASR models agree) where frame
   posteriors were low, **Then** a `adjudicator/missed_speech` presence vote is added (C9) and presence
   uncertainty in those buckets increases or the belief flips — never silently.
3. **Given** identity uncertainty at a diarization boundary and a clear fine-hop cosine change-point,
   **Then** an `embedding_changepoint/<model>` evidence vote supports the nearest model boundary and
   re-aggregated identity uncertainty in the region decreases.
4. **Given** a region whose overlap posterior mean ≥ 0.5, **Then** identity/utterance rows in it gain
   `aleatoric_floor` ≥ overlap posterior and, if uncertainty stays high, `irreducible_reason =
   "overlapping_speech"`.

---

### User Story 4 - Robust conclusion with audit trail (Priority: P2)

When the loop stops, the researcher gets a single fused answer, not nine parquets to reconcile: a
word-level consensus transcript (text, start, end, speaker, calibrated confidence, alternates where
contested), a final diarization with boundary confidence, a fused presence track, and `iterations.json`
recording every rule that fired, why, what it cost, and what it changed.

**Why this priority**: "Come to a robust conclusion" is the user-facing deliverable; everything else is
machinery.

**Independent Test**: Run on a tutorial clip; verify `final/transcript.json` words are monotone in time,
every word carries speaker + confidence ∈ [0,1], contested words list alternates, and every entry in
`iterations.json` references trigger values present in the belief store.

**Acceptance Scenarios**:

1. **Given** a completed run, **Then** `final/` contains `transcript.json`, `diarization.json`,
   `presence.parquet`, `convergence.json`, and `iterations.json` conforming to
   `contracts/final-outputs.md`.
2. **Given** a word where ≥ 2 model families agree, **Then** its confidence exceeds that of any
   single-family word, and family weights prevent double counting (two Whisper-derived models ≠ two
   independent votes).
3. **Given** contested words, **Then** alternates carry their vote shares and the LS bundle shows the
   final consensus track alongside the existing per-model tracks.

---

### User Story 5 - Bounded, budgeted convergence (Priority: P3)

An operator caps compute: `--max-rounds`, per-round and total intervention budgets by cost class, and
per-region intervention caps. The loop reports what it spent, what converged, and what it would have
done next with more budget.

**Why this priority**: Makes the loop safe to run unattended and its cost predictable.

**Independent Test**: Run with a tiny budget (`--budget-medium 2 --budget-heavy 0`) on a degraded clip;
verify the loop stops within budget, `convergence.json.budget` accounts for every spent unit, and
`next_actions` lists the top unfunded interventions.

**Acceptance Scenarios**:

1. **Given** exhausted medium budget, **Then** pending medium interventions defer, appear in
   `next_actions`, and the loop proceeds to fusion.
2. **Given** `--max-rounds 1`, **Then** no interventions run and outputs are the baseline single-shot
   set (see FR-024 compatibility).
3. **Given** any run, **Then** Σ logged costs in `iterations.json` = `convergence.json.budget.spent`.

## Requirements *(mandatory)*

### Loop control & rounds

- **FR-001**: The workflow MUST run as ordered rounds: triage (0), baseline evidence (1), intervention
  rounds (2..K, K = `--max-rounds`, default 3), fusion (final). Rounds MUST be skippable by
  configuration but not reorderable.
- **FR-002**: Triage MUST run only light signals (quality DSP, Brouhaha, segmentation-3.0 posteriors,
  AST/YAMNet, openSMILE) and MUST reuse the existing content-addressable cache keys (same task names,
  params) so triage results are shared with the baseline round.
- **FR-003**: With `--enhancement auto`, the enhanced pass MUST run only if triage finds at least one
  region with speech posterior ≥ θ_speech and quality degradation ≥ θ_enh. `always` and `never` MUST be
  supported; `--no-enhancement` remains an alias for `never`.
- **FR-004**: If triage finds no bucket with P(speech) ≥ θ_speech, the run MUST stop after emitting
  presence outputs, with `run_state: "no_speech"` and exit code 0.

### Belief store & aggregation

- **FR-005**: All model evidence MUST be recorded as votes in a persistent belief store
  (`contracts/belief-store.md`), keyed by (axis, bucket, source) where source = (model_id, stream,
  scope, round). Existing comparator votes are the round-1 population of this store.
- **FR-006**: Aggregation MUST be re-runnable from the belief store alone, without model inference
  (harvest/aggregate split). Re-aggregation after an intervention MUST touch only buckets covered by new
  votes.
- **FR-007**: Vote merge semantics: a region-scoped vote from (model, stream) MUST shadow the same
  (model, stream)'s file-scoped vote in covered buckets; votes from distinct models or streams coexist.
  Shadowed votes remain in the store (provenance), excluded from aggregation.
- **FR-008**: Cross-model aggregation MUST apply model-family weights (declared in policy) so that
  same-family models (e.g. Whisper-derived) do not double-count as independent evidence.
- **FR-009**: Each belief row MUST carry: round of last update, status ∈ {open, converged, irreducible,
  budget_exhausted}, elected stream, epistemic and aleatoric components, and (when irreducible) a
  machine-readable `irreducible_reason`.

### Region proposal & interventions

- **FR-010**: After each aggregation, contiguous high-uncertainty regions MUST be proposed per axis:
  seed at ≥ θ_high (default 0.66), expand while ≥ θ_low (default 0.33), merge gaps < g (default 0.5 s),
  pad ± pad_s (default 1.0 s) snapped outward to presence troughs where available, capped at top-N
  (default 8) by uncertainty mass per round.
- **FR-011**: Interventions MUST be declared as data (trigger predicate over belief state, action, cost
  class, guards, priority function) per `contracts/interventions.md`; the engine MUST rank runnable
  interventions by expected-gain/cost and execute within budget (`contracts/policy-engine.md`).
- **FR-012**: The v1 catalog MUST include at minimum: P2 fine-grid posterior re-analysis, P3
  hallucination adjudication, C9 missed-speech correction, U1 region re-ASR on elected stream, U2
  reserve-model escalation, U3 consensus re-alignment, I1 boundary refinement via embedding
  change-points, I4 overlap detection via per-class segmentation posteriors, and S1 stream election.
- **FR-013**: Region re-processing MUST crop with context padding via `extract_segments`, map timestamps
  back to file time, and merge back only words/frames whose midpoint lies in the core (unpadded) region
  (`contracts/region-reprocessing.md`).
- **FR-014**: Cropped-region model calls MUST flow through the existing cache (`cache_key` on the crop's
  own audio signature), so repeated runs and overlapping regions replay cached results.
- **FR-015**: Stream election (C5) MUST pick, per region, the stream whose belief scores best (presence
  confidence, quality, utterance agreement), MUST apply the enhancement-artifact guard (reject enhanced
  when raw-side phonetic evidence contradicts enhanced-side speech), and MUST record the election and its
  inputs.
- **FR-016**: Segmentation-3.0 per-class (powerset) posteriors MUST be exposed alongside the existing
  collapsed P(speech) (extension of `frame_posteriors.py:78-88`), yielding an overlap posterior used by
  I4 and the aleatoric floor (C7).

### Convergence & budget

- **FR-017**: A region converges when its axis uncertainty ≤ θ_low; it becomes irreducible when an
  intervention improves it by < ε (default 0.05) and `--max-region-rounds` (default 2) is exhausted, or
  when its aleatoric floor ≥ its uncertainty. `irreducible_reason` MUST name the dominant floor
  (`overlapping_speech`, `snr_floor`, `non_speech_vocalization`, `single_model_coverage`, ...).
- **FR-018**: Budgets MUST be enforced per cost class (light/medium/heavy) per round and per run;
  exceeding budget defers interventions into `convergence.json.next_actions` rather than dropping them
  silently.
- **FR-019**: The loop MUST terminate: hard cap `--max-rounds`; a round with zero fired interventions
  ends iteration early.
- **FR-020**: Every policy decision (fired or deferred) MUST be logged to `iterations.json` with
  trigger values, cost, and post-hoc uncertainty delta (`contracts/final-outputs.md`).

### Fusion & outputs

- **FR-021**: Fusion MUST produce `final/transcript.json` by time-aligned word-level voting over the
  final vote set (family-weighted, confidence-weighted), with speaker attribution from the unified
  clustering, alternates for contested slots, and confidences calibrated via the synthetic harness
  (prior spec US5) when a calibration profile exists — otherwise raw vote shares, flagged
  `calibrated: false`.
- **FR-022**: Fusion MUST produce `final/diarization.json` (unified clusters, refined boundaries with
  confidence), `final/presence.parquet` (fused p_voice + status), and `final/convergence.json`
  (per-axis convergence counts, uncertainty mass per round, budget accounting, irreducible regions with
  reasons).
- **FR-023**: The consensus transcript and final presence/diarization MUST be attached to the LS bundle
  as additional tracks; existing per-model tracks are unchanged.

### Compatibility, determinism, failure

- **FR-024**: `--max-rounds 1 --enhancement always` MUST reproduce today's outputs: the existing 9
  uncertainty parquets, per-task JSONs, LS bundle, disagreements.json and summary.json keys are
  unchanged (new keys/files strictly additive). Default flag values MUST preserve current behavior for
  the existing artifact set.
- **FR-025**: The loop MUST be deterministic given (audio, args, policy file, senselab version): stable
  intervention ordering (priority, then axis priority utterance > identity > presence, then region
  start), seeded clustering, and a policy hash recorded in provenance.
- **FR-026**: Any intervention failure MUST be caught, logged with the exception, leave the belief store
  unchanged for its buckets, and not abort the run (mirrors the comparator's failure envelope,
  `scripts/analyze_audio.py:2036-2040`).
- **FR-027**: All thresholds (θ_speech, θ_enh, θ_low, θ_high, ε, g, pad_s, N, budgets, family weights,
  reserve models) MUST live in a versioned policy file with CLI overrides — no hardcoded parameters
  (constitution VIII).

## Success Criteria *(mandatory)*

- **SC-001**: On a validation suite of clips with localized injected degradation (noise, clip, reverb
  spans), ≥ 70% of injected spans are proposed as regions in round 2, and ≥ 50% of them either reduce
  utterance uncertainty by ≥ 0.15 or carry a correct `irreducible_reason` by loop end.
- **SC-002**: On clean clips, `--enhancement auto` skips the enhanced pass and total wall-clock is
  ≤ 60% of today's two-pass run (warm cache excluded from measurement).
- **SC-003**: On silent clips, the run stops after triage in ≤ 15% of the full-run wall-clock.
- **SC-004**: Two consecutive runs with identical inputs and warm cache produce byte-identical
  `iterations.json` and `final/convergence.json`.
- **SC-005**: `--max-rounds 1 --enhancement always` output passes the existing regression suite with no
  changes to pre-existing artifacts (schema and values), verified by diffing a golden run.
- **SC-006**: Hallucination adjudication removes ≥ 80% of ASR tokens injected into music-only spans of
  the validation suite from utterance pairing, with ≤ 5% false-purge of true speech tokens.
- **SC-007**: Fused word confidences are monotonically related to correctness on the synthetic suite
  (higher-confidence deciles have lower WER); with a calibration profile, ECE ≤ 0.10.
- **SC-008**: Every intervention in `iterations.json` on the validation suite links to trigger values
  reproducible from the persisted belief store (audit-trail completeness = 100%).

## Out of Scope

- Streaming / online processing; the loop operates on complete files.
- LLM-driven planning or an agentic controller in the control path (a post-hoc advisory hook over
  `convergence.json` may be added later; explicitly not in v1).
- Human-in-the-loop actions mid-run; Label Studio remains post-hoc review.
- Model training / fine-tuning; per-model ensemble or MC-dropout uncertainty (already deferred by the
  compare-uncertainty spec).
- Separation-based speaker-attributed re-ASR of overlap regions (C11/U4): specified as v2, implemented
  behind a default-off flag only if time permits.
- Changing the existing 9-parquet contract, LS track names, or per-task JSON shapes (additive only).
- Corpus-level adaptation across files (each run is independent).

## Key Entities

See [data-model.md](./data-model.md): VoteStore, BeliefRow, Region, InterventionRecord, StreamElection,
RoundSummary, ConvergenceReport, FinalWord.

## Assumptions & Dependencies

- Builds on the merged comparator (compare-uncertainty spec) and the implemented US1–US3 of
  scene-quality-utterance (quality columns, sound sources, frame posteriors, per-axis grids). The
  utterance rework (US4: token logits) and calibration harness (US5) are pending there; this feature
  degrades gracefully without them (P3 uses fewer indicators; FR-021 flags `calibrated: false`).
- `extract_segments` (`senselab.audio.tasks.preprocessing`) provides cropping; content-addressable cache
  keys already incorporate the crop's audio signature, so region-level caching needs no cache change.
- Diarization remains whole-file (global speaker continuity); boundary repair is local via embeddings
  (research.md D3). AST cannot run on crops < 10.24 s — short-crop scene checks use YAMNet and frame
  posteriors (research.md D2).
