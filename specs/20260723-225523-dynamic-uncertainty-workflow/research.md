# Research & Decision Record: Uncertainty-driven adaptive analysis workflow

**Branch**: `20260723-225523-dynamic-uncertainty-workflow` | **Date**: 2026-07-23

Codebase findings this design is grounded in:

- The pipeline is a single feed-forward pass; uncertainty is computed once and only reported
  (`scripts/analyze_audio.py:1861-2264`, `compute.py:48-130`). No spec to date proposes acting on it —
  compare-uncertainty explicitly defers ensemble-style refinement, and scene-quality-utterance's only
  cross-signal element (`scene_quality_coupling`, FR-019) is a static one-directional multiplier.
- `compute_uncertainty_axes` interleaves expensive harvesting (embedding extraction per model per pass,
  frame posteriors, Brouhaha, per-bucket g2p) with cheap pure aggregation, and mutates the caller's
  `passes` dict (synthetic silhouette source, `compute.py:211-236`). There is no re-aggregate entry
  point: changing only the aggregator re-runs all GPU inference.
- The content-addressable cache keys on the *audio signature* of the exact waveform
  (`scripts/analyze_audio.py:741-806`), so cropped segments cache correctly with zero cache changes.
  ASR and alignment caches are already independent (FR-024 of the asr-extensions spec).
- `frame_posteriors.py:75-88` computes segmentation-3.0's full powerset class posteriors and then
  collapses them to a scalar P(speech) = `1 − P(no-speaker class)` (max over the class axis only as a
  multilabel fallback); the per-class posteriors — including the overlap classes — are discarded before
  return, so exposing an overlap posterior is a small additive extension.
- Existing one-directional couplings to preserve: silhouette voter injection, empirical cosine
  calibration floors, presence intensity mask (`compute.py:211-236, 441-456, 405-418`).

## D1 — Control: deterministic policy engine (not an LLM agent)

**Decision**: The loop is driven by declarative rules: `trigger predicate over belief state → action →
cost class → guards → priority = expected_gain / cost`. Stable tiebreaks (priority, axis priority
utterance > identity > presence, region start). Policy is data (D10); the engine is a pure function of
(belief state, regions, budget, policy).

**Rationale**: senselab's value proposition is reproducible pipelines; every existing artifact carries
provenance and cache keys. A stochastic planner would break SC-004 (byte-identical decision logs) and
make irreducibility claims unauditable. Rules are unit-testable without models.

**Alternatives considered**: (a) LLM agent planner — flexible, but non-deterministic, unauditable,
and adds a serving dependency to a batch pipeline; rejected for the control path, noted as a possible
post-hoc *advisor* over `convergence.json` (out of scope). (b) Bandit/RL selection — needs reward
data we don't have; the expected-gain heuristics encode the same idea transparently. (c) Hybrid
(deterministic core + LLM for irreducible regions) — deferred with (a); the hook point is
`next_actions` in convergence.json.

## D2 — Re-processing unit: cropped regions with padding and midpoint merge-back

**Decision**: Interventions operate on `extract_segments` crops of `[start − pad, end + pad]`
(pad_s = 1.0 default), boundaries first quantized to the reporting grid, snapped outward to the nearest
presence trough (p_voice local minimum < 0.2) within the pad when one exists. Crop-local outputs are
offset-mapped to file time; only words/frames whose **midpoint lies inside the core region** merge back
into the vote store. Min-length rules: ASR crops ≥ 1.0 s post-pad; AST is never run on crops (10.24 s
native window, `scripts/analyze_audio.py:61-67`) — short-crop scene checks use YAMNet (0.96 s) and
frame posteriors.

**Rationale**: Whole-file re-runs make per-region iteration unaffordable; crops bound cost by
uncertainty mass. Padding gives ASR acoustic context; midpoint merge-back avoids double-edged words;
grid quantization makes crop signatures repeatable → cache hits across rounds and runs.

**Alternatives**: whole-file re-runs with different models only (cost scales with file, not with
uncertainty; rejected as the primary mechanism, still reachable as a heavy-class rule); VAD-segment
units (couples the unit to one signal — the thing we're uncertain about); fixed tiling (ignores
uncertainty structure).

## D3 — Diarization stays whole-file; identity repair is local

**Decision**: Never re-run diarizers on crops. Identity interventions are: fine-hop re-embedding +
calibrated change-point evidence (I1), overlap posterior (I4), and re-clustering with updated evidence.

**Rationale**: Diarizers cluster globally; a crop severs speaker continuity and returns labels that
cannot be mapped back reliably. Embeddings already provide a diarization-independent, localizable
identity signal (`embeddings.py:91`, cosine calibration `embeddings.py:666`), and label unification
across models exists (`clustering.py:202`).

## D4 — Enhancement: conditional whole-file + per-region stream election (no per-region enhancement in v1)

**Decision**: `--enhancement auto` runs SepFormer once on the whole file iff triage finds degraded
speech (C1). Downstream, election (S1) picks per region which stream's evidence to trust and which
stream re-processing uses (C5), guarded: the enhanced stream cannot win where raw-side phonetic evidence
(PPG activity, ASR agreement on raw) contradicts enhanced-side speech — SepFormer can hallucinate
speech-like energy from noise.

**Rationale**: Keeps the raw/enhanced pairing that the delta axes and existing contracts assume
(`raw_vs_enhanced` parquets), while removing the always-2× cost and adding the per-region choice the
one-shot pipeline can't make. Per-region enhancement creates boundary artifacts at crop seams and a
combinatorial pass space; deferred.

## D5 — Vote merge semantics and family weights

**Decision**: Vote key = (axis, bucket, model_id, stream, scope, round). Region-scoped votes **shadow**
file-scoped votes of the same (model_id, stream) in covered buckets (shadowed rows persist for
provenance, excluded from aggregation — most-specific-scope-wins, latest-round-wins within equal
scope). Distinct models/streams coexist. Aggregation weights each model by `1 / family_size` from a
policy-declared family map (whisper-derived = {openai/whisper-*, nyralabs/CrisperWhisper*}, nemo,
qwen, granite, ...), so intra-family agreement doesn't masquerade as independent evidence.

**Alternatives**: averaging region+file votes of the same model (double-counts the same model on the
same audio); full covariance modeling between models (no data to fit it; family buckets are an honest
coarse prior).

## D6 — Convergence thresholds and budget classes

**Decision**: Reuse the established bins — θ_high = 0.66 (seed), θ_low = 0.33 (converged) — matching
disagreements/LS semantics. Improvement floor ε = 0.05 per intervention; `--max-region-rounds 2`;
budget classes: **light** (no model load: DSP, re-aggregation, adjudication over existing evidence),
**medium** (one model forward on one crop: fine posteriors, re-embed, one ASR on crop, re-align),
**heavy** (reserve-model load, separation, any whole-file re-run). Defaults: medium ≤ 24/run,
heavy ≤ 4/run, both overridable (FR-018).

**Rationale**: consistent thresholds keep parquet/LS/loop semantics aligned; cost classes make budgets
hardware-portable where wall-clock caps aren't.

## D7 — Epistemic vs aleatoric split, and irreducibility

**Decision**: Per bucket: `epistemic` = the cross-source disagreement component (what more/better
evidence can reduce); `aleatoric_floor` = calibrated max of quality-driven floor (from SNR/clip/reverb
degradation) and overlap posterior (C7). A region is `irreducible` when interventions stop helping
(FR-017) **and** the floor explains the residual (residual ≤ floor + ε), with `irreducible_reason`
naming the dominant term; otherwise it exhausts budget as `budget_exhausted` (distinct status — honest
about "we stopped" vs "nothing more to learn").

**Rationale**: "Robust conclusion" requires saying *why* uncertainty remains. Overlap and SNR floors
are the two dominant, measurable aleatoric sources in this pipeline.

## D8 — Harvest/aggregate split with a persistent VoteStore (prerequisite)

**Decision**: Split `compute_uncertainty_axes` into `harvest_all(...) → VoteStore` (expensive,
model-touching) and `aggregate_all(VoteStore, policy) → axis results` (pure, cheap, covered-bucket
incremental). The synthetic silhouette source becomes store rows instead of a `passes` mutation. The
public signature is preserved by a thin wrapper.

**Rationale**: Iteration is only affordable if re-scoring is free. Also fixes two flagged defects:
caller-visible mutation and no re-aggregate entry point (aggregator sweeps currently re-run all GPU
work).

## D9 — Fusion: time-aligned word-level voting + consensus re-alignment

**Decision**: Build word slots by grouping final-vote words across models with midpoint containment /
time-IoU ≥ 0.3; candidates weighted by family weight × model confidence (logprob / CTC / alignment
score when present); winner by weighted share, alternates kept when the margin is below a policy
threshold. Speaker attribution = unified cluster at word midpoint. Then C8: force-align the consensus
text once (Qwen aligner default, MMS fallback — same backends as the script) for authoritative
timestamps. Confidence calibration via the synthetic-harness profile when present (`calibrated: false`
otherwise).

**Alternatives**: pick-single-best-model (discards the ensemble the pipeline already paid for);
lattice/confusion-network combination at decoder level (backends don't expose lattices uniformly).

## D10 — Policy-as-data

**Decision**: All thresholds, budgets, family weights, reserve models, and rule enable/disable live in
`adaptive/policy/default.yaml`; CLI flags override; `sha256(policy_yaml)` recorded in every round's
provenance and in `iterations.json`.

## D11 — Failure envelope

**Decision**: Each intervention runs inside the same failure envelope as the comparator
(`scripts/analyze_audio.py:2036-2040`): exceptions are caught, logged into `iterations.json` with
`status: "failed"`, the belief store is untouched for its buckets, and the loop continues.

## D12 — Deferred to v2

Separation-based speaker-attributed re-ASR for overlap regions (U4/C11 — SepFormer separation + per-source
embedding matching + per-source ASR; expensive and needs its own validation), LID-gated re-ASR (U6),
LLM advisory hook over `convergence.json`, per-region enhancement, corpus-level policy learning.
