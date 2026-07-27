# Implementation Plan: Uncertainty-driven adaptive analysis workflow

**Branch**: `20260723-225523-dynamic-uncertainty-workflow` | **Date**: 2026-07-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/20260723-225523-dynamic-uncertainty-workflow/spec.md`

## Summary

Turn the single-shot analyze_audio pipeline into a bounded, deterministic, uncertainty-driven loop.
The three existing axes (presence / identity / utterance) become the control signal: a cheap triage
round gates enhancement and heavy tasks (C1/C2); a baseline round populates a persistent **belief
store** of provenance-tagged votes; intervention rounds propose high-uncertainty regions and execute a
declarative, budgeted **intervention catalog** on cropped audio (re-ASR on the elected stream, model
escalation, hallucination adjudication, boundary refinement, overlap detection); a fusion round emits a
consensus transcript, refined diarization, fused presence, and a complete decision audit trail. The
prerequisite refactor is a **harvest/aggregate split** of `compute_uncertainty_axes` so aggregation is
a pure, cheap function of the vote store. Decision record in [research.md](./research.md) (D1–D12).

## Technical Context

**Language/Version**: Python 3.11–3.12 (repo `requires-python = ">=3.11,<3.15"`), managed via uv
**Primary Dependencies**: no new runtime deps. Reuses pyannote-audio (segmentation-3.0 per-class
posteriors, Brouhaha), transformers, speechbrain (embeddings, enhancement), existing subprocess-venv ASR
backends, pandas/pyarrow, jiwer; PyYAML (already transitive via pre-commit/HF) for the policy file —
promote to explicit if not already a direct dep.
**Storage**: File-based — belief store + per-round artifacts under `<run_dir>/rounds/<k>/`; final
outputs under `<run_dir>/final/`; existing 9 uncertainty parquets, per-task JSONs, LS bundle unchanged.
Model-call caching stays in `artifacts/analyze_audio_cache/` (crop calls get distinct audio signatures).
**Testing**: pytest via `uv run pytest`; unit tests under `src/tests/audio/workflows/audio_analysis/`
(policy engine, merge semantics, region proposal, fusion — all pure-function testable without models);
GPU CI for end-to-end loop on the validation suite.
**Target Platform**: macOS arm64 (unit CI) + Linux/EC2 GPU (model-heavy paths); library + thin script.
**Project Type**: Single-project Python library (senselab) — extends `audio/workflows/audio_analysis`.
**Performance Goals**: Triage ≤ 15% of full-run cost; interventions bounded by explicit budget; clean
audio ≤ 60% of today's wall-clock (SC-002/003); re-aggregation after an intervention is O(covered
buckets), no model inference (FR-006).
**Constraints**: Backward compatibility (FR-024): `--max-rounds 1 --enhancement always` reproduces
today's artifact set; new outputs strictly additive. Determinism (FR-025): policy hash + seeded
clustering + stable ordering. No hardcoded parameters (FR-027).
**Scale/Scope**: One new subpackage (~8 files), a refactor of `compute.py` into harvest/aggregate, one
small extension to `frame_posteriors.py`, script restructuring into stage functions, policy YAML,
tests. ~10 new files, ~6 edited files.

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. UV-Managed Python | ✅ | All commands `uv run …`; any dep promotion via `uv add`. |
| II. Encapsulated Testing | ✅ | Policy engine / merge / region / fusion are pure functions — unit-tested without models; model paths mocked; loop e2e on GPU CI. |
| III. Commit Early and Often | ✅ | One commit per phase (A–E below). |
| IV. CI Must Stay Green | ✅ | Phase A lands behavior-neutral; regression goldens guard FR-024. |
| V. Anti-Pattern Avoidance | ✅ | `logger` not `print` in library code; `monkeypatch.setattr` mock isolation; optional-model `(ImportError, RuntimeError)` guards. |
| VI. No Unnecessary API Calls | ✅ | No new gated models; existing loaders (`ensure_hf_model`, pinned revisions) reused; interventions only load models already declared in policy. |
| VII. Simplicity First | ⚠️ justified | A loop + policy engine is intrinsically more machinery than a feed-forward script. Mitigation: policy-as-data, pure aggregation core, `--max-rounds 1` degenerates to today's pipeline. See Complexity Tracking. |
| VIII. No Hardcoded Parameters | ✅ | Every threshold/budget/weight in `policy/default.yaml` with CLI overrides (FR-027). |

**Gate result**: PASS (one justified deviation recorded below).

## Architecture

### Control flow

```text
main()
 ├─ round 0  TRIAGE (raw_16k only)
 │    quality DSP + brouhaha + seg3-posteriors(+per-class) + AST/YAMNet + openSMILE
 │    → presence prior, quality map, source masses, overlap posterior
 │    → decisions: no_speech? (FR-004 stop)   enhancement auto? (FR-003)
 ├─ round 1  BASELINE
 │    run remaining stages on raw (+ enhanced if elected): diar, ASR, align, PPG
 │    comparator HARVEST → VoteStore ;  AGGREGATE → BeliefState v1
 │    stream election per coarse region (S1)
 ├─ rounds 2..K  INTERVENE (while budget ∧ open regions ∧ k ≤ max_rounds)
 │    propose_regions(belief, axis)               [regions.py]
 │    plan = policy.match_and_rank(belief, regions, budget)   [policy.py]
 │    for each planned intervention: execute → new votes → VoteStore
 │    AGGREGATE (covered buckets only) → BeliefState v(k) ; convergence marks
 ├─ FUSION
 │    consensus transcript (family/confidence-weighted word voting) + C8 re-alignment
 │    final diarization (unified clusters + refined boundaries), fused presence
 │    convergence report + iterations log
 └─ existing outputs (9 parquets, LS bundle, disagreements.json, plot) + final/ + rounds/
```

### New module layout

```text
src/senselab/audio/workflows/audio_analysis/adaptive/
├── __init__.py          # run_adaptive_loop(...) public API
├── belief.py            # VoteStore, BeliefState, merge/shadow semantics (FR-005..009)
├── regions.py           # region proposal (FR-010), presence-trough snapping
├── policy.py            # rule schema, ranking, budget, determinism (FR-011, FR-025)
├── interventions.py     # catalog: P2,P3,C9,U1,U2,U3,I1,I4,S1 (FR-012)
├── reprocess.py         # crop/pad/offset-map/merge-back via extract_segments (FR-013..014)
├── election.py          # stream election + enhancement-artifact guard (FR-015)
├── convergence.py       # thresholds, irreducibility, budget accounting (FR-017..019)
├── fusion.py            # word-level voting, consensus re-alignment, final outputs (FR-021..023)
└── policy/default.yaml  # all thresholds/budgets/family weights (FR-027)
```

Touched existing code:

- `compute.py` — split into `harvest_all(...) -> VoteStore` and `aggregate_all(store, ...) ->
  {(pass,axis): AxisResult}`; `compute_uncertainty_axes` becomes a thin wrapper (harvest → aggregate)
  preserving its signature and stopping the in-place mutation of `passes` (`compute.py:236`) by
  returning the synthetic-source injection as store entries instead.
- `frame_posteriors.py` — expose per-class powerset posteriors + `overlap_posterior` alongside the
  existing collapsed P(speech) (`:78-88`); additive return field, existing callers unchanged (FR-016).
- `scripts/analyze_audio.py` — `run_pass` decomposed into per-stage functions (`stage_diarization`,
  `stage_scene`, `stage_features`, `stage_asr`, `stage_alignment`, `stage_ppg`) with unchanged task
  names/params (cache-key stability); `main` delegates to `run_adaptive_loop` with a legacy path for
  `--max-rounds 1 --enhancement always`.
- `disagreements.py`, `labelstudio.py`, `plot.py` — additive: final consensus track, per-round deltas.

### Key design points (full record in research.md)

- **D1 deterministic policy engine** — declarative rules, priority = expected_gain/cost, stable
  tiebreaks; no LLM in the control path.
- **D2 region cropping** — pad ± 1.0 s, snap to presence troughs, midpoint merge-back, min-length rules
  (AST never on crops; YAMNet/posteriors for short crops); cache-native via crop audio signature.
- **D3 diarization stays whole-file** — global clustering context; local repair via embedding
  change-points (I1) instead of re-diarizing crops.
- **D4 conditional whole-file enhancement + per-region stream election** — not per-region enhancement
  in v1; enhancement-artifact guard before trusting the enhanced stream.
- **D5 vote merge** — shadow same (model, stream) at region scope; family weights against
  double-counting.
- **D6 convergence** — θ reuse (0.33/0.66), ε = 0.05, max-region-rounds = 2, cost classes
  light/medium/heavy.
- **D7 epistemic/aleatoric split** — epistemic = cross-source disagreement; aleatoric floor =
  f(quality, overlap posterior); irreducibility requires the floor to explain the residual.
- **D8 harvest/aggregate split** — prerequisite for cheap iteration; VoteStore is the only interface
  between rounds.
- **D9 fusion** — ROVER-style time-aligned word voting; C8 consensus re-alignment for authoritative
  timestamps.
- **D10 policy-as-data** — YAML, hashed into provenance.
- **D11 failure envelope** — intervention failures logged, belief untouched, run continues (FR-026).
- **D12 v2 deferrals** — separation-based overlap re-ASR (U4/C11), LID re-run (U6), LLM advisor.

## Project Structure

### Documentation (this feature)

```text
specs/20260723-225523-dynamic-uncertainty-workflow/
├── spec.md
├── plan.md              # This file
├── research.md          # Decision record D1–D12
├── data-model.md        # VoteStore / BeliefRow / Region / InterventionRecord / ...
├── quickstart.md        # How to run & validate
├── contracts/
│   ├── belief-store.md
│   ├── policy-engine.md
│   ├── interventions.md
│   ├── region-reprocessing.md
│   ├── final-outputs.md
│   └── cli.md
└── tasks.md             # created by /speckit.tasks (NOT here)
```

## Phasing (each phase lands green and independently valuable)

- **Phase A — harvest/aggregate split + VoteStore (behavior-neutral)**: refactor `compute.py`;
  `compute_uncertainty_axes` output bit-identical on goldens; VoteStore persisted under
  `rounds/1/belief/`. Unlocks everything; zero user-visible change.
- **Phase B — triage + gating (C1, C2) + stage decomposition of run_pass**: `--enhancement auto`,
  no-speech early exit, cache-key stability tests. Immediate compute wins (US1).
- **Phase C — loop core (US2 + parts of US3)**: regions.py, policy.py, budget/convergence,
  interventions P2, P3, C9, U1, U2, S1; `rounds/<k>/` artifacts; iterations.json.
- **Phase D — identity/overlap repair + fusion (US3 + US4)**: I1, I4 (needs FR-016 posterior
  extension), U3 consensus re-alignment, fusion.py, final/ outputs, LS consensus track,
  convergence.json.
- **Phase E (v2, optional)**: U4 separation re-ASR behind `--enable-overlap-separation`, U6 LID,
  advisory hook.

Dependencies: A → B → C → D → E. B is shippable without C. Regression goldens (FR-024/SC-005) are
created in A and enforced from B onward.

## Complexity Tracking

| Deviation | Why needed | Simpler alternative rejected because |
|---|---|---|
| Policy engine + intervention catalog (VII) | The loop's value is *selective* compute; selection logic must be inspectable, testable, and overridable | Hardcoded if/else loop: untestable in isolation, violates FR-027, invites hidden coupling |
| Persistent per-round belief artifacts | Determinism audits (SC-004/008) and post-hoc debugging require replayable state | In-memory only: audit trail impossible, irreducibility unverifiable |
| Two execution paths in script (legacy vs loop) during transition | FR-024 golden-diff guarantee while phases land | Big-bang switch: violates "CI stays green", high regression risk |

## Risks & Mitigations

- **Oscillation / thrash** (intervention helps one model, hurts another): monotonicity guard (FR-017)
  + per-region round cap; interventions never delete evidence, only add or shadow.
- **Crop-boundary artifacts**: context padding + midpoint merge-back (D2); tests with words straddling
  crop edges.
- **Enhancement hallucination** (SepFormer inventing speech): election guard (FR-015) requires raw-side
  phonetic corroboration before the enhanced stream wins.
- **Correlated evidence inflation**: family weights (FR-008) with tests that adding whisper-small to
  whisper-large barely moves aggregated confidence.
- **Cache stampede on tiny region variations**: region boundaries quantized to the reporting grid before
  cropping, so re-runs hit identical crop signatures.
- **US4/US5 of scene-quality-utterance unfinished**: P3 and calibration degrade gracefully (spec
  Assumptions); loop lands independent of token-logit plumbing.
