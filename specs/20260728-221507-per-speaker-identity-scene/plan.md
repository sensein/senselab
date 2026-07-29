# Implementation Plan: Per-Speaker Identity Uncertainty and Background Scene Characterization

**Branch**: `20260728-221507-per-speaker-identity-scene` | **Date**: 2026-07-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260728-221507-per-speaker-identity-scene/spec.md`

## Summary

Four related changes to the audio-analysis workflow:

1. **Per-speaker identity uncertainty** in the final convergence — a speaker-count
   distribution and per-speaker presence tracks replacing the single per-bucket
   identity scalar, with full participation in the adaptive loop under
   uncertainty-gated mutual influence.
2. **Pin the classifiers' level sensitivity** — both AST and YAMNet were measured to be
   amplitude-sensitive; turn that finding into a regression guard, document each
   model's floor, and fix a score-comparability defect the audit surfaced.
3. **Background characterization** on a foreground-suppressed variant, where detection
   works by per-band noise-floor subtraction rather than amplification.
4. **A background mask** marking target-free regions with its own uncertainty, driving
   introspection of what those regions contain.

**Technical approach.** Detection is reframed away from gain: estimate a bias-corrected
per-band noise floor, subtract it, and apply a single 3/6/10 dB margin ladder that holds
at every source distance. Amplification is demoted to classifier-input conditioning
capped at ~10 dB. The background mask is derived from existing presence/diarization
signals plus task metadata, and is expected to carry more evidential weight than deep
suppression, since a 30 dB suppression baseline was measured to fail outright.

**Sequencing is not priority order.** The P1 story (identity) is blocked on PR #537,
which edits the same files. The P2 stories are unblocked and ship first.

## Technical Context

**Language/Version**: Python 3.11–3.14 (repo `requires-python = ">=3.11,<3.15"`), managed via `uv`
**Primary Dependencies**: numpy, scipy, pandas + pyarrow (parquet), torch/torchaudio, transformers (AST), TensorFlow Hub (YAMNet), pyannote-audio (diarization, brouhaha SNR/C50), speechbrain (embeddings, enhancement), librosa (**promote transitive → explicit**: `pcen`, `A_weighting`), pyloudnorm (**new**, BS.1770 LUFS; numpy/scipy only)
**Storage**: File-based — parquet under `<run_dir>/<pass>/uncertainty/` and `<run_dir>/final/`, JSON for convergence/decision logs, content-addressable cache under `artifacts/analyze_audio_cache/`
**Testing**: `uv run pytest`, tests mirror package structure in `src/tests/`, files named `*_test.py`; GPU-gated tests behind the `ec2-gpu-test` label
**Target Platform**: macOS arm64 (dev + CI unit tests), Linux CUDA (GPU CI)
**Project Type**: Python library (`senselab`) plus a CLI wrapper script (`scripts/analyze_audio.py`)
**Performance Goals**: Default runs (no foreground suppression requested) complete within 10% of current wall-clock (SC-009). Amplitude-invariance probe runs on cached models only, no downloads.
**Constraints**: Uncalibrated audio — only relative/band-referenced quantities are defensible (FR-021c). Mono input; no directional localization. Lab-like close-microphone scope. Byte-reproducible outputs under mutual influence (FR-011f, SC-029). Gain hard-capped at ~10 dB (clipping inflection).
**Scale/Scope**: Recordings of seconds to tens of minutes, 1–5 speakers. 4 user stories, 82 functional requirements, 33 success criteria.

### Resolved unknowns

No `NEEDS CLARIFICATION` markers remain. The four research questions were resolved
before planning (see [research.md](./research.md)); implementation-level unknowns were
resolved by direct inspection of this repository and its virtualenv rather than by
further research, because they are facts about installed code:

| Unknown | Resolved |
|---|---|
| Does either classifier self-normalize level? | No. Both amplitude-sensitive; mechanisms quoted in research.md |
| Is `librosa.pcen` / `A_weighting` available? | Yes, librosa 0.11.0 — but librosa is **not declared** in `pyproject.toml` |
| Does a bias-corrected band floor estimator exist in-repo? | No. `quality_control/metrics.py:140,443` have uncorrected P10 floors |
| Are per-frame SNR and C50 available? | Yes, `scene_quality/brouhaha.py` `BrouhahaFrames.snr_db` / `.c50_db` |
| What does #537 touch? | `identity.py`, `clustering.py`, `stages.py`, `stage_context.py` + 4 new diarizers |

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Assessed against constitution v1.1.0.

| Principle | Status | Notes |
|---|---|---|
| **I. UV-Managed Python** | ✅ Pass | All commands via `uv run`. The amplitude probe and calibration scripts are `uv run` entry points. |
| **II. Encapsulated Testing** | ✅ Pass | `uv run pytest`; no host packages. Model-touching tests gated and cache-backed. |
| **III. Commit Early and Often** | ✅ Pass | Phasing below yields one commit per sub-task; each phase is independently committable. |
| **IV. CI Must Stay Green** | ✅ Pass | Each phase lands with its tests. The amplitude-invariance regression guard (FR-017b) must not require network — it uses cached models or skips. |
| **V. Memory-Driven Anti-Pattern Avoidance** | ✅ Pass | Registry items 1–4 all apply here; see Anti-pattern watch list below. |
| **VI. No Unnecessary API Calls** | ⚠️ Attention | The gain sweep runs classifiers repeatedly. MUST reuse cached models with `local_files_only`, and MUST cache probe results — a naive implementation would re-download or re-infer per gain. |
| **VII. Simplicity First** | ⚠️ Justified violation | 82 FRs and a mutual-influence architecture. See Complexity Tracking. |
| **VIII. No Hardcoded Parameters** | ✅ Pass, load-bearing | Every threshold in this feature (3/6/10 dB ladder, −23 LUFS, −45 dBFS reject, +10 dB cap, guard intervals, percentile *q*, occupancy, padding limits) MUST live in the versioned policy profile, not as literals. FR-022/FR-023 already require this. |

**Anti-pattern watch list for this feature** (from the registry):

1. *Mock cache pollution* — the amplitude probe tests will mock classifier pipelines;
   use `monkeypatch.setattr` to swap a fresh dict, never `.clear()` on the shared
   `_pipelines` / `_hf_cache`.
2. *Circular imports* — new modules live under `workflows/audio_analysis/`; the floor
   estimator must not be imported from `utils/dependencies.py`, and `utils/`-level code
   must not import `audio/` (the `cached_inference` Protocol pattern is the precedent).
3. *Debug prints* — use `senselab.utils.data_structures.logging.logger`. Note
   `stages.py` currently uses bare `print()` for pass banners; do not extend that.
4. *Broad exception catches for optional imports* — catch `(ImportError, RuntimeError)`
   for librosa/pyloudnorm/TensorFlow paths.

**Gate result: PASS** with one justified violation (VII) and one attention item (VI)
carried into the design.

## Project Structure

### Documentation (this feature)

```text
specs/20260728-221507-per-speaker-identity-scene/
├── plan.md              # This file
├── research.md          # Phase 0 — decisions, rationale, alternatives
├── data-model.md        # Phase 1 — entities and their fields
├── quickstart.md        # Phase 1 — how to run and validate
├── contracts/           # Phase 1 — output and CLI contracts
│   ├── cli.md
│   ├── speaker-identity.md
│   ├── background-mask.md
│   ├── background-sources.md
│   ├── level-verdicts.md
│   └── policy-profile.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (already present)
└── tasks.md             # Phase 2 — created by /speckit.tasks, NOT here
```

### Source Code (repository root)

```text
src/senselab/
├── audio/
│   ├── tasks/
│   │   ├── classification/          # AST + YAMNet — level probe targets
│   │   │   ├── huggingface.py       # MODIFY: score-comparability fix (FR-017c)
│   │   │   ├── yamnet.py            # MODIFY: amplify-before-serialize (FR-019b)
│   │   │   └── level_probe.py       # NEW: gain sweep + invariance verdicts (US2)
│   │   ├── quality_control/
│   │   │   └── metrics.py           # UNCHANGED: documents its own floor bias
│   │   └── speech_enhancement/      # reused for the suppression residual
│   └── workflows/
│       └── audio_analysis/
│           ├── level.py             # NEW: LUFS targets, gain policy, headroom (US2/US3)
│           ├── noise_floor.py       # NEW: per-band bias-corrected floor (US3)
│           ├── background_mask.py   # NEW: mask + uncertainty + introspection (US4)
│           ├── foreground.py        # NEW: suppressed variant + depth measure (US3)
│           ├── sources.py           # NEW: margin ladder, noise-character guards (US3)
│           ├── speaker_identity.py  # NEW: count posterior, per-speaker tracks (US1)
│           ├── identity.py          # MODIFY (⚠ #537 collision)
│           ├── clustering.py        # MODIFY (⚠ #537 collision)
│           ├── stages.py            # MODIFY (⚠ #537 collision)
│           ├── stage_context.py     # MODIFY (⚠ #537 collision)
│           ├── calibration.py       # MODIFY: detection-margin profile
│           └── adaptive/
│               ├── influence.py     # NEW: uncertainty-gated weighting (FR-011b/c)
│               ├── provenance.py    # NEW: revision attribution (FR-011g)
│               ├── belief.py        # MODIFY: per-speaker state
│               ├── convergence.py   # MODIFY: oscillation detection (FR-011e)
│               ├── interventions.py # MODIFY: count-disagreement trigger
│               ├── fusion.py        # MODIFY: per-speaker final outputs
│               └── policy/default.yaml  # MODIFY: all new thresholds
├── utils/                           # unchanged
scripts/
├── analyze_audio.py                 # MODIFY: new flags (⚠ #523 collision, minor)
├── probe_classifier_levels.py       # NEW: standalone amplitude probe
└── calibrate_detection_margin.py    # NEW: margin derivation + provenance

src/tests/audio/
├── tasks/classification/level_probe_test.py          # NEW
└── workflows/audio_analysis/
    ├── noise_floor_test.py                           # NEW
    ├── background_mask_test.py                       # NEW
    ├── sources_test.py                               # NEW
    ├── speaker_identity_test.py                      # NEW
    └── adaptive/influence_test.py                     # NEW
```

**Structure Decision**: Extend the existing `workflows/audio_analysis/` package rather
than create a new top-level module. That package already owns the three-axis comparator,
the adaptive loop, and the calibration bridge, and this feature is squarely more of the
same. New concerns get their own modules (`level`, `noise_floor`, `background_mask`,
`foreground`, `sources`, `speaker_identity`) instead of being added to `stages.py`,
which is already ~600 lines and is being edited by #537 — keeping new code out of it
minimises merge conflict surface. The adaptive additions (`influence.py`,
`provenance.py`) are new files for the same reason.

## Implementation Phasing

Ordered by *unblocked-ness and risk*, not by spec priority. Rationale: the P1 story
collides with an open PR, while both P2 stories are independent of it.

| Phase | Story | Blocked by | Independently shippable |
|---|---|---|---|
| **A** | US2 — pin level sensitivity, fix score comparability | — | Yes |
| **B** | US4 — background mask + uncertainty + introspection | — | Yes |
| **C** | US3 — noise floor, margin ladder, suppression, background sources | A, B | Yes |
| **D** | US1 guards — influence weighting, provenance, oscillation detection | — | No (infrastructure) |
| **E** | US1 — per-speaker identity + full loop participation | #537, D | Yes |

**Phase D before E is mandatory**, per the spec's Dependencies note: the guards against
self-confirmation and oscillation must exist before any influence path can exercise
them, so the loop is never able to confirm its own edits even transiently.

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1. Gate result: **PASS**, with the same single justified
violation.

| Principle | Post-design status |
|---|---|
| **VI. No Unnecessary API Calls** | ✅ **Resolved from ⚠**. `contracts/cli.md` requires `probe_classifier_levels.py` to use cached checkpoints only and to skip with a clear message when a model is absent; `contracts/level-verdicts.md` requires the regression guard to run without network. Probe results are persisted to `artifacts/level_probe/`, so the sweep runs once rather than per consumer. |
| **VII. Simplicity First** | ⚠️ Still violated, still justified — see Complexity Tracking. Phasing mitigates it: each of A/B/C/E ships independently, so the 82-requirement surface is never landed at once. |
| **VIII. No Hardcoded Parameters** | ✅ Strengthened. `contracts/policy-profile.md` makes every threshold in the feature a profile entry with a default: the 3/6/10 dB ladder, −23 LUFS, −45 dBFS reject, +10 dB cap, quantile *q*, guard interval, occupancy, flatness limit, and the quarantine label list. The bias correction is *computed* from *q* rather than stored, so the two cannot drift apart. |
| **III / IV. Commit cadence, CI green** | ✅ Each phase carries its own tests; the level-probe guard is offline-safe so it cannot flake on network. |
| **V. Anti-patterns** | ✅ Watch list carried into the design; the four registry items relevant here are named above. |

**New consideration introduced by the design**: two dependency changes (promote `librosa`,
add `pyloudnorm`). Both are justified in research D14. Note a related finding — the prior
scene-quality spec (`20260722-175022`) listed "librosa (promote from transitive →
explicit)" in its technology set, but `pyproject.toml` still has no `librosa` entry, so
that promotion never landed. This feature depends on `librosa.pcen` and
`librosa.A_weighting` directly, so the promotion must actually happen here rather than be
assumed done.

## Complexity Tracking

> Constitution principle VII (Simplicity First) is violated. Justifications:

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| **Mutual-influence architecture** (FR-011a–h): bidirectional signal influence with uncertainty-gated weighting, provenance, and oscillation detection | Explicitly chosen by the user over report-only and trigger-only options: "all signals should be able to iteratively influence each other towards convergence, with uncertainty gating use" | Report-only (option A) and trigger-only (option B) were both offered and declined. Report-only would leave the count-disagreement signal inert — it would detect the multi-speaker case that motivated this work and then do nothing with it. |
| **82 functional requirements in one spec** | The four asks are genuinely coupled: the level finding determines the gain policy, the gain policy needs the floor, the floor needs the mask to know where it can be trusted, and the mask needs task metadata. Splitting them would create specs that cannot be validated independently. | Four separate specs rejected because US3's success criteria are unverifiable without US2's measured floors and US4's mask. The phasing above recovers the benefit — each phase ships alone. |
| **Six new workflow modules** | Distinct concerns with distinct test surfaces; `stages.py` is already large and under concurrent edit by #537 | Adding to `stages.py` rejected on merge-conflict grounds alone. Fewer, larger modules rejected because the floor estimator and the mask have no shared state and would only be co-located by accident. |
| **New dependency: pyloudnorm** | BS.1770 loudness is correctness-critical (gated integrated loudness, K-weighting filter coefficients, true-peak oversampling) and easy to get subtly wrong by hand | Hand-implementing K-weighting rejected: the standard's two biquad stages plus the two-stage gating and 4× oversampled true-peak are ~150 lines of standards-compliance code we would then own and have to validate. pyloudnorm is numpy/scipy-only, has no transitive weight, and is validated to ±0.1 LU. |
| **Promoting librosa to an explicit dependency** | `pcen` and `A_weighting` are used directly; relying on it transitively is already fragile and would break silently on an upstream change | Not a real alternative — using a transitive dependency directly is the bug, not the simplification. |
