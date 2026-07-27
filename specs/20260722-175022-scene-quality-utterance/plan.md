# Implementation Plan: Scene-aware presence axis + improved utterance uncertainty

**Branch**: `20260722-175022-scene-quality-utterance` | **Date**: 2026-07-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/20260722-175022-scene-quality-utterance/spec.md`

## Summary

Rework the `audio_analysis` workflow's **presence** axis into a scene-aware dimension and improve the **utterance** estimator, without breaking downstream consumers. The presence axis keeps its name and `aggregated_uncertainty` output; all new signals are **additive columns**. The work adds (1) per-bucket audio-quality degradation scores (SNR, clipping, reverb, bandwidth) built from existing senselab DSP plus a new `pyannote/brouhaha` model, (2) background sound-source category masses (speech/people/machine/environment) from a hand-authored AudioSet map over full AST/YAMNet distributions, (3) continuous frame-level speech posteriors (`pyannote/segmentation-3.0` + Brouhaha VAD) with per-axis grids and a confidence/uncertainty split, and (4) an overlapping word-scale utterance grid, calibrated confidence, token-level entropy (via newly-plumbed Whisper token logits + extended `ScriptLine`), and recorded scene-quality coupling. A synthetic clean+noise+RIR calibration harness fits the dB→[0,1] normalizations. See [research.md](./research.md) for the decision record (D1–D10).

## Technical Context

**Language/Version**: Python 3.11–3.12 (repo `requires-python = ">=3.11,<3.15"`), managed via uv
**Primary Dependencies**: pyannote-audio (existing — adds `segmentation-3.0` raw-scores + `brouhaha` via `Model`/`Inference`), transformers (AST + Whisper token logits), torchaudio + torchaudio-squim (existing), librosa (promote from transitive → explicit), numpy/scipy (calibration), pandas/pyarrow (existing parquet), jiwer (existing)
**Storage**: File-based — parquet under `<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet`; checked-in category map JSON and calibration profile JSON under the package; validation artifacts under `artifacts/`
**Testing**: pytest via `uv run pytest`; unit tests under `src/tests/audio/workflows/audio_analysis/` and `src/tests/audio/tasks/{voice_activity_detection,speech_to_text}/`
**Target Platform**: macOS arm64 (unit CI) + Linux/EC2 GPU (model-heavy paths); library, no server
**Project Type**: Single-project Python library (senselab) — extends existing `audio` module
**Performance Goals**: Model inference is per-pass (once on whole 16 kHz mono audio), then bucketed; no per-bucket model calls. Quality DSP runs on a coarse 0.5 s analysis window. Adds ≤2 pyannote forward passes (segmentation-3.0, brouhaha) per pass beyond today's pipeline.
**Constraints**: Backward compatibility — `presence` axis name, parquet paths, LS track names, and `aggregated_uncertainty` values unchanged; new signals additive and null-safe when a model is unavailable (FR-023). Gated model loads must use `ensure_hf_model`/`local_files_only` (constitution VI).
**Scale/Scope**: One workflow module + one new VAD frame-posterior function + one new model loader + `ScriptLine`/Whisper token plumbing + a scripts calibration helper + tests. ~5 new files, ~8 edited files.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. UV-Managed Python | ✅ | All commands `uv run …`; librosa added via `uv add`. |
| II. Encapsulated Testing | ✅ | Tests under `uv run pytest`; model-heavy paths guarded/mocked in unit tests, real-model checks on GPU CI. |
| III. Commit Early and Often | ✅ | One commit per phase/sub-task (spec already committed separately). |
| IV. CI Must Stay Green | ✅ | pre-commit + macOS unit tests per push; no watch loops. |
| V. Anti-Pattern Avoidance | ✅ | No `print` (use `logger`); mock isolation via `monkeypatch.setattr` (memory `feedback_mock_cache_pollution`); `(ImportError, RuntimeError)` guards for optional model imports; no circular imports in `dependencies.py`. |
| VI. No Unnecessary API Calls | ✅ | Brouhaha + segmentation-3.0 load via `ensure_hf_model` + `local_files_only`; models validated/pinned by revision (memory `project_model_revision_pinning`). |
| VII. Simplicity First | ⚠️ justified | New signals are always-on additive columns (no feature flag). Backward-compat of `aggregated_uncertainty`/paths is a real need (shared library, b2aiprep consumers), not a speculative shim — see Complexity Tracking. |
| VIII. No Hardcoded Parameters | ✅ | Grids, model ids/revisions, `--scene-top-k`, category-map path, calibration clip/SNR/RT60/output paths all parameters with defaults. |

**Gate result**: PASS (one justified deviation recorded below). Re-check after Phase 1: **PASS — no new violations introduced by the design.**

## Project Structure

### Documentation (this feature)

```text
specs/20260722-175022-scene-quality-utterance/
├── plan.md              # This file
├── spec.md              # Feature spec (committed)
├── research.md          # Phase 0 — decision record D1–D10
├── data-model.md        # Phase 1 — entities & column schema
├── quickstart.md        # Phase 1 — how to run & validate
├── contracts/           # Phase 1 — interface contracts
│   ├── frame_posteriors.md
│   ├── quality.md
│   ├── sound_sources.md
│   ├── presence-parquet-columns.md
│   ├── utterance.md
│   └── cli.md
└── tasks.md             # Phase 2 — created by /speckit.tasks (NOT here)
```

### Source Code (repository root)

```text
src/senselab/audio/
├── tasks/
│   ├── voice_activity_detection/
│   │   ├── frame_posteriors.py        # NEW — segmentation-3.0 + brouhaha per-frame arrays
│   │   └── api.py                     # (unchanged public API; new fn exported)
│   ├── scene_quality/                 # NEW package — brouhaha loader (VAD/SNR/C50)
│   │   ├── __init__.py
│   │   └── brouhaha.py
│   ├── classification/
│   │   └── api.py                     # EDIT — honor full top_k already; call sites raise top_k
│   ├── speech_to_text/
│   │   └── huggingface.py             # EDIT — Whisper output_scores → token entropy/avg_logprob
│   └── features_extraction/
│       └── (reuse torchaudio_squim.py, quality_control/metrics.py — no edits)
├── workflows/audio_analysis/
│   ├── grid.py                        # EDIT — (unchanged struct; presence_grid threaded in compute)
│   ├── quality.py                     # NEW — per-bucket quality vector harvester
│   ├── sound_sources.py               # NEW — AudioSet→category masses harvester
│   ├── data/audioset_source_map.json  # NEW — versioned display_name→category map
│   ├── presence.py                    # EDIT — frame-posterior voters + coarse-voter demotion
│   ├── aggregate.py                   # EDIT — presence split surfacing; utterance token-entropy sub-signal
│   ├── utterance.py                   # EDIT — overlap grid handling; token-entropy vote; quality coupling
│   ├── compute.py                     # EDIT — presence_grid param; wire quality+sources; new columns
│   ├── types.py                       # EDIT — new UncertaintyRow fields (defaulted)
│   ├── io.py                          # EDIT — new pa.array columns
│   ├── plot.py / labelstudio.py / disagreements.py  # EDIT — surface new sub-signals
│   └── calibration.py                 # NEW — load/apply fitted normalization profile
└── data_structures/
    └── (ScriptLine lives in senselab/utils/data_structures/script_line.py — EDIT: optional fields)

scripts/
├── analyze_audio.py                   # EDIT — --scene-top-k, --presence-grid, wire new outputs
└── calibrate_scene_quality.py         # NEW — synthetic noise+RIR calibration harness

src/tests/audio/
├── workflows/audio_analysis/
│   ├── quality_test.py                # NEW
│   ├── sound_sources_test.py          # NEW (incl. SC-003 full-coverage check)
│   ├── frame_posteriors_test.py       # NEW
│   ├── grid_test.py                   # NEW (per-axis grid)
│   ├── compute_uncertainty_axes_test.py  # EDIT — new columns, presence_grid
│   └── aggregate_test.py              # EDIT — presence split, token-entropy sub-signal
└── tasks/speech_to_text/ + voice_activity_detection/  # EDIT/NEW — token logits, frame posteriors
```

**Structure Decision**: Single-project library extension. New signal-processing/model units are isolated files (`frame_posteriors.py`, `scene_quality/brouhaha.py`, `quality.py`, `sound_sources.py`, `calibration.py`) each with one responsibility and a narrow interface (contracts in `contracts/`); edits to existing workflow files are additive. The one cross-cutting edit outside the workflow is `ScriptLine` + Whisper token plumbing (D7), isolated to phase 4.

## Implementation Phases (independently testable, map to spec user stories)

- **Phase 1 — Scene quality (US1, P1)**: `scene_quality/brouhaha.py` loader; `quality.py` harvester (Brouhaha SNR/C50 + DSP clip + librosa bandwidth + estimator-spread uncertainty); new `quality_*` columns; librosa explicit dep. Tests: `quality_test.py`, synthetic noised/clipped/band-limited fixtures. Delivers FR-001…FR-006, SC-001/SC-002.
- **Phase 2 — Sound sources (US2, P1)**: raise `top_k`; `data/audioset_source_map.json`; `sound_sources.py` harvester; `src_*` columns. Tests: `sound_sources_test.py` incl. full-coverage assertion. Delivers FR-007…FR-010, SC-003.
- **Phase 3 — Temporal resolution (US3, P2)**: `frame_posteriors.py` (segmentation-3.0 + Brouhaha VAD); `presence_grid` param; frame-posterior voters + coarse-voter demotion; `presence_confidence`/`presence_uncertainty` columns. Tests: `frame_posteriors_test.py`, `grid_test.py`. Delivers FR-011…FR-015, SC-004.
- **Phase 4 — Utterance rework (US4, P2)**: overlap grid handling; Whisper `output_scores`→token entropy; `ScriptLine` optional fields; calibrated confidence; scene-quality coupling column. Tests: speech_to_text token test, `aggregate_test.py` updates. Delivers FR-016…FR-019, SC-005/SC-006. **Highest risk — see Complexity Tracking; splittable to a follow-up if it destabilizes core `ScriptLine`.**
- **Phase 5 — Calibration (US5, P3)**: `scripts/calibrate_scene_quality.py`; `calibration.py` profile load/apply; validation artifact. Delivers FR-020…FR-022, SC-007.
- **Cross-cutting**: FR-023 (null-safe degradation), FR-024 (LS/plot/disagreements surface new signals), FR-025 (Brouhaha via HF-token path) threaded through all phases; SC-008 (regression) guarded continuously.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Backward-compat of `presence` axis name, parquet paths, LS tracks, and `aggregated_uncertainty` (constitution VII "no compat shims") | senselab is a shared library; b2aiprep and existing LS bundles read the `presence` parquet and track names. Renaming/altering them breaks external consumers with no in-repo signal. The maintainer explicitly chose "keep presence, additive columns." | Renaming to a `scene` axis (the clean-slate option) was considered and rejected in brainstorming precisely because of downstream breakage. Additive columns are the minimal change, not a speculative shim. |
| `ScriptLine` gains optional `avg_logprob`/`no_speech_prob`/`token_entropy` fields (touches a core data structure used across tasks) | FR-017 requires a token-level utterance signal; no backend exposes token scores today and `ScriptLine` drops extras, so the fields must be declared. This also revives the already-dead `avg_logprob` presence/utterance signal. | Carrying token scores in a side-channel dict keyed by span was considered; rejected because every existing consumer already expects data on `ScriptLine`, and a parallel structure would fork the transcript representation. Fields are optional/defaulted (null blast radius). |
