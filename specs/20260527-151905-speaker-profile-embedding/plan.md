# Implementation Plan: Speaker Profile Embedding for analyze_audio

**Branch**: `20260527-151905-speaker-profile-embedding` | **Date**: 2026-05-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260527-151905-speaker-profile-embedding/spec.md`

## Summary

Add a **standalone speaker-profile stage** that runs *before* `analyze_audio`. For a given subject, it pools per-window speaker embeddings across all of that subject's files, clusters them, and stores the **dominant cluster's centroid** as a contamination-tolerant **speaker profile**. `analyze_audio` then optionally consumes that profile to (1) score each analyzed window's similarity to the target voice — flagging likely **other-voice** regions (foreground or background) — and (2) derive a **target-speaker recording-quality** indicator. The profile stage and `analyze_audio` share the existing content-addressable cache so expensive per-file tasks (diarization, speaker embeddings, scene classification) are computed once.

The design **reuses existing workflow code**: `embeddings.extract_per_window_embeddings`, `clustering.cluster_pass_speakers` (dominant-cluster selection with outlier rejection, spectral/k-means + silhouette + merge-close-clusters), and `clustering.calibrate_cosine_uncertainty` / `_empirical_calibration_band`. The new work is (a) **cross-file** pooling/aggregation into a persisted profile artifact, (b) leave-one-file-out scoring, and (c) wiring the profile into `analyze_audio`'s identity axis as an additional reference signal.

## Technical Context

**Language/Version**: Python ≥3.11,<3.15 (per `pyproject.toml`)
**Primary Dependencies**: senselab audio stack — `Audio`, `extract_speaker_embeddings_from_audios` (SpeechBrain: ECAPA / ResNet-TDNN, already in use), `diarize_audios` (pyannote / Sortformer), scene classification (AST / YAMNet), openSMILE features; **NEW**: HuggingFace `transformers` WavLM SV backend (`WavLMForXVector`, default `microsoft/wavlm-base-plus-sv`) for the third consensus model (FR-019); scientific stack: `numpy`, `scikit-learn` (SpectralClustering/KMeans, silhouette), `torch`, `pyarrow` (parquet)
**Storage**: Filesystem only — JSON profile artifact + the existing content-addressable cache at `artifacts/analyze_audio_cache/`; per-recording outputs alongside existing `analyze_audio` JSON/parquet
**Testing**: `pytest` (with `pytest-xdist`, `pytest-mock`, `pytest-cov`); `ruff` lint; `mypy` type-check (matches repo gates)
**Target Platform**: Linux (CPU/CUDA); clinical-research batch use on HPC (the senselab usage context)
**Project Type**: Single Python library + CLI scripts (senselab `src/` package + `scripts/` entry points)
**Performance Goals**: No re-computation of cached per-file tasks across the two stages (cache-hit reuse); profile build for a typical subject (3–5 solid files) completes within the time of one `analyze_audio` per-file pass over those files
**Constraints**: Embedding windows ≥~1s contiguous speech (ECAPA stat-pooling); confident profile needs ~20–30s aggregate speech-present audio (floor ~20s); profile comparison gated on speech presence (`p_voice`); sub-1s intrusion localization may be coarse; clinical assumption — subject is present and is the dominant voice
**Scale/Scope**: Per-subject profiles over datasets with many subjects; each subject typically O(5–20) files, mostly short (some sub-1s), a few long (free-speech / reading)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution at `.specify/memory/constitution.md` is an **unpopulated template** (placeholder principles only) — there are no ratified gates to evaluate against. No constitutional violations are possible, so the gate passes vacuously. In its place, this plan adheres to the **established senselab repo conventions** observed in the codebase:

- **Library-first**: core logic lives under `src/senselab/...` as importable, independently testable functions; CLI scripts are thin wrappers (mirrors `analyze_audio.py` + `workflows/audio_analysis/`).
- **Reuse over reinvention**: build on existing `embeddings.py` / `clustering.py` rather than new clustering code.
- **Graceful degradation**: per-model / per-file failures are captured as structured reasons, not aborts (matches existing `failures` dict pattern).
- **Auditability**: provenance (params, models, usage record) stamped into artifacts (matches parquet provenance / cache-key stamping).
- **Quality gates**: `ruff` + `mypy` + `pytest` must pass.

**Result**: PASS (no gates defined; repo conventions adopted). Re-checked post-design — still PASS (no new cross-cutting concerns introduced).

## Project Structure

### Documentation (this feature)

```text
specs/20260527-151905-speaker-profile-embedding/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── speaker-profile.schema.md       # Persisted profile artifact (JSON)
│   ├── build-profile-cli.md            # build_speaker_profile CLI command schema
│   └── analyze-audio-profile.md        # New analyze_audio inputs + profile outputs
└── checklists/
    └── requirements.md  # Created by /speckit.specify
```

### Source Code (repository root)

```text
src/senselab/audio/tasks/
└── speaker_embeddings/             # EXTEND
    ├── api.py                      #   dispatch WavLM handle (FR-019, T005)
    └── wavlm.py                    # NEW WavLMForXVector backend (T004)

src/senselab/audio/workflows/
├── audio_analysis/                 # EXISTING — reused + lightly EXTENDED
│   ├── embeddings.py               #   reuse extract_per_window_embeddings, WindowEmbedding
│   ├── clustering.py               #   reuse cluster_pass_speakers, calibrate_cosine_uncertainty, _empirical_calibration_band
│   ├── identity.py                 #   EXTEND: profile votes as identity-axis model_votes (T023)
│   ├── presence.py                 #   EXTEND: promote _speech_window_mask helper (T009a)
│   └── global_summary.py           #   EXTEND: profile sub-signals on single_speaker + quality claims (T024a, T027)
└── speaker_profile/                # NEW package
    ├── __init__.py                 # (T001)
    ├── types.py                    # SpeakerProfile, ProfileSourceFile, ClusterStats, ProfileParams,
    │                               #   ProfileComparisonResult, RecordingQualityIndicator, RecordingOtherVoiceSummary (T002)
    ├── constants.py                # documented thresholds (T003)
    ├── cache.py                    # shared cache-key helper / module-hash keying (T007)
    ├── build.py                    # speech gate, cross-file pooling → dominant-cluster centroid (T009–T016)
    ├── compare.py                  # leave-one-file-out scoring, other-voice flags, quality (T019–T021, T026)
    └── io.py                       # load/save profile JSON (T010)

scripts/
├── analyze_audio.py                # EXTEND: --speaker-profile input + profile-derived outputs (T022, T024, T024a, T027)
├── build_speaker_profile.py        # NEW thin CLI (T017)
└── gen_synthetic_test_audio.py     # NEW one-time SpeechT5 fixture generator (T010a)

src/tests/audio/tasks/
└── speaker_embeddings_test.py      # EXTEND: WavLM backend tests (T006)

src/tests/audio/workflows/speaker_profile/   # NEW tests
├── build_test.py                   # (T011)
├── compare_test.py                 # (T018, T025)
├── cache_test.py                   # (T008)
├── io_test.py                      # (T010c)
├── fixtures.py                     # deterministic composers (T010b)
└── regression_test.py              # SC-006 byte-identical (T029)

src/tests/data_for_testing/synthetic/        # NEW committed FLAC clips + manifest.json (T010a)
```

**Structure Decision**: Single-project senselab layout. Core logic is a new importable package `src/senselab/audio/workflows/speaker_profile/` that depends on the existing `audio_analysis` modules; a thin `scripts/build_speaker_profile.py` mirrors the `analyze_audio.py` pattern. `analyze_audio.py`, `identity.py`, `presence.py`, and `global_summary.py` are extended additively (behind an optional profile input); profile outputs ride the existing identity axis and `single_speaker`/`quality` claims, so existing behavior is unchanged when no profile is supplied (FR-011).

## Complexity Tracking

> No constitution gates are defined, and the design introduces no flagged complexity violations. The main risk (cross-stage cache reuse vs. the script-source wrapper hash) is addressed in `research.md` rather than by adding architectural complexity. Table intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
