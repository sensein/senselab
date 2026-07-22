# Implementation Plan: HuggingFace Model Cache & Version Consistency

**Branch**: `20260706-213755-hf-model-cache` | **Date**: 2026-07-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260706-213755-hf-model-cache/spec.md`

## Summary

Provide a single shared mechanism that every senselab model backend uses to obtain, cache, version-verify, and load HuggingFace models, eliminating the per-load Hub version checks that cause 429 rate-limit failures under heavy parallelism. The mechanism resolves a requested reference (branch/tag/`"main"`/sha) to an immutable commit SHA **at most once per freshness window** (default 7 days, coordinated across processes via a file lock), records the resolved SHA + last-verified timestamp in a shared sidecar, and thereafter loads models **pinned by SHA with `local_files_only=True`** — never re-hitting the Hub. Two freeze scopes (run-scoped parameter, system-level env var) layer on top with precedence system ⊇ run ⊇ default. All 31 existing loading sites are migrated onto it incrementally (strangler-fig) behind unchanged public APIs, each with a behavior-preserving test.

**Key technical driver (from research):** in the installed `huggingface_hub` 1.11 / `transformers` 4.55, offline mode is captured at *import time*, so mid-process `os.environ["HF_HUB_OFFLINE"]="1"` toggling (today's `hf_offline_loading`) does **not** reliably suppress the network revision-check. The mechanism therefore relies on **SHA-pinned `local_files_only=True` loads** (in-process) and the already-correct **fresh-import subprocess env** (subprocess workers), not env toggling.

## Technical Context

**Language/Version**: Python ≥3.11,<3.15 (per `pyproject.toml`); main venv currently CPython 3.12 (CI) / 3.14 (dev)
**Primary Dependencies**: `huggingface_hub` 1.11 (`HfApi.model_info`, `snapshot_download`, `try_to_load_from_cache`, `scan_cache_dir`), `filelock` (already used), `transformers` 4.55, `speechbrain` 1.0.3, `pyannote.audio` 3.4, NeMo (subprocess venv), `sentence-transformers`. **No new third-party dependency.**
**Storage**: Filesystem only — a shared per-`(repo_id, ref)` **cache-record sidecar** (JSON) under `{HF_HOME}/senselab_cache/`, alongside the HF Hub cache it describes. Atomic write-then-`os.replace`; explicit `schema_version` (mirrors `speaker_profile/io.py`, ranking store).
**Testing**: `cd src && uv run pytest`; `uv run ruff check .`; `uv run mypy` (pre-commit). Unit tests use monkeypatched `HfApi.model_info` / temp cache dirs (no network); per-backend behavior-preserving tests; GPU end-to-end verification on SLURM.
**Target Platform**: Linux shared-compute (SLURM GPU nodes) with a shared HF cache location across jobs/nodes.
**Project Type**: Single project (library) — `src/senselab/…`.
**Performance Goals**: Zero avoidable Hub calls for a cached model within its window (SC-002); at most one coordinated re-check per window across ≥100 concurrent jobs (SC-001, SC-005); download-once (SC-003).
**Constraints**: No per-load Hub traffic; graceful degradation when Hub unreachable but cached (FR-007); loud failure on version mismatch/unavailable (FR-003/008); byte-behavior preserved per backend (public APIs unchanged).
**Scale/Scope**: 31 model-loading sites (11 migrated, 3 partial, 17 unmigrated) across speech-to-text, forced alignment, speaker embeddings/diarization, VAD, enhancement, classification/SER, TTS, SSL, text embeddings; plus 6 non-HF-Hub loaders (TF-Hub/s3prl/SPARC/Coqui) that are out of the HF-verification scope but share the download-once pattern.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Test-Driven Development (NON-NEGOTIABLE)** — ✅ PASS (planned). The shared mechanism's behaviors (freshness-window skip, coordinated single re-check, version-mismatch failure, run/system freeze, Hub-unreachable fallback) each get a failing unit test first. Every backend migration is a refactor guarded by a behavior-preserving test written/confirmed before the migration edit.
- **II. Reproducible Environments via uv** — ✅ PASS. All tests/lint/type-checks/scripts run via `uv run`. No new environment manager; the freeze mechanism itself improves reproducibility.
- **III. Documented Thresholds & Defaults** — ✅ PASS. The freshness window (`DEFAULT_FRESHNESS_WINDOW_DAYS = 7`, env `SENSELAB_HF_FRESHNESS_DAYS`), retry limit (`SENSELAB_HF_MAX_RETRIES = 3`, existing), heartbeat/stale thresholds (existing), and freeze env (`SENSELAB_HF_FREEZE`) are all named, documented constants surfaced in the PR description (FR-012).
- **Tech & Tooling Constraints** — ✅ PASS. No new dependency (reuses `huggingface_hub`/`filelock`). Long-running model/GPU verification runs on SLURM, not login nodes.

**Result (initial): no violations → Complexity Tracking is empty.**
**Result (post-Phase 1 re-check): still no violations — the design adds one module + one sidecar record, introduces no new dependency, and keeps every public API unchanged.**

## Project Structure

### Documentation (this feature)

```text
specs/20260706-213755-hf-model-cache/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── model-cache-api.md
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/senselab/utils/
├── dependencies.py          # Existing HF helpers — reworked (see below); re-exports for back-compat
└── hf_model_cache.py        # NEW: cache-record + resolver + freeze (the shared mechanism)

src/senselab/utils/data_structures/
└── model.py                 # HFModel — gains resolved-SHA plumbing (already calls ensure_hf_model)

src/senselab/audio/tasks/**  # 25 audio loading sites migrated to the shared mechanism
src/senselab/text/tasks/**   #  2 text-embedding loading sites migrated

src/tests/utils/
├── hf_model_cache_test.py   # NEW: window/freeze/verification/coordination unit tests
└── dependencies_test.py     # Extended
src/tests/**                 # Per-backend behavior-preserving tests (existing files extended)
```

**Structure Decision**: Single-project library. The new logic lives in a cohesive `src/senselab/utils/hf_model_cache.py` (cache-record dataclass, `resolve_model`, freeze context, freshness/coordination). `dependencies.py` keeps its download-once primitives (`ensure_hf_model`, `_HeartbeatLock`, `retry_on_transient_error`, SpeechBrain CWD helpers) and re-exports the new surface so no import path breaks. Backends call one entry point (`resolve_model` / reworked `load_hf_resilient` / `hf_subprocess_env`) and never contain bespoke caching/version code (SC-006, SC-010).

## Complexity Tracking

> No Constitution violations — no entries.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
