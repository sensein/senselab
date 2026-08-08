# Implementation Plan: Test Classification, Dependency Updates, and Modular Architecture

**Branch**: `20260419-133236-test-classification-deps` | **Date**: 2026-04-19 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260419-133236-test-classification-deps/spec.md`

## Summary

Systematically upgrade all senselab dependencies, classify tests into CPU/GPU tiers, re-enable macOS CI for CPU tests, create a feature/dependency compatibility matrix, and isolate conflicting backends into runtime subprocess venvs. All changes target `alpha` first.

## Technical Context

**Language/Version**: Python 3.11-3.12, Bash
**Primary Dependencies**: torch, transformers, speechbrain, pyannote-audio, coqui-tts, ppgs/espnet, sentence-transformers
**Testing**: pytest (CPU on GitHub Actions, GPU on EC2)
**Target Platform**: GitHub Actions (macOS/ubuntu) + AWS EC2 (GPU)
**Project Type**: Python library with ML backends
**Constraints**: Core must install cleanly on Python 3.12; legacy backends isolated via subprocess venvs

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | All Python via uv; subprocess venvs created by uv |
| II. Encapsulated Testing | PASS | CPU tests on GHA, GPU on EC2, isolated venvs |
| III. Commit Early and Often | PASS | Each phase is independently committable |
| IV. CI Must Stay Green | PASS | macOS re-enabled; EC2 tested separately |
| V. Memory-Driven Anti-Patterns | PASS | local_files_only lesson applied |
| VI. No Unnecessary API Calls | PASS | HF caching layer preserved |
| VII. Simplicity First | PASS | Subprocess venvs only where needed |
| VIII. No Hardcoded Parameters | PASS | Venv specs driven by compatibility matrix, not hardcoded |

**PASS** — No violations.

## Project Structure

### Source Code Changes

```text
src/senselab/
├── utils/
│   ├── subprocess_venv.py    # NEW: uv venv manager for isolated backends
│   └── compatibility.py      # NEW: feature/dep matrix + runtime checks
├── audio/tasks/
│   ├── voice_cloning/coqui.py     # MODIFIED: use subprocess venv
│   └── features_extraction/       # MODIFIED: ppgs via subprocess venv
.github/workflows/
└── tests.yaml                     # MODIFIED: re-enable macOS, test tiers
pyproject.toml                     # MODIFIED: dependency upgrades + restructure
docs/
└── compatibility-matrix.md        # NEW: human-readable matrix
```

## Phase 0: Research

### Current Dependency Landscape

**Known conflicts in current install:**
1. `cv2` (opencv) and `av` (PyAV) both bundle `libavdevice` → duplicate symbol warnings, potential crashes
2. `torchcodec` fails on macOS (FFmpeg RPATH issues) and sentence-transformers was pinned `<5.4` to avoid it
3. `coqui-tts~=0.27` not installed (likely conflicts with current torch)
4. `ppgs` depends on `espnet` which pulls massive deps and pins old versions
5. `snorkel>=0.10.0,<0.11.0` pins old versions
6. `lightning~=2.4.0` pins old version (ppgs dependency)
7. `sentence-transformers>=5.1,<5.4` pinned due to torchcodec import issue

**Dependency tiers (from resolution analysis):**

| Tier | Packages | Strategy |
|------|----------|----------|
| Core | torch, torchaudio, torchvision, transformers, datasets, pydantic, huggingface-hub, accelerate | Upgrade to latest, must coexist |
| Core Audio | speechbrain, pyannote-audio, praat-parselmouth, audiomentations, opensmile | Upgrade to latest, keep in core |
| Core Text | sentence-transformers, pylangacq, nltk, jiwer | Upgrade (unpin sentence-transformers after torchcodec fix) |
| Core Video | opencv-python-headless, ultralytics, av | Upgrade, resolve cv2/av conflict |
| Isolated | coqui-tts, ppgs+espnet+snorkel+lightning | Move to subprocess venvs |
| Utility | scikit-learn, umap-learn, matplotlib, joblib, pycountry | Upgrade freely |

### Decision 1: Subprocess venv architecture

**Decision**: `src/senselab/utils/subprocess_venv.py` — a utility that manages isolated uv venvs at runtime.
**Rationale**: Keeps the core clean. Uses JSON over stdin/stdout for IPC. Venvs cached in `~/.cache/senselab/venvs/` (or `$SENSELAB_VENV_CACHE`).
**Key operations**: `ensure_venv(name, requirements, python_version)` → creates/reuses venv. `run_in_venv(name, function, args)` → executes a Python function in the isolated venv via subprocess.

### Decision 2: Compatibility matrix format

**Decision**: Python dict in `src/senselab/utils/compatibility.py` + auto-generated markdown in `docs/`.
**Rationale**: Code artifact enables runtime checking (graceful errors). Markdown enables human reading. Both generated from one source of truth.

### Decision 3: Test classification

**Decision**: No new pytest markers needed. GPU tests already use `@pytest.mark.skipif(not torch.cuda.is_available())`. CPU tests run naturally on GitHub Actions. macOS job re-enabled with current test suite — GPU tests auto-skip.
**Rationale**: Simplest approach. No test infrastructure changes needed.

### Decision 4: Dependency upgrade approach

**Decision**: Phased upgrade on `alpha` branch.
1. First: upgrade GitHub Actions versions (low risk)
2. Second: upgrade core packages (torch, transformers, etc.)
3. Third: upgrade audio/text/video packages (speechbrain, pyannote, etc.)
4. Fourth: move conflicting packages to subprocess venvs
5. Each phase: run EC2 tests, fix breakage, commit.
**Rationale**: Incremental approach catches breakage early. Each phase is independently revertable.

### Decision 5: Handling the cv2/av conflict

**Decision**: Move to `opencv-python-headless` only (no GUI deps) and keep `av` (PyAV) for video I/O. Both are needed but their bundled FFmpeg libs clash. Resolution: use system FFmpeg via `imageio-ffmpeg` binary (already proven in EC2 CI) and disable bundled FFmpeg in one of them.
**Rationale**: The duplicate `libavdevice` symbols cause runtime crashes. Preferring `av` for video decoding and using headless opencv for image processing avoids the conflict.

## Phase 1: Implementation Steps

### Step 1: Re-enable macOS tests

- Remove `if: false` from `macos-tests` job in `tests.yaml`
- macOS runs full test suite — GPU tests auto-skip via existing markers
- No pytest config changes needed

### Step 2: Merge GitHub Actions version bumps

- Retarget PRs #421, #432, #433, #434, #435 to `alpha`
- Merge sequentially, verify CI after each

### Step 3: Create subprocess venv utility

- `src/senselab/utils/subprocess_venv.py`:
  - `ensure_venv(name, requirements, python_version)` — creates uv venv if not exists
  - `run_in_venv(name, module, function, args, kwargs)` — runs function via subprocess
  - JSON serialization for args/results over stdin/stdout
  - File paths for large data (audio waveforms)
  - Venv cache at `~/.cache/senselab/venvs/{name}/`
  - Lock file for concurrent access

### Step 4: Create compatibility matrix

- `src/senselab/utils/compatibility.py`:
  - Dict mapping function → {deps, python_versions, torch_versions, isolated}
  - `check_compatibility(function_name)` — raises clear error if deps missing
  - Auto-decorate public API functions with compatibility checks
- `docs/compatibility-matrix.md` — auto-generated from the dict

### Step 5: Move conflicting backends to subprocess venvs

- **coqui-tts**: `voice_cloning/coqui.py` → call via subprocess venv
- **ppgs/espnet**: `features_extraction/` → call via subprocess venv
- **snorkel**: used by ppgs → goes with ppgs venv
- **lightning**: used by ppgs → goes with ppgs venv
- Remove these from `pyproject.toml` core dependencies
- Add venv specs to compatibility matrix

### Step 6: Upgrade all core dependencies

- Remove version pins, run `uv lock --upgrade`
- Fix any API breakage in senselab wrapper code
- Unpin `sentence-transformers` (torchcodec issue mitigated)
- Upgrade speechbrain, pyannote-audio to latest
- Resolve cv2/av conflict
- `python-dotenv` already in runtime deps (from earlier PR)

### Step 7: Test on EC2 + merge

- Run full test suite on EC2 with upgraded deps
- Fix remaining failures
- Merge `alpha` → `main`

## Risks

- **Upstream API breakage**: speechbrain/pyannote major version upgrades may change APIs. Mitigated by incremental testing on `alpha`.
- **Subprocess venv overhead**: First call to an isolated backend is slow (venv creation + install). Mitigated by persistent caching.
- **Serialization limits**: JSON over stdin/stdout can't handle large tensors. For audio data, pass file paths or use temp files. Functions in isolated venvs must accept/return serializable data.
- **cv2/av conflict**: May require subprocess isolation for video tasks too if headless opencv still clashes.
- **Python version matrix**: Some isolated backends may need Python 3.11 while core runs 3.12. uv handles multi-version management.
