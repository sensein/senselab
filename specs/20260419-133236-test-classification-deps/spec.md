# Feature Specification: Test Classification, Dependency Updates, and Modular Architecture

**Feature Branch**: `20260419-133236-test-classification-deps`
**Created**: 2026-04-19
**Status**: Draft
**Input**: User description: "Classify tests into CPU-only vs GPU-required, update dependency bumps from open PRs, re-enable macOS tests for CPU-only subset, and target alpha branch. Also upgrade dependency packages (esp speechbrain and pyannote audio to latest releases), create a feature matrix of functions and identify which require being on specific python and torch versions, and update the installer and the code to limit functionality for specific dependencies. Consider moving certain tasks into external separate uv venvs that can install legacy or direct github committish dependencies, and keep the core modern and lightweight."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - CPU-only tests run on labeled PRs via GitHub Actions (Priority: P1)

A maintainer labels a PR with a macOS test label. The CI runs CPU-only tests on a GitHub Actions macOS runner without needing GPU infrastructure. Tests that require GPU are excluded from this run. This gives fast feedback (under 10 minutes) without EC2 costs, and only runs when requested — avoiding unnecessary macOS runner consumption on trivial PRs.

**Why this priority**: macOS runners cost credits. Making tests label-triggered (like EC2 GPU tests) gives maintainers control over when to spend those credits, while still providing fast CPU test coverage when needed.

**Independent Test**: Label a PR with the macOS test label, verify CPU-safe tests run, GPU tests are skipped, and all pass.

**Acceptance Scenarios**:

1. **Given** a PR is labeled with the macOS test label, **When** CI runs on GitHub Actions, **Then** only tests classified as CPU-safe execute, and tests requiring GPU are skipped.
2. **Given** a PR without the macOS test label, **When** CI checks are viewed, **Then** the macOS test job does not run (skipped).
3. **Given** the CPU-only test suite, **When** it completes, **Then** it finishes in under 10 minutes on a macOS runner.

---

### User Story 2 - GPU tests run on EC2 when labeled (Priority: P1)

A maintainer labels a PR with `ec2-gpu-test`. The EC2 runner provisions a GPU instance and runs the full test suite — including tests that require CUDA. Tests that were skipped on GitHub Actions now execute with GPU access.

**Why this priority**: GPU tests validate ML model loading, inference, and device handling — critical for a library that processes audio/video with deep learning. These must run before merging significant changes.

**Independent Test**: Label a PR with `ec2-gpu-test`, verify all tests (including GPU-requiring ones) execute and GPU is detected.

**Acceptance Scenarios**:

1. **Given** a PR labeled `ec2-gpu-test`, **When** EC2 tests run, **Then** the full test suite executes including tests that were skipped on GitHub Actions.
2. **Given** the EC2 test run completes, **When** checking results, **Then** the passed count is higher than the GitHub Actions run (because GPU tests now execute instead of being skipped).

---

### User Story 3 - Full dependency upgrade with conflict resolution (Priority: P2)

ALL dependencies are upgraded to their latest releases. A dependency resolution process determines which packages can coexist in the core environment and which must be isolated into subprocess venvs. This is not limited to specific packages — every pinned version in pyproject.toml is evaluated for upgrade. Open dependabot PRs are also merged.

**Why this priority**: Stale dependencies accumulate security vulnerabilities, compatibility issues, and block new features. A systematic resolution approach prevents ad-hoc version pinning and ensures the core stays modern.

**Independent Test**: Run `uv lock --upgrade` on the core extras, identify conflicts, isolate conflicting packages, verify all tests pass.

**Acceptance Scenarios**:

1. **Given** open dependabot PRs for GitHub Actions (checkout v5→v6, upload-artifact v5→v7, etc.), **When** they are retargeted to `alpha` and merged, **Then** CI workflows still function correctly.
2. **Given** all dependencies are upgraded to latest, **When** conflict resolution identifies incompatible packages, **Then** those packages are moved to subprocess venvs and their senselab wrappers updated.
3. **Given** the upgraded dependency set on `alpha`, **When** the full test suite runs on EC2, **Then** all tests pass (including those using isolated backends).
4. **Given** dependency updates on `alpha`, **When** all tests pass, **Then** `alpha` is merged to `main` to propagate the updates.

---

### User Story 4 - Feature/dependency compatibility matrix (Priority: P2)

A maintainer or contributor can consult a matrix that maps each senselab function to its required dependencies, supported Python versions, and supported torch versions. This matrix identifies which functions work on which configurations and is used to gate functionality at runtime.

**Why this priority**: Without a compatibility matrix, users encounter cryptic import errors when optional dependencies are missing. The matrix enables graceful degradation and clear error messages.

**Independent Test**: Generate the matrix, verify it covers all public API functions, and confirm that calling a function with a missing dependency produces a clear error pointing to the required package.

**Acceptance Scenarios**:

1. **Given** the compatibility matrix exists, **When** a user calls a function whose dependency is not installed, **Then** they receive a clear error message naming the missing package and install command.
2. **Given** a function that requires a specific Python version (e.g., >=3.11), **When** running on an unsupported version, **Then** the function raises an error at call time (not at import time).

---

### User Story 5 - Legacy/conflicting backends isolated in subprocess venvs (Priority: P3)

Heavy or legacy backends (e.g., coqui-tts, espnet/ppgs) that conflict with modern Python/torch versions are moved to runtime subprocess venvs. The core senselab installation stays lightweight and modern. When a user calls a function backed by an isolated dependency, senselab automatically creates and manages a separate venv behind the scenes.

**Why this priority**: Some dependencies (coqui-tts, espnet) pin old torch versions or require Python <=3.11. Isolating them prevents version conflicts and keeps the core installable on Python 3.12+.

**Independent Test**: Install senselab core (no legacy extras). Call a function that uses an isolated backend. Verify the subprocess venv is auto-created, the function executes, and results are returned to the caller.

**Acceptance Scenarios**:

1. **Given** senselab is installed without coqui-tts, **When** a user calls `clone_voices()`, **Then** senselab auto-provisions an isolated venv with coqui-tts and executes the operation via subprocess.
2. **Given** the isolated venv already exists from a prior call, **When** the function is called again, **Then** the existing venv is reused (no reinstall).
3. **Given** senselab core runs on Python 3.12, **When** a legacy backend requires Python 3.11, **Then** the subprocess venv is created with Python 3.11 (managed by uv) and the function still works.

---

### User Story 6 - Some tests have both CPU and GPU variants (Priority: P3)

Tests that can meaningfully run on both CPU and GPU (e.g., model loading, basic inference) should execute on CPU during GitHub Actions runs and on GPU during EC2 runs. This maximizes test coverage without duplicating test code.

**Why this priority**: Some tests currently skip entirely on CPU even though the CPU code path is valid. Running them on CPU catches regressions in the CPU fallback path.

**Independent Test**: Identify tests currently marked GPU-only that can also run on CPU, modify their skip conditions, verify they pass on both runners.

**Acceptance Scenarios**:

1. **Given** a test that loads an HF model and runs inference, **When** running on GitHub Actions (CPU), **Then** it executes on CPU (not skipped).
2. **Given** the same test running on EC2, **When** GPU is available, **Then** it executes on GPU.

---

### Edge Cases

- What happens when a dependency bump breaks GPU-specific tests but not CPU tests? The CPU tests pass on the PR, and the breakage is only caught when `ec2-gpu-test` is added. This is acceptable — GPU tests are opt-in and should be run before merging significant dependency changes.
- What happens when a test is incorrectly classified? Tests that need GPU but aren't marked will fail on GitHub Actions. The fix is to add the appropriate skipif marker.
- What happens when dependabot PRs conflict with each other? Merge them one at a time to `alpha`, resolving conflicts sequentially.
- What happens when a subprocess venv fails to install a legacy backend? The function raises a clear error with the installation failure details. The caller's environment is not affected.
- What happens when a subprocess venv's Python version is unavailable? uv downloads and manages the required Python version automatically.
- What happens when speechbrain or pyannote-audio's latest release has breaking API changes? Tests catch these during the `alpha` testing phase. API adapters are added in the senselab wrapper layer before merging to `main`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The CI system MUST run CPU-safe tests on GitHub Actions macOS runners when a PR is labeled with a macOS test label (label-triggered, not on every PR).
- **FR-002**: The CI system MUST skip GPU-requiring tests when running on GitHub Actions (no GPU available).
- **FR-003**: The CI system MUST run the full test suite (including GPU tests) on EC2 when a PR is labeled `ec2-gpu-test`.
- **FR-004**: The macOS test job MUST be re-enabled as a label-triggered job (similar to EC2 GPU tests), not running on every PR.
- **FR-005**: Tests MUST use existing pytest skipif markers (`torch.cuda.is_available()`) to auto-detect GPU availability — no separate test configuration files needed.
- **FR-006**: Open GitHub Actions version bump PRs (checkout, upload-artifact, download-artifact, setup-uv, configure-aws-credentials) MUST be reviewed and merged to `alpha`.
- **FR-007**: The production dependencies bump PR MUST be reviewed, tested on EC2, and merged to `alpha`.
- **FR-008**: ALL dependencies MUST be upgraded to their latest compatible releases — not just speechbrain and pyannote-audio. The dependency resolution process MUST identify which packages can coexist in one environment and which require isolation.
- **FR-009**: A feature/dependency compatibility matrix MUST be created that maps each public API function to its required dependencies, supported Python versions, and supported torch versions.
- **FR-010**: Functions MUST degrade gracefully when optional dependencies are missing — raising clear errors that name the missing package and install command, not cryptic import tracebacks.
- **FR-011**: Any dependency that cannot coexist with the latest core packages MUST be automatically isolated into a runtime subprocess venv managed by uv. The determination of which packages conflict is driven by the dependency resolution process, not a hardcoded list.
- **FR-012**: Subprocess venvs MUST be auto-provisioned on first use and reused on subsequent calls.
- **FR-013**: Subprocess venvs MUST support different Python versions from the host (e.g., Python 3.11 for legacy backends when host runs 3.12).
- **FR-014**: Tests that can run on both CPU and GPU SHOULD be updated to not skip on CPU (remove unnecessary `cuda.is_available()` skipif where CPU fallback works).
- **FR-015**: All changes MUST target the `alpha` branch first, then be merged to `main` after verification.

### Key Entities

- **CPU-Safe Test**: A test file or test function that passes without GPU access. May use optional dependencies (torchaudio, speechbrain) but does not require CUDA.
- **GPU-Required Test**: A test that needs `torch.cuda.is_available() == True` to execute. Skipped on GitHub Actions, runs on EC2.
- **Dual-Mode Test**: A test that can run on both CPU and GPU, testing the appropriate code path based on hardware availability.
- **Compatibility Matrix**: A structured document (or code artifact) mapping each senselab function to its required dependencies, Python version constraints, and torch version constraints.
- **Subprocess Venv**: An isolated virtual environment managed by uv, created at runtime to host legacy or conflicting dependencies. Called via subprocess from the main senselab process.
- **Core Installation**: The minimal senselab install with modern dependencies (torch, transformers, etc.) that works on the latest supported Python. No legacy or conflicting packages.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: CPU-only test suite passes on GitHub Actions on every PR in under 10 minutes.
- **SC-002**: GPU test suite passes on EC2 with 491+ tests passing (matching current baseline).
- **SC-003**: macOS test job is re-enabled and runs CPU-safe tests.
- **SC-004**: All open GitHub Actions version bump PRs (5 PRs) are merged to `alpha`.
- **SC-005**: All dependencies are upgraded to latest compatible releases on `alpha`.
- **SC-006**: Zero dependency version conflicts in the core installation (packages that conflict are isolated).
- **SC-007**: Compatibility matrix covers 100% of public API functions.
- **SC-008**: All backends that cannot coexist with the core are moved to subprocess venvs with working tests.
- **SC-009**: Core senselab installs successfully on Python 3.12 without any dependency conflicts.
- **SC-010**: At least 5 tests currently marked GPU-only are identified as dual-mode and updated to run on CPU too.

## Clarifications

### Session 2026-04-19

- Q: How should tasks with conflicting or legacy dependencies be isolated? → A: Runtime subprocess venvs. Core senselab stays lightweight. Heavy/legacy backends are auto-installed into separate venvs managed by uv and called via subprocess. The user sees one API.
- Q: Which packages should be isolated? → A: Not a hardcoded list. ALL packages are upgraded to latest. Any package that cannot coexist with the core (due to Python version, torch version, or transitive conflicts) is automatically isolated. The dependency resolution process determines this, not manual selection.
- Q: Should macOS tests run on every PR? → A: No. macOS tests should be label-triggered (like EC2 GPU tests) to control runner credit consumption. Only run when a maintainer adds the macOS test label.

## Assumptions

- The existing `@pytest.mark.skipif(not torch.cuda.is_available())` markers correctly identify GPU-requiring tests — no custom markers needed.
- Tests using `TORCHAUDIO_AVAILABLE` or `SPEECHBRAIN_AVAILABLE` skipif markers are CPU-safe when those packages are installed (they don't require GPU).
- The `alpha` branch is the staging area for all changes before merging to `main`.
- Dependabot PRs can be retargeted to `alpha` without code changes.
- The production dependencies bump (PR #436: torch 2.10, transformers, datasets, etc.) may require code changes if APIs changed.
- uv can create venvs with specific Python versions and install packages into them programmatically.
- Subprocess communication between the host process and isolated venvs uses JSON serialization over stdin/stdout.
- The compatibility matrix is maintained as a code artifact (not just documentation) so it can be used at runtime for graceful error messages.
