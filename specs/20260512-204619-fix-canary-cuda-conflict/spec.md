# Feature Specification: Resolve NeMo Canary torch/torchaudio CUDA mismatch on newer-CUDA hosts

**Feature Branch**: `20260512-204619-fix-canary-cuda-conflict`
**Created**: 2026-05-12
**Status**: Draft
**Input**: User description: "we are running into an issue on a system with cuda 12.9 where the nemo canary model runs into an installation conflict between torch and torchaudio compiled with different cuda versions"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - First-time install on a newer-CUDA host (Priority: P1)

A researcher on a workstation whose system CUDA is newer than the CUDA toolkits the project's pinned PyTorch wheels default to (concrete reference at the time of this spec: PyTorch's most-recent default-wheels CUDA is 12.8 (`cu128`); "newer-CUDA host" means a host with system CUDA > 12.8, e.g. CUDA 12.9; the same class of issue will recur as CUDA progresses past 12.9, 13.x, ...) follows the project's documented setup steps and runs the audio analysis script for the first time. The script triggers preparation of the isolated environment that hosts the NeMo Canary model, then transcribes audio with that model. Today this fails with a `torch` / `torchaudio` symbol/ABI mismatch because the two libraries inside the isolated environment were resolved to binaries compiled for different CUDA versions.

**Why this priority**: This is the first observable failure for any newer-CUDA-host user trying to use the project, and the failure happens at install/first-run — the worst place for a stop-the-world incompatibility. Without a fix, the Canary ASR backend is unusable for these hosts and the audio analysis script cannot run end-to-end with the documented default model list.

**Independent Test**: On a host whose system CUDA matches the newer reference (e.g. CUDA 12.9), run the project's documented install steps, then run the audio analysis entry point on a known short audio sample with the default ASR model list. Verify that the Canary backend's environment is prepared without import error and that the script produces a Canary transcript for that sample.

**Acceptance Scenarios**:

1. **Given** a host with system CUDA newer than the PyTorch ecosystem's most recent default-wheels CUDA version (e.g. CUDA 12.9 against a project pin of torch ≥ 2.8 whose default wheels are CUDA 12.x), **When** the user runs the documented install + first-run command, **Then** preparation of the Canary backend's environment completes without a torch/torchaudio ABI error and the script transcribes the audio sample with the Canary backend.

2. **Given** a host with system CUDA at or below the PyTorch ecosystem's most recent default-wheels CUDA version, **When** the same install + first-run command is executed, **Then** behavior is unchanged from today (no regression for already-working hosts).

3. **Given** a host where the Canary backend's environment was previously partially prepared and left in a broken state by the original mismatch, **When** the user re-runs the documented install + first-run command, **Then** the broken environment is rebuilt cleanly and the Canary backend works without manual cleanup by the user.

---

### User Story 2 - Diagnostic when no compatible binary set is available (Priority: P2)

A user on a host whose system CUDA is so new that no compatible `torch` + `torchaudio` binary pair exists in the public package index yet (the temporary state that always exists for a window of days–weeks after a fresh CUDA major release) runs the project. The system must fail loudly and informatively, naming the specific incompatibility, rather than producing an obscure ABI-symbol error from inside a model load.

**Why this priority**: Catching the unsupported-CUDA case explicitly turns a confusing 30-minute debugging session into a one-line actionable message. It does not unblock users on those exact hosts (only upstream wheel availability can), but it removes the support burden from project maintainers and gives the user a path forward (downgrade CUDA, switch to CPU, or wait for upstream wheels).

**Independent Test**: Simulate the unsupported-CUDA case by removing or shadowing the binary index used to install the Canary backend's environment. Run the install + first-run command. Verify that the failure is reported with a message that names the host's CUDA version, the failing package pair, and the recommended user action.

**Acceptance Scenarios**:

1. **Given** a host whose system CUDA has no matching `torch` + `torchaudio` binary pair available, **When** the install + first-run command is executed, **Then** the failure message names the host's CUDA version, the unavailable binary pair, and at least one concrete user-actionable path (CPU fallback, downgrade CUDA, wait for upstream wheels).

2. **Given** the same unsupported-CUDA host, **When** the user re-runs the command after taking one of the recommended actions, **Then** the previous diagnostic does not block the new run.

---

### User Story 3 - Same fix shape applies to other isolated-environment backends (Priority: P3)

Other backends in the project that ship their own isolated environment with their own pinned `torch` family (e.g. the Qwen ASR backend, and any future backend that follows the same isolation pattern) inherit the same risk on newer-CUDA hosts because the failure mode is purely about how the isolation step resolves `torch` and `torchaudio` together. Whatever mechanism resolves the Canary case should be reusable for those backends without ad-hoc duplication.

**Why this priority**: Avoids the next user-visible recurrence of the same class of bug from a sibling backend a few weeks after the Canary fix lands. Lower priority because the user-visible damage today is only Canary.

**Independent Test**: For each isolated-environment backend covered by the fix's scope, prepare its environment on a newer-CUDA host and confirm the same compatibility behavior as in User Story 1. No call-site change should be required to make the other backends benefit from the fix.

**Acceptance Scenarios**:

1. **Given** a newer-CUDA host and a project release that includes the Canary fix, **When** the user runs any isolated-environment backend covered by the fix's scope, **Then** the environment is prepared with a compatible `torch` + `torchaudio` pair without per-backend manual workarounds.

---

### Edge Cases

- **Host has no GPU at all** (CPU-only): the isolated environment must still build a working `torch` + `torchaudio` pair and the Canary backend must run on CPU. This is the safest fallback and must not silently fall back without telling the user.
- **Host has an NVIDIA GPU but no system CUDA toolkit** (driver-only install): the runtime CUDA version is what matters; the spec applies to that.
- **Host is non-Linux** (macOS arm64, Windows): macOS has no CUDA in any case; the Canary backend's environment must build on Mac without CUDA-related failures. Windows is out of scope per existing project constraints unless the user reports the same failure there.
- **CI environment** (CPU-only, ephemeral image): the install + first-run sequence must complete in CI on existing CPU-only runners with no regression in run time.
- **Pre-existing broken environment from prior failed install attempt**: the fix must include a path that detects and rebuilds a stale environment so users who hit the original bug can recover without manual `rm -rf` of the cache directory.
- **Concurrent first-run from two processes** (e.g. two test workers): environment preparation must not deadlock or corrupt the cache.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The project MUST successfully prepare the Canary backend's isolated environment on hosts whose system CUDA version is newer than the PyTorch ecosystem's most-recent default-wheels CUDA version, for every CUDA version that has both a public `torch` and a public `torchaudio` binary available (whether that pair is built for the host's exact CUDA version or for a back-compatible older one, as long as the two come from the same toolchain).

- **FR-002**: Inside the Canary backend's prepared environment, `torch` and `torchaudio` MUST be from the same toolchain build — same CUDA version (or both CPU) and ABI-compatible — such that importing `torchaudio` after `torch` produces no symbol/ABI errors and no version-mismatch warnings.

- **FR-003**: The project MUST NOT regress on hosts that already work today (system CUDA at or below the project's pinned default). The Canary backend's environment on those hosts must continue to use the same effective `torch` family it uses today.

- **FR-004**: On a host where no compatible `torch` + `torchaudio` binary pair is publicly available, the project MUST fail with a single, named, actionable error message that includes: the detected system CUDA version, the package(s) for which no compatible binary was found, and at least one concrete recommended user action.

- **FR-005**: When a previously-prepared Canary backend environment exists in a stale or partially-broken state (e.g. left over from a failed install attempt under the original bug), the project MUST detect that state on the next run and rebuild the environment from a clean baseline without requiring the user to manually delete cache directories.

- **FR-006**: The mechanism used to resolve the conflict MUST be reusable for any other backend in the project that follows the same isolated-environment pattern, without per-backend duplication of resolution logic. (Lower-priority backends may opt in over time; the contract is that nothing in the resolution mechanism is hard-coded to Canary.)

- **FR-007**: On hosts with no GPU / no CUDA, the Canary backend's environment MUST be prepared with a CPU-only `torch` + `torchaudio` pair and the backend MUST run on CPU without raising CUDA-related errors during import or model load.

- **FR-008**: The project's documentation MUST identify which combinations of host CUDA version are supported, which are gracefully degraded (CPU fallback), and which are unsupported (with the diagnostic from FR-004). Users on a previously-broken host MUST be able to find the recovery steps (re-run the install command) in the documentation.

### Key Entities

- **Host CUDA version**: The CUDA runtime version visible to the project at install time on the user's machine. Drives every downstream binary selection.

- **Isolated backend environment**: A project-managed environment, separate from the user's main project environment, that hosts a specific model (Canary, Qwen ASR, ...) along with its own pinned `torch` family. Created on demand; cached for reuse across runs.

- **torch / torchaudio binary pair**: The two libraries that must come from the same toolchain build to function together. The unit of compatibility for this spec.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a host with system CUDA 12.9, a user who runs the project's documented install + first-run command on a known short audio sample receives a Canary transcript for that sample in under 10 minutes, excluding model-download time (which depends on network bandwidth; baseline measured at ≥ 100 Mbps), with zero manual intervention beyond the documented commands.

- **SC-002**: On hosts that work today (system CUDA ≤ the PyTorch ecosystem's most-recent default-wheels CUDA), end-to-end run time of the project's documented install + first-run command does not increase by more than 5% compared to the project's last working release before this fix.

- **SC-003**: On a host whose system CUDA has no compatible public `torch` + `torchaudio` pair, the project's failure surface for this exact case is a single, named, actionable error message — no stack traces from deep inside `torchaudio` imports, no symbol-not-found errors, no incomplete cache directories left behind.

- **SC-004**: 100% of users who hit the original conflict in a partially-prepared environment can recover with one re-run of the documented install + first-run command (no manual cache deletion required).

- **SC-005**: At least one other isolated-environment backend in the project benefits from the same resolution mechanism in the same release (or is documented as out-of-scope-this-release with a clear future-work pointer).

## Assumptions

- The PyTorch ecosystem will continue to publish `torch` and `torchaudio` binaries that share a toolchain build for at least one CUDA version covering the majority of current Linux hosts (i.e. there is always *some* compatible pair to choose, even if not the host's exact CUDA version).

- CPU-only fallback is acceptable when no compatible CUDA pair exists. Canary on CPU is slower but functional for development / validation; production GPU workflows are explicitly acknowledged as needing a supported CUDA version.

- macOS support is unchanged — Canary on Mac runs on CPU/MPS as it does today; Mac is not the source of this bug.

- The current isolated-environment scheme (separate cache directories per backend) is acceptable to keep. The fix changes how dependencies are *resolved* inside those environments, not the existence or location of the environments themselves.

- The fix is observable at install/first-run time, not at long-term model accuracy or inference quality. Existing accuracy/perf metrics on hosts that work today are presumed unchanged.
