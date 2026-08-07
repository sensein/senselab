# Feature Specification: Scalene-Based Profiling Tooling

**Feature Branch**: `20260503-235625-scalene-profiling`
**Created**: 2026-05-03
**Status**: Draft
**Input**: User description: "can we first add scalene based proper profiling to senselab without changing anything so we can use that to profile any part of the codebase not just imports."

## Clarifications

### Session 2026-05-04

- Q: What is the actual "no changes" constraint for adding the profiling tool? → A: API stability — the senselab public API (importable names, function signatures, class interfaces, runtime behavior) must not change so that existing user code and tutorials continue to work unmodified. Additive changes are permitted: new files, a new optional dependency entry in `pyproject.toml`, new tests for the profiling tool, and new documentation sections. Modifications to existing runtime code in `src/senselab/` that would alter the API or its behavior are out of scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Profile a Tutorial or Code Snippet (Priority: P1)

As a senselab developer, I want a one-command way to profile any Python script, tutorial, or code snippet using Scalene so that I can see line-level CPU, memory, and (when applicable) GPU usage without having to remember Scalene's flags or set up a configuration each time.

**Why this priority**: Without an easy entry point, developers will not run profiling regularly. The primary value of adding this tool is making profiling a low-friction routine activity, not a special expedition.

**Independent Test**: Pick any tutorial notebook or example script, run the new profiling command against it, and verify a Scalene HTML/JSON report is produced with line-level annotations covering both senselab code and third-party calls.

**Acceptance Scenarios**:

1. **Given** an existing Python script (e.g., a small senselab usage example), **When** the developer runs the profiling command pointing to that script, **Then** Scalene executes the script under profiling and writes an HTML report identifying the hottest lines and their CPU/memory cost.
2. **Given** a Jupyter notebook of senselab code, **When** the developer runs the profiling command pointing to that notebook, **Then** the notebook is executed (cells run) under Scalene and a profiling report is produced for the executed cells.
3. **Given** the resulting profile report, **When** the developer opens it, **Then** they can identify the top 10 slowest functions and the top 10 highest-memory lines in senselab code.

---

### User Story 2 - Profile Specific Functions or Code Regions (Priority: P2)

As a senselab developer, I want to profile only a targeted region of code (e.g., a single function call) without paying the profiling overhead for setup/teardown code, so that the report focuses on the part I'm actually optimizing.

**Why this priority**: When investigating a specific bottleneck (e.g., why does `enhance_audios()` use so much memory?), full-script profiling adds noise from imports and setup. Scoped profiling provides a clearer signal.

**Independent Test**: Wrap a specific function call in a profiling decorator or context manager and verify the resulting report contains only timing/memory data from that scope.

**Acceptance Scenarios**:

1. **Given** a Python script that performs setup work followed by a target function call, **When** the developer marks the target call for scoped profiling, **Then** the report attributes time and memory only to the target call, not to setup.
2. **Given** a scoped profile, **When** the developer compares it to a full-script profile, **Then** the scoped report has fewer entries and clearer attribution to the function under study.

---

### User Story 3 - Discoverable Documentation and Examples (Priority: P3)

As a new senselab contributor, I want clear instructions and worked examples for using the profiling tool so that I can profile my changes before submitting a pull request without reading external Scalene documentation.

**Why this priority**: A profiling tool that exists but isn't documented gets used by no one but its author. Lowering the barrier to entry maximizes the value of the tool.

**Independent Test**: A new contributor reads the project documentation and successfully runs at least one profiling example end-to-end without needing to consult external resources.

**Acceptance Scenarios**:

1. **Given** the project documentation, **When** a new contributor searches for "profiling" or "performance", **Then** they find a section explaining how to invoke the profiler with at least one runnable example.
2. **Given** the documentation, **When** the contributor follows the example, **Then** they produce a working profile report on their first attempt.

---

### Edge Cases

- What happens when the script being profiled crashes mid-execution? The profiler exits non-zero and surfaces the target script's stack trace; no profile report is produced for an incomplete run. (Scalene's `run` subcommand produces a complete profile or none — partial reports are not supported by the underlying tool.)
- How does the tool handle scripts that require GPU access? Scalene supports GPU profiling but only when CUDA is available; on CPU-only machines (e.g., macOS ARM64) the report should gracefully omit GPU columns.
- What happens when profiling a notebook that requires interactive input? Notebooks intended for profiling should be non-interactive; this is a developer-facing constraint to document.
- How does the tool handle profiling under a multiprocessing or subprocess workload? Scalene has child-process support but it must be enabled explicitly; this should be configurable.
- What if Scalene is not installed in the environment? The profiling command must fail with a clear, actionable error message pointing to the installation step.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a single command-line entry point that profiles any specified Python script using Scalene and produces a report.
- **FR-002**: System MUST produce a profile report in a human-readable format (HTML or web-viewable) that shows line-level CPU and memory attribution.
- **FR-003**: System MUST support profiling Jupyter notebook files in addition to plain Python scripts.
- **FR-004**: System MUST allow the developer to scope profiling to a specific function or code region rather than the entire script.
- **FR-005**: System MUST preserve the existing senselab public API. Importable names, function signatures, class interfaces, and runtime behavior under `src/senselab/` MUST remain unchanged so that current user code and tutorials continue to work without modification. Additive changes that support the profiling capability (new files, a new optional dependency entry in `pyproject.toml`, new tests for the profiling tool itself, new documentation sections) are permitted.
- **FR-006**: System MUST work on the developer's local machine (macOS ARM64 and Linux) without requiring GPU hardware. GPU profiling is enabled when CUDA is available but never required.
- **FR-007**: System MUST surface a clear, actionable error message when invoked in an environment where the profiling tool is not available.
- **FR-008**: System MUST be installable through the project's existing dependency management workflow without disrupting the default install.
- **FR-009**: System MUST include at least one worked example demonstrating how to profile a senselab tutorial or representative code snippet end-to-end.
- **FR-010**: System MUST place generated profile reports in a location separate from source code (e.g., a project-level artifacts directory) so reports do not pollute the package.

### Key Entities

- **Profiling Target**: The Python script, notebook, or scoped function being measured. Attributes: file path, optional argument list, optional scope identifier.
- **Profile Report**: The output artifact produced by a profiling run. Attributes: generated timestamp, target reference, output path, format type (HTML, JSON), summary metrics (top hotspots, peak memory).
- **Profiling Configuration**: The runtime options used (whether to include GPU profiling, child processes, sampling rate). Attributes: configuration name and the options set.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A senselab developer can profile any tutorial in the repository in under 5 minutes from cold start (locating the command, running it, opening the report).
- **SC-002**: The generated profile report identifies, at minimum, the top 10 slowest functions and the top 10 highest-memory lines for any non-trivial profiled script.
- **SC-003**: Adding the profiling capability preserves the senselab public API. Existing user code that imports from `senselab.*` continues to work unmodified, the existing CI test suite passes without changes, and a default `uv sync` (without the new profiling extra) installs the same runtime dependency set as before this feature.
- **SC-004**: The profiling tool can be invoked end-to-end (script -> report) without referencing any external documentation, using only the project's own documentation.
- **SC-005**: Scoped profiling on a single function produces a report whose top entries are within that function's call stack at least 90% of the time.

## Assumptions

- Profiling is performed on a developer workstation, not in production. CI does not run profiling automatically.
- The profiling tool is intended for development use, not as part of the user-facing senselab API. End users of senselab are not expected to invoke it.
- The default invocation profiles CPU and memory; GPU profiling is opt-in and only meaningful when CUDA is available.
- Reports are stored locally and are not automatically uploaded or shared.
- The profiling tool is added as an optional dependency group so that production installs of senselab do not pay the cost of the profiler unless explicitly requested.
- Notebook profiling assumes the notebook is non-interactive (does not require manual input during execution).
- The existing import-time profiling tool from the previous feature continues to coexist; the new tool is for general-purpose profiling, not a replacement.
