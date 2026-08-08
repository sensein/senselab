# Feature Specification: Optimize Import Times

**Feature Branch**: `20260501-154228-optimize-import-times`
**Created**: 2026-05-01
**Status**: Draft
**Input**: User description: "let's optimize import times. take each import block from the tutorials and evaluate how much time it takes, and check where bottlenecks in import exist."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Profile All Tutorial Import Bottlenecks (Priority: P1)

As a senselab developer, I want to measure the wall-clock time of every import statement used across the tutorial notebooks so that I can identify which modules and third-party dependencies are the slowest to load.

**Why this priority**: Understanding where the time goes is a prerequisite for any optimization work. Without a profiling baseline, improvements cannot be measured.

**Independent Test**: Run a timing measurement against each distinct import from the tutorials and produce a ranked report of import durations. The report itself is the deliverable.

**Acceptance Scenarios**:

1. **Given** a fresh process with no cached modules, **When** each distinct import line from the 20 tutorial notebooks is timed individually, **Then** a report is produced showing per-import wall-clock time, sorted from slowest to fastest.
2. **Given** the timing report, **When** a developer reviews it, **Then** imports taking longer than 2 seconds are clearly flagged as bottlenecks.

---

### User Story 2 - Identify Internal vs External Bottlenecks (Priority: P2)

As a senselab developer, I want to know whether slow imports originate inside senselab (e.g., heavy top-level code, eager model loading) or from third-party dependencies (e.g., torch, speechbrain, pyannote) so that I can prioritize optimization strategies.

**Why this priority**: The fix strategy differs drastically: internal bottlenecks can be addressed with lazy imports or deferred initialization; external ones may require subprocess isolation or optional dependency groups.

**Independent Test**: The timing report distinguishes senselab-internal import time from transitive third-party dependency time for each import.

**Acceptance Scenarios**:

1. **Given** a senselab import like `from senselab.audio.tasks.speech_enhancement import enhance_audios`, **When** it is profiled, **Then** the report shows both total import time and a breakdown of which child imports (torch, speechbrain, etc.) contribute most to the total.
2. **Given** the breakdown, **When** a developer reviews it, **Then** they can determine whether the bottleneck is in senselab module-level code or in a transitive dependency.

---

### User Story 3 - Per-Tutorial Import Cost Summary (Priority: P3)

As a senselab user running a tutorial notebook, I want to know how long the first code cell (imports) will take on a cold start so that I can set expectations and choose lighter tutorials when time is limited.

**Why this priority**: Users experience import time as a "waiting wall" before any tutorial code runs. Knowing the cost per tutorial helps both users and maintainers prioritize which tutorials benefit most from optimization.

**Independent Test**: Produce a summary table showing, for each tutorial notebook, the aggregate cold-start import time.

**Acceptance Scenarios**:

1. **Given** the 20 tutorial notebooks with imports, **When** each tutorial's combined import block is timed in a fresh process, **Then** a per-tutorial summary table is produced showing total import time.
2. **Given** the summary table, **When** sorted by total import time, **Then** the top 5 slowest tutorials are identified for priority optimization.

---

### Edge Cases

- What happens when an import requires authentication (e.g., `google.colab.userdata`)? These should be excluded from timing or noted as environment-specific.
- How does the system handle imports that fail on the current platform (e.g., CUDA-only, Colab-only)? Failed imports should be recorded with their error rather than crashing the profiling run.
- What about import order dependencies? Some imports are faster when another module has already loaded a shared dependency (e.g., `torch`). The profiling must account for this by isolating each import in a separate process.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST time every distinct import statement found across all tutorial notebooks individually, each in an isolated process to avoid caching effects.
- **FR-002**: System MUST produce a ranked list of imports sorted by wall-clock time, from slowest to fastest.
- **FR-003**: System MUST flag imports exceeding a configurable threshold (default: 2 seconds) as bottlenecks.
- **FR-004**: System MUST break down senselab imports to show which transitive dependencies contribute most to load time.
- **FR-005**: System MUST produce a per-tutorial summary showing the aggregate cold-start import time for each notebook.
- **FR-006**: System MUST gracefully handle imports that fail on the current platform (missing packages, auth requirements) and record the failure without aborting.
- **FR-007**: System MUST isolate each import measurement in a fresh subprocess to prevent module caching from skewing results.

### Key Entities

- **Import Statement**: A single Python import line extracted from a tutorial notebook, with attributes: module path, source notebook, timing result, success/failure status.
- **Tutorial Notebook**: A Jupyter notebook file containing one or more import blocks, with attributes: file path, list of imports, aggregate import time.
- **Bottleneck Report**: The output artifact containing ranked imports, per-tutorial summaries, and internal-vs-external breakdowns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A complete timing report covers 100% of distinct import statements found across all 20 tutorial notebooks.
- **SC-002**: The report clearly identifies the top 10 slowest imports with their wall-clock times.
- **SC-003**: For each senselab import flagged as a bottleneck, the report attributes the majority of load time to specific child dependencies.
- **SC-004**: A per-tutorial summary table is produced, enabling developers to identify the 5 slowest tutorials by import time.
- **SC-005**: The profiling process completes without manual intervention, handling all platform-specific import failures gracefully.

## Assumptions

- Profiling is performed on the developer's local machine (macOS ARM64) rather than in Colab, since the goal is to identify inherent import costs, not network/environment latency.
- The environment has all optional senselab dependencies installed (articulatory, text, video, senselab-ai extras plus dev group).
- Colab-specific imports (`google.colab`) will be skipped or flagged as unavailable rather than treated as failures.
- Each import is timed in a fresh subprocess to ensure cold-start measurements; warm-cache timings are out of scope.
- The profiling script is a development-time tool, not a user-facing feature; it does not need to be packaged or documented beyond the spec outputs.
