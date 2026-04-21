# Feature Specification: Fix All Tutorials to Run on Google Colab

**Feature Branch**: `20260420-212321-fix-colab-tutorials`
**Created**: 2026-04-20
**Status**: Draft
**Input**: User description: "review and fix all tutorials to run on colab"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run Any Tutorial on Colab Without Errors (Priority: P1)

A researcher opens any senselab tutorial notebook on Google Colab and runs all cells from top to bottom. The notebook completes successfully without any import errors, missing dependency errors, or runtime failures. The notebook installs senselab and any required system dependencies (FFmpeg, uv) automatically.

**Why this priority**: Tutorials are the primary onboarding path for new users. If they fail on Colab, users abandon the project.

**Independent Test**: Open each notebook in Colab (or a Colab-equivalent CI environment with Python 3.12), run all cells, verify zero errors and expected outputs appear.

**Acceptance Scenarios**:

1. **Given** a fresh Colab runtime, **When** a user opens any tutorial notebook and runs all cells, **Then** all cells complete without errors and produce expected outputs
2. **Given** a fresh Colab runtime, **When** the first cell installs senselab, **Then** all dependencies including FFmpeg for torchcodec and uv for subprocess venvs are available
3. **Given** a tutorial that requires GPU, **When** run on a CPU-only Colab runtime, **Then** the notebook gracefully skips GPU-only operations or provides a clear message

---

### User Story 2 - Automated Tutorial Testing in CI (Priority: P2)

A developer pushes changes to senselab and CI automatically validates that all tutorial notebooks still execute correctly. CPU tutorials run on GitHub Actions, GPU tutorials run on EC2. Failures report the specific notebook and cell that failed.

**Why this priority**: Prevents tutorial regression. Tutorials break silently when library APIs change; automated testing catches this before users do.

**Independent Test**: CI runs all notebooks via an execution tool, reports pass/fail per notebook with cell-level error details.

**Acceptance Scenarios**:

1. **Given** a PR is opened, **When** CI runs, **Then** all CPU tutorial notebooks are executed and pass
2. **Given** a PR has the GPU test label, **When** EC2 CI runs, **Then** all GPU tutorial notebooks are executed and pass
3. **Given** a tutorial cell fails, **When** CI reports results, **Then** the specific notebook path and failing cell number are shown

---

### User Story 3 - Standardized Tutorial Structure (Priority: P3)

Every tutorial notebook has a consistent structure: Colab badge, standardized setup cell, clear section headers, and runtime/hardware requirements stated upfront. Users know what to expect before running.

**Why this priority**: Reduces friction and support burden. A consistent structure across all tutorials makes the experience predictable.

**Independent Test**: Validate each tutorial has the required structure elements (badge, setup cell, requirements note).

**Acceptance Scenarios**:

1. **Given** any tutorial notebook, **When** opened on GitHub, **Then** an "Open in Colab" badge is visible at the top
2. **Given** any tutorial notebook, **When** the first code cell is run on Colab, **Then** senselab and all dependencies are installed within 3 minutes
3. **Given** a GPU tutorial, **When** opened, **Then** a note at the top states "Requires GPU runtime"

---

### Edge Cases

- Tutorials using models that require HuggingFace authentication (HF_TOKEN) must include token setup instructions
- GPU-only tutorials should detect and warn when running on CPU-only runtimes
- Tutorials using subprocess venvs (voice cloning, PPGs, SPARC) need uv installed first
- Large model downloads may exceed Colab disk or RAM limits for free-tier users
- Colab's pre-installed package versions may conflict with senselab's requirements

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every tutorial notebook MUST execute without errors on a fresh Google Colab runtime (Python 3.12)
- **FR-002**: Every tutorial notebook MUST have a standardized first cell that installs senselab and system dependencies
- **FR-003**: The setup cell MUST install FFmpeg shared libraries for torchcodec
- **FR-004**: The setup cell MUST ensure uv is available for subprocess venv operations
- **FR-005**: Tutorials requiring GPU MUST be clearly marked and MUST gracefully handle CPU-only runtimes
- **FR-006**: Tutorials requiring HuggingFace authentication MUST include HF_TOKEN setup instructions
- **FR-007**: CI MUST execute all tutorial notebooks on every PR (CPU on GitHub Actions, GPU on EC2)
- **FR-008**: CI MUST report the specific notebook and cell that failed
- **FR-009**: Notebook outputs MUST be cleared before committing
- **FR-010**: Each tutorial MUST include an "Open in Colab" badge at the top

### Key Entities

- **Tutorial Notebook**: A Jupyter notebook in `tutorials/` demonstrating a senselab feature
- **Setup Cell**: Standardized first code cell that installs dependencies
- **Tutorial Manifest**: Configuration mapping each notebook to its requirements (CPU/GPU, extras, HF_TOKEN)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of tutorial notebooks (17 notebooks) execute without errors on a fresh Colab runtime
- **SC-002**: Tutorial setup completes in under 3 minutes on Colab
- **SC-003**: CI catches tutorial breakage within the same PR that causes it
- **SC-004**: Every tutorial has a working "Open in Colab" link
- **SC-005**: A new user can go from zero to running their first tutorial in under 5 minutes

## Assumptions

- Google Colab uses Python 3.12 with torch pre-installed
- Colab has pip but not uv; setup cell bootstraps uv via `pip install uv`, then uses uv for all package management
- Colab GPU runtimes have CUDA available
- Colab has ~12GB RAM and ~100GB disk on free tier
- Tutorials install senselab via `pip install --pre senselab` (latest pre-release from PyPI)
- HuggingFace models in tutorials are either public or include token instructions
- CI uses papermill or nbclient for notebook execution
- The spec branch targets alpha

## Clarifications

### Session 2026-04-20

- Q: Which version of senselab should tutorials install? → A: Latest pre-release (`pip install --pre senselab`)
