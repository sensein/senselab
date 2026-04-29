# Tasks: Alpha Prerelease Process

**Input**: Design documents from `/specs/20260418-104204-alpha-prerelease-process/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md

**Tests**: No automated tests — this is CI/CD configuration. Verification is manual via PR merge + workflow observation.

**Organization**: Tasks are grouped by user story to enable independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No setup needed — this feature modifies existing files only.

(No tasks)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Configuration changes that enable both alpha releases and preserve stable releases.

- [x] T001 [P] Update `.autorc` — set `onlyPublishWithReleaseLabel` to `false` and add `prereleaseBranches: ["alpha"]`
- [x] T002 [P] Update `.github/workflows/release.yaml` — change trigger from `push` to `pull_request: closed` on branches `[main, alpha]`, add merged + label/branch condition, upgrade Auto from v11.1.2 to v11.2.1
- [x] T002a [P] Add `if: !github.event.release.prerelease` condition to `.github/workflows/docs.yaml` to skip doc builds on alpha releases

**Checkpoint**: Configuration ready — branch creation and verification can proceed.

---

## Phase 3: User Story 2 - Alpha branch created from main (Priority: P1)

**Goal**: Create the `alpha` branch from `main` and push to remote.

**Independent Test**: Verify `alpha` branch exists on GitHub and PRs can target it.

### Implementation for User Story 2

- [x] T003 [US2] Create `alpha` branch from current `main` and push to origin (`git checkout main && git pull origin main && git checkout -b alpha && git push -u origin alpha`)
- [x] T004 [US2] Verify `alpha` branch is visible on GitHub and accepts PR targeting

**Checkpoint**: Alpha branch exists and is ready to receive PRs.

---

## Phase 4: User Story 1 - Automatic alpha releases on PR merge (Priority: P1)

**Goal**: Merge a test PR to `alpha` and verify end-to-end: tag creation, GitHub pre-release, PyPI publish.

**Independent Test**: Create a small test PR against `alpha`, merge it, verify that an alpha version tag appears, a GitHub pre-release is created with release notes, and the package is published to PyPI.

### Implementation for User Story 1

- [x] T005 [US1] Create a test PR against `alpha` that adds a comment to `pyproject.toml` (e.g., `# alpha branch test`)
- [x] T006 [US1] Merge the test PR and verify the release workflow runs
- [x] T007 [US1] Verify an alpha version tag is created (e.g., `1.4.0-alpha.0`) without a "v" prefix
- [x] T008 [US1] Verify a GitHub pre-release is published with auto-generated release notes
- [x] T009 [US1] Verify the PyPI publish workflow triggers and the package is available via `pip install senselab --pre`

**Checkpoint**: End-to-end alpha release pipeline confirmed working.

---

## Phase 5: User Story 3 - Stable releases remain on main (Priority: P2)

**Goal**: Confirm stable release process is unaffected by the changes.

**Independent Test**: Merge a PR to `main` with `release` label and verify stable release; merge a PR without label and verify no release.

### Implementation for User Story 3

- [x] T010 [US3] Merge a PR to `main` WITHOUT the `release` label and verify NO release is created
- [x] T011 [US3] Merge a PR to `main` WITH the `release` label and verify a stable release is created (no alpha suffix)

**Checkpoint**: Stable release process verified unchanged.

---

## Phase 6: User Story 4 - CI tests run on alpha branch PRs (Priority: P2)

**Goal**: Confirm CI runs on PRs targeting `alpha`.

**Independent Test**: Open a PR against `alpha` and verify macOS-tests and pre-commit run.

### Implementation for User Story 4

- [x] T012 [US4] Open a PR against `alpha` and verify that macOS-tests and pre-commit workflows trigger
- [x] T013 [US4] Verify CI results are visible on the PR checks tab

**Checkpoint**: CI gates confirmed active on alpha branch.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and cleanup.

- [x] T014 Update `quickstart.md` in specs with actual version numbers observed during verification
- [x] T015 Document the alpha release process in project README or CONTRIBUTING.md (optional, maintainer decision)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: No dependencies — can start immediately. T001 and T002 are parallel (different files).
- **US2 (Phase 3)**: Depends on Phase 2 (config must be in place before branch is meaningful)
- **US1 (Phase 4)**: Depends on US2 (alpha branch must exist to merge PRs)
- **US3 (Phase 5)**: Depends on Phase 2 only (tests stable release on main)
- **US4 (Phase 6)**: Depends on US2 (alpha branch must exist to open PRs against it)
- **Polish (Phase 7)**: Depends on all verification phases

### User Story Dependencies

- **US2 (P1)**: Depends on Foundational only — creates the branch
- **US1 (P1)**: Depends on US2 — needs alpha branch to exist for end-to-end test
- **US3 (P2)**: Independent of US1/US2 — tests main branch behavior
- **US4 (P2)**: Depends on US2 — needs alpha branch for PR targeting

### Parallel Opportunities

- T001 and T002 can run in parallel (different files)
- US3 and US4 can run in parallel after US2 completes
- T005-T009 are sequential (each depends on previous verification)

---

## Implementation Strategy

### MVP First (US2 + US1)

1. Complete T001 + T002 (config changes) — commit together
2. Complete T003 (create alpha branch)
3. Complete T005-T009 (verify alpha release pipeline)
4. **STOP and VALIDATE**: Alpha releases work end-to-end

### Full Delivery

5. Complete T010-T011 (verify stable releases unaffected)
6. Complete T012-T013 (verify CI on alpha PRs)
7. Complete T014-T015 (documentation)

---

## Notes

- Total tasks: 15
- Tasks per story: US1: 5, US2: 2, US3: 2, US4: 2, Foundational: 2, Polish: 2
- T005-T013 are verification tasks (manual observation), not code tasks
- Only T001 and T002 involve file edits; the rest are git operations and manual verification
- Commit T001 + T002 together as a single commit before creating the alpha branch
