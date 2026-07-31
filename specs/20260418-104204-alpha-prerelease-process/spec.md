# Feature Specification: Alpha Prerelease Process

**Feature Branch**: `20260418-104204-alpha-prerelease-process`  
**Created**: 2026-04-18  
**Status**: Draft  
**Input**: User description: "write a new spec to create a prerelease process off a dedicated branch (alpha) for senselab. this should branch off main. the pre-release process should use the same setup as ~/software/neuronets/nobrainer."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automatic alpha releases on PR merge (Priority: P1)

A maintainer merges a pull request into the `alpha` branch. The system automatically creates a new pre-release version (e.g., `2.0.0-alpha.0`, `2.0.0-alpha.1`), tags the commit, generates a GitHub pre-release with release notes, and publishes the package to PyPI as a pre-release.

**Why this priority**: This is the core value of the feature — enabling automated pre-release distribution without manual version management. It allows early adopters and collaborators to test new features before they reach stable.

**Independent Test**: Can be fully tested by merging a PR into the alpha branch and verifying that a GitHub pre-release is created, a git tag is applied, and the package appears on PyPI with an alpha version suffix.

**Acceptance Scenarios**:

1. **Given** a PR is merged into the `alpha` branch, **When** the release workflow runs, **Then** a new alpha version tag is created (incrementing the alpha counter), a GitHub pre-release is published with auto-generated release notes, and the package is published to PyPI.
2. **Given** the alpha branch has no prior alpha tags, **When** the first PR is merged, **Then** the version bump type is determined by PR labels (`major`, `minor`, or `patch`; default is `patch` if no label), applied to the latest stable tag, and suffixed with `-alpha.0` (e.g., a `minor`-labeled PR after stable `1.3.0` produces `1.4.0-alpha.0`). Subsequent alpha merges increment the counter (`1.4.0-alpha.1`, etc.). A new bump label resets the counter for the new version.
3. **Given** a PR is merged into `alpha` without a release label, **When** the release workflow runs, **Then** an alpha pre-release is still created (no label required for alpha releases).

---

### User Story 2 - Alpha branch created from main (Priority: P1)

A maintainer creates the `alpha` branch from the current state of `main`. The branch serves as the integration branch for pre-release work. PRs targeting new features or breaking changes can be directed to `alpha` instead of `main`.

**Why this priority**: The alpha branch is a prerequisite for the entire prerelease workflow. Without it, no alpha releases can happen.

**Independent Test**: Can be tested by creating the `alpha` branch from `main`, verifying it exists on the remote, and confirming CI workflows recognize it.

**Acceptance Scenarios**:

1. **Given** the `main` branch exists with the latest stable release, **When** the `alpha` branch is created, **Then** it starts from the same commit as `main` and is pushed to the remote repository.
2. **Given** the `alpha` branch exists, **When** a developer opens a PR targeting `alpha`, **Then** the PR is accepted and CI tests run against it.

---

### User Story 3 - Stable releases remain on main (Priority: P2)

The existing stable release process on `main` continues to work unchanged. Stable releases are triggered by merging PRs with a `release` label into `main`. The alpha prerelease process does not interfere with the stable release flow.

**Why this priority**: Preserving the existing stable release process ensures no disruption to current users and maintainers.

**Independent Test**: Can be tested by merging a PR with the `release` label into `main` and verifying that a stable release (not alpha) is created and published.

**Acceptance Scenarios**:

1. **Given** a PR with the `release` label is merged into `main`, **When** the release workflow runs, **Then** a stable version tag and GitHub release are created (no alpha suffix).
2. **Given** the alpha branch has active alpha releases, **When** a stable release is created on `main`, **Then** the stable release is unaffected by alpha version history.

---

### User Story 4 - CI tests run on alpha branch PRs (Priority: P2)

Pull requests targeting the `alpha` branch trigger the same CI test suite as PRs targeting `main`. This ensures pre-release code meets the same quality bar as stable releases.

**Why this priority**: Quality gates on alpha ensure pre-releases are usable by early adopters and don't ship obviously broken code.

**Independent Test**: Can be tested by opening a PR against `alpha` and verifying that macOS tests and pre-commit checks run.

**Acceptance Scenarios**:

1. **Given** a PR is opened targeting the `alpha` branch, **When** code is pushed, **Then** the test workflow (macOS-tests, pre-commit) runs automatically.
2. **Given** CI tests fail on an alpha PR, **When** the maintainer reviews, **Then** they can see the failures and the PR is not auto-merged.

---

### Edge Cases

- What happens when the alpha branch diverges significantly from main? Maintainers are responsible for periodically rebasing or merging `main` into `alpha` to prevent large divergence.
- What happens if a stable release on `main` leapfrogs the alpha version? The `auto` tool's semantic versioning handles this — the next alpha tag will be based on the new stable version.
- What happens if the PyPI publish step fails for an alpha release? The GitHub pre-release is still created; the maintainer can manually re-trigger the publish workflow or fix and merge a new PR.
- What happens if someone accidentally merges to `alpha` without intending a release? Every merge to `alpha` triggers an alpha release. This is by design (matching the nobrainer pattern). Maintainers should protect the alpha branch accordingly.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST create an `alpha` branch from the current `main` branch.
- **FR-002**: The release automation MUST recognize `alpha` as a prerelease branch and automatically create alpha-suffixed version tags on PR merge.
- **FR-003**: Alpha releases MUST NOT require a `release` label on the PR (auto-release on every merge to `alpha`).
- **FR-004**: Stable releases on `main` MUST continue to require the `release` label (no behavior change).
- **FR-005**: The system MUST publish alpha releases to PyPI as pre-release packages (installable with `--pre` flag).
- **FR-006**: The system MUST generate release notes for each alpha release based on merged PR titles and descriptions.
- **FR-007**: Alpha version tags MUST NOT include a "v" prefix (e.g., `2.0.0-alpha.0`, not `v2.0.0-alpha.0`), consistent with existing senselab conventions.
- **FR-008**: The CI test workflow MUST run on PRs targeting the `alpha` branch, with the same checks as PRs to `main`.
- **FR-009**: The documentation build workflow MUST NOT trigger on alpha releases (docs are only published for stable releases).
- **FR-010**: The system MUST use the same versioning and release tooling already in use by senselab, extended to support the alpha branch.

### Key Entities

- **Alpha Branch**: A long-lived branch forked from `main` that receives pre-release work. PRs merged here trigger automatic alpha releases.
- **Alpha Version Tag**: A semantic version with alpha suffix (e.g., `2.0.0-alpha.3`) applied to commits on the alpha branch.
- **Pre-release Package**: A PyPI package published with a pre-release version that users install via `pip install senselab --pre`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Merging a PR to the `alpha` branch produces a new alpha version tag and GitHub pre-release within 5 minutes.
- **SC-002**: The alpha package is installable from PyPI using `pip install senselab --pre` within 10 minutes of PR merge.
- **SC-003**: Stable releases on `main` continue to work identically to before (zero regression).
- **SC-004**: CI tests run on 100% of PRs targeting the `alpha` branch before merge.
- **SC-005**: No manual version management is required — versions are fully automated from git tags.

## Clarifications

### Session 2026-04-18

- Q: How will the pre-release version be set? → A: PR labels (`major`/`minor`/`patch`) control the version bump from the latest stable tag. Default bump is `patch` if no label. Auto determines the version and appends `-alpha.N`. Subsequent merges increment the alpha counter; a new bump label resets it.

## Assumptions

- The existing `auto` tool (Intuit Auto) and `.autorc` configuration will be extended, not replaced.
- The existing `hatchling` + `hatch-vcs` build system can handle alpha version tags without modification (hatch-vcs supports PEP 440 pre-release versions).
- The `alpha` branch will be created once and maintained as a long-lived branch (not recreated per release cycle).
- Branch protection rules for `alpha` will be configured manually by a maintainer (not automated in this spec).
- The same `PYPI_TOKEN` and `AUTO_ORG_TOKEN` secrets used for stable releases will work for alpha releases.
- The nobrainer pattern of "every merge to alpha triggers a release" is the desired behavior (no opt-out per PR).
