# Implementation Plan: Alpha Prerelease Process

**Branch**: `20260418-104204-alpha-prerelease-process` | **Date**: 2026-04-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260418-104204-alpha-prerelease-process/spec.md`

## Summary

Add an `alpha` prerelease branch to senselab, mirroring nobrainer's setup. PRs merged to `alpha` automatically create alpha version tags and publish pre-release packages to PyPI. Stable releases on `main` are unchanged. Implementation requires modifying 3 files and creating the alpha branch.

## Technical Context

**Language/Version**: N/A (CI/CD configuration only — YAML, JSON)
**Primary Dependencies**: Intuit Auto (v11.2.1), hatch-vcs, GitHub Actions
**Storage**: N/A
**Testing**: Manual verification via PR merge + workflow run
**Target Platform**: GitHub Actions (ubuntu-latest runners)
**Project Type**: CI/CD pipeline configuration
**Performance Goals**: Release pipeline completes within 5 minutes of PR merge
**Constraints**: Must not break existing stable release process on `main`
**Scale/Scope**: 3 files modified, 1 branch created

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution is a default template (not project-specific). No gates defined. **PASS.**

## Project Structure

### Documentation (this feature)

```text
specs/20260418-104204-alpha-prerelease-process/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: Reference analysis
├── data-model.md        # Phase 1: Configuration model
└── checklists/
    └── requirements.md  # Specification quality checklist
```

### Source Code (repository root)

```text
.autorc                              # Modified: add prereleaseBranches
.github/workflows/release.yaml      # Modified: PR-based trigger for main+alpha
.github/workflows/tests.yaml        # Already handles alpha (PR-based trigger)
```

**Structure Decision**: This is purely a CI/CD configuration change. No source code modifications. Only 2 config files are modified and the alpha branch is created.

## Phase 0: Research

### Key Findings

**Nobrainer vs Senselab release workflow differences:**

| Aspect | Nobrainer | Senselab (current) | Senselab (target) |
|--------|-----------|--------------------|--------------------|
| Release trigger | PR closed (merged) | Push to main | PR closed (merged) |
| Trigger branches | master, alpha | main | main, alpha |
| Release condition | `release` label OR alpha branch | Any push | `release` label OR alpha branch |
| `.autorc` prereleaseBranches | `["alpha"]` | (absent) | `["alpha"]` |
| `.autorc` onlyPublishWithReleaseLabel | `false` | `true` | `false` |
| Auto version | v11.2.1 | v11.1.2 | v11.2.1 |
| GH_TOKEN secret name | AUTO_USER_TOKEN | AUTO_ORG_TOKEN | AUTO_ORG_TOKEN |
| Publish workflow | Triggered by release published | Triggered by release published | No change needed |

**Key decision**: Change `onlyPublishWithReleaseLabel` from `true` to `false`. With `true`, Auto only creates releases when the `release` label is present. With `false`, Auto uses its default behavior which, combined with `prereleaseBranches: ["alpha"]`, means:
- PRs to `main` with `release` label → stable release
- PRs to `alpha` (any) → alpha pre-release
- PRs to `main` without label → no release (this is controlled by the workflow `if` condition, not by `.autorc`)

**Publish workflow**: Already triggers on `release: types: [published]`, which fires for both stable and pre-releases. No modification needed.

**Tests workflow**: Already uses PR-based triggers (`pull_request: types: [opened, synchronize, reopened, labeled]`). PRs against `alpha` will already trigger tests. No modification needed.

**Docs workflow**: Triggers on `release: types: [published]`. Auto marks alpha releases as `prerelease: true` in GitHub, but the docs workflow triggers on any `published` event. If FR-009 (docs only for stable) is required, the docs workflow needs a condition to skip pre-releases. However, this is a minor concern and can be addressed separately.

## Phase 1: Implementation Steps

### Step 1: Update `.autorc`

Current:
```json
{
    "onlyPublishWithReleaseLabel": true,
    "baseBranch": "main",
    "author": "bot <fabiocat@mit.edu>",
    "noVersionPrefix": true,
    "plugins": ["git-tag"]
}
```

Target:
```json
{
    "onlyPublishWithReleaseLabel": false,
    "baseBranch": "main",
    "prereleaseBranches": ["alpha"],
    "author": "bot <fabiocat@mit.edu>",
    "noVersionPrefix": true,
    "plugins": ["git-tag"]
}
```

Changes:
- `onlyPublishWithReleaseLabel`: `true` → `false`
- Add `prereleaseBranches: ["alpha"]`

### Step 2: Update `.github/workflows/release.yaml`

Current trigger: `on: push: branches: [main]`

Target trigger (matching nobrainer):
```yaml
on:
  pull_request:
    branches: [main, alpha]
    types: [closed]
```

Current condition: none (runs on every push to main)

Target condition:
```yaml
if: >-
  github.event.pull_request.merged == true &&
  (
    contains(github.event.pull_request.labels.*.name, 'release') ||
    github.event.pull_request.base.ref == 'alpha'
  )
```

Also update Auto version from v11.1.2 to v11.2.1 for consistency with nobrainer.

### Step 3: Create the `alpha` branch

```bash
git checkout main
git pull origin main
git checkout -b alpha
git push -u origin alpha
```

### Step 4: Verify

1. Merge a test PR to `alpha` → verify auto creates alpha tag + GitHub pre-release
2. Verify PyPI publish triggers on the pre-release
3. Merge a PR to `main` WITHOUT `release` label → verify no release is created
4. Merge a PR to `main` WITH `release` label → verify stable release is created

## Risks

- **Changing `onlyPublishWithReleaseLabel` to `false`**: The workflow `if` condition now gates releases (label required for main, always for alpha). If the `if` condition is wrong, pushes to main could create unwanted releases. Mitigated by matching the proven nobrainer pattern exactly.
- **Auto version upgrade**: Upgrading from v11.1.2 to v11.2.1 could introduce behavioral changes. Low risk — both are minor versions in the v11 series.
