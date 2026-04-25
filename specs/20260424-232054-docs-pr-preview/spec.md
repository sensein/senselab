# Feature Specification: Docs PR Preview + Coverage Audit

**Feature Branch**: `20260424-232054-docs-pr-preview`
**Created**: 2026-04-24
**Status**: Draft
**Input**: Review the docs generation GitHub workflow and update it so PR previews are generated and then deleted after a PR is merged or closed. Check the generated docs page to make sure all aspects of senselab are covered and it's easy for a user to navigate. Also check any readme or other docs for consistency with the current codebase.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Previews Docs Changes in PR (Priority: P1)

A developer opens a PR that modifies code with docstrings, tutorials, or doc.md files. A docs preview is automatically built and deployed to a temporary URL. The PR comment includes a link to the preview. When the PR is merged or closed, the preview is automatically cleaned up.

**Why this priority**: Docs changes are currently only visible after a release is published. Developers and reviewers have no way to preview how documentation will look, leading to formatting issues and broken links reaching production.

**Independent Test**: Open a PR that modifies a docstring, verify a preview link appears as a PR comment, click the link and see the rendered docs, then merge/close the PR and verify the preview is removed.

**Acceptance Scenarios**:

1. **Given** a PR is opened with code changes, **When** CI runs, **Then** a docs preview is built and a comment with the preview URL is posted on the PR.
2. **Given** a docs preview exists for a PR, **When** the PR is merged, **Then** the preview deployment is automatically deleted within minutes.
3. **Given** a docs preview exists for a PR, **When** the PR is closed without merging, **Then** the preview deployment is automatically deleted.
4. **Given** a PR is updated with new commits, **When** CI runs again, **Then** the preview is rebuilt with the latest changes (same URL).

---

### User Story 2 - All Modules Have Documentation (Priority: P2)

A user browsing the docs site can find documentation for every module in senselab. No module is missing from the generated documentation. Each module has a doc.md providing context beyond auto-generated API docs.

**Why this priority**: 6 core audio modules currently lack doc.md files. Users looking for preprocessing, plotting, quality control, or I/O documentation find gaps.

**Independent Test**: Generate docs locally and verify every module under audio/tasks/, video/tasks/, text/tasks/, and utils/ appears with meaningful content.

**Acceptance Scenarios**:

1. **Given** the docs are generated, **When** a user navigates to the plotting module, **Then** they find documentation for `plot_aligned_panels` and other plotting functions with usage examples.
2. **Given** the docs are generated, **When** a user navigates to any audio task module, **Then** every module has a doc.md with context about what it does and when to use it.
3. **Given** new functions were added (SPARC decode/convert, PPG phoneme analysis, SpeechBrain SER, plot_aligned_panels), **When** docs are generated, **Then** these functions appear with complete docstrings.

---

### User Story 3 - README and Docs Are Consistent (Priority: P2)

A user reading the README finds accurate installation instructions, feature descriptions, and links that all point to the correct documentation site. No dead links or outdated references.

**Why this priority**: The README currently has inconsistent documentation URLs (sensein.group vs sensein.github.io) and the feature list may not reflect all recent additions.

**Independent Test**: Review README links, verify they resolve, and compare feature descriptions against the actual codebase.

**Acceptance Scenarios**:

1. **Given** the README, **When** a user clicks any documentation link, **Then** they reach the correct, live documentation page.
2. **Given** the README feature list, **When** compared to the codebase, **Then** every major capability (including SPARC decode/convert, PPG phoneme analysis, SpeechBrain SER, plot_aligned_panels, recording widgets) is mentioned or its category is represented.
3. **Given** the tutorials/README.md, **When** compared to actual tutorial files, **Then** all tutorials are listed with correct names and descriptions.

---

### Edge Cases

- What happens if the docs build fails on a PR? (The preview comment should indicate failure, not post a broken link.)
- What happens if a PR has no docs-relevant changes? (Preview should still build — all code has docstrings.)
- What happens if multiple PRs are open simultaneously? (Each gets its own isolated preview URL.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The CI system MUST build a docs preview for every non-draft PR and post a comment with the preview URL.
- **FR-002**: The CI system MUST delete the docs preview when the PR is merged or closed.
- **FR-003**: Every module under `audio/tasks/`, `video/tasks/`, `text/tasks/`, and `utils/` MUST have a `doc.md` file with a brief description of the module's purpose and typical use cases.
- **FR-004**: The README.md MUST have consistent documentation links pointing to the canonical docs URL (sensein.github.io/senselab).
- **FR-005**: The tutorials/README.md MUST list all current tutorials with accurate names and descriptions matching the actual notebook files.
- **FR-006**: All new public functions added in recent PRs (SPARC decode/convert, PPG phoneme analysis, SpeechBrain SER, plot_aligned_panels) MUST have complete Google-style docstrings that render correctly in pdoc.
- **FR-007**: The docs preview for a given PR MUST be updated when new commits are pushed to that PR.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of open PRs with code changes get a docs preview comment with a working link within the CI run time.
- **SC-002**: 100% of merged/closed PRs have their preview deleted within 5 minutes.
- **SC-003**: 0 modules under the task directories lack a doc.md file.
- **SC-004**: 0 broken links in README.md or tutorials/README.md.
- **SC-005**: All 7 missing doc.md files are created with meaningful content (not just placeholders).

## Assumptions

- GitHub Pages is used for docs deployment (already configured on the `docs` branch).
- PR previews can be deployed as subdirectories under the docs branch (e.g., `pr-123/`) or use a separate deployment mechanism (e.g., Netlify, Surge.sh, or GitHub Pages environments).
- pdoc is the documentation generator (already configured with custom theme).
- The docs workflow currently triggers only on releases; the new PR preview workflow triggers on pull_request events.
- Preview cleanup triggers on pull_request closed event (covers both merged and abandoned PRs).
