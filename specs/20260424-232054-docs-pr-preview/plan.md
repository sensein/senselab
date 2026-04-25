# Implementation Plan: Docs PR Preview + Coverage Audit

**Branch**: `20260424-232054-docs-pr-preview` | **Date**: 2026-04-24 | **Spec**: [spec.md](spec.md)

## Summary

Add PR docs preview workflow (build + deploy on PR, cleanup on close), fill 7 missing doc.md files, fix README URL inconsistency, and verify tutorials/README.md accuracy.

## Technical Context

**Language/Version**: YAML (GitHub Actions), Markdown
**Primary Dependencies**: pdoc, JamesIves/github-pages-deploy-action@v4, peter-evans/create-or-update-comment
**Storage**: GitHub Pages (`docs` branch)
**Testing**: Open a test PR and verify preview appears
**Target Platform**: GitHub Actions CI
**Project Type**: CI/CD workflow + documentation
**Constraints**: Must not break existing release-triggered docs deployment

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Docs build uses `uv run pdoc` |
| II. Encapsulated Testing | PASS | CI builds docs in isolated runner |
| III. Commit Early and Often | PASS | Incremental changes |
| IV. CI Must Stay Green | PASS | New workflow, doesn't modify existing |
| VII. Simplicity First | PASS | Uses existing GitHub Pages infrastructure |

## Project Structure

```text
.github/workflows/
├── docs.yaml                    # EXISTING (release docs) — unchanged
└── docs-preview.yaml            # NEW (PR preview + cleanup)

src/senselab/audio/tasks/
├── preprocessing/doc.md         # NEW
├── input_output/doc.md          # NEW
├── plotting/doc.md              # NEW
├── quality_control/doc.md       # NEW
├── ssl_embeddings/doc.md        # NEW
└── speaker_diarization_evaluation/doc.md  # NEW

src/senselab/text/tasks/
└── embeddings_extraction/doc.md # NEW

README.md                        # UPDATED (fix URL)
tutorials/README.md              # VERIFIED (add new tutorials if missing)
```

## Implementation Phases

### Phase 1: Create PR Docs Preview Workflow

Create `.github/workflows/docs-preview.yaml` with two jobs:

**Job 1: build-and-deploy** (triggers on PR opened/synchronize/reopened)
1. Build docs with pdoc (same command as release workflow)
2. Deploy to `docs` branch under `pr-{PR_NUMBER}/` subdirectory
3. Post/update PR comment with preview URL

**Job 2: cleanup** (triggers on PR closed)
1. Remove `pr-{PR_NUMBER}/` from docs branch
2. Update PR comment to note preview was removed

### Phase 2: Add Missing doc.md Files

Create 7 doc.md files with 3-5 sentence descriptions.

### Phase 3: Fix README and Verify

1. Fix README.md line 39 URL
2. Verify tutorials/README.md lists all 20 tutorials including new ones
3. Verify all feature descriptions match codebase
