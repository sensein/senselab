# Research: Docs PR Preview + Coverage Audit

**Date**: 2026-04-24

## R1: PR Docs Preview Approach

**Decision**: Use GitHub Pages with subdirectory-based PR previews on the `docs` branch. Each PR gets deployed to `docs/pr-{number}/`, and cleanup removes that directory on PR close/merge.

**Rationale**: The existing docs infrastructure uses `JamesIves/github-pages-deploy-action@v4` deploying to the `docs` branch. PR previews as subdirectories keep everything in one place — no external services, no secrets, no additional costs.

**Alternatives considered**:
- Netlify Deploy Previews: Requires Netlify account and secret token
- Surge.sh: Requires account and CI token
- GitHub Environments: More complex, designed for staging/production, overkill for docs previews
- Separate gh-pages branch per PR: Branch pollution

## R2: PR Preview Workflow Design

**Decision**: Two workflow triggers:
1. `pull_request: [opened, synchronize, reopened]` → build docs and deploy to `pr-{number}/` subdirectory
2. `pull_request: [closed]` → delete `pr-{number}/` directory from docs branch

The build step posts/updates a PR comment with the preview URL using `peter-evans/create-or-update-comment`.

**Rationale**: The `synchronize` trigger ensures rebuilds on new commits. The `closed` trigger covers both merged and abandoned PRs. Comment posting gives reviewers a direct link.

## R3: Missing doc.md Files

**Decision**: Create doc.md files for these 7 modules:
1. `src/senselab/audio/tasks/preprocessing/doc.md` — resample, downmix, normalize, chunk
2. `src/senselab/audio/tasks/input_output/doc.md` — read/write audio files
3. `src/senselab/audio/tasks/plotting/doc.md` — waveform, spectrogram, play_audio, plot_aligned_panels
4. `src/senselab/audio/tasks/quality_control/doc.md` — QC framework, metrics, checks
5. `src/senselab/audio/tasks/ssl_embeddings/doc.md` — self-supervised learning embeddings
6. `src/senselab/audio/tasks/speaker_diarization_evaluation/doc.md` — DER metrics
7. `src/senselab/text/tasks/embeddings_extraction/doc.md` — text embeddings

Each doc.md should be 3-5 sentences: what the module does, when to use it, and a usage pointer.

## R4: README URL Fix

**Decision**: Standardize all documentation URLs to `https://sensein.github.io/senselab` (the canonical GitHub Pages URL). Fix README line 39 which currently points to `https://sensein.group/senselab/senselab.html`.

**Rationale**: pyproject.toml and the badge both use `sensein.github.io`. The `sensein.group` URL may be an old redirect or alternate domain.
