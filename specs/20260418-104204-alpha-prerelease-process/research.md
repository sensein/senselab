# Research: Alpha Prerelease Process

## Decision 1: Release workflow trigger mechanism

**Decision**: Change from `push` trigger to `pull_request: closed` trigger
**Rationale**: The push trigger fires on every commit to main, relying on Auto's internal logic to decide whether to create a release. The PR-closed trigger is more explicit — it only fires when a PR is merged, and the workflow `if` condition controls whether a release is created based on branch + labels. This matches nobrainer's proven pattern and gives finer control.
**Alternatives considered**: Keep push trigger and add branch filtering in Auto — rejected because it doesn't extend cleanly to alpha prereleases.

## Decision 2: `onlyPublishWithReleaseLabel` setting

**Decision**: Set to `false`
**Rationale**: With `true`, Auto itself gates releases by label. With `false`, the workflow `if` condition handles gating. This is necessary because alpha releases should NOT require a label. The workflow condition `contains(labels, 'release') || base.ref == 'alpha'` provides the exact behavior needed: label-gated for main, automatic for alpha.
**Alternatives considered**: Keep `true` and add a separate alpha-specific workflow — rejected as unnecessary duplication.

## Decision 3: Auto version

**Decision**: Upgrade from v11.1.2 to v11.2.1
**Rationale**: Aligns with nobrainer. v11.2.1 has better prerelease branch support. Minor version bump with no breaking changes.
**Alternatives considered**: Stay on v11.1.2 — viable but v11.2.1 is tested with prerelease branches in nobrainer.

## Decision 4: Docs workflow handling for alpha releases

**Decision**: No change for now (docs may build on alpha releases)
**Rationale**: The docs workflow triggers on `release: published`. GitHub marks alpha releases as pre-releases, but the `published` event fires regardless. Adding a skip condition (`if: !github.event.release.prerelease`) is trivial but out of scope for the initial implementation. Can be added as a follow-up if alpha docs builds cause issues.
**Alternatives considered**: Add pre-release skip condition immediately — deferred to keep scope minimal.
