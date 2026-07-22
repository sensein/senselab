# Feature Specification: HuggingFace Model Cache & Version Consistency

**Feature Branch**: `20260706-213755-hf-model-cache`
**Created**: 2026-07-06
**Status**: Draft
**Input**: User description: "given the previous context of generating solutions for solving 429 errors, we want to create a more consistent way for dealing with caching of huggingface models, including loading them and checking that they have the correct versions"

## Clarifications

### Session 2026-07-06

- Q: Can every backend's version be verified, even loaders without a version parameter? → A: Yes. Version verification is a cache/hub-layer capability — compare the local copy's resolved immutable identity against the hub's current identity for the requested reference — and is independent of the loader's API. A loader lacking a version parameter is simply pointed at the version-identified local copy; it never weakens verification. Exact verification therefore applies to all backends (no best-effort exception); only a model source with no version concept at all would be out of scope.
- Q: Default freshness / version-refresh policy, and how are reproducible runs pinned? → A: Default is bounded auto-refresh — re-check only at a documented window boundary (coordinated across jobs, never per load) and adopt updates found then. Two freeze scopes override the default: (1) a **run-scoped freeze** (parameter) that snapshots resolved versions for the duration of the current run so they cannot shift mid-run (e.g. a 2+ day batch); (2) a **system-level freeze** (environment variable / config) that pins versions across all runs and disables re-checks for reproducible-science environments. Precedence: system freeze ⊇ run freeze ⊇ default.
- Q: What is this feature's completion boundary — how many backends get migrated? → A: Deliver the shared mechanism AND migrate all existing model backends within this feature, done incrementally (strangler-fig) behind unchanged public APIs with per-backend tests that preserve current behavior. Partial migration would leave the 429 problem unsolved for un-migrated backends.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reliable model loading under heavy parallelism (Priority: P1)

A researcher launches a large batch of audio-analysis jobs across many machines at once. Every
job needs the same set of models. Today, each job independently contacts the model hub to check
the model version at load time; at scale those simultaneous checks trip the hub's rate limits and
some jobs fail or silently produce empty/"missing" results. The user wants every job in the batch
to load its models successfully, regardless of how many run concurrently.

**Why this priority**: This is the originating problem (the 429 failures). Without it, large batch
runs are unreliable and results are silently incomplete — the highest-impact pain.

**Independent Test**: Launch many concurrent jobs that all load the same already-available model
and confirm they all succeed with no rate-limit-induced failures and no missing outputs.

**Acceptance Scenarios**:

1. **Given** a model already available locally, **When** many jobs load it at the same time,
   **Then** all loads succeed and none fail due to model-hub rate limiting.
2. **Given** a model already available locally, **When** a single job loads it, **Then** the load
   completes without any avoidable network request to the model hub.

---

### User Story 2 - Correct, verified model version (Priority: P1)

A user relies on results being produced by a specific, known version of a model. They want the
system to guarantee that the model it loads is the version that was requested — and to fail loudly
if it cannot, rather than silently substituting a different or stale version.

**Why this priority**: Silent version drift undermines the correctness and reproducibility of every
downstream result; it is as damaging as an outright failure but harder to detect.

**Independent Test**: Request a specific version and confirm the loaded model matches it; request a
version that is not available and confirm a clear error instead of a silent substitution.

**Acceptance Scenarios**:

1. **Given** a requested model version, **When** the model is loaded, **Then** the system confirms
   the loaded model is that exact version.
2. **Given** a requested version that is not present and cannot be retrieved, **When** a load is
   attempted, **Then** the system fails with a clear error naming the model and version, and never
   silently loads a different version.

---

### User Story 3 - Bounded freshness without per-load hub calls (Priority: P2)

A user occasionally wants the system to pick up a newer published version of a model, but not at the
cost of a hub call on every single load (which reintroduces the rate-limit problem). They want a
documented, predictable freshness policy: reuse the known version for a defined window, and re-check
for updates only when that window lapses.

**Why this priority**: Balances reproducibility against staying current, and is what keeps the
solution from silently regressing back into per-load hub traffic.

**Independent Test**: With a model cached and its upstream "latest" pointer moved, verify the system
keeps using the cached version until the freshness window elapses, then performs a single
coordinated re-check and adopts the new version.

**Acceptance Scenarios**:

1. **Given** a cached model within its freshness window, **When** it is loaded, **Then** no version
   re-check network call is made.
2. **Given** a cached model whose freshness window has elapsed and whose upstream version has
   changed, **When** it is loaded, **Then** exactly one coordinated re-check occurs and the updated
   version is adopted.
3. **Given** a run-scoped freeze is active, **When** models load repeatedly during a long run whose
   freshness window lapses partway through, **Then** every load uses the versions resolved at run
   start and no version changes mid-run.
4. **Given** a system-level freeze is active, **When** models load across multiple separate runs,
   **Then** every run uses the same pinned versions and no version re-check occurs.

---

### User Story 4 - One consistent mechanism across all model backends (Priority: P2)

A maintainer adds a new model backend to senselab. They want to inherit the same caching, loading,
and version-verification behavior automatically, instead of re-implementing it (differently, and
often with bugs) for each model family.

**Why this priority**: Inconsistent per-model handling is the root cause of the divergent behavior
and defects observed today; a single shared mechanism is what makes the guarantees durable.

**Independent Test**: Add a model backend through the shared mechanism and confirm it exhibits the
same loading, version-verification, and freshness guarantees without any backend-specific caching
code.

**Acceptance Scenarios**:

1. **Given** the shared mechanism, **When** a new model backend is added, **Then** it obtains the
   caching/version guarantees by implementing only the model-specific load step.
2. **Given** two different model families, **When** each is loaded, **Then** both exhibit identical
   caching and version-verification behavior.

### Edge Cases

- The model hub is unreachable but a valid version is already cached → the cached version is used
  and a warning is surfaced (no hard failure).
- Many jobs request a not-yet-downloaded model simultaneously → it is downloaded at most once and
  the others reuse it (no duplicate downloads, no corrupted cache).
- A requested version does not exist at all → clear, actionable error.
- The local cache is partial or corrupt for a model → the system detects it and re-obtains rather
  than loading a broken model or silently succeeding.
- Two distinct versions of the same model are used within one run → each loads correctly with no
  cross-contamination.
- A version re-check is due but the hub call fails → the system falls back to the last known-good
  cached version instead of failing the load.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST load any supported model without issuing an avoidable network request
  to the model hub when the requested version is already present locally.
- **FR-002**: The system MUST ensure a requested model and version are present locally before
  loading, downloading each at most once even when many jobs request it concurrently.
- **FR-003**: The system MUST verify — for every backend — that the model it loads is the requested
  version, by comparing the local copy's resolved version identity against the requested version at
  the cache/hub layer (independent of whether the model's own loader exposes a version parameter),
  and MUST fail with a clear, actionable error rather than silently loading a different or stale
  version.
- **FR-004**: The system MUST record the resolved, concrete version identity of each cached model so
  later loads can verify the version without a network call.
- **FR-005**: In the default (non-frozen) mode, the system MUST re-check for an updated version
  only after a documented freshness window (default: 7 days, configurable) has elapsed — never on
  every load — and MUST adopt the newer version when one is found.
- **FR-006**: When a version re-check is due, the system MUST perform at most one coordinated check
  shared across concurrent jobs (not one per job).
- **FR-007**: When the model hub is unreachable but a valid cached version exists, the system MUST
  use the cached version and continue, surfacing a warning.
- **FR-008**: When a requested model or version is neither cached nor retrievable, the system MUST
  fail with a clear error that identifies the model and the requested version.
- **FR-009**: The caching, loading, and version-verification behavior MUST be provided by a single
  shared mechanism used uniformly by all model backends, so behavior is consistent and new backends
  inherit it without re-implementation. All existing backends MUST be migrated onto this mechanism
  within this feature, incrementally and behind unchanged public APIs, with per-backend tests that
  preserve current behavior.
- **FR-010**: The system MUST provide the same guarantees (FR-001 through FR-009) regardless of the
  execution context in which a model runs.
- **FR-011**: The system MUST behave correctly when multiple distinct versions of the same model are
  used within a single run, with no cross-contamination between them.
- **FR-012**: All tuning values (e.g., the freshness window and retry limits) MUST be named,
  documented defaults that are configurable, consistent with the project's threshold-documentation
  policy.
- **FR-013**: The system MUST provide a **run-scoped freeze** (a simple parameter) that snapshots
  the resolved model versions at the start of a run and holds them for the run's entire duration,
  so versions cannot change mid-run (e.g., during a multi-day batch) even if a freshness window
  lapses during the run.
- **FR-014**: The system MUST provide a **system-level freeze** (an environment variable or
  configuration) that pins model versions across all runs and disables version re-checks entirely,
  for reproducible-science environments. Freeze controls take precedence over default refresh, and
  a system-level freeze takes precedence over run-scoped behavior (system ⊇ run ⊇ default).
- **FR-015**: Version identity MUST be determined at the cache/hub layer — the resolved immutable
  identity of the local snapshot versus the hub's current identity for the requested reference —
  not from the model loader. A loader without a version parameter MUST NOT weaken verification; the
  system points it at the version-identified local copy instead.

### Key Entities

- **Model reference**: What the caller asks for — a model identity plus a requested version selector
  (an exact version, a named channel such as "latest", or a tag).
- **Resolved version identity**: The concrete, immutable version that a selector maps to; what the
  system verifies a loaded model against.
- **Cached model record**: A locally available model together with its resolved version identity and
  the time its version was last verified (used to decide freshness without a network call).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a batch of at least 100 concurrent jobs loading the same already-available model,
  100% load successfully with zero model-hub rate-limit failures and zero missing outputs.
- **SC-002**: Loading an already-available model within its freshness window performs zero network
  calls to the model hub.
- **SC-003**: A model requested simultaneously by many first-time jobs is downloaded at most once.
- **SC-004**: When the requested version differs from what is cached, the system either loads the
  correct version or fails with a clear error 100% of the time — it never silently serves a
  different version.
- **SC-005**: A changed upstream version is adopted within one freshness window (and not before),
  and the check that detects it is performed at most once across concurrent jobs.
- **SC-006**: Adding a new model backend requires implementing only the model-specific load step and
  introduces no backend-specific caching or version-checking code.
- **SC-007**: When the model hub is unreachable, every model that is already cached still loads
  successfully (zero hard failures for cached models), with a warning surfaced.
- **SC-008**: With a run-scoped freeze active, all loads within a single run use identical model
  versions from start to finish, even if a freshness window lapses mid-run (zero mid-run version
  changes).
- **SC-009**: With a system-level freeze active, repeated runs load identical model versions and
  perform zero version re-checks, regardless of upstream changes.
- **SC-010**: Every existing model backend loads through the shared mechanism; none retains
  bespoke caching or version-checking code (verified by inspection at feature completion).

## Assumptions

- Models are obtained from a shared external model hub that enforces per-account/per-source rate
  limits; exceeding them is the cause of the observed failures.
- Every supported model comes from a versioned hub that exposes a resolvable, immutable version
  identity for a given reference (so version verification is always possible); a source with no
  version concept at all is out of scope.
- Jobs frequently run many-at-once on shared compute and share a common local cache location.
- Existing cross-process download coordination (single-downloader locking) can be built upon rather
  than reinvented.
- The set of affected backends is the existing senselab speech/audio model loaders (speech-to-text,
  forced alignment, speaker embeddings, and similar) plus any added later.
- **Default freshness is bounded, not per-load**: by default the system re-checks for a newer
  version only after a documented window (default 7 days, configurable) and adopts updates found
  then; it never checks on every load. Two freeze scopes provide reproducibility: a run-scoped
  freeze (parameter) that holds versions steady for the duration of one run, and a system-level
  freeze (environment variable / config) that pins versions across all runs and disables re-checks
  (see FR-013, FR-014). The 7-day default window is a documented, configurable threshold.

## Out of Scope

- Selecting which models to use, or evaluating model quality/accuracy.
- The external model hub's own availability or rate-limit policy.
- Changing what the models compute or their outputs.

## Dependencies

- Read/write access to a shared local model cache location across all participating jobs.
- The external model hub for first-time downloads and for version re-checks when a freshness window
  lapses.
