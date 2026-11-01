# Full review: architecture as built vs design (target `3a468aff`, branch `2x/phase-1`)

Angle: package structure, module boundaries and public surfaces of `senselab-worker` and `senselab-core`
against `design.md`, `execution.md`, `engine-choice.md`, `workflow-engines.md`,
`pydra-extension-points.md`, and the pydra companion (`~/software/nipype/pydra-ext`, branch
`enh/execution-extensions` at `c697d8f9`, `proposals/0001-execution-extensions.md`); plus a code-quality sample.
Paths below are relative to the worktree unless they say otherwise. `spec/` stands for
`specs/20261030-140022-senselab-2x-api/`.

Counts used throughout (from `wc -l`, at `3a468aff`):
- package source: 11,853 lines (`packages/*/src`);
- package tests: about 23,450 lines;
- release gate: `scripts/release_gate.py` 1,092, plus `tests/release_gate_test.py` 1,843 and `tests/release_fixture.py` 293;
- fixture budget: `scripts/check_fixture_budget.py` 455, plus its test 423;
- the lint and guard layer: `packages/senselab-core/tests/lint/` 3,800 lines in 15 files (the 262-line import resolver among them);
- the spec docs grew 3,744 lines between `3a36b4be` (the last design commit) and `3a468aff`. `design.md` went from 860 to 1,066 lines, ADR-006 grew by 265 and ADR-003 by 106.

## What matches the design

Recorded here so the findings below are read in proportion.
- **The worker is stdlib plus numpy.** Every import under `packages/senselab-worker/src` is stdlib, numpy or
  `senselab_worker`. `pyproject.toml` declares `numpy>=1.23` only, with `requires-python>=3.10`. The
  `worker-is-the-base` contract in `.importlinter:19-29` forbids core and pydantic, and the floor job runs
  the built wheel on 3.10 (`.github/workflows/2x-tests.yaml:114-131`).
- **The kernel is separate from the runtime, and the direction is enforced.**
  `.importlinter:31-36` (`kernel-below-runtime`) and `:47-54` (`layers`: cli|testing → runtime → kernel →
  worker) are in place, and `packages-not-1x` forbids importing 1.x. A grep finds no kernel import of
  `runtime` or `testing`.
- **Core installs no framework.** `core-is-framework-free` holds. The declared dependencies are numpy,
  pydantic, pyyaml, huggingface-hub and soundfile, with the worker pinned exactly.
- **Hashing is for exactness only, and the default is `equivalent`.** `LogicalKey`
  (`kernel/identity.py:207-235`) holds no code, environment or context. `ExecutionContext` is recorded,
  not keyed (`:238-256`). `ReusePolicy` has exact, equivalent and tolerant, and `DEFAULT_REUSE_POLICY =
  EQUIVALENT` (`:259-267`). The store records a reuse activity for every non-exact hit
  (`runtime/store.py:884`). This reflects owner items 20 and 21 and ADR-002.
- **vLLM-omni is kept**, answering owner item 21 ("why drop vllm omni?"). See `design.md:96,282,575,941,1058`,
  `principles.md:423` and `execution.md:16`. No document drops it.
- **The graph is senselab's (ADR-001).** The ADR is accepted, and `declare.py:1-3` says "Declarations
  only: no graph, planner or execution". The graph format itself is not built, and Phase 1 did not schedule it.

## Phase 1 scope: built, partly built, overbuilt

The scope is design §6.2 Phase 1 (`design.md:892-907`) plus what `plans/phase-1.md:5-15` added from `docs.md`.

| Item | Status | Where |
|---|---|---|
| Workspace, three packages | built | `pyproject.toml`, `packages/*/pyproject.toml` |
| Worker codec, handshake (I-2), wire spec, floor test | built | `senselab_worker/{codec,serve,launch}.py`, `spec/wire-protocol.md`, the floor job |
| Signal/Audio, timeline, annotation, embedding | built | `kernel/{signals,time,results}.py` |
| Model/ResolvedModel, rewired revision, sha constants as data | built | `kernel/identity.py`, `runtime/{models,revision}.py`, `runtime/data/weights.yaml` |
| Result store with ADR-002 key, ProvStore writer | built | `runtime/store.py`, `runtime/provenance.py`, `runtime/bep028.py` |
| Torch-free hardware probe | built, and it carries 1.x-only code (F-9) | `runtime/hardware.py` |
| License record | built (52 lines) | `kernel/license.py` |
| `fake` stack and generated contract tests | built | `senselab_worker/fake.py`, `runtime/fake_backend.py`, `testing/contracts.py` |
| Data fetch, generators, fixture budget (docs.md Phase 1) | built; the budget is overbuilt (F-1) | `runtime/fetch.py`, `testing/synth.py`, `scripts/check_fixture_budget.py` |
| Task/backend declarations, `TaskParams` | **pulled forward**: design puts these in Phase 2 (`design.md:911`) | `kernel/declare.py` (265 lines). Acceptable, because the contract tests need a registry |
| Exit: core-only on 3.11 and 3.14 | built | `2x-tests.yaml:62` |
| Exit: 1.x tests pass after rewiring, triage outputs unchanged | superseded by the ADR-009 amendment, but design.md still states it (F-4) | `design.md:907` |
| **Not designed, built in hardening** | release gate (3.2k lines with tests), workflow-expression guard (424), written-names, destructive-ops, owned-deletion and digest-binding AST guards over a 262-line import resolver, generated value contracts (`testing/values.py`, 746), the dir-fd storage sandbox (`runtime/storage.py`, 737), process-group supervision in `worker_client.py` (1,004), a Unicode assigned-code-point table in the worker (`data/unicode-assigned.txt`, `scripts/unicode_table.py`) | F-1, F-2 |

**Execution status.** Pools, substrates, resident resources, GPU/CPU placement, Slurm, `Env.server`,
`VllmServer` and swappable stacks are all Phase 2–3 (`design.md:909-942`). None of them exists in the
packages, and none is overdue. Nothing in the code contradicts the execution design: faults are typed at
the source as ADR-009 says (`senselab_worker/faults.py`), the probe needs no torch, and `numerics_class`
feeds the context, not the key. The contradictions are in the documents (F-5, F-6). Two items are
decided but not placed in any plan:
- the version-bump CI check (`design.md:409`, `testing.md:139`);
- the advisory prose review by a small model (P-31 check, `engineering.md:179`; owner item 22). See F-1.

---

## Critical

### F-1. The hardening grew a list-driven guard and gate layer that the design never asked for, against owner items 14, 15 and 22

- **Departs from:**
  - owner item 14 (`owner-requests.md:20`): "we want to make verbosity minimal for the human and any agent.
    code itself should be clear to read and compact/efficient (think fastio code)";
  - item 15 (`:21`): "elegant, testable, robust algorithmic efficiency";
  - item 22 (`:28`): "seems brittle making assumptions. why not use agents to sharpen rather than CI through
    lists. perhaps some lightweight models like nimble could evaluate during CI runs";
  - P-35 (`engineering.md:51`): "Each break is fixed **by an architectural change**, not a patch for that case";
  - the designed fixture budget (`docs.md:64`): "at most 256 KB per file and 2 MB in total, checked in CI".
- **What the branch does instead:**
  - **Release gate** (`scripts/release_gate.py`, 1,092 lines; tests 2,136). The pre-implementation design has no
    gate at all: `design.md` at `3a36b4be` §5 says only "single git tag ... exact same-version pin". It now
    occupies `design.md:707-784`, 78 lines of build-tool detail: `.gitignore` location, `ENV_KEPT`,
    `safe.directory`, pinned `vcs-versioning`. That is a release-engineering product in its own right, for a
    package with no release yet.
  - **Fixture budget** (455 + 423 lines). It binds caps to a freeze commit with content-hashed allowlists,
    parses uv and venv lock files line by line (`design.md:795-799`), and derives caps from the whole 1.x
    tree. The caps are 512 KiB per code file and about 21 MB code total (`scripts/data/fixture_budget.yaml:1-15`).
    The design call was a 256 KB / 2 MB budget.
  - **Lint layer** (3,800 lines). It is AST sweeps driven by allowlists:
    - `ALLOWED_RAW` (`tests/lint/written_names_test.py`);
    - `DESTRUCTIVE_MODULE_CALLS` (`destructive_ops_test.py`);
    - `ALLOWED` (`owned_deletion_test.py:12-15`);
    - `MOVED_HELPERS` (`moved_modules_test.py:56`);
    - an 18-entry banned-api list (`pyproject.toml:160-187`);
    - a 23-name `ignore_imports` worklist of 1.x scripts (`.importlinter:56-90`);
    - a workflow-expression trust list (`design.md:806-828`).

    Each round added a list for the case that broke. That is the "CI through lists" item 22 rejects.
  - **Runtime armour.** `runtime/storage.py` (737) has no symlink following, grammar-checked names and
    descriptor-only modes. The quarantine, set-aside, restore and reseal machinery in `store.py:524-716`
    is about 190 lines. `worker_client.py` (1,004) supervises process groups and defers signals. The design
    asked for "atomic rename, first writer wins; failures are not cached" (`design.md:426-431` as
    originally written) and for "persistent, with a JSON control line" (`design.md:294`).
  - What item 22 asked for (model and agent sharpening in CI) is not built and not in any plan.
- **Scale:** gate, budget and lints come to about 9,000 lines, against 11,853 lines of package source.
- **Kind:** deviation, and drift: the H-R rulings in hardening superseded owner item 22 without the owner.
- **Correction:**
  1. Cut the release gate to what a first alpha needs: build, install each wheel alone at its floor, import, and check the pins. That is about 150 lines. Drop the member derivation, the RECORD re-derivation, the sdist rebuild and the environment allowlist until a release defect is measured.
  2. Replace the fixture budget with the designed two numbers over `packages/` and test data.
  3. Keep import-linter contracts that state architecture (layers, worker base, framework-free).
  4. Retire the allowlist-driven AST guards (written names, destructive ops, owned deletion, digest bindings, banned-api list, 1.x script worklist). The invariant they protect belongs in the one API that owns it (`StorageDir`, `digest`), with a test of that API.
  5. Build the advisory model and agent prose and architecture review that P-31 and item 22 call for.
  6. Move the as-built gate, budget and guard text out of `design.md` (see F-2).

## Important

### F-2. The authority documents were rewritten during implementation into as-built specifications

- **Departs from:** the brief's authority order (the owner, then principles, then ADRs, then design; plans
  and rulings are not authority). Also P-31 (`engineering.md:176-177`): "say each thing once, in the fewest
  exact words. Reasons and history live in `specs/`".
- **What the branch does instead:** implementation commits edited `design.md` 12 times
  (`git log 3a36b4be..3a468aff -- spec/design.md`). Examples: `d74e64ad` "design.md §5 states the release gate,
  fixture budget and workflow guard as built", `9d31cd7c`, `33e7e96a` and `ccf58520`. The edits added:
  - `design.md:157-179`, the construction path, with measured timings and pydantic internals;
  - `:180-192`, the source-stamp costs;
  - `:432-442`, NTFS caseless folding and the 2770/0440 mode semantics;
  - `:444-470`, the provenance token families;
  - `:707-828`, the gate, budget and workflow guard.

  `docs.md:72` gained a 9-line `BlobStamp` and `os.link` race paragraph. ADR-006 grew by 265 lines and
  ADR-003 by 106. A design meant for owner review now carries controller-ruled implementation detail.
  That also makes the controller's own rulings look like design authority to the next reviewer.
- **Kind:** drift.
- **Correction:** restore `design.md` to design-level statements. Move the as-built detail into
  `wire-protocol.md`, the ADR consequences, or a `spec/as-built/` note. Get the owner's sign-off on the
  design-level changes the hardening needs, such as "committed entries are read-only".

### F-3. The kernel decodes files and hashes the whole file on every read

- **Departs from:**
  - ADR-009 (`adr/ADR-009-migration-kernel.md:33-35`): the kernel is "data protocols and types, identity,
    the graph format, serialisation";
  - `plans/phase-1.md:20`: "a **kernel** (types, identity, declarations; no I/O frameworks)";
  - `design.md:152`: "Loading is lazy, by path, byte range, stream or time range";
  - `design.md:555`: decoding is per-modality, not shared kernel;
  - owner item 15 (algorithmic efficiency) and P-33.
- **What the branch does instead:** `kernel/signals.py:14` imports `soundfile`. `:179-216` opens, stamps
  and decodes files, and keeps a process-wide memo (`OrderedDict` plus a lock). Every decoded `read`
  re-hashes the entire file (`content_identity(raw)` at `:187`). `design.md:189-192` documents the cost:
  "a 30 s window `read` 0.063 s ... a read therefore costs one full hash of the file whatever its window,
  about 0.5 s per GB". A windowed pass over hour-long recordings is therefore O(file) per window. That is
  a for-loop cost model, not a fastio one.
- **Kind:** deviation.
- **Correction:**
  1. Move decoding and stamping to a runtime or audio-adapter module behind a `RegularSeries` reader protocol. The kernel keeps the types and the identity.
  2. Verify freshness by `stat` plus per-chunk digests of the decoded window (ADR-002 already chunks identity). The design already names this path as unbuilt at `design.md:191`.

### F-4. The owner's corrections landed in ADR-009 but not in design.md or engineering.md

- **Departs from:**
  - owner item 30 (`owner-requests.md:39`): "are PRs being merged into a 2.x branch or to alpha? should be the former";
  - the ADR-009 amendment (`adr/ADR-009-migration-kernel.md:19-23`): `src/` is a frozen 1.x pool,
    "not tested and not linted ... 1.x code that breaks because core changed is not fixed there".
- **What the branch still says:**
  - `design.md:832` "Strategy (in place on `alpha`)";
  - `:838-839` "every caller is rewired in the same PR";
  - `:842-844` triage rewired per port, "1.x import side effects stay until Phase 5";
  - `:907` Phase 1 exit "1.x tests pass after the rewiring, and triage outputs unchanged";
  - `:1042` "in place on `alpha`";
  - `engineering.md:127` (P-27) "2.x is built in place on `alpha`".

  In addition, `.importlinter:5` puts the 1.x `scripts/` at the repo root (66 files, including the 15
  `extend_*.py` drivers) under 2.x lint contracts with a per-file ignore list. That contradicts "not linted".
- **Kind:** omission (the correction is half-landed).
- **Correction:** amend `design.md` §6.1, §6.2 (the Phase 1 exit) and §8, and P-27, to the 2.x-branch and
  frozen-pool model. Treat 1.x `scripts/` like `src/`: drop `scripts` from `root_packages` and delete the
  `H-R19` worklist.

### F-5. Environment identity in the key: three companion documents and the pydra proposal still put it there, against ADR-002

- **Departs from:**
  - ADR-002 (`adr/ADR-002-identity-v2.md:54,95`): the env lock is execution context, and "Environment
    identity in the key (R-40)" is a rejected alternative;
  - I-2 (`design.md:417`);
  - P-10 (laptop and cluster plans share a `LogicalPlan` hash).
- **What the documents say instead:**
  - `engine-choice.md:22`, G3: "caller-controlled keys (model sha, env lock sha, params)";
  - `pydra-extension-points.md:197-199` (the key policy includes "env lock"), `:213` and `:287` ("key policy
    via `__bytes_repr__` and the environment checksum");
  - `workflow-engines.md:194`: "caller-defined identity incl. commit and env lock";
  - proposal 0001 §3.5 step 1 (`proposals/0001-execution-extensions.md:142`): "Add `Environment.checksum()`,
    folded into the job checksum". §3.11 (`:205-206`) then records the same environment checksum "outside the
    checksum". The proposal contradicts itself.

  If senselab's key is pydra's checksum (ADR-001) and G1 folds the environment in, then every env rebuild
  and every laptop-to-cluster move changes keys. That defeats the `equivalent` default.
- **Kind:** drift. ADR-002 came after these documents, and they were not updated.
- **Correction:** update G3 and the three document passages to "keys exclude environment; context is
  recorded". In the proposal, make G1 an identity that is recorded and offered to the reuse policy, not
  folded into the checksum by default, or make folding opt-in so that K1 is coherent.

### F-6. ADR-001's "pydra's checksum equals senselab's key" has no proposal item that delivers it

- **Departs from:** ADR-001 (`adr/ADR-001-graph-format.md:44-46`): "gives each node its ADR-002 logical key.
  pydra's checksum then equals senselab's key, and pydra never hashes senselab function code". Also
  `design.md:490-491`.
- **What the companion offers instead:** K1 (`proposals/0001-execution-extensions.md:196-211`) offers a
  declared `version=` in place of code, plus type byte representations. pydra still computes its own
  checksum over (version, input bytes), so the result is not an externally supplied key.
  - D1's `ResultStore.lookup(checksum)` (`:101-106`) has no context or policy argument.
  - The reuse policy is `Submitter(reuse="exact"|"any")` with "richer policies" left undefined.
    senselab's `equivalent` therefore has no hook to plug into.
  - senselab's citation tables predate K1 and list "A0…J2" only (`design.md:500`, `engine-choice.md:8`,
    the table at `execution.md:153-163`). K1 is not among the Phase 3 gate prerequisites (E1, D1, F2) either.
- **Kind:** drift between the companion and the design.
- **Correction:** add to the proposal an explicit "node-supplied checksum" hook, or state in ADR-001 that
  the pydra checksum is derived from the logical-key digest via `__bytes_repr__` of a single key object.
  Add a reuse-policy protocol (stored context, current context, backend → bool) to K1, and pass a context
  to `ResultStore.lookup`. Add K1 to the citation tables and to the Phase 3 gate list.

### F-7. `runtime/retry.py` retries permanent errors, keeps rationale prose in code, and reads an ambient knob

- **Departs from:**
  - P-32 and P-34 (robustness);
  - CLAUDE.md "Rationale does not go in code";
  - P-37 (one implementation per behaviour);
  - I-1 (no ambient reads, `design.md:416`).
- **What the branch does instead:**
  - `OSError` is in `_TRANSIENT_EXCEPTIONS` (`retry.py:13-17`), and any member with no HTTP status
    retries (`:93-96`). A `FileNotFoundError`, `PermissionError` or `IsADirectoryError` raised inside a
    retried call is therefore retried with 1 s and 2 s backoff before it surfaces.
  - The comments at `:36-42` and `:96-100` are review history ("Without this, a dropped connection ... was
    classified non-transient").
  - `requests` is imported optionally (`:26-33`) although it is not a dependency of core.
  - `SENSELAB_HF_MAX_RETRIES` is read from the environment (`:127`).
  - `storage.storage_fault` (`runtime/storage.py:1-4`, "the one mapping from an `OSError` to a retryable
    `StoreUnavailable`") is a second transient classifier.
- **Kind:** deviation (a module carried over from 1.x, not brought to the 2.x standard).
- **Correction:** classify transient errors only for network types (`httpx.TransportError`,
  `ConnectionError`, `TimeoutError`, `urllib.error.URLError`) and for status 429 or ≥500. Delete the
  `requests` branch and the prose. Pass the retry count from the execution profile, or as a parameter.

## Minor

### F-8. Configuration through environment variables in the core runtime, and one environment write

- **Departs from:** I-1 (`design.md:416`), the execution profile owning roots and caps (`design.md:349`),
  and "no side effects" (`design.md:511`).
- **What the branch does instead:**
  - `SENSELAB_CACHE` (`runtime/revision.py:92`);
  - `SENSELAB_DATA_CACHE` (`runtime/fetch.py:57,210`);
  - `SENSELAB_HF_MAX_RETRIES` (`retry.py:127`);
  - `SENSELAB_TORCH_INDEX_URL` (`hardware.py:16,179`);
  - `revision.py:86`, which writes `os.environ["SENSELAB_RUN_ID"]`, a process-global mutation from a library call;
  - `senselab_core/__init__.py:38`, which runs a metadata check at import.
- **Correction:** pass these as explicit arguments now, so the execution profile can supply them in
  Phase 2. Return the run id rather than exporting it. Move the version check into the worker handshake or the CLI.

### F-9. The core runtime carries code that only frozen 1.x uses

- `hardware.py:30-237` (`_PYTORCH_INDEX_MAP`, `detect_host_cuda`, `pick_torch_index` and
  `SenselabCudaCompatibilityError`) has no caller in `packages/`. Its only callers are
  `src/senselab/utils/subprocess_venv.py` and 1.x tests. ADR-009 says 1.x is not fixed, so this is dead
  weight in core until Phase 2 env building is designed. Its docstring and comments (`:1-16`, `:30-35`,
  `:208`) are 1.x rationale prose, and it uses `Optional[...]` throughout.
- **Correction:** remove it from core. Re-add the index picker when the env builder is ported (execution.md §5).

### F-10. Small duplicates that break P-37 ("there should not be multiple functions that do the same thing", owner item 18)

- `senselab_worker/codec.py:133-151` (`_refuse_constant`, `_finite_float`, `_unique_object`) re-implement
  `canonical.py:376-393` (`_parse_float`, `_refuse_constant`, `_unique_object`).
- `codec.py:160-185` (`_height`) is a second depth and cycle walker beside `canonical.Walk`/`entered`.
  The design says "Cycles, depth and work go through one `canonical.Walk`" (`design.md:469`).
- There are two content-addressed commit protocols: `runtime/store.py:365-417` (stage a directory, then
  rename) and `runtime/fetch.py:323-375` (hard-link first, with keep-or-replace). Each also has its own
  `AddressedFormat` and staging.
- `kernel/declare.py:46-56` refuses unknown keys by hand on a model that already has `extra="forbid"`.
  The suggestion could come from one error formatter.
- `runtime/storage.py:701-705` imports `store` and `fetch` lazily through a module `__getattr__`, so the
  lower module reaches upward.

### F-11. The reuse-policy plumbing does not match its declared home

- `EQUIVALENCE` is a code literal: "a class is added by editing this declaration"
  (`kernel/identity.py:269-273`). The design and CLAUDE.md put measured values in `data/` with a
  derivation, and I-4 has verify sampling feed the classes (`design.md:419`, ADR-002:73).
  `DEFAULT_REUSE_POLICY` is referenced only by a test. `ResultStore.get` requires a policy and has no
  default (`store.py:817`).
- **Correction:** load the classes from a packaged data file with a derivation. Have the run-config
  resolver (Phase 2) default to `DEFAULT_REUSE_POLICY`, or give `get` that default now.

### F-12. Stale design statements against what was built

- `design.md:591-592` "Two implementations of weight hashing ... cross-implementation golden test". The
  build has one: core uses `senselab_worker.hashing.weights_identity` (`runtime/models.py:12`). The build
  is the better outcome, and the text should say so.
- The store layout is `store@4/<d[:2]>/<d>/<fingerprint>/` (`store.py:5`), against `schema-<N>/<k[:2]>/<k>/`
  (`design.md:427`). The per-context subdirectory is a reasonable consequence of the reuse policy, and the
  design should record it.
- `engine-choice.md:32` G13 says "the graph layer must sit inside core", but core's dependency list
  (`design.md:684`) has no pydra. ADR-001 places the pydra executor in the runtime. Phase 3 should say
  whether pydra becomes a core dependency or an extra.
- The `senselab` console script maps to `senselab_core.cli:main` (`pyproject.toml:38`) with only `data
  fetch`. Design §4.10 places the CLI on the root facade.

### F-13. The pydra benchmarks are copied in two repositories

- `spec/bench/pydra_overhead.py` and `pydra_partial_failure.py` duplicate
  `benchmarks/proposal_0001/*.py` on the pydra branch. The two copies have already diverged (flake8
  reformatting at `c697d8f9`).
- **Correction:** keep the pydra copy and cite it by commit.

## Code-quality sample

Thirteen modules were read or skimmed. Docstring share comes from an AST count.

| Module | Lines | Docstring share | Verdict |
|---|---|---|---|
| `kernel/_base.py` | 525 | 23% | compact and vectorised (`require_finite:432-453` uses a numpy mask and a bit view). The sealing work (`:290-412`) guards against interpreter bypasses that P-35 does not ask for (F-1) |
| `kernel/identity.py` | 296 | 20% | clean; `acceptable` is a `match` with `assert_never`. F-11 |
| `kernel/declare.py` | 265 | 23% | clean; F-10 |
| `kernel/signals.py` | 542 | 20% | I/O in the kernel and a full-file hash per read (F-3) |
| `senselab_worker/canonical.py` | 687 | 23% | carries a vendored Unicode assigned table to decide text portability (overbuilt for Phase 1) |
| `senselab_worker/codec.py` | 541 | 16% | duplicates canonical's helpers (F-10) |
| `runtime/store.py` | 887 | 21% | the core logic is fine. About 190 lines (`:524-716`) are quarantine, set-aside, restore and reseal machinery. Each `get` reads every context's `context.json` before choosing a candidate (`:858-876`) |
| `runtime/storage.py` | 737 | 28% | a dir-fd sandbox with grammar-checked names, not designed (F-1) |
| `runtime/worker_client.py` | 1,004 | 13% | 53 functions; process-group polling (`_live_members:255`, `_members_gone:277`), signal deferral and four grace constants (`:50-56`). The largest single host-side surface, for a one-request protocol |
| `runtime/provenance.py` | 1,163 | **32%** | the largest and most docstring-heavy module. It wraps `worker_digest.file_digest` again (`:70-90`) and returns error strings instead of raising typed errors |
| `runtime/fetch.py` | 777 | 16% | sound; second commit protocol (F-10) |
| `runtime/retry.py` | 144 | 29% | **worst offender**: F-7 |
| `runtime/hardware.py` | 430 | 22% | **second worst**: 1.x prose and dead 1.x API (F-9) |

**Worst offenders:**
1. `runtime/retry.py:13-17,36-42,93-100,127`;
2. `runtime/hardware.py:1-16,30-35,161-237`;
3. `runtime/worker_client.py:186-338,788-930`, where supervision code outweighs the protocol;
4. `runtime/provenance.py` as a whole: 1,163 lines and 32% docstrings for a writer the design describes in two paragraphs;
5. `kernel/signals.py:179-216` (F-3).

Outside the modules carried over from 1.x, rationale prose in code is rare. Most docstrings state behaviour.
The verbosity problem is scale: every guarantee is stated in full, in code and again in the design.
