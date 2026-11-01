# senselab 2.x Phase 2 Implementation Plan: tasks, backends, W3C PROV, benchmarks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 1 kernel into something a person or an agent can call:
- declared tasks and backends with `Params`, generated functions and JSON Schemas, and `describe`;
- the runtime catalogue, a minimal execution-profile resolver, `Env.host`/`Env.isolated`/`Env.server`;
- one worker protocol for real backends and one model cache;
- provenance as W3C PROV (ADR-010) with export redaction (ADR-007);
- the first ported backends (Whisper, CrisperWhisper, pyannote, speechbrain, PII review, one DSP primitive);
- the benchmark harness with its manifests, smoke suite and verify sampling;
- cross-context tolerance profiles, and parity against 1.x within them, on CPU and on one CUDA host;
- the pydra gate prerequisites D1, E1, F1, F2, G1, K1, K2 working on the pydra branch.

**Architecture:**
- `senselab_worker` (stdlib + numpy) gains the model cache, so host and isolated backends share one
  implementation (P-37).
- `senselab_core.kernel` gains the decorators, `Params`, `backend_params`, the JSON Schema of a declaration, the
  kernel types later tasks consume (`RegularSeries`, `Text`, `Transcript`, `TextSpans`, `Trials`, `Metric`), clock
  re-binding and the PROV vocabulary types. It still does no I/O (ADR-009).
- `senselab_core.runtime` gains the key builder, the PROV-O store, the catalogue, the profile resolver, the env
  builder, `VllmServer`, and `run_call`, the single-call path that Phase 3's `InProcess` executor replaces.
- The 2.x `senselab` package (facade, CLI, task declarations, backends, metric tasks, catalogue data, env locks)
  lives at `packages/senselab/src/senselab/` and is built by the root `senselab` distribution (ruling P2-1). 1.x
  modules reach 2.x only by a rename-only `git mv` commit followed by a modify commit (ADR-009, D-6, ruling P2-2 as
  amended by P2-35).

**Tech stack:** as Phase 1, plus `pyld` in the core test group (JSON-LD expansion in contract tests), the `torch`
extra of the root distribution for the one host backend, and per-backend uv locks for isolated envs.

**Spec:** `design.md` §4.2–§4.4, §4.8, §4.10, §4.12, §4.13, §6.1, §6.2 (Phase 2); ADR-001, 002, 003, 005, 006, 007,
009, 010 (as amended); `execution.md`; `testing.md`; `docs.md`; `wire-protocol.md`; `benchmarks.md`;
`migration-ledger.md`; `review-register.md` rows owned by Phase 2; `rulings.md` H-R49..H-R59. Pre-flight findings
(a-*, b-*, c-*, d-*, Table 1 #n, Table 2 rows) are resolved by rulings P2-33..P2-86.

**Branches.** Code: `2x/phase-2` → `2.x` (worktree `~/software/sensein/senselab-2x-impl`). Spec changes, ADR
amendments, register rows and `STATE.md`: `2x/phase-2-spec` → `2.x` (worktree `~/software/sensein/senselab-2x`;
D-7). pydra: `~/software/nipype/pydra-ext`, per-PR branches merged into `enh/execution-extensions` (ruling P2-31).

## Global Constraints

- **No backwards compatibility** (P-14): no shims, aliases, re-exports or readers of retired formats. A retired
  format is refused with a typed error that names the migration statement. Before the first 2.x release a major
  format bump has no previous-major reader (ADR-006 amendment, ruling P2-44).
- **One implementation per behaviour** (P-37): a variant is a parameter, strategy or backend over one engine. A
  ported module that duplicates a core engine (locks, hashing, worker protocol, model cache, config load, word
  confidence, SPDX parsing, retry policy) is folded into that engine in its modify commit, never kept beside it.
- **Rationale lives in specs** (P-21, P-31): docstrings say what and how; measurements, history and rejected
  alternatives go to `specs/`. No register, round or ruling id outside `specs/`.
- **Order of work in every task** (D-1 as amended, owner item 39, ruling P2-34): implement and run the local tests
  → review and compact locally (the named reviewer model, never the implementer's model and context, P-40, against
  `principles.md` and the `engineering.md` checklist; duplicates folded, verbosity cut) → commit and push; GitHub CI
  green → ORCD run at that pushed SHA. An ORCD failure gets a fix commit that is itself reviewed and compacted
  before it is committed, then a scoped ORCD rerun at the new SHA. ORCD evidence is cited by SHA (`<sha> <job id>
  <tier summary line>`) in the SDD ledger and, at phase exit, in `STATE.md`. Nothing is compacted after evidence
  without rerunning. A data file generated from an ORCD run (tolerance profiles, env-grouping proposal) is its own
  reviewed commit and needs no rerun, because no code changed.
- **GitHub CI** (ruling P2-33, superseding P2-28): logic, contract, and limits that need no model and no subprocess
  venv. No model weights load and no backend env is built in GitHub CI; no model runs there for review either.
- **ORCD** (ruling P2-33): every model-backed, venv-backed or GPU tier and every use case, submitted by an agent from
  a pinned SHA: CPU on partition `pi_satra`, node `node3805`; GPU on typed H100 (`--gres=gpu:h100:1`); never
  `node2803`; never a login or remote-control node. Each orcd step uses the job shape below.
- **Local machine** (owner items 27, 28, ruling P2-85): logic and contract tests only, in the existing `.venv`, run
  with `uv run --no-sync`, scoped to the changed files, serial. **No `uv sync` locally**, no new venvs, no model
  downloads, no subprocess-venv builds, no full suites. A task that adds a package path or a dependency tests locally
  with that path's `src` on `PYTHONPATH` and the existing `.venv`, and leaves any test that needs the new dependency
  to CI, where the locked install proves it. Lock resolution without install (`uv lock`, `uv pip compile`) is
  allowed. Watch load, memory and disk.
- **Weights pinned by commit** (P-4, P-16): every test or benchmark path that loads weights reads its model from a
  catalogue entry whose revision is a 40-hex commit or a 64-hex weights digest, and loads by that identity
  (resolve, then load with `revision=<sha>`). The `catalogue@1` schema refuses a ref.
- **Tiers** (`testing.md`, H-R49): every test carries exactly one of `logic` (default), `contract`, `limits`,
  `env`, `usecase`; `gpu`, `network`, `slow`, `model` are sub-markers. A test that loads weights or needs a backend
  env carries `model` (ruling P2-33). PRs run `logic or contract`; the nightly adds `limits and not model`;
  everything carrying `model`, every `env` build of a backend lock and every `usecase` runs on ORCD.
- **Generated contract cases** (ruling P2-62): run each backend's worker (`load` then `handle`) against the
  backend's declared load stub (the fake stack's model, no weights); they are `contract` and local. The same cases
  with real weights are the `env` tier (`model`) on ORCD.
- **Moves** (ADR-009, D-6, ruling P2-35 amending P2-2): merge `origin/alpha` and advance the area's ledger mark first
  (P-39). A rename-only commit moves whole files with `git mv` and nothing else, to
  `packages/senselab/src/senselab/<1.x relative path>` by default, or to
  `packages/senselab-core/src/senselab_core/runtime/` (1.x basename kept) when the file is runtime infrastructure core
  needs. Merging files and renaming identifiers, modules or directories happen in the following modify commit. A file
  is moved by `git mv` at most once in Phase 2, and no later task renames it. A file only partly ported moves whole;
  its unported functions are removed in the modify commit and listed in the ledger with their `alpha` source (ruling
  P2-3). The 1.x dispatch layer (`api.py`) never moves (ruling P2-36).
- **Identity:** ADR-002 key and context fields exactly; nothing hashes function code, Python objects or
  environments. Every new persisted artifact carries a `format` tag registered per ADR-006.
- **Commits** (P-27): one commit per unit of work, pushed promptly; each task's last code commit has a green 2.x CI
  and records the implementer and reviewer models in its message.

### ORCD job shape (every `orcd` step)

- **Checkout:** a fresh clone from GitHub per campaign at the pushed SHA under test
  (`$SCRATCH/senselab-2x/<sha>/`), never another scratch checkout, never reset while jobs run from it. `git log
  origin/2x/phase-2..HEAD` is empty before `sbatch`. The SHA is written into the job name and the job log.
- **CPU:** `#SBATCH -p pi_satra -w node3805 -c 8 --mem=32G -t 02:00:00`.
- **GPU:** `#SBATCH --gres=gpu:h100:1 -c 8 --mem=64G -t 04:00:00 --exclude=node2803`, on the partition the
  `orcd-remote` skill reports for typed H100 under this account.
- **Environment:** `PATH="$HOME/.local/bin:$PATH"`; `LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"`;
  `HF_HOME=/orcd/data/satra/002/huggingface`; `HF_TOKEN` from `~/.cache/huggingface/token`;
  `SENSELAB_RUN_ID=slurm-$SLURM_JOB_ID`; `SENSELAB_CACHE=$SCRATCH/senselab-2x-cache`;
  `UV_CACHE_DIR=$SCRATCH/uv-cache`; execution profile `orcd` (env roots under `$SCRATCH/senselab-2x-venvs`).
- **Steps:** `uv sync --locked --all-extras --group test` as its own step, then `uv run --no-sync pytest <scope> -m
  "<tier>"`, serial. Check the next maintenance reservation before choosing `-t`.
- **Who waits:** an agent writes and submits the job; the main thread holds the wait. `COMPLETED` is not evidence:
  check elapsed time and grep the log for the tier's summary line, then record `<sha> <job id> <summary>`.

## Review Focus

1. **PROV identity.** An activity's `@id` is the content address of its execution record, never the logical key; two
   contexts for one key give two activities; a reuse is its own activity; one execution activity per call. Pinned in
   Task 3b (`test_one_key_two_contexts_two_activities`) and Task 10b (`test_one_activity_per_call`).
2. **No path leaves a run.** No provenance document, export, error or log line carries a file name or path; the
   label map stays in the private run directory. Pinned in Task 4 (`test_document_holds_no_path`).
3. **Params, backend params and backend version reach the key exactly once.** Output-affecting batch formation and
   chunk grid enter the key only when declared; execution-context and placement fields never do. Pinned in Task 5b
   (`key_ignores_<field>` through `build_key`, the builder `run_call` uses).
4. **Stack switch by profile alone.** Changing `stack_preferences` changes the backend and the recorded context, never
   the task call or the output schema. Pinned in Task 10b (`test_stack_switch_by_profile`) and Task 32.
5. **Pinned loads.** No backend loads through a ref; every load passes the catalogue's SHA. Pinned in Task 7
   (`test_catalogue_refuses_ref_revision`) and each port's ORCD `env` run.
6. **One worker protocol, total errors.** A real backend's per-item failure is a typed fault with `gap: false`;
   zero gaps over its fault fixtures (ADR-009 amendment). Pinned per port (`content_fault_isolated` on the declared
   stub locally, with real weights on ORCD, R-57).
7. **Tolerance is never fitted to what it judges.** Profiles come from 2.x cross-context agreement on the calibration
   split; parity and cross-stack checks run on the held-out split; a failure is a finding, never a refit. Pinned in
   Tasks 22a and 22b.

## File Structure

```
pyproject.toml                                     # root `senselab`: wheel packages = packages/senselab/src/senselab; torch extra; senselab = senselab.cli:main
.importlinter                                      # senselab above senselab_core.runtime; core and worker never import senselab
packages/senselab/
  src/senselab/
    __init__.py                                    # facade: tasks, describe, models, generated classes; lazy sl.audio, sl.text, sl.signal
    cli.py                                         # `senselab` console script: core's `data fetch`, `bench run`
    _api/                                          # generated task classes and functions (committed; drift-checked)
    declarations.py                                # @task declarations of the Phase 2 tasks
    audio/tasks/speech_to_text/{huggingface,token_confidence,crisperwhisper}.py   # moved from src/, then ported
    audio/tasks/speaker_diarization/pyannote.py    # moved, ported
    audio/tasks/speaker_embeddings/{speechbrain,windowing}.py    # moved, ported
    text/tasks/pii_detection/{redaction_review,redaction_policy,redaction_review_vllm}.py, data/   # moved, ported
    signal/resample.py                             # DSP primitive (new module; backend and worker in one)
    metrics/{wer,der,eer,span_f1,resample_error}.py   # metric tasks
    data/catalogue.yaml                            # catalogue@1: every Phase 2 model, pinned by commit
    data/env_locks/<env>.txt                       # Phase 2 locks: moved (crisperwhisper, vllm-server, pii-review) or generated (pyannote, speechbrain)
    data/prov/senselab-audio.jsonld                # vocabulary terms the ported tasks add
    data/tolerance/<backend>.yaml                  # tolerance@1, generated from cross-context runs, with derivations
    data/env_grouping.yaml                         # env-grouping proposal (recorded, not adopted)
    data/source_digests.yaml                       # written at the first release
  tests/                                           # incl. parity/
packages/senselab-core/src/senselab_core/
  kernel/declare.py                                # @task/@backend(name, version, ...), TaskParams, BackendParams, Env kinds, json_schema()
  kernel/signals.py                                # + RegularSeries protocol, Text, chunked SourceStamp
  kernel/results.py                                # + Transcript, TextSpans, Trials, Metric
  kernel/time.py                                   # rebind(output, stored_anchor, reader_anchor, timemap)
  kernel/prov.py                                   # PROV record and literal types, vocabulary model (no I/O)
  runtime/keys.py                                  # build_key(call, placement, context): the one key builder
  runtime/provenance.py                            # ProvStore: PROV-O JSON-LD writer/reader (provenance@3.0)
  runtime/bep028.py                                # BEP028 profile check and file layout (new; no translation)
  runtime/data/prov/{context,senselab}.jsonld      # published @context and core vocabulary
  runtime/catalogue.py                             # Catalogue, CatalogueEntry (loader takes YAML text or path)
  runtime/licenses.py                              # SPDX parser (moved out of scripts/check_licenses.py), vendored table, derived flags
  runtime/profiles.py                              # ExecutionProfile, resolve_execution()
  runtime/data/profiles/{laptop,orcd}.yaml         # execution-profile@1 (retry policy folded in)
  runtime/envs.py                                  # from moved subprocess_venv + uv_tools + venv_lock + cuda_probe
  runtime/vllm_server.py                           # moved; VllmServer and its HTTP client helpers
  runtime/call.py                                  # run_call(): resolve, place, env, key, store, provenance
packages/senselab-worker/src/senselab_worker/
  model_cache.py                                   # ModelCache: LRU by (backend, model identity, device, dtype) under a budget
benchmarks/
  harness.py, manifests/<dataset>.yaml, suites/smoke.yaml   # outside pytest; metric tasks live in the package
scripts/check_source_digests.py, scripts/generate_task_api.py, scripts/env_grouping.py,
scripts/generate_tolerance.py, scripts/regenerate_equivalence.py
```

**Execution order** (dependencies are authoritative; numbering is kept stable by suffixes):
0 → 1 → 2a, 2b → 3a → 3b → 3c → 4 → 5a → 5b → 6 → 7 → 8 → 9a1 → 9a2 → 9b → 10a → 10b → 11a, 11b → 12 → 13 → 14,
15, 16 → 17a → 17b → 18, 19 → 20a → 20b → 20c → 22a → 21 → 22b → 23a → 23b → 32 → 33 → 34. The pydra tasks (24–31b)
run in parallel with everything from Task 1 on.

---

### Task 0: P-39 reconciliation at Phase 2 start (complete)

**Model:** sonnet implements (effort lower, worktree); sonnet reviews (ruling P2-45). **Tier:** none.
**Branches:** merge on `2x/phase-2` (code worktree); ledger on `2x/phase-2-spec`. **Done:** `alpha` 22d2c103 was
already in `2x/phase-2` (no merge); marks set; docs b0f91256.

- [x] **Step 1 (local):** `git fetch origin alpha && git log <mark>..origin/alpha --oneline -- src scripts tutorials`
  for each ledger row.
- [x] **Step 2:** In the code worktree, merge `origin/alpha` into `2x/phase-2` (no edit under `src/`), push. Set each
  mark to the merged `alpha` commit; for each commit touching an area Phase 2 ports, add a row naming the port task
  that carries it.
- [x] **Step 3:** Commit `docs(2x): P-39 reconciliation at Phase 2 start` on `2x/phase-2-spec`; push.

### Task 1: The 2.x `senselab` package, its CLI and its import contracts

**Model:** sonnet implements; opus reviews (layering). **Tier:** logic, contract. **Implements:** design §5, ADR-009,
P-13, rulings P2-1, P2-52, P2-71, P2-73. **Depends on:** Task 0.

**Files:** Create `packages/senselab/src/senselab/{__init__,cli}.py`, `packages/senselab/tests/`. Modify
`pyproject.toml` (wheel `packages`, `dev-mode-dirs`, an empty `[project.optional-dependencies] torch`, `[project.scripts]
senselab = "senselab.cli:main"`), `uv.lock`, `.importlinter`, `.github/workflows/2x-tests.yaml`, `packages/conftest.py`.

**Interfaces:**
- `import senselab as sl` resolves to `packages/senselab/src/senselab`; `src/` is on no 2.x path.
- `senselab.cli:main` is the console script; it mounts core's `data fetch` (core's `senselab_core.cli` keeps the
  implementation) and is where later tasks add subcommands. Core never imports `senselab`.
- `.importlinter`: `packages-not-1x` restated as "`senselab_core` and `senselab_worker` never import `senselab`";
  the `layers` contract gains `senselab` above `senselab_core.cli | senselab_core.testing`. Lazy framework imports are
  checked by the subprocess test only, not by import-linter (grimp counts function-level imports).
- CI: `core-only` runs `packages/senselab-core packages/senselab-worker`; a new `senselab-lean` job (3.11, 3.14)
  installs the root distribution without extras from `uv.lock` and runs `packages/senselab/tests -m "logic or
  contract"`.

- [ ] **Step 1: Failing tests (local).** `test_senselab_resolves_to_packages` (`senselab.__file__` under
  `packages/senselab/`); `test_senselab_imports_no_framework` (subprocess import of `senselab` and of every backend
  module registered so far loads no torch, transformers, speechbrain, pyannote, ctranslate2, vllm; rerun unchanged
  by every port task); `test_cli_mounts_data_fetch`. `PYTHONPATH=packages/senselab/src uv run --no-sync pytest
  packages/senselab/tests -m "logic or contract"` fails.
- [ ] **Step 2: Implement**; `uv lock` (resolution only); rerun the Step 1 command; `uv run --no-sync lint-imports`.
- [ ] **Step 3:** Review and compact, commit `feat(2x): the 2.x senselab package and its contracts`; push; CI green
  (the `senselab-lean` job proves the locked install).

### Task 2a: Phase 1 hardening carried forward: worker and wire

**Model:** opus implements (worker protocol); fresh sonnet reviews. **Tier:** logic, contract. **Implements:**
R-146, R-147, R-158 (wire and worker groups), R-159, R-123; ruling P2-55. **Depends on:** Task 1.

**Files:** `senselab_worker/{faults,codec,serve}.py`, `senselab_core/runtime/{faults,worker_client}.py`, tests; the
shared bounded helper goes in `packages/senselab-core/tests/runtime/bounded_probe_fixture.py` (beside `run_bounded`),
replacing the copies in `shutdown_schedule_fixture.py` and `worker_shutdown_schedule_test.py`; the worker test copy in
`senselab-worker/tests/launch_test.py` becomes one worker test-local helper (worker-floor installs no core).

**Interfaces:** `is_transient` walks a bounded cause chain and classifies SSL errors by type; an array reference
accepts exactly `{path, sha256, dtype, shape}` with integer shape elements; `ShutdownGraces` refuses overflow with
`ValueError`; `_has_exited` folded into the single exit wait; `max_line_bytes` bounds responses too; the scripted
fixture raises a typed error when replies run out.

- [ ] **Step 1: Failing tests (local)**, one per item: `test_cause_cycle_terminates`,
  `test_ssl_error_classified_by_type`, `test_array_ref_schema_is_strict`, `test_graces_overflow_is_value_error`,
  `test_exit_observed_by_one_wait`, `test_response_line_bounded`, `test_scripted_fixture_exhausted_raises_typed`.
- [ ] **Step 2: Implement**; fold the eight wire duplicate groups of R-158 or record in the register why one stays;
  move the rationale prose of R-159 to `wire-protocol.md` (spec commit).
- [ ] **Step 3 (local):** `uv run --no-sync pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `fix(2x): worker and wire residuals`; push; CI green.

### Task 2b: Phase 1 hardening carried forward: store, leases, fetch, retry

**Model:** opus implements (store and leases); fresh sonnet reviews. **Tier:** logic, contract. **Implements:**
R-148, R-151, R-152, R-153, R-125, R-127, R-158 (store and kernel groups); rulings P2-50, P2-60. **Depends on:** Task 1.

**Files:** `runtime/{store,locks,fetch,retry}.py`, `runtime/data/retry_policy.yaml` (read, unchanged), tests.

**Interfaces:** run ids over `NAME_MAX` refused at validation; `collect` covers run-directory, root-level and
in-place set-aside objects; a displaced lease re-reads and merges the manifest under the lease; the stat fast path
compares ctime; a re-`put` verifies by recorded digests; fetch removes superseded version links and retries by
`faults.is_transient` under the packaged `retry_policy.yaml` (Task 8 folds that file into the profile). The
`equivalence.yaml` checks (R-155) move to Task 21 (ruling P2-60).

- [ ] **Step 1: Failing tests (local)**, one per item: `test_run_id_over_name_max_refused`,
  `test_collect_covers_every_set_aside_location`, `test_displaced_lease_keeps_manifest_update` (planted delay at the
  `os` boundary, `syscall_faults_fixture`), `test_ctime_change_defeats_stat_fast_path`,
  `test_reput_verifies_recorded_digests`, `test_superseded_version_links_removed`,
  `test_fetch_retries_transient_by_packaged_policy`.
- [ ] **Step 2: Implement.** ADR-003 gains no text; the group-member limit and the by-path reads are stated in the
  register row's resolution.
- [ ] **Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{store,locks,fetch,retry}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `fix(2x): store, lease and fetch residuals`; push; CI green.

### Task 3a: PROV record types and the `senselab:` vocabulary (ADR-010)

**Model:** opus implements (identity and provenance); fresh opus reviews. **Tier:** logic, contract.
**Implements:** ADR-010 as amended, R-126, R-150; rulings P2-5, P2-7, P2-40, P2-58, P2-74. **Depends on:** Task 2b.

**Files:** Create `kernel/prov.py`, `runtime/data/prov/{context,senselab}.jsonld`. Modify `kernel/formats.py`
(`senselab-vocab@1.0`, `prov-activity@1`), core test group in `packages/senselab-core/pyproject.toml` (`pyld`),
`uv.lock`.

**Interfaces:**
- Records are `prov:Entity`, `prov:Activity`, `prov:Agent` and subtypes (`prov:Plan`, `prov:Collection`,
  `prov:SoftwareAgent`); core terms only: `senselab:Run`, `senselab:SoftwareAgent`, `senselab:Environment`,
  `senselab:File`, `senselab:Plan`, and `senselab:modelSource` on the Agent (R-110).
- Relations are PROV relations with qualified forms (`prov:qualifiedUsage`, `prov:qualifiedGeneration`,
  `prov:qualifiedAssociation`) where a role or time attaches; the association role carries the node id; no
  `prov:hadPlan` until Phase 3.
- `@id`s: an entity's is its content address (`content:`/`output:`); an activity's is
  `identity_digest("prov-activity@1", record without @id)`, with the logical key as `senselab:logicalKey`
  (ruling P2-6); agents and environments are content-addressed over their recorded fields.
- Values are JSON-LD literals: JSON scalars as is; non-finite and over-range numbers as XSD typed literals
  (`{"@value": "-INF", "@type": "xsd:double"}`, big integers as `xsd:integer` strings, decimals as `xsd:decimal`);
  arrays as content-addressed `senselab:File` entities; sets as sorted `@set`; an enum as its `str` value; bytes,
  paths and Python objects refused with `UnrecordableValue` naming the attribute path (ruling P2-5).
- An attribute outside the loaded vocabulary, or outside its declared range, is refused at the boundary. Packages
  add terms by shipping a vocabulary file registered with their declarations; core holds no workflow type.
- The pyld contract test expands through an offline document loader that serves the packaged `@context`; no test
  reaches the network for it.

- [ ] **Step 1: Failing tests (local):** `test_unknown_term_refused`, `test_value_out_of_range_refused`,
  `test_minus_inf_round_trips_as_xsd_double`, `test_unrecordable_value_names_attribute`; contract
  `test_document_expands_with_pyld` (expanded terms resolve to `prov:` or the published vocabulary; no relative IRI
  survives; offline loader).
- [ ] **Step 2: Implement.** Hash JCS of the compact record, never an expansion (ADR-010). `uv lock` (resolution
  only).
- [ ] **Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel -m "logic or contract" -k "not
  pyld"` (pyld is not in the local venv; CI runs the contract from the locked install).
- [ ] **Step 4:** Review and compact, commit `feat(2x): PROV record types and the senselab vocabulary`; push; CI
  green. Spec commit: ADR-010 amendment (the vocabulary is checked by a contract test until the first 2.x release,
  then by the release gate; added to D-5's deferred list, ruling P2-40); ADR-006 format table rows.

### Task 3b: `ProvStore` writer and reader

**Model:** opus implements; fresh opus reviews. **Tier:** logic, contract. **Implements:** ADR-010, D-4, R-149;
rulings P2-4, P2-6. **Depends on:** Task 3a.

**Files:** Rewrite `runtime/provenance.py` (writer, reader, union, `merge`); tests.

**Interfaces:** each writer replaces `prov/<writer-uuid>.jsonld` atomically; a run's provenance is the union of its
writers' documents by `@id`, and two different nodes under one `@id` are refused (ruling P2-4). Accessors return in
`@id` order, so `merge` is order-independent (R-149).

- [ ] **Step 1: Failing tests (local):** `test_one_key_two_contexts_two_activities`, `test_reuse_is_an_activity`,
  `test_writers_union_by_id`, `test_conflicting_node_refused`, `test_merge_is_order_independent`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{provenance,provenance_checksum}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): provenance is PROV-O JSON-LD`; push; CI green.

### Task 3c: Store integration, token removal, `provenance@2` refusal

**Model:** opus implements; fresh sonnet reviews. **Tier:** logic, contract. **Implements:** ADR-010, ADR-006 as
amended; rulings P2-44, P2-57. **Depends on:** Task 3b.

**Files:** Modify `runtime/store.py`, `kernel/formats.py` (`provenance@3.0`), `kernel/_base.py` (`FrozenDict` and
the token helpers removed), `testing/provenance.py` (`revised` removed). Delete the old `runtime/bep028.py`
translation and its tests (Task 4 creates the profile check from scratch).

**Interfaces:**
- The store writes the result entity and any reuse activity; it never writes an execution activity (that is
  `run_call`'s, Task 10b; ruling P2-57).
- Removed: `$path`, `$enum`, `$dec`, `$bytes`, `$set`, `EnumMember`, `FrozenDict`, `revised`, `recorded`,
  `plain_values`, `PROV_TYPE`, `PATH_KEY`, `CHECKSUM_KEY`, `SIZE_KEY`, `MTIME_KEY`, the bare relation tuples,
  `provenance@2.0` and its JSONL reader (a `provenance@2` file is refused with `ProvenanceFormatRefused`).

- [ ] **Step 1: Failing tests (local):** `test_provenance2_refused`, `test_store_writes_entity_and_reuse_only`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{provenance,store}_test.py packages/senselab-core/tests/kernel -m "logic or contract" -k "not pyld"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): the store writes PROV; token families removed`; push; CI
  green. Spec commit: ADR-006 amendment (before the first 2.x release a major bump has no previous-major reader,
  H-R2; ruling P2-44); ADR-010 amendment log line for `provenance@3.0`.

### Task 4: Export redaction and the BEP028 profile

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** ADR-007 (amended), ADR-010
(BEP028 is a profile, not a translation), ruling P2-8. **Depends on:** Task 3c.

**Files:** Create `runtime/bep028.py` (profile check plus file layout); modify `runtime/provenance.py`,
`runtime/store.py`, `runtime/fetch.py` (file entities by label).

**Interfaces:** a file entity carries its content address and a per-run label `file-<n>`; the label→path map is
written to `<run>/private/labels@1.json` (mode 0600) and never exported; `export(run, dest)` copies the documents and
checks the BEP028 profile (`check_profile(doc) -> list[str]`).

- [ ] **Step 1: Failing tests (local):** `test_document_holds_no_path` (property: random path components never occur
  in any written document, error or log); `test_label_map_is_private`; `test_export_conforms_to_bep028`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{bep028,provenance}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): exports are redacted by construction`; push; CI green.

### Task 5a: Declarations: `@task`, `@backend`, `Params`, `Env`, JSON Schema

**Model:** opus implements (kernel identity); fresh opus reviews. **Tier:** logic, contract. **Implements:** design
§4.2, §4.4, ADR-001 (ports), ADR-002 §1; R-55, R-131; rulings P2-9, P2-47, P2-75. **Depends on:** Task 1.

**Files:** `kernel/declare.py`, `runtime/fake_backend.py` (`env="host"` → `Env.host()`, name and version declared, its
model is the load stub), tests.

**Interfaces:**
- `@task(name, version, kind)` over a class with `Params(TaskParams)`, `inputs`, `outputs` (port name → kernel
  type).
- `@backend(name, version, task, stack, models, env, devices, batch_invariant, tunables, output_affecting, worker,
  load_stub, params=BackendParams subclass)`: `name` and `version` are required and have no default (ruling P2-47);
  `load_stub` names the no-weights model state the contract cases load (ruling P2-62).
- `Env.host()`, `Env.isolated(name)`, `Env.server(name)` replace the free `env: str`.
- `BackendSpec.license` is removed: a backend's code license is derived from its env lock (Task 7, ruling P2-75).
- `json_schema(task)` emits JSON Schema 2020-12 (ruling P2-9) for inputs, params, backend params and outputs.

- [ ] **Step 1: Failing tests (local):** `test_unknown_param_suggests`, `test_backend_requires_name_and_version`,
  `test_backend_params_validated_per_backend`, `test_env_kinds_replace_free_string`, contract
  `test_schema_validates_2020_12` (every generated schema under the 2020-12 metaschema).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel -m "logic or contract" -k "not pyld"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): task and backend declarations`; push; CI green. Spec commit:
  pin the dialect in `wire-protocol.md`; close R-55, R-131.

### Task 5b: The key builder and generated contract cases

**Model:** opus implements (identity); fresh opus reviews. **Tier:** logic, contract. **Implements:** ADR-002 §1;
R-56, R-57, R-58, R-122; rulings P2-48, P2-62. **Depends on:** Task 5a.

**Files:** Create `runtime/keys.py`. Modify `kernel/declare.py` (`VERDICT_TYPES`), `testing/contracts.py`, tests.

**Interfaces:**
- `build_key(call, placement, context) -> LogicalKey` is the one key builder (ruling P2-48): it takes `backend`,
  `backend_version`, validated `backend_params` and the declared output-affecting components from the placement, and
  receives the execution context only to prove it ignores it. `key_for` becomes its kernel half; `run_call` (Task
  10b) calls `build_key`.
- `contract_cases` runs every backend that names a worker: `load(load_stub)` then `handle`, including
  `content_fault_isolated` (R-57); `FAULT_RUNNERS` keyed by stack and `NOT_YET_WIRED` are gone. A model-less host
  backend has a worker module too (Task 12).
- `VERDICT_TYPES` is derived from the kernel types marked as decisions, not a literal (R-58).

- [ ] **Step 1: Failing tests (local):** `test_output_affecting_enters_key_only_when_declared`,
  `test_backend_version_enters_key`, `test_key_ignores_context_through_builder` (two execution contexts, one key),
  `test_worker_backend_gets_contract_cases`, `test_contract_cases_load_the_declared_stub`, `test_verdict_types_derived`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel packages/senselab-core/tests/testing packages/senselab-core/tests/runtime/keys_test.py -m "logic or contract" -k "not pyld"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): one key builder and contract cases for every worker`; push;
  CI green.

### Task 6: Generated API and `describe` from declarations

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.2, §4.10, P-1;
rulings P2-10, P2-46, P2-63. **Depends on:** Task 5b.

**Files:** `scripts/generate_task_api.py`, `packages/senselab/src/senselab/_api/`, `senselab/__init__.py`.

**Interfaces:** declarations only (ruling P2-63); Task 10b wires execution and the catalogue.
- Per task: a configured immutable class (`SpeechToText(model=..., **params)`, `.with_params()`, `.call(x) ->
  TaskCall`, the validated call value) and the function `speech_to_text(x, **params)` building the same call;
  `__call__` is added in Task 10b; `.map` waits for Phase 3.
- Generated at development time into committed modules from the declarations (real signatures for IDEs and pdoc);
  a contract test regenerates and refuses drift.
- `sl.tasks(accepts=None)`; `sl.describe(name)` returns the task's JSON Schema with its backends, stacks and devices
  (models, licenses and default model are added in Task 10b). The facade exposes the generated classes, `tasks` and
  `describe`; `run_call` stays in `senselab_core.runtime.call` (ruling P2-46). Errors use one envelope `{code,
  message, fix, field, suggestions}`.

- [ ] **Step 1: Failing tests (local):** `test_generated_api_has_no_drift`, `test_function_and_class_build_same_call`
  (fake task), `test_describe_is_valid_schema`, `test_tasks_filters_by_accepted_type`, `test_error_envelope_shape`.
- [ ] **Step 2: Implement; Step 3 (local):** `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): generated task API and describe`; push; CI green.

### Task 7: Runtime catalogue and license records

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.3, §4.13, P-23,
R-88, R-116 (flags and closed `how`); rulings P2-11, P2-49, P2-53, P2-64, P2-75. **Depends on:** Task 5a.

**Files:** `runtime/catalogue.py`, `runtime/licenses.py` (the SPDX parser moved out of `scripts/check_licenses.py`,
plus a vendored SPDX table as data), `scripts/check_licenses.py` (imports the parser), `kernel/license.py`,
`packages/senselab/src/senselab/data/catalogue.yaml`; core tests over a toy catalogue; a contract in
`packages/senselab/tests` over the packaged one.

**Interfaces:**
- `catalogue@1` entry: `{id, revision (40-hex or 64-hex), tasks, backends, stacks, license, resources: {}}`;
  `Catalogue.load(text_or_path)`, `Catalogue.default(task)`, `Catalogue.entry(model_id)`, `Catalogue.sources()` (the
  hub organisations of its entries). `senselab` passes its own packaged resource; core never reads `senselab` data.
  Resource tables stay empty until the Phase 3 scale ladder.
- Every Phase 2 model entry is added here, pinned by commit (resolved by a metadata query, no weights):
  `openai/whisper-large-v3-turbo`, the CrisperWhisper model, the pyannote diarization pipeline and its segmentation
  model, the speechbrain ECAPA speaker model, the PII review model 1.x uses (ruling P2-64).
- One SPDX parser (`parse_expression`, `ids_of`, `allows`) in `runtime/licenses.py` (ruling P2-49). License flags are
  derived from the parsed expression against the vendored table; `verified.how` is one of `hf-card | pep639 |
  manual`; a free flag that contradicts the expression is refused.
- `backend_license(backend, lock_records)` derives a backend's code license from its env lock's package records (the
  root lock for `Env.host`); Task 18 supplies the records (ruling P2-75).

- [ ] **Step 1: Failing tests (local):** core `test_flags_derived_from_spdx`, `test_malformed_expression_refused`,
  `test_catalogue_refuses_ref_revision` (the one schema test for pins), `test_backend_license_from_lock_records`;
  senselab contract `test_packaged_catalogue_loads`; `check_licenses_test.py` unchanged and green.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/catalogue_test.py packages/senselab-core/tests/kernel/license_test.py packages/senselab-core/tests/lint/check_licenses_test.py -m "logic or contract"`; `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests -m contract -k catalogue`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): runtime catalogue and SPDX-derived licenses`; push; CI green.

### Task 8: Minimal execution-profile resolver

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.6 (execution
profile), §4.4 tunables, P-2, P-7, P-8, P-18, ADR-007 allowlist, R-127; rulings P2-13, P2-42, P2-50, P2-68, P2-76.
**Depends on:** Tasks 2b, 7.

**Files:** `runtime/profiles.py`, `runtime/data/profiles/{laptop,orcd}.yaml`; `runtime/retry.py` reads the profile;
delete `runtime/data/retry_policy.yaml`.

**Interfaces:** `execution-profile@1` with `stack_preferences: {task: [stack, ...]}`, `devices`, `env_roots`,
`envs.torch_index` (an optional override of the probed CUDA index, ruling P2-42), `model_sources` (default
`Catalogue.sources()`; ruling P2-68), `model_cache_budget_bytes`, `retry` (the former `retry_policy.yaml`),
`network`; packaged `laptop` default, an `orcd` site profile; user YAML deep-merged; the merged mapping's digest is
recorded in the execution context, never keyed. `resolve_execution(task, model_entry, profile, hardware, registry)
-> Placement` picks backend, stack, env and device, with `Placement.explain()`. A GPU-only backend with no GPU class
fails with the node named; no silent CPU fallback.

- [ ] **Step 1: Failing tests (local):** `test_stack_preference_picks_backend`, `test_profile_digest_not_in_key`,
  `test_gpu_only_without_gpu_fails`, `test_unknown_profile_key_suggests`, `test_unlisted_source_refused`,
  `test_retry_policy_read_from_profile`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{profiles,retry,fetch}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): minimal execution-profile resolver`; push; CI green.

### Task 9a1: Env builder (`Env.host`, `Env.isolated`) and CUDA routing

**Model:** opus implements (concurrency, leases); fresh sonnet reviews. **Tier:** logic, contract. **Implements:**
design §4.4, P-6, P-25, ADR-003 (D-2 leases), the subprocess-venv CUDA routing; rulings P2-35, P2-42, P2-70, P2-76.
**Depends on:** Task 8.

**Files:** Rename-only commit: `git mv src/senselab/utils/{subprocess_venv,uv_tools,venv_lock,cuda_probe}.py
packages/senselab-core/src/senselab_core/runtime/`. Modify commit: fold the four into `runtime/envs.py`; `file_lock`
is not moved (leases replace it).

**Interfaces:** `ensure_env(env, lock, root) -> EnvHandle{python, lock_sha, wheel_set, torch_index}`, `root` from the
profile's `env_roots`, builds under a D-2 lease (no `FileLock`, no heartbeat) and installs the worker runtime as an
overlay. The CUDA-aware torch index routing (`pick_torch_index` over `nvidia-smi`/`nvcc` probes) keeps its semantics;
the profile's `envs.torch_index` overrides it; `SENSELAB_TORCH_INDEX_URL` is removed (P-14, P-18). The chosen index and
the env are recorded in the execution context. Digest sites from the migration ledger's H-R20 table in these modules
use `identity_digest` with registered tags.

- [ ] **Step 1: Failing tests (local):** `test_concurrent_build_one_install` (two processes, planted delay at the `os`
  boundary, no real install: the installer is the `Resolver`-style seam), `test_cuda_index_routing` (table-driven over
  probe outputs), `test_profile_torch_index_overrides_probe`, `test_torch_index_recorded_in_context`.
- [ ] **Step 2: Implement** (rename-only commit, then modify). Ledger: mark the moved modules, list unported
  functions (ruling P2-3).
- [ ] **Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/envs_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): env builder under leases`; push; CI green.

### Task 9a2: Phase 2 env locks and the `Env.host` refusal

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract; env on ORCD. **Implements:** design §4.4, P-6;
rulings P2-35, P2-77. **Depends on:** Task 9a1.

**Files:** Rename-only commit: `git mv src/senselab/utils/data/venv_locks/{crisperwhisper,vllm-server,pii-redaction-review}.txt
packages/senselab/src/senselab/utils/data/venv_locks/`. Modify commit: the directory becomes
`packages/senselab/src/senselab/data/env_locks/` and `pii-redaction-review.txt` becomes `pii-review.txt`; headers
regenerated. The other 26 locks stay in `src/` until their port (ruling P2-77).

**Interfaces:** lock headers carry `identity_digest` with a registered tag in place of `requirements_sha256`, rewritten
by the moved lock tool (no resolution, no install); `Env.host()` is refused unless the root lock covers the
backend's requirements.

- [ ] **Step 1: Failing tests (local):** `test_host_refused_when_lock_lacks_backend`, `test_lock_header_is_tagged_digest`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/envs_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): Phase 2 env locks`; push; CI green.
- [ ] **Step 5 (orcd, CPU node3805, at that SHA):** `-m "env and model" -k envs` builds the `crisperwhisper` lock and
  round-trips every wire data type through its worker. Failures follow the P2-34 loop.

### Task 9b: `VllmServer` and `Env.server`

**Model:** opus implements (service lifecycle); fresh sonnet reviews. **Tier:** logic, contract; limits on ORCD GPU.
**Implements:** design §4.4, `execution.md` §3.5, I-8, P-37 (vLLM server row); rulings P2-14, P2-38, P2-78.
**Depends on:** Task 9a2.

**Files:** Rename-only commit: `git mv src/senselab/utils/vllm_server.py packages/senselab-core/src/senselab_core/runtime/`.
Modify commit: `VllmServer` and the HTTP client helpers (`chat`, `complete`, `content_part`) stay; `ensure_vllm_venv`
is replaced by `ensure_env` (ruling P2-78).

**Interfaces:** `VllmServer(env, model: ResolvedModel, settings)` as a context manager: starts in `Env.server("vllm")`
(lock `vllm-server`), loads by commit, health-checks, attests `{engine_version, model_identity, config_digest}` on
every response (I-8), stops with the owning process. Phase 2 has no cross-process service registry (ruling P2-14).

- [ ] **Step 1: Failing tests (local):** against a stdlib stand-in server process: `test_attestation_enters_context`,
  `test_restart_with_new_config_changes_attestation`, `test_server_stops_with_owner`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/vllm_server_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): VllmServer behind Env.server`; push; CI green. Spec commit:
  ADR-003 "staged" amendment (services are process-scoped in Phase 2; the leased registry and the `svc` pool worker of
  design §4.4 arrive with lease renewal in Phase 3; ruling P2-38).
- [ ] **Step 5 (orcd, GPU H100, at that SHA):** `-m "limits and gpu and model" -k vllm_server` with the catalogue's PII
  model (Task 7).

### Task 10a: Model cache and the host in-process path

**Model:** opus implements (worker protocol); fresh opus reviews. **Tier:** logic, contract. **Implements:** design
§4.4 (one worker protocol, one model cache), §4.12, ADR-009 (typed faults), P-5; rulings P2-15, P2-81.
**Depends on:** Tasks 2a, 5b, 8, 9a1.

**Files:** `senselab_worker/model_cache.py`, `senselab_worker/serve.py`, `runtime/worker_client.py`, tests.

**Interfaces:**
- `ModelCache(budget_bytes)`: `get_or_load((backend, model_identity, device, dtype), loader, size_of)`, LRU under the
  budget; one implementation used in-process (`Env.host`) and inside workers.
- Backend modules keep the wire contract (`load(init)`, `handle(state, items, params)`); `Env.host` calls the same
  `handle` in-process, so a backend has one code path for both env kinds.
- Test seam (declared, ruling P2-81): for the fake stack, `Env.isolated` resolves to the running interpreter in a
  worker subprocess; no env is built locally.

- [ ] **Step 1: Failing tests (local, fake stack):** `test_model_loaded_once_across_calls`,
  `test_cache_evicts_under_budget`, `test_host_and_isolated_share_handle`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): one model cache for host and workers`; push; CI green.

### Task 10b: `run_call` and the wired generated API

**Model:** opus implements (identity); fresh opus reviews. **Tier:** logic, contract. **Implements:** design §4.12,
ADR-002, P-1, P-8; rulings P2-16, P2-37, P2-43, P2-46, P2-57, P2-63. **Depends on:** Tasks 3c, 6, 7, 8, 9b, 10a.

**Files:** `runtime/call.py`, `scripts/generate_task_api.py`, `packages/senselab/src/senselab/_api/` (regenerated),
`senselab/__init__.py`, tests.

**Interfaces:**
- `run_call(call, *, execution=None, run=None, store=None) -> Result` in `senselab_core.runtime.call` (ruling P2-43):
  `execution` is the execution profile, `run` the run config (`reuse`, and from Task 21 `verify`), `store` the store;
  `None` means the packaged default. It resolves the model (catalogue pin or `resolve()`), places it
  (`resolve_execution`), opens or reuses the env and worker, builds the key with `build_key`, serves `store.get` under
  the reuse policy, else runs and `store.put`s. It writes the execution activity and its association, with
  `senselab:modelSource` on the agent, and sets `ExecutionContext.model_source` (ruling P2-37); the store writes the
  entity and any reuse activity. Phase 3's `InProcess` executor replaces it; nothing wraps it.
- The generated classes gain `__call__(x)` → one result through `run_call`; `sl.describe` gains models, licenses
  and default model from the catalogue; `sl.models(task=None)` is added.

- [ ] **Step 1: Failing tests (local, fake stack):** `test_stack_switch_by_profile`, `test_second_call_hits_store`,
  `test_one_activity_per_call`, `test_model_source_recorded`, `test_function_equals_configured_call`,
  `test_call_from_describe_alone` (a call built only from `describe()` JSON runs on the fake stack),
  `test_describe_lists_catalogue_models`, `test_generated_api_has_no_drift` (rerun).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/call_test.py -m "logic or contract"`; `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): run_call`; push; CI green. Spec commit: ADR-002 amendment
  line (Phase 2's `run_call` sets `model_source`; Phase 3 executors keep it) and R-110's owner updated (ruling P2-37).

### Task 11a: Kernel types the ported tasks consume

**Model:** opus implements (kernel); fresh sonnet reviews. **Tier:** logic, contract. **Implements:** ADR-005 (task
inputs are protocols), ADR-006 (format tags), design §4.1; ruling P2-67. **Depends on:** Task 1.

**Files:** `kernel/signals.py`, `kernel/results.py`, `kernel/formats.py`, tests.

**Interfaces:**
- `RegularSeries` protocol (ADR-005): what a task over regularly sampled channels reads; `RegularGroup` and `Audio`
  satisfy it.
- `Text` (ADR-005 protocol, with one content-identified implementation, `text@1.0`).
- `Transcript` (`transcript@1.0`): a tree of annotations whose word spans are integer ns on a clock, with optional
  per-word confidence.
- `TextSpans` (`text-spans@1.0`): columnar character spans `[start, end)` over a `Text` content identity, with label
  and score.
- `Trials` (`trials@1.0`): pairs of embeddings with a same/different label (speaker verification scoring input).
- `Metric` (`metric@1.0`): a named value with its unit and item count (metric task output).

- [ ] **Step 1: Failing tests (local):** `test_regular_group_satisfies_regular_series`,
  `test_transcript_word_spans_are_ns`, `test_text_spans_within_text_bounds`, `test_every_new_type_round_trips_by_format_tag`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel -m "logic or contract" -k "not pyld"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): kernel types for the Phase 2 tasks`; push; CI green. Spec
  commit: ADR-006 format table rows.

### Task 11b: Clock re-binding, chunked stamps, local identity rule

**Model:** opus implements (kernel); fresh sonnet reviews. **Tier:** logic, contract. **Implements:** R-114 (H-R59),
R-109, R-118; rulings P2-21, P2-22, P2-66. **Depends on:** Task 10b.

**Files:** `kernel/time.py`, `kernel/signals.py` (`SourceStamp` format bump), `runtime/call.py` (hit path),
`runtime/audio_io.py`, `runtime/models.py`, tests.

**Interfaces:**
- `rebind(output, stored_anchor, reader_anchor, timemap)` in `kernel/time.py` re-binds every clock-relative field of
  an output; `run_call`'s store-hit path calls it, so a cached output is served on the reader's anchor; never keyed by
  anchor.
- Source stamps record per-chunk digests over the `content_identity` chunk grid (64 MiB); a windowed read verifies
  only the chunks it touches; a `cp -p` rewrite inside a read chunk still raises `StaleSource`.
- Local model identity hashes every regular file under the root except VCS metadata directories (`.git`, `.hg`,
  `.svn`); symlink rules stay as R-135 fixed them.

- [ ] **Step 1: Failing tests (local):** `test_rebound_equals_reclocked_compute` (through `run_call`: one content under
  a session anchor, then standalone, one computation), `test_window_read_hashes_touched_chunks_only` (bytes read
  counted at the `os` boundary), `test_rewrite_in_touched_chunk_is_stale`, `test_vcs_dirs_do_not_change_identity`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel/time_test.py packages/senselab-core/tests/runtime/{audio_io,models,call}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): clock re-binding, chunked stamps, local identity rule`; push;
  CI green. Spec commits: ADR-005 amendment (whole-file conflict resolved by chunked stamps), ADR-002 amendment log
  line.

### Task 12: DSP primitive `sl.signal.resample`

**Model:** sonnet implements; opus reviews. **Tier:** logic (property), contract. **Implements:** design §4.15
tier 1, P-26, P-13; rulings P2-20, P2-59. **Depends on:** Tasks 6, 10b, 11a.

**Files:** Create `packages/senselab/src/senselab/signal/resample.py` (backend and its worker module in one) and the
`resample` declaration in `declarations.py`. No `git mv`: the 1.x `resample_audios` (torch/speechbrain) is rewritten,
not ported; the ledger row says so (ruling P2-59).

**Interfaces:** task `resample` (`transform`, input `RegularSeries`, param `target_rate: int`), backend `numpy-sinc`
(name and version declared, stack `dsp`, `Env.host`, no model, load stub = its parameter-free state):
rational-ratio Kaiser-windowed sinc, generic over channel groups. Pass-band ripple and stop-band attenuation bounds
are computed from the filter's design (Kaiser β, transition width), never measured.

- [ ] **Step 1: Failing tests (local):** Hypothesis `test_duration_preserved_within_one_sample`,
  `test_passband_within_design_ripple`, `test_stopband_below_design_attenuation`,
  `test_channel_order_and_gain_invariance`; generated contract cases for the backend.
- [ ] **Step 2: Implement; Step 3 (local):** `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests/signal -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): sl.signal.resample`; push; CI green.

### Task 13: Port HF Whisper on transformers (host)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; env and limits on ORCD. **Implements:** design
§6.2, P-4, P-8, P-16; rulings P2-18, P2-36, P2-56, P2-62, P2-65, P2-73, P2-82. **Depends on:** Tasks 7, 10b, 11a, 12.

**Files:** P-39 check. Rename-only commit: `git mv src/senselab/audio/tasks/speech_to_text/{huggingface,token_confidence}.py`
to the mirrored path. Modify commit: `@backend(name="whisper-transformers", version=..., task="speech_to_text",
stack="transformers", env=Env.host(), batch_invariant=False, load_stub=...)` with its worker module; `token_confidence`
is the one word-confidence engine; `api.py` does not move and any helper needed from it is rewritten, the ledger
naming the source (ruling P2-36); declaration `SpeechToText` in `declarations.py`; the root `torch` extra filled and
`uv.lock` committed (`uv lock`, resolution only).

**Interfaces:** output `Transcript` with word spans in integer ns; timestamps per `Params.timestamps`; typed faults
(`ContentFault` for undecodable input, `ResourceFault` for OOM); the torch import stays inside `load`. Batch
formation is declared non-invariant until Task 22b certifies otherwise (ruling P2-82).

- [ ] **Step 1: Failing tests (local, contract):** generated contract cases for the backend on its load stub;
  `test_senselab_imports_no_framework` (rerun, now covering this backend).
- [ ] **Step 2: Implement.** Ledger mark and unported functions.
- [ ] **Step 3:** Review and compact, commit `feat(2x): Whisper on transformers`; push; CI green.
- [ ] **Step 4 (orcd, CPU node3805 then GPU H100, at that SHA):** `-m "env and model" -k whisper_transformers` (the
  contract cases with real weights) and `-m "limits and model" -k whisper_transformers`: the generic edge set (empty,
  one-sample, silence, clipping, long input, each rate), zero gaps over fault fixtures, `content_fault_isolated` on
  the real backend. Batch-of-1 and mixed-batch outputs are recorded with their context in the store for Task 22a; no
  tolerance is asserted (ruling P2-65).

### Task 14: Port CrisperWhisper on ct2 and transformers (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** design
§6.2, P-6, P-8; rulings P2-56, P2-65. **Depends on:** Tasks 9a2, 13.

**Files:** P-39 check. Rename-only commit: `git mv src/senselab/audio/tasks/speech_to_text/crisperwhisper.py`. Modify
commit: two backends over one module (`stack="ct2"` and `stack="transformers"`, both `Env.isolated("crisperwhisper")`,
each with name, version and load stub), sharing the decoding code, with word confidence from `token_confidence` (one
engine, ruling P2-56); the ct2 conversion cache keyed by `identity_digest`.

- [ ] **Step 1 (local):** generated contract cases for both backends on their stubs; `test_two_stacks_one_output_schema`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3:** Review and compact, commit `feat(2x): CrisperWhisper on ct2 and transformers`; push; CI green.
- [ ] **Step 4 (orcd, CPU then GPU H100, at that SHA):** `-m "(limits or env) and model" -k crisperwhisper`; both
  stacks' outputs recorded with their context; cross-stack agreement is asserted in Task 22b.

### Task 15: Port pyannote diarization (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** design
§6.2; ruling P2-69. **Depends on:** Tasks 9a2, 10b.

**Files:** P-39 check. Rename-only commit: `git mv src/senselab/audio/tasks/speaker_diarization/pyannote.py`. Modify
commit: `@backend(name="pyannote-diarization", version=..., task="diarize", stack="pyannote",
env=Env.isolated("pyannote"), load_stub=...)`; output `Intervals` of speaker turns; gated model read with `HF_TOKEN`;
`data/env_locks/pyannote.txt` generated with the moved lock tool (`uv pip compile`, resolution only, no install) from
the backend's requirement list, pinning a `lightning` version outside the quarantined 2.6.2/2.6.3.

- [ ] **Step 1 (local):** generated contract cases on the stub; **Step 2:** implement and generate the lock;
  **Step 3:** review and compact, commit `feat(2x): pyannote diarization`; push; CI green.
- [ ] **Step 4 (orcd, CPU then GPU H100, at that SHA):** `-m "(limits or env) and model" -k pyannote`; outputs recorded
  with their context.

### Task 16: Port speechbrain speaker embeddings (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** rulings
P2-36, P2-69, P2-82. **Depends on:** Tasks 9a2, 10b.

**Files:** P-39 check. Rename-only commit: `git mv src/senselab/audio/tasks/speaker_embeddings/{speechbrain,windowing}.py`.
Modify commit: `@backend(name="speechbrain-ecapa", version=..., task="speaker_embeddings", stack="speechbrain",
env=Env.isolated("speechbrain"), output_affecting={"batch_formation"}, load_stub=...)`; `windowing`'s call into
`speaker_embeddings/api.py` (which does not move) is rewritten; `data/env_locks/speechbrain.txt` generated as in Task
15.

**Interfaces:** output `Embedding` with `space` = the resolved model identity; batches formed by duration bucket;
`batch_formation` is output-affecting until Task 22b certifies invariance, then dropped in a later commit with a
version bump (ruling P2-82, I-3).

- [ ] **Step 1 (local):** generated contract cases on the stub; **Step 2:** implement and generate the lock;
  **Step 3:** review and compact, commit `feat(2x): speechbrain speaker embeddings`; push; CI green.
- [ ] **Step 4 (orcd, CPU then GPU H100, at that SHA):** `-m "(limits or env) and model" -k speechbrain`, including
  duration-matched controls for short spans; outputs recorded with their context.

### Task 17a: PII review task and its transformers backend

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** design
§6.2, P-8, P-12, P-17; rulings P2-19, P2-77. **Depends on:** Tasks 9a2, 10b, 11a.

**Files:** P-39 check. Rename-only commit: `git mv src/senselab/text/tasks/pii_detection/{redaction_review,redaction_policy}.py`
and its `data/`. Modify commit: task `pii_review` (`measure`; input `Text`; output `TextSpans` with categories and
scores; no verdict); one prompt-and-parsing engine; backend `stack="transformers"` (`Env.isolated("pii-review")`,
lock `pii-review`, name, version, load stub); dataset-specific guidance and policy text become a profile slice with a
derivation (P-12), keyed as `profile:`. Ledger: unported PII modules (`rules`, `local_llm`, `subprocess_backend`,
`api`) listed with their `alpha` source.

- [ ] **Step 1 (local):** generated contract cases on the stub; `test_prompt_built_from_profile_slice`;
  `test_findings_parsed_into_text_spans`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(2x): PII review on transformers`; push; CI green.
- [ ] **Step 4 (orcd, CPU node3805 and GPU H100, at that SHA):** `-m "(limits or env) and model" -k pii_review_transformers`.

### Task 17b: PII review on vLLM through `VllmServer`

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits on ORCD GPU. **Implements:** P-37 (second
vLLM server folded), I-8; rulings P2-51, P2-77. **Depends on:** Tasks 9b, 17a.

**Files:** Rename-only commit: `git mv src/senselab/text/tasks/pii_detection/redaction_review_vllm.py`. Modify commit:
backend `stack="vllm"` (`Env.server("vllm")`, through `VllmServer`) over Task 17a's engine; its process management is
removed. The `pii-review-vllm` lock is not ported (superseded by `Env.server("vllm")`); the ledger says so.

- [ ] **Step 1 (local):** generated contract cases on the stub; `test_backend_server_is_attested_vllmserver` (against
  the stdlib stand-in: the attestation recorded in a `run_call` context equals the `VllmServer` instance's
  attestation; ruling P2-51).
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(2x): PII review on vLLM`; push; CI green.
- [ ] **Step 4 (orcd, GPU H100, at that SHA):** `-m "limits and gpu and model" -k pii_review_vllm`.

### Task 18: License backfill, env-lock licenses, env-grouping proposal

**Model:** sonnet implements; opus reviews. **Tier:** contract; env on ORCD. **Implements:** design §4.13 (CI gate
equals run-time default), §4.11 env grouping, §6.4, R-116; rulings P2-27, P2-61. **Depends on:** Tasks 13–17b.

**Files:** `scripts/check_licenses.py` (a lock-file reader), `packages/senselab-core/tests/lint/check_licenses_test.py`,
`packages/senselab/src/senselab/data/catalogue.yaml` (license records), `scripts/env_grouping.py`,
`packages/senselab/src/senselab/data/env_grouping.yaml`.

**Interfaces:** every catalogue model and every locked package of every shipped env (transitive) has a license
record; unknowns need a dated manual record. `check_licenses.py` gains a lock-file reader whose records come from the
installed metadata of a built env's wheels; the existing `licenses` job (already on `ubuntu-latest`) keeps checking
the root install. `scripts/env_grouping.py` resolves pairwise unions of backend locks and writes
`data/env_grouping.yaml` (`env-grouping@1`), recorded, not adopted (Phase 4).

- [ ] **Step 1 (local):** `uv run --no-sync pytest packages/senselab-core/tests/lint/check_licenses_test.py -m "logic or
  contract"`: the catalogue's model records, and the lock reader over a toy lock with fixture wheel metadata.
- [ ] **Step 2:** Review and compact, commit `feat(2x): license backfill and env-lock license reader`; push; CI green.
- [ ] **Step 3 (orcd, CPU node3805, at that SHA):** build every shipped env and run `check_licenses.py` against each;
  `uv run --no-sync python scripts/env_grouping.py`.
- [ ] **Step 4:** commit the generated `env_grouping.yaml` and any manual license records the run named, reviewed;
  push.

### Task 19: CI cadence and process items

**Model:** sonnet implements; opus reviews. **Tier:** contract. **Implements:** R-115, R-113 (H-R54), R-117 (P-40),
R-133, R-134, R-157; rulings P2-29, P2-33, P2-54. **Depends on:** Task 13.

**Files:** `scripts/check_source_digests.py` (and its entry in the `digests-are-tagged` source list),
`.github/workflows/2x-tests.yaml`, `2x-nightly.yaml` (on `alpha`), `ci_tiers_test.py`, `packages/conftest.py` (the
`model` sub-marker), the 2.x PR template. Spec: `testing.md` §2 and §5.

**Interfaces:**
- `scripts/check_source_digests.py` recomputes each task's and backend's source digest and fails when it changed
  without a `version` (task) or `backend_version` (backend) bump. Until the first release it compares against the
  digests at the merge-base with `2.x`; from the first release, against
  `packages/senselab/src/senselab/data/source_digests.yaml` (ruling P2-54).
- `2x-nightly.yaml` on `alpha` (a `skip-release` PR), checking out `2.x`, runs every PR job plus `limits and not
  model`; its first green run recorded in `STATE.md`. testing.md's tier table and cadence table are amended in the same
  task: limits needing a model or backend env, every backend env build and every use case are ORCD-only, submitted
  from a pinned SHA (ruling P2-33); `ci_tiers_test.py` enforces the new cadence.
- The 2.x PR template carries implementer and reviewer model fields; a check refuses equal values.
- R-133: each list-driven guard test is kept only if it guards a design decision and duplicates no other check; the
  register row's resolution lists each one kept or removed with its reason. R-134: the size-budget derivation is
  written in `engineering.md` (ruling P2-29).

- [ ] **Step 1 (local):** `test_source_edit_without_bump_fails`, `test_source_edit_with_bump_passes`,
  `test_baseline_is_merge_base_before_first_release`, `test_equal_models_refused`, `ci_tiers_test.py` updated.
- [ ] **Step 2: Implement**; open the `alpha` workflow PR with `skip-release`.
- [ ] **Step 3:** Review and compact, commit `ci(2x): source digests, nightly, PR model fields`; push; CI green. Spec
  commit: `testing.md` §2 and §5 amended, R-133 and R-134 resolutions.

### Task 20a: Metric tasks

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** `testing.md` §6,
`benchmarks.md` (reference scorers); ruling P2-80. **Depends on:** Tasks 10b, 11a.

**Files:** `packages/senselab/src/senselab/metrics/{wer,der,eer,span_f1,resample_error}.py` and their declarations.

**Interfaces:** each metric is a declared `measure` task with a host backend (no model) returning `Metric`: `wer`
(`Transcript` pair, HF Open ASR normaliser), `der` (`Intervals` pair, pyannote-metrics definition), `eer` (`Trials`),
`span_f1` (`TextSpans` pair), `resample_error` (`RegularSeries` pair).

- [ ] **Step 1 (local):** `test_metric_matches_reference_rows` (small published tables per metric); generated contract
  cases.
- [ ] **Step 2: Implement; Step 3 (local):** `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests/metrics -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): metric tasks`; push; CI green.

### Task 20b: Benchmark harness, manifests, smoke suite, CLI

**Model:** sonnet implements; opus reviews. **Tier:** logic (harness on the fake stack). **Implements:**
`testing.md` §6, `benchmarks.md`, docs.md (first fetch consumer), H-R49, P-30, P-41; rulings P2-25, P2-26, P2-71,
P2-80. **Depends on:** Tasks 12–17b, 20a.

**Files:** `benchmarks/harness.py`, `benchmarks/manifests/{librispeech-test-clean,voxconverse-v0.3,gretel-synthetic-pii}.yaml`,
`benchmarks/suites/smoke.yaml`, `packages/senselab/src/senselab/cli.py` (`bench run`), `kernel/formats.py`
(`benchmark-result@1`).

**Interfaces:** `senselab bench run <suite> --execution <profile>` fetches each manifest through `senselab data
fetch`, runs `run_call` per item, scores with the metric tasks, and writes results with full provenance to
`$SENSELAB_CACHE/2x/benchmarks/` (`benchmark-result@1`). Each manifest assigns every item to `calibration` or
`heldout` by the digest of its item id (ruling P2-41). Smoke items: LibriSpeech test-clean 50 utterances (ASR),
VoxConverse test 5 files (diarization), LibriSpeech test-clean speaker trials (speaker embeddings, ruling P2-25),
gretel synthetic PII 200 documents (ruling P2-26), synthetic tones (resample).

- [ ] **Step 0 (spec):** read the gretel README; record its license and terms in `benchmarks.md` (spec commit). If
  the terms do not allow the use, gretel leaves the suite and the ledger says why.
- [ ] **Step 1 (local):** harness logic on the fake stack and a toy manifest: `test_harness_records_provenance`,
  `test_manifest_split_is_deterministic`, `test_bench_cli_runs_toy_suite`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(2x): benchmark harness and smoke suite`; push;
  CI green.

### Task 20c: Smoke benchmarks on ORCD

**Model:** sonnet submits; the main thread holds the wait. **Tier:** benchmark (ORCD). **Depends on:** Task 20b.

- [ ] **Step 1 (orcd, CPU node3805, at Task 20b's SHA):** `senselab bench run smoke --execution orcd`, under 15 min.
- [ ] **Step 2 (orcd, GPU H100, same SHA):** the same suite on GPU. Evidence recorded by SHA; failures follow the
  P2-34 loop.

### Task 21: Verify sampling and per-backend equivalence

**Model:** opus implements (identity, reuse policy); fresh sonnet reviews. **Tier:** logic, contract; limits on
ORCD. **Implements:** ADR-002 §3, §4, R-112, R-155, I-4; rulings P2-23, P2-43, P2-60, P2-83. **Depends on:** Tasks
10b, 20c, 22a.

**Files:** `runtime/equivalence.py`, `runtime/data/equivalence.yaml` (`equivalence@2`), `runtime/call.py`,
`kernel/formats.py` (`equivalence-evidence@1`), `scripts/regenerate_equivalence.py`.

**Interfaces:** `equivalence@2`: classes per backend, keyed by numerics class, each with a derivation; duplicate keys
and empty derivations refused (R-155). `verify: k` is a run-config key beside `reuse` (ruling P2-43): `run_call`
recomputes a fraction `k` of cross-context hits and compares them within the backend's `tolerance@1` profile (Task
22a); agreements append `equivalence-evidence@1` records to the store, disagreements split the class and are
reported; `equivalent` with no class behaves as `exact`. `equivalence.yaml` is regenerated from evidence by
`scripts/regenerate_equivalence.py`, never hand-edited (ruling P2-83).

- [ ] **Step 1 (local, fake stack):** `test_unmeasured_equivalent_is_exact`, `test_verify_agreement_adds_evidence`,
  `test_verify_disagreement_splits_class`, `test_equivalence1_refused`, `test_equivalence_duplicate_key_refused`,
  `test_empty_derivation_refused`, `test_regenerated_equivalence_matches_evidence`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{equivalence,call}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): verify sampling and per-backend equivalence`; push; CI green.
  Spec commit: ADR-002 amendment (classes per backend by numerics class; verify sampling owned).
- [ ] **Step 5 (orcd, CPU node3805 and GPU H100, at that SHA):** a smoke run with `verify: 0.1` across the two
  contexts records evidence; then `scripts/regenerate_equivalence.py` output committed, reviewed.

### Task 22a: Cross-context tolerance profiles

**Model:** sonnet implements; opus reviews. **Tier:** limits (`model`) on ORCD. **Implements:** `testing.md` §4,
ADR-002 §3, P-17; rulings P2-41, P2-65. **Depends on:** Tasks 13–17b, 20c.

**Files:** `scripts/generate_tolerance.py`, `kernel/formats.py` (`tolerance@1`),
`packages/senselab/src/senselab/data/tolerance/<backend>.yaml`.

**Interfaces:** a profile is the spread of one 2.x backend's outputs across contexts (repeat runs, CPU vs CUDA,
batch 1 vs N, and both stacks for CrisperWhisper and PII review) on the **calibration** split, per output field, with a
derivation naming the SHA, job ids and items. Profiles are generated, never hand-edited, and never fitted to data
they judge.

- [ ] **Step 1 (local):** `test_profile_derived_from_spread_with_derivation`, `test_profile_reads_calibration_only`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(2x): tolerance profile generator`; push; CI green.
- [ ] **Step 4 (orcd, CPU node3805 and GPU H100, at that SHA):** run every Phase 2 backend over the calibration split
  in each context; `scripts/generate_tolerance.py` writes the profiles.
- [ ] **Step 5:** commit the generated profiles, reviewed; push.

### Task 22b: Parity with 1.x on the held-out split

**Model:** sonnet implements; opus reviews. **Tier:** limits (`model`, parity, migration only) on ORCD.
**Implements:** `testing.md` §4 parity, P-17, design §6.2 exit; rulings P2-24 (as amended), P2-41, P2-79, P2-82.
**Depends on:** Task 22a.

**Files:** `packages/senselab/tests/parity/`, `benchmarks/parity_reference.py` (the `parity-reference@1` converter),
`kernel/formats.py` (`parity-reference@1`).

**Interfaces:** reference outputs come from the 1.x code at the merged `alpha` commit, run on ORCD from its own
pinned checkout over the **held-out** split, and are converted from 1.x output files to `parity-reference@1`
(registered per ADR-006) as benchmark artifacts, never committed. `tests/parity/` asserts on the held-out split:
2.x versus the 1.x reference within the backend's `tolerance@1`; cross-stack agreement (CrisperWhisper, PII review)
within its profile; and any batch-invariance claim a port wants to make. A failure is a finding (a register row),
never a refit or a regenerated bound. vLLM parity is CUDA only; the PII CPU parity runs on the transformers stack.

- [ ] **Step 1 (local):** `test_converter_reads_1x_output_files` (fixture outputs), `test_parity_uses_heldout_only`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `test(2x): parity with 1.x on the held-out split`;
  push; CI green.
- [ ] **Step 4 (orcd, CPU node3805 and GPU H100):** produce 1.x references at the pinned `alpha` SHA; then `-m "limits
  and model" packages/senselab/tests/parity` at Task 22b's SHA. Red results become register rows; any fix follows the
  P2-34 loop.

### Task 23a: Phase 2 reference pages and status page

**Model:** sonnet implements; opus reviews. **Tier:** contract (docs build). **Implements:** docs.md Phase 2, P-41,
R-130. **Depends on:** Tasks 10b, 18, 20b, 22b.

**Interfaces:** reference pages generated from declarations (schemas, default models, stacks, licenses); the 2.x
status page (each task declared, implemented, ported, benchmarked or at parity; phase exits; consolidation
inventory; ledger marks; open register rows) and per-page badges in the PR preview.

- [ ] **Step 1 (local):** `uv run --no-sync --group docs pdoc` builds (if the docs group is not in the local venv,
  the build is proved in the docs-preview CI job); the status page test reads the registry.
- [ ] **Step 2:** Review and compact, commit `docs(2x): Phase 2 reference and status page`; push; CI green.

### Task 23b: Phase 2 spec pass

**Model:** sonnet implements; opus reviews. **Tier:** none. **Branch:** `2x/phase-2-spec`. **Implements:** D-7;
ruling P2-39. **Depends on:** Task 23a.

- [ ] **Step 1:** design §6.1 rewritten for merge-based reconciliation; the ledger trimmed; the 14 unclassified items
  classified; design §6.2's owner list amended so A1 is a Phase 2 F1 prerequisite (ruling P2-39); every ADR amendment
  the tasks above name is present and dated.
- [ ] **Step 2:** Review, commit `docs(2x): Phase 2 spec pass`; push.

### pydra gate prerequisites (Tasks 24–31b)

Work in `~/software/nipype/pydra-ext`. Each PR is a branch `enh/ee-<id>-<slug>` cut from pydra `main`, or from the
branch it depends on; a **draft** PR to `nipype/pydra` `main` citing proposal 0001 (#891); merged into
`enh/execution-extensions` once green so Phase 3 pins one commit (ruling P2-31). senselab takes no pydra dependency in
Phase 2 (ruling P2-32). Tests: pydra's own CI on each draft PR, plus ORCD (CPU node3805) runs in a locked venv built
in the job (`uv venv` + `uv pip install -e ".[test]"` from the PR's lock), `pytest pydra -k <area>` serially;
nothing runs locally (no pydra venv exists and none is created).

| Task | PR | Model (implements / reviews) | Interfaces and tests | Depends on |
|---|---|---|---|---|
| 24 | **D1** `ResultStore` protocol + `PickleStore` | opus / fresh sonnet | `lookup(checksum, context=None)`, `acquire(checksum)`, `commit(lease, result)` at the four call sites; behaviour-identical; `debug` and `cf` parity | — |
| 25 | **G1** `Environment.checksum()` | sonnet / opus | recorded in context beside the result, never in the job checksum; `Docker`/`Singularity` accept `identity=`; a rebuilt env leaves checksums unchanged | — |
| 26 | **K1** declared task identity, context-aware reuse | opus / fresh sonnet | `@python.define(version=)` replaces the code hash; entries record context; `Submitter(reuse="exact"\|"any")`, default `exact` | D1, G1 |
| 27 | **K2** node-supplied checksum, reuse hook | opus / fresh sonnet | `__checksum__`/`checksum=`; `Job.checksum` equals the supplied value exactly; `ReusePolicy.accept(stored, current, task)`; built-in `exact`, `any` | K1 |
| 28 | **A1** Slurm requeue budget, `CANCELLED` (pulled forward, rulings P2-30, P2-39) | sonnet / opus | a requeue budget; a user's `CANCELLED` is not requeued; Slurm test on ORCD | — |
| 29 | **F1** `ComputeResources` + Slurm/SGE mapping | sonnet / opus | set by the define machinery, overridable per node, never in the checksum; mapped to `sbatch`/`qsub` args; Slurm test on ORCD | A1 |
| 30 | **F2** worker routing | sonnet / opus | `Submitter(workers={...}, router=...)` called at the three dispatch sites; definitions name no worker | F1 |
| 31a | **E1a** per-element readiness (no re-split case) | opus, effort higher / fresh opus | a downstream element runs when its own upstream indices succeed; full splitter/combiner matrix and `debug` parity | — |
| 31b | **E1b** `on_error` propagation | opus / fresh opus | an errored element marks only its descendants; siblings and their descendants complete; `debug` parity | E1a |

Each task: failing tests first in pydra's suite; implement; review and compaction by the named reviewer; push the
branch; open the draft PR; ORCD step (CPU node3805, the PR head SHA) runs the area's tests and, for A1 and F1,
submits a real Slurm job from the test; merge into `enh/execution-extensions`.

### Task 32: Phase 2 exit checks

**Model:** sonnet implements; opus reviews; the P-1 check uses a fresh sonnet agent with no repository access.
**Tier:** all. **Depends on:** Tasks 0–31b.

- [ ] **Step 1 (CI):** `2x-tests` green at the head of `2x/phase-2`: `core-only` (3.11, 3.14), `senselab-lean`,
  `worker-floor` (3.10), `lint-imports`, pre-commit, `licenses`; the nightly green on `2.x`.
- [ ] **Step 2 (orcd, CPU node3805 and GPU H100, head SHA):** the full packages suite, every tier, serial; env builds
  for every shipped lock; the R-124 fd-0 probe (`strace -f -e trace=openat,getrandom` beside `readlink
  /proc/self/fd/0`), its result recorded in `wire-protocol.md` or the row closed with the negative result.
- [ ] **Step 3 (P-8, orcd):** each two-stack call (CrisperWhisper ct2/transformers, PII vLLM/transformers) switches by
  `stack_preferences` alone; the task call and output schema are unchanged; the context records the stack.
- [ ] **Step 4 (P-1, orcd):** the fresh agent receives only `sl.describe()` output for each task and a public input
  path, writes the call JSON, and `senselab_core.runtime.call.run_call` runs it.
- [ ] **Step 5:** parity green on CPU and on the H100 host (Task 22b); the pydra prerequisites green on their draft
  PRs and merged into `enh/execution-extensions` (gate evidence for Phase 3's first task).
- [ ] **Step 6:** P-39 reconciliation again (Task 0's steps).

### Task 33: Phase 2 exit adversarial round (P-35)

**Model:** opus lead, effort higher, with up to 5 sonnet subagents; claims verified by haiku (P-40). **Tier:**
read-only. **Depends on:** Task 32.

- [ ] **Step 1:** Brief the lead with the Phase 2 surfaces: declarations and `Params`, `build_key`, `run_call`, the
  catalogue and pins, the profile resolver, envs and leases, `VllmServer`, the model cache, the PROV-O store and
  redaction, chunked stamps and re-binding, the new kernel types, the ported backends, the benchmark harness, verify
  sampling, tolerance and parity, the pydra PRs. Goal: inputs, races or misuse that break ADR-002, 003, 006, 007, 009,
  010 or the Phase 2 exit; plus the P-37 duplicate-finding pass. Every finding carries an architectural resolution.
- [ ] **Step 2:** Findings become `review-register.md` rows from the next free id, with lens `EXT2`, on
  `2x/phase-2-spec`; findings saved under `reviews/phase2-exit-adversarial/`.
- [ ] **Step 3:** Commit `docs(2x): Phase 2 exit adversarial round`; push.

### Task 34: Phase 2 exit fix round

**Model:** per finding, from the roster (opus for kernel, concurrency and identity; sonnet otherwise); each fix
reviewed by a different model. **Depends on:** Task 33.

- [ ] **Step 1:** Fix every high and medium finding by an architectural change (P-35); schedule the rest with an
  owner in the register. Each fix is reviewed and compacted before its commit (P2-34).
- [ ] **Step 2:** CI green; rerun the affected ORCD steps of Task 32 at the new head.
- [ ] **Step 3:** `STATE.md`: Phase 2 done, with the head SHA and ORCD evidence by SHA; log line. Commit `docs(2x):
  Phase 2 done` on `2x/phase-2-spec`; push.

---

## Self-review notes

- **Spec coverage.** design §6.2 Phase 2: declarations, `Params`, generated functions and schemas → 5a, 5b, 6, 10b;
  profile resolver and `Env.server` → 8, 9b; catalogue → 7; one worker protocol and model cache → 10a, 10b; kernel
  types → 11a; ports → 12–17b; ADR-010 → 3a–3c, 4; benchmark harness, manifest, smoke → 20a–20c; license backfill and
  env grouping → 18; tolerance and parity → 22a, 22b; exit → 21, 32. docs.md Phase 2 → 23a. Spec pass → 23b. Gate
  prerequisites → 24–31b. Register rows owned by Phase 2: R-55, R-56, R-57, R-58, R-83 (per backend), R-88, R-109,
  R-110 (owner moved by the ADR-002 amendment), R-112, R-113, R-114, R-115, R-116, R-117, R-118, R-122..R-127, R-130,
  R-131, R-146..R-153, R-155, R-158, R-159; R-133, R-134, R-157 taken in Task 19 because no phase owned them.
- **Not in Phase 2:** `Plan`, `.map`, `Pool`, Slurm substrates, the leased service registry and lease renewal
  (R-111), taint propagation (R-116, Phase 3), resource tables (scale ladder), `prov:hadPlan`, the 26 env locks of
  unported backends.

## Rulings made in this plan

Each is the planner's choice where the design left one open; the owner reviews them. A ruling that changes an ADR
decision takes effect only through the dated amendment its task commits (ADR-000).

- **P2-1** ruling: The 2.x `senselab` import package lives at `packages/senselab/src/senselab/`, built by the root `senselab`
  distribution; `src/` is on no 2.x path; `packages-not-1x` becomes "core and worker never import `senselab`" — the root name is the 2.x umbrella, and the 1.x pool shares the import name.
- **P2-2** ruling (amended by P2-35): A moved file keeps its 1.x relative path under `packages/senselab/src/senselab/`,
  or its 1.x basename under `senselab_core/runtime/` when it is runtime infrastructure; it is `git mv`'d once and never
  renamed by a later task — `git log --follow` and blame stay trivial, and the 1.x layout already matches
  `sl.<modality>`.
- **P2-3** ruling: A partly ported file moves whole; the modify commit removes its unported functions and the ledger lists
  them with their `alpha` source — a file leaves `src/` only by `git mv`, and `alpha` keeps the 1.x copy current.
- **P2-4** ruling: A run's provenance is the union of per-writer documents `prov/<writer-uuid>.jsonld`, each replaced
  atomically; conflicting nodes under one `@id` are refused — append-only without a lease, and Phase 3
  executors write per fragment.
- **P2-5** ruling: Non-JSON values are XSD typed literals; arrays are content-addressed file entities; sets are sorted
  `@set`; enums are their `str`; bytes, paths and objects are refused — replaces the token families with PROV-O's
  own literal model.
- **P2-6** ruling: An activity's `@id` is `identity_digest("prov-activity@1", record)` over the compact record without
  `@id` — D-4 requires an execution address, and JCS of the compact form is what ADR-010 hashes.
- **P2-7** ruling: The `senselab-vocab@1.0` vocabulary is JSON-LD; each package ships its terms as package data registered
  with its declarations; the vocabulary is checked by a contract test until the first release (P2-40) — D-5 keeps the
  release gate minimal.
- **P2-8** ruling: Redaction is by construction: no document holds a path; labels map to paths only in
  `<run>/private/labels@1.json` (0600) — ADR-010 makes store and export one document.
- **P2-9** ruling: Task and backend schemas use JSON Schema 2020-12 — pydantic v2 emits it natively (R-55, R-131).
- **P2-10** ruling: The task API is generated at development time into committed modules, with a regeneration drift test — IDEs and pdoc need real signatures, from one source.
- **P2-11** ruling: Catalogue revisions are full commit SHAs or weights digests, never refs; a ref is accepted only in a
  user's call and resolved — P-16 for every test and benchmark path.
- **P2-12** ruling (amended by P2-68): The default model-source allowlist is `Catalogue.sources()`, the hub organisations
  of the catalogue entries; the execution profile owns it — ADR-007 needs an allowlist, and the catalogue is the only
  source list that exists.
- **P2-13** ruling: `execution-profile@1` carries stack preferences, devices, env roots, the torch index override, model
  sources, the model-cache budget, retry and network policy; `laptop` and `orcd` are packaged — the minimum Phase 2
  needs for P-8 and ADR-007.
- **P2-14** ruling: `Env.server` is process-scoped in Phase 2 (owned by the calling process); the leased service registry
  waits for lease renewal in Phase 3 — leases have no renewal (R-111), and a server outlives the stale bound. Enacted
  through the ADR-003 amendment of P2-38.
- **P2-15** ruling: The model cache lives in `senselab_worker`, one implementation for host and workers — P-37 and design
  §4.12.
- **P2-16** ruling: `run_call` is Phase 2's single-call path; Phase 3's `InProcess` replaces it, nothing wraps it — Phase
  2's exits need execution before `Plan` exists.
- **P2-17** ruling (enacted by P2-37): `run_call` sets `ExecutionContext.model_source` — it builds the context; R-110's
  Phase 3 work reduces to keeping that when `InProcess` replaces it.
- **P2-18** ruling: Only HF Whisper runs `Env.host` (the root lock's `torch` extra); every other ported backend is isolated,
  one env per backend, plus the shared `vllm` server env — keeps the senselab venv small and exercises the one worker
  protocol.
- **P2-19** ruling: PII review is a `measure` task emitting `TextSpans` findings; policy and guidance text are a profile
  slice — P-20 and P-12; the 1.x module mixes dataset guidance into code.
- **P2-20** ruling: The DSP primitive is `sl.signal.resample`, numpy-only Kaiser-windowed sinc over rational ratios — core
  is torch-free and the algorithm is generic over signals (P-26).
- **P2-21** ruling: Whole-file hashing (R-109) is replaced by per-chunk stamps on the `content_identity` grid, verified for the
  chunks a read touches — keeps the `cp -p` guarantee and ADR-005's "never hashed whole".
- **P2-22** ruling: Local model identity excludes only VCS metadata directories (R-118) — incidental files are content;
  VCS state is not.
- **P2-23** ruling: `equivalence@2` keys classes per backend by numerics class — R-112; global fingerprints mix backends.
- **P2-24** ruling (amended by P2-41): Parity references are 1.x outputs produced on ORCD from the pinned `alpha`
  checkout over the held-out split, converted to `parity-reference@1` and stored as benchmark artifacts; tolerances are
  2.x cross-context profiles from the calibration split; vLLM parity is CUDA only — 2.x cannot import 1.x, and vLLM on
  CPU is not a supported context.
- **P2-25** ruling: The speaker-embedding smoke uses LibriSpeech test-clean trials — VoxCeleb1 needs a request.
- **P2-26** ruling: The gretel synthetic PII set enters the smoke suite only after its README is read and its terms
  recorded (Task 20b Step 0) — `benchmarks.md` lists its README as unread.
- **P2-27** ruling: The env-grouping proposal is computed on ORCD by resolving pairwise unions of backend locks, recorded as
  `env-grouping@1`, not adopted — §6.4 defers the solver; resolution needs the network.
- **P2-28** ruling (superseded by P2-33): No model weights load in GitHub CI in Phase 2; the PR-triggered CPU smoke of
  `testing.md` is deferred.
- **P2-29** ruling: List-driven guard tests (R-133) stay only where they guard a design decision and duplicate no other check;
  size budgets (R-134) are set from ruff's complexity and statement limits plus a file-length limit, with the
  derivation written in `engineering.md` — D-1 and owner item 14; no phase owned these rows.
- **P2-30** ruling: pydra A1 is pulled into Phase 2 — F1 depends on it in proposal 0001.
- **P2-31** ruling: Each pydra PR is its own branch and draft PR, merged into `enh/execution-extensions` for pinning — the
  maintainers review per PR; Phase 3 pins one commit.
- **P2-32** ruling: senselab takes no pydra dependency in Phase 2; the gate evidence is pydra's tests on its CI and ORCD — the gate decision is Phase 3's first task.
- **P2-33** ruling (a-1): GitHub CI (PR and nightly) runs logic, contract, and limits that need no model and no subprocess
  venv; a test that loads weights or needs a backend env carries the `model` sub-marker. Every model-backed,
  venv-backed or GPU tier and every use case runs on ORCD only, submitted by an agent from a pinned SHA. testing.md's
  tier and cadence tables are fixed in Task 19, the task that adds the nightly workflow.
- **P2-34** ruling (a-2): Per task: implement → review and compact locally (owner item 39) → commit and push → ORCD run at
  that SHA. An ORCD failure gets a fix commit that is itself reviewed, then a scoped ORCD rerun at the new SHA. ORCD
  evidence is cited by SHA. Nothing is compacted after evidence without rerunning.
- **P2-35** ruling (a-3, amends P2-2): A rename-only commit moves whole files with `git mv` and nothing else, to
  `packages/senselab/src/senselab/<1.x relative path>` by default or `packages/senselab-core/src/senselab_core/runtime/`
  for runtime infrastructure core needs. Merging files and renaming identifiers, modules or directories
  (`venv_locks` → `env_locks`) happen in the following modify commit. A file is moved at most once in Phase 2.
- **P2-36** ruling (a-4, Table 1 #31, c-10): `api.py` never moves; helpers a port needs from it are rewritten in the modify
  commit and the ledger names the source.
- **P2-37** ruling (a-5, Table 1 #9): Task 10b's spec commit adds the ADR-002 amendment line by which Phase 2's `run_call`
  sets `model_source`, and moves R-110's owner; Task 3a adds the `senselab:modelSource` term.
- **P2-38** ruling (a-6): Task 9b's spec commit adds an ADR-003 "staged" amendment: services are process-scoped in Phase 2;
  the leased registry and `svc` pool worker arrive in Phase 3.
- **P2-39** ruling (a-7): Task 23b amends design §6.2's owner list: A1 is a Phase 2 prerequisite of F1.
- **P2-40** ruling (a-8): ADR-010 is amended: the vocabulary is checked by a contract test until the first 2.x release,
  then by the release gate; the gate check joins D-5's deferred list.
- **P2-41** ruling (a-9, b-1): Tolerance profiles are cross-context agreement (testing.md, ADR-002): the spread of 2.x
  outputs across repeat runs, CPU vs CUDA and batch 1 vs N on a calibration split. Parity with 1.x is asserted against
  them on a held-out split. No profile is fitted to the data it judges, and no bound is refitted to turn a failure
  green; a parity failure is a finding.
- **P2-42** ruling (a-10): The torch index override is the profile key `envs.torch_index`, recorded in the env context;
  `SENSELAB_TORCH_INDEX_URL` is removed. The probed CUDA routing keeps its semantics.
- **P2-43** ruling (a-11, Table 1 #15): One signature, `run_call(call, *, execution=None, run=None, store=None)`; `verify`
  is a run-config key beside `reuse`.
- **P2-44** ruling (a-12): ADR-006 is amended: before the first 2.x release a major format bump has no previous-major
  reader (H-R2); retired formats are refused.
- **P2-45** ruling (a-13): Task 0 used sonnet, effort lower, in a worktree, per the P-40 roster.
- **P2-46** ruling (a-14): `run_call` stays in `senselab_core.runtime.call`; the facade exposes the generated classes,
  `tasks`, `describe` and `models`; the P-1 exit check calls the runtime function.
- **P2-47** ruling (a-15, Table 1 #11): `@backend` declares `name` and `version`, both required; `backend_version` enters
  the key per ADR-002 and is what the source-digest check compares.
- **P2-48** ruling (b-2, c-2, Table 1 #12): `build_key(call, placement, context)` in `runtime/keys.py` is the one key
  builder, created in Task 5b; `key_ignores_<field>` cases go through it; `run_call` calls it.
- **P2-49** ruling (b-3, Table 1 #19): The SPDX parser moves from `scripts/check_licenses.py` into `runtime/licenses.py`;
  the script imports it.
- **P2-50** ruling (b-4, c-3, Table 1 #5): Task 2b uses the packaged `retry_policy.yaml`; Task 8 folds it into the profile's
  `retry` key and deletes the file.
- **P2-51** ruling (b-5): PII review's single-server guarantee is a behavioural test: the attestation in the `run_call`
  context is the `VllmServer` instance's (I-8); no source grep.
- **P2-52** ruling (b-6, Table 2 row 1): One check for lazy framework imports: Task 1's subprocess test, extended to every
  backend module and rerun per port; no import-linter contract and no per-port duplicate.
- **P2-53** ruling (b-7): One schema test that a ref revision is refused (`test_catalogue_refuses_ref_revision`).
- **P2-54** ruling (b-8): Until the first release the source-digest check compares against the digests at the merge-base
  with `2.x`.
- **P2-55** ruling (b-9, Table 2 row 2a): The shared bounded helper lives in `bounded_probe_fixture.py`; the worker
  package keeps one test-local helper.
- **P2-56** ruling (b-10, Table 1 #30): `token_confidence.py` moves with Whisper in Task 13 and is the one word-confidence
  engine; CrisperWhisper uses it.
- **P2-57** ruling (b-11, Table 1 #8): The store writes the result entity and any reuse activity; `run_call` writes only
  the execution activity and its association; pinned by `test_one_activity_per_call`.
- **P2-58** ruling (b-12, Table 2 row 3): The pyld contract uses an offline document loader serving the packaged context.
- **P2-59** ruling (b-13, Table 2 row 12): `resample` is a new module (no `git mv`, ledger "rewritten, not ported"); its
  bounds come from the filter design.
- **P2-60** ruling (b-14, Table 1 #6): The `equivalence.yaml` checks (R-155) are written once, for `equivalence@2`, in
  Task 21.
- **P2-61** ruling (b-15, Table 1 #36, c-15): `check_licenses.py` gains a lock-file reader fed by built envs' wheel
  metadata; Task 18's ORCD step builds every shipped env and runs it there.
- **P2-62** ruling (b-16, Table 1 #13, #14): Generated contract cases run each backend's worker against its declared load
  stub (the fake stack's model, no weights), locally; real-weight runs of the same cases are the `env` tier on ORCD.
  Every backend, including a model-less host backend, has a worker module.
- **P2-63** ruling (c-1, Table 1 #16): Task 6 uses declarations only; wiring the generated API to `run_call` and the
  catalogue moves into Task 10b.
- **P2-64** ruling (c-4, Table 1 #18, Table 2 row 7): Task 7 adds every Phase 2 catalogue entry, pinned, up front; the
  loader takes YAML text or a path; core tests use a toy catalogue and the packaged one is checked in
  `packages/senselab/tests`.
- **P2-65** ruling (c-5, Table 1 #32): Ports record outputs with their context and assert behaviour, not tolerance;
  tolerance profiles are produced in Task 22a, before Task 21 (which consumes them) and Task 22b.
- **P2-66** ruling (c-6, Table 1 #26, #27): Task 11b follows Task 10b and wires `rebind(output, stored_anchor,
  reader_anchor, timemap)` into `run_call`'s hit path.
- **P2-67** ruling (c-7, Table 1 #28, #29): Task 11a adds `RegularSeries`, `Text`, `Transcript`, `TextSpans`, `Trials` and
  `Metric` as kernel types under ADR-005/006 with their format tags.
- **P2-68** ruling (c-8, Table 1 #17): Task 7 supplies `Catalogue.sources()`; Task 8 owns `model_sources` and the
  refusal test.
- **P2-69** ruling (c-9, Table 1 #23): Tasks 15 and 16 generate their `pyannote` and `speechbrain` locks with the moved
  lock tool (resolution only, no install).
- **P2-70** ruling (c-11, Table 1 #25): Task 9a1's rename-only commit moves `cuda_probe.py` into core runtime; the
  CUDA-aware torch index routing keeps its `ensure_venv` semantics in `ensure_env`.
- **P2-71** ruling (c-12, Table 1 #3): The console script moves to `senselab.cli:main` in Task 1, reusing core's
  `data fetch`; Task 20b adds `bench run` there.
- **P2-72** ruling (c-13, c-14, Table 1 #4, #37): Task 10a depends on Task 2a; Task 23a depends on Tasks 18 and 22b.
- **P2-73** ruling (Table 1 #1, #2): Task 1's import test is `test_senselab_imports_no_framework`; Task 1 adds an empty
  `torch` extra and Task 13 fills it; both commit `uv.lock`.
- **P2-74** ruling (Table 1 #9): The `senselab:modelSource` term is part of the core vocabulary from Task 3a.
- **P2-75** ruling (Table 1 #10): `@backend` takes no license; `BackendSpec.license` is removed; a backend's code license
  is derived from its env lock's package records (the root lock for `Env.host`).
- **P2-76** ruling (Table 1 #20, #21): `resolve_execution(task, model_entry, profile, hardware, registry)` and
  `ensure_env(env, lock, root)` with `root` from the profile.
- **P2-77** ruling (Table 1 #22, #24): Env names are `Env.server("vllm")` (lock `vllm-server`) and
  `Env.isolated("pii-review")` (lock renamed from `pii-redaction-review` in Task 9a2's modify commit); only the Phase 2
  backends' locks move; `pii-review-vllm` is not ported.
- **P2-78** ruling (Table 1 #33): Task 9b keeps `VllmServer` and its HTTP client helpers; `ensure_vllm_venv` is replaced
  by `ensure_env`.
- **P2-79** ruling (Table 1 #34): 1.x reference outputs are converted by a `parity-reference@1` converter registered per
  ADR-006.
- **P2-80** ruling (Table 1 #35, Table 2 row 20): Metric tasks are declared in `packages/senselab/src/senselab/metrics/`;
  `benchmarks/` holds only manifests, suites and the harness.
- **P2-81** ruling (Table 2 row 10): Task 10a's `Env.isolated` test is a declared seam: for the fake stack it resolves to
  the running interpreter in a worker subprocess.
- **P2-82** ruling (Table 2 rows 13, 16): A port declares the conservative batch behaviour (`batch_invariant=False`, or
  `batch_formation` output-affecting); it is relaxed only after Task 22b certifies invariance, in a later commit with a
  version bump.
- **P2-83** ruling (Table 2 row 21): Verify evidence goes to the store as `equivalence-evidence@1`; `equivalence.yaml` is
  regenerated from it by a script, never hand-edited (P-17).
- **P2-84** ruling (d-1..d-8): Tasks 3, 5, 9a, 10, 17, 20, 23 and 31 are split into 3a–3c, 5a–5b, 9a1–9a2, 10a–10b,
  17a–17b, 20a–20c, 23a–23b and 31a–31b; Task 11 into 11a–11b for the added kernel types, and Task 22 into 22a–22b for
  the two artifacts of P2-41.
- **P2-85** ruling (local limits): No `uv sync` locally. A task that adds a package path or dependency tests locally with
  the new `src` on `PYTHONPATH` and the existing `.venv` via `uv run --no-sync`; the locked install is proved in CI.
- **P2-86** ruling (Table 2 rows 18, 19, 23): Each of Tasks 18, 19 and 23a names its files; Task 19 names the R-133 and
  R-134 outputs; Task 23's code and spec commits are separate tasks on separate branches.

## Design gaps left open

None. The ADR-010 vocabulary check is resolved by P2-40.
