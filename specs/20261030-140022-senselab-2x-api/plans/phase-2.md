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
- parity against 1.x within measured tolerances, on CPU and on one CUDA host;
- the pydra gate prerequisites D1, E1, F1, F2, G1, K1, K2 working on the pydra branch.

**Architecture:**
- `senselab_worker` (stdlib + numpy) gains the model cache, so host and isolated backends share one
  implementation (P-37).
- `senselab_core.kernel` gains the decorators, `Params`, `backend_params`, the JSON Schema of a declaration, clock
  re-binding and the PROV vocabulary types. It still does no I/O (ADR-009).
- `senselab_core.runtime` gains the PROV-O store, the catalogue, the profile resolver, the env builder,
  `VllmServer`, and `run_call`, the single-call path that Phase 3's `InProcess` executor replaces.
- The 2.x `senselab` package (facade, task declarations, backends, catalogue data, env locks) lives at
  `packages/senselab/src/senselab/` and is built by the root `senselab` distribution (ruling P2-1). 1.x modules reach
  it only by a rename-only `git mv` from `src/` followed by a modify commit (ADR-009, D-6).

**Tech stack:** as Phase 1, plus `pyld` in the core test group (JSON-LD expansion in contract tests), the `torch`
extra of the root distribution for the one host backend, and per-backend uv locks for isolated envs.

**Spec:** `design.md` §4.2–§4.4, §4.8, §4.10, §4.12, §4.13, §6.1, §6.2 (Phase 2); ADR-001, 002, 003, 005, 006, 007,
009, 010 (as amended); `execution.md`; `testing.md`; `docs.md`; `wire-protocol.md`; `benchmarks.md`;
`migration-ledger.md`; `review-register.md` rows owned by Phase 2; `rulings.md` H-R49..H-R59.

**Branches.** Code: `2x/phase-2` → `2.x` (worktree `~/software/sensein/senselab-2x-impl`). Spec changes, ADR
amendments, register rows and `STATE.md`: `2x/phase-2-spec` → `2.x` (worktree `~/software/sensein/senselab-2x`;
D-7). pydra: `~/software/nipype/pydra-ext`, per-PR branches merged into `enh/execution-extensions` (ruling P2-31).

## Global Constraints

- **No backwards compatibility** (P-14): no shims, aliases, re-exports or readers of retired formats. Every retired
  format is refused with a typed error that names the migration statement.
- **One implementation per behaviour** (P-37): a variant is a parameter, strategy or backend over one engine. A
  ported module that duplicates a core engine (locks, hashing, worker protocol, model cache, config load) is folded
  into that engine in its modify commit, never kept beside it.
- **Rationale lives in specs** (P-21, P-31): docstrings say what and how; measurements, history and rejected
  alternatives go to `specs/`. No register, round or ruling id outside `specs/`.
- **Review and compaction during development** (D-1 as amended, owner item 39). Each task's diff is reviewed by
  the named reviewer model (never the implementer's model and context, P-40) against `principles.md` and the
  `engineering.md` checklist, and compacted (duplicates folded, verbosity cut) before it is committed. No model runs
  in GitHub CI, for review or for inference (ruling P2-28).
- **Local machine** (owner items 27, 28): logic and contract tests only, in the existing `.venv` (re-synced with
  `uv sync --locked --group test` when a dependency changes), scoped to the changed files, serial. No new venvs, no
  model downloads, no subprocess-venv builds, no full suites. Watch load, memory and disk.
- **ORCD** for anything needing a model, a subprocess venv, a GPU, the full suite or a Slurm substrate: CPU on
  partition `pi_satra`, node `node3805`; GPU on typed H100 (`--gres=gpu:h100:1`); never `node2803`; never a login or
  remote-control node. Each orcd step uses the job shape below.
- **Weights pinned by commit** (P-4, P-16): every test or benchmark path that loads weights reads its model from a
  catalogue entry whose revision is a 40-hex commit or a 64-hex weights digest, and loads by that identity
  (resolve, then load with `revision=<sha>`). A contract test refuses a catalogue entry used by a test whose
  revision is a ref.
- **Tiers** (`testing.md`, H-R49): every test carries exactly one of `logic` (default), `contract`, `limits`,
  `env`, `usecase`; `gpu`, `network`, `slow` are sub-markers. PRs run `logic or contract`. Model-backed tests are
  `limits` or `usecase` and run on ORCD.
- **Moves** (ADR-009, D-6): merge `origin/alpha` and advance the area's ledger mark first (P-39); then a rename-only
  `git mv` commit; then a modify commit. The moved path keeps its 1.x relative path under
  `packages/senselab/src/senselab/` (ruling P2-2). A file only partly ported moves whole; its unported functions are
  removed in the modify commit and listed in the ledger with their `alpha` source (ruling P2-3).
- **Identity:** ADR-002 key and context fields exactly; nothing hashes function code, Python objects or
  environments. Every new persisted artifact carries a `format` tag registered per ADR-006.
- **Commits** (P-27): one commit per unit of work, pushed promptly; each task's last commit has a green 2.x CI and
  records the implementer and reviewer models in its message.

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
  `UV_CACHE_DIR=$SCRATCH/uv-cache`; isolated envs under `$SCRATCH/senselab-2x-venvs`.
- **Steps:** `uv sync --locked --all-extras --group test` as its own step, then `uv run --no-sync pytest <scope> -m
  "<tier>"`, serial. Check the next maintenance reservation before choosing `-t`.
- **Who waits:** an agent writes and submits the job; the main thread holds the wait. `COMPLETED` is not evidence:
  check elapsed time and grep the log for the tier's summary line.

## Review Focus

1. **PROV identity.** An activity's `@id` is the content address of its execution record, never the logical key; two
   contexts for one key give two activities; a reuse is its own activity. Pinned in Task 3
   (`test_one_key_two_contexts_two_activities`).
2. **No path leaves a run.** No provenance document, export, error or log line carries a file name or path; the
   label map stays in the private run directory. Pinned in Task 4 (`test_document_holds_no_path`).
3. **Params and backend params reach the key exactly once.** Output-affecting batch formation and chunk grid enter the
   key only when declared; execution-context and placement fields never do. Pinned in Task 5
   (`key_ignores_<field>` built through `run_call`'s key builder).
4. **Stack switch by profile alone.** Changing `stack_preferences` changes the backend and the recorded context, never
   the task call or the output schema. Pinned in Task 10 (`test_stack_switch_by_profile`) and Task 32.
5. **Pinned loads.** No backend loads through a ref; every load passes the catalogue's SHA. Pinned in Task 7
   (`test_test_catalogue_entries_are_pinned`) and each port's limits test.
6. **One worker protocol, total errors.** A real backend's per-item failure is a typed fault with `gap: false`;
   zero gaps over its fault fixtures (ADR-009 amendment). Pinned per port (`content_fault_isolated` on the real
   backend, R-57).

## File Structure

```
pyproject.toml                                     # root `senselab`: wheel packages = packages/senselab/src/senselab; torch extra
.importlinter                                      # senselab above senselab_core.runtime; core and worker never import senselab
packages/senselab/
  src/senselab/
    __init__.py                                    # facade: tasks, describe, models, run_call; lazy sl.audio, sl.text, sl.signal
    _api/                                          # generated task classes and functions (committed; drift-checked)
    audio/tasks/speech_to_text/{huggingface,crisperwhisper}.py   # moved from src/, then ported
    audio/tasks/speaker_diarization/pyannote.py    # moved, ported
    audio/tasks/speaker_embeddings/speechbrain.py  # moved, ported
    text/tasks/pii_detection/redaction_review.py   # moved, ported (vLLM + transformers)
    signal/resample.py                             # DSP primitive
    declarations.py                                # @task declarations of the Phase 2 tasks
    data/catalogue.yaml                            # catalogue@1: models, pins, licenses, backends, stacks
    data/env_locks/<env>.txt                       # moved from src/senselab/utils/data/venv_locks/
    data/prov/senselab-audio.jsonld                # vocabulary terms the ported tasks add
    data/tolerance/<backend>.yaml                  # tolerance@1 profiles with derivations
    data/env_grouping.yaml                         # env-grouping proposal (recorded, not adopted)
  tests/
packages/senselab-core/src/senselab_core/
  kernel/declare.py                                # @task/@backend, TaskParams, BackendParams, Env kinds, json_schema()
  kernel/time.py                                   # rebind(span, from_anchor, to_anchor, timemap)
  kernel/prov.py                                   # PROV record types, vocabulary model (no I/O)
  runtime/provenance.py                            # ProvStore: PROV-O JSON-LD writer/reader (provenance@3.0)
  runtime/bep028.py                                # BEP028 profile check and file layout (no translation)
  runtime/data/prov/{context,senselab}.jsonld      # published @context and core vocabulary
  runtime/catalogue.py                             # Catalogue, CatalogueEntry
  runtime/profiles.py                              # ExecutionProfile, resolve_execution()
  runtime/envs.py                                  # moved subprocess_venv + uv_tools + venv_lock: build/lock/locate envs
  runtime/vllm_server.py                           # moved; VllmServer behind Env.server
  runtime/call.py                                  # run_call(): resolve, place, env, cache, store, provenance
  runtime/licenses.py                              # SPDX parse, vendored table, derived flags
packages/senselab-worker/src/senselab_worker/
  model_cache.py                                   # ModelCache: LRU by (backend, model identity, device, dtype) under a budget
benchmarks/
  harness.py, manifests/<dataset>.yaml, suites/smoke.yaml, metrics/   # outside pytest
scripts/check_source_digests.py, scripts/generate_task_api.py, scripts/env_grouping.py
```

Tasks 1–12 are core-only and land before any port (13–17). Ports land before licenses-and-grouping (18), the
benchmark harness (20, 21) and parity (22). The pydra tasks (24–31) run in parallel with everything from Task 1 on.

---

### Task 0: P-39 reconciliation at Phase 2 start

**Model:** haiku implements (Explore, effort lower); sonnet reviews. **Tier:** none. **Branch:** `2x/phase-2-spec`.

- [ ] **Step 1 (local):** `git fetch origin alpha && git log <mark>..origin/alpha --oneline -- src scripts tutorials`
  for each ledger row.
- [ ] **Step 2:** In the code worktree, merge `origin/alpha` into `2x/phase-2` (no edit under `src/`), push. Set each
  mark to the merged `alpha` commit; for each commit touching an area Phase 2 ports, add a row naming the port task
  that carries it.
- [ ] **Step 3:** Commit `docs(2x): P-39 reconciliation at Phase 2 start` on `2x/phase-2-spec`; push.

### Task 1: The 2.x `senselab` package and its import contracts

**Model:** sonnet implements; opus reviews (layering). **Tier:** logic, contract. **Implements:** design §5, ADR-009,
P-13, ruling P2-1. **Depends on:** Task 0.

**Files:** Create `packages/senselab/src/senselab/__init__.py`, `packages/senselab/tests/`. Modify `pyproject.toml`
(wheel `packages`, `dev-mode-dirs`, `[project.optional-dependencies] torch`), `.importlinter`,
`.github/workflows/2x-tests.yaml`, `packages/conftest.py`.

**Interfaces:**
- `import senselab as sl` resolves to `packages/senselab/src/senselab`; `src/` is on no 2.x path.
- `.importlinter`: `packages-not-1x` restated as "`senselab_core` and `senselab_worker` never import `senselab`";
  the `layers` contract gains `senselab` above `senselab_core.cli | senselab_core.testing`; `senselab` (without the
  `torch` extra) imports no model framework at module level (`backends-import-lazily`).
- CI: `core-only` runs `packages/senselab-core packages/senselab-worker`; a new `senselab-lean` job (3.11, 3.14)
  installs the root distribution without extras from `uv.lock` and runs `packages/senselab/tests -m "logic or
  contract"`.

- [ ] **Step 1: Failing tests (local).** `test_senselab_resolves_to_packages` (`senselab.__file__` under
  `packages/senselab/`); `test_declarations_import_with_core_only` (subprocess import of `senselab` loads no torch,
  transformers, speechbrain, pyannote, vllm). `uv run pytest packages/senselab/tests -m "logic or contract"` fails.
- [ ] **Step 2: Implement**, then `uv sync --locked --group test` (local) and rerun; `uv run lint-imports`.
- [ ] **Step 3:** Review, compact, commit `feat(2x): the 2.x senselab package and its contracts`; push; CI green.

### Task 2a: Phase 1 hardening carried forward: worker and wire

**Model:** opus implements (worker protocol); fresh sonnet reviews. **Tier:** logic, contract. **Implements:**
R-146, R-147, R-158 (wire and worker groups), R-159, R-123. **Depends on:** Task 1.

**Files:** `senselab_worker/{faults,codec,serve}.py`, `senselab_core/runtime/{faults,worker_client}.py`, tests;
`packages/senselab-core/tests/bounded.py` (the one shared bounded helper).

**Interfaces:** `is_transient` walks a bounded cause chain and classifies SSL errors by type; an array reference
accepts exactly `{path, sha256, dtype, shape}` with integer shape elements; `ShutdownGraces` refuses overflow with
`ValueError`; `_has_exited` folded into the single exit wait; `max_line_bytes` bounds responses too; the scripted
fixture raises a typed error when replies run out.

- [ ] **Step 1: Failing tests (local)**, one per row, named by invariant: `test_cause_cycle_terminates`,
  `test_array_ref_schema_is_strict`, `test_graces_overflow_is_value_error`, `test_response_line_bounded`.
- [ ] **Step 2: Implement**; fold the eight wire duplicate groups of R-158 or record in the register why one stays;
  move the rationale prose of R-159 to `wire-protocol.md` (spec commit).
- [ ] **Step 3 (local):** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `fix(2x): worker and wire residuals`; push.

### Task 2b: Phase 1 hardening carried forward: store, leases, fetch, retry

**Model:** opus implements (store and leases); fresh sonnet reviews. **Tier:** logic, contract. **Implements:**
R-148, R-151, R-152, R-153, R-155, R-125, R-127, R-158 (store and kernel groups). **Depends on:** Task 1.

**Interfaces:** run ids over `NAME_MAX` refused at validation; `collect` covers run-directory, root-level and
in-place set-aside objects; a displaced lease re-reads and merges the manifest under the lease; the stat fast path
compares ctime; `equivalence.yaml` refuses duplicate keys and empty derivations; a re-`put` verifies by recorded
digests; fetch removes superseded version links and uses `faults.is_transient` with the profile's `RetryPolicy`.

- [ ] **Step 1: Failing tests (local)**, including `test_displaced_lease_keeps_manifest_update` under a planted
  delay at the `os` boundary (`syscall_faults_fixture`), and `test_equivalence_duplicate_key_refused`.
- [ ] **Step 2: Implement.** ADR-003 gains no text; the group-member limit and the by-path reads are stated in the
  register row's resolution.
- [ ] **Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/{store,locks,fetch,equivalence,retry}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `fix(2x): store, lease and fetch residuals`; push.

### Task 3: PROV-O JSON-LD store and the `senselab:` vocabulary (ADR-010)

**Model:** opus implements (identity and provenance); fresh opus reviews. **Tier:** logic, contract.
**Implements:** ADR-010 as amended, D-4, R-126, R-149, R-150; rulings P2-4..P2-7. **Depends on:** Task 2b.

**Files:** Create `kernel/prov.py`, `runtime/data/prov/{context,senselab}.jsonld`. Rewrite `runtime/provenance.py`.
Modify `runtime/store.py` (writes PROV records), `kernel/formats.py` (`provenance@3.0`, `senselab-vocab@1.0`,
`prov-activity@1` tag), `testing/provenance.py`. Delete `testing/values.py` parts that serve the token families.

**Interfaces:**
- Records are `prov:Entity`, `prov:Activity`, `prov:Agent` and subtypes (`prov:Plan`, `prov:Collection`,
  `prov:SoftwareAgent`); core terms only: `senselab:Run`, `senselab:SoftwareAgent`, `senselab:Environment`,
  `senselab:File`, `senselab:Plan`.
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
- Persistence: each writer replaces `prov/<writer-uuid>.jsonld` atomically; a run's provenance is the union of its
  writers' documents by `@id`, and two different nodes under one `@id` are refused (ruling P2-4). Accessors return
  in `@id` order, so `merge` is order-independent (R-149).
- Removed: `$path`, `$enum`, `$dec`, `$bytes`, `$set`, `EnumMember`, `FrozenDict`, `revised`, `recorded`,
  `plain_values`, `PROV_TYPE`, the bare relation tuples, `provenance@2.0` and its JSONL reader (a `provenance@2`
  file is refused with `ProvenanceFormatRefused`).

- [ ] **Step 1: Failing tests (local):** `test_one_key_two_contexts_two_activities`, `test_reuse_is_an_activity`,
  `test_unknown_term_refused`, `test_value_out_of_range_refused`, `test_minus_inf_round_trips_as_xsd_double`,
  `test_writers_union_by_id`, `test_conflicting_node_refused`, `test_merge_is_order_independent`,
  `test_provenance2_refused`; contract `test_document_expands_with_pyld` (expanded terms resolve to `prov:` or the
  published vocabulary; no relative IRI survives).
- [ ] **Step 2: Implement.** Hash JCS of the compact record, never an expansion (ADR-010).
- [ ] **Step 3 (local):** `uv sync --locked --group test`; `uv run pytest packages/senselab-core/tests/runtime/{provenance,provenance_checksum,store}_test.py packages/senselab-core/tests/kernel -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): provenance is PROV-O JSON-LD`; push. Spec commit: ADR-006
  format table and ADR-010 amendment log line for `provenance@3.0`.

### Task 4: Export redaction and the BEP028 profile

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** ADR-007 (amended), ADR-010
(BEP028 is a profile, not a translation), ruling P2-8. **Depends on:** Task 3.

**Files:** Rewrite `runtime/bep028.py` as profile check plus file layout; modify `runtime/provenance.py`,
`runtime/store.py`, `runtime/fetch.py` (file entities by label).

**Interfaces:** a file entity carries its content address and a per-run label `file-<n>`; the label→path map is
written to `<run>/private/labels@1.json` (mode 0600) and never exported; `export(run, dest)` copies the documents and
checks the BEP028 profile (`check_profile(doc) -> list[str]`).

- [ ] **Step 1: Failing tests (local):** `test_document_holds_no_path` (property: random path components never occur
  in any written document, error or log); `test_label_map_is_private`; `test_export_conforms_to_bep028`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/{bep028,provenance}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): exports are redacted by construction`; push.

### Task 5: Declarations: `@task`, `@backend`, `Params`, `backend_params`, JSON Schema

**Model:** opus implements (kernel identity); fresh opus reviews. **Tier:** logic, contract. **Implements:** design
§4.2, §4.4, ADR-001 (ports), ADR-002 §1; R-55, R-131, R-56, R-57, R-58, R-122; ruling P2-9. **Depends on:** Task 1.

**Files:** `kernel/declare.py`, `testing/contracts.py`, `runtime/fake_backend.py`, tests.

**Interfaces:**
- `@task(name, version, kind)` over a class with `Params(TaskParams)`, `inputs`, `outputs` (port name → kernel
  type); `@backend(task, stack, models, env, devices, batch_invariant, tunables, output_affecting, worker,
  params=BackendParams subclass)`.
- `Env.host()`, `Env.isolated(name)`, `Env.server(name)` replace the free `env: str`.
- `BackendSpec.worker` names the module serving `load`/`handle`; the generated contract cases run any backend with a
  worker module (R-57). `NOT_YET_WIRED` is gone.
- `key_for` takes `backend_params` (validated against the chosen backend) and the declared output-affecting
  components; `key_ignores_<field>` builds keys through the same builder `run_call` uses, under two execution
  contexts (R-56).
- `VERDICT_TYPES` is derived from the kernel types marked as decisions, not a literal (R-58).
- `json_schema(task)` emits JSON Schema 2020-12 (ruling P2-9) for inputs, params, backend params and outputs;
  a contract validates every generated schema under the 2020-12 metaschema.

- [ ] **Step 1: Failing tests (local):** `test_unknown_param_suggests`, `test_backend_params_validated_per_backend`,
  `test_output_affecting_enters_key_only_when_declared`, `test_key_ignores_context_through_builder`,
  `test_schema_validates_2020_12`, `test_real_backend_gets_contract_cases` (fake backend declared through
  `@backend(worker=...)`).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/kernel packages/senselab-core/tests/testing -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): task and backend declarations`; push. Spec commit: pin the
  dialect in `wire-protocol.md`; close R-55, R-131.

### Task 6: Generated API and `describe`

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.2, §4.10, P-1;
ruling P2-10. **Depends on:** Task 5.

**Files:** `scripts/generate_task_api.py`, `packages/senselab/src/senselab/_api/`, `senselab/__init__.py`.

**Interfaces:**
- Per task: a configured immutable class (`SpeechToText(model=..., **params)`, `.with_params()`, `__call__(x)` →
  one result through `run_call`) and the function `speech_to_text(x, **params)` ≡ `SpeechToText(**params)(x)`;
  `.map` waits for Phase 3.
- Generated at development time into committed modules from the declarations (real signatures for IDEs and pdoc);
  a contract test regenerates and refuses drift.
- `sl.tasks(accepts=None)`, `sl.describe(name)` (the task's JSON Schema plus models, backends, stacks, devices,
  licenses, default model), `sl.models(task=None)`.
- One call shape: `sl.run_call({"task", "inputs", "params", "model"})`; errors use one envelope `{code, message,
  fix, field, suggestions}`.

- [ ] **Step 1: Failing tests (local):** `test_function_equals_configured_call` (fake task),
  `test_generated_api_has_no_drift`, `test_describe_is_valid_schema`, `test_call_from_describe_alone` (a call built
  only from `describe()` JSON runs on the fake stack).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab/tests -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): generated task API and describe`; push.

### Task 7: Runtime catalogue and license records

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.3, §4.13, P-23,
R-88, R-116 (flags and closed `how`); rulings P2-11, P2-12. **Depends on:** Task 5.

**Files:** `runtime/catalogue.py`, `runtime/licenses.py` (with a vendored SPDX table as data), `kernel/license.py`,
`packages/senselab/src/senselab/data/catalogue.yaml`.

**Interfaces:**
- `catalogue@1` entry: `{id, revision (40-hex or 64-hex), tasks, backends, stacks, license, resources: {}}`;
  `Catalogue.default(task)`, `Catalogue.entry(model_id)`. Resource tables stay empty until the Phase 3 scale ladder.
- License flags are derived from a parsed SPDX expression against the vendored table; `verified.how` is one of
  `hf-card | pep639 | manual`; a free flag that contradicts the expression is refused.
- The model-source allowlist (ADR-007) defaults to the hub organisations of catalogue entries.

- [ ] **Step 1: Failing tests (local):** `test_flags_derived_from_spdx`, `test_malformed_expression_refused`,
  `test_test_catalogue_entries_are_pinned`, `test_unlisted_source_refused`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/catalogue_test.py packages/senselab-core/tests/kernel/license_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): runtime catalogue and SPDX-derived licenses`; push.

### Task 8: Minimal execution-profile resolver

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.6 (execution
profile), §4.4 tunables, P-2, P-7, P-8, P-18, ADR-007 allowlist, R-127; ruling P2-13. **Depends on:** Task 7.

**Files:** `runtime/profiles.py`, `runtime/data/profiles/{laptop,orcd}.yaml`.

**Interfaces:** `execution-profile@1` with `stack_preferences: {task: [stack, ...]}`, `devices`, `env_roots`,
`model_sources`, `model_cache_budget_bytes`, `retry`, `network`; packaged `laptop` default, an `orcd` site profile;
user YAML deep-merged; the merged mapping's digest is recorded in the execution context, never keyed.
`resolve_execution(task, profile, hardware) -> Placement` picks backend, stack, env and device, with
`Placement.explain()`. A GPU-only backend with no GPU class fails with the node named; no silent CPU fallback.

- [ ] **Step 1: Failing tests (local):** `test_stack_preference_picks_backend`, `test_profile_digest_not_in_key`,
  `test_gpu_only_without_gpu_fails`, `test_unknown_profile_key_suggests`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/profiles_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): minimal execution-profile resolver`; push.

### Task 9a: Env builder (`Env.host`, `Env.isolated`)

**Model:** opus implements (concurrency, leases); fresh sonnet reviews. **Tier:** logic, contract; env on ORCD.
**Implements:** design §4.4, P-6, P-25, ADR-003 (D-2 leases), the subprocess-venv CUDA routing. **Depends on:**
Task 8.

**Files:** `git mv src/senselab/utils/{subprocess_venv,uv_tools,venv_lock}.py` and
`src/senselab/utils/data/venv_locks/` (rename-only commit), then modify into `runtime/envs.py` and
`packages/senselab/src/senselab/data/env_locks/`.

**Interfaces:** `ensure_env(env, lock) -> EnvHandle{python, lock_sha, wheel_set}` builds under a D-2 lease (no
`FileLock`, no heartbeat), installs the worker runtime as an overlay, keeps the CUDA-aware torch index routing and
`SENSELAB_TORCH_INDEX_URL`, records the env in the execution context; `Env.host()` is refused unless the root lock
covers the backend's requirements. Digest sites from the migration ledger's H-R20 table in these modules use
`identity_digest` with registered tags.

- [ ] **Step 1: Failing tests (local):** `test_concurrent_build_one_install` (two processes, planted delay at the `os`
  boundary, no real install: the installer is the `Resolver`-style seam), `test_host_refused_when_lock_lacks_backend`,
  `test_cuda_index_routing` (table-driven over probe outputs).
- [ ] **Step 2: Implement.** Ledger: mark the moved modules, list unported functions (ruling P2-3).
- [ ] **Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/envs_test.py -m "logic or contract"`.
- [ ] **Step 4 (orcd, CPU node3805, pinned SHA):** `-m env -k envs` builds the `crisperwhisper` lock and round-trips
  every wire data type through its worker.
- [ ] **Step 5:** Review, compact, commit `feat(2x): env builder under leases`; push.

### Task 9b: `VllmServer` and `Env.server`

**Model:** opus implements (service lifecycle); fresh sonnet reviews. **Tier:** logic, contract; limits on ORCD GPU.
**Implements:** design §4.4, `execution.md` §3.5, I-8, P-37 (vLLM server row); ruling P2-14. **Depends on:** Task 9a.

**Files:** `git mv src/senselab/utils/vllm_server.py` then modify into `runtime/vllm_server.py`.

**Interfaces:** `VllmServer(env, model: ResolvedModel, settings)` as a context manager: starts in its `Env.server`
env, loads by commit, health-checks, attests `{engine_version, model_identity, config_digest}` on every response
(I-8), stops with the owning process. Phase 2 has no cross-process service registry (ruling P2-14).

- [ ] **Step 1: Failing tests (local):** against a stdlib stand-in server process: `test_attestation_enters_context`,
  `test_restart_with_new_config_changes_attestation`, `test_server_stops_with_owner`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/vllm_server_test.py -m "logic or contract"`.
- [ ] **Step 4 (orcd, GPU H100, pinned SHA):** `-m "limits and gpu" -k vllm_server` with the catalogue's PII model.
- [ ] **Step 5:** Review, compact, commit `feat(2x): VllmServer behind Env.server`; push.

### Task 10: Model cache, worker protocol for backends, and `run_call`

**Model:** opus implements (worker protocol, identity); fresh opus reviews. **Tier:** logic, contract.
**Implements:** design §4.4 (one worker protocol, one model cache), §4.12, ADR-002, ADR-009 (typed faults), P-5,
P-8; rulings P2-15, P2-16, P2-17. **Depends on:** Tasks 3, 6, 8, 9a, 9b.

**Files:** `senselab_worker/model_cache.py`, `senselab_worker/serve.py`, `runtime/call.py`, `runtime/worker_client.py`.

**Interfaces:**
- `ModelCache(budget_bytes)`: `get_or_load((backend, model_identity, device, dtype), loader, size_of)`, LRU under the
  budget; one implementation used in-process (`Env.host`) and inside workers.
- Backend modules keep the wire contract (`load(init)`, `handle(state, items, params)`); `Env.host` calls the same
  `handle` in-process, so a backend has one code path for both env kinds.
- `run_call(call, *, profile, store, prov) -> Result`: resolve the model (catalogue pin or `resolve()`), place it
  (`resolve_execution`), open or reuse the env and worker, build the key with `key_for`, `store.get` under the reuse
  policy, else run, `store.put`, write PROV records; sets `ExecutionContext.model_source` (ruling P2-17). Phase 3's
  `InProcess` executor replaces it; nothing wraps it.

- [ ] **Step 1: Failing tests (local, fake stack):** `test_model_loaded_once_across_calls`,
  `test_cache_evicts_under_budget`, `test_stack_switch_by_profile`, `test_second_call_hits_store`,
  `test_provenance_written_per_call`, `test_host_and_isolated_share_handle` (fake backend in both env kinds).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/call_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): model cache and run_call`; push.

### Task 11: Kernel time and identity follow-ups

**Model:** opus implements (kernel); fresh sonnet reviews. **Tier:** logic, contract. **Implements:** R-114 (H-R59),
R-109, R-118; rulings P2-21, P2-22. **Depends on:** Task 2b.

**Interfaces:**
- `rebind(output, stored_anchor, reader_anchor, timemap)` in `kernel/time.py`: a cached clock-relative output is
  re-bound to the reader's anchor when served; never keyed by anchor.
- Source stamps record per-chunk digests over the `content_identity` chunk grid (64 MiB); a windowed read verifies
  only the chunks it touches; a `cp -p` rewrite inside a read chunk still raises `StaleSource`; stamp format bump.
- Local model identity hashes every regular file under the root except VCS metadata directories (`.git`, `.hg`,
  `.svn`); symlink rules stay as R-135 fixed them.

- [ ] **Step 1: Failing tests (local):** `test_rebound_equals_reclocked_compute` (one content under a session anchor,
  then standalone, one computation), `test_window_read_hashes_touched_chunks_only` (bytes read counted at the `os`
  boundary), `test_rewrite_in_touched_chunk_is_stale`, `test_vcs_dirs_do_not_change_identity`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/kernel/time_test.py packages/senselab-core/tests/runtime/{audio_io,models}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): clock re-binding, chunked stamps, local identity rule`; push.
  Spec commits: ADR-005 amendment (whole-file conflict resolved by chunked stamps), ADR-002 amendment log line.

### Task 12: DSP primitive `sl.signal.resample`

**Model:** sonnet implements; opus reviews. **Tier:** logic (property), contract. **Implements:** design §4.15
tier 1, P-26, P-13; ruling P2-20. **Depends on:** Tasks 6, 10.

**Files:** move `src/senselab/audio/tasks/preprocessing/preprocessing.py` (rename-only) to the mirrored path, then
modify: `resample` moves to `packages/senselab/src/senselab/signal/resample.py`; the other functions are removed and
listed in the ledger (ruling P2-3).

**Interfaces:** task `resample` (`transform`, input `RegularSeries`, param `target_rate: int`), backend `numpy-sinc`
(stack `dsp`, `Env.host`, no model): rational-ratio Kaiser-windowed sinc, generic over channel groups.

- [ ] **Step 1: Failing tests (local):** Hypothesis: duration preserved within one output sample; band-limited tone
  preserved within a measured band; channel-order and gain invariance; contract cases generated for the backend.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab/tests/signal -m "logic or contract"`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): sl.signal.resample`; push.

### Task 13: Port HF Whisper on transformers (host)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits on ORCD. **Implements:** design §6.2,
P-4, P-8, P-16; ruling P2-18. **Depends on:** Tasks 7, 10, 12.

**Files:** P-39 check; `git mv src/senselab/audio/tasks/speech_to_text/huggingface.py` (and the parts of `api.py`
it needs) to the mirrored path; modify into a `@backend(task="speech_to_text", stack="transformers",
env=Env.host())` with a worker-contract module; declaration `SpeechToText` in `declarations.py`; catalogue entry
`openai/whisper-large-v3-turbo` pinned by commit; root lock `torch` extra.

**Interfaces:** output `Transcript` (kernel type) with word spans in integer ns; timestamps per `Params.timestamps`;
typed faults (`ContentFault` for undecodable input, `ResourceFault` for OOM); the torch import stays inside `load`.

- [ ] **Step 1: Failing tests (local, contract):** generated contract cases for the backend (no weights: `load` is
  not called); `test_no_torch_import_at_module_level`.
- [ ] **Step 2: Implement.** Ledger mark and unported functions.
- [ ] **Step 3 (orcd, CPU node3805 then GPU H100, pinned SHA):** `-m limits -k whisper_transformers`: the generic
  edge set (empty, one-sample, silence, clipping, long input, each rate), zero gaps over fault fixtures,
  `content_fault_isolated` on the real backend, batch of 1 vs mixed batch within tolerance or
  `batch_invariant=False`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): Whisper on transformers`; push.

### Task 14: Port CrisperWhisper on ct2 and transformers (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** design
§6.2, P-6, P-8. **Depends on:** Task 13.

**Files:** P-39 check; `git mv src/senselab/audio/tasks/speech_to_text/crisperwhisper.py`; modify into two backends
over one module (`stack="ct2"` and `stack="transformers"`, both `Env.isolated("crisperwhisper")`), sharing the
decoding and word-confidence code (P-37); the ct2 conversion cache keyed by `identity_digest`.

- [ ] **Step 1 (local):** generated contract cases for both backends; `test_two_stacks_one_output_schema`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3 (orcd, CPU then GPU H100, pinned SHA):** `-m "limits or env" -k crisperwhisper`; the two stacks agree
  within the measured tolerance (recorded in Task 22).
- [ ] **Step 4:** Review, compact, commit `feat(2x): CrisperWhisper on ct2 and transformers`; push.

### Task 15: Port pyannote diarization (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Implements:** design
§6.2. **Depends on:** Task 10.

**Files:** P-39 check; `git mv src/senselab/audio/tasks/speaker_diarization/pyannote.py`; modify into
`@backend(task="diarize", stack="pyannote", env=Env.isolated("pyannote"))`; output `Intervals` of speaker turns;
lock pins a `lightning` version outside the quarantined 2.6.2/2.6.3; gated model read with `HF_TOKEN`.

- [ ] **Step 1 (local):** generated contract cases; **Step 2:** implement;
  **Step 3 (orcd, CPU then GPU H100, pinned SHA):** `-m "limits or env" -k pyannote`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): pyannote diarization`; push.

### Task 16: Port speechbrain speaker embeddings (isolated)

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits and env on ORCD. **Depends on:** Task 10.

**Files:** P-39 check; `git mv src/senselab/audio/tasks/speaker_embeddings/{speechbrain,windowing}.py`; modify into
`@backend(task="speaker_embeddings", stack="speechbrain", env=Env.isolated("speechbrain"))`; output `Embedding`
with `space` = the resolved model identity; batches formed by duration bucket, declared `output_affecting:
{batch_formation}` unless the limits test certifies batch invariance (I-3).

- [ ] **Step 1 (local):** generated contract cases; **Step 2:** implement;
  **Step 3 (orcd, CPU then GPU H100, pinned SHA):** `-m "limits or env" -k speechbrain`, including duration-matched
  controls for short spans.
- [ ] **Step 4:** Review, compact, commit `feat(2x): speechbrain speaker embeddings`; push.

### Task 17: Port text PII review on vLLM and transformers

**Model:** sonnet implements; opus reviews. **Tier:** contract local; limits on ORCD. **Implements:** design §6.2,
P-8, P-37 (second vLLM server folded), P-12, P-17; ruling P2-19. **Depends on:** Tasks 9b, 10.

**Files:** P-39 check; `git mv src/senselab/text/tasks/pii_detection/{redaction_review,redaction_review_vllm,redaction_policy}.py`
and its `data/`; modify into task `pii_review` (`measure`: findings with spans, categories and scores; no verdict)
with backends `stack="vllm"` (`Env.server("pii-review")`, through `VllmServer`) and `stack="transformers"`
(`Env.isolated("pii-review")`), one prompt and parsing engine; dataset-specific guidance and policy text become a
profile slice with a derivation (P-12), keyed as `profile:`.

- [ ] **Step 1 (local):** generated contract cases for both backends; `test_prompt_built_from_profile_slice`;
  `test_no_second_server_implementation` (the module imports `VllmServer`, holds no process management).
- [ ] **Step 2: Implement.** Ledger: unported PII functions (rules, local LLM, subprocess backend) listed.
- [ ] **Step 3 (orcd, GPU H100 for vLLM, CPU node3805 and H100 for transformers, pinned SHA):** `-m limits -k pii_review`.
- [ ] **Step 4:** Review, compact, commit `feat(2x): PII review on vLLM and transformers`; push.

### Task 18: License backfill, Linux licenses job, env-grouping proposal

**Model:** sonnet implements; opus reviews. **Tier:** contract; env on ORCD. **Implements:** design §4.13 (CI gate
equals run-time default), §4.11 env grouping, §6.4, R-116 (Linux job); ruling P2-27. **Depends on:** Tasks 13–17.

**Interfaces:** every catalogue model and every locked package of every shipped env (transitive) has a license
record; unknowns need a dated manual record; the `licenses` CI job runs on Linux; `scripts/env_grouping.py`
resolves pairwise unions of backend locks and writes `data/env_grouping.yaml` (`env-grouping@1`), recorded, not
adopted (Phase 4).

- [ ] **Step 1 (local):** `uv run pytest packages/senselab-core/tests/lint/check_licenses_test.py -m "logic or contract"`
  over the new catalogue and locks.
- [ ] **Step 2 (orcd, CPU node3805, pinned SHA):** `uv run --no-sync python scripts/env_grouping.py`; commit the
  proposal file.
- [ ] **Step 3:** Review, compact, commit `feat(2x): license backfill and env-grouping proposal`; push.

### Task 19: CI and process items

**Model:** sonnet implements; opus reviews. **Tier:** contract. **Implements:** R-115, R-113 (H-R54), R-117 (P-40),
R-133, R-134, R-157; ruling P2-29. **Depends on:** Task 13.

**Interfaces:**
- `scripts/check_source_digests.py`: recomputes each task's and backend's source digest and fails when it changed
  without a `version` bump, against the digests recorded in `packages/senselab/src/senselab/data/source_digests.yaml`
  at the last release (first release: the file is created).
- `2x-nightly.yaml` on `alpha` (a `skip-release` PR) running limits and use cases against `2.x`; its first green run
  recorded in `STATE.md`.
- The 2.x PR template carries implementer and reviewer model fields; a check refuses equal values.
- Each list-driven guard test of R-133 is kept only if it guards a design decision and duplicates no other check;
  size budgets get a written derivation (ruling P2-29).

- [ ] **Step 1 (local):** tests for the digest check (a source edit without a bump fails; with a bump passes).
- [ ] **Step 2: Implement**; open the `alpha` workflow PR with `skip-release`.
- [ ] **Step 3:** Review, compact, commit `ci(2x): source digests, nightly, PR model fields`; push.

### Task 20: Benchmark harness, manifests and smoke suite

**Model:** sonnet implements; opus reviews. **Tier:** logic (harness on the fake stack); benchmarks on ORCD.
**Implements:** `testing.md` §6, `benchmarks.md`, docs.md (first fetch consumer), H-R49, P-30, P-41; rulings P2-28,
P2-25, P2-26. **Depends on:** Tasks 12–17.

**Files:** `benchmarks/harness.py`, `benchmarks/manifests/{librispeech-test-clean,voxconverse-v0.3,gretel-synthetic-pii}.yaml`,
`benchmarks/suites/smoke.yaml`, metric tasks `wer`, `der`, `eer`, `span_f1`, `resample_error` declared as senselab
tasks.

**Interfaces:** `senselab bench run <suite> --execution <profile>` fetches each manifest through `senselab data
fetch`, runs `run_call` per item, scores with the metric tasks (implementations checked against the reference
scorers: the HF Open ASR normaliser, pyannote metrics), and writes results with full provenance to
`$SENSELAB_CACHE/2x/benchmarks/` (`benchmark-result@1`). Smoke items: LibriSpeech test-clean 50 utterances (ASR),
VoxConverse test 5 files (diarization), LibriSpeech test-clean speaker trials (speaker embeddings, ruling P2-25),
gretel synthetic PII 200 documents once its README is read and its terms recorded (ruling P2-26), synthetic tones
(resample).

- [ ] **Step 1 (local):** harness logic on the fake stack and a toy manifest: `test_harness_records_provenance`,
  `test_metric_matches_reference_rows` (small tables per metric).
- [ ] **Step 2: Implement.**
- [ ] **Step 3 (orcd, CPU node3805, pinned SHA):** `senselab bench run smoke --execution orcd`, under 15 min.
- [ ] **Step 4 (orcd, GPU H100, pinned SHA):** the same suite on GPU.
- [ ] **Step 5:** Review, compact, commit `feat(2x): benchmark harness and smoke suite`; push.

### Task 21: Verify sampling and per-backend equivalence

**Model:** opus implements (identity, reuse policy); fresh sonnet reviews. **Tier:** logic, contract; limits on
ORCD. **Implements:** ADR-002 §3, §4, R-112, I-4; ruling P2-23. **Depends on:** Tasks 10, 20.

**Interfaces:** `equivalence@2`: classes per backend, keyed by numerics class, each with a derivation;
`run_call(..., verify=k)` recomputes a fraction `k` of cross-context hits and compares within the backend's
tolerance profile; agreements append evidence, disagreements split the class and are reported; `equivalent` with no
class behaves as `exact`.

- [ ] **Step 1 (local, fake stack):** `test_unmeasured_equivalent_is_exact`, `test_verify_agreement_adds_evidence`,
  `test_verify_disagreement_splits_class`, `test_equivalence1_refused`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run pytest packages/senselab-core/tests/runtime/{equivalence,call}_test.py -m "logic or contract"`.
- [ ] **Step 4 (orcd, CPU node3805 and GPU H100, pinned SHA):** a smoke run with `verify: 0.1` across the two
  contexts records evidence.
- [ ] **Step 5:** Review, compact, commit `feat(2x): verify sampling and per-backend equivalence`; push. Spec commit:
  ADR-002 amendment (classes per backend by numerics class; verify sampling owned).

### Task 22: Parity with 1.x and tolerance profiles

**Model:** sonnet implements; opus reviews. **Tier:** limits (parity, migration only) on ORCD. **Implements:**
`testing.md` §4 parity, P-17, design §6.2 exit; ruling P2-24. **Depends on:** Tasks 13–17, 20.

**Interfaces:** reference outputs come from the 1.x code at the merged `alpha` commit, run on ORCD from its own
pinned checkout over the smoke items, stored as benchmark artifacts (never committed); `tests/parity/` compares 2.x
outputs with them; `data/tolerance/<backend>.yaml` (`tolerance@1`) are generated from the measured spread with a
derivation, never hand-edited. vLLM parity is CUDA only; the PII CPU parity runs on the transformers stack.

- [ ] **Step 1 (orcd, CPU node3805 and GPU H100):** produce 1.x references at the pinned `alpha` SHA.
- [ ] **Step 2 (orcd, same hosts, 2.x pinned SHA):** `-m limits packages/senselab/tests/parity`; generate the
  tolerance profiles; rerun to green.
- [ ] **Step 3:** Review, compact, commit `test(2x): parity with 1.x within measured tolerances`; push.

### Task 23: Docs for Phase 2 and the spec pass

**Model:** sonnet implements; opus reviews. **Tier:** contract (docs build). **Implements:** docs.md Phase 2, P-41,
R-130, D-7. **Depends on:** Tasks 6, 20.

**Interfaces:** reference pages generated from declarations (schemas, default models, stacks, licenses); the 2.x
status page (each task declared, implemented, ported or benchmarked; phase exits; consolidation inventory; ledger
marks; open register rows) and per-page badges in the PR preview. Spec commits on `2x/phase-2-spec`: design §6.1
rewritten for merge-based reconciliation, the ledger trimmed, the 14 unclassified items classified, and the ADR
amendments the tasks above name, each dated.

- [ ] **Step 1 (local):** `uv run --group docs pdoc` builds; the status page test reads the registry.
- [ ] **Step 2:** Review, compact, commit `docs(2x): Phase 2 reference and status page`; push.

### pydra gate prerequisites (Tasks 24–31)

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
| 28 | **A1** Slurm requeue budget, `CANCELLED` (pulled forward, ruling P2-30) | sonnet / opus | a requeue budget; a user's `CANCELLED` is not requeued; Slurm test on ORCD | — |
| 29 | **F1** `ComputeResources` + Slurm/SGE mapping | sonnet / opus | set by the define machinery, overridable per node, never in the checksum; mapped to `sbatch`/`qsub` args; Slurm test on ORCD | A1 |
| 30 | **F2** worker routing | sonnet / opus | `Submitter(workers={...}, router=...)` called at the three dispatch sites; definitions name no worker | F1 |
| 31 | **E1** per-element readiness (no re-split case) + `on_error` | opus, effort higher / fresh opus | a downstream element runs when its own upstream indices succeed; an errored element marks only its descendants; full splitter/combiner matrix and `debug` parity | — |

Each task: failing tests first in pydra's suite; implement; review and compaction by the named reviewer; push the
branch; open the draft PR; ORCD step (CPU node3805, the PR head SHA) runs the area's tests and, for A1 and F1,
submits a real Slurm job from the test; merge into `enh/execution-extensions`.

### Task 32: Phase 2 exit checks

**Model:** sonnet implements; opus reviews; the P-1 check uses a fresh sonnet agent with no repository access.
**Tier:** all. **Depends on:** Tasks 0–31.

- [ ] **Step 1 (CI):** `2x-tests` green at the head of `2x/phase-2`: `core-only` (3.11, 3.14), `senselab-lean`,
  `worker-floor` (3.10), `lint-imports`, pre-commit, `licenses` on Linux.
- [ ] **Step 2 (orcd, CPU node3805 and GPU H100, head SHA):** the full packages suite, every tier, serial; env builds
  for every shipped lock; the R-124 fd-0 probe (`strace -f -e trace=openat,getrandom` beside `readlink
  /proc/self/fd/0`), its result recorded in `wire-protocol.md` or the row closed with the negative result.
- [ ] **Step 3 (P-8, orcd):** each two-stack call (CrisperWhisper ct2/transformers, PII vLLM/transformers) switches by
  `stack_preferences` alone; the task call and output schema are unchanged; the context records the stack.
- [ ] **Step 4 (P-1, orcd):** the fresh agent receives only `sl.describe()` output for each task and a public input
  path, writes the call JSON, and `run_call` runs it.
- [ ] **Step 5:** parity green on CPU and on the H100 host (Task 22); the pydra prerequisites green on their draft
  PRs and merged into `enh/execution-extensions` (gate evidence for Phase 3's first task).
- [ ] **Step 6:** P-39 reconciliation again (Task 0's steps).

### Task 33: Phase 2 exit adversarial round (P-35)

**Model:** opus lead, effort higher, with up to 5 sonnet subagents; claims verified by haiku (P-40). **Tier:**
read-only. **Depends on:** Task 32.

- [ ] **Step 1:** Brief the lead with the Phase 2 surfaces: declarations and `Params`, `run_call`, the catalogue and
  pins, the profile resolver, envs and leases, `VllmServer`, the model cache, the PROV-O store and redaction,
  chunked stamps and re-binding, the ported backends, the benchmark harness and verify sampling, the pydra PRs.
  Goal: inputs, races or misuse that break ADR-002, 003, 006, 007, 009, 010 or the Phase 2 exit; plus the P-37
  duplicate-finding pass. Every finding carries an architectural resolution.
- [ ] **Step 2:** Findings become `review-register.md` rows from the next free id, with lens `EXT2`, on
  `2x/phase-2-spec`; findings saved under `reviews/phase2-exit-adversarial/`.
- [ ] **Step 3:** Commit `docs(2x): Phase 2 exit adversarial round`; push.

### Task 34: Phase 2 exit fix round

**Model:** per finding, from the roster (opus for kernel, concurrency and identity; sonnet otherwise); each fix
reviewed by a different model. **Depends on:** Task 33.

- [ ] **Step 1:** Fix every high and medium finding by an architectural change (P-35); schedule the rest with an
  owner in the register.
- [ ] **Step 2:** CI green; rerun the affected ORCD steps of Task 32 at the new head.
- [ ] **Step 3:** `STATE.md`: Phase 2 done, with the head SHA and ORCD job ids; log line. Commit `docs(2x): Phase 2
  done` on `2x/phase-2-spec`; push.

---

## Self-review notes

- **Spec coverage.** design §6.2 Phase 2: declarations, `Params`, generated functions and schemas → 5, 6;
  profile resolver and `Env.server` → 8, 9b; catalogue → 7; one worker protocol and model cache → 10; ports → 12–17;
  ADR-010 → 3, 4; benchmark harness, manifest, smoke → 20; license backfill and env grouping → 18; exit → 21, 22, 32.
  docs.md Phase 2 → 23. Gate prerequisites → 24–31. Register rows owned by Phase 2: R-55, R-56, R-57, R-58, R-83
  (per backend), R-88, R-109, R-112, R-113, R-114, R-115, R-116, R-117, R-118, R-122..R-127, R-130, R-131,
  R-146..R-153, R-155, R-158, R-159; R-133, R-134, R-157 taken in Task 19 because no phase owned them.
- **Not in Phase 2:** `Plan`, `.map`, `Pool`, Slurm substrates, the leased service registry and lease renewal
  (R-111), taint propagation (R-116, Phase 3), resource tables (scale ladder), `prov:hadPlan`.

## Rulings made in this plan

Each is the planner's choice where the design left one open; the owner reviews them. A ruling that changes an ADR
decision takes effect only through the dated amendment its task commits (ADR-000).

- **P2-1** ruling: The 2.x `senselab` import package lives at `packages/senselab/src/senselab/`, built by the root `senselab`
  distribution; `src/` is on no 2.x path; `packages-not-1x` becomes "core and worker never import `senselab`" — the root name is the 2.x umbrella, and the 1.x pool shares the import name.
- **P2-2** ruling: A moved file keeps its 1.x relative path under `packages/senselab/src/senselab/`; no second rename in
  Phase 2 — `git log --follow` and blame stay trivial, and the 1.x layout already matches `sl.<modality>`.
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
  with its declarations; the vocabulary is checked by a contract test, not the release gate, until the first
  release — D-5 keeps the release gate minimal, which ADR-010's "the release gate checks it" contradicts.
- **P2-8** ruling: Redaction is by construction: no document holds a path; labels map to paths only in
  `<run>/private/labels@1.json` (0600) — ADR-010 makes store and export one document.
- **P2-9** ruling: Task and backend schemas use JSON Schema 2020-12 — pydantic v2 emits it natively (R-55, R-131).
- **P2-10** ruling: The task API is generated at development time into committed modules, with a regeneration drift test — IDEs and pdoc need real signatures, from one source.
- **P2-11** ruling: Catalogue revisions are full commit SHAs or weights digests, never refs; a ref is accepted only in a
  user's call and resolved — P-16 for every test and benchmark path.
- **P2-12** ruling: The default model-source allowlist is the hub organisations of the catalogue entries — ADR-007 needs
  an allowlist, and the catalogue is the only source list that exists.
- **P2-13** ruling: `execution-profile@1` carries stack preferences, devices, env roots, model sources, the model-cache
  budget, retry and network policy; `laptop` and `orcd` are packaged — the minimum Phase 2 needs for P-8 and
  ADR-007.
- **P2-14** ruling: `Env.server` is process-scoped in Phase 2 (owned by the calling process); the leased service registry
  waits for lease renewal in Phase 3 — leases have no renewal (R-111), and a server outlives the stale bound.
- **P2-15** ruling: The model cache lives in `senselab_worker`, one implementation for host and workers — P-37 and design
  §4.12.
- **P2-16** ruling: `run_call` is Phase 2's single-call path; Phase 3's `InProcess` replaces it, nothing wraps it — Phase
  2's exits need execution before `Plan` exists.
- **P2-17** ruling: `run_call` sets `ExecutionContext.model_source` — it builds the context; R-110's Phase 3 work reduces
  to keeping that when `InProcess` replaces it.
- **P2-18** ruling: Only HF Whisper runs `Env.host` (the root lock's `torch` extra); every other ported backend is isolated,
  one env per backend — keeps the senselab venv small and exercises the one worker protocol.
- **P2-19** ruling: PII review is a `measure` task emitting findings; policy and guidance text are a profile slice — P-20
  and P-12; the 1.x module mixes dataset guidance into code.
- **P2-20** ruling: The DSP primitive is `sl.signal.resample`, numpy-only Kaiser-windowed sinc over rational ratios — core
  is torch-free and the algorithm is generic over signals (P-26).
- **P2-21** ruling: Whole-file hashing (R-109) is replaced by per-chunk stamps on the `content_identity` grid, verified for the
  chunks a read touches — keeps the `cp -p` guarantee and ADR-005's "never hashed whole".
- **P2-22** ruling: Local model identity excludes only VCS metadata directories (R-118) — incidental files are content;
  VCS state is not.
- **P2-23** ruling: `equivalence@2` keys classes per backend by numerics class — R-112; global fingerprints mix backends.
- **P2-24** ruling: Parity references are 1.x outputs produced on ORCD from the pinned `alpha` checkout, stored as benchmark
  artifacts; tolerances are generated from measured spread; vLLM parity is CUDA only — 2.x cannot import 1.x,
  and vLLM on CPU is not a supported context.
- **P2-25** ruling: The speaker-embedding smoke uses LibriSpeech test-clean trials — VoxCeleb1 needs a request.
- **P2-26** ruling: The gretel synthetic PII set enters the smoke suite only after its README is read and its terms recorded — `benchmarks.md` lists its README as unread.
- **P2-27** ruling: The env-grouping proposal is computed on ORCD by resolving pairwise unions of backend locks, recorded as
  `env-grouping@1`, not adopted — §6.4 defers the solver; resolution needs the network.
- **P2-28** ruling: No model weights load in GitHub CI in Phase 2; model-backed tiers and smoke benchmarks run on ORCD; the
  PR-triggered CPU smoke of `testing.md` is deferred — owner item 39 and the local and CI limits.
- **P2-29** ruling: List-driven guard tests (R-133) stay only where they guard a design decision and duplicate no other check;
  size budgets (R-134) are set from ruff's complexity and statement limits plus a file-length limit, with the
  derivation written — D-1 and owner item 14; no phase owned these rows.
- **P2-30** ruling: pydra A1 is pulled into Phase 2 — F1 depends on it in proposal 0001.
- **P2-31** ruling: Each pydra PR is its own branch and draft PR, merged into `enh/execution-extensions` for pinning — the
  maintainers review per PR; Phase 3 pins one commit.
- **P2-32** ruling: senselab takes no pydra dependency in Phase 2; the gate evidence is pydra's tests on its CI and ORCD — the gate decision is Phase 3's first task.

## Design gaps left open

- ADR-010 says the release gate checks the vocabulary; D-5 freezes the gate's scope until the first release. P2-7
  bridges it with a contract test; the owner decides whether ADR-010 is amended or D-5's deferred list grows.
