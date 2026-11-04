# senselab 2.x Phase 3 Implementation Plan: plans and executors

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a declared task over a dataset as an immutable, hashed plan, on a laptop and on a Slurm cluster, so that:
- `sl.plan(...)` returns a `LogicalPlan` (the run identity) and a `Placement` (recorded, never keyed), with `explain()`;
- `InProcess` and `Pool` executors behind one `Executor` protocol run it, the pool on local and Slurm substrates;
- plan-fragment items, an idempotent coordinator with an epoch file, renewed leases and faults typed at the source make
  an interrupted run resume from its plan hash and recompute only what is missing (P-5);
- cross-record batching obeys I-3, a minimal BIDS adapter feeds `.map`, and the scale ladder writes the catalogue
  resource tables;
- the pydra gate is decided first, and the Phase 3 pydra PRs land on `enh/execution-extensions`.

**Architecture:**
- `senselab_core.kernel` gains the graph document (`graph@1.0`), the `Dataset`/`Record` protocol and input manifest,
  and the `LogicalPlan` with its hash. It still does no I/O (ADR-009).
- `senselab_core.runtime` gains the planner (`planner.py`), the dataset adapters (`datasets.py`, `bids.py`), and the
  package `runtime/execution/`: the `Executor` protocol and `PlanFragment`, the job engine that replaces `run_call`,
  the `InProcess` and `Pool` executors, the run lifecycle and events (ADR-008), queues, pool workers and their
  supervisor, substrates, the service registry and batching.
- **The seam is the `Executor` protocol** (ADR-001). A graph runner walks the plan and hands each ready node's jobs to
  an executor as a `PlanFragment` (`submit`), then reads their outcomes (`results`). Under the gate's pass the runner
  is pydra's `Submitter` over the compiled workflow, with a senselab worker plugin whose `run(job)` submits a fragment;
  under its failure it is senselab's `NativeRunner`. Only the gate-selected runner is built (rulings P3-2, P3-3).
- A single call (`ConfiguredTask.__call__`) is one fragment submitted straight to `InProcess`; it needs no runner.

**Tech stack:** as Phase 2; under Route A, pydra from `enh/execution-extensions` at one pinned commit (P3-4); pandas
imported lazily by `ResultSet.to_frame()` only.

**Spec:** `design.md` §4.5–§4.9, §6.2 (Phase 3), §6.3, §6.4; ADR-001, 002, 003, 004, 006, 008, 009, 010 (as
amended); `execution.md`; `engine-choice.md`; `pydra-extension-points.md`; `testing.md`; `benchmarks.md`;
`engineering.md` (P-27..P-40); `principles.md`; `owner-requests.md` items 1–55; `rulings.md`; `review-register.md`
rows owned by Phase 3; `plans/phase-2.md` rulings P2-1..P2-95, which bind here unless a P3 ruling supersedes one.

**Placeholders.** `$SENSELAB_2X_IMPL` is the code worktree, `$SENSELAB_2X_SPEC` the spec worktree, `$PYDRA_EXT` the
pydra worktree, `$SCRATCH` the cluster scratch root, `$CPU_PARTITION`/`$CPU_NODE` the CPU partition and pinned node,
`$GPU_PARTITION` the partition the `orcd-remote` skill reports for typed H100, `$EXCLUDE_NODES` the nodes excluded in
Phase 2, `$SHARED_HF_HOME` the group's shared Hugging Face cache. No identifier, user name or personal path is written
into code, data or specs (P-28).

**Branches** (ruling P3-5). Code: `2x/phase-3`, which starts at the head of `2x/phase-2` (`79b417a8`; Phase 2 is not
yet merged into `2.x`), → `2.x` after `2x/phase-2` merges. Spec changes, ADR amendments, register rows and
`STATE.md`: `2x/phase-3-spec` → `2.x`. pydra: per-PR branches in `$PYDRA_EXT`, draft PRs into
`enh/execution-extensions` (ruling P3-6).

## Global Constraints

- **Interpreters are uv-managed only** (Phase 2, unchanged): `--python-preference only-managed` and a named version on
  every interpreter-selecting `uv venv`, in senselab and in pydra work.
- **Default Python is Colab's, 3.13** (owner items 54, 55): every CI job runs on 3.13; the rest of 3.11–3.15 runs when
  a release-tagged PR merges (owner item 54) and nightly.
- **No backwards compatibility** (P-14, P2-44): no shims, aliases, re-exports or readers of retired formats. A retired
  format is refused with a typed error naming the migration. `run_call` is removed, not wrapped (P2-16).
- **One implementation per behaviour** (P-37): one graph runner (the one the gate selects), one job engine, one queue,
  one lease, one YAML loader, one format gate, one `ResourceClass`, one assignment routine.
- **Rationale lives in specs** (P-21, P-31): docstrings say what and how; no register, round or ruling id outside
  `specs/`.
- **Thresholds and budgets are data with a derivation** (P-17): quanta, lease renewal fractions, retry budgets,
  bucket edges, idle timeouts, worker and request timeouts, scale budgets. A value not yet measured says so in its
  derivation and names the task that measures it.
- **Agreement is a standard comparison** (P2-95, owner item 48): hash equality, exact equality or `isclose` under
  run-config thresholds, by kernel type. No tolerance is fitted. Where a spec says "measured tolerance", read this.
- **Env versions are distinct agents** (P2-92, owner item 45); **licenses are transparent and the user's policy
  decides** (P2-93, owner item 46), with a missing license raising by default (P-23, owner item 11; ruling P3-20).
- **Order of work in every task** (P2-34): implement and run the local tests → review and compact locally by the
  named reviewer (never the implementer's model and context, P-40) → commit and push; GitHub CI green → ORCD run at
  that pushed SHA. ORCD evidence is cited `<sha> <job id> <summary line>`.
- **GitHub CI** (P2-33): logic, contract, and limits that need no model and no backend env. The 2.x jobs are
  `detect`, `pre-commit`, `core-only`, `senselab-lean`, `core-limits`, `source-digests`, `pr-models`, `core-env`,
  `licenses`, `worker-floor`; import contracts run inside `core-only`. Under Route A a `core-pydra` job (3.13) installs
  the `pydra` extra from `uv.lock` and runs every test that walks a plan.
- **Local machine** (owner items 27, 28, 42; P2-85): logic, contract and bounded model-free limits only, `uv run
  --no-sync`, scoped to the changed files, serial, never `pytest -n auto`. No `uv sync`, no new venvs, no model or
  dataset downloads, no backend envs, no full suites. Under Route A, tests that walk a plan need pydra and run in CI and
  on ORCD; executor, queue and pool tests submit fragments directly, so they run locally. Watch load, memory and disk; a
  local probe bounds its wall time and resident memory and stops on breach.
- **ORCD** (P2-33): every model, backend-env, GPU, Slurm-substrate, multi-node and scale run, submitted by an agent
  from a pinned SHA; the main thread holds the wait (`engineering.md` roster).
- **Weights pinned by commit** (P-4, P-16): loads go through catalogue pins; a plan records the resolved identity and
  `resume` never re-resolves a ref.
- **Tiers** (`testing.md`): exactly one of `logic`, `contract`, `limits`, `env`, `usecase`; sub-markers `gpu`,
  `network`, `slow`, `model`. A test that spawns processes is `logic` only if it finishes in under a second on the fake
  stack or a fake substrate; otherwise it is `limits` (model-free limits run in `core-limits`).
- **Moves** (P2-35, ADR-009): merge `origin/alpha` first (P-39); rename-only `git mv` commit, then a modify commit. A
  1.x file whose 1.x importers have not moved is not moved; its 2.x replacement is rewritten and the ledger names the
  source (rulings P3-13, P3-31).
- **pydra work follows senselab's principles** (owner item 50): rationale out of code, tiered tests, typed errors,
  full-tier CI before a merge to `main` (owner item 52), tests on the `cf` worker and on Slurm, never only `debug`
  (owner item 53).
- **Commits** (P-27): one commit per unit of work, pushed promptly; each task's last code commit has a green CI and
  records the implementer and reviewer models.

### ORCD job shape (every `orcd` step)

- **Checkout:** a fresh clone from GitHub per campaign at the pushed SHA (`$SCRATCH/senselab-2x/<sha>/`), never
  reset while jobs run from it; `git log origin/2x/phase-3..HEAD` empty before `sbatch`; the SHA in the campaign's job
  names and logs (a campaign convention; the runtime names jobs by plan hash and attempt).
- **CPU:** `#SBATCH -p $CPU_PARTITION -w $CPU_NODE -c 8 --mem=32G -t 02:00:00`.
- **GPU:** `#SBATCH -p $GPU_PARTITION --gres=gpu:h100:1 -c 8 --mem=64G -t 04:00:00 --exclude=$EXCLUDE_NODES`.
- **Pool campaigns** (Tasks 16, 21): the coordinator is a CPU job of the shape above with `-c 2 --mem=4G`; it submits
  worker jobs through the `slurm` substrate within the profile's caps. Concurrency is the QOS CPU cap divided by cpus
  per task, not the array `%` limit.
- **Environment:** `PATH="$HOME/.local/bin:$PATH"`; `LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"`;
  `HF_HOME=$SHARED_HF_HOME`; `HF_TOKEN` from the user's token file; `SENSELAB_RUN_ID=slurm-$SLURM_JOB_ID` (the attempt
  id, P3-17); `SENSELAB_CACHE=$SCRATCH/senselab-2x-cache`; `UV_CACHE_DIR=$SCRATCH/uv-cache`; execution profile `orcd`.
- **Steps:** `uv sync --locked --all-extras --group test` as its own step, then `uv run --no-sync pytest <scope> -m
  "<tier>"`, serial. Check the next maintenance reservation before choosing `-t`.
- **Evidence:** `COMPLETED` is not evidence: check elapsed time, grep the log for the summary line, record `<sha>
  <job id> <summary>`. Timing figures count only from an idle node (`benchmarks.md` §5).

## Review Focus

1. **A plan's hash never sees placement.** The `laptop` and `orcd` profiles give one `LogicalPlan` hash for one
   input; any change to a model pin, params, profile slice, manifest or reuse policy changes it. Pinned in Task 8
   (`test_plan_hash_ignores_execution_fields`), Task 7 (`test_plan_hash_changes_with_each_logical_field`) and Task 21
   Step 2.
2. **Duplicates cause duplicate work, never wrong results.** Two coordinators, two claims of one item, or a resume
   during a live run yield one result per key and no torn entry. Pinned in Task 14
   (`test_two_coordinators_one_result_per_key`) and Task 13 (`test_double_claim_commits_once`).
3. **A long job keeps its lease; a dead holder loses it.** Pinned in Task 12 (`test_renewed_lease_survives_stale_bound`,
   `test_renew_during_breaker_aside_window_keeps_lease`, `test_dead_holder_lease_broken_after_bound`) and its
   cross-node ORCD step.
4. **Faults charge the right budget.** Only a typed content fault charges an item; a gap fault never marks an item
   dead; one broken env is reported once, not per item. Pinned in Task 14 (`test_gap_fault_never_dead`,
   `test_broken_env_reported_once`) and Task 15a (`test_failing_batch_bisected`).
5. **Resume recomputes only what is missing and never re-resolves a ref.** Pinned in Task 11
   (`test_resume_executes_only_missing`, `test_resume_never_reresolves_ref`) and Task 21 Step 4.
6. **Batching never silently changes a non-invariant backend.** Until Task 18 a non-invariant backend runs at batch
   size 1 (formation `single`); from Task 18 its bucket is in the key. Pinned in Task 9
   (`test_noninvariant_backend_sent_single_until_buckets`) and Task 18
   (`test_noninvariant_backend_batches_by_keyed_bucket_or_single`).
7. **A BIDS tree in the wild.** Non-BIDS files, caseless name collisions, symlinks out of the root and a file
   rewritten between plan and run are ignored, refused or stale, never misread. Pinned in Task 6
   (`test_bids_ignores_non_bids_files`, `test_bids_caseless_collision_refused`, `test_bids_symlink_out_of_root_refused`)
   and Task 9 (`test_rewritten_input_is_stale_at_run`).

## File Structure

```
packages/senselab-core/src/senselab_core/
  kernel/graph.py                          # Graph, Node, Edge (graph@1.0), validate(), order(), graph_digest()
  kernel/dataset.py                        # Record, Dataset protocol, InputManifest (input-manifest@1.0)
  kernel/plan.py                           # LogicalPlan (logical-plan@1.0), plan_hash()
  kernel/license.py                        # + LicenseSet (new), taint union
  kernel/time.py                           # rebind over declared clock-relative fields (R-162)
  runtime/config.py                        # load_yaml(source, what), ConfigError: the one YAML loader (R-195)
  runtime/datasets.py                      # from_records(), shard(), build_manifest()
  runtime/bids.py                          # bids_dataset(): the minimal BIDS adapter
  runtime/planner.py                       # resolve(), plan(), Plan, PlanPlacement (placement@1.0), explain(), resume()
  runtime/host_worker.py                   # renamed from runtime/in_process.py (InProcessWorker → HostWorker)
  runtime/call.py                          # TaskCall, RunConfig, Result, bind_input: the call shape; run_call removed
  runtime/locks.py                         # Lease.renew()
  runtime/storage.py                       # probe_capabilities()
  runtime/revision.py                      # run_id() names the attempt (P3-17)
  runtime/profiles.py, data/profiles/      # execution-profile@2.0
  runtime/data/{batching.yaml,graph@1.0.schema.json}
  runtime/execution/
    executor.py                            # Executor protocol, Handle, Status, JobRef, PlanFragment, ResultRef, FaultRecord
    job.py                                 # Job, execute(): the one job engine (from run_call)
    workers.py                             # WorkerSet: bounded checkout/return of worker clients and servers
    runner.py                              # Task 9b: PydraRunner (pass) or NativeRunner (fail); never both
    in_process.py                          # InProcess executor
    run.py                                 # Run, RunStatus, ResultSet
    events.py                              # Event (run-event@1.0), EventLog, BrokenEnv, status_from()
    queue.py                               # Queue, QueueSet, Claim (rewritten from the ADR-003 design)
    pool.py                                # Pool executor: epoch, release, fault effects
    pool_worker.py                         # serve_pool(): claim, quantum, execute, renew, bisect
    supervisor.py                          # subreaper supervision of worker processes
    substrates.py                          # Substrate protocol, Local, Slurm, scale()
    services.py                            # ServiceRegistry, consumer leases, the svc pool worker
    batching.py                            # form_batches(), BucketGrid
  cli.py                                   # execute, coordinate, status, explain
packages/senselab/src/senselab/
  __init__.py                              # sl.plan, sl.run, sl.resume, sl.use, sl.validate, sl.Dataset
  _api/                                    # regenerated: .map on every task class
  data/resources/<task>/<backend>.yaml     # resource-table@1.0, generated by the scale ladder
benchmarks/scale/{ladder.py,budgets.yaml,bids_tree.py,suites/ladder.yaml}
scripts/generate_resources.py
```

**Execution order** (dependencies are authoritative): 0 (first; Tasks 1–8 run alongside it) · 1 → 2, 3, 4a → 4b → 5
→ 6 → 7 → 8 → 9 → 9b (needs Task 0's outcome) → 10 → 11 → 12 → 13 → 14 → 15a → 15b → 16 → 17 → 18 → 20 → 21 → 22 →
23 → 24. The pydra tasks (40–53) run in parallel from Task 0 on. There is no Task 19 (folded into 9b).

---

### Task 0: The pydra decision gate (design §4.9)

**Model:** opus lead, effort higher (gate synthesis); fresh sonnet verifies each evidence line against the pydra
branch and CI (P-40). **Tier:** pydra CI, ORCD, and one senselab spike. **Implements:** design §4.9, §6.2, §6.3;
`execution.md` §4; ADR-001; rulings P3-1, P3-2, P3-3, P3-4, P3-7. **Depends on:** nothing. Its outcome selects the
Task 9b variant.

**Evidence to gather** (each a line `<item> <commit or PR> <CI run> <ORCD job> <verdict>` in the gate record):
- **Merged on the branch:** A1, D1, E1a, E1b, F1, F2, G1, K1 into `enh/execution-extensions` (closed draft PRs
  #894–#901 and the integration merge); #903 (workflow-construction cache, merged as `2b24ac27`); #904 (cf
  forkserver, merged, branch head `277bee99` at planning); #902 (CI tiers, merged into the branch, open to `main`);
  #891 the draft carrying the branch to `main`.
- **K2 is checked, not assumed.** At planning the branch has no `__checksum__`, no `ReusePolicy` and no K2 PR;
  `ResultStore.lookup(checksum, context)` exists from D1/K1. Task 40 delivers K2 inside this gate (ruling P3-1).
- **Pinned commit:** one commit `P` of `enh/execution-extensions` holding all of the above plus K2.

**Pass criteria (all):**
1. A1, D1, E1a, E1b, F1, F2, G1, K1 and K2 are merged into `enh/execution-extensions` at `P`.
2. pydra's **full** CI tier (the `full-ci` label of #902: full Python matrix, `cf`, Slurm, singularity) is green at
   `P` (owner items 52, 53).
3. ORCD (CPU node, a locked venv built in the job): `pytest pydra -k "store or checksum or reuse or readiness or
   on_error or resources or router or slurm"` passes at `P` under the `cf` worker, and the A1/F1 tests submit a real
   Slurm job.
4. The senselab spike (Step 3) passes its four contracts at `P`.

**Outcomes:**
- **Pass → Route A.** Task 9b builds `PydraRunner` and carries the spike's extra, pin and `core-pydra` job into
  `2x/phase-3` under review. `NativeRunner` is not built (design §6.3: the fallback only if triggered).
- **Fail → own runner.** Task 9b builds `NativeRunner` behind the same `Executor` protocol; Phase 4 extends it to
  route, loop and nesting (the 2–4k-line fallback of design §4.9). The pydra tasks continue; the gate is re-evaluated
  at Task 21 and at Phase 4 start, and a later pass replaces `NativeRunner` with `PydraRunner` in one commit.

- [ ] **Step 1 (read-only):** gather the evidence lines (`git log origin/enh/execution-extensions`, `gh pr view 891
  902 903 904`, a grep of the branch for `__checksum__` and `ReusePolicy`); write the record to
  `reviews/phase3-gate/evidence.md` on `2x/phase-3-spec`.
- [ ] **Step 2:** if K2 is missing, run Task 40 to completion (draft PR into the branch, merged on a green full tier).
  K2 is already opus work; if it fails review twice it moves to fable (roster rule); if fable fails on it, the gate
  fails.
- [ ] **Step 3 (spike):** on a throwaway branch `2x/p3-gate-spike` of the code repo, the spike adds what it needs
  itself: the optional `pydra` extra in `packages/senselab-core/pyproject.toml`, `[tool.uv.sources] pydra = { git =
  "https://github.com/nipype/pydra", rev = "<P>" }` in the root `pyproject.toml`, `uv lock` (resolution only), and a
  `core-pydra` job (3.13) in `.github/workflows/2x-tests.yaml`. `packages/senselab-core/tests/runtime/execution/
  pydra_spike_test.py` (tier `contract`) runs a fake three-node graph (map, task, gather) on the fake stack, each pydra
  job's `run` calling the fake backend directly: `test_supplied_checksum_equals_logical_key` (every `Job.checksum`
  equals the senselab key digest), `test_no_result_pickle_written` (no `_result.pklz` under any cache root;
  `_job.pklz` may remain until B2 and D2 land, ruling P3-49), `test_errored_element_fails_only_its_descendants`
  (`on_error="raise_at_end"`), `test_router_sends_resource_classes_to_two_workers`. The spike branch is deleted after
  the decision.
- [ ] **Step 4:** decide; record the outcome, `P` and the evidence in `STATE.md`; commit on `2x/phase-3-spec` a
  design §9 amendment line and, because P3-1 moves K2 into Phase 3, the §4.9 and §6.2 amendments that say so; push.
  Commit `docs(2x): Phase 3 pydra gate`.

### Task 1: P-39 reconciliation at Phase 3 start

**Model:** sonnet implements (effort lower, worktree); haiku verifies (P-40). **Tier:** none. **Branches:** merge on
`2x/phase-3`; ledger on `2x/phase-3-spec`. **Implements:** P-39, ruling P3-8.

- [ ] **Step 1:** `git fetch origin alpha && git log <mark>..origin/alpha --oneline -- src scripts tutorials` per
  ledger row; merge `origin/alpha` into `2x/phase-3` (no edit under `src/`); set the marks; push.
- [ ] **Step 2:** the register statuses were set from the pre-flight check in the plan's revision commit; recheck the
  rows that cite `2x/phase-2` code after the merge, and add a row for any alpha change that reopens one.
- [ ] **Step 3:** commit `docs(2x): P-39 reconciliation at Phase 3 start`; push.

### Task 2: Core carry-overs: one YAML loader, the provenance format gate, declared clock fields, stamp layout

**Model:** sonnet implements; opus reviews (kernel). **Tier:** logic, contract. **Implements:** R-195, R-190 (and
the provenance and label-map remainder of R-172), R-162, R-201; rulings P3-8, P3-9. **Depends on:** Task 1.

**Files:** Create `runtime/config.py`, `tests/runtime/config_test.py`. Modify every YAML site in `packages/`:
`runtime/{envs,models,profiles,licenses,fetch,storage,catalogue,equivalence,compare}.py`,
`packages/senselab/src/senselab/text/tasks/pii_spans/{vllm_backend,profile}.py`,
`packages/senselab/src/senselab/signal/resample.py`, `packages/senselab/src/senselab/reference/_render.py`, and
`benchmarks/harness.py`; `.importlinter` (a `forbidden` contract: within `senselab_core`, `senselab` and `benchmarks`,
only `senselab_core.runtime.config` imports `yaml`). `scripts/` stay outside the contract (lint roots, ruling P3-9).
Modify `runtime/provenance.py` (`_declared_format` → `formats.check` over the whole document), `kernel/prov.py` (ID
patterns exported), `packages/senselab/src/senselab/metrics/der.py` (its `assign` replaced by
`runtime.compare.best_assignment`), `runtime/compare.py`, `runtime/bep028.py` (label map as JCS),
`kernel/{time,results,signals}.py`, `runtime/audio_io.py`, tests.

**Interfaces:**
- `load_yaml(source: str | Path | Traversable, what: str) -> dict[str, Any]` raising `ConfigError(what, label,
  message)`; dates stay text; a duplicate key is refused.
- `best_assignment(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]` (existing, `compare.py:301`) becomes
  the one routine: with a tie it returns the lexically smallest optimal `(row, column)` sequence, pinned by a test;
  `der.assign` is deleted.
- Each kernel type that holds clock-relative fields declares them (`clock_fields: ClassVar[tuple[str, ...]]`);
  `rebind` shifts only those (R-162).
- A source stamp records the channel layout it was taken under; `stamp → decode → stamp` round-trips the layout and a
  re-stamp under another layout is a different stamp (R-201).

- [ ] **Step 1: Failing tests (local):** `test_one_yaml_loader_refuses_duplicates_and_keeps_text_dates`,
  `test_indented_retired_provenance_is_refused_not_malformed`, `test_best_assignment_tie_break_is_pinned`,
  `test_der_uses_core_assignment`, `test_rebind_leaves_duration_fields`, `test_stamp_pins_channel_layout`,
  `test_label_map_is_jcs`; the import-linter contract fails against today's tree.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel packages/senselab-core/tests/runtime/{config,provenance,compare,audio_io,profiles,catalogue,equivalence,bep028}_test.py -m "logic or contract" -k "not pyld"`; `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests/metrics packages/senselab/tests/text -m "logic or contract"`; `uv run --no-sync lint-imports`.
- [ ] **Step 4:** Review and compact, commit `refactor(core): one YAML loader, one format gate, declared clock fields`;
  push; CI green. Close R-162, R-190, R-195, R-201 and R-172's remainder.

### Task 3: Benchmark carry-overs: ratio aggregates, malformed references, declared split units

**Model:** sonnet implements; opus reviews. **Tier:** logic. **Implements:** R-192, R-193, R-194; ruling P3-10.
**Depends on:** Task 2.

**Files:** `kernel/results.py` (`Metric` → `metric@2.0`), `kernel/formats.py`, `packages/senselab/src/senselab/metrics/
{wer,der,eer,span_f1,resample_error}.py`, `benchmarks/harness.py`, tests.

**Interfaces:**
- `Metric` carries `numerator: float | None` and `denominator: float | None` beside `value`, `unit`; an aggregate over
  items is `sum(numerators) / sum(denominators)`; EER has both `None` and aggregates over pooled trials (ruling
  P3-10). `metric@1.0` is refused.
- An adapter declares its split unit (`split_unit: Literal["item", "speaker", "recording"]`) and `split_of(unit_id)`
  hashes that unit; the trials adapter's unit is the speaker, diarization's the recording.
- A reference span outside `0 <= start <= end <= len` is a `ContentFault` of that item, counted in the row.

- [ ] **Step 1: Failing tests (local):** `test_der_aggregate_is_ratio_of_sums`, `test_eer_aggregates_pooled_trials`,
  `test_metric1_refused`, `test_trials_split_by_speaker`, `test_malformed_reference_is_item_fault`.
- [ ] **Step 2: Implement; Step 3 (local):** `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests/metrics packages/senselab/tests/bench -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `fix(bench): ratio aggregates, item faults for bad references, split units`;
  push; CI green.

### Task 4a: Backend carry-overs: minimum input, transcript extent, overlap scoring, per-request faults

**Model:** sonnet implements; opus reviews. **Tier:** contract local; env and limits on ORCD. **Implements:** R-198,
R-188 (remainder), R-197, R-191; ruling P3-11. **Depends on:** Task 2.

**Files:** `kernel/declare.py` (`BackendSpec.min_input_ns`), `kernel/results.py` (`Transcript` extent), the shared
audio input check, `packages/senselab/src/senselab/audio/tasks/speech_to_text/{crisperwhisper,window_confidence}.py`,
`.../speaker_diarization/pyannote.py`, `.../speaker_embeddings/speechbrain.py` (its `min_samples()` folded into the
declaration), `testing/contracts.py`, `runtime/fake_backend.py`, `senselab_worker/fake.py`, `packages/conftest.py`;
create `packages/senselab/tests/data/tier_budgets.yaml`; tests.

**Interfaces:**
- `BackendSpec.min_input_ns: int | None`, a duration: a sample minimum converts at the backend's model input rate
  (speechbrain's 640 samples at 16 kHz is 40 ms), stated in the declaration's derivation; the one input check refuses
  shorter input with `ContentFault` before the backend runs; crisperwhisper and pyannote declare theirs from the
  library's frame or receptive field.
- `Transcript` validates every span against a declared extent (`extent_ns`), not only its clock (R-188).
- Confidence of a long-form segment averages only the ids that survive the window merge (R-197).
- Fault injection is a request field (`params["_fault"]`, reserved by the fake and the contract cases), never bound at
  `load`; a contract case loads once per test. Each tier's time budget is data in `tier_budgets.yaml` with a
  derivation from the measured CPU run (Step 4).

- [ ] **Step 1: Failing tests (local):** `test_input_below_declared_minimum_is_content_fault` (generated per backend
  on its stub), `test_transcript_span_past_extent_refused`, `test_overlap_dropped_ids_not_scored`,
  `test_fault_injection_per_request_one_load`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `fix(senselab): declared minimum input, transcript
  extent, scored ids, per-request faults`; push; CI green.
- [ ] **Step 4 (orcd, CPU then GPU H100, at that SHA):** `-m "(env or limits) and model" -k "crisperwhisper or
  pyannote or speechbrain or whisper_transformers"`; record each tier's wall time; commit `tier_budgets.yaml` with
  those numbers as derivations, reviewed. Close R-188, R-191, R-197, R-198.

### Task 4b: CrisperWhisper scoring cost, measured (R-196)

**Model:** sonnet submits; opus reads the numbers. **Tier:** limits (`model`) on ORCD. **Implements:** R-196.
**Depends on:** Task 4a.

- [ ] **Step 1 (orcd, idle CPU node and idle H100, at Task 4a's SHA):** per-window wall time of decoding and of the
  confidence pass on both engines, 50 LibriSpeech items, with confidence on and off.
- [ ] **Step 2:** the register row records both numbers and the decision: if the confidence pass is a material share
  of decoding, a follow task exposes the encoder output from the library and scores from it; otherwise the row closes
  with the measurement. Spec commit; push.

### Task 4c: CrisperWhisper confidence from the decode's own logits (R-202)

**Model:** opus implements; opus reviews. **Tier:** logic local; env and limits on ORCD. **Implements:** R-202 (follow
task of R-196's measurement). **Depends on:** Task 4b.

- [ ] **Step 1:** expose the encoder output and the per-step logits of the final decode of each window from the
  library (both engines), and compute token and word confidences from them; the teacher-forcing pass goes.
- [ ] **Step 2:** the confidences equal the teacher-forced ones within the comparison thresholds on the Task 4a
  long-form item; re-measure with `benchmarks/crisperwhisper_confidence_cost.py` on CPU and H100 (ruling P3-59);
  record the share in R-202.

### Task 5: The graph document (`graph@1.0`)

**Model:** opus implements (graph format); fresh opus reviews. **Tier:** logic, contract. **Implements:** ADR-001,
design §4.5 (fan-out), P-1, P-9; rulings P3-12, P3-13. **Depends on:** Task 2.

**Files:** Create `kernel/graph.py`, `runtime/data/graph@1.0.schema.json`, `scripts/generate_graph_schema.py`;
modify `kernel/formats.py`, `packages/senselab/src/senselab/__init__.py` (`sl.validate`).

**Interfaces:**
- `PortRef = str` of the form `<node>.<port>` or `input.<name>`.
- `Node(Frozen)`: `id: NodeId`, `task: str`, `task_version: str`, `params: Mapping[str, JsonValue]`,
  `backend_params: Mapping[str, Mapping[str, JsonValue]]` (per backend name; ruling P3-50), `model: str | None` (a
  reference string).
- `Edge(Frozen)`: `source: PortRef`, `target: PortRef`, `kind: Literal["plain", "map", "gather"]`.
- `Graph(Formatted)`: `format = "graph@1.0"`, `inputs: Mapping[str, str]` (name → kernel port type, `list[T]` for a
  record list), `nodes: tuple[Node, ...]`, `edges: tuple[Edge, ...]`, `outputs: Mapping[str, PortRef]`. `route`,
  `loop` and nested pipeline nodes are not in `graph@1.0` (ruling P3-12).
- `validate(graph: Graph, registry: Registry) -> list[GraphIssue]`, `GraphIssue{code, node, port, message}` with codes
  `unknown_task`, `version_mismatch`, `unknown_port`, `type_mismatch`, `unmapped_list` (a list into an item port without
  `map`), `gather_without_map`, `cycle`, `unbound_input`.
- `order(graph) -> tuple[NodeId, ...]`: topological, lexically smallest ready id first (ruling P3-13).
- `graph_digest(graph) -> str` = `identity_digest("graph@1", <canonical JSON>)`; `graph_schema() -> dict` (JSON Schema
  2020-12, P2-9); the published file is regenerated by the script and drift-checked.
- `sl.validate(graph_or_json) -> list[GraphIssue]`.

- [ ] **Step 1: Failing tests (local):** `test_list_into_item_port_without_map_is_unmapped_list`,
  `test_cycle_reported_with_nodes`, `test_order_is_lexical_among_ready`, `test_graph_round_trips_json`,
  `test_graph_digest_ignores_key_order`, contract `test_published_graph_schema_has_no_drift`,
  `test_graph_schema_validates_2020_12`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel/graph_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): the graph document`; push; CI green. Spec commit: ADR-006
  format row `graph@1.0`.

### Task 6: Datasets: `Record`, `Dataset`, input manifest, in-memory and minimal BIDS adapters

**Model:** sonnet implements; opus reviews (identity of the manifest). **Tier:** logic, contract. **Implements:**
design §4.6 (protocol, binding, adapters), I-7, R-121 (decode errors as kernel types); rulings P3-14, P3-15, P3-16.
**Depends on:** Task 5.

**Files:** Create `kernel/dataset.py`, `runtime/datasets.py`, `runtime/bids.py`; modify `kernel/formats.py`,
`kernel/signals.py` and `runtime/audio_io.py` (a non-audio file raises the kernel `UndecodableSource`, not
`LibsndfileError`); tests; `packages/senselab-core/tests/runtime/bids_tree_fixture.py` (a generated tree, no committed
audio).

**Interfaces:**
- `Record(Frozen)`: `id: RecordId` (stable: the root-relative POSIX path for BIDS, the given id in memory), `entities:
  Mapping[str, str]`, `inputs: Mapping[str, InputRef]` (a path under the root or a value), `sidecar: Mapping[str,
  JsonValue]`.
- `Dataset` protocol (kernel): `records() -> Iterator[Record]` (sorted by id), `entities() -> Mapping[str, tuple[str,
  ...]]`, `filter(**entities: str | Sequence[str]) -> Dataset`, `shard(by: str, index: int, count: int) -> Dataset`
  (closed under the grouping level `by`, I-7). The protocol has no constructors (ruling P3-14).
- `InputManifest(Formatted)`: `format = "input-manifest@1.0"`, `rows: tuple[ManifestRow, ...]` sorted by id, each `{id,
  entities, inputs: {port: InputIdentity}}`; `digest() -> str`.
- Runtime constructors: `datasets.from_records(rows: Iterable[Mapping[str, Any]], *, id_key: str = "id") -> Dataset`;
  `bids.bids_dataset(root: Path, *, datatype: str | None = None, suffix: str | None = None, extensions: tuple[str, ...]
  = (".wav", ".flac")) -> Dataset`; `datasets.build_manifest(dataset: Dataset, ports: Mapping[str, str]) ->
  InputManifest`, whose identities come from `audio_io` stamping (chunked stamps, P2-21).
- BIDS: entities parsed once from file names (`sub`, `ses`, `task`, `acq`, `run`, `recording`, and any `key-value`
  pair), suffix and datatype from the path; `participants.tsv` columns and the nearest sidecar JSON (BIDS inheritance)
  are read into `Record.sidecar` and never keyed (ruling P3-15). Files outside the naming grammar are ignored and
  counted; names that collide caselessly are refused (`BidsCollision`); a symlink whose target leaves the root is
  refused.
- Binding: a record's input named like a graph input binds to it; `bind={"audio": "<suffix or port>"}` overrides.

- [ ] **Step 1: Failing tests (local):** `test_record_ids_stable_across_walk_order`, `test_shard_closed_under_subject`
  (a shard by `sub` never splits a subject; misaligned shards refused), `test_manifest_digest_changes_with_content_only`
  (moving the root leaves it; one rewritten sample changes it), `test_bids_entities_parsed_once`,
  `test_bids_ignores_non_bids_files`, `test_bids_caseless_collision_refused`, `test_bids_symlink_out_of_root_refused`,
  `test_sidecar_not_in_manifest_identity`, `test_non_audio_file_is_undecodable_source`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel/dataset_test.py packages/senselab-core/tests/runtime/{datasets,bids,audio_io}_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): datasets, input manifests and a minimal BIDS adapter`; push;
  CI green. Spec commit: ADR-006 row `input-manifest@1.0`.

### Task 7: `LogicalPlan`, resolution and the plan hash

**Model:** opus implements (identity, planner); fresh opus reviews. **Tier:** logic, contract. **Implements:** ADR-004,
ADR-002 (identity, I-1), design §4.7, R-163 (profile slices), P-2, P-10; rulings P3-17, P3-18. **Depends on:** Task 6.

**Files:** Create `kernel/plan.py`; `runtime/planner.py` (resolution); modify `runtime/keys.py` (`Call.profile_slices`
populated), `runtime/call.py` (`RunConfig` with `reuse`), `kernel/formats.py`, tests.

**Interfaces:**
- `LogicalPlan(Formatted)`: `format = "logical-plan@1.0"`, `graph: Graph`, `models: Mapping[NodeId, ResolvedModel |
  None]` (catalogue defaults resolved here), `profile_slices: Mapping[NodeId, Mapping[str, ProfileSliceIdentity]]`,
  `manifest_digest: str`, `bind: Mapping[str, str]`, `reuse: ReusePolicy`. No hardware, env, device, batch size,
  partition, backend or profile digest (ADR-004).
- `plan_hash(plan: LogicalPlan) -> str` = `identity_digest("logical-plan@1", <canonical JSON>)`; it is the run id
  (ruling P3-17).
- `resolve(target: Graph | TaskCall, dataset: Dataset | None, *, config: RunConfig, dataset_profile: Mapping | None =
  None) -> tuple[LogicalPlan, InputManifest]` takes no execution profile and no hardware: a single call or a
  configured task over a dataset becomes a one-node graph whose input edge is `map` over the record list; each node's
  model is pinned (catalogue pin or `resolve()`d commit); each node's declared profile slices are cut from the run
  config or dataset profile and keyed by identity (R-163).
- The backend is placement (ADR-004): a node's `backend_params` hold one entry per backend the node may be placed on,
  and only the placed backend's entry reaches its key (P3-50).

- [ ] **Step 1: Failing tests (local, fake stack):** `test_resolve_takes_no_execution_input` (its signature has no
  profile or hardware parameter), `test_plan_hash_changes_with_each_logical_field` (table over model pin, params,
  backend params, profile slice, manifest, reuse policy), `test_ref_resolved_to_commit_in_plan`,
  `test_declared_slice_keyed_by_identity`, `test_undeclared_profile_key_perturbation_leaves_plan` (I-1).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/kernel/plan_test.py packages/senselab-core/tests/runtime/planner_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): the logical plan and its hash`; push; CI green. Spec commit:
  ADR-006 row `logical-plan@1.0`; close R-163.

### Task 8: Placement, `explain`, licenses and taint, cost, caps; `execution-profile@2.0`

**Model:** opus implements (planner); fresh sonnet reviews. **Tier:** logic, contract. **Implements:** ADR-004
(placement), design §4.7 (announce env builds, cost, licenses), §4.13 and I-9 (taint), `execution.md` §3.4, P-7,
P-23, R-116 (taint), R-199; rulings P3-19, P3-20, P3-21, P3-22, P3-23, P3-50. **Depends on:** Task 7.

**Files:** `runtime/planner.py` (placement), `runtime/keys.py` (`Placement` fields), `runtime/profiles.py`,
`runtime/data/profiles/{laptop,orcd}.yaml`, `kernel/license.py` (new `LicenseSet`, taint union), `runtime/catalogue.py`
(resource tables joined by `(id, revision)`), `runtime/call.py` (`_hit`'s license rule), `kernel/formats.py`
(`placement@1.0`, `execution-profile@2.0`, `catalogue@2.0`, `resource-table@1.0`), tests.

**Interfaces:**
- `Placement` gains `resource_class: str` (`cpu`, `gpu:<type>`, `mps`, `svc:<id>`), `batch_size: int`, `partition:
  str | None`. `build_key` reads exactly what it reads today: `backend` (name and version), `backend_params`, and
  `batch_formation` and `chunk_grid` only when the backend declares them output-affecting (ADR-002). It never reads
  `device`, `env_root`, `why`, `resource_class`, `batch_size` or `partition`.
- `PlanPlacement(Formatted)`: `format = "placement@1.0"`, `profile_digest`, `nodes: Mapping[NodeId, Placement
  record]`, `env_builds: tuple[str, ...]` (envs not yet built), `licenses: Mapping[NodeId, LicenseSet]` (the node's
  own and the union of its upstreams', I-9), `cost: Cost`.
- `Cost(Frozen)`: `wall_s`, `gpu_hours`, `download_bytes` (each `float | int | None`), `unmeasured: tuple[str, ...]`
  naming every node or model without a resource table; an unmeasured figure is `None`, never estimated (P3-19).
- `Plan`: `logical: LogicalPlan`, `manifest: InputManifest`, `placement: PlanPlacement`, `hash: str`; `explain() ->
  str` lists per node backend, stack, env, device, batch, resource class, partition, licenses, cost and the reason.
- `plan(target, dataset=None, *, config=None, dataset_profile=None, execution=None, bind=None) -> Plan`. The placed
  backend's `backend_params` entry is validated at plan time; a node with no entry for its placed backend uses that
  backend's defaults, and an entry its schema rejects raises `BackendParamsMismatch(node, backend, errors)` (P3-50).
- Taint (I-9): a derived output's `LicenseSet` is the union of its node's and its inputs' recorded sets; the run
  config's `licenses` policy is checked at plan time on every node and on every hit served (R-116). A hit whose context
  records no env license is refused by default and recomputed; a `RunConfig.licenses` policy may allow it, and the
  allowance is recorded on the reuse activity (ruling P3-20, R-199).
- Caps: per resource class, the static memory cap must cover the models resident at once, from the resource tables;
  a node without a table is admitted and listed under `unmeasured` (P3-21).
- `execution-profile@2.0` adds, each with a derivation: `resource_classes: {name: ResourceClass{device, gpu_type,
  cpus, mem_bytes, time_s, partition, gres, max_workers, memory_cap_bytes}}` (the one `ResourceClass`; the 1.x
  `work_pool.ResourceClass` and `PartitionSpec` are not ported), `substrate: {kind: local | slurm, ...}`, `pool:
  {quantum_items, quantum_s, poll_s, lease_renew_fraction, segment_items}`, `faults: {content_attempts,
  infrastructure_replacements, resource_replacements}`, `timeouts: {load_s, request_s}` (R-189), `services:
  {idle_exit_s}`. One bump carries every Phase 3 key (P3-22); `execution-profile@1.0` is refused.
- Resource tables are generated files `packages/senselab/src/senselab/data/resources/<task>/<backend>.yaml`
  (`resource-table@1.0`); the catalogue joins them by `(id, revision)`; `catalogue@2.0` drops the inline `resources`
  field (P3-23).

- [ ] **Step 1: Failing tests (local, fake stack):** `test_plan_hash_ignores_execution_fields` (laptop and orcd
  profiles, two hardware probes, one hash), `test_gpu_only_node_without_gpu_class_fails_naming_node`,
  `test_explain_lists_every_placement_field`, `test_env_builds_announced`, `test_unmeasured_cost_is_none_and_named`,
  `test_taint_unions_upstream_licenses`, `test_policy_refuses_tainted_hit`,
  `test_hit_without_env_license_refused_by_default`, `test_policy_allowing_unlicensed_hit_is_recorded`,
  `test_caps_refuse_resident_models_over_cap`, `test_backend_params_mismatch_at_plan_time`, `test_profile1_refused`,
  `test_nonkey_placement_fields_never_in_key` (each of `device`, `env_root`, `why`, `resource_class`, `batch_size`,
  `partition` varied through `build_key`; one key).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/{planner,profiles,catalogue,keys,call}_test.py packages/senselab-core/tests/kernel/license_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): placement, explain, taint and execution-profile@2.0`; push;
  CI green. Spec commit: ADR-006 rows; close R-199 and R-116's taint part.

### Task 9: The job engine, the `Executor` protocol and `InProcess` (replaces `run_call`)

**Model:** opus implements (identity, worker protocol); fresh opus reviews. **Tier:** logic, contract. **Implements:**
ADR-001 (`Executor`, `PlanFragment`), ADR-009 (gap faults never charged or cached), I-3 (until Task 18), R-110, R-121
(executor part), R-160, R-166, R-167, R-187 (producer kept), R-200; rulings P2-16, P3-24, P3-25, P3-26, P3-51.
**Depends on:** Task 8.

**Files:** Create `runtime/execution/{__init__,executor,job,workers,in_process}.py`. Rename `runtime/in_process.py` →
`runtime/host_worker.py` (`InProcessWorker` → `HostWorker`, P3-24). Modify `runtime/call.py` (remove `run_call`,
`close_workers`, `stop_servers`, `_execute` and the worker pool; keep `TaskCall`, `RunConfig`, `Result`,
`bind_input`, `packaged`), `runtime/keys.py` (`request_components(backend, request_size)`), `senselab_worker/
model_cache.py` (evict before load, reentrant lock, `used_bytes` under the lock, R-160), `packages/senselab/src/
senselab/_api/call.py`, `benchmarks/harness.py`, `scripts/run_calls.py`, `packages/senselab/tests/parity/`, tests.

**Interfaces:**
- `JobRef(Frozen)`: `plan_hash: str`, `node: NodeId`, `element: tuple[int, ...]` (map indices; `()` unmapped).
- `PlanFragment(Formatted)`: `format = "plan-fragment@1.0"`, `plan_hash`, `node`, `jobs: tuple[JobRef, ...]`,
  `inputs: Mapping[JobRef, Mapping[str, InputIdentity]]`, `placement: Placement record`, `attempt: int`. Never a
  pickled object.
- `FaultRecord(Frozen)`: `fault: FaultClass`, `type: str`, `message: str`, `gap: bool`, `charged: bool`, built from
  the worker's `error_record`.
- `Executor` protocol: `submit(fragment: PlanFragment) -> Handle`, `status(handle) -> Status`, `cancel(handle) ->
  None`, `results(handle) -> Iterator[ResultRef]`; `ResultRef{job: JobRef, key_digest: str, output_identity:
  OutputIdentity | None, failure: FaultRecord | None}`. This is the seam both graph runners meet (P3-2).
- `Job(Frozen)`: `ref: JobRef`, `call: keys.Call`, `values: Mapping[str, object]`, `anchors`.
- `execute(jobs: Sequence[Job], placement: Placement, *, store: ResultStore, run: RunConfig, profile:
  ExecutionProfile, workers: WorkerSet) -> list[JobOutcome]`: per job, build the key with `build_key` under the
  formation of the request actually sent (`request_components(backend, request_size)` returns `single` for one item;
  R-187), serve under `run.reuse` with verify sampling, the taint check and re-binding, else send the unserved jobs and
  store per item; one activity per execution with `senselab:modelSource` (R-110). A non-batch-invariant backend is sent
  one item per request until Task 18 adds keyed buckets; an invariant backend's multi-item formation is placement only.
  File-backed inputs are keyed only after `audio_io.check` (R-121). `JobOutcome = Completed(result) |
  Failed(FaultRecord)`; a failure is never stored.
- `WorkerSet(bound: int)`: `checkout(backend, env, model) -> ContextManager[Worker]`; LRU-evicts idle workers beyond
  `bound` (R-167); a server start holds a per-(env, model) lease, not a global lock (R-166).
- `InProcess(*, store=None, profile=None, run=None)`: an `Executor` whose `submit` runs the fragment's jobs through
  `execute` in this process, in requests of `placement.batch_size` (subject to the rule above).
- `ConfiguredTask.__call__(x)` resolves a one-node plan and submits its single fragment to `InProcess`; no runner. The
  harness, `scripts/run_calls.py` and the parity tests submit fragments or run plans, never `run_call` (P3-25).

- [ ] **Step 1: Failing tests (local, fake stack):** the `run_call` tests ported onto `InProcess`
  (`test_second_call_hits_store`, `test_one_activity_per_call`, `test_model_source_recorded`,
  `test_stack_switch_by_profile`, `test_call_from_describe_alone`), plus `test_failure_never_stored`,
  `test_gap_fault_record_never_charged`, `test_rewritten_input_is_stale_at_run`,
  `test_noninvariant_backend_sent_single_until_buckets`, `test_request_components_match_request_sent`,
  `test_worker_set_bounded_and_evicts_idle`, `test_server_start_lock_per_env_and_model`,
  `test_model_cache_evicts_before_load`.
- [ ] **Step 2: Implement** (rename commit for `host_worker.py`, then the change). The `run_call` docstring's rationale
  moves to `execution.md` (R-200).
- [ ] **Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution packages/senselab-core/tests/runtime/keys_test.py packages/senselab-worker/tests -m "logic or contract"`; `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core)!: one job engine behind the Executor protocol; run_call
  removed`; push; CI green. Close R-110, R-121's executor part, R-160, R-166, R-167, R-200.
- [ ] **Step 5 (orcd, CPU then GPU H100, at that SHA):** `-m "(env or limits) and model"` over `packages/senselab/tests`
  (every port runs through `InProcess`); parity rerun at this SHA with no change in result.

### Task 9b: The graph runner the gate selects

**Model:** opus implements (graph, identity); fresh opus reviews. **Tier:** logic, contract (`core-pydra` under
Route A); limits on ORCD. **Implements:** design §4.9, §6.3, ADR-001, `execution.md` §4, `pydra-extension-points.md`
§7, E1 semantics; rulings P3-2, P3-3, P3-4, P3-43, P3-52. **Depends on:** Task 0 (outcome), Task 9.

**Files:** Create `runtime/execution/runner.py`; under Route A also modify `packages/senselab-core/pyproject.toml`
(optional extra `pydra`), the root `pyproject.toml` (`[tool.uv.sources]` pin at `P`; `pydra` a dependency of the root
`senselab` distribution, P3-4), `uv.lock` (resolution), `.github/workflows/2x-tests.yaml` (`core-pydra`, 3.13),
`.importlinter` (only `runtime.execution.runner` imports `pydra`); tests.

**Interfaces (both variants):** `run_plan(plan: Plan, executor: Executor, *, events: EventLog) -> RunStatus` walks the
plan: for each ready node it builds the node's jobs from upstream output identities, submits them as fragments, reads
`results`, and marks only the descendants of a failed element as failed. The runner owns readiness and descendant
failure; the executor owns execution, release, retries and fault effects.

- **Pass (Route A), `PydraRunner`:** `compile_plan(plan: Plan) -> pydra Workflow` (a `map` edge becomes `.split()`, a
  `gather` edge `.combine()`); each node is a `python.define` task whose `__checksum__` (K2) returns the job's senselab
  key digest from the resolved upstream identities, so pydra never hashes senselab code. `SenselabStore`: pydra's
  `ResultStore` (D1) over senselab's store; `lookup(checksum, context)` applies the run's reuse policy through K2's
  `ReusePolicy` (`accept` = `kernel.identity.acceptable`). `ExecutorWorker`: a pydra `Worker` whose `run(job)` submits
  the job's fragment to the given senselab `Executor` and returns when `results` yields its outcome, raising on a
  failure so E1b marks its descendants; F2's router maps the node's `ComputeResources` tags to resource classes. The
  `Submitter` drives the loop (`on_error="raise_at_end"`); senselab adds no `ready()`/`complete()` to pydra (that split
  is proposal 0001 §3.10, Route B, out of scope).
- **Fail, `NativeRunner`:** a ready-set loop over `kernel.graph.order` with per-element readiness, using the same
  `Executor` calls.

- **Job-lease heartbeat (P3-53):** under Route A pydra holds and renews each job's lease (Task 45); `PydraRunner` passes the execution profile's `lease.stale_after_s` and `lease.renew_fraction` (data, with derivations, bounded by the root's measured attribute delay, Task 16) to pydra's `LeaseSettings`; senselab does not renew job leases itself.

- [ ] **Step 1: Failing tests (fake stack; `core-pydra` CI under Route A, local under the fallback):**
  `test_map_node_submits_one_fragment_per_batch`, `test_errored_element_fails_only_its_descendants`,
  `test_gather_waits_for_every_element`, `test_inprocess_and_pool_give_same_store` (a fake-substrate `Pool` from Task 14
  joins this test there); under Route A also `test_pydra_checksum_equals_logical_key` and `test_no_result_pickle_written`.
  The pydra import confinement is the import-linter contract.
- [ ] **Step 2: Implement** (under Route A, the gate spike rebuilt in reviewed form; `uv lock`, resolution only).
- [ ] **Step 3:** Review and compact, commit `feat(core): the plan runner (<variant>)`; push; CI green. Spec commit:
  the reviewer checks the build against ADR-001's executor text; an amendment is written only if the gate outcome
  requires one (P3-2).

### Task 10: Run lifecycle, events, `ResultSet`, `.map` and the facade

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.2 (`.map`), §4.6
(`sl.Dataset`), §4.7, §4.10, ADR-008 (events, `status`, `explain`), P-1; rulings P3-17, P3-27, P3-28.
**Depends on:** Task 9b.

**Files:** Create `runtime/execution/{run,events}.py`; modify `runtime/planner.py` (`Plan.run`), `runtime/revision.py`
(the attempt id), `cli.py` (`status`, `explain`), `scripts/generate_task_api.py`, `packages/senselab/src/senselab/
{__init__,cli}.py`, `_api/` (regenerated), `kernel/formats.py` (`run-event@1.0`), the root test dependency group
(pandas, so `senselab-lean` runs the frame test), tests.

**Interfaces:**
- Run directory `$SENSELAB_CACHE/2x/runs/<plan_hash>/`: `plan.json`, `manifest.json`, `placements/<attempt>.json`,
  `events/<writer-uuid>.jsonl`, `dead/`, `prov/`, `queues/`, `epoch/`. Provenance `run_id` is the plan hash;
  `revision.run_id()` (from `SENSELAB_RUN_ID`, Slurm or a uuid) names the attempt (P3-17).
- `Event(Formatted)`: `format = "run-event@1.0"`, `kind` in `queued | claimed | started | finished | failed | yielded |
  cache_hit | verify`, `node`, `key`, `job`, `attempt`, `placement_digest`, `fault: FaultClass | None`, `reuse:
  ReusePolicy | None`, `env: str | None`, `t_ns`. `EventLog(run_dir).append(event)` writes to this writer's file only.
- `BrokenEnv(Frozen)`: `env: str`, `fault: FaultRecord`, `failures: int`.
- `status_from(events: Iterable[Event]) -> RunStatus`; `RunStatus{counts: Mapping[JobState, int], faults:
  Mapping[FaultClass, int], broken: tuple[BrokenEnv, ...], done: bool}`; metrics derive from events only (ADR-008).
- `Plan.run(executor: Executor | None = None) -> Run` (default from the profile's substrate: `InProcess` for none,
  `Pool` otherwise); `Run(run_dir)`: `status()`, `wait(timeout: float | None = None) -> RunStatus`, `results() ->
  ResultSet`, `cancel() -> None` (stops releasing; running items finish or are abandoned to their lease).
- `ResultSet`: iterates `(Record, Result | FaultRecord)` in record order; `outputs(port)`; `to_frame()` imports pandas
  lazily and indexes by entities, raising `MissingExtra("pandas")` without it (P3-27).
- Facade: `sl.plan`, `sl.run(target, dataset=None, **plan_kwargs) -> ResultSet`, `sl.resume`, `sl.use(execution=...)`
  (a context-variable session default), `sl.validate`, and `sl.Dataset`, a namespace with `bids(...)` and
  `from_records(...)` over the runtime constructors; every generated task class gains `.map(dataset, *, bind=None) ->
  ResultSet`.
- CLI: `senselab status <run>`, `senselab explain <run> <node|key>` (events, fault, stderr tail, placement reason).

- [ ] **Step 1: Failing tests (fake stack):** `test_map_equals_per_record_calls`, `test_status_derived_from_events_only`,
  `test_two_writers_one_event_log`, `test_cancel_stops_release`, `test_to_frame_indexed_by_entities` (runs in
  `senselab-lean`), `test_explain_names_fault_and_placement`, `test_attempt_id_distinct_from_run_id`,
  `test_sl_dataset_bids_builds_bids_dataset`, `test_generated_api_has_no_drift` (rerun),
  `test_function_and_class_build_same_call` (rerun).
- [ ] **Step 2: Implement; Step 3:** locally the tests that need no runner (`-k "not map and not cancel"`); the rest in
  `core-pydra` (Route A) or locally (fallback): `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution -m "logic or contract"`; `PYTHONPATH=packages/senselab/src uv run --no-sync pytest packages/senselab/tests -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(2x): run lifecycle, events and .map`; push; CI green. Spec commit:
  ADR-006 row `run-event@1.0`; ADR-008 amendment (OpenTelemetry export deferred, P3-28).

### Task 11: `resume` and the plan link in provenance

**Model:** opus implements (identity); fresh sonnet reviews. **Tier:** logic, contract. **Implements:** ADR-004
(restart), ADR-010 (`prov:hadPlan`), `execution.md` §3.1, P-5; ruling P3-29. **Depends on:** Task 10.

**Files:** `runtime/planner.py` (`resume`), `runtime/execution/job.py`, `runtime/provenance.py`, `kernel/prov.py`,
`runtime/data/prov/senselab.jsonld` (if a term is added), tests.

**Interfaces:**
- `resume(plan_hash: str, *, execution: ExecutionProfile | str | None = None, store=None) -> Run`: reads `plan.json`
  and `manifest.json`, re-checks the manifest's stamps (a changed input is `StaleSource`, never a silent re-plan),
  re-probes hardware, writes a fresh placement and runs. It never resolves a ref. A fresh placement that picks another
  backend keeps the plan hash and recomputes that node, because the backend is keyed (ADR-002).
- Every execution and reuse activity carries `prov:qualifiedAssociation` with `prov:hadPlan` → the `senselab:Plan`
  entity `plan:<plan_hash>` (ADR-010); the plan entity is written once per run.

- [ ] **Step 1: Failing tests (fake stack):** `test_resume_executes_only_missing` (interrupt after k of n by a planted
  stop at the job boundary; resume executes n−k, reuses k), `test_resume_never_reresolves_ref` (the resolver seam
  counts zero calls), `test_resume_with_other_profile_same_backend_reuses`, `test_resume_with_other_backend_recomputes`,
  `test_resume_refuses_stale_manifest`, `test_activity_had_plan`.
- [ ] **Step 2: Implement; Step 3:** as Task 10 Step 3, scoped to `{planner,provenance}_test.py` and
  `runtime/execution`.
- [ ] **Step 4:** Review and compact, commit `feat(core): resume by plan hash; activities name their plan`; push; CI
  green. Spec commit: ADR-010 amendment log line (the plan link is recorded).

### Task 12: Lease renewal (R-111)

**Model:** opus implements (leases); fable reviews (lease protocol proof, roster rule). **Tier:** logic; limits on
ORCD (cross-node NFS). **Implements:** ADR-003 (amendment 2026-11-03), H-R40 (no modification in place), `execution.md` §3.2,
R-111; ruling P3-30. **Depends on:** Task 2.

**Files:** `runtime/locks.py`, `packages/senselab-core/tests/runtime/{locks,locks_timing}_test.py`, the cross-node
probe beside the Phase 1 NFS checks.

**Interfaces** (ADR-003 amendment 2026-11-03, owner item 58: the lease becomes a directory; this task replaces
`_try_take`'s `O_EXCL` link and `_take_aside`'s `(dev, ino, mtime)` judgement):
- **Take:** `mkdir <name>.lease`, then `mkdir <name>.lease/<token>`; an `EEXIST` on the second with the holder's own
  token is its own (lost NFS reply). Every path is inside `<name>.lease`, so nothing crosses a filesystem.
- `Lease.keep_alive() -> ContextManager[LeaseState]`: a renewer thread owned by the lease, independent of the work,
  creates `<token>/beat-<n+1>` and removes `<token>/beat-<n>` every `pool.lease_renew_fraction × stale_after` (data,
  with a derivation; bounded by the root's measured attribute delay, Task 16). This is the heartbeat; a task that
  writes nothing keeps its lease. A beat `mkdir` failing with `ENOENT` sets `LeaseState.lost`; the holder stops and
  does not commit as owner (a content-addressed result committed by rename stays idempotent).
- **Judge and break:** the lease's age is a fresh `stat` of `<token>/` against the storage clock (a probe
  directory's mtime), never the node clock; past `stale_after` a breaker renames `<name>.lease` aside and takes its
  own. **Release:** rename aside, then remove.
- No file is written or rewritten under the lease (D-2, H-R40); no pid-liveness takeover.

- [ ] **Step 1: Failing tests (local):** `test_keep_alive_renews_while_the_work_writes_nothing`,
  `test_beat_after_break_marks_lost` (`ENOENT` on the beat), `test_lost_mkdir_reply_with_own_token_is_held`
  (`EEXIST` injected at the `os` boundary), `test_dead_holder_lease_broken_after_bound`,
  `test_breaker_never_breaks_a_beating_lease` (storage-clock time advanced at the `os` boundary),
  `test_no_file_is_written_under_the_lease`, `test_root_without_atomic_mkdir_is_refused`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/locks_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): a lease is a directory, renewed by beats`;
  push; CI green. Spec commit: ADR-003 amendment (renewal); close R-111.
- [ ] **Step 5 (orcd, three CPU nodes on the shared filesystem, at that SHA):** a holder on node A renews past
  `stale_after` while B and C try to break it; then A is SIGKILLed and B takes it within one bound. Recorded beside the
  Phase 1 checks (a)–(f).

### Task 13: Queues and plan-fragment items

**Model:** opus implements (concurrency); fresh opus reviews. **Tier:** logic; limits on ORCD. **Implements:**
ADR-003 (index segments, oldest-first, O(1) amortised claims), `execution.md` §3.2, P-37; rulings P3-31, P3-32.
**Depends on:** Tasks 9, 12.

**Files:** Create `runtime/execution/queue.py`, rewritten from the ADR-003 design (P3-31): `src/senselab/utils/
work_pool.py` is not moved, because `src/senselab/utils/reproduce/{slurm,spec,restricted}.py` and five 1.x tests still
import it (R-128); the ledger names it as the source, and its `payload_digest` and `pool_tag` rows close as replaced by
the fragment digest and a label digest. Tests.

**Interfaces:**
- `QueueSet(root: Path)`; `queue(name: str) -> Queue`, `name` a resource class plus `#<model identity>` for a resident
  model (`gpu:h100#<id>`).
- `Queue.put(fragment: PlanFragment) -> None`: writes the item under its `identity_digest("plan-fragment@1", …)`
  (first writer wins) and appends one line to the head index segment `segments/<seq>.jsonl` under a segment lease; a
  segment rolls at `pool.segment_items`.
- `Queue.claim(worker: str, max_items: int) -> Claim | None`: oldest unclaimed item first from the oldest live segment;
  a claim is a `Lease` on the item; `Claim.renew()`, `Claim.commit()` (to `done/`), `Claim.release()` (back to ready),
  `Claim.dead(fault: FaultRecord)` (to `dead/`).
- `Queue.depth() -> int`, `Queue.oldest_age_s() -> float | None`, both O(segments), not O(items).

- [ ] **Step 1: Failing tests (local):** `test_claims_oldest_first_across_segments`, `test_double_claim_commits_once`
  (two processes on the fake stack, planted delay at the `os` boundary, under a second, so `logic`),
  `test_expired_claim_reclaimed`, `test_claim_cost_flat_in_queue_size` (syscalls counted at the `os` boundary at 10²
  and 10⁴ items, not timed), `test_fragment_is_never_pickled`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/queue_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): queues of plan fragments with index segments`; push; CI green.
  Ledger rows for `work_pool.py` (spec commit).
- [ ] **Step 5 (orcd, three CPU nodes, at that SHA):** 3×8 claimer processes over 10⁴ fake items on the shared
  filesystem: every item committed exactly once, duplicates counted, claim order monotone per segment.

### Task 14: The `Pool` executor: epoch, release, fault effects

**Model:** opus implements (concurrency, faults); fresh opus reviews. **Tier:** logic; limits on ORCD. **Implements:**
ADR-003 (idempotence, epoch), ADR-009 (typed faults; gap faults), `execution.md` §3.1, §3.3, R-83 (executor rule);
rulings P3-33, P3-34, P3-36. **Depends on:** Task 13.

**Files:** Create `runtime/execution/pool.py`; modify `cli.py` (`senselab coordinate <run>`: takes an epoch and runs
`run_plan` with a `Pool`), tests.

**Interfaces:**
- `Epoch(run_dir)`: `take() -> int` (the next epoch by `mkdir epoch/<n>`, ADR-003 amendment 2026-11-03), `current() -> int`.
- `Pool(run_dir, *, queues: QueueSet, store: ResultStore, events: EventLog, profile: ExecutionProfile, substrate:
  Substrate, epoch: int)`: an `Executor`. `submit(fragment)` reads `Epoch.current()` and raises `StaleCoordinator` when
  a newer epoch exists, then puts the fragment on its queue; `results(handle)` reads outcomes from the store and
  `dead/` only, never from worker memory (P3-34).
- Fault effects, owned here once (P3-33, P3-36):
  - **content** fault with `gap: false`: the item's attempt count rises; at `faults.content_attempts` it goes to
    `dead/` and `results` yields its failure, so the runner fails its descendants;
  - **resource** fault on a single-item request (the pool worker has already bisected, Task 15a): re-submitted on the
    next larger memory class the profile lists, uncharged; after `faults.resource_replacements` it is a resource
    failure;
  - **infrastructure** fault, or any fault with `gap: true`: re-submitted uncharged and reported as a classification
    gap when `gap`; after `faults.infrastructure_replacements` failures of one env the env is reported once as
    `BrokenEnv` and its queue is held, never failing its items one by one.

- [ ] **Step 1: Failing tests (local, fake stack, fake substrate, fragments submitted directly):**
  `test_two_coordinators_one_result_per_key`, `test_stale_coordinator_stops_after_new_epoch`,
  `test_content_fault_dead_after_budget`, `test_gap_fault_never_dead`,
  `test_resource_fault_moves_to_larger_class_uncharged`, `test_broken_env_reported_once`,
  `test_results_read_from_store_only`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/pool_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): the Pool executor with an epoch fence and typed fault
  effects`; push; CI green. Close R-83's executor part.

### Task 15a: Pool workers: claim, quantum, execute, renew, bisect

**Model:** opus implements (worker protocol, concurrency); fresh sonnet reviews. **Tier:** logic; limits on ORCD.
**Implements:** `execution.md` §3.2, §3.4 (quantum), ADR-003 (yield), R-119, R-189, R-161; rulings P3-35, P3-36.
**Depends on:** Task 14.

**Files:** Create `runtime/execution/pool_worker.py`; modify `cli.py` (`senselab execute --pool <dir> --classes
<list>`), `runtime/worker_client.py` (timeouts from the profile; exit −9 with a cgroup OOM record → `ResourceFault`),
`senselab_worker/faults.py` (traceback limit as a total across the chain), `senselab_worker/serve.py` and the `init`
record (model identity, device and dtype, so worker loads go through `ModelCache`, R-161; `wire-protocol.md` version
bump), tests.

**Interfaces:**
- `serve_pool(pool_dir: Path, classes: Sequence[str], *, profile: ExecutionProfile, store: ResultStore) -> int`: a host
  `senselab_core` process that drives each claim's backend worker through `WorkerSet` (one worker protocol, P3-35);
  picks the class with the oldest work; yields after `pool.quantum_items` items or `pool.quantum_s` seconds when another
  of its classes has older work; renews its claims from its wait loop; exits when its queues stay empty for `pool.poll_s
  × 3` or on SIGTERM after committing or releasing its claims.
- A request that fails as a whole (`WorkerFatal`, `WorkerLost`, a resource fault) is bisected inside the claim until
  the failing item is alone; that item's fault goes to the `Pool` effects (Task 14). Per-item failures inside a request
  are never bisected (P3-36).
- `WorkerClient(load_timeout=profile.timeouts.load_s, request_timeout=profile.timeouts.request_s)`; a timeout is a
  `ResourceFault` (R-189). The traceback limit is 8 frames in total across the cause chain (R-119).

- [ ] **Step 1: Failing tests (local, fake stack; process-spawning tests under a second are `logic`, the rest
  `limits`):** `test_worker_yields_after_quantum_to_older_queue`, `test_failing_batch_bisected` (one poisoned item in 8;
  7 commit), `test_claim_renewed_during_long_item`, `test_timeout_is_resource_fault`, `test_oom_kill_is_resource_fault`
  (exit −9 with a planted cgroup event), `test_traceback_limit_total_across_chain`,
  `test_worker_loads_through_model_cache`, `test_sigterm_releases_claims`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/pool_worker_test.py packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py -m "logic or contract or (limits and not model)"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): pool workers with quanta, renewal and bisection`; push; CI
  green. Spec commit: `wire-protocol.md` (init record, traceback limit); close R-119, R-161, R-189.

### Task 15b: Worker supervision

**Model:** opus implements (process lifecycle); fresh sonnet reviews. **Tier:** logic; limits on ORCD (Linux).
**Implements:** R-120, P-34; ruling P3-37. **Depends on:** Task 15a.

**Files:** Create `runtime/execution/supervisor.py`; modify `runtime/worker_client.py`, `runtime/vllm_server.py`,
`runtime/launch.py`, tests.

**Interfaces:** `Supervisor()` as a context manager in the pool worker: on Linux it sets `PR_SET_CHILD_SUBREAPER`, so
orphaned backend descendants (fork without exec, `setsid` escapees) re-parent to it and are reaped and killed when
their worker ends; group liveness is judged from its own child table, never from `ps` or `/proc` scans. On macOS it
uses the process group and states the limit in `wire-protocol.md` (P3-37).

- [ ] **Step 1: Failing tests (Linux-gated in CI):** `test_setsid_escapee_killed_with_worker`,
  `test_fork_without_exec_child_reaped`, `test_liveness_from_supervisor_not_ps`.
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(core): supervised worker processes`; push; CI
  green (the Linux job runs the gated tests). Close R-120.
- [ ] **Step 4 (orcd, CPU node, at that SHA):** `-m limits -k supervisor` with a host SIGKILL of the pool worker: no
  descendant survives.

### Task 16: Substrates: local processes and Slurm; storage capability probe

**Model:** sonnet implements; opus reviews (Slurm, caps). **Tier:** logic (fake substrate), limits (local pool);
limits on ORCD (Slurm). **Implements:** design §4.9 (local and Slurm), `execution.md` §3.6, P-29 (QOS caps); rulings
P3-38, P3-39. **Depends on:** Task 15b.

**Files:** Create `runtime/execution/substrates.py` (rewritten; the 1.x `Controller` is its named source, P3-31);
modify `runtime/storage.py` (`probe_capabilities`), `runtime/planner.py` (probe at plan time),
`runtime/data/profiles/{laptop,orcd}.yaml` (substrate and resource classes), `cli.py`, tests.

**Interfaces:**
- `Substrate` protocol: `start(resource_class: str, count: int) -> list[WorkerHandle]`, `running(resource_class) ->
  int`, `stop(handles) -> None`.
- `Local(max_workers: int)` spawns `senselab execute` processes; `Slurm(classes: Mapping[str, ResourceClass],
  qos_cpu_cap: int)` submits one `sbatch` job per worker with the class's partition, gres, cpus, memory and time, named
  by plan hash and attempt id; it never targets a node the profile excludes.
- `scale(queues: QueueSet, substrate: Substrate, profile: ExecutionProfile) -> None`: per class, target workers =
  min(queue depth ÷ `pool.quantum_items`, `max_workers`, QOS CPU cap ÷ cpus per task), called by the coordinator each
  round.
- `probe_capabilities(root: Path) -> StorageCapabilities{atomic_mkdir, directory_rename, dir_mtime_server_clock,
  atomic_rename, attribute_delay_s}`, measured on the root itself (not assumed from its filesystem type), so NFS
  v3/v4, Lustre, GPFS, SMB and local disks are each judged by what they do; a store with no directories (an object
  store) fails `atomic_mkdir`. A plan whose store or run directory fails it is refused with `StorageUnsupported`
  naming the missing primitive; `attribute_delay_s` bounds the lease renew fraction (Task 12).
- A pool runs only where its store is reachable from every substrate (P3-39).

- [ ] **Step 1: Failing tests (local, fake `sbatch` on `PATH`):** `test_scale_bounded_by_qos_cap`,
  `test_slurm_args_from_resource_class`, `test_excluded_node_never_targeted`, `test_unsupported_store_refused_at_plan`;
  `limits`: `test_local_pool_runs_toy_plan` (fake stack, 2 workers, ≤ 10 s).
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/substrates_test.py packages/senselab-core/tests/runtime/storage_test.py -m "logic or contract or (limits and not model)"` (the plan-walking local-pool test runs in `core-pydra` under Route A).
- [ ] **Step 4:** Review and compact, commit `feat(core): local and Slurm substrates`; push; CI green.
- [ ] **Step 5 (orcd, coordinator CPU job, at that SHA):** a 200-item fake-stack plan on `Pool` with the `slurm`
  substrate: workers scale up and down within the cap, every item committed once, events complete.

### Task 17: Services: the leased registry, consumer leases, the `svc` pool worker

**Model:** opus implements (service lifecycle, leases); fresh sonnet reviews. **Tier:** logic; limits on ORCD GPU.
**Implements:** ADR-003 (services, Phase 3 stage), `execution.md` §3.5, I-8, design §4.4 (`svc`); ruling P3-40.
**Depends on:** Tasks 12, 16.

**Files:** Create `runtime/execution/services.py`; modify `runtime/vllm_server.py` (started by the `svc` worker),
`runtime/execution/{workers,pool_worker}.py`, `kernel/formats.py` (`service-entry@1.0`), tests.

**Interfaces:**
- `ServiceRegistry(root: Path)`: `register(identity: Attestation, endpoint: str) -> ServiceLease` (an entry
  `{identity, endpoint}` under a `Lease` kept alive by `keep_alive`; the lease renewal is the liveness signal, so the entry has no separate heartbeat field); `lookup(identity) -> ServiceEntry | None`;
  `consume(identity) -> ConsumerLease`.
- A `svc:<id>` queue is served by a pool worker that starts the server, registers it and renews its lease; it exits
  when no consumer lease has been live for `services.idle_exit_s`; queue depth does not decide this.
- Every reply's attestation is checked against the entry (I-8).

- [ ] **Step 1: Failing tests (local, stdlib stand-in server):** `test_service_registered_under_renewed_lease`,
  `test_service_exits_after_idle_consumers`, `test_two_consumers_one_server`, `test_attestation_checked_per_reply`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/services_test.py packages/senselab-core/tests/runtime/vllm_server_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): leased services and the svc pool`; push; CI green. Spec
  commit: ADR-003 amendment (the Phase 3 service stage built; the registry entry is `{identity, endpoint}` under a
  renewed lease, replacing `{identity, endpoint, heartbeat}`).
- [ ] **Step 5 (orcd, GPU H100, at that SHA):** `-m "limits and gpu and model" -k services`: two plans share one
  vLLM server for the PII model; restarting with a new config changes keys (I-8).

### Task 18: Cross-record batching under I-3

**Model:** opus implements (identity); fresh sonnet reviews. **Tier:** logic; limits (`model`) on ORCD.
**Implements:** I-3, ADR-004 (batch formation), P2-82, R-164; rulings P3-41, P3-42. **Depends on:** Task 14.

**Files:** Create `runtime/execution/batching.py`, `runtime/data/batching.yaml` (`batching@1.0`); modify
`runtime/keys.py` (`request_components` takes the batch), `runtime/execution/{pool,job,in_process}.py`, tests.

**Interfaces:**
- `BucketGrid(Frozen)`: `id: str`, `edges_ns: tuple[int, ...]` from `batching.yaml` with its derivation.
- `form_batches(jobs: Sequence[Job], backend: BackendSpec, *, batch_size: int, grid: BucketGrid) -> list[Batch]`;
  `Batch{jobs, formation: str}`, and the key of every job in a batch is built with that formation in this same task:
  - `batch_invariant=True`: any jobs up to `batch_size`, formation `mixed` (placement only, not keyed);
  - not invariant, `batch_formation` output-affecting: grouped by length bucket, formation `bucket:<grid id>:<index>`,
    keyed;
  - not invariant otherwise: batch size 1, formation `single` (as since Task 9).
- Batches form across records of one node and one placement only.

- [ ] **Step 1: Failing tests (local, fake stack):** `test_noninvariant_backend_batches_by_keyed_bucket_or_single`,
  `test_invariant_backend_formation_not_in_key`, `test_bucket_in_key_changes_key`, `test_batches_never_cross_nodes`.
- [ ] **Step 2: Implement; Step 3 (local):** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/batching_test.py packages/senselab-core/tests/runtime/keys_test.py -m "logic or contract"`.
- [ ] **Step 4:** Review and compact, commit `feat(core): cross-record batching under I-3`; push; CI green.
- [ ] **Step 5 (orcd, CPU then GPU H100, at that SHA):** per Phase 2 backend, batch of 1 against bucketed batches on
  the held-out split, compared by kernel type under the run-config thresholds (P2-95). Agreement is recorded as
  evidence; a backend that agrees on every item may be declared invariant in a later commit with a version bump
  (P2-82). R-164 closes with the measured confidence differences as a finding.

### Task 20: The scale ladder and the catalogue resource tables

**Model:** sonnet implements; opus reviews. **Tier:** benchmark (not pytest) on ORCD; logic for the generator.
**Implements:** P-33, P-7, design §4.3 (resource tables), §6.4 (packed tier and token arbiter wait on this), R-73,
R-165; rulings P3-23, P3-44. **Depends on:** Task 18.

**Files:** Create `benchmarks/scale/{ladder.py,budgets.yaml,bids_tree.py,suites/ladder.yaml}`,
`scripts/generate_resources.py`; generated `packages/senselab/src/senselab/data/resources/<task>/<backend>.yaml`;
modify `runtime/audio_io.py` and `kernel/signals.py` for R-73 (a range read keeps the chunk list and reads only its
chunks), `packages/senselab/src/senselab/data/server_settings/pii_spans_vllm.yaml` (derivations, R-165), tests.

**Interfaces:**
- `bids_tree.py`: `synthetic(root, n: int, seed: int)` writes a deterministic BIDS tree of `n` tone recordings
  (byte-identical on every host); `from_librispeech(root, manifest, n)` lays `n` LibriSpeech test-clean utterances into
  BIDS (`sub-<speaker>/ses-01/audio/sub-<speaker>_ses-01_task-read_run-<k>_audio.flac`) from the Phase 2 fetch manifest.
- `ladder.py run <suite> --execution <profile>`: the suite file names tasks and scale points (1, 10², 10⁴, long);
  no per-knob flags (design §4.10). It runs each plan, reads the events and writes `scale-result@1.0` (per-item wall,
  peak RSS, files written, bytes moved, store stats, coordinator and claim overhead) to
  `$SENSELAB_CACHE/2x/benchmarks/scale/`.
- `budgets.yaml` (`scale-budgets@1.0`): per-item ceilings and the flatness rule (per-item cost at 10⁴ within the
  budget's ratio of the cost at 10²), each with a derivation from the first idle-node run; never fitted at run time.
- `generate_resources.py`: from scale results, writes `resource-table@1.0` per (task, backend, hardware class):
  `load_s`, `peak_memory_bytes`, `s_per_input_s`, with the SHA and job ids as derivation; never hand-edited. The vLLM
  `max_model_len` and `max_num_seqs` derivations come from the same H100 run (R-165).

- [ ] **Step 1 (local):** `test_synthetic_tree_is_byte_identical_for_a_seed`, `test_resource_table_derivation_names_jobs`,
  `test_range_read_reads_only_its_chunks` (bytes read counted at the `os` boundary, R-73).
- [ ] **Step 2: Implement; Step 3:** review and compact, commit `feat(bench): the scale ladder and resource tables`;
  push; CI green.
- [ ] **Step 4 (orcd, idle CPU node and idle H100, at that SHA):** the ladder for `resample`, the fake stack and every
  Phase 2 backend at 1, 10², 10⁴ (CPU, `resample` and the fake stack) and a long signal (two hours of audio); store
  budgets measured.
- [ ] **Step 5:** commit the generated resource tables, the vLLM derivations and the first `budgets.yaml`
  derivations, reviewed; close R-73 and R-165. A scale point that is not flat is a register row, not a budget change.

### Task 21: Phase 3 exit checks

**Model:** sonnet implements; opus reviews; the P-1 check uses a fresh sonnet agent with no repository access.
**Tier:** all. **Depends on:** Tasks 0–20 and the pydra tasks' status.

- [ ] **Step 1 (CI):** `2x-tests` green at the head of `2x/phase-3`: `detect`, `pre-commit`, `core-only` (with the
  import contracts), `senselab-lean`, `core-limits`, `source-digests`, `pr-models`, `core-env`, `licenses`,
  `worker-floor`, and `core-pydra` under Route A; the nightly green on `2.x`.
- [ ] **Step 2 (P-10, laptop and ORCD):** `bids_tree.synthetic(n=1000, seed=0)` on both hosts;
  `sl.signal.Resample(target_rate=8000).map(sl.Dataset.bids(root))` planned under `laptop` and under `orcd`: one
  `LogicalPlan` hash; a field-level diff of the two plans lists differences only inside `placement`. Run on the laptop
  with `InProcess` and with `Pool` on `local` (2 workers), bounded at 15 min wall and 2 GB resident with load and
  memory watched; run on ORCD with `Pool` on `slurm`. Equal output identities on both hosts (P3-45).
- [ ] **Step 3 (P-10 with a model, ORCD):** `bids_tree.from_librispeech(n=1000)`; `SpeechToText().map(...)` planned
  under `orcd` and under `laptop` on an ORCD CPU node: one hash; run under `orcd` with `Pool` on `slurm` (H100 class).
- [ ] **Step 4 (P-5, ORCD):** Step 3's run is interrupted (`scancel` of the coordinator and every worker once a third
  of the items are committed), then `resume(<hash>)`: events and provenance show executions = items not committed
  before the interrupt, reuses = items committed, and no ref resolution.
- [ ] **Step 5 (ORCD, CPU and GPU H100, head SHA):** the full packages suite, every tier, serial; env builds for every
  shipped lock; the Task 12 and 13 cross-node checks.
- [ ] **Step 6 (P-1, ORCD):** the fresh agent receives only `sl.describe()` output and a dataset path, writes a plan
  call (task, params, dataset), and `sl.plan(...).run()` runs it.
- [ ] **Step 7:** if the gate failed at Task 0 it is re-evaluated; each pydra task 40–53 is merged into
  `enh/execution-extensions` or carried by a register row naming its owner and reason. P-39 reconciliation again.

### Task 22: Phase 3 docs and spec pass

**Model:** sonnet implements; opus reviews. **Tier:** contract (docs build) for the pages; none for the spec pass.
**Branches:** pages on `2x/phase-3`; spec on `2x/phase-3-spec`. **Implements:** docs.md (Phase 3 onwards), D-7;
rulings P3-46, P3-47. **Depends on:** Task 21.

- [ ] **Step 1 (pages):** reference pages for plans, runs, executors, datasets and the profile keys, generated from the
  code; one example page on public data (the LibriSpeech BIDS `.map` of Task 21 Step 3, its plan and `explain()`); the
  status page gains plan and executor rows. `uv run --no-sync --group docs pdoc` or the docs-preview job. Commit
  `docs(2x): Phase 3 reference and examples`; push; CI green.
- [ ] **Step 2 (spec):** design §4.9 and §9 (gate outcome), §6.2 (new integrations accepted by P2-95 comparisons),
  §4.8 I-3 and I-4 check column and `testing.md` §4 (P2-95 comparisons replace tolerance profiles, P3-46), `testing.md`
  §2 and §5 (3.13 default; `core-pydra` under Route A), `execution.md` §3.2 (the pool worker drives the env's worker,
  P3-35), ADR-003 (renewal, services), ADR-008, ADR-010; each amendment dated. ADR-001 changes only if Task 9b's check
  found the gate outcome requires it. Commit `docs(2x): Phase 3 spec pass`; push.

### Task 23: Phase 3 exit adversarial round (P-35)

**Model:** opus lead, effort higher, with up to 5 sonnet subagents; claims verified by haiku (P-40). **Tier:**
read-only. **Depends on:** Task 22.

- [ ] **Step 1:** brief the lead with the Phase 3 surfaces: `graph@1.0` and `validate`, the BIDS adapter and manifests,
  `LogicalPlan` and its hash, placement, taint and caps, the job engine and `InProcess`, the selected runner, run
  lifecycle and events, `resume`, lease renewal, queues and claims, the `Pool` and epoch, pool workers, bisection and
  supervision, substrates and the storage probe, services, batching, the scale ladder, and the pydra PRs of Tasks
  40–53. Goal: inputs, races, crashes and misuse that break ADR-001, 002, 003, 004, 008, 009, 010 or the Phase 3 exit;
  plus the P-37 duplicate-finding pass over `packages/` and `benchmarks/`. Every finding carries an architectural
  resolution.
- [ ] **Step 2:** findings become `review-register.md` rows from the next free id, lens `EXT3`, on `2x/phase-3-spec`;
  findings saved under `reviews/phase3-exit-adversarial/`.
- [ ] **Step 3:** commit `docs(2x): Phase 3 exit adversarial round`; push.

### Task 24: Phase 3 exit fix round

**Model:** per finding, from the roster (opus for kernel, concurrency, identity, leases; fable reviews a lease or store
protocol fix; sonnet otherwise); each fix reviewed by a different model. **Depends on:** Task 23.

- [ ] **Step 1:** fix every high and medium finding by an architectural change (P-35); schedule the rest with an owner.
  Each fix is reviewed and compacted before its commit (P2-34).
- [ ] **Step 2:** CI green; rerun the affected ORCD steps of Task 21 at the new head.
- [ ] **Step 3:** `STATE.md`: Phase 3 done, with the head SHA, the gate outcome and ORCD evidence by SHA; log line.
  Commit `docs(2x): Phase 3 done` on `2x/phase-3-spec`; push.

### pydra: K2 and the Phase 3 PRs (Tasks 40–53)

Work in `$PYDRA_EXT`. Each PR is a branch `enh/ee-<id>-<slug>` cut from `enh/execution-extensions` at its current head
when it depends on a PR merged there, else from `main`; a **draft** PR into `enh/execution-extensions` citing proposal
0001 (#891), merged once pydra's full CI tier is green (P3-6). Tests: failing tests first in pydra's suite, on `debug`
**and** `cf`, and on Slurm where the area touches it (owner items 52, 53); review and compaction by the named reviewer
against senselab's principles (owner item 50); ORCD run on a CPU node in a locked venv built in the job (uv-managed
3.13), `pytest pydra -k <area>` serially; nothing runs locally. None of Tasks 41–53 blocks a senselab task. senselab's
Route A uses its own queues through `ExecutorWorker` (Task 9b); pydra's I is a generic plugin and senselab does not use
it in Phase 3 (P3-52).

| Task | PR | Model (implements / reviews) | Interfaces and tests | Depends on |
|---|---|---|---|---|
| 40 | **K2** node-supplied checksum, reuse hook (inside Task 0) | opus / fresh opus | `__checksum__(self, inputs) -> str` or `define(checksum=callable)`; `Job.checksum` equals the supplied value exactly and no input is hashed; `ResultStore.lookup(checksum, context)` compares through `ReusePolicy.accept(stored, current, task) -> bool`; built-in `exact`, `any`; `Submitter(reuse=ReusePolicy \| "exact" \| "any")`; `test_supplied_checksum_is_job_checksum`, `test_reuse_hook_marks_stored_context`, `test_tasks_without_hook_keep_checksums` | K1 |
| 41 | **A0** Slurm `--error` fix | sonnet / opus | both sites fixed; a test submits with `--error` set | — |
| 42 | **A2** batched polling, arrays | sonnet / opus | one `squeue --me` per poll cycle; `sbatch --array` for split states through the tuple path; ORCD Slurm test of 100 elements | A1 |
| 43 | **B1** benchmarks | sonnet / opus | proposal 0001's scripts in `benchmark.py`; per-item cold and cached cost printed for `debug` and `cf` | — |
| 44 | **B2** cheap submission | opus / fresh sonnet | no re-pickle for in-process workers; `_job.pklz` only for remote and failed jobs; one `scandir` per cache root per round; targets < 0.5 ms cold and < 0.1 ms cached per trivial item, measured by B1 on an idle node | B1 |
| 45 | **C** leases and atomic commit inside the store | opus / fable (lease proof) | the job runs in `<checksum>/` under an ADR-003 lease (amendment 2026-11-03): a directory taken by `mkdir` with the holder's token, kept alive by beat directories from a renewer independent of the job (a long job that writes nothing keeps its lease), broken by rename aside once its age on the storage clock passes `stale_after`, no pid-liveness takeover (P3-48); the heartbeat is pydra's: `stale_after` and `renew_fraction` are a pydra setting (`Submitter(lease=LeaseSettings(stale_after=, renew_fraction=))`, store-level default documented) so pydra keeps its own job leases alive without senselab; every path stays inside the lease or the target's directory, so nothing crosses a filesystem; result committed by write-then-rename; `clean_stale_locks` becomes lease expiry, and `Submitter._check_locks`' age test (it removes any lockfile older than the current submitter's start, on by default for the debug worker, so on a shared cache it can remove a live holder's lock) is replaced by the lease's liveness (found in the #907 review); tests: two-process race, a job running past `stale_after` that writes nothing keeps its lease, SIGKILL then takeover after `stale_after`, a root without atomic `mkdir` refused, read-only shared cache. The same PR amends proposal 0001 §3.3, and the senselab spec commit for this task amends `pydra-extension-points.md` §5's "a lease with a heartbeat" line to match | D1 |
| 46 | **D2** `PortableStore` | opus / fresh sonnet | JSON for scalars and attrs records, `.npy` arrays, files as files, pickle only for unregistered types; versioned layout `tasks-v<N>/`; no reader of an older layout (owner item 52) | D1 |
| 47 | **E2** partial `State` refactor | opus, effort higher / fresh opus | per-element readiness for the re-split case; the full splitter/combiner matrix and `debug`/`cf` parity | E1a, E1b |
| 48 | **G2** `Venv` environment and runner | sonnet / opus | `Venv(path= \| python=, lock=)`; same-version tasks cloudpickled, others by import reference with `PortableStore` inputs; a `__main__`/closure function raises a clear error; uv-managed interpreters only | D2, G1 |
| 49 | **H0** Python tasks without `chdir` | sonnet / opus | opt-in; default unchanged; two jobs share one process without touching each other's cwd | — |
| 50 | **H** provided objects and batching | opus / fresh sonnet | `Arg(provided=<name>)`; only the provider's key enters the checksum; `Submitter(providers=...)` LRU under a budget; batched implementations bisected on failure | H0, F1, D1 |
| 51 | **I** `pydra-workers-pool` | opus / fresh opus | a generic lease-queue worker on a shared filesystem with index segments and oldest-first claims (the ADR-003 design of Task 13, written for pydra's job model); controller scales from queue depth within caps; local and Slurm | C, D2, F2 |
| 52 | **J1** provenance via messengers | sonnet / opus | one JSON-lines record per job: used checksums, outputs, environment checksum, host, times, status | G1, D1 |
| 53 | **J2** `Submitter.plan()` | sonnet / opus | nodes, split sizes, resources, worker per node, expected hits; run-time-built workflows marked deferred | F2, B2 |

---

## Self-review notes

- **Spec coverage, design §6.2 Phase 3.** Immutable `Plan`, `explain`, run lifecycle, `resume` → Tasks 7, 8, 10, 11.
  `InProcess` and `Pool` on local and Slurm → Tasks 9, 9b, 13–16. Plan-fragment items → 9, 13; coordinator with an
  epoch file → 14; faults typed at the source → 14, 15a. Cross-record batching under I-3 → 9 (single until keyed), 18.
  Minimal BIDS adapter → 6. Scale ladder and resource tables → 20. Exit P-10 → 21 Steps 2–3; exit P-5 → 11 and 21
  Step 4. The gate → 0; its runner → 9b. The pydra items A0, A2, B1, B2, C, D2, E2, G2, H0, H, I, J1, J2 → 41–53; K2
  → 40.
- **Also served:** ADR-003 services (17) and lease renewal (12); ADR-008 events (10); ADR-010 `prov:hadPlan` (11); the
  graph document of ADR-001 (5); taint I-9 (8).
- **Register rows owned by Phase 3:** R-73 → 20; R-83 (executor rule) → 14; R-110 → 9; R-111 → 12; R-116 (taint) → 8;
  R-119 → 15a; R-120 → 15b; R-121 → 6 (decode errors), 9 (keying after check), Phase 5 (other codecs); R-128 → per
  module move (Task 13 rewrites rather than moves `work_pool.py`; the `reproduce` modules move after Phase 3); R-160 →
  9; R-161 → 15a; R-162 → 2; R-163 → 7; R-164 → 18; R-165 → 20; R-166 → 9; R-167 → 9; R-172 (remainder) → 2; R-187
  (producer) → 9; R-188 (remainder) → 4a; R-189 → 15a; R-190 → 2; R-191 → 4a; R-192, R-193, R-194 → 3; R-195 → 2;
  R-196 → 4b; R-197, R-198 → 4a; R-199 → 8; R-200 → 9; R-201 → 2. None deferred.
- **Not in Phase 3:** `Pipeline`, route, loop, nesting and `Absent` (Phase 4); dataset profiles and sidecar mapping
  (Phase 4); the packed store tier and the GPU token arbiter (§6.4, unless Task 20 shows the need); laptop-to-cluster
  submission (P3-39); OpenTelemetry export (P3-28); Route B (proposal 0001 §3.10); vLLM-omni and vLLM ASR (after Phase
  3, accepted against the transformers stack by P2-95 comparisons).
- **Type consistency.** `JobRef`, `PlanFragment`, `FaultRecord`, `Job`, `JobOutcome`, `Executor`, `Placement` (with
  `resource_class`, `batch_size`, `partition`), `Plan`, `LogicalPlan`, `InputManifest`, `WorkerSet`, `BrokenEnv`,
  `ResourceClass` and `LicenseSet` are each defined once (Tasks 5–10) and used by those names after.

## Rulings made in this plan

Each is the planner's choice where the specs left one open, unless marked as a controller ruling; the owner reviews
them. A ruling that changes an ADR decision takes effect only through the dated amendment its task commits (ADR-000).

- **P3-1** ruling (owner attention): K2 was a Phase 2 gate prerequisite (design §6.2) but was not delivered; the
  branch has no `__checksum__` and no `ReusePolicy`. Task 0 delivers K2 (Task 40) before deciding, rather than failing
  the gate at once. This deviates from §4.9's "if … not working on the branch", so Task 0 Step 4 commits the §4.9 and
  §6.2 amendments. If K2 fails at fable after escalation, the gate fails. Tasks 1–8 proceed meanwhile.
- **P3-2** controller ruling (supersedes the planner's): `NativeRunner` is built only if the gate fails (design §4.9,
  §6.3, "fallback only if triggered"). `InProcess` runs plans on the runner the gate selects. ADR-001 is not narrowed
  beyond what the gate decides, and no ADR-001 change is planned unless the outcome requires one. The runners meet at
  the `Executor` protocol; neither is forced into a `ready()/complete()` protocol pydra lacks.
- **P3-3** ruling: "Working on the branch" in design §4.9 means: merged into `enh/execution-extensions` at one pinned
  commit, green on pydra's full CI tier, passing the area tests on ORCD under `cf` and Slurm, and passing the senselab
  spike's four contracts at that commit.
- **P3-4** ruling: Under Route A pydra is an optional `pydra` extra of `senselab-core` (so `core-only` stays light,
  P-13) and a dependency of the root `senselab` distribution (so `.map` works out of the box), pinned by commit through
  `[tool.uv.sources]` and `uv.lock`; only `runtime.execution.runner` imports it (import-linter).
- **P3-5** ruling: Phase 3 code is on `2x/phase-3`, which starts at `2x/phase-2`'s head because Phase 2 is not yet
  merged into `2.x`; spec work on `2x/phase-3-spec`.
- **P3-6** ruling: A Phase 3 pydra PR targets `enh/execution-extensions` (as #903 and #904 did), cut from it when it
  depends on a PR merged there; #891 carries the branch to `main` after a full-tier run (owner item 52).
- **P3-7** ruling: The gate's spike is throwaway and carries its own extra, pin, lock and CI job; Task 9b rebuilds
  what Route A keeps under review, so no unreviewed code reaches `2.x`.
- **P3-8** ruling: The register statuses checked in the pre-flight (R-169..R-176, R-181, R-182, R-184, R-185, R-187,
  R-188, with R-172 and R-188 partial) are set in the plan's revision commit; R-162, R-164 and R-165 move to Tasks 2,
  18 and 20.
- **P3-9** ruling: One YAML loader, `runtime/config.py:load_yaml`, replaces every loader in `packages/` and
  `benchmarks/`; the check is an import-linter forbidden contract, not a pytest test; `scripts/` are outside it because
  they are lint roots that may read YAML before the packages are importable.
- **P3-10** ruling: `Metric` carries optional numerator and denominator (`metric@2.0`); aggregates are ratios of sums;
  EER aggregates over pooled trials.
- **P3-11** ruling: A backend's minimum input is a declared duration, `BackendSpec.min_input_ns`, checked by the one
  input check; speechbrain's `min_samples()` folds into it. Fault injection is a reserved request field.
- **P3-12** ruling: `graph@1.0` holds task nodes and plain, `map` and `gather` edges only; route, loop and nested nodes
  arrive in Phase 4 with a format bump (ADR-006: no previous-major reader before the first release).
- **P3-13** ruling: The topological order is rewritten from `reproduce/dag.py`'s rule (lexically smallest ready node
  first) rather than reused as design §4.9 suggests for the fallback, because that module imports the 1.x reproduce
  spec and metrics and its importers have not moved. The ledger says so.
- **P3-14** ruling: The `Dataset` and `Record` protocol and `InputManifest` are kernel types without constructors;
  constructors and manifest building are runtime functions, and `sl.Dataset` is a facade namespace over them (ADR-009).
- **P3-15** ruling: The minimal BIDS adapter parses entities from names and reads sidecars without keying them; mapping
  sidecar fields to entities is a dataset-profile job (Phase 4). Any `key-value` entity is accepted.
- **P3-16** ruling: A BIDS `Record.id` is its root-relative POSIX path, so moving the root changes neither ids nor the
  manifest digest.
- **P3-17** ruling: A run's id is its plan hash, and provenance `run_id` is the plan hash; each `run()` or `resume` is
  an attempt whose id is `revision.run_id()` (from `SENSELAB_RUN_ID`, Slurm or a uuid), with its own placement file.
- **P3-18** ruling: A single task call or `.map` is planned as a one-node graph with a `map` input edge, so every run
  has a `LogicalPlan`; a single call submits its one fragment straight to `InProcess`.
- **P3-19** ruling: A cost figure with no measured resource table is `None` and named under `unmeasured`.
- **P3-20** controller ruling (supersedes the planner's): A cached hit without a recorded env license is refused by
  default (owner item 11, P-23); a user policy in `RunConfig.licenses` may allow it (owner item 46, P2-93), and the
  allowance is recorded.
- **P3-21** ruling: Memory caps are checked from resource tables; a node without a table is admitted and listed as
  unmeasured, because an OOM is a resource fault that re-places the job (ADR-003).
- **P3-22** ruling: One `execution-profile@2.0` bump in Task 8 carries every Phase 3 key, each with a derivation; retry
  budgets are execution policy, because they change which results exist, not their identity.
- **P3-23** ruling: Resource tables are generated files joined to catalogue entries by `(id, revision)`
  (`catalogue@2.0`).
- **P3-24** ruling: `runtime/in_process.py` (`InProcessWorker`) is renamed `runtime/host_worker.py` (`HostWorker`), so
  `InProcess` names only the executor.
- **P3-25** ruling: `run_call` is removed (P2-16); its body is `execution.job.execute`; no test asserts its absence.
- **P3-26** ruling: A failure is never stored; a `FaultRecord` is run-scoped (`dead/`, events) and records whether it
  was charged; a gap fault is never charged (ADR-009).
- **P3-27** ruling: `ResultSet.to_frame()` imports pandas lazily; core gains no pandas dependency; the root test group
  gains pandas so `senselab-lean` exercises it.
- **P3-28** ruling: ADR-008's OpenTelemetry export is deferred until a user asks; events, `status` and `explain` are
  built.
- **P3-29** ruling: `resume` re-checks the manifest's stamps and refuses a changed input; a changed dataset is a new
  plan.
- **P3-30** controller ruling (corrected 2026-11-03, owner item 58): A lease is a directory taken by `mkdir` with the
  holder's token; renewal is the heartbeat, beat directories created and removed by a renewer owned by the lease
  (`Lease.keep_alive`), independent of the work; nothing is written or rewritten under the lease (D-2, H-R40); no
  pid-liveness takeover (ADR-003 amendment 2026-11-03).
- **P3-31** ruling: `work_pool.py` is not moved, because 1.x `reproduce` modules and tests still import it (R-128,
  ADR-009: a 1.x module goes when its last use case moves); `execution/queue.py` and `execution/substrates.py` are
  rewritten from the ADR-003 design with it as the named source.
- **P3-32** ruling: Queue names are resource classes, with `#<model identity>` for a resident model; segment roll size
  is profile data.
- **P3-33** ruling: Fault effects follow `execution.md` §3.3, with budgets from the profile's `faults` keys, owned once
  by the `Pool`; a fault with `gap: true` is treated as infrastructure and reported as a classification gap.
- **P3-34** ruling: The `Pool` reads outcomes only from the store and `dead/`, so a restarted coordinator sees the same
  state.
- **P3-35** ruling: The pool worker is a host `senselab_core` process that drives the queue's env through the one
  worker protocol; `execution.md`'s "runs inside the env the queue needs" means it runs that env's worker (amended in
  Task 22).
- **P3-36** ruling: Only a request that fails as a whole is bisected, by the pool worker within its claim; the `Pool`
  alone re-places a single item's resource fault and counts `resource_replacements`.
- **P3-37** ruling: Worker supervision uses `PR_SET_CHILD_SUBREAPER` on Linux; macOS supervises by process group with
  the limit stated.
- **P3-38** ruling: Slurm scaling targets min(queue depth ÷ quantum, class `max_workers`, QOS CPU cap ÷ cpus per task).
- **P3-39** ruling: A pool runs only where its store is reachable from every substrate; laptop-to-cluster submission is
  deferred (`execution.md` §3.6 admits it).
- **P3-40** ruling: Services are registry entries `{identity, endpoint}` under renewed leases, with consumer leases
  deciding idle exit; this completes ADR-003's staged service amendment and drops the `heartbeat` field.
- **P3-41** ruling: Bucket edges are packaged data (`batching@1.0`) with a derivation; batch formation is keyed in the
  task that introduces it (18); before then a non-invariant backend runs at batch size 1 (I-3).
- **P3-42** ruling: Batch invariance is certified by P2-95 comparisons on the held-out split; certification informs a
  later declaration change with a version bump and never relaxes a backend at run time.
- **P3-43** ruling: Under Route A the epoch check, release and fault effects stay in senselab's `Pool`, reached through
  the `Executor` calls of `ExecutorWorker`; pydra supplies graph expansion, readiness and descendant failure.
- **P3-44** ruling: Scale budgets and resource tables are generated from idle-node runs and never fitted at run time; a
  non-flat scale point is a finding. The ladder takes a suite file, not per-knob flags.
- **P3-45** ruling (owner attention): The laptop half of the P-10 exit runs a model-less task (`resample`) over a
  synthetic, byte-identical 1,000-record BIDS tree, because the local machine runs no models and downloads no data
  (owner items 27, 28); it shows only a profile difference, not a hardware-driven one. The model-backed `.map` runs on
  ORCD, where the laptop profile's plan is computed on a cluster node to show the equal hash.
- **P3-46** ruling: Where design §4.8 (I-3, I-4), §6.2 and `testing.md` §4 say "measured tolerance", Phase 3 uses P2-95
  type comparisons under run-config thresholds; Task 22 amends the text.
- **P3-47** ruling: The docs example for Phase 3 is the LibriSpeech BIDS `.map`.
- **P3-48** ruling (owner attention, revised 2026-11-03): pydra's C uses ADR-003 leases as amended 2026-11-03 (owner
  item 58): a directory taken by `mkdir`, renewed by beat directories (the heartbeat), takeover once its age on the
  storage clock passes `stale_after`, no pid-liveness takeover, in place of proposal 0001 §3.3's "owner, host, pid,
  heartbeat, TTL takeover" (owner item 50). Task 45 amends the proposal and `pydra-extension-points.md` §5.
- **P3-53** ruling (owner, 2026-11-03: "the heartbeat setting should be in pydra"): the job-lease heartbeat lives in
  pydra (Task 45, `LeaseSettings`), configured by senselab's execution profile through `PydraRunner` (Task 9b);
  senselab's own leases (queue items, coordinator epoch, service registry; Tasks 12–17) keep the same ADR-003
  directory protocol in senselab core.
- **P3-54** ruling (superseded 2026-11-03 by owner item 60): the static-or-error rule for closures and globals is
  withdrawn. pydra's function checksum follows the adopted design (`.superpowers` review of 2026-11-03, owner item 60):
  the key carries the code part at a `code_hash` tier (`function`, `module`, `user`; default `user`, user code
  inferred from install paths with an override) with closure and global values hashed by the ordinary value rules
  and library objects by qualified name; interpreter, platform, environment and the versions of packages a task
  reaches are recorded and compared by `reuse` (`exact`, `consistent` default, `any`); `version=` replaces the code
  part only. Delivered as #906 (closures only) and four follow-up pydra PRs: a version-neutral AST encoding with
  docstring stripping; global values and the tiers; the dependency record and the `consistent` default; `version=`
  with the source guard.
- **P3-59** ruling: Task 4b's CPU timing runs non-exclusive on the private CPU node (an exclusive node exceeds the
  group's CPU cap, and the general partition's exclusive start was 11 days out); the decision reads the on/off share,
  which alternation within one job keeps robust to contention; absolute times are upper bounds; node load is recorded.
- **P3-49** ruling: The spike and Task 9b assert that no `_result.pklz` is written; `_job.pklz` may remain until B2
  and D2 land, and its absence is not claimed in Phase 3.
- **P3-50** ruling: A node's `backend_params` are keyed by backend name; only the placed backend's entry reaches its
  key, it is validated at plan time against that backend's schema, and a rejected entry raises
  `BackendParamsMismatch`. Because the backend is keyed, a placement that picks another backend keeps the plan hash
  but recomputes that node.
- **P3-51** ruling: Task 9 keeps R-187's producer when `run_call` goes: the key's batch formation is derived from the
  request actually sent, and a non-invariant backend is sent single items until Task 18.
- **P3-52** ruling: Route A uses senselab's own queues through `ExecutorWorker`; pydra's I (Task 51) is a generic
  plugin informed by Task 13's design and is not a senselab dependency in Phase 3, so the queue protocol has one
  senselab implementation.
- **P3-55** ruling: A split whose summed denominator is 0 has no defined value: the aggregate is absent (None), unless
  the metric declares an empty-case value (`Metric.empty_value`). Span F1 declares 1.0, since nothing expected and nothing
  found is perfect agreement; error rates such as WER and DER declare none, since 1.0 would mean total error.
- **P3-56** ruling: A transcript's extent is `extent: Span | None`, the stretch of the recording it covers. A timed span
  lies inside it, on its clock, both bounds checked, so `rebind` moves the extent with the spans. A transcript with no
  span may declare none (a reference transcript has no audio).
- **P3-57** ruling: A backend's minimum input is the shortest audio its front end or receptive field takes, in ns from
  the model's input rate: crisperwhisper and the Whisper `transformers` backend 400 samples at 16 kHz, the Whisper STFT
  window, n_fft 400 (25 ms); pyannote 991 samples, the receptive field of
  the segmentation SincNet and convolutions (61.9375 ms, pinned to the loaded model by an env test); speechbrain 640
  samples, the 5 frames the ECAPA encoder's widest dilated convolution needs, at hop 160 (40 ms).
- **P3-58** ruling: A long-form segment's confidence is scored over the ids of the words the window merge keeps, counted
  from the library's per-window chunk text. When the chunks do not match the captured windows, or the kept-word and
  word counts disagree, the mean and the word confidences are both None: confidence is an optional measurement. Word
  confidences are grouped per window, so a window's first piece without a leading space does not join the previous
  window's last word. The request key `_diagnostics` is reserved (as `_fault` is): it adds a `diagnostics` output,
  the number of windows decoded and of chunks the library reported, so a real-weights test can assert that they match.

**Low pre-flight findings not changed, with reasons.** None remains open: each low finding is fixed in its task or
covered by a ruling above (X17 by P3-2, X18 by P3-40, X19 in Task 2, X20 in Task 0, X21 in Task 0 Step 2, X22 by P3-49,
X23 in Task 21, X24 in the job shape and Task 16, X25 by P3-5; per-task lows in Tasks 2, 3, 4a, 6, 9, 13, 15a, 16, 20
and the import contract of Task 9b).

## Design gaps left open

None. Spec conflicts found while planning are resolved by P3-1 (K2 missing), P3-2 (runner selection), P3-8 (stale
register owners), P3-20 (P-23 against as-built reuse), P3-31 (the 1.x importers of `work_pool.py`), P3-35
(`execution.md` §3.2), P3-45 (laptop exit against local limits) and P3-46 (tolerance wording).
