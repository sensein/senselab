# senselab 2.x Phase 4 Implementation Plan: pipelines, datasets, signals

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose declared tasks into nestable pipelines with routing, loops and typed absence, run them over any
dataset through dataset profiles, and port the audio use cases onto them, so that:
- `sl.Pipeline` builds a `graph@2.0` document (ports, nesting, `map`/`gather`, `over=`, `route`, `loop`, `Absent`)
  that compiles to a pydra workflow and runs on the warm pool exactly as a single task does;
- one loader reads run configs, dataset profiles and execution profiles as `name:version` layers;
- datasets are full BIDS trees or in-memory mappings, with every dataset fact in a dataset profile;
- the generic `sl.signal` tier runs unchanged on audio and on accelerometry;
- explore_conversation, health_measurements, audio_analysis, triage's measurement tasks and one cross-modal pipeline
  run as 2.x pipelines, and backends share environments by the computed grouping (P-24);
- duplicate tasks, and variants written as separate functions that differ only by model, backend or parameters, are
  merged into one task with typed parameters, each merge proven by parity at every former variant's configuration
  (P-37's consolidation inventory, emptied by Phase 5's exit).

**Architecture:**
- **pydra does the graph work.** senselab already drives pydra's `GraphRunner` and runs jobs on its `WarmPool`
  (execution.md amendments route-B S1b..S4). Phase 4 compiles the new graph constructs onto what pydra
  `integration/senselab-r2` provides: nested workflows; `when=`/`unless=` gates and `select(n)` for routes;
  `workflow.loop` (a runtime while-loop) for loops; `Absent` with `propagate`/`accept`/`drop`/`fill` modes; split over
  upstream outputs with partial state for nested fan-out; `plan()` for what would run or be reused. Where senselab
  needs something pydra lacks, the plan names a pydra follow-up (§ pydra follow-ups), not a senselab workaround.
- **The kernel gains data, not behaviour:** `Absent` with its cause class, `graph@2.0`, the `Pipeline` builder (it only
  builds a `Graph`), `DatasetProfile`, `Accelerometry`. It still does no I/O (ADR-009).
- **The runtime gains** the compiler for the new constructs (`runtime/execution/runner.py`), the merged loader
  (`runtime/layers.py`), the dataset adapters and the series reader.
- **Ported tasks** live in the umbrella package under their namespace (`senselab.signal`, `senselab.audio`,
  `senselab.multimodal`); pipelines are builder modules under `senselab.pipelines`.

**Tech stack:** as Phase 3; pydra from `integration/senselab-r2` at one pinned commit (Task 0); scipy for `sl.signal`;
new isolated envs for NeMo, Qwen3-ASR, TensorFlow (YAMNet), parselmouth, openSMILE and PyAV.

**Spec:** `design.md` §4.1, §4.5, §4.6, §4.11, §4.15, §6.2 (Phase 4 and its exit), §6.4; ADR-001, 002, 005, 006, 009
(as amended); `execution.md` (amendments route-B S1b..S4); `route-b.md`; `roadmap.md`; `migration-ledger.md` (moving
a use case; triage classification); `use-cases.md` §1, §4, §5.2; `user-scenarios.md` UX-11, UX-26; `principles.md`;
`engineering.md` (P-27..P-40); `rulings.md`; `review-register.md` rows scheduled below; `plans/phase-3.md` rulings
P3-1..P3-52, which bind here unless a P4 ruling supersedes one.

**Placeholders.** `$SENSELAB_2X_IMPL` the code worktree, `$SENSELAB_2X_SPEC` the spec worktree, `$PYDRA_EXT` the pydra
worktree, `$SCRATCH` the cluster scratch root, `$CPU_PARTITION`, `$GPU_PARTITION` (typed H100), `$SHARED_HF_HOME`,
`$B2AI_BIDS_ROOT` the controlled B2AI BIDS tree on the cluster, `$B2AI_PROFILE` the B2AI dataset profile file. No
identifier, user name or personal path is written into code, data or specs (P-28).

**Branches** (P-27). Code: `2x/phase-4`, cut from `2.x` after Task 0 merges, → `2.x`. Spec: `2x/phase-4-spec` → `2.x`
(this plan, ADR amendments, register rows, ledger, `STATE.md`). pydra follow-ups: per-PR branches in `$PYDRA_EXT` onto
`integration/senselab-r2`.

## Global Constraints

- **Interpreters are uv-managed only:** `--python-preference only-managed` and a named version on every
  interpreter-selecting `uv venv`, in senselab and in pydra work.
- **Default Python is 3.13** (owner items 54, 55); 3.11-3.14 nightly and on a `full-matrix` label.
- **No backwards compatibility** (P-14): `graph@1.0` and every retired profile format are refused with a typed error
  naming the migration; no shims, aliases or readers of old formats.
- **One implementation per behaviour** (P-37, P-32): one graph compiler, one profile loader, one Absent type, one BIDS
  parser, and one task per behaviour as `consolidation.md` (Task 1b) groups them. A port implements a group's single
  replacement, never a member; a variant is a typed parameter, a backend or a profile, never a second task or a flag
  per former caller. A former member's hard-coded constants become documented settings: `Params` defaults or `data/`
  values with derivations (P-17).
- **Rationale lives in specs** (P-21, P-31): docstrings say what and how; no register, round or ruling id outside
  `specs/`.
- **Thresholds and budgets are data with a derivation** (P-17): grids, windows, peak profiles, QC bounds, fitted
  thresholds. A value fitted on one dataset lives in that dataset's profile, never in a packaged default.
- **Dataset knowledge only in dataset profiles and adapters** (P-12): no dataset name, task id, sidecar field or path
  pattern in pipeline or task code; Task 8's check enforces it.
- **Agreement is a standard comparison** (P2-95): hash equality, exact equality or `isclose` under run-config
  thresholds. Parity against 1.x uses the 1.x `alpha` head merged in Task 1 (P-39).
- **Licenses are transparent and the user's policy decides** (P-23); every new backend and env carries its license
  record in the PR that adds it (GPL parselmouth, audEERING openSMILE and NVIDIA model licenses included).
- **Weights pinned by commit** (P-4, P-16); every new catalogue entry has a 40-hex revision.
- **Order of work in every task** (P2-34): implement and run the local tests → review and compact by the named
  reviewer (a different model and context, P-40) → commit and push; CI green → ORCD at that SHA. Evidence is cited
  `<sha> <job id> <summary line>`.
- **GitHub CI:** `detect`, `pre-commit`, `core-only`, `core-pydra`, `senselab-lean`, `core-limits`, `source-digests`,
  `pr-models`, `core-env`, `licenses`, `worker-floor`. Pipeline tests walk a plan, so they run in `core-pydra`.
- **Local machine** (P-29): logic, contract and bounded model-free limits only, `uv run --no-sync`, scoped to the
  changed files, serial, never `pytest -n auto`, never `uv sync`, no new venvs, no model or dataset downloads.
- **ORCD** (P2-33): every model, backend env, GPU, Slurm, multi-node and dataset run; an agent submits from a pinned
  SHA and the main thread holds the wait. H100 by default; A100 only as overflow.
- **Tiers** (`testing.md`): exactly one of `logic`, `contract`, `limits`, `env`, `usecase`. A pipeline run on public
  data is `usecase` (it populates the tier R-265 found empty).
- **Moves** (ADR-009, ledger "Moving a use case"): merge `origin/alpha` first; rename-only `git mv` commit, then a
  modify commit. A 1.x module whose 1.x importers have not moved is not moved; its replacement is rewritten and the
  ledger names the source.
- **Commits** (P-27): one per unit of work; each task's last code commit has green CI and records the implementer and
  reviewer models.

### ORCD job shape (every `orcd` step)

- **Before every `sbatch`:** the `orcd-remote` skill's submit plan (`--plan`): earliest realistic start, typed GPU,
  healthy nodes, allocation guard. No node-name exclusions.
- **Checkout:** a fresh clone from GitHub per campaign at the pushed SHA (`$SCRATCH/senselab-2x/<sha>/`), never reset
  while jobs run from it; `git log origin/2x/phase-4..HEAD` empty before `sbatch`.
- **CPU:** `#SBATCH -p $CPU_PARTITION -c 8 --mem=32G -t 02:00:00`.
- **GPU:** `#SBATCH -p $GPU_PARTITION --gres=gpu:h100:1 -c 8 --mem=64G -t 04:00:00`.
- **Environment:** `PATH="$HOME/.local/bin:$PATH"`; `LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"`;
  `HF_HOME=$SHARED_HF_HOME`; `HF_TOKEN` from the user's token file; `SENSELAB_RUN_ID=slurm-$SLURM_JOB_ID`;
  `SENSELAB_CACHE=$SCRATCH/senselab-2x-cache`; `UV_CACHE_DIR=$SCRATCH/uv-cache`; execution profile `orcd` (a site
  file, P-42).
- **Steps:** `benchmarks/exit/orcd/step-prepare.sh` fills the shared uv cache; `step-tier.sh` runs a tier, serial, each
  test bounded (120 s CPU, 300 s GPU).
- **Evidence:** `COMPLETED` is not evidence: check elapsed time, grep the summary line, record `<sha> <job id>
  <summary>`. A step that measures a property names the number it must show before it runs. Timing counts only from
  an idle node.

## Review Focus

1. **A resumed run takes the branch and the rounds it took.** A route's predicate and a loop's `until` are stored
   results; a resume with one plan hash re-executes neither, and a changed predicate implementation is a new plan
   hash, not a silent re-route. Pinned in Task 6 (`test_resume_takes_recorded_branch`,
   `test_resume_replays_loop_rounds_from_store`).
2. **Environment-caused absence is never served from the store.** An `Absent` of class `environment` (awaiting review,
   a missing optional resource) is recomputed on resume; a content one (`unfitted`, `not-routed`, `no-speech`) is
   served. Pinned in Task 2 (`test_environment_absent_never_served`) and Task 6
   (`test_environment_absent_recomputed_on_resume`).
3. **A dataset profile change re-plans only what reads it.** Changing a fitted threshold changes the keys of the nodes
   that declare that slice and no other; changing a sidecar mapping changes only nodes that read the mapped field.
   Pinned in Task 8 (`test_threshold_change_rekeys_only_its_readers`).
4. **A real BIDS tree is read as stated.** Every file of a requested extension is a record or counted as ignored with a
   reason; directory and filename entities agree; derivatives never become raw records. Pinned in Task 9
   (`test_every_requested_file_is_record_or_ignored`, `test_derivatives_never_raw_records`).
5. **A nested pipeline keys as it does standalone.** A pipeline used as a node gives each inner node the key it has
   when the pipeline runs alone on the same inputs, so its results are reused across both uses. Pinned in Task 5
   (`test_nested_node_keys_equal_standalone`) and Task 31 Step 3.
6. **A merge silently changes one variant's defaults.** When former tasks become one task with typed parameters, a
   member called with its old arguments must give its old outputs; a default that differs from a member's former
   default is listed in that group's row and passed explicitly in the member's parity configuration. Pinned in every
   merge task by `test_member_<name>_parity` (one per former member, at its recorded configuration) and
   `test_defaults_declared_per_member` (Task 1b's `members.yaml` against the merged task's `Params` defaults).

## File Structure

```
packages/senselab-core/src/senselab_core/
  kernel/absent.py                 # Absent, AbsentClass, AbsentMode (absent@1.0)               Task 2
  kernel/graph.py                  # graph@2.0: node kinds task|pipeline|route|loop, over=      Task 3
  kernel/pipeline.py               # Pipeline builder, Composable protocol, listing()           Task 4
  kernel/declare.py                # TaskSpec.absent (per input port), profile slice kinds      Tasks 2, 8
  kernel/dataset.py                # Dataset.groups(), Dataset.layers(), IgnoredReason           Task 9
  kernel/dataset_profile.py        # DatasetProfile (dataset-profile@1.0), TaskProtocol          Task 8
  kernel/signals.py                # Accelerometry(Recording)                                    Task 10
  kernel/media.py                  # Video (a stamped container; frames not decoded)             Task 29
  runtime/layers.py                # load_profile(): the one loader for run, dataset, execution  Task 7
  runtime/profiles.py              # ExecutionProfile reads through layers.py                    Task 7
  runtime/call.py                  # RunConfigFile reads through layers.py                       Task 7
  runtime/planner.py               # plans pipelines; unfitted slices become Absent at plan time Tasks 5, 6, 8
  runtime/bids.py, datasets.py     # full BIDS, frames, HF datasets, mappings                    Task 9
  runtime/series_io.py             # open_series(): CSV/Parquet series under a declared layout   Task 10
  runtime/execution/runner.py      # compile_plan(): nesting, partial splits, over=, route, loop Tasks 5, 6
  runtime/execution/absence.py     # kernel Absent <-> pydra Absent; reuse rule for absence      Tasks 2, 6
  runtime/execution/warm.py        # batched WorkerCall, workers_per_device                      Task 0b
  runtime/envs.py, data/           # adopted env groups                                          Task 30
packages/senselab/src/senselab/
  __init__.py                      # sl.Pipeline, sl.validate(pipeline|graph|json), sl.Dataset.*
  signal/{filter,rescale,channels,windows,spectrum,envelope}.py                                Task 11
  signal/{activity,spans,qc}.py                                                                 Task 12a
  signal/{synchronize,impute,regularize,peaks}.py, signal/data/peak_profiles/                   Task 12b
  audio/tasks/voice_features/      # praat (parselmouth), opensmile; 1.x phonation merged        Task 13
  audio/tasks/embeddings/          # speaker and SSL embeddings as one task                      Task 14
  audio/tasks/classify/            # ast (transformers), yamnet (tf, isolated); SER by model     Task 16
  audio/tasks/voice_activity/      # pyannote and energy strategies                              Task 19b
  audio/tasks/enhancement/         # sepformer (speechbrain)                                     Task 17
  audio/tasks/speaker_diarization/sortformer*, speech_to_text/canary_qwen*   (nemo env)          Task 18
  audio/tasks/speech_to_text/qwen_asr*, audio/tasks/forced_alignment/  (qwen, mms)               Task 19
  audio/tasks/uncertainty/         # audio_analysis L1 votes and L2 fold                         Tasks 21, 22
  audio/tasks/{word_consensus,background_floor,speaker_enrollment}/   # merges C-8..C-10         Tasks 20a-20c
  audio/tasks/triage/              # triage measurement tasks                                    Tasks 25-27
  multimodal/demux.py              # video -> audio (pyav, isolated)                             Task 29
  pipelines/{explore_conversation,health_measurements,audio_analysis,adaptive,
             voice_measurements,speech_pii}.py                                                   Tasks 15, 22, 23, 28, 29
  data/catalogue.yaml, env_licenses.yaml, env_locks/, env_grouping.yaml
packages/*/tests/...               # mirrors the above
tests/fixtures/profiles/{adult-speech-fixture,synthetic-cohort}/   # neutral dataset profiles   Task 24
scripts/check_dataset_coupling.py  # P-12 static check                                           Task 8
specs/.../consolidation.md         # the task-level consolidation inventory (spec branch)        Task 1b
packages/senselab/tests/consolidation/members.yaml   # each former member's configuration      Task 1b
packages/senselab-core/tests/lint/consolidation_inventory_test.py                               Task 1b
```

**Execution order** (dependencies are authoritative): 0 → 0b · 1 → 1b (alongside 0; 1b before any port or merge) ·
2 → 3 → 4 → 5 → 6 · 7 → 8 → 9 · 10 → 11 → 12a → 12b · 13 (after 1b), 14 (after 1b, 11) → 15 (needs 5, 9) · 16, 17,
18, 19 (after 0b, 1b) → 19b (needs 12a) → 20 → 20a, 20b, 20c → 21 → 22 (needs 5) → 23 (needs 6) · 24 (needs 8, and
Task 1's merge of the triage stack) → 25 (needs 12b, 20b) → 26 (needs 20a) → 27 (needs 13, 16, 20c) → 28 · 29 (needs 5, 6) · 30 (after
13-19b and 29) · 31 → 32 → 33 → 34.

---

### Task 0: Land the pydra repin (#662) and the batching measurement

**Model:** opus implements (the PR's author continues); fresh opus reviews. **Tier:** contract, ORCD. **Implements:**
route-b.md §5 (the pin moves with every integration merge); R-233, R-235, R-264 re-measured on the warm pool.
**Depends on:** pydra `integration/senselab-r2` rebuilt on the E16 stack top.

PR #662 (draft, `2x/pydra-repin-2cdd817b`) moves the pin to `integration/senselab-r2` and declares pydra's protocol
members. It stays draft until the integration branch is rebuilt.

- [ ] **Step 1:** repin to the rebuilt head `R` of `integration/senselab-r2` (one commit: `[tool.uv.sources]` rev,
  `uv lock`, `env_licenses.yaml`); delete what the merge makes unnecessary, each removal with the test that fails when
  its reason is gone (route-b.md §5). `digest_contract_test` passes unchanged, or the PR bumps the affected tags.
- [ ] **Step 2:** CI green at the head, `core-pydra` included.
- [ ] **Step 3 (ORCD, CPU and H100):** full packages suite, every tier, serial; env builds for every shipped lock.
  Must show: every populated tier passes, 0 env build failures.
- [ ] **Step 4 (ORCD, H100):** `benchmarks/exit/orcd/step-batch.sh` (from #662): per-item time of the speechbrain
  `speaker_embeddings` worker at explicit request sizes 1, 4, 16, 64 and under pydra's `probe_batch`; every output
  equal to the record's size-1 output under the run-config thresholds. Record the table in `execution.md` amendment
  "route-B S4".
- [ ] **Step 5 (ORCD, CPU, same campaign):** `fake.echo` and `signal.resample` at 10² and 10⁴ items on the warm pool,
  recording queue-to-claim wait (R-233), each pool worker's peak RSS (R-235) and per-item time for items of 1 and 8 jobs
  (R-264). Must show, for each row to close: median wait under one `poll_s` at 10² (R-233); peak RSS at 10⁴ within 1.2×
  of 10² (R-235); per-job slope below the per-item spread (R-264). A row that fails becomes a pydra follow-up (F7, F8,
  F9 below) with the numbers.
- [ ] **Step 6:** mark #662 ready; merge `--no-ff` into `2.x`; spec commit on `2x/phase-4-spec`: register rows R-233,
  R-235, R-264 closed or re-pointed with `<sha> <job id>`; `STATE.md` log line.

### Task 0b: Batched calls and `workers_per_device` (R-327, R-328)

**Model:** opus implements; fresh opus reviews (execution semantics). **Tier:** logic, contract, ORCD H100.
**Implements:** R-327, R-328; R-257 and R-258 checked against pydra's bisection. **Depends on:** Task 0.

**Files:** modify `runtime/execution/warm.py`, `runtime/execution/providers.py`, `runtime/profiles.py`,
`runtime/data/profiles/{laptop,slurm}.yaml`, `config-derivations.md`; tests in
`packages/senselab-core/tests/runtime/execution/{warm_batch,warm_pool}_test.py`.

**Interfaces:**
- `WorkerCall` is defined with pydra's `batched=` and no constant `batch`, so `MeasuredFit` probes N per backend and
  class. A backend whose formation is `keyed` (bucket-keyed, I-3) is submitted through a second class without
  `batched`, one job per request; the class is chosen by `keyed_formation(backend)` at submit.
- Execution profile `execution-profile@7.0`: `classes.<name>.workers_per_device: int = 1` (≥ 1), passed to pydra's
  `WorkerClass.workers_per_device`. Every packaged and site profile is re-tagged; `@6.x` is refused naming the key.

- [ ] **Step 1: Failing tests (local, fake stack):** `test_invariant_backend_submits_batched_class`,
  `test_keyed_backend_one_job_per_request`, `test_batched_outputs_equal_single_outputs` (fake backend, 64 items),
  `test_resource_fault_at_load_not_bisected` (R-257: a `ResourceFault` raised in `load` costs one worker start),
  `test_mixed_content_and_oom_request_resends_resource_items` (R-258: the content item fails alone, the resource items
  are re-placed at a smaller batch), `test_workers_per_device_reaches_worker_class`,
  `test_profile_6x_refused_naming_workers_per_device`.
- [ ] **Step 2:** if pydra's bisection fails R-257 or R-258, the test is marked `xfail(strict=True)` with the pydra
  follow-up number (F5, F6) and the rule is written in `execution.md` (amendment "Phase 4 batching"); senselab adds no
  bisection of its own.
- [ ] **Step 3: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/warm_batch_test.py -m "logic or contract"`.
- [ ] **Step 4 (ORCD H100):** Task 0 Step 4 rerun through the batched class; must show per-item time at the probed N
  below the size-1 time, and every output equal to size 1. `workers_per_device` stays 1 in packaged profiles; its
  derivation names this measurement.
- [ ] **Step 5:** review, compact, commit `feat(core): batched worker calls and workers per device`; CI green; merge;
  register R-327, R-328 closed, R-257, R-258 closed or re-pointed to the pydra follow-up.

### Task 1: P-39 reconciliation and the open 1.x work

**Model:** sonnet implements; haiku verifies each ledger line against `git log`. **Tier:** none (merge, spec).
**Implements:** P-39; the owner request of 2026-11-28 (open 1.x work, below). **Depends on:** nothing; repeated before
Tasks 15, 23 and 24, and in Task 31.

- [ ] **Step 1:** `git fetch origin alpha`; merge `origin/alpha` into `2x/phase-4` (`src/` takes `alpha`'s side; the
  2.x `uv.lock` and workflows keep 2.x's side). Record the merged head as each ledger row's mark.
- [ ] **Step 2:** for every commit since `c4bb902b` under `src`, `scripts`, `tutorials`, add a ledger line
  `<commit> → <2.x handling>` naming the Phase 4 task that carries it, or "none, not ported" with the reason.
- [ ] **Step 3:** update the ledger's "Open 1.x work" section to the current PR states (merged, open, closed) and each
  PR's handling as in the section below; a newly merged PR in that list moves the parity reference, so the task it
  names reruns its parity step.
- [ ] **Step 4:** commit `chore(2x): P-39 reconciliation at Phase 4 start` (code) and the ledger commit (spec); push.

### Task 1b: The task consolidation inventory (P-37)

**Model:** opus builds the inventory (judgement); two haiku verifiers, each on a disjoint half, check every member's
file:line and that each "distinct" verdict names what differs. **Tier:** none (spec), then logic for the lint test.
**Implements:** P-37 (the consolidation inventory, emptied by Phase 5's exit), P-31, P-32, P-17 (fixed parameters as
documented settings), design §6.1; owner request 2026-11-28 (consolidation). **Depends on:** Task 1. Every port and
merge task below reads it.

**Files:** create `specs/20261030-140022-senselab-2x-api/consolidation.md` (spec branch; design §6.1's table keeps the
infrastructure rows and links here for tasks), `packages/senselab/tests/consolidation/members.yaml`,
`packages/senselab-core/tests/lint/consolidation_inventory_test.py`.

- [ ] **Step 1 (spec, read-only scan):** inventory every declared 2.x task (`@task` in `packages/`) and every public
  task function of the 1.x areas Phase 4 ports (`src/senselab/audio/tasks/*`, `audio/workflows/{audio_analysis,
  explore_conversation.py,health_measurements,triage}`, and the scripts those call). Two members are one group when
  they compute the same output type from the same input type and differ only by model, backend, parameters, window,
  units or a constant. Start from the candidate groups in § Consolidation groups; a candidate the scan finds distinct
  is kept with the reason (a null result is valid).
- [ ] **Step 2 (spec):** one row per group in `consolidation.md`: `id` (C-n), members (file:line each), the single
  replacement (task name and version), its `Params` and backends, each member's former defaults and constants and where
  each now lives (a `Params` default, a profile value, or `data/` with a derivation), the identity consequence (task
  name or version changes, so every key of every member changes; old entries are never served, no alias, P-14), the
  parity tests (one `test_member_<name>_parity` per member at its configuration), the owning task, and status.
- [ ] **Step 3:** `members.yaml`: per group, per member, the configuration that reproduces it (`task`, `params`,
  `backend`, `model`) and the 1.x call it replaces. `test_defaults_declared_per_member` (shared helper
  `testing/consolidation.py`) loads it in every merge task.
- [ ] **Step 4: Failing test, then implement:** `consolidation_inventory_test.py`:
  `test_every_open_group_names_an_owner_task`, `test_closed_group_members_absent_from_registry` (no task declared under
  a replaced member's name), `test_open_group_count_never_increases` (against the count recorded at this task's
  commit). Run `uv run --no-sync pytest packages/senselab-core/tests/lint/consolidation_inventory_test.py -m logic`.
- [ ] **Step 5:** review (opus, fresh context), compact, commit the spec on `2x/phase-4-spec` and the test on
  `2x/phase-4`; push; CI green. design §6.1 amendment: the inventory's task rows live in `consolidation.md`.

### Task 2: `Absent` and its cause class (I-6)

**Model:** sonnet implements; opus reviews (identity). **Tier:** logic, contract. **Implements:** design §4.5
(Absence), I-6, H-4; ADR-006 row. **Depends on:** Task 0.

**Files:** create `kernel/absent.py`, `runtime/execution/absence.py`; modify `kernel/declare.py`, `kernel/formats.py`
(`ABSENT_FORMAT`), `runtime/execution/portable.py` (reuse rule), `runtime/execution/outputs.py` (decode),
`senselab_worker/codec.py` (wire form); tests `kernel/absent_test.py`, `runtime/execution/absence_test.py`.

**Interfaces:**
- `AbsentClass = Literal["content", "environment"]`; `AbsentMode = Literal["propagate", "accept", "drop", "fill"]`.
- `Absent(Formatted)`: `format = "absent@1.0"`, `cause_class: AbsentClass`, `cause: NonEmptyStr` (from the vocabulary
  in `runtime/data/absent_causes.yaml`: `unfitted`, `not-routed`, `no-speech`, `out-of-protocol`, `awaiting-review`,
  `resource-unavailable`, each with its class), `source: str | None` (provenance only, excluded from the digest).
  Registered in `DIGEST_TAGS`.
- `TaskSpec.absent: Mapping[str, AbsentMode]`, one entry per input port that is not `propagate`; declared as
  `@task(..., absent={"port": "accept"})`; a `fill` mode names its fill value in the task's `Params`.
- `absence.to_pydra(a: Absent) -> pydra.engine.absent.Absent` with cause `"<class>:<cause>"`; `from_pydra(p) ->
  Absent`; pydra's `not-routed` maps to class `content` (a recorded predicate decides it).
- The reuse rule (`portable.open_store`) never serves an entry any of whose outputs holds an `environment` Absent.

- [ ] **Step 1: Failing tests:** `test_absent_digest_ignores_source`, `test_unknown_cause_refused_naming_vocabulary`,
  `test_cause_class_fixed_by_vocabulary`, `test_absent_round_trips_wire_and_pydra`,
  `test_environment_absent_never_served` (an entry written with one is a miss), `test_content_absent_served`,
  `test_fill_mode_requires_fill_param`.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/kernel/absent_test.py packages/senselab-core/tests/runtime/execution/absence_test.py -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): typed Absent with a cause class`; push; CI green. Spec commit:
  ADR-006 row `absent@1.0`; pydra follow-up F1 if pydra writes an entry it will never serve.

### Task 3: `graph@2.0`: nesting, routes, loops, `over=`, nested fan-out

**Model:** opus implements (kernel format); fresh opus reviews; fable architecture review of the format only.
**Tier:** logic, contract. **Implements:** design §4.5, ADR-001 (amendment), P-9; rulings P4-2, P4-3. **Depends on:**
Task 2.

**Files:** modify `kernel/graph.py`, `kernel/formats.py` (`GRAPH_FORMAT = "graph@2.0"`), `runtime/data/graph@2.0.schema.json`
(regenerated by `scripts/generate_graph_schema.py`; `graph@1.0.schema.json` deleted); tests `kernel/graph_test.py`.

**Interfaces:**
- `Node.kind: Literal["task", "pipeline", "route", "loop"] = "task"`. `task` as in `graph@1.0`. `pipeline`:
  `graph: Graph` inline. `route`: `when: PortRef` (a `bool` output), `then: Graph`, `otherwise: Graph | None`; the
  node's outputs are the outputs `then` declares, and `otherwise`, when given, declares the same names and types.
  `loop`: `body: Graph`, `until: PortName` (a body output typed `bool`), `carry: Mapping[PortName, PortName]` (body
  input ← body output of the previous round), `max_rounds: int | None` (≥ 1; a safety stop only); outputs are the
  body's plus `rounds: int`, `converged: bool`, `stopped: bool`.
- `Node.over: str | None`: a dataset grouping key; the node runs once per group, its record-bound inputs gathered per
  group in record order.
- `Graph.input_absent: Mapping[PortName, AbsentMode]` (default `propagate`).
- A `map` over a mapped node's `list[T]` output is allowed (a partial split); `nested_fan_out` remains only for a
  `map` that would split a node twice over one source.
- New issue codes: `route_outputs_differ`, `until_not_bool`, `carry_unknown_port`, `unknown_grouping_key` (checked at
  plan time against the dataset's entities).
- `graph_digest` covers nested graphs; node ids are unique per level, and a nested node's path id is
  `<outer>/<inner>`.

- [ ] **Step 1: Failing tests:** `test_graph_1_refused_naming_migration`, `test_nested_graph_round_trips_lossless`,
  `test_route_outputs_must_match`, `test_loop_until_must_be_bool_output`, `test_loop_carry_names_ports`,
  `test_map_over_mapped_output_validates`, `test_nested_digest_changes_with_inner_params`,
  `test_schema_validates_every_example` (the examples in `design.md` §4.5 as JSON).
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/kernel/graph_test.py -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): graph@2.0 with nesting, routes, loops and groups`; push; CI
  green. Spec commit: ADR-001 amendment (route as `then`/`otherwise` on one `bool` port; loop as runtime while-loop
  with optional `max_rounds`; `over`; nested fan-out), ADR-006 row, ADR-001 amendment log.

### Task 4: The `Pipeline` builder

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.5 (example), P-9
(listing without running), ruling P4-1. **Depends on:** Task 3.

**Files:** create `kernel/pipeline.py`; modify `packages/senselab/src/senselab/__init__.py`, `_api/describe.py`
(pipelines in `sl.describe()`), `_api/facade.py` (`sl.validate`); tests `kernel/pipeline_test.py`,
`packages/senselab/tests/facade_test.py`.

**Interfaces:**
- `Composable` (kernel protocol): `spec: TaskSpec`, `params: Params`, `backend_params`, `model: str | None`. Every
  `ConfiguredTask` and every `Pipeline` satisfies it.
- `Pipeline(name: str)`; `input(name: str, type: str) -> Port`; `add(node: Composable, *, id: str | None = None, over:
  str | None = None, **inputs: Port | Mapped | Gathered) -> NodeRef` (id defaults to the task's last name segment, then
  `_2`, `_3`); `map(port: Port) -> Mapped`; `gather(port: Port) -> Gathered`; `route(when: Port, then: "Pipeline",
  otherwise: "Pipeline | None" = None, **inputs) -> NodeRef`; `loop(body: "Pipeline", *, until: str, carry:
  Mapping[str, str], max_rounds: int | None = None, **inputs) -> NodeRef`; `output(**ports: Port) -> None`; `graph() ->
  Graph`; `Pipeline.from_graph(name: str, graph: Graph) -> Pipeline`; `listing() -> str` (one block per level: nodes,
  ports, edges; nested levels indented); `spec -> TaskSpec` (kind `transform`, version = `graph_digest` prefix).
- `NodeRef.out.<port>` returns a `Port`; an unknown port raises at the call with close-match suggestions.
- `sl.validate(target: Pipeline | Graph | str | Path) -> list[GraphIssue]`, no planning.

- [ ] **Step 1: Failing tests:** `test_design_example_builds_and_validates`, `test_pipeline_is_composable_node`,
  `test_from_graph_round_trips`, `test_list_into_item_port_without_map_is_type_error`, `test_unknown_port_suggests`,
  `test_listing_shows_each_level_without_running`, `test_describe_lists_packaged_pipelines`.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/kernel/pipeline_test.py packages/senselab/tests/facade_test.py -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): Pipeline builder`; push; CI green.

### Task 5: Compile nesting, partial splits and groups; `plan()` in `explain`

**Model:** opus implements; fresh opus reviews (keys). **Tier:** logic, contract (`core-pydra`). **Implements:** design
§4.5, §4.7; P-9, P-10; R-305 checked. **Depends on:** Task 4.

**Files:** modify `runtime/execution/runner.py` (`compile_plan`, `_wired`), `runtime/planner.py` (nested nodes,
groups), `kernel/plan.py`, `runtime/explain.py`; tests `runtime/execution/{runner_nesting,runner_groups}_test.py`,
`runtime/explain_test.py`.

**Interfaces:**
- A `pipeline` node compiles to a pydra nested workflow whose checksum is supplied (`plan-<hash>/<path>`); each inner
  node keeps its own senselab key, computed exactly as if the inner graph were planned alone on the same inputs.
- A `map` over a mapped output compiles to pydra's split over upstream outputs with partial state; a `gather` combines
  over that split only.
- `over=k` compiles to a split over the plan's group index (`Plan.groups[k]: tuple[tuple[int, ...], ...]`, record
  rows per group value, sorted), each group's record-bound inputs gathered.
- `Plan.explain()` gains, per node, pydra `plan()`'s would-run and would-reuse counts and `deferred` for nodes built at
  run time (loop rounds, routed branches).

- [ ] **Step 1: Failing tests:** `test_nested_node_keys_equal_standalone`, `test_nested_pipeline_reused_across_uses`,
  `test_map_over_segments_runs_per_segment` (diarize → map segments → fake ASR, 3 records), `test_partial_split_fails_only_its_element`,
  `test_over_participant_gathers_each_group_in_record_order`, `test_unknown_grouping_key_refused_at_plan`,
  `test_explain_lists_would_run_and_reuse`, `test_equal_inputs_in_one_item_execute_once` (R-305: pydra settles the
  twin; if it does not, the test documents the count and R-305 stays open with owner).
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/runtime/execution/runner_nesting_test.py packages/senselab-core/tests/runtime/execution/runner_groups_test.py packages/senselab-core/tests/runtime/explain_test.py -m "logic or contract"`
  (needs the `pydra` extra; otherwise CI `core-pydra`).
- [ ] **Step 3:** review, compact, commit `feat(core): compile nested pipelines, partial splits and groups`; push; CI
  green.

### Task 6: Compile routes, loops and absence

**Model:** opus implements; fresh opus reviews; fable reviews the resume semantics only. **Tier:** logic, contract
(`core-pydra`). **Implements:** design §4.5 (routing, loops, absence), I-6, H-4, ADR-001; P-5 for pipelines.
**Depends on:** Task 5.

**Files:** modify `runtime/execution/runner.py`, `runtime/execution/absence.py`, `runtime/execution/records.py`
(round and branch ids in provenance); tests `runtime/execution/{runner_route,runner_loop,runner_absent}_test.py`,
`runtime/execution/resume_test.py`.

**Interfaces:**
- `route`: the `then` subgraph's nodes are added with `when=<predicate port>`, `otherwise`'s with `unless=`; each
  output merges through `select(2)`. With no `otherwise`, a false predicate gives `Absent(content, not-routed)`.
- `loop`: `workflow.loop(body, until=, carry=, max_rounds=)`; round `k`'s inner node ids are `<loop>/r<k>/<node>` in
  events and provenance; `rounds`, `converged`, `stopped` are outputs.
- A task's `TaskSpec.absent` becomes each pydra input field's `absent=`; a worker's returned kernel `Absent` is
  converted by `absence.to_pydra`.

- [ ] **Step 1: Failing tests:** `test_route_runs_only_taken_branch`, `test_route_without_otherwise_gives_not_routed`,
  `test_resume_takes_recorded_branch`, `test_loop_rounds_have_distinct_ids`, `test_loop_stops_on_repeated_round_checksum`,
  `test_loop_max_rounds_sets_stopped`, `test_resume_replays_loop_rounds_from_store`,
  `test_absent_propagates_without_dispatch`, `test_accept_mode_passes_absent`, `test_drop_mode_gathers_present_only`,
  `test_absent_not_accepted_fails_typed`, `test_environment_absent_recomputed_on_resume`,
  `test_content_absent_reused_on_resume`.
- [ ] **Step 2: Implement; run** the four test files with `-m "logic or contract"` (CI `core-pydra`).
- [ ] **Step 3:** review, compact, commit `feat(core): compile routes, loops and absence`; push; CI green. Spec:
  ADR-010 amendment (round and branch ids in activity names), ADR-001 amendment log.

### Task 7: The merged config loader

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.6 (three profile
kinds), P-17, P-18, P-42; R-282 (owner ruling: exclusions are a floor), R-320; ruling P4-7. **Depends on:** Task 0.

**Files:** create `runtime/layers.py`; modify `runtime/config.py` (keeps `load_yaml`, loses `merge_over`),
`runtime/profiles.py`, `runtime/call.py` (`RunConfigFile.load`), `runtime/keys.py` (`profile_slices` reads
`LoadedProfile`), `scripts/check_config_derivations.py`; tests `runtime/layers_test.py`, `runtime/profiles_test.py`.

**Interfaces:**
- `ProfileKind = Literal["run", "dataset", "execution"]`; `ProfileRef.parse("name:version") -> ProfileRef`.
- `load_profile(kind: ProfileKind, source: str | Path, *, overrides: Mapping[str, Any] | None = None) -> LoadedProfile`:
  a `name:version` resolves to a packaged profile, a path to a user file; layers are packaged default → `extends:`
  chain (each a `name:version` or path) → the file → `overrides`.
- Merge: mappings deep-merge; lists replace, except keys the kind's schema marks additive (`placement.exclude` and
  every `*.exclude`), which take the union (R-282).
- `LoadedProfile(Frozen)`: `kind`, `ref`, `document: FrozenMapping`, `layers: tuple[str, ...]`, `digest() -> str`
  (over `document` only; the layer list is recorded, not keyed).
- Variables: only `${VAR}` and `${VAR:-default}` expand; a bare `$VAR` stays text (R-320 N-4); the dead unset branch
  in `config.expand` goes (N-6).
- Every key of a packaged profile has a derivation section; `check_config_derivations.py` covers all three kinds.

- [ ] **Step 1: Failing tests:** `test_extends_chain_merges_in_order`, `test_extends_cycle_refused`,
  `test_exclusions_are_a_floor`, `test_digest_ignores_layer_paths`, `test_bare_dollar_not_expanded`,
  `test_execution_profile_loads_through_layers`, `test_run_config_loads_through_layers`,
  `test_unknown_key_refused_with_suggestion`, `test_packaged_key_without_derivation_fails_check`.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/runtime/layers_test.py packages/senselab-core/tests/runtime/profiles_test.py packages/senselab-core/tests/lint/check_config_derivations_test.py -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): one loader for run, dataset and execution profiles`; push; CI
  green. Spec: ADR-004 sentence on exclusions (R-282 closed), R-320 closed, `config-derivations.md`.

### Task 8: Dataset profiles, unfitted values, and the P-12 check

**Model:** opus implements (keying); fresh opus reviews. **Tier:** logic, contract. **Implements:** design §4.6
(dataset profile row, profile slices), P-12, I-1, UX-11, UX-26; rulings P4-5, P4-6. **Depends on:** Task 7.

**Files:** create `kernel/dataset_profile.py`, `scripts/check_dataset_coupling.py`; modify `kernel/declare.py`
(slice kinds `run.`, `dataset.`), `runtime/planner.py`, `runtime/keys.py`; tests `kernel/dataset_profile_test.py`,
`runtime/planner_profile_test.py`, `tests/lint/dataset_coupling_test.py`.

**Interfaces:**
- `DatasetProfile(Formatted)`, `format = "dataset-profile@1.0"`, loaded by `load_profile("dataset", ...)`:
  `entities: Mapping[str, EntitySpec]` (`EntitySpec{source: str, level: int}`: a BIDS key or record field, and its
  grouping level), `sidecar: Mapping[str, str]` (canonical field → sidecar key path), `tasks: Mapping[str,
  TaskProtocol]`, `lexicons: Mapping[str, ArtifactRef]`, `cohort: Mapping[str, JsonValue]`, `fitted: Mapping[str,
  float | None]` (`None` = unfitted), `artifacts: Mapping[str, ArtifactRef]` (fitted-profile artifacts by content
  identity).
- `TaskProtocol(Frozen)`: `kind: str` (the task family, e.g. `reading`, `sustained`, `syllable-train`,
  `breath-series`, `free`), `expected_text_field: str | None`, `lexicon: str | None`, `categories: tuple[str, ...]`,
  `traits: Mapping[str, JsonValue]`. Field names follow the 1.x stack's `tasks.yaml` (#627), renamed only where a 2.x
  name exists.
- `DatasetProfile.protocol_of(record: Record) -> TaskProtocol | Absent` (`Absent(content, out-of-protocol)` when the
  record's task entity has no entry).
- A task declares `profile_slices=("dataset.fitted.<group>", ...)`; the slice is in its key (I-1). At plan time a
  declared slice holding `None` makes that node's output `Absent(content, unfitted)` for every record, recorded in the
  plan, and no job is formed (`explain` lists it).
- `check_dataset_coupling.py`: every string literal and attribute name in `packages/*/src/**/{tasks,pipelines,signal,
  multimodal}/**/*.py` is checked against the profile names, task ids, sidecar keys and entity sources of every
  profile under `tests/fixtures/profiles/` and of `$B2AI_PROFILE` when set; any hit fails with file:line.

- [ ] **Step 1: Failing tests:** `test_threshold_change_rekeys_only_its_readers`,
  `test_sidecar_mapping_change_rekeys_only_readers`, `test_unfitted_slice_plans_absent_without_job`,
  `test_record_outside_protocol_is_out_of_protocol_absent`, `test_profile_slice_absent_from_key_refused` (a node reading
  an undeclared slice raises), `test_coupling_check_flags_planted_task_id`, `test_coupling_check_clean_on_packages`.
- [ ] **Step 2: Implement; run** the three test files with `-m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): dataset profiles and the dataset coupling check`; push; CI green.
  Spec: ADR-006 row `dataset-profile@1.0`; P-12's check names the script.

### Task 9: Full dataset adapters

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.6 (adapters), P-11,
I-7, H-6; R-281. **Depends on:** Task 8.

**Files:** modify `kernel/dataset.py`, `runtime/bids.py`, `runtime/datasets.py`; tests `runtime/{bids,datasets}_test.py`,
`runtime/bids_tree_fixture.py`.

**Interfaces:**
- `IgnoredReason = Literal["outside-grammar", "derivative", "symlink-out-of-root", "entity-mismatch", "caseless-collision",
  "unreadable"]`; `BidsDataset.ignored: Mapping[IgnoredReason, tuple[str, ...]]`.
- `bids_dataset(root: Path, *, profile: DatasetProfile | None = None, datatype=None, suffix=None, extensions=(".wav",
  ".flac"))`: names matched with `fullmatch`; a filename entity must equal its directory's (`sub-`, `ses-`); sidecar
  fields read through `profile.sidecar`; entities and grouping levels from `profile.entities` (BIDS defaults without a
  profile).
- `Dataset.groups(by: str) -> Iterator[tuple[str, Dataset]]` (sorted, closed under `by`, I-7);
  `Dataset.layers() -> Mapping[str, Dataset]` (BIDS `derivatives/<pipeline>/` trees as annotation layers, never raw
  records).
- `datasets.from_frame(frame, *, id_column="id")`, `datasets.from_hf(dataset, *, columns: Mapping[str, str])`,
  `datasets.from_mapping(mapping: Mapping[str, Mapping[str, Any]])`; pandas and `datasets` imported lazily.

- [ ] **Step 1: Failing tests:** `test_every_requested_file_is_record_or_ignored` (generated tree with each reason
  planted), `test_fullmatch_names`, `test_filename_entity_must_equal_directory`, `test_derivatives_never_raw_records`,
  `test_layers_expose_derivatives`, `test_groups_closed_under_level`, `test_sidecar_read_through_profile_mapping`,
  `test_same_pipeline_runs_on_bids_and_frame` (P-11: one fake pipeline, identical outputs).
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab-core/tests/runtime/bids_test.py packages/senselab-core/tests/runtime/datasets_test.py -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): full BIDS, frame, HF and mapping datasets`; push; CI green.
  Register R-281 closed.

### Task 10: Accelerometry and the series reader

**Model:** sonnet implements; opus reviews (ADR-005). **Tier:** logic, contract. **Implements:** design §4.1, §4.15,
ADR-005, P-26. **Depends on:** Task 1.

**Files:** modify `kernel/signals.py`; create `runtime/series_io.py`, `runtime/data/units.yaml` (unit vocabulary and
conversions); tests `kernel/signals_test.py`, `runtime/series_io_test.py`, `packages/senselab-core/src/senselab_core/testing/synth.py`
(`accelerometry(seconds, rate, seed)`).

**Interfaces:**
- `Accelerometry(Recording)`: one regular group, channels of `type="accel"`, units `g` or `m/s^2`, 1 or 3 channels;
  validated as `Audio` is.
- `SeriesLayout(Frozen)`: `time_column: str | None`, `columns: Mapping[str, ChannelInfo]`, `rate: float | None` (a
  regular series when set, else irregular from `time_column`), `clock_id: str`. Carried by a dataset profile's record
  view (`DatasetProfile.views[<name>].layout`).
- `open_series(path: Path, layout: SeriesLayout, *, kind: type[Recording]) -> Recording` for CSV and Parquet; the file
  is stamped with the chunked source stamp `audio_io` uses (one stamping implementation, moved to
  `runtime/stamping.py`).

- [ ] **Step 1: Failing tests:** `test_accelerometry_units_validated`, `test_accelerometry_refuses_nan`,
  `test_series_csv_regular_and_irregular`, `test_series_parquet_round_trip`, `test_series_rewrite_is_stale`,
  `test_one_stamping_implementation` (audio and series stamps share `stamping.stamp`).
- [ ] **Step 2: Implement; run** the two test files with `-m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(core): accelerometry and a series reader`; push; CI green. Spec:
  ADR-005 amendment (accelerometry specialisation).

### Task 11: `sl.signal` transforms

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.15 tier 1
(filter, units, windowing, spectral and envelope features), use-cases.md §4(a); merges C-2 (envelope), C-3
(spectral representations) and C-13 (1.x preprocessing). **Depends on:** Tasks 1b, 10.

**Files:** create `signal/{filter,rescale,channels,windows,spectrum,envelope}.py`, `signal/data/{filter,spectrum}.yaml`;
add scipy to the root `pyproject.toml` with its license record; tests under `packages/senselab/tests/signal/`.

**Interfaces** (each `@task(name="signal.<name>", version="1", kind="transform")` on `RegularSeries`, backend
`numpy` on stack `dsp`, host env, as `signal.resample`):
- `signal.filter`: Params `kind: Literal["lowpass", "highpass", "bandpass", "bandstop", "notch", "detrend"]`,
  `cutoff_hz: tuple[float, ...]`, `order: int`; zero-phase SOS.
- `signal.rescale`: Params `to_units: str` (from `units.yaml`); refuses a channel without units.
- `signal.channels`: Params `op: Literal["select", "vector_magnitude", "mean", "rereference"]`, `channels: tuple[str,
  ...]`.
- `signal.windows`: series → `Intervals` on a grid; Params `length_s`, `hop_s`, `align: Literal["start", "events"]`.
- `signal.spectrum`: series + `Intervals` → `Intervals` with value columns `band_power[<band>]`, `spectral_entropy`,
  `dominant_hz`; Params `bands: Mapping[str, tuple[float, float]]` (no default bands in code; a modality's bands come
  from a profile in `signal/data/spectrum.yaml`, chosen by channel `type`).
- `signal.envelope`: Params `method: Literal["hilbert", "rms"]`, `smooth_s: float` → `RegularSeries` track.

- [ ] **Step 1: Failing tests:** per task, a known-answer test on synthetic input (a tone's band power, a step's
  envelope, a notch's attenuation at the line frequency) with tolerances from the run config, plus
  `test_spectrum_runs_unchanged_on_audio_and_accelerometry` (the P-26 exit test, same task object, both inputs) and
  the generated contract tests for each declaration; `test_member_<name>_parity` for each C-2, C-3 and C-13 member
  (the 1.x `envelope`, `band_profile`, `gammatone`, `spectral_continuity` and `preprocessing` functions, outputs
  recorded from the merged 1.x
  head on synthetic input) and `test_defaults_declared_per_member`.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab/tests/signal -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(signal): filter, rescale, channels, windows, spectrum, envelope`; push;
  CI green; `config-derivations.md` sections for each data file.

### Task 12a: `sl.signal` detection, span algebra and QC

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** design §4.15 tier 1 (activity,
span algebra, QC as measures not verdicts); merges C-4 (spans, change points, disruptions, clipping). **Depends on:**
Task 11.

**Files:** create `signal/{activity,spans,qc}.py`, `signal/data/qc.yaml`; tests.

**Interfaces:**
- `signal.activity`: track → `Intervals`; Params `on_level`, `off_level` (hysteresis, in the track's units or a
  quantile when `relative=True`), `min_duration_s`, `min_gap_s`.
- `signal.spans`: `Intervals` (+ `Intervals`) → `Intervals`; Params `op: Literal["union", "intersect", "subtract",
  "merge_gaps", "min_duration"]`, `gap_s`, `duration_s`.
- `signal.qc`: series → `Intervals` (gaps, flatline, clipping) + `Intervals` per window with `snr_db`,
  `line_noise_db`, `clipped_fraction`, `out_of_range_fraction`. Every bound in `qc.yaml` per channel type; no verdict.

- [ ] **Step 1: Failing tests:** `test_activity_hysteresis_no_chatter`, `test_span_algebra_properties` (hypothesis:
  union commutative and idempotent, subtract ∪ intersect = original), `test_qc_finds_planted_clip_flatline_gap`,
  `test_qc_emits_measures_not_labels`; contract tests; `test_member_<name>_parity` for each C-4 member and
  `test_defaults_declared_per_member`.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab/tests/signal -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(signal): activity, span algebra and QC measures`; push; CI green.

### Task 12b: `sl.signal` synchronisation, gaps and the peak family

**Model:** sonnet implements; opus reviews (TimeMap semantics). **Tier:** logic, contract. **Implements:** design
§4.15 tiers 1 and 2 (synchronisation → `TimeMap`, imputation, regularisation, `detect_peaks` with modality profiles),
ADR-005. **Depends on:** Task 12a.

**Files:** create `signal/{synchronize,impute,regularize,peaks}.py`, `signal/data/peak_profiles/{speech_syllable,
cough_onset,breath,accel_step,ecg_rpeak}.yaml`; tests.

**Interfaces:**
- `signal.synchronize`: two series → `TimeMap` (offset and drift by cross-correlation over windows, with uncertainty).
- `signal.impute`: series + gap `Intervals` → series + `Intervals` of what was imputed; Params `method:
  Literal["linear", "hold"]`, `max_gap_s`.
- `signal.regularize`: an irregular group → regular at `rate`; gaps beyond `max_gap_s` are `Intervals`, never zero.
- `signal.detect_peaks`: series → `Events` with amplitude and prominence; Params `profile: str | None` (default by
  channel `type`); a profile holds band, refractory period, prominence and rate prior. A profile value that a dataset
  must fit is `null` in the packaged profile and read from `dataset.fitted.peaks.<profile>` (Task 8: unfitted →
  `Absent`).

- [ ] **Step 1: Failing tests:** `test_synchronize_recovers_planted_offset_and_drift`, `test_impute_records_what_was_imputed`,
  `test_regularize_never_zero_fills`, `test_detect_peaks_profile_by_channel_type`, `test_unfitted_peak_value_is_absent`,
  `test_detect_peaks_same_engine_all_profiles` (one engine, P-37); contract tests.
- [ ] **Step 2: Implement; run** `uv run --no-sync pytest packages/senselab/tests/signal -m "logic or contract"`.
- [ ] **Step 3:** review, compact, commit `feat(signal): synchronisation, imputation, regularisation, peak family`;
  push; CI green.

### Task 13: Merge C-5: voice features (Praat, openSMILE, 1.x phonation tracks)

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** the tasks
health_measurements and explore_conversation need; P-23; merge C-5. **Depends on:** Task 1b.

**Files:** `git mv` of `src/senselab/audio/tasks/features_extraction/{praat_parselmouth,opensmile}.py`,
`src/senselab/audio/tasks/phonation/` and their tests into `packages/senselab/src/senselab/audio/tasks/voice_features/`
(rename-only commit), then the modify commit: `voice_features/{praat,opensmile}_backend.py`, env locks `praat`,
`opensmile`, catalogue and license records.

**Interfaces:** `audio.voice_features` (`Audio` + optional `Intervals` → `Intervals` with value columns per feature,
or a `RegularSeries` track per feature when `output="track"`); Params `feature_set: Literal["praat", "egemaps"]`,
`features: tuple[str, ...] | None` (a subset: `f0`, `hnr`, `formants`, `jitter`, `shimmer`, `cpp`, ...), `output:
Literal["summary", "track"]`, `f0_range_hz: tuple[float, float] | None` (None = derived per recording, as 1.x
`derive_f0_range`), `win_length_s`, `hop_length_s`; backends `praat` (parselmouth, isolated, GPL-3.0 record) and
`opensmile` (isolated, audEERING LicenseRef). The 1.x `f0_track`, `hnr_track`, `formant_track`, `period_marks` and
`extract_praat_parselmouth_features_from_audios` are members, not tasks.

- [ ] **Step 1: Failing tests:** contract tests; `test_praat_known_tone_f0` and `test_egemaps_columns_stable` (local
  logic on a synthetic tone, fake env); `test_member_<name>_parity` per C-5 member at its `members.yaml`
  configuration against the merged 1.x head on the LibriSpeech fixture (tier `env`, ORCD);
  `test_defaults_declared_per_member`.
- [ ] **Step 2: Implement; local run** of the logic and contract tests; **ORCD:** env builds and the parity test; must
  show parity within the run-config thresholds on every feature column.
- [ ] **Step 3:** review, compact, commit; push; CI green; ledger rows.

### Task 14: Merge C-1: one embeddings task (speaker and SSL), on the `signal.windows` grid

**Model:** opus implements (it re-keys a shipped 2.x task); fresh opus reviews. **Tier:** logic, contract, env (ORCD
H100). **Implements:** the embeddings explore_conversation needs; alpha PR #649's behaviour (open 1.x work); merge
C-1; owner decision 8. **Depends on:** Tasks 1b, 11 (rerun Task 1 if #649 merged since).

**Files:** rename-only `git mv` of `src/senselab/audio/tasks/ssl_embeddings/`, then the modify commit:
`audio/tasks/embeddings/` holding the existing speechbrain backend (moved from `speaker_embeddings/`) and a new
`transformers` backend; `speaker_embeddings/windowing.py` deleted, its windows taken from `signal.windows` (one
windowing engine); `_api/declarations.py` declares `audio.embeddings` and drops `speaker_embeddings`; the catalogue,
`describe` output and the pyannote/EER metric callers updated.

**Interfaces:** `audio.embeddings` (`Audio` + optional `Intervals` → `list[Embedding]`; `space` is the resolved model
identity, so a speaker vector and an SSL vector never compare); Params `layers: Literal["output", "last", "all"] |
tuple[int, ...] = "output"` (`output` is the model's embedding head, the former `speaker_embeddings`), `pool:
Literal["mean", "none"]`, `window_s: float | None`, `hop_s: float | None`; backends `speechbrain` (isolated, as
today) and `transformers` (host); encoder-decoder models use the encoder only (#649). Identity: the task name changes,
so every `speaker_embeddings` key changes; nothing is migrated (P-14).

- [ ] **Step 1: Failing tests:** contract tests; `test_encoder_decoder_uses_encoder_only` (a tiny config from the
  test data, no download); `test_member_speaker_embeddings_parity` (the 2.x task's recorded outputs at its Phase 3
  parity configuration), `test_member_ssl_embeddings_parity` (1.x, ORCD, cpu and H100);
  `test_windows_from_signal_grid`; `test_defaults_declared_per_member`; `test_speaker_embeddings_name_refused` (the old
  name raises naming `audio.embeddings`).
- [ ] **Step 2: Implement; local logic and contract run; ORCD parity**; must show `isclose` under run-config
  thresholds per vector for both members.
- [ ] **Step 3:** review, compact, commit; push; CI green; `consolidation.md` C-1 closed; ledger rows (if #649 is still
  open, the row says the behaviour is ported from the PR head `<sha>` and Task 1 reconciles it on merge).

### Task 15: explore_conversation and health_measurements as pipelines

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, usecase (ORCD). **Implements:** design §6.2
(port explore_conversation and health_measurements), P-9, P-11. **Depends on:** Tasks 5, 9, 13, 14.

**Files:** create `pipelines/explore_conversation.py`, `pipelines/health_measurements.py`; rename-only `git mv` of the
two 1.x modules into `pipelines/` first; tests `packages/senselab/tests/pipelines/`.

**Interfaces:** each module exposes `build(config: LoadedProfile | None = None) -> Pipeline`, model choices and
windows from the run config (`run.pipelines.<name>`):
- explore_conversation: `signal.resample` → `signal.channels(mean)` → `diarize` → `map(segments)` →
  `speech_to_text`, `audio.embeddings` (speaker model, then SSL model: two nodes of one task), `audio.voice_features`
  → `gather`.
- health_measurements: `audio.voice_features(praat)` per record, `over=` the dataset's top grouping level for the
  per-participant summary (`signal.spans` + summary task `audio.feature_summary`).

- [ ] **Step 1: Failing tests:** `test_explore_conversation_graph_validates`, `test_health_measurements_graph_validates`,
  both with the fake stack end to end (logic); usecase test on a two-speaker mix of two LibriSpeech fixture utterances
  and on the LibriSpeech BIDS fixture.
- [ ] **Step 2: Implement; local logic run; ORCD usecase run**, CPU and H100: every node completes; outputs compared to
  the 1.x workflow on the same inputs under run-config thresholds.
- [ ] **Step 3:** review, compact, commit `feat(pipelines): explore_conversation and health_measurements`; push; CI
  green; ledger rows moved.

### Task 16: Merge C-6: one audio classification task (AST, YAMNet, emotion models)

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** audio_analysis's
scene stage; triage's YAMNet and label uses; merge C-6. **Depends on:** Tasks 0b, 1b.

**Files:** rename-only `git mv` of `src/senselab/audio/tasks/classification/` and the triage `label_membership.py`,
`classifier_ontology.py`; then `audio/tasks/classify/{transformers,transformers_backend,yamnet,yamnet_backend}.py`,
`audio/tasks/classify/data/` (YAMNet class names; label ontology maps by model), env lock `yamnet` (TensorFlow,
isolated), catalogue entries pinned by commit.

**Interfaces:** `audio.classify` (`Audio` → `Intervals` with label scores per window); Params `win_length_s`,
`hop_length_s`, `top_k: int | None`, `floor: float | None` (label membership, from triage), `ontology: str | None`
(a label map in `data/`, e.g. onto AudioSet nodes); backends `transformers` (AST and speech emotion models by `model`,
host) and `yamnet` (tf, isolated). Scene, speech emotion recognition and label membership are configurations, not
tasks.

- [ ] **Step 1–3:** contract tests; `test_member_<name>_parity` per C-6 member (1.x `classify_audios` with AST, with
  YAMNet, with an emotion model; triage label membership) on the fixture (ORCD); `test_defaults_declared_per_member`;
  review, commit, CI green, C-6 closed, ledger rows.

### Task 17: Speech enhancement (SepFormer)

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD H100). **Implements:**
audio_analysis's enhanced pass. **Depends on:** Task 0b.

**Files:** rename-only `git mv` of `src/senselab/audio/tasks/speech_enhancement/` speechbrain path; then
`audio/tasks/enhancement/{speechbrain,speechbrain_backend}.py` in the existing speechbrain env.

**Interfaces:** `audio.enhance` (`Audio` → `Audio`, same rate and length); backend `sepformer` (speechbrain).

- [ ] **Step 1–3:** contract tests (length and rate preserved); ORCD parity against 1.x on the fixture; review, commit,
  CI green, ledger rows.

### Task 18: The NeMo stack: Sortformer diarization and Canary-Qwen ASR

**Model:** sonnet implements; opus reviews (env and license). **Tier:** contract, env (ORCD H100). **Implements:**
audio_analysis's second diarizer and second ASR; P-8 (a second stack); C-7 (a model or stack is a backend of the
existing task, never a task). **Depends on:** Tasks 0b, 1b.

**Files:** one isolated env `nemo` (lock excludes the quarantined `lightning` 2.6.2 and 2.6.3); backends of `diarize`
(`speaker_diarization/sortformer*.py`) and of `speech_to_text` (`speech_to_text/canary_qwen*.py`), rewritten from the
1.x subprocess workers (their 1.x importers have not moved); catalogue, env and model license records. No new task
name; `test_no_task_per_backend` (registry: each new backend declares an existing task).

- [ ] **Step 1–3:** contract tests; ORCD: env build, then parity against 1.x on the fixture (DER for Sortformer, WER and
  word times for Canary-Qwen) under run-config thresholds; review, commit, CI green, ledger rows.

### Task 19: Qwen3-ASR and forced alignment (Qwen3 aligner, MMS)

**Model:** sonnet implements; opus reviews. **Tier:** contract, env (ORCD H100). **Implements:** audio_analysis's
third ASR and its alignment stage; C-7. **Depends on:** Tasks 0b, 1b.

**Files:** env `qwen-asr`; backends `speech_to_text/qwen_asr*.py`; one task `audio.align` in
`audio/tasks/forced_alignment/` with backends `qwen-aligner` (qwen-asr env) and `mms` (transformers, host, per-language
adapter pinned by commit). Qwen3-ASR's native timestamps and the separate align stage of 1.x are one path: the run
config chooses whether `audio.align` runs after a recognizer that has word times.

**Interfaces:** `audio.align` (`Audio` + `Transcript` → `Transcript` with word spans); Params `language: Language |
None`.

- [ ] **Step 1–3:** contract tests; ORCD parity against 1.x for both aligners (word boundary error under run-config
  thresholds), `test_member_<name>_parity` per aligner; review, commit, CI green, ledger rows.

### Task 19b: Merge C-11: voice activity detection

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** merge C-11: the 1.x
`detect_human_voice_activity_in_audios` (derived from a diarization model), audio_analysis's speech-presence evidence
and the energy detector become one task. **Depends on:** Tasks 12a, 18.

**Files:** rename-only `git mv` of `src/senselab/audio/tasks/voice_activity_detection/`; then
`audio/tasks/voice_activity/{__init__,strategies}.py`.

**Interfaces:** `audio.voice_activity` (`Audio` → `Intervals`); Params `strategy: Literal["diarization", "energy"]`,
`model: str | None` (the diarization model; any `diarize` backend), `energy_profile: str | None` (a `signal.activity`
profile in `data/`); the `diarization` strategy calls `diarize` and unions its speaker intervals, the `energy` strategy
is `signal.envelope` + `signal.activity`. No VAD code of its own.

- [ ] **Step 1–3:** `test_member_<name>_parity` per C-11 member on the fixture (ORCD for the diarization strategy);
  `test_defaults_declared_per_member`; contract tests; review, commit, CI green, C-11 closed, ledger rows.

### Task 20: audio_analysis classification

**Model:** opus (classification judgement); a haiku verifier checks each row's file:line. **Tier:** none (spec).
**Implements:** ledger "Moving a use case". **Depends on:** Tasks 1b, 16-19b.

- [ ] **Step 1 (spec):** add an "audio_analysis" table to `migration-ledger.md` like triage's: every module of
  `src/senselab/audio/workflows/audio_analysis/` (81 files) and `adaptive/` classed `task` (naming its C-n group when it
  is a member), `pipeline`, `plot` (not ported in Phase 4: plots, `labelstudio.py`, `layout.py`, the review page) or
  `drop` (superseded by the 2.x runtime: `io.py`, `keys.py`, `stage_io.py`, `stage_context.py`, `run_config.py`, the
  cache), with its reason. New groups found here are added to `consolidation.md`.
- [ ] **Step 2:** commit `docs(2x): audio_analysis classification`; push.

### Task 20a: Merge C-8: word consensus

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** merge C-8: 1.x
`speech_to_text_ensemble.fuse_word_streams`, audio_analysis's word fusion and triage's `consensus.py` and
`word_arbitration.py` become `audio.word_consensus`. **Depends on:** Task 20.

**Interfaces:** `audio.word_consensus` (`list[Transcript]` → `Transcript` with per-word agreement and member
confidence); Params `grouping: Literal["sequence", "time"]`, `similarity: Literal["phonemic", "orthographic"]`,
`timing: Literal["consensus", "arbitrate"]` (triage's word timing arbitration), `calibrator: str | None` (a fitted
artifact by identity).

- [ ] **Step 1–3:** `test_member_<name>_parity` per C-8 member on recorded fixture hypotheses;
  `test_defaults_declared_per_member`; contract tests; review, commit, CI green, C-8 closed.

### Task 20b: Merge C-9: background floor

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** merge C-9: 1.x
`envelope.global_floor_dbfs`, audio_analysis `noise_floor.py`/`floors.py`, triage `background_model.py` and
`session_background.py` become `audio.background_floor`. **Depends on:** Tasks 12a, 20.

**Interfaces:** `audio.background_floor` (`Audio` + optional speech `Intervals` → `Intervals` with floor level, hum
frequency and power, impulse events); Params `scope: Literal["recording", "residual", "session"]` (session runs with
`over=` the session grouping key), `exclude: Literal["none", "speech"]`; bounds in `data/` with derivations.

- [ ] **Step 1–3:** `test_member_<name>_parity` per C-9 member; `test_background_floor_on_planted_hum_and_noise`;
  `test_defaults_declared_per_member`; contract tests; review, commit, CI green, C-9 closed.

### Task 20c: Merge C-10: speaker enrollment and verification

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** merge C-10: 1.x
`estimate_speaker_embedding_from_audios` and `speaker_verification`, audio_analysis `speaker_identity.py`, triage
`enrollment.py` become `audio.speaker_enrollment` and one scoring task. **Depends on:** Tasks 14, 20.

**Interfaces:** `audio.speaker_enrollment` (`list[Embedding]` (+ `Intervals`) → `Embedding`, run `over=` the
participant key); Params `aggregate: Literal["mean", "medoid"]`, `min_speech_s`. `audio.speaker_similarity`
(`Embedding` × `Embedding` → `Annotation` with a score); Params `measure: Literal["cosine", "centred_cosine"]` (the
1.x `centred_cosine_similarity` is this measure).

- [ ] **Step 1–3:** `test_member_<name>_parity` per C-10 member; `test_enrollment_vector_from_n_recordings`;
  `test_defaults_declared_per_member`; contract tests; review, commit, CI green, C-10 closed.

### Task 21: audio_analysis L1: per-axis measurement tasks

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract. **Implements:** audio_analysis `doc.md` (one
grid, four axes, one voter per model), P-20's L1/L2 split as measurement tasks. **Depends on:** Tasks 20a-20c.

**Files:** rename-only `git mv` of the `task`-classed L1 modules into `audio/tasks/uncertainty/`; then the modify
commit; the grid (`0.1 s` window and hop) and each axis's parameters move to the run config with their derivations
from `doc.md`.

**Interfaces:** `audio.uncertainty.votes` (one task, Params `axis: Literal["speech_presence", "speaker", "asr",
"background_mask"]`) → `Intervals` on the run-config grid with `uncertainty`, `epistemic_uncertainty`, `confidence`,
`variability` columns and the voter in `producer`; a bucket no evidence reaches carries no row, never `0.0`.

- [ ] **Step 1–3:** `test_every_axis_on_one_grid`, `test_no_vote_where_no_evidence`, `test_votes_match_1x_on_fixture`
  (recorded 1.x outputs on the fixture, under thresholds); contract tests; review, commit, CI green.

### Task 22: audio_analysis L2 and the analysis pipeline

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, usecase (ORCD H100). **Implements:** the fold
(`fuse.py`, `reliability.py`, `support.py`, `aggregators.py`), P-9 (one pass pipeline nested twice). **Depends on:**
Tasks 5, 21.

**Files:** `audio/tasks/uncertainty/fold.py`; `pipelines/audio_analysis.py` with `pass_pipeline(config) -> Pipeline`
(diarizers, ASRs, embeddings, `audio.classify`, alignment, votes) and `build(config) -> Pipeline` (the pass on raw audio, the same
pass on `audio.enhance`'s output, then `audio.uncertainty.fold` over both).

- [ ] **Step 1–3:** `test_pass_nested_twice_keys_equal_standalone`, `test_fold_aggregator_from_run_config`; ORCD usecase
  run on the public fixture against `scripts/analyze_audio.py` output from the merged 1.x head: every axis row equal
  under thresholds; review, commit, CI green; ledger rows moved.

### Task 23: The adaptive loop as `p.loop`

**Model:** opus implements (loop semantics); fresh opus reviews. **Tier:** logic, contract, usecase (ORCD).
**Implements:** audio_analysis `adaptive/` as a loop whose rounds are recorded (design §4.5; owner answer: `p.loop`
when a use case needs it). **Depends on:** Tasks 6, 22.

**Files:** `pipelines/adaptive.py`: `build(config) -> Pipeline` whose body takes the previous round's fold, picks
interventions (`audio.uncertainty.interventions`, a task), runs them through routes, refolds, and outputs `done: bool`
(`converged` from the policy's convergence rule).

- [ ] **Step 1–3:** `test_round_k_reads_round_k_minus_1_fold`, `test_loop_converges_on_fixture_policy`,
  `test_resume_mid_loop_replays_rounds`; ORCD usecase against `scripts/adaptive_loop.py` on the fixture (same rounds,
  same interventions); review, commit, CI green; the policy digest moves to a registered tag (ledger "Untagged digests").

### Task 24: Triage: reclassification, the b2ai-voice and synthetic profiles

**Model:** opus (classification); sonnet (profiles); a haiku verifier on rows. **Tier:** none (spec), logic, contract.
**Implements:** design §6.2 (B2AI coupling in a `b2ai-voice` dataset profile and task protocol), P-12; owner decision
1 below. **Depends on:** Task 8; Task 1 rerun after the triage stack (#607-#642, open 1.x work) merges into `alpha`.

- [ ] **Step 1 (spec):** refresh the triage table in `migration-ledger.md` at the merged `alpha` head: modules the stack
  added, moved to b2aiprep (#639) or split (#627, #628) get rows; `consolidation.md` groups are refreshed and new members
  added (a group already closed reopens only if a new member is found).
- [ ] **Step 2:** the `b2ai-voice` profile (`dataset-profile@1.0`): entities, sidecar mapping (the per-recording
  stimulus text and speech type fields), the task protocol built from the 1.x profile's `tasks.yaml` (48 entries),
  lexicons, cohort conditions and every fitted value with the run it came from. It lives where owner decision 1 puts
  it; senselab's tests use only `tests/fixtures/profiles/adult-speech-fixture/` (neutral names, as #639's fixture).
- [ ] **Step 3:** `tests/fixtures/profiles/synthetic-cohort/` and its generator `testing/synth_cohort.py`: a different
  entity scheme (`participant`, `visit`, `activity` instead of `sub`, `ses`, `task`), a different task protocol
  (four task families with other ids), and `fitted` values left `null`; recordings synthesised (tones, noise, a
  two-voice mix) as an in-memory dataset and as a BIDS-like tree.
- [ ] **Step 4:** `test_both_profiles_load`, `test_synthetic_has_disjoint_task_ids_and_entities`,
  `test_coupling_check_clean_with_both_profiles`; review, commit, CI green.

### Task 25: Triage measurement tasks: signal level

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** triage `task` rows
for admit checks, background, session background, breath and cough patterns, the spectrogram grid. **Depends on:**
Tasks 12b, 20b, 24.

**Files:** rename-only `git mv` per module into `audio/tasks/triage/`, then modify: admit checks become
`signal.qc` profiles (no new task); `background_model.py` and `session_background.py` are C-9 members already merged
(Task 20b; this task wires their profile values);
breath and cough onsets are `signal.detect_peaks` with the `breath` and `cough_onset` profiles over
`signal.envelope`; `review_page/spectrogram.py` → `signal.spectrum`'s grid. Thresholds fitted on B2AI move to the
profile's `fitted` section (#628's split of method and numbers).

- [ ] **Step 1–3:** per behaviour a known-answer test on synthetic input and a parity test against the merged 1.x
  modules on the fixture profile; `test_unfitted_cough_rise_is_absent_on_synthetic`; review, commit, CI green, ledger
  rows moved.

### Task 26: Triage measurement tasks: transcript and task content

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD). **Implements:** triage `task` rows
`stimulus`, `task_extent`, `task_events`, `task_content`, `task_lexicon`, `task_speech`, `residue`,
`finding_placement`, `task_audio`; `nodes/redact.py` is a C-12 member (Task 29). **Depends on:** Tasks 20a, 25.

**Interfaces:** each a task over `Transcript` (+ `Audio`, + the record's `TaskProtocol` from the dataset profile as
an input, never read from a sidecar inside the task): e.g. `audio.task_extent(transcript, protocol) -> Intervals`,
`audio.stimulus_alignment(transcript, protocol) -> Annotation` list. Word consensus is `audio.word_consensus` (Task 20a).

- [ ] **Step 1–3:** parity per task against the merged 1.x module on the fixture profile; on the synthetic profile
  every task runs or returns a typed `Absent` (`out-of-protocol`, `unfitted`); review, commit, CI green, ledger rows.

### Task 27: Triage measurement tasks: speakers, voice and quality

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, env (ORCD H100). **Implements:** triage `task`
rows `speaker_share`, `voice_phonation` (phonation measures from `audio.voice_features`, C-5),
`ddk_task`/`nodes/ddk.py` (syllable events via `signal.detect_peaks(speech_syllable)`), `nodes/taxonomy.py` (over
`audio.classify`, C-6), `background_speech.py` (`audio.classify` with YAMNet on the enhancement residual),
`quality_join.py`, `task_squim.py` (new backend `squim`, torchaudio, isolated). `enrollment`, `label_membership` and
`classifier_ontology` are already merged (C-10, C-6). Disordered phonation is measured, not flagged. **Depends on:**
Tasks 13, 16, 20c, 26.

- [ ] **Step 1–3:** parity per task on the fixture profile; synthetic profile runs or is typed-absent; review, commit,
  CI green, ledger rows.

### Task 28: The voice measurement pipeline (the exit pipeline)

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, usecase (ORCD). **Implements:** design §6.2
Phase 4 exit (one pipeline from the ported tasks on two dataset profiles), P-12. **Depends on:** Task 27.

**Files:** `pipelines/voice_measurements.py`: `build(config) -> Pipeline`, the triage measurement chain (admit QC →
background → diarize, ASR ×n → word consensus → task extent and events → voice, airway and syllable measures → quality
→ PII) with routes on the record's protocol `kind` (a declared task `audio.protocol_kind(protocol) -> bool` per kind),
nothing dataset-specific in code.

- [ ] **Step 1–3:** `test_pipeline_has_no_dataset_literals` (Task 8's check on this module), fake-stack end-to-end run
  on both fixture profiles (logic); review, commit, CI green; ledger: triage `pipeline` rows the chain covers.

### Task 29: The cross-modal pipeline: video → audio → text

**Model:** sonnet implements; opus reviews. **Tier:** logic, contract, usecase (ORCD H100). **Implements:** design
§4.11 (bridges in `sl.multimodal`), P-24's check (no conversion code between nodes), §6.4 (one cross-modal bridge);
R-268; owner decision 5; merge C-12 (redaction). **Depends on:** Tasks 1b, 5, 6.

**Files:** `kernel/media.py` (`Video(Recording)`: a stamped container with stream metadata; frames are not decoded);
`multimodal/demux.py` (`multimodal.demux`: `Video` → `Audio`, backend `pyav`, isolated); `audio/tasks/redact/` (merge
C-12: 1.x `redaction.plan_redactions`/`apply_redactions` and triage `nodes/redact.py` become `audio.redact`, `Audio` +
`TextSpans` + `Transcript` → `Audio` with `Intervals` masked; Params `pad_s`, `merge_gap_s`, `mask: Literal["silence",
"tone", "noise"]`); `pipelines/speech_pii.py` (`demux` → `signal.resample` → `speech_to_text` → `pii_spans` →
`audio.redact`, spans on the video's clock through the transcript's word spans; the 1.x `detect_pii_in_audios` is this
pipeline, not a task); `text/tasks/pii_spans/transformers.py` loads the w4a16 checkpoint without dequantising, or
the profile prefers the `vllm` stack for `pii_spans` (R-268, whichever the ORCD measurement supports).

- [ ] **Step 1–3:** `test_demux_audio_shares_video_clock`, `test_speech_pii_has_no_conversion_nodes` (every edge joins
  equal kernel types), usecase on a public CC-licensed clip (docs.md fixture policy); ORCD peak GPU memory of
  `pii_spans` recorded (must show ≤ 40 GiB on H100 for the transformers path, or the profile routes to vllm);
  `test_member_<name>_parity` per C-12 member and `test_defaults_declared_per_member`; review, commit, CI green;
  register R-268 closed, C-12 closed.

### Task 30: Adopt the environment grouping (P-24)

**Model:** sonnet implements; opus reviews (env identity). **Tier:** contract, env (ORCD). **Implements:** design
§4.11, P-24 ("adopted in Phase 4"), §6.4 row amended by owner decision 3. **Depends on:** Tasks 13-19, 29.

**Files:** `scripts/env_grouping.py` rerun (ORCD, network) over every lock; `data/env_grouping.yaml` (`env-grouping@1`)
becomes an input of `scripts/lock_envs.py`: one lock per group; `runtime/envs.py` resolves a backend's env to its
group's; each backend's recorded execution context names the group env.

- [ ] **Step 1:** failing tests: `test_backend_env_resolves_to_group`, `test_group_change_changes_recorded_context_not_key`,
  `test_group_lock_covers_every_member_requirement`.
- [ ] **Step 2 (ORCD):** env count and per-worker resident memory of the audio_analysis pipeline before and after, on
  one H100 node; the numbers go into the `env_grouping.yaml` derivation. Must show fewer envs and no member's parity
  test failing in its group env.
- [ ] **Step 3:** review, compact, commit; CI green; spec: design §6.4 row and §4.11 amended.

### Task 31: Phase 4 exit checks

**Model:** sonnet implements; opus reviews; the P-1 check uses a fresh sonnet agent with no repository access.
**Tier:** all. **Depends on:** Tasks 0-30.

- [ ] **Step 1 (CI):** `2x-tests` green at the head of `2x/phase-4`, `core-pydra` included; the nightly (alpha PR #643)
  green on `2.x`; one `full-matrix` run at the head.
- [ ] **Step 2 (P-12, ORCD H100):** `voice_measurements.build()` planned and run on `$B2AI_BIDS_ROOT` (200 recordings
  across every task family) under `$B2AI_PROFILE`, and on the synthetic cohort under `synthetic-cohort`, changing only
  the dataset profile: the two plans' graph digests are equal; every node of every record completes or is a typed
  `Absent`; the synthetic run's unfitted values yield `Absent(content, unfitted)` with no job formed;
  `check_dataset_coupling.py` finds nothing in `packages/`. Must show: 0 failed elements on both runs.
- [ ] **Step 3 (P-9):** the audio_analysis pass pipeline run standalone, then nested in `audio_analysis.build()` on
  the same inputs: every inner node of the nested run is a reuse.
- [ ] **Step 4 (P-26):** `signal.spectrum` and `signal.activity` run as one task object on the LibriSpeech fixture and on
  a public accelerometry recording read through `open_series`; both runs complete.
- [ ] **Step 5 (P-10, P-5 for pipelines, ORCD):** `explore_conversation` over 1000 LibriSpeech records planned under
  `laptop` and `orcd`: one plan hash, placement-only diff; interrupted at a third and resumed: executions = elements not
  committed, no ref resolution, routes and rounds replayed from the store.
- [ ] **Step 6 (ORCD, CPU and H100):** the full packages suite, every tier, serial; env builds for every shipped lock;
  R-306's provenance read and write cost at 10× Step 5's size, recorded.
- [ ] **Step 7 (P-1):** the fresh agent receives only `sl.describe()` output and a dataset path, composes a two-node
  pipeline as `graph@2.0` JSON, `sl.validate` returns no issue and `sl.plan(...).run()` runs it.
- [ ] **Step 7b (P-37):** every `consolidation.md` group whose members Phase 4 ported is closed, with its parity
  evidence `<sha> <job id>` per member; the open-group count is below Task 1b's; every remaining open group names a
  Phase 5 task.
- [ ] **Step 8:** Task 1 again (P-39); every pydra follow-up below is merged into `integration/senselab-r2` or carried
  by a register row naming its owner.
- [ ] **Step 9:** evidence in `reviews/phase4-exit-evidence.md` on `2x/phase-4-spec`, one line per step
  `<sha> <job id> <summary>`.

### Task 32: Phase 4 docs and spec pass

**Model:** sonnet implements; opus reviews. **Tier:** contract (docs build). **Depends on:** Task 31.

- [ ] **Step 1 (pages):** reference pages for pipelines, `graph@2.0`, datasets and dataset profiles, `sl.signal`,
  generated from the code; example pages on public data (explore_conversation on LibriSpeech, the speech-PII clip, the
  accelerometry spectrum); the status page gains pipeline, dataset-profile and signal rows; README states what 2.x now
  delivers.
- [ ] **Step 2 (spec):** design §4.5, §4.6, §4.11, §4.15, §6.2 (Phase 4 done), §6.4; ADR-001, 005, 006, 010 amendments
  dated; `testing.md` (usecase tier populated); `execution.md` (batched calls, workers per device).
- [ ] **Step 3 (spec):** the dated review of the triage decision layer against `roadmap.md` (ADR-009, P-36):
  `reviews/phase4-flywheel-review.md`, one line per decision-layer module: still needed by a running 1.x campaign,
  or superseded; owner sign-off requested.

### Task 33: Phase 4 exit adversarial round (P-35)

**Model:** opus lead, effort high, with up to 5 sonnet subagents; claims verified by haiku (P-40). **Tier:**
read-only. **Depends on:** Task 32.

- [ ] **Step 1:** brief the lead with the Phase 4 surfaces: `Absent` and the reuse rule, `graph@2.0` and `validate`,
  the builder, the compiler (nesting, partial splits, groups, routes, loops, absence), resume of routed and looped
  runs, the loader and its layering, dataset profiles and unfitted values, the coupling check, the adapters, the series
  reader, `sl.signal`, every ported backend and pipeline, the env grouping, every C-n merge. Goal: inputs, races and
  misuse that break ADR-001, 002, 005, 006, 009, 010 or the Phase 4 exit; plus the P-37 duplicate pass over
  `packages/` (any two tasks that differ only by configuration is a finding), and a check that no merged task's
  default differs from a member's without that member's row saying so.
- [ ] **Step 2:** findings become register rows from the next free id, lens `EXT4`; saved under
  `reviews/phase4-exit-adversarial/`; commit on `2x/phase-4-spec`.

### Task 34: Phase 4 exit fix round

**Model:** per finding from the roster; each fix reviewed by a different model. **Depends on:** Task 33.

- [ ] **Step 1:** fix blocking and important findings by an architectural change (P-35); nits to the exit register.
- [ ] **Step 2:** CI green; rerun the affected Task 31 steps at the new head.
- [ ] **Step 3:** `STATE.md`: Phase 4 done, head SHA, evidence by SHA; merge `2x/phase-4` into `2.x`.

---

## pydra follow-ups

Each is a draft PR onto `integration/senselab-r2` in `$PYDRA_EXT`, with tests on `debug` and `cf` and on Slurm where
the area touches it, reviewed against senselab's principles; none blocks a senselab task unless named.

| Id | What senselab needs | Raised by | Blocks |
|---|---|---|---|
| F1 | A store hook so an output a reuse rule will never serve is not written (environment-class `Absent`) | Task 2 | nothing (senselab's reuse rule already refuses it) |
| F2 | A public claim-lease heartbeat reader (R-319) | carried | nothing |
| F3 | `Job.context` memoised per task class and environment (R-325) | carried | nothing |
| F4 | Look up a job's entry before loading its provider (RFC 0005 §4.3) | carried | nothing |
| F5 | A resource fault at worker start or load is not bisected (R-257), if Task 0b's test fails | Task 0b | Task 0b's xfail |
| F6 | A mixed content-and-resource request re-places only the resource items (R-258), if Task 0b's test fails | Task 0b | Task 0b's xfail |
| F7 | A pool worker woken when its queue gains work (R-233), if Task 0 Step 5 measures the idle-poll wait | Task 0 | nothing |
| F8 | Pool worker memory bounded per item (R-235), if Task 0 Step 5 measures growth | Task 0 | nothing |
| F9 | `MeasuredFit`'s item cost as `a0 + a1·jobs` (R-264), if Task 0 Step 5 measures a slope | Task 0 | nothing |
| F10 | `SlurmSubstrate` reports a queued worker's expected start (R-267) | register | nothing |

## Consolidation groups (initial, for Task 1b)

Candidates from a scan of `src/senselab/audio/tasks/*/api.py`, the two workflow trees and the 2.x registry; Task 1b
verifies each with file:line, adds what the full scan finds and records "distinct" with a reason where a candidate is
not a variant. Identity: every replacement is a new task name or version, so every key of every member changes and no
old entry is served (P-14); a group's row says which store entries that invalidates.

| Id | Members | One replacement (typed parameters) | Merge task |
|---|---|---|---|
| C-1 | 2.x `speaker_embeddings` (and its `windowing.py`); 1.x `ssl_embeddings` | `audio.embeddings` (`layers`, `pool`, window); windows from `signal.windows` | 14 |
| C-2 | 1.x `envelope` (`analytic_envelope`, `hilbert_envelope_dbfs`, `envelope_dbfs`, `analytic_magnitude`) | `signal.envelope` (`method`, units) | 11 |
| C-3 | 1.x `band_profile` (`bin_power`, `long_term_average_spectrum`, `rolloff_hz`), `gammatone`, `spectral_continuity` | `signal.spectrum` (`representation: welch \| ltas \| gammatone`, measures) | 11 |
| C-4 | 1.x `spans` (`propose_spans`, `segments_between_change_points`), `disruptions`, `clipping` | `signal.activity`, `signal.spans`, `signal.qc` | 12a |
| C-5 | 1.x `features_extraction` (praat, opensmile), `phonation` tracks; triage `voice_phonation` measures | `audio.voice_features` (`feature_set`, `features`, `output`) | 13 |
| C-6 | 1.x `classification` (AST, YAMNet, emotion models); triage `label_membership`, `classifier_ontology` | `audio.classify` (`model`, `top_k`, `floor`, `ontology`) | 16 |
| C-7 | per-model ASR, diarization and alignment entry points of 1.x (NeMo, Qwen, MMS) | backends of `speech_to_text`, `diarize`, `audio.align` | 18, 19 |
| C-8 | 1.x `fuse_word_streams`; audio_analysis word fusion; triage `consensus`, `word_arbitration` | `audio.word_consensus` | 20a |
| C-9 | 1.x `global_floor_dbfs`; audio_analysis `noise_floor`, `floors`; triage `background_model`, `session_background` | `audio.background_floor` (`scope`, `exclude`) | 20b |
| C-10 | 1.x `estimate_speaker_embedding_from_audios`, `speaker_verification`, `centred_cosine_similarity`; audio_analysis `speaker_identity`; triage `enrollment` | `audio.speaker_enrollment`, `audio.speaker_similarity` (`measure`) | 20c |
| C-11 | 1.x `voice_activity_detection`; audio_analysis speech-presence evidence; energy VAD | `audio.voice_activity` (`strategy`) | 19b |
| C-12 | 1.x `redaction`, `pii_detection.detect_pii_in_audios`; triage `nodes/redact` | `audio.redact` (and the `speech_pii` pipeline) | 29 |
| C-13 | 1.x `preprocessing` (resample, downmix, filter, normalise) | `signal.resample`, `signal.channels`, `signal.filter`, `signal.rescale` | 11 |
| — | 2.x metrics `wer`, `der`, `eer`, `span_f1`, `resample_error` | checked by Task 1b; expected distinct (different algorithms, not configurations) | — |

## Phase 3 register rows, rechecked against the new pydra

| Row | Finding (short) | Phase 4 disposition |
|---|---|---|
| R-233 | 10² per-item figure is the deleted pool worker's idle poll | The code is gone (route-B S1b). Task 0 Step 5 measures the warm pool; closed, or F7 |
| R-235 | Pool worker memory grows per item | The code is gone. Task 0 Step 5 measures the warm pool worker; closed, or F8 |
| R-255 | Theil-Sen slope arbitrary on near-equal lengths (`runtime/probe.py`) | Defer with owner to R-303 (calibration cleanup): placement now uses pydra's `MeasuredFit`, so the probe's estimate may be deleted rather than fixed |
| R-257 | Pool bisects a load-time resource fault | Bisection is pydra's; it becomes live with batched calls. Task 0b tests it; closed, or F5 |
| R-258 | Mixed content and OOM request split by neither executor | As R-257: Task 0b states the rule in `execution.md` and tests it; closed, or F6 |
| R-264 | Item cost grows per job | The measured slope was in the deleted pool. Task 0 Step 5 re-measures; closed, or F9; R-304 keeps the GPU half |
| R-267 | No expected start for a queued worker | Defer with owner: F10 in pydra, then `senselab status` shows it in Phase 5 (surfaces) |
| R-268 | `pii_spans` transformers dequantises to 58 GiB | Task 29 (the cross-modal pipeline runs `pii_spans` on H100) |
| R-270 | Float output identities differ across CPU architectures | Defer with owner: recommended close under N-8 (no measured classes ship; a group seeds `numpy-sinc` evidence with `verify` sampling), with a docs line in Task 32 |
| R-271 | `bounds.pool-slurm` has no wall bound | Task 31: the exit harness takes each run's bound from `Plan.explain()`'s expected work times the profile's margin, and documents cancel-by-prefix |

Also scheduled: R-281 → Task 9; R-282 → Task 7; R-305 → Task 5; R-306 → Task 31 Step 6; R-320 → Task 7; R-327,
R-328 → Task 0b; R-113 → alpha PR #643 (Task 31 Step 1). R-303, R-304 stay "after the Phase 3 exit"; R-297 stays
Phase 5; R-265 is resolved by the `usecase` tests of Tasks 15, 22, 23, 29.

## Open 1.x work (alpha, 2026-11-28)

`origin/alpha` is `c4bb902b`, the parity reference (P-39). Open PRs into `alpha` and the stack onto it; "moves the
reference" means its merge changes `src/` behaviour that a Phase 4 parity step compares against.

| PR | What | Phase 4 handling | Phase 4 waits? | Moves the reference |
|---|---|---|---|---|
| #607-#642 (15 stacked: #607 design, #608, #609, #612, #613, #618, #627, #628, #633, #634, #635, #637, #638, #639, #642) | triage dataset profiles: identity once, profiles supply B2AI data, tasks as data, method vs numbers, B2AI modules to b2aiprep (#639), drivers into the package (#642) | Ported: the profile shape and task protocol (Task 24), the measurement tasks (Tasks 25-27), the measurement pipeline (Task 28). The decision layer stays on 1.x. The B2AI hint builder and registry leave senselab (#639), so the `b2ai-voice` profile is read from where owner decision 1 puts it | Yes: Tasks 24-28 start after alpha A (#607-#638) merges, and after alpha B (#639, #642) if owner decision 2 says so | Yes, heavily (triage, scripts) |
| #660 | `resample_audios` computes in float64; results record their environment | Already carried: `signal.resample` computes in float64 and 2.x records the environment by design. Task 1 adds the ledger line | No | Yes (`audio/tasks/preprocessing`, `utils/provenance`) |
| #649 | `ssl_embeddings` uses the encoder of encoder-decoder models | Ported in Task 14 | Task 14 waits for its merge or close, or ports the PR head and Task 1 reconciles on merge | Yes (`audio/tasks/ssl_embeddings`) |
| #641 | ppgs venv pins `espnet<202609` | Left on 1.x: PPGs are not a Phase 4 task (triage's `extend_ppg_praat` driver is not ported); Phase 5 ports ppgs with the pin | No | Yes (venv lock only; PPG outputs bit-identical per the PR) |
| #646, #636, #625 | video `ssl_embeddings`, `event_localization`, `standardize_encoding` | Left on 1.x; Phase 5 ports video tasks. Task 29's `demux` does not depend on #625 | No | Yes (`video/tasks`) |
| #645 | evaluation: grouped splits, bootstrap, Holm, linear probe; an `events` data structure | Left on 1.x for Phase 4. The `events` structure is superseded by the kernel's `Events`/`Intervals`; evaluation belongs with benchmarks or a Phase 5 task | No | Yes (`utils/evaluation`, new) |
| #643 | schedules the 2.x nightly from the default branch | Not a port; closes R-113 | Task 31 Step 1 needs it merged | No (`.github` only) |
| #640 | CLAUDE.md cache bullet | None (1.x guidance) | No | No |
| #622, #623 | dependabot bumps of the 1.x lock | None: the 2.x line keeps its own `uv.lock` in Task 1's merge | No | Possibly (1.x dependency versions); Task 1 notes any parity step affected |

PRs #524, #236 and #207 target other branches (`main`, a feature branch) and are out of Phase 4's reference.

## Self-review notes

- **Spec coverage, design §6.2 Phase 4.** `Pipeline` with ports, nesting, routing, loops, `Absent` → Tasks 2-6. Full
  dataset adapters and profiles → 8, 9 (and 24 for B2AI). Merged config loader → 7. Generic `sl.signal` tier → 10-12b.
  Env grouping adoption → 30. Port audio_analysis → 16-23; explore_conversation and health_measurements → 13-15. One
  cross-modal pipeline → 29. Triage measurement tasks, B2AI coupling in a `b2ai-voice` profile → 24-28; decision layer
  stays on 1.x and gets its dated review → 32 Step 3.
- **Exit criteria, verbatim.** Two dataset profiles changing only the profile, synthetic with another entity scheme and
  protocol, unfitted → typed `Absent`, every node runs or typed-absent, static check → 31 Step 2 (built by 8, 24, 28).
  Pipelines nest unchanged → 5, 22, 31 Step 3. One generic task on audio and accelerometry → 11, 31 Step 4.
- **Brief facts.** Task 0 = #662 and R-327/R-328 (0, 0b). The 10 P3 rows → the table above. pydra features used, not
  rebuilt: conditions → 6; `Absent` modes → 2, 6; `workflow.loop` → 6, 23; `plan()` → 5; split over upstream outputs
  → 5; `max_concurrent` stays unused (the profile's worker classes bound pool work); warm pools, batch probe and
  `workers_per_device` → 0b.
- **Consolidation (owner request 2026-11-28).** Inventory early → Task 1b (before every port); each group's merge is
  one task (§ Consolidation groups, last column), each with per-member parity and declared defaults (Review Focus 6);
  identity consequences in each row; P-37 exit progress → Task 31 Step 7b; adversarial duplicate pass → Task 33.
- **Type consistency.** `Absent`, `AbsentClass`, `AbsentMode` (Task 2); `Node.kind`, `over`, `input_absent` (3);
  `Pipeline`, `Composable`, `Port`, `NodeRef` (4); `Plan.groups` (5); `ProfileKind`, `ProfileRef`, `LoadedProfile`,
  `load_profile` (7); `DatasetProfile`, `TaskProtocol`, `EntitySpec` (8); `IgnoredReason`, `Dataset.groups`,
  `Dataset.layers` (9); `Accelerometry`, `SeriesLayout`, `open_series` (10); `Video` (29). Later tasks use these names.
- **Proportion.** Port tasks (13-29) name the task, ports, backends and parity test and leave the bodies to the moved
  1.x code; core tasks (2-12b) carry the interfaces.

## Rulings made in this plan

The planner's choices where the specs leave one open; the owner reviews them. A ruling that changes an ADR takes effect
only through its task's dated amendment (ADR-000).

- **P4-1** The `Pipeline` builder is kernel code: it only builds a `Graph`, so `core-only` can build and validate
  pipelines without the runtime.
- **P4-2** Nested pipelines are inlined in `graph@2.0`, not referenced by name, so a document is lossless and its digest
  covers every level.
- **P4-3** A route is `then`/`otherwise` on one `bool` port; a many-way choice is nested routes. It compiles to
  `when=`/`unless=` and `select(2)`.
- **P4-4** pydra's `not-routed` is content-class absence: a recorded predicate decides it, so it is reused on resume.
- **P4-5** Environment-class absence is refused by senselab's reuse rule; that it is still written is pydra follow-up F1.
- **P4-6** An unfitted dataset value makes the node `Absent(content, unfitted)` at plan time, so no job is formed and
  `explain` shows it.
- **P4-7** One loader for the three profile kinds; `ExecutionProfile` and `RunConfigFile` move onto it (P-37).
- **P4-8** Packaged pipelines are builder modules (`build(config) -> Pipeline`); their `graph@2.0` JSON is generated
  for docs and `sl.describe`, not hand-kept.
- **P4-9** Duplicates and configuration variants are merged where they are ported, one reviewable task per group
  (Task 1b's inventory); a port never lands a member as its own task first. A group's members may be retired in one
  merge although some members' pipelines port later; their pipelines call the replacement.
- **P4-10** Task 0 stays in Phase 4 (the brief's default): Phase 4 adds about a dozen GPU backends, and batching must
  be measured before they are.
- **P4-11** Triage tasks come last in the phase, after the alpha stack lands, so they are ported once from the reference.
- **P4-12** audio_analysis plots, the Label Studio export and the review page are not ported in Phase 4 (owner answer:
  review-page tooling stays outside the package).

## Owner decisions needed

1. Where the `b2ai-voice` dataset profile lives now that #639 moves B2AI data to b2aiprep: a user file supplied by
   b2aiprep (recommended; senselab ships only the neutral fixture), or a packaged senselab profile.
2. Whether Tasks 24-28 wait for alpha B (#639, #642) or only alpha A (#607-#638).
3. Env grouping: adopt in Phase 4 as P-24's check says (recommended, measured in Task 30), or keep design §6.4's
   "one env per backend until measured as a problem".
4. audio_analysis backend set: every 1.x default model (Sortformer, Canary-Qwen, Qwen3-ASR, Qwen3 aligner, MMS, AST,
   YAMNet, SepFormer; recommended, for parity), or a minimal set with the rest in Phase 5.
5. The cross-modal pipeline: video → audio → text PII (recommended; it exercises P-24's check) or audio + text only.
6. Time-series foundation-model embeddings (design §4.15 tier 1 "embeddings") deferred to Phase 5.
7. R-270 closed under N-8; R-255 folded into R-303; R-267 deferred to Phase 5 with pydra F10.
8. C-1: the shipped 2.x `speaker_embeddings` task becomes `audio.embeddings` with SSL embeddings (all its keys change;
   recommended), or speaker and SSL embeddings stay two tasks.
