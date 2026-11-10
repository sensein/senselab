# Phase 3 exit evidence (Task 21)

Every line below is a job id, a commit, a verdict line or a file. Job ids are the cluster scheduler's. Commits are
short SHAs of `2x/phase-3` (merge heads) unless a branch is named. "The group site profile" is the group's own
execution-profile file, copied to each job and never committed; "the group's high-priority partition" (4 h job cap,
per-user QOS limits of 4 submitted jobs, 1 GPU, 1 node) and "a preemptable partition" (256 jobs, 32 H100) are the two
partitions the jobs ran on. Result directories: `results/t21-<step>-<job>/` and `t21-<step>-<job>.out` under the
run's scratch directory.
Files named `task-21-*.md`, `progress.md` and `tmp/...` are controller notes outside the repository; what is
load-bearing in them is inlined here.

## Verdict table

| Step | Verdict | Job / commit | Verdict line or key number |
|---|---|---|---|
| 1 CI | PARTIAL: `2x-tests` green, nightly unmet | CI 42764434206 at 32a47d2c; 42767644065 at e134f625 (merged fd513f0c); 42772697382 (merged b6385e91); 42779865067 (merged 5990b4f5); 42808716059 (push at e8ae516b; 42806809412 ran the same tree at f7729038) | Green except `pr-models`, skipped in every run (it runs on `pull_request` events; Phase 3 merges are pushes). No nightly workflow exists (`gh workflow list`): carried to Task 22, R-113 reopened. |
| 2 P-10 | PASS on the amended criterion | laptop at 32a47d2c; ORCD 28565606 (fd513f0c) | one plan hash `bd57268d...`, 2 diffs, both in `placement`; 1000 items on every executor; outputs agree per item under the run-config `samples` threshold (atol 1e-6, rtol 0): measured max 3.678e-18; 981 of 1000 identities equal across architectures, 19 differ |
| 3 P-10 model | PASS 6/6 | 28578691 (ab62fd60) | one hash `97b8bdc5...`, 5 diffs all in placement; 1000/1000 committed in 1329 s; placed on `gpu:h100` |
| 4 P-5 | PASS on events (28580776/7 at ab62fd60); provenance gaps found and fixed (T-POOL-PROV c461f2b0, T-RESUME-REUSE 1f2ca81f); final chain 28604079/28604080 at 1f2ca81f, provenance checker 11/11 on the retrieved store, job rerun pending | start 28580776, resume 28580777 (ab62fd60) | interrupted at 421 committed (target 334); resume: executions 480 + late hits 97 = 577 = 1000 - 423; reuses 520; same hash `97b8bdc5...` |
| 5 suite | PASS (usecase tier empty; cpu at ab62fd60) | cpu 28578692 (ab62fd60); gpu 28584535 (e8ae516b); cross-node 28564914 (fd513f0c) | cpu 6/6 incl. env builds of every lock; gpu 6/6; cross-node 6/6 |
| 6 P-1 | PASS | 28582061 (ab62fd60) | the fresh agent's call planned and ran; 20/20 committed in 218 s |
| 7 pydra, P-39 | PASS (K2 merged; 41-53 carried) | no job | gate PASS Route A at pydra 0c1bbcd9; K2 = #905 merged; Tasks 41-53 carried by R-238..R-250 |
| R-253 | closed with a measurement | 28564898 (fd513f0c) | slope 7.29 +- 0.46 ms per job; see below |

## Step 1 (CI)

`2x-tests` green on the merge heads named above (progress.md 2026-11-10; the opus review checked the push run
42808716059 at the head e8ae516b and 42806809412 at f7729038, same tree). `pr-models` is skipped in every Task 21
run: it runs only on `pull_request` events and the Phase 3 merges are pushes; it first runs on the Phase 3 to `2.x`
pull request. The nightly on `2.x` is unmet: `gh workflow list` shows no nightly workflow in the repository. R-113 is
recorded closed (Phase 2 Task 19, `2x-nightly.yaml` "on alpha via a skip-release PR") but the workflow never landed,
and Phase 2 had the same clause. The nightly is the only job that runs the 3.11 to 3.14 matrix (`workflow_dispatch`
has no `ref` input, so a dispatch runs 3.13 only), so that matrix never ran in Phase 3. Ruling: the nightly clause is
carried to Task 22 (owner), R-113 is reopened, and one full-matrix run (3.11 to 3.14) at the head, for example a
`full-matrix`-labelled pull request, is required before the merge into `2.x`. Step 1 is therefore not ticked.

## Step 2 (P-10)

Laptop part, commit 32a47d2c, 10-CPU 32 GB machine, bound 900 s and 2 GB, load and memory sampled every 30 s
(nothing high, no run stopped; `task-21-report.md`):

| executor | wall | peak resident | done / failed |
|---|---|---|---|
| InProcess | 291 s | 491 MB | 1000 / 0 |
| Pool on `local`, 2 workers | 80.7 s (repeat 79.7 s) | 687 MB (repeat 670 MB) | 1000 / 0 |

The plan under `laptop` and under the group site profile has one `LogicalPlan` hash,
`bd57268db8863bdb199c228228f7fc28c69798a240c6c7294b24405ffe30e9af`; the field diff has two differences, both inside
`placement` (`placement.nodes.call.partition`, `placement.profile_digest`). Every run's hash equals the plan hash;
the three laptop identity files agree (1000/1000).

ORCD run 28565606 (Pool on `slurm`; coordinator on the group's high-priority partition, workers on the preemptable
one): `progress: 1000/1000 items committed after 535 s`; checks 7, failed 1. The six passing checks: plan one hash
and placement-only diff, identities written, every item done, none failed, an identity per item, run hash equals the
plan's and the expected hash. The failing check, as printed: `FAIL output identities equal the laptop's` (sub-0020:
`output:8f99ef97... != output:b68aee4d...`; the verdict line prints the two sides in the opposite order to the stored
files, laptop 8f99ef97, ORCD b68aee4d).

Earlier attempts, for the record: 28564896 (and 28564897 for Step 3) failed at about 5 minutes, every worker sbatch
refused with "Job violates accounting/QOS policy" because the four coordinators filled the group's high-priority
partition's per-user submit limit of 4; the product behaved correctly (the class was broken after 7 starts and the
message names the profile values). Resubmitted as 28565606 / 28565607.

### Ruling that changed the criterion (cross-architecture numerics)

Diagnosis (controller note `task-21-step2-identity-diag.md`, read-only; its load-bearing figures are all here): 19 of
1000 output identities differ between the laptop (arm64) and the cluster (x86-64): sub-0020, 0022, 0064, 0188, 0206,
0229, 0270, 0317, 0402, 0556, 0588, 0655, 0668, 0685, 0711, 0714, 0784, 0787, 0843. The input tree is byte-identical on
both hosts (one sha256 over the sorted per-file digests, `66e59c53...b55f`). The `numpy-sinc` resampler sums in float64
with `numpy.einsum`; zero-crossing samples hold rounding residue whose value depends on summation order and fused
multiply-add. All 19 arrays are float32 of equal shape; every sample above 1e-6 in magnitude is bit-equal; the largest
absolute difference is 3.678e-18 against a signal peak of about 0.24 to 0.26; differing samples per item run from 333
to 7535. The output identity hashes the array bytes only (`output@3`); context (platform, wheel set) is stored beside
the result and is not hashed (ADR-002 rule 2). Not measured: the cluster's numpy build and SIMD dispatch (no
interpreter was run on a login node), so the FMA / SIMD explanation is inferred; why only 19 items carry such residue
was not enumerated.

Where the wording came from. "Equal output identities on both hosts" was in the plan itself: `plans/phase-3.md` Task 21
Step 2, citing P3-45. It contradicted the design: design.md §4.8 (lines 384-385) lists the numerics class (device,
architecture, precision) under "Execution context, recorded per result and not keyed", and ADR-002 rule 2 and I-4 say
the same, with the output identity a byte hash. Equal identities across arm64 and x86-64 cannot be required of a float
kernel without making numerics part of the key. The plan is amended in the spec patch for that reason.

Ruling (2026-11-10, controller): Step 2's criterion is one plan hash, a placement-only diff, and per-item output
agreement under `runtime.compare` with the run-config thresholds: sample data compares `close: samples`, atol 1e-6,
rtol 0 (ADR-002 amendment 2026-11-02, P2-95; P3-46 is the rule that Phase 3 uses those comparisons and not tolerance
profiles). Output identities are hashes and are compared for equality only, which holds within one numerics class; their
cross-architecture differences are recorded as a measurement. No product change: hashing rounded floats would put an
invented tolerance into the identity. Step 2 PASS on that criterion: the measured maximum absolute difference is
3.678e-18, against an atol of 1e-6, and every sample above 1e-6 is bit-equal, in all 19 differing items. The check was
the direct per-sample difference; `runtime.compare` itself was not run on the 19 pairs.

Consequence recorded (review N-10): a downstream step's keys follow input identities, so 19 of 1000 items would miss
the cache across architectures (there is no early cutoff on equal output within a threshold); the measurement could
seed `numpy-sinc` equivalence evidence (verify sampling, ADR-002 amendment 2026-11-02) instead of staying a note (R-270).

## Step 3 (P-10 with a model)

Job 28578691 at ab62fd60 (one H100 worker, see the placement ruling below): `stt: one plan hash 97b8bdc5...47 under
laptop, <site profile>; 5 differences, all in placement`; `progress: 1000/1000 items committed after 1329 s`; six
checks PASS (plan, identities written, every item done, identity per item, run hash equals the plan's, `PASS placed on
gpu:h100`). Verdict line: `{"pass": true, "failed": [], "checks": 6, "hash": "97b8bdc549994632929072d206841a2dc282070ad1bc6235057f3eaa43bbff47", "wall_s": 1329.19}`.

### Ruling that changed the placement (serialised GPU steps)

The first Step 3, 4 and 6 runs on the preemptable partition stalled: their H100 workers sat in the scheduler's
Priority state with estimated starts 11-11 20:00 to 11-12 06:00, nothing committed, and the Step 6 run 28570301 ended
TIMEOUT at 2 h with 0/20. Ruling: cancel 28565607, 28572583, 28573230 and their workers; serialise the GPU steps through
the group's high-priority partition (QOS: 1 GPU, 4 jobs per user) with a copy of the site profile whose every GPU class
has `max_workers` 1, coordinators `-t 04:00` to cover the worker's queue wait. This changes the number of workers (one),
not the plan (the hash is placement-independent). Step 3 therefore ran with one H100 worker, not a pool of several.

## Step 4 (P-5)

Start 28580776 (ab62fd60): checks 3/3: wrote its report, one run hash, interrupted at or past a third. Interrupt target
334; 421 committed at the interrupt; hash `97b8bdc5...`. Progress lines show the pace (0 committed for the first 426
s, which is the worker's cold start; 308 at 728 s). The first submission, 28580433/4, was refused at setup (exit 2)
because the cancelled run's directory existed; resubmitted with a fresh directory.

Resume 28580777 (afterok on the start): checks 5/5: accounting `{"committed": 423, "executions": 480, "late": 97,
"reuses": 520, "items": 1000, "hash": "97b8bdc5..."}`. Executions 480 plus late hits 97 equals the 577 items not
committed before the resume (1000 - 423); reuses 520 = 1000 - 480; no fewer committed than at the interrupt; the
resume is of the interrupted run; the driver asserts no ref resolution.

What "committed" means. The 97 "late" items are key groups in the store with no result event before the resume:
results committed between the coordinator's last event and its SIGKILL (commits arrive in bursts: 308 at 728 s, 421 at
the interrupt). The resume refuses to start while any job of the run's prefix is live, so none of the 97 came from a
surviving worker. Committed in the store is 423 + 97 = 520 = reuses, and executions are 480 = 1000 - 520. The plan's
equalities hold when "committed" means committed in the store; counted by result events (423) they would not, because
events lagged the store by 97 items at the SIGKILL.

Amendments. The interrupt was made by the driver inside the start job (cancel the run's scheduler jobs, SIGKILL the
coordinator process, sweep until none is live), not an external `scancel` of a coordinator job; it is at least as
harsh, and the SIGTERM-first path of a real `scancel` is untested. The checks read events and the resolution manifest.
The plan says "events and provenance", and the first Step 4 run (28580776, 28580777) checked events only. The
provenance clause (ADR-010: every store hit is a `senselab:Reuse` activity and every execution an activity) was then
checked post hoc, and the check found two product gaps in turn, both fixed and then confirmed by a final rerun.

1. Provenance was absent before c461f2b0. The run's `prov/` held one writer document of 227 bytes, the plan entity and
   no activity, because Pool workers wrote no provenance document (only the coordinator did, at the end of a resume).
   Found by this check, not by the events; fixed by Task T-POOL-PROV (merged as c461f2b0 on `2x/phase-3`; spec
   c2956454): a Pool worker writes its document into the run's `prov/` after each claim and at serve end, so a SIGKILL
   loses at most the claim in flight and the one just committed (ADR-010 as amended).
2. At c461f2b0 (start 28588819, resume 28588820, check 28589000) provenance was written: 1000 logical keys, one
   execution activity per key, one plan entity, no duplicate. But the resume's accounting reported 460 reuses and the
   union held 177 `senselab:Reuse` activities: the other 283 items were settled by the coordinator from an earlier
   epoch's outcome (`Pool._settle`, which does not go through a store hit), so they had an execution and no Reuse.
   Fixed by Task T-RESUME-REUSE (merged as 1f2ca81f; spec 4ef3fdbf, ADR-010 amendment, R-263): the coordinator records
   a `senselab:Reuse` for each item it settles from an earlier epoch. Two expectations of the first provenance check
   were wrong and are corrected: the resume's `executions` counts `finished` events, not executions (items claimed
   twice in the resume epoch were executed by the first claim and served on the second, which the coordinator logged
   as `cache_hit`), and the invariant is at most one execution activity per key, not exactly one.

Final chain at 1f2ca81f: start 28604079 (372 committed at the interrupt), resume 28604080 (accounting: executions 565,
late 63, reuses 435, committed 372), provenance check 28604081 and its rerun with the corrected checker (below).
Checked locally on the retrieved store with the corrected checker (11 checks, all PASS):
- 3 writer documents: the start worker's (372 activities, all executions), the resume coordinator's (284, all
  `senselab:Reuse`) and the resume worker's (777: 626 executions and 151 `senselab:Reuse`);
- 1000 distinct logical keys, each with an execution or a Reuse; one plan entity;
- `senselab:Reuse` = 435 = the resume's reuses (284 by the coordinator's writer, 151 by the worker);
- 998 execution activities, no key executed twice; the 2 keys without an execution activity each have a Reuse, within
  the loss ADR-010 allows (a kill loses the claim in flight and the one just committed, at most 2 x quantum_items 100
  per cancelled worker, here 1 worker: 200);
- resume-phase executions 626 = items 1000 - committed 372 - late start 0 - the 2 keys without an execution;
- information: 61 items claimed twice within the resume epoch (events). An item executed and committed by one claim was
  claimed again later and served from the store; it cost no recomputation but is why the events undercount executions.

The check is `task-21-orcd-4-prov.sh` over `start.json` and the resume accounting; each writer document is assigned to
a phase as a whole by its activity times against the resume's first event. The job 28604081 ran an earlier version of
the checker (it compared resume-phase executions to the accounting's `executions` and assigned phase by file time);
the corrected checker was verified on the retrieved store and is to be rerun as a job (pending). Events verdicts of
the final chain: start 28604079 PASS 3/3 (372 committed, target 334 by the one-third rule), resume 28604080 PASS 5/5.

## Step 5 (suite, env builds, cross-node)

| job | commit | tiers |
|---|---|---|
| cpu 28578692 | ab62fd60 | logic 3780 tests, contract 3022, limits 64, env 59, 0 failures and 0 errors each; env builds PASS for every shipped lock, 5 of them (crisperwhisper, pii-spans, pyannote, speechbrain, vllm-server; "every shipped lock is covered by this driver"); combined 6 checks PASS |
| gpu 28584535 | e8ae516b | logic 3780, contract 3022, limits 64 (4 skipped), env 59 (0 skipped), 0 failures; envs-verdict PASS; combined 6 checks PASS |
| cross-node 28564914 | fd513f0c (progress.md) | lease probe over 3 hosts: held 360 s, takeover after kill 120 s, 0 overlapping holds, 0 unreleased takes, 0 lost, pass; queue probe: 10000 items, 10000 done, 0 dead, 0 committed twice or more, 1 duplicate commit event, 0 lost claims, 3 distinct hosts, pass |

Notes. The CPU suite was not rerun at e8ae516b. `git diff --stat fd513f0c e8ae516b` touches two test files only
(`substrates_test.py`, `transformers_backend_test.py`) and `ab62fd60..e8ae516b` only the second, so every ORCD step ran
the product code that is at the head; the second file holds the Whisper real-weights fixture, whose tests skip on a CPU
host, so a CPU run at the head would execute the same code. The gpu job ran every tier at e8ae516b, including logic and
contract, so the full suite did run at that head. Skips (both hosts): contract 70 (generic-variation and no-lock-records
skips); logic 3, two of them the `nfs` parametrisation of `locks_test.py`, skipped because `SENSELAB_WORK_POOL_NFS_DIR`
was unset on the one host that could run it (the cross-node probe covers the semantics; set it in the next Step 5
script); limits 3 parity tests (campaign directory unset). The usecase tier is empty: no
test carries `@pytest.mark.usecase` (the marker is declared and has a budget row), so the script reports an empty
tier and leaves it out of the combined verdict. Nothing is substantiated about a usecase test.

Failures on the way, all fixed on branches merged before the passing runs:
- 28564899 (cpu at fd513f0c): 4 failures. Job venv lacked the test group (pandas, pyld), 3 tests; the substrates test
  chmod'd its fake `sbatch` non-executable while the host PATH was searched, so the cluster's real `sbatch` ran against
  the synthetic fixture partition. Fix 72df2815, merged b6385e91.
- 28569745 (cpu) and 28569747 (gpu) at b6385e91: the env-builds step crashed in both (`'dict' object has no attribute
  'isolated'`) and the combined verdict globbed only existing parts, so the earlier "PASS 5/5" did not cover the env
  builds (progress.md CORRECTION). The driver and script were fixed; the passing runs above include `envs-verdict.json`.
  The gpu run also failed a logic test (a GPU allocation exports a device variable that the test expected unset), fixed
  by 5990b4f5.
- 28582376 (gpu, ab62fd60): `pii_spans_transformers` real-weights test CUDA OOM. Root cause: an earlier test in the
  same pytest process loaded Whisper on the card and never released it (10.79 GiB reserved, identical to the digit in
  two runs); the pii worker, which dequantises the w4a16 checkpoint to about 58 GiB on a 79.18 GiB card, then had 336
  MiB free. A test-fixture lifetime defect; fix f7729038 (frees the state, `gc.collect`, `empty_cache`), merged e8ae516b.
  The compute-app listing prints nothing inside the job's container, so the holder is established from the OOM message
  and the test order, not from a listing.
- Cross-node 28564914 on the preemptable partition because the job needs 3 distinct nodes and the high-priority
  partition has 2.

## Step 6 (P-1)

A fresh sonnet agent with no repository, web or command access received one file: the instruction text, the dataset
path and the `describe()` output of the 10 tasks (`task-21-step6-agent-prompt.md`). Its literal answer
(`tmp/t21-step6/call.json`, echoed in the verdict line of 28582061):

```json
{"dataset": "<the job's synthetic 20-tone dataset path>", "format": "agent-plan@1.0",
 "model": "pyannote/speaker-diarization-community-1@3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
 "params": {}, "task": "diarize"}
```

(`<...>` stands for the scratch path the verdict line prints.) The call validated, `sl.plan(...)` planned it and
`.run()` finished: `progress: 20/20 items committed after 218 s`; verdict `{"pass": true, "failed": [], "checks": 1}`.
Earlier run 28570301 timed out at 2 h with 0/20 for the placement reason under Step 3. P-1 holds for this agent and
this task; it is one agent and one task, not a statistic.

## Step 7

Task 0 gate: PASS Route A at pydra `0c1bbcd9`; it did not fail, so there is no re-evaluation. pydra
`enh/execution-extensions` head `0e3bbbbd` (#912); PRs #903-#912 into it are all merged. Task 40 (K2) merged as #905
`0c1bbcd9`. Tasks 41-53 have no PR; each is carried by a register row naming owner ("senselab pydra track") and reason
(none blocks a senselab task; Route A uses senselab's own queues, P3-52): 41 R-238, 42 R-239, 43 R-240, 44 R-241,
45 R-242 (R-210 is the senselab-side consumer row), 46 R-243, 47 R-244, 48 R-245, 49 R-246, 50 R-247, 51 R-248,
52 R-249, 53 R-250. P-39: `origin/alpha` is `c4bb902b`, equal to the ledger mark; the ledger line of 2026-11-08 records
`3300d8c9` and `c4bb902b` as reconciled; the "mark c4bb902b" at Task 21 Step 7 is the P-39 reconciliation mark, not a
merge into `2x/phase-3`, so no merge is owed (progress.md ledger correction). Not substantiated here: the pydra PR
states were read on 2026-11-10 by the earlier leg (`task-21-report.md`) and not re-queried in this close-out.

## R-253 measurement: the per-item cost `a` against jobs per item

Job 28564898 at fd513f0c; `signal.resample` on a CPU node; the point probed once per jobs-per-item J, four
repetitions, plus `pool.quantum_items` 4 and 256 at J 2. Verdict line: `{"pass": false, "failed": ["every single-item
size measured"], "checks": 53, "slope_s_per_job": 0.007290121055228376, "slope_stderr": 0.0004643104200750078}`. The
job's own check failed because J 64 had no measured items (each J 64 repetition measured only 1-job items); every
other check passed.

| jobs per item | n | median `a` (s) |
|---|---|---|
| 1 | 286 | 0.09837 |
| 2 | 18 | 0.10531 |
| 4 | 12 | 0.12154 |
| 8 | 12 | 0.15197 |
| 16 | 12 | 0.22686 |
| 32 | 12 | 0.34187 |
| 64 | 0 | not measured |

Slope of `a` on jobs: 7.29 +- 0.46 ms per job (least squares). Ruling: R-253 closed with this result; `a` varies with
an item's job count, so item cost is not job-count-free. `single_item_jobs` 2 stays as the probe's minimum. The
model `a0 + a1 * jobs` is R-264, after the Phase 3 exit. The coverage gap at J 64 is recorded, not repaired.
The local laptop smoke earlier read about 4 ms per job from 3 items per J; it was a preliminary indication, not the
result.

## Other findings carried to the register

- The coordinators count against the group's per-user submit limit, which no placement request sees: the first Step 2
  and 3 runs were refused (28564896/7) and the GPU steps had to be serialised. `calibration-probe.md` §10 (T20R-12)
  already has the hook's answer and `SiteRefused` on a QOS refusal; what is new is the coordinators' own count, and the
  site-profile docs must say senselab does not read QOS (R-266).
- A coordinator waits on a queued worker without a start estimate (the Step 4/6 stalls above); §10.3's placement answer
  has no expected-start field, §10.5 puts start estimates in the hook (R-267).
- The usecase tier is declared with a budget but no test carries it; use-case tests are Phase 5's (R-265).
- `pii_spans` on the transformers backend dequantises the w4a16 checkpoint to about 58 GiB on an 80 GiB card; the vllm
  backend serves it at about 20 GiB (R-268). The 58 GiB is from the OOM message (58.09 GiB allocated, 9.35 reserved);
  the 20 GiB figure is from T20R-10 (`reviews/t20r10-evidence.md`) and the fix report, not measured in Task 21.
- The setup line in the 2.x `CLAUDE.md` omits `--group test` and `--group benchmarks` (Task 22 docs item, R-269).
- `exit.yaml` `progress_s` (60) was added with the progress printing (e134f625); its derivation is owed to
  `config-derivations.md` (done in the spec patch).
- The preflight still says "equal output identities"; the plan's Step 2 is amended in the spec patch (Task 22 fixes the preflight).
- `bounds.pool-slurm` has no wall bound; a stuck pool run ends only at the coordinator's `--time` and its workers run
  to their own limit (task-21-report.md concern 3). Registered as R-271.

## Commits (verified by the opus review with `git rev-parse` on each job's checkout)

Step 2 28565606 and cross-node 28564914: fd513f0c. Steps 3 (28578691), 4 (28580776, 28580777), 6 (28582061) and Step 5
cpu (28578692): ab62fd60. Step 5 gpu (28584535): e8ae516b. R-253 (28564898): fd513f0c. `fd513f0c..e8ae516b` changes two
test files only.

## Could not be substantiated

- Per-job names of each CI run (see Step 1).
- The architecture-numerics mechanism (FMA / SIMD) is inferred, not measured, and `runtime.compare` was not run on the
  19 pairs.
- Who held GPU memory in 28569747: superseded by the fixture explanation above, itself inferred from the OOM message.
- The corrected checker ran as cluster job 28608948 at 1f2ca81f: PASS 11/11 (435 senselab:Reuse = the resume's reuses, 284 by the coordinator and 151 by workers; at most one execution per key; 2 keys with a Reuse and no execution, within the kill allowance; one plan entity).
- Not exercised at the exit: a multi-worker GPU pool, an interrupt with more than one worker, the SIGTERM path of an
  external `scancel`, and the "null for the default" model path of the Step 6 call (the agent named a model).
