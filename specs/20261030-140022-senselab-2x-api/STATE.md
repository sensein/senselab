# senselab 2.x: working state

Read this first when resuming. Update it at the end of every work session, and whenever a phase,
task or decision changes state. Newest log entries go first.

## Branches and PRs

| What | Where |
|---|---|
| Integration branch | `2.x`. All 2.x work merges here, never into `alpha` (owner item 30). It carries the 2.x workflows (`2x-tests.yaml`, `2x-docs-preview.yaml`). |
| Spec (current) | `2x/phase-3-spec` → `2.x` (P-27, ruling P3-5). Every spec change is made here: ADRs, principles, design, plans, registers, rulings, reviews and this file. Phase 1 used `senselab-2x-api` (PR #617). |
| Code (current) | `2x/phase-3`, cut from the head of `2x/phase-2` → `2.x` (P-27, ruling P3-5). Code and tests only. Phase 1 used `2x/phase-1` (PR #619). |
| 2.x workflows | merged: #620 (on `alpha`, 1.x and 2.x workflows separated) and #621 (on `2.x`). Previews are at `sensein.group/senselab/2x/pr-<N>/`. |
| 1.x | developed on `alpha`; freeze branch `freeze-1x-2026-10-30` → `19834fdd` (= `1.4.0-alpha.11`). |
| pydra proposal | branch `enh/execution-extensions` → draft PR nipype/pydra#891; worktree `${SOFTWARE_ROOT}/nipype/pydra-ext`. |

## Where the records are (this directory)

- **Owner's words:** [`owner-requests.md`](owner-requests.md), verbatim, items 1..38. The authority.
- **Owner decisions:** [`decisions-2026-11-01.md`](decisions-2026-11-01.md), D-1..D-7 and the items that
  follow from them without a further choice.
- **Controller rulings:** [`rulings.md`](rulings.md), R1..R42 and H-R1..H-R57 with their status. They are
  not design authority; a ruling that changes a decision is in force only through an ADR amendment
  (ADR-000).
- **Reviews:** [`reviews/full-review-2026-11-01/`](reviews/full-review-2026-11-01/) (five reviews and
  `consolidated.md`), [`reviews/phase1-adversarial.md`](reviews/phase1-adversarial.md) (A-1..A-39),
  [`reviews/phase1-reattack/`](reviews/phase1-reattack/) (X*-n). Register rows R-n are in
  [`review-register.md`](review-register.md).
- **Rules:** `principles.md` (P-1..P-41), `engineering.md` (standards, checklist, roster, P-27..P-35,
  P-39, P-40).
- **What to build:** `design.md`, `adr/` (ADR-001..010), `execution.md`, `testing.md`, `docs.md`,
  `wire-protocol.md`.
- **Plans:** `plans/phase-1.md`, `plans/phase-1-hardening.md`, `plans/phase-2.md`.
- **Migration:** `migration-ledger.md`, a mark per 1.x area (P-39) and the triage classification.
- The controller's working notes under `.superpowers/` in the code worktree are git-excluded scratch.
  Nothing here cites them; anything a later session needs is in the files above.

## Decisions (2026-11-01)

The owner accepted the full review's recommendations ("accept recommendations and move specs to the docs
PR - clarify the decisions"):

- **D-1** Guards and gates cut back to the design: keep the import-linter contracts, the identifier check,
  the fixture budget (256 KB per file, 2 MB total) and the workflow-expression guard; delete the
  spelling-based bypass lints, `moved_modules_test.py` and tests that only make other tests exist; review and
  compact during development, no model in CI (owner item 39).
- **D-2** Locks are leases by rename and `O_EXCL` (ADR-003); revert the `flock`/heartbeat/takeover protocol
  and the 1.x-only lock modes; keep H9b's path safety.
- **D-3** Canonical text is NFC except text with code points unassigned in Unicode 13.0, for portability
  across interpreters (ADR-006).
- **D-4** ADR-010: an activity's `@id` is the execution; the plan link waits for Phase 3; Phase 2
  implements ADR-010, including the removal of the 1.x token families and export redaction.
- **D-5** The release gate checks version grammar, exact sibling pins and build-and-install from `dist/`
  until the first 2.x release.
- **D-6** `src/` is restored to the freeze commit and then merged with `alpha`; 2.x never edits `src/`.
- **D-7** Design in #617, code in #619; ADRs change only by dated amendments; no spec cites the ledger.

## Status

| Phase | State |
|---|---|
| Design | accepted 2026-10-30; amended 2026-10-31 and 2026-11-01 in #617 |
| 0: freeze marker | done |
| 1: kernel, worker runtime, identity, store, data fetch | T0–T15a done. Hardening done: H1–H8 (H8a–c), the H9 series (H9a–f, H9-compat, H9-collect, H9-cleanup, H9-ci), H10, **H11a** and **H11b** (including fix round 1). 2.x CI first fully green at `97b617c5`; Linux CI green after D-2 fix round 1 (`b76681a9`). Full review done 2026-11-01; its decisions are applied. **Phase 1 done 2026-11-01** at `195406b7`: CI green, exit adversarial round fixed (R-135..R-159), ORCD full packages suite and release gate green, cross-node NFS lock checks (a)-(f) pass. |
| 2: tasks, backends, benchmarks, parity, ADR-010 | **done 2026-11-03** at `79b417a8`, except the pydra gate prerequisite K2, which was not delivered and moves to Phase 3 Task 0 (P3-1) (`2x/phase-2`): CI green; exit adversarial round fixed (R-168..R-195, groups A–E); ORCD verify A–D agree (equivalence classes from runs p2cmd-27735057, p2cmd-27735059); parity at `75a1c65c` 8 passed, 9 skipped (jobs 27748712..27748718), CrisperWhisper ct2 against transformers within 4e-6 on cpu and 4e-4 on H100 float16 |
| 3: plans and executors | **done 2026-11-14** at `f37f1ae4` (`2x/phase-3`; ORCD evidence at `3c3dfced`, which differs by a tests-only diff, see `reviews/phase3-exit-evidence.md`): plan `plans/phase-3.md` (41 tasks); Task 0 gate passed 2026-11-03 at pydra `0c1bbcd9`, Route A (`reviews/phase3-gate/evidence.md`; CI 41606617714, ORCD 27757266); CI green at the head (run 43505114802); full matrix (3.10 worker floor, 3.11-3.14, core-only, lean, pydra, limits, licenses, source digests, pre-commit) 43566375276; exit adversarial round (Task 23, R-273..R-297) fixed by Task 24 (T24-1..T24-12, merge heads `7b56922d`..`3c3dfced`, then the Python 3.11 test fix) with every minor scheduled in `review-register.md`; ORCD exit evidence at the one head Steps 2 (29047611), 3 (29047612), 4 start, resume and provenance (29047613, 29047616, 29055900), 5 CPU, GPU and cross-node (29053174, 29047615, 29047619) and 6 (29049402) all pass at product `3c3dfced` (`reviews/phase3-exit-evidence.md`). The T24-7 I-5 residual (an unresolved unit is not fit-checked) was decided by the controller and confirmed by the owner on 2026-11-14. |
| 4–5 | not started |
| pydra PRs (A0…K2) | Phase 2 prerequisites (D1, G1, K1, A1, F1, F2, E1a, E1b) merged into `enh/execution-extensions` and closed; CI speed #902 open for review; workflow-cache fix #903. #891 carries the branch to `main` as a draft |

## Next actions, in order

Done on 2026-11-01: specs merged into `2x/phase-1` (`d90b5057`); **D-6** `src/` = `alpha` (`ffdf06e6`,
merge `a2be7b27`); **D-1** guard layer cut (`f7026f54`; the CI prose review of `b2db96be` was removed per owner item 39); **D-5** release gate trimmed (`a3ea1894`); **D-2** locks are leases by rename and `O_EXCL` (`5a325c29..12853f8b`; ADR-003 amended to
match what was built), plus **fix round 1** (`b76681a9`): a holder is identified by its uuid token and a judged lease by device, inode and mtime, because ext4 reuses inode numbers at once; Linux CI green; the D-1 fixture budget numbers (256 KB per file, 2 MB total) now live in
`scripts/data/fixture_budget.yaml` with their derivations.

1. **H-R49, done:** **H11a** (`fb1e774d`, `117a8641`, `621ad187`, `d574fe8e`, `7157ba7e`): regressions renamed by
   invariant and folded into component test files, no review ids in tests, default tier `logic`, PRs run
   logic + contract, limits on the `limits` label or nightly, use cases nightly only (H-R54), wall-clock
   checks in `limits`. **H11b** (`80b3aef4..e47860c2`, `05369e68`, `c21d424f`, `7bf9f7e9`, `a0d6b949`) and its
   **fix round 1** (`5db856fa`): internal monkeypatches replaced by seams (`Resolver`, `ResultStore(equivalence=)`,
   `WorkerClient` constructor arguments) and faults injected at the `os` boundary; the only substitutions of
   senselab internals are `to_wire` and `_wait_exit` in a spawned child, and an AST guard
   (`tests/lint/internal_patches_test.py`) enforces it (`testing.md` section 3). Policy checks as
   `scripts/check_*.py` remain for the rest of the forbidden list.
2. **Follows items done 2026-11-01** (commits on `2x/phase-1`, code PR #619):
   - **F1** `7e179167..e363dc87`: portable model identity (`logical-key@4`), `ResultStore.get` applies
     `DEFAULT_REUSE_POLICY`, equivalence classes as data with a derivation, one transient classifier
     (`faults.is_transient`) and `RetryPolicy` from data (`SENSELAB_HF_MAX_RETRIES` gone).
   - **F2** (senselab `ef205267`, pydra-ext `010bdf98`): environment recorded not keyed; K2 added; owners for
     the pydra PRs in design section 6.2.
   - **F3** `56563623..2cd31918`: H-R56 (`logical-key@5`, `execution-context@2.0`, `model_source`); audio
     decoding, stamping and the memo moved to `runtime/audio_io.py` under the `kernel-does-no-io` contract;
     the classifier reads `response.status_code` from any HTTP client.
   - **F4** `409c0789`: every 2.x CI job installs with `uv sync --locked` except `worker-floor`; the
     equivalence table loads in the runtime (`runtime/data/equivalence.yaml`); `importlib` is forbidden in
     the kernel; `huggingface-hub` bounded `>=1.3,<3`.

   **Gates and parked items, done:** the engineering and license gates, including the locked 3.10 floor
   (`73931a6d..ab402590`), F1 to F4, and the parked items (all in `review-register.md`).

   **Follows still open** are rows in `review-register.md`, not text here: `model_source` R-110, lease renewal R-111,
   verify sampling R-112, the nightly workflow on `alpha` R-113, the clock re-binding R-114, the source-digest CI
   check R-115, per-task licences and taint R-116, the P-40 roster and worktree isolation R-117, whole-file
   hashing R-109.
   Parked items: all in review-register.md (R-110..R-134); none live only in the ledger.
3. **Phase 1 exit**, remaining steps in order (`plans/phase-1.md` Task 15, as amended; final review of
   `82a986e1`):
   1. X1 code fixes: done.
   2. CI green at the final head: done, `195406b7` (2026-11-01).
   3. Docs fixes: done.
   4. ORCD at `195406b7`: done. Packages 3023 passed, 4 skipped; release gate 17 passed; cross-node NFS lock
      checks (a)-(f) pass on 3 nodes, NFS 4.2 (jobs 27543616, 27543968, 27545193). (c) forces the lost-reply path by
      injection, (d) passes within a tolerance, (f) plants the unreadable lease as the same user. `src/tests` is not
      run in the 2.x environment: the root is a metadata umbrella and `src/` equals `origin/alpha`, whose CI covers it.
   5. The Task 15 adversarial round with a duplicate-finding pass (P-37): done against `c54e9712`
      (`reviews/exit-adversarial-2026-11-01/`, R-135..R-159); 11 high/medium findings fixed and reviewed
      (`c294151c..195406b7`).
   6. Resolve or schedule every register finding: done; every open row names an owner.
   7. Mark Phase 1 done, merge #617 into `2.x`, and retarget #619: done 2026-11-01 (owner: "do 2").
4. **Phase 2:** ADR-010 (PROV-O JSON-LD store, `senselab:` vocabulary, token-family removal, export
   redaction) and the benchmark harness with its data manifest and smoke suite, alongside the tasks and
   backends of design §6.2. Plan: [`plans/phase-2.md`](plans/phase-2.md) (Tasks 0–34); its planner rulings
   P2-1..P2-32 await owner review.

## How to resume

1. In the spec checkout, `git pull`. Read this file, then `decisions-2026-11-01.md`, then
   `principles.md` and `engineering.md` (roster and checklist).
2. Code work happens on the current code branch above; spec changes happen on the current spec branch.
3. Check 1.x drift: `git fetch origin alpha && git log 19834fdd..origin/alpha --oneline -- src scripts tutorials`.
4. PR state: `gh pr list --base 2.x`; 2.x CI runs on every `2x/**` branch.
5. Brief every delegated task as `engineering.md` requires; file-modifying agents use worktree isolation.

## Log

- **2026-11-14 (Phase 3 done):** Task 24 fixed what the exit adversarial round and the Task 21 reruns found, in twelve tasks, each merged `--no-ff` onto the phase head with CI green and reviewed by a different model:
  - **T24-1** `7b56922d`: every run-state write goes through `StorageDir` (`write_atomically`, `publish`), event appends loop and truncate, readers quarantine a torn line, an AST lint refuses a hand-rolled write (R-290, R-285).
  - **T24-3** `88eeb813`: a fault class survives every wrapper; one `FaultRecord` constructor pair and one server-fault mapper; a load-time fault is the environment's, never split (R-287, R-289, R-292, R-257; `run-event@2.2`).
  - **T24-2** `a4191872`: one run lease combined across epochs by the longest bound, an epoch current only once its attempt record exists, the license policy persisted and never loosened, a resumed coordinator follows the live item that holds a job instead of re-cutting it, `RunStatus.quarantined` (R-273, R-277, R-283, R-280; `run-event@2.3`).
  - **T24-4** `2dbdb536`: one bisection engine for both executors; an item that crashes two fresh workers is charged to the item, a crash another item survives stays infrastructure (R-288; ADR-009 amendment).
  - **T24-5** `fc4d203e`: shutdown is total, every step runs under `ExitStack` and a fault at each step is injected (R-291), with the two CI-hang test fixes (R-274, R-275).
  - **T24-6** `6dec9d5c`: fixed parameters become profile keys with derivations, the caller's profile reaches the store, probe and fetch, and `equivalence.yaml` leaves the package (R-219, R-254, the vLLM timeout, R-300).
  - **T24-7** `bb419411`: a load-time resource fault records an unfit-for-this-class fact; placement skips an unfit or too-small class and stops the plan as `Unplaceable` when none is left; `senselab unfit [--clear]`.
  - **T24-8** `527f6925`: `WorkerClient.close()` is safe in a signal handler and every wait it reaches is lock-free (R-276, R-295).
  - **T24-9** `a065218b`: the epoch is the one exclusion for every executor; the coordinator lease file and the pydra save lock are gone, each attempt has its own pydra root, `InProcess` has an epoch fence and a `superseded` status. A resume on another host no longer waits forever (R-302 closed).
  - **T24-10** `33ceb341`: a worker's provenance for a claim is durable before its outcome is committed, one provenance document per claim (ADR-010 amendment).
  - **T24-11** `1fc88492`: cluster-only defects caught locally: abrupt stops at every boundary of the commit path, a host-shape fixture, a lint refusing a host-bound lock, and the provenance completeness check as a product function (`senselab status --check-provenance`).
  - **T24-12** `3c3dfced`: the ORCD step drivers live in `benchmarks/exit` under mypy-strict with a CI smoke test, one stall watchdog, and diagnostics that check their own joins.
  Format tags bumped: `execution-profile@5.0`, `run-lease@2`, `server-settings@2.0`; new: `measure-unfit@1.0`; event tags `run-event@2.2` and `2.3`. Each older tag is refused naming both, and the group site profiles were re-tagged before the reruns.
  Owner rulings recorded in `progress.md`: profile node exclusions are a floor (R-282; ADR-004 sentence now, code with Phase 4's merged loader); an item that crashes two fresh workers is charged to the item, the count a profile key (R-288); `equivalence.yaml` moves out of the package and no measured classes ship (N-8); a load-time resource fault records an unfit class and placement reroutes it ("New task now", T24-7); a run's license policy can be tightened and never loosened; Step 2 compares arrays under the run-config `samples` threshold (atol 1e-6), not identities; T24-11 and T24-12 added to Task 24 (2026-11-13); the pydra #902 replies approved.
  Confirmed by the owner on 2026-11-14: T24-7's I-5 residual. A unit whose digest cannot be resolved has no key for an unfit record, so it is not fit-checked; its load OOM still holds the environment (T24-3) and is visible. The controller accepted it per run-without-stalls.
  Deferred, with owners in `review-register.md`: 7a'-2 and T20R-11 after the exit (R-303, R-304), R-264 and R-281 to Phase 4, R-297 to Phase 5, within-item dedup of identical inputs (R-305), a per-commit delta provenance document at 10x scale (R-306); the nits the reviews deferred are in the register's Phase 3 exit register section. Phase 4 starts from `2x/phase-3` at the head above.
- **2026-11-03 (Phase 3 gate):** K2 delivered as nipype/pydra#905 (Task 40); the gate passed at `0c1bbcd9` on all four criteria, so Task 9b builds `PydraRunner` (Route A). Fixes found on the way: #903 workflow-construction cache keyed by class, #904 cf workers never forked.
- **2026-11-03 (Phase 2 done):** parity compared against 1.x for the first time (34i/34j) and found three differences: the 1.x embedding reference was one zero-padded batch, the 1.x ASR reference decoded with word timestamps, and the CrisperWhisper ct2 engine reported no confidence. Task 34k (`3ef57813..75a1c65c`) fixed all three; confidence is computed once per window on its final ids, by teacher forcing, on both engines. The 1.x producer takes its params from the items file, and `run` refuses an items file of another benchmark, backend or items. Rows R-196..R-201.
- **2026-11-01 (final review, docs fixes):** the pydra environment fold corrected to ADR-002; as-built detail
  removed from ADR-003 and ADR-009; register rows R-132..R-134 added (prose review inert, config-guard tests,
  size budgets); stale plan text fixed.
- **2026-11-01 (F1, F3, F4 done):** model key is the weights digest and the source is context (`logical-key@5`,
  `execution-context@2.0`; H-R56); decoding and stamping out of the kernel; one transient classifier and a data
  retry policy; equivalence classes as data; CI locked except `worker-floor` (H-R57). ADR-002, 003, 005, 006 and
  009, `design.md`, `testing.md`, the plans and rulings updated.
- **2026-11-01 (D-2, H11a, H11b done):** locks are leases (D-2) with fix round 1 (`b76681a9`, Linux CI green);
  H11a renamed and re-tiered the tests; H11b replaced internal patches with seams and guarded the rest
  (fix round 1 `5db856fa`). Docs pass: ADR-003 identity wording, `testing.md` fault model and cadence,
  register citations checked against `origin/2x/phase-1`, H-R53 wording. ORCD full runs await owner sign-in.
- **2026-11-01 (decisions applied to specs):** owner decisions D-1..D-7 recorded. Specs moved from
  `2x/phase-1` into #617. ADR-003, ADR-006, ADR-009, design.md and docs.md restored to decisions, with
  dated amendments and amendment logs; ADR-007 and ADR-010 amended; ADR-000 gains the amendment rule;
  P-27, P-31 and P-35 amended in `engineering.md`; rulings' statuses set; `2.x` merged into
  `senselab-2x-api`.
- **2026-11-01 (full review):** five reviews and a consolidation at `3a468aff`
  (`reviews/full-review-2026-11-01/`). Implementation had rewritten the design under unchanged headers;
  the guard layer had grown to about 76% of package source.
- **2026-11-01 (H10, H-R48, H-R50..H-R52):** `src/` stays on the 2.x line as the 1.x source pool, imported
  by nothing in `packages/`. The root `senselab` distribution is the 2.x umbrella. Core provenance has only
  the generic `artifact` type, `run_id` is required, and run manifests live under `$SENSELAB_CACHE/2x/`.
- **2026-11-01 (2.x workflows):** #620 and #621 merged; the `2.x` branch runs its own workflows.
- **2026-10-31 (H9 series):** path safety under storage roots (H9b), values and source stamps (H9c),
  portable canonical text (H9d), typed digests (H9e), release gate round (H9f).
- **2026-10-30 (Phase 1 adversarial round):** A-1..A-39 against `94e81479`, recorded as R-59..R-98;
  hardening plan H1–H8.
- **2026-10-30 (Phase 1 start):** P-39 scan found no drift (`origin/alpha` = `19834fdd`).
- **2026-10-30:** design completed through five reviews and an architect review; ADR-001..009; Phase 0
  freeze branch; Phase 1 delivered T0–T14.
