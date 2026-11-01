# senselab 2.x: working state

Read this first when resuming. Update it at the end of every work session, and whenever a phase,
task or decision changes state. Newest log entries go first.

## Branches and PRs

| What | Where |
|---|---|
| Integration branch | `2.x`. All 2.x work merges here, never into `alpha` (owner item 30). It carries the 2.x workflows (`2x-tests.yaml`, `2x-docs-preview.yaml`). |
| Design (docs PR) | `senselab-2x-api` → `2.x`, draft PR sensein/senselab#617. Every spec change is made here: ADRs, principles, design, plans, registers, rulings, reviews and this file (D-7). Worktree `~/software/sensein/senselab-2x`. |
| Code (code PR) | `2x/phase-1` → `senselab-2x-api` today, draft PR sensein/senselab#619. Code and tests only; it merges `senselab-2x-api` to stay current and is retargeted to `2.x` once #617 merges (D-7). Worktree `~/software/sensein/senselab-2x-impl`. |
| 2.x workflows | merged: #620 (on `alpha`, 1.x and 2.x workflows separated) and #621 (on `2.x`). Previews are at `sensein.group/senselab/2x/pr-<N>/`. |
| 1.x | developed on `alpha`; freeze branch `freeze-1x-2026-10-30` → `19834fdd` (= `1.4.0-alpha.11`). |
| pydra proposal | branch `enh/execution-extensions` → draft PR nipype/pydra#891; worktree `~/software/nipype/pydra-ext`. |

## Where the records are (this directory)

- **Owner's words:** [`owner-requests.md`](owner-requests.md), verbatim, items 1..38. The authority.
- **Owner decisions:** [`decisions-2026-11-01.md`](decisions-2026-11-01.md), D-1..D-7 and the items that
  follow from them without a further choice.
- **Controller rulings:** [`rulings.md`](rulings.md), R1..R42 and H-R1..H-R54 with their status. They are
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
- **Plans:** `plans/phase-1.md`, `plans/phase-1-hardening.md`.
- **Migration:** `migration-ledger.md`, a mark per 1.x area (P-39) and the triage classification.
- The controller's working notes under `.superpowers/` in the code worktree are git-excluded scratch.
  Nothing here cites them; anything a later session needs is in the files above.

## Decisions (2026-11-01)

The owner accepted the full review's recommendations ("accept recommendations and move specs to the docs
PR - clarify the decisions"):

- **D-1** Guards and gates cut back to the design: keep the import-linter contracts, the identifier check,
  the fixture budget (256 KB per file, 2 MB total) and the workflow-expression guard; delete the
  spelling-based bypass lints, `moved_modules_test.py` and tests that only make other tests exist; add the
  advisory agent prose review.
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
| 1: kernel, worker runtime, identity, store, data fetch | T0–T15a done. Hardening done: H1–H8 (H8a–c), the H9 series (H9a–f, H9-compat, H9-collect, H9-cleanup, H9-ci), H10, **H11a** and **H11b** (including fix round 1). 2.x CI first fully green at `97b617c5`; Linux CI green after D-2 fix round 1 (`b76681a9`). Full review done 2026-11-01; its decisions are applied. Remaining: the "follows" items and the Phase 1 exit (next actions). |
| 2–5 | not started |
| pydra PRs (A0…K2) | proposed in #891; none implemented. Owners named in design §6.2 (Phase 2: gate prerequisites; Phase 3: the rest); dates not set |

## Next actions, in order

Done on 2026-11-01: specs merged into `2x/phase-1` (`d90b5057`); **D-6** `src/` = `alpha` (`ffdf06e6`,
merge `a2be7b27`); **D-1** guard layer cut and advisory prose review added (`f7026f54`, `b2db96be`;
needs the `ANTHROPIC_API_KEY` repository secret); **D-5** release gate trimmed (`a3ea1894`); **D-2** locks are leases by rename and `O_EXCL` (`5a325c29..12853f8b`; ADR-003 amended to
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
2. **Next: the "follows" items** of the decision record: portable model identity (A I-1), read
   `DEFAULT_REUSE_POLICY` (Ru I-1), equivalence classes as data (A I-8), owners for the clock re-binding
   (A I-9) and the source-digest CI check (A I-10); audio decoding out of the kernel and whole-file
   hashing against ADR-005 (Ar F-3, A I-5, R-109); the pydra corrections (Ar F-5, F-6) and owners for the
   pydra PRs (**F2, done 2026-11-01**: environment is recorded not keyed; K2 added to the proposal; K1/K2 in the Phase 3 gate; owners in design §6.2; benchmark copy removed from `specs/`, kept in pydra `benchmarks/proposal_0001/`); one transient classifier (Ar F-7); the engineering and license gates (T F5–F7); the P-40
   roster and worktree isolation; a scheduled workflow on `alpha` (the default branch) that checks out `2.x` for nightly limits and use cases, landed as a `skip-release` PR (GitHub runs `schedule` only from the default branch); parked items moved into tracked registers (Ru I-11, I-12); lease renewal not yet implemented, owner:
   execution phase (Phase 3), since `Lease` has no renewal and `execution.md` 3.2 says workers renew.
3. **Then the Phase 1 exit** (`plans/phase-1.md` Task 15, as amended): 2.x CI green; `src/` equals `alpha`; no 1.x
   tests on the 2.x line; P-39 re-scan; `STATE.md` updated. The ORCD full runs are **pending owner sign-in**. Run these on ORCD shared NFS scratch from a pinned checkout, in a submitted job
   (never on login or claude-rc nodes), together with the full local-tier runs H11b did not make (the env tier and the later-boundary sweep): (a) `SENSELAB_WORK_POOL_NFS_DIR=<dir>` run of
   `packages/senselab-core/tests/runtime/locks_test.py`, whose two `[nfs]` cases cover one taker and
   concurrent breakers; (b) cross-node exclusion: processes on 2-4 nodes calling `record_resolution` with
   distinct SHAs against one `SENSELAB_CACHE`, expecting one SHA, one manifest entry and no lease litter;
   (c) lost-reply path: `link` EEXIST with `st_nlink == 2` under load; (d) clock skew: a lease planted on
   one node is aged by `fs_now()` on another, within `acregmin`/`acregmax`, and the lstat pre-check is not
   defeated by attribute caching; (e) stale-break race across nodes: one break log, and any "displaced"
   warning matches a holder whose release logs it; (f) an unreadable 0o000 stale lease left by another
   user in a group-writable run directory is still broken. Then merge #617 into `2.x` and retarget #619.
4. **Phase 2:** ADR-010 (PROV-O JSON-LD store, `senselab:` vocabulary, token-family removal, export
   redaction) and the benchmark harness with its data manifest and smoke suite, alongside the tasks and
   backends of design §6.2.

## How to resume

1. `cd ~/software/sensein/senselab-2x && git pull`. Read this file, then `decisions-2026-11-01.md`, then
   `principles.md` and `engineering.md` (roster and checklist).
2. Code work happens in `~/software/sensein/senselab-2x-impl` on `2x/phase-1`. Spec changes happen here,
   on `senselab-2x-api`, and reach the code branch by merge.
3. Check 1.x drift: `git fetch origin alpha && git log 19834fdd..origin/alpha --oneline -- src scripts tutorials`.
4. PR state: `gh pr view 617`, `gh pr view 619`; 2.x CI runs on both.
5. Brief every delegated task as `engineering.md` requires; file-modifying agents use worktree isolation.

## Log

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
