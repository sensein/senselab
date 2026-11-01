# Full review: traceability (owner requests and P-1..P-41 to decisions and code)

Target: `2x/phase-1` at `3a468aff` (worktree `senselab-2x-impl`). Read-only review; git reads, `gh` reads and
file reads only. Spec paths are relative to `specs/20261030-140022-senselab-2x-api/` unless they start with
another top-level directory. "Scheduled" means a phase in `design.md` §6.2, `docs.md` §Phasing, `roadmap.md` or
`plans/` assigns the item to a later phase; Phase 1 was not meant to build it.

Verdicts: **conforms** (built as decided, or scheduled for a later phase); **deviates** (built differently);
**omitted** (decided, due by now, not built, or due and assigned to no phase); **drift** (a later ruling or edit
contradicts an earlier decision, or an owner correction did not land everywhere it binds).

## Owner requests (owner-requests.md)

| # | Carrying decision | Implementation or schedule | Verdict |
|---|---|---|---|
| 1a | API for humans and agents: P-1 principles.md:21-31; design §4.2 design.md:221-260, §4.10 design.md:504-534 | Phase 1 has dataclass declarations only (`kernel/declare.py:121-226`; plans/phase-1.md:870 "decorators and generated functions come in Phase 2"). `describe`, schemas, generated functions: Phase 2 (design.md:909-926, exit "an agent runs each task from `describe` output alone"). MCP and CLI: Phase 5 (design.md:969-978). Root `senselab` ships no code yet (README.md:16) | conforms (scheduled) |
| 1b | versioning/provenance/hashing models: P-3, P-4 principles.md:44-60; design §4.3, §4.8; ADR-002 | `kernel/identity.py:67-296` (ModelRef, ResolvedModel, LogicalKey, ExecutionContext, ReusePolicy), `runtime/models.py:118` `resolve`, `runtime/provenance.py`, `runtime/store.py` | conforms |
| 1c | subprocess envs, local and clusters, GPU vs CPU, inference stacks: P-5..P-8; design §4.4, §4.9; execution.md | worker runtime and client (`senselab_worker/serve.py`, `runtime/worker_client.py`), torch-free probe `runtime/hardware.py:357`. Env build/lock, stacks: Phase 2 (design.md:909-926); Pool/Slurm: Phase 3 (design.md:928-943) | conforms (scheduled) |
| 1d | nested pipelines; adapt to environment, hardware, dataset (BIDS or in-memory mapping): P-9, P-10, P-11 principles.md:102-129; design §4.5-§4.7; ADR-001, ADR-004 | Plan/placement: Phase 3 (design.md:928-943, exit "plans differing only in execution fields"); minimal BIDS adapter Phase 3; `Pipeline` with nesting, full adapters (`Dataset.from_records` in-memory) Phase 4 (design.md:944-967) | conforms (scheduled) |
| 1e | core as light as possible: P-13 principles.md:142-150; design §5 design.md:671-701 | three distributions (`packages/senselab-worker`, `packages/senselab-core`, root umbrella); import-linter `core-is-framework-free` (.importlinter); `core-only` CI job (.github/workflows/2x-tests.yaml) | conforms |
| 2 | P-12 principles.md:131-140; design §4.6 design.md:328-354; Phase 4 exit design.md:958-966 | `packages/*/src` hold no dataset names (grep `b2ai|stimulus_text` finds only a spec path in a docstring, `runtime/bep028.py:5`). Dataset profiles and the static check: Phase 4 | conforms (scheduled) |
| 3 | P-14 principles.md:152-158 | compatibility removed (H-R47): `runtime/provenance.py:177-197` reads only `provenance@2.0`; no shims found in `packages/` | conforms |
| 4 | P-15 principles.md:160-167; change rules principles.md:7-15 | register exists and is cited; but not kept current: no change-log entry for 2026-11-01 (principles.md:382-433), P-27 contradicts owner #30, and P-ids are mis-cited (CLAUDE.md:42) | deviates (F2, M1) |
| 5 | design §5 "Torch reach" design.md:701-702; §3 anti-goals design.md:98 | core and worker torch-free (import-linter `core-is-framework-free`, `worker-is-the-base`); release gate imports every core and worker module in its own venv (design.md:752-760) | conforms |
| 6 | P-24 principles.md:202-211; design §4.11 design.md:551-576 | shared kernel types `kernel/time.py`, `kernel/results.py`, `kernel/signals.py` (Recording with typed channel groups). Env grouping report Phase 2, adoption Phase 4 (design.md:924, 951) | conforms (scheduled) |
| 7a | core across subprocess venvs and Pythons: P-22 principles.md:169-181; design §4.12 design.md:578-594 | `senselab_worker` stdlib+numpy, `requires-python >=3.10` (packages/senselab-worker/pyproject.toml); handshake checks wire and runtime digest (`runtime/worker_client.py:640-650`); floor job 3.10/numpy 1.23.5 (2x-tests.yaml `worker-floor`) | conforms |
| 7b | licenses clear and machine-readable: P-23 principles.md:183-200; design §4.13 design.md:596-622 | `kernel/license.py:32-52` (record, `require_license`); `BackendSpec.license` required (`kernel/declare.py:148`); `DataManifest.license` (`runtime/fetch.py:159`). No CI license gate over shipped locked packages; policy, sets, taint and `sl.licenses` assigned to no phase | omitted (F7) |
| 8 | P-22 principles.md:171; changelog principles.md:389-391 | core `requires-python >=3.11`, root `>=3.11,<3.15` (pyproject.toml) | conforms |
| 9 | P-22 principles.md:172; resolved tension principles.md:375-377 | worker floor 3.10 (packages/senselab-worker/pyproject.toml) | conforms |
| 10 | design §4.14 design.md:624-652 (hyperparameters at construction, `.with_params`, `fit`; not fit/transform/predict, not `X`) | decision recorded; Task objects Phase 2; `sl.sklearn` adapter deferred (design.md:1002) | conforms (decided; scheduled) |
| 11a | isolated locked envs, no system libs: P-25 principles.md:213-223; design §4.4, §5 design.md:697-699 | `ExecutionContext.env_lock` (`kernel/identity.py:239`); env creation/locking Phase 2. CI installs core unlocked (`uv pip install` in 2x-tests.yaml `core-only`, `core-env`; 2x-docs-preview.yaml) | deviates (M5) |
| 11b | pydra stable/debuggable, a PR to pydra, be critical of engines: design §4.9 design.md:488-505; engine-choice.md; workflow-engines.md §4 | draft PR nipype/pydra#891 (STATE.md:13); STATE.md:49 "none implemented"; Phase 3 gate needs E1, D1, F2 working (design.md:495-499) but no plan owns them | conforms (decided), M11 |
| 11c | in place on alpha; core its own distribution | three distributions conform (design §5). "In place on alpha" superseded by owner #30; see #30 | conforms / F1 |
| 11d | no second dataset, design standpoint: design.md:1043, Phase 4 exit design.md:958-963 | synthetic dataset with another entity scheme: Phase 4 | conforms (scheduled) |
| 11e | commit+push early often: P-27 engineering.md:124-129 | branch pushed (`origin/2x/phase-1` = HEAD, 0 ahead/behind). P-27 statement still says "PR to `alpha`", "built in place on `alpha`" (engineering.md:127) | drift (F1) |
| 11f | no identifiers leak: P-28 engineering.md:131-139 | `participant-ids` pre-commit hook (.pre-commit-config.yaml). Tracked `.claude/projects/-Users-<username>-.../memory/*.md`; personal ORCD paths in inherited 1.x specs (e.g. specs/20261005-breathing-pattern/design.md:50) | deviates (M2) |
| 11g | local and ORCD tests within limits: P-29 engineering.md:141-151 | CLAUDE.md:27-36; ORCD runs recorded in the (untracked) ledger | conforms |
| 11h | use cases and models research incl. actigraphy, ECG: use-cases.md §1-§5 (e.g. use-cases.md:13-21, 106) | research done; catalogue entries come with ports | conforms |
| 11i | generic vs modality-specific: P-26 principles.md:225-236; design §4.15 design.md:654-669; use-cases.md §4 (use-cases.md:189-248) | kernel `Recording` with typed channel groups and `IrregularGroup` (`kernel/signals.py:242-417`); `sl.signal` tier Phase 4 (design.md:950, exit "one generic task runs unchanged on audio and on accelerometry") | conforms (scheduled) |
| 11j | missing license raises by default; license sets: P-23 principles.md:188-193; design.md:609-617 | `MissingLicense` exists; no policy object, no sets, no `on_missing`; unscheduled | omitted (F7) |
| 12a | tests: logic vs limits vs use case: P-30 engineering.md:153-171; testing.md §2-§5 | markers registered (pyproject.toml markers); 215 tests named for finding ids, every tier runs on every PR and a test enforces it; correction (H-R49, "H11") not yet run | drift (F4) |
| 12b | benchmarks separate on public data: testing.md §6 testing.md:203-222; benchmarks.md; docs.md:85-88 | no `benchmarks/`, no manifest, no smoke suite; docs.md puts the first manifest consumer (benchmark smoke) in Phase 1 | omitted (F9) |
| 12c | worker pools, GPU-only tasks, Slurm; senselab vs pydra; pydra alpha via uv branches: execution.md §3-§5; pydra-extension-points.md; design.md:493-494 | design only; Phase 3 | conforms (scheduled) |
| 13 | engine-choice.md §1-§3 (requirements table vs engines) | done | conforms |
| 14a | reviewers over all pieces; draft pydra PR | review-register.md R-1..R-98; architect-review.md; pydra#891 draft | conforms |
| 14b | engineering principles on every commit; minimal verbosity; profiling at scales; brittleness; agents break the system: P-31..P-35 engineering.md:42-66, 173-226 | adversarial round and re-attack done (reviews/phase1-adversarial.md; tests/adversarial). Per-commit gates (mypy strict, complexity, duplicate check, size budgets, PR-template checklist, prose review, blocking id check) not built | omitted (F5, F6) |
| 15 | P-32 engineering.md:181-195 (fastio as algorithmic exemplar; best practice) | standards written; their enforcement listed in engineering.md:47 not built | omitted (F6) |
| 16 | engine-choice.md; pydra-extension-points.md §7 | design only | conforms |
| 17 | P-36 principles.md:257-268; roadmap.md:1-56 | `Annotation.producer`, `guideline`, `uncertainty` (`kernel/results.py:26-47`) cover hooks H-1, H-5; no flywheel features built | conforms (M3) |
| 18 | P-37 principles.md:270-281; engineering.md:48; design §6.1 inventory design.md:869-885 | `Registry` refuses duplicate names and signatures (`kernel/declare.py:163-193`). No duplicate-code check; no duplicate-finding pass in the Phase 1 adversarial brief (plans/phase-1.md:838-844) | omitted (F6) |
| 19 | architect-review.md; ADR-000-index.md | done | conforms |
| 20 | ADR-001 (graph format, compiles to pydra); P-38 principles.md:283-299; ADR-002 | `LogicalKey` keys declared facts only (`kernel/identity.py:207-237`); reuse policies with `equivalent` default (`kernel/identity.py:259-296`). Graph format Phase 4. Version-bump CI check unscheduled | conforms (M7) |
| 21 | vLLM-omni kept: design.md:1058-1059, P-8; P-27..P-35 moved to engineering.md (principles.md:419-422) | vLLM-omni after Phase 3 (design.md:941-943); move done | conforms |
| 22 | P-31 check engineering.md:46, 179; design.md:1005 | no prose-review model in CI and no phase owns it; the one blocking id check is not built | omitted (F5) |
| 23 | P-39 engineering.md:228-239; design.md:853-867; migration-ledger.md | ledger marks `19834fdd`; Phase 1 start scan done (STATE.md:88-89). Phase 1 exit re-scan pending (plans/phase-1.md:846) | conforms (M13) |
| 24 | P-40 engineering.md:76-117, 241-252 | plan tasks name models (plans/phase-1.md "Model:" lines). Implementer visible only as `Co-Authored-By` trailer (145 Opus, 31 Sonnet); reviewer models recorded only in the untracked ledger; one PR (#619, 177 commits) instead of one per task | deviates (M6) |
| 25 | P-41 principles.md:241-255; docs.md | 2.x preview workflow (pdoc API only) at `/senselab/2x/pr-N/` (2x-docs-preview.yaml); status page Phase 2 (docs.md:89-90); fixture budget (`scripts/check_fixture_budget.py`, `scripts/data/fixture_budget.yaml`); generators `testing/synth.py`. Phase 1 manifest/benchmark smoke missing | omitted (F9), M9 |
| 26 | STATE.md:1-4 ("update at the end of every work session") | STATE.md stale; rulings H-R1..H-R52 only in a git-ignored ledger | omitted (F2) |
| 27 | process (no spec line) | CLAUDE.md:31-36 | conforms (not checkable from the repo) |
| 28 | process | CLAUDE.md:33-35 ("do not ... create new venvs or download models locally") | conforms |
| 29 | process | ledger records a green 2x-tests run at 97b617c5 (untracked) | conforms (not checkable from the repo) |
| 30 | owner correction | CLAUDE.md:20 and 2x workflows target `2.x`; P-27 (engineering.md:127), design.md:832, 1042, STATE.md:54, plans/phase-1.md:63, 200, plans/phase-1-hardening.md:67 and the PR #619 body still say `alpha` | drift (F1) |
| 31 | owner correction | `2x-tests.yaml`, `2x-docs-preview.yaml`; 1.x `tests.yaml`/`docs-preview.yaml` `branches-ignore: [2.x, senselab-2x-api, 2x/**]`; merged to alpha (#620) and 2.x (#621) | conforms |
| 32 | P-14; ruling H-R47 | landed (see #3) | conforms |
| 33 | testing.md; audit (ledger), ruling H-R49 | audit found 14 deviations; conformance task "H11" queued, not run | drift (F4) |
| 34 | owner correction | CLAUDE.md rewritten for 2.x. AGENTS.md still 1.x (Python 3.11-3.12, `uv run pytest -n auto`, AGENTS.md:7, 28); tracked 1.x memory notes under `.claude/projects/` | deviates (M2) |
| 35 | owner correction | CLAUDE.md is short and 2.x-only, but cites P-41 for one implementation per behaviour (CLAUDE.md:42) | deviates (M1, M2) |
| 36 | ADR-009 amended ADR-009:19-26; migration-ledger.md:28-38 | `src/` kept as frozen pool; import-linter `packages-not-1x`; ruff/mypy/pre-commit exclude `src/`. design.md §6.1 and the Phase 1 exit still describe rewired, tested 1.x | drift (F3) |
| 37 | P-36; ADR-009:27-30; migration-ledger.md §Triage | ledger classifies every triage module; design.md:953-958 still says orchestration stays on the 1.x tag and triage is not ported as a pipeline | drift (M3) |
| 38 | this review | five reviewers at 3a468aff | conforms |

## Principles P-1..P-41

| P | Carrying decision | Implementation or schedule | Verdict |
|---|---|---|---|
| P-1 | principles.md:21-31; design §4.2, §4.10 | declarations `kernel/declare.py`; schemas and `describe` Phase 2; MCP Phase 5 | conforms (scheduled) |
| P-2 | principles.md:33-42; design §4.7; ADR-004 | key vs context split `kernel/identity.py:207-257`; `TaskSpec` has no device/env field; plans Phase 3 | conforms |
| P-3 | principles.md:44-52; design §4.8; ADR-002, ADR-010 | `runtime/provenance.py`, `ExecutionContext`, `ResolvedModel`; framework-written provenance around every node Phase 3. ADR-010 model unscheduled (F8) | conforms (F8) |
| P-4 | principles.md:54-61; design §4.3 | `ModelRef.parse`, `ResolvedModel`, `UnpinnedModel` (`kernel/identity.py:43-180`); `runtime/models.py:118` | conforms |
| P-5 | principles.md:63-70; design §4.9 | store and fetch cache in place; executors, resume Phase 3 | conforms (scheduled) |
| P-6 | principles.md:72-80; design §4.4 | one worker protocol (`senselab_worker/serve.py`, `runtime/worker_client.py`); env build Phase 2 | conforms (scheduled) |
| P-7 | principles.md:82-90 | `probe_hardware`, `numerics_class` (`runtime/hardware.py:357-430`); resource tables Phase 3 | conforms |
| P-8 | principles.md:92-100 | Phase 2 two-stack exit (design.md:925); vLLM-omni after Phase 3 | conforms (scheduled) |
| P-9 | principles.md:102-109; ADR-001 | Phase 4 | conforms (scheduled) |
| P-10 | principles.md:111-119; ADR-004 | Phase 3 | conforms (scheduled) |
| P-11 | principles.md:121-129; design §4.6 | Phase 3 (minimal BIDS), Phase 4 (full, in-memory) | conforms (scheduled) |
| P-12 | principles.md:131-140 | packages clean; static check Phase 4 | conforms |
| P-13 | principles.md:142-150; design §5 | import-linter contracts; `core-only` job; gate imports every module | conforms (M8) |
| P-14 | principles.md:152-158 | H-R47 landed | conforms |
| P-15 | principles.md:160-167 | register not updated for 2026-11-01; P-27 stale; mis-citations | deviates (F2, M1) |
| P-16 | principles.md:344-346 | `runtime/revision.py`; no 2.x loaders yet | conforms |
| P-17 | principles.md:348-351 | `runtime/data/weights.yaml`, `runtime/data/store_policy.yaml`, `scripts/data/fixture_budget.yaml` with derivations; `EQUIVALENCE` is a code literal | conforms (M7) |
| P-18 | principles.md:353-356 | `senselab data fetch <manifest> [--cache]` (`cli.py`) | conforms |
| P-19 | principles.md:358-360 | `store@4` key space (`runtime/store.py:78`) | conforms |
| P-20 | principles.md:362-365 | `_measure_has_no_verdict` (`testing/contracts.py:159-161`) | conforms |
| P-21 | principles.md:367-369 | not audited line by line here (other reviewers) | conforms (not audited) |
| P-22 | principles.md:169-181 | worker floor, handshake. Core CI: 3.11, 3.14 (`core-only`), 3.12 (`core-env`); 3.13 untested; no nightly | deviates (M4) |
| P-23 | principles.md:183-200; design §4.13 | record type only; no gate, policy, sets, taint; unscheduled | omitted (F7) |
| P-24 | principles.md:202-211 | shared kernel types; grouping Phase 2/4 | conforms |
| P-25 | principles.md:213-223 | context field; env tier later; CI installs unlocked | deviates (M5) |
| P-26 | principles.md:225-236 | typed channel groups; `sl.signal` Phase 4 | conforms (scheduled) |
| P-27 | engineering.md:124-129 | pushed; statement contradicts owner #30 | drift (F1) |
| P-28 | engineering.md:131-139 | identifier hook; personal paths and tracked memory notes | deviates (M2) |
| P-29 | engineering.md:141-151 | CLAUDE.md; serial scoped runs | conforms |
| P-30 | engineering.md:153-171; testing.md | id-named tests, all tiers per PR, enforced | drift (F4) |
| P-31 | engineering.md:46, 173-179 | no prose model; no blocking id check; ids in tests and comments | omitted (F5) |
| P-32 | engineering.md:47, 181-195 | ruff without C901/PL; mypy not strict; no duplicate check or size budgets | omitted (F6) |
| P-33 | engineering.md:49, 197-205 | scale ladder Phase 3 (design.md:939) | conforms (scheduled), M10 |
| P-34 | engineering.md:50, 207-215 | property and fault tests present (hypothesis, bounded probes); PR template lacks the brittleness checklist | omitted (F6) |
| P-35 | engineering.md:51, 217-226 | Phase 1 adversarial round and re-attack, findings in review-register.md | conforms |
| P-36 | principles.md:257-268; roadmap.md | hooks kept; nothing built | conforms (M3) |
| P-37 | principles.md:270-281 | registry refusal; no duplicate-code check | omitted (F6) |
| P-38 | principles.md:283-299; ADR-002 | key/context/reuse built; version-bump check unscheduled | conforms (M7) |
| P-39 | engineering.md:228-239 | ledger current at Phase 1 start | conforms (M13) |
| P-40 | engineering.md:241-252 | reviewer models not recorded in PR or commits; one PR | deviates (M6) |
| P-41 | principles.md:241-255; docs.md | preview built; Phase 1 data items missing | omitted (F9), M9 |

## Findings

### F1 (Important, drift) The owner's 2.x-branch correction did not land in the specs that bind agents

- **Departs from:** owner #30, "are PRs being merged into a 2.x branch or to alpha? should be the former." The
  correction landed in CLAUDE.md:20 ("2.x work merges into the `2.x` integration branch, never `alpha`") and
  in the 2x workflows.
- **Still saying alpha:**
  - P-27 statement, engineering.md:126-127: "Work in progress stays visible on a branch with a PR to `alpha`.
    2.x is built in place on `alpha`." This is an owner principle; only the owner amends it (principles.md:12).
  - design.md:832 "§6.1 Strategy (in place on `alpha`, ...)"; design.md:1042 "Where the work happens: in place
    on `alpha`".
  - STATE.md:54 "one PR per task to `alpha`"; plans/phase-1.md:63 and :200; plans/phase-1-hardening.md:67
    and :796 (protect `alpha` and `main`, not `2.x`).
  - PR #619 body: "Merging into `alpha` is left to the owner."
- **Why it matters:** a later session resumes from STATE.md and the plans (CLAUDE.md:10), not CLAUDE.md alone,
  and is told to open PRs to `alpha`.
- **Correction:** propose the P-27 amendment to the owner (2.x is built on `2.x`, PRs to `2.x`; owner #11's
  "in place on alpha" superseded by #30) and log it in principles.md; update design §6.1 heading and §8,
  STATE.md, both plans, the A-39 branch-protection step (protect `2.x`) and the PR #619 description.

### F2 (Important, omitted) State that later sessions can pick up is incomplete; the rulings are untracked

- **Departs from:** owner #26, "make sure a state is maintained where this or future sessions can pick up
  from"; P-15 principles.md:162-167 ("so that architectural intent survives across sessions"); STATE.md:3-4
  ("Update it at the end of every work session").
- **What the branch does:**
  - STATE.md is stale: ADR-001..009 (STATE.md:26, :93; ADR-010 exists), "hardening plan H1–H8" (STATE.md:47,
    :85; H9a-f and H10 have run), ledger path `.superpowers/sdd/phase-1/progress.md` (STATE.md:15; the live
    ledger is `.superpowers/sdd/phase-1-hardening/progress.md`), next actions to `alpha` (STATE.md:54).
  - The 52 controller rulings live only in `progress.md`, excluded by `.git/info/exclude:13` (`.superpowers/`).
    Several change decided behaviour and are recorded nowhere tracked except where folded into an ADR: H-R2
    (wire v2, store modes), H-R12 (bounds as code constants, not P-17 data), H-R31/H-R32 (tests named
    `test_a<n>_*`), H-R45 (two frozen types), H-R49 (test conformance task, benchmark harness assignment), and
    the scheduling of ADR-010's implementation ("implementation task after the full review").
  - PR #619 says "Progress and rulings are tracked in `STATE.md`", which is not true.
  - principles.md's change log ends at 2026-10-30 (principles.md:432-433); nothing logs the 2026-11-01
    changes (src/ pool, ADR-010, compat removal).
- **Correction:** move every ruling that changes or interprets a decision into a tracked file under the spec
  directory (review-register.md rows or a `rulings.md`), each citing the P-id/ADR it interprets; rewrite
  STATE.md's status, document map, ledger pointer and next actions; add the 2026-11-01 change-log entries.

### F3 (Important, drift) Phase 1 exit and §6.1 still describe a rewired, tested 1.x tree

- **Departs from:** ADR-009:19-23 as amended 2026-11-01 (owner #36): "`src/` is the 1.x source pool: frozen,
  not packaged ..., not tested and not linted ... 1.x code that breaks because core changed is not fixed
  there."
- **What the specs still say:** design.md:907 Phase 1 exit "1.x tests pass after the rewiring, and triage
  outputs are unchanged"; design.md:836-840 "1.x code may import core", "Moves are rewires ... every caller is
  rewired in the same PR"; plans/phase-1.md:597 "Exit requirement: triage outputs unchanged";
  plans/phase-1.md:832-835 (Task 15 Step 2: run the full 1.x suite on ORCD with the freeze's pass counts).
  `.github/workflows/2x-tests.yaml` no longer has a 1.x job (H10 removed `cpu-tests`), so the exit as written
  cannot be met and the real exit is undefined.
- **Correction:** amend design §6.1 (one-way imports, moves-are-rewires) and the Phase 1 exit in §6.2 to the
  source-pool model; rewrite plans/phase-1.md Task 15 Step 2 and Task 9's exit requirement.

### F4 (Important, drift) Test naming and CI cadence contradict P-30 and testing.md; the correction is queued but not run

- **Departs from:** testing.md:186-187 ("Forbidden in 2.x tests: spec, register or round ids in names, files or
  headers"); P-30 check engineering.md:170-171 ("No parametrize id is a register number"); testing.md:43 ("The
  default run is `logic` only"); testing.md:176-184 (PR: logic + contract; limits on label; use cases
  nightly/weekly); testing.md:106 ("Monkeypatching senselab internals is forbidden").
- **What the branch does:** 215 test functions named `test_a<n>_*`/`test_x*<n>_*` (e.g.
  `packages/senselab-core/tests/core_packaging_test.py:64`); round-named files
  `tests/adversarial/phase1_regression_test.py`, `reattack_*_test.py`; a test that requires the id names
  (`phase1_regression_test.py:1124-1128`); `tests/lint/ci_tiers_test.py` enforces that PR jobs select every
  tier, and `2x-tests.yaml` `core-only` runs `-m "logic or contract or limits or usecase"`; `pyproject.toml`
  `addopts` sets no default `-m logic`. The id naming was prescribed by rulings H-R31/H-R32 and
  plans/phase-1-hardening.md:875, a later ruling against testing.md. H-R49 schedules the fix ("H11") before
  Phase 1 exit; it has not run, and H-R49 itself is untracked (F2).
- **Correction:** run the conformance task: rename regressions by invariant and fold them into component test
  files, drop the name-enforcing test (keep the A-row coverage as register rows naming tests), add
  `-m logic` as the default, rewrite `ci_tiers_test.py` to enforce testing.md's cadence, move policy checks to
  `scripts/check_*.py`, replace internal monkeypatches with injected seams. Record H-R49 in a tracked file.
- **Status 2026-11-01: resolved by H11a (`fb1e774d`, `117a8641`, `621ad187`, `d574fe8e`) and H11b (`80b3aef4..5db856fa`).** Tests are named by invariant, the default run is `logic`, `ci_tiers_test.py` enforces the cadence, internal patches are replaced by seams and faults at the `os` boundary, and `tests/lint/internal_patches_test.py` guards the two declared child-only substitutions (`testing.md` section 3). H-R49 is recorded in `rulings.md`. Not resolved: further policy checks as `scripts/check_*.py`, cut back by D-1.

### F5 (Important, omitted) P-31's two checks are absent, and ids appear outside `specs/`

- **Departs from:** engineering.md:46 ("a small pinned model ... reads the prose in the diff ... The one
  blocking rule is structural: an exact-pattern check for spec, register and round ids (`F-n`, `R-n`, `rNN`)
  outside `specs/`"); owner #22 ("why not use agents to sharpen rather than CI through lists. perhaps some
  lightweight models like nimble could evaluate during CI runs").
- **What the branch does:** no prose-review step in `2x-tests.yaml` and no phase or plan task owns one
  (only design.md:1005 lists it as "replaced, not deferred"). No id-pattern hook in `.pre-commit-config.yaml`.
  Ids sit in code and config: `.importlinter` (`# H-R19: scripts, ...`, contract name "...the H-R19
  worklist"), `tests/lint/written_names_test.py:9` "(H-R40, H-R42)", and the F4 test names.
- **Correction:** add the blocking id check (pre-commit, exact patterns incl. `H-R\d+`, `A-\d+`, `XK-\d+`,
  outside `specs/`), fix what it finds, and assign the advisory prose-review job to a named phase in
  plans/ (it applies "to every commit", so Phase 1 or the first Phase 2 task).

### F6 (Important, omitted) The per-commit quality gates of P-32, P-34 and P-37 are not built

- **Departs from:** engineering.md:3 ("These apply to every commit"); engineering.md:47 ("ruff (including
  complexity `C901` and `PL` limits), mypy strict on core, a duplicate-code check, and file and function size
  budgets reported per PR"); engineering.md:28 ("mypy strict on core"); engineering.md:50 and P-34 check ("A
  brittleness checklist in the PR template"); engineering.md:55 ("The PR template carries this checklist");
  engineering.md:48 / P-37 check (duplicate-code check; "an adversarial 'find the duplicates' agent pass each
  phase").
- **What the branch does:** `pyproject.toml` `[tool.ruff.lint] select = ["ANN","D","E","F","I","TID251",...]`
  (no `C901`, no `PL`); `[tool.mypy]` and the pre-commit mypy hooks set no `strict`; no duplicate-code tool or
  size budget anywhere; `.github/PULL_REQUEST_TEMPLATE.md` is the generic 1.x template with no P-31..P-40
  checklist; the Phase 1 adversarial brief (plans/phase-1.md:838-844) has no duplicate-finding pass.
- **Correction:** add `C901`/`PL` with limits as data, `strict = true` for `packages/senselab-core` and
  `senselab-worker`, a token/AST duplicate check with a threshold, per-PR size budgets, and replace the PR
  template with engineering.md's checklist; add the duplicate pass to every phase's adversarial brief.

### F7 (Important, omitted) Licenses: no CI gate over what Phase 1 ships, and policy/sets/taint belong to no phase

- **Departs from:** P-23 principles.md:188-200 ("Policies are expressed against named license sets"; "A
  missing or unknown license raises by default"; check: "The CI gate equals the run-time default: it is
  applied to every catalogue model and locked package (transitive included) shipped at that phase"); design
  §4.13 design.md:609-622; owner #7, #11 ("missing licenses can be a flag that defaults to raising exception and
  the process could be generalized to licensing sets").
- **What the branch does:** only the record and `require_license` (`kernel/license.py:32-52`). Phase 1 ships
  `senselab-core` with numpy, pydantic, pyyaml, huggingface-hub and soundfile (packages/senselab-core/pyproject.toml;
  soundfile's wheel bundles libsndfile, LGPL) plus their transitive set in `uv.lock`, and no job derives or
  checks their license records. design §6.2 assigns only "Backfill licenses for the ported backends" (Phase 2,
  design.md:923); `license_policy`, the packaged named sets, `on_missing`, I-9 taint and `sl.licenses(plan)` are
  in no phase's build list (design.md:892-978).
- **Correction:** add the license gate to Phase 1 for the locked packages it ships (records harvested from
  PEP 639/classifiers, unknowns raise unless a dated manual record exists); assign the policy schema, named
  sets and `on_missing` to Phase 2 and I-9 taint plus `sl.licenses(plan)` to Phase 3 explicitly in §6.2.

### F8 (Important, omitted) ADR-010 is accepted but contradicted by design.md and owned by no plan

- **Departs from:** ADR-010 (accepted 2026-11-01): "The model is W3C PROV-DM ... One serialisation: PROV-O in
  JSON-LD ... `provenance@2.0` and its JSONL format are retired, with no reader."
- **What the branch does:** design.md:442-446 and §4.8 still describe the custom ProvStore with "BEP028 export is
  how runs are published" and the `provenance@2.0` file format; `runtime/provenance.py` writes and reads
  `provenance@2.0` (`runtime/provenance.py:177-197`). No phase in design §6.2, no plans/ task and no roadmap row
  schedules the change; its scheduling exists only in the untracked ledger ("implementation task after the full
  review").
- **Correction:** amend design §4.8 to point at ADR-010; add an ADR-010 task to plans/ (before Phase 1 exit,
  since the provenance format is a Phase 1 deliverable and pre-alpha bumps are free) or name its phase.

### F9 (Important, omitted) docs.md's Phase 1 data items are not built; benchmarks have no owner

- **Departs from:** docs.md:85-88 ("Phase 1: the data manifest and fetch, used first by the benchmark smoke
  suite; the fixture budget check; the generators that replace the committed fixtures"); testing.md:203-218
  (benchmarks as senselab pipelines in `benchmarks/`); P-41 check principles.md:253-255 ("Every example dataset
  has a manifest with hashes and a license record").
- **What the branch does:** fetch exists (`runtime/fetch.py`, `cli.py`), the budget exists, generators exist
  (`testing/synth.py`), but there is no data manifest in the tree and no `benchmarks/` directory or smoke
  suite. The ledger's audit says the per-task harness is unowned; H-R49 only promises to "assign it to a named
  phase" (untracked).
- **Correction:** either add one public-data manifest and a CPU benchmark smoke run that consumes it before
  Phase 1 exit, or amend docs.md's phasing; in both cases assign the benchmark harness to a phase in plans/.

### M1 (Minor, deviates) Principle ids are mis-cited

CLAUDE.md:42 "One implementation per behaviour (P-41)": that principle is P-37 (principles.md:270); P-41 is docs.
Rulings H-R31 and H-R42 and the H8 parked list in the ledger repeat the error. Correction: cite P-37.

### M2 (Minor, deviates) 1.x agent guidance and personal paths remain on the 2.x line

- AGENTS.md is the 1.x file (AGENTS.md:7 "requires Python 3.11-3.12", :28 `uv run pytest -n auto`), against owner
  #34/#35 ("simplify claude.md to be clear and specific to 2.x"; "is there a reason to keep 1.x stuff").
- `.claude/projects/-Users-<username>-software-sensein-senselab/memory/*.md` are tracked 1.x memory notes whose path
  carries a home-directory name (P-28 engineering.md:134: "no personal path enters the repository").
- Inherited 1.x specs carry personal ORCD paths, e.g. specs/20261005-breathing-pattern/design.md:50
  (`<cluster path>`), specs/20261001-nimble-second-opinion/design.md:50-55.
- Correction: rewrite AGENTS.md to point at CLAUDE.md (or delete it); remove the tracked `.claude/projects/`
  files; redact or drop 1.x spec directories not needed on 2.x (they remain in git history and on `alpha`).

### M3 (Minor, drift) design.md's triage text disagrees with ADR-009 and the ledger

design.md:953-958 "Triage is not ported as a pipeline ... Its decision layer and orchestration stay on the frozen
1.x tag" versus ADR-009:27-30 and migration-ledger.md §Triage ("pipeline: ports as a generic pipeline"). The ledger
also classes `b2ai_hints.py` as "pipeline" while describing it as "a dataset adapter"; under P-12 it is a dataset
adapter/profile, not pipeline code. Correction: amend design §6.2 Phase 4 to the ADR-009 wording; reclassify
`b2ai_hints.py` as dataset adapter (into the `b2ai-voice` profile).

### M4 (Minor, deviates) Core CI does not cover every non-EOL Python

P-22 check principles.md:179 ("core CI covers every non-EOL Python"); testing.md:182 (nightly full matrix). The
branch runs core on 3.11 and 3.14 (`core-only`) and 3.12 (`core-env`, release-gate subset); 3.13 runs nowhere and
there is no scheduled workflow. Correction: add 3.12/3.13 to a nightly matrix (or to `core-only`).

### M5 (Minor, deviates) CI installs core from unlocked resolution

P-25 principles.md:215-217 ("senselab never runs on unlocked or system state. senselab itself is installed into a
venv built from a lock"); design.md:697. `2x-tests.yaml` `core-only`/`core-env` and `2x-docs-preview.yaml` use
`uv pip install ./packages/...`, which resolves fresh versions. Correction: install from `uv.lock`
(`uv sync --locked --package ...` or an exported lock) in test and docs jobs; keep an unlocked range check only as
a separately labelled job if wanted.

### M6 (Minor, deviates) Reviewer models are not recorded where P-40 checks them

P-40 check engineering.md:251 ("each PR records its implementer and reviewer models, and they differ");
engineering.md:65. Implementers appear only as `Co-Authored-By` trailers; reviewer models appear only in the
untracked ledger; Phase 1 is one draft PR (#619, 177 commits) instead of "one PR per task" (plans/phase-1.md:63).
Correction: record per-task implementer and reviewer models in tracked form (PR body table or commit trailer
`Reviewed-by-model:`).

### M7 (Minor, omitted) The version-bump check is unscheduled; equivalence classes are a code literal

testing.md:139-140 and P-38 check principles.md:298 ("a source change without a version bump fails CI") is not
built and no phase owns it. `EQUIVALENCE` (`kernel/identity.py:268-272`) is "added by editing this declaration",
while ADR-002/I-4 (design.md:426) say classes come only from measured tolerance profiles, which P-17 puts in data.
Correction: assign the version-bump check to Phase 2 (when real backends exist); load equivalence classes from a
`data/` profile with derivation.

### M8 (Minor, deviates) The kernel decodes audio

plans/phase-1.md:23-24: the kernel holds "types, identity, declarations; no I/O frameworks"; design.md:553-558
puts decoding with what stays per modality. `kernel/signals.py:14, 179-212, 327` imports `soundfile` and opens and
decodes files inside `RegularGroup`/`SourceStamp`. Correction: move decoding and stamping to a runtime reader
behind a protocol, keeping the kernel's `Recording` decode-free (the ADR-conformance reviewer may rank this higher).

### M9 (Minor, deviates) docs.md names the wrong preview path

docs.md:47 says previews are at `sensein.group/senselab/pr-<N>/`; the 2.x previews are at `/senselab/2x/pr-<N>/`
(2x-docs-preview.yaml; README.md:53). Correction: update docs.md.

### M10 (Minor, drift) P-33 enforcement and the Phase 3 schedule disagree

engineering.md:49 says the scale ladder "runs per PR on touched modules" as a per-commit standard; design.md:939
builds `benchmarks/scale` in Phase 3. Phase 1 modules (store, fetch, codec) have no scale measurement. Correction:
state in engineering.md that the ladder applies from Phase 3, or add a minimal ladder for the Phase 1 hot paths.

### M11 (Minor, omitted) The pydra work the Phase 3 gate depends on has no owner

design.md:495-499: the Phase 3 gate falls back to senselab's own runner "if E1, D1 and F2 are not working on the
branch". STATE.md:49 "none implemented"; STATE.md:56 next action "Start pydra PR A0 ... and D1" is stale and no plan
in plans/ covers any pydra PR. Correction: add a pydra plan (A0, D1, E1, F2 at least) with a date before Phase 3.

### M12 (Minor, drift) testing.md contradicts itself on the contract marker

testing.md:48 gives the contract tier the marker `logic` (generated), while P-30 (engineering.md:158), testing.md
§2's own rule (one tier per marker) and the branch (`pyproject.toml` marker `contract`) use `contract`.
Correction: fix testing.md:48.

### M13 (Minor, note) P-39 bookkeeping before Phase 1 exit

The ledger's marks are `19834fdd`; `origin/alpha` has four CI commits since (#620, touching
`src/tests/scripts/docs_deploy_test.py`), and `origin/alpha` is not an ancestor of the branch, while ADR-009:24-26
makes "merge `alpha` into the 2.x line" the reconciliation step. A stack of draft 1.x triage PRs (#607-#618,
"dataset profiles, B2AI out of triage") is in flight against `alpha` and bears on P-12 and the triage
classification; the ledger does not mention it. Correction: run plans/phase-1.md Task 15 Step 4 (re-scan, merge
or record), and note the in-flight PR stack in migration-ledger.md.

## Counts

Important: 9 (F1-F9). Minor: 13 (M1-M13). Critical: 0.
