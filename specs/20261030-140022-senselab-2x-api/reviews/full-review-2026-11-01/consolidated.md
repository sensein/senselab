# Consolidated findings, full review 2026-11-01

This file merges the five reviews in this directory, which are traceability (T), ADRs (A), architecture (Ar),
rulings audit (Ru) and delivery (D). Duplicates are merged and the source ids are kept. Items marked
**owner** need the owner's decision before they are fixed. Everything else follows from the owner's
requests, the principles and the ADRs.

## Decisions for the owner

1. **The guard and gate layer (Critical).** Hardening grew about 9,000 lines of list-driven guards and
   gates: a 1,092-line release gate plus its tests, the fixture budget, and a 3,800-line lint layer. That is
   about 76% of the package source (Ar F-1, Ru I-2/I-3). It contradicts items 14, 15 and 22 ("why not use
   agents to sharpen rather than CI through lists") and P-35. Proposal:
   - Cut it back to what design §5 and engineering.md ask for.
   - Delete the escalated digest-bypass lints and keep the import contract.
   - Reduce the release gate to version, pins and an installable dist.
   - Add the agent prose review that item 22 asked for, which no phase builds yet.
2. **Lock semantics (Ru I-13).** H-R40..H-R42 introduced `flock`, a heartbeat and takeover. ADR-003
   decided on rename/O_EXCL only and rejected fencing. Either amend ADR-003 or revert to its design.
3. **Unicode in canonical text (Ru I-6, A).** H-R33/H-R34 changed ADR-006's "UTF-8 NFC" twice, on a
   triage premise (emoji in LLM reviews). Either keep the Unicode-13 rule or return to plain NFC.
4. **ADR-010 details (A I-3).**
   - "An activity's `@id` is its logical key" merges executions, because one key can have several
     results (one per context, plus reuse activities). The proposed fix is that the activity id is the
     execution and the logical key is an attribute of it.
   - "The graph is the plan" needs ADR-001/004 artifacts that arrive in Phase 3/4.
   - Decide which phase implements ADR-010.
5. **The release gate in Phase 1 (Ru I-3).** H-R1's premise for keeping it lapsed with item 30.
   Decide whether to defer it to the first 2.x release.
6. **The state of `src/` (Ru I-9).** 2.x rounds edited 97 files under `src/` (Task 7/9 rewires, H9c,
   H-R42). The pool should equal 1.x as it stands. Proposal: restore `src/` to the freeze state, then
   merge `alpha`.

## Critical

- **Triage-shaped machinery is still in core** (Ru C-1). This covers the token families `$path`, `$enum`,
  `$dec`, `$bytes`, `$set`, and `EnumMember`, `FrozenDict`, `revised`, `recorded` and `plain_values`
  (provenance.py). They came from keeping 1.x triage green (P-14, P-36, ADR-002). Remove them when
  ADR-010 is implemented. `$enum` also hashes `<module>.<qualname>` into durable ids (A I-6).
- **Implementation rewrote the design** (Ru C-2, Ar F-2). ADR-006 grew from 250 to 3,793 words and
  ADR-003 from 459 to 2,090. design.md grew from 860 to 1,066 lines, and docs.md grew too. All of it was
  as-built detail and ruling content, under unchanged "accepted 2026-10-30" headers and citing ledger ids.
  Restore ADRs and design to decisions: move implementation detail out, and record each real change as a
  dated amendment.

## Important

- **State and the branch target.**
  - STATE.md is stale (D-2, T F2).
  - plans/, P-27 (engineering.md:127), design.md:832/1042, design §6.1/§6.2 exit and §8, and the bodies
    of #617 and #619 still say "in place on alpha", or require 1.x to be rewired and its tests to pass
    (D-3, T F1/F3, Ar F-4).
  - `senselab-2x-api` (#617) lacks `2.x` merged in, so it runs 1.x CI (D-1).
  - principles.md has no change-log entry for 2026-11-01 (T F2).
- **Kernel boundary and cost.** The kernel imports soundfile and decodes audio (Ar F-3, T M8). Since H9c
  it hashes the whole file at open and again on every read, which contradicts ADR-005's "never hashed
  whole" (A I-5). The owners disagree, Phase 2 versus Phase 3 (R-73, R-109).
- **Identity.**
  - A `local:` or `url:` model puts its path or URL into the portable key (A I-1).
  - `DEFAULT_REUSE_POLICY = EQUIVALENT` is dead code, although the owner confirmed `equivalent` as the
    default (Ru I-1).
  - `EQUIVALENCE` is a code literal, with no measured classes and no verify sampling (A I-8, T M7).
  - The clock re-binding bullet in ADR-002 is not implemented and has no owner (A I-9).
  - ADR-002's source-digest CI check is missing (A I-10).
- **Privacy.** The BEP028 exporter writes file names and paths (A I-2), which contradicts ADR-007.
- **Formats.** ADR-006 contradicts itself on previous-major readers versus the tag bumps made without
  migrations. It also has six internal contradictions, C-1..C-6 (A I-4). NaN and big-integer rules
  accreted across six surfaces, and the signal NaN rule sits in design.md rather than ADR-005 (Ru I-7).
- **pydra.**
  - engine-choice G3, pydra-extension-points.md, workflow-engines.md and proposal §3.5 put
    environment identity into the key, against ADR-002 (Ar F-5).
  - No proposal item delivers "pydra checksum equals senselab key". K1 is missing from the citation
    tables (Ar F-6).
  - No plan owns the pydra PRs that the Phase 3 gate depends on (T M11).
  - Phase 1 has no graph document, so pydra compatibility is unshown (A I-11).
- **Faults.** retry.py treats every `OSError` as transient. It carries an optional undeclared `requests`
  import and an environment knob, and it is a second classifier next to `storage_fault` (Ar F-7).
- **Tests and benchmarks** (audit, T F4/F9). This item is H-R49's scope.
  - 264 tests are named for finding ids, and one test enforces those names.
  - Every tier runs on every PR, and there is no default tier.
  - 86 tests monkeypatch internals.
  - Wall-clock assertions sit in the fast tiers.
  - No benchmark harness, data manifest or smoke suite exists, and none has an owner.
- **Engineering gates missing** (T F5/F6). These are absent: mypy strict, ruff C901/PL, a duplicate-code
  check, size budgets, the P-34 PR checklist, the P-37 duplicate-finding pass, the P-31 ids-outside-specs
  check, and the agent prose review.
- **Licensing** (T F7). P-23 has only the License record. There is no CI license gate over locked
  packages (soundfile/libsndfile and transitive dependencies), and no license sets, `on_missing` or taint
  handling.
- **Process** (Ru I-11/I-12, T M6).
  - Implementers shared one worktree, and all ran on opus. The P-40 roster calls for sonnet outside the
    kernel, and for worktree isolation.
  - Phase 1 is one PR with 177 commits.
  - About 51 parked items live only in a git-excluded ledger. Examples: local model identity includes
    `.git`, `_bounded` is copied four times, branch protection.
  - Every fix round grew the code: package source went from 6,043 to 11,853 lines and tests from 6,535
    to 23,252, against the plan's "No new layers".
- **1.x leftovers in 2.x** (Ru I-10, A I-7).
  - `moved_modules_test` duplicates the import contract.
  - The fixture budget is bound to the 1.x freeze SHA.
  - ADR-003 has 1.x lock paragraphs and a stale count of `ALLOWED_RAW` (the table has 3 entries).
  - `SharedFileLock` keeps negative-timeout and shared modes that only 1.x used.
  - About 60 1.x scripts are still lint roots.
  - design.md:953 says triage is "not ported", which contradicts H-R52.
  - AGENTS.md is 1.x guidance (T M2).

## Minor

These are listed in each review's file: T M1–M13, A minors, Ar minors, D-7..D-13 and Ru minors. Examples:
- CLAUDE.md cites P-41 where it should cite P-37.
- `uv pip install` is unlocked in CI.
- The docs.md preview path is stale.
- Core CI does not run 3.13.
- `hardware.py` has dead CUDA index code.
- The pydra benchmarks are copied into both repos.

## What held

- The worker is stdlib plus numpy.
- The kernel-below-runtime and layer contracts are enforced.
- Core installs no ML framework.
- Keys hash for exactness only, and `equivalent` is the documented default.
- vLLM-omni is kept.
- Execution (pools, Slurm, stacks) is correctly unbuilt, for Phase 2–3.
- Every registered digest tag hashes what ADR-006 says, apart from A I-1 and A I-6.
- The 2.x and 1.x workflows are separate.
- The 2.x tree has no identifiers.
- Today's corrections landed: the compatibility read path is gone, `src/` is a pool, and triage tasks
  port.
