# Decisions, 2026-11-01

The owner reviewed the consolidated findings of the 2026-11-01 full review
([`reviews/full-review-2026-11-01/consolidated.md`](reviews/full-review-2026-11-01/consolidated.md)) and
answered: "accept recommendations and move specs to the docs PR - clarify the decisions".

Each decision below states what changes, which documents carry it, and which findings it resolves. An ADR
that changes gets a dated amendment section. Its original decision text stays where it is, and the
amendment says what replaces it.

## D-1. Cut the guard and gate layer back to the design

- **Keep:**
  - the import-linter contracts (layers, worker independence, `packages-not-1x`, digests-are-tagged);
  - the identifier check (P-28);
  - the fixture budget at docs.md's numbers (256 KB per file, 2 MB total);
  - the workflow-expression guard.
- **Delete:**
  - the spelling-based lints that chase in-repo bypasses: the digest-binding data-flow and strict-mode
    checks, the written-names lint, and the dynamic-import allowance machinery;
  - `moved_modules_test.py`, because the import contract supersedes it;
  - every test whose only job is to make another test exist, such as the coverage test that requires
    finding-id names.
- **Add** the agent prose review the owner asked for (item 22). A small model reviews each PR's diff
  against the principles and engineering checklist and comments; it does not block. This is the
  "sharpen with agents, not lists" route.
- **Carried by:** engineering.md (P-31, P-35) and design §5.
- **Resolves:** Ar F-1, Ru I-2, Ru I-10.

## D-2. Locks are leases by rename and O_EXCL, as ADR-003 decided

- **Revert:**
  - the `flock`, heartbeat and takeover protocol from H-R40..H-R42;
  - the negative-timeout and shared modes that only 1.x used.
- **Keep from H9b**, because they are path safety and not lock semantics:
  - no in-place modification under a storage root;
  - `O_NOFOLLOW`/`dir_fd` component walks;
  - wrong-kind objects quarantined;
  - private staging.
- **Carried by:** ADR-003, which records the reverted ruling in an amendment.
- **Resolves:** Ru I-13, A I-7.

## D-3. Canonical text: NFC, except text with code points unassigned in Unicode 13.0

- **The rule (H-R34) stays.** Text containing a code point unassigned in Unicode 13.0 is kept as given and
  never normalised. All other text is NFC.
- **The reason is portability, not triage.** NFC depends on the interpreter's Unicode version. Python 3.10
  and 3.12 normalise one string differently (re-attack XC-1, reproduced). Plain NFC would give the same
  value different digests on different supported interpreters.
- **Carried by:** ADR-006, in an amendment that replaces the "UTF-8 NFC" line and records H-R33 (refuse)
  as superseded.
- **Resolves:** Ru I-6, A (C-1..C-6 where they concern text).

## D-4. ADR-010 details and its phase

- **An activity's `@id` identifies the execution.** It is the content address of the execution record,
  not the logical key. The logical key is a `senselab:` property of the activity. One key can therefore
  carry several executions: one per context, plus reuse activities.
- **The plan link is staged.** It is recorded once ADR-001's graph document exists (Phase 3). Until then an
  activity carries its node id and logical key, with no `prov:hadPlan`.
- **Phase 2 implements ADR-010.** The work covers:
  - the PROV-O JSON-LD store;
  - the `senselab:` vocabulary;
  - removing the triage-shaped token families and helpers (`$path`, `$enum`, `$dec`, `$bytes`, `$set`,
    `EnumMember`, `FrozenDict`, `revised`, `recorded`, `plain_values`);
  - the redaction ADR-007 requires for exports.
- **Carried by:** ADR-010 (amended), design §6.2 (Phase 2 deliverable), ADR-006 (its provenance sections
  marked superseded by ADR-010), and ADR-007.
- **Resolves:** A I-3, Ru C-1, A I-2, A I-6.

## D-5. The release gate is minimal until the first 2.x release

- **What it checks now:**
  - the version grammar;
  - exact sibling pins;
  - that each distribution builds and installs from `dist/`.
- **Deferred** to the first 2.x release and specified then: member-set completeness, the sdist rebuild
  identity, strict `WHEEL`/`METADATA` parsing, per-floor installs, cache isolation and import-every-module.
- **Carried by:** design §5 and ADR-006 (version and tag grammar).
- **Resolves:** Ru I-3.

## D-6. `src/` is 1.x as it stands on alpha

- **Restore `src/` to its state at the freeze commit (19834fdd).** That undoes the edits 2.x rounds made to
  97 files. Then merge `alpha`, so the pool carries 1.x as it stands (P-39).
- **2.x never edits `src/`.** A use case leaves it only by a rename-only `git mv` commit into `packages/`,
  followed by a modify commit.
- **Carried by:** ADR-009 and migration-ledger.md.
- **Resolves:** Ru I-9.

## D-7. Design lives in the docs PR, code in the code PR

- **#617 (`senselab-2x-api` → `2.x`) is where design is decided.** It carries every spec change: ADRs,
  principles, design, plans, registers, `STATE.md`, rulings and reviews.
- **#619 (`2x/phase-1`) carries code and tests only.** It merges `senselab-2x-api` to stay current. Once
  #617 merges into `2.x`, #619 is retargeted to `2.x`.
- **Specs record decisions and their reasons.**
  - As-built detail that hardening wrote into ADR-003, ADR-006, design.md and docs.md moves out. Either it
    becomes a dated amendment, if it changed a decision, or it is removed, if it only describes code.
  - No spec cites the git-excluded ledger. Rulings are cited from `rulings.md`.
- **Carried by:** ADR-000 (the amendment rule), principles.md (change log), and `STATE.md`.
- **Resolves:** Ru C-2, Ar F-2, Ru I-5, D-1, D-2, D-3, T F1, T F2, T F3, Ar F-4.

## Follows from the decisions without a further choice

- **Tests and benchmarks:** H-R49 (test conformance), plus the benchmark harness assigned to Phase 2.
- **Identity:**
  - portable model identity: content, not path or URL (A I-1);
  - read `DEFAULT_REUSE_POLICY` (Ru I-1);
  - equivalence classes as data with an owner for verify sampling (A I-8);
  - assign owners for the clock re-binding (A I-9) and the source-digest CI check (A I-10).
- **Kernel:** move audio decoding out of the kernel, and decide whole-file hashing against ADR-005
  (Ar F-3, A I-5, R-109).
- **pydra:** correct environment-in-key in engine-choice, extension-points, workflow-engines and the
  proposal, and add checksum equality to the proposal (Ar F-5, F-6). Assign owners for the pydra PRs.
- **Faults:** one transient classifier, with `OSError` not blanket-transient (Ar F-7).
- **Gates:** the engineering gates and the license gate (T F5–F7).
- **Process:** P-40 roster and worktree isolation; parked items moved into tracked registers (Ru I-11,
  I-12).
