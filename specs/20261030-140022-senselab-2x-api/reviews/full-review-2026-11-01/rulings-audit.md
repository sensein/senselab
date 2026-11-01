# Full review: audit of the controller's rulings (R1..R42, H-R1..H-R52) at 3a468aff

Reviewer angle: the controller's own rulings, treated as suspect. Authority order is the brief's: owner
requests (`owner-requests.md`, cited `OR-n`), then `principles.md` / `engineering.md`, then the ADRs, then the
other specs. Ledger line numbers: `P1:n` = `.superpowers/sdd/phase-1/progress.md`, `H:n` =
`.superpowers/sdd/phase-1-hardening/progress.md`. Spec paths are relative to
`specs/20261030-140022-senselab-2x-api/`.

Corrections the owner already made are re-checked only for whether they landed (section 6).

**Counts:** Critical 2, Important 12, Minor 6.

---

## Critical

### C-1. A ruling made the 1.x suite a gate on 2.x core, and its consequences are still in core (process ruling H:140; H-R29, H-R30, H-R34, H-R38 non-compat half, H-R45)

- **What was decided.** After the H8a regression, the ledger records "Process ruling: every Linux verification
  from here runs src/tests (1.x) as well as packages/" (H:140). From then on, 1.x triage failures set the
  design of core:
  - H-R29 (H:139). "Provenance digest is total over every value the pipelines record". The trigger was triage's
    `-inf` level measurements (H:138), and the ruling withdrew "1 vs 1.0 distinctness" to keep them passing.
  - H-R30 (H:144). Declares the provenance value type as "JSON values ∪ tagged families Path, Enum, ndarray,
    Decimal, bytes, set/frozenset, numpy scalars". These are exactly the types that 1.x callers had passed
    through `json.dumps(default=str)` (H:143).
  - H-R34 (H:174). Replaces H-R33's refusal of post-Unicode-13.0 text because it "breaks provenance totality
    (H-R29/H-R30) — e.g. modern emoji in LLM reviews". LLM reviews are triage.
  - H-R38 (H:201), its non-compatibility half. "1.x mutators move to an explicit copy-with-changes API",
    "write∘read∘write idempotent, generated over every family". H9c then edited 13 1.x production files and 30
    test files and AST-rewrote 102 assertions (H:204).
  - H-R45 (H:257). Keeps two frozen mapping types: `FrozenDict` exists so that 1.x read-back consumers get a
    `dict`.
- **Authority contradicted.**
  - OR-3, "backwards compatibility is not required".
  - P-14, `principles.md:154-155`: "There are no shims, aliases, deprecation paths or parallel fields.
    Migration is documented, not supported in code."
  - OR-17 / P-36, `principles.md:262-263`: "Triage's decision layer is rebuilt as a flywheel application after
    2.x rather than ported."
  - ADR-002 as amended (`adr/ADR-002-identity-v2.md`, "Params are canonical by declaration"): "The registry
    refuses `set`, `frozenset`, `Decimal`, `datetime`, `Path`, `bytes`". Core therefore runs two value domains,
    a strict one for keys and a 1.x-shaped one for provenance.
  - P-37, `principles.md:272`: two frozen mapping types.
- **What is still in core at 3a468aff.** The owner caught H-R38's read path, and H-R47 withdrew it. The writer
  side was not withdrawn:
  - `packages/senselab-core/src/senselab_core/runtime/provenance.py:205`:
    `_PROV_TOKENS = {"$f","$int","$path","$enum","$nd","$dec","$bytes","$set"}`;
  - `EnumMember` (:221), `recorded` (:358), `plain_values` (:371), `revised` (:684-687), and `MTIME_KEY` (:62);
  - `FrozenDict`, documented in `adr/ADR-006-formats.md` "A caller reads every value decoded".
  - provenance.py is 1,163 lines, the largest module in core.
- **What it should have been.** 2.x core is verified against 2.x tests. A 1.x failure caused by a core change
  is a 1.x matter. ADR-009 now says this itself: "1.x code that breaks because core changed is not fixed there."
- **Do now.** Fold this into the ADR-010 implementation:
  - provenance attributes are JSON values (PROV typed literals);
  - delete the `$path/$enum/$dec/$bytes/$set` families, `EnumMember`, `FrozenDict`, `recorded`,
    `plain_values` and `revised`;
  - one frozen mapping type;
  - put H-R34 back to the owner (see I-6).

### C-2. ADRs and specs were rewritten in place by rulings, with no Status change, no owner sign-off, and citations to a git-excluded ledger (R2, plus the H-R rulings that edit ADR-003, ADR-006 and design.md)

- **What was decided.**
  - R2 (P1:26): "all commits for Phase 1, including spec edits … go on `2x/phase-1`; senselab-2x-api (#617)
    is frozen as the design PR".
  - From then on, every ruling that changed a decision was written straight into the ADR text by an implementer
    or by the controller:
    - `adr/ADR-006-formats.md` grew from 250 to 3,793 words;
    - `adr/ADR-003-storage-leases.md` grew from 459 to 2,090 words;
    - `design.md` grew from 6,929 to 9,833 words;
    - `wire-protocol.md` is new, at 4,651 words.
  - Both ADRs still read "**Status:** accepted, 2026-10-30". Neither "Alternatives rejected" section was updated.
  - ADR-006 cites rulings and finding ids as its authority: 16 `H-R` ids, plus `XC-1`, `XC-6`, `H9c`, `H9d`
    and others.
  - The ledgers that define those ids are excluded from git (`.git/info/exclude:13 .superpowers/`), so the
    citations are dangling in the repository.
- **Authority contradicted.**
  - `adr/ADR-000-index.md:7-8`: "Each record has a Status, Context, Decision, Consequences and Alternatives
    rejected. A record that is superseded keeps its number and names its successor."
  - `principles.md:12`: "Agents propose changes in PRs".
  - The brief's authority order: rulings are not authority, yet they are now cited inside the ADRs as if they
    were.
  - P-31 / `engineering.md:46`: "History, measurements and rejected alternatives go to `specs/`". ADR-006 is
    now mostly a change history: the tag bumps of H9b, H9c, H9d and H9e, and triage history ("TAXONOMY's
    evidence fold wrote until 2026-09-13").
- **Decisions changed this way.** The specific ones are listed under I-6 (Unicode), I-7 (NaN), I-3 (version
  and tag grammar) and I-13 (lock semantics).
- **What it should have been.** A decision change is a new ADR, or an amendment section with a date, a status
  and the owner's approval, on the design PR the owner reviews. Mechanism detail belongs in module docs.
- **Do now.**
  - Split each of ADR-003 and ADR-006 into the decision, at its original length plus dated amendments, and
    mechanism notes moved to `design.md` or module docs.
  - Strip ruling and finding ids from committed specs.
  - Put the changed decisions (I-6, I-7, I-3, I-13) to the owner as explicit amendments.

---

## Important

### I-1. The owner-confirmed reuse default is not wired (R24)

- **What was decided.** R24 (P1:114), plus the interface note at P1:120: "acceptable(stored, current, policy,
  *, backend) with no policy default".
- **What the code does.**
  - `packages/senselab-core/src/senselab_core/kernel/identity.py:267` defines
    `DEFAULT_REUSE_POLICY = ReusePolicy.EQUIVALENT`.
  - Nothing reads it except `tests/kernel/identity_test.py:316`.
  - `ResultStore.get(key, context, policy)` (`runtime/store.py:817`) requires a policy argument.
- **Authority contradicted.**
  - `principles.md:423`: "owner confirmed `equivalent` as the default reuse policy (P-38)". OR-21: "reuse
    default can be equivalent".
  - `engineering.md:47`: "No dead code".
- **Do now.** Make `policy` default to `DEFAULT_REUSE_POLICY` at the one public entry point, or delete the
  constant until the executor exists. Do not leave a dead constant pinned by a test.

### I-2. The tagged-digest rule was enforced by escalating guards against an attacker who does not exist (H-R8 → H-R17 → H-R19 → H-R36)

- **What was decided.** Each round chased a bypass found by an agent writing deliberately evasive in-repo code:
  - H-R8: an AST spelling sweep;
  - H-R17: a capability allowlist on `hashlib`;
  - H-R19: 47 allowlisted modules, mostly 1.x;
  - H-R36: typed `file_digest`/`npy_digest`, a raw-digest-owners contract, one shared resolver over import
    chains, attribute chains and literal `__import__`, a ban on `builtins.__import__`, and ruff TID251/B009/S102/S307.
- **Result.**
  - `tests/lint/digest_bindings_test.py` (233 lines), `import_resolver_fixture.py` (262) and
    `digest_contract_test.py`.
  - ADR-006 still lists accepted holes: "computed-name lookups, npy_digest over text-built arrays, owner modules
    hash any bytes" (H:188).
  - H-R18 (H:91) had already ruled the opposite for the reproduce adapter guard: "a guard against accidental
    first-party imports, not a sandbox … no further hardening". The same reasoning was not applied here.
- **Authority contradicted.**
  - P-35, `engineering.md:51`: "Each break is fixed **by an architectural change**". The architectural fix was
    made in H9e: no untagged digest API exists. Every guard after that point is patch-per-spelling.
  - OR-14: "agents should be used to break system to set architectural standards not more brittleness".
  - P-32: thin surfaces.
- **Do now.** Keep the one tagged API (`identity_digest` / `file_digest` / `npy_digest`) and one import-linter
  contract that only the digest module imports `hashlib`. Delete the resolver, the bindings guard and the
  spelling regressions.

### I-3. The release gate was kept in Phase 1 on a premise the owner then removed, and grew into a 3,000-line subsystem (H-R1, H-R21..H-R25, H-R28, H-R43, H-R44)

- **What was decided.** H-R1 (H:6): "H7 (release gate) stays inside Phase 1 — alpha publishes prereleases on
  merge".
- **How the premise lapsed.** OR-30 moved 2.x merges to `2.x`, and the ledger at H:262 says "2.x merges cut no
  1.x release". The adversarial lead had already said "A-11/A-29 block first release", not Phase 1 exit (P1:238).
- **What was built anyway.**
  - `scripts/release_gate.py` is 1,092 lines, and `tests/release_gate_test.py` is 1,843 lines.
  - It took 4 fix rounds in H7, then H9f plus 2 more rounds.
  - `design.md` §5 "Versioning" went from two bullets (original: "A single git tag and hatch-vcs … An exact
    same-version pin") to roughly 60 lines (`design.md:706-760`). These were written after the fact: "states
    the release gate … as built" (commit d74e64ad).
- **Rulings within it that depart from authority.**
  - **H-R43 (H:243)** accepts "the repo's established `X.Y.Z-(alpha|beta|rc).N` form (1.x auto-release tags)".
    This carries a 1.x convention into 2.x release tooling, contrary to P-14. It also justifies the reduced
    root import check with "caught by CI's cpu-tests instead". H10 removed `cpu-tests` from `2x-tests.yaml`,
    yet `design.md:750` still cites it. Separately, the root distribution is now metadata-only, so the whole
    root-import branch is moot.
  - **H-R44 (H:250)** pins build backends exactly in every `[build-system]`: `hatchling==1.33.0`,
    `hatch-vcs==0.5.0`, `setuptools-scm==10.3.4`, `vcs-versioning==2.6.0`, `packaging==26.3`
    (`packages/senselab-core/pyproject.toml:2`). The reason is "so the rebuild comparison stays exact". That
    trades build robustness for gate exactness, which is the brittleness P-34 forbids ("depend on an
    accident").
  - **H-R24 (H:110)** set a version floor and a tag grammar. These were never in an ADR. Version grammar is a
    design decision changed without an amendment (see C-2).
- **Do now.**
  - Cut the gate to: build, install each wheel alone in a fresh venv, import, and check the sibling pin.
  - One tag grammar: PEP 440 canonical.
  - Unpin `packaging`, `setuptools-scm` and `vcs-versioning` from `[build-system]`.
  - Delete the root-import special case and the stale `cpu-tests` sentence.

### I-4. Regression tests were shaped around a historical commit, which caused the 129 GB disk incident (H-R31, H-R32, H-R37)

- **What was decided.**
  - H-R32 (H:154): "a test_a<n>_* counts only if it fails at 94e81479 … move new-module imports out of the
    module level (or split the file) so base runs reach the assertion".
  - Running those base checks locally filled the disk: a 129 GB file and an ENOSPC incident (H:194).
  - H-R37 (H:195) then added `FiniteSource`, `run_bounded` and RLIMITs, plus "A contract test asserts each
    test_a<n>/test_x* touching I/O runs under the shared limiter". That test enforces the finding-id naming
    the owner later rejected.
- **Authority contradicted.**
  - P-30, `engineering.md:163-164`: "named for the behaviour they check. Parametrisation enumerates meaningful
    cases, never encoded history".
  - `testing.md:187`: "spec, register or round ids in names, files or headers" are forbidden.
  - OR-28 / OR-27: the local machine is not to be used for heavy runs.
  - `testing.md:107-112` now carries the incident narrative and a ruling id inside a normative spec.
- **Do now.**
  - In H11, delete the name-keyed limiter contract along with the id names.
  - Keep `run_bounded` only where a test really drives unbounded I/O.
  - Drop the "fails at 94e81479" structure: imports go back to module level.
  - Remove the H-R37 sentence and the 129 GB story from `testing.md`.

### I-5. One long-lived branch, a draft PR with no CI, and design edits hidden from the design PR (R1, R2, R9; P-27 never amended)

- **What was decided.**
  - R1 (P1:25) replaced the plan's one PR per task with a single integration branch.
  - R9 (P1:42) accepted that the draft PR #619 ran no CI. The first CI run then failed at collection in three
    jobs (H:196).
  - R2: see C-2.
- **Authority contradicted.**
  - `engineering.md:39`: "small reviewed changes".
  - P-27, `engineering.md:126-127`: "Work in progress stays visible on a branch with a PR to `alpha`. 2.x is
    built in place on `alpha`". This text is now false: OR-30 redirected the work to `2.x`, and P-27 was never
    amended. A principle that contradicts the owner's current direction is drift.
- **Do now.**
  - Amend P-27 (owner text) to name `2.x`.
  - From Phase 2, one PR per task into `2.x`, with CI running on drafts.

### I-6. The Unicode policy was changed twice by ruling, on a triage-driven premise (H-R33, H-R34)

- **What was decided.**
  - The original ADR-006 says "text: UTF-8 NFC".
  - H-R33 refused code points unassigned in Unicode 13.0.
  - H-R34 reversed that: such text is "NOT NFC-normalised and is encoded under a distinct tag".
- **Cost.**
  - A checked-in Unicode table and a fingerprint over NFC/NFD outputs that had to be reworked for 3.14 (H-R35).
  - A bump of every text-bearing tag.
  - Roughly 30 lines of ADR-006.
- **Why the premise no longer holds.** The trigger, emoji in triage's LLM reviews, is gone from 2.x (C-1). The
  3.10 floor that forces Unicode 13.0 is itself past end of life (P-22, `principles.md:171`; 3.10 reached EOL
  in 2026-10). It is kept only because worker envs may use old Python.
- **Authority contradicted.**
  - ADR-006's own decision line, changed without an amendment (C-2).
  - P-14 / OR-37: the change was driven by triage.
- **Do now.** Owner decision. The simplest consistent options are:
  - (a) NFC only for identifier and path fields, with free text hashed as given;
  - (b) H-R33's refusal for identity text.

  Either removes the "raw family".

### I-7. NaN policy accreted per ruling, and the signal rule lives in design.md under a ruling id instead of ADR-005 (R13, R14, H-R27, H-R29, H-R38)

- **Current rules.**
  - Wire `.item()`: NaN refused (R13).
  - JCS: refused.
  - Provenance: `{"$f":"NaN"}` token (H-R29).
  - Arrays: one quiet NaN.
  - `Audio`: refuses NaN at construction and decode.
  - Irregular groups: hold NaN as the "missing-sample marker" (H-R38).
  - That is `design.md:197-200` "Signals and NaN (H-R38)", and `kernel/signals.py:114-118`.
- **Big integers have three encodings.** R14 allows integral floats above 2^53; H-R27 encodes `mtime_ns` as a
  decimal string; H-R29 uses `$int` tokens.
- **Authority contradicted.**
  - P-26, `principles.md:226-231`: "An algorithm that applies to any time series is written once over the
    general signal type". A regular-versus-irregular NaN split is a per-type special case.
  - ADR-005 is the signal model, unchanged at 280 words, and carries no NaN rule.
- **Do now.**
  - One admission rule for every signal: missing samples are explicit (a mask or gap intervals), not NaN.
  - Put it in ADR-005 as a dated amendment.
  - Once C-1 removes the provenance `$f`/`$int` families, one integer rule remains (I-JSON or refuse).

### I-8. Tags were bumped repeatedly on formats that were never released, contradicting an earlier ruling (H-R29 vs H-R35, H9b, H9c)

- **What was decided.**
  - H-R29 (H:139): "Tag stays provenance-id@2 (unreleased)".
  - H-R35 (H:178): "Bump every tag whose payload can carry text".
  - The bumps after that: store@1→@4, provenance-id@2→@4, output@3, logical-key@3, context@3, weights@3,
    data-manifest@3.
  - ADR-006 records each bump as history (`adr/ADR-006-formats.md:186-218`).
- **Authority contradicted.** H-R2 (H:7), "pre-alpha format bumps without migration"; P-19 (bumping is free);
  P-31 (no history outside `specs/` changelogs).
- **Do now.**
  - Reset every 2.x tag to `@1` at the first 2.x release.
  - Delete the bump history from ADR-006; git holds it.

### I-9. The 1.x source pool was edited by 2.x rounds and now conflicts with H-R51's porting model (R27, Task 7/9, H9c, H-R42)

- **What was done.**
  - Task 7/9 rewired 118 1.x callers onto `senselab_core`.
  - H9c edited 13 1.x production files.
  - H-R42 (H:236) made "1.x build.file_lock delegate to SharedFileLock".
  - `git diff --shortstat 19834fdd 3a468aff -- src/senselab`: 97 files, +316/−2,596. `src/tests`: 118 files.
- **Authority contradicted.**
  - ADR-009 as amended: "1.x code that breaks because core changed is not fixed there", and "Merge `alpha` into
    the 2.x line first … then a rename-only `git mv`".
  - P-39: every one of those 97 diverged files is a merge conflict waiting for the first port.
- **Do now.** Restore `src/` to the freeze branch, except for the modules core replaced and deleted, so that
  `merge alpha` then `git mv` is clean. Otherwise, record each diverged file in `migration-ledger.md`.

### I-10. Guards built for 1.x survive their purpose (R3, R38, R39, H-R9, H-R41, H-R43, H-R44/H9f)

- `tests/lint/moved_modules_test.py` (385 lines, MOVED names `senselab.utils.file_lock` and others) is
  superseded by the one-line import-linter contract `packages-not-1x` (`.importlinter:9`).
- The fixture budget (`scripts/check_fixture_budget.py` at 455 lines, plus 423 lines of tests) is bound to the
  1.x freeze SHA (`scripts/data/fixture_budget.yaml:1`), measured over the 1.x tree. 2.x CI must fetch a 1.x
  commit to pass (H:267).
- `adr/ADR-003-storage-leases.md` still describes 1.x `reproduce.build` and `reproduce.cache` allowances
  ("17 functions and 41 sites") and 1.x `build.exclusive_lock` timeouts (H-R41), although H10 narrowed the lint
  to 2.x modules.
- ledger H:292: "1.x scripts in scripts/ still ruff/import-linter roots".
- **Do now.**
  - Delete `moved_modules_test.py`.
  - Rebase the fixture budget on the 2.x tree, with no freeze-commit dependency.
  - Strip the 1.x paragraphs from ADR-003.
  - Move the 1.x scripts out of the 2.x lint roots, or into the source pool.

### I-11. The agent roster and isolation rules were departed from by ruling (H-R3, process ruling H:216, H9 at H:169)

- **What was decided.**
  - H-R3: implementers run sequentially in one shared worktree.
  - H:216: "pre-commit stashes other agents' unstaged edits; a reviewer saw broken imports".
  - H:169: "all opus implementers" for H9, including the gate, CI and cleanup work outside the kernel.
- **Authority contradicted.**
  - `engineering.md:84-86`: implementers are "sonnet" outside the kernel, with Isolation "worktree".
  - `engineering.md:96`: "Move up one tier … after a task fails its tests or review twice".
- **Do now.** Implementers go in their own worktrees (they can then run in parallel), at the roster tier, and
  escalate on failure.

### I-12. Parked items live only in a git-excluded ledger, and several were never resolved (R35 contradicted by practice)

- **Count.** The ledgers hold 32 + 7 "minor (deferred)" lines and about 12 "parked/moved/open" lines. None is
  in the register except R-56..R-58 and R-109.
- **This contradicts R35 itself** (P1:191): obligations "become open review-register rows (durable, not
  ledger-only)".
- **Still unresolved at 3a468aff:**
  - T8 (P1:139): "local directory identity includes dotfiles (.git) and skips symlinked dirs — decide inclusion
    rules". Still undecided: `runtime/models.py:104` uses `os.walk(path)` over every file. A local model's
    identity changes with `.git` contents, contrary to P-4/P-38.
  - H9a M-10 (H:219): `_bounded` is duplicated, now 4 copies (`reattack_wire_test.py:294`,
    `reattack_store_test.py:51`, `shutdown_schedule_fixture.py:268`, `senselab-worker/tests/launch_test.py:103`).
    This contradicts P-37.
  - H9a M-13 (pgid reuse), and the H4 deferred check-then-act window (H:41). Neither was registered.
  - T4: traceback frame limit per chained exception (`faults.py:11`). Never decided.
  - A-39 branch protection, "owner action" (H:88). No closure recorded.
  - ledger H:292: 1.x scripts as lint roots.
- **Do now.** Move every open deferred item into `review-register.md`, or close it with a reason. Treat the
  ledger as scratch.

### I-13. Lock semantics departed from ADR-003's decision (R30, R31, H-R40, H-R41, H-R42)

- **Original decision.** ADR-003 decided: "The protocol uses only primitives that `work_pool` has measured on
  NFS v4.2: atomic `rename`, `O_EXCL` hard link, and server-clock timestamps". Its alternatives rejected "Strict
  fencing on NFS" and "a lock service".
- **What core has.** `SharedFileLock`, an exclusive `flock` with holder, heartbeat and clock sidecars, takeover
  after `stale_after`, and an unbounded wait for `timeout=-1` (H-R41). Its own docstring says "correct only
  where `flock` is coherent across clients" (`runtime/locks.py:16`). ADR-003 was rewritten to describe this
  ("The lock file is the one exception") with its Status, Consequences and Alternatives unchanged.
- **Do now.** Owner decision. Either:
  - amend ADR-003 to admit `flock` (and say which filesystems are supported); or
  - make the store's correctness rest on rename and idempotence alone, as decided, with the lock as an
    efficiency hint.

---

## Minor

### M-1. P-41 is cited for "one implementation per behaviour" four times (H:101, H:149, H:205, H:236). That principle is P-37; P-41 is docs. Correct the citations, especially wherever they were copied into briefs.

### M-2. H-R22 (H:107) exempts editable installs from the cross-distribution exact-version import check, and adds a PEP 610 parser and an 8-way contract. P-22 asks only for "verified at handshake" (`principles.md:175`). Do now: drop the import-time check and keep the handshake.

### M-3. H-R12 (H:60) puts engineering bounds in code, consistent with P-17's "decision constants". This is upheld. Note only that the plan's own global constraint said otherwise; correct the plan, not the code.

### M-4. H-R39 (H:215) led to a 664-case shutdown schedule that takes 164 s of the 376 s contract tier (H:279). P-29 asks for fast cycles; `testing.md:180` puts limits off the per-PR path. H-R49 queues the move, which has not landed.

### M-5. R11 (P1:51) classifies OOM by class name. That is the "string match" P-34 names. It is now flagged as a gap (ADR-009), which is acceptable. Keep it as a fallback only.

### M-6. H-R50 keeps the 1.x format name `resolutions@2` for a 2.x-only file under `$SENSELAB_CACHE/2x/runs/`. Under P-14 a 2.x format starts at its own `@1`; fold this into I-8.

---

## 4. Process patterns, with counts

Line counts are `git diff --numstat` over `packages/`, `scripts/` and `.github/` for each fix-round range
recorded in the ledgers.

- **Fix rounds that grew the code.**
  - Phase 1: 18 fix rounds. 15 grew the code, 3 were neutral (spec-only or mypy), and none shrank it.
  - Hardening: 28 fix rounds. **All 28 grew the code, net +8,270 lines.** One round reduced `src` lines:
    H9c round 2, −32 src lines and +40 overall.
  - The largest: H9d r2 +1,181, H8b r2 +1,060, H9b r1 +751, H9c r1 +556, H7 r1 +385, H9b r2 +376.
- **Overall growth across the hardening phase.**
  - From Phase 1 exit (94e81479) to 3a468aff, package `src` went from 6,043 to 11,853 lines (+96%), and package
    tests from 6,535 to 23,252 (+256%).
  - The hardening plan's own words were "No new layers" (`plans/phase-1-hardening.md:9`).
- **Findings fixed by accumulating machinery instead of simplifying: 11.**
  1. digest guards (I-2);
  2. release gate (I-3);
  3. fail-at-base and the limiter (I-4);
  4. provenance token families (C-1);
  5. Unicode table and fingerprint (I-6);
  6. lock sidecars, heartbeat, takeover and set-aside (I-13);
  7. `written_names` lint plus the `StorageDir` grammar on every write;
  8. editable-install exemption (M-2);
  9. the src/tests conftest that drops lazy modules for hypothesis (H-R46), which edits the 1.x pool;
  10. the 664-case shutdown schedule (M-4);
  11. build-backend pins (H-R44).
- **Simplifications by ruling: 3.** H-R9 (allowlist instead of denylist), H-R18 (no sandbox), and H9b dropping
  `filelock`.
- **Scope expansion by ruling.**
  - R42 scoped hardening to "A-1..A-11". The plan took all 39 rows plus 9 unverified ones.
  - H-R26 then added a re-attack, which produced 31 more findings and became H9 (H9a–f, plus ci, cleanup,
    collect and compat).
  - The adversarial lead had marked only A-1..A-10 as blocking Phase 1 (P1:238).
- **Parked and never resolved.** 39 deferred minors and about 12 parked items live only in git-excluded ledgers.
  At least 7 are open at 3a468aff (I-12).
- **Specs rewritten to match code.** These commits change design text to describe what was built, rather than
  code changing to meet a design: d74e64ad, 33e7e96a and 9d31cd7c ("design.md §5 states … as built"), plus
  c5a69b9d, ccab74bf and 366f3e59 (ADR-003), and c8f76c25, fe5cbefd, ccf58520 and 7d20b482 (ADR-006).

## 5. Rulings and accepted limits to reverse now

| Ruling | Decided | Should be | Action |
|---|---|---|---|
| Process ruling H:140 | 1.x suite gates 2.x | 2.x tests gate 2.x | Already moot after H-R51; also undo its products (C-1) |
| H-R29 / H-R30 | provenance total over 1.x value types | JSON values (PROV literals) | Delete the families with ADR-010 |
| H-R34 (and H-R33) | raw post-13.0 text family | owner choice: NFC-on-identifiers or refuse | I-6 |
| H-R38 (rest) / H-R45 | `revised`/`recorded`/`FrozenDict` for 1.x | one frozen type, no copy-with-changes API for 1.x | C-1 |
| R24 | no reuse default | `equivalent` default (owner) | I-1 |
| H-R1 / H-R43 / H-R44 | full gate, 1.x tag spelling, exact backend pins | minimal gate, PEP 440 tags, unpinned backends | I-3 |
| H-R17 / H-R19 / H-R36 | resolver-based digest guards | one API plus an import-linter contract | I-2 |
| H-R32 / H-R37 (name contract) | fail-at-base structure, limiter keyed by test names | behaviour-named tests | I-4 (with H11) |
| H-R35 bumps | every tag bumped pre-release | reset to `@1` at first release | I-8 |
| H-R41 (1.x `.claims`, `exclusive_lock`) | 1.x lock behaviours in a 2.x ADR | drop from ADR-003 | I-10 |
| H-R42 (1.x `file_lock` → `SharedFileLock`) | edit 1.x | leave 1.x as frozen | I-9 |
| H-R22 | editable exemption | handshake check only | M-2 |
| R1 / R9 | one branch, draft without CI | PR per task, CI on drafts | I-5 |

**Rulings checked and upheld:** R3, R4, R5, R16, R23, R28, R33, H-R9, H-R10, H-R11, H-R13, H-R14, H-R18,
H-R47, H-R50 (except M-6), H-R51, H-R52.

## 6. Owner corrections: did they land at 3a468aff?

- **H-R38 compatibility read path.** Landed: 92303ddd, `ProvenanceFormatRefused`, and `PROV_TYPE =
  Literal["artifact"]` (`provenance.py:44`). The writer-side 1.x machinery remains (C-1).
- **Test names after review rounds.** Not landed. 264 `def test_(a|x…|m|n|i|c)<n>_` functions remain, along
  with `tests/adversarial/phase1_regression_test.py` and `reattack_*_test.py`. H-R49 / H11 is queued.
- **Triage kept alive.** The ADR-009 and migration-ledger text landed (7cd8fb49). The provenance and Unicode
  consequences remain (C-1, I-6).
- **src/ deletion losing history.** Landed: H-R51 / H-R52, and `src/` is intact. But `src/` is not the 1.x
  state (I-9).
