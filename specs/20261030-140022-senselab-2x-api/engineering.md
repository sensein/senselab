# Engineering standards for senselab 2.x

These apply to every commit, by humans and agents alike. Principles P-31..P-35 in
[`principles.md`](principles.md); testing tiers in [`testing.md`](testing.md).

## Exemplar

`src/senselab/utils/fastio.py`, 334 lines, shows the pattern:
- one private engine (`_pool`) with thin public functions over it;
- typed, keyword-only options;
- docstrings that say what a thing does and how to call it, and nothing else;
- work done where the data is (in the worker threads), never in a second serial pass.

## Standards

| Standard | Rule | Enforced by |
|---|---|---|
| **Minimal verbosity** (P-31) | Say a thing once, in the fewest words that are exact. A docstring states what and how to call it. A comment exists only where code cannot say it. History, measurements and rejected alternatives go to `specs/` (P-21). The same applies to logs, errors, CLI output and agent-facing text: an error names the cause and the fix in one line. | lint over prose: docstring and comment length budgets; banned history words ("originally", "used to", "no longer"); register and round ids outside `specs/`. Review |
| **Clear, compact code** (P-32) | Share engines and keep surfaces thin. No duplicated logic: a second copy is factored before merge. Functions are small, with typed, keyword-only options. No dead code, no flags for removed behaviour (P-14). Dependencies stay out of core (P-13). | ruff (including complexity `C901` and `PL` limits), mypy strict on core, a duplicate-code check, and file and function size budgets reported per PR |
| **Efficient at every scale** (P-33) | Code is measured at four scale points: 1 item, 10², 10⁴, and a long signal (hours of audio, days of actigraphy). Time, peak memory, files written and bytes moved per item must not grow with scale. Profiles find the hot spots, not guesses. | `benchmarks/scale/` ladder runs per PR on touched modules (small points) and nightly (large points). Scalene profiles attached on regression. Per-item budgets stored as data with derivations (P-17); a regression beyond a band blocks merge |
| **Brittleness checked always** (P-34) | A change must not make the system depend on an accident: timing, ordering, one dataset, one host, one model's quirk, a magic number, a string match on output. Every boundary is exercised with bad input. Failures are typed and isolated. | property tests and fault injection in the logic tier (killed worker, stale lease, NFS rename race, network off, missing license, corrupted cache entry, truncated audio). A brittleness checklist in the PR template. Review rejects special-casing |
| **Adversarial agents set standards** (P-35) | Agents are used to break the system: hostile inputs, races, scale, misuse of the API, and attempts to make a test pass by special-casing. Each break is fixed **by an architectural change**, not a patch for that case: a new invariant, a type, a contract test or a lint rule. The fix is generalised into the standards here or into a principle. | each phase ends with an adversarial review round. Findings go to the register with their architectural resolution. A fix that only adds an `if` for the found case is rejected |

## Per-commit checklist

The PR template carries this checklist. Pre-commit and CI check what they can.

- [ ] Prose is minimal: no history, no ids, docstrings say what and how (P-31).
- [ ] No duplicated logic; surfaces are thin over shared engines (P-32).
- [ ] The scale ladder passes for touched modules; per-item cost is flat (P-33).
- [ ] New boundaries have bad-input, fault and property tests; there is no special-casing (P-34).
- [ ] No identifiers; tests are scoped and serial (P-28, P-29).
- [ ] Every new task, backend or profile is declared, so contract tests cover it (P-30).

## What "special-casing" means (rejected in review)

- A branch on a dataset, recording, subject, model id or host name.
- A threshold literal tuned to make one case pass (P-17).
- A retry, sleep or timeout added to hide a race instead of removing it.
- A test that asserts the current output of a model, or a measured number, outside `benchmarks/`.
- A try/except that turns an unknown failure into a default value.
