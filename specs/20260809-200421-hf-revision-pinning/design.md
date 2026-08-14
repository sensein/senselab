# Every result knows which commit produced it

Status: design approved 2026-08-09, not yet implemented.

senselab loads HuggingFace models by mutable ref. Every production call site uses `revision="main"`,
the run config has no `revision` key at all, and no output artifact records which commit actually
ran. A result produced today and a result produced next month are indistinguishable in the record
even when the upstream repo changed underneath them.

## What already works, and is not being rebuilt

Resolution to an immutable SHA already happens, reliably, at load time:

- `resolve_model(repo_id, revision) -> (sha, snapshot_path)` returns the 40-hex commit SHA.
- `load_hf_resilient(...)` injects `revision=<sha>` into the loader call, which also skips the Hub
  version check that rate-limits (429) under parallelism.
- `hf_load_coverage_test.py` enforces by AST that every in-process load site routes through those
  helpers, that subprocess-venv backends stage weights via `hf_subprocess_env`, and that the raw-load
  exemption list stays tiny and justified.

The wrapping is done. The defect is that the SHA is computed and then dropped.

## The four gaps

**1. Cache-key poisoning. This is a correctness bug, not a provenance gap.**
`cache_key()` hashes `{schema, audio_signature, task, model, params, code_version, senselab_version}`,
where `model` is a bare id such as `openai/whisper-large-v3-turbo`. Every model loads at
`revision="main"`. When upstream publishes a new commit, `resolve_model` loads the *new* weights
while `cache_key()` produces the *same* hash — so a cached result computed from the old commit is
silently served as current. Only a hand-bumped `CACHE_SCHEMA_VERSION` flushes it, and nothing ties
that bump to an upstream release.

**2. No output records a SHA.** `provenance_for()` records `task`, `model_id`, `params`, `device`,
`code_version`, `senselab_version`, `cache_schema_version`, `timestamp_utc` — no revision of any
kind. One path is wired further: `SignalRow.revision` reaches `L1/signals/<signal>.parquet` through
`io.write_signal_parquet`. Its value is the literal string `"main"`.

**3. The subprocess boundary loses it.** Parents send the *ref* to workers
(`"revision": model.revision or "main"` in `input_json`), and each worker re-resolves `main` against
its own cache. `hf_subprocess_env` calls `resolve_model` purely to stage the snapshot and decide
whether to set `HF_HUB_OFFLINE=1`; it returns an env dict and discards the SHA.

**4. Nothing pins.** Zero production call sites pass a revision other than `"main"`.
`BROUHAHA_REVISION` looks like a pin but is defined as `"main"`.

## The constraint that shapes the design

**The cache key must contain the SHA, and the cache key is computed *before* the model loads** —
deciding whether to load at all is its entire purpose. So the SHA cannot be collected as a side
effect of loading, and the obvious "registry that `resolve_model` writes into as it resolves" is
wrong: it populates after the key is needed.

Resolution must therefore happen *above* the load, and must be cheap enough to sit in front of a
cache hit that would otherwise avoid all model work.

## What makes this cheap

`HFModel`'s `revision` field validator calls `check_hf_repo_exists`, which for models calls
`ensure_hf_model(repo_id, revision)` — **which already resolves and returns the commit SHA** — and
then discards it to return a bool.

The SHA is already computed at model construction, above every load, on the exact path that needs it.
Capturing it adds no network call and no download. This is a plumbing change, not a new mechanism.

The second piece of leverage: `HFModel.revision` is already threaded to all four layers. Backends
read it, subprocess parents already forward it into worker `input_json`, and the Brouhaha path
already writes it to parquet. Those readers need a SHA-bearing field to read, not new wiring.

## Resolution and load are two separate calls, and the second must carry the SHA

This is the rule the whole design rests on, and getting it wrong makes everything else cosmetic.

Resolving `(repo_id, "main") -> sha` does **not** bind anything to that SHA. `snapshot_download`
called with `revision="main"` materialises `snapshots/<sha>/` and writes `refs/main`, but the caller
is still ref-addressed: a later load that passes `revision="main"` sends `huggingface_hub` back
through `refs/main`, which may by then point somewhere else. Recording a SHA while loading through a
ref would produce provenance that is confidently wrong — the worst possible outcome for a change
whose entire purpose is knowing what ran.

So every load is **two calls**:

1. Resolve the ref to a SHA.
2. Load again, explicitly passing `revision=<sha>`.

The second call downloads nothing. The blobs are already on disk from step 1, and a full 40-hex SHA
triggers `huggingface_hub`'s commit-hash shortcut, which returns the cached files with **zero**
network traffic — not even a HEAD. This is why it is free to insist on it everywhere.

`load_hf_resilient` already does exactly this (it injects `revision=<sha>`), which is the proof the
pattern works. What is missing is that the other paths do not: `ensure_hf_model` downloads
ref-addressed, `HFModel` construction never re-binds, and subprocess parents ship the *ref* to
workers that then resolve it themselves. Each of those becomes a two-call sequence.

The rule is directly testable, and cheaply: assert that whatever a loader (or a worker's
`input_json`) receives as `revision` is 40-hex, never a ref name. That single assertion is what stops
the design decaying back into ref-addressed loads.

### A consequence of local-first resolution, worth stating

`ensure_hf_model` already checks locally before going online — `is_hf_model_cached` is a
filesystem-only check, then senselab's own on-disk result cache, and only on a miss does it take the
lock and call `snapshot_download`. That ordering is correct and stays.

But the local check is keyed on the **ref**. Once `refs/main` exists locally, `main` never
re-resolves: the fast path returns the cached SHA and never asks the Hub whether `main` moved. So a
warm machine keeps returning January's commit while a cold machine downloads August's — same
senselab version, same config, different weights, no signal anywhere.

This design does not change that behaviour, and deliberately so: re-checking the Hub on every
resolution would reintroduce the 429 rate-limiting that `load_hf_resilient` exists to avoid.

Recording the SHA makes the divergence *visible*. It does not make it *stop*, and for one important
case visibility is not enough — see the next section.

## One run, one SHA: the resolution manifest

A cluster sweep is not one process. It is an array of jobs across nodes, each spawning subprocess
venvs, running over hours or days. If upstream pushes to `main` partway through, the tasks that
resolve after the push get different weights from the ones before it. Every task would record its own
SHA correctly, and the run as a whole would be quietly inhomogeneous — results from two different
models pooled into one analysis, each individually well-documented. Per-task provenance *documents*
that split; it does not prevent it, and a split run is usually worthless rather than merely annotated.

**A run resolves each `(repo_id, ref)` exactly once, and every participant binds to that answer.**

The run is named by `SENSELAB_RUN_ID`:

- If set, it is inherited — a Slurm submission exports one value (a UUID, or `$SLURM_JOB_ID`) and
  every node, task, and subprocess venv in that submission shares it.
- If unset, senselab generates a UUID4 at first use and exports it to every subprocess it spawns. A
  bare `python -m senselab ...` is therefore its own run, self-consistent, with no configuration
  required. This is the "for a given process launch" case: the launch is the run.

Resolution consults a per-run manifest before anything else:

```
$SENSELAB_CACHE/runs/<run_id>/resolutions.json
    {"openai/whisper-large-v3-turbo@main": "abc123…", …}
```

`resolve_revision(repo_id, ref)` becomes: manifest hit → return that SHA, no network and no local
`refs/` read; manifest miss → resolve as described above, record the entry, return it. The
first participant to need a model decides for the whole run; everyone after follows, including a task
that starts on a cold node a day later.

**Writes take a `SharedFileLock`** on the manifest — this is precisely the multi-user,
multi-node, read-modify-write case that lock was built for, and the manifest lives in the same shared
tree. Two nodes resolving the same model concurrently must not lose one another's entries, and the
loser of the race adopts the winner's SHA rather than overwriting it. Manifest writes are therefore
append-if-absent, never replace: an entry, once recorded, is immutable for the life of the run. That
immutability is the whole guarantee.

**The manifest is also the run's provenance.** It is a single small file listing every model the run
used and the exact commit of each — recoverable without parsing per-artifact metadata, and the
natural thing to attach to a paper or a bug report.

Two consequences worth being explicit about:

- **A long-lived run pins to increasingly old commits.** That is the intent: within a run, consistency
  beats freshness. A new run gets a new id and re-resolves.
- **The manifest is authoritative over the local cache.** If the manifest names a SHA a node does not
  have, that node downloads that SHA rather than using whatever its `refs/main` points at. This is the
  case that makes the guarantee real on a heterogeneous cluster, and it is where the two-call rule
  above stops being a formality: the load *must* pass the SHA, or the node silently uses its own.

Run directories are small JSON files and accumulate; pruning is a housekeeping concern, not part of
this design's correctness. They are safe to delete once a run is finished.

## Design

**Two fields, because they answer two different questions.**

```python
revision: str = "main"          # what was ASKED for: a ref, tag, or SHA
commit_sha: Optional[str] = None   # what it RESOLVED to: 40-hex, immutable
```

This is not a compatibility shim (which the pre-alpha convention forbids) — it is two distinct
values. `revision="main", commit_sha="abc123…"` records *both* that the run tracked `main` and which
commit that was on the day it ran. Collapsing to one field would leave provenance unable to
distinguish "pinned to abc123" from "tracked main, got abc123", and drift is only diagnosable when
you can tell those apart.

`commit_sha` is populated at construction, from the SHA `ensure_hf_model` already returns.

**One cheap resolver, cache-first.** `resolve_revision(repo_id, ref) -> sha` is the single choke
point, and it must not reintroduce the 429 storms `load_hf_resilient` exists to avoid:

1. If `ref` is already 40-hex, return it — no I/O.
2. Read `$HF_HUB_CACHE/models--<org>--<repo>/refs/<ref>`, which holds the SHA. Works offline.
3. Only on miss, one metadata call (`HfApi().model_info(...).sha`) — metadata, never a download.
4. Memoize per process, keyed `(repo_id, ref)`.

Steps 1–2 make the common path free, which is what lets this sit in front of a cache lookup.

**Unresolvable is a hard error.** If no SHA can be obtained — offline with a cold cache, network
down, repo gated — raise. A result whose provenance says "unknown commit" is worth less than no
result, and it would still be cached under a key that cannot distinguish it from any other commit.
This follows `scripts/calibrate_detection_margin.py`, which refuses to emit a profile from
insufficient measurement rather than emitting a weak one.

**Record-only, no lockfile.** Each run resolves fresh against the ref and records what it got.
Rerunning months later picks up whatever `main` is then — and correctly *misses* the cache, because
the SHA is in the key. Freezing SHAs across runs is a separate concern with its own refresh workflow;
this design makes it possible later by making the SHA a recorded value, and does not build it now.

### Where it lands

| Gap | Change |
|---|---|
| Cache-key poisoning | `cache_key()` payload gains the resolved SHA; `cache_key_for()` resolves before the lookup |
| No SHA in outputs | `provenance_for()` records `revision` **and** `commit_sha`; Brouhaha's existing parquet column starts carrying a real SHA |
| Subprocess boundary | Parents forward `commit_sha`; workers load by SHA instead of re-resolving `main` |
| Nothing pins | Backends already pass `revision=`; they pass the SHA once the field carries one |

Call sites that hold a bare `model_id` string rather than an `HFModel` — the workflow's cache path
among them — call `resolve_revision` directly at the stage boundary. Same helper, no second mechanism.

## Testing

**No test downloads a real model of consequence.** The suite currently has no tiny fixture model at
all — it uses a fake `"org/model"` for mocked paths and real names like `Qwen/Qwen3-ASR-1.7B`
elsewhere. Two tiers, and the split is the point:

- **One tiny real model**, from the `hf-internal-testing/tiny-random-*` family (public, ungated,
  well under a megabyte, stable). It exists to prove the *real* `huggingface_hub` contract that
  mocks cannot: that resolving a ref yields a 40-hex SHA, and that the second call with
  `revision=<sha>` returns the cached files without network. A mock asserting our own beliefs about
  `huggingface_hub` would pass just as happily when those beliefs are wrong, which is exactly the
  failure this design must not have. Confine it to the handful of tests that genuinely exercise the
  Hub contract, and pin the fixture model's own SHA so the fixture cannot drift.
- **Everything else is mocked** — monkeypatch `check_hf_repo_exists` / `ensure_hf_model` /
  `resolve_revision` per test. Tests must never construct an unmocked `HFModel`: its validator calls
  `ensure_hf_model`, which downloads the full snapshot. An earlier revision of the diarization tests
  did this and pulled 20 GB, and would have on every cold CI run. Each test monkeypatches
  independently rather than relying on a sibling having warmed a session-lifetime cache.

The specific assertions:

- **Resolution without network.** Write a `refs/<ref>` file into a temp `HF_HUB_CACHE` and assert
  `resolve_revision` returns its contents; assert a 40-hex input short-circuits with no I/O at all.
- **The two-call rule.** Assert every loader call and every worker `input_json` receives a 40-hex
  `revision`, never a ref name. This is the regression guard against decaying back to ref-addressed
  loads, and it is a cheap string assertion.
- **The cache-key bug gets a test that fails against current code**: same audio, same task, same
  `model_id`, two different SHAs must produce two different keys. Today they collide — that is the
  bug, and a test that passes both before and after would not be testing it.
- **Hard-error path**: cold cache plus unreachable Hub raises rather than falling back to the ref.
- **Provenance round-trip**: run a stage, read the artifact back, assert both `revision` and
  `commit_sha` are present and that `commit_sha` is 40-hex.
- **The manifest pins across an upstream move.** Seed a manifest with a known SHA, then make the
  resolver's underlying lookup return a *different* SHA (as an upstream push would) and assert
  `resolve_revision` still returns the manifest's. This is the test that proves the run-consistency
  guarantee, and it needs no network — the "upstream moved" half is a monkeypatch.
- **Concurrent first-resolution does not lose an entry.** Two processes resolving the same model at
  once must end with one agreed SHA, and the loser must adopt the winner's rather than overwrite it.
  Worth one real `multiprocessing` test: mocking the lock would prove nothing about the case the
  lock exists for.
- **Run id propagates.** Assert a spawned subprocess venv's environment carries the parent's
  `SENSELAB_RUN_ID`, and that an unset id is generated once rather than per-call.

## What this does not do

- It does **not** add a lockfile, or any way to replay a *past* run's SHAs. The manifest pins a run
  to itself while it runs; it is not a checked-in artifact you re-run against months later.
- It does **not** change which models are used or any default model list.
- It does **not** make `HFModel` construction more expensive — the resolution it captures is one the
  constructor already performs and discards.
- It does **not** touch the AST coverage test's contract. Load sites already route through the
  helpers; only what those helpers record changes.
- It does **not** attempt to detect that an upstream model changed. It makes the change *visible*
  after the fact and stops it silently reusing a stale cache entry.

## Success criteria

A result artifact names the 40-hex commit that produced it. An upstream push to `main` invalidates
the cache entries that depended on it, without a `CACHE_SCHEMA_VERSION` bump. A subprocess worker
loads the same commit its parent resolved, rather than resolving one for itself.

And the one that guards all three: **no load anywhere passes a ref.** Every loader call and every
worker `input_json` carries a 40-hex SHA, so the commit named in the provenance is provably the
commit whose weights ran — not a ref that merely pointed there at some earlier moment.
