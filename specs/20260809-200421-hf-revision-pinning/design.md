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

- **Resolution is assertable without network.** Write a `refs/<ref>` file into a temp `HF_HUB_CACHE`
  and assert `resolve_revision` returns its contents; assert a 40-hex input short-circuits with no
  I/O at all.
- **The cache-key bug gets a regression test that fails today**: same audio, same task, same
  `model_id`, two different SHAs must produce two different keys. Against current code they collide,
  which is the bug.
- **Hard-error path**: cold cache plus unreachable Hub raises rather than falling back.
- **Provenance round-trip**: run a stage, read the artifact back, assert both `revision` and
  `commit_sha` are present and that `commit_sha` is 40-hex.
- **Subprocess**: assert the worker `input_json` carries a SHA, not `"main"`.
- Tests must not construct an unmocked `HFModel` — that triggers a real `snapshot_download` (an
  earlier revision of the diarization tests pulled 20 GB). Monkeypatch per test.

## What this does not do

- It does **not** add a lockfile, or any way to replay an old run's SHAs. Record-only by decision.
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
