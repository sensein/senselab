# HF Revision Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every senselab result can name the 40-hex commit that produced it, an upstream push invalidates the cache entries that depended on it, and every participant in one run uses the same commit.

**Architecture:** A new `senselab.utils.model_revision` module owns run identity and a run-scoped resolution manifest. `resolve_revision(repo_id, ref)` consults the manifest, then the local cache, then the Hub, and hard-errors rather than silently falling back to a ref. `HFModel` gains a `commit_sha` field populated at construction from the SHA `ensure_hf_model` already computes and currently discards. Cache keys and provenance then carry the SHA, and subprocess parents forward it instead of shipping a ref for the worker to re-resolve.

**Tech Stack:** Python 3.12, pydantic v2, `huggingface_hub`, `filelock` (via senselab's own `SharedFileLock`), pytest, uv.

## Global Constraints

- **Read the design first:** `specs/20260809-200421-hf-revision-pinning/design.md`. It carries the reasoning; this plan carries the steps.
- **Every load is two calls.** Resolve the ref to a SHA, then load again explicitly passing `revision=<sha>`. Resolving alone binds nothing — a later load passing `"main"` goes back through `refs/main`, which may point elsewhere by then. The second call downloads nothing: a full 40-hex SHA triggers `huggingface_hub`'s commit-hash shortcut, returning cached files with zero network, not even a HEAD.
- **Never fall back to a ref.** An unresolvable revision raises. A result whose provenance says "unknown commit" is worth less than no result, and it would still be cached under an indistinguishable key.
- **`lock_holder()` returning `None` is not evidence a lock is free** — `filelock` opens the lock file `O_TRUNC` on every poll including failed ones. Never treat `None` as "unheld".
- **Never break a lock on a timeout path.** Reaching a timeout proves a live holder (a crashed process's flock is kernel-released). Catch `TimeoutError`, log, and retry — never unlink. `ensure_hf_model` in `dependencies.py` is the reference pattern.
- **`dependencies.py` must keep zero imports from `senselab.utils.data_structures`** (circular-import constraint). `model_revision.py` is imported *by* `dependencies.py`, so it must import only stdlib, `filelock`, `huggingface_hub`, and `senselab.utils.file_lock`.
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims. `revision` and `commit_sha` are two distinct values, not a compatibility pair.
- **Tests must never construct an unmocked `HFModel`** — its validator calls `ensure_hf_model`, which downloads a full snapshot. An earlier revision of the diarization tests pulled 20 GB. Each test monkeypatches independently rather than relying on a sibling having warmed a cache.
- **One tiny real model only**, from `hf-internal-testing/tiny-random-*`, confined to tests that must prove the real `huggingface_hub` contract. Everything else is mocked.
- `uv run` for every command — never bare `python`/`pip`. Never `pytest -n auto` (it OOMs the machine). Run scoped test files, in the foreground.
- Google-style docstrings, line length 120, type hints required. Comments explain *why*, not *what*.
- Stage commits with explicit pathspecs. **Never `git add -A` or `git add .`** — this tree has untracked local artifacts.
- Commit messages end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `src/senselab/utils/model_revision.py` (new) | Run id, the run manifest, and `resolve_revision`. The single choke point. |
| `src/senselab/utils/dependencies.py` | `_get_cached_commit_hash`'s silent ref-fallback becomes a hard error; `hf_subprocess_env` returns the SHA. |
| `src/senselab/utils/data_structures/model.py` | `HFModel.commit_sha`, populated at construction. |
| `src/senselab/utils/tasks/cached_inference.py` | `cache_key()` payload gains the SHA. |
| `src/senselab/audio/workflows/audio_analysis/stage_context.py` | `cache_key_for` / `provenance_for` resolve and record. |
| `src/tests/utils/model_revision_test.py` (new) | Resolution, manifest, run id, concurrency. |

---

### Task 1: `resolve_revision`, run identity, and the run manifest

**Files:**
- Create: `src/senselab/utils/model_revision.py`
- Test: `src/tests/utils/model_revision_test.py`

**Interfaces:**
- Consumes: `SharedFileLock` from `senselab.utils.file_lock` — `SharedFileLock(path, *, timeout=600.0, heartbeat_interval=30.0, stale_after=120.0)`, a context manager over a *resource* path (it derives `.lock`/`.heartbeat` by string append). Raises `TimeoutError` on timeout, never `filelock.Timeout`.
- Produces:
  - `run_id() -> str`
  - `manifest_path(run: str | None = None) -> Path`
  - `read_manifest(run: str | None = None) -> dict[str, str]`
  - `record_resolution(repo_id: str, ref: str, sha: str, run: str | None = None) -> str`
  - `resolve_revision(repo_id: str, ref: str = "main", *, token: str | None = None) -> str`
  - `RevisionResolutionError(RuntimeError)`
  - `manifest_key(repo_id: str, ref: str) -> str` returning `f"{repo_id}@{ref}"`

- [ ] **Step 1: Write the failing tests**

Create `src/tests/utils/model_revision_test.py`:

```python
"""Tests for run-scoped HF revision resolution."""

import json
import os
from pathlib import Path

import pytest

from senselab.utils.model_revision import (
    RevisionResolutionError,
    manifest_key,
    manifest_path,
    read_manifest,
    record_resolution,
    resolve_revision,
    run_id,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point SENSELAB_CACHE at a temp dir and clear any inherited run id."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "senselab"))
    monkeypatch.delenv("SENSELAB_RUN_ID", raising=False)
    import senselab.utils.model_revision as mr

    mr._RUN_ID = None
    mr._MEMO.clear()


def test_run_id_is_generated_once_and_exported() -> None:
    first = run_id()
    second = run_id()
    assert first == second, "run_id must be stable within a process"
    assert os.environ["SENSELAB_RUN_ID"] == first, "run id must be exported so subprocesses inherit it"


def test_run_id_is_inherited_when_already_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENSELAB_RUN_ID", "job-12345")
    import senselab.utils.model_revision as mr

    mr._RUN_ID = None
    assert run_id() == "job-12345"


def test_a_full_sha_short_circuits_without_touching_the_manifest() -> None:
    assert resolve_revision("org/model", SHA_A) == SHA_A
    assert not manifest_path().exists(), "a 40-hex ref must resolve with no I/O at all"


def test_record_then_read_round_trips() -> None:
    record_resolution("org/model", "main", SHA_A)
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


def test_manifest_entries_are_immutable_for_the_run() -> None:
    record_resolution("org/model", "main", SHA_A)
    returned = record_resolution("org/model", "main", SHA_B)
    assert returned == SHA_A, "the loser of a race adopts the winner's SHA, never overwrites it"
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


def test_the_manifest_pins_across_an_upstream_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once recorded, a run keeps its SHA even after upstream moves the ref."""
    record_resolution("org/model", "main", SHA_A)

    def _moved(repo_id: str, ref: str, token: object = None) -> str:
        raise AssertionError("resolution must not be attempted when the manifest already has an answer")

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _moved)
    assert resolve_revision("org/model", "main") == SHA_A


def test_an_unresolvable_ref_raises_rather_than_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fails(repo_id: str, ref: str, token: object = None) -> str:
        raise RevisionResolutionError("cold cache, hub unreachable")

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _fails)
    with pytest.raises(RevisionResolutionError):
        resolve_revision("org/model", "main")


def test_a_resolved_ref_is_recorded_so_the_rest_of_the_run_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _once(repo_id: str, ref: str, token: object = None) -> str:
        calls.append((repo_id, ref))
        return SHA_A

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _once)
    assert resolve_revision("org/model", "main") == SHA_A
    import senselab.utils.model_revision as mr

    mr._MEMO.clear()  # force the manifest, not the memo, to answer the second call
    assert resolve_revision("org/model", "main") == SHA_A
    assert len(calls) == 1, "the manifest must answer after the first resolution"


def test_a_corrupt_manifest_does_not_crash_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text("{not json")
    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", lambda *a, **k: SHA_A)
    assert resolve_revision("org/model", "main") == SHA_A
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest src/tests/utils/model_revision_test.py -q --noconftest -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'senselab.utils.model_revision'`

- [ ] **Step 3: Write the module**

Create `src/senselab/utils/model_revision.py`:

```python
"""Run-scoped resolution of HuggingFace refs to immutable commit SHAs.

A cluster sweep is an array of jobs across nodes, each spawning subprocess venvs,
running over hours or days. If upstream pushes to ``main`` partway through, tasks
that resolve after the push get different weights from those before it — every
task recording its own SHA correctly, and the run as a whole quietly
inhomogeneous. Per-task provenance documents that split; it does not prevent it.

So a run resolves each ``(repo_id, ref)`` exactly once and every participant binds
to that answer. Entries are append-if-absent and immutable for the run's life;
that immutability is the entire guarantee.

This module imports nothing from ``senselab.utils.data_structures`` —
``dependencies.py`` imports it, and that module carries a circular-import
constraint.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from senselab.utils.file_lock import SharedFileLock

logger = logging.getLogger("senselab")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Process-local. _RUN_ID caches the run identity; _MEMO avoids re-reading the
# manifest from disk for a pair already resolved in this process.
_RUN_ID: Optional[str] = None
_MEMO: dict[str, str] = {}


class RevisionResolutionError(RuntimeError):
    """A ref could not be resolved to an immutable commit SHA.

    Raised rather than falling back to the ref: a result whose provenance says
    "unknown commit" is worth less than no result, and it would still be cached
    under a key that cannot distinguish it from any other commit.
    """


def manifest_key(repo_id: str, ref: str) -> str:
    """Return the manifest key for a ``(repo_id, ref)`` pair."""
    return f"{repo_id}@{ref}"


def run_id() -> str:
    """Return this run's id, generating and exporting one if unset.

    Inherited from ``SENSELAB_RUN_ID`` when present, so a single Slurm submission
    that exports one value shares it across every node, task and subprocess venv.
    When absent, a UUID4 is generated once and exported, so a bare launch is its
    own self-consistent run with no configuration required — and any subprocess
    senselab spawns inherits it through the environment.
    """
    global _RUN_ID
    if _RUN_ID is None:
        existing = os.environ.get("SENSELAB_RUN_ID")
        _RUN_ID = existing if existing else str(uuid.uuid4())
        os.environ["SENSELAB_RUN_ID"] = _RUN_ID
    return _RUN_ID


def _cache_root() -> Path:
    """Return senselab's cache root, honouring ``SENSELAB_CACHE``.

    Duplicated from ``dependencies._senselab_cache_dir`` rather than imported:
    ``dependencies`` imports *this* module, and importing back would close a cycle.
    """
    override = os.environ.get("SENSELAB_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "senselab" / "hf"


def manifest_path(run: Optional[str] = None) -> Path:
    """Return the path to a run's resolution manifest."""
    return _cache_root() / "runs" / (run or run_id()) / "resolutions.json"


def read_manifest(run: Optional[str] = None) -> dict[str, str]:
    """Return a run's recorded resolutions, or an empty mapping.

    A manifest that is missing, empty, or unparseable reads as empty rather than
    raising: a corrupt manifest must degrade to "resolve again", never to a crash
    that takes down every job in the run.
    """
    path = manifest_path(run)
    try:
        text = path.read_text()
    except OSError:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Run manifest %s is unparseable; treating as empty", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def record_resolution(repo_id: str, ref: str, sha: str, run: Optional[str] = None) -> str:
    """Record ``(repo_id, ref) -> sha`` for this run and return the binding SHA.

    Append-if-absent under a :class:`SharedFileLock`: two nodes resolving the same
    model concurrently must not lose one another's entries, and the loser of the
    race adopts the winner's SHA rather than overwriting it. The returned value is
    therefore the *authoritative* SHA for the run, which may not be ``sha``.
    """
    path = manifest_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = manifest_key(repo_id, ref)
    lock = SharedFileLock(path)
    while True:
        try:
            lock.__enter__()
            break
        except TimeoutError:
            # A timeout proves a live holder: a crashed process's flock is
            # kernel-released, so it cannot hold continuously through the window.
            # Waiting is correct; breaking the lock would displace a live writer.
            logger.info("Waiting to record a resolution for %s@%s in the run manifest", repo_id, ref)
            continue
    try:
        current = read_manifest(run)
        winner = current.get(key)
        if winner is not None:
            return winner
        current[key] = sha
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        return sha
    finally:
        lock.__exit__(None, None, None)


def _resolve_uncached(repo_id: str, ref: str, token: Optional[str] = None) -> str:
    """Resolve a ref to a SHA without consulting the manifest or the memo.

    Filesystem first, network only on a miss: re-checking the Hub on every
    resolution would reintroduce the 429 rate-limiting that ``load_hf_resilient``
    exists to avoid.
    """
    from senselab.utils.dependencies import _get_cached_commit_hash

    try:
        local = _get_cached_commit_hash(repo_id, ref)
    except Exception:  # noqa: BLE001 — any local-read failure just means "ask the Hub"
        local = None
    if local and _SHA_RE.match(local):
        return local

    try:
        from huggingface_hub import HfApi

        sha = HfApi(token=token).model_info(repo_id=repo_id, revision=ref).sha
    except Exception as exc:
        raise RevisionResolutionError(
            f"Cannot resolve {repo_id}@{ref} to a commit SHA: {exc}. "
            "Refusing to load through a mutable ref — the result's provenance would name no commit."
        ) from exc
    if not sha or not _SHA_RE.match(str(sha)):
        raise RevisionResolutionError(f"Hub returned no usable commit SHA for {repo_id}@{ref} (got {sha!r})")
    return str(sha)


def resolve_revision(repo_id: str, ref: str = "main", *, token: Optional[str] = None) -> str:
    """Return the immutable commit SHA this run uses for ``(repo_id, ref)``.

    Order: a ref that is already a full SHA short-circuits with no I/O; then this
    process's memo; then the run manifest; then local cache and Hub. A freshly
    resolved SHA is recorded so the rest of the run follows it.

    Raises:
        RevisionResolutionError: If no SHA can be obtained.
    """
    if _SHA_RE.match(ref):
        return ref

    key = manifest_key(repo_id, ref)
    memoized = _MEMO.get(key)
    if memoized is not None:
        return memoized

    recorded = read_manifest().get(key)
    if recorded is not None:
        _MEMO[key] = recorded
        return recorded

    sha = _resolve_uncached(repo_id, ref, token)
    binding = record_resolution(repo_id, ref, sha)
    _MEMO[key] = binding
    return binding
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest src/tests/utils/model_revision_test.py -q --noconftest -p no:cacheprovider`
Expected: PASS (10 tests)

- [ ] **Step 5: Add the concurrency test**

Mocking the lock would prove nothing about the case the lock exists for. Append to `src/tests/utils/model_revision_test.py`:

```python
def test_concurrent_first_resolution_agrees_on_one_sha(tmp_path: Path) -> None:
    """Two processes recording different SHAs must converge on one winner."""
    import multiprocessing as mp

    cache = tmp_path / "shared"

    def _worker(sha: str, out: "mp.Queue[str]") -> None:
        os.environ["SENSELAB_CACHE"] = str(cache)
        os.environ["SENSELAB_RUN_ID"] = "shared-run"
        import importlib

        import senselab.utils.model_revision as mr

        importlib.reload(mr)
        out.put(mr.record_resolution("org/model", "main", sha))

    ctx = mp.get_context("spawn")
    q: "mp.Queue[str]" = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(s, q)) for s in (SHA_A, SHA_B)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = {q.get(timeout=5) for _ in procs}
    assert len(results) == 1, f"both processes must adopt one SHA, got {results}"
```

- [ ] **Step 6: Run it and verify it passes**

Run: `uv run pytest src/tests/utils/model_revision_test.py -q -p no:cacheprovider`
Expected: PASS (11 tests). Drop `--noconftest` here — `multiprocessing` spawn needs the package importable.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/utils/model_revision.py src/tests/utils/model_revision_test.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/utils/model_revision.py src/tests/utils/model_revision_test.py
git commit -m "feat(utils): run-scoped resolution of HF refs to commit SHAs

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `_get_cached_commit_hash` stops silently returning a ref

**Files:**
- Modify: `src/senselab/utils/dependencies.py:246` (`_get_cached_commit_hash`)
- Test: `src/tests/utils/dependencies_test.py`

**Interfaces:**
- Consumes: `RevisionResolutionError` from Task 1.
- Produces: `_get_cached_commit_hash` returns a 40-hex SHA or raises; it never returns its `revision` argument unchanged unless that argument is itself a SHA.

**Why:** its docstring already admits "Returns `revision` unchanged only as a last resort". That last resort is the silent ref-fallback the design forbids — every caller downstream then believes it holds a SHA and records a ref.

- [ ] **Step 1: Write the failing test**

Add to `src/tests/utils/dependencies_test.py`:

```python
def test_cached_commit_hash_raises_rather_than_returning_a_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold cache must raise, not hand back the mutable ref as if it were a SHA."""
    from senselab.utils.dependencies import _get_cached_commit_hash
    from senselab.utils.model_revision import RevisionResolutionError

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(RevisionResolutionError):
        _get_cached_commit_hash("org/never-downloaded", "main")
```

- [ ] **Step 2: Run it and verify it fails**

Run: `uv run pytest src/tests/utils/dependencies_test.py -k cached_commit_hash_raises -q`
Expected: FAIL — it returns `"main"` instead of raising.

- [ ] **Step 3: Replace the last-resort return**

In `_get_cached_commit_hash`, replace the final `return revision` with:

```python
    # Returning `revision` here would hand a mutable ref to callers that will
    # record it as a commit -- provenance that is confidently wrong, which is
    # worse than none. Refuse instead; the caller can go to the Hub.
    raise RevisionResolutionError(
        f"{repo_id}@{revision} is not resolvable from the local cache "
        f"(no refs/{revision} pointer and no snapshot directory)."
    )
```

Add the import at the top of the function body (deferred, to keep module import cheap):

```python
    from senselab.utils.model_revision import RevisionResolutionError
```

Update the docstring: delete the "Returns `revision` unchanged only as a last resort" sentence and state that it raises instead.

- [ ] **Step 4: Run the scoped tests**

Run: `uv run pytest src/tests/utils/dependencies_test.py -q`
Expected: PASS. If any existing test relied on the ref-fallback, that test was asserting the bug — update it to expect the raise and say so in the commit message.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/utils/dependencies.py src/tests/utils/dependencies_test.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/utils/dependencies.py src/tests/utils/dependencies_test.py
git commit -m "fix(dependencies): a ref is not a commit hash, so stop returning one

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `HFModel.commit_sha`, populated at construction

**Files:**
- Modify: `src/senselab/utils/data_structures/model.py:61-95` (`HFModel`), `:230` (`check_hf_repo_exists`)
- Test: `src/tests/utils/model_test.py`

**Interfaces:**
- Consumes: `resolve_revision` from Task 1.
- Produces: `HFModel.commit_sha: Optional[str]` — a 40-hex SHA after construction of a remote model; `None` only for a local `Path`-backed model. `revision` keeps its existing meaning: the ref that was *asked for*.

**Why this is nearly free:** `check_hf_repo_exists` already calls `ensure_hf_model(repo_id, revision)`, which resolves and returns the SHA, and then discards it to return a bool. The SHA is already computed at construction, above every load.

- [ ] **Step 1: Write the failing test**

Add to `src/tests/utils/model_test.py`:

```python
def test_hf_model_records_the_resolved_commit_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """revision keeps the ref asked for; commit_sha carries what it resolved to."""
    sha = "c" * 40
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kw: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)

    from senselab.utils.data_structures.model import HFModel

    model = HFModel(path_or_uri="org/model", revision="main")
    assert model.revision == "main", "the requested ref must survive"
    assert model.commit_sha == sha, "the resolved commit must be recorded"
```

- [ ] **Step 2: Run it and verify it fails**

Run: `uv run pytest src/tests/utils/model_test.py -k records_the_resolved -q`
Expected: FAIL — `HFModel` has no attribute `commit_sha`.

- [ ] **Step 3: Add the field and populate it**

In `HFModel`, add beside `revision`:

```python
    commit_sha: Optional[str] = None
    """The immutable 40-hex commit this run pins to, resolved at construction.

    Distinct from ``revision``, which records what was *asked for*. Keeping both
    lets provenance distinguish "pinned to abc123" from "tracked main, which
    resolved to abc123" -- drift is only diagnosable when those are tellable apart.
    """
```

Add a `model_validator(mode="after")` that populates it, after the existing `revision` validator has confirmed the ref exists:

```python
    @model_validator(mode="after")
    def _resolve_commit_sha(self) -> "HFModel":
        """Pin this model to an immutable commit, once, at construction.

        Skipped for local paths, which have no Hub revision. The resolution is one
        the constructor already performs -- ``check_hf_repo_exists`` calls
        ``ensure_hf_model``, which computes this SHA and discards it -- so this
        adds no network call and no download.
        """
        if isinstance(self.path_or_uri, Path) or self.commit_sha is not None:
            return self
        from senselab.utils.model_revision import resolve_revision

        object.__setattr__(self, "commit_sha", resolve_revision(str(self.path_or_uri), self.revision))
        return self
```

Import `model_validator` from `pydantic` alongside the existing `field_validator`.

- [ ] **Step 4: Run the scoped tests**

Run: `uv run pytest src/tests/utils/model_test.py -q`
Expected: PASS. Any test constructing an `HFModel` must monkeypatch **both** `check_hf_repo_exists` and `resolve_revision` — an unmocked construction downloads a full snapshot.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/utils/data_structures/model.py src/tests/utils/model_test.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/utils/data_structures/model.py src/tests/utils/model_test.py
git commit -m "feat(model): record the commit a model resolved to, not just the ref asked for

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Cache keys carry the SHA

**Files:**
- Modify: `src/senselab/utils/tasks/cached_inference.py:64` (`CACHE_SCHEMA_VERSION`), `:344-363` (`cache_key`)
- Modify: `src/senselab/audio/workflows/audio_analysis/stage_context.py:172-181` (`cache_key_for`)
- Test: `src/tests/utils/cached_inference_test.py`

**Interfaces:**
- Consumes: `resolve_revision` from Task 1.
- Produces: `cache_key(*, audio_sig, task, model_id, params, code_version, senselab_ver, commit_sha: str | None) -> str`. `commit_sha` is keyword-only and required — an omitted argument must be a type error, not a silent `None`.

**Why this is the most serious of the four gaps:** today the key contains only `model_id`, so an upstream push makes `resolve_model` load *new* weights while `cache_key` produces the *same* hash. A result from the old commit is served as current.

- [ ] **Step 1: Write the failing test**

Add to `src/tests/utils/cached_inference_test.py`:

```python
def test_two_commits_of_one_model_do_not_share_a_cache_key() -> None:
    """An upstream push must invalidate, not silently reuse."""
    from senselab.utils.tasks.cached_inference import cache_key

    common = dict(
        audio_sig="sig",
        task="asr",
        model_id="openai/whisper-large-v3-turbo",
        params={"device": "cpu"},
        code_version="v1",
        senselab_ver="0.1.0",
    )
    assert cache_key(**common, commit_sha="a" * 40) != cache_key(**common, commit_sha="b" * 40)
```

- [ ] **Step 2: Run it and verify it fails**

Run: `uv run pytest src/tests/utils/cached_inference_test.py -k two_commits -q`
Expected: FAIL — `cache_key() got an unexpected keyword argument 'commit_sha'`.

- [ ] **Step 3: Add the parameter to the payload**

In `cache_key`, add `commit_sha: str | None,` to the keyword-only signature and `"commit_sha": commit_sha,` to the `payload` dict. Document why:

```python
        # Without this, an upstream push to a tracked ref loads new weights under
        # an unchanged key and a stale result is served as current.
        "commit_sha": commit_sha,
```

Bump `CACHE_SCHEMA_VERSION` from `22` to `23` — every existing entry predates SHA-awareness and cannot be attributed to a commit, so it must not be reused.

- [ ] **Step 4: Thread it through `cache_key_for`**

In `stage_context.py`, resolve before the lookup and pass it:

```python
    def cache_key_for(self, task: str, model_id: str | None, params: Mapping[str, Any]) -> str:
        """Cache key for one (task, model, params) call in this pass."""
        return cache_key(
            audio_sig=self.audio_signature,
            task=task,
            model_id=model_id,
            params=dict(params),
            code_version=stage_code_version(task),
            senselab_ver=self.senselab_ver,
            commit_sha=self._commit_sha_for(model_id),
        )
```

Add the helper to the same class:

```python
    def _commit_sha_for(self, model_id: str | None) -> str | None:
        """Resolve ``model_id`` to this run's commit SHA, or ``None`` if not a Hub id.

        Resolution has to happen here, above the load, because the cache key is
        computed to decide *whether* to load at all -- a SHA harvested during
        loading would arrive too late to key on.
        """
        if not model_id or "/" not in model_id:
            return None
        from senselab.utils.model_revision import resolve_revision

        return resolve_revision(model_id)
```

- [ ] **Step 5: Run the scoped tests**

Run: `uv run pytest src/tests/utils/cached_inference_test.py src/tests/audio/workflows/ -q`
Expected: PASS. Existing callers of `cache_key` that now fail to type-check must be updated to pass `commit_sha`.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/utils/tasks/cached_inference.py src/senselab/audio/workflows/audio_analysis/stage_context.py src/tests/utils/cached_inference_test.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/utils/tasks/cached_inference.py src/senselab/audio/workflows/audio_analysis/stage_context.py src/tests/utils/cached_inference_test.py
git commit -m "fix(cache): key on the commit, so an upstream push invalidates

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Provenance records both the ref and the commit

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/stage_context.py:202-222` (`provenance_for`)
- Modify: `src/senselab/audio/workflows/audio_analysis/scene_quality/brouhaha.py:57` (`BROUHAHA_REVISION`)
- Test: `src/tests/audio/workflows/audio_analysis/stage_context_test.py`

**Interfaces:**
- Consumes: `_commit_sha_for` from Task 4.
- Produces: `provenance_for` output gains `"revision"` and `"commit_sha"` keys.

- [ ] **Step 1: Write the failing test**

```python
def test_provenance_records_the_commit_that_produced_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "d" * 40
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    ctx = _make_stage_context()  # existing helper in this test module
    prov = ctx.provenance_for("asr", "openai/whisper-large-v3-turbo", {"device": "cpu"})
    assert prov["commit_sha"] == sha
    assert prov["revision"] == "main"
```

- [ ] **Step 2: Run it and verify it fails**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/stage_context_test.py -k records_the_commit -q`
Expected: FAIL — `KeyError: 'commit_sha'`.

- [ ] **Step 3: Add both keys**

In `provenance_for`'s returned dict, after `"model_id": model_id,`:

```python
            # Both, deliberately: "revision" is what was asked for, "commit_sha" is
            # what ran. Recording only the second cannot distinguish a deliberate pin
            # from a tracked ref that happened to resolve there on the day.
            "revision": "main",
            "commit_sha": self._commit_sha_for(model_id),
```

- [ ] **Step 4: Make Brouhaha's existing parquet column carry a real SHA**

`SignalRow.revision` already reaches `L1/signals/<signal>.parquet`, carrying the literal `"main"`. In `brouhaha.py`, leave `BROUHAHA_REVISION = "main"` as the ref, and pass the resolved SHA into the row's `revision` field at the point the row is built, so the column stops recording a ref. Add a comment naming what changed and why.

- [ ] **Step 5: Run the scoped tests**

Run: `uv run pytest src/tests/audio/workflows/ -q`
Expected: PASS

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/audio/workflows/audio_analysis/stage_context.py src/senselab/audio/workflows/audio_analysis/scene_quality/brouhaha.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/audio/workflows/audio_analysis/stage_context.py src/senselab/audio/workflows/audio_analysis/scene_quality/brouhaha.py src/tests/audio/workflows/audio_analysis/stage_context_test.py
git commit -m "feat(audio_analysis): provenance names the commit that produced the result

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The subprocess boundary carries the SHA, and a guard that it stays that way

**Files:**
- Modify: `src/senselab/utils/dependencies.py:523` (`hf_subprocess_env`)
- Modify: `src/senselab/audio/tasks/speech_to_text/canary_qwen.py:104`, `src/senselab/audio/workflows/audio_analysis/scene_quality/brouhaha.py:238-245`, `src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py:617`
- Test: `src/tests/utils/revision_pinning_guard_test.py` (new)

**Interfaces:**
- Consumes: `resolve_revision` from Task 1.
- Produces: `hf_subprocess_env(...) -> dict` unchanged in shape, but the returned env carries `SENSELAB_RUN_ID` so the worker joins the parent's run.

**Why:** parents currently send the *ref* (`"revision": model.revision or "main"`) and each worker re-resolves it against its own cache — so two nodes in one run can load different commits.

- [ ] **Step 1: Write the failing guard test**

Create `src/tests/utils/revision_pinning_guard_test.py`:

```python
"""No load anywhere may pass a ref where a commit SHA belongs.

This is the regression guard for the design's central rule. Without it the
codebase decays back to ref-addressed loads one call site at a time, and the
provenance keeps reporting commits it did not actually load.
"""

import re

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_worker_input_json_carries_a_sha_not_a_ref(monkeypatch) -> None:
    sha = "e" * 40
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    from senselab.audio.workflows.audio_analysis.scene_quality import brouhaha

    payload = brouhaha._build_worker_input(audio_path="/tmp/a.wav")  # noqa: SLF001
    assert SHA_RE.match(payload["revision"]), f"worker got a ref, not a commit: {payload['revision']!r}"


def test_subprocess_env_propagates_the_run_id(monkeypatch) -> None:
    monkeypatch.setenv("SENSELAB_RUN_ID", "run-abc")
    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", lambda *a, **k: ("f" * 40, "/tmp"))
    from senselab.utils.dependencies import hf_subprocess_env

    env = hf_subprocess_env("org/model", "main")
    assert env["SENSELAB_RUN_ID"] == "run-abc", "a worker must join its parent's run, not start its own"
```

If `brouhaha` has no `_build_worker_input` helper, extract one in Step 3 — the payload construction must be callable without spawning a subprocess, or this rule cannot be tested at all.

- [ ] **Step 2: Run it and verify it fails**

Run: `uv run pytest src/tests/utils/revision_pinning_guard_test.py -q`
Expected: FAIL — the payload carries `"main"`, and the env has no `SENSELAB_RUN_ID`.

- [ ] **Step 3: Send the SHA and the run id**

In `hf_subprocess_env`, add the run id to the returned env:

```python
    # The worker must join the parent's run rather than starting its own, or it
    # would resolve refs independently and could pin to a different commit.
    from senselab.utils.model_revision import run_id

    env["SENSELAB_RUN_ID"] = run_id()
```

At each of the three parent call sites, replace `model.revision or "main"` in the `input_json` with the resolved SHA:

```python
    from senselab.utils.model_revision import resolve_revision

    revision = model.commit_sha or resolve_revision(model_name, model.revision if model else "main")
```

Extract `_build_worker_input` in `brouhaha.py` if needed so the payload is constructible without spawning.

- [ ] **Step 4: Run the scoped tests**

Run: `uv run pytest src/tests/utils/revision_pinning_guard_test.py src/tests/utils/dependencies_test.py -q`
Expected: PASS

- [ ] **Step 5: Add the tiny-model contract test**

This is the one test that touches the real Hub, and it exists because a mock asserting our own beliefs about `huggingface_hub` would pass just as happily when those beliefs are wrong:

```python
import pytest


@pytest.mark.slow
def test_the_two_call_rule_holds_against_the_real_hub(tmp_path, monkeypatch) -> None:
    """Resolve yields a SHA; loading with that SHA needs no network."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    from senselab.utils.model_revision import resolve_revision

    repo = "hf-internal-testing/tiny-random-gpt2"
    sha = resolve_revision(repo, "main")
    assert SHA_RE.match(sha)

    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=sha)  # populate
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert snapshot_download(repo, revision=sha), "a full SHA must resolve from cache with no network"
```

- [ ] **Step 6: Run it, lint, type-check, commit**

```bash
uv run pytest src/tests/utils/revision_pinning_guard_test.py -q
uv run ruff format src/senselab/utils/dependencies.py src/tests/utils/revision_pinning_guard_test.py
uv run ruff check src/
uv run mypy src/senselab/
git add src/senselab/utils/dependencies.py src/senselab/audio/workflows/audio_analysis/scene_quality/brouhaha.py src/senselab/audio/tasks/speech_to_text/canary_qwen.py src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py src/tests/utils/revision_pinning_guard_test.py
git commit -m "feat(subprocess): workers load the commit their parent resolved

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Document the behaviour change

**Files:**
- Modify: `CLAUDE.md`
- Modify: `src/senselab/audio/workflows/audio_analysis/doc.md`

- [ ] **Step 1: Record the cache consequence in `CLAUDE.md`**

Under the existing "Cache invalidation is free" bullet, add that `CACHE_SCHEMA_VERSION` reached 23 because cache keys became commit-aware, and that **the first run after this change recomputes everything** — existing entries cannot be attributed to a commit, so they are not reused. This is a one-time full recompute, not a silent no-op, and someone will otherwise report it as a regression.

- [ ] **Step 2: Document `SENSELAB_RUN_ID` in the workflow doc**

Add a short section to `doc.md`: what the run id is for, that a Slurm submission should export one value so every node and subprocess shares it, that leaving it unset makes each launch its own run, and where the manifest lives (`$SENSELAB_CACHE/runs/<run_id>/resolutions.json`). Note that the manifest doubles as the run's provenance — one file naming every model and its exact commit.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md src/senselab/audio/workflows/audio_analysis/doc.md
git commit -m "docs: commit-aware caches recompute once, and SENSELAB_RUN_ID pins a run

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Design section → task: the four gaps → Tasks 4 (cache keys), 5 (provenance), 6 (subprocess), 3 (nothing pins). "Resolution and load are two separate calls" → Task 6's guard test plus Task 2's refusal to return a ref. "One run, one SHA" → Task 1. "Hard error" → Tasks 1 and 2. "Two fields" → Task 3. Testing section → tests in every task, with the tiny-model contract test in Task 6 Step 5. Record-only (no lockfile) → nothing builds one.

**Known gap, deliberately left:** the design notes `hf_subprocess_env` discards `resolve_model`'s SHA. Task 6 adds the run id to the env so the worker resolves to the same commit through the manifest, rather than changing the function's return type. If a future caller needs the SHA in the parent, it calls `resolve_revision` — the same answer, from the same choke point.

**Type consistency.** `resolve_revision(repo_id, ref="main", *, token=None) -> str` is used identically in Tasks 2–6. `commit_sha` is the field name in Task 3, the `cache_key` parameter in Task 4, the provenance key in Task 5, and the `HFModel` attribute read in Task 6. `manifest_key` is defined and used only in Task 1.

**Ordering.** Task 1 must land first (everything imports it). Task 2 depends on Task 1's exception type. Tasks 4 and 5 both touch `stage_context.py`, and Task 5 uses the helper Task 4 adds — keep them in order. Task 6 reads `HFModel.commit_sha` from Task 3.
