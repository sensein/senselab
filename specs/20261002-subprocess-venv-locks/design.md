# Subprocess venvs install from committed locks

**Date:** 2026-10-02. **Owner's ask:** "fix 5": every subprocess venv must be reproducible.

## The problem

`ensure_venv` installed each backend's requirement list (`_PPGS_REQUIREMENTS` and so on)
by resolving it fresh against PyPI. Most entries are ranges or bare names, so a venv built
today differed from one built next month. That already broke CI once: `huggingface-hub` 2.x
pushed the `ppgs` resolution back to `transformers` 4.12.2 and `tokenizers` 0.10.3, which do
not build. `6fdb9a50` fixed it with a `huggingface-hub<2` constraint, after the fact.

## The design

- **One lock per venv name.** Locks live in `src/senselab/utils/data/venv_locks/<name>.txt`.
  `scripts/lock_subprocess_venvs.py` compiles them with:

  ```
  uv pip compile --universal --python-version <venv python> --generate-hashes --no-sources \
      --exclude-newer <timestamp> -c <quarantine constraints>
  ```

  The compile input is the backend's own requirement list plus the IPC dependencies
  (`safetensors`, `numpy`). The list in code stays the single source of the specification,
  and the lock is derived from it. A lock is universal, so one file serves linux x86_64 and
  macOS arm64 through environment markers.
- **The header binds the lock to its inputs.** `requirements-sha256` digests the sorted
  requirements, the IPC list, the Python version and the CUDA cap. `ensure_venv` refuses a
  missing lock, or one whose digest differs from the call's inputs (`VenvLockError`, naming
  the regenerate command). It does this before creating any directory. The test
  `src/tests/utils/venv_lock_test.py` runs the same check for every `ensure_venv` call it
  finds by AST, so a requirement edit without a regenerated lock fails in CI.
- **torch stays CUDA-routed.**
  - The lock body leaves out `torch` and `torchaudio`, plus the CUDA-variant packages that
    only they pull in: `nvidia-*`, `triton`, `pytorch-triton*` and `cuda-*`. A package
    counts as torch-only when every parent in uv's `# via` annotations is already torch-only.
    `sympy` and `filelock`, for example, stay in the body.
  - The exact pins go into the header's `torch:` line.
  - `ensure_venv` keeps its two stages:
    1. Install the header pins from the host's PyTorch index, chosen as before by
       `pick_torch_index` or `SENSELAB_TORCH_INDEX_URL`, with no other index named.
    2. Install the body with `--no-deps --no-sources`.

    Nothing is resolved at install time. The `nvidia-*` and `triton` versions follow from the
    exact torch pin and the index, because torch pins them exactly per variant.
- **The pinned torch exists where it will be installed.**
  - For each venv, the compile checks the PyTorch simple index. The pinned
    `torch`/`torchaudio` must have a wheel for the venv's Python on:
    - the newest CUDA index the venv can route to (`cu128`, or the cap's tag) for linux x86_64;
    - the `cpu` index for linux x86_64 and for macOS arm64.
  - If any check fails, the compile adds `torch<that version` and resolves again. Each
    skipped version is recorded as a `# note:` line in the header.
  - This check found that `s3prl` (`torch<2.5`) could not build on any CUDA ≥ 12.8 host,
    because `cu128` carries no torch below 2.7. It now declares `max_cuda_version=(12, 4)`,
    as `unasdiff` and `child_adult` already did.
- **The venv's identity is the lock.**
  - The completion marker stores `lock_sha256`, the digest of the whole lock file. A changed
    lock rebuilds the venv; a marker from the requirement-list era has no `lock_sha256`, so
    it rebuilds too.
  - The directory naming (`name`, or `name-<index tag>`), the `SharedFileLock` and the
    `.senselab-installed` marker are unchanged.
- **Quarantined releases are excluded at compile time.** Every compile carries
  `lightning!=2.6.2,!=2.6.3` and `pytorch-lightning!=2.6.2,!=2.6.3`. Those PyPI releases were
  malware; see the memory note "lightning quarantine". A test asserts that no committed lock
  holds either release. As of 2026-10-02 PyPI serves lightning again, and the locks resolve
  2.6.6 (brouhaha, diarizen) and 2.4.0 (both NeMo venvs).
- **`--no-sources`.** NeMo's own `pyproject.toml` routes `torch` through the PyTorch CPU index
  in `tool.uv.sources`, and uv rejected that as a second index for `torch`. Locks are
  compiled, and installed, ignoring any dependency's `tool.uv.sources`.
- **`--exclude-newer`** is the day the lock was compiled. It is recorded in the header, so a
  regeneration with the same timestamp and uv version reproduces the same lock.

## What is not locked

- **Compatibility probes** (`compatibility_test_runner`) explore version ranges, so a
  committed lock cannot exist for them. They call `ensure_venv(..., compile_lock=True)`,
  which compiles a lock into the cache at build time and installs through the same path.
  Such a venv is not reproducible, by construction.
- **unasdiff's flash-attn opt-in** (`SENSELAB_UNASDIFF_FLASH_ATTN`). `flash-attn==2.5.8`
  cannot be compiled into a universal lock, because building its metadata needs `torch`.
  It was also not installable through the previous isolated-build path. It is now built,
  once, into the already-locked venv with `--no-build-isolation --no-deps`, after unpinned
  `ninja`/`packaging`/`psutil`/`wheel`/`setuptools`. A marker file in the venv records the
  build. This needs a host with `nvcc` and has not been run.
- **The interpreter patch level.** `uv venv --python 3.11` takes the newest 3.11 patch uv has.

## Size

- **On disk:** 22 locks, 3.0 MB with hashes. Most of it is per-platform wheel hashes; the
  largest are the two NeMo locks, at about 4,300 lines each.
- **In the wheel:** the locks ship in it, because `ensure_venv` reads them at run time.

## Regenerating

```
uv run python scripts/lock_subprocess_venvs.py            # all venvs, exclude-newer = today
uv run python scripts/lock_subprocess_venvs.py ppgs s3prl # some
uv run python scripts/lock_subprocess_venvs.py --check    # report missing or stale locks
```

## Measured

- **ppgs on macOS arm64** (CPU index), built from its lock into an empty cache: 17 s.
  - `torch` 2.8.0, `torchaudio` 2.8.0, `ppgs` and `huggingface_hub` 1.33.0 import.
  - `uv pip check` reports all 120 packages compatible.
  - The second `ensure_venv` call reused the venv in 0.02 s.
- **Not yet built:** none of the CUDA-index installs (cu121/cu124/cu128) has been built from a
  lock yet. That needs a GPU node.
