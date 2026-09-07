# A subprocess venv's directory must reflect what it installs

## The defect

`ensure_venv` in `src/senselab/utils/subprocess_venv.py` named a venv's directory from the backend
name alone (`_cache_dir() / name`), before resolving anything about the install itself. The
PyTorch wheel index (`cu128` / `cu126` / `cu124` / `cu121` / `cpu` / an operator's `override`) was
resolved afterwards, from the host's own CUDA probe, and stored in the completion marker for a
defensive comparison on reuse.

A CPU node and a GPU node both target `venvs/crisperwhisper`. One resolves `cpu`, the other
`cu128`. Each sees the other's marker as a mismatch and rebuilds — over the venv the other host is
using. Measured on ORCD: two jobs collided twice in one run and left a venv corrupted, missing
`libscipy_openblas64_*.so` (`specs/20260817-triage-workflow-dag/benchmarks/orcd-scheduling-2026-09-08.md`).

`cdde8ed8` closed a related race — `ensure_venv` now refuses to certify a venv if it lost its lock
mid-build — but that only serializes the rebuilds. It does not stop two device classes from
targeting one directory in the first place.

## The rule

A venv's directory reflects the dependencies it actually installs:

- **Device-independent → one shared directory.** A backend with no `torch` / `torchaudio` in
  `requirements` resolves no index (`torch index: n/a (torch-free)` in the existing log line) and
  keeps the bare `name`. Every host builds the identical venv, so nothing is rebuilt needlessly.
- **Device-dependent → keyed by the dependency.** A backend that does resolve an index gets that
  index's `TorchIndex.tag` folded into the directory: `crisperwhisper-cu128`, `crisperwhisper-cpu`,
  `crisperwhisper-override`, and so on. Two different tags can no longer collide on one directory.

## The reorder, and why it is safe

Naming the directory by tag requires resolving the tag before choosing the directory. The
resolution — `_torch_install_specs(requirements)`, then (if non-empty) `detect_host_cuda()` and
`pick_torch_index(probed, env_override=..., max_cuda_version=...)` — depends only on
`requirements`, the host probe, `SENSELAB_TORCH_INDEX_URL`, and the caller's `max_cuda_version`.
None of those four inputs reads `venv_dir`, the lock, or the marker, so moving the whole block
ahead of `venv_dir = _cache_dir() / dir_name` changes nothing about what it computes — only when.
Confirmed by reading `ensure_venv` top to bottom before moving anything: no other statement between
the old and new call sites touches `torch_specs`, `host_cuda`, or `torch_index`.

The per-call `SharedFileLock` still locks the (now tag-suffixed) `venv_dir`, so two builds racing
for the *same* resolved tag are still serialized exactly as before; they simply can no longer be
two builds for *different* tags fighting over one path.

## The marker comparison stays, demoted to defensive

`ensure_venv` still compares the marker's stored `torch_index.url` against the freshly resolved
one before trusting a cache hit. Before this change that comparison was load-bearing: it was the
only thing standing between a CPU node's marker and a GPU node's expectations, in the one directory
they shared. After this change it is unreachable in that scenario, because a `cu128` resolution and
a `cpu` resolution no longer address the same directory at all. It stays as a defensive check —
if a marker inside a `t-different-index-cu128` directory ever carries a `cu121` index, the mismatch
means `dir_name` itself was computed wrong, not that a legitimate device change occurred, and the
new `logger.error` on that path says so.

## Migration: pre-alpha, no shim

Rename and replace outright — no parallel bare-name path, no fallback, no alias. Every existing
device-dependent venv at its old bare path (`crisperwhisper`, `nemo-canary-qwen`, `qwen-asr`, and
any other torch-bearing backend) becomes an **orphan** the moment this ships: nothing will look
there again, `ensure_venv` will build fresh under the tag-suffixed name on next use, and the old
directory is never cleaned up automatically.

**Operators should delete the old bare-name directories under `~/.cache/senselab/venvs/`** (or
wherever `SENSELAB_VENV_CACHE` points) once they've confirmed the new tag-suffixed ones are in
place — they are dead weight, not a fallback.

**The first run after this ships rebuilds each affected venv once.** Measured on ORCD:
590–800 s per torch-bearing venv (a Stage-1 + Stage-2 `uv pip install` of `torch` + `torchaudio` +
the backend's own dependencies). This is a one-time cost per host per backend, not a regression —
worth saying plainly before someone reports the first post-migration run as slow.

Torch-free venvs (`yamnet`, `hear`, `continuous-ser`) are unaffected: their directory name does not
change, so nothing rebuilds for them.

## Known pre-existing gap: three GPU-integration tests hardcode the old bare path

`src/tests/audio/tasks/speech_to_text/crisperwhisper_test.py`,
`src/tests/audio/tasks/speech_to_text/qwen_test.py`, and
`src/tests/audio/tasks/speech_to_text/canary_qwen_test.py` each build
`Path.home() / ".cache" / "senselab" / "venvs" / "<name>"` directly (not via `ensure_venv` /
`_cache_dir`) to gate a `skipif` for their real-model integration test. All three backends
(`crisperwhisper`, `qwen-asr`, `nemo-canary-qwen`) declare `torch`, so after this change their
actual directories carry a tag suffix (`crisperwhisper-cu128`, etc.) that these hardcoded paths do
not know about. Left as-is, the `skipif` gate now always evaluates "not provisioned" on a
freshly-migrated host, and the GPU integration tests silently skip forever instead of running when
a provisioned venv exists.

This is outside this change's stated scope (`subprocess_venv.py`, its helpers, and their tests) and
is not fixed here. It is recorded so it is not rediscovered as a mystery regression: any follow-up
touching these three test files should route their skip-gate through `ensure_venv`'s own directory
resolution (or otherwise account for the tag suffix) rather than reconstructing the path by hand.
