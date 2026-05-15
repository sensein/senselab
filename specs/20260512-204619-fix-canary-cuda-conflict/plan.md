# Implementation Plan: Resolve NeMo Canary torch/torchaudio CUDA mismatch on newer-CUDA hosts

**Branch**: `20260512-204619-fix-canary-cuda-conflict` | **Date**: 2026-05-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260512-204619-fix-canary-cuda-conflict/spec.md`

## Summary

Subprocess-venv backends (Canary-Qwen, NeMo Conformer/Sortformer, Qwen-ASR) provision their own isolated `uv pip install`-ed environment via `src/senselab/utils/subprocess_venv.py::ensure_venv`. The current install call lets the default PyPI index choose `torch` and `torchaudio` binaries, which on hosts with system CUDA newer than the default-wheels CUDA can resolve the two packages to mismatched CUDA toolchains and break their ABI contract at import time. The fix: detect the host CUDA version once, select a matching official PyTorch wheel index (cuXX), and route the venv's `torch`+`torchaudio` install through that index — falling back to CPU wheels when no compatible CUDA index exists, with a clear named diagnostic in the unsupported case. Done at the shared `ensure_venv` level so all three current subprocess backends (and any future one) get the fix without per-backend code change.

## Technical Context

**Language/Version**: Python 3.11–3.14 (matches senselab's `requires-python = ">=3.11,<3.15"`).
**Primary Dependencies**: `uv` (managed installer), `torch>=2.8,<2.9`, `torchaudio>=2.8,<2.9` (currently pinned at the subprocess-venv definitions), `nemo_toolkit[asr,tts]`, `qwen-asr`. Fix introduces no new runtime dependency.
**Storage**: File-based — venvs live under `~/.cache/senselab/venvs/<name>/`, marker file `.senselab-installed` records the current resolved requirement set.
**Testing**: `uv run pytest` for unit + integration on the new CUDA-detection / index-selection helper. End-to-end validation on a real CUDA 12.9 host is out-of-band (no CUDA 12.9 in the project's CI runners).
**Target Platform**: Linux x86_64 (primary — that's where CUDA hosts are), macOS arm64 (must remain CPU/MPS, no regressions), Windows (out of scope per existing project posture).
**Project Type**: Python library + CLI wrapper. The change lands in the library (`src/senselab/utils/subprocess_venv.py`); the CLI (`scripts/analyze_audio.py`) is unaffected.
**Performance Goals**: First-run venv preparation on a CUDA host should not exceed today's runtime by more than ~10% (one extra `nvidia-smi` / `nvcc --version` invocation at venv-build time). Subsequent runs hit the existing marker-file cache and are unchanged.
**Constraints**: No network calls beyond what `uv pip install` already performs; the CUDA probe runs only once per `ensure_venv` invocation; the resolved CUDA index is recorded in the marker so re-runs don't need to re-probe.
**Scale/Scope**: Affects every subprocess-venv backend (today: `nemo-canary-qwen`, `nemo`, `qwen-asr`). All three call the same `ensure_venv` helper, so a single change covers them.

## Constitution Check

The repo constitution at `.specify/memory/constitution.md` enforces eight principles. Evaluation:

- **I. UV-Managed Python** — ✅ The fix continues to use `uv pip install` via the existing `_find_uv()` path. No bare `python`/`pip`. The CUDA probe itself shells out to `nvidia-smi` / `nvcc` — those are external tools, not Python package managers, so they don't violate this principle.
- **II. Encapsulated Testing** — ✅ Tests for the new CUDA-detection helper are pure-Python unit tests under `src/tests/utils/` and run inside the project's uv-managed test environment. No host-CUDA-required tests will land in CI.
- **III. Commit Early and Often** — ✅ Plan partitions the work into discrete commits per task in `tasks.md` (one for the probe helper, one for the install-route plumb, one for marker-schema bump, one for docs).
- **IV. CI Must Stay Green** — ✅ All new tests are CPU-only. The existing pre-commit + cpu-tests (3.12) lanes must pass on the PR.
- **V. Memory-Driven Anti-Pattern Avoidance** — ✅ Two prior memories apply directly and are honored:
  - `feedback_subprocess_hangs_are_machine_contention.md` — don't disable backends to work around timeouts. This fix is about install-time resolution, not runtime, so the memory doesn't conflict.
  - `project_subprocess_error_propagation.md` — propagate original error types through subprocess boundary. The new diagnostic in FR-004 is itself a structured failure with a named cause, satisfying this.
- **VI. No Unnecessary API Calls** — ✅ The CUDA probe runs at most once per `ensure_venv` call. Result is cached in the marker file so subsequent runs hit the existing cache fast-path and don't re-probe.
- **VII. Simplicity First** — ✅ The simplest viable mechanism is: probe CUDA → pick `cpu` or one of a small fixed set of `cuXX` indexes → pass `--index-url` to the existing `uv pip install`. No new abstraction layer, no plugin registry — one helper function, one extra argument plumbed through.
- **VIII. No Hardcoded Parameters** — ⚠️→✅ The PyTorch wheel index URL pattern (`https://download.pytorch.org/whl/cu<NN>`) is the canonical upstream URL. The fix introduces this URL as a constant in `cuda_probe.py` but exposes the resolved CUDA tag via the marker file (auditable post-install) and accepts an env-var override (`SENSELAB_TORCH_INDEX_URL`) for users who mirror PyPI internally. This satisfies the principle's intent — operators can override without code change.

No gates fail.

## Project Structure

### Documentation (this feature)

```text
specs/20260512-204619-fix-canary-cuda-conflict/
├── plan.md              # This file
├── research.md          # Phase 0 output — CUDA detection options + index choice rationale
├── data-model.md        # Phase 1 — marker schema bump + probe-result entity
├── quickstart.md        # Phase 1 — install + first-run validation on a CUDA-12.9 host
├── contracts/
│   ├── ensure_venv.md   # Phase 1 — the public function's contract change
│   └── env-vars.md      # Phase 1 — SENSELAB_TORCH_INDEX_URL override semantics
├── checklists/
│   └── requirements.md  # Spec quality checklist (passed)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/senselab/utils/
├── subprocess_venv.py      # ensure_venv: add CUDA-aware torch index routing
└── cuda_probe.py           # NEW: detect_host_cuda() + pick_torch_index() helpers

src/senselab/audio/tasks/speech_to_text/
├── canary_qwen.py          # Unchanged code path; reads new behavior transparently via ensure_venv
├── nemo.py                 # Same
└── qwen.py                 # Same (picks up CUDA-routed torchaudio appended by ensure_venv)

src/tests/utils/
├── cuda_probe_test.py      # NEW: unit tests for detection + index picking (mocked subprocess)
└── subprocess_venv_test.py # NEW or amended: marker-schema upgrade behavior
```

The fix is small: one new file (`cuda_probe.py`), one modified file (`subprocess_venv.py`), two test files. No backend file needs to change.

## Complexity Tracking

| Concern | Why it's worth complexity-budget | What controls it |
|---|---|---|
| Probe is shell-shellouty | Has to work without already-installed `torch` (chicken-and-egg) — so `torch.version.cuda` is not the primary probe; `nvidia-smi` and `nvcc --version` are. | Probe is one ~40-line function, well-tested with mocked `subprocess.run`. |
| Mapping CUDA→index | PyTorch supports a fixed set of `cuXX` indexes that lags real CUDA versions; we have to pick the highest available index `≤` host CUDA. | Map is a literal list of supported indexes, kept short (4–5 entries) and updated explicitly when PyTorch adds a new index. |
| Marker extension | Old marker files written by today's code don't carry the resolved CUDA tag; we need to re-resolve on first run after upgrade. | Add a `torch_index` field; absence of the field counts as mismatch → rebuild (same path as a changed `requirements` list). |
| Env-var override | Operator escape hatch; required by Constitution VIII. | Single env var name `SENSELAB_TORCH_INDEX_URL`; documented in contracts/env-vars.md. |

No complexity item rises to the "needs a separate design doc" threshold.

## Phase 0 — Research

Stored at `research.md` (this section is the source-of-record). Key decisions:

### D1: How to detect host CUDA without importing `torch`

**Decision**: `nvidia-smi --query-gpu=driver_version --format=csv,noheader` (driver-derived CUDA) → fallback `nvcc --version` → fallback "no CUDA, use CPU index".
**Rationale**: At `ensure_venv` time, the target venv doesn't exist yet, so we can't `import torch`. The host's main venv might have a `torch` build that disagrees with the system CUDA, so trusting `torch.version.cuda` from the main venv is wrong. `nvidia-smi` ships with the NVIDIA driver and is the single source of truth for the runtime CUDA available to user processes.
**Alternatives considered**: (a) parsing `/usr/local/cuda/version.txt` — fragile, distro-dependent, won't work in container with bind-mounted CUDA. (b) reading `LD_LIBRARY_PATH` for `libcudart.so.X` — also fragile.

### D2: How to map host CUDA to a PyTorch wheel index

**Decision**: Pick the highest official `cuXX` index `≤` host CUDA from a fixed list updated explicitly when PyTorch adds support. As of this spec: `cu128`, `cu126`, `cu124`, `cu121`. Host CUDA 12.9 → pick `cu128` (back-compatible with newer driver). No matching index → `cpu`.
**Rationale**: PyTorch wheels for cuXX are back-compatible to that CUDA's minor via the NVIDIA driver's forward-compat shim. The fixed list avoids an HTTP probe at install time.
**Alternatives considered**: HTTP-probe `download.pytorch.org/whl/cuXX/` listings at runtime — adds network dependency and is slower than a static list. The static list cost is a one-line update when PyTorch publishes `cu129`/`cu130`.

### D3: Where to plumb the index URL

**Decision**: Inside `ensure_venv`, after computing the full `all_reqs` list, prefix the `uv pip install` argv with `--index-url <chosen>` and `--extra-index-url https://pypi.org/simple` so non-torch packages still resolve from PyPI.
**Rationale**: Single call site, applies uniformly to every subprocess backend. The `--extra-index-url` lets `nemo_toolkit[asr,tts]` etc. continue to resolve from PyPI while `torch` / `torchaudio` go through the PyTorch index.
**Alternatives considered**: A second `uv pip install` call (one for torch via the special index, one for everything else from PyPI). Rejected — increases install time and makes resolution non-atomic; uv may end up with two different `torch` builds in the same env.

### D4: How to communicate "no compatible CUDA wheels exist" to the user

**Decision**: After `pick_torch_index()` returns either a `cuXX` choice or `cpu`, attempt the install once. On `uv pip install` failure with a recognizable signature (no matching distribution), wrap the error into a `SenselabCudaCompatibilityError` carrying: detected CUDA version, attempted index, the failing package(s), and a one-line action ("downgrade CUDA, fall back to CPU by setting `SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`, or wait for upstream wheels").
**Rationale**: Satisfies FR-004 (single named actionable error) and SC-003 (no deep stack traces).
**Alternatives considered**: Pre-flight HEAD-request to the index URL before installing. Rejected — adds a network call and still doesn't catch all failure modes (e.g. index exists but specific version missing). The "try install, wrap failure" pattern is simpler and catches everything.

### D5: Stale marker / partial-install recovery

**Decision**: Extend the marker dict to include a `torch_index` field. Absence of `torch_index` in an existing marker counts as a mismatch → venv rebuild. This is exactly the existing `stored.get("requirements") == sorted(requirements)` mismatch path, with one new clause. No explicit `schema_version` — the marker is internal cache state, not a published contract.
**Rationale**: Satisfies FR-005 (auto-recovery from prior failed install) and SC-004 (single re-run recovery). Uses the existing rebuild path so no new logic.
**Alternatives considered**: A separate "broken-venv" detection by trying to import torch in the venv and catching failures. Rejected — too much heuristic, brittle in subtle import-error landscape.

### D6: Env-var override for mirrored PyPI / internal indexes

**Decision**: `SENSELAB_TORCH_INDEX_URL` — when set, overrides everything in `pick_torch_index()` and uses the operator's URL verbatim. Recorded in the marker so an unexpected change of env-var triggers a rebuild.
**Rationale**: Constitution VIII (No Hardcoded Parameters). Operators on air-gapped or internally-mirrored hosts can route through their mirror.
**Alternatives considered**: A config file at `~/.config/senselab/torch_index.toml`. Rejected — heavier than necessary; env-var is the standard escape hatch for this kind of operator override in Python tooling.

## Phase 1 — Design & Contracts

### Data model (`data-model.md`)

Two entities:

**HostCuda** — value object produced by `cuda_probe.detect_host_cuda()`:
- `version`: `tuple[int, int]` or `None` (None = no CUDA)
- `source`: one of `"nvidia-smi"`, `"nvcc"`, `"none"`
- `raw`: raw string the probe parsed (for diagnostics)

**TorchIndex** — value object produced by `cuda_probe.pick_torch_index(host_cuda, env_override=None)`:
- `url`: full HTTPS URL to the index
- `tag`: short identifier (`"cu128"`, `"cpu"`, `"override"`)
- `cuda_version`: `tuple[int, int]` or `None` (the index's CUDA, not the host's)
- `source`: one of `"static-map"`, `"env-override"`

**Marker schema** (`.senselab-installed`):
- Today: `{"requirements": [...], "python_version": "3.12"}`
- After this fix: `{"requirements": [...], "python_version": "3.12", "torch_index": {"tag": "cu128", "url": "..."}}`
- Compatibility: marker is internal cache state, not a published contract. No explicit version field — absence of `torch_index` in an existing marker counts as a mismatch → rebuild, same as any other field change.

### Contracts (`contracts/`)

Two contracts:

1. **`ensure_venv.md`** — the function's interface contract change: argv it passes to `uv pip install` now includes `--index-url ... --extra-index-url https://pypi.org/simple`; marker dict gains a `torch_index` field; behavior unchanged on the cached fast-path.
2. **`env-vars.md`** — `SENSELAB_TORCH_INDEX_URL` override semantics: when set to any value, it is passed verbatim as `--index-url`; when unset, the static map applies. Listed in repo CLAUDE.md and the docs.

### Quickstart (`quickstart.md`)

Steps to validate on a CUDA-12.9 host:

1. `uv sync --extra text --extra video --extra senselab-ai --extra nlp --extra pii --group dev`
2. `rm -rf ~/.cache/senselab/venvs/nemo-canary-qwen` (recovery from any pre-upgrade broken venv)
3. `uv run python scripts/analyze_audio.py path/to/short.wav --asr-models nvidia/canary-qwen-2.5b --no-enhancement --skip comparisons`
4. Expected: Canary transcript appears in `artifacts/analyze_audio/.../raw_16k/asr/nvidia_canary_qwen_2_5b.json`, marker file at `~/.cache/senselab/venvs/nemo-canary-qwen/.senselab-installed` carries `"torch_index": {"tag": "cu128", ...}`.
5. Verify `torch.__version__` and `torchaudio.__version__` inside the venv have matching CUDA suffix: `~/.cache/senselab/venvs/nemo-canary-qwen/bin/python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"` — both should end in `+cu128` (or `+cpu` on a no-GPU host).

Negative case (simulated unsupported CUDA):

1. `SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/nonexistent uv run python scripts/analyze_audio.py path/to/short.wav --asr-models nvidia/canary-qwen-2.5b`
2. Expected: single named error `SenselabCudaCompatibilityError`, exit non-zero, no half-built venv at `~/.cache/senselab/venvs/nemo-canary-qwen/`.

### Agent context update

Plan invokes `.specify/scripts/bash/update-agent-context.sh claude` so the project's `CLAUDE.md` adds:

- `SENSELAB_TORCH_INDEX_URL` to the env-var index
- A note in the "Subprocess venv" section about the CUDA-aware install routing

The script preserves any manual additions between markers; no risk of clobbering existing CLAUDE.md content.

## Re-evaluation: Constitution Check (post-design)

All eight principles re-checked against the Phase 0+1 design:

- I, II, III: unchanged.
- IV: still CPU-only tests; OK.
- V: no new anti-patterns introduced; relevant memories honored.
- VI: probe runs once per ensure_venv invocation, cached in marker; OK.
- VII: ~40 lines new probe code + ~10 lines plumb + marker schema bump. No new abstractions beyond the two value objects (HostCuda, TorchIndex) which are necessary to test the probe in isolation.
- VIII: env-var override exists; resolved index URL is auditable in the marker.

No new gate violations.

---

**Plan complete.** Phase 2 (`/speckit.tasks`) will translate this into a task list with concrete file paths and acceptance criteria per task.
