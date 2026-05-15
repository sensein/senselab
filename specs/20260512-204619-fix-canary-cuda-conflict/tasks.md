# Tasks: Resolve NeMo Canary torch/torchaudio CUDA mismatch on newer-CUDA hosts

**Feature**: 20260512-204619-fix-canary-cuda-conflict
**Spec**: [spec.md](spec.md)  |  **Plan**: [plan.md](plan.md)
**Stack**: Python 3.11–3.14, uv, existing senselab subprocess-venv harness. New module: `src/senselab/utils/cuda_probe.py`. Tests: `src/tests/utils/`.

User stories from spec (priority order):
- **US1 (P1)** — First-time install on a newer-CUDA host (Canary works on CUDA 12.9)
- **US2 (P2)** — Diagnostic when no compatible binary set is available
- **US3 (P3)** — Same fix shape applies to other isolated-environment backends

---

## Phase 1 — Setup

Nothing to set up — the project is established, `uv` + the subprocess-venv harness are already in place. Skip to Phase 2.

---

## Phase 2 — Foundational (blocks every user story)

The new probe helper + value objects + error class are required by all three stories. Land them first.

- [X] T001 Create `src/senselab/utils/cuda_probe.py` with the module docstring explaining the role of the probe (host CUDA detection at `ensure_venv` time, must work without `torch` installed, parses `nvidia-smi` / `nvcc` output directly with no driver→CUDA table), and define the three exported names: `HostCuda` (frozen dataclass), `TorchIndex` (frozen dataclass), `SenselabCudaCompatibilityError` (RuntimeError subclass). Fields per `data-model.md`.
- [X] T002 [P] Implement `detect_host_cuda() -> HostCuda` in `src/senselab/utils/cuda_probe.py` using the D1 probe order: `nvidia-smi` (no extra args, parse the `CUDA Version: X.Y` line from its default header) → `nvcc --version` (parse `release X.Y` line) → `HostCuda(version=None, source="none", raw="")`. Honor a 5-second per-probe subprocess timeout. No driver→CUDA mapping needed — both probes report the CUDA version directly.
- [X] T003 [P] Implement `pick_torch_index(host_cuda, env_override=None) -> TorchIndex` in `src/senselab/utils/cuda_probe.py` with the D2 static map: `[("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1))]` (highest first). When `env_override` is set + non-empty → return `TorchIndex(url=env_override, tag="override", cuda_version=None, source="env-override")`. When `host_cuda.version is None` → `TorchIndex(url="https://download.pytorch.org/whl/cpu", tag="cpu", cuda_version=None, source="static-map")`. Otherwise pick the highest map entry whose CUDA tuple is `≤` host's; if none match → CPU index.
- [X] T004 [P] Write unit tests for the probe + picker in `src/tests/utils/cuda_probe_test.py`. Cover at minimum: (a) `nvidia-smi` reports a known driver string for CUDA 12.9 → `HostCuda(version=(12, 9), source="nvidia-smi")`; (b) `nvidia-smi` missing → falls through to `nvcc`; (c) both missing → `HostCuda(version=None, source="none")`; (d) host CUDA 12.9 → picker chooses `cu128`; (e) host CUDA 12.1 → picker chooses `cu121`; (f) host CUDA 11.8 (below map) → CPU; (g) `env_override` non-empty → `tag="override"`, `source="env-override"`; (h) `env_override = ""` (empty string) → treated as unset. Mock `subprocess.run`; do NOT shell out on CI.
- [X] T005 Run `uv run pytest src/tests/utils/cuda_probe_test.py -v` and `uv run ruff check src/senselab/utils/cuda_probe.py src/tests/utils/cuda_probe_test.py` to confirm Phase 2 lands green before any subprocess_venv changes.

**Checkpoint**: After Phase 2, `cuda_probe` is self-contained, tested, and unused. `ensure_venv` still has its v1 behavior. No backend has been touched. CI is green.

---

## Phase 3 — User Story 1 (P1): First-time install on a newer-CUDA host

**Goal**: A CUDA 12.9 host runs `analyze_audio.py` with `--asr-models nvidia/canary-qwen-2.5b` and gets a Canary transcript with no manual workaround.

**Independent test** (matches spec US1 Acceptance Scenarios):

1. On a CUDA 12.9 host, with no senselab venv pre-built, run `uv run python scripts/analyze_audio.py path/to/short.wav --asr-models nvidia/canary-qwen-2.5b --no-enhancement --skip comparisons`.
2. Verify the Canary subprocess venv builds without import error, its marker carries `"torch_index": {"tag": "cu128", ...}`, and a transcript JSON appears in `artifacts/analyze_audio/.../raw_16k/asr/nvidia_canary_qwen_2_5b.json`.
3. On a host with CUDA ≤ 12.8, the same command takes within +5% of pre-fix wall time (no regression).

### Tasks for US1

- [X] T010 [US1] In `src/senselab/utils/subprocess_venv.py`, import `detect_host_cuda`, `pick_torch_index`, `SenselabCudaCompatibilityError`, and `os` (for env-var read) at the module top. No behavior change yet — pure imports.
- [X] T011 [US1] In `ensure_venv` in `src/senselab/utils/subprocess_venv.py`, extend the marker-match check (after the marker-read block) to also require `stored.get("torch_index", {}).get("url") == <resolved index URL>`. Absence of `torch_index` in an existing marker counts as mismatch → rebuild. To make that check meaningful before the install runs, resolve the index FIRST inside the lock, in this exact order so a configured override avoids any unnecessary `nvidia-smi` shellout:

    1. `env_override = os.getenv("SENSELAB_TORCH_INDEX_URL") or None` (treat empty string as unset)
    2. If `env_override` is set, **skip** `detect_host_cuda()` (the override wins regardless of host CUDA) and pass `HostCuda(version=None, source="none", raw="")` to the picker — the picker short-circuits to a `tag="override"` `TorchIndex`. Otherwise call `detect_host_cuda()`.
    3. `torch_index = pick_torch_index(host_cuda, env_override=env_override)`

    Cache `torch_index` on a local variable for the marker comparison, install argv, and marker write.
- [X] T012 [US1] In the same function in `src/senselab/utils/subprocess_venv.py`, modify the `uv pip install` argv (at the existing `uv pip install ... all_reqs` shellout site) to prepend `--index-url <torch_index.url>` and `--extra-index-url https://pypi.org/simple` before the existing arguments. Leave the rest of the argv (`--python`, the requirements list) intact.
- [X] T013 [US1] In `ensure_venv` in `src/senselab/utils/subprocess_venv.py`, extend the marker-write dict to include the resolved index: `{"requirements": sorted(requirements), "python_version": py_ver, "torch_index": {"tag": torch_index.tag, "url": torch_index.url, "source": torch_index.source}}`.
- [X] T014 [P] [US1] Write the marker-mismatch tests in `src/tests/utils/subprocess_venv_test.py`: (a) marker without `torch_index` key (old shape) → mismatch, rebuild triggered; (b) marker with matching `torch_index.url` → cache hit (no rebuild); (c) marker with non-matching `torch_index.url` → mismatch, rebuild triggered. Use `tmp_path` for the cache dir; monkeypatch `_cache_dir()`; mock `subprocess.run` so no actual uv invocation happens.
- [X] T015 [P] [US1] Add an integration test in `src/tests/utils/subprocess_venv_test.py` that calls `ensure_venv("test-venv", ["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"])` with `subprocess.run` mocked to record the argv. Assert the recorded argv contains `--index-url` followed by the URL that `pick_torch_index(detect_host_cuda())` would return on the test host, and `--extra-index-url https://pypi.org/simple`.
- [X] T016 [US1] Run `uv run pytest src/tests/utils/ -v` and `uv run pre-commit run --all-files`. Confirm green before merge of US1 scope.

**Checkpoint**: US1 is complete. On a CUDA 12.9 host, the Canary venv now installs with matching `cu128` torch+torchaudio. On a CPU host, it installs with matching `cpu` wheels. The diagnostic path (US2) is not yet implemented — failures still surface as raw `CalledProcessError`.

---

## Phase 4 — User Story 2 (P2): Diagnostic when no compatible binary set is available

**Goal**: When the chosen index has no installable wheel pair, the failure is a single named actionable error instead of an opaque uv stack trace.

**Independent test** (matches spec US2 Acceptance Scenarios):

1. Run with `SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/nonexistent` to force the install to fail with "no matching distribution".
2. Verify the raised exception is `SenselabCudaCompatibilityError`, its `__str__` contains the host CUDA version, the attempted index, and the failing packages.
3. Verify `~/.cache/senselab/venvs/<name>/` does not exist after the failure (cleanup happened before the raise).

### Tasks for US2

- [X] T020 [US2] In `ensure_venv` in `src/senselab/utils/subprocess_venv.py`, wrap the `uv pip install` call (the one modified by T012) in a `try/except subprocess.CalledProcessError`. On failure: (a) delete `venv_dir` with `shutil.rmtree(venv_dir, ignore_errors=True)`; (b) classify the failure via a new private helper `_classify_uv_failure(stderr: str) -> list[str] | None` (returns the list of failing packages when stderr matches "no matching distribution" / "could not find a version that satisfies" patterns, else `None`); (c) if classified: raise `SenselabCudaCompatibilityError(host_cuda=..., attempted_index=..., failing_packages=...)` with `from exc`; (d) else: re-raise the original `CalledProcessError`.
- [X] T021 [US2] Implement `SenselabCudaCompatibilityError.__str__` in `src/senselab/utils/cuda_probe.py` to produce the multi-line message defined in `contracts/ensure_venv.md`: host CUDA, attempted index URL, failing packages, and the one-line action including the `SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` recommendation.
- [X] T022 [P] [US2] Add a unit test for `_classify_uv_failure` in `src/tests/utils/subprocess_venv_test.py` covering: (a) uv stderr containing "No solution found when resolving dependencies: ... no matching distribution found for torch==..." → returns `["torch==..."]`; (b) uv stderr "Could not find a version that satisfies the requirement torchaudio>=2.8,<2.9" → returns `["torchaudio>=2.8,<2.9"]`; (c) uv stderr from an unrelated failure (network timeout, permission error) → returns `None`.
- [X] T023 [P] [US2] Add an integration test in `src/tests/utils/subprocess_venv_test.py` that mocks `subprocess.run` to raise `CalledProcessError` with a "no matching distribution" stderr. Call `ensure_venv("test-venv", ["torch>=2.8,<2.9"])` and assert: (a) `SenselabCudaCompatibilityError` is raised; (b) `str(exc)` contains "Host CUDA", "Attempted index", and "Failing packages"; (c) the test venv directory does not exist after the call.
- [X] T024 [P] [US2] Add an integration test in `src/tests/utils/subprocess_venv_test.py` for the unrelated-failure pass-through path: mock `subprocess.run` to raise `CalledProcessError` with a network-error stderr. Assert: (a) `CalledProcessError` is re-raised (not `SenselabCudaCompatibilityError`); (b) the test venv directory does not exist (cleanup still happened).
- [X] T025 [US2] Run `uv run pytest src/tests/utils/ -v` and confirm green.

**Checkpoint**: US2 is complete. Unsupported-CUDA failures are now named and actionable; the cache directory is always cleaned on failure regardless of which error path triggered.

---

## Phase 5 — User Story 3 (P3): Same fix shape applies to other isolated-environment backends

**Goal**: Confirm `nemo` and `qwen-asr` subprocess venvs automatically pick up the same CUDA-aware install behavior with no per-backend code change.

**Independent test** (matches spec US3 Acceptance Scenarios):

1. With the fix landed, build the `nemo` venv (e.g. via `--asr-models nvidia/parakeet-tdt-0.6b-v3`) and the `qwen-asr` venv (e.g. via `--asr-models Qwen/Qwen3-ASR-1.7B`) on a CUDA 12.9 host.
2. Verify both venvs have markers carrying `"torch_index": {"tag": "cu128", ...}` and their respective `torch.__version__` / `torchaudio.__version__` end in `+cu128`.
3. Confirm no call-site code in `canary_qwen.py`, `nemo.py`, or `qwen.py` had to change.

### Tasks for US3

- [X] T030 [US3] Verify by reading `src/senselab/audio/tasks/speech_to_text/nemo.py` and `src/senselab/audio/tasks/speech_to_text/qwen.py` that both still call `ensure_venv(name, REQUIREMENTS, python_version=...)` with no extra arguments and no per-backend `uv pip install` shellout that bypasses it. If both backends use `ensure_venv` as expected, this task is complete — no code change. If either backend bypasses `ensure_venv`, FILE A NEW TASK (T034) under Phase 5 in this same `tasks.md` to migrate it to `ensure_venv` before this feature merges. Do not mark T030 complete with an outstanding bypass. **Verified**: both backends call `ensure_venv` correctly; only their post-install worker subprocess calls remain (lines 132 + 180), which are not install-time bypasses.
- [X] T031 [P] [US3] Add a parameterized integration test in `src/tests/utils/subprocess_venv_test.py` that walks the three real backend requirement lists — `_CANARY_REQUIREMENTS` from `canary_qwen.py`, `_NEMO_REQUIREMENTS` from `nemo.py`, `_QWEN_REQUIREMENTS` from `qwen.py` — through `ensure_venv` (with `subprocess.run` mocked). For each, assert that the recorded `uv pip install` argv contains the same `--index-url <pick_torch_index>.url --extra-index-url https://pypi.org/simple` pair. Catches any future regression where a backend bypasses the shared resolver.
- [X] T032 [P] [US3] If `qwen-asr` does not list `torch` / `torchaudio` directly in its requirements (it pins them transitively via the `qwen-asr` package), add a docstring note to `src/senselab/audio/tasks/speech_to_text/qwen.py` near `_QWEN_REQUIREMENTS` pointing out that the CUDA-routed `torchaudio` appended by `ensure_venv` is what protects this venv from the mismatch, and recommending that any future direct `torch`/`torchaudio` pins added here continue to flow through `ensure_venv`.
- [X] T033 [US3] Run `uv run pytest src/tests/utils/ -v` and confirm green. Optionally (gated on availability of a CUDA-12.9 host) trigger an end-to-end run via the `nemo` and `qwen-asr` backends per the quickstart's "Smoke check for the other subprocess venvs" section and attach the marker JSONs as validation artifacts.

**Checkpoint**: US3 is complete. The three current subprocess backends provably share the resolution mechanism; the regression test in T031 guards against any future bypass.

---

## Phase 6 — Polish & Cross-Cutting Concerns

- [X] T040 Update `CLAUDE.md` (root): add a `## CUDA host configuration` subsection explaining that subprocess venvs auto-pick the matching PyTorch wheel index, and listing `SENSELAB_TORCH_INDEX_URL` as the override. Cross-link to `specs/20260512-204619-fix-canary-cuda-conflict/quickstart.md`.
- [X] T041 [P] Update `README.md`: add a "Troubleshooting / CUDA hosts" section pointing at the env-var override and the quickstart's negative-path recipe.
- [X] T042 [P] Add a short entry to `pyproject.toml`'s top-of-file comment block (or to the spec's own doc) noting that `torch` / `torchaudio` pins inside subprocess venvs are no longer enough on their own — the `ensure_venv` resolution is what guarantees the ABI match. **Landed in-source instead** (`canary_qwen.py`, `nemo.py`, `qwen.py`) — `pyproject.toml` top-comment is stripped by `pretty-format-toml`, and the in-source location is where developers will actually see the warning when editing the requirements list.
- [X] T043 Save a feedback memory note to `~/.claude/projects/-Users-satra-software-sensein-senselab/memory/feedback_subprocess_venv_cuda_routing.md` per Constitution V: "ensure_venv now routes torch/torchaudio via a CUDA-aware PyTorch index; any change to subprocess-venv install logic must preserve this routing. Test in src/tests/utils/subprocess_venv_test.py guards the contract." Update `MEMORY.md` to point at it.
- [X] T044 Open a draft PR against `alpha` containing all commits from this feature branch. PR description summarizes: the bug, the fix shape (probe → pick index → route install), the marker schema bump, the env-var override, the test coverage, and a link to the quickstart's validation steps. CI: pre-commit + build-and-deploy + cpu-tests (3.12). **Opened as https://github.com/sensein/senselab/pull/516.**
- [X] T045 After CI is green, run an ultrareview-style local review using the `general-purpose` agent against the latest two commits (probe + ensure_venv changes). Focus areas: silent-failure hunting (any path that swallows the CalledProcessError without classification?), marker-rebuild semantics (is there any way a marker can survive an aborted install and incorrectly satisfy the next cache-hit check?), env-var sanitization (does an empty string vs unset matter?). Address findings before requesting human review. **Review run; 2 MEDIUM + 3 LOW findings landed as a follow-up commit (host CUDA always probed on override path for accurate diagnostics; failing-package regex anchored to no-matching-distribution phrase; uv venv cleanup on failure; index-map sorted defensively; debug-only logging on the wrap-into-compat-error path).**

---

## Dependencies

```text
Phase 2 (T001–T005) — foundational, blocks everything else
         │
         ├─→ Phase 3 (US1: T010–T016) — primary fix path
         │           │
         │           └─→ Phase 4 (US2: T020–T025) — diagnostic on top of US1
         │
         └─→ Phase 5 (US3: T030–T033) — independent of US2; can land alongside US1

Phase 6 (Polish: T040–T045) — after US1/US2/US3 land
```

### Parallel-execution opportunities

Within Phase 2: T002, T003, T004 can be done in parallel (different functions/test cases, but all in `cuda_probe.py` / `cuda_probe_test.py`). T005 gates the rest.

Within Phase 3: T011, T012, T013 are all in `ensure_venv` and must be sequential (same function body). T014 and T015 are independent test files and can be parallel. T016 gates merge.

Within Phase 4: T020, T021 touch different files and can be parallel after T020's helper is named (one line of pre-coordination); the tests T022, T023, T024 are independent. T025 gates.

Within Phase 5: T031 and T032 are parallel (different files). T030 is a read-only verification.

Within Phase 6: T040, T041, T042 are doc-only and all parallel. T043 is memory-only and parallel with everything. T044 sequences after Phase 5's CI is green. T045 gates merge.

---

## Implementation strategy

**MVP scope** = Phase 2 + Phase 3 (US1).

That alone delivers:
- A CUDA 12.9 user can run analyze_audio with Canary today.
- The cache schema upgrade is in place so old broken venvs auto-rebuild.
- The mechanism is in `ensure_venv` so US3's other backends also benefit (just without the dedicated regression test).

**Incremental delivery**:

1. Ship MVP (Phase 2 + 3) — gets the user unblocked.
2. Ship US2 (Phase 4) — turns scary failures into named errors.
3. Ship US3 (Phase 5) — locks in the regression guard.
4. Polish (Phase 6) — docs + memory + review.

If schedule pressure forces a single-commit minimal fix, T001–T013 is the strict minimum (probe + picker + routed install). T014–T016 (tests + lint) are required to land cleanly in this repo per Constitution IV (CI must stay green). Skipping them is not an option.

---

## Format validation

All tasks above are `- [ ] TXXX [P?] [USx?] description with file path` per the required format:

- ✅ Checkbox at the head of every line.
- ✅ Sequential T001..T045 IDs.
- ✅ `[P]` marker only where the task is parallel-safe (different files or read-only).
- ✅ `[USx]` story label only inside Phase 3/4/5 (story phases); absent in Phase 1/2/6.
- ✅ Every implementation/test task names an exact file path.
