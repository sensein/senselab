# senselab 2.x Phase 1 Implementation Plan: kernel, worker runtime, identity, store, data fetch

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the torch-free foundations of senselab 2.x on `alpha`:
- the uv workspace with `senselab_worker` and `senselab_core`;
- the worker runtime and its wire protocol;
- kernel data and identity types;
- model resolution;
- the result store and provenance;
- the hardware probe;
- the declaration registry with the `fake` stack;
- public-data fetch.

All 1.x behaviour stays unchanged.

**Architecture:**
- `senselab_worker` (stdlib + numpy, Python ≥3.10) holds canonical serialisation, hashing, the
  wire codec and the serve loop.
- `senselab_core` (Python ≥3.11) is a **kernel** (types, identity, declarations; no I/O
  frameworks) plus a **runtime** (resolution, store, provenance, probe, data fetch).
- 1.x modules that move into core are **rewired in the same PR**, with no re-exports. 1.x may
  import core; core never imports 1.x.

**Tech Stack:**
- Python 3.11–3.14 for core and 3.10+ for the worker;
- uv workspace, hatchling + hatch-vcs;
- numpy, pydantic v2, pyyaml, filelock, huggingface-hub;
- pytest + hypothesis;
- import-linter.

**Spec:** `specs/20261030-140022-senselab-2x-api/`, chiefly:
- `design.md` §4.1, §4.8, §4.12, §4.13, §5, §6;
- ADR-002, ADR-005, ADR-006, ADR-009;
- `testing.md`, `docs.md`, `engineering.md`, `principles.md`.

## Global Constraints

- Core: `requires-python = ">=3.11"`. Worker: `requires-python = ">=3.10"` with numpy `>=1.23`
  and no other runtime dependency (P-22).
- Core and worker never import `torch`, `transformers`, `speechbrain`, `pyannote` or `senselab`
  (1.x) (P-13, ADR-009). The worker never imports `pydantic` or `senselab_core`.
- Dependency direction: `senselab_worker` ← `senselab_core.kernel` ← `senselab_core.runtime` ←
  `senselab` (1.x and later 2.x backends). Enforced by import-linter.
- One git tag versions all three distributions (hatch-vcs, root at the repository). Core pins
  `senselab-worker==<same version>` and senselab pins `senselab-core==<same version>` via a
  metadata hook.
- Canonical forms (ADR-006):
  - JSON is RFC 8785 JCS;
  - arrays are little-endian `.npy` format version 1.0, C order;
  - text is UTF-8 NFC;
  - hashes are sha256, lower-case hex.
- Every persisted artifact carries `format: "<name>@<major>.<minor>"` (ADR-006).
- Logical key fields are exactly those of ADR-002 §1. Execution-context fields are exactly those of
  ADR-002 §2. Neither contains function code, Python objects or environments.
- Reuse policies: `exact`, `equivalent` and `tolerant`. With no measured equivalence classes,
  `equivalent` behaves as `exact` (ADR-002, owner default `equivalent`).
- Time is integer nanoseconds on a `clock_id` (ADR-005).
- Tests: `uv run pytest <scoped path>`, serial, never `-n auto` (P-29). New tests carry exactly
  one tier marker (`logic`, `contract`, `limits`, `usecase` or `env`) (P-30). No identifiers in
  fixtures (P-28).
- Prose: docstrings say what and how; no history, no register ids outside `specs/` (P-31).
- Each task is one PR from `2x/p1-<slug>` to `alpha`, merged with a green CI. The PR records the
  implementer and reviewer models (P-40) and runs the `engineering.md` checklist.

## Review Focus

1. **Canonical JSON edge values.** `NaN`, `inf`, `-0.0`, integers above 2⁵³, numpy scalars and
   non-NFC strings in params must either canonicalise deterministically or raise `ValueError`.
   They must never hash two different ways. Pinned in Task 3 (`test_jcs_edge_values`).
2. **Arrays across numpy majors.** An array written by numpy 2.x on the host must decode
   bit-identically in a worker on numpy 1.23, and object and structured dtypes must be refused.
   Pinned in Task 4 (`test_array_roundtrip_across_numpy_majors`, run by the floor job).
3. **Concurrent writers and crash leftovers on a shared store.** Two processes writing the same key
   leave exactly one entry. A crash mid-write leaves no readable entry and is cleaned on the next
   write. Pinned in Task 10 (`test_two_writers_one_entry`, `test_crash_leaves_no_entry`).
4. **Interrupted or tampered downloads.** A partial file, a hash mismatch or a mirror serving
   different bytes must never enter the cache. Pinned in Task 13 (`test_partial_download_not_cached`,
   `test_mirror_hash_mismatch_refused`).
5. **Incomplete rewires.** No import of a moved 1.x module path may remain. That includes strings
   inside worker scripts executed in isolated envs and AST-guard allowlists. Pinned in Tasks 7 and 9
   (`test_no_moved_module_paths_remain`).

## File Structure

```
pyproject.toml                         # root `senselab`: workspace, markers, depends on senselab-core
hatch_build_pins.py                    # metadata hook: exact same-version pins
.importlinter                          # layering contracts
packages/senselab-worker/
  pyproject.toml
  src/senselab_worker/
    __init__.py                        # WIRE_VERSION, runtime_identity()
    canonical.py                       # jcs_dumps, canonical_npy_bytes, nfc
    hashing.py                         # sha256_bytes, content_identity, weights_identity
    codec.py                           # encode/decode messages, array refs
    faults.py                          # FaultClass, classify(exc)
    serve.py                           # serve(load, handle, batch=None)
    fake.py                            # FakeFaults, fake load/handle (the `fake` stack's worker side)
  tests/                               # logic tier; also run from the built wheel by the floor job
packages/senselab-core/
  pyproject.toml
  src/senselab_core/
    kernel/
      time.py                          # TimeAnchor, TimeMap, Span
      signals.py                       # ChannelInfo, RegularGroup, IrregularGroup, Recording, Audio
      results.py                       # Annotation, Embedding, Intervals, Events
      license.py                       # License
      identity.py                      # ModelRef, ResolvedModel, LogicalKey, ExecutionContext, ReusePolicy, logical_key()
      declare.py                       # TaskSpec, BackendSpec, Registry, REGISTRY
      formats.py                       # FORMATS registry, Formatted base
    runtime/
      locks.py                         # moved from senselab/utils/file_lock.py
      revision.py                      # moved from senselab/utils/model_revision.py
      models.py                        # resolve(), verify_weights(), weights manifest loader
      data/weights.yaml                # the former literal sha constants
      provenance.py                    # moved from senselab/utils/prov_store.py
      bep028.py                        # moved from senselab/utils/prov_bep028.py
      store.py                         # ResultStore
      hardware.py                      # probe_hardware(); moved cuda_probe content
      fetch.py                         # DataManifest, fetch()
      worker_client.py                 # WorkerClient (host side of the wire protocol)
    testing/
      synth.py                         # tone, noise, chirp generators
      contracts.py                     # generated contract suite over REGISTRY
    cli.py                             # `senselab data fetch`
  tests/
scripts/check_fixture_budget.py
specs/20261030-140022-senselab-2x-api/wire-protocol.md   # R-37
```

Tasks 2–4, 7–9 and 11 can run in parallel once Task 1 has merged. Dependencies are listed per task.

---

### Task 0: P-39 reconciliation scan at phase start

**Model:** haiku implements (Explore, effort lower); sonnet reviews. **Tier:** none (ledger).
**Implements:** P-39. **Branch:** commits on `senselab-2x-api` (the ledger lives in specs).

- [ ] **Step 1:** Run `git fetch origin alpha && git log 19834fdd..origin/alpha --oneline -- src scripts tutorials`.
- [ ] **Step 2:** If the output is empty, set every `migration-ledger.md` mark to the fetched
  `origin/alpha` sha and log "no drift" in `STATE.md`. Otherwise add one row per commit with its
  planned handling (port in task N, fold into a profile, or drop with a reason).
- [ ] **Step 3:** Commit `docs(2x): P-39 reconciliation at Phase 1 start` and push.

### Task 1: Workspace, packaging and test-tier markers

**Model:** sonnet implements; opus reviews. **Tier:** logic. **Implements:** §5, P-13, P-22, P-30, ADR-009.
**Depends on:** none.

**Files:**
- Create: `packages/senselab-worker/pyproject.toml`, `packages/senselab-worker/src/senselab_worker/__init__.py`.
- Create: `packages/senselab-core/pyproject.toml`, `packages/senselab-core/src/senselab_core/{__init__,kernel/__init__,runtime/__init__}.py`.
- Create: `hatch_build_pins.py`, `.importlinter`.
- Modify: `pyproject.toml` (`[tool.uv.workspace] members = ["packages/*"]`; dependency
  `senselab-core`; `[tool.uv.sources] senselab-core = { workspace = true }`; register markers
  `logic contract limits usecase env gpu network slow large_model`, adding `--strict-markers`).
- Modify: `.github/workflows/tests.yaml` (add the `core-only` job: Python 3.11 and 3.14).
- Test: `packages/senselab-core/tests/test_packaging.py`, `packages/senselab-worker/tests/test_packaging.py`.

**Interfaces:**
- Produces: importable `senselab_worker` and `senselab_core` (empty kernel and runtime); the marker
  set above; `uv run lint-imports` passing.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_core_imports_without_heavy_frameworks():
      out = subprocess.run([sys.executable, "-c", "import senselab_core, sys; print(sorted(m for m in sys.modules if m.split('.')[0] in {'torch','transformers','speechbrain','pyannote','senselab'}))"], capture_output=True, text=True, check=True)
      assert out.stdout.strip() == "[]"

  @pytest.mark.logic
  def test_versions_are_pinned_equal():
      assert metadata.version("senselab-core") == metadata.version("senselab-worker")
      assert f"senselab-worker=={metadata.version('senselab-worker')}" in metadata.requires("senselab-core")
  ```
  Add the worker twin: importing `senselab_worker` loads no `pydantic` and no `senselab_core`.
- [ ] **Step 2:** `uv run pytest packages/senselab-core/tests/test_packaging.py -v` fails (no
  packages yet).
- [ ] **Step 3: Implement.**
  - Each member's `pyproject.toml` uses hatch-vcs with `[tool.hatch.version.raw-options] root = "../.."`
    and the root's `version_scheme`/`local_scheme`.
  - `hatch_build_pins.py` is a hatchling metadata hook (`[tool.hatch.metadata.hooks.custom]`) that
    writes `senselab-worker=={version}` (core) and `senselab-core=={version}` (root) into
    `dependencies`.
  - Core dependencies: `numpy>=1.23`, `pydantic>=2,<3`, `pyyaml`, `filelock`, `huggingface-hub>=1.3`.
  - Worker dependencies: `numpy>=1.23`.
  - `.importlinter` has three contracts:
    - `senselab_worker` forbidden from `senselab_core`, `senselab`, `pydantic`;
    - `senselab_core.kernel` forbidden from `senselab_core.runtime` and `senselab`;
    - `senselab_core` forbidden from `senselab` and from `torch`, `transformers`, `speechbrain`,
      `pyannote`.
  - The `core-only` CI job runs `uv venv -p <py> .core && uv pip install ./packages/senselab-worker ./packages/senselab-core pytest hypothesis && pytest packages -m "logic or contract"`, plus `lint-imports`.
- [ ] **Step 4:** Run `uv sync --all-extras --group dev`, then
  `uv run pytest packages -m "logic or contract" -v`, which passes, then `uv run lint-imports`,
  which passes. Run `uv run pytest src/tests/utils -q` to confirm 1.x still collects with
  `--strict-markers`.
- [ ] **Step 5:** Commit `build(2x): uv workspace with senselab-worker and senselab-core` and open
  the PR to `alpha`.

### Task 2: Wire protocol spec (R-37)

**Model:** opus implements; sonnet reviews. **Tier:** none (spec). **Implements:** R-37, I-2, ADR-009 faults.
**Depends on:** none (parallel with Task 1).

**Files:**
- Create: `specs/20261030-140022-senselab-2x-api/wire-protocol.md` (committed on `senselab-2x-api`).

**Decisions to write down:**
- **Framing.** UTF-8 JSON lines on stdin and stdout. The worker's fd 1 is duplicated to stderr
  after the handshake, as `venv_worker.py` does today. `WIRE_VERSION = 1`.
- **Hello** (worker → host, first line):
  `{"hello": {"wire": 1, "runtime": "<sha256>", "python": "3.10.14", "numpy": "1.23.5", "pid": 123}}`.
  - `runtime` is `runtime_identity()`: the sha256 over the sorted `(relative path, bytes)` of the
    `senselab_worker/*.py` files.
  - The host accepts only if both `wire` and `runtime` equal its own. Otherwise it sends
    `{"stop": true}` and raises `ProtocolMismatch`, an `infrastructure` fault.
- **Init:** `{"init": {...}}`, answered by `{"ready": {"load_s": float}}` or
  `{"error": {...}, "fatal": true}`.
- **Request:** `{"id": "<uuid>", "items": [{"id": "<item>", "inputs": {...}}], "params": {...}}`.
- **Response:** `{"id": "<uuid>", "results": [{"id": "<item>", "outputs": {...}} | {"id": "<item>", "error": {...}}]}`.
  There is one entry per item, so a failure in one item never fails its batch-mates.
- **Arrays:** `{"__array__": {"path": "<dir>/<sha256>.npy", "sha256": "<hex>", "dtype": "<f4", "shape": [2, 16000]}}`.
  The file is canonical `.npy` in a shared temporary directory given at init. The reader verifies
  the sha256.
- **Error:** `{"type": "ValueError", "message": "...", "fault": "content" | "resource" | "infrastructure"}`.
- **Stop:** `{"stop": true}`.
- **Fault table (defaults):**
  - `ValueError`, `TypeError`, `KeyError` → `content`;
  - `MemoryError`, `TimeoutError` → `resource`;
  - anything unclassified → `infrastructure`, reported as a classification gap.

- [ ] **Step 1:** Write `wire-protocol.md` with the decisions above and one example exchange.
- [ ] **Step 2:** Commit `docs(2x): wire protocol v1 and handshake (R-37)`. Mark R-37 resolved in
  `review-register.md`. Push.

### Task 3: Canonical serialisation and hashing

**Model:** opus implements; sonnet reviews. **Tier:** logic. **Implements:** ADR-006, ADR-002 §1, P-4.
**Depends on:** Task 1.

**Files:**
- Create: `packages/senselab-worker/src/senselab_worker/{canonical,hashing}.py`.
- Test: `packages/senselab-worker/tests/test_canonical.py`, `test_hashing.py`.

**Interfaces:**
- Produces:
  - `jcs_dumps(obj: object) -> bytes` (raises `ValueError` on NaN or inf);
  - `canonical_npy_bytes(a: numpy.ndarray) -> bytes` (raises `ValueError` on object or structured
    dtype);
  - `sha256_bytes(b: bytes) -> str`;
  - `ContentId = dict` with keys `algo`, `chunk_size`, `chunks`, `root`;
  - `content_identity(path: str | os.PathLike[str], *, chunk_size: int = 64 * 2**20) -> ContentId`;
  - `weights_identity(paths: Sequence[str | os.PathLike[str]], *, root: str | os.PathLike[str]) -> str`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_jcs_rfc8785_vectors():
      assert jcs_dumps({"b": 2, "a": [1, "x"]}) == b'{"a":[1,"x"],"b":2}'
      assert jcs_dumps(1e21) == b"1e+21" and jcs_dumps(0.0) == b"0" and jcs_dumps(-0.0) == b"0"

  @pytest.mark.logic
  @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
  def test_jcs_edge_values(bad):
      with pytest.raises(ValueError):
          jcs_dumps({"x": bad})

  @pytest.mark.logic
  def test_jcs_normalises_strings_to_nfc():
      assert jcs_dumps("é") == jcs_dumps("é")

  @pytest.mark.logic
  def test_canonical_npy_is_little_endian_c_order():
      a = numpy.arange(6, dtype=">f4").reshape(2, 3).T
      b = canonical_npy_bytes(a)
      assert numpy.load(io.BytesIO(b)).dtype.str == "<f4" and b[6:8] == b"\x01\x00"

  @pytest.mark.logic
  def test_content_identity_chunks(tmp_path):
      p = tmp_path / "x.bin"; p.write_bytes(b"a" * 10)
      cid = content_identity(p, chunk_size=4)
      assert len(cid["chunks"]) == 3 and cid["root"] == sha256_bytes("".join(cid["chunks"]).encode())

  @pytest.mark.logic
  def test_weights_identity_ignores_listing_order(tmp_path): ...
  ```
  Also add a Hypothesis property: `jcs_dumps(json.loads(jcs_dumps(x))) == jcs_dumps(x)` for JSON
  values built from ints, finite floats and NFC strings.
- [ ] **Step 2:** `uv run pytest packages/senselab-worker/tests -v` fails.
- [ ] **Step 3: Implement.**
  - `jcs_dumps` follows RFC 8785 §3.2: UTF-16 key sort; ECMAScript number serialisation via
    `repr` normalisation; NFC applied before encoding. Numpy scalars convert with `.item()`.
    Integers above 2⁵³ raise `ValueError`, as RFC 8785 requires I-JSON.
  - `content_identity` streams the file and never reads it whole.
  - `weights_identity` hashes `jcs_dumps(sorted([relpath, sha256(file)]))`.
- [ ] **Step 4:** `uv run pytest packages/senselab-worker/tests -v` passes.
- [ ] **Step 5:** Commit `feat(worker): canonical JSON, arrays and content identity`.

### Task 4: Worker runtime (codec, faults, serve loop, `fake` worker, floor job)

**Model:** opus implements; sonnet reviews. **Tier:** logic + env (floor).
**Implements:** wire-protocol.md, I-2, ADR-009 faults, testing.md fake fault model, P-22.
**Depends on:** Tasks 1, 2, 3.

**Files:**
- Create: `packages/senselab-worker/src/senselab_worker/{codec,faults,serve,fake}.py`.
- Modify: `packages/senselab-worker/src/senselab_worker/__init__.py` (`WIRE_VERSION = 1`, `runtime_identity()`).
- Create: `packages/senselab-core/src/senselab_core/runtime/worker_client.py`.
- Modify: `.github/workflows/tests.yaml` (add the `worker-floor` job).
- Test: `packages/senselab-worker/tests/test_codec.py`, `test_serve.py`, `test_fake.py`.
- Test: `packages/senselab-core/tests/runtime/test_worker_client.py`.

**Interfaces:**
- Consumes: `jcs_dumps`, `canonical_npy_bytes`, `sha256_bytes` (Task 3).
- Produces:
  - `runtime_identity() -> str`;
  - `encode(msg: dict, *, array_dir: Path) -> bytes`;
  - `decode(line: bytes, *, array_dir: Path) -> dict`;
  - `class FaultClass(str, Enum): CONTENT = "content"; RESOURCE = "resource"; INFRASTRUCTURE = "infrastructure"`;
  - `classify(exc: BaseException) -> FaultClass`;
  - `serve(load: Callable[[dict], object], handle: Callable[[object, list[dict], dict], list[dict]]) -> None`;
  - `@dataclass(frozen=True) class FakeFaults: fail_items: frozenset[str] = frozenset(); oom_items: frozenset[str] = frozenset(); crash_after: int | None = None; batch_coupled: bool = False; nondeterministic: bool = False; latency_s: float = 0.0`;
  - `fake_load(init: dict) -> object` and `fake_handle(state, items, params) -> list[dict]`. Each
    output is `{"value": sha256(jcs(item inputs + params))}`, perturbed per `FakeFaults`;
  - host side: `class WorkerClient` with `__init__(self, python: Path, module: str, *, init: dict, array_dir: Path)`,
    `run(self, items: list[dict], params: dict) -> list[dict]` and `close(self) -> None`;
  - `class ProtocolMismatch(RuntimeError)`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_runtime_identity_is_stable_and_hex():
      assert runtime_identity() == runtime_identity() and len(runtime_identity()) == 64

  @pytest.mark.logic
  def test_array_roundtrip_across_numpy_majors(tmp_path):
      a = numpy.linspace(-1, 1, 16000, dtype="<f4").reshape(1, -1)
      msg = decode(encode({"x": a}, array_dir=tmp_path), array_dir=tmp_path)
      assert msg["x"].tobytes() == a.tobytes()

  @pytest.mark.logic
  def test_one_item_failure_isolated(worker):
      out = worker.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {"faults": {"fail_items": ["b"]}})
      assert "outputs" in out[0] and out[1]["error"]["fault"] == "content"

  @pytest.mark.logic
  def test_oom_is_resource_fault(worker): ...

  @pytest.mark.logic
  def test_handshake_mismatch_refuses(tmp_path, monkeypatch):
      # host built with WIRE_VERSION 2 against a worker on 1
      with pytest.raises(ProtocolMismatch):
          WorkerClient(sys.executable, "senselab_worker.fake", init={}, array_dir=tmp_path)
  ```
  Here `worker` is a fixture that starts `WorkerClient(sys.executable, "senselab_worker.fake", ...)`.
  Write `test_mismatch` so that it builds the client with an overridden expected identity through
  a constructor argument `expect_runtime: str | None = None`, not with `monkeypatch`.
- [ ] **Step 2:** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/test_worker_client.py -v` fails.
- [ ] **Step 3: Implement.**
  - The serve loop follows `wire-protocol.md`.
  - `__main__` of `senselab_worker.fake` calls `serve(fake_load, fake_handle)`.
  - `WorkerClient` starts `python -m <module>`, checks the hello, and maps errors to `FaultClass`.
  - The `worker-floor` CI job:
    1. `uv build packages/senselab-worker`;
    2. `uv venv -p 3.10 .floor`;
    3. `uv pip install --python .floor dist/senselab_worker-*.whl numpy==1.23.5 pytest`;
    4. `.floor/bin/pytest packages/senselab-worker/tests -m logic`.
- [ ] **Step 4:** Scoped pytest passes locally. Run the floor sequence locally with Python 3.10,
  and it passes.
- [ ] **Step 5:** Commit `feat(worker): wire protocol v1, typed faults, fake worker, floor job`.

### Task 5: Kernel time, signal and result types

**Model:** opus implements; sonnet reviews. **Tier:** logic.
**Implements:** ADR-005, design §4.1, P-24, P-26, roadmap hooks H-1 and H-5, ADR-006 `format`.
**Depends on:** Tasks 1, 3.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/kernel/{formats,time,signals,results}.py`.
- Test: `packages/senselab-core/tests/kernel/test_time.py`, `test_signals.py`, `test_results.py`.

**Interfaces:**
- Consumes: `content_identity`, `canonical_npy_bytes`, `sha256_bytes` (Task 3).
- Produces (all pydantic v2, frozen):
  - `class Formatted(BaseModel)` with `format: str` validated as `^[a-z0-9_.-]+@\d+\.\d+$`;
  - `TimeAnchor(clock_id: str, absolute_start_ns: int | None)`;
  - `TimeMap(source: str, target: str, offset_ns: int, drift_ppm: float = 0.0, uncertainty_ns: int = 0)`
    with `map(t_ns: int) -> int`;
  - `Span(start_ns: int, end_ns: int, clock_id: str)`, with a validator `end_ns >= start_ns`;
  - `ChannelInfo(name: str, type: str, units: str, placement: str | None = None, reference: str | None = None)`;
  - `RegularGroup(channels: tuple[ChannelInfo, ...], sampling_rate: float, data: numpy.ndarray | None = None, source: Path | None = None)`
    with `read(start_ns: int | None = None, end_ns: int | None = None) -> numpy.ndarray`;
  - `IrregularGroup(channels, timestamps_ns: numpy.ndarray, data: numpy.ndarray)`;
  - `Recording(groups: dict[str, RegularGroup | IrregularGroup], anchor: TimeAnchor, identity: str)`;
  - `Audio(Recording)`: one regular group named `"audio"`, every channel `type == "audio"`;
    `Audio.from_path(path) -> Audio` (lazy; `identity = content_identity(path)["root"]`) and
    `Audio.from_array(a, sampling_rate) -> Audio`;
  - `Annotation(span: Span, label: str | None, value: float | None, score: float | None, producer: str, guideline: str | None = None, uncertainty: float | None = None)`;
  - `Embedding(vector: numpy.ndarray, space: str)`;
  - `Intervals(start_ns: numpy.ndarray, end_ns: numpy.ndarray, label: numpy.ndarray, clock_id: str)`;
  - `Events(time_ns: numpy.ndarray, label: numpy.ndarray, value: numpy.ndarray | None, clock_id: str)`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_span_rejects_negative_duration():
      with pytest.raises(ValidationError): Span(start_ns=10, end_ns=5, clock_id="c")

  @pytest.mark.logic
  def test_timemap_offset_and_drift():
      m = TimeMap(source="a", target="b", offset_ns=1_000, drift_ppm=10.0)
      assert m.map(1_000_000_000) == 1_000_000_000 + 1_000 + 10_000

  @pytest.mark.logic
  def test_multirate_recording():
      r = Recording(groups={"ecg": RegularGroup(channels=(ChannelInfo(name="I", type="ecg", units="mV"),), sampling_rate=250.0, data=numpy.zeros((1, 250))),
                            "acc": RegularGroup(channels=tuple(ChannelInfo(name=a, type="acc", units="g") for a in "xyz"), sampling_rate=50.0, data=numpy.zeros((3, 50)))},
                    anchor=TimeAnchor(clock_id="dev", absolute_start_ns=None), identity="x")
      assert r.groups["ecg"].sampling_rate == 250.0 and r.groups["acc"].data.shape == (3, 50)

  @pytest.mark.logic
  def test_audio_identity_is_raw_bytes(tmp_path):
      p = write_wav(tmp_path)  # helper from senselab_core.testing.synth once Task 14 lands; inline here
      assert Audio.from_path(p).identity == content_identity(p)["root"]

  @pytest.mark.logic
  def test_audio_time_range_read_is_lazy(tmp_path): ...

  @pytest.mark.logic
  def test_annotation_carries_producer_and_guideline(): ...
  ```
- [ ] **Step 2:** `uv run pytest packages/senselab-core/tests/kernel -v` fails.
- [ ] **Step 3:** Implement. `Audio` reads with `soundfile`, a core dependency added in this task,
  by frame range for time-range reads. `numpy` arrays use pydantic `arbitrary_types_allowed`.
- [ ] **Step 4:** Scoped pytest passes, and `uv run lint-imports` passes.
- [ ] **Step 5:** Commit `feat(core): kernel time, signal and result types`.

### Task 6: Kernel license record and identity types

**Model:** opus implements; sonnet reviews. **Tier:** logic + contract.
**Implements:** ADR-002, design §4.8, §4.13, P-3, P-4, P-23, P-38.
**Depends on:** Tasks 3, 5.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/kernel/{license,identity}.py`.
- Test: `packages/senselab-core/tests/kernel/test_identity.py`, `test_license.py`.

**Interfaces:**
- Produces:
  - `License(expression: str, commercial_use: Literal["yes","no","unknown"], copyleft: Literal["none","weak","strong","network"], attribution: bool, gated: bool, source: str, verified: dict[str, str])`;
  - `ModelRef(source: Literal["hf","url","local"], id: str, ref: str)` with `ModelRef.parse("org/name@rev")`;
  - `ResolvedModel(ref: ModelRef, identity: str, unresolved_reason: str | None = None, license: License | None = None)`;
  - `class ReusePolicy(str, Enum): EXACT = "exact"; EQUIVALENT = "equivalent"; TOLERANT = "tolerant"`;
  - `ExecutionContext(env_lock: str, platform: str, wheel_set: str, numerics: dict[str, str], decoder: str | None, driver: dict[str, str], host: str, protocol: str)`
    with `fingerprint() -> str` (a sha256 over the fields except `host`);
  - `LogicalKey(task: str, task_version: str, backend: str, backend_version: str, model: str | None, params: dict, inputs: dict[str, str], profile_slices: dict[str, str] = {}, batch_formation: str | None = None, chunk_grid: str | None = None, schema: int = 1)`
    with `digest() -> str` (sha256 of `jcs_dumps(self.model_dump())`);
  - `acceptable(stored: ExecutionContext, current: ExecutionContext, policy: ReusePolicy) -> bool`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_key_ignores_param_order_and_unicode_form():
      a = LogicalKey(task="t", task_version="1", backend="b", backend_version="1", model=None, params={"x": 1, "y": "é"}, inputs={})
      b = LogicalKey(task="t", task_version="1", backend="b", backend_version="1", model=None, params={"y": "é", "x": 1}, inputs={})
      assert a.digest() == b.digest()

  @pytest.mark.logic
  @pytest.mark.parametrize("field,value", [("task_version","2"),("backend_version","2"),("model","m"),("params",{"x":2}),("inputs",{"a":"h"}),("profile_slices",{"p":"s"})])
  def test_key_changes_with_each_identity_field(field, value): ...

  @pytest.mark.logic
  def test_context_is_not_in_key():
      assert "host" not in LogicalKey.model_fields and "env_lock" not in LogicalKey.model_fields

  @pytest.mark.logic
  def test_equivalent_without_measurement_is_exact():
      c1, c2 = ctx(platform="linux"), ctx(platform="macos")
      assert acceptable(c1, c1, ReusePolicy.EQUIVALENT) and not acceptable(c1, c2, ReusePolicy.EQUIVALENT)
      assert acceptable(c1, c2, ReusePolicy.TOLERANT)

  @pytest.mark.logic
  def test_modelref_parse():
      assert ModelRef.parse("openai/whisper-large-v3-turbo@main") == ModelRef(source="hf", id="openai/whisper-large-v3-turbo", ref="main")
  ```
- [ ] **Step 2:** Scoped pytest fails.
- [ ] **Step 3:** Implement. `acceptable` consults an equivalence-class table that is empty in
  Phase 1 (`EQUIVALENCE: dict[str, frozenset[str]] = {}`), so `equivalent` behaves as `exact`.
- [ ] **Step 4:** Scoped pytest passes.
- [ ] **Step 5:** Commit `feat(core): identity v2 types and license record`.

### Task 7: Move `file_lock` and `model_revision` into core runtime (rewire)

**Model:** sonnet implements (effort lower); opus reviews (locking code). **Tier:** logic (moved
tests) plus the 1.x suite for the callers.
**Implements:** §6.1 (moves are rewires; no re-exports), ADR-009, P-14, P-37.
**Depends on:** Task 1.

**Callers today:**
- `senselab.utils.file_lock`: 8 source files and 6 test files.
- `senselab.utils.model_revision`: 27 source files and 33 test files.
- `model_revision` imports `file_lock`, so both move together.
- Find them with: `grep -rln "senselab.utils.file_lock\|senselab.utils.model_revision\|utils import .*\(file_lock\|model_revision\)" src scripts`.

**Files:**
- Move: `src/senselab/utils/file_lock.py` → `packages/senselab-core/src/senselab_core/runtime/locks.py`.
- Move: `src/senselab/utils/model_revision.py` → `packages/senselab-core/src/senselab_core/runtime/revision.py`.
- Move their tests to `packages/senselab-core/tests/runtime/`.
- Modify every caller found by the grep above.
- Create: `src/tests/utils/moved_modules_test.py`.

**Interfaces:**
- Produces:
  - `senselab_core.runtime.locks.SharedFileLock` and `lock_holder`;
  - `senselab_core.runtime.revision.resolve_revision`, `RevisionResolutionError`, `run_id`,
    `record_resolution`;
  - all signatures unchanged.

- [ ] **Step 1: Write the failing guard test**
  ```python
  MOVED = {"senselab.utils.file_lock", "senselab.utils.model_revision"}

  @pytest.mark.logic
  def test_no_moved_module_paths_remain():
      hits = [p for p in Path("src").rglob("*.py") for m in MOVED if m in p.read_text()]
      hits += [p for p in Path("scripts").rglob("*.py") for m in MOVED if m in p.read_text()]
      assert hits == []
  ```
  The scan covers worker-script strings too, because it greps text.
- [ ] **Step 2:** `uv run pytest src/tests/utils/moved_modules_test.py -v` fails.
- [ ] **Step 3:** Move with `git mv`, rewrite the imports, and update any path strings in
  `src/tests/utils/revision_pinning_guard_test.py` allowlists. There is no shim at the old path.
- [ ] **Step 4:** Run `uv run pytest src/tests/utils/moved_modules_test.py packages/senselab-core/tests/runtime -v`,
  which passes. Then run the 1.x directories with callers, serially:
  `uv run pytest src/tests/utils src/tests/audio/tasks/speech_to_text src/tests/audio/tasks/speaker_diarization -q`.
  They pass, with the same pass/skip counts as on `alpha`.
- [ ] **Step 5:** Commit `refactor(2x): move file_lock and model_revision into senselab_core.runtime`.

### Task 8: Model resolution and the weights manifest

**Model:** sonnet implements; opus reviews. **Tier:** logic, plus `network` for one live
resolution test.
**Implements:** design §4.3, P-4, P-16, P-37 (one place for weight hashes).
**Depends on:** Tasks 3, 6, 7.

**The four literal sha constants become data:**
- `MEDIAPIPE_MODEL_SHA256` (`video/tasks/facial_analysis/backends.py:34`);
- `YUNET_SHA256` and `SFACE_SHA256` (`video/tasks/facial_analysis/identity.py:16,22`);
- `_S3FD_SHA256` (`utils/clearvoice.py:278`);
- `CPC_SHA256` (`audio/tasks/turn_taking_projection/maai.py:104`).

**Files:**
- Create: `packages/senselab-core/src/senselab_core/runtime/models.py`, `runtime/data/weights.yaml`.
- Modify: the four modules above, which call `verify_weights(name, path)` instead of comparing a
  literal.
- Test: `packages/senselab-core/tests/runtime/test_models.py`.

**Interfaces:**
- Consumes: `ModelRef`, `ResolvedModel` (Task 6); `resolve_revision` (Task 7); `weights_identity`,
  `sha256_bytes` (Task 3).
- Produces:
  - `resolve(ref: ModelRef, *, token: str | None = None) -> ResolvedModel` (hf → 40-hex commit;
    url/local → `weights_identity`; otherwise `unresolved_reason`);
  - `verify_weights(name: str, path: Path) -> None`, which raises `WeightsMismatch` on mismatch;
  - `WEIGHTS: dict[str, str]`, loaded from `weights.yaml`, with the format `weights@1.0`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_weights_manifest_holds_the_former_constants():
      assert WEIGHTS["yunet"] == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
      assert WEIGHTS["maai-cpc"] == "d0f474de968c7243c20f43dee1626681618f2f150d3e638e82b79e22563b17f1"

  @pytest.mark.logic
  def test_verify_weights_rejects_other_bytes(tmp_path): ...

  @pytest.mark.logic
  def test_resolve_local_uses_weights_identity(tmp_path): ...

  @pytest.mark.network
  def test_resolve_hf_returns_commit(): ...
  ```
  Extend `MOVED` in `moved_modules_test.py` with a check that no `*_SHA256 = "<64 hex>"` literal
  remains in `src/senselab` outside `reproduce/`.
- [ ] **Step 2:** Scoped pytest fails.
- [ ] **Step 3:** Implement and rewire the four modules.
- [ ] **Step 4:** `uv run pytest packages/senselab-core/tests/runtime/test_models.py src/tests/utils/moved_modules_test.py src/tests/video src/tests/audio/tasks/turn_taking_projection -q`
  passes.
- [ ] **Step 5:** Commit `feat(core): model resolution and weights manifest`.

### Task 9: Move `prov_store` and `prov_bep028` into core runtime (rewire)

**Model:** sonnet implements (effort lower); opus reviews. **Tier:** logic (moved tests) plus the
1.x triage suite.
**Implements:** §6.1, ADR-009, P-3, P-14. Exit requirement: triage outputs unchanged.
**Depends on:** Task 1.

**Callers today:**
- `senselab.utils.prov_store`: 50 source files and 68 test files, mostly
  `audio/workflows/triage` and `utils/reproduce`.
- `senselab.utils.prov_bep028`: 2 source files and 2 test files.

**Files:**
- Move to `packages/senselab-core/src/senselab_core/runtime/provenance.py` and `runtime/bep028.py`.
- Modify all callers.
- Extend `MOVED` in `src/tests/utils/moved_modules_test.py`.

- [ ] **Step 1:** Add both paths to `MOVED`. The guard test fails.
- [ ] **Step 2:** `git mv`, then rewire the imports. The store format and ids are unchanged.
- [ ] **Step 3:** Run `uv run pytest src/tests/utils/moved_modules_test.py packages/senselab-core/tests/runtime -v`,
  which passes. Then run `uv run pytest src/tests/audio/workflows/triage src/tests/utils/reproduce -q`
  serially, which passes with the same counts as on `alpha`.
  - It is long. Run it on ORCD if the laptop is loaded (P-29).
- [ ] **Step 4:** Commit `refactor(2x): move prov_store and prov_bep028 into senselab_core.runtime`.

### Task 10: Result store with reuse policy and provenance

**Model:** opus implements; sonnet reviews. **Tier:** logic.
**Implements:** ADR-002, ADR-003 (atomic rename, first-writer-wins), ADR-006, P-3, P-19, P-38.
**Depends on:** Tasks 6, 9.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/runtime/store.py`.
- Test: `packages/senselab-core/tests/runtime/test_store.py`.

**Interfaces:**
- Consumes: `LogicalKey`, `ExecutionContext`, `ReusePolicy`, `acceptable` (Task 6); `ProvStore`
  (Task 9); `jcs_dumps`, `canonical_npy_bytes` (Task 3).
- Produces:
  - `class ResultStore` with `__init__(self, root: Path)`;
  - `put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> str`,
    which returns the output identity;
  - `get(self, key: LogicalKey, context: ExecutionContext, policy: ReusePolicy) -> StoredResult | None`;
  - `StoredResult(outputs: dict[str, object], output_identity: str, context: ExecutionContext, policy_used: ReusePolicy)`.
- **Layout:**
  - `root/store@1/<d[:2]>/<d>/<ctx-fingerprint>/{result.json, *.npy, context.json}`, where `d`
    is `key.digest()`;
  - written into `root/tmp/<uuid>/` and committed by `os.rename`;
  - an existing target means another writer won; discard and return the existing identity;
  - `result.json` carries `format: "result@1.0"`;
  - the root differs from `cached_inference`'s, so the 1.x cache is untouched.
- **Output identity:** the sha256 over `jcs_dumps` of the outputs, with arrays replaced by their
  canonical `.npy` sha256.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_put_get_exact(tmp_path): ...

  @pytest.mark.logic
  def test_cross_context_follows_policy(tmp_path):
      s.put(k, {"y": 1}, ctx(platform="linux"))
      assert s.get(k, ctx(platform="macos"), ReusePolicy.EQUIVALENT) is None
      assert s.get(k, ctx(platform="macos"), ReusePolicy.TOLERANT).policy_used is ReusePolicy.TOLERANT

  @pytest.mark.logic
  def test_two_writers_one_entry(tmp_path):
      # two processes via multiprocessing (spawn) put the same key concurrently
      assert len(list((tmp_path / "store@1").rglob("result.json"))) == 1

  @pytest.mark.logic
  def test_crash_leaves_no_entry(tmp_path):
      # a staging dir without rename is invisible to get() and removed by the next put()
      ...

  @pytest.mark.logic
  def test_output_identity_enables_early_cutoff(tmp_path):
      assert s.put(k1, {"y": numpy.ones(3, "<f4")}, c) == s.put(k2, {"y": numpy.ones(3, "<f4")}, c)
  ```
- [ ] **Step 2:** Scoped pytest fails.
- [ ] **Step 3:** Implement. `put` also appends one provenance activity (used: the key inputs;
  generated: the output identity; environment: the context) through a `ProvStore` the caller
  passes or the store owns.
- [ ] **Step 4:** Scoped pytest passes.
- [ ] **Step 5:** Commit `feat(core): result store with reuse policy and provenance`.

### Task 11: Torch-free hardware probe

**Model:** sonnet implements; opus reviews. **Tier:** logic.
**Implements:** design §4.4, §4.9, P-7, P-13.
**Depends on:** Task 1.

**Callers:** `senselab.utils.cuda_probe` has 7 callers, `subprocess_venv.py` among them.

**Files:**
- Move: `src/senselab/utils/cuda_probe.py` → `packages/senselab-core/src/senselab_core/runtime/hardware.py`,
  and rewire the 7 callers.
- Extend `MOVED`.
- Test: `packages/senselab-core/tests/runtime/test_hardware.py`.

**Interfaces:**
- Produces, beside the moved `HostCuda`, `detect_host_cuda` and `pick_torch_index`:
  - `HardwareInfo(cpu_count: int, memory_bytes: int, gpus: tuple[GpuInfo, ...], apple_silicon: bool, platform: str)`;
  - `GpuInfo(name: str, memory_bytes: int, driver: str | None)`;
  - `probe_hardware(*, run: Callable[[list[str]], str] = _run) -> HardwareInfo`;
  - `numerics_class(info: HardwareInfo, device: str, precision: str) -> dict[str, str]`.

- [ ] **Step 1:** Write failing tests that inject `run`, never a real `nvidia-smi`:
  - `test_parses_two_gpus_from_nvidia_smi_csv`;
  - `test_no_nvidia_smi_means_no_gpus`;
  - `test_apple_silicon_detected_from_platform`;
  - `test_probe_imports_no_torch`.
- [ ] **Step 2:** The tests fail.
- [ ] **Step 3:** Implement with
  `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`,
  `os.cpu_count()`, `os.sysconf` or `sysctl hw.memsize`, and `platform.machine()`.
- [ ] **Step 4:** Scoped pytest passes. `uv run pytest src/tests/utils/subprocess_venv_test.py -q`
  also passes.
- [ ] **Step 5:** Commit `feat(core): torch-free hardware probe; move cuda_probe`.

### Task 12: Declaration registry, the `fake` stack and generated contract tests

**Model:** opus implements; sonnet reviews. **Tier:** contract.
**Implements:** testing.md §3 (generated contracts, the fake stack with its fault model), ADR-001
(declarations only; no graph), P-1, P-20, P-30.
**Depends on:** Tasks 4, 6, 10.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/kernel/declare.py`.
- Create: `packages/senselab-core/src/senselab_core/testing/contracts.py`.
- Create: `packages/senselab-core/src/senselab_core/runtime/fake_backend.py`.
- Test: `packages/senselab-core/tests/test_contracts.py`.

**Interfaces:**
- Produces:
  - `TaskSpec(name: str, version: str, kind: Literal["measure","transform","decide","fit"], inputs: dict[str, str], outputs: dict[str, str], params: type[BaseModel])`;
  - `BackendSpec(task: str, name: str, version: str, stack: str, env: str, devices: tuple[str, ...], license: License, output_affecting: frozenset[str] = frozenset())`;
  - `class Registry` with `add_task(spec)` and `add_backend(spec)`. Both raise on a duplicate
    name, or on a second task with the same `(kind, inputs, outputs)` signature (P-37).
    Also `tasks() -> list[TaskSpec]` and `backends(task: str) -> list[BackendSpec]`;
  - `REGISTRY: Registry`, holding `fake.echo` and its backend `fake` (stack `fake`, env `host`,
    license Apache-2.0) after `import senselab_core.runtime.fake_backend`;
  - `run_fake(items: list[dict], params: dict, faults: FakeFaults) -> list[dict]`, which drives
    `WorkerClient` with `senselab_worker.fake`;
  - `contract_cases(registry: Registry) -> list[tuple[str, Callable[[], None]]]`, which
    `test_contracts.py` parametrises over, with ids equal to the backend names.
- **Phase 1 contract cases:**
  - params schema round-trip and unknown-key rejection;
  - a license record is present;
  - the key is deterministic, and changes per ADR-002 field;
  - the key is unchanged by context fields;
  - `measure` outputs carry no verdict fields (P-20);
  - a content fault is isolated per item.

- [ ] **Step 1:** Write `test_contracts.py`:
  `@pytest.mark.contract @pytest.mark.parametrize("name,case", contract_cases(REGISTRY), ids=lambda c: c if isinstance(c, str) else "")`.
  Add a registry logic test, `test_duplicate_signature_refused`.
- [ ] **Step 2:** The tests fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `uv run pytest packages/senselab-core/tests/test_contracts.py -v` passes. The
  parametrised `fake` cases are listed.
- [ ] **Step 5:** Commit `feat(core): declaration registry, fake stack, generated contract tests`.

### Task 13: Data manifests and `senselab data fetch`

**Model:** sonnet implements; opus reviews (cache concurrency). **Tier:** logic, plus `network`
for one live fetch.
**Implements:** docs.md data policy, P-41, P-11, P-37 (one fetch for docs, benchmarks and tests).
**Depends on:** Tasks 3, 7.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/runtime/fetch.py`, `src/senselab_core/cli.py`.
- Modify: root `pyproject.toml`, adding `[project.scripts] senselab = "senselab_core.cli:main"`.
- Test: `packages/senselab-core/tests/runtime/test_fetch.py`.

**Interfaces:**
- Produces:
  - `DataFile(path: str, url: str, sha256: str, mirrors: tuple[str, ...] = ())`;
  - `DataManifest(format: Literal["data-manifest@1.0"], name: str, version: str, license: License, terms: str, files: tuple[DataFile, ...])`;
  - `DataManifest.load(path: Path) -> DataManifest`;
  - `fetch(manifest: DataManifest, *, cache_root: Path, opener: Callable[[str], BinaryIO] = _urlopen) -> dict[str, Path]`;
  - CLI: `senselab data fetch <manifest.yaml> [--cache DIR]`.
- **Cache layout:** `cache_root/blobs/<sha256>`, written to `tmp/<uuid>` and renamed. The hash is
  verified before the rename. On mismatch the next mirror is tried. If none matches, it raises
  `FetchError` naming the file and every URL tried. Per-name symlinks live under
  `cache_root/datasets/<name>@<version>/`.

- [ ] **Step 1:** Write failing tests with an injected `opener` serving bytes from memory:
  - `test_fetch_verifies_and_caches`;
  - `test_partial_download_not_cached` (the opener raises midway);
  - `test_mirror_hash_mismatch_refused`;
  - `test_mirror_with_matching_hash_accepted`;
  - `test_second_fetch_is_offline`, where the opener asserts it is never called.
- [ ] **Step 2:** The tests fail.
- [ ] **Step 3:** Implement. The default `cache_root` is `$SENSELAB_DATA_CACHE`, or
  `~/.cache/senselab/data`.
- [ ] **Step 4:** Scoped pytest passes. `uv run senselab data fetch --help` prints usage.
- [ ] **Step 5:** Commit `feat(core): data manifests and senselab data fetch`.

### Task 14: Synthetic generators and the fixture budget

**Model:** sonnet implements; a fresh sonnet reviews. **Tier:** logic.
**Implements:** docs.md data policy, P-41, P-30.
**Depends on:** Task 5.

**Files:**
- Create: `packages/senselab-core/src/senselab_core/testing/synth.py`.
- Create: `scripts/check_fixture_budget.py`.
- Create: `scripts/fixture_budget_allowlist.txt`, listing today's six files in
  `src/tests/data_for_testing/`, which are grandfathered until their tests are ported.
- Modify: `.pre-commit-config.yaml` (a local hook running the budget check).
- Test: `packages/senselab-core/tests/testing/test_synth.py`, `src/tests/scripts/check_fixture_budget_test.py`.

**Interfaces:**
- Produces:
  - `tone(freq_hz: float, *, duration_s: float, sampling_rate: int = 16_000, channels: int = 1, amplitude: float = 0.5) -> Audio`;
  - `noise(*, duration_s: float, sampling_rate: int = 16_000, seed: int = 0) -> Audio`;
  - `chirp(f0_hz: float, f1_hz: float, *, duration_s: float, sampling_rate: int = 16_000) -> Audio`;
  - `write_wav(audio: Audio, path: Path) -> Path`;
  - `check(paths: Sequence[Path], *, per_file: int = 256 * 1024, total: int = 2 * 1024 * 1024, allow: set[str]) -> list[str]`,
    which returns violation messages.

- [ ] **Step 1:** Write failing tests:
  - `test_tone_frequency_peak` (the FFT peak is within one bin of `freq_hz`);
  - `test_noise_is_seeded`;
  - `test_chirp_sweeps`;
  - `test_budget_flags_large_new_file`;
  - `test_budget_allows_grandfathered`.
- [ ] **Step 2:** The tests fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Scoped pytest passes, and `uv run pre-commit run --all-files` passes.
- [ ] **Step 5:** Commit `feat(core): synthetic signal generators and fixture budget check`.

### Task 15: Phase 1 exit check and adversarial round

**Model:**
- the exit checks: sonnet implements, opus reviews;
- the adversarial round: an opus lead (effort higher) with up to 5 sonnet subagents, verified by
  haiku (P-35, P-40).

**Tier:** all of the above.
**Depends on:** Tasks 0–14.

- [ ] **Step 1:** Confirm on `alpha` CI that `core-only` is green on Python 3.11 and 3.14,
  `worker-floor` is green on 3.10 with numpy 1.23.5, `lint-imports` passes, and pre-commit passes.
- [ ] **Step 2:** Run the full 1.x suite serially on ORCD from a pinned checkout of `alpha`. Use a
  CPU partition named in the job and respect the QOS cap (P-29). It passes with the pass and skip
  counts of `freeze/1x-2026-10-30` plus the moved tests.
- [ ] **Step 3: Adversarial round brief.**
  - Give the opus lead the Phase 1 surfaces: the worker protocol, canonical serialisation, identity
    types, the store, the fetch cache, the import layering and the rewires.
  - Its goal: find inputs, races or misuse that break ADR-002, ADR-003 or ADR-006 or the Phase 1
    exit criteria.
  - Every finding needs an architectural resolution (P-35).
  - Findings become `review-register.md` rows `R-55…`.
- [ ] **Step 4:** Resolve or schedule each finding. Run Task 0's reconciliation again. Update
  `STATE.md` to "Phase 1 done" and add the log line.
- [ ] **Step 5:** Commit `docs(2x): Phase 1 exit` and push.

---

## Self-review notes

- **Spec coverage.** Every §6.2 Phase 1 bullet maps to a task:
  - workspace → 1;
  - worker runtime, handshake, wire spec and floor → 2, 4;
  - Signal and Audio, timeline, annotation, embedding → 5;
  - `Model`/`ResolvedModel` with the rewires → 6, 7, 8;
  - store and ProvStore writer → 9, 10;
  - probe → 11;
  - License type → 6;
  - fake stack and contract module → 12.

  The docs.md Phase 1 items are covered by 13 and 14, and P-39 by tasks 0 and 15.
- **Decisions made here** that the spec left open:
  - **sha256 for every identity.** xxhash is not used for identity.
  - **One weight-hash implementation, in `senselab_worker`.** Core imports it, so there is no
    second implementation (P-37), which removes the cross-implementation golden test that
    design §4.12 anticipated.
  - **Core-package tests** live under `packages/*/tests`, so the core-only job runs without the
    root package.
  - **Existing fixtures are grandfathered** by an allowlist until ported.
  - **Phase 1 declarations are dataclass specs.** Decorators and generated functions come in
    Phase 2.
- **Spec gap.** ADR-009 writes the dependency direction as "kernel ← `senselab_worker`" (the worker
  depends on the kernel). That conflicts with design §4.12 and §5: the worker is stdlib + numpy
  with no core dependency, and core depends on the worker. This plan follows §4.12 and §5, and
  ADR-009 should be corrected.
