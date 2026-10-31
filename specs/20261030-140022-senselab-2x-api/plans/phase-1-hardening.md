# senselab 2.x Phase 1 Hardening Plan: invariants from the adversarial round

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 1 adversarial findings ([`reviews/phase1-adversarial.md`](../reviews/phase1-adversarial.md),
register rows R-59…R-98) by landing the invariants they violate, each with a contract test that covers
every type, format or call site the invariant governs. Phase 1 exits only after H1–H8 merge.

**Architecture:** No new layers. Each task makes one existing surface enforce one stated invariant:
- kernel values own their bytes and compare by canonical digest (H1);
- the canonical encoder and the key accept only exact, typed values (H2);
- a stored artifact is valid only at the address its content derives (H3);
- the worker is a process group with a private channel and owned files (H4);
- every path from a backend object to wire bytes is total (H5);
- repository guards resolve code instead of scanning text (H6);
- a release is installable before it is uploaded (H7).

**Tech Stack:** as `plans/phase-1.md`: Python 3.11–3.14 (core), 3.10+ (worker); numpy, pydantic v2;
pytest + hypothesis; import-linter + grimp; hatchling + hatch-vcs.

**Spec:** `specs/20261030-140022-senselab-2x-api/`, chiefly:
- [`reviews/phase1-adversarial.md`](../reviews/phase1-adversarial.md) (the findings `A-n`, by row);
- [`review-register.md`](../review-register.md) R-59…R-98 (the same findings as register rows);
- ADR-002 (identity), ADR-003 (store), ADR-006 (formats), ADR-009 (faults);
- `wire-protocol.md`, `testing.md`, `engineering.md` (P-34, P-35, P-37, P-40), `principles.md`.

| Finding | Row | Task |
|---|---|---|
| A-1, A-12, A-13, A-14, A-16, A-37 | R-59, R-70, R-71, R-72, R-74, R-95 | H1 |
| A-4, A-5, A-6, A-23, A-24, A-31, A-38 | R-62, R-63, R-64, R-81, R-82, R-89, R-96 | H2 |
| A-2, A-3, A-17, A-18, A-19, A-20, A-35, A-36 | R-60, R-61, R-75, R-76, R-77, R-78, R-93, R-94 | H3 |
| A-7, A-8, A-9, A-21, A-22 | R-65, R-66, R-67, R-79, R-80 | H4 |
| A-10, A-25 (types and gap rule), A-34 | R-68, R-83, R-92 | H5 |
| A-26, A-27, A-28, A-33, A-39 (workflow) | R-84, R-85, R-86, R-91, R-97 | H6 |
| A-11, A-29, A-32 | R-69, R-87, R-90 | H7 |
| every probe expressible in-repo; R-98 investigations | R-98 | H8 |

**Deferred, with owners:**
- **A-15 (R-73) → Phase 3.** Range identity has its first consumer in the declared chunk grid (I-7),
  and "bytes read at open" is a scale-ladder measurement (P-33); both land in Phase 3. No Phase 1 or
  Phase 2 caller forms a range key. Phase 3 keeps the chunk list with the source and adds the
  bounded-read test.
- **A-25 per-backend contract (R-83) → Phase 2 and Phase 3.** H5 lands the typed faults and the
  rule that the fallback table always marks `gap`. The generated contract "zero gaps over each
  backend's fault fixtures" needs a real backend (Phase 2, with R-57). "A gap fault is never cached
  as a verdict" is executor behaviour (Phase 3).
- **A-30 (R-88) → Phase 2 license backfill.** Only the fake stack records a license in Phase 1. The
  SPDX parser and vendored table are validated by the first real records, which arrive with the
  backfill (R-33). Gate: no non-fake `License` is recorded before it lands.
- **A-39 branch protection → owner.** Required checks and protection rules are repository settings
  (H6 Step 6 lists the exact settings).

## Global Constraints

- Everything in `plans/phase-1.md` Global Constraints still holds (P-13, P-22, P-28..P-31, ADR-006
  canonical forms, one tag for three distributions).
- **Every fix is an invariant with a generated or exhaustive test (P-35).** A test that pins only the
  probe's input is a regression test (H8), never the fix. Review rejects an `if` for the found case.
- **One implementation per behaviour (P-37).** The canonical scalar encoder, the storage-fault mapping,
  owned-file deletion and the wire-value validator each exist once and are imported by every caller.
- **Pre-alpha: rename and replace outright.** Changed digests bump the store format
  (`store@1` → `store@2`) and the manifest formats. No migration, no alias.
- **Constants with a threshold character** (retention ages, byte caps) go in a `data/` policy file with
  a `derivation:` block, not as code literals.
- Tests: `uv run pytest <scoped path>`, serial, never `-n auto` (P-29). Each new test carries exactly
  one tier marker (P-30; H6 makes this a collection error). Test files are named `*_test.py`.
- Each task is one PR from `2x/p1h-<slug>` to `alpha`. It records implementer and reviewer models,
  which differ (P-40), and runs the `engineering.md` checklist.
- Rationale stays in `specs/`: docstrings say what and how. Register ids never appear in code (P-31).

## Review Focus

1. **Immutability that survives every copy path.** pickle, `copy`, `deepcopy`, `model_copy`,
   `model_validate(model_dump())`, and the `.base` chain. Pinned in H1 (`value_arrays_immutable`).
2. **Canonical encoder closure.** No `numpy.generic` subtype, Enum mix-in, masked array or subclass
   reaches `.item()`. Pinned in H2 (`test_every_numpy_generic_type_raises_or_encodes_exactly`).
3. **Destructive actions only after a completed, failed verification.** Pinned in H3
   (`test_no_destructive_call_under_oserror_handler`, `test_transient_error_never_quarantines`).
4. **Process-tree shutdown within a bound, on Linux and macOS.** Pinned in H4
   (`test_forking_backend_leaves_no_survivor`).
5. **Totality of the wire path under hostile backends.** Pinned in H5 (`test_serve_is_total_fuzz`).

## File Structure

```
packages/senselab-worker/src/senselab_worker/
  canonical.py          # H2: exact-type whitelist; canonical_scalar(); tagged_digest()
  hashing.py            # H2: DIGEST_TAGS; content root covers chunk_size
  ownership.py          # H4 (new): remove_owned(), the one deletion helper
  launch.py             # H4: _claim_channel() (fd 0 and fd 1 private)
  codec.py              # H4/H5: Deadline-aware array reads; validate_message(); strict parse_line
  serve.py              # H5: one result line per item (wire v2); total _result()
  faults.py             # H5: ContentFault, ResourceFault; total error_record(); gap on fallback
  __init__.py           # H5: WIRE_VERSION = 2
packages/senselab-core/src/senselab_core/
  kernel/_base.py       # H1: frozen_array(), Frozen copy/eq/hash/validation; bounded Nanoseconds; Rate
  kernel/signals.py     # H1: SourceStamp, StaleSource, dtype policy, derived frames
  kernel/time.py        # H1: bounded drift
  kernel/identity.py    # H2: ModelIdentity, ContentIdentity, OutputIdentity; digest at construction; policy coercion
  kernel/declare.py     # H2: check_canonical_params(); key_for(model: ResolvedModel | None)
  runtime/storage.py    # H3 (new): StoreUnavailable, StorageFull, storage_fault(), StorePolicy
  runtime/data/store_policy.yaml   # H3 (new): retention ages, world-readable default, derivations
  runtime/store.py      # H3: address binding, read-only entries, compare-and-rename quarantine
  runtime/fetch.py      # H3: files/ namespace, prefix-free paths, size bound, https allowlist, read-only blobs
  runtime/revision.py   # H2: manifest_key by JCS
  runtime/worker_client.py  # H4/H5: process group, owned array dir, Deadline, non-reentrant run
  testing/values.py     # H1 (new): sample registry and generated value contracts over every Frozen type
  testing/contracts.py  # H2: key_independent_of_hash_seed case
packages/conftest.py    # H6 (new): exactly-one-tier collection hook
scripts/release_gate.py # H7 (new)
.github/workflows/{publish,tests}.yaml   # H6, H7
.importlinter           # H6
src/tests/utils/moved_modules_test.py    # H6: AST resolution and tombstones
scripts/check_fixture_budget.py          # H6
```

Dependency order: H2 and H4 first, in parallel. H1 and H3 use H2's `tagged_digest` and typed
identities, and H3 uses H4's `remove_owned`. Then H1, H3 and H5 (after H4, same files) in parallel.
H6 and H7 can run at any time. H8 is last.

---

### Task H1: Kernel values own their bytes, compare by digest, and revalidate

**Model:** opus implements; sonnet reviews (fresh context). **Tier:** logic + contract.
**Implements:** A-1, A-12, A-13, A-14, A-16, A-37 (R-59, R-70, R-71, R-72, R-74, R-95); P-34, P-35, ADR-005.
**Depends on:** H2 (`tagged_digest`, `ContentIdentity`).

**Files:**
- Modify: `packages/senselab-core/src/senselab_core/kernel/{_base,signals,time,results}.py`.
- Create: `packages/senselab-core/src/senselab_core/testing/values.py`.
- Test: `packages/senselab-core/tests/kernel/value_contracts_test.py` (generated),
  `packages/senselab-core/tests/kernel/signals_test.py`, `time_test.py` (additions).

**Interfaces:**
- `_base.py`:
  - `frozen_array(a: numpy.ndarray) -> numpy.ndarray`: a C-order little-endian copy of `a` whose
    buffer is an immutable `bytes` object, so neither it nor any array on its `.base` chain can be
    made writeable. Replaces `readonly()`; every array validator (`as_ns_array`, `as_numeric_array`,
    `as_label_array`, `SignalArray`) calls it.
  - `class Frozen(BaseModel)` gains:
    - `canonical_digest(self) -> str`: `tagged_digest("value@1", {"type": <qualified name>, "fields": <dump with each array as its canonical-npy sha256>})`;
    - `__eq__(self, other: object) -> bool`: same type and same `canonical_digest`;
    - `__hash__(self) -> int`: `hash(self.canonical_digest())`;
    - `__reduce__`, `__copy__`, `__deepcopy__(memo)`: rebuild through `type(self).model_validate(...)`;
    - `model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self`:
      `model_validate` of the dump merged with `update`, so every validator runs;
    - `model_construct(cls, ...)`: raises `TypeError("Frozen values are built through validation")`.
  - `Nanoseconds`: adds `Field(ge=-(2**63), le=2**63 - 1)`.
  - `Rate = Annotated[float, Strict(), Field(gt=0, allow_inf_nan=False)]`: the one sampling-rate type;
    `bool`, `str` and non-finite values are refused.
  - `FrozenMapping` stores a `types.MappingProxyType` over a private copy and defines `__setattr__`
    raising `AttributeError` after `__init__`.
- `signals.py`:
  - `class SourceStamp(Frozen)`: `size: int`, `mtime_ns: int`, `inode: int`, `device: int`,
    `header: str` (sha256 of the JCS of `soundfile.info` fields: frames, channels, samplerate,
    format, subtype). `RegularGroup.source_stamp: SourceStamp | None`, set with `source`.
  - `class StaleSource(RuntimeError)`: raised by `RegularGroup.read` when the current stamp differs
    from `source_stamp`.
  - `RegularGroup.frames` is a property computed from `data.shape` or `source_stamp`; the
    `_frames` private attribute is removed.
  - **Dtype policy** (`SIGNAL_DTYPES`): audio samples are held and read as float32 full scale. `from_array`
    converts integer PCM by its dtype full scale, the same mapping `soundfile` applies, and refuses
    complex, boolean, object and empty arrays. `sampling_rate` is `Rate`.
  - `Audio.identity` is derived: the validator recomputes it (from `data`, or from `source` content
    identity) and refuses a supplied value that differs.
- `time.py`: `TimeMap.drift_ppm` is bounded to `(-1e6, 1e6)`, so a map is strictly increasing.
- `testing/values.py`:
  - `register_samples(cls: type[Frozen], factory: Callable[[], Sequence[Frozen]], *, invalid_updates: Sequence[Mapping[str, object]] = ()) -> None`;
  - `frozen_types() -> list[type[Frozen]]`: every concrete `Frozen` subclass defined under
    `senselab_core.kernel` and `senselab_core.runtime`, found by walking `__subclasses__` after
    importing both packages;
  - `value_cases() -> list[tuple[str, Callable[[], None]]]`: ids `"<Type>::<contract>"`.
  Samples must include at least one multi-element array per array field.

- [ ] **Step 1: Write the generated contracts (failing)**
  ```python
  @pytest.mark.contract
  def test_every_frozen_type_has_samples():
      assert {t for t in frozen_types()} <= registered_types(), "register samples for every Frozen type"

  @pytest.mark.contract
  @pytest.mark.parametrize(("case_id", "case"), value_cases(), ids=lambda c: c if isinstance(c, str) else "")
  def test_value_contract(case_id, case):
      case()
  ```
  The contracts `value_cases()` generates per type:
  - `value_arrays_immutable`: for each sample and each of `pickle.loads(pickle.dumps(v))`,
    `copy.copy(v)`, `copy.deepcopy(v)`, `v.model_copy()`, `type(v).model_validate(v.model_dump())`,
    every ndarray reachable from the fields, and every array on each one's `.base` chain, raises
    `ValueError` on `flags.writeable = True`.
  - `value_owns_bytes`: mutating the array passed to the constructor after construction leaves
    `canonical_digest()` and every `read()` result unchanged.
  - `eq_hash_consistent`: `v == copy.deepcopy(v)` and their hashes are equal; for two samples with
    different digests, `!=` holds; neither raises.
  - `copy_revalidates`: each registered `invalid_updates` entry raises `ValidationError` through
    `model_copy(update=...)`.
  - `construct_refused`: `type(v).model_construct()` raises `TypeError`.
  - `identity_derived` (types with an `identity` field): `model_validate({**dump, "identity": "f" * 64})`
    raises `ValidationError`.
- [ ] **Step 2: Unit tests (failing)**
  - `test_overwritten_source_raises_stale_source(tmp_path)`: write a WAV, `Audio.from_path`, rewrite
    it with other samples (same and different length), assert `read()` raises `StaleSource`.
  - `test_from_array_matches_from_path(dtype)` parametrized over `int16`, `int32`, `float32`,
    `float64`: `numpy.testing.assert_allclose(Audio.from_array(x, sr).read(), Audio.from_path(write(x)).read(), atol=<one LSB of dtype>)`.
  - `test_rate_refuses_bool_str_nonfinite`: `True`, `"16000"`, `inf`, `0`, `-5` each raise.
  - `test_from_array_refuses_complex_bool_empty`.
  - `test_nanoseconds_bounded_to_int64`: `Span(start_ns=0, end_ns=2**200)` raises.
  - `test_timemap_is_strictly_increasing` (hypothesis over offset, drift in bounds, t).
  - `test_frozenmapping_has_no_mutable_reach`: `m._items["k"] = 1` raises `TypeError`.
- [ ] **Step 3:** `uv run pytest packages/senselab-core/tests/kernel -v` fails as expected.
- [ ] **Step 4: Implement** the interfaces above. Register samples for every kernel and runtime
  `Frozen` type in the same PR (a type without samples fails `test_every_frozen_type_has_samples`).
  Add an AST test `test_no_model_construct_in_package_sources` under
  `packages/senselab-core/tests/lint/` that refuses any `model_construct` call in `packages/*/src`.
- [ ] **Step 5:** `uv run pytest packages/senselab-core/tests/kernel packages/senselab-core/tests/lint -v`
  passes; then `uv run pytest packages -m "logic or contract" -q` (serial).
- [ ] **Step 6:** Commit `fix(kernel): values own their bytes, compare by digest and revalidate on copy`.
  Mark R-59, R-70, R-71, R-72, R-74, R-95 "resolved (H1)".

### Task H2: Canonical encoder whitelist, typed identities, hash-seed contract

**Model:** opus implements; sonnet reviews (fresh context). **Tier:** logic + contract.
**Implements:** A-4, A-5, A-6, A-23, A-24, A-31, A-38 (R-62, R-63, R-64, R-81, R-82, R-89, R-96); ADR-002, ADR-006, P-37.
**Depends on:** none.

**Files:**
- Modify: `packages/senselab-worker/src/senselab_worker/{canonical,hashing}.py`.
- Modify: `packages/senselab-core/src/senselab_core/kernel/{identity,declare}.py`,
  `runtime/{store,revision,models}.py` (callers), `testing/contracts.py`.
- Modify: `specs/20261030-140022-senselab-2x-api/adr/ADR-006-formats.md` (the digest-tag rule) and
  `ADR-002-identity-v2.md` (typed identities).
- Test: `packages/senselab-worker/tests/{canonical,hashing}_test.py`;
  `packages/senselab-core/tests/kernel/{identity,declare}_test.py`; `tests/runtime/revision_test.py`;
  `tests/testing/contracts_test.py`.

**Interfaces:**
- `canonical.py`:
  - `canonical_scalar(v: object) -> None | bool | int | float | str`: dispatch on `type(v)` exactly.
    Python `None`, `bool`, `int`, `float`, `str` pass. A `numpy.generic` passes only if
    `type(v).__module__ == "numpy"` and `v.dtype.kind in "biufU"`. Everything else raises `TypeError`:
    `datetime64`, `timedelta64`, `void`, `complex`, `object_`, any subclass, and Enum members
    (including `(int, Enum)` and `(str, Enum)`). The one scalar encoder: `jcs_dumps` and
    `store._encode` both call it.
  - `jcs_dumps` containers: `type(obj) in (list, tuple)` and `type(obj) is dict` exactly, with `str`
    keys; a cycle raises `ValueError` (tracked by `id`).
  - `canonical_npy_bytes`: requires `type(a) is numpy.ndarray` (masked arrays and subclasses
    refused) and `a.dtype.kind in "biufU"`.
  - `tagged_digest(tag: str, value: object) -> str`: `sha256(tag.encode("ascii") + b"\x00" + jcs_dumps(value))`;
    raises `ValueError` if `tag not in DIGEST_TAGS`.
- `hashing.py`:
  - `DIGEST_TAGS: frozenset[str] = frozenset({"logical-key@2", "output@2", "context@2", "content-root@2", "audio@2", "weights@2", "value@1", "data-manifest@2"})`;
  - `content_identity` root is `tagged_digest("content-root@2", {"chunk_size": n, "chunks": [...]})`;
  - `weights_identity` uses `tagged_digest("weights@2", ...)`.
- `identity.py`:
  - `ContentIdentity = Annotated[str, StringConstraints(pattern=r"^content:[0-9a-f]{64}$")]`;
  - `OutputIdentity = Annotated[str, StringConstraints(pattern=r"^output:[0-9a-f]{64}$")]`;
  - `InputIdentity = ContentIdentity | OutputIdentity`;
  - `class ModelIdentity(Frozen)`: `source: Literal["hf", "url", "local"]`, `id: NonEmptyStr`,
    `identity: NonEmptyStr`. Built only by `ResolvedModel.key_identity(self) -> ModelIdentity`, which
    raises `UnpinnedModel(ValueError)` when `not self.pinned`;
  - `LogicalKey.model: ModelIdentity | None`; `inputs: Mapping[str, InputIdentity]`;
    `profile_slices: Mapping[str, Annotated[str, pattern r"^profile:[0-9a-f]{64}$"]]`;
  - `LogicalKey` computes its digest in an `after` validator (stored in a private attribute), so a
    key that cannot be digested is a `ValidationError`; `digest()` returns it. The digest is
    `tagged_digest("logical-key@2", ...)`, and `ExecutionContext.fingerprint()` uses `"context@2"`;
  - `acceptable(stored, current, policy: ReusePolicy | str, *, backend: str) -> bool` coerces with
    `ReusePolicy(policy)` (an unknown value raises `ValueError`) and dispatches with `match` over the
    three members and `typing.assert_never`.
- `declare.py`:
  - `class NonCanonicalParams(TypeError)`;
  - `check_canonical_params(model: type[BaseModel]) -> None`: walks the field annotations
    recursively. Allowed: `None`, `bool`, `int`, `float`, `str`, `Literal[...]` of those, a
    `str`-valued `Enum`, `tuple[...]`, `list[...]`, `Mapping[str, ...]`/`dict[str, ...]`, `Optional`
    and unions of allowed types, nested `TaskParams`, and `Annotated` constraints over these. Anything
    else raises `NonCanonicalParams`, naming the field and type: `set`, `frozenset`, `Decimal`,
    `datetime`/`date`/`time`, `Path`, `bytes`, `Any` and `object` among them;
  - `Registry.add_task` calls it;
  - `key_for(..., inputs: Mapping[str, InputIdentity], model: ResolvedModel | None = None, ...)` puts
    `model.key_identity()` into the key.
- `revision.py`: `manifest_key(repo_id: str, ref: str) -> str` returns `jcs_dumps([repo_id, ref]).decode()`;
  the manifest format tag is bumped.
- `store.py`: `output_identity` returns `OutputIdentity` (`"output:" + tagged_digest("output@2", tree)`).
  `STORE_DIR = "store@2"`.
- `testing/contracts.py`: new backend case `key_independent_of_hash_seed`. It builds the base call's
  key in two subprocesses with `PYTHONHASHSEED=1` and `PYTHONHASHSEED=2`, importing the declaring
  module named by `TaskSpec.params.__module__`, and asserts equal digests.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_every_numpy_generic_type_raises_or_encodes_exactly():
      for t in _all_numpy_scalar_types():   # walk numpy.generic.__subclasses__ recursively
          for v in _samples_of(t):           # includes NaT, timedelta64 in ns and ps, void, clongdouble
              try:
                  b = jcs_dumps(v)
              except TypeError:
                  continue
              assert t.__module__ == "numpy" and numpy.dtype(t).kind in "biufU"
              assert json.loads(b) == v.item() and type(json.loads(b)) is type(v.item())

  @pytest.mark.logic
  @given(json_values())
  def test_jcs_round_trips_every_accepted_value(v):
      assert jcs_dumps(json.loads(jcs_dumps(v))) == jcs_dumps(v)

  @pytest.mark.logic
  @pytest.mark.parametrize("v", [numpy.datetime64("NaT"), numpy.timedelta64(1, "ns"), numpy.ma.array([1, 2], mask=[0, 1]), IntEnumMember, numpy.zeros(1, dtype=[("a", "<i4")])[0]])
  def test_lossy_values_refused(v):
      with pytest.raises(TypeError):
          jcs_dumps({"a": v})

  @pytest.mark.logic
  def test_digest_tags_distinct_and_prefix_free():
      tags = sorted(DIGEST_TAGS)
      assert all(not b.startswith(a) for a in tags for b in tags if a != b)

  @pytest.mark.logic
  def test_output_identity_differs_from_key_digest_of_same_tree(): ...

  @pytest.mark.logic
  def test_content_root_covers_chunk_size(tmp_path): ...  # same bytes, two chunk sizes, two roots

  @pytest.mark.logic
  def test_key_refuses_untyped_identities():
      for bad in [{"model": "main"}, {"inputs": {"a": "/home/x/a.wav"}}, {"inputs": {"a": "f" * 64}}]:
          with pytest.raises(ValidationError):
              LogicalKey(**{**BASE, **bad})

  @pytest.mark.logic
  def test_key_identity_requires_pinned_model(): ...  # UnpinnedModel

  @pytest.mark.logic
  @given(params=json_objects_including_large_floats())
  def test_constructible_key_is_digestible(params):
      try:
          key = LogicalKey(**{**BASE, "params": params})
      except ValidationError:
          return
      assert len(key.digest()) == 64

  @pytest.mark.logic
  @pytest.mark.parametrize("policy", ["exact", "bogus", 1, None])
  def test_acceptable_coerces_policy(policy): ...  # "exact" behaves as EXACT; others raise ValueError

  @pytest.mark.logic
  @pytest.mark.parametrize("annotation", [set[str], frozenset[int], Decimal, datetime, Path, bytes, Any])
  def test_registry_refuses_non_canonical_param_types(annotation): ...

  @pytest.mark.logic
  def test_manifest_key_is_injective():
      assert manifest_key("a@b", "c") != manifest_key("a", "b@c")
  ```
  The generated `key_independent_of_hash_seed` case runs over the fake registry from
  `contracts_test.py`.
- [ ] **Step 2:** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/kernel packages/senselab-core/tests/testing -v` fails.
- [ ] **Step 3: Implement** the interfaces. Delete the `.item()` branch in `store._encode` in favour
  of `canonical_scalar` (P-37). Update ADR-006 with "every digest is `tagged_digest(tag, value)` with
  a registered tag" and ADR-002 with the typed identities.
- [ ] **Step 4:** The tests above pass. `uv run pytest packages -m "logic or contract" -q` passes.
  `uv run lint-imports` passes.
- [ ] **Step 5:** Commit `fix(identity): exact-type canonical encoder, typed identities, tagged digests`.
  Mark R-62, R-63, R-64, R-81, R-82, R-89, R-96 "resolved (H2)".

### Task H3: Store and fetch: address binding, read-only entries, transient I/O is not corruption

**Model:** opus implements; sonnet reviews (fresh context). **Tier:** logic + contract.
**Implements:** A-2, A-3, A-17, A-18, A-19, A-20, A-35, A-36 (R-60, R-61, R-75, R-76, R-77, R-78, R-93, R-94); ADR-003, ADR-006, P-34, P-37.
**Depends on:** H2, H4 (`ownership.remove_owned`).

**Files:**
- Create: `packages/senselab-core/src/senselab_core/runtime/storage.py`,
  `packages/senselab-core/src/senselab_core/runtime/data/store_policy.yaml`.
- Modify: `packages/senselab-core/src/senselab_core/runtime/{store,fetch}.py`.
- Test: `packages/senselab-core/tests/runtime/{store,fetch}_test.py`,
  `packages/senselab-core/tests/runtime/addressed_formats_test.py` (generated),
  `packages/senselab-core/tests/lint/destructive_ops_test.py`.

**Interfaces:**
- `storage.py`:
  - `class StoreUnavailable(OSError)`: retryable; carries the original `errno`;
  - `class StorageFull(StoreUnavailable)`: ENOSPC, EDQUOT;
  - `storage_fault(err: OSError, path: Path) -> StoreUnavailable`: the one mapping from `OSError` to
    a storage fault, used by store and fetch. Fetch's `StorageError` is deleted in favour of
    `StorageFull` (P-37);
  - `class StorePolicy(Frozen)`: `world_readable: bool = False`, `quarantine_retention_s: int`,
    `staging_stale_s: int`. `StorePolicy.load_default() -> StorePolicy` reads `data/store_policy.yaml`
    (values with `derivation:`);
  - `committed_modes(policy: StorePolicy) -> tuple[int, int]`: file and directory modes for committed
    objects, `0o440`/`0o2550`, plus the world read bits when `world_readable`. Staging is `0o2770`/`0o660`.
  - `AddressedFormat` (Protocol): `name: str`; `address_of(path: Path) -> str` re-derives the address
    from the bytes read; `address_at(path: Path) -> str` reads it from the location.
    `ADDRESSED_FORMATS: tuple[AddressedFormat, ...]` registers the store entry, the fetch blob and the
    fetch dataset marker.
- `store.py`:
  - `class MisfiledEntryError(CorruptEntryError)`;
  - `_read_result(entry, *, digest: str, fingerprint: str)` validates `LogicalKey` from `record["key"]`.
    It requires `key.digest() == record["key_digest"] == digest` (the path's), and it requires the
    fingerprint recomputed from `context.json` to equal the directory name. Any mismatch is
    `MisfiledEntryError`;
  - `_guarded` maps `OSError` through `storage_fault` (re-raised, never quarantined). Only
    `ValueError`, `KeyError`, `TypeError`, `ValidationError` and `CorruptEntryError` raised by a
    completed read become `CorruptEntryError`. A non-directory at the entry path is corrupt;
  - `_commit` chmods every staged file and directory to `committed_modes(policy)` before the rename;
  - `_quarantine(entry: Path, *, digest: str, fingerprint: str) -> None`: compare-and-rename. It
    renames into `quarantine/<uuid>`, re-verifies the moved copy, and if it verifies, renames it back
    with no-replace (a rename onto a now-existing non-empty entry fails with
    `ENOTEMPTY`/`EEXIST` and the moved copy is dropped). It records a provenance activity with the
    reason either way;
  - `ResultStore.__init__(self, root: Path, provenance: ProvStore | None = None, *, policy: StorePolicy | None = None)`;
  - `ResultStore.collect(self, now: float | None = None) -> int`: removes quarantine and staging
    directories older than the policy ages, through `remove_owned` (H4). `put` calls it
    opportunistically, as it does stale staging today;
  - `get` and `put` raise `StoreUnavailable` instead of returning `None` or quarantining when I/O fails.
- `fetch.py`:
  - payload lives at `datasets/<name>@<version>/files/<path>` and the marker at
    `datasets/<name>@<version>/.complete`;
  - `DataFile.path` is refused unless every `PurePosixPath` part is non-empty and not `.` or `..`.
    `DataManifest` refuses a path set that is not prefix-free (no path is a directory prefix of
    another);
  - `DataFile.size: Annotated[int, Field(ge=0)]` is required; `MANIFEST_FORMAT = "data-manifest@2.0"`;
  - `_download(url, opener, staged, *, size: int)` aborts with `FetchError` when more than `size`
    bytes arrive, or fewer at EOF;
  - `_urlopen` builds an opener whose redirect handler refuses non-`https` targets. Any non-`https`
    URL raises `FetchError` before a connection. `ALLOWED_SCHEMES = frozenset({"https"})`;
  - blobs are written with `committed_modes`. `_blob_state(blob: Path, sha256: str, recorded: Mapping[str, BlobStamp], verify) -> Literal["absent", "present", "verified"]`
    uses `os.lstat` and `stat.S_ISREG`, so a symlink or non-regular file is `absent`. `"size"`
    trust requires the full recorded `BlobStamp(size, mtime_ns, inode)` to match; otherwise the blob
    is re-hashed.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.contract
  @pytest.mark.parametrize("fmt", ADDRESSED_FORMATS, ids=lambda f: f.name)
  def test_artifact_at_wrong_address_never_reads(fmt, tmp_path):
      a, b = fmt.make_two(tmp_path)          # two valid artifacts at their own addresses
      fmt.swap(a, b)                          # copy b's bytes onto a's address
      with pytest.raises(CorruptEntryError):  # fetch's formats raise their ManifestError subclass
          fmt.read(a)

  @pytest.mark.logic
  def test_forged_result_with_recomputed_identity_is_not_served(tmp_path): ...  # misfiled key digest

  @pytest.mark.logic
  def test_committed_entry_is_read_only(tmp_path):
      store.put(KEY, OUT, CTX)
      for p in entry.rglob("*"):
          assert not (p.stat().st_mode & 0o222)

  @pytest.mark.logic
  @pytest.mark.parametrize("err", [errno.EACCES, errno.ESTALE, errno.EIO, errno.EMFILE])
  def test_transient_error_never_quarantines(err, tmp_path, monkeypatch):
      store.put(KEY, OUT, CTX)
      monkeypatch.setattr(Path, "read_bytes", raising_once(OSError(err, "x")))
      with pytest.raises(StoreUnavailable):
          store.get(KEY, CTX, ReusePolicy.EXACT)
      assert entry.is_dir() and not any((root / "quarantine").glob("*"))
      assert store.get(KEY, CTX, ReusePolicy.EXACT).outputs == OUT

  @pytest.mark.logic
  def test_quarantine_restores_an_entry_healed_in_between(tmp_path): ...

  @pytest.mark.logic
  def test_collect_removes_only_aged_quarantine(tmp_path): ...

  @pytest.mark.logic
  def test_umask_077_gives_no_world_bits(tmp_path): ...

  @pytest.mark.logic
  def test_stray_file_at_entry_is_corrupt_not_fatal(tmp_path): ...

  @pytest.mark.logic
  def test_enospc_is_storage_full(tmp_path, monkeypatch): ...  # store and fetch both

  @pytest.mark.logic
  @given(paths=manifest_path_sets())
  def test_no_manifest_path_addresses_store_owned_name(paths): ...  # accepted sets are prefix-free and never reach .complete

  @pytest.mark.logic
  def test_symlinked_or_dangling_blob_is_absent(tmp_path): ...

  @pytest.mark.logic
  def test_same_size_rewrite_is_not_trusted(tmp_path): ...

  @pytest.mark.logic
  def test_download_aborts_past_declared_size(tmp_path): ...  # opener yields endless bytes

  @pytest.mark.logic
  @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x/y", "http://x/y"])
  def test_non_https_refused(url): ...

  @pytest.mark.logic
  def test_redirect_to_non_https_refused(): ...
  ```
  Lint (`destructive_ops_test.py`, logic tier): parse every module under `packages/*/src`. For each
  `except` handler whose type is `OSError`, a subclass of it, `Exception`, `BaseException` or bare,
  fail if its body, or a same-module function it calls (followed transitively), calls `os.rename`,
  `os.replace`, `os.unlink`, `os.remove`, `os.rmdir`, `Path.unlink`, `Path.rename`, `Path.rmdir`,
  `shutil.rmtree`, `shutil.move`, `remove_owned` or a method named `_quarantine`.
- [ ] **Step 2:** `uv run pytest packages/senselab-core/tests/runtime/store_test.py packages/senselab-core/tests/runtime/fetch_test.py packages/senselab-core/tests/runtime/addressed_formats_test.py packages/senselab-core/tests/lint -v` fails.
- [ ] **Step 3: Implement.** Probe the R-98 lock and clock items while in these files: `SharedFileLock`'s
  non-atomic holder write, and staleness judged by mtime against the local clock. Add a failing test
  for each that reproduces. Fix a reproduced one by the same invariant (atomic write; staleness from
  the store's own clock via a touched sentinel). Record each outcome in R-98.
- [ ] **Step 4:** The scoped tests pass; then `uv run pytest packages -m "logic or contract" -q`.
- [ ] **Step 5:** Commit `fix(store): bind entries to their address, keep I/O faults non-destructive, own every deletion`.
  Mark R-60, R-61, R-75, R-76, R-77, R-78, R-93, R-94 "resolved (H3)".

### Task H4: Worker process lifecycle: process group, private channel, owned array directory, deadlines

**Model:** opus implements; sonnet reviews (fresh context). **Tier:** logic (+ `limits` for the timing bound).
**Implements:** A-7, A-8, A-9, A-21, A-22 (R-65, R-66, R-67, R-79, R-80); wire-protocol.md, P-34.
**Depends on:** none.

**Files:**
- Create: `packages/senselab-worker/src/senselab_worker/ownership.py`.
- Modify: `packages/senselab-worker/src/senselab_worker/{launch,codec}.py`;
  `packages/senselab-core/src/senselab_core/runtime/worker_client.py`.
- Modify: `specs/20261030-140022-senselab-2x-api/wire-protocol.md` (framing: private fds; array-dir
  ownership).
- Test: `packages/senselab-worker/tests/{launch,codec}_test.py`;
  `packages/senselab-core/tests/runtime/worker_client_test.py`;
  `packages/senselab-core/tests/runtime/worker_lifecycle_contract_test.py` (new);
  `packages/senselab-core/tests/lint/owned_deletion_test.py` (new).
- Fixtures (new backends under `packages/senselab-core/tests/runtime/backends/`): `forking_backend.py`,
  `stdin_reading_backend.py`, `subdir_backend.py`, `slow_reader_backend.py`.

**Interfaces:**
- `ownership.py`:
  - `ARRAY_NAME = re.compile(r"^[0-9a-f]{64}\.npy$")`;
  - `remove_owned(root: Path, path: Path, *, pattern: re.Pattern[str]) -> bool`. It removes `path`
    only if it is a direct child of `root` whose name matches `pattern`. By `lstat`, the child must be
    a regular file, or a real directory (a staging or quarantine directory named by a uuid pattern),
    which is removed as a tree without following symlinks. It returns whether it removed something,
    and raises nothing for absence;
  - this is the one deletion helper in `packages/*/src`.
- `launch.py`:
  - `_claim_channel() -> tuple[BinaryIO, BinaryIO]` replaces `_claim_stdout`. It duplicates fd 0 and
    fd 1 to private descriptors (non-inheritable, as `os.dup` makes them), points fd 0 at
    `/dev/null` and fd 1 at fd 2, and sets `sys.stdin` to `/dev/null` and `sys.stdout` to
    `sys.stderr`, before the backend is imported.
- `codec.py`:
  - `class Deadline`: `__init__(self, seconds: float | None)`; `remaining(self) -> float | None`;
    `expired(self) -> bool`; `check(self, what: str) -> None`, which raises `TimeoutError` naming `what`;
  - `from_wire(value, *, array_dir: Path, deadline: Deadline | None = None)`. `_read_array` opens
    with `os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)` and requires
    `stat.S_ISREG(os.fstat(fd).st_mode)`, else `ArrayIntegrityError`, checking the deadline between
    chunks.
- `worker_client.py`:
  - `WorkerClient.__init__(self, python, module, *, init, array_parent: Path | None = None, expect_runtime=None, load_timeout=None, request_timeout=None)`:
    `array_dir` is renamed outright. The client creates its own directory with
    `tempfile.mkdtemp(prefix="senselab-arrays-", dir=array_parent)` and exposes it read-only as
    `WorkerClient.array_dir`;
  - `_start` passes `start_new_session=True`;
  - `_shutdown` sends `SIGTERM` to the process group with `os.killpg(pgid, ...)`, waits
    `STOP_GRACE_S`, then sends `SIGKILL` to the group. It joins pump threads with a bound, and closes
    pipes only after the group is dead. It removes array files with `remove_owned(..., pattern=ARRAY_NAME)`
    and the directory with `os.rmdir` (a non-empty directory is left and logged). `close()` returns
    within `SHUTDOWN_BOUND_S = STOP_GRACE_S + KILL_GRACE_S + JOIN_GRACE_S`;
  - `run(self, items, params)` holds a `threading.Lock` acquired with `blocking=False`. A second
    concurrent call raises `ConcurrentUse(RuntimeError)`. One `Deadline(request_timeout)` covers
    the send (a writer thread with a bounded join, or `select` on the pipe), the receive and the decode;
  - `_remove_arrays` is total: each failure is logged with the path and the run continues, and a
    cleanup failure never replaces a result or raises;
  - `__del__` closes a leaked client (R-98 finaliser item; a `ResourceWarning` is emitted).
- Limit: a descendant that calls `setsid` itself leaves the group and is not reached. The contract
  states it, and the docstring of `WorkerClient` says so in one line.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.limits
  def test_forking_backend_leaves_no_survivor(tmp_path):
      client = WorkerClient(sys.executable, "backends.forking_backend", init={}, request_timeout=2)
      t0 = time.monotonic()
      with pytest.raises(TimeoutError):
          client.run([{"id": "a", "inputs": {}}], {})
      client.close()
      assert time.monotonic() - t0 < 2 + SHUTDOWN_BOUND_S
      assert not _alive(_grandchild_pid(tmp_path))

  @pytest.mark.logic
  def test_backend_child_cannot_read_requests():
      with WorkerClient(sys.executable, "backends.stdin_reading_backend", init={}, request_timeout=10) as c:
          for i in range(3):
              assert c.run([{"id": str(i), "inputs": {}}], {})[0]["outputs"]

  @pytest.mark.logic
  def test_client_deletes_only_what_it_created(tmp_path):
      parent = tmp_path / "p"; parent.mkdir(); (parent / "thesis.txt").write_text("x")
      with WorkerClient(sys.executable, "senselab_worker.fake", init={}, array_parent=parent) as c:
          (c.array_dir / "notes.md").write_text("y")
          c.run(ITEMS, {})
      assert (parent / "thesis.txt").exists() and (c.array_dir / "notes.md").exists()

  @pytest.mark.logic
  def test_concurrent_run_is_refused(): ...  # ConcurrentUse; the first call's results are intact

  @pytest.mark.logic
  def test_backend_subdir_in_array_dir_does_not_fail_runs(): ...

  @pytest.mark.limits
  def test_send_to_wedged_worker_respects_deadline(): ...  # 8 MB params, request_timeout=2

  @pytest.mark.logic
  def test_fifo_array_ref_is_refused_without_blocking(tmp_path): ...  # ArrayIntegrityError, < 1 s

  @pytest.mark.logic
  def test_symlinked_array_ref_is_refused(tmp_path): ...
  ```
  The contract (`worker_lifecycle_contract_test.py`, contract tier) parametrizes the fixture backends
  × {normal close, timeout, crash, `close()` from another thread}. For each, after `close()`: no
  process of the group remains, the call returned within its bound, and the client's parent
  directory holds exactly the files it held before.
  The lint (`owned_deletion_test.py`, logic tier) fails on any call to `os.unlink`, `os.remove`,
  `Path.unlink`, `shutil.rmtree` or `os.rmdir` in `packages/*/src` outside `ownership.remove_owned`.
  `os.rmdir` is also allowed in `worker_client._shutdown`, on the directory the client created.
- [ ] **Step 2:** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py packages/senselab-core/tests/runtime/worker_lifecycle_contract_test.py packages/senselab-core/tests/lint -v` fails.
- [ ] **Step 3: Implement.** Move every existing deletion call in `packages/*/src` to `remove_owned`,
  so the lint passes: store and fetch staging cleanup, and the locks module. Probe the R-98 item "a `.pth`/sitecustomize print before
  fd 1 is claimed" with a test venv carrying such a `.pth`. If it reproduces, the host's hello reader
  discards non-JSON lines before the hello and logs them, and the hello is the first JSON object.
  Record the outcome in R-98.
- [ ] **Step 4:** Run the scoped tests on Linux (ORCD CPU job or CI) and on macOS locally; both pass.
- [ ] **Step 5:** Commit `fix(worker): kill the process group, keep the channel private, delete only owned files`.
  Mark R-65, R-66, R-67, R-79, R-80 "resolved (H4)".

### Task H5: Total wire path: one line per item, total error records, typed faults, strict values

**Model:** opus implements; sonnet reviews (fresh context). **Tier:** logic + contract.
**Implements:** A-10, A-25 (types and gap rule), A-34 (R-68, R-83, R-92); R-5, R-52; wire-protocol.md.
**Depends on:** H4.

**Files:**
- Modify: `packages/senselab-worker/src/senselab_worker/{__init__,codec,serve,faults,fake}.py`;
  `packages/senselab-core/src/senselab_core/runtime/worker_client.py`.
- Modify: `specs/20261030-140022-senselab-2x-api/wire-protocol.md` (wire v2), `adr/ADR-009-migration-kernel.md`
  (typed faults at the source).
- Test: `packages/senselab-worker/tests/{serve,faults,codec}_test.py`,
  `packages/senselab-worker/tests/serve_fuzz_test.py` (new),
  `packages/senselab-core/tests/runtime/worker_client_test.py`.

**Interfaces:**
- **Wire v2** (`WIRE_VERSION = 2`): a response is one line per item, `{"id": <req>, "item": <item id>, "outputs": {...}}`
  or `{"id": <req>, "item": <item id>, "error": {...}}`, in request order, then
  `{"id": <req>, "done": <n>}`. The line budget `MAX_LINE_BYTES` applies per item. An item whose line
  would exceed it is sent as that item's error,
  `{"type": "ResponseTooLarge", "fault": "resource", "gap": false, ...}`. There is no aggregate budget
  and no batch-mate effect.
- `faults.py`:
  - `class ContentFault(Exception)` and `class ResourceFault(Exception)`: what a backend raises to
    state the class. `InfrastructureFault` exists;
  - `error_record(exc: BaseException) -> dict[str, Any]` is total. `message` is `_safe_text(exc)`:
    `str(exc)`, else `repr(exc)`, else `f"<unprintable {type(exc).__name__}>"`. The result is encoded
    with `errors="backslashreplace"` so lone surrogates survive, and the traceback is formatted under
    the same guard. `gap` is `False` only for `ContentFault`, `ResourceFault` and
    `InfrastructureFault` instances. Classification by the fallback table always sets `gap: true`.
- `serve.py`:
  - `_result(...)` is total. Any exception from `to_wire` or `dump_line` becomes the item's error
    record, and a `BaseException` other than `KeyboardInterrupt`/`SystemExit` raised by `handle` is
    the batch's per-item error. `serve` loops until stop or EOF, and nothing an item does ends it;
  - a message with a top-level `stop` key that is not exactly `true` is a `ProtocolError` reply, not
    an exit (R-98 item).
- `codec.py`:
  - `validate_message(msg: object, kind: Literal["hello", "init", "ready", "request", "result", "done", "fatal", "stop"]) -> dict[str, Any]`:
    the one structural validator, called by the writer before sending and the reader after parsing.
    Integers are `type(v) is int`, so `True` and `1.0` are not wire versions;
  - `parse_line` uses `object_pairs_hook` refusing duplicate keys and `parse_float` refusing
    non-finite results (`1e999`). Nesting deeper than `MAX_DEPTH = 64` is a `ProtocolError`
    (R-98 item), and `jcs_dumps` refuses the same depth.
- `worker_client.py`:
  - reads per-item lines until `done`, matching `item` ids in order;
  - decodes `fatal` totally: an unknown fault or missing `gap` becomes `infrastructure`, `gap: true`,
    and the worker's message and type are kept;
  - EOF before `done` raises `WorkerLost(returncode=<code>)`.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.contract
  @settings(max_examples=300, deadline=None)
  @given(behaviour=handle_behaviours())   # outputs: JSON, non-JSON, huge strings, surrogates, arrays;
                                           # errors: raising __str__/__repr__, surrogate messages,
                                           # BaseException subclasses; wrong length; non-list; deep nesting
  def test_serve_is_total_fuzz(behaviour):
      replies = _serve_in_process(behaviour, n_items=behaviour.n)   # BytesIO rx/tx, real serve()
      assert [r["item"] for r in replies[:-1]] == behaviour.item_ids
      assert replies[-1] == {"id": REQ, "done": behaviour.n}
      for r in replies[:-1]:
          assert ("outputs" in r) != ("error" in r)
          validate_message(r, "result")
      assert _serve_in_process(GOOD, n_items=1)[0]["outputs"]   # the loop is still serving

  @pytest.mark.logic
  def test_one_oversize_item_does_not_fail_batch_mates(): ...  # 20 × 1 MB: oversize items are resource errors, others outputs

  @pytest.mark.logic
  def test_error_record_is_total(): ...  # raising __str__ and __repr__, lone surrogate

  @pytest.mark.logic
  def test_fallback_classification_marks_gap():
      assert error_record(KeyError("k"))["gap"] is True
      assert error_record(ContentFault("bad input"))["gap"] is False

  @pytest.mark.logic
  @pytest.mark.parametrize("wire", [True, 1.0, "1"])
  def test_hello_wire_must_be_int(wire): ...

  @pytest.mark.logic
  def test_parse_line_refuses_overflow_and_duplicate_keys(): ...

  @pytest.mark.logic
  def test_unknown_fatal_keeps_message(): ...

  @pytest.mark.logic
  def test_crash_mid_reply_is_worker_lost_with_returncode(): ...

  @pytest.mark.logic
  def test_stop_false_is_a_protocol_error_not_an_exit(): ...
  ```
  One subprocess case of the fuzz property runs in `worker_client_test.py` with the fake backend
  driven by `FakeFaults`. Extend `FakeFaults` with `oversize_items` and `unprintable_items`.
- [ ] **Step 2:** `uv run pytest packages/senselab-worker/tests packages/senselab-core/tests/runtime/worker_client_test.py -v` fails.
- [ ] **Step 3: Implement.** Update `wire-protocol.md` to v2 (framing, per-item lines, `done`, the
  per-item budget, `validate_message` as the schema) and ADR-009 (backends raise typed faults; the
  table is a fallback that marks `gap`).
- [ ] **Step 4:** The tests pass. The `worker-floor` job (Python 3.10, numpy 1.23.5) passes the
  worker suite.
- [ ] **Step 5:** Commit `fix(worker): wire v2 with per-item lines, total error records and typed faults`.
  Mark R-68 and R-92 "resolved (H5)". Set R-83 to "partly resolved (H5); per-backend contract Phase 2;
  executor rule Phase 3".

### Task H6: Repository guards resolve code; tiers and triggers enforced

**Model:** sonnet implements; opus reviews. **Tier:** logic.
**Implements:** A-26, A-27, A-28, A-33, A-39 workflow part (R-84, R-85, R-86, R-91, R-97); P-28, P-30, ADR-009 layering.
**Depends on:** none.

**Files:**
- Modify: `src/tests/utils/moved_modules_test.py`, `scripts/check_fixture_budget.py`,
  `scripts/fixture_budget_allowlist.txt` (gains sha256 and size columns), `.importlinter`,
  `.github/workflows/tests.yaml`.
- Create: `packages/conftest.py`, `packages/senselab-core/tests/lint/{dependencies,ci_tiers}_test.py`,
  `scripts/data/fixture_budget.yaml` (code-file byte cap, with derivation).

**Interfaces:**
- Moved-module guard. `_imports(path: Path) -> set[str]` returns absolute module names:
  - sources are `.py` and `.pyi` files, and the code cells of `.ipynb` files;
  - it resolves relative imports against the file's package;
  - it covers `import a.b`, `from a import b` (as `a.b` and `a`) and `from a import *`;
  - it covers `importlib.import_module(<expr>)` and `__import__(<expr>)` when `<expr>` constant-folds
    (string literals, `+`, f-strings of constants). A non-foldable argument is reported unless the
    call site is on an allowlist;
  - it covers attribute chains rooted at an imported moved package (`senselab.utils.file_lock.X`).
  `TOMBSTONES` maps each moved module to its old file path. The test asserts the file does not exist
  and `importlib.util.find_spec(old)` is `None`.
- Fixture budget:
  - the allowlist line format is `<path>\t<sha256>\t<bytes>`;
  - every entry must exist in `git ls-tree -r freeze-1x-2026-10-30` with the same path, and its
    current size must be ≤ the frozen size, with matching sha256;
  - the scan covers every `git ls-files` path, not only test trees;
  - a code file (`.py`, `.pyi`, `.ipynb`) over `code_file_max_bytes` from the policy file fails;
  - a tracked symlink (mode `120000`) fails.
- Tier hook (`packages/conftest.py`). `pytest_collection_modifyitems` collects, for each item under
  `packages/`, the set of its marks in `{"logic", "contract", "limits", "usecase", "env"}`. If a set
  does not have exactly one member, it raises `pytest.UsageError` listing the offending node ids.
- `ci_tiers_test.py`: parses `.github/workflows/tests.yaml` and asserts that the union of the `-m`
  expressions of jobs running on `packages` selects every tier.
- `dependencies_test.py`:
  - `test_every_package_is_a_root_package`: the import names under `packages/*/src/*` equal
    `root_packages` in `.importlinter`;
  - `test_packages_import_only_declared_dependencies`: for each package, the external modules in its
    grimp graph are a subset of its `[project.dependencies]` import names, `sys.stdlib_module_names`
    and its sibling packages;
- `.importlinter` adds a `layers` contract: `senselab_core.cli | senselab_core.testing`,
  `senselab_core.runtime`, `senselab_core.kernel`, `senselab_worker`.
- `tests.yaml`: adds `push: branches: [alpha, main, "2x/**"]`. The core job's selection covers
  `logic or contract or limits`; `env` stays in `worker-floor`.

- [ ] **Step 1: Write failing tests**
  - `test_relative_import_of_moved_module_is_found`, `test_importlib_concat_is_found`,
    `test_notebook_cell_import_is_found`, `test_star_import_is_found`, `test_pyi_is_scanned`,
    `test_tombstones_hold`. Each builds a temporary tree with one bypass from the report and asserts
    `_imports` finds it.
  - `test_new_allowlist_entry_is_refused`, `test_grown_fixture_is_refused`,
    `test_large_code_file_is_refused`, `test_symlink_is_refused`, `test_non_test_tree_is_scanned`.
  - `test_untiered_test_fails_collection` (`pytester`), `test_two_tiers_fail_collection`.
  - `test_ci_selects_every_tier`, `test_every_package_is_a_root_package`,
    `test_packages_import_only_declared_dependencies`.
- [ ] **Step 2:** The scoped tests fail. Run `uv run pytest packages -q --co` to list the untiered
  tests the hook will reject, such as `test_leak_fixture_records_a_child_until_it_is_waited_on`, and
  give each its tier.
- [ ] **Step 3: Implement**, then `uv run pytest src/tests/utils/moved_modules_test.py packages/senselab-core/tests/lint -v`,
  `uv run lint-imports` and `uv run python scripts/check_fixture_budget.py`; all pass.
- [ ] **Step 4:** Commit `fix(guards): resolve imports, bind the fixture allowlist to the freeze tag, enforce tiers and layers`.
- [ ] **Step 5:** Mark R-84, R-85, R-86, R-91 "resolved (H6)". Set R-97 to "workflow resolved (H6);
  protection: owner".
- [ ] **Step 6: Owner action (A-39).** Protect `alpha` and `main`. Require the `core-only`,
  `worker-floor`, `lint-imports` and pre-commit checks, and require PRs (admins may bypass on `main`
  as today). Record the date in `STATE.md`.

### Task H7: Release gate: build all, install from `dist/`, refuse unreleasable versions

**Model:** sonnet implements; opus reviews. **Tier:** logic + env.
**Implements:** A-11, A-29, A-32 (R-69, R-87, R-90); R-30; design §5.
**Depends on:** none.

**Files:**
- Create: `scripts/release_gate.py`.
- Modify: `.github/workflows/publish.yaml`, `packages/senselab-core/hatch_build_pins.py` (the one
  pin hook; the root uses the same file), `packages/senselab-core/src/senselab_core/__init__.py`.
- Test: `packages/senselab-core/tests/core_packaging_test.py`, `packages/senselab-core/tests/release_gate_test.py`.

**Interfaces:**
- `scripts/release_gate.py`:
  - `main(dist: Path, *, floor: str) -> int`;
  - `check_versions(wheels: list[Path], floor: Version) -> None`: all wheels share one version that is
    `>= floor`, with no `.dev`, no local segment and no pre-release unless the tag is one;
  - `check_closure(wheels: list[Path]) -> None`: every `Requires-Dist` naming `senselab`,
    `senselab-core` or `senselab-worker` is satisfied by a wheel in `dist/`;
  - `check_installs(dist: Path, python: str) -> None`: `uv venv` into a temporary directory, then
    `uv pip install --no-index --find-links dist senselab`, then
    `python -c "import senselab_worker, senselab_core, senselab"`.
- `RELEASE_FLOOR = "2.0.0"` lives in the root `pyproject.toml` under `[tool.senselab.release]`.
- `hatch_build_pins.py`: refuses (raises) a version with a local segment, so `==X+local` cannot be
  written.
- `senselab_core/__init__.py`: at import it asserts
  `metadata.version("senselab-worker") == metadata.version("senselab-core")`, raising `ImportError`
  that names both versions.
- `publish.yaml`:
  - checkout with `fetch-depth: 0` and `fetch-tags: true`;
  - run `uv build --all-packages`;
  - run `uv run python scripts/release_gate.py dist --floor <from pyproject>`;
  - run `uv publish dist/*`, only after the gate passes.

- [ ] **Step 1: Write failing tests**
  ```python
  @pytest.mark.logic
  def test_version_is_at_least_the_release_floor():
      assert Version(metadata.version("senselab-core")) >= Version(RELEASE_FLOOR)

  @pytest.mark.logic
  @pytest.mark.parametrize("v", ["0.1.dev3", "2.0.0+g1234", "1.9.0", "2.0.0.dev1"])
  def test_release_gate_refuses_version(v, tmp_path): ...  # fabricated wheel metadata in tmp_path

  @pytest.mark.logic
  def test_release_gate_refuses_missing_sibling(tmp_path): ...  # root wheel only

  @pytest.mark.env
  def test_release_gate_installs_dist(tmp_path): ...  # builds --all-packages into tmp_path, runs the gate

  @pytest.mark.logic
  def test_core_refuses_mismatched_worker(monkeypatch): ...
  ```
- [ ] **Step 2:** The tests fail. `test_version_is_at_least_the_release_floor` fails in a shallow
  clone, which is intended. CI checkouts that build packages use `fetch-depth: 0`.
- [ ] **Step 3: Implement.** Dry-run the publish job on a fork or with `workflow_dispatch` and
  `uv publish --dry-run`.
- [ ] **Step 4:** Commit `fix(release): build every package and gate on an installable dist`. Mark R-69, R-87, R-90
  "resolved (H7)" and R-30 "resolved (H7)".

### Task H8: The adversarial probes as regression tests, and a scoped re-attack

**Model:** sonnet implements the regression tests (mechanical translation of the report rows); opus
reviews. The re-attack is an opus lead (effort higher) with sonnet subagents, verified by haiku (P-35, P-40).
**Tier:** logic + limits.
**Depends on:** H1–H7.

**Files:**
- Create: `packages/senselab-core/tests/adversarial/phase1_regression_test.py`,
  `packages/senselab-worker/tests/adversarial/phase1_regression_test.py`.
- Modify: `review-register.md` (R-98 outcomes), `reviews/phase1-adversarial.md` (a §7 "re-run"
  section).

The probe scripts were kept outside the repository in the adversarial round's scratch directory. The
tests are rebuilt from the break column of `reviews/phase1-adversarial.md`, which gives each probe's
inputs and wrong outcome. Each test is named `test_a<n>_<slug>` and asserts that the outcome no
longer occurs. These pin the found cases. The invariants are pinned by H1–H7's contracts.

| Report row | Regression test (asserts) | In-repo? |
|---|---|---|
| A-1 | flip-flag mutate raises; pickle/deepcopy writeable is False | yes |
| A-2 | B's entry copied onto A's path is not served; forged `result.json` is not served | yes |
| A-3 | one ESTALE on read: entry in place, nothing quarantined, `StoreUnavailable` raised | yes |
| A-4 | `NaT`, `timedelta64` ns/ps, masked array, `np.void`, `(int, Enum)` each raise | yes |
| A-5 | `model="main"`, `inputs={"a": "<path>"}` raise | yes |
| A-6 | `set` param refused at registration; four seeds give one digest for an allowed param model | yes (subprocesses) |
| A-7 | grandchild-holding backend: `run()` returns within bound; grandchild dead | yes (limits) |
| A-8 | child with default stdin: three requests answered | yes |
| A-9 | `thesis.txt` and `notes.md` survive | yes |
| A-10 | raising `__str__`, surrogate message, 20 × 1 MB: per-item records, worker alive | yes |
| A-11 | `uv build --all-packages` gives three wheels; gate passes | yes (env) |
| A-12 | multi-element `==` and `hash()` | yes |
| A-13 | `model_copy(update={"sampling_rate": -5})`, `end_ns=-1` raise | yes |
| A-14 | overwritten and shrunk WAV raise `StaleSource` | yes |
| A-16 | int16 `from_array` vs WAV read equal; `sampling_rate=True` raises | yes |
| A-17 | heal-between-check-and-rename keeps the entry | yes |
| A-18 | manifest with `.complete`, `.`, `a/b`+`a` refused | yes |
| A-19 | same-size blob rewrite re-hashed; dangling symlink absent | yes |
| A-20 | endless body aborts at size+1; `file://` refused | yes |
| A-21 | 8 MB send to wedged worker bounded; FIFO ref refused | yes (limits) |
| A-22 | concurrent `run()` refused; subdir does not fail runs | yes |
| A-23 | `"exact"` coerced, `"bogus"` raises | yes |
| A-24 | `{"x": 1e20}` either refused at construction or digestible | yes |
| A-25 | `KeyError` record has `gap: true` | yes |
| A-26 | relative, importlib, notebook, shim bypasses caught | yes |
| A-27 | new entry + 9 MB, 5 MB base64 `.py`, symlink refused | yes |
| A-28 | untiered test fails collection | yes (`pytester`) |
| A-29 | gate refuses `0.1.devN` | yes |
| A-31 | output identity ≠ key digest of the same tree | yes |
| A-32 | `2.0.0+local` refused by the pin hook | yes |
| A-33 | an undeclared external import in a package fails | yes |
| A-34 | `wire: true`, `1e999`, duplicate keys refused | yes |
| A-35 | umask 077: no world bits | yes |
| A-36 | stray file at entry: corrupt, not `NotADirectoryError`; ENOSPC: `StorageFull` | yes |
| A-37 | `Span(0, 2**200)`, `drift_ppm=-3e6`, forged identity raise | yes |
| A-38 | `manifest_key("a@b","c") != manifest_key("a","b@c")` | yes |
| A-15, A-30 | deferred (Phase 3, Phase 2) | no: owners in R-73, R-88 |
| A-39 | branch protection | no: repository settings (owner) |

- [ ] **Step 1:** Write the regression tests in the table. Each fails on `94e81479` and passes on the
  hardened branch. Check this by running the file at both commits from a scratch worktree.
- [ ] **Step 2:** Resolve the remaining R-98 items: H3, H4 and H5 probed theirs. ADR-007 path
  redaction joins R-48's owner (Phase 3 exports), and `Absent` joins Phase 4 (`Pipeline` with
  `Absent`). Close R-98 with one line per item.
- [ ] **Step 3: Scoped re-attack.** Brief an opus lead with the hardened surfaces and the invariants
  of H1–H7. The goal is to break an invariant, not to re-find a pinned case. New findings become rows
  R-99 onward with architectural resolutions. Append the outcome to `reviews/phase1-adversarial.md` §7.
- [ ] **Step 4:** Run `plans/phase-1.md` Task 15 Steps 1–2 again (CI green, the 1.x suite on ORCD
  from a pinned checkout). Update `STATE.md` to "Phase 1 done" with the log line.
- [ ] **Step 5:** Commit `test(2x): Phase 1 adversarial findings as regression tests; re-attack`.

---

## Self-review notes

- **Coverage.** All high findings (A-1..A-11) have tasks. A-11 and A-29 share H7, the release gate.
  Of the mediums, 17 are assigned (A-12..A-14, A-16..A-29), A-25 split as stated, and two are
  deferred (A-15 to Phase 3, A-30 to Phase 2). All lows are assigned (A-31..A-38), with A-39 split
  between H6 (workflow) and the owner (protection).
- **Generality.** Each task's fix test runs over a registry, not the found input:
  - H1: every `Frozen` type;
  - H2: every numpy scalar type and every declared task under two seeds;
  - H3: every addressed format, and every `OSError` handler by AST;
  - H4: every fixture backend × every shutdown path, and every deletion call by AST;
  - H5: generated `handle` behaviours;
  - H6: every import form and every tracked file;
  - H7: every built wheel.
- **Format bumps** (pre-alpha, no migration): `store@1` → `store@2` (H2 tags and typed identities,
  H3 modes); `data-manifest@1.0` → `@2.0` (H3 `size`); wire 1 → 2 (H5); the revision manifest tag
  (H2).
- **Risks.**
  - H1's derived `Audio.identity` re-hashes a source file on `model_validate`. Phase 3's scale ladder
    measures that cost with A-15.
  - H4's process-group kill does not reach a descendant that calls `setsid`. This is stated, not
    hidden.
  - H5 changes the wire protocol, so the `worker-floor` job must pass before merge.
