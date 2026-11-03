# Full review, ADR angle: ADR-001..ADR-010 against `packages/` at 3a468aff

Reviewer angle: ADR-by-ADR conformance. Branch `2x/phase-1`, worktree `senselab-2x-impl`, commit `3a468aff`.
Paths: `S/` = `specs/20261030-140022-senselab-2x-api/`; `W/` = `packages/senselab-worker/src/senselab_worker/`;
`K/` = `packages/senselab-core/src/senselab_core/kernel/`; `R/` = `packages/senselab-core/src/senselab_core/runtime/`.

Phase ownership comes from `S/design.md` §6.2 (lines 892-976), `S/plans/phase-1.md`, `S/plans/phase-1-hardening.md`
and `S/roadmap.md`. Plans are cited only to say who owns an item. They are not treated as authority.

Verdicts: **built** (conforms), **partial**, **deviation** (built differently), **omission** (decided, and due
by now, but not built), **deferred** (a named later phase owns it), **unowned** (no phase names it), **stale-text**
(the ADR no longer describes the code).

Counts: Critical 0, Important 11, Minor 12.

---

## Part A. Bullet-by-bullet conformance

### ADR-001 Graph format, compiled to pydra

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| The graph is a serialisable JSON document against a published schema | None. `K/declare.py:1-3` reads "Declarations only: no graph, planner or execution". `S/plans/phase-1.md:717` says "declarations only; no graph". Phase 3 builds the `Plan` and Phase 4 the `Pipeline` (design §6.2) | deferred. No phase names the JSON schema or `sl.validate()` as a deliverable, so it is effectively unowned (finding I-11) |
| A node is a declared task or pipeline reference with a version, params and typed ports | `TaskSpec` `K/declare.py:118-133` gives name, version, kind, params and ports. Ports are `StrMap` type *names*, free strings never checked against a type vocabulary | partial |
| An edge joins two ports and the port types must match | none | deferred (Phase 3/4) |
| Fan-out is explicit (`map`/`gather`), and `over=` names a grouping key | none | deferred (Phase 4) |
| `route`/`loop` take a declared decide task as predicate | `TaskKind` includes `decide` (`K/declare.py:19`). `VERDICT_TYPES` is restricted to decide tasks by the generated contract (`packages/senselab-core/src/senselab_core/testing/contracts.py:160`) | partial. The predicate wiring is deferred (Phase 4) |
| Nesting: a pipeline is a node with ports | none | deferred (Phase 4) |
| The plan hash and result keys come from the document plus ADR-002 identities, never from code | `LogicalKey` digests only declared fields (`K/identity.py:207-236`) | built at the task level. The plan hash is deferred |
| `Executor` protocol (`submit/status/cancel/results`) lives in the runtime | none | deferred (Phase 3) |
| `PlanFragment` is logical nodes plus placement, serialised | none | deferred (Phase 3) |
| Implementations are `InProcess`, `Pool` and `Pydra` | none | deferred (Phase 3, pydra gate) |
| A compiler maps map/gather to split/combine, nesting to a nested workflow, and route/loop likewise | none | deferred. No phase names the compiler |
| senselab registers byte representations with pydra so pydra's checksum equals the logical key | none | deferred |

**Does a serialisable graph document exist in Phase 1, and is it pydra-compatible?** No. Phase 1 was never meant
to build one (`S/plans/phase-1.md:717`). The only graph-adjacent piece is `TaskSpec`, whose ports are untyped strings,
so nothing yet shows pydra compatibility. The one Phase-1 piece that pydra compatibility depends on is the
`LogicalKey` digest (`K/identity.py:232`). It is code-free and seed-independent
(`contracts.py:16,48`), so it is a valid candidate for pydra's checksum hook.

### ADR-002 Identity v2

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| §1 Key = {schema, task, task_version, backend, backend_version, model identity, params (JCS), inputs, profile slices, batch/chunk when declared} | `LogicalKey` `K/identity.py:207-229` | partial. The model identity is not portable (I-1). `batch_formation`/`chunk_grid` are accepted as free strings whether or not the backend declares them, and `key_for` never sets them (`K/declare.py:233-234`). `contracts.py:63-64` names the owner as "Phase 2 planner", but design §6.2 places the planner and batching in Phase 3 (m-9) |
| Input identity is raw bytes for source data and the upstream output identity for derived data (early cutoff) | `ContentIdentity`/`OutputIdentity` `K/identity.py:30-37`; `output_identity` `R/store.py:150-165`; file-backed audio `K/signals.py:479`, `hashing.py:37-56` | built |
| Cached outputs are clock-relative: an output with a `clock_id` is re-bound to the consumer's anchor when served | Nothing in `R/store.py` reads or re-binds a `clock_id`. The bullet was added in commit b2b2e986, whose message is "store method inputs stay dict-typed…" | omission / unowned (I-9) |
| Long inputs use chunked identity, and a range read = chunks touched + range | `content_identity` `W/hashing.py:37-56` | partial. Range identity is deferred to Phase 3 (R-73) |
| Identities are typed (`ModelIdentity`, `content:`/`output:`, `profile:`), and an unpinned model raises | `K/identity.py:30-40,43,152-178` | built. `ModelIdentity.id` still admits a path (I-1) |
| Params are canonical by declaration (registry refuses set, Decimal, Path…) | `check_canonical_params` `K/declare.py:67-116`, called at `add_task` `K/declare.py:172` | built |
| Two `PYTHONHASHSEED` values give one digest | `testing/contracts.py:16,48,319` | built |
| Digest taken at construction (`logical-key@3`) | `K/identity.py:231-236` | built |
| Versions are declared, and a CI check fails a source change without a bump (source digest vs last release) | No check in `packages/`, `scripts/` or `.github/` | omission / unowned (I-10) |
| Never hashed: function code, Python objects, environments | Keys: conforms. Provenance ids hash an enum's `<module>.<qualname>` (`W/canonical.py:575,623-625`, `R/provenance.py:228-237`) | partial (I-6) |
| §2 Context: env lock, platform, wheel-set, numerics class, decoder **and resampler**, driver, host, protocol | `ExecutionContext` `K/identity.py:239-256` | partial. There is no resampler field, and `numerics` is a free `StrMap`. No production code builds a context from the host: the only constructor is `testing/values.py:480` (m-4) |
| Context is recorded, never keyed | `context.json` beside each entry (`R/store.py:9,391`); not part of `LogicalKey` | built |
| §3 Reuse policy `exact` / `equivalent` (default) / `tolerant` | `ReusePolicy`, `DEFAULT_REUSE_POLICY`, `acceptable` `K/identity.py:259-296`; `ResultStore.get` narrowest-first `R/store.py:817-888` | built. `DEFAULT_REUSE_POLICY` is used by no production caller: `get` requires the policy argument, and the run config that applies the default is Phase 3 |
| `equivalent` with no measurement behaves as `exact` | `EQUIVALENCE = FrozenMapping({})` `K/identity.py:269` | built |
| `tolerant` results are labelled in provenance | `_record_reuse` `R/store.py:801-815,884` records `policy_used` | partial. `provenance` is optional (`R/store.py:474`), and with none a tolerant hit is silently unlabelled (m-5) |
| §4 Verify sampling: a fraction `k` is recomputed, agreements add evidence, disagreements split classes | none. `EQUIVALENCE` is documented "Read-only: a class is added by editing this declaration, never at run time" (`K/identity.py:269-273`) | unowned, and the code contradicts the ADR (I-8) |
| §5 I-2 / I-4 / I-5 revised | I-2: `fingerprint` excludes `protocol` (`K/identity.py:254-256`). I-4: built. I-5 (composite = logical-plan hash) is deferred (Phase 3/4) | built / deferred |

**Portable logical key.** The key is portable for `hf` models only (see I-1). **Context recorded, not keyed.**
Holds. **Default `equivalent`.** Declared but applied by no caller yet, because the run config is Phase 3. With no
classes it degenerates to `exact`, as the ADR says.

### ADR-003 Storage and leases

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| Only rename, `O_EXCL` link and server-clock time | `StorageDir` `R/storage.py:222-224,250`; `fs_now` `R/storage.py:654` | built |
| Results written to their key by rename, first writer wins | `_commit` `R/store.py:365-417` | built |
| Coordinator epoch file as an efficiency fence | none | deferred (Phase 3) |
| Append-only index segments and oldest-first claims | none (`work_pool` is 1.x, under `src/`) | deferred (Phase 3) |
| Quantum yielding | none | deferred (Phase 3) |
| GPU admission by host-local `flock` budget; 2.0 uses static caps; OOM is a resource fault | OOM is classified as resource in `W/faults.py:56-72`, but as a gap. Caps are absent | partial / deferred (Phase 3) |
| Services: registry entry under lease, consumer leases, attestation | none | deferred (Phase 3) |
| Packed tier | none | deferred (design §6.4) |
| One path primitive (`O_DIRECTORY | O_NOFOLLOW`, fd-relative) | `R/storage.py:222-224,250-382` | built |
| Nothing modified in place: temp `0o600` + `fchmod` + rename | `StorageDir.write_atomically` `R/storage.py:620`, `write_atomically` `:662` | built |
| Lock file is a regular file with `st_nlink == 1`, else moved aside or `LockUnavailable` | `R/locks.py:244-260` | built |
| Holders and waits; a negative timeout waits unbounded "(1.x `build.exclusive_lock` and `file_lock`)" | `R/locks.py:148-149,172` | stale-text and a compat leftover (I-7) |
| Heartbeat stops at release | `R/locks.py:500` | built |
| Private staging for downloads `tmp/<uuid>/` | `R/fetch.py:71,297,363` | built |
| Every written name listed, plus a lint with `ALLOWED_RAW` "17 functions and 41 sites in 1.x reproduce.build/cache" | `WRITTEN_NAMES` in storage/locks/revision/fetch. `packages/senselab-core/tests/lint/written_names_test.py:74-78` has **3** entries and no 1.x module | stale-text (I-7) |
| Wrong kind at an owned path is moved aside, else `WrongKindError` | `R/storage.py:170`; store quarantine `R/store.py:524-650` | built |
| Ids that become path components are one component | `path_component` `R/storage.py:185`; `R/revision.py:104` | built |
| Modes from `StorePolicy` | `R/storage.py:93` | built |
| Entry bound to address and context (`result@2.0`, `store@4`) | `R/store.py:78,81,272-320` | built |
| Lock takeover read from the holder file | `R/locks.py:161,397,428` | built |

### ADR-004 Logical plan versus placement

Every bullet (`LogicalPlan`, `Placement`, `sl.plan`, `resume(logical_plan_hash)`, cost estimate) is **deferred to
Phase 3**: design §6.2 "the immutable `Plan`, `explain`, run lifecycle and `resume`". Nothing in Phase 1
contradicts it. The reuse policy is passed per call (`R/store.py:817`), not baked into anything hashed, which keeps
the bullet "the LogicalPlan holds the reuse policy" open. The only mismatch is the owner string in
`contracts.py:63-64` (m-9).

### ADR-005 Signal and time model

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| Tasks accept protocols `RegularSeries`, `IrregularSeries`, `FrameSequence`, `Text` | none. Phase-1 Task 5 claims "Implements: ADR-005" (`S/plans/phase-1.md:377`) but its interface list (`:385-403`) drops the protocols | unowned. The closest is Phase 4's "generic `sl.signal` tier" (m-7) |
| Concrete types validate units and layout | `Audio` `K/signals.py:431-442` | built for audio |
| Recording = named channel groups, regular or irregular | `RegularGroup` `K/signals.py:242`, `IrregularGroup` `:336`, `Recording` `:365` | built |
| Integer ns on a named `clock_id` | `K/time.py:21-69` | built |
| `TimeAnchor` may be date-shifted for privacy | no field or flag records a shift | partial |
| `TimeMap` (offset+drift **or piecewise**, with uncertainty) as a declared input | `K/time.py:28-55`: offset and drift only | partial |
| Spans `{start_ns, end_ns, clock_id}`; durations and values carry units | `Span` `K/time.py:58-69`. `Annotation.value` has no unit field, and provenance extents are float seconds with no clock (`R/provenance.py:413`, `R/bep028.py:162`) | partial (m-8) |
| VFR video is a `FrameSequence` | none | deferred (Phase 4/5) |
| Long recordings read lazily by range; a multi-day recording "is never hashed or loaded whole" | `SourceStamp.of` hashes every byte at open (`K/signals.py:187`). Every decoded read re-hashes the whole file (`S/design.md:180-192`, H9c R-103) | deviation, introduced by hardening without amending the ADR (I-5) |

### ADR-006 Formats

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| Canonical JSON (JCS), `.npy` 1.0 LE C-order, UTF-8 text | `jcs_dumps` `W/canonical.py:249`, `canonical_npy_bytes` `:476` | built |
| Tables are Parquet with a fixed writer config | none | unowned (m-10) |
| Canonical text: NFC when all code points are in Unicode 13.0, else raw | `canonical_text` `W/canonical.py:40,79-94` | built |
| Exact-type whitelist; one `canonical_scalar` | `W/canonical.py:113` | built |
| One walk, one bound (`MAX_DEPTH`, `MAX_CANONICAL_CHARS`) | `W/canonical.py:32-34,197` | built |
| `jcs_loads` accepts only fixed points | `W/canonical.py:336-367` | built. The ADR's wording is wrong (C-1 below) |
| `value@2` tokens | `Tokeniser` `W/canonical.py:562-680`; `value_digest` `K/_base.py:157-175` | built |
| Durable identities never name a Python class | Recording identity uses the `"recording"` token (`K/signals.py:375,416`), which conforms. Provenance ids hash `$enum: [<module>.<qualname>, v]` | deviation (I-6) |
| Canonical equivalences list | `canonical_npy_bytes`, `Tokeniser` | built |
| Three digest functions, "no other digest function exists" | `W/digest.py:44-100`, plus a fourth, `_raw_digest.bytes_digest` (`W/_raw_digest.py`) | built. The text contradicts itself (C-4) |
| Enforcement: `digests-are-tagged` over root packages "`senselab_worker`, `senselab_core`, `senselab`, `scripts`"; binding test at `src/tests/utils/digest_bindings_test.py` | `.importlinter` root packages are `senselab_worker`, `senselab_core`, `scripts` (no `senselab`). The test is at `packages/senselab-core/tests/lint/digest_bindings_test.py`, and its allowances no longer include 1.x modules | stale-text (I-7) |
| Registered identity tags (12) and blob tags (2) | `W/digest.py:15-37`, each checked below | built |
| `provenance@2.0` file, reader, vocabulary | `R/provenance.py:44,170-202,1040,1139-1155` | built. Superseded by ADR-010 but not marked as superseded (I-3) |
| 2.x shares no mutable state with 1.x (`2x/runs/…/resolutions@2.json`) | `R/revision.py:35-40,104` | built |
| Every artifact carries `format: <name>@<major>.<minor>` | `Formatted` `K/formats.py:22-45`; `result@2.0`, `execution-context@1.0`, `provenance@2.0`, `timemap@1.0`, `data-manifest@2.0`, `dataset-marker@2.0`, `store-policy@1.0`. The run manifest `resolutions@2.json` carries no `format` field (`R/revision.py:36-38`) | partial (C-2) |
| Readers support the current and previous major within 2.x | `Formatted` reads only its own major (`K/formats.py:43-45`) | deviation (I-4) |
| Migrations are declared tasks | none | unowned (I-4) |
| A schema bump no longer implies a wipe | store@1→@2→@3→@4, data-manifest@1.0→@2.0, result@1.0→@2.0 and wire 1→2 were each new key spaces with no migration | deviation (I-4) |

**Every tag in `DIGEST_TAGS`.** The plan's name `DIGEST_TAGS` became `IDENTITY_TAGS` | `BLOB_TAGS`
(`W/digest.py:15-37`). The ADR list (`ADR-006:175-184`) equals the registry, and `test_digest_tags_distinct_and_prefix_free` pins it.

| Tag | Site | What it hashes vs ADR | Verdict |
|---|---|---|---|
| `logical-key@3` | `K/identity.py:232` | `model_dump()` of the key, including `schema_version` | conforms. The model `id` may be a path (I-1) |
| `output@3` | `R/store.py:151,160-165` | encoded output tree, arrays as `.npy` digests | conforms |
| `context@3` | `K/identity.py:254-256` | every field except `host` and `protocol` | conforms. The ADR does not say that `host` is excluded (m-4) |
| `content-root@2` | `W/hashing.py:44-55` | `{chunk_size, chunks}`, with chunks hashed `file@1` | conforms |
| `audio@3` | `K/signals.py:454,481` | `{npy, sampling_rate}` | conforms |
| `weights@3` | `W/hashing.py:59-82` | sorted `[relpath, file@1]` pairs | conforms |
| `value@2` | `K/_base.py:31,157-180` | `{"type": "<module>.<qualname>", fields}` | names a Python class. Durable only if persisted (C-3) |
| `provenance-id@4` | `R/provenance.py:202,466-484` | `[run_id, fields]`, truncated to 16 hex, with enum class names | conforms to ADR-006. Conflicts with ADR-002 and ADR-006 line 90 (I-6) and with ADR-010 (I-3) |
| `data-manifest@3` | `R/fetch.py:186-187` | `model_dump(mode="json")`, which includes `format: "data-manifest@2.0"` | conforms. Two version numbers sit under one name (m-11) |
| `worker-runtime@1` | `W/__init__.py:14-25` | `[[relpath, file@1]]` | conforms |
| `fake-output@1` | `W/fake.py:100,139` | inputs, params and nonce | conforms |
| `contract-probe@1` | `testing/contracts.py:172,240,248` | stand-in ids, labelled `content:` although they hash no content | conforms (test only) |
| `file@1` | `W/digest.py:60-82`; `W/hashing.py:50`; `R/models.py:80,98`; `R/fetch.py:410,429` | plain sha256 of a file or chunk | conforms |
| `npy@1` | `W/digest.py:85-100`; `R/store.py:145,209`; `K/_base.py:118-120` | plain sha256 of canonical `.npy` | conforms |

### ADR-007 Trust and privacy

| Decision bullet | Code / owner | Verdict |
|---|---|---|
| Model sources are allowlisted in the execution profile | `resolve` `R/models.py:118-136` accepts any `hf`/`url`/`local` reference | deferred. Phase 2's "execution-profile resolver" is the nearest owner, and it is not named (m-12) |
| Remote code only in an isolated env with the network guard | the network guard is 1.x only (`src/senselab/utils/network_guard.py`) | deferred (Phase 2) |
| MCP and agent roles cannot add sources | none | deferred (Phase 5) |
| Keys and identities are hashes with no identifiers | `LogicalKey.model.id` holds a `local:` path verbatim (`K/identity.py:56-57,160`) | deviation (I-1) |
| Provenance and BEP028 exports redact paths and file names by default | `R/bep028.py:148-155` writes `Label = Path(path).name` and `AtLocation = str(path)` | deviation (I-2). The deferral to "Phase 3 exports" is recorded only in a plan (`phase-1-hardening.md:921-922`), and design §6.2 Phase 3 does not list it |
| Stores per project, with filesystem permissions | `StorePolicy` modes exist (`R/storage.py:93`). There is no project notion and no recorded cross-project copy | partial / unowned |

### ADR-008 Observability

Every bullet is **unowned**: the event stream (`queued … verify`), metrics derived from events, `senselab status`,
`senselab explain`, the OpenTelemetry export, and a versioned event schema. Design §4.10a restates them, but no phase
build list in §6.2 names them. Phase 3 ("run lifecycle") is the natural owner. The CLI has only `data fetch`
(`packages/senselab-core/src/senselab_core/cli.py:13-16`). Today the closest thing to a `cache_hit` event is the
provenance activity `store.reuse` (`R/store.py:801-815`), and it is emitted only for non-exact hits. See I-11.

### ADR-009 Migration, kernel, faults (amended 2026-11-01), against H-R51 and H-R52

| Decision bullet | Code / state | Verdict |
|---|---|---|
| Migrate one use case at a time; `use-cases.md` lists them | `S/use-cases.md` | built (docs) |
| Each use case has a coexistence budget (dates) | No dates in `S/use-cases.md` or `S/migration-ledger.md` | omission / unowned (m-6) |
| A 1.x module is deleted when its last use case moves | `src/senselab/utils/` no longer has `file_lock`, `model_revision` or `prov_store` (core replaced them; H-R51/H-R52 "H10 deletes only 1.x modules core already replaced") | conforms to H-R52. **ADR-009 does not state the "already replaced by core" deletion rule** that H-R51 and H-R52 add (m-1) |
| 1.x on `alpha` and the freeze branch; 2.x on `2.x` | `origin/freeze-1x-2026-10-30`, `origin/2.x`; `.github/workflows/2x-tests.yaml:5-8` (`2.x`, `2x/**`) | built |
| `src/` is the 1.x pool: not packaged, tested or linted | `pyproject.toml` `testpaths=["packages"]`, ruff/mypy/codespell exclude `src`, `.pre-commit-config.yaml:2-3` | built |
| Nothing in `packages/` imports it | `.importlinter` `packages-not-1x` | built |
| 1.x code outside `src/` | About 60 1.x drivers in `scripts/` (`extend_*`, `triage_*`) import `senselab` (1.x), are an import-linter root, and are ruffed | stale-text: ADR-009 is silent on `scripts/` and `tutorials/` (I-7) |
| How a use case moves: merge alpha, then `git mv`, then modify | process only | n/a |
| Triage: tasks and pipelines port, and the decision layer is deleted when its last use case moves | `S/migration-ledger.md:35-90` classifies each module | conforms to H-R52. **`S/design.md:953-958` still says "Triage is not ported as a pipeline … Its decision layer and orchestration stay on the frozen 1.x tag"**, which contradicts H-R52's "pipelines port like any use case (generalised per dataset)" (I-7). ADR-009 also omits H-R52's "pipelines generalised per dataset" (m-1) |
| Kernel/runtime split; dependency direction worker ← kernel ← runtime ← backends | `.importlinter` `worker-is-the-base`, `kernel-below-runtime`, `layers` | built. The direction was reversed in commit 10981014 without an amendment mark (m-2) |
| Faults typed at the source; typed faults have `gap: false` | `W/faults.py:15-79` | built |
| Fallback table always marks a gap | `W/faults.py:55-73,110-125` | built |
| Gap reported by the host | `R/worker_client.py:745-746,989-1003` | built |
| Zero gaps per backend contract | owner: Phase 2 | deferred |
| Never cache a gap fault | owner: Phase 3 | deferred |
| Only `content` charges the retry budget | owner: Phase 3 | deferred |

### ADR-010 W3C PROV (accepted 2026-11-01, not implemented)

**How far the code is.** Nothing of ADR-010 is built. `R/provenance.py` is still the custom model ADR-010 rejects:
- the `Entity`, `Activity`, `Agent` and `Environment` dataclasses (`:499-656`);
- relations as a closed `RELATION` literal of tuples (`:52-54`);
- the closed lists `PROV_TYPE = Literal["artifact"]` (`:44`), `AGENT_TYPE` (`:50`) and `ENVIRONMENT_KIND` (`:51`);
- the JSONL `provenance@2.0` file (`:170,1040`);
- ids of the form `<prefix>-<16 hex>` of `provenance-id@4` over `[run_id, …]` (`:202,466-484`). These are
  run-scoped, truncated and not content addresses. ADR-010 instead wants an entity `@id` to be the content address
  and an activity `@id` to be its logical key.

`R/bep028.py` is still a translation layer (`to_bep028_graph` `:236`). There is no `prov:Plan`, no qualified
association, and no `senselab:` vocabulary or published `@context`.

**What implementing it touches:**
- `R/provenance.py`, rewritten as a PROV-O JSON-LD writer and reader;
- `R/bep028.py`, which collapses into a profile check (`check_graph` `:366`), along with its `ALLOWED_RAW` entry in
  `written_names_test.py:76`;
- `R/store.py:773-815`: `_record_put`, `_result_entity` and `_record_reuse` use `prov_type="artifact"` and
  `node`/`step`;
- `packages/senselab-core/src/senselab_core/testing/provenance.py`;
- the shared tokeniser's provenance branch (`$nd`, `$enum`) and `K/_base.py` `FrozenDict`/`thawed`;
- `W/digest.py` (`provenance-id@4` is retired or bumped) and `test_digest_tags_distinct_and_prefix_free`;
- the `.importlinter` `raw-digest-owners` (unchanged);
- the release gate (`scripts/release_gate.py`), which must check the vocabulary file;
- the migration-ledger statement `R/provenance.py:176`;
- ADR-006 lines 221-272 and the provenance rows of its tag list;
- `S/design.md:456`, which still points at `provenance@2.0`;
- ADR-003's WRITTEN_NAMES and BEP-028 sidecar text;
- the tests `provenance_test.py`, `provenance_checksum_test.py`, `bep028_test.py`, `store_test.py` and the
  adversarial canon and kernel tests.

The plan-related parts (`prov:Plan` = graph document, `hadPlan`, node id as role) depend on ADR-001's graph document
and ADR-004's plan hash, which do not exist until Phase 3/4. See I-3 for the sequencing and id conflicts.

---

## Part B. Findings

### Important

#### I-1. A `local:` or `url:` model puts a path or URL into the "portable" logical key
- **Departs from:** ADR-002 §1, which says "Logical key: exact, and portable across machines" and "The key refuses
  a ref, a path or bare hex where an identity belongs" (`S/adr/ADR-002-identity-v2.md:18,34-35`). It also departs
  from ADR-007 "Keys and identities are hashes and carry no identifiers" (`S/adr/ADR-007-trust-privacy.md:22`).
- **What the branch does:** `ModelRef` parses `local:<path>` with `id` = the path verbatim (`K/identity.py:56-57`).
  `key_identity()` copies `ref.id` into `ModelIdentity.id` (`K/identity.py:160`), which enters the key
  (`K/identity.py:222,232`). The same weights at `/home/user/m` and `/scratch/b/m`, or behind two mirror URLs, get two
  keys, and a key can carry a username or participant directory.
- **Kind:** deviation.
- **Correction:** for `local` and `url`, the key's model identity should carry only `source` and the weight
  `identity`, with `id` recorded in context or provenance. Keep `id` for `hf`, where it is portable. Bump
  `logical-key@3` to `@4`.

#### I-2. The Phase-1 BEP028 exporter writes paths and file names; redaction is "deferred" only in a plan
- **Departs from:** ADR-007 "Provenance and BEP028 exports redact paths and file names to entity-free labels by
  default", whose alternatives rejected "Redacting at publish time only. Too easy to forget"
  (`ADR-007:23-24,35`).
- **What the branch does:** `R/bep028.py:148-155` emits `"Label": Path(path).name` and `"AtLocation": str(path)`
  for any entity with a `path` attribute. The deferral exists only in `plans/phase-1-hardening.md:921-922` ("ADR-007
  path redaction joins R-48's owner (Phase 3 exports)"). Design §6.2 Phase 3 (`S/design.md:928-942`) lists no
  export or redaction work, and R-48 still reads "resolved in ADR-007".
- **Kind:** deviation. A shipped exporter does the opposite of the default.
- **Correction:** make the exporter emit the entity id as `Label` and no `AtLocation` unless the caller passes an
  explicit private mapping. Alternatively, remove `write_bep028_files` from Phase 1 until redaction lands. Add
  "ADR-007 redaction" to the design §6.2 phase that owns it, and re-open R-48.

#### I-3. ADR-010 has no owning phase; ADR-006 is not marked superseded; its id rule conflicts with ADR-002
- **Departs from:**
  - ADR-000 "A record that is superseded keeps its number and names its successor" (`ADR-000:6-7`).
  - ADR-010 "`provenance@2.0` and its JSONL format are retired" and "an entity's `@id` is the content address of
    what it names, and an activity's is its logical key" (`ADR-010:24,38-39`).
- **What the branch does:**
  - ADR-006 still decides the provenance file, reader, vocabulary and `provenance-id@4` (`ADR-006:175-272`), with
    no pointer to ADR-010.
  - `S/design.md:456` points at ADR-006 `provenance@2.0`.
  - No plan, design phase or `STATE.md` line owns ADR-010.
- **ADR-010's id rule:**
  - It collides with ADR-002/ADR-003. One logical key can have several results, one per context fingerprint
    (`R/store.py:5-9`), plus reuse activities, so "activity `@id` = logical key" merges distinct executions.
  - It also collides with ADR-010's own "two different records under one id are refused" rule, which it inherits.
  - "The pipeline graph is the plan" needs the ADR-001 document and the ADR-004 hash (Phase 3/4), but provenance is
    written from Phase 1 onward.
- **Kind:** omission (an owner decision without an implementer) plus an internal ADR conflict.
- **Correction:**
  - Mark ADR-006's provenance bullets "superseded by ADR-010" and update ADR-000 and `design.md:456`.
  - Assign ADR-010 to a phase in design §6.2.
  - Amend ADR-010 so that an activity's `@id` is the key digest plus the context fingerprint, plus an attempt
    where needed.
  - Say what `hadPlan` points to for a single task call before a graph exists.

#### I-4. Format majors were bumped as new key spaces, against ADR-006's reader and migration rules
- **Departs from:** ADR-006 "Readers support the current and the previous major version, within 2.x", "A major
  change requires a migration", "Migrations are declared tasks", and "A schema bump no longer implies a wipe … A
  wipe is reserved for an identity change" (`ADR-006:282-288`).
- **What the branch does:**
  - Hardening bumped `store@1→@4`, `data-manifest@1.0→@2.0`, `result@1.0→@2.0` (an added required field, not an
    identity change) and wire 1→2.
  - Each was a new key space, with no previous-major reader and no migration task
    (`plans/phase-1-hardening.md` "Format bumps (pre-alpha, no migration)"; `ADR-006:215-219`).
  - `Formatted` accepts only its own major (`K/formats.py:43-45`).
- **ADR-006 now contradicts itself:** lines 215-219 call `result@2.0` "a new key space, not a migration", but
  line 287 forbids a wipe for anything but an identity change. The consequence "Every reader needs a migration test
  from the previous major" (`:292`) has no test anywhere.
- **Kind:** drift (a hardening rule contradicts the ADR, which was never amended).
- **Correction:** amend ADR-006 with an explicit pre-release clause, for example "before 2.0.0 a major bump is a new
  key space; from 2.0.0 the previous-major reader and migration rules apply". Name the release that starts the
  rule.

#### I-5. ADR-005 says a long recording is never hashed whole; Phase 1 hashes it on open and on every read
- **Departs from:** ADR-005 "Long recordings are accessed lazily, by time range and in chunks. Identity is chunked
  (ADR-002), so a multi-day recording is never hashed or loaded whole" (`ADR-005:24-25`).
- **What the branch does:** `SourceStamp.of` hashes every byte (`K/signals.py:187`). Since H9c (R-103), every
  decoded `read` stamps the file again, content included, which is a full hash per window read
  (`S/design.md:180-192`). ADR-005 was never edited (its git log has only 235d83a1).
- **Kind:** drift (hardening changed behaviour the ADR fixes, with no amendment).
- **Correction:** either amend ADR-005 to state the Phase-1 cost and the R-73/R-109 end state, or treat it as a
  defect. In the second case the owners also disagree: R-73 says Phase 3 and R-109 says Phase 2. Pick one.

#### I-6. Durable provenance ids hash Python class names
- **Departs from:** ADR-006 "Durable identities never name a Python class" (`ADR-006:90`) and ADR-002 "What is
  never hashed: function code, Python objects" (`ADR-002:51`).
- **What the branch does:** the shared tokeniser writes an enum as `{"$enum": ["<module>.<qualname>", v]}`
  (`W/canonical.py:575,623-625`). Provenance records it as `EnumMember.type_name` (`R/provenance.py:221-237`), and
  that token is hashed into `provenance-id@4` (`R/provenance.py:466-484`). Moving or renaming the enum's module
  changes every id that holds it. `value@2` also hashes `<module>.<qualname>` (`K/_base.py:147-149,179`), and
  ADR-006 itself documents both tokens (`:79-80`).
- **Kind:** deviation, and an ADR self-contradiction.
- **Correction:** refuse enums and types in provenance values. ADR-010's declared-range attributes do this anyway.
  Alternatively, require a declared token on the enum, as `Recording.IDENTITY_TYPE` does. State in ADR-006 that
  `value@2` is an in-process equality digest and is never persisted.

#### I-7. ADR-003, ADR-006 and design.md describe the pre-H-R51 tree; 1.x compat survives in `locks`
- **Departs from:**
  - owner request 32/33 ("wasn't one of the principles no backwards compatibility?", "check other packages for 1.x
    compat leftovers");
  - ADR-009 as amended ("src/ is the 1.x source pool: frozen … not tested and not linted").
- **Stale text in ADR-003:**
  - "1.x `reproduce.build.file_lock` takes the same lock" (`ADR-003:90-91`).
  - "a negative `timeout` (1.x `build.exclusive_lock` and `file_lock`)" (`:97-99`).
  - The lint covers "1.x `reproduce.build` and `reproduce.cache` … 17 functions and 41 sites" (`:115-122`). The
    actual `ALLOWED_RAW` is 3 entries with no 1.x module (`written_names_test.py:74-78`).
  - "the HF cache's `runs/`" (`:69`). It is now `2x/runs/` (`R/revision.py:104`).
- **Stale text in ADR-006:**
  - The root packages include `senselab` (`:141-142`); `.importlinter` has no `senselab`.
  - The binding test is at `src/tests/utils/digest_bindings_test.py` (`:148-149`); it is now at
    `packages/senselab-core/tests/lint/`.
  - The allowances are "the 1.x modules and scripts of H-R19" (`:145-146`); only scripts remain.
- **Compat leftovers in code:** `SharedFileLock`'s negative-timeout unbounded wait and `shared=True` mode
  (`R/locks.py:148-149,157-158`) exist for the 1.x callers named in ADR-003. The only 2.x caller is
  `R/revision.py:193`, which uses neither.
- **ADR-009 gaps:** ADR-009 says nothing about the 1.x drivers in `scripts/`. These are about 60 `extend_*` and
  `triage_*` files that import `senselab`, are an import-linter root, are linted, and carry `digests-are-tagged`
  allowances.
- **design.md:** `S/design.md:953-958` ("Triage is not ported as a pipeline … decision layer and orchestration stay
  on the frozen 1.x tag") contradicts H-R52 and amended ADR-009.
- **Kind:** stale-text plus drift; the owner's correction has not fully landed.
- **Correction:**
  - Rewrite those ADR-003 and ADR-006 sentences against the present tree.
  - Remove negative timeout and `shared` from `SharedFileLock`, or name a 2.x caller.
  - State in ADR-009 where 1.x `scripts/` and `tutorials/` live: move them under `src/` or exclude them like `src/`.
  - Fix design §6.2 Phase 4 to match H-R52.

#### I-8. Equivalence classes are a frozen code literal of full fingerprints, which ADR-002 §4 rules out
- **Departs from:** ADR-002 §3/§4: `equivalent` uses "a measured equivalence class … (the tolerance profiles,
  P-17)", and "Agreements add evidence to the equivalence class. Disagreements split the class"
  (`ADR-002:67,72-75`). It also departs from the Consequence "Changing a decoder or driver no longer invalidates the
  cache" (`:88-89`).
- **What the branch does:**
  - `EQUIVALENCE` is a read-only in-code mapping of whole context fingerprints, "added by editing this declaration,
    never at run time" (`K/identity.py:269-273`), and an adversarial test pins it read-only
    (`tests/adversarial/reattack_kernel_test.py:299`).
  - The fingerprint covers `env_lock`, `wheel_set`, `decoder` and `driver` (`K/identity.py:254-256`). So under the
    default policy any driver or lock change makes earlier results unreusable until someone edits code. That is the
    R-40 behaviour ADR-002 replaced.
  - Classes are not loaded from tolerance profiles, and verify sampling has no owner (design §4.8 and `testing.md:137`
    describe it; no phase builds it).
- **Kind:** deviation (hardening fixed the class source as code), plus an unowned item.
- **Correction:**
  - Load classes from the data-side tolerance profiles (P-17), keyed by the numerics class, not by the whole
    fingerprint.
  - Give verify sampling a phase.
  - Amend ADR-002 if code-declared classes are intended for 2.0.

#### I-9. Clock re-binding of served outputs was added to ADR-002 silently and has no implementation or owner
- **Departs from:** ADR-002 "a stored output carrying a `clock_id` is re-bound to the consumer's anchor when it is
  served" (`ADR-002:29-31`).
- **What the branch does:**
  - The bullet entered in commit b2b2e986, whose subject is "store method inputs stay dict-typed for callers'
    literal inference (H9c round 1)".
  - `ResultStore.get` returns stored outputs as read (`R/store.py:853-888`).
  - Nothing in `packages/` re-binds `clock_id`.
  - No plan or phase names it.
  - `Audio` names its clock by its identity (`K/signals.py:483-486`), so two consumers on different session clocks
    receive the producer's clock.
- **Kind:** omission, unowned; the ADR was amended without a decision record.
- **Correction:** record the decision (owner or ruling) in ADR-002's status. Assign re-binding to the executor
  (Phase 3), or make the store's `get` take the consumer anchor.

#### I-10. The "declared versions" CI check exists nowhere and is owned by no phase
- **Departs from:** ADR-002 "A CI check fails when a task's or backend's source changes without a version bump,
  comparing a source digest against the last released one" (`ADR-002:47-50`), whose Consequence is "The CI
  source-digest check enforces it" (`:90`). See also `S/design.md:409`.
- **What the branch does:** there is no such check in `packages/`, `scripts/` or `.github/workflows/`. The only
  declared backend is `fake`. The guarantee that makes "versions are declared, not hashed" safe is absent, and no
  phase plan lists it.
- **Kind:** omission (unowned).
- **Correction:** add it to Phase 2 (the first real backends) in design §6.2. A source digest per declared module,
  stored at release and checked by the release gate, is enough.

#### I-11. ADR-001's graph schema and ADR-008's event stream have no phase deliverable
- **Departs from:** ADR-001 "A senselab graph is a serialisable document (JSON against a published schema)" with
  "Agents author pipelines as JSON against the graph schema and get errors from `sl.validate()`"
  (`ADR-001:14-15,52`). Also ADR-008's events, metrics, `status`/`explain` and OTel (`ADR-008:10-22`).
- **What the branch does:** neither exists, which is correct for Phase 1. But design §6.2 names neither:
  - Phase 3 lists "the immutable `Plan`, `explain`, run lifecycle and `resume`";
  - Phase 4 lists "`Pipeline` with ports, nesting, routing, loops";
  - neither names the JSON schema, `sl.validate()`, the pydra compiler, the event schema or
    `senselab status`/`explain <run>`.

  ADR-010 now depends on the graph document.
- **Kind:** omission at the planning level (unowned).
- **Correction:** add "graph schema + `validate` + pydra compiler" to Phase 3 or 4, and "event stream + derived
  metrics + `status`/`explain`" to Phase 3, in design §6.2.

### Minor

- **m-1. ADR-009 does not fully restate H-R51 and H-R52 as they now read.** It omits two things: H-R51/H-R52's
  "H10 deletes … 1.x modules core already replaced" (a deletion before any use case moves, which conflicts with the
  ADR-009 line "deleted when its last use case moves"), and H-R52's "pipelines generalised per dataset". Add both,
  or reconcile the deletion rule.
- **m-2. Accepted ADRs are edited in place with no amendment record.**
  - ADR-002 (4 edits), ADR-003 (5) and ADR-006 (17) keep "Status: accepted, 2026-10-30".
  - ADR-009 marks only the 2026-11-01 bullet. Its dependency-direction reversal (10981014) and fault rewrite
    (1e333faf) are unmarked.

  Add an "Amended" line per change naming the ruling.
- **m-3. ADR-006 reads as a changelog.** The H9b/H9c/H9d/H9e bump history (`:186-219`) mixes current rules with
  superseded states (`store@3` at `:211-213` and `provenance-id@3` at `:201`). Move the history out and keep the
  current rule.
- **m-4. `ExecutionContext` departs from ADR-002 §2 in four ways:**
  - it has no resampler identity;
  - `numerics` is untyped, where the ADR names device type, architecture and precision;
  - the fingerprint excludes `host`, which the ADR does not state;
  - no production code builds a context from the hardware probe.
- **m-5. A tolerant hit is unlabelled without a provenance store** (`R/store.py:474,884`). ADR-002 says such results
  "are labelled `tolerant` in provenance". Either require provenance for non-exact `get`, or record the policy in
  the returned result's provenance at the caller.
- **m-6. ADR-009's coexistence budget has no dates.** `use-cases.md` and `migration-ledger.md` carry none, and no
  phase owns setting them.
- **m-7. Plan Task 5 claims "Implements: ADR-005" but drops the protocols.** `RegularSeries`, `IrregularSeries`,
  `FrameSequence` and `Text` are absent, and so are piecewise `TimeMap` and the date-shift marker. Name an owner
  (Phase 4 `sl.signal`).
- **m-8. Provenance entity extents are float seconds with no `clock_id`.** See `R/provenance.py:413-427` and
  `R/bep028.py:162` (`ExtentStartSeconds`). ADR-005 says spans are integer ns on a clock.
- **m-9. Owners named in code disagree with design §6.2.** `contracts.py:63-64` says `batch_formation`/`chunk_grid`
  are owned by the "Phase 2 planner (ADR-004)", but design puts the planner and batching in Phase 3.
- **m-10. Parquet tables have no implementation and no owner** (ADR-006 "tables: Parquet with a fixed writer
  configuration"). `Events`/`Intervals` exist as columnar kernel types (`K/results.py:59-90`).
- **m-11. Format tags and identity tags share one `name@N` syntax with unrelated numbers.** For example, the format
  is `data-manifest@2.0` (`R/fetch.py:55`) while the identity tag is `data-manifest@3`. ADR-006 should say the two
  namespaces are distinct, or rename one, for example `data-manifest-id@3`.
- **m-12. The ADR-007 source allowlist and isolated remote code have no named owner.** Phase 2's
  "execution-profile resolver" is the obvious home. R-48 says "resolved in ADR-007", although nothing in it is built.

### ADR-006 internal contradictions (part of the ADR-006 question)

- **C-1.** Line 73 says `jcs_loads` refuses "non-NFC text", but lines 28-33 keep post-13.0 text raw and never
  normalise it. The code refuses only non-fixed points (`W/canonical.py:336-367`), so raw non-NFC text is accepted.
  Line 73 is wrong.
- **C-2.** Line 279 says "Every artifact carries `format: <name>@<major>.<minor>`", but lines 276-277 keep the run
  manifest as `resolutions@2` (no minor, and no `format` field in the file).
- **C-3.** Line 90 says "Durable identities never name a Python class", but lines 79-80 define the `$enum`/`$type`
  tokens with qualified names in `value@2` and provenance (I-6).
- **C-4.** Line 119-120 says "no other digest function exists" (three, typed by source), but line 134 adds
  `_raw_digest.bytes_digest`.
- **C-5.** Lines 94-97 say a store and `merge` keep "whichever arrived first" of `1` and `1.0`, so the bytes depend
  on order. Lines 232-234 claim "the union is order-independent". That claim holds only for timings.
- **C-6.** Lines 215-219 call the store@4 bump a new key space, while line 287 says a wipe is reserved for an
  identity change (I-4). Lines 253-260 (no `provenance@1` reader) agree with line 283 ("within 2.x"), but
  lines 221-272 are superseded by ADR-010 (I-3).

### Hardening changes to ADR-fixed behaviour without an ADR amendment

| ADR | Change | Task | Amended? |
|---|---|---|---|
| ADR-002 | Equivalence classes read-only code literal | H8/H9 reattack | no (I-8) |
| ADR-002 | Clock re-binding bullet added | H9c round 1 | added without a ruling (I-9) |
| ADR-005 | Full-file hash at open and on every read | H1, H9c (R-103) | no (I-5) |
| ADR-006 | Majors bumped as new key spaces, no migration | H2, H3, H9b, H9d | partly, and self-contradictory (I-4) |
| ADR-003 | 1.x lock semantics kept after H-R51 | H9b, H-R41 | text stale (I-7) |
| ADR-009 | Dependency direction reversed; faults rewritten | 10981014, H5 | unmarked (m-2) |
| ADR-001, -004, -007, -008 | none | n/a | n/a. These ADRs are unbuilt, and the text still matches intent |

**ADRs whose text no longer describes the code:** ADR-003 (1.x lock and lint paragraphs), ADR-005 (long-recording
hashing), ADR-006 (enforcement paragraph, `jcs_loads` wording, previous-major readers, provenance now under ADR-010),
and ADR-002 (clock re-binding is stated as fact but not built; equivalence-class evolution).
