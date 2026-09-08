# Triage provenance: file checksums and a BEP028 PROV-O JSON-LD serializer

Two additions to `src/senselab/utils/prov_store.py`'s model, and the measurements behind them.

Target specification: BIDS BEP028 (`bids-standard/bids-specification` PR #2099, head
`bclenet:BEP028_spec`, read at 315 commits / 26 changed files). The authoritative field definitions
are `src/schema/objects/metadata.yaml` (`Activities`, `Software`, `Files`, `Datasets`, `ProvEntity`,
`Environments`, `Checksum`), the JSON-LD context is `src/provenance-context.json`, and the prose is
`src/modality-agnostic-files/provenance.md`.

## Corrections to the brief that commissioned this work

Three claims in the brief do not survive contact with the specification.

1. **The `Checksum` object's keys are `ChecksumAlgorithm` and `ChecksumValue`, not `Algorithm` and
   `Value`.** `metadata.Checksum.items` requires both, `ChecksumAlgorithm` is a URI whose value
   "MUST be as provided by SPDX (for example: `spdx:checksumAlgorithm_sha256`)", and
   `ChecksumValue` is "a lower case hexadecimal encoded digest value". `Checksum` is an **array** of
   such objects, not a single object.
2. **`prov-<label>_io.json` is one of four suffixes, not the provenance file.** BEP028 defines
   `_act` (Activities), `_soft` (Software), `_io` (Files, Datasets, `prov:Entity`) and `_env`
   (Environments), under `prov/[<subdir>/]`. A single aggregated JSON-LD document is separately
   sanctioned for the graph view but is given no filename.
3. **`Entity` does not carry `AtLocation`.** `metadata.Files.items` has `AtLocation`;
   `metadata.ProvEntity.items` does not. An entity with a location must therefore be classified as
   a `Files` object, which decides the partition below.

Verified in the codebase, against the brief:

- Nothing populates `Activity.started` / `ended`. A grep for `started=` / `ended=` across
  `src/senselab/` outside `prov_store.py` returns nothing, and all 40 activities in the battery
  store carry `"started": null, "ended": null`. The brief was right to doubt it.
- The entity id format is at `prov_store.py:158`, as stated, but is `f"{prov_type}-{digest}"` for
  every `prov_type`, not only `stream`.
- `store.jsonl` in a real battery run is **4.65 MB**, not 1.4 MB.

## Addition 1 — file checksums

### What gets one

Every entity carrying a `path`. In the battery run measured below that is **21 of 761 entities**:
6 `stream` (the input recording plus `plain`, `preemphasised`, `normalized`, `enhanced`,
`residual`) and 15 `measurement` (the `derivatives/*.json` and `derivatives/*.npz` sidecars). The
remaining 740 — spans, words, verdicts, assertions, kinds — have no bytes on disk and get nothing.

`store.jsonl` itself gets no checksum. It cannot: the digest would have to be a record inside the
file it digests. `to_bep028_graph` emits `StoreFingerprint` (the store's own order-independent
content hash, `prov_store.py:fingerprint`) instead, which identifies the store's *content*
independently of its serialization. A digest of `store.jsonl` belongs to whatever writes the run
directory's manifest, not to the store.

### Algorithm

SHA-256, full 64-character lowercase hex. BEP028 names it; `prov_store.py` already imported
`hashlib`. Note that the store's own `_digest` truncates SHA-256 to 16 hex characters — that is an
*identifier*, deliberately short, and is a different thing from a checksum. The two must not be
confused, so the checksum is stored under its own key (`checksum_sha256`) and validated to 64
characters, which a truncated id cannot pass.

### Where it is computed

In the process that wrote the file, immediately after the write, via
`nodes/common.py:path_attributes`. Never at read time.

The decisive measurement: of the 21 path-carrying entities in the battery run, **8 name files that
no longer exist** on the machine doing the conversion — `streams/preemphasised.flac`,
`streams/normalized.flac`, and the six `derivatives/*.npz` — and a ninth, the input recording, sits
at an absolute `/orcd/data/...` path that does not resolve locally. That run executed on ORCD and
only part of it was retrieved. A read-time digest would have been unavailable for 9 of 21 files and,
worse, would have described the files *as they are now* rather than as the pipeline used them. That
distinction is the entire point of recording a digest.

### The write-while-reading window

`nodes/admit.py` decodes at `:76-77` (`Audio(filepath=...)` then the forced `audio.waveform`) and
creates the input `stream` entity afterwards. A digest taken at entity-creation time is therefore a
*second, later* read of the file. If a regenerated source dataset overwrites the path in between,
the store would record a digest for bytes the pipeline never processed.

ADMIT now digests **before** the decode and **again after** it. If the two agree the digest is
provably of the decoded bytes and is recorded. If they differ the recording is failed through the
existing `_fail` path with `why` naming the cause; neither digest is recorded and nothing is raised.
This is the same class of finding ADMIT already makes about a file that will not decode, is
all-zero, or is constant.

**What this does not protect against.** senselab does not own the source tree, so there is no lock
that makes the read atomic. A file rewritten *and* rewritten back within the window, or rewritten
between the second digest and a later reader's use of the path, is undetectable. The guarantee is
detection, not prevention. `size_bytes` and `mtime_ns` are recorded alongside the digest for the
same reason: they are free from the `stat` we already need, and they let a later reader detect a
swap even where a digest is unavailable.

### When it cannot be computed

The store's existing convention (`Agent.unresolved_reason`) is to record *why* something is unknown
rather than record null or raise. `file_digest` returns `(digest, reason)` with exactly one set;
reasons are `"file not found"`, `"path is a directory"`, `"permission denied"` and
`"read failed: <OSError subclass>"`. `_check_entity_attributes` refuses, at write time and at
read-back alike, a digest and a reason together, a digest that is not 64 lowercase hex characters,
an empty reason, and either without a `path` naming the file it describes — mirroring
`_check_agent_fields`.

### Rejected: a first-class `Entity.checksum` field

Structurally stronger — the invariant would be on the dataclass rather than on a free-form
dictionary. Rejected because `_RECORD_KEYS["entity"]` is an exact key set: adding a field makes
`read_jsonl` reject every `store.jsonl` already on disk. Converting existing run directories is the
requirement this work exists to serve, so the checksum lives in `attributes` next to the `path` it
describes. A store written before this change reads back with no checksum keys — truthfully what it
has — rather than failing.

### Consequence: identity follows content

Entity ids digest the attributes (`prov_store.py:158`), so putting the checksum in the attributes
makes the id change when the file's bytes change. Two files at the same path with different content
are different entities. This is deliberate and desirable in a content-addressed store, and
`prov_store_checksum_test.py:test_entity_identity_follows_content` pins it.

Nothing depended on the id being stable across a content change, because it never was stable across
runs: `run_id` is mixed into every id (`prov_store.py:158`) and is `layout.root.name`
(`run.py:359`) — the run stem plus a UTC timestamp. Adding `mtime_ns`, which changes on every
write, therefore removes no stability that existed.

### Measured cost

On the battery run directory
`sub-0032892c…_task-Caterpillar-Passage_20260908-033704` (15 files, 18.00 MB, warm cache, 1 MB
chunks, best of 3):

| what | files | bytes | SHA-256 |
| --- | --- | --- | --- |
| whole run directory | 15 | 18.00 MB | **8.6 ms** (2103 MB/s) |
| `run/` only (what triage writes) | 13 | 16.96 MB | 8.3 ms |
| `store.jsonl` alone | 1 | 4.65 MB | 2.2 ms |
| the input recording, digested twice | 1 | ~1 MB × 2 | < 1 ms |

Against the 92–192 s PREPROCESS spends per recording this is 0.004–0.009%. The machine was under
other load, so these are upper bounds (see the memory note on resource measurements needing
isolation); a disk ten times slower still costs 86 ms. The decision was made independently of this
number — the measurement documents it rather than gating it.

## Addition 2 — a PROV-O JSON-LD serializer

`src/senselab/utils/prov_bep028.py`. A pure serializer over the store, not a second writer: the
store stays the single source of truth and cannot diverge from the graph. `convert_store_file`
takes a written `store.jsonl`, so an existing run directory converts without re-running anything.

### The mapping

| store | BEP028 | note |
| --- | --- | --- |
| `Entity` with a `path` | `Records.Files[]` | `AtLocation`, `Checksum`, `Type: ["sl:<prov_type>"]` |
| `Entity` without a `path` | `Records["prov:Entity"][]` | `prov:Entity` has no `AtLocation`, which decides the split |
| `Activity` | `Records.Activities[]` | `Label` is `<node>` or `<node> / <step>` |
| `Agent` (`software`) | `Records.Software[]` | `"senselab 1.3.1a45.dev542"` splits into `Label` + `Version` |
| `Agent` (`model`) | `Records.Software[]` | `Label` = `model_id`, `Version` = the resolved 40-hex commit |
| `wasGeneratedBy` `used` `wasAssociatedWith` `wasAttributedTo` `wasDerivedFrom` | `GeneratedBy` `Used` `AssociatedWith` `AttributedTo` `DerivedFrom` | all arrays |
| `wasInvalidatedBy` | `InvalidatedBy` | **no BEP028 term**; see below |
| `Entity.extent` | `ExtentStartSeconds` / `ExtentEndSeconds` | senselab vocabulary |
| `Entity.attributes` (remainder) | `Attributes` | nested, so no key can collide with a BEP028 term |
| `Activity.parameters` | `Parameters` | likewise |

### Identity

BEP028 wants IRIs, and specifies `bids:[<dataset>]:prov#entity-<label>` for a non-file
`prov:Entity` and `bids:[<dataset>]:prov#<label>-<uid>` for activities, software and environments.
Ours map so that **the fragment is the store id verbatim**, with `entity-` prepended for entities:

```
stream-c2654bfe83ec21d8   ->  bids::prov#entity-stream-c2654bfe83ec21d8
act-d4b1fa9a63c8a12a      ->  bids::prov#act-d4b1fa9a63c8a12a
agent-09fe01972fae0380    ->  bids::prov#agent-09fe01972fae0380
```

`store_id` reverses it by stripping `bids:<dataset>:prov#` and then a leading `entity-`. The
reversal is total and needs no knowledge of the record kind: entity ids always begin with a
`prov_type` name, none of which is `act` or `agent`. Human readability lives in `Label`, which is
where BEP028 puts it, rather than in the identifier.

Rejected: `bids::<relative path>` for the stream files, which is what BEP028 specifies for a **BIDS
file**. A triage run directory is not a BIDS dataset — `run/streams/plain.flac` is a sidecar of a
run, not a BIDS file — and a `bids::` path URI would assert dataset membership that does not hold.
The battery run directories happen to be named `sub-…_ses-…_task-…`, but that is the battery
harness's naming, not BIDS membership. The `dataset` argument exists for the day a run *is* placed
in a BIDS derivative dataset.

### The context

`@context` is an array: BEP028's own context object embedded verbatim, then a senselab extension.
Embedding rather than referencing the URL is deliberate — BEP028 is an unmerged PR with no stable
published IRI, and a graph that needs the network to mean anything is not a provenance record.
`BEP028_CONTEXT_SOURCE` names where the copy came from.

The extension adds exactly three things and changes no BEP028 term:

- `"InvalidatedBy": {"@id": "prov:wasInvalidatedBy", "@type": "@id"}` — PROV-O has the term, BEP028's
  context simply omits it.
- `"AtLocation": {"@id": "prov:atLocation"}` — BEP028's context defines `Atlocation` (lowercase L)
  while its schema and every example use `AtLocation`. Adding the cased spelling maps both to the
  same predicate rather than silently dropping the one the examples use.
- `"@vocab": "https://senselab.sensein.group/prov#"` — see below.

### `@vocab`, and why it is load-bearing

BEP028's model has no place for arbitrary per-entity attributes, and **JSON-LD silently drops any
term the context does not define**. Without a vocabulary, every measurement the store actually
holds — every span kind, every classifier score, every activity parameter — would vanish from the
graph while the file still parsed as JSON and still looked complete. `check_graph` therefore treats
a missing `@vocab` as a defect.

The consequence is an unbounded predicate vocabulary: the battery graph has **519 distinct
`sl:` predicates**, most of them AudioSet label names (`sl:Accordion`, `sl:Afrobeat`) and band
edges (`sl:0_200`) lifted out of `Attributes` maps. This is ugly RDF. The rejected alternative was
to serialize each attribute map as one opaque JSON string literal — bounded and tidy, but no longer
queryable, which defeats the purpose of emitting RDF at all. Losslessness won.

### Where BEP028 and the store disagree in kind

Not naming — these are things the specification cannot represent.

1. **`Command` is REQUIRED on every activity, and we have none.** Our activities are in-process node
   steps, not shell invocations. BEP028 offers `Command: null` to mean "performed manually", which
   is false for us. We emit `null` anyway, because in JSON-LD a null value **produces no triple** —
   so the required key is present for the schema and no false statement enters the graph — and add
   the `Description` that BEP028 recommends exactly when `Command` is null, plus
   `Type: ["sl:workflow-node"]`. This is the least-bad of three bad options; the honest fix is a
   BEP028 term for "computed in-process".
2. **`Software.Version` is REQUIRED and has no way to say "unknown, and here is why".** Two of the
   eight agents in the battery run have no commit: a TF-Hub URL pin, and torchaudio's bundled
   `SQUIM_OBJECTIVE` weights. We **omit** `Version` and emit `sl:VersionUnresolvedReason`. The
   record is then schema-invalid and truthful, rather than schema-valid and wrong. Recording a
   version while none is resolved is the one outcome worse than recording nothing.
3. **`Checksum` cannot say "unavailable, and here is why".** It is either present and true or absent
   and silent. For the 9 files whose digest cannot be taken we omit `Checksum` and emit
   `sl:ChecksumUnresolvedReason`.
4. **`wasAttributedTo` is in BEP028's context but is not a field of any BEP028 object.** Neither
   `Files.items` nor `ProvEntity.items` lists it. We emit `AttributedTo` — 761 triples in the
   battery graph — which is valid JSON-LD and valid PROV-O but not a schema-validated sidecar key.
   The aggregated graph, which BEP028 explicitly sanctions, is the only place it can go.
5. **Node/step containment is unrepresentable, because the store does not record it.** Our
   activities are created per node *and* per step (`preprocess.py` makes one per block, 40 in the
   battery run, 36 of them `PREPROCESS` steps) with no relation linking a step to its node.
   `prov:wasInformedBy` is the PROV-O term for it and BEP028 defines `InformedBy`, but inventing the
   edge at serialization time would be fabrication. The gap is in the store, not in BEP028.
6. **`AtLocation` is specified as a "relative path to the file on disk"; the input recording's is
   absolute.** `admit.py:95` records `str(source.resolve())`, which for the battery run is an
   `/orcd/data/...` path outside any dataset. We keep it absolute because it is true. A relative
   path would need a root the graph does not name.
7. **BEP028's `ChecksumAlgorithm` is not coerced to an IRI.** Its schema says `format: uri`, but the
   context maps it to `spdx:ChecksumAlgorithm` with no `"@type": "@id"`, so
   `"spdx:checksumAlgorithm_sha256"` expands to a plain string literal rather than a resource
   (confirmed with pyld). We do **not** override this: a local fix would make our graphs disagree
   with every other BEP028 graph. It is a defect to report upstream.
8. **`Environments` is emitted by nothing.** See below.

### Where the file goes

BEP028: `prov/[<subdir>/]prov-<label>_{act,soft,io,env}.json`. Triage: `prepare_run_layout`
(`run.py:130-166`) creates `<root>/run/` (with `streams/` and `derivatives/`), `<root>/released/`
and `<root>/summary/`, and the store at `<root>/run/store.jsonl`.

Reconciliation: a `prov/` directory at the **run root**, sibling to `run/` and `released/`, so that
when a run root is dropped into a BIDS derivative dataset `prov/` already sits where BEP028 puts
it. `write_bep028_files` writes the three non-empty BEP028-named files there; the aggregated
single-document graph is the same content under one `Records` object and is what a reader loads into
a triple store.

## Addition 3 — environment

Was: one string, `agent_type="software", version=f"senselab {version('senselab')}"`
(`nodes/common.py:68`, unchanged — see "Where it goes" below). Now: a fourth record kind,
`Environment` (`prov_store.py`), captured at run time by `run_triage` and serialized to BEP028's
`_env` suffix.

### What the brief got right, and one correction

The five-venv table in the commissioning brief was verified against this host and matches exactly:
`clearvoice-cpu` (torch 2.14.0), `crisperwhisper`/`crisperwhisper-cpu` (torch 2.13.0/2.14.0,
transformers 5.14.1/5.16.1), `qwen-asr`/`qwen-asr-cpu` (torch 2.8.0, transformers 4.57.6),
`yamnet` (tensorflow 2.21.0, keras 3.14.0, numpy 2.4.4), `hear` (tensorflow 2.21.0, keras 3.15.1,
numpy 2.4.6) — reproduced by `env_capture_measure_scan.py` in the scratchpad. One thing the brief
did not say: `preprocess.py:1923-1926` runs **both** ASR backends (`asr_crisperwhisper` and
`asr_qwen`) for consensus, not one or the other, so a single run's own environment picture is
`clearvoice(-cpu)` + `crisperwhisper(-cpu)` + `qwen-asr(-cpu)` + `yamnet` + `hear` — five venvs,
not a choice among them.

### Where it is captured

`ensure_venv` (`subprocess_venv.py`) is the single choke point every one of the five backends
already calls, and for a torch-bearing backend it calls `detect_host_cuda()` (a subprocess probe)
on *every* call, cache hit or not, because the resolved directory name needs the CUDA tag before
the cache marker can even be checked (`_ensure_venv_once`, `torch_specs = ...; if torch_specs:
detect_host_cuda()`, before `dir_name` is built). Capturing by calling `ensure_venv` again — a
tempting way to recover `venv_dir` in a node that only sees the backend's high-level result — would
double that probe cost on every recording. Instead, `subprocess_venv.record_venv_use()` opens a
`contextvars.ContextVar`-backed recorder; `ensure_venv` notes `(name, venv_dir)` into it, when one
is open, right after `_ensure_venv_once` returns — no second probe, no second lock acquisition.
`run_triage` (`run.py`) opens the recorder around ADMIT through VERDICT (every node that can reach a
subprocess venv sits inside that span) and calls `capture_environments(store, used_venvs)` once,
after the span closes and before `store.write_jsonl`. A run that fails at ADMIT or never reaches a
venv-backed node still gets a host environment; `used_venvs` is simply empty.

This is the same rule the checksum addition follows for the same reason: captured by the process
that did the work, never reconstructed later. `venv_environment()` (`subprocess_venv.py`) reads
`pyvenv.cfg`'s `version_info` line for the interpreter version and globs
`lib/python*/site-packages/*.dist-info` for installed distributions — pure filesystem work, no
interpreter start, no `uv pip freeze`. Measured on this host: 6.4 ms for the 26 provisioned venvs
(`env_capture_measure_scan.py`), the same order of magnitude as the brief's 35.6 ms on a different
host and cache state.

### Where it goes in the store

A **new record kind**, `environment` (`prov_store.py:Environment`, fields `id`, `kind` —
`"host"`/`"venv"` — `label`, `python_version`, `operating_system`, `dependencies`,
`dependencies_digest`, `senselab_version`), not a field added to `Agent`.

Rejected: widening `Agent`. `_RECORD_KEYS["agent"]` is an exact key set, checked on every line
`read_jsonl` reads (`prov_store.py:470`), so adding a field to it makes every `store.jsonl` already
on disk fail to read back — the same reason the checksum addition above chose `attributes` over a
first-class `Entity.checksum` field. A new record kind carries no such risk: an old store simply
never has a line whose `"record"` is `"environment"`, and the read loop's per-line dispatch
(`prov_store.py:466`) does not require every kind to appear.

Verified, not assumed: `env_capture_backcompat_check.py` ran `ProvStore.read_jsonl` over all
**388** `store.jsonl` files under `triage_battery_20260908/out/*/run/` with the new code — 388/388
parsed, and each read back with `store.environments() == []`, which is what a store predating this
change should report. `src/tests/utils/prov_store_test.py::TestEnvironments` pins the same
property on a synthetic store as a regression test.

The software agent itself (`nodes/common.py:software_agent`) is unchanged: same fields, same
`version=f"senselab {version('senselab')}"` string, same id. The host `Environment` is the fuller
record the brief asked for — python version, `platform.platform()`, senselab version — and is what
a reader should use for the host's environment; the software agent keeps its narrower job
(`wasAttributedTo`/`wasAssociatedWith` targets) because changing its fields is exactly the
`_RECORD_KEYS` hazard above.

### Serializing to `_env`

BEP028's `Environments` schema (`src/schema/objects/metadata.yaml` on `bclenet:BEP028_spec`, read
directly): `Id` and `Label` required; `AlternativeIdentifier`, `EnvironmentVariables`,
`OperatingSystem`, `Dependencies` optional. No `Type` field — unlike `Files`, `Activities` and
`Software`, an `Environments` object carries no BEP028-defined type slot. `prov_bep028.py:
_environment_record` maps `operating_system` → `OperatingSystem` and `dependencies` → `Dependencies`
directly; `python_version`, `dependencies_digest` and `senselab_version` have no BEP028 term, so
they go through `_term()` into the senselab vocabulary (`PythonVersion`, `DependenciesDigest`,
`SenselabVersion`), the same mechanism `SIZE_KEY`/`MTIME_KEY` already use on `Files` objects — the
`@vocab` catch-all (see Addition 2's own section on it) means they expand rather than vanish.

**Where BEP028 and the store disagree in kind, again — a ninth item for the list above.** The
schema's own `Activities.Used` field description says an activity's `Used` array's identifiers "MAY
name … environment(s) … described as specified in the Environments section" — so a graph *should*
say which activity ran in which environment. The store does not record that: `record_venv_use()`
is scoped to the whole run, not to one activity, because attributing a venv use to the one
PREPROCESS step (of the ~36 in a battery run) that actually called `transcribe_audios` would need
call-site changes across `preprocess.py` and `speech.py`, and the id of *which* step called which
API is not tracked today. Inventing that edge at serialization time would be exactly the kind of
fabrication item 5 above already refuses for node/step containment. `Environments` are therefore
emitted unlinked: present in the graph, named by no `Activities[].Used` entry — a gap in the store,
not a BEP028 limitation, and one this design does not close.

### Volume: full listing vs. a declared subset plus a digest

A full dist-info listing runs 36-93 packages per venv (measured: `hear` 36, `yamnet` 39,
`crisperwhisper` 43, `clearvoice-cpu` 62, `qwen-asr` 93). Storing all of it, per venv, per run,
across 44,786 recordings, is disproportionate to what any reader would query — nobody diffs two
runs' `six` or `charset_normalizer` versions. `venv_environment()` instead names
`_DECLARED_ENV_PACKAGES` in full — torch, torchaudio, torchcodec, transformers, tensorflow,
tensorflow-hub, keras, numpy, and each of the three backends with their own installed library
(`crisperwhisper`, `qwen-asr`/`qwen_asr`, `clearvoice`) — plus a SHA-256 over the *full* sorted
listing, so a mismatch against a fresh scan is detectable without paying to store the other
30-90 packages that did not change.

Measured (`env_capture_measure_store_cost.py`), one run's environment records (host + the five
venvs actually reachable from triage) as `store.jsonl` lines:

| | bytes / run | × 44,786 recordings |
| --- | --- | --- |
| full dist-info listing | 7,276 B | 325.9 MB |
| declared subset + digest | 2,136 B | 95.7 MB |
| difference | 5,140 B | 230.2 MB |

**Decision: declared subset + digest.** 230 MB is not free at battery scale, and the subset already
carries the packages the "three different torch versions inside triage alone" argument in the
brief is actually about. A reader who needs the excluded 90% re-scans the venv directly (still on
disk, still 6-36 ms) and compares its own digest against `DependenciesDigest`; nothing about the
excluded packages is lost, only stored redundantly.

### Tests

`src/tests/utils/prov_store_test.py::TestEnvironments` — the record kind's write/read/merge/
fingerprint contract, an unknown `kind` refused on read, and the 388-store backward-compatibility
property reproduced as a unit test. `src/tests/utils/subprocess_venv_test.py::TestRecordVenvUse`,
`::TestVenvEnvironment` — the recorder (including the "outside the context manager, nothing is
recorded, no error" case) and the dist-info/`pyvenv.cfg` parsing, against a synthetic on-disk venv
tree. `src/tests/utils/prov_bep028_test.py::test_a_recorded_environment_reaches_the_graph`,
`::test_write_bep028_files_emits_the_env_suffix` — the `_env` mapping and file split.
`src/tests/audio/workflows/triage/nodes/common_test.py::TestHostEnvironment`,
`::TestCaptureEnvironments` — the host record's fields and the host+venv aggregation.
`src/tests/audio/workflows/triage/run_test.py::test_the_store_carries_a_host_environment` — a full
(fake-graph) `run_triage()` call writes exactly one host environment, end to end.

### Artefacts

`env_capture_demo_graph.jsonld` and `prov/prov-demo_env.json` (scratchpad): a live capture — the
real host plus the five real on-disk venvs this host has provisioned for triage's backends,
`capture_environments` run for real, not reconstructed from a converted store. `check_graph` on the
aggregated document: zero problems. Converting an actual pre-existing run directory
(`sub-0032892c…_task-Cape-V-sentences-(v2)-1_20260908-025853/run/store.jsonl`) still emits no
`Environments` key at all (`env_capture_convert_real_run.py`) — correct, since that run predates
this change and never called `capture_environments`; inventing one at conversion time is exactly
what this design refuses to do.

## Validation of the emitted graph

Structural, in `check_graph` (no dependency): the context is BEP028's plus a `@vocab`, every
`Records` key is a context term, no `Id` appears twice, every `Id` reverses to a store id, every
relation names an `Id` the document declares, every `ChecksumValue` is 64 lowercase hex under the
SPDX SHA-256 term, and no key would be dropped by JSON-LD. Zero problems on both artefacts.

Semantic, with ephemeral tools (`uv run --with …`, added to no project manifest):

- **pyld 3.3.0** (the reference implementation) expands the battery graph to **78,681 triples**
  across 528 predicates in 1.1 s. The six PROV-O relation counts —
  761 `wasGeneratedBy` + 761 `wasAttributedTo` + 862 `used` + 40 `wasAssociatedWith`
  + 1242 `wasDerivedFrom` = **3,666** — equal the store's relation count exactly. Nothing was lost
  and nothing was added.
- **rdflib 7.6.0** parses the same file to **2 triples**. This is an rdflib limitation, not a defect
  in the document: BEP028's `"Records": {"@container": "@type", "@id": "@graph"}` is a JSON-LD 1.1
  type-container, pyld accepts the term definition and expands it correctly, and rdflib's JSON-LD
  parser does not implement it. Anyone loading a BEP028 aggregated graph with rdflib will get an
  empty graph and no error. Worth reporting upstream, and worth knowing before choosing a reader.

## Artefacts

| file | source | size | triples |
| --- | --- | --- | --- |
| `prov-triage_graph.jsonld` | `…_task-Caterpillar-Passage_20260908-033704/run/store.jsonl`, converted in 0.13 s | 6,130,944 B | 78,681 |
| `prov/prov-triage_{io,act,soft}.json` | the same graph, split BEP028-style | 5,774,631 / 69,343 / 3,671 B | — |
| `prov-admit_graph.jsonld` | a live `admit()` over `run/streams/plain.flac`, showing a real `Checksum` | 4,529 B | 41 |

The battery graph carries **0 of 21** `Files` with a `Checksum`: that run executed before this
change, so it recorded no digests, and retro-computing them at conversion time is exactly what the
write-time rule forbids. The second artefact exists because of that — a real ADMIT over a real file,
whose stream entity carries
`checksum_sha256=cbb4a15c0a5d1458cfd9e11611b28acda6a3118ed50172d2dadf7faedcfa0fc7`,
`size_bytes=1069155`, `mtime_ns=1788838624000000000`, emitted as
`{"ChecksumAlgorithm": "spdx:checksumAlgorithm_sha256", "ChecksumValue": "cbb4a15c…"}`.
