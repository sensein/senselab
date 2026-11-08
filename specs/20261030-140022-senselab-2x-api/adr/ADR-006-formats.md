# ADR-006: Formats, versioning and evolution

**Status:** accepted 2026-10-30; amended 2026-10-31, 2026-11-01.

## Context
Architect review §5 (R-42) found:
- three serialisers in use;
- no canonical serialisation;
- no format version on plans, profiles, provenance or fitted artifacts;
- "a new schema directory per bump" as the only evolution path, which means a full corpus
  recompute.

## Decision
- **Canonical serialisation,** used wherever bytes are hashed:
  - JSON: RFC 8785 (JCS);
  - arrays: little-endian `.npy` version 1.0, C order;
  - tables: Parquet with a fixed writer configuration;
  - text: UTF-8 NFC.

  Output identity (ADR-002) is the hash of this form.
- **Every artifact carries `format: <name>@<major>.<minor>`.** That covers results, logical plans,
  placements, profiles, provenance records, fitted artifacts, store indexes and graph documents.
  - A minor change is additive, and readers ignore unknown fields.
  - A major change requires a migration.
- **Readers** support the current and the previous major version.
- **Migrations are declared tasks** (`migrate:<name>@<from>→<to>`). They are run lazily on read, or
  in bulk as a pipeline, and recorded in provenance.
- **A schema bump no longer implies a wipe.** The store keeps entries whose format is readable. A
  wipe is reserved for an identity change (ADR-002), which creates a new key space.

## Consequences
- 2.x releases can evolve result models without recomputing corpora.
- Every reader needs a migration test from the previous major version.

## Alternatives rejected
- **Wiping on every bump** (1.x `CACHE_SCHEMA_VERSION`). Acceptable for a cache, not for results
  and fitted artifacts that people rely on.

## Amendment 2026-10-31: canonical values and digests (made 2026-10-30 and 2026-10-31)

Adds to the Decision's canonical serialisation; the text rule it first carried is replaced by the
2026-11-01 amendment. Resolves R-105 and R-106 ([`review-register.md`](../review-register.md)); the
findings are in [`reviews/phase1-reattack/canon.md`](../reviews/phase1-reattack/canon.md).

- **The canonical encoders accept an exact-type whitelist.** JSON scalars are exactly `None`, `bool`,
  `int`, `float` and `str`, or a numpy scalar of dtype kind `b i u f U`. Containers are exactly `list`,
  `tuple` and `dict` with `str` keys. Arrays are `numpy.ndarray` of kind `b i u f U`. Everything else
  raises: subclasses, `Enum` members, datetimes, complex, masked arrays. Durable identities never name a
  Python class.
- **One walk, one bound.** Every canonical encoder walks containers through one shared guard: a depth limit
  of 64 containers, a memo for shared containers, and a work budget of 64 Mi characters of JCS text, four
  times the wire's 16 MiB line, so any value a worker can return encodes. A shared DAG (`x = [x, x]` 63
  times) was expanded as a tree (XC-5); under the bound it is refused in about 0.2 s.
- **`jcs_loads` accepts only what `jcs_dumps` writes** (`jcs_dumps(jcs_loads(b)) == b`): a repeated key,
  `NaN`, an overflowing number or any non-canonical spelling is refused (XC-6).
- **A kernel value's canonical form (`value@2`)** is `{"type": <name>, "fields": <fields>}`, each field as
  the value holds it. Every family outside JSON (array, set, path, type, held value, out-of-range integer,
  non-finite float) gets its own one-key token, and a held key starting with `$` gains one more `$`, so no
  two families share an encoding (XC-2: `value@1` compared an array equal to its hash).
- **Canonical equivalences.** This is the one list. Values that differ in Python but are one value here
  share every digest:
  - numbers: `1`, `1.0`, numpy `1`, and `-0.0`, `0.0`, `0` compare by value; arrays write `-0.0` as `0.0`;
  - NaN: every NaN is one value, written as the positive quiet NaN with no payload. `inf - inf` gives
    different sign bits on arm64 and x86 (XC-4), so without this one computation could hash two ways;
  - bools: every nonzero byte of a bool array is `True`, written as 1;
  - text: equal after the canonical text form (2026-11-01 amendment); two keys of one object with one form
    are refused, not merged;
  - containers: a list and a tuple; a set and a frozenset; two set elements with one JCS form are one
    element (XC-7); in `value@2`, any mapping and a `dict` with the same items;
  - paths: one `as_posix()` text;
  - arrays: equal in dtype, shape and canonical values whatever their byte order and layout; a 0-d array
    is never equal to a one-element array;
  - bytes: `bytes` and `bytearray` of the same content.

  The worker wire is transport, not identity: it keeps every value's bits. The result store is identity:
  it writes canonical forms, so a cached output reads back canonical where a fresh one keeps what the
  backend returned.
- **Every digest names a registered tag and states what it hashes; no other digest function exists.**
  There are three, typed by source:
  - `identity_digest(tag, value)` = `sha256(tag || 0x00 || JCS(value))`. No tag is a prefix of another. A
    new kind of identity registers a new tag; changing what an identity covers bumps its tag's major.
  - `file_digest(tag, path)` = the plain sha256 of a file on disk.
  - `npy_digest(tag, array)` = the plain sha256 of the array's canonical `.npy` bytes.

  A file or npy digest's tag is checked, not hashed, so its value equals the published sha256 of the
  content. Hashing serialised data is not reachable through the public API. The import-linter contract
  `digests-are-tagged` forbids hash modules everywhere except the digest module, and `raw-digest-owners`
  names the few owners allowed to hash bytes already in memory.

  **Registered identity tags:** `logical-key@7`, `output@3`, `context@3`, `content-root@2`, `audio@4`, `audio-file@1`,
  `weights@3`, `value@2`, `provenance-id@4`, `data-manifest@3`, `worker-runtime@1`, `fake-output@1`,
  `contract-probe@1`, `prov-activity@1`, `prov-node@1`, `wheel-set@1`, `env-lock@1`, `env-overlay@1`, `launch-env@1`, `vllm-config@1`, `source@1`. **Registered blob tags:** `file@1`, `npy@1`. **Data formats** (a
  `format:` line in a packaged YAML file or in a store entry): `equivalence@2.0`, `equivalence-evidence@1.0`, `retry-policy@1.0`, `catalogue@1.0`, `spdx-licenses@1.0`, and
  `execution-context@2.0` in a store entry's `context.json`; `env-record@1.0`, an env root's `<name>[-<tag>].json`; `senselab-vocab@1.1`, a vocabulary document's
  `owl:versionInfo` (ADR-010).
- **The store is `store@4` and an entry's `result.json` is `result@2.0`,** which adds the fingerprint of the
  context it was written in (R-101, ADR-003). A `result@1.0` entry cannot show which context it belongs to,
  so none is read: `store@4` is a new key space.

## Amendment 2026-11-01: canonical text, readers within 2.x, release versions

Owner decisions D-3, D-4 and D-5 ([`decisions-2026-11-01.md`](../decisions-2026-11-01.md)), and rulings
H-R2, H-R43, H-R47, H-R48 and H-R50 ([`rulings.md`](../rulings.md)).

- **Canonical text: NFC, except text with code points unassigned in Unicode 13.0** (D-3, ruling H-R34).
  Replaces "text: UTF-8 NFC" in the Decision.
  - Text whose code points are all assigned in Unicode 13.0 is put in NFC. Text holding any other code
    point is kept exactly as given and never normalised. It is never refused; only a lone surrogate, which
    is not Unicode, raises.
  - **Why: portability.** NFC is computed from the interpreter's Unicode database, which differs between
    supported Pythons (3.10 has 13.0, 3.12 has 15.0). Unicode's stability policy fixes the NFC of strings
    whose code points are assigned in a given version, but a code point assigned later can change NFC as a
    combining mark: `"a\U0001E08F̣"` normalises one way on 3.10 and another on 3.12 (XC-1, reproduced).
    Plain NFC would give one value different digests on different supported interpreters.
  - **Cost:** two post-13.0 strings equal under a later NFC have two digests. Each digest is the same on
    every interpreter.
  - The allowed code points are a checked-in table generated from Unicode 13.0.0. CI checks that 3.10, 3.12
    and 3.14 normalise every code point in it identically.
  - **Superseded:** H-R33, which refused such text, and the triage premise first given for keeping it (emoji
    in LLM reviews). The reason is portability.
- **Readers support the previous major within 2.x** (H-R50). A 1.x format is not a previous major of a
  2.x one: 2.x reads no 1.x artifact (P-14). Before the first 2.x release a major bump starts a new key
  space and carries no migration (H-R2); the Decision's previous-major readers and migration tests apply
  from the first 2.x release.
- **2.x shares no mutable state with 1.x** (H-R50). Run manifests live under `$SENSELAB_CACHE/2x/`, and
  the result store and data cache hold 2.x formats only, so one cache root can serve both lines.
- **Release versions and tags** (D-5, H-R43). A version is PEP 440 canonical (`X.Y.Z`, `X.Y.ZaN`,
  `X.Y.ZbN`, `X.Y.ZrcN`). A tag is that string or the repository's spelling `X.Y.Z-(alpha|beta|rc).N`, and
  `Version(tag)` must equal the built version exactly. Epoch, local, post and dev segments and a `v` prefix
  are refused. The three distributions pin each other exactly. Until the first 2.x release the release
  gate checks only this grammar, the exact sibling pins, and that each distribution builds and installs
  from `dist/` (design §5).
- **Provenance format: superseded by [ADR-010](ADR-010-w3c-prov.md)** (D-4), implemented in Phase 2. Until
  then:
  - a provenance file is `provenance@2.0`, whose first line is its format record; a record's id is
    `identity_digest("provenance-id@4", ...)` over its content, and `run_id` is required;
  - the reader reads exactly what the writer writes (H-R47): a file without the `provenance@2.0` record is
    refused as a 1.x store, and there is no converter;
  - the entity vocabulary is generic, `artifact` alone (H-R48); a use case adds its types when it ports;
  - the value token families that came from 1.x provenance (`$path`, `$enum`, `$dec`, `$bytes`, `$set`)
    are removed when ADR-010 is implemented.

## Amendment 2026-11-01: retired formats are refused; the 1.x value tokens are removed

Ruling P2-44 ([`plans/phase-2.md`](../plans/phase-2.md)) and ADR-010 as amended; implemented in Phase 2 Task 3c.

- **Before the first 2.x release a major bump has no previous-major reader** (H-R2, P2-44). This replaces
  "Readers support the current and the previous major version" and "Every reader needs a migration test from
  the previous major version" until the first 2.x release. A file of a retired format is refused with a typed
  error that names the format and the migration statement. No reader, converter or alias is written for it
  (P-14). From the first 2.x release the Decision's previous-major readers and migration tests apply.
- **`provenance@3.0`** is the format of a provenance document (ADR-010). It is the document's top-level
  `format`, beside `@context` and `@graph`. The published context maps `format` to null, so a JSON-LD
  processor ignores it. `provenance@2.0`, its JSONL reader and `provenance-id@4` are retired. A file in a
  run's `prov/` that declares any other format is refused with `ProvenanceFormatRefused`.
- **`value@2` has no set, enum, path, bytes or `Decimal` family.**
  - In the `value@2` bullet, "(array, set, path, type, held value, out-of-range integer, non-finite float)"
    becomes "(array, type, held value, out-of-range integer, non-finite float)".
  - *Canonical equivalences* loses three entries: "a set and a frozenset; two set elements with one JCS form
    are one element", "paths: one `as_posix()` text" and "bytes: `bytes` and `bytearray` of the same content".
  - The tag stays `value@2`: every value the encoder still accepts encodes byte-identically (checked on 30
    representative values). Three in-memory digests do change: `BackendSpec` and a `RegularGroup` with a
    `source` (posix text, sorted tuple), and the equivalence digest. None is persisted; each feeds only equality,
    hashing or the in-memory verified key.
- **`FrozenMapping` is the one frozen mapping type.** Removing `FrozenDict` leaves a built `Frozen` value's
  `__dict__` a plain dict: a direct `__dict__` write is not refused, though attribute writes, private-state writes
  and `__setstate__` are. Such a write does not change equality, hash or `canonical_digest`.
  - A value of these families is refused (`NoCanonicalForm`). A kernel value holds a path as its posix text
    (`RegularGroup.source`) and a set as a sorted tuple (`BackendSpec.output_affecting`).
  - The tag stays `value@2`. The encoding of every value the encoder still accepts is unchanged; the change
    is a refusal. The two kernel fields whose type changed enter no persisted address: a recording's identity
    stands a source group as its content root, not its path.

## Amendment 2026-11-03: index-free tolerance paths; observations name their flatten

Phase 2 Task 22f.

- **`tolerance@2.0`.** A tolerance profile's field `path` is index-free: every `[n]` of a flattened output path is
  `[*]` (`runtime.tolerance.index_free`). A field's spreads pool every index of its path, and a comparison judges
  each flattened path by the field of its index-free form, so an output with more turns or segments than any
  calibration item is covered. A path with no field is a disagreement. A `tolerance@1.0` profile (indexed paths, or
  a whole label array as one path) is refused with `ToleranceProfileRetired`, naming the regeneration.
- **`tolerance-observations@2.0`** carries `flatten`, the version of the leaf flattening that wrote it
  (`flatten@2`: numbers, None and numeric arrays as float lists with None where a value is absent, NaN read as
  None; a non-numeric array element by element; anything else as its JSON). `generate` refuses an observation file
  of another format or flatten, naming the file; nothing is upgraded. `tolerance-observations@1.0` files, written
  by `flatten@1` (a non-numeric array as its truncated repr at one path), are all re-measured.

## Amendment log

- 2026-10-30: canonical encoders take an exact-type whitelist; tagged digests; `value@1`.
- 2026-10-31: `value@2` per-family tokens; canonical `.npy` values; one walk and bound (R-105; H9d).
- 2026-10-31: tags bumped for canonical `.npy` and raw post-13.0 text: `output@3`, `audio@3`, `value@2`,
  `logical-key@3`, `context@3`, `weights@3`, `provenance-id@3`, `data-manifest@3` (H-R35; H9d).
- 2026-10-31: canonical text refuses post-13.0 code points (H-R33), then keeps them raw (H-R34).
  Restated 2026-11-01 (D-3).
- 2026-10-31: three typed digest functions; `worker-runtime@1`, `fake-output@1`, `contract-probe@1`
  registered (R-106, H-R36; H9e).
- 2026-10-31: `provenance@2.0` and `provenance-id@4` (R-104; H9c).
- 2026-10-31: `store@4`, `result@2.0` with the context fingerprint (R-101; H9b).
- 2026-11-01: provenance reader reads only `provenance@2.0`; generic vocabulary (H-R47, H-R48).
- 2026-11-01: readers within 2.x; 2.x namespace under the cache root (H-R50).
- 2026-11-01: canonical text restated with its portability reason (D-3); version and tag grammar (D-5);
  provenance sections superseded by ADR-010 (D-4).
- 2026-11-01: `logical-key@3` → `@4`: a `url` or `local` model is keyed by its weights' digest, not its path or
  URL (F1, A I-1). `@4` → `@5`: the model's source leaves the key and is recorded in the execution context
  (F3, H-R56). Both bumped rather than folded, so no earlier key addresses an entry.
- 2026-11-01: `execution-context@1.0` → `@2.0`: `model_source` is recorded and not fingerprinted (F3, H-R56).
- 2026-11-01: new data formats `equivalence@1` (`runtime/data/equivalence.yaml`) and `retry-policy@1.0`
  (`runtime/data/retry_policy.yaml`), each entry carrying its derivation (F1, F4).
- 2026-11-01: new data formats `catalogue@1.0` (`senselab/data/catalogue.yaml`, written `catalogue@1` in the plan; the tag
  pattern needs a minor) and `spdx-licenses@1.0` (`runtime/data/spdx_licenses.yaml`, one derivation per group) (Task 7).
- 2026-11-01: identity tags `prov-activity@1` (an activity's compact PROV-O node without `@id`) and `prov-node@1` (an
  agent's or environment's), and the data format `senselab-vocab@1.0` (`runtime/data/prov/senselab.jsonld`), for
  ADR-010 (Phase 2 Task 3a; rulings P2-5, P2-6, P2-7).
- 2026-11-01: retired formats refused before the first 2.x release (P2-44); data format `provenance@3.0`;
  `provenance@2.0` and `provenance-id@4` retired; `value@2` loses the set, enum, path, bytes and `Decimal`
  families (Phase 2 Task 3c).

## Amendment 2026-11-01: the label map's format

- 2026-11-01: new data format `labels@1.0`, the run's private label map `<run>/private/labels@1.json`
  (ADR-007 amendment, Phase 2 Task 4); a reader reads any minor of `labels@1`.

## Amendment 2026-11-01: backend params in the logical key

- 2026-11-01: `logical-key@5` → `@6`: the key gains `backend_params` (ADR-002 amendment, Phase 2 Task 5b).
- 2026-11-07: `logical-key@6` → `@7`: the key gains `service`, a server backend's attestation (I-8; Phase 3 Task 17).
  New data format `service-entry@1.0`, a service's registry entry `{identity, endpoint}` (ADR-003).

## Amendment 2026-11-01: env builds

- 2026-11-01: identity tag `wheel-set@1`, the sorted `[name, version]` pairs of an env's `*.dist-info` names
  (`EnvHandle.wheel_set`, the context's `wheel_set`); replaces 1.x `venv_environment`'s untagged
  `dependencies_digest` (H-R20). Data format `env-record@1.0`: the current build of one env under its root,
  `{directory, lock_sha, torch_index, python_version, wheel_set}`, replaced by rename under the env's lease
  (Phase 2 Task 9a1).
- 2026-11-01: identity tag `env-lock@1`, the digest of everything a lock is compiled from: the backend's requirement
  specs, the base requirements, the quarantine constraints, the pinned `exclude_newer`, the Python version, the CUDA
  ceiling and the excluded distributions. It replaces the untagged `requirements_sha256`; the 1.x format has no
  reader, and locks are recompiled. A lock's header is `# senselab env lock: <name>`, `# identity-digest: <tag> <64 hex>`,
  `# python:`, `# torch:`, `# torch-index:`, `# exclude-newer:`, `# uv:`, then an optional `# exclude:` and any `# note:`
  lines. Data format `env-lock-inputs@1.0`: `data/env_locks/inputs.yaml`, each value with its derivation.
- 2026-11-01: identity tag `vllm-config@1`, the ordered `vllm serve` arguments of a server's settings (each
  `--flag=value` or switch); the server's attestation carries it as `config_digest`, with the reported engine version
  and the model identity, and the execution context records all three in `driver` (I-8). It replaces 1.x
  `engine_identity` (Phase 2 Task 9b).
- 2026-11-01: identity tag `source@1`, the digest of a declaration's normalised source (docstrings and the `version`
  argument left out) and of its worker module's, taken by `scripts/check_source_digests.py`. Data format
  `source-digests@1`: `data/source_digests.yaml`, written at a release, mapping `task:<name>` and
  `backend:<task>/<name>` to `version` and `digest`; before the first release the baseline is the tree at the
  merge-base with `2.x` (P2-54, Phase 2 Task 19).
- 2026-11-02: identity tag `env-overlay@1`, the digest of an env's overlay: the worker runtime identity and each
  worker module's `[dotted name, file@1 digest]`, sorted. A built env is named by it beside its lock and torch index,
  so an env is never changed once published; another overlay is another build (R-179, Phase 2 Task 34).
- 2026-11-02: identity tag `launch-env@1`, the digest of how a worker or server process is started: its ambient
  variables of a recorded family with its declared variables, sorted `[name, value]` pairs. The context records it as
  `driver.launch_env`. Data format `launch-environment@1.0`: `data/launch_environment.yaml`, the families of ambient
  variables a launched process inherits (linker, TLS, proxy, Hugging Face, GPU, threads, locale, process; GPU and
  threads recorded) and the `never` family (`PYTHON*`, `VIRTUAL_ENV`, `CONDA_*`), each with its derivation. A server's
  attestation adds the observed `served_model` and names its observed entries in `driver.observed` (R-178, Phase 2
  Task 34).

## Amendment 2026-11-01: kernel types of the Phase 2 tasks and their wire forms

- 2026-11-01: data formats `text@1.0` (`PlainText`: canonical text and its content identity, `content:` +
  `value@2` over type token `text`), `transcript@1.0` (a tree of nodes `{text, span?, confidence?, children}`, every
  span integer ns on one clock), `text-spans@1.0` (columnar code-point spans `[start, end)` over a text's content
  identity and length, with label and optional score), `trials@1.0` (row-paired embedding matrices `first`, `second`
  in one `space`, with a bool `same` column) and `metric@1.0` (`{name, value, unit, items}`, `items >= 1`). Task
  ports may also name the ADR-005 protocols `RegularSeries` (implemented by `RegularGroup`, `Audio`) and `Text`
  (implemented by `PlainText`) (Phase 2 Task 11a).
- 2026-11-01: **wire forms.** Every kernel value a port may carry has a wire form: its `model_dump()` with `format`
  set to its tag, arrays sent as worker-wire array references. A `Formatted` type's tag is its format; the signal
  types register `recording@1.0`, `audio@1.0` and `regular-group@1.0` (`kernel.wire.WIRE_FORMATS`). A file-backed
  regular group travels as its samples decoded on the host together with the file's source stamp, so no path
  reaches a worker and the decoder recorded in the context is the one that produced them. A regular group may hold
  samples with the stamp of the file they were decoded from (their shape must match the stamp's channels and
  frames); it is then identified as the file is, by the stamp's content root. A wire form is read by the registered
  type of its tag's name and major version, keeping its `identity` and `anchor` (the host is trusted), so a
  file-backed recording reads back with the file's identity, anchor and samples; only the path is gone. `run_call` sends isolated workers wire forms and host workers the values
  themselves. Outputs are symmetric: an output on a port declared as a kernel type is stored as its wire form (a
  host backend's kernel value is converted) and returned to the caller, fresh or from the store, as the declared
  type read by `from_wire_form`; an output whose tag reads as another type is refused (`OutputTypeMismatch`) before
  it is stored. A `str` passed for a `Text` port is held as `PlainText` (Phase 2 Task 11a).
- 2026-11-01: a source stamp carries `chunk_size` and `chunks`, the per-chunk digests its `content` root is taken
  over (ADR-005 amendment); the signal wire formats become `recording@2.0`, `audio@2.0` and `regular-group@2.0`; an
  `@1` form is refused with `WireFormatRetired`, naming its format and the migration statement (P2-44) (Phase 2 Task
  11b).
- 2026-11-02: `equivalence@1` → `equivalence@2.0` (`runtime/data/equivalence.yaml`, classes of numerics classes per
  backend; `@1` refused with `EquivalenceFormatRetired`); new data format `equivalence-evidence@1.0`, one verify
  comparison per file under a result store's `equivalence-evidence@1/`, named by its canonical digest;
  `senselab-vocab@1.0` → `@1.1`, adding `senselab:verifyFraction`, `senselab:verifySampling` and
  `senselab:verifies` (Phase 2 Task 21).
- 2026-11-03: `tolerance@1.0` → `tolerance@2.0` (index-free field paths); `tolerance-observations@1.0` → `@2.0`
  carrying `flatten@2`; stale observations refused by name (Phase 2 Task 22f).

## Amendment 2026-11-02: exit adversarial findings, group A (Phase 2 Task 34a)

- `audio@3` → `audio@4`: a held audio's identity covers its channels (`ChannelInfo` of each channel) beside its
  samples and rate, so two audios differing only in channel names or units no longer share an identity (R-181). A
  stamped audio's identity stays its file's content root (ADR-005 amendment 2026-11-01).
- `recording@2.0`, `audio@2.0` and `regular-group@2.0` → `@2.1`: a regular group gains `decoded`, the `npy@1` digest
  of the samples a reader decoded from its stamped file. Held samples carry a file's stamp only as
  `RegularGroup.decoded_from` builds them, and must still have that digest; any other held group with a stamp is
  refused, so a transform that copies a decoded group with new samples cannot keep the file's identity (R-171). The
  change is additive: an `@2.0` form reads as before, except a held group with a stamp and no `decoded`, which only the
  host's wire wrote and which is now refused. `decoded` stays out of a recording's identity.
- **One format gate** (R-172): `kernel.formats.check(written, found, retired)` returns `accepted` (the written
  name and major, any minor), `retired` (a `<name>@<major>` in the reader's retired set), `unknown` (any other tag,
  such as a later major) or `malformed` (not a tag). The store, the wire and the label map read through it. In the
  result store only `malformed` is corruption: a retired member raises `StoreFormatRetired`, naming the migration
  statement, and an `unknown` one, or a later minor of `execution-context@2` with fields this version does not know
  (the fingerprint covers every field, so they cannot be ignored), raises `StoreFormatUnknown`; `get` skips the
  latter, and neither is quarantined. `result@1` and `execution-context@1` are the retired store formats. Provenance
  is not yet on the gate (R-190, Task 34b's file).
- `tolerance@2.0` → `@2.1`: a profile's derivation states its `coverage` (R-186). A field's tolerance, the largest
  spread over n calibration items, is a distribution-free one-sided upper tolerance bound covering a fraction
  `attained = (1 - confidence)^(1/n)` of items with probability `confidence`; the profile records the packaged
  `target` and `confidence` and `required_items = ceil(ln(1 - confidence) / ln(target))`, and validation refuses a
  stated coverage its item count does not give. New data format `tolerance-coverage@1.0`
  (`runtime/data/tolerance_coverage.yaml`: target 0.95, confidence 0.95, so 59 items), each value with its derivation.
  An `@2.0` profile reads as before, with no stated coverage. The 12-item `speech_to_text/transformers` and
  `diarize/pyannote` profiles state 0.779 when regenerated; reaching the target is a re-measurement.
- 2026-11-02: `equivalence@2.0` → `@3.0` (members are a numerics class and an env lock; `@2.0` refused);
  `equivalence-evidence@1.0` → `@2.0` (each side's env lock), under `equivalence-evidence@2/`;
  `execution-context@2.0` → `@3.0` (`env_license`); `store@4` → `store@5`; `senselab-vocab@1.1` → `@1.2`
  (`senselab:envLock`, `senselab:license`, `senselab:storedAgent`); `execution-context@2` joins the retired store
  formats (P2-92, P2-93; Phase 2 Task 34d).
- 2026-11-02: new data formats `comparison@1.0` (`runtime/data/comparison.yaml`, how each kernel type's fields
  compare) and `run-config@1.0` (`runtime/data/run_config.yaml`, the named comparison thresholds, each with its
  derivation); `equivalence-evidence@2.0` → `@3.0` (`thresholds`, a digest, replaces `tolerance`), under
  `equivalence-evidence@3/`; `parity-reference@2.0` → `@3.0` (each 1.x projection carries its kernel value's
  `format` leaf); new `parity-comparison@1.0` (`parity_reference.py compare --config`: one parity decision, its
per-item differences and the digest of the thresholds it was made under); `senselab-vocab@1.2` → `@1.3`
(`senselab:verifyThresholds`). Retired with nothing to read them:
  `tolerance@2` (and `@1`), `tolerance-observations@2` (and `@1`) and `tolerance-coverage@1` (P2-95; Phase 2 Task
  34e).

## Amendment 2026-11-02: the data formats list (P2-95; Phase 2 Task 34e)

Restates the **Data formats** list of the 2026-10-31 amendment, which this supersedes: `equivalence@3.0`,
`equivalence-evidence@3.0`, `retry-policy@1.0`, `catalogue@1.0`, `spdx-licenses@1.0`, `comparison@1.0`,
`run-config@1.0`, and `execution-context@3.0` in a store entry's `context.json`; `env-record@1.0`, an env root's
`<name>[-<tag>].json`; `senselab-vocab@1.3`, a vocabulary document's `owl:versionInfo` (ADR-010). The tolerance
profile formats (`tolerance@`, `tolerance-observations@`, `tolerance-coverage@`) are retired and not listed.

## Amendment 2026-11-03: core carry-overs (Phase 3 Task 2)

- `recording@2.1`, `audio@2.1` and `regular-group@2.1` → `@3.0`: a source stamp's `channels` is the channel layout
  (each `ChannelInfo`) the stamp was taken under, not a count; non-additive, so a major. `@2` is retired.
- New identity tag `audio-file@1`: a file-backed audio's identity is `"content:" + identity_digest("audio-file@1",
  {content: <root>, channels: <layout>})` (R-201). Every file-backed audio therefore takes a new identity, and so a
  new clock id and new cache keys, not only a relabelled one. A `Recording` of one stamped group no longer takes the
  file's content root: it takes the groups digest (`value_digest(IDENTITY_TYPE, {"groups": ...})`).
- Supersedes the 2026-11-02 amendment's "a stamped audio's identity stays its file's content root", and the 2026-10-31
  line that a held group with the file's stamp "is identified as the file is, by the stamp's content root": it is
  identified by `audio-file@1` over that root and the layout.
- `labels@1.0` → `labels@2.0`: the run's private label map is canonical JSON (JCS), read strictly. `labels@1` is
  retired and refused as such.
- `comparison@1.0` keys `recording@2`, `audio@2` and `regular-group@2` become `@3`.

## Amendment 2026-11-03: benchmark carry-overs (Phase 3 Task 3)

- `metric@1.0` → `metric@2.0`: `Metric` adds `numerator` and `denominator` (finite floats, the denominator not
  negative, both set or both absent; a zero denominator has a zero numerator) beside `value`; non-additive for a reader
  that aggregates, so a major (P3-10). `metric@1` is retired: `from_wire_form` refuses it with `WireFormatRetired`.
  `comparison@1.0` keys `metric@1` becomes `metric@2`. `Metric` also carries an optional `empty_value`, the metric's
  declared value for nothing to
  score, allowed only with the ratio parts. `span_f1` of two empty span sets is `(0, 0)` with value and `empty_value`
  1.0, so it adds nothing to a pooled ratio.
- `benchmark-result@1.0` → `benchmark-result@2.0`: a row carries `numerator` and `denominator`. A split's aggregate is
  the sum of its rows' numerators over the sum of their denominators; when that denominator is 0 the aggregate is
  absent (None) unless the metric
  declares an `empty_value` (P3-55). A split whose
  rows carry no ratio parts (EER, a pooled metric, `resample_error`) aggregates as the `items`-weighted mean of its
  values; a split mixing the two is refused with `MixedRatios`. `benchmark-result@1` is retired and refused with
  `BenchmarkResultRetired`.
- Data formats list (2026-11-02 amendment): adds `metric@2.0`, a kernel value's wire form, and `benchmark-result@2.0`,
  a benchmark run's `result.json`.

## Amendment 2026-11-03: transcript extent (Phase 3 Task 4a)

- `transcript@1.0` → `transcript@2.0`: `Transcript` adds `extent`, a `Span` (R-188, P3-56). A transcript with a timed
  span declares it, and every span lies inside it on its clock. Non-additive for a reader that checks spans, so a major.
  `transcript@1` is retired and refused with `WireFormatRetired`. `comparison@1.0` keys `transcript@1` becomes
  `transcript@2`, adding `extent` (optional), `extent.start_ns` and `extent.end_ns` (close, `time_ns`) and
  `extent.clock_id` (exact).
- Data formats list (2026-11-02 amendment): `transcript@1.0` is replaced by `transcript@2.0`.

## Amendment 2026-11-04: the graph document (Phase 3 Task 5)

- Data format `graph@1.0`, the graph document of ADR-001 (task nodes; `plain`, `map` and `gather` edges; no route,
  loop or nested nodes until a format bump), published as `runtime/data/graph@1.0.schema.json` (JSON Schema 2020-12,
  regenerated by `scripts/generate_graph_schema.py` and drift-checked). A reader reads any minor of `graph@1`; later
  minors may add top-level fields only, and a document without `format` is read as `graph@1.0`. `Node` and `Edge` are
  closed.
- Identity tag `graph@1`: the digest of a graph's canonical JSON without `format` (nodes sorted by id, edges by source,
  target and kind). A graph of a later minor has no digest under a `graph@1.0` reader (rulings P3-12, P3-13, P3-50).

## Amendment 2026-11-04: the input manifest (Phase 3 Task 6)

- Data format `input-manifest@1.0`: `{format, rows: [{id, entities: {name: value}, inputs: {port: "content:<64hex>" |
  "output:<64hex>"}}]}`, rows sorted by id, ids unique. A document, not a port value.
- Identity tag `input-manifest@1`: `identity_digest("input-manifest@1", {"rows": rows})`, from content stamps (chunked
  file content roots) and value digests, never from paths or sidecars, so moving the root leaves the digest unchanged.
  Ids are root-relative POSIX paths in NFC. A reader of `input-manifest@1` ignores fields added in a later minor, and a
  manifest of a later minor has no digest under a 1.0 reader (rulings P3-14, P3-15, P3-16).

## Amendment 2026-11-04: the logical plan (Phase 3 Task 7)

- Data format `logical-plan@1.0`: `{format, graph: graph@1.0, models: {node: ResolvedModel | null},
  profile_slices: {node: {slice: "profile:<64hex>"}}, manifest_digest: <64hex>, bind: {input: record input},
  reuse: "exact" | "equivalent" | "tolerant"}`; `models` and `profile_slices` name exactly the graph's nodes. A
  document, not a port value. No hardware, env, device, batch, partition, backend or profile digest (ADR-004).
- Identity tag `logical-plan@1`: `identity_digest("logical-plan@1", fields)` over every field but `format`; it is the
  run id (ruling P3-17). The graph enters as the `graph@1` digest of its nodes without `model`, with params completed
  by their defaults; each model enters as its key identity (hub id and commit, or the weights digest; no reference
  text, source path or license). A plan (or graph) of a later minor has no hash under a 1.0 reader.
- Identity tag `profile-slice@1`: a declared profile slice's identity is `"profile:" + identity_digest("profile-slice@1",
  <slice value>)`. A task declares slices as `run.<path>` (run config settings) or `dataset.<path>` (dataset profile);
  only declared slices are read (I-1).

## Amendment 2026-11-05: placement, profiles, catalogue, resource tables (Phase 3 Task 8)

- **`placement@1.0`** (new). A plan's placement document: `profile_digest` (the canonical digest of the execution
  profile placed under); `nodes: {node id -> NodePlacement{backend, backend_version, backend_params, stack, env (host
  | <kind>:<name>), device, batch_size, resource_class, partition, env_root, why}}`; `env_builds` (the `<kind>:<name>`
  envs not yet built); `licenses: {node id -> LicenseSet{records: {subject -> License | null}}}`, the node's own env and
  model licenses unioned with every upstream node's; and `cost: Cost{wall_s, gpu_hours, download_bytes, unmeasured}`.
  A figure is null when a node or model it needs has no resource table, and is never estimated. Placement never enters
  an identity: `build_key` reads none of `device`, `env_root`, `why`, `resource_class`, `batch_size` or `partition`,
  and the plan hash is the `logical-plan@1.0` hash. No identity tag. `port_value: false`.
- **`execution-profile@2.0`** (major; `execution-profile@1` retired, refused with `ProfileFormatRetired`). Adds, each
  with a derivation in `config-derivations.md`: `resource_classes: {name -> ResourceClass{device, gpu_type, cpus,
  mem_bytes, time_s, partition, gres, max_workers, memory_cap_bytes}}`, where a name is `cpu`, `mps`, `gpu`,
  `gpu:<type>` or `svc:<id>`, optionally `:<variant>`, and must match the class's device and GPU type;
  `substrate{kind: local | slurm, qos_cpu_cap, exclude_nodes}`; `pool{quantum_items, quantum_s, poll_s,
  lease_renew_fraction, segment_items}`; `faults{content_attempts, infrastructure_replacements,
  resource_replacements}`; `timeouts{load_s, request_s}` (null for no bound, passed to the worker client); and
  `services{idle_exit_s}`. All the 1.0 keys are kept. Migration: rewrite the file as 2.0. Migrating one is not reading
  it.
- **`catalogue@2.0`** (major; `catalogue@1` retired, refused with `CatalogueFormatRetired`). An entry is `{id, revision,
  tasks, backends, stacks, license, default_for}`, and the inline `resources` field is dropped. Measured resources are
  `resource-table@1.0` files joined to entries by `(id, revision)`. A catalogue file holds no tables. Migration: drop
  each entry's `resources`.
- **`resource-table@1.0`** (new). A generated file `data/resources/<task>/<backend>.yaml`, never hand-edited:
  `{task, backend, rows: [{id, revision, download_bytes, classes: {resource class -> {load_s, peak_memory_bytes,
  s_per_input_s}}, measured: {sha (40-hex), jobs}}]}`. `id` and `revision` are both null for a backend without a model.
  A row's model must be pinned by a catalogue entry serving that task and backend. A table's path must match its task
  and backend. `port_value: false`.
- **`senselab-vocab@1.4`** (minor, additive). Adds `senselab:unlicensedReuse` (domain `senselab:Reuse`, range
  `xsd:boolean`): the stored result served records no env license, and the run's license policy allowed it.
- **`context@4`** (identity tag; replaces `context@3`). The context fingerprint adds whether an env license is
  recorded, so a recompute after a refused unlicensed hit gets its own slot. Logical keys and the plan hash are
  unchanged; every stored entry is recomputed once.
- Execution profiles carry no derivation prose: the loader neither requires nor reads `derivations`, an override
  carrying one is refused (`UnknownProfileKey`), and each key's derivation is in `config-derivations.md`, with a
  one-line `#` comment per key in the YAML. `plan(hardware=...)` places under a given hardware probe and never enters
  the plan hash. A license union that cannot be formed raises `LicenseConflict`, naming the node.

## Amendment 2026-11-05: plan fragments (Phase 3 Task 9)

- **`plan-fragment@1.0`** (new). Jobs of one node of one admitted plan: `{format, plan_hash: <64hex>, node: NodeId,
  admission: <executor's admission id>, jobs: [JobRef{plan_hash, node, element: [int >= 0]}], inputs: [{port:
  "content:<64hex>" | "output:<64hex>"}] (one per job, in the order of jobs), upstream: {"output:<64hex>":
  UpstreamRef{key_digest, outputs: <the stored entry's output identity>, port}} (exactly the output identities the
  jobs read, alone or gathered, each equal to its ref's port identity), gathered: {"gather:<64hex>":
  ["output:<64hex>", ...]} (each gather input the jobs read, with its element port identities in element order),
  placement: NodePlacement (`placement@1.0`'s node record),
  attempt: int >= 1}`. Every job names the fragment's plan hash and node, once. Identities only, never values or
  pickled objects. Not an identity: no identity tag; the plan hash is unchanged. `port_value: false`. Results come
  back as `ResultRef{job, key_digest, output_identity | failure: FaultRecord{fault, type, message, gap, charged}}`;
  `charged` is true only for a typed content fault that is not a gap. Results and failures are run-scoped and a
  failure is never stored (ruling P3-26).
- Identity tag `port-output@1` (new): an `output:` input identity names one output port of one upstream job,
  `"output:" + identity_digest("port-output@1", {"key": <upstream key digest>, "port": <port>})`. The upstream
  execution context never enters it.
- Identity tag `port-gather@1` (new): a gathered input's identity is `"gather:" + identity_digest("port-gather@1",
  [<element port-output identities, in element order>])`. A logical key's input is now `content:`, `output:` or
  `gather:` and 64 hex digits.

## Amendment 2026-11-04: run events (Phase 3 Task 10)

- **`run-event@1.0`** (new). One state change of one job, a line of `<run>/events/<writer uuid>.jsonl` (a writer appends
  to its own file only): `{format, kind: queued | claimed | started | finished | failed | yielded | cache_hit | verify |
  blocked | cancelled, node: NodeId, key: <64hex> | null, job: JobRef | null, attempt: <attempt id>, ordinal: int >= 1,
  writer: <writer uuid>, seq: int >= 0, placement_digest: <64hex> | null, fault: "content" | "resource" |
  "infrastructure" | null, failure: FaultRecord | null, reuse: "exact" | "equivalent" | "tolerant" | null, env: <string>
  | null, output: "output:<64hex>" | null, t_ns: int >= 0}`.
  - `attempt` is the id of the run or resume that wrote the event (never the fragment submission count); `ordinal` is
    that attempt's place among the run's attempts, taken when it is claimed (`attempts/<n>` created exclusively);
    `writer` names the file the event is in; `seq` is the event's position in that file, from 0.
  - A `failed` event carries its `failure`, and `fault` is that record's class; no other kind carries either.
    `reuse` is held as the policy's value string and only a `cache_hit` carries it. `job`, when present, names `node`.
  - A job's state machine is queued, then claimed, started, then one terminal kind: `finished`, `cache_hit`, `failed`,
    `blocked` (a descendant of a failed element) or `cancelled` (not released after a cancel). `yielded` returns a job
    to queued; `verify` changes no state. A job's state is that of its latest attempt (highest `ordinal`): the terminal
    event with precedence `finished` = `cache_hit` > `failed` > `blocked` > `cancelled` (ties: greater `writer`, then
    greater `seq`), so a result is never hidden by another writer's not-run event; else that attempt's last other event
    in the time merge. `seq` is compared within one writer only, and no clock order decides a terminal state. A run is
    done when every job it queued has a terminal event; it is cancelled when any job is cancelled. Cancelled and blocked
    jobs are not faults.
  - Not an identity: no identity tag; the plan hash is unchanged. `port_value: false`.
- A run directory is `$SENSELAB_CACHE/2x/runs/<plan hash>/`: `plan.json`, `inputs.json` (what each record's inputs are
  read from: a path, a JSON value, or `{"memory": <input identity>}` for an object given in the planning process, which
  only that process can run; the rest are re-bound and identity-checked on every run, before anything is written),
  `manifest.json` (plan hash and result store; no status), `placements/<attempt>.json`, `attempts/<n>` (the attempt id
  claimed `n`-th), `events/`, `dead/`, `prov/`, `queues/`, `epoch/`. The provenance `run_id` is the plan hash; each
  `run()` or `resume` has a fresh attempt id (the one `SENSELAB_RUN_ID` or Slurm pins, with `-<n>` for a second attempt
  in one process, else a uuid4), which names its placement file; placement files are never replaced (ruling P3-17).
  While an attempt runs, `revision.attempt_id()` (read by equivalence evidence and the revision manifest) is that
  attempt's id; outside a run it is the process's.
- `ResultRef` gains `served: "exact" | "equivalent" | "tolerant" | null`: the reuse policy under which a stored
  result gave it; the runner logs it as `cache_hit`.

## Amendment 2026-11-07: config files hold values (Phase 3 Task ENF)

Each data file below drops its inline `derivation`/`derivations` keys, which move to a `##` section of
`config-derivations.md` keyed by the file's path (P-17); the major version moves because the shape changes:
`store-policy@2.0`, `launch-environment@2.0` and `fixture-budget@2.0` and `size-budget@2.0` (a key maps to its value),
`run-config@2.0` (a threshold is exactly `rtol` and `atol`; a user's run config may not carry a derivation either),
`env-lock-inputs@2.0` (`quarantine` and `base_requirements` are lists), `license-policy@2.0` (a set is its `ids`),
`default-python@2.0` (`default` alone), and `equivalence@4.0` (a class is exactly its `members`; `@3.0` is retired
and refused, and `scripts/regenerate_equivalence.py` prints each class's derivation for the spec). No identity tag
changes: none of these values entered a digest with its derivation. `signal/data/resample.yaml` carries no format tag
and loses its `derivations` block. `spdx-licenses@1.0` still holds one derivation per group; its groups are unnamed,
so moving them needs a format change, left open.

## Amendment 2026-11-07: recorded lengths in results (Phase 3 Task 18)

- **`plan-fragment@2.0`** replaces `@1.0` outright. `UpstreamRef` gains `length_ns: int >= 0 | null`, the span in ns
  of the port's value when it is a recording. Results come back as `ResultRef{..., lengths_ns: {port: int >= 0}}`, the
  span of each recording output, taken when the result is given. A job's batch formation is measured from these, so
  keying reads no stored value. Queue items are named under identity tag `plan-fragment@2`, which replaces
  `plan-fragment@1`. Logical keys and the plan hash are unchanged.

## Amendment 2026-11-07: sizing from measures (Phase 3 Task 20)

- **`execution-profile@3.0`** (major; `execution-profile@2` retired, refused with `ProfileFormatRetired`). A resource
  class has no `batch_size`: a placement's batch is sized (`size`) from the resource tables, 1 while unmeasured.
  Adds `sizing{load_share, memory_headroom, target_wall_s, calibration_items, calibration_workers, device_query_s}`; `pool.quantum_items` and
  `pool.quantum_s` are ceilings of the sized quantum. Migration: drop each class's `batch_size`, add `sizing` from the
  packaged profile of the same name, and set `format: execution-profile@3.0`.
- **`resource-table@1.0`** (amended in place: no table had been generated). A row is `{id, revision, download_bytes,
  classes: {resource class -> {load_s, claim_s, resident_bytes, peak_memory_bytes, device_resident_bytes,
  device_peak_bytes, s_per_input_s, input_s_per_item, batched: {batch_size, s_per_input_s} | null, measured: {sha, run,
  jobs}}}}` (`batched` added by Phase 3 Task 20e: the seconds per input second of items sent in the largest batch
  measured, so sizing keeps a batch only where it is not slower per item): each class carries its own
  derivation, and writing one class replaces that class alone. `download_bytes` is null when unmeasured; the host
  pair is null where no resident figure could be read (no cgroup v2), the device pair off a device or unread; `input_s_per_item` is null for items without a recording, `s_per_input_s` being then per
  item; `sha` is null for a calibration round, whose `run` names the senselab-core version and the run, and `jobs`
  holds at least one id. Packaged tables are generated by `scripts/generate_resources.py`; a calibration round writes
  under `<cache>/2x/resources/<senselab-core version>+<12 hex of identity_digest("core-runtime@1", the package's *.py files)>/`, read before the packaged tables and never by other code.
- **`scale-result@1.0`**, **`scale-suite@1.0`**, **`scale-budgets@1.0`** (new; benchmarks, not port values): the ladder's
  per-point result, its suite file and its budgets file (`benchmarks/scale/`).
- `core-runtime@1` (new identity tag, Phase 3 Task 20h): `identity_digest("core-runtime@1", [[relpath, file_digest("file@1", path)], ...])` over senselab-core's `*.py` files, sorted by POSIX relative path; it names the calibration-row directory, never a key.
