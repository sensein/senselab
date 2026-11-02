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

  **Registered identity tags:** `logical-key@6`, `output@3`, `context@3`, `content-root@2`, `audio@3`,
  `weights@3`, `value@2`, `provenance-id@4`, `data-manifest@3`, `worker-runtime@1`, `fake-output@1`,
  `contract-probe@1`, `prov-activity@1`, `prov-node@1`, `wheel-set@1`, `env-lock@1`, `vllm-config@1`, `source@1`. **Registered blob tags:** `file@1`, `npy@1`. **Data formats** (a
  `format:` line in a packaged YAML file or in a store entry): `equivalence@1`, `retry-policy@1.0`, `catalogue@1.0`, `spdx-licenses@1.0`, and
  `execution-context@2.0` in a store entry's `context.json`; `env-record@1.0`, an env root's `<name>[-<tag>].json`; `senselab-vocab@1.0`, a vocabulary document's
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
