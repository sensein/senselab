# ADR-006: Formats, versioning and evolution

**Status:** accepted, 2026-10-30.

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
  - arrays: little-endian `.npy` version 1.0, C order, each value in its canonical form (below);
  - tables: Parquet with a fixed writer configuration;
  - text: UTF-8, in its canonical form (below).

  Output identity (ADR-002) is the hash of this form.
- **Canonical text is portable by construction (H-R33, amended by H-R34).** NFC is computed from the
  interpreter's Unicode database, which differs between supported Pythons (3.10 has Unicode 13.0, 3.12 has
  15.0). Unicode's normalization stability policy fixes the NFC of every string whose code points are
  assigned in a given version, but a code point assigned later can change NFC as a combining mark:
  `"a\U0001E08F\u0323"` normalised one way on 3.10 and another on 3.12 (XC-1). `canonical.canonical_text`,
  the one text form every canonical encoder uses (JCS strings and keys, provenance and value keys, `U`
  arrays, weight paths), therefore has two branches:
  - text whose code points are all assigned in Unicode 13.0, the database of the 3.10 floor, is put in NFC;
  - text holding any other code point (an emoji from 14.0 or 15.0, a later script, an unassigned code
    point) is kept exactly as given, with no normalisation, so every interpreter writes the same bytes.

  It is never refused; only a lone surrogate, which is not Unicode, raises `InvalidText`. Refusing it (H-R33
  as first written) broke provenance totality (H-R29, H-R30), since LLM review text carries modern emoji.
  - The raw form is its own family, and its tag is the post-13.0 code point it holds. NFC and NFD of every
    allowed code point hold only allowed code points (`test_normalising_text_in_the_table_never_leaves_it`),
    so no normalised string equals a raw one, and two raw strings are equal only when identical. No
    `{"$u": ...}` wrapper is used: a JSON string stays a JSON string, so stored outputs and records read back
    as the text that was written, and `jcs_loads` keeps its fixed points.
  - The cost is that two raw strings equal under a later NFC (`e\u0301` + emoji against `\u00e9` + emoji)
    have two digests. Each digest is the same on every interpreter.
  - The allowed code points are `senselab_worker/data/unicode-assigned.txt`: the ranges whose general
    category in `unicodedata` 13.0.0 is neither `Cn` nor `Cs`. `packages/senselab-worker/scripts/unicode_table.py`
    regenerates it and refuses to run on any other Unicode version.
  - The file carries a normalisation fingerprint that hashes outputs only, never database metadata (whose
    API shapes change: CPython 3.14's `unicodedata.decomposition` returns Hangul syllable mappings that
    3.10-3.13 left empty): the NFD and NFC of every allowed code point; the NFC of each of the floor's 941
    canonical composition pairs; and, for every code point that reorders against a class-1 or class-240
    probe mark, its NFD before one reference mark per combining class. The pairs and the reference marks are
    read once from 3.10's database and stored in the table.
    `tests/unicode_table_test.py` regenerates the table under 3.10 and, under any later Python, recomputes the
    fingerprint from that interpreter's database; both must equal the file. 3.10, 3.12 and 3.14 agree, and the
    digests of raw strings pinned under 3.10 are reproduced under 3.12 and 3.14. The 3.10 floor CI job runs
    the regeneration.
- **The canonical encoders accept an exact-type whitelist.** JSON scalars are exactly `None`,
  `bool`, `int`, `float` and `str`, or a numpy scalar defined by numpy whose dtype kind is one of
  `b i u f U` and whose `.item()` is one of those types. Containers are exactly `list`, `tuple` and
  `dict` with `str` keys. Arrays are exactly `numpy.ndarray` with dtype kind `b i u f U`. Everything
  else raises: subclasses, `Enum` members, `datetime64`, `timedelta64`, `void`, complex, masked
  arrays. One scalar encoder (`canonical_scalar`) serves every caller.
- **One walk, one bound.** Every canonical encoder (JCS, the tokeniser behind `value@2` and provenance, the
  store's output encoder) walks containers through one `canonical.Walk`: the shared cycle and depth guard
  (`MAX_DEPTH` = 64 containers), a memo that writes a container shared by many parents once per depth, and a
  work budget of `MAX_CANONICAL_CHARS` = 64 Mi characters of JCS text written or replayed. A shared DAG
  (`x = [x, x]` 63 times) was expanded as a tree (XC-5); it is now refused in about 0.2 s, and 23 levels
  (42 MB of text, 16 s before) encode in 0.3 s. A replayed container is appended as one joined string, so
  memory follows the text, not the number of tokens: 161 MB peak for those 23 levels and 75 MB for the
  refused 63, against 1.04 GB each when replay copied one list entry per token. The budget is four
  times the wire's 16 MiB line (`MAX_LINE_BYTES`): any value a worker can return encodes with room for NFC
  expansion and number spelling.
- **`jcs_loads` accepts only what `jcs_dumps` writes** (`jcs_dumps(jcs_loads(b)) == b`) and raises
  `NotCanonicalJSON` otherwise: a repeated key, `NaN`/`Infinity` or a number that overflows (`1e400`),
  nesting past what the parser can hold, and any non-canonical spelling (`-0.0`, whitespace, unsorted keys,
  non-NFC text) (XC-6).
- **A kernel value's canonical form (`value@2`)** is `{"type": <name>, "fields": <fields>}`. `<fields>` holds
  each field as the value holds it (not its pydantic dump), written by the shared tokeniser
  (`canonical.Tokeniser`), which gives every family outside JSON its own one-key token:
  - an array: `{"$array": blob_digest("npy@1", <its canonical .npy>)}`;
  - a set or frozenset: `{"$set": [<elements, one per JCS form, sorted by JCS bytes>]}`;
  - an enum member: `{"$enum": [<qualified type name>, <its value>]}`; a path: `{"$path": <posix text>}`;
  - a type: `{"$type": <qualified name>}`; a held `Frozen` value: `{"$value": {"type": ..., "fields": ...}}`;
  - an integer outside the I-JSON range (±(2**53 − 1)): `{"$int": "<decimal>"}`; a non-finite float:
    `{"$f": "NaN" | "Infinity" | "-Infinity"}`; a `Decimal`: `{"$dec": ...}`; bytes: `{"$bytes": <hex>}`;
  - any other mapping becomes an object, and a key that starts with `$` gains one more `$`, so no held
    mapping reads as a token.

  So no two families share an encoding: an array and its sha256 hex string, a path and its text, an enum
  and its value, a held value and a mapping of its fields all differ (XC-2). `value@1` replaced arrays,
  paths and enums with plain strings, and a field typed `Any` compared an array equal to its hash.

  Durable identities never name a Python class: a recording's `content:` identity hashes a
  declared type token (`recording`, or `audio@3` for in-memory audio), not a module path.
- **Canonical equivalences.** This is the one list. Values that differ in Python but are one value here share
  every digest, and each canonical form writes them alike:
  - numbers: `1`, `1.0`, a numpy `1` and `1.0`, and `-0.0`, `0.0` and `0` compare by value, in JSON and in
    arrays, where `-0.0` is written as `0.0`;
  - NaN: every NaN, whatever its sign or payload, is one value: in arrays it is written as the positive quiet
    NaN with no payload (`0x7E00`, `0x7FC00000`, `0x7FF8000000000000`), in tokens as `{"$f": "NaN"}`. `inf -
    inf` gives `0x7FF8…` on arm64 (measured) and sets the sign bit on x86 (XC-4, inferred from the
    architecture), so without this one computation could hash two ways;
  - bools: every nonzero byte of a bool array is `True` and is written as 1;
  - text: two strings, two keys, or two strings of a `U` array, equal after NFC when every code point is
    assigned in Unicode 13.0 (two such keys in one object are refused, not merged; a `U` array keeps a width
    no narrower than its own); text holding any other code point is equal only to itself;
  - containers: a list and a tuple; a set and a frozenset; two elements of one set with one JCS form are one
    element (XC-7); in `value@2`, any `Mapping` (a `FrozenMapping`, `MappingProxyType`, `OrderedDict`) and a
    `dict` with the same items;
  - paths: a `PurePosixPath`, a `PureWindowsPath` and a `Path` with one `as_posix()` text (`$path` is that text);
  - arrays: equal in dtype name, shape and canonical values whatever their byte order and layout; a 0-d array
    is never equal to a one-element array;
  - bytes: `bytes` and `bytearray` of the same content (provenance and `value@2`).

  Provenance and `value@2` keep every other family apart as tagged tokens. The worker wire is transport,
  not identity: it writes `npy_bytes`, the same layout keeping every value's bits, and only an identity
  canonicalises them. The result store is identity: it writes and reads back the canonical forms, so a cached
  output has `-0.0` as `0.0`, one NaN without payload, bools as 0/1 and text in canonical form, where a freshly
  computed one keeps what the backend returned.
- **Every digest names a registered tag, and no other digest function exists** (`senselab_worker.digest`,
  H9e, R-106). There are two:
  - `identity_digest(tag, value)` = `sha256(tag || 0x00 || JCS(value))` over a canonical value. No tag is a
    prefix of another, so two kinds of identity never coincide over the same value. A new kind registers a
    new tag; changing what an identity covers bumps its tag's major.
  - `blob_digest(tag, bytes | path | binary stream)` = the plain lowercase sha256 of raw content. The tag is
    checked against the blob registry and not hashed, so a blob digest equals the content's published
    sha256: the store's and the wire's array file names, fetch manifest checksums, weight pins and provenance
    file checksums keep their values. Both functions refuse an unregistered tag.

  The raw primitives (`sha256_bytes`, `sha256_file`, the untagged `tagged_digest` re-export) are gone, not
  private: `digest.py` imports `hashlib` inside its two functions, so no module namespace in `senselab_worker`
  or `senselab_core` holds a digest module or a function from one.

  **Enforcement.** One import-linter contract, `digests-are-tagged` in `.importlinter`, forbids every root
  package (`senselab_worker`, `senselab_core`, `senselab`, `scripts`) to import `hashlib`, `_hashlib`, `hmac`,
  `blake3`, `xxhash`, `mmh3`, `Crypto`, `Cryptodome`, `cryptography` or the `_md5`/`_sha*`/`_blake2`
  extension modules. Its only 2.x allowance is `senselab_worker.digest -> hashlib`; the others are the 1.x
  modules and scripts of H-R19, the migration worklist (`migration-ledger.md`). An allowance that matches no
  import fails the contract. A digest reachable without importing a digest module is a name, not a module,
  so ruff's banned-api (TID251) bans it, per call site: name-based UUIDs (`uuid.uuid3`, `uuid.uuid5`), other
  libraries' hashes (`joblib.hash`, `zlib.crc32`/`adler32`, `binascii.crc32`/`crc_hqx`, sklearn's
  `murmurhash3_32`, dask's `tokenize`), and the dynamic-import entry points that would reach a digest module
  unseen (`importlib.import_module`, `importlib.__import__`, `importlib.util`, `importlib.machinery`,
  `sys.modules`). `getattr` with a literal name (B009), `exec` (S102) and `eval` (S307) are reported too. An
  existing site carries its own `# noqa`, so a second site in the same file is reported, and a mark that no
  longer suppresses anything is reported by `digest_contract_test.py`. Tests are exempt: they use `hashlib`
  as the independent oracle.

  **What the contract does not see.** Files outside every root package (`hatch_pin_loader.py`, the
  floor-interpreter generator `packages/senselab-worker/scripts/unicode_table.py`, which runs without the
  worker installed and hashes normalisation output, and the network guard's `sitecustomize.py`) are listed in
  the test that checks every production file is in the graph, so a new one fails it. Reflection with a
  computed name (`getattr(m, name)`, the builtin `__import__`, `vars(m)[...]`) is not decidable statically.
  A 1.x worklist module can still hash serialised data until it is ported (P-39).

  **Registered identity tags** (`digest.IDENTITY_TAGS`): `logical-key@3` (a task's logical key),
  `output@3` (a result's outputs, arrays as canonical `.npy`), `context@3` (an execution context),
  `content-root@2` (a file's chunked content root), `audio@3` (in-memory audio), `weights@3` (a weight
  set), `value@2` (a kernel value), `provenance-id@3` (a provenance record's id), `data-manifest@3`
  (a dataset manifest), `worker-runtime@1` (the worker runtime's source files) and `fake-output@1` (the
  fake backend's output value).

  **Registered blob tags** (`digest.BLOB_TAGS`): `file@1` (a file's or a source's bytes, and each chunk of
  a content root) and `npy@1` (an array's `.npy` bytes).

  H9e registered two identity tags for digests that were untagged: `worker-runtime@1` replaces the sha256 of
  the concatenated `path NUL bytes NUL` records of the worker's source, now
  `identity_digest("worker-runtime@1", [[relpath, blob_digest("file@1", path)], ...])`; `fake-output@1`
  replaces the sha256 of the fake backend's JCS payload, whose random nonce is now a payload field. Both
  values changed; neither addresses stored data (the runtime identity is compared live in the handshake).

  H9d changed two things every digest can see, and bumped every tag whose payload can carry them:
  - canonical `.npy` now writes each value in its one form: `output@2` → `output@3`, `audio@2` → `audio@3`,
    and `value@1` → `value@2`, which also took per-family tokens and held fields;
  - text holding a code point unassigned in Unicode 13.0 is now written raw (H-R34), where before it was put
    in NFC, so its JCS bytes changed on every interpreter: every tag whose payload can hold such text is
    bumped. That is `logical-key@3` (task, backend, model and parameter text), `context@3` (environment,
    platform and driver text), `weights@3` (relative paths), `provenance-id@3` (attributes and parameters,
    which also merge a set's elements of one JCS form) and `data-manifest@3` (paths, licence and terms), along
    with `output@3` and `value@2` above.
  - `content-root@2` (an integer and sha256 hex strings) and `audio@3` (a sha256 hex string and a rate) carry
    no text and keep their majors.

  The store moved to `store@3`, a new key space, because its addresses are `logical-key` digests and its
  entries carry `output` identities. Every provenance id changes, for 13.0-only inputs too, since the tag is
  hashed with the payload: 1.x triage records written before H9d do not share ids with records written after.

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
