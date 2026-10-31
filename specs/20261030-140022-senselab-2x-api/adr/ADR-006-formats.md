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
  - text: UTF-8 NFC, over portable text only (below).

  Output identity (ADR-002) is the hash of this form.
- **Canonical text is portable by construction (H-R33).** NFC is computed from the interpreter's Unicode
  database, which differs between supported Pythons (3.10 has Unicode 13.0, 3.12 has 15.0). Unicode's
  normalization stability policy fixes the NFC of every string whose code points are assigned in a given
  version, but a code point assigned later can change NFC as a combining mark: `"a\U0001E08F\u0323"`
  normalised one way on 3.10 and another on 3.12 (XC-1). Canonical text therefore holds only code points
  assigned in Unicode 13.0, the database of the 3.10 floor; anything else, and a lone surrogate, raises
  `UnportableText` (`canonical.nfc`, the one NFC every canonical encoder calls: JCS strings and keys,
  provenance and value keys, `U` arrays, weight paths).
  - The allowed code points are `senselab_worker/data/unicode-assigned.txt`: the ranges whose general
    category in `unicodedata` 13.0.0 is neither `Cn` nor `Cs`. `packages/senselab-worker/scripts/unicode_table.py`
    regenerates it and refuses to run on any other Unicode version.
  - The file carries an NFC fingerprint: the sha256 over every allowed code point of its combining class,
    decomposition mapping, NFD and NFC, which together determine NFC on every string over those code points.
    `tests/unicode_table_test.py` regenerates the table under 3.10 and, under any later Python, recomputes the
    fingerprint from that interpreter's database; both must equal the file. 3.10 and 3.12 agree.
  - Refusing, rather than vendoring a frozen NFC implementation, keeps one NFC (the interpreter's) and costs
    only text using characters added after 2020; such text has no portable identity to give.
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
  (42 MB of text, 16 s before) encode in 0.3 s. The budget is four
  times the wire's 16 MiB line (`MAX_LINE_BYTES`): any value a worker can return encodes with room for NFC
  expansion and number spelling.
- **`jcs_loads` accepts only what `jcs_dumps` writes** (`jcs_dumps(jcs_loads(b)) == b`) and raises
  `NotCanonicalJSON` otherwise: a repeated key, `NaN`/`Infinity` or a number that overflows (`1e400`),
  nesting past what the parser can hold, and any non-canonical spelling (`-0.0`, whitespace, unsorted keys,
  non-NFC text) (XC-6).
- **A kernel value's canonical form (`value@2`)** is `{"type": <name>, "fields": <fields>}`. `<fields>` holds
  each field as the value holds it (not its pydantic dump), written by the shared tokeniser
  (`canonical.Tokeniser`), which gives every family outside JSON its own one-key token:
  - an array: `{"$array": <sha256 of its canonical .npy>}`;
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
  - text: two strings, two keys, or two strings of a `U` array, equal after NFC (two such keys in one object
    are refused, not merged; a `U` array keeps a width no narrower than its own);
  - containers: a list and a tuple; a set and a frozenset; two elements of one set with one JCS form are one
    element (XC-7);
  - arrays: equal in dtype name, shape and canonical values whatever their byte order and layout;
  - provenance only: `bytes` and `bytearray` of the same content.

  Provenance and `value@2` keep every other family apart as tagged tokens. The worker wire is transport,
  not identity: it writes `npy_bytes`, the same layout keeping every value's bits, and only an identity
  canonicalises them.
- **Every digest is `tagged_digest(tag, value)` with a registered tag:**
  `sha256(tag || 0x00 || JCS(value))`. No tag is a prefix of another, so two kinds of digest never coincide
  over the same value. A new kind of digest registers a new tag; changing what a digest covers bumps its
  tag's major.

  **Registered digest tags** (`canonical.DIGEST_TAGS`): `logical-key@2` (a task's logical key),
  `output@3` (a result's outputs, arrays as canonical `.npy`), `context@2` (an execution context),
  `content-root@2` (a file's chunked content root), `audio@3` (in-memory audio), `weights@2` (a weight
  set), `value@2` (a kernel value), `provenance-id@2` (a provenance record's id) and `data-manifest@2`
  (a dataset manifest).

  H9d bumped `output@2` → `output@3` and `audio@2` → `audio@3` (canonical `.npy` now writes each value in
  its one form) and `value@1` → `value@2` (per-family tokens, held fields). The store moved to `store@3`, a new
  key space, because entries under `store@2` carry `output@2` identities. `provenance-id@2` is unchanged: its
  arrays were already written by value, and its digests differ only for a set holding two elements of one
  JCS form, which it counted twice against this list; 1.x triage records keep their ids.

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
