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
