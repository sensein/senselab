# ADR-002: Identity v2. Exact keys over portable facts, a recorded context, and a reuse policy

**Status:** accepted, 2026-10-30. Supersedes the key in design §4.8 as of commit `6c6e4dd5`.
Principle P-38.

## Context
- Architect review §2 (R-39): keys hashed upstream *keys*, so identical upstream output still
  invalidated downstream.
- §3 (R-40): env identity was the whole lock plus the platform, giving near-zero hits across hosts.
- §4 (R-41): the decoder, system libraries and driver stack change outputs but were not keyed.
- §8 (R-44): identity of long and large inputs.
- The owner: hashes judge exactness, not distance. Function and environment hashes differ between
  macOS and Linux (pydra #777 for function hashing, #776 for generic objects), and a user may accept
  outputs that are the same or close.

## Decision

**1. Logical key: exact, and portable across machines.**
```
{schema, task, task_version, backend, backend_version,
 model weight identity, params (RFC 8785 JCS canonical JSON),
 input identities, declared profile slices,
 batch formation and chunk grid, when the backend declares them output-affecting}
```
- **Input identity** is the hash of the raw content bytes for source data. For derived data it is
  the upstream **output** identity: the hash of the canonical serialised output (ADR-006). An
  upstream change that reproduces identical output therefore leaves downstream keys unchanged
  (early cutoff).
- **Long and large inputs** use chunked content identity: a hash list over fixed-size byte chunks.
  A time-range read is identified by the chunks it touches plus the range.
- **Identities are typed, not free strings.** The key refuses a ref, a path or bare hex where an
  identity belongs:
  - the model is a `ModelIdentity` (`source`, `id`, `identity`), built from a pinned
    `ResolvedModel` by `key_identity()`; an unpinned model raises `UnpinnedModel`;
  - an input is `content:<64 hex>` (source bytes) or `output:<64 hex>` (an upstream output
    identity);
  - a profile slice is `profile:<64 hex>`.
- **Params are canonical by declaration.** A task's params fields must have one canonical JSON
  form (JSON scalars, `Literal`s of them, `str`-valued enums, lists, tuples, `str`-keyed mappings,
  unions and nested params). The registry refuses `set`, `frozenset`, `Decimal`, `datetime`,
  `Path`, `bytes`, `Any` and `object`. A generated contract builds every declared backend's key
  under two `PYTHONHASHSEED` values and requires one digest.
- **The digest is taken at construction** (`tagged_digest("logical-key@2", fields)`, ADR-006), so a
  key that validates can be digested.
- **Versions are declared, not hashed.** `task_version` and `backend_version` are bumped by
  developers. A CI check fails when a task's or backend's source changes without a version bump,
  comparing a source digest against the last released one.
- **What is never hashed:** function code, Python objects and environments.

**2. Execution context: recorded with every result, never part of the key.**
- the env lock sha, platform tag and installed wheel-set hash;
- the numerics class: device type, architecture, precision mode;
- decoder and resampler identity;
- driver stack (CUDA driver, cuDNN);
- host;
- the protocol identity (worker runtime and wire version).

**3. Reuse policy,** in the run config, decides whether a stored result with key K and context C′
may serve a request in context C:

| Policy | A result from context C′ is reused when | Notes |
|---|---|---|
| `exact` | the context fingerprint of C′ equals that of C | reproduction runs |
| `equivalent` (default) | C′ and C are in a **measured equivalence class** for that backend (the tolerance profiles, P-17) | where no measurement exists, behaves as `exact` |
| `tolerant` | always | the user accepts any context; such results are labelled `tolerant` in provenance |

**4. Verify sampling (optional).**
- A fraction `k` of cross-context hits is recomputed in the current context and compared within
  the backend's tolerance.
- Agreements add evidence to the equivalence class. Disagreements split the class and are
  reported.
- Over time this turns "close enough" into measured equivalence.

**5. Invariants revised** (design §4.8):
- I-2 now separates the context from the protocol identity, and neither is in the key.
- I-4: the numerics class is context, governed by the reuse policy.
- I-5: a composite pipeline's identity is its logical-plan hash (ADR-004). Its outputs' keys follow
  rule 1, which means early cutoff.

## Consequences
- **Results move between laptop and cluster** under `equivalent` once the tolerance measurements
  exist, and under `tolerant` immediately.
- **Hits under each policy are a metric** (ADR-008), and verify sampling feeds the tolerance
  profiles.
- **Changing a decoder or driver no longer invalidates the cache.** Its effect is caught by
  equivalence measurement and verify sampling, not by key churn.
- **Declared versions put a duty on developers.** The CI source-digest check enforces it.

## Alternatives rejected
- **Hashing function code.** It is not portable across platforms, and it misses called functions
  (pydra #777).
- **Environment identity in the key** (R-40): it gives almost no reuse and still cannot measure
  distance.
- **Input-addressed composite keys:** they defeat early cutoff (R-39).
