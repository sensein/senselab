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
 model weight identity, params and backend params (RFC 8785 JCS canonical JSON),
 input identities, declared profile slices,
 batch formation and chunk grid, when the backend declares them output-affecting}
```
- **Input identity** is the hash of the raw content bytes for source data. For derived data it is
  the upstream **output** identity: the hash of the canonical serialised output (ADR-006). An
  upstream change that reproduces identical output therefore leaves downstream keys unchanged
  (early cutoff).
- **Cached outputs are clock-relative.** An input's identity is its content, not its `TimeAnchor`, so two
  recordings of one content on two clocks share a key; a stored output carrying a `clock_id` is re-bound to
  the consumer's anchor when it is served.
- **Long and large inputs** use chunked content identity: a hash list over fixed-size byte chunks.
  A time-range read is identified by the chunks it touches plus the range.
- **Identities are typed, not free strings.** The key refuses a ref, a path or bare hex where an
  identity belongs:
  - the model is a `ModelIdentity` (`id`, `identity`), built from a pinned
    `ResolvedModel` by `key_identity()`; an unpinned model raises `UnpinnedModel`. A hub model is its repo
    id plus the resolved 40-hex commit; a `url` or `local` model has `id` null and the digest of its weights
    (see the 2026-11-01 amendment);
  - an input is `content:<64 hex>` (source bytes) or `output:<64 hex>` (an upstream output
    identity);
  - a profile slice is `profile:<64 hex>`.
- **Params are canonical by declaration.** A task's params fields must have one canonical JSON
  form (JSON scalars, `Literal`s of them, `str`-valued enums, lists, tuples, `str`-keyed mappings,
  unions and nested params). The registry refuses `set`, `frozenset`, `Decimal`, `datetime`,
  `Path`, `bytes`, `Any` and `object`. A generated contract builds every declared backend's key
  under two `PYTHONHASHSEED` values and requires one digest.
- **The digest is taken at construction** (`identity_digest("logical-key@6", fields)`, ADR-006), so a
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
- the protocol identity (worker runtime and wire version);
- how the model's weights were reached (`model_source`: `hf`, `url` or `local`).

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

## Amendment 2026-11-01: portable model identity, reuse default, equivalence as data

Rulings H-R56 and H-R57 ([`rulings.md`](../rulings.md)); tags in ADR-006.

- **A model's logical key is the content digest of its weights, or the hub repo plus the resolved commit.**
  `ModelIdentity` is `{id, identity}`. For a hub model `id` is the repo id and `identity` is the 40-hex
  commit. For a `url` or `local` model `id` is null and `identity` is the 64-hex digest of the weights
  (`file@1` for one file, `weights@3` for a directory), whichever way they were reached.
- **The source is context, not key (H-R56).** Whether the weights came from the hub, a URL or a local path is
  recorded in `ExecutionContext.model_source` and is not in the context fingerprint. The same weights reached
  by `local:` and by `url:...#sha256=` give one key. Nothing sets `model_source` yet: Phase 3 executors set
  it where they build the context.
- **`equivalent` is applied when no policy is given.** `ResultStore.get(key, context, policy=DEFAULT_REUSE_POLICY)`
  uses `equivalent`; the kernel's `acceptable()` still takes the policy as a required argument.
- **Equivalence classes are data.** They are read from `runtime/data/equivalence.yaml` (format `equivalence@1`)
  by the runtime, not declared in code. Each class lists its context fingerprints and must carry a derivation;
  `parse_equivalence` refuses a class without one. The file holds no class, so `equivalent` behaves as `exact`.
- **Verify sampling (§4) has no owner yet.** No phase builds it, so nothing adds evidence to a class or splits
  one; classes are added by hand from measurement until an owner is named.
- Logical keys written under `logical-key@3`, `@4` or `@5` are not addressed; this is accepted before release.

## Amendment 2026-11-02: classes per backend by numerics class; verify sampling owned

Rulings P2-23, P2-43, P2-60, P2-83, P2-89 (Phase 2 Task 21; R-112, R-155). This amendment supersedes the
2026-11-01 statements that classes list context fingerprints and that verify sampling has no owner.

- **A class relates numerics classes, per backend.** `runtime/data/equivalence.yaml` is `equivalence@2.0`: per
  backend, classes whose members are numerics classes (`ExecutionContext.numerics`, compared as `numerics_key`,
  its JCS text), each with a derivation. Under `equivalent`, a stored context serves the current one when their
  fingerprints are equal or their numerics classes are in one class of the backend; the other context fields do
  not enter that test (P2-23). A numerics class is in at most one class of a backend; repeated YAML keys at any
  level and empty derivations are refused; `equivalence@1` is refused with `EquivalenceFormatRetired`, naming the
  regeneration. With no class `equivalent` behaves as `exact`.
- **A class is a clique of measured pairs.** Two numerics classes are measured equivalent when some comparison
  of theirs agreed within tolerance and none disagreed. Tolerance is not transitive, so every pair of a class is
  measured: A~B and B~C with A-C unmeasured never put A and C in one class. `scripts/regenerate_equivalence.py
  regenerate <stores>` takes maximal cliques largest first (ties by sorted members), each from the numerics
  classes no earlier class holds. Each class's derivation counts agreements, disagreements and keys and names the
  runs and tolerance profile digests. The table is regenerated, never hand-edited.
- **Verify sampling (§4) is owned by `run_call`.** `RunConfig.verify` is the fraction `k` in [0, 1] of
  cross-context hits (a hit not admitted by `exact`) recomputed. A hit is sampled when the first 32 bits of
  `value_digest("verify-sample", {key, stored fingerprint, current fingerprint})` fall below `k`, so the decision
  is reproducible. A sampled hit is not served: the backend's `tolerance@1.0` profile (from the folder the
  declaring package hands `run_call`, `Packaged.tolerance`) is read before execution, and a backend with none
  raises `NoToleranceProfile` with nothing executed, stored or recorded. The recomputation is compared with the
  stored result before anything is stored; the comparison is stored as an `equivalence-evidence@1.0` record under
  the store's `equivalence-evidence@1/`; then the outputs are stored and returned.
- **Provenance of verification.** A served hit records one `senselab:Reuse` activity; a sampled hit records one
  execution activity and no reuse, carrying `senselab:verifies`, the evidence record's canonical digest. Every
  reuse and execution activity carries `senselab:verifyFraction` (`k`) and `senselab:verifySampling` (the
  sampling digest's type, `verify-sample`) (`senselab-vocab@1.1`).
- **A disagreement splits the class, persistently.** A disagreeing record takes both numerics classes out of
  the class holding them. The record is the persisted split: every `ResultStore` over the same root applies the
  disagreements recorded in it when it first reads a non-empty table. It is logged with the mismatched output
  paths, never their values.
- **One comparison engine.** The tolerance profile model, `flatten`, `within` and the speaker relabelling are in
  `senselab_core.runtime.tolerance`. `within` compares the labels of every flattened `Intervals` (found by its
  `format` leaf under any path, never by a task or port name) under the optimal one-to-one time-overlap label map (P2-89; a pure-Python Hungarian assignment, so core needs no scipy); an unmatched speaker
  is a mismatch. `run_call` verification and `parity_reference.compare` both use it.

## Amendment log

- 2026-11-01: model key is the weights digest or hub repo plus commit; source is context (`logical-key@5`,
  `execution-context@2.0`; H-R56); `equivalent` applied by default; equivalence classes are data with a
  derivation; verify sampling unowned (F1, F3, F4).
- 2026-11-01: backend params enter the key; batch formation and chunk grid enter only when the backend
  declares them output-affecting; one key builder, `runtime/keys.build_key(call, placement, context)`, with
  `kernel.declare.key_for` its kernel half (`logical-key@6`; P2-47, P2-48; Phase 2 Task 5b).
- 2026-11-01: Phase 2's `runtime/call.run_call` builds the `ExecutionContext` and sets `model_source` from
  `ResolvedModel.ref.source` (None for a model-less call); Phase 3 executors keep this when `InProcess` replaces
  `run_call`. The execution profile's `canonical_digest` is recorded on each execution activity
  (`senselab:executionProfile`), never in the key or the context fingerprint (P2-17, P2-37; Phase 2 Task 10b).
- 2026-11-01: clock re-binding is `kernel.time.rebind(output, stored_anchor, reader_anchor, timemap)`; `run_call`
  stores outputs on the clocks their recording inputs' identities name and re-binds them to the caller's anchors on
  every serve, so no key carries an anchor and one computation serves one content under every anchor; two recording
  inputs of one content on different anchors are refused (`AmbiguousClock`) (R-114, H-R59, P2-66). Chunked identity
  is verified per chunk on read and on `check` (ADR-005 amendment; R-109, P2-21). A local model's identity leaves out
  `.git`, `.hg` and `.svn` directories at any depth; every other file counts (R-118, P2-22) (Phase 2 Task 11b).

## Amendment 2026-11-02: tolerance profiles are keyed by task and backend

- A backend can serve more than one task, and its agreement differs by task. A tolerance profile is therefore
  keyed by task and backend, never by backend alone (P2-87); one profile per `(task, backend)` pair, each with its
  derivation (P2-41).
- A profile is measured on the calibration split and judged on the held-out split; a failure is a finding and no
  bound is refitted to it (P2-41).
- The `equivalent` policy above reads the profile of the call's own task and backend.
- 2026-11-02: `equivalence@2.0` classes per backend by numerics class, cliques of measured pairs; duplicate keys
  and empty derivations refused; verify sampling owned by `run_call` (`RunConfig.verify`), decided before storing,
  evidence `equivalence-evidence@1.0` in the store, splits persisted as evidence, verification recorded in
  provenance; table regenerated by `scripts/regenerate_equivalence.py` (R-112, R-155; P2-23, P2-43, P2-60, P2-83,
  P2-89; Phase 2 Task 21).
- 2026-11-03: the profile a verify comparison reads is `tolerance@2.0`, whose fields name index-free paths (ADR-006
  amendment 2026-11-03; Phase 2 Task 22f).
