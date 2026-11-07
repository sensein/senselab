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
- **The digest is taken at construction** (`identity_digest("logical-key@7", fields)`, ADR-006), so a
  key that validates can be digested.
- **A server backend's key holds its service's attestation** (`service`: engine version, served model, model
  identity, config digest), as design I-8 requires; `key_for` refuses a server backend's call without it and any
  other backend's call with it (ADR-003, amendment 2026-11-07).
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

- 2026-11-07: a server backend's key holds the attestation of the service answering the call (`service`,
  `logical-key@7`), implementing I-8's "the key uses the attested value" (Phase 3 Task 17, P3-40).
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
- 2026-11-02: equivalence members are (numerics class, env lock), `equivalence@3.0`, `equivalence-evidence@2.0`;
  consistency (`exact`, `equivalent`) vs availability (`tolerant`); `ExecutionContext.env_license`,
  `execution-context@3.0`, `store@5`; `RunConfig.licenses` (P2-92, P2-93; Phase 2 Task 34d).

## Amendment 2026-11-02: the producers of batch formation and chunk grid (Phase 2 Task 34a)

- `run_call` derives the placement's `batch_formation` and `chunk_grid` from the request it sends
  (`runtime.keys.request_components`): every request holds one item, so the batch formation is `single`, and the chunk
  grid is the backend's declared `BackendSpec.chunk_grid` (a grid that is a backend param is keyed as a param). A
  backend that declares either output-affecting therefore always has a value for the key (R-187).
- The backend contract gains `output_affecting_produced`: every declared output-affecting component has a value from
  `request_components`, and a backend that is not batch-invariant declares `batch_formation` output-affecting.
  `speech_to_text/transformers` now declares it, so its keys carry `batch_formation: single`.

## Amendment 2026-11-02: env versions are agents; equivalence members carry the env lock

Ruling P2-92 (owner item 45; supersedes the P2-23 reading in R-185). Phase 2 Task 34d.

- **A member of an equivalence class is a numerics class in one env lock.** `runtime/data/equivalence.yaml` is
  `equivalence@3.0`: per backend, classes of `{numerics, env_lock}` members (`member_key`, the JCS text of both),
  each with a derivation. `equivalence@1` and `@2.0` are refused with `EquivalenceFormatRetired`. Under
  `equivalent` a stored context serves the current one when their fingerprints are equal or their members are in one
  class of the backend; platform, wheel set, decoder and driver still do not enter that test (P2-23).
- **A class never spans env locks without evidence for that pair.** `EvidenceSide` records the env lock
  (`equivalence-evidence@2.0`, stored under `equivalence-evidence@2/`), and regeneration keys each pair by both
  members, so agreement within lock L1 and within lock L2 never joins L1 and L2. Splits act on members.
- **The packaged table is empty again.** Its one class (pyannote, cpu and cuda sm_90) was measured from
  `equivalence-evidence@1.0` records that carry no env lock, so it cannot be attributed to a lock pair; it returns
  when the verify-sampling run is repeated and the table regenerated.
- **Which policy is which.** `exact` and `equivalent` are the consistency policies: neither serves a result from
  another env lock unless a measured class holds that pair (`exact` never does). `tolerant` is the availability
  policy: it serves another env lock's result, and the reuse records both agents (ADR-010 amendment of this date).
  No policy was added.
- **Provenance agent of a result** carries its env lock (`senselab:envLock`), so a difference in env is always
  visible.

## Amendment 2026-11-02: the env license is context

Ruling P2-93 (owner item 46). Phase 2 Task 34d.

- `ExecutionContext.env_license` (a `License` or None) is the code license of the env, `backend_license` over its
  env lock's license records. It is recorded, never fingerprinted, like `host`, `protocol` and `model_source`.
  Stored contexts are `execution-context@3.0`; entries live under `store@5/`, and `execution-context@2` is a
  retired store format (the store format gate, Task 34a).
- `RunConfig.licenses` (`LicensePolicy`, the user's allowed and refused SPDX ids, `allow` None meaning every id not
  refused) is the only thing that refuses a call for its license: a refused env or model license raises
  `LicenseRefused` before the env opens, and a stored result whose env license the policy refuses is not served.
  A subject with no license record is refused when a policy is named. With no policy, no license refuses a call.

## Amendment 2026-11-02: agreement is a standard comparison under configured thresholds

Ruling P2-95 (owner items 48, 49). Phase 2 Task 34e. Supersedes, in the amendments above, every statement that a
comparison is judged within a backend's tolerance profile, that a sampled hit's backend needs a profile
(`NoToleranceProfile`), that `Packaged.tolerance` hands `run_call` a folder of profiles, and that a class derivation
names tolerance profile digests.

- **Three comparisons only.** Two outputs agree when their hashes are equal, or leaf by leaf when each leaf is equal
  (`exact`) or within `numpy.isclose` of a named threshold (`close`, `|actual - expected| <= atol + rtol *
  |expected|`). `runtime.compare.compare` short-circuits on equal hashes and otherwise returns `Comparison`: agree,
  the field-level differences (path, rule, threshold name and bounds, both sides summarised, and why) and the
  thresholds' canonical digest.
- **Each kernel type declares how its fields compare,** as data: `runtime/data/comparison.yaml`
  (`comparison@1.0`), keyed by `<name>@<major>` of every wire format, each field an index-free path pattern with
  `exact` or `{close: <threshold>}`. A kernel value is found by its `format` leaf under any path; a leaf under a type
  with no rule for it, or on one side only, is a difference; a leaf outside any kernel value is exact. Identities and
  text are exact. `Intervals` declares `match_labels`, so its labels are compared after the optimal one-to-one
  time-overlap map (P2-89), the one relabelling there is.
  A kernel value whose type has no entry in the table is refused (`UndeclaredComparison`, naming its format tag),
  never compared as exact.
- **Thresholds are run config values,** never fitted. `runtime/data/run_config.yaml` (`run-config@1.0`) holds
  `thresholds`, each `{rtol, atol, derivation}`; `Thresholds.load(override)` deep-merges a user's run config (a
  mapping or YAML) over it and refuses an unknown name with the closest one. `RunConfig.thresholds` defaults to the
  packaged ones. The packaged defaults (`time_ns`, `probability`, `score`, `embedding`, `samples`, `value`) cite the
  ORCD measurements taken for the retired profiles as evidence for the written value. A benchmark may later evaluate
  which thresholds suit a backend; it informs a default and never sets one at run time.
- **Verify sampling** compares a sampled hit with its recomputation under `run.thresholds`; nothing is read before
  execution and no backend lacks what verification needs. `EquivalenceEvidence` records `thresholds`, the digest,
  in place of `tolerance` (`equivalence-evidence@3.0`, under the store's `equivalence-evidence@3/`); a class
  derivation names the threshold digests. Every activity a call writes records `senselab:verifyThresholds` (ADR-010
  amendment of this date).
- **Parity uses the same thresholds.** `parity_reference.py compare --config` and the parity tests
  (`$SENSELAB_PARITY_CONFIG`) load the run config through `Thresholds.load`; each decision is a
  `parity-comparison@1.0` record carrying the thresholds' digest.
- **Retired:** the tolerance profiles (`senselab/data/tolerance/`), `scripts/generate_tolerance.py` (`measure`,
  `generate`), `runtime.tolerance` with its coverage policy, `NoToleranceProfile` and its envelope row, and R-186's
  coverage target.
- 2026-11-02: per-type comparison as data, thresholds in the run config, verify evidence by thresholds digest;
  tolerance profiles retired (P2-95; Phase 2 Task 34e).
