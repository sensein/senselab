# senselab 2.x: architecture review

Reviewed: `specs/20261030-140022-senselab-2x-api/` (D = design.md, X = execution.md, EC = engine-choice.md, T = testing.md, PR = principles.md, RM = roadmap.md), the alpha tree (`src/senselab`, about 182k lines), and pydra proposal 0001 with pydra source (`pydra-ext`). Code paths are relative to `src/senselab/`. Five independent deep dives covered execution code, identity, pydra coupling, the data model, and the factual claims. I formed my findings before reading `review-register.md` and only then mapped them to it.

## Executive summary

**Verdict: the direction is sound, but the proposal is not ready to build as written. Approve with conditions.**

The vocabulary is good: declared tasks and backends, profiles in place of flags, an immutable printable plan, typed absence, and identity and provenance owned by the framework. It correctly diagnoses 1.x, and the current-state claims check out against the code (36 `api.py`, 29 locks, 15 `extend_*` drivers; `model_registry.yaml` is read by no `.py` file at all, which is worse than claimed).

The weaknesses are in the parts that will become hardest to change after 2.0:
- the graph semantics are delegated to an engine whose required changes do not exist yet;
- the cache key is input-addressed with whole-lock environment identity, so it will rarely hit;
- the signal and annotation model is too concrete to cover the modalities it promises;
- several execution guarantees that the five earlier reviews marked "resolved in design" have no mechanism that could implement them.

The proposal also frames 2.x as "keeps the machinery" (D:20). For execution that framing is wrong. The pool runs one `Popen` per item (`utils/work_pool.py:1100`), has no batches, no resident models, no typed faults and no handshake. L1, L2 and the coordinator are mostly new code.

**Top 5 risks**
1. **No senselab-owned graph IR.** pydra's hashing and State would define keys, fan-out identity and resume semantics. Its gate items (E1, D1, F2) exist only as prose: the branch diff is the proposal plus two benchmark scripts.
2. **Cache keys that do not hit.** Merkle-of-keys gives no early cutoff, and env identity covers the whole lock and the platform. Any lock bump recomputes the whole graph, and hits between laptop and cluster are near zero. Users will route around the cache.
3. **On-disk formats frozen at 2.0 without an evolution story.** There is no canonical serialisation spec. "New schema directory per bump" means a full corpus recompute. Plans, graphs, profiles and ProvStore have no format versioning.
4. **Unenforceable execution guarantees.** Coordinator fencing on plain files, "starvation is impossible", the admission arbiter, service lifecycle and packed-tier commit are all asserted without a protocol.
5. **A long half-migrated state.** In-place migration with no compatibility layer runs for about 23–34 person-weeks of Phases 1–4 before Phase 5 deletes 1.x. Through that time there are two stores, two worker protocols and two graph runners, and triage stays on a frozen tag indefinitely.

**New findings: 14 of 18** (4 re-open recorded items whose resolution is inadequate).

## Scorecard

| Area | Rating | Reason |
|---|---|---|
| 1 Decomposition and layering | adequate | The nouns are right. "Core" is a runtime rather than a kernel, Plan conflates logical and physical, and there is no owned graph IR. |
| 2 Identity and caching | weak | Input-addressed without early cutoff. Lock-wide env identity is too strict. Decoder, system libraries and backend code are too loose. No serialisation spec. |
| 3 Execution | weak | Fencing, fairness, arbiter, services and packed tier lack protocols. Observability is absent. Most of L1/L2 is greenfield. |
| 4 Signals, time, annotations | weak | A single-ndarray `Signal`, float spans with no unit, TimeMaps only from sync, two annotation forms, and identity that needs a full decode. |
| 5 API and extensibility | adequate | Declarations and generated schemas are strong. There is no post-2.0 evolution policy for tasks, plans or on-disk formats. |
| 6 pydra coupling | weak | Keys go "through pydra's hashing hooks" (X:156). There is no `Executor` protocol, and the gate depends on unwritten work. |
| 7 Security, licensing, reproducibility, performance | adequate | Licensing is thorough, arguably too thorough. There is no model-code trust boundary and no multi-user store access model. Identifiers reach provenance. |
| 8 Migration | weak | No strangler boundary or budget for coexistence. Triage is orphaned on 1.x. |
| 9 Scope | weak | A 2.0 that carries nine modality namespaces, license taint algebra, an env-merging solver, a packed store and MCP is too large. |

## Ranked findings

### 1. Graph semantics must be senselab-owned; pydra becomes a compiler target. NEW (extends R-21, R-29)

**Evidence**
- `Executor` is only named (D:383, X:137, EC:110). A single line, `submit(jobs) -> futures` (pydra-extension-points.md:267), is the only thing like a definition.
- No document defines the IR that `Pipeline` compiles to, or whether the graph schema and plan hash derive from senselab or from pydra structures.
- The key policy is "supplied through pydra's hashing hooks" (X:156), via `__bytes_repr__`. pydra has already moved hashes twice (commits 1c44c750 and 81dca642).
- On the branch `enh/execution-extensions`, `git diff c5081a69..HEAD` contains only the proposal and two benchmark scripts. The barrier is unchanged at `pydra/engine/submitter.py:850-853`.
- Route A still needs senselab-only semantics that pydra lacks:
  - an immutable plan and `resume(plan_hash)` (pydra re-constructs nested workflows at run time);
  - coordinator fencing;
  - `Absent` cause classes;
  - route and loop primitives (EC:47 credits loops "via nested workflows").
- EC scores G1 ● for pydra, but its printable plan is J2, which is unbuilt. G14 (governance) is not an engine property. Rows G4–G10 and G12 are by EC's own account "execution, which senselab owns anyway" (EC:72).

**Why it matters.** This decision is the hardest to undo. Once pydra's hashing, State indices and lazy-field semantics shape keys, fan-out identity and the serialised graph, the "fallback behind the same `Executor` protocol" has no protocol to sit behind. The claim "Moving A→B changes no user code" (X:131) holds only if an IR exists.

**Resolution (abstraction change)**
- Define `senselab_core.graph`: nodes, typed ports, fan-out, route, loop, `Absent`, and expansion to a frozen job set, with keys computed by senselab alone.
- **Invariant:** *the plan hash, the graph schema and every result key are functions of the senselab IR only; no executor contributes bytes to them.*
- `InProcess`, `Pool` and `Pydra` are executors of that IR.
- Upstream D1, C, G1, F1, H and I to pydra on their merits, but take E1 off senselab's critical path.
- Re-score EC without G14, with execution rows marked neutral. The IR plus an in-process executor already meets G1, G2, G3, G11 and G13.
- The fallback estimate (2–4k lines) is credible for the graph layer alone. The reuse it counts is overstated: `reproduce/dag.py` is a 174-line toposort, and `unrolled_contracts` (`contracts.py:991`) is a static check of one workflow. Build the IR now, not at the gate.

### 2. Keys must carry output identity (early cutoff). NEW; the R-6 resolution causes this

**Evidence**
- I-5 says "Merkle hash of its inner node *keys*" (D:349). Downstream inputs are "identity of every declared input" (D:332), and the document never says whether that means the producer's key or its output's content hash.
- Today's `reproduce/cache.py:249,293,439-454` hashes upstream keys.

**Why it matters**
- An env refresh, a `task_version` bump or a re-commit with identical weights invalidates every downstream node, even when the upstream bytes are unchanged.
- Combined with finding 3, every lock bump recomputes the whole graph at corpus scale.
- This is the Nix input-addressed and CA-derivations distinction, and the Bazel action-cache plus CAS split.

**Resolution (invariant I-10).** *An edge carries the content identity of the produced value. A node key = H(own task/backend/code/env/model/params, input content ids).*
- Keep an action cache (key→output id) separate from the CAS (output id→bytes).
- I-5 survives only for planning estimates ("would this rerun?").
- Dagster's `data_version` vs `code_version` is the closest product analogue.

### 3. Env identity is too strict; use per-backend closures and equivalence classes. RECORDED R-4 (the resolution fixes protocol vs numerics but not granularity)

**Evidence**
- I-2 keys lock sha + platform tag + installed wheel-set hash (D:345).
- torch and torchaudio come from a host-routed CUDA index rather than the lock (`utils/venv_lock.py:44-45`), so cu121 and cu128 hosts already diverge on the same lock.
- I-4 keys the numerics class unless a tolerance profile exists, and none exists.
- Estimated hit rates:
  - same host: about 95%;
  - within one cluster across GPU types: 60–90%;
  - laptop↔cluster or macOS↔Linux: about 0;
  - after any lock bump: about 0.
- Env merging (D:441-444) makes this worse by design: "Merging changes numerics identity, which is correct."

**Why it matters.** P-10's promise that laptop and cluster plans differ only in execution fields (D:315) is true of plans and false of results. A cache that never hits across sites stops being part of the workflow.

**Resolution**
- Numerics identity = the hash of the *wheel closure reachable from the backend's declared requirements*. `reproduce/cache.py:211-226` already computes such a closure. It is not the whole lock.
- Make **equivalence classes first-class**: a measured tolerance profile maps a set of numerics identities to one equivalence id, and the key uses that id.
- With I-10, an irrelevant bump stops at the first node.

### 4. Inputs that change output but are unkeyed. NEW (partly within R-1's perturbation contract, which cannot detect these)

**Evidence**
- **Identity "after canonical decode"** (D:147) leaves the decoder and resampler unidentified. System libraries are "probed, recorded" and not keyed (D:236).
- **Device and runtime state.** The CUDA driver, cuDNN, TF32 and `cudnn.benchmark` flags, and BLAS thread count are absent from the key.
- **Nondeterministic backends** are labelled (D:346) but still cached under first-writer-wins.
- **`task_version` is manual** (D:171).
- **Backend implementation code is in no key field at all** (D:328-333 has `backend` as a name). design.md never says how backend code reaches an isolated env, given that the worker runtime is "stdlib + numpy" and outside identity (D:454, I-2).
- **I-1's perturbation contract** perturbs env vars and profile keys. It cannot reveal a library version that is constant within one host.

**Resolution**
- `Decoder` is a versioned task, and its identity feeds data identity.
- Backend source hash (the module closure) is a key field. The overlay exclusion applies only to the wire runtime.
- Driver, cuDNN, determinism flags and thread env join numerics identity, or **an unprobed numerics input makes the result uncacheable** (an invariant).
- Nondeterministic backends are cached only as samples, with a run id in the key, or not at all.
- Golden-output contract tests force `task_version` bumps: an output hash changed without a bump fails CI.

### 5. A canonical serialisation spec and evolution of every on-disk format. NEW

**Evidence**
- There are three serialisers today:
  - `utils/tasks/cached_inference.py:316-318` (`default=str`, NaN allowed);
  - `utils/prov_store.py:202` (`default=str`, truncated to 16 hex);
  - `utils/reproduce/cache.py:116` (strict).
- The audio signature does not hash the dtype (`cached_inference.py:338-358`).
- `Audio.generate_id` hashes `str(self.waveform)` (`audio/data_structures/audio.py:187`), and torch truncates that repr, so long recordings can collide today.
- The only versioning in 2.x is "a new schema directory per bump" (D:358).
- Pipelines serialise "against a published graph schema" (D:267). Plans, profiles, ProvStore and fitted artifacts have no format version or reader policy.
- P-14 forbids compatibility code *within* the 1.x→2.x break. Nothing governs 2.0→2.1.

**Why it matters**
- "Cache invalidation is free" (P-19) holds for a laptop cache. It does not hold for 10⁶ GPU items on a shared cluster.
- A frozen plan (`resume(plan_hash)`) that a newer senselab cannot read is a broken resume.

**Resolution**
- Write one `canonical_bytes` spec with golden vectors, implemented in core and worker (as for weight hashing, D:462):
  - RFC 8785 JSON;
  - params dumped after defaults are filled;
  - enums by value;
  - arrays as `(dtype.str, shape, C-bytes)`;
  - NaN refused.
- Make the key a typed, tagged record (`KeyV<n>`), so a new field is a scoped bump, not a global wipe.
- **Invariant:** *every persisted artifact (store entry, plan, graph, profile, prov record, fitted artifact) carries `format: name/version`. Readers support N and N-1, and upgrades are explicit `senselab migrate` tasks, recorded in provenance.*
- Define semver for the public API and the meaning of `task_version` for users who pin.

### 6. Execution guarantees without mechanisms. RECORDED R-3, R-8, R-9, R-12 (resolutions inadequate)

**Evidence**
- **Fencing (X:48-50).** The store and queues are plain files. "Reject releases with a stale token" is a check-then-act race without compare-and-swap. The controller today is guarded only by `fcntl.lockf` (`utils/work_pool.py:1567`), which is advisory on NFSv3 (`utils/file_lock.py:11-16`). Item tokens fence state transitions but not the item's output: a zombie child and its replacement can both write (`work_pool.py:1084-1086`).
- **Fairness (X:75-77).** `claim` takes the lexicographically first item (`work_pool.py:618-632`), and every claim reaps and lists the whole queue, which is O(N) NFS metadata calls. A bounded quantum needs preemptible or length-bounded items, and the design states neither.
- **Arbiter (X:79-82).** The design gives no process model and no way to measure a non-cooperating vLLM. Today there is a start lock and a static `utilisation()` (`utils/vllm_server.py:277-300,437`).
- **Services (X:89-93).** "Alive while pending consumers" needs graph state inside an `svc` worker, and the design gives no signal path. Identity is assembled on the client side (`vllm_server.py:412-433`), and responses carry none.
- **Packed tier (X:109).** There is no commit protocol for columnar shards under per-item first-writer-wins.

**Resolution (contracts)**
- **Epoch-scoped writes.** A coordinator writes under `epoch-<n>/`, and the live epoch is selected by one atomic rename. Readers ignore other epochs.
- **Output fencing.** Results are committed as `<key>/<claim-token>` and promoted by link, so a zombie's write is never promoted.
- **Sharded queues** with round-robin claim across classes. Item max duration is declared, and a quantum is enforced only at item boundaries. State the guarantee as "bounded by max item length", not "impossible".
- Make the arbiter a per-device ledger file holding reservations against catalogue tables. Over-use is a resource fault. Do not claim more.
- **The packed tier** is append-only segment files plus an index rebuilt from segments, with per-key first-writer resolved at index build.
- Write these as a formal storage-protocol spec and model-check it (TLA+ or a Hypothesis state machine over the `fake` FS) before Phase 3.

### 7. The data model must be protocol-shaped, with time first-class. NEW

**Evidence**
- `Signal.data` is "a numpy array with a dtype per channel" (D:133). A numpy array has one dtype.
- A single `sampling_rate` (D:134) cannot represent a mixed-rate wearable on one clock, or video PTS.
- `IrregularSignal` is a second type (D:142), against the anti-goal at D:104.
- Spans have no unit (D:150), and loading rounds float seconds (`audio.py:165`).
- TimeMaps come only from sync (D:151). Trims and resamples record only `wasDerivedFrom`, so spans on derived signals cannot be projected back.
- `Annotation` and `Events/Intervals` are two forms of one thing (D:154,158). A Transcript "tree" (D:159) cannot hold overlapping speakers or n-best.
- The design drops shapes that audio_analysis already needs: Categorical, windowed embedding, track window/hop, `GridRelation` (`audio/workflows/audio_analysis/shapes.py:13-27`).
- `Embedding.space` = model identity (D:160), which ignores layer and pooling.

**Resolution**
- `Signal` is a protocol: one sample axis, a `TimeBase` (rate, or timestamps, plus a window for tracks), homogeneous channel groups, and `read(range)` over ndarray, memmap or Zarr.
- A `Device{clock_id}` owns its rate groups. Every signal carries `gaps: Intervals`.
- Spans are integer ticks on the source TimeBase. Every time-altering transform emits a child→parent piecewise-linear `TimeMap` (an invariant).
- `Intervals` (columnar, with tier, parent_id, hypothesis rank, producer, guideline, confidence ≠ score) is the canonical form, and `Annotation` is its row view.
- `space = (model id, layer, pooling, window, hop)`.
- Text is content anchored by Spans, not a Signal.

### 8. Identity for long and large inputs. NEW

**Evidence.** Identity after full decode (D:147) is computed per run, with no memoisation (no mtime/stat index anywhere in D or X). `reproduce/weights.py:162-237` already has the stat-stamp pattern.

**Resolution**
- Identity is a Merkle hash over canonical-decoded chunks on the Signal's grid. It is computed once at ingest into the dataset manifest (H-6).
- A range view's identity is `(parent id, tick range)` and needs no decode.
- A stat-stamped identity index `(realpath, size, mtime_ns, inode, fs, decoder id) → id` is shared with the weight verifier.
- Chunked span-emitting tasks declare a halo and a stitch rule. I-7 names the receptive field but not the merge.

### 9. Plan conflates logical identity with physical placement. NEW

**Evidence**
- The Plan includes the hardware probe, partitions and chunk grid and is immutable (D:310, X:45-47).
- Yet resource faults "re-place with a larger grant or smaller batch" (X:69), the controller scales pools (X:98), and `resume` "never re-probes hardware" (X:47). A resume on a cluster whose free GPU type changed has no legal move.

**Resolution**
- Split `Plan` = **LogicalPlan** (identity-bearing: graph, resolutions, keys; hashed; immutable) + **Placement** (mutable and journaled: classes, partitions, batches, grants).
- `resume(logical_hash)` re-places, and placement decisions go to provenance.
- P-10 becomes literally checkable: same logical hash across sites.

### 10. "Core" is a runtime, not a kernel; dependency direction is unenforced below it. NEW

**Evidence.** Core holds the planner, executors, `work_pool`, `fastio`, `vllm_server`, `uv_tools`, the catalogue and huggingface-hub (D:554-562). `vllm_server` is a stack-specific service manager. Backends, agents and other tools that only need types and identity inherit the whole execution runtime.

**Resolution**
- Three layers with import-linter contracts:
  - `kernel`: types, declarations, identity, canonical bytes, store/prov interfaces, graph IR;
  - `runtime`: planner, executors, pools, substrates, catalogue, profile resolution;
  - `stacks`: vLLM, NeMo and other service managers, as backends.
- Backends depend on kernel only. The executor knows services through a `Service` protocol, not through `VllmServer`.
- Publish kernel and runtime in one distribution if needed, but the layering is a contract.

### 11. Observability is unspecified. NEW

**Evidence**
- None of D, X, T or PR mentions logs, metrics, tracing or per-item diagnostics. A grep for observab, metric, logging and tracing finds only benchmark metrics.
- Today the child's stderr is mixed into a Slurm `%x-%j.out` per job (`work_pool.py:1409-1410`). `failed/<id>.json` holds an exit code and a host (`work_pool.py:739-748`).
- Debugging a 100k-item failure means joining `dead/`, `workers/` and log timestamps by hand.

**Resolution**
- **Invariant:** *every execution attempt has an attempt id linking the plan node, item key, worker, substrate job, env identity and a per-attempt log artifact, stored with the failure record.*
- Emit structured events (OpenTelemetry-compatible spans: plan→node→attempt). `senselab runs explain-failure <run>` groups failures by fault class, env and exception signature.
- Progress and rate metrics are store queries, not log scraping.

### 12. Model-code trust boundary and multi-user store privacy. NEW (R-14 covers paths and budgets, not code or data)

**Evidence**
- `model` is any reference string (D:202, D:209). Through MCP an agent can make senselab download and run arbitrary hub code (`trust_remote_code`, pickled `.bin`).
- The store is shared on NFS, content-addressed and cross-user. A hit reveals that another user processed identical content, and there is no access model.
- Provenance records input paths, and BIDS paths carry `sub-`/`ses-` identifiers. BEP028 export publishes them (D:362). `absolute_start` is identifying (use-cases.md notes date-shifting) but its relation to identity is unstated.

**Resolution**
- A **code-trust policy** in the execution profile, peer of `license_policy`: allowlisted model sources, `trust_remote_code` and pickle loading as role capabilities, and untrusted code only in isolated envs with no network after resolve.
- Store **namespaces with ACLs** (per project), and a cross-namespace hit only by explicit share.
- Provenance stores **record ids and content ids, never paths**, with paths in a separate, non-exported site manifest.
- `absolute_start` is excluded from identity and redactable.

### 13. Migration needs a strangler boundary and a budget for coexistence. NEW (R-31 covers campaigns only)

**Evidence**
- Phases 1–4 (23–34 person-weeks, D:705-712) run 1.x and 2.x side by side. 1.x imports core (D:583), and the 1.x side effects stay until Phase 5 (D:591).
- Triage's decision layer stays on the frozen tag until a post-2.x flywheel (D:679-682, RM:55) that has no date. The project's most-used pipeline is therefore on an unmaintained branch for an unbounded period.
- P-14 forbids shims, so alpha users get a moving, mixed API for months.

**Resolution**
- Migrate **by vertical, not by layer**. Each phase ends with one use case wholly on 2.x and its 1.x path deleted in the same phase, so the consolidation inventory shrinks monotonically. Make "inventory count never increases" a CI check.
- Cap the frozen tag's life with a dated decision: port triage's decision layer as a plain `decide` pipeline, or retire it.
- Release 2.x core from a branch to PyPI as `senselab-core` pre-releases, so alpha is never in two halves for external users.

### 14. Route and loop predicates must be tasks. NEW

**Evidence.** `p.route(selector, …)` and `p.loop(…, until=…)` (D:260-262) take callables, but the graph must "serialise losslessly" (D:267), be hashed, and have routes stored as results.

**Resolution.** Selectors and `until` predicates are declared `decide` tasks with keyed inputs. Lambdas are rejected at build time.

### 15. Implicit fan-out is a typing footgun. RECORDED R-19 (the resolution creates it)

**Evidence.** Wiring a list output into an item input fans out implicitly (D:257). A task whose input is itself a list type, or a nested list, is ambiguous, and the agent-facing schema cannot show the arity change.

**Resolution.** Fan-out is a property of the edge in the IR (`map` edge), inferred by the builder but always explicit in the serialised graph and in `explain()`.

### 16. Fault classification must originate in the worker. NEW in mechanism (R-5 is resolved in taxonomy only)

**Evidence**
- An exit code is the only signal (`work_pool.py:1127-1128`), so an OOM kill (137) spends the content budget.
- `venv_worker._reconstruct` collapses errors to `ValueError`, `TypeError` or `RuntimeError` (`utils/venv_worker.py:119-128`).
- There is no handshake version (`venv_worker.py:90-91`), and arrays have no channel (`default=str`, line 263).

**Resolution.** The wire protocol's error frame carries `fault_class` as a required, typed field (spec in Phase 1, closing R-37). An unclassified error is infrastructure by default, never content.

### 17. Over-engineering for 2.0. NEW

**Cut or defer, each behind an interface that keeps the hook:**
- the env-merging solver (D:441-447): keep one env per backend;
- the packed store tier and the arbiter: use static caps + OOM-as-resource-fault until the scale ladder shows a need;
- license *taint algebra* and named-set combinators: keep record + plan-time check + transitive CI;
- pixi envs;
- vLLM-omni;
- `sl.eeg`/`sl.eyetrack`/`sl.motion` namespaces before any task exists in them;
- `sklearn` adapter;
- Route B;
- prose-lint banned-word rules (PR P-31): a review norm, not CI.

**Keep:** kernel, IR, store, provenance, one worker protocol, InProcess + Pool on local/Slurm, the audio port, one cross-modal bridge, and the MCP surface (cheap once schemas are generated).

Also: 37 principles plus 9 invariants plus generated contract suites are a heavy governance load for a small team. Merge the process principles (P-27..P-35) into `engineering.md`.

### 18. Several "facts" need correction. NEW (minor)

**Evidence**
- BIDS entity regexes: four files found by `sub-[`/`sub-(` (`triage/recording_vectors.py`, `triage/speaker_vectors.py`, `triage/nodes/background.py`, `scripts/evaluate_task_events.py`), against "six" claimed (D:79).
- "Keeps the machinery" (D:20) overstates reuse in execution (see Executive summary).

**Resolution.** Re-baseline the effort for Phase 3 to include L1/L2 as new code.

## The three decisions most likely to be regretted

1. **Letting pydra define graph semantics and hashing** instead of owning a small IR (finding 1). It couples senselab's keys and resume to an unbuilt State refactor in another project.
2. **Input-addressed, lock-wide identity** (findings 2–3). It is correct in theory, but it never hits in practice, so users bypass the cache, which defeats P-3/P-5.
3. **Freezing a concrete `Signal`/Span/Annotation model and a schema-wipe-only format policy at 2.0** (findings 5, 7). Every store, profile and BEP028 export inherits those choices.

## Recommended ADRs

1. **ADR: senselab graph IR and Executor protocol.** IR schema, expansion, keys from IR only, and pydra as a backend.
2. **ADR: identity model v2.** Output-addressed edges (I-10), canonical bytes, typed key record, numerics closure, equivalence classes, uncacheable-when-unprobed.
3. **ADR: storage and lease protocol.** Epochs, output fencing, sharded queues, packed tier, a model-checked spec, and the supported-filesystem matrix.
4. **ADR: LogicalPlan vs Placement.** What is hashed, what is journaled, and resume semantics.
5. **ADR: Signal/TimeBase/Span/Intervals.** Integer ticks, mandatory TimeMaps, chunked identity at ingest.
6. **ADR: format versioning and evolution.** Per-artifact `format`, N/N-1 readers, `migrate` tasks, API semver, `task_version` semantics.
7. **ADR: trust and privacy boundary.** Code-trust policy, store namespaces and ACLs, path-free provenance.
8. **ADR: observability.** Attempt ids, structured events, failure triage tooling.
9. **ADR: migration by vertical.** Inventory-monotone CI and a dated triage decision.
