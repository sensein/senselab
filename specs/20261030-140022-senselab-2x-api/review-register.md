# Review register

Each row records one review finding, how it was resolved, and its status. Ids `R-n` are never
reused. New rows come from the adversarial round that ends each phase (P-35).

Reviews of 2026-10-30, all against `senselab-2x-api`, each with a different lens:
- **C**: consistency;
- **A**: adversarial;
- **U**: usability;
- **F**: feasibility;
- **M**: pydra maintainer;
- **P**: principal-architect review with subagents ([`architect-review.md`](architect-review.md)).

| Id | Src | Finding | Resolution | Status |
|---|---|---|---|---|
| R-1 | A | Inputs a node reads that are not in its key (run-config thresholds, fitted profiles, hints, TimeMaps) are served stale; whole-profile hashing over-invalidates | Canonical key with declared profile slices and input identities; I-1 with a perturbation contract (`design.md` §4.8, `testing.md` §4) | resolved in design |
| R-2 | A | Cross-record batching makes a record's result depend on its batch-mates | I-3: declared and certified batch invariance, otherwise canonical buckets in the key or batch size 1 | resolved in design |
| R-3 | A | A coordinator restart re-plans and mixes commits; two coordinators run at once | Immutable hashed Plan, `resume(plan_hash)`, lease with a fencing token, routes and loop rounds stored as results (`execution.md` §3.1) | resolved in ADR-003 |
| R-4 | A, C, F | Env identity is wrong both ways: cross-platform locks, an undefined host digest, and the worker version in the requirements-sha (per-release invalidation, per-commit rebuild storms) | I-2: numerics identity separate from protocol identity; worker runtime as an overlay keyed by source hash | resolved in design |
| R-5 | A | Infrastructure faults are charged to items; one failure fails a whole batch | Fault taxonomy, per-item batch results, bisection (`execution.md` §3.3) | resolved in design |
| R-6 | A | Composite caching goes stale; environment-caused `Absent` is cached | I-5 (Merkle composite key), I-6 (cause class on `Absent`) | resolved in design |
| R-7 | A | Shards cut groups; chunk length changes outputs | I-7: shards closed under grouping, member-manifest hash, declared chunk grid | resolved in design |
| R-8 | A | Resident-model affinity starves other queues; GPU packing OOMs | A resident model is a cache, not a reservation; bounded quantum; per-device admission arbiter (`execution.md` §3.4) | resolved in ADR-003 |
| R-9 | A | Service restarts keep a stale identity; scale-to-zero kills a live service | I-8 attestation; services as leased endpoints, alive while consumers are pending (`execution.md` §3.5) | resolved in ADR-003 |
| R-10 | A, C | License taint does not propagate; CI gate and run-time gate differ; two policy schemas | I-9; one schema `{allow, on_missing}`; CI gate equals the run-time default (`design.md` §4.13) | resolved in design |
| R-11 | A | Fake stack and generated contracts cannot see batching, fault or fairness failures | Fake stack with a declared fault model; limits-tier identity contracts (`testing.md` §3–4) | resolved in design |
| R-12 | A | Store inode explosion; stat-heavy cache-hit estimation | Packed tier with key index; hits estimated from the index (`execution.md` §3.7) | resolved in ADR-003 (packed tier deferred, design §6.4) |
| R-13 | A | Storage without atomic rename, exclusive link or a server clock breaks leases | Storage capability probe as a `SystemRequirement` (`execution.md` §3.6) | resolved in design |
| R-14 | A, U | Agents can relax policy, overspend, or reach arbitrary paths | Caps in the execution profile, acknowledged cost estimate, allowlisted roots, relaxations as role capabilities (`design.md` §4.10) | resolved in design |
| R-15 | U | No run lifecycle; `sl.run` means two things; `.map` return undefined | `plan.run() -> Run`; one meaning of `sl.run`; lazy `ResultSet` (`design.md` §4.7) | resolved in design |
| R-16 | U, C | Two call shapes; profile argument names; `@` overloaded | One call shape; `config`/`dataset_profile`/`execution`; `@rev` for models, `name:version` for profiles | resolved in design |
| R-17 | U, C | `fit` underspecified; two fitted-profile paths; `kind` enum missing `fit` | `fit(inputs, labels=)` returns a Profile artifact; decide tasks take `profile=`; single path (`design.md` §4.14) | resolved in design |
| R-18 | U | No discovery for pipelines and datasets, no graph schema, no error envelope or result references | `design.md` §4.10 | resolved in design |
| R-19 | U | Explicit fan-out ceremony; unspecified binding | Fan-out inferred from list ports; `over=` only for grouping keys; bind by name with `bind={}` | resolved in design |
| R-20 | U, C | "Nine nouns" false; design docs verbose | Honest vocabulary (§4.0); design.md cut and de-duplicated | resolved in design |
| R-21 | C, F | Three pydra PR numbering schemes; pydra's role stated three ways; fallback trigger contradictory | Proposal ids only; Route A stated once (`design.md` §4.9); gate at the start of Phase 3 | resolved in design |
| R-22 | C | `Env.host` "whatever is installed" contradicts P-25 | Host = locked senselab venv with the I-2 identity | resolved in design |
| R-23 | C, F | Phase exits depend on later phases (planner, BIDS adapter, vLLM) | Minimal execution-profile resolver and `Env.server` in P2; minimal BIDS adapter in P3 | resolved in design |
| R-24 | C | `Signal`/P-26 never built; scale ladder, prose lint and adversarial rounds not scheduled | `Signal` in P1, `sl.signal` in P4, ladder in P3, rounds per phase | resolved in design |
| R-25 | C | Test tiers, cadence and cache-key definitions disagree across documents | P-30 names the tiers; one cadence table (`testing.md` §5 = `benchmarks.md` §5); one key | resolved in design |
| R-26 | C | Check fields missing or contradictory (P-7, P-17, P-18, P-19, P-20, P-23, P-24, P-25) | Fixed in `principles.md`, logged | resolved |
| R-27 | C | Facts: 37 `api.py` (36), 16 `extend_*` (15), `unrolled_contracts` location | Corrected | resolved |
| R-28 | F | Triage zero-changed-verdict gate can never close | Adjudicated-verdict gate (`design.md` §6.2) | resolved in design |
| R-29 | F | pydra on the critical path; fallback undersized | Branch via `uv`; gate at Phase 3 start; fallback 2–4k lines, 3–5 person-weeks | resolved in design |
| R-30 | F | Packaging pitfalls: tag prefixes, exact pin, workspace `requires-python` | Single tag, metadata-hook pin, worker floor tested from the wheel (`design.md` §5) | resolved in design |
| R-31 | F | Phase 2 silently changes triage; plan-fragment items break running campaigns | Phase 0 frozen 1.x tag; rewire triage per ported backend; new store schema (`design.md` §6.1) | resolved in design |
| R-32 | F | CI cost unaffordable | Cadence table: per PR min/max Python, changed locks; nightly and weekly for the rest | resolved in design |
| R-33 | F | License backfill blocks Phase 1 | Record type in P1; backfill per ported backend from P2 | resolved in design |
| R-34 | M | Nine factual errors in proposal 0001 (FIXME location, memoisation, sync `Worker.run`, dispatch sites, existing opt-outs, SGE `qsub_args`, audit fetch) | Proposal revised (`98965678` on `enh/execution-extensions`) | resolved |
| R-35 | M | Proposal API sketches clash with pydra 1.0 idioms; defaults would change; PR order needs splitting; senselab specifics leak | Proposal revised: `workflow.add(resources=)`, `Submitter(workers=, router=)`, `provided=` `Arg` attribute, Audit/messengers, defaults kept, A0…J2 split, generic examples | resolved |
| R-36 | A | Weight hashing implemented in both core and the worker runtime | Cross-implementation golden test in the env tier (`design.md` §4.12) | resolved in design |
| R-37 | C | Undefined: wire-protocol version, handshake format | Wire and handshake spec is a Phase 1 deliverable; the JSON Schema dialect moved to R-55 | resolved in wire-protocol.md |
| R-38 | P | senselab does not own its graph format; keys pass through pydra hashing; `Executor` is undefined | senselab defines the graph format, plan hash and keys; pydra is a compiler target and one executor (P §1) | resolved in ADR-001 |
| R-39 | P | Composite keys hash upstream keys, not outputs, so identical upstream output still invalidates downstream (I-5) | output-addressed early cutoff: downstream keys use upstream output identity (P §2) | resolved in ADR-002 |
| R-40 | P | Env identity covers the whole lock plus platform; near-zero hits across hosts and after lock bumps | per-backend dependency closures and measured equivalence classes (P §3, extends R-4) | resolved in ADR-002 |
| R-41 | P | Unkeyed inputs change outputs: decoder and resampler, system libraries, driver and cuDNN, backend source | key the decoder and resampler identity and the backend source hash; record and class the driver stack (P §4) | resolved in ADR-002 |
| R-42 | P | No canonical serialisation or evolution plan for stores, plans, profiles, provenance and fitted artifacts | canonical serialisation spec; format version on every artifact; migrations instead of wipes (P §5) | resolved in ADR-006 |
| R-43 | P | Signal model is one array and one rate; it cannot cover mixed-rate wearables, VFR video or long recordings | protocol-shaped data model with first-class time (P §7) | resolved in ADR-005 |
| R-44 | P | Identity of long and large inputs (hashing cost, chunked identity) | chunked content identity (P §8) | resolved in ADR-002 |
| R-45 | P | Plan conflates logical identity with physical placement | split the logical plan from placement (P §9) | resolved in ADR-004 |
| R-46 | P | Core is a runtime, not a kernel; dependency direction is unenforced below it | a small kernel plus enforced layering (P §10) | resolved in ADR-009 |
| R-47 | P | Observability is unspecified | logs, metrics and per-attempt diagnostics as a design section (P §11) | resolved in ADR-008 |
| R-48 | P | A model reference can make senselab download and run arbitrary hub code, including via MCP; provenance file labels can carry participant ids | a model-code trust boundary; store privacy; path redaction in exports (P §12) | resolved in ADR-007 |
| R-49 | P | Migration has no strangler boundary or coexistence budget | migrate one use case at a time, with a time-boxed coexistence budget (P §13) | resolved in ADR-009 |
| R-50 | P | Route and loop predicates are not tasks | predicates are declared decide tasks (P §14) | resolved in ADR-001 |
| R-51 | P | Implicit fan-out from list-typed ports is a typing trap (from R-19) | explicit fan-out (P §15) | resolved in ADR-001 |
| R-52 | P | Fault classification must originate in the worker | typed faults raised at the source (P §16, refines R-5) | resolved in ADR-009 |
| R-53 | P | Over-engineering for 2.0 | cut list (P §17) | resolved in design §6.4 |
| R-54 | P | Factual corrections, e.g. the model registry is read by no Python code, and the work pool starts one subprocess per item | fix the wording in design §2 (P §18) | resolved in design §2 (verified) |
| R-55 | R-37 | JSON Schema dialect for task schemas | a builder decision | open — builder decision, Phase 2 |
| R-56 | SDD | The generated `key_ignores_<field>` contract is structural: a context field is neither a `LogicalKey` field nor a `key_for` argument. It would miss a context value reaching the key through params, inputs, model or a planner-built key | once keys are built with an execution context in hand, the case builds the key under two contexts and asserts the same digest | open — owner: Phase 2 planner/executor |
| R-57 | SDD | Contract runners exist only for the `fake` stack; `contract_cases` refuses any other stack (`NoContractRunner`), because `BackendSpec` names no worker module | add a worker module or a stack→runner mapping to `BackendSpec` so `content_fault_isolated` runs a real backend's worker | open — owner: Phase 2, first non-fake backend |
| R-58 | SDD | `VERDICT_TYPES = {"Verdict"}` in `kernel/declare.py` names a type that does not exist yet, so `measure_has_no_verdict` is vacuous for real declarations and type-name based | replace the literal with the real decision output types, or derive the set from the type registry | open — owner: Phase 2 (first decide task or Verdict type) |
