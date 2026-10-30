# Review register

Each row records one review finding, how it was resolved, and its status. Ids `R-n` are never
reused. New rows come from the adversarial round that ends each phase (P-35).

Reviews of 2026-10-30, all against `senselab-2x-api`, each with a different lens:
- **C**: consistency;
- **A**: adversarial;
- **U**: usability;
- **F**: feasibility;
- **M**: pydra maintainer.

| Id | Src | Finding | Resolution | Status |
|---|---|---|---|---|
| R-1 | A | Inputs a node reads that are not in its key (run-config thresholds, fitted profiles, hints, TimeMaps) are served stale; whole-profile hashing over-invalidates | Canonical key with declared profile slices and input identities; I-1 with a perturbation contract (`design.md` §4.8, `testing.md` §4) | resolved in design |
| R-2 | A | Cross-record batching makes a record's result depend on its batch-mates | I-3: declared and certified batch invariance, otherwise canonical buckets in the key or batch size 1 | resolved in design |
| R-3 | A | A coordinator restart re-plans and mixes commits; two coordinators run at once | Immutable hashed Plan, `resume(plan_hash)`, lease with a fencing token, routes and loop rounds stored as results (`execution.md` §3.1) | resolved in design |
| R-4 | A, C, F | Env identity is wrong both ways: cross-platform locks, an undefined host digest, and the worker version in the requirements-sha (per-release invalidation, per-commit rebuild storms) | I-2: numerics identity separate from protocol identity; worker runtime as an overlay keyed by source hash | resolved in design |
| R-5 | A | Infrastructure faults are charged to items; one failure fails a whole batch | Fault taxonomy, per-item batch results, bisection (`execution.md` §3.3) | resolved in design |
| R-6 | A | Composite caching goes stale; environment-caused `Absent` is cached | I-5 (Merkle composite key), I-6 (cause class on `Absent`) | resolved in design |
| R-7 | A | Shards cut groups; chunk length changes outputs | I-7: shards closed under grouping, member-manifest hash, declared chunk grid | resolved in design |
| R-8 | A | Resident-model affinity starves other queues; GPU packing OOMs | A resident model is a cache, not a reservation; bounded quantum; per-device admission arbiter (`execution.md` §3.4) | resolved in design |
| R-9 | A | Service restarts keep a stale identity; scale-to-zero kills a live service | I-8 attestation; services as leased endpoints, alive while consumers are pending (`execution.md` §3.5) | resolved in design |
| R-10 | A, C | License taint does not propagate; CI gate and run-time gate differ; two policy schemas | I-9; one schema `{allow, on_missing}`; CI gate equals the run-time default (`design.md` §4.13) | resolved in design |
| R-11 | A | Fake stack and generated contracts cannot see batching, fault or fairness failures | Fake stack with a declared fault model; limits-tier identity contracts (`testing.md` §3–4) | resolved in design |
| R-12 | A | Store inode explosion; stat-heavy cache-hit estimation | Packed tier with key index; hits estimated from the index (`execution.md` §3.7) | resolved in design |
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
| R-37 | C | Undefined: wire-protocol version, handshake format, JSON Schema dialect | Wire and handshake specs are Phase 1 deliverables; the dialect is a builder decision | open (Phase 1) |
