# ADR-010: Provenance is W3C PROV

**Status:** accepted 2026-11-01 (owner: "use w3c prov, add ADR-010"); amended 2026-11-01 (four times).

## Context

- 2.x core carried over the 1.x provenance store. That store uses its own record classes (`Entity`,
  `Activity`, `Agent`, `Environment`), a closed `PROV_TYPE` list that was mostly triage vocabulary,
  relations held as bare `(relation, source, target)` tuples, and its own JSONL format
  (`provenance@2.0`). W3C PROV appears only at export, where `bep028.py` maps the custom model onto
  `prov:` terms.
- No ADR chose that model. It came over in the port.
- There are no plans in it: no `prov:Plan` and no qualified association, although the pipeline graph
  (ADR-001) is exactly the plan a run follows.

## Decision

- **The model is W3C PROV-DM.** Every record is a `prov:Entity`, `prov:Activity` or `prov:Agent`, or a
  PROV subtype (`prov:Plan`, `prov:Collection`, `prov:SoftwareAgent`). Relations are PROV relations,
  including the qualified forms (`prov:qualifiedAssociation`, `prov:qualifiedUsage`,
  `prov:qualifiedGeneration`) wherever a role, time or plan is attached.
- **One serialisation: PROV-O in JSON-LD.** It uses a published senselab `@context`. The store and the
  export are the same document. BEP028 is the profile it conforms to, not a separate translation.
  `provenance@2.0` and its JSONL format are retired, with no reader for them (P-14).
- **Domain types are a vocabulary, not a list in core.** senselab terms live in a `senselab:`
  namespace, are typed as subclasses of PROV classes, and are published with the context. Core defines
  only the PROV structure and the terms every run needs: the run, the software agent, the environment,
  the plan and the content-addressed file. Tasks and pipelines add their own terms. Core holds no
  workflow-specific type (H-R48).
- **The pipeline graph is the plan.**
  - A run's graph document (ADR-001) is recorded as a `prov:Plan` entity. Its identifier is the plan
    hash.
  - Each activity is associated with the agent that executed it through `prov:qualifiedAssociation`,
    with `prov:hadPlan` pointing at the graph and the node id carried as the association's role.
  - A logical-plan node (ADR-004) and the activity that ran it are linked through the plan, not through
    a field of the activity.
- **Identity stays as decided.**
  - Identifiers are tagged digests (ADR-002, ADR-006): an entity's `@id` is the content address of what
    it names, and an activity's is its logical key.
  - Canonical encoding (JCS) is what is hashed.
  - JSON-LD is how it is written; it is never re-expanded to compute identity.
- **Attributes are namespaced.** A senselab attribute is a `senselab:` property with a declared range.
  An attribute outside the vocabulary is refused at the boundary (ADR-006's value rules still apply).
- **Privacy is unchanged.** Paths and file names are redacted to entity-free labels by default (ADR-007).

## Consequences

- Provenance can be read by general PROV tooling and by BIDS without senselab installed.
- `ProvStore` becomes a writer and reader of PROV-O JSON-LD. Several pieces disappear: `bep028.py`
  stops being a translation layer, the closed `PROV_TYPE` list, and the custom record shapes.
- The vocabulary file is part of the release, and the release gate checks it.
- Every provenance tag changes, so the next store tag is a new major.

## Alternatives rejected

- **Keep the custom store and translate at export.** Two models stay in step by hand. The export is
  lossy wherever the custom model has no PROV term, and plans cannot be expressed.
- **PROV-JSON instead of PROV-O / JSON-LD.** It is not linked data, it is not what BEP028 publishes,
  and it has no namespacing for a domain vocabulary.
- **A closed enum of senselab types in core.** It recreates the triage-vocabulary problem: every new
  task edits core.

## Amendment 2026-11-01: activity identity, a staged plan link, Phase 2 (D-4)

Owner decision D-4 ([`decisions-2026-11-01.md`](../decisions-2026-11-01.md)). Resolves A I-3, A I-2,
A I-6 ([`reviews/full-review-2026-11-01/adrs.md`](../reviews/full-review-2026-11-01/adrs.md)) and
Ru C-1 ([`rulings-audit.md`](../reviews/full-review-2026-11-01/rulings-audit.md)).

- **An activity's `@id` identifies the execution.** Replaces "and an activity's is its logical key" under
  *Identity stays as decided*. The `@id` is the content address of the execution record. The logical key
  (ADR-002) is a `senselab:` property of the activity. One key can carry several executions, one per
  execution context, plus reuse activities; an `@id` equal to the key would merge them.
- **The plan link is staged.** Replaces the timing of *The pipeline graph is the plan*, whose decision
  stands. `prov:hadPlan` is recorded once ADR-001's graph document exists (Phase 3). Until then an
  activity carries its node id and its logical key, and no `prov:hadPlan`.
- **Phase 2 implements this ADR** (design §6.2):
  - the PROV-O JSON-LD store;
  - the `senselab:` vocabulary;
  - removal of the token families and helpers carried over from 1.x provenance (`$path`, `$enum`,
    `$dec`, `$bytes`, `$set`, `EnumMember`, `FrozenDict`, `revised`, `recorded`, `plain_values`).
    `$enum` hashes a Python qualified name into durable ids, which ADR-006 forbids;
  - the export redaction ADR-007 requires; today's BEP028 exporter writes file names and paths.

## Amendment 2026-11-01: the vocabulary check is staged (P2-40)

Ruling P2-40 ([`plans/phase-2.md`](../plans/phase-2.md)); implemented in Phase 2 Task 3a.

- **Replaces** "the release gate checks it" under *Consequences*. Until the first 2.x release the vocabulary is
  checked by a contract test: the packaged `@context` and `senselab:` vocabulary (`senselab-vocab@1.0`) expand
  under pyld through an offline document loader serving the packaged context; every expanded term is a `prov:` or
  `senselab:` IRI, no term is dropped and no relative IRI survives (P2-58). From the first 2.x release the release
  gate runs that check; it joins D-5's deferred list (design §5).
- **Identifiers as built.** An activity's `@id` is `activity:` + `identity_digest("prov-activity@1", <compact node
  without @id>)` (P2-6). An agent's is `agent:` and an environment's `environment:` +
  `identity_digest("prov-node@1", <compact node without @id>)`: they are addressed over their recorded fields, and
  `prov-node@1` is a separate tag because ADR-006 gives each kind of identity its own. An entity's is its content
  address, and an array value is a `senselab:File` entity addressed by its canonical `.npy` digest (P2-5).
- **Entities carry no relation.** An output's address depends on its content alone, so two executions may generate one
  entity. Every relation is written on the activity: `prov:used`, `prov:generated` and `prov:wasAssociatedWith`, with
  `prov:qualifiedUsage` and `prov:qualifiedAssociation`, and the entity's `prov:qualifiedGeneration` written inside the
  activity's node through JSON-LD `@reverse`. An entity node is then the same in every writer's document.
- **Core terms.** Classes `senselab:Run`, `senselab:SoftwareAgent`, `senselab:Environment`, `senselab:File`,
  `senselab:Plan`; properties `senselab:logicalKey` (on an activity, `rdf:JSON`), `senselab:modelSource` (on an
  agent, P2-74), and `senselab:nodeId` and `senselab:port` (on a `prov:Role`: the association's node id, and the port a
  usage or generation was on).
- **Numbers follow their range.** A numeric literal takes the form its property's declared range gives it, not its
  Python type: `xsd:decimal` is always the typed decimal string, `xsd:double` a JSON number (non-finite as the
  typed `INF`, `-INF`, `NaN`), `xsd:integer` a JSON integer, or the typed string beyond +/-(2**53 - 1). A value the
  range cannot hold exactly is refused. Equal numbers then give one node, and one `@id`, whatever type a writer held.
- **Ids are checked addresses.** A record is built only by the builders, and a document refuses any node whose `@id`
  is not a `content:`, `output:`, `activity:`, `agent:` or `environment:` address; an error names the argument,
  never the refused id.
- **Times are inside the activity's address.** `prov:startedAtTime`, `prov:endedAtTime` and `prov:atTime` are part of
  the hashed node, so two writers give one activity only with identical times. The executing side builds the
  activity once (Task 3b), and every other writer copies its bytes rather than rebuilding it.
- **The namespace** is `https://w3id.org/senselab/ns#` and the context `https://w3id.org/senselab/ns/context.jsonld`.
  It retires `https://senselab.sensein.group/prov#` (`SENSELAB_VOCABULARY` in `runtime/bep028.py`, and the triage
  BEP028 design), which is deleted with that exporter in Task 4. The w3id redirect is an external registration the
  owner confirms before it is filed, before the first 2.x release; no test dereferences either IRI.

## Amendment 2026-11-01: the store as built (P2-4, R-149, R-150)

Rulings P2-4, P2-6; implemented in Phase 2 Task 3b.

- **One document per writer.** A writer holds the records it adds and replaces its own document,
  `<run>/prov/<writer uuid4 hex>.jsonld`, atomically with the JCS text of `{"@context": ..., "@graph": [...]}`.
  A run's provenance is the union of its writers' documents by `@id`; two different nodes under one `@id` are refused
  (`ConflictingNode`). A temporary of an interrupted write is skipped; any other file in `prov/` is refused. A writer
  is one process: a store used after `fork` mints a new writer id in the child, because children sharing the parent's
  id would replace one another's document and lose records. An `add` that meets a conflict adds nothing.
- **A pool worker writes when it finishes a claim.** A worker's store holds its own `ProvStore`, so the executions and
  `senselab:Reuse` activities of its items are the worker's records. It replaces its document, as above, when each
  claim ends and when it returns, so a worker that dies loses at most the claim it is running or has just committed (its results stand). Its document also holds the
  run's `senselab:Plan` entity, which the coordinator adds too (equal bytes, one node), so a worker's document names
  only nodes the union holds even before the coordinator writes.
- **A reader rebuilds every node.** Each node read is rebuilt through the record builders over the reader's vocabulary
  and must equal its bytes, so an `@id` that is not the node's address, a term outside the vocabulary or a literal
  outside its range is refused (`MalformedDocument`) rather than wrapped as a record. Read errors name the writer and
  a checked `@id`, never a file name, a refused id or a value; an OS error is reported by its class name, and a
  literal the reader cannot parse by `literal_value`'s non-echoing `ValueError`. Attribute references nest at most
  `MAX_DEPTH` deep, the canonical depth rule.
- **Endpoints are checked over the union (R-150).** Every `prov:used`, `prov:generated` and `prov:wasAssociatedWith`
  endpoint, and every record an attribute names, must be a node of the union (`DanglingReference`); an endpoint may sit
  in another writer's document. A writer's own document is not checked alone.
- **Accessors return in `@id` order (R-149),** so `merge` gives the same store for any order of its inputs.
- **An activity is copied, not rebuilt.** Its times are in its address, so a second writer adds the record the executing
  side built (or one read back) and the union holds one node.
- **Until the result store moves over** (Task 3c), the `provenance@2.0` store stays beside this one as `JsonlProvStore`
  in `runtime/provenance_jsonl.py`; Task 3c deletes that module, and registers the document's format, `provenance@3.0`.

## Amendment 2026-11-01: the result store records PROV (P2-57)

Rulings P2-44, P2-57; implemented in Phase 2 Task 3c.

- **The store writes the result entity and the reuse activities, nothing else** (P2-57).
  - A `put` adds the result's entity, `output:<hex>`, with no attributes.
  - Every `get` hit, `exact` included, adds a `senselab:Reuse` activity that used that entity, with its start
    and end times. It carries `senselab:logicalKey`, `senselab:context` (the current context),
    `senselab:storedContext`, `senselab:reusePolicy`, and `senselab:equivalence` when `equivalent` admitted it.
  - A call is then one activity whether it executed or reused. The execution activity is `run_call`'s
    (Task 10b).
  - A quarantine is logged, not recorded. Its record held the entry's path.
- **Core terms added.** These extend the "Core terms" list of the P2-40 amendment.
  - `senselab:Reuse` is a subclass of `prov:Activity`.
  - `senselab:context` is on `prov:Activity`, with range `rdf:JSON`: an `execution-context@2.0` record. An
    execution activity uses the same term.
  - `senselab:storedContext` (`rdf:JSON`), `senselab:reusePolicy` (`xsd:string`) and `senselab:equivalence`
    (`xsd:string`, the table's `value@2` digest) are on `senselab:Reuse`.
  - `senselab:executionProfile` is on `prov:Activity`, with range `xsd:string`: the `canonical_digest` of the
    execution profile an execution ran under (Phase 2 Task 10b).
  - `run_call`'s execution activity is associated with a `senselab:SoftwareAgent` whose `senselab:modelSource` is
    `hf:<org>/<name>@<commit>` for a hub model and the bare source kind (`url`, `local`) otherwise, so no path or
    URL enters it; a model-less call's agent carries none. The activity uses each input's `content:` entity and
    generated the result entity the store adds.
- **The document's format is `provenance@3.0`**, its top-level `format`: `{"@context": ..., "format":
  "provenance@3.0", "@graph": [...]}`. The published context maps `format` to null, so a JSON-LD processor
  ignores the key by declaration, not by dropping an unknown term.
  - A reader reads any minor of `provenance@3`.
  - A file in `prov/` whose first line declares any other format, `provenance@2.0` included, is refused with
    `ProvenanceFormatRefused`, which names the format and the migration statement (ADR-006 as amended, P2-44).
  - A writer's document without a readable `format` is `MalformedDocument`.
- **Deleted with `provenance@2.0`:**
  - `runtime/provenance_jsonl.py` and its tests;
  - `runtime/bep028.py` and its tests (Task 4 writes the profile check);
  - `testing/provenance.py`;
  - the token families `$path`, `$enum`, `$dec`, `$bytes` and `$set`, and `$nd`;
  - `EnumMember`, `FrozenDict`, `FrozenList`, `revised`, `recorded`, `plain_values`, `PROV_TYPE` and the
    checksum, path, size and mtime keys.
- **One frozen mapping type.** A built `Frozen` value's `__dict__` is a plain `dict`. Attribute writes are
  refused, and its digest is computed when it is built; a write through `__dict__` itself is not refused.

## Amendment 2026-11-01: the BEP028 profile (ruling P2-8; Phase 2 Task 4)

Source: bids-standard/BEP028_BIDSprov, `bep028spec.md` at master 162bea57 (2026-10-07), licence CC-BY-4.0.

- **What the draft states.**
  - Records go in `prov/` below the dataset root as `prov-<label>_<suffix>.<json|jsonld>`, suffix one of `act`,
    `ent`, `agt`, `soft`, `env`, `base`, `all`; `all` is `.jsonld` (3.1.3, 3.2.1).
  - A document has `@context`, `BIDSProvVersion` and `Records` (1.4); `.jsonld` content is JSON-LD.
  - Provenance SHOULD be anonymised (1.3).
- **What senselab does.** `export` copies the run's PROV-O documents, unchanged, as `prov/prov-<writer>_all.jsonld`
  (the writer's uuid4 hex is the label). It meets the naming rule, `@context`, JSON-LD and anonymisation (ADR-007).
  The draft's own record vocabulary (`Records`, `Activities`, `Id`, `Label`, `Used`) is not PROV-O and is not
  produced: that would be the translation this ADR rejects. `format` stands where `BIDSProvVersion` does. A BIDS
  validator reading the draft's vocabulary will not read these documents until the context is aligned (open).
- **`check_profile(doc)`** adds nothing to what `ProvStore.from_documents` already validates, except an activity's
  end needs a start and is not before it, and a `senselab:fileLabel` names one node.
- **Core term added:** `senselab:fileLabel`, on `senselab:File`, `xsd:string`.
- **`senselab:File` from an array has no label.** It names no path; only a path-sourced file
  (`DataManifest.file_entities`) gets one.

## Amendment 2026-11-02: env versions are agents; licenses are recorded (P2-92, P2-93; Phase 2 Task 34d)

- **The software agent is `agent_record(key, context)`**, shared by `run_call` and the store. It carries
  `senselab:envLock` (the env lock sha), `senselab:license` (the env's `License` record as `rdf:JSON`, when the
  context has one) and, with a model, `senselab:modelSource` as before. An env at another lock is another agent.
- **A reuse records both agents.** A `senselab:Reuse` activity is associated with the current context's agent
  (`prov:qualifiedAssociation`, role the task) and names the agent that computed the served result in
  `senselab:storedAgent` (`xsd:string`, its `@id`); both agent records are added. The stored agent is rebuilt
  from the stored context, so it is the same node the execution that computed the result recorded.
- **Core terms added** (`senselab-vocab@1.2`): `senselab:envLock` (`prov:Agent`, `xsd:string`),
  `senselab:license` (`prov:Agent`, `rdf:JSON`), `senselab:storedAgent` (`prov:Activity`, `xsd:string`).
  `senselab:context` and `senselab:storedContext` are `execution-context@4.0` records (was `@3.0`/`@3.1`; ADR-006 amendment 2026-11-08, Task 20R-S2).

## Amendment 2026-11-02: the thresholds of a verify decision are recorded (P2-95; Phase 2 Task 34e)

- **Core term added** (`senselab-vocab@1.3`): `senselab:verifyThresholds` (`prov:Activity`, `xsd:string`), the
  canonical digest of the comparison thresholds of the run config the call ran under. Every reuse and execution
  activity carries it beside `senselab:verifyFraction` and `senselab:verifySampling`; a sampled execution's
  `senselab:verifies` evidence record holds the same digest.

## Amendment 2026-11-06: the plan link is recorded (Phase 3 Task 11)

- **`prov:hadPlan` is recorded,** ending the staging of the 2026-11-01 amendment. Every execution and reuse activity of
  a planned job carries `prov:qualifiedAssociation` with `prov:hadRole` (`senselab:nodeId`, now the graph node id, no
  longer the task name) and `prov:hadPlan` → `plan:<plan hash>`. The plan is a `prov:Plan` / `senselab:Plan` entity
  with no attributes, written once per run by the run that starts it (and by a single call for its own plan);
  `plan:` joins the checked `@id` addresses, and a reader refuses a `prov:hadPlan` that names no plan of the union
  (`DanglingReference`). A reuse recorded outside a plan (a direct `ResultStore.get`) names its agent alone: no
  `prov:hadRole` and no plan, so `senselab:nodeId` always means a graph node id. No vocabulary term is added
  (`senselab:Plan` was a core term), so `senselab-vocab` stays at 1.4.
- **`plan.json` is written once** and holds the logical plan and input manifest only; each attempt's placement is
  `placements/<attempt>.json`, and a run's current placement is that of its latest claimed attempt (ADR-004).

## Amendment 2026-11-10: a reuse across the attempts of one run is a Reuse, whichever writer settles it (Phase 3 Task T-RESUME-REUSE)

- **Resuming a run reuses what the earlier attempt finished.** An item whose outcome on the run's queue was written by
  an earlier epoch is settled by the resuming coordinator from the store without being sent to a worker; that is a
  reuse in the sense of this ADR and is recorded as a `senselab:Reuse` activity. The writer is whichever process
  settles the item: the resuming coordinator, in its own store's provenance document, or a worker that serves it.
- **A settled reuse has its own shape.** The coordinator ran nothing, so its `senselab:Reuse` carries no
  `senselab:context` and no agent. It carries `senselab:storedContext` and `senselab:storedAgent` (the context and
  agent that computed the result), `senselab:logicalKey`, `prov:used` of the result, the verify settings and
  `senselab:reusePolicy`, the policy the item's outcome was served under in its own attempt (`exact`, or `equivalent`
  with `senselab:equivalence`). Its `prov:qualifiedAssociation` has `prov:hadRole` (`senselab:nodeId`, the graph node)
  and `prov:hadPlan` (`plan:<hash>`) and no `prov:agent`, and the activity has no `prov:wasAssociatedWith` (PROV-DM
  5.3.3: the agent of an association is optional when a plan is given). This amends the 2026-11-02 clause "a reuse
  records both agents" and the 2026-11-06 clause "every execution and reuse activity of a planned job carries an
  association with `prov:hadRole` and `prov:hadPlan`" only as far as the agent is absent; the node and the plan are
  recorded as before. No vocabulary term is added.
- **The bound.** A writer records at most one `senselab:Reuse` per item per attempt: the coordinator records an item
  once per graph node however often it reads the outcome. The union may hold a worker's Reuse and the coordinator's for
  one item when the worker's commit raced a surviving earlier-epoch worker that linked its outcome first (the worker's
  reuse was recorded before its commit lost); the two differ by time and are not merged.
- **The coordinator writes its document when it has settled items from an earlier epoch,** after each poll that
  recorded any and when its pool closes, with the run's plan entity. A failed write in a poll is logged and retried at
  the next poll and at close, so settled results are never lost to it; `close` raises if its final write fails, unless
  an error already ends the run.

## Amendment 2026-11-10: the Phase 3 head as built, checked (Phase 3 Task 22a)

Adds no rule. It records what the Phase 3 head holds, so the amendments above can be read against it.

- **Formats.** A run's documents are `provenance@3.0` (`kernel/formats.py:26`); the core vocabulary is
  `senselab-vocab@1.4` (`runtime/data/prov/senselab.jsonld:7`), the version the 2026-11-06 amendment names.
- **Writers.** Each writer replaces its own `<run>/prov/<writer uuid4 hex>.jsonld` and a reader takes the union by
  `@id` (`runtime/provenance.py:1-6`); the coordinator and each pool worker are writers (clauses above).
- **Checked on a run.** The provenance checker passed 11 of 11 on the store retrieved from the interrupted and resumed
  speech-to-text run, with 435 `senselab:Reuse` activities from the resume and at most one execution per key
  (`reviews/phase3-exit-evidence.md`, Step 4). That is evidence of the 2026-11-10 amendment above, not a new clause.
