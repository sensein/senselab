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
