# ADR-010: Provenance is W3C PROV

**Status:** accepted 2026-11-01 (owner: "use w3c prov, add ADR-010"); amended 2026-11-01.

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
