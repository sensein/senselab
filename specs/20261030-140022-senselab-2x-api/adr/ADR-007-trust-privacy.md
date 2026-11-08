# ADR-007: Trust and privacy

**Status:** accepted 2026-10-30; amended 2026-11-01, 2026-11-02, 2026-11-08.

## Context
Architect review §12 (R-48):
- any model reference could make senselab download and run arbitrary hub code, including through
  MCP;
- provenance exports carried file labels that can contain participant ids;
- shared stores had no access model.

## Decision

**Model code**
- **No implicit remote code.** Model sources (hub organisations, URLs, local roots) are allowlisted
  in the execution profile.
- **Remote code runs only in an isolated env,** with the network guard on after its weights and
  code are downloaded and pinned. Weights are loaded by identity (ADR-002).
- **MCP and agent roles cannot add sources** or relax the allowlist (design §4.10).

**Privacy**
- **Keys and identities are hashes** and carry no identifiers.
- **Provenance and BEP028 exports redact paths and file names** to entity-free labels by default.
  The mapping from labels to paths stays in the private run directory.
- **Stores are per project, with filesystem permissions.**
  - A shared store does not mix projects that have different data agreements.
  - Cross-project reuse is an explicit copy, recorded in provenance.

## Consequences
- A first use of a new model source needs a one-line profile change.
- Exports are publishable by default.

## Alternatives rejected
- **Trusting `trust_remote_code` per call.** Unauditable.
- **Redacting at publish time only.** Too easy to forget.

## Amendment 2026-11-01: export redaction is built with ADR-010 (D-4)

Owner decision D-4 ([`decisions-2026-11-01.md`](../decisions-2026-11-01.md)); finding A I-2
([`reviews/full-review-2026-11-01/adrs.md`](../reviews/full-review-2026-11-01/adrs.md)). The decision
stands. The Phase 1 BEP028 exporter writes file names and paths and does not meet it, so no Phase 1
export is publishable. The redaction is built in Phase 2, with the PROV-O store of ADR-010, and the
exporter's tests require it.

## Amendment 2026-11-01: redaction as built (P2-8; Phase 2 Task 4)

- **Scope.** Paths may appear in local operator logs and in storage-infrastructure errors (`StorageDir`,
  `storage_fault`, `WrongKindError`, the store's corrupt-entry errors). They never appear in a PROV document, in an
  export, or in an error raised by the provenance read, merge and write APIs or the export API.
- **A file is named by its content address and a label.** A `senselab:File` entity carries `senselab:fileLabel`,
  `file-<n>`, allocated per run, one per content address; a second path of the same bytes is added to that label's
  paths. The grammar is checked where a record enters a store (`ProvStore.add`) and where one is read back.
- **The map is private.** `<run>/private/labels@1.json` (`labels@1.0`, mode 0600, in a 0700 directory) holds
  `{label: {address, paths}}`, written under a lease. No document holds it, and `export` never copies it.
- **`senselab:modelSource` is a source kind and an identity, never a local path** (ADR-002); a path-shaped value
  (absolute, `~`, `./`, `../`, a drive letter, a backslash) is refused at the same boundary.
- **`export(run, dest)`** reads the run's documents once, validates, profile-checks and scans those bytes, and
  copies exactly them to `<dest>/prov/prov-<writer>_all.jsonld`. The scan refuses a document with any string, key
  included, equal to a segment or stem of a path in the label map however it is spelt (`./`, `//`, backslashes), or
  holding a BIDS `sub-`/`ses-` label. Nothing is copied on a refusal (`ProfileViolation`).
- **Errors.** `ProvStore.write` and the label map raise `RunIOError`: the errno text, no filename, cause or context.
  The original error is logged for the operator. An unreadable `prov/` raises `MalformedDocument` naming an error
  class.
- **Limits.** A path never recorded in the map and carrying no `sub-`/`ses-` label, such as an operator's model
  directory in a free-text attribute, is not detected. It belongs to the writer; `modelSource` is the one such
  attribute core refuses a path in. A per-run `senselab:fileLabel` sits inside a content-addressed node, so the same
  `content:` entity labelled differently in two runs conflicts in a cross-run union; export stays per run.

## Amendment 2026-11-02: catalogue metadata binds to the pinned revision (Phase 2 Task 34a)

- A catalogue entry's license, placement and resources describe `(id, revision)`, not `id`. A model reference that
  resolves to the entry's revision carries them; one that resolves to any other commit (`org/model@main` after the
  repository moved) carries `license: None`, unverified, and is placed as an uncatalogued model (R-176).

## Amendment 2026-11-08: identifier visibility is per audience

Owner, 2026-11-08: "identifier specification may depend on use-case, that shouldn't be a blanket rule but
dependent on who sees the identifiers". [`user-scenarios.md`](../user-scenarios.md) (Terms, UX-28, UX-45,
UX-50) governs; this amendment brings the decision in line.
- **Policy.** Visibility of participant and record identifiers (and of the accounts, hosts and sites that ran
  the work) is decided per audience by a policy the data owner sets for a project or dataset: real values,
  pseudonymous ids, or none. An AI agent sees what its principal may see.
- **Default.** With no policy set, the data owner sees real identifiers and every other audience sees
  pseudonymous ids.
- **Pseudonyms are stable per dataset**: one identifier maps to one pseudonym across runs, outputs and every
  audience allowed pseudonyms. This supersedes "allocated per run" for labels shown to an audience.
- **Exports and shared reports render for their recipient audience**, never above what the policy allows that
  audience. "Publishable by default" and the export refusal of paths and `sub-`/`ses-` labels describe the
  rule for a public audience, not for every recipient.
