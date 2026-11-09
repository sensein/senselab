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

## Amendment 2026-11-09: placement records (Phase 3 Task 20R-12)

- The writers of a run's readable part are unchanged: the run's event log carries the `placement` events. Partitions,
  accounts, node names (as written and as expanded from `${VAR}`), the site hook's `why` and the notes are rendered
  under the audience rule by one helper before they are stored or raised, for every source of the value (profile,
  override, hook answer), a value a later layer replaced included, as is the text of `SiteRefused`; both render from
  one set of names collected as the layers are applied. The operator's text goes to the log.

## Amendment 2026-11-08: the host token's guarantee (Phase 3 Task 20R-S1)

- `host_token()` is a digest of a random per-host, per-boot value (Linux boot id, macOS boot session uuid, else a
  private file in the host's temporary directory), never of the host name. It says only whether two records share a
  host on one boot.
- Anyone with a shell on that node can compute the token and so link records to the node; a group reader without
  node access cannot name the host from it, and can tell that two runs shared a host on one boot. That is the stated
  guarantee. Naming a host to its user is the audience rule above, rendered from the user's local state.
- Service registries, vLLM start locks and server logs are keyed by `host_token()`, so no shared path or file holds a
  host name.
- A run's readable part is written only by the run, its event log and its queues, as path-free documents; every
  substrate's state (pydra's pickles, Slurm worker logs, the run's own profile file, placements, manifest, inputs) sits
  under `<run>/private/`, mode 0700. On an NFSv4-ACL filesystem an inherited ACE may survive that `chmod`; that is
  residual, and a site that shares runs over such ACLs restricts the parent.
- A stored fault message renders this host's name to `<host>`. IP address literals in a fault message are not
  rendered; they are out of this amendment.
