# ADR-007: Trust and privacy

**Status:** accepted 2026-10-30; amended 2026-11-01.

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
