# ADR-007: Trust and privacy

**Status:** accepted, 2026-10-30.

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
