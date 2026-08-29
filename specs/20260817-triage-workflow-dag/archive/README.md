# Archive

Superseded documents, kept because dated notes elsewhere cite them and because deleting an argument
makes it look like nobody made it.

**Nothing in the current design depends on any of these.** The live DAG is `admit.md`,
`preprocess.md`, `store.md`, `taxonomy.md`, `branch-airway.md`, `branch-speech.md`,
`branch-voice.md`, `redact.md`, `verdict.md`, `routing.md`, and `report.md`, with derivations in
`benchmarks/` and the current call graph in [`../dag.md`](../dag.md).

| file | superseded by | why |
| --- | --- | --- |
| `design.md` | the node documents | it indexed a graph of phases against a findings register, which is not the goal. Its §8 required-measurements list is triaged in [`../benchmarks/open.md`](../benchmarks/open.md) |
| `decisions.md` | the node documents | D1–D27 were taken against the round-based workflow. Six (D6, D8, D11, D12, D16, D19) are still cited from dated notes in the parent directory, which is why this file is kept rather than deleted |
| `ports.md` | [`../store.md`](../store.md) | it defined a normative port contract between nodes. Nodes now write to an append-only store and read what they find, so a declared-port discipline is not the interface any more |
| `plan-foundation.md`, `plan-nodes-1.md`, `plan-nodes-2.md`, `plan-review.md`, `plan-v2-1.md`, `plan-v2-2.md` | the node documents and the shipped code | agentic-worker task checklists for building the store, the DSP tasks, and the nine nodes. Every task is built; a plan is a to-do list for work that no longer needs doing, not a description of what shipped. `plan-foundation.md`'s Task 1 (PII finding offsets) is the one open item still tracked, from [`../benchmarks/open.md`](../benchmarks/open.md) |
| `routing-failure.md` | [`../routing.md`](../routing.md)'s "When ROUTING itself fails" | a scoped, completed checklist for one failure path. Its decision is now stated directly in the canonical doc; nothing here it didn't already say |
| `unvoiced-phonation-routing.md` | [`../preprocess.md`](../preprocess.md) | a scoped, completed checklist for the non-periodic formant-continuity limb and the word-aligned phonation path. Both are now documented in `preprocess.md`'s own phonation-span section, verbatim down to the `unvoiced_max_formant_bandwidth_hz` and `word_aligned_min_evidence_fraction` config keys |
| `consensus-timing-authority-2026-08-26.md` | [`../taxonomy.md`](../taxonomy.md), [`../preprocess.md`](../preprocess.md) | a completed checklist deciding that consensus ASR, not a second forced-alignment pass, is the sole timing authority. Both canonical docs now state this directly ("the consensus transcript is the authoritative ASR product"; PREPROCESS runs no alignment pass) |

Read these as a record of what was decided and when, never as a description of the current graph.
