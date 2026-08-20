# Archive

Superseded documents, kept because dated notes elsewhere cite them and because deleting an argument
makes it look like nobody made it.

**Nothing in the current design depends on any of these.** The live DAG is `admit.md`, `preprocess.md`,
`store.md`, `taxonomy.md`, `branch-airway.md`, `branch-speech.md`, `branch-voice.md`, `redact.md`,
`verdict.md`, with derivations in `benchmarks/`.

| file | superseded by | why |
| --- | --- | --- |
| `design.md` | the node documents | it indexed a graph of phases against a findings register, which is not the goal. Its §8 required-measurements list is triaged in [`../benchmarks/open.md`](../benchmarks/open.md) |
| `decisions.md` | the node documents | D1–D27 were taken against the round-based workflow. Six (D6, D8, D11, D12, D16, D19) are still cited from dated notes in the parent directory, which is why this file is kept rather than deleted |
| `ports.md` | [`../store.md`](../store.md) | it defined a normative port contract between nodes. Nodes now write to an append-only store and read what they find, so a declared-port discipline is not the interface any more |

Read these as a record of what was decided and when, never as a description of the current graph.
