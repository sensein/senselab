# Migration ledger (P-39)

One row per 1.x area: the last `alpha` commit whose 1.x changes are reconciled into 2.x. Update it
in the same PR as the reconciliation. A 1.x module may be deleted only when its mark equals
`alpha`.

Base: the design branch was cut from `alpha` at `19834fdd`. No commits had landed on `alpha`
since then as of 2026-10-30.

| Area (1.x paths) | Mark | Reconciled changes (commit → 2.x handling) |
|---|---|---|
| `src/senselab/audio/tasks/` | `19834fdd` | — |
| `src/senselab/video/tasks/`, `src/senselab/text/tasks/` | `19834fdd` | — |
| `src/senselab/audio/workflows/audio_analysis/` | `19834fdd` | — |
| `src/senselab/audio/workflows/triage/` | `19834fdd` | — |
| `src/senselab/utils/` | `19834fdd` | — |
| `scripts/`, `tutorials/` | `19834fdd` | — |
