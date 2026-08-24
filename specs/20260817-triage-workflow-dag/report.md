# REPORT — what every run emits

Runs last, after [`verdict.md`](verdict.md). **Every run emits both products below, on every file, on
every outcome** — including a file ADMIT refused, where the report says that and nothing else.

## Signature

```
report(store, verdict) -> artifacts
```

Reads the [element store](store.md). Writes artifacts beside the store; **it writes no elements and
asserts nothing**. A rendering is not evidence, and nothing downstream reads the report to learn a
fact the store does not already hold.

## Two products

| product | form | what it is for |
| --- | --- | --- |
| **summary** | one PDF **or** one image per file | a human reading one recording |
| **summary JSON** | one JSON per file | a consumer reading many |

The two carry the same claims. The image form exists for a file whose summary fits on one page; the
choice is the config key `report.format` and it does not change the content.

## The summary — one file, one page

On a single shared time axis:

| layer | drawn from |
| --- | --- |
| waveform and energy envelope with its floor | PREPROCESS |
| every `span`, with its `peak_over_floor_db` | PREPROCESS |
| `phonation_spans` and glides, with `duration_s` and production mode | PREPROCESS |
| window label sets — YAMNet, AST, HeAR — as label lanes over their own grids | PREPROCESS |
| speech spans with their speaker attribution and any `nontarget` marking | SPEECH |
| airway-labelled spans with their labels and confirmations | AIRWAY |
| voiced runs and their extents | VOICE |
| redacted extents | REDACT |

Beside the axis, one block per step:

- **which branches ran, which were skipped, and which a hint forced** — the `branch_decision`
  elements;
- **each branch's conclusion and its flags**;
- **TAXONOMY's classification beside the resolved kinds**, with the per-kind agreement or mismatch;
- **the verdict**: `triage`, `release`, and every reason, with a REDACT non-pass shown whatever the
  triage axis says.

**The summary respects the PII marking.** A `word` element the scan marked is rendered redacted, and
no matched text appears anywhere in either product. Neither product is a **released artifact**: both
carry element ids, which are a join key back into the store.

## The summary JSON

The verdict's product ([`verdict.md`](verdict.md)), plus per-step summaries, plus **provenance
embedded rather than referenced**:

```
{
  file:        { path, duration_s, sample_rate, channels },
  verdict:     { triage, release, reasons[], kinds{}, screened{}, agreement{}, hints{} },
  branches:    { branch: { will_run, forced_by_hint, kind_state, verdict?, flags[] } },
  steps:       { step: { summary fields, element_ids[] } },
  provenance:  {
    config_hash: str,                  # the merged config mapping's hash
    config:      { ... },              # the merged mapping itself
    commit:      str,                  # the senselab commit the run was made at
    models:      [ { model_id, revision, task, node } ],   # every agent, with its resolved commit
    run_id:      str,
    started, ended
  }
}
```

- **`config_hash` and the merged mapping both appear.** A hash identifies a run; the mapping is what
  makes it readable without the repository.
- **Every model carries its resolved commit**, taken from the store's Agent records. An agent whose
  commit could not be resolved appears with its `unresolved_reason`, never with a bare ref.
- **Every claim names the store elements behind it.** Each entry under `steps` carries the
  `element_ids` it summarises, so any number in the JSON is traceable to the assertion that produced
  it. This is what makes the JSON a view of the store rather than a second copy of it.

## Placement

```
<run_dir>/
  store.jsonl              # the store
  summary/
    summary.pdf | summary.png
    summary.json
  released/                # REDACT's artifacts, only on a REDACT pass
```

`summary/` sits beside the store, under the run directory, and is **not** under `released/`: it
carries element ids and marked words' extents, so it inherits the store's sensitivity.

## Out of scope

Deciding anything, writing elements, computing any measurement not already in the store, and
producing a releasable artifact — that is [`REDACT`](redact.md)'s.

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `report.format` | `pdf` or `png`; a presentation choice, owed no measurement, but it must be declared rather than defaulted silently |
