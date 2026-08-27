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
| **summary** | one paginated PDF **or** one image per file | a human reading one recording |
| **summary JSON** | one JSON per file | a consumer reading many |

The PDF is the concise human view and the JSON is the complete analytic record. They agree on the
decision claims; the choice is the config key `report.format` and does not alter the workflow's
stored evidence.

- **`pdf`, the packaged form, uses one aligned US Letter landscape evidence page per 10 seconds of
  recording, followed by concise US Letter landscape decision page(s).** Each evidence page shares
  its fixed 10-second recording-time axis across all panels. Every artist is filtered to that page's
  time window before it is drawn, so an off-page label can never alter the printed page dimensions.
  The complete audit record remains in the accompanying JSON rather than turning the decision page
  into a diagnostic dump.
- **`png` is one image** carrying the panels and the blocks together, for a viewer that scrolls
  rather than pages.

## The summary — the panels

On a single shared time axis, one row each:

| layer | drawn from |
| --- | --- |
| waveform amplitude on the left y-axis, energy envelope and its floor in dBFS on a twin right axis, and every `span` as a translucent overlay annotated with its `peak_over_floor_db` — **one row**. Sub-resolution post-filter values are gaps rather than dBFS readings, so numerical ringing cannot pull the displayed floor downward. | PREPROCESS |
| `phonation_spans` and glides, with `duration_s` and production mode | PREPROCESS |
| YAMNet and HeAR as fixed-row top-K probability rasters over their own native grids; a cell's color is the thresholded score retained for that classifier window, and an empty cell means below the reporting threshold. AST is a raster only when its stored hop is under 8 s, otherwise it is a coarse-window summary in Supporting Evidence | PREPROCESS |
| speech spans with their speaker attribution and any `nontarget` marking | SPEECH |
| consensus ASR words — a compact multi-row token lane: one bar per fused consensus word at its own extent with the authoritative word text drawn on a light confidence-ordered fill, never as a y-tick. The colors are presentation only; the authoritative numeric confidence remains in JSON. | PREPROCESS |
| airway-labelled spans with their labels and confirmations | AIRWAY |
| voiced runs and their extents | VOICE |
| redacted transcript words, when PII marking changed at least one consensus word — a parallel compact token lane whose placeholders show exactly what a released transcript would replace | REDACT |
| spectrogram | the conditioned stream |

The waveform, the envelope and the envelope spans are three readings of one signal and share one
row. The right-hand y-label is simply `dBFS`; each span carries its own dB-over-floor annotation,
so the axis does not repeat a second, unrelated label. A lane that shares that row is still a
**declared lane**, and the ABSENT block reports it as drawn.

AST's packaged 10.24 s window and hop are intentionally rendered as a summary rather than a timeline:
each fired label is a property of that broad acoustic context, not a local event boundary. The JSON
records `evidence.label_presentations.ast` with the stored window and hop plus `mode: summary_only`,
so an analytical consumer receives the same distinction as the human reader. A run whose stored AST
hop is below 8 s retains its time-aligned probability raster; new configs reject such a hop.

The JSON mirrors every drawn YAMNet, HeAR and time-resolved AST window in
`evidence.classifier_windows`. Each item carries its provenance entity id, timing, retained
`label_scores` and `thresholded_labels`; the report never makes a page-only probability claim.

**The title is short.** The task token the run id names, the date, and the two verdict axes —
`task-… · 2026-08-25 · triage: flag · release: not_assessed`. The full run id and the file path are
provenance and appear in the blocks; every block line is folded to the block width, so nothing runs
off the page.

## The summary — the blocks

The final PDF page, the foot of the PNG. One block per step:

- **which branches ran, which were skipped, and which a hint forced** — the `branch_decision`
  elements;
- **each branch's conclusion and its flags**;
- **TAXONOMY's classification beside the resolved kinds**, with the per-kind agreement or mismatch
  and the per-kind hint reading — the row of verdict.md's hint table this kind fell on. The decision
  page also gives each kind's explicit evidence path (line evidence, floor and line state), naming
  the consensus lexical line as decisive for speech and acoustic labels as corroboration;
- **the verdict**: `triage`, `release`, and every reason, with a REDACT non-pass shown whatever the
  triage axis says.

**The summary distinguishes evidence from release presentation.** The consensus lane and
`evidence.consensus_transcript_tokens` preserve PREPROCESS's authoritative words. The separate
redacted lane and `evidence.redacted_transcript_tokens` apply the PII marking. The summary remains
sensitive because its consensus text and provenance identifiers can identify the recording; it is not
a release artifact. Both products carry element ids, which are a join key back into the store.

## The summary JSON

The verdict's product ([`verdict.md`](verdict.md)), plus per-step summaries, plus **provenance
embedded rather than referenced**:

```
{
  file:        { path, duration_s, sample_rate, channels },
  verdict:     { triage, release, reasons[], kinds{}, screened{}, agreement{}, hints{} },
  screening:   { screened_kinds{}, resolved_kinds{}, agreement{}, decision_paths{} },
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
<out_dir>/<stem>_<utc-timestamp>/     # the run root; the three trees below are siblings
  run/
    store.jsonl                       # the store
    streams/  derivatives/            # the sidecars measurements point at
    run.json                          # the runner's own record
  summary/
    summary.pdf | summary.png             # pdf: Letter <=10 s evidence pages, then decision pages
    summary.json
  released/                           # REDACT's artifacts, only on a REDACT pass
```

`summary/` sits **beside** the store tree rather than inside it, and is **not** under `released/`: it
carries element ids and marked words' extents, so it inherits the store's sensitivity and is not a
releasable artifact. Three siblings rather than two nested trees is what lets a publish step sweep
`released/` without reaching either of the other two.

## Out of scope

Deciding anything, writing elements, computing any measurement not already in the store, and
producing a releasable artifact — that is [`REDACT`](redact.md)'s.

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `report.format` | `pdf` (packaged) or `png`; a presentation choice, owed no measurement, but it must be declared rather than defaulted silently. The derivation beside the key records why the earlier `png` default was withdrawn |
