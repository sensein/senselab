# Branch evidence carried every branch's spans and only AIRWAY's assertions

`_branch_evidence` in `nodes/report.py` builds the `evidence.branches` block of the summary JSON —
the audit record a reader follows from a decision back to what was observed. Until this change it
carried, for every branch, that branch's spans and measurements, but an `assertion` only when the
producing activity's node was `AIRWAY`:

```python
if entity.prov_type == "assertion" and branch != "AIRWAY":
    continue
```

SPEECH and VOICE write their typed deviations as assertions, through `write_findings`
(`nodes/branches.py`), against the closed `DEVIATION_TYPES` vocabulary. Both branches therefore
emitted located, typed, evidenced observations that VERDICT could fold and no reader of the report
could see.

## Sizing, measured on the finished corpus

Swept `/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out/**/store.jsonl` read-only in a
`mit_quicktest` job (32 procs, 81 s wall). Counts are of live assertions — not invalidated, with a
`wasGeneratedBy` edge to a branch activity — grouped by the producing node, the assertion's `verb`
and its `deviation_type`/`claim`.

| | recordings | share of corpus | assertions |
| --- | --- | --- | --- |
| corpus | 62,529 | 100% | — |
| **SPEECH or VOICE typed finding (invisible)** | **18,858** | **30.2%** | **140,528** |
| SPEECH typed finding | 17,577 | 28.1% | 137,317 |
| VOICE typed finding | 1,281 | 2.0% | 3,211 |
| AIRWAY typed finding (rendered) | 4,156 | 6.6% | 11,359 |

The evidence the filter dropped is 12.4× the evidence it kept, on 4.5× as many recordings. Nearly
one recording in three carried a typed finding that reached no reader.

By type:

| node | verb | type | n |
| --- | --- | --- | --- |
| SPEECH | contest | lexical_transcript | 89,245 |
| SPEECH | deviate | stimulus_mismatch | 28,547 |
| SPEECH | deviate | omission | 13,430 |
| SPEECH | deviate | off_task_extent | 3,429 |
| SPEECH | deviate | filler | 1,216 |
| SPEECH | deviate | truncation | 488 |
| SPEECH | deviate | repeat_reading | 483 |
| SPEECH | deviate | repeated_item | 479 |
| VOICE | deviate | omission | 1,141 |
| VOICE | deviate | lexical_content | 1,120 |
| VOICE | deviate | repeat_attempt | 820 |
| VOICE | deviate | sweep_direction_mismatch | 96 |
| VOICE | deviate | truncation | 34 |
| AIRWAY | deviate | off_task_extent | 10,813 |
| AIRWAY | deviate | truncation | 546 |

Eight of the ten declared deviation types appear in the corpus; the two that do not
(`repeat_attempt` under SPEECH, and any SPEECH/AIRWAY `lexical_content`) are branch-specific.

VOICE writes 1,141 assertions with no extent and 2,070 with one. The extent-free count equals the
`omission` count exactly: `voice.py` mints omissions per recording rather than per region, so a
report record for one carries `timing: null` rather than a fabricated span. That is the reason the
rendering keys off the type and lets `_timing` speak for the extent, rather than requiring one.

## What now renders

The branch test is gone. An assertion is carried when its `verb` is in
`_EVIDENCE_ASSERTION_VERBS` — `deviate` and `contest`, which is exactly the set `write_findings`
writes — and is described branch-agnostically as `<branch> <verb>: <type>`. For AIRWAY that formula
reproduces the previous string character for character (`airway deviate: off_task_extent`), so the
existing AIRWAY evidence test is unchanged and the corpus's rendered evidence does not move.

The evidence payload is still only `entity_id`, `description`, `timing` and `provenance`. A
deviation's `evidence` mapping (which for `filler` and `repeated_item` carries the matched word's
text) is not copied, so the block stays free of transcript text.

### Two assertion verbs deliberately stay out

- **`label` (271,766 in the corpus, all SPEECH)** — the per-word PII markings written at
  `speech.py:1981`. They are the mechanism behind the redacted transcript, which the report already
  renders as `redacted_transcript_tokens`; carrying them here would add up to several hundred
  records per recording that restate a lane already present, and each names a PII category against
  a word extent. A test asserts they stay out.
- **`exempt` (5,637, all REDACT)** — a redaction not made because the task expected the content.
  Dropped by the same removed line, so the same shape of defect, but not a typed deviation and out
  of this change's scope. Owed a decision: it is a reader-relevant observation with a category and
  no text, and nothing about it argues for staying invisible.

## What the printed page shows, and the 4-item cap left alone

The decision page prints a branch's deduplicated deviation **type names** from its `branch_report`
(`deviation: stimulus_mismatch`), and separately the **first four** items of that branch's evidence
block with their extents, then `N additional item(s) in summary JSON`. So the types were never the
missing thing on the page; the located, per-instance evidence was, and it was missing from the JSON
outright.

Ranking each recording's SPEECH and VOICE evidence items exactly as the report orders them
(`branch-evidence-assertions-rank-sweep.py`, same job shape, 96 GB):

| | recordings | typed findings |
| --- | --- | --- |
| with a SPEECH or VOICE typed finding | 18,858 | 140,528 |
| at least one now printed within the 4-item cap | 10,198 (54.1%) | 14,291 (10.2%) |
| none printed; all past the cap, all in the JSON | 8,660 (45.9%) | — |

Ordering typed findings ahead of spans inside the cap would take the first row to 100%. It was not
done: the page already names every type above the evidence lines, so what the reordering buys is
one instance's extent in place of a span's, and it would displace the branch's own proposals from
the only place they are printed. The JSON carries all 140,528 either way.

## Method

`specs/20260817-triage-workflow-dag/branch-evidence-assertions-sweep.py` is the sweep, stdlib-only
so it runs under the cluster's Python 3.6. It parses `store.jsonl` directly rather than importing
senselab, reads nothing but the store files, and writes nothing under the run tree.
