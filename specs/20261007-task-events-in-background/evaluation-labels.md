# Owner labels as evaluation targets

`scripts/evaluate_task_events.py` scores triage decisions against the owner's listening labels,
using `data/evaluation/owner_label_map.yaml` and the split in `data/evaluation/task_events_split.yaml`.
The label table itself (`triage_listening_labels_*.csv`) carries participant stems and stays out of
the repository; the split carries 8-hex subject prefixes only.

## Targets

| Target | Agrees with | From the owner |
|---|---|---|
| present | pass | "should not be discarded", "should pass", task heard as performed |
| present_flagged | review | background intercom inside a cough task ("should have been flagged for quality"); mic shutoff during phonation (flag) |
| present_or_review | pass, review | breath heard but swamped by background: "hard to analyze unless enhanced keeps the breath"; "it's ok for now to triage some noisy recordings" |
| review_acceptable | review, discard | "OK if flagged or discarded"; "leave as contested"; "discard quiet ones"; background noise heard instead of breath |
| absent | discard | no breath, no cough, no task, no audio |
| excluded | – | not judged for content |

A rerun or a recording missing from the decisions never agrees.

Extent and event remarks (`remark: extent`, `events`, `task_mismatch`) are counted, not scored:
the labels say where the extent or events were wrong, not where they should be.

## Split

Per listen set, subjects (8-hex prefixes) are stratified by their majority target and 30% of each
stratum is held out, with `numpy.random.default_rng(20261007)`; one subject's recordings stay on one
side. The file is written once; a subject labelled later is held out.
