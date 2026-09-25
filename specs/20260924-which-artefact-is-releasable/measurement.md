# The corpus effect

Two changes reach the same replay and are reported apart: the release vocabulary (this spec) and
the recall conformance withdrawal (`specs/20260924-recall-conformance-is-production/design.md`),
which landed at `d5029681` and is in the same tip. They touch different axes — release and
conformance — so the differential separates them by axis rather than by run.

## The r3 baseline

Every `summary/summary.json` under `/orcd/scratch/bcs/002/satra/triage_r3_20260923/out/`, read
directly. Slurm array `23702231`, 32 tasks, `ou_bcs_normal --qos=normal`. **62,548 summaries, 0
unreadable.**

| `release` | `release_ground` | n | % |
| --- | --- | ---: | ---: |
| `nothing_to_redact` | `NO_TRANSCRIPT` | 19,097 | 30.53 |
| `nothing_to_redact` | `NOTHING_BEYOND_STIMULUS` | 13,810 | 22.08 |
| `releasable` | — | 12,046 | 19.26 |
| `nothing_to_redact` | `SCAN_FOUND_NOTHING` | 11,526 | 18.43 |
| `withheld` | — | 4,647 | 7.43 |
| `nothing_to_redact` | `NO_LEXICAL_WORD` | 1,421 | 2.27 |
| `not_assessed` | `SPEECH_UNREAD` | 1 | 0.00 |

`nothing_to_redact` totals **45,854 (73.31%)**, which is the population §3 of
[`design.md`](design.md) splits.

Every recording in the corpus carries `llm_redaction.status` `disabled` or no annotation at all, so
no reviewer reading contributed to any of these figures.

## What the new table predicts

Holding the grounds fixed and applying the new table row by row:

| `release` | from | n | % |
| --- | --- | ---: | ---: |
| `release_without_redaction` | `NO_LEXICAL_WORD` + `NOTHING_BEYOND_STIMULUS` + `SCAN_FOUND_NOTHING` | 26,757 | 42.78 |
| `not_assessed` | `NO_TRANSCRIPT` + the one `SPEECH_UNREAD` | 19,098 | 30.53 |
| `release_with_redaction` | the old `releasable` | 12,046 | 19.26 |
| `withheld` | the old `withheld` | 4,647 | 7.43 |

**No recording leaves `withheld`, and none enters it.** The only movement across a permission
boundary is 19,097 recordings from a determination to an admission of ignorance, which is the
conservative direction.

## The replay

TO BE FILLED: job ids, row count, and the differential's own release and conformance matrices.
