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

The same tree was read a second time by a second script, [`axes_census.py`](axes_census.py), Slurm
array `23710001` on `pi_satra`: **62,548 recordings**, and the release table above reproduced value
for value. The two scripts share no code, so the agreement is a check on both. `axes_census.py` also
records the r3 baseline on the other two axes, which is what the recall change will be read against:

| axis | r3 |
| --- | --- |
| `triage` | 52,255 `pass` / 10,264 `flag` / 29 `discard` |
| task conformance | 48 declared families |

`compare_census.py` run over r3 against itself reports zero movement on every section, which is the
comparator's own null.

## The replay

`scripts/extend_replay_decisions.py` at `84ebb3b5107b1176ef9f01f305156d578bf4ffb9`, over the same
manifest and the same finished-corpus sources r3 replayed —
`/orcd/scratch/bcs/002/satra/triage_r3_20260923/replay_manifest.jsonl`, 62,548 rows, `--hints
<design>/scope` — into a fresh `--out-root` at
`/orcd/scratch/bcs/002/satra/triage_r4_20260924/out`.

240 slices in three disjoint ranges, one array per partition:

| array | partition | slices |
| --- | --- | --- |
| `23709700` | `pi_satra` | 0-39 |
| `23709701` | `ou_bcs_normal --qos=normal` | 40-139 |
| `23709702` | `mit_preemptable` | 140-239 |

Two earlier submissions were cancelled before doing work and are recorded because the reason is
operational and will recur: `23707723` on `ou_bcs_normal` ran 14 of a 160-throttle array for twelve
minutes against another user holding 230 jobs there, and `23709702`'s predecessor `23709076` on
`mit_preemptable` scheduled none of 240 in five minutes behind 1,496 other pending jobs. A first
submission of 400 slices was refused outright: the `ou_bcs_normal` **partition** QOS caps submitted
jobs per user at 256 whatever `--qos` names, and every array task counts.

Before the array, one 8-recording slice (`23707189`) ran the pinned checkout end to end and its
summaries came back `triage-summary/v9` carrying `release_without_redaction` and `not_assessed` with
the new `NO_TRANSCRIPT` wording. A `find -newermt` over the whole finished corpus tree for any
`*.flac` or `store.jsonl` touched in that window returns nothing: the mirror does not write through.

**Completion is verified on the row count, not on the job states.** Two earlier passes lost 156 and
60 rows to timed-out slices and reported success:

```
cat <run>/rows/slices/* | python3 -c 'import sys,json,collections
c=collections.Counter()
for line in sys.stdin:
    line=line.strip()
    if line: c[json.loads(line)["replay"]]+=1
print(sum(c.values()), dict(c))'
```

must read **62,548**. Any slice missing is resubmitted over the same roots: the driver skips a store
already carrying this configuration's replay marker, so a re-run costs nothing on what is done.

TO BE FILLED once the arrays drain: the row count, the r3-against-r4 census, and the in-store
differential (`r4-diff.sbatch`, whose baseline is the original corpus decision and therefore spans
more than these two changes — it is the cross-check, not the measurement).
