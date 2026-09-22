# What the first build measured

The 2026-09-19 corpus run, scanned on 2026-09-22 at commit `e55e8d2d` in the pinned checkout
`/orcd/scratch/bcs/002/satra/senselab-rvec`, 32 Slurm tasks over
`/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out`, merged to
`/orcd/scratch/bcs/002/satra/recording_vectors_20260922/recording_vectors.parquet` (mode 600, in a
mode-700 tree).

## Size

| | |
| --- | ---: |
| rows written | **62,488** |
| columns | 101 |
| parquet, zstd | **91.3 MB** |
| **bytes per recording** | **1,462** |
| binary + string payload before parquet compression | 146.3 MB (2,342 B/row) |

The ≈2 KB budget in the design was set before the owner's 2026-09-22 correction put the transcript
back. It survived the correction anyway, at 1,462 B/row compressed. Words cost 92 B/row of text
plus 94 B/row of extents — 13% of the file — because the median recording carries 7 words and only
the 5th percentile of tasks are passages. The distribution is the reason: p50 7 words, p95 83, max
1,089 (a `caterpillar-passage`). No split is needed; 91 MB is one fetch.

Where the bytes actually go, uncompressed:

| block | MB | B/row |
| --- | ---: | ---: |
| `wave_minmax` | 32.0 | 511.8 |
| `span_label_name` | 25.8 | 413.6 |
| `env_dbfs` | 16.0 | 255.9 |
| `continuity` | 16.0 | 255.9 |
| `span_labels` | 14.0 | 223.6 |
| `spans` | 8.7 | 139.8 |
| `span_squim` | 8.7 | 139.7 |
| `branch_lane_role` | 8.3 | 132.8 |
| `asr_words` | 5.9 | 94.3 |
| `asr_word_text` | 5.8 | 92.0 |
| `branch_lanes` | 4.0 | 64.2 |
| `pii_category` | 0.7 | 10.9 |
| `pii_marks` | 0.5 | 7.4 |

The three fixed-width traces are 1,024 B/row and 44% of the uncompressed payload, and parquet's
zstd is what brings the file to 1,462 B/row overall. The largest variable cost is not the
transcript — it is `span_label_name`, the classifier label repeated per span per classifier, which
dictionary-encodes well but is still the second-largest block. If the file ever needs to shrink,
that is the place, and the fix is a corpus label vocabulary, not a shorter transcript.

## Coverage

| | |
| --- | ---: |
| manifest stems | 62,548 |
| stems with a `store.jsonl` when the shards ran | 62,488 |
| rows written | 62,488 |
| unreadable | 0 |
| incomplete (a store with no VERDICT fold) | 0 |
| superseded (a stem with more than one timestamped run) | 0 |
| measurement names with no column | 0 |

**60 recordings are not in the file, and none of them failed.** The `triage-corpus` array
(`23430535_1`) was still running when the scan went out: 56 stems had no run directory at all and 4
more appeared in the minutes between the shards finishing and the reconciliation. Re-running the
scan picks them up without moving any other row, because sharding is `sha1(stem) % 32` rather than
an enumeration index.

`superseded == 0` is worth stating: `measure-distributions.md` records that the manifest carries
62,578 lines for 62,548 distinct stems, 29 appearing twice and one three times. Those duplicates
are manifest lines, not run directories — the tree holds exactly one run directory per stem.

## What the nulls say

| column | null | reading |
| --- | ---: | --- |
| `duration_s`, `time_scale_s`, every trace | 29 | the 29 `discard` verdicts, all ADMIT failures: the file never decoded, so there is no stream, no derivative and no route |
| `spans`, `span_labels`, `span_squim`, `branch_lanes`, `route_*`, `conformance_quality` | 32 | 29 ADMIT failures plus 3 that reached PREPROCESS and no further |
| `pii_findings_n`, `pii_marks`, `pii_category`, `conformance_speech` | 19,074 (30.5%) | SPEECH did not run, so no PII scan happened. **These are the rows the null-not-zero rule exists for**: as `0` they would have read as 30.5% of the corpus scanned and clean |
| `conformance_airway` | 38,387 (61.4%) | AIRWAY declined |
| `conformance_voice` | 39,878 (63.8%) | VOICE declined |
| `grounds` | 62,459 | nothing was discarded on a ground; the 29 discards are ADMIT failures with no ground recorded |
| `m_sweep_extent` | 62,483 | the rarest measurement in the corpus, 5 readings |

Verdicts: 51,698 `pass`, 10,761 `flag`, 29 `discard`.
Release: 44,526 `not_assessed`, 13,555 `releasable`, 4,407 `withheld`.

## The reductions, and why each is what it is

**A scalar measurement column is the mean of that recording's readings.** 13 of the 17 numeric
names are written at most once per recording and the mean is then the reading. Four are written per
span or per track — `breath_peak_over_floor_db`, `cough_peak_over_floor_db`,
`phonation_onset_to_offset_s`, `interruptions` — and on a 481-store probe they repeated on 15/20,
5/6, 7/18 and 7/18 of the recordings that carried them at all. A single column per name therefore
needs a reduction whatever it is; the mean is reported beside `_n` so a reader can see when it
folded more than one number and go back to the store if that matters. No reduction is claimed to be
the right summary of a per-span reading — `_n` is there so the claim does not have to be made.

**`carrier_rejected` keeps the distinct gate names, not the count of each.** It is the
most-written measurement in the graph (383,212 readings over the corpus, a mean of six per
recording that carries it) and what a reader wants from it is *which* gate discarded a carrier.
`_n` carries the total.

**The envelope decimates by max and continuity by mean.** An envelope is peaky and the panel is
read for its peaks; a 50 ms event in a 30 s recording occupies a sixth of one bucket and a mean
would flatten it by 11 dB. Continuity is slowly varying and a single-sample excursion is not a
bucket. Both reductions are asserted through `extract`, not only through `encode_trace`, because
the mutation that swapped them at the call site survived a test that only exercised the helper.

## Mutation testing

`mutants.py` in this directory patches the module 18 ways and reports which the suite kills. The
first run killed 13; the five survivors were holes in the tests, not properties of the data, and
each is now covered:

| mutation | first run | after |
| --- | --- | --- |
| pack the blocks big-endian | not applied — the anchor matched `unpacker` too, so encode *and* decode flipped together and the round trip still passed | killed by the literal-bytes worked example |
| `TIME_SCALE = 65536` | survived — every assertion compared against the constant, and the worked example used a dyadic fraction that quantises identically under both | killed: full scale must be exactly 65535 and must still pack into a `uint16` |
| envelope decimated by mean | survived — the discriminating test called `encode_trace` directly, so mutating the call site in `extract` changed nothing it saw | killed through `extract` |
| continuity decimated by max | survived, same reason | killed through `extract` |
| `TRACE_POINTS = 255` | survived — the width assertions were written against the constant | killed: the widths are literal 256 and 512 |

That is the case for mutation testing in one table. A round trip is symmetric and cannot see a
symmetric error; an assertion written against the constant it is checking asserts nothing. Both
bugs were in tests that passed.

The remaining 13, all killed on the first run: drop a measure from the enumerated set; write zero
instead of null for an unmeasured scalar; report a never-scanned recording as clean; write an empty
block where the producer did not run; let an out-of-range value wrap; swap the `pesq` and `si_sdr`
ranges; index the label and SQUIM blocks against the unfiltered span list; order the ASR lane by
store order; collapse a repeated measurement's count to one; read invalidated entities as live; put
`verdict` before `task`; drop the words from the ASR lane; drop the PII category.

18/18 killed.
