# The generated tables are parquet, not JSON — 2026-09-12

Everything `routing_analysis` generates is a table: one row per recording, per detector, per family,
per gate and budget. All of it was JSON — a 1.40 GB JSONL shard, a 129 kB detector profile, a
`ruleset_score.json` holding four different shapes at once. Every sweep in this project reads two or
three of the shard's columns and parsed all 1.40 GB of JSON to get them.

The tables are now parquet, zstd. The corpus totals, which are not a table, stayed JSON.

## The measurement

5,000 real rows out of the corpus, flattened to the 836 columns they carry:

```
jsonl bytes for those rows :  111.66 MB
parquet (snappy)           :   18.28 MB  (16.4% of jsonl)
parquet (zstd)             :   15.63 MB  (14.0% of jsonl)

parse 5,000 rows from jsonl :  1.32 s
read the whole parquet      :  0.24 s
read 3 columns from parquet :  0.03 s
```

Seven times smaller and 44x faster on the access pattern that dominates. Extrapolated over all
62,547 recordings the shard goes from 1.40 GB to about 0.20 GB, and a two-column sweep from 16.5 s
of JSON parsing to well under a second.

zstd over snappy: 14.5% smaller for a read that is already bounded by the footer rather than by
decompression. No reason was found to prefer snappy.

## Flattened columns, not parquet structs

`RecordingFeatures` has fifteen dict-valued fields with open key sets. Two shapes were available.

A **`MAP<string, double>` column** per field keeps the schema at 27 columns whatever the corpus
holds. It also makes every read a whole-map read: a detector wanting `peaks["plain|yamnet|Speech"]`
would decode all 238 peak keys of all 62,547 recordings to find it. The 44x number above is the
whole reason for the change and the map shape gives it back.

**One `<field>.<key>` column each** is what ships. `pq.read_table(path, columns=[...])` then touches
only the column chunks it was asked for. Splitting a column name on its first `.` is unambiguous
because no field name contains one, so a praat key spelled `mean_f0.hertz` round trips as
`praat.mean_f0.hertz` and comes back whole.

### The schema is nearly closed, which is why the width is affordable

A wide sparse schema is where parquet stops being free, so the width was bounded rather than
assumed. Counted off the static vocabularies in `labels.py` and `features.py`:

| field | columns | bounded by |
| --- | --- | --- |
| `span_label_stats` | 333 | `TRACKED_LABELS` per span classifier x 9 statistics |
| `peaks` | 238 | 3 streams x 3 classifiers + the consensus stream's 2, x their tracked labels |
| `span_stats` | 80 | 4 measures x 20 statistics |
| `squim` | 78 | 3 populations x (2 + 3 metrics x 8 statistics) |
| `span_label_set_stats` | 54 | 6 non-empty `LABEL_SETS` entries x 9 statistics |
| `ppg` | 19 | `PPG_SUMMARY_KEYS` |
| everything else closed | 50 | word outcomes, span measures, level, disruptions, silence, verdicts |
| **closed total** | **852** | |

Three fields are open-keyed: `praat` takes whatever the `praat_features` measurement wrote (forty
numbers, fixed by the extractor), `residual` takes every numeric attribute of the `residual`
measurement (about ten), and `bracketed_types` is keyed by normalised bracket tokens, which is the
only genuinely unbounded one and is small in practice.

So the ceiling is roughly 900 columns, and 5,000 rows already reached 836 of them. The corpus-wide
schema is at most some 8% wider than the sample that was measured; it does not grow with the corpus
in any way that matters.

### What 900 columns costs, measured

Synthetic tables at the real shape — 900 float columns at 60% density — with the footer read off
`ParquetFile.metadata.serialized_size`:

```
rows=  4000 group= 8192 size=  2.98 MB  footer=0.33 MB (11.1%)  read=0.06s  3col=0.009s
rows= 62547 group= 8192 size= 38.59 MB  footer=1.39 MB ( 3.6%)  read=0.14s  3col=0.018s
rows= 62547 group= 1024 size= 76.79 MB  footer=9.50 MB (12.4%)  read=0.82s  3col=0.080s
```

Row-group size is the knob that decides whether the width is affordable. At 1,024 rows per group the
file doubles, the footer takes an eighth of it and the three-column read is 4.4x slower, because each
of 62 groups carries 900 column-chunk headers with statistics. `ROW_GROUP_ROWS` is 8,192.

## Absent stays distinct from zero

A detector reading `None` drops the recording from its scoring; a detector reading `0.0` scores it as
a negative. `specs/20260912-detector-grids/design.md` records what that confusion cost the last time
it happened — three detectors reporting a plausible null result over the whole corpus.

The mapping is direct: a key a record's dict does not carry is not written, so the column is null
there, and `load_features` drops nulls rather than defaulting them. A key carrying `0.0` is written
and reads back `0.0`. `RecordingFeatures.duration_s` is `float | None` and its column is nullable, so
an unmeasured extent is null and not zero. Four tests in `TestFeaturesShard` assert exactly this,
including that `load_feature_column` on a key one record lacks reads `[None, 0.4]` and not
`[0.0, 0.4]`.

The schema is derived from `RecordingFeatures` itself through `get_type_hints`, not written down
beside it: a `dict[str, int]` field becomes nullable `int64` columns, `dict[str, str]` becomes
`string`, `list[str]` becomes one `list<string>` column. A field carrying an annotation with no rule
raises at import rather than being guessed at.

## The shard is a directory of parts

`analyze_routing_evidence.py` is resumable, and parquet cannot be appended to. Each batch of
`PART_ROWS` (4,000) extracted recordings is written as `features/part-NNNNN.parquet`, and a resume
reads the `stem` column alone out of the parts already on disk — the one read that was previously a
full JSON parse of everything extracted so far. A killed run loses at most the part it was filling.

That resumability has a price, measured over 16 parts at the real shape:

```
16 parts  46.98 MB  footers 5.30 MB (11.3%)
3-column read over the parts: 0.235s    one consolidated file: 0.018s
```

Thirteen times the narrow-read latency of a single file, and 11% of the bytes spent on repeated
footers. Against the 16.5 s the JSONL cost for the same read it is still a seventy-fold improvement,
and the alternative is re-extracting the corpus after every kill.

`missing.jsonl` and `manifest.jsonl` stay JSONL. Both are append-only logs whose whole purpose is
that a line can be added without rewriting the file, and neither is read by column.

## `ruleset_score.json` split three ways

One file held a corpus total, a per-branch 2x2, a per-family tally and a per-gate recall curve. Three
of those repeat a row shape; one does not.

| output | shape |
| --- | --- |
| `ruleset_score.json` | config hash, budgets, route-state counts, per-axis branch counts, the per-branch 2x2 |
| `families.parquet` | one row per task family, `<axis>.<branch>` columns |
| `recall_at_budget.parquet` | one row per routing gate and budget |

The per-branch scores stayed in the JSON although they are a table: there are as many of them as
there are branches, and at that count a parquet footer is larger than the data it describes. They
belong with the totals they are read beside.

`recall_at_budget.parquet` is denormalised — each row repeats its gate's threshold, configured 2x2
and curve header. Dictionary and RLE encoding make the repetition nearly free, and it keeps
"the recall of every gate at the 5% budget" a filter rather than a traversal.

The header that describes a whole table rather than any row — config hash, budgets, recording
count — is carried in the parquet file's key-value metadata, so the table stays one file and one
shape.

## The detector profile got smaller and slower

175 rows of quantiles in JSON clothing, read at import by `detectors.py` to derive every grid. It is
now one 175-row table, each detector a row, the quantile ladder as `q0.001` … `q0.999` columns, and
the sweep header (`profile_version`, `generated`, `corpus`, `n_recordings`, `quantiles`) in the
file's key-value metadata.

```
json     129,121 bytes     read 0.90 ms
parquet   25,224 bytes     read 2.56 ms
```

**This one is slower.** Five times smaller on disk, 2.8x slower to load, because a 25 kB parquet file
pays a footer parse and 24 column decodes where `json.loads` of 129 kB is one pass. The absolute cost
is 1.7 ms once per process — `load_detector_profile` is `lru_cache`d — so it was taken in exchange
for the profile being one shape with every other generated table rather than the only JSON left. If
import time ever matters, this is the row to revisit.

A `constant` detector's row has null `min`, `max`, `distinct` and quantiles, and the reader leaves
those keys out of the entry rather than reading them as zeros. `_validate` then rejects the entry for
its state, as it did before, rather than for a ladder of zeros it would have looked like it had.

## A derived grid stops where the data stops

`ddk.ppg_distinct_phonemes` has polarity `below`, a profiled minimum of 1, and a grid whose lowest
threshold was 0. A `below` cut at 0 fires on nothing. The integer union in `derive_thresholds`
enumerated `range(0, ceiling + 1)` for every count-unit detector regardless of what the feature
reached, so one row of every sweep over that detector was dead.

The grid is now clamped to what the contributing ladders attain. Two detectors changed:

| detector | polarity | profiled | was | now |
| --- | --- | --- | --- | --- |
| `ddk.ppg_distinct_phonemes` | below | 1 … 40 | 0 … 40 | 1 … 40 |
| `ddk.ppg_repetition_lag_segments` | above | 1 … 1199 | 0 … 1199 | 1 … 1199 |

Those are the only two count-valued detectors whose profiled minimum is not 0. Nothing was dead at
the top end: the ceiling was already `min(DENSE_INTEGER_SPAN, floor(max))`, and every count-unit
feature in the profile has an integral maximum, so no rounded threshold overshoots it.

The clamp is general rather than a special case for counts — any point outside `[min, max]` of the
contributing ladders is dropped, whether a `below` cut under the minimum that fires on nothing or an
`above` cut under the minimum that duplicates the route-everything endpoint. Only counts were
affected in this profile, because only the integer union ever proposed a point the ladder had not
measured.

**The declared cut points are unioned in after the clamp.** `taxonomy.consolidation_floor` and the
three pinned sign tests are properties of the quantity rather than of this corpus, which is the whole
argument for pinning them; a corpus that happens not to reach 0.2 must not be able to delete the row
that reports what the configured floor is costing.

### Four gated detectors keep a row that fires on nothing, deliberately

| detector | own max | top threshold |
| --- | --- | --- |
| `airway.residual_energy_fraction+hear>=0.5` | 0.999762 | 0.999879 |
| `cough.yamnet_cough_labels.plain+short_span` | 0.999977 | 0.999990 |
| `cough.level_crest_db+hear_cough>=0.5` | 41.7785 | 46.1112 |
| `glide.longest_amplitude_span+singing>=0.05` | 57.8552 | 68.6229 |

Each takes its grid from the ungated sibling reading the same primary feature, which is the design
`specs/20260912-detector-grids/design.md` argues for: a gate selects which recordings are scored, it
does not change what is read. The feature does attain those values; the gated view of it does not.
Clamping a gated detector to its own maximum would break the invariant that a gated grid carries its
primary feature's whole ladder, for four rows out of 3,064. They stay.

## Pre-alpha

No JSONL fallback, no format sniffing, no dual-format reader. `dump_features` writes parquet,
`load_features` reads parquet, `2026-09-12.json` was deleted rather than kept beside its parquet, and
existing shards on the cluster are regenerated.
