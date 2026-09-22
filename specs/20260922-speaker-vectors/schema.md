# `speaker_vectors.parquet` — the schema a reader decodes against

One row per speaker. `schema_version` 1, carried twice: as an `int32` column on every row and as
arrow metadata `senselab.speaker_vectors.schema_version`. No column changes without this file
changing with it.

**Sensitivity.** A speaker embedding is a voice biometric derived from human-subject audio, and
the row names the recordings it came from. Written mode 600, gitignored
(`speaker_vectors*.parquet`, `*.report.json`, `*.summary.json`), never copied to a shared
location. On ORCD it lives under a mode-700 scratch directory.

## Writer settings

`compression="zstd"`, `row_group_size=1024`, `write_page_index=True`, `os.chmod(path, 0o600)` —
identical to `scripts/triage_recording_vectors.py`, and for the same reason recorded in
`specs/20260922-compact-recording-vectors/schema.md`: the default `row_group_size` of 1,048,576
puts the whole file in one row group, and a column chunk is per row group, so reading one row's
columns reads those columns entire. There the two costs crossed at 1024. **Here the file is far
smaller** — one row per speaker, ~1,500 rows against 62,488, and no 256-byte trace blobs — so the
selection cost that drove that measurement barely applies. 1024 is kept anyway: it costs nothing
at this size, and a reader that opens both files should not have to hold two different
assumptions about how they are laid out.

## Columns

### Identity and support — first, in this order

| column | type | meaning |
| --- | --- | --- |
| `speaker_id` | `string` | The BIDS `sub-<id>`. The only speaker key the corpus has; see `coverage.md` for what that assumes. |
| `vector` | `list<double>` | The pooled embedding, unit norm. Length `dim`. **Raw, not corpus-centred.** Centring the corpus improves verification and costs identification; `design.md` D-7 has both numbers, and the corpus mean is recoverable from this column in one pass. |
| `dim` | `int32` | 192 for ECAPA. |
| `n_extents` | `int32` | Task extents that contributed a window. |
| `n_recordings` | `int32` | Distinct stems behind those extents, and **the divisor the vector is actually pooled over**. Smaller than `n_extents` whenever a recording minted more than one extent — which the replayed tree does on 3,309 AIRWAY recordings (`design.md` D-9) and the design tree never did. |
| `extent_seconds` | `float64` | Total seconds of task extent that went in. |

### Provenance

| column | type | meaning |
| --- | --- | --- |
| `model_id` | `string` | `speechbrain/spkrec-ecapa-voxceleb`. |
| `model_commit_sha` | `string` | The **resolved 40-hex commit** the vector was produced with. Never a ref. Null only when resolution failed. |
| `unresolved_reason` | `string` | Why the sha is null. Non-null exactly when the sha is null. |
| `method` | `string` | The pooling rule, `recording_equal_spherical_mean`: windows → extent centroid → recording centroid → speaker vector. Measured against the one-stage window-weighted alternative in `design.md` D-7; the recording rather than the extent is the unit for the reason in D-9. |
| `window_s`, `hop_s` | `float64` | The window grid, 2.0 and 1.0. |
| `schema_version` | `int32` | 1. |
| `corpus_root` | `string` | The tree the extents were read from — which matters, because the replay moves task extent. |

### Counts a reader needs to weight a row

| column | type | meaning |
| --- | --- | --- |
| `n_sessions` | `int32` | Distinct `ses-` labels contributing. |
| `n_windows_used` | `int32` | Windows that reached the centroid. |
| `n_windows_dropped` | `int32` | Zero-norm windows excluded. |
| `n_effective_windows` | `float64` | `total windowed duration / window_s` — about `n/2` at a 2.0 s window on a 1.0 s hop. Any null whose width scales as `n^-1/2` is about √2 overconfident without this discount. |
| `n_extents_failed` | `int32` | Extents whose extraction raised or produced nothing. Non-zero means the estimate is thinner than `n_extents` suggests. |

### Diagnostics — is this one speaker?

Each is paired with a closed-form null where one exists, so nothing here needs a fitted threshold.
None of them is a verdict. **All of them describe the window cloud**, not the per-extent
centroids the stored vector is pooled from — one embedding pass feeds both, and the distribution
block is computed over every window with its extent id as the file id.

| column | type | meaning |
| --- | --- | --- |
| `rbar` | `float64` | Mean resultant length of the pooled windows. |
| `rbar_null` | `float64` | `1/sqrt(n)`, the value independent directions would give. |
| `cos_to_centroid_loo_q05`, `_q50` | `float64` | Leave-one-out cosine of each window to the centroid. `q05` is where an intruder shows up. |
| `cos_extent_centroid_to_pooled_min` | `float64` | The worst-agreeing extent. |
| `cos_extent_pairwise_q50` | `float64` | Median pairwise cosine between extent centroids. **Null with one extent** — no pair exists, and a number would be invented. |
| `auc_same_extent_vs_diff_extent` | `float64` | Mann-Whitney AUC of same-extent against different-extent window pairs. Exact null 0.5. **High is bad here**: it means extent identity, not speaker identity, explains the geometry. Null with one extent. |
| `cos_mean_vs_trimmed10`, `cos_mean_vs_medoid` | `float64` | Whether the centroid depends on how it was aggregated. |
| `leave_one_extent_out_cos_min` | `float64` | Jackknife along the cross-extent axis: the largest move any single extent's removal causes. |
| `participation_ratio`, `participation_ratio_null` | `float64` | How many directions the window set occupies, against the Marchenko-Pastur reference `d*n/(d+n)`. |
| `pc1_share_centred` | `float64` | Centred PC1 share. High is the signature of bimodality — two speakers, or two recording conditions. |

### The per-extent parallel lists

Nine columns, all the same length, all indexed by the same position, all in the order the extents
were handed to the estimator. A reader zips them.

| column | type |
| --- | --- |
| `extent_stem` | `list<string>` |
| `extent_run_dir` | `list<string>` |
| `extent_session` | `list<string>` |
| `extent_task` | `list<string>` |
| `extent_family` | `list<string>` |
| `extent_start_s`, `extent_end_s` | `list<double>` |
| `extent_windows_n` | `list<int32>` |
| `extent_leave_one_out_cos` | `list<double>` |
| `extent_centroid_to_pooled_cos` | `list<double>` |

`extent_start_s` and `extent_end_s` are the **exact** store seconds, not quantised. The
per-recording sibling quantises its times to `uint16` because it carries hundreds of them per row
for drawing; there are a few dozen here and they are provenance, so they stay exact.

## Null discipline

Inherited unchanged from the per-recording sibling, because a reader of both must not have to
learn two rules:

- **null** = not produced.
- **`[]`** = produced and empty.
- **`0`** = a measured zero.

`to_table` builds each column with `row.get(name)` against the declared schema and substitutes
nothing, so a key a row does not carry becomes null rather than a default. The single-extent case
is the one that exercises this in practice: `cos_extent_pairwise_q50` and
`auc_same_extent_vs_diff_extent` are null there, not 0.0 and not 0.5.

## Shards and the merge

Sharding is by **subject**, not by recording — a speaker's extents must all land in one worker or
the pooling is wrong. `shard_of(subject, slices)` is `sha1(subject)[:8] % slices`, content-
addressed so a growing tree never reshuffles a subject into another worker.

`scripts/triage_speaker_vectors.py` writes `speaker_vectors.NNN.parquet` plus
`speaker_vectors.NNN.report.json` per shard, and `--merge` concatenates them into
`speaker_vectors.parquet` with a `speaker_vectors.summary.json` carrying `schema_version`,
`shards`, `rows`, `bytes`, `bytes_per_row` and per-column `null_counts`. An empty shard still
writes a schema-correct empty parquet, so the merge is always a concat and never a cast. A shard
that wrote no row exits 1.

The report is not decoration. It carries `subjects_without_extent` and `subjects_all_refused`
separately — nothing to embed and everything below the floor are different facts about a speaker
— plus `recordings_seen`, `recordings_with_extent`, `recordings_unreadable`, `extents_admitted`,
`extents_refused_short`, `extents_missing_audio` and `subjects_failed`. A thin result must never
be mistaken for a clean one.

## Running it

```bash
# one shard
uv run python scripts/triage_speaker_vectors.py CORPUS_ROOT --out DIR --slice N --slices 64
# then, once every shard has written
uv run python scripts/triage_speaker_vectors.py --merge DIR
```

`CORPUS_ROOT` is a tree of finished run directories. **For the production artefact it is the
replayed tree**, `/orcd/scratch/bcs/002/satra/triage_replay_20260922/out`, because task extent is
one of the things the replay moves; every measurement in `design.md` was taken on the design
tree, which is the same shape and is what validated the reading path.

The staged submission is `/orcd/scratch/bcs/002/satra/speaker_vectors_20260922/production.sbatch`
(64 shards on `pi_satra`, output to `.../replay_vectors/`, checkout pinned by a commit check that
exits 74 on a mismatch).

**Submit it only after the replay array has finished every slice.** Sharding is by subject, so a
subject whose recordings straddle a finished and an unfinished replay slice would be pooled from
a partial supply — and the row would carry no sign of it, because `n_extents` would simply be
smaller. There is no marker in the tree that distinguishes "this speaker had four extents" from
"this speaker has four extents so far".
