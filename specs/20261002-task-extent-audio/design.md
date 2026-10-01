# Task-extent audio

Owner's request (2026-10-01): *"a set of routines that generate task-extent extracted audio from both
plain and from enhanced when available. there should also be a redacted stream to which the same
task extent is applied."*

Code: `src/senselab/audio/workflows/triage/task_audio.py` (library),
`scripts/extend_task_audio.py` (driver), `nodes/redact.py:released_audio` (the masked copy, shared
with `settle_release`). Tests: `src/tests/audio/workflows/triage/task_audio_test.py`,
`src/tests/scripts/extend_task_audio_test.py`.

## 1. One extent definition

`task_extent(store, padding_s)` → `TaskExtent`, recorded as `definition:
hull_of_live_task_extent_spans`:

1. Take every **live** span whose `role` is `task_extent`.
2. The hull is the earliest span's start to the latest span's end.
3. Pad each side outward by `task_audio.padding_s`.
4. Clamp to `[0, recording_s]`, where `recording_s` is the end of the live `recording` stream's
   extent. The measurement records `clamped_start` and `clamped_end`.
5. If no span exists, or the clamped extent is empty, there is no extent and nothing is cut.

**Why the hull of every live span, not only the owning branch's span.** This is the same set
`recording_vectors._extent_columns` reads for `task_extent_start_s` and `task_extent_end_s`, so the
parquet's trim columns and the cut describe the same interval.

The obvious alternative was to read only the span from the branch that owns the declared family:
SPEECH for reading, VOICE for phonation, AIRWAY for respiration, DDK for diadochokinesis. That was
rejected for two reasons:

- SPEECH already guarantees that at most one of its own spans survives (`TASK_EXTENT_ROLE` in
  `nodes/speech.py`).
- A non-owning branch that wrote a task extent found the task there too. Dropping its span could cut
  off performed task audio. Keeping it can only add lead-in.

If the owner wants owner-only extents, that is a second definition name beside this one, not a
change to this one.

**`padding_s` = 0.25 s, CONVENTIONAL.** See `config-derivations.md#task_audio`. It is the same width
as the mask pad, so a mask padded at the task's edge lies inside the cut. The pad costs at most 0.5 s
per recording; §5 sets that against the savings measured on r9.

## 2. Sample alignment

Each stream is cut at its own rate, with no resampling. Given an extent `[s, e)` in seconds and a
rate `sr`, the cut keeps samples `lo = floor(s·sr)` to `hi = ceil(e·sr)`, both clamped to the stream
(`sample_bounds`). This is the same rule `apply_redactions` uses to mask, which has two consequences:

- A mask inside the extent lands on exactly the same samples of the cut as of the full stream,
  shifted by the integer `lo`. No second rounding is involved.
- Each bound lies within one sample of the extent in seconds, at every rate. The tests check this at
  8, 16, 22.05, 44.1 and 48 kHz.

`plain` and `enhanced` share the 16 kHz working rate. PREPROCESS lag-aligns `enhanced` to `plain`,
so the two cuts start on the same sample index. The redacted copy is masked from `recording` at its
native rate and channel count, and is cut at that rate. All three cuts cover the same seconds.

## 3. Which audio each cut takes

| cut | source | present when |
|---|---|---|
| `task_plain` | the live `plain` stream | always, once PREPROCESS conditioned the recording |
| `task_enhanced` | the live `enhanced` stream | PREPROCESS wrote one |
| `task_redacted` | the **released** redacted copy (`released_audio`) | the live fold's `release` is `release_with_redaction` |

**The redacted cut is the released copy, never REDACT's stream by itself.** The fold may change the
masks after REDACT ran: the ledger's `final_masks` can unmask a word that a reviewer released, or
add masks. What `settle_release` writes to `released/` is therefore one of two things:

- REDACT's `redacted` stream, where the final masks equal the planned ones;
- otherwise, `recording` masked again with the final masks.

`released_audio` is now the single function that both `settle_release` and the cut call. As a
result, `task_redacted` always carries exactly the masks the released copy carries. When the copy
was masked again, the cut is derived from `recording` and the `pii_ledger` measurement, not from the
`redacted` stream.

**When the release is not `release_with_redaction`, there is no redacted cut.** This is the safest
output for each of the other releases:

- `release_without_redaction`: the recording itself may be handed on, so `task_plain` is the
  releasable cut. A masked cut would be a second, unreleased artefact that someone could mistake for
  the release.
- `withheld` and `not_assessed`: no masked copy may be handed on, so none is made. REDACT's stream
  can still exist in `run/streams/`. Cutting it would produce a file named "redacted" that nothing
  vouches for.

If a re-fold moves the release away from `release_with_redaction`, the next pass retires the
standing `task_redacted` stream and deletes its file. The test is
`test_a_release_that_moves_retires_the_redacted_cut`.

`task_plain` and `task_enhanced` are internal products, like `plain` and `enhanced` themselves. They
live under `run/streams/`, not under `released/`. Whether either may leave the tree depends on the
recording's release, and the derivatives sync must check that (§6).

## 4. Masks are verified, not assumed

For `task_redacted`, `masks_in_cut` computes each final mask's sample range inside the cut:
`max(lo_mask, lo) − lo` to `min(hi_mask, hi) − lo`. Masks outside the extent are dropped, and a mask
that straddles an edge is clipped to it.

After the cut is written, `verify_masks` reads the file back and checks two things:

- its shape equals the masked copy's `[lo, lo + n)`;
- under a `silence` fill, every sample inside every mask range is exactly zero.

If either check fails, the pass raises and nothing is registered. The records in cut coordinates are
stored on the stream as `masks`, with `masks_verified: true`.

The tests check stronger properties on synthetic stores:

- The only silent samples in the cut are the masks' samples. A mask outside the extent leaves no
  trace, and a mask inside it lies on `floor(start·sr) − lo` to `ceil(end·sr) − lo`.
- A ledger that widens a mask produces a re-masked cut that is silent over the wider range and is
  derived from the ledger.

## 5. Store, files and cache

- **Streams.** `task_plain`, `task_enhanced` and `task_redacted` are stream entities. Each is
  generated by a `TASK_AUDIO`/`cut` activity and carries:
  - `path`, `checksum_sha256`, `sampling_rate`, `channels` and `write_gain`;
  - `source`, `source_path`, `source_sha256` and `source_samples: [lo, hi]`;
  - `start_s`, `end_s`, `cut_key` and `process_version`;
  - for `task_redacted` only: `fill`, `remasked`, `masks` and `masks_verified`.

  Each stream is `wasDerivedFrom` its source stream and the `task_audio` measurement, and the
  re-masked redacted cut is also derived from the ledger. Files are at
  `run/streams/task_<source>.flac`.
- **Measurement.** `task_audio` carries the extent (`TaskExtent.as_dict()`), `cuts` and
  `process_version`. It is derived from the task-extent spans.
- **Sidecar.** `run/derivatives/task_audio.json` holds the extent and, per source, either the cut's
  record (stream id, path, digest, source path and digest, sample range, masks) or `{"absent":
  reason}`.
- **Cache.** `cut_key` hashes the process and `PROCESS_VERSION`, the source's recorded digest, the
  definition, `[lo, hi]`, and, for the redacted cut, the masks and fill. On a rerun, a cut is
  `present` when its key matches and the file on disk still hashes to its recorded digest. When
  every cut is present, the store is not touched at all, so its fingerprint and the files' mtimes are
  unchanged. The cache is per run rather than the global result cache because the cut is a slice. A
  global entry would duplicate the audio and save nothing.
- **Parquet (schema 16).** `task_audio_start_s`, `task_audio_end_s`, `task_audio_duration_s` and
  `task_audio_cuts`.

## 6. Measurement on r9 (2026-10-01)

**Setup.**

- Sample: 49 r9 run roots, drawn by a seeded shuffle of `refold/manifest.jsonl`. The draw was
  stratified to take up to 15 recordings under `release_with_redaction`; 14 were found in the first
  600 rows. The sample was then copied with `copytree` to
  `/orcd/scratch/bcs/002/satra/task_audio_20261001/copies/` (321 MB).
- Run: `extend_task_audio.py` at `f4561d18`, on one 2-CPU srun, against the copies only.
- Caveat: the stores were copied while the v6 re-fold was still writing the tree, so each copy's
  release is whatever fold stood at copy time.

**Releases in the sample.** 34 `release_without_redaction`, 14 `release_with_redaction`, 1
`not_assessed`.

**Outcomes.**

- 45 recordings were cut and 4 had no task extent.
- `task_plain` and `task_enhanced` were written on all 45.
- `task_redacted` was written on 13. The 14th recording released with redaction is one of the 4
  with no extent. The other 32 recordings have no redacted cut because they are released without
  redaction.
- **Masks:** all 13 redacted cuts passed `verify_masks`. 23 final masks fell inside the cuts.

**Cost.** 25.9 s wall-clock for all 49 rows (median 0.1 s and maximum 2.1 s per row), with a peak
RSS of 1.2 GB, which is mostly the import.

**Rerun.** All 45 cut rows came back `present` and no store was rewritten.

**Duration saved.** Over the 45 cut recordings, the recordings total 821.8 s and the cuts remove
112.8 s (13.7%).

| | min | q1 | median | q3 | max |
|---|---|---|---|---|---|
| saved, s | 0.00 | 0.97 | 1.54 | 3.79 | 8.39 |
| saved, fraction | 0.000 | 0.08 | 0.14 | 0.29 | 0.605 |

The largest savings are where the task is short inside a long take:

| task | n | median saving |
|---|---|---|
| prolonged-vowel | 4 | 6.1 s (54%) |
| respiration-and-cough-cough-1 | 1 | 4.8 s (61%) |
| glides-low-to-high | 1 | 4.1 s (55%) |
| word-color-stroop | 2 | 7.7 s (10%) |

Read passages and free speech fill their recordings: caterpillar-passage saves 1.8%, and free
speech saves 5–22%.

**Empty extent (no cut), 4 recordings.**

- `random-item-generation-v2`: 1
- `respiration-and-cough-breath-1`: 1
- `respiration-and-cough-breath-2`: 1
- `respiration-and-cough-cough-2`: 1

No branch wrote a task-extent span on any of these. Respiration accounts for 3 of the 4; whether
AIRWAY should have found the breaths is a question for that branch, not for the cut.

**Whole file.**

- **1 recording**, `cape-v-sentences-5` (2.83 s), was clamped at both ends. Its cut is the whole
  file, because the task fills the recording to within the pad on both sides.
- **12 recordings were clamped at the end only**: the task runs to within 0.25 s of the recording's
  end. That is a truncation signature, and the measurement records it as `clamped_end`.
- **3 recordings were clamped at the start only.**

## 7. Staged corpus run

`/orcd/scratch/bcs/002/satra/task_audio_20261001/stage/task_audio.sbatch` is staged and **not
submitted**. It runs 64 CPU slices over `refold/manifest.jsonl`:

- `--config review_on.yaml`, the configuration the fold ran under;
- `TRIAGE_EXPECTED_COMMIT=PIN_ME`, with the exit-74 guard;
- excluding node2119, node2621 and node3002.

It must run only **after the v6 re-fold has finished**, for two reasons. The redacted cut follows the
live release. And the array rewrites `store.jsonl` where it cuts, which would race the re-fold.

At 0.1–2 s per row, a 64-way array finishes 62,550 rows in well under an hour.

## 8. BIDS derivatives naming (proposal)

Each cut maps to a BIDS-derivatives file named after the source recording's own stem, with a `desc`
entity naming the stream:

| store stream | derivatives file |
|---|---|
| `task_plain` | `sub-<s>/ses-<x>/audio/sub-<s>_ses-<x>_task-<t>_desc-taskplain_audio.flac` |
| `task_enhanced` | `…_desc-taskenhanced_audio.flac` |
| `task_redacted` | `…_desc-taskredacted_audio.flac` |

Each file has a sidecar `…_desc-task<source>_audio.json`, carrying the following:

- **`Sources`:** the `bids::` URI of the raw recording.
- **`TaskExtent`:** `start_s`, `end_s`, `padding_s`, `definition`, `clamped_start` and
  `clamped_end`.
- **`SamplingFrequency` and `Channels`.**
- **`SourceSamples`:** `[lo, hi]`.
- **The cut's checksum.**
- **`Masks`, for `taskredacted`:** in seconds on the cut's own time base. Categories only, never
  text.

The `desc` values have no hyphens or underscores, as BIDS entities require. `audio` is the suffix the
b2ai tree already uses.

**How the sync would pick them up.** The run tree's derivatives sync already walks
`out/sub-*/ses-*/<stem>_<stamp>/`. It would read the sidecar `run/derivatives/task_audio.json`
rather than globbing for files, so that a stream the store has retired is never copied, and for each
`cuts.<source>` record that is not `absent` it would:

1. Copy `run/<path>` to the name above. The record's `checksum_sha256` must match the file, or the
   sync refuses the copy.
2. Write the JSON sidecar from the record and the `extent` block.
3. Apply the release gate:
   - **`taskredacted`:** synced only when `cuts.redacted` exists. It exists only under
     `release_with_redaction`, so the gate is inherited.
   - **`taskplain` and `taskenhanced`:** synced to a releasable derivatives tree only under
     `release_without_redaction`. Under any other release they go only to the internal tree, the
     same rule that applies to `plain` and `enhanced`.

   The sync reads the release off the live VERDICT, never off the presence of the file.
