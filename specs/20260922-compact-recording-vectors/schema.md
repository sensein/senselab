# `recording_vectors.parquet` — the interface

> **Sensitive artefact.** This file carries transcript text and the extents and categories of every
> detected PII finding. It is written **mode 600**, lives only on the owner's machine or in the
> **mode-700** tree on ORCD scratch, is **never committed** (`.gitignore` carries its name), never
> published and never copied to a shared location. The page that reads it inherits that status: it
> is opened from disk by the person the data belongs to, not hosted. This exemption is one file for
> one reader — `corpus_decisions.json`, the measurement distributions and everything in `specs/`
> stay counts and categories only.

Produced by `senselab.audio.workflows.triage.recording_vectors` and
`scripts/triage_recording_vectors.py`. **One row per recording.** `schema_version` is `1`; any
change to a column or a byte layout bumps it and changes this file with it.

---

## 1. How to read this file

Three shapes of column:

1. **Scalars** — `string`, `double`, `int32`. Nullable; the nullability table below says what null
   means for each.
2. **Lists** — `list<string>` or `list<double>`. Null means absent; `[]` means present and empty.
3. **Binary blocks** — `binary`. A block is a **flat array of fixed-width little-endian records,
   with no header, no padding and no framing**. The record count is `len(bytes) / record_size`.
   Every block's layout is in §5. Null means absent; `b""` (zero length) means present and empty.

Two quantisers, used everywhere:

| | encode | decode |
| --- | --- | --- |
| **time** (`uint16`) | `round(t / time_scale_s * 65535)`, clamped to `0..65535` | `code / 65535 * time_scale_s` |
| **value** (`uint8`) | `round((v - lo) / (hi - lo) * 255)`, clamped to `0..255` | `lo + code / 255 * (hi - lo)` |

**Every time in every block is normalised against `time_scale_s`, not `duration_s`.** `duration_s`
is the *source* recording's duration; `time_scale_s` is the conditioned stream's, which is the axis
the graph's extents live on. Use `time_scale_s` and nothing else to decode a time.

One part in 65,535 of a 30 s recording is 0.46 ms, below anything the graph measures.

In JavaScript, every block is read with one `DataView`:

```js
function decodeSpans(bytes, timeScaleS) {           // 5 bytes per record: u8, u16, u16
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = [];
  for (let o = 0; o + 5 <= bytes.byteLength; o += 5) {
    out.push({
      row: "ECASG"[view.getUint8(o)],
      t0: view.getUint16(o + 1, true) / 65535 * timeScaleS,   // true = little-endian
      t1: view.getUint16(o + 3, true) / 65535 * timeScaleS,
    });
  }
  return out;
}
```

---

## 2. Identity and the decision record

Owner-directed: `participant`, `task`, `verdict` are the first three columns, in that order.

| column | type | units / vocabulary | null means |
| --- | --- | --- | --- |
| `participant` | string | `sub-<uuid>`, from the stem | the stem carries no `sub-` |
| `task` | string | everything after `task-` in the stem, e.g. `diadochokinesis-pa` | the stem carries no `task-` |
| `verdict` | string | `pass` \| `flag` \| `discard` — VERDICT's `triage` | never; a row exists only when the fold does |
| `session` | string | `ses-<uuid>` | the stem carries no `ses-` |
| `stem` | string | the BIDS stem, the run directory's name minus its `_YYYYmmdd-HHMMSS` suffix | never |
| `run_dir` | string | the run directory, relative to the scan root | never |
| `declared_family` | string | the declared task family, e.g. `story-recall-v2` | nothing was declared |
| `release` | string | `releasable` \| `withheld` \| `not_assessed` | the fold wrote none |
| `grounds` | string | VERDICT's `discard_ground` | **nothing was discarded** — the common case |
| `route_state` | string | e.g. `routed`, `declined` | the fold wrote none |
| `duration_s` | double | seconds, the **source** recording | ADMIT's `recording` stream entity is absent |
| `duration_conditioned_s` | double | seconds, the conditioned stream | PREPROCESS wrote no stream |
| `time_scale_s` | double | seconds — **the denominator for every `uint16` time** | neither duration is known |
| `sampling_rate` | int32 | Hz, of the conditioned stream | no conditioned stream |
| `schema_version` | int32 | `1` | never |
| `malformed_store_lines` | int32 | lines of `store.jsonl` that did not parse; `0` is the normal value | never |
| `flags_n` | int32 | how many node verdicts in the fold carry outcome `flag`, `fail` or `discard` — the same filter `report.py` calls a flag | never |
| `flag_nodes` | list\<string\> | which nodes those were, e.g. `["SPEECH"]` | never; `[]` when none |
| `conformance_airway` \| `_speech` \| `_voice` \| `_quality` | string | `true` \| `false` \| `undetermined` | **that node wrote no branch report** — it did not run |
| `route_airway` \| `_speech` \| `_voice` | string | e.g. `routed`, `declined` | routing wrote no state for it |
| `pii_findings_n` | int32 | how many `pii` entities the store holds | **the PII scan did not run.** A recording never scanned is null, not `0`. `0` means scanned and clean |
| `wave_peak` | double | amplitude; the scale `wave_minmax` is encoded against | no stream decoded |
| `floor_dbfs` | double | dBFS; the noise floor, a **scalar** — see §6 | `energy_envelope` is absent |
| `spans_unrowed_n` | int32 | general spans with no five-row code, left out of `spans` — see §6 | PREPROCESS did not run |

---

## 3. The measurements

The 29 names `specs/20260817-triage-workflow-dag/measure-distributions.md` enumerates. Each gets
**two** columns, and one gets a third:

- **`m_<name>`** — the reading. **Null when the graph wrote none for this recording.**
- **`m_<name>_n`** — `int32`, **never null**, how many readings the store held. `0` is a fact: the
  graph ran and wrote nothing.

> **The invariant, enforced by a test:** `m_<name> IS NULL` ⟺ `m_<name>_n == 0`. A reading of
> `0.0` is a measurement and reads as `0.0` with `_n == 1`. Null is not zero anywhere in this file.

**17 scalar numerics** — `double`, the **arithmetic mean** of the recording's readings. Identical
to the reading when `_n == 1`, which is the case for 13 of them. Four are written per span or per
track and routinely repeat — `breath_peak_over_floor_db`, `cough_peak_over_floor_db`,
`phonation_onset_to_offset_s`, `interruptions` — so read `_n` before treating one as a single
measurement.

`breath_coverage_fraction`, `breath_peak_over_floor_db`, `cough_peak_over_floor_db`,
`ddk_ppg_period_dispersion`, `ddk_repetition_count_from_ppg_decode`,
`ddk_syllable_rate_from_envelope_modulation_hz`, `ddk_syllable_rate_from_ppg_decode_hz`,
`expected_sequence_repeat_fraction`, `glide_extent_semitones`, `interruptions`,
`pause_fraction_of_response`, `phonation_onset_to_offset_s`, `source_content_coverage`,
`speech_rate_from_consensus_words_per_s`, `train_fraction_of_recording`,
`verbatim_overlap_fraction`, `voiced_duration_s`.

**1 vector numeric** — `list<double>`, nulls preserved inside the list.
`m_ddk_position_realised_mass` is one value per syllable position, each `null` when that position
was not realised.

**1 matrix numeric** — `list<double>`, **flattened row-major**, with
`m_ddk_cv_instrument_reading_width` giving the row width so the matrix can be rebuilt as
`n / width` rows. Rows are repetitions, columns are syllable positions.

**10 categorical** — `list<string>`, the **distinct** values written, sorted. Nine of them
(`category_membership`, `defines_its_cue`, `effort_absolute`, `measured_route`, `phonation_extent`,
`repetition_rule`, `route`, `source_overlap`, `sweep_extent`) only ever carry the single sentinel
`NOT_SEPARABLE_BY_THIS_DESIGN`, so for those the column is effectively a presence flag and `_n` is
the number of times it fired. `m_carrier_rejected` is the real one: the distinct gate names that
discarded a carrier, e.g. `["production_min_s", "voiced_fraction_min"]`, with `_n` the total number
of rejections (it repeats on ~97% of recordings that carry it at all).

---

## 4. Parallel list columns

Five binary blocks carry an open vocabulary that a byte cannot index. Each has a **parallel
`list<string>` column with one element per record of its block, in the same order**. The rule is:
*fixed enum → a byte in the block; open vocabulary → a parallel string list.*

| block | parallel column |
| --- | --- |
| `span_labels` | `span_label_name` — the classifier label, e.g. `"Speech"` |
| `asr_words` | `asr_word_text` — the word, e.g. `"caterpillar"` |
| `pii_marks` | `pii_category` — e.g. `"PERSON"`, `"DATE_TIME"` |
| `branch_lanes` | `branch_lane_role` — the role kind, e.g. `"speech_run"`, or the redaction category |

A parallel column is null exactly when its block is null, and `[]` exactly when its block is `b""`.

---

## 5. The binary blocks

All little-endian. `u8` = `getUint8`, `u16` = `getUint16(offset, true)`.

### `wave_minmax` — the conditioned waveform, 512 bytes

Fixed length `256 × 2 = 512`. Bucket `i` occupies bytes `2i` (min) and `2i + 1` (max), each `u8`
over `[-wave_peak, +wave_peak]`. Bucket `i` covers
`[i/256 × time_scale_s, (i+1)/256 × time_scale_s)`.

Decoded from the `preemphasised` stream, falling back to `plain`. Null when neither decodes.

### `env_dbfs` — the energy envelope, 256 bytes

Fixed length `256`. Byte `i` is `u8` over **`[-100.0, 0.0]` dBFS**, the **maximum** of the
per-sample envelope within bucket `i`. Max, not mean: an envelope is peaky and the panel reads its
peaks. Null when `energy_envelope.npz` is absent or unreadable.

### `continuity` — the continuity trace, 256 bytes

Fixed length `256`. Byte `i` is `u8` over **`[0.0, 1.05]`** (the axis the figure draws), the
**mean** within bucket `i`. Mean, not max: continuity is slowly varying. Null when
`continuity_trace.npz` is absent.

### `spans` — the five-row span lane, 5 bytes per record

| offset | size | field |
| ---: | ---: | --- |
| 0 | u8 | `row` — index into `["E","C","A","S","G"]` |
| 1 | u16 | `t0` |
| 3 | u16 | `t1` |

`E` = envelope amplitude, `C` = continuity, `A` = ASR, `S` = normalised amplitude, `G` = gap —
`_SPAN_ROWS` in `nodes/figure.py`, and a test asserts the two agree. Records are in store order;
**a record's index is the `span_index` the next two blocks refer to.**

Null when PREPROCESS wrote no verdict; `b""` when it ran and no span survived.

### `span_labels` — the top classifier label per span, 4 bytes per record

| offset | size | field |
| ---: | ---: | --- |
| 0 | u16 | `span_index` into `spans` |
| 2 | u8 | `classifier` — index into `["yamnet","hear","ast"]`; `255` = not one of those |
| 3 | u8 | `score` — `u8` over `[0.0, 1.0]` |

The label's name is `span_label_name[i]`. One record per (span, classifier) that produced any
score; the label is the argmax of the per-window maxima, which is the reduction the raster panel
draws. Absent for a span that no classifier scored — omission, not a sentinel.

### `span_squim` — SQUIM per span, 5 bytes per record

| offset | size | field | range |
| ---: | ---: | --- | --- |
| 0 | u16 | `span_index` into `spans` | |
| 2 | u8 | `stoi` | `[0.0, 1.0]` |
| 3 | u8 | `pesq` | `[1.0, 4.5]` |
| 4 | u8 | `si_sdr` | `[-10.0, 30.0]` dB |

The ranges are `FigureStyle.squim_ranges`, and a test asserts they agree. A span SQUIM refused is
simply absent from this block.

### `asr_words` — the consensus ASR lane, 5 bytes per record

| offset | size | field |
| ---: | ---: | --- |
| 0 | u16 | `t0` |
| 2 | u16 | `t1` |
| 4 | u8 | `outcome` — index into `["agreement","variant","insertion"]`; `255` = none of those |

The word itself is `asr_word_text[i]`. Records are in the consensus `index` order, which is the
only legal ordering. Null when no `consensus_transcript` measurement exists; `b""` when one exists
with no words.

### `pii_marks` — detected PII, 4 bytes per record

| offset | size | field |
| ---: | ---: | --- |
| 0 | u16 | `t0` |
| 2 | u16 | `t1` |

The category is `pii_category[i]`. Extents are word extents, so a mark can be overlaid on the ASR
lane directly. **Null when the PII scan did not run** — a recording that was never scanned must not
render as clean. `b""` means scanned and nothing found.

### `branch_lanes` — the branch-proposal lanes, 5 bytes per record

| offset | size | field |
| ---: | ---: | --- |
| 0 | u8 | `lane` — index into `["AIRWAY","SPEECH","VOICE","REDACT"]` |
| 1 | u16 | `t0` |
| 3 | u16 | `t1` |

The role is `branch_lane_role[i]`: for the three branches, the span's `role` with its trailing
instance index stripped (`speech_run_12` → `speech_run`); for REDACT, the redaction's category.
Null when routing wrote no `branch_decision`; `b""` when it routed and nothing proposed a span.

### A worked example

One span, on the ASR row, from 1.0 s to 2.0 s of a recording whose `time_scale_s` is `4.0`:

- `row` = index of `"A"` in `["E","C","A","S","G"]` = `2` → byte `0x02`
- `t0` = `round(1.0 / 4.0 × 65535)` = `16384` = `0x4000` → little-endian bytes `0x00 0x40`
- `t1` = `round(2.0 / 4.0 × 65535)` = `32768` = `0x8000` → little-endian bytes `0x00 0x80`

The block is the five bytes `02 00 40 00 80`. Decoding: `0x4000 / 65535 × 4.0 = 1.00002 s`,
`0x8000 / 65535 × 4.0 = 2.00003 s`. This example is asserted byte-for-byte by
`test_the_worked_example_in_the_schema_document_decodes_as_written`.

---

## 6. Where the design and the stores disagreed

The design was written from the figure code. Three things read differently against the stores.

**The floor is not a trace.** `energy_envelope.npz` stores `floor_dbfs` as
`np.full_like(envelope, floor)` — a constant array the length of the recording — and the figure
takes element `[0]` and draws a horizontal rule. So the "four decimated polylines" are three
polylines and one scalar, and `floor_dbfs` is a `double` column, not a block. Decimating a constant
to 256 points would have cost 256 bytes a recording to say one number 256 times.

**The five-row span lane has a sixth kind of span in its source set.** REDACT writes its redaction
spans with `{"name": "redaction", "category": …}` and **no `family`**, so the figure's own filter
(`family is None`) admits them and then assigns them the row code `"?"`, which is not one of the
five rows — they are drawn nowhere. This extractor excludes spans with no row code from the `spans`
block and counts them in `spans_unrowed_n`, so the omission is visible rather than silent. The
redactions themselves are in `branch_lanes` under the `REDACT` lane, which is where the figure
draws them.

**"Every numeric measurement" is 17 of the 29 names, not 29.** Of the enumerated names, 17 carry
numbers, 1 carries a vector, 1 a matrix, and 10 carry strings — 9 of those only ever the sentinel
`NOT_SEPARABLE_BY_THIS_DESIGN`. Each still gets a column, typed by what the graph actually writes;
only the 17 are usable as a parallel-coordinate axis directly.

A fourth, smaller one: **a measurement name is not unique within a recording.** Four of the 17
numerics are written per span or per track, and `carrier_rejected` is written a mean of six times.
A single column per name therefore needs a reduction, which is why `_n` exists beside every one.

---

## 7. Running it

```bash
# one shard
uv run python scripts/triage_recording_vectors.py RUN_ROOT --out DIR --slice N --slices 32
# merge the shards
uv run python scripts/triage_recording_vectors.py --merge DIR --out DIR
```

Sharding is **content-addressed** — `sha1(stem) % slices` — so a tree that grows while the array is
running never moves a recording from one shard to another, and a shard that is re-run covers the
same set. A stem with more than one timestamped run directory keeps the latest and reports the
rest as `superseded`. A store with no VERDICT fold is `incomplete` and yields no row.

The first build over the 2026-09-19 corpus wrote **62,488 rows in 91.3 MB — 1,462 bytes per
recording**, transcripts included. What each block costs, what the nulls mean over the whole
corpus, and the mutation results are in
[`measurements.md`](measurements.md).

Each shard writes `recording_vectors.NNN.parquet` (mode 600) and `recording_vectors.NNN.report.json`
with `considered`, `written`, `incomplete`, `unreadable`, `superseded` and `anomalies` — the last
being measurement names the store carried that this schema has no column for. The merge writes
`recording_vectors.parquet` (mode 600) and `recording_vectors.summary.json`, which carries the
per-column null counts.
