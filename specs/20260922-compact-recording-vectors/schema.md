# `recording_vectors.parquet` — the interface

> **Sensitive artefact.** This file carries transcript text and the extents and categories of every
> detected PII finding. It is written **mode 600**, lives only on the owner's machine or in the
> **mode-700** tree on ORCD scratch, is **never committed** (`.gitignore` carries its name), never
> published and never copied to a shared location. The page that reads it inherits that status: it
> is opened from disk by the person the data belongs to, not hosted. This exemption is one file for
> one reader — `corpus_decisions.json`, the measurement distributions and everything in `specs/`
> stay counts and categories only.

Produced by `senselab.audio.workflows.triage.recording_vectors` and
`scripts/triage_recording_vectors.py`. **One row per recording.** `schema_version` is `3`; any
change to a column or a byte layout bumps it and changes this file with it. The same number is in
the parquet's own key-value metadata, under `senselab.recording_vectors.schema_version`, so a
reader can check it before decoding a byte.

What each bump added:

| version | added |
| --- | --- |
| **1** | the identity and decision columns, the measurements and the binary blocks |
| **2** | `release_ground` beside `release`, and with it the release vocabulary's fourth state, `nothing_to_redact` — a recording that needed no redaction is a determination, not an absence of one |
| **3** | VERDICT's gates (§6), the multi-speaker instrument (§7), the enhanced/residual levels (§8), and four measurement names |

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
| `release` | string | `releasable` \| `withheld` \| `nothing_to_redact` \| `not_assessed` | the fold wrote none |
| `release_ground` | string | why the release axis reads as it does — one of the seven controlled grounds `vocabulary.py` declares: four behind `nothing_to_redact`, three behind `not_assessed` | **REDACT itself decided**, so the state stands on its own verdict and needs no ground |
| `grounds` | string | VERDICT's `discard_ground` | **nothing was discarded** — the common case |
| `route_state` | string | e.g. `routed`, `declined` | the fold wrote none |
| `duration_s` | double | seconds, the **source** recording | ADMIT's `recording` stream entity is absent |
| `duration_conditioned_s` | double | seconds, the conditioned stream | PREPROCESS wrote no stream |
| `time_scale_s` | double | seconds — **the denominator for every `uint16` time** | neither duration is known |
| `sampling_rate` | int32 | Hz, of the conditioned stream | no conditioned stream |
| `schema_version` | int32 | `3` | never |
| `malformed_store_lines` | int32 | lines of `store.jsonl` that did not parse; `0` is the normal value | never |
| `flags_n` | int32 | how many node verdicts in the fold carry outcome `flag`, `fail` or `discard` — the same filter `report.py` calls a flag | never |
| `flag_nodes` | list\<string\> | which nodes those were, e.g. `["SPEECH"]` | never; `[]` when none |
| `conformance_airway` \| `_speech` \| `_voice` \| `_quality` | string | `true` \| `false` \| `undetermined` | **that node wrote no branch report** — it did not run |
| `route_airway` \| `_speech` \| `_voice` | string | e.g. `routed`, `declined` | routing wrote no state for it |
| `pii_findings_n` | int32 | how many `pii` entities the store holds | **the PII scan did not run.** A recording never scanned is null, not `0`. `0` means scanned and clean |
| `wave_peak` | double | amplitude; the scale `wave_minmax` is encoded against | no stream decoded |
| `floor_dbfs` | double | dBFS; the noise floor, a **scalar** — see §9 | `energy_envelope` is absent |
| `spans_unrowed_n` | int32 | general spans with no five-row code, left out of `spans` — see §9 | PREPROCESS did not run |

Three further column groups belong to the decision record and are large enough to have sections of
their own: the gates VERDICT resolved (§6), the multi-speaker instrument (§7) and the
enhanced/residual levels (§8).

---

## 3. The measurements

The 33 names the graph writes: the 29
`specs/20260817-triage-workflow-dag/measure-distributions.md` enumerates over the 2026-09-21
corpus, and the four the multi-speaker instrument added after that run
(`extent_dominant_speaker_share`, `extent_secondary_source_s`, `extent_source_active_s`,
`extent_speaker_count`), whose distributions nothing has measured yet. Each gets **two** columns,
and one gets a third:

- **`m_<name>`** — the reading. **Null when the graph wrote none for this recording.**
- **`m_<name>_n`** — `int32`, **never null**, how many readings the store held. `0` is a fact: the
  graph ran and wrote nothing.

> **The invariant, enforced by a test:** `m_<name> IS NULL` ⟺ `m_<name>_n == 0`. A reading of
> `0.0` is a measurement and reads as `0.0` with `_n == 1`. Null is not zero anywhere in this file.

**21 scalar numerics** — `double`, the **arithmetic mean** of the recording's readings. Identical
to the reading when `_n == 1`, which is the case for 13 of the 17 that the 2026-09-21 corpus
measured. Four of those are written per span or per track and routinely repeat —
`breath_peak_over_floor_db`, `cough_peak_over_floor_db`, `phonation_onset_to_offset_s`,
`interruptions` — so read `_n` before treating one as a single measurement. The four the
multi-speaker instrument added are written **once per task extent**, and `extent_source_active_s`
once per separated source inside each extent, so all four repeat on any recording carrying more
than one extent; how often, over the corpus, is not yet measured.

`breath_coverage_fraction`, `breath_peak_over_floor_db`, `cough_peak_over_floor_db`,
`ddk_ppg_period_dispersion`, `ddk_repetition_count_from_ppg_decode`,
`ddk_syllable_rate_from_envelope_modulation_hz`, `ddk_syllable_rate_from_ppg_decode_hz`,
`expected_sequence_repeat_fraction`, `extent_dominant_speaker_share`,
`extent_secondary_source_s`, `extent_source_active_s`, `extent_speaker_count`,
`glide_extent_semitones`, `interruptions`, `pause_fraction_of_response`,
`phonation_onset_to_offset_s`, `source_content_coverage`,
`speech_rate_from_consensus_words_per_s`, `train_fraction_of_recording`,
`verbatim_overlap_fraction`, `voiced_duration_s`.

**`m_extent_dominant_speaker_share` is the mean, and `extent_dominant_speaker_share_min` beside it
is the min.** The two disagree whenever a recording carries more than one task extent; §7 says
which one VERDICT's gate reads.

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

## 6. The gates

`schema_version` 3 carries every gate VERDICT resolved. A **gate** says what reading is good
enough. Its bound resolves **family → group → default**, most specific first and key by key, so
two rows of the same corpus can be judged against different bounds for the same gate — which is
why the bound is stored per row rather than looked up from the config at read time.
`nodes/gates.py` holds the one declaration of the set; `GATE_NAMES` is pinned against `GATE_KEYS`
by a test, so a gate added to the registry fails that test until this schema is bumped with it.
The design is `specs/20260921-gates-in-verdict/design.md`.

Three columns per gate, 22 gates, 66 columns:

| column | type | what it carries |
| --- | --- | --- |
| `gate_<name>` | double | the reading VERDICT read |
| `gate_<name>_bound` | double | the bound it resolved for this recording |
| `gate_<name>_passed` | string | `true` \| `false` \| `undetermined` |

Nullability, one case at a time, each stated by its own test:

- **All three are null when the fold did not apply that gate.** No layer named a bound for this
  recording's task group, or the fold resolved no group at all. Not applied is not failed.
- **`_passed` is `undetermined` when either the reading or the bound was absent.** A gate whose
  reading is absent yields UNDETERMINED, **never `false`** — nothing was measured, so nothing
  refused. `gate_<name>` is then null while `gate_<name>_bound` may still carry the bound it
  would have been read against.
- **A reading of `0.0` is a measured value and is not absent.** `gate_train_min_s == 0.0` with
  `_passed == "false"` is a train of zero seconds against a bound of one second, which is a real
  refusal. Null is not zero in these columns either.

A gate is **either a conformance term or a flag ground, never both** — `gates.py` raises at import
if a name appears in both tables — and **both kinds land in the same three columns**. A flag gate
says something about the recording's circumstances rather than about whether the participant
performed the instruction, so it decides no conformance; which kind a refusal was is read off
`gate_flagged_names`, which carries only the flag gates that failed, not off the per-gate
columns, which are identical for the two kinds. `FLAG_GATES` today names one gate,
`dominant_speaker_share_min`.

The 22, in column order, with the reading each is read against and the direction it compares:

| gate | reading | op |
| --- | --- | --- |
| `production_min_s` | `carrier_duration_s` | `at_least` |
| `voiced_fraction_min` | `carrier_voiced_fraction` | `at_least` |
| `f0_spread_max_semitones` | `carrier_f0_spread_semitones` | `at_most` |
| `continuity_min` | `carrier_continuity` | `at_least` |
| `dominant_segment_min_fraction` | `sweep_dominant_fraction` | `at_least` |
| `monotone_tolerance_semitones` | `sweep_monotone_reversal_semitones` | `at_most` |
| `expected_tokens_matched_min` | `expected_tokens_matched` | `at_least` |
| `omissions_max` | `expected_tokens_omitted` | `at_most` |
| `response_min_s` | `response_duration_s` | `at_least` |
| `coverage_min` | `source_content_coverage` | `at_least` |
| `dominant_speaker_share_min` | `extent_dominant_speaker_share` | `at_least` |
| `items_min` | `items_produced` | `at_least` |
| `events_min` | `airway_events_found` | `at_least` |
| `repetitions_min` | `ddk_repetitions_found` | `at_least` |
| `repeat_overlap_min` | *located* | `at_least` |
| `echo_overlap_max` | *located* | `at_most` |
| `verbatim_overlap_max` | *located* | `at_most` |
| `gap_off_task_min_s` | *located* | `at_least` |
| `interval_max_s` | *located* | `at_most` |
| `score_min` | *located* | `at_least` |
| `train_min_s` | *located* | `at_least` |
| `rate_prominence_min` | *located* | `at_least` |

A gate marked *located* produces a finding that carries an extent — a rejected carrier, a located
deviation, a per-event count — and is therefore applied inside the reporting node that knows where
the extent is, against the same bound this table names. It reaches these columns the same way,
through the fold's record.

Beside them, the fold's own summary, once per row:

| column | type | null means |
| --- | --- | --- |
| `gate_node` | string | no task group resolved |
| `gate_group` | string | no task group resolved |
| `gate_family` | string | no task group resolved, or the recording declared no family — the out-of-family mode, which reads only the group and default layers |
| `gate_applied_n` | int32 | never null; `0` means no group resolved |
| `gate_flagging_n` | int32 | never null |
| `gate_failed_n` | int32 | never null; `0` means nothing refused |
| `gate_undetermined_n` | int32 | never null; `0` means every applied gate could be answered |
| `gate_failed_names` | list\<string\> | never null; `[]` when nothing refused |
| `gate_flagged_names` | list\<string\> | never null; `[]` when no flag gate refused |

`gate_applied_n` and `gate_flagging_n` count the records the fold wrote, whatever they are named.
`gate_failed_n`, `gate_undetermined_n` and `gate_failed_names` are read off the outcomes of the
gates **this schema has a column for**, so a name the registry has gained and this file has not
lands in the two counts and nowhere else — which is the observable that the pinning test exists to
make loud. A gate the fold records in both lists holds one outcome, not two.

---

## 7. The multi-speaker instrument

What the separation and localisation steps found, as five scalars. The design is
`specs/20260922-the-multi-speaker-instrument/design.md`, and the task-extent readings are
`specs/20260922-speakers-within-the-task-extent/design.md`.

| column | type | units | null means |
| --- | --- | --- | --- |
| `separated_n` | int32 | sources | never null; **`0` means separation did not run** |
| `secondary_extent_n` | int32 | runs | never null |
| `secondary_extent_s` | double | seconds | no secondary run was located |
| `solo_extent_s` | double | seconds | no solo run was located |
| `extent_dominant_speaker_share_min` | double | fraction | no task extent carried a share reading |

`separated_n` counts the live streams whose name begins `separated_`. `secondary_extent_n` counts
the spans with role `secondary_source_extent`, and `secondary_extent_s` totals their seconds;
`solo_extent_s` totals the seconds of the spans with role `solo_extent`. The count is never null
and the seconds are, because a count of zero is a fact about a run that happened and a total of
zero seconds would be indistinguishable from one.

**`extent_dominant_speaker_share_min` is folded by `min`, and that is the whole reason it exists
beside `m_extent_dominant_speaker_share`.** The min is the fold VERDICT's flag gate applies — the
worst extent answers, because one extent another speaker holds is the finding, whatever the others
did — while `m_extent_dominant_speaker_share`, like every `m_` column in §3, is the **arithmetic
mean**. The two disagree whenever a recording carries more than one task extent: a row reading,
say, 0.65 in the `m_` column against 0.4 in the `_min` is not an inconsistency, it is two extents,
one of them shared.

The secondary and solo spans also reach `branch_lanes` (§5), on the `SPEECH` lane, under the roles
`secondary_source_extent` and `solo_extent` — so *where* the second voice is can be drawn without
reading these scalars back.

---

## 8. The enhanced and residual levels

Five doubles, **all null together when PREPROCESS wrote no residual decomposition**.

| column | type | units |
| --- | --- | --- |
| `residual_peak_dbfs` | double | dBFS |
| `residual_rms_dbfs` | double | dBFS |
| `enhanced_rms_dbfs` | double | dBFS |
| `enhanced_over_residual_rms_db` | double | dB |
| `enhanced_over_residual_rms_fitted_db` | double | dB |

**Only the residual's own levels are stored by the graph.** The `residual` measurement carries
`peak_dbfs` and `rms_dbfs` for the residual stream and nothing equivalent for the enhanced one.
The other three columns are **reconstructed** from the two energy fractions the same measurement
carries, `enhanced_energy_fraction` and `energy_fraction`: they share one denominator and one
aligned length, so their ratio is a ratio of mean square amplitudes and

```
enhanced_over_residual_rms_db = 10 * log10(enhanced_energy_fraction / energy_fraction)
```

is **exactly** the RMS difference in dB, not an approximation of it. `enhanced_rms_dbfs` is then
`residual_rms_dbfs` plus that difference. The `_fitted_` variant adds the decomposition's own
`gain_db` — `plain = g · enhanced + residual`, so the fitted variant is the **gain-fitted speech
component** measured against the residual, which is the pair the decomposition actually solved
for. The derivation is `specs/20260923-enhanced-over-residual/design.md`.

Either fraction absent, zero or negative yields null rather than an infinity — the residual's own
stored levels still stand in that case. `enhanced_rms_dbfs` additionally needs `rms_dbfs`, since
it is that level plus the difference; without it the difference stands alone. `gain_db` absent
yields null for the fitted column alone.

**The enhanced stream's peak is not derivable this way and is therefore not carried.** A peak is
one sample, an energy fraction is a sum over all of them, and no ratio of sums recovers an
extremum — a column would have to be either a second stored measurement or a guess, and a guess
about a clipping headroom is worse than an absent column.

---

## 9. Where the design and the stores disagreed

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

**"Every numeric measurement" is 21 of the 33 names, not 33.** Of the names the graph writes, 21
carry numbers, 1 carries a vector, 1 a matrix, and 10 carry strings — 9 of those only ever the
sentinel `NOT_SEPARABLE_BY_THIS_DESIGN`. Each still gets a column, typed by what the graph
actually writes; only the 21 are usable as a parallel-coordinate axis directly.

A fourth, smaller one: **a measurement name is not unique within a recording.** Four of the 21
numerics are written per span or per track, four more once per task extent, and `carrier_rejected`
is written a mean of six times. A single column per name therefore needs a reduction, which is why
`_n` exists beside every one — and why one reading, the dominant speaker's share, carries a second
column under a second reduction (§7).

---

## 10. Running it

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
[`measurements.md`](measurements.md). **Those figures are the `schema_version` 1 build.** The
columns sections 6, 7 and 8 add are scalars and short string lists rather than blocks, so they
cost far less per row than a block does — but the schema-3 build is still running on the cluster,
so no size, no null count and no coverage figure for any of them is measured yet, and none is
stated here.

Each shard writes `recording_vectors.NNN.parquet` (mode 600) and `recording_vectors.NNN.report.json`
with `considered`, `written`, `incomplete`, `unreadable`, `superseded` and `anomalies` — the last
being measurement names the store carried that this schema has no column for. The merge writes
`recording_vectors.parquet` (mode 600) and `recording_vectors.summary.json`, which carries the
per-column null counts.

## Row groups: why 1024

The first build wrote one row group for all 62,488 rows, which is pyarrow's default
(`row_group_size` 1,048,576). Parquet's unit of skippable IO is the column chunk, and a column
chunk is per row group, so a single row group means a reader wanting one recording's eight blob
columns must read those columns *entire*. Measured on that file: **79.9 MB to open one recording.**
The viewer hid this behind decode caching, but it is 87% of the file for one row.

Rewriting the same table at several row-group sizes, with `write_page_index=True` throughout, and
costing the two reads the viewer actually performs — the ten axis columns over all rows at first
paint, and the eight blob columns for one row on selection:

| rows/group | groups | file MB | first paint MB | one selection MB |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 123 | 100.8 | 1.47 | 0.65 |
| **1024** | **62** | **97.3** | **1.33** | **1.29** |
| 2048 | 31 | 95.9 | 1.23 | 2.60 |
| 4096 | 16 | 94.1 | 0.96 | 5.23 |
| 8192 | 8 | 93.1 | 0.72 | 10.49 |
| 1048576 (default) | 1 | 91.3 | 0.46 | 79.91 |

The two costs cross at 1024, which is what the writer uses. It is the right side of the trade for
this page regardless of the crossing: first paint happens once and 1.33 MB is a third of the 4.06 MB
that painted in 498 ms, while a selection is the interaction the reader repeats and falls **62×**.
The file grows 6.6%, on a private local artefact.

`write_page_index` is separate and additive: it writes the offset and column indexes, so a reader
can seek within a chunk instead of decompressing from its start. It costs nothing measurable in
file size and is what makes a narrow read narrow at page granularity rather than row-group
granularity.
