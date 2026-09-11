# Posteriorgrams and Praat features over a completed triage run

## What this is

A batch driver, `scripts/extract_ppg_praat.py`, that reads the `enhanced.flac` of every recording
in a finished triage run and writes two things per recording: the full phonetic posteriorgram, and
the Praat/Parselmouth feature set. It is thin. Both extractions already exist — in
`senselab.audio.tasks.features_extraction.ppg` and `...praat_parselmouth` — and the driver calls
them; what is new here is the batching, the sharding, the resume and the output layout, plus the
handful of things that were missing from those two modules and were added there.

Input is the manifest at `/orcd/scratch/bcs/002/satra/ppg_20260911/manifest.jsonl`, 60,202 rows,
one JSON object per line carrying `stem`, `enhanced`, `family`, `duration_s` and `lexical`.

```
uv run python scripts/extract_ppg_praat.py MANIFEST OUT --slice-index I --slice-count N \
    [--batch-size 500] [--device cpu]
```

Task *i* of *n* takes `rows[i::n]` — a stride, not a block, so no task draws the long tail of the
corpus on its own.

## Batch size is the whole lever; the device is noise beside it

Measured on 32 real `enhanced.flac` tracks, mean duration 5.4 s, on an A100 node:

|      | batch 8              | batch 32             |
| ---- | -------------------- | -------------------- |
| CPU  | 15.0 s (1.87 s/file) | 15.7 s (0.49 s/file)  |
| CUDA | 11.2 s (1.41 s/file) | 10.1 s (0.32 s/file)  |

Twenty-four extra files cost **0.7 s** on CPU. Fitting the two CPU points gives roughly 14.8 s of
fixed cost per call — subprocess spawn plus model load — and about 0.005 s per second of audio
after that, about 185× realtime at the margin. Moving to CUDA buys 3.8 s off the fixed cost and
nothing meaningful off the marginal cost; going from batch 8 to batch 32 divides the per-file cost
by 3.8 on CPU because it amortises that fixed cost across four times as many files.

So the run targets CPU nodes, which are plentiful, and makes the batch large. The packaged default
is **500 recordings per `extract_ppgs_from_audios` call** (`DEFAULT_BATCH_SIZE`): at the mean
duration that is ~2700 s of audio, ~13.5 s of marginal work against 14.8 s of fixed cost, so the
call spends roughly half its time on the model load and the amortisation is past the knee. Larger
batches keep improving, but by then the fixed cost is already a minority of the call and the memory
held — 500 decoded waveforms, ~173 MB at 16 kHz float32, plus 500 WAVs in the subprocess's temp
directory — starts to be the binding constraint rather than the throughput.

The batch size is a parameter (`--batch-size`), because the two numbers above were measured at 5.4 s
mean duration and a corpus with a different duration distribution moves the knee.

Praat is separate: `extract_praat_parselmouth_features_from_audios` costs 0.59 s/file (9.1×
realtime) and returns 40 keys. It loads no model and spawns no subprocess, so it has no fixed cost
to amortise and nothing to gain from batching. The driver calls it once per recording, inside the
batch loop, which is what gives each recording its own error isolation for free.

## The venv must exist before the array is submitted

The first `extract_ppgs_from_audios` call on a host builds the isolated ppgs venv. That took
**810 s**. `ensure_venv`'s lock timeout is **600 s**, so an array task that starts while another is
mid-build does not get the venv — it waits out the window, logs, and (depending on the lock's
retry) either waits again or raises. Dozens of tasks racing one cold build is the worst case: all
of them idle on a compute allocation for the duration.

The driver therefore does not build one. `main` checks `ppgs_venv_is_provisioned()` — added to
`ppg.py`, over `provisioned_venv_dirs("ppgs")`, so it counts only a tree carrying the
`.senselab-installed` completion marker and never a half-built one — and if there is none it exits
2 with a message naming the pre-build step:

```
uv run python -c \
    "from senselab.audio.tasks.features_extraction import ensure_ppgs_venv; ensure_ppgs_venv()"
```

`ensure_ppgs_venv()` is also new in `ppg.py`, and is what `extract_ppgs_from_audios` itself now
calls, so the pre-build and the run cannot drift on requirements or Python version.

## Output layout

Mirroring the BIDS tree the run already uses, the entity path taken from the stem by
`entity_subdir` in `senselab.audio.workflows.triage.run` — not a second entity parser, and not a
flat directory. A flat directory is what this replaces: 125,263 entries in one directory exceeded
`ARG_MAX` and made the corpus impossible to enumerate.

```
<OUT>/sub-<label>/ses-<label>/<stem>_ppg.npz         the posteriorgram and its phoneme order
<OUT>/sub-<label>/ses-<label>/<stem>_features.json   the outcome row, with the Praat features
<OUT>/slices/slice-<i>-of-<n>.jsonl                  every row this task handled
<OUT>/slices/slice-<i>-of-<n>.summary.json           the task's counts and its provenance
```

`<stem>_ppg.npz` holds:

- `posteriorgram` — **float16**, frame-major `(frames, phonemes)` via `to_frame_major_posteriorgram`.
  The array is 40 phonemes at ~116 frames/s; at float16 the whole corpus is about 3.3 GB, which is
  what makes storing every frame rather than a summary affordable. float16 is lossless enough for a
  posterior: the values are in [0, 1] and float16 resolves ~10 bits of mantissa there, far finer
  than the model's own agreement with itself across runs.
- `phonemes` — the 40 labels in the posteriorgram's own column order, so a consumer never has to
  guess the inventory or its order. `PHONEME_LABELS` was private (`_PHONEME_LABELS`) and is now
  public for exactly this.
- `seconds_per_frame`, `duration_s`, `sampling_rate` — enough to put a frame index on the
  recording's clock without reopening the audio.

`<stem>_features.json` is the manifest row plus three added keys: `ppg` (its status, the npz's
name, and the frame and phoneme counts), `praat` (its status and the 40-key `features` dict), and a
top-level `status` folded from the two — `ok` when both landed, `partial` when one did, `error`
when neither did.

## Resume, and what counts as done

`<stem>_features.json` is written **last**, after the npz, and its existence is the completion
marker — the same shape as the triage driver skipping a recording that already has a summary. An
array task that dies mid-batch restarts and redoes only the recordings whose row never landed; a
recording with an npz but no row is redone, because the npz alone is not a completed recording.
Both files are written to a `.partial` sibling and `replace`d, so a task killed mid-write leaves no
truncated file that a later run would trust.

The skip is unconditional on the row's existence, including for an `error` row. To retry a
recording, delete its row. This is deliberate: a deterministic failure retried by every re-run
costs the whole corpus's worth of load attempts on every pass, and the row already says what
happened.

## One bad recording does not lose the other 499

Three failure modes, each recorded rather than raised:

- **The audio will not load.** Caught per recording during the load loop; that recording gets an
  `error` row and never enters the batch.
- **The model raises on one recording.** `extract_ppgs_from_audios` already returns a scalar NaN
  tensor for that file rather than raising. The driver treats it as an outcome: `ppg.status` is
  `"nan"`, no npz is written, Praat still runs, and the row is `partial`.
- **The whole ppgs call fails** — a subprocess that dies, a worker that returns the wrong number of
  posteriorgrams. Caught around the one call; every recording in the batch gets that message in its
  `ppg` block and keeps its Praat features. The remaining batches still run.

## What was added to the task modules rather than to the driver

- `ppg.PHONEME_LABELS` and `ppg.PPGS_SAMPLE_RATE`, public, both resolved from the `ppgs` library
  when it is importable and falling back to the 0.0.9 inventory and 16 kHz when it is not (it
  normally is not: ppgs lives in the subprocess venv).
- `ppg.ensure_ppgs_venv()` and `ppg.ppgs_venv_is_provisioned()`, as above.
- A sampling-rate guard in `extract_ppgs_from_audios`, beside the existing mono guard. The worker
  passes `ppgs.SAMPLE_RATE` to `ppgs.from_audio` for whatever it read off disk, so a file at any
  other rate was analysed as if it were at 16 kHz and came back a plausible-looking wrong answer.
  It is now refused, and the driver downmixes and resamples before calling.

## `voice.f0_range_hz` was null on a false premise

`voice.f0_range_hz` was `null`, and PREPROCESS reads it. A null there raised inside the
`phonation_tracks` block, so that block was recorded absent — which is why the phonation tracks are
missing from all 62,547 recordings of the completed run, and with them the F0 and formant tracks
the whole voice kind rests on.

The key's premise, written beside it, was that no single search range serves both a low adult male
fundamental and an infant voice, so the caller must state which population is being measured before
anything can run. The first half is true. The second does not follow, and `extract_pitch_values` in
`praat_parselmouth.py` had already shown why: it runs a wide 50–600 Hz autocorrelation search,
trims the outliers at |z| ≤ 2, and narrows to one of the two published settings by where the
trimmed mean falls — (60, 250) below 170 Hz, (100, 500) above. That is the pitch-range
standardization method it cites (doi:10.3758/BRM.41.2.318). No fixed corpus-wide range is needed,
because the range is derivable per recording, and a derived one is strictly better than a declared
one: it serves the adult male and the infant in the same run.

**Resolved in place, no flag beside it.** `voice.f0_range_hz` is **deleted**.
`voice.f0_search_range_hz: [50.0, 600.0]` replaces it — the wide search the narrowing starts from,
which is a property of the method and not a population guess. A new
`derive_f0_range(audio, search_floor_hz=, search_ceiling_hz=)` in
`senselab.audio.tasks.phonation.api` wraps `extract_pitch_values` and returns the narrowed
`(f0_min_hz, f0_max_hz)`. PREPROCESS's `_phonation_tracks` and VOICE's `_f0_range` both call it, both
off the `plain` stream, so the two still cannot hold ranges that drift — the no-drift property that
motivated one shared key survives, now as one shared derivation.

`voice.f0_range_by_population` stays and still wins when the hint declares a population: a caller
who knows the population may still state the range and skip the derivation. Nothing else changes.
`voice.f0_range_ratio_max` now bounds the resolved range, whichever way it was resolved.

Two consequences worth saying plainly:

- **PREPROCESS behaves differently and `config_hash` changes.** The phonation-tracks pass now runs
  under the packaged config instead of being recorded absent. The existing corpus is not affected —
  this driver reads stored `enhanced.flac`, not the store — but any *new* triage run carries a
  different config hash and a phonation-tracks measurement its predecessors did not have.
- **`extract_pitch_values` no longer guesses on silence.** With no voiced frame at all, `np.mean`
  over an empty array gave NaN, `NaN < 170` is `False`, and the function fell through to the
  "female and child" branch and returned (100, 500) — a guessed range that read as a derived one.
  It now returns NaN for both, and `derive_f0_range` raises on that, so an underivable range is an
  absence. In PREPROCESS the block is recorded absent, as it was; in VOICE the branch refuses
  before the store is written to.

## `phonation.hnr_floor_interval_db` and `phonation.rms_floor_interval` stay null

Checked for the same resolution and it does not apply. These two are the near-edge intervals VOICE
reads a span's onset gate values against (N22). They were measured in normalised-autocorrelation
units — (0.44, 0.933) and (0.0007, 0.0161) — and the implementation reads Praat harmonicity in dB
and RMS, so the numbers do not transfer.

Praat does not self-calibrate either of them. Its harmonicity analysis has a `silence_threshold`
relative to the global peak, which is already a separate config key
(`phonation.silence_threshold: 0.1`), and it exposes no dB floor and no RMS interval at all. There
is nothing to derive per recording the way the F0 range is derived, so resolving these two requires
a measurement in the implementation's own units that nobody has taken.

They stay null. While null the near-edge row is inert and the verdict records
`gate_interval: "unmeasured"` — the honest state, and distinguishable from `"partial"` (exactly one
supplied) and `"measured"` (both). Only the config comments change, from "prior work measured it in
units that do not transfer" to naming why derivation is not available here either.

## Tests

`src/tests/scripts/extract_ppg_praat_test.py`, over synthetic tones and a temporary manifest.
`extract_ppgs_from_audios` is monkeypatched in the driver module and
`ppgs_venv_is_provisioned` is stubbed true — nothing here builds a venv or loads the model. Praat
is real, because it is fast and has no venv. Covered: the BIDS-shaped path for a stem with sub and
ses and for a stem with neither; resume skipping a completed recording and redoing a half-written
one; the ragged final batch (7 recordings at batch 3 is 3 + 3 + 1); a NaN posteriorgram recorded as
an outcome with the rest of the batch intact; an unreadable recording and a whole-batch failure,
same; the phoneme order and float16 layout persisted with the data; and the missing-venv refusal
naming `ensure_ppgs_venv`.

`derive_f0_range` has its own tests in `src/tests/audio/tasks/phonation_test.py`: a 110 Hz buzz and
a 230 Hz buzz narrow to different ranges, and silence refuses rather than returning one.
