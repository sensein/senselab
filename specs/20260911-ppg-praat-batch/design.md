# Posteriorgrams and Praat features as PREPROCESS blocks, and the pass that back-fills them

## What this is

Two new PREPROCESS blocks — `ppg_posteriorgram` and `praat_features` — plus one driver,
`scripts/extend_ppg_praat.py`, that runs those same two blocks over runs that finished before they
existed. Every derivative lands in the recording's own `run/derivatives/` and every one registers a
store entity, so both reach `store.jsonl` and the four BEP028 files by the route every other
derivative already takes.

## What the first attempt got wrong

`d45de3ba` shipped `scripts/extract_ppg_praat.py`: a standalone batch driver writing
`<OUT>/sub-*/ses-*/<stem>_ppg.npz` and `<stem>_features.json` into a **second output tree**, keyed
by the same BIDS entity path as the runs but sharing nothing else with them. It never touched
`ProvStore`.

The owner rejected that shape, and the reasons are structural rather than stylistic:

- **No provenance.** Nothing recorded which model produced the posteriorgram, at what settings, from
  which stream, under which environment. The npz was a file on a disk with a name.
- **No checksum, size or mtime.** `path_attributes(...)` is what puts those three into the store and
  out to `prov-triage_io.json`. A derivative that skips it cannot be verified later, and a corpus
  half of whose derivatives can be verified is a corpus none of whose derivatives can be trusted.
- **No environment capture.** ppgs runs in its own interpreter. That interpreter's torch version
  decides the numbers, and no record of it existed.
- **Nothing else reads it.** Every consumer in the graph — TAXONOMY, the branches, REPORT, the
  BEP028 export — reads the store. A tree beside the store is a tree with no readers.

The `f0_range_hz` → `f0_search_range_hz` rename in the same commit is a separate, correct change and
stays; see the section at the end.

## The two blocks

Both are module-level functions in `nodes/preprocess.py` with a one-line closure in the block list,
rather than closures like every other block. That is the one deviation from the existing pattern and
it is forced: the driver has to call the same code, and a closure over `preprocess`'s local `state`
cannot be called from outside it. Everything else is the pattern — `_activity(...)` for the
sub-activity, `write_measurement(...)` for the entity, `path_attributes(...)` for the file facts,
`derived_from` naming the `enhanced` stream entity.

### `ppg_posteriorgram`

`derivatives/ppg_posteriorgram.npz` holds:

- `posteriorgram` — **float16**, frame-major `(frames, phonemes)` via `to_frame_major_posteriorgram`.
  40 phonemes at ~116 frames/s; at float16 the whole corpus is about 3.3 GB. float16 is lossless
  enough for a posterior: the values are in [0, 1] and float16 resolves ~10 bits of mantissa there,
  far finer than the model's own agreement with itself across runs.
- `phonemes` — the 40 labels in the posteriorgram's own column order.
- `seconds_per_frame`, `duration_s`, `sampling_rate`.

The entity carries the path, its SHA-256, its size, its mtime, the frame and phoneme counts, the
frame rate, the dtype and the layout. **It never carries the array.** At 3.3 GB corpus-wide, inlining
it would put the posteriorgram inside `store.jsonl` and inside every BEP028 export of it. The rule is
the same one `energy_envelope.npz` and `gammatone.npz` already follow: an array is referenced by
path and digest.

### `praat_features`

Forty scalars, and they are attributes. A sidecar for forty numbers would add an indirection and a
second file to checksum, and buy nothing. A non-finite scalar — Praat placing no jitter on an
unvoiced recording — is recorded as `null`, which is JSON's only representation of an absent number;
`NaN` in a JSON document is a Python extension that a strict reader of the exported `prov/` files
would reject.

The two settings come from the config: `praat_features.time_step_s` and
`praat_features.window_length_s`, with their derivations in
`specs/20260817-triage-workflow-dag/config-derivations.md`.

### Both run on `enhanced`

Owner-specified. Neither runs on `plain`.

## Read-if-present, and how it is guaranteed

The requirement is that an extend pass over an existing run and a fresh full run write the same
entities. The guarantee is mechanical rather than argued: **neither block reads `preprocess`'s
`state` at all.** Both call `resolve_stream(store, run_dir, "enhanced")`, which is the store's shared
read rule — invalidated entities skipped, latest live write wins — and load the audio from the
sidecar the stream entity names.

So in a fresh run, where `_residual` has just written the `enhanced` stream and holds the same audio
in memory, the block still goes back to the store and back to the flac. It costs one decode of a
~5 s file against a ~15 s model call. What it buys is that the fresh path and the extend path are
the same three lines over the same bytes, and the entity's `derived_from` names the same stream id
in both, because `resolve_stream` returns exactly the entity `_residual` wrote.

`ppg_input` is the single conditioning step (downmix to mono, resample to `PPGS_SAMPLE_RATE` when the
run's `resample.target_hz` differs). The driver calls it to build its batch and the block calls it
when it extracts for one recording, so a batched call and a single call hand the model identical
samples.

What is *not* identical between the two paths, and cannot be: the PREPROCESS **verdict**. A fresh
run's verdict lists both blocks in its `derivatives` map, or names them in its `absent` map. An
extend pass cannot amend a verdict that is already written — the store is append-only and a verdict's
id is a digest of its attributes. The measurement entities, their activities, their agent and their
`wasDerivedFrom` edges are identical; the verdict of an extended run is the one its own run wrote.
Said plainly here so nobody reads a diff of two fingerprints and concludes the blocks diverged.

## Absence is typed

`extract_ppgs_from_audios` used to return `torch.tensor(float("nan"))` for a recording the model
raised on. That is a sentinel travelling as data: inside a block it would have been written to an npz,
checksummed and registered as a posteriorgram.

It now returns `list[torch.Tensor | PpgsPosteriorgramUnavailable]` — one entry per input, the entry
being the typed error where the model produced nothing, carrying the worker's own message.
`require_posteriorgram(entry)` unwraps or raises. `PpgsPosteriorgramUnavailable` subclasses
`ValueError`, so PREPROCESS's block runner catches it in the same `(ValueError, LookupError)` arm as
`SpanTooShortForYAMNet`, `AudioTooShortForAST` and `CrisperWhisperDecoderPositionsExceeded`: the
recording records a named absence and every other block still runs.

Returning the error object rather than raising is what keeps the batch: one bad recording in 500 must
not lose the other 499, and a positional list cannot express "nothing here" without either a sentinel
or a typed placeholder. `asyncio.gather(return_exceptions=True)` is the same trade.

`extract_features_from_audios` now calls `require_posteriorgram`, so a NaN can no longer reach a
feature dict either.

## Environments

The ppgs venv is a different interpreter with its own torch, so it needs its own environment record.

In a fresh run this needs no new code and gets none: `run_triage` already wraps node execution in
`record_venv_use()`, `extract_ppgs_from_audios` calls `ensure_ppgs_venv()` → `ensure_venv(...)`, which
notes the resolved directory, and `capture_environments` turns the noted directory into an
`Environment` record beside the host's. The block inherits the property.

The driver reproduces it explicitly: it opens `record_venv_use()` around the whole slice and calls
`capture_environments(store, used_venvs)` before writing each recording's store. The first batch's
ppgs call resolves the venv before any store is written, so every recording in the slice carries the
record.

## The driver

```
uv run python scripts/extend_ppg_praat.py MANIFEST --slice-index I --slice-count N \
    [--batch-size 500] [--device cpu] [--log-dir DIR] [--config OVERRIDE.yaml]
```

Input is the manifest at `/orcd/scratch/bcs/002/satra/ppg_20260911/manifest.jsonl`, 60,202 rows, one
JSON object per line carrying `stem`, `enhanced`, `family`, `duration_s` and `lexical`. **Those
60,202 rows are not the corpus** — 2,376 of the 62,578 stores on disk are absent from this manifest
and hold neither measurement; see *The manifest is not the corpus* below before quoting either
measurement's coverage.

**What it relies on that the manifest does not state.** The manifest carries no run root. The driver
derives one from `enhanced`: `<run_root>/run/streams/enhanced.flac`, so the run root is the path's
third parent, the store is `<run_root>/run/store.jsonl` and the BEP028 tree is `<run_root>/prov/`.
This is `prepare_run_layout`'s layout and the layout `specs/20260908-triage-prov-bep028/design.md`
fixed for `prov/`. A path of any other shape is refused by name rather than guessed at, because a
wrong guess extends the wrong recording's store.

Per recording: read `store.jsonl` with `ProvStore.read_jsonl(path, run_id=<run root name>)`, run
whichever of the two blocks the store is missing, capture environments, write the store back, and
re-export `prov/`.

**`run_id` is the run root's own name, not the default `"read"`.** Entity ids are
`sha256([run_id, prov_type, extent, attributes])`, so reading under any other id would mint the new
entities in a namespace the rest of the store does not share, and a second pass under a third id
would mint them again.

### Convergence

The module docstring of `prov_store.py` says merging two stores is a set union and is
order-independent. That holds for what this driver writes, but it is not by itself enough to make a
rerun converge, and the difference matters:

- Entities, activities, agents and environments are dicts keyed by a content digest, and relations
  are membership-checked, so adding the *same* record twice is a no-op. Union, order-independent.
- But `path_attributes` includes `mtime_ns`. Re-running the ppg block would write a byte-identical
  npz at a new mtime, which is a different attribute set, which is a **different entity id**. The
  union would then hold two posteriorgram entities, both live, differing only in mtime.

So convergence comes from the driver being read-if-present at the store level: a recording whose
store already holds a live `ppg_posteriorgram` measurement does not re-extract, and one that already
holds `praat_features` does not re-run Praat. A completed recording is skipped whole and its store is
not rewritten at all. A recording that has one of the two gains only the other. A task killed
mid-slice restarts and redoes only what never landed. `store.jsonl` is written to a `.partial`
sibling and `replace`d, so a task killed mid-write leaves no truncated store for the next pass to
read. The test asserts `store.fingerprint()` is unchanged across a second full pass.

A recording whose posteriorgram the model refuses is retried on the next pass, because the store
holds nothing to skip on. That is right for a transient failure and cheap for a deterministic one:
the slice log names it, and an operator excludes it from the manifest. The alternative — writing an
"absent" entity so the next pass skips it — would make the extend path's graph differ from the fresh
path's, where an absence lives in the PREPROCESS verdict's `absent` map and nowhere else.

### BEP028 re-export

After each recording is extended, `to_bep028_graph(store)` + `write_bep028_files(graph, <run_root>/prov)`
rewrite `prov-triage_{io,act,soft,env}.json` from the merged store. Without this, `prov/` would name a
graph the store no longer holds — the export exists precisely so a reader can work from `prov/`
without parsing `store.jsonl`, and a stale one is worse than an absent one.

### Failure isolation

Three modes, each recorded rather than raised:

- **The store will not open, or the path names no run root.** That recording gets an `error` record
  and never enters the batch.
- **The model produced no posteriorgram for one recording.** The typed absence is that recording's
  `ppg` outcome; no npz is written, Praat still runs, and the record is `absent`.
- **The whole ppgs call fails** — a dead subprocess, a wrong-length result. Every recording in the
  batch gets that message in its `ppg` field and keeps its Praat features. The remaining batches run.

The per-slice log goes to `<log-dir>/slices/slice-<i>-of-<n>.jsonl` with a `.summary.json` beside it.
That is the only thing written outside a recording's own run root, and it is a log, not a derivative.

## Batch size is the whole lever; the device is noise beside it

Measured on 32 real `enhanced.flac` tracks, mean duration 5.4 s, on an A100 node:

|      | batch 8              | batch 32              |
| ---- | -------------------- | --------------------- |
| CPU  | 15.0 s (1.87 s/file) | 15.7 s (0.49 s/file)  |
| CUDA | 11.2 s (1.41 s/file) | 10.1 s (0.32 s/file)  |

Twenty-four extra files cost **0.7 s** on CPU. Fitting the two CPU points gives roughly 14.8 s of
fixed cost per call — subprocess spawn plus model load — and about 0.005 s per second of audio after
that, about 185× realtime at the margin. CUDA buys 3.8 s off the fixed cost and nothing meaningful
off the marginal cost; batch 8 → 32 divides the per-file cost by 3.8 on CPU purely by amortising that
fixed cost over four times as many files.

So the run targets CPU nodes, which are plentiful, and makes the batch large. The default is
**500 recordings per `extract_ppgs_from_audios` call** (`DEFAULT_BATCH_SIZE`): at the mean duration
that is ~2700 s of audio, ~13.5 s of marginal work against 14.8 s of fixed cost, so the call spends
roughly half its time on the model load and the amortisation is past the knee. Larger batches keep
improving, but by then the fixed cost is a minority of the call and the memory held — 500 decoded
waveforms, ~173 MB at 16 kHz float32, plus 500 WAVs in the subprocess's temp directory — becomes the
binding constraint. `--batch-size` stays a flag because the numbers were measured at 5.4 s mean
duration and a corpus with a different duration distribution moves the knee.

Praat is separate: `extract_praat_parselmouth_features_from_audios` costs 0.59 s/file (9.1× realtime)
and returns 40 keys. It loads no model and spawns no subprocess, so it has no fixed cost to amortise
and nothing to gain from batching. The driver calls it once per recording, which is also what gives
each recording its own error isolation for free.

This is why the block and the driver split where they do. Inside PREPROCESS the ppg block extracts
for one recording and pays the 14.8 s fixed cost, which is unavoidable when a run is one recording.
The driver, which has 60,202 of them, calls `ppg_input` per recording, `extract_ppgs_from_audios`
once per 500, and `write_ppg_posteriorgram` per recording — the same write path, fed from a batch.

`rows[i::n]` sharding is kept from `d45de3ba`: task *i* of *n* takes a stride, not a block, so no task
draws the corpus's long tail on its own.

## The venv must exist before the array is submitted

The first `extract_ppgs_from_audios` call on a host builds the isolated ppgs venv. That took **810 s**.
`ensure_venv`'s lock timeout is **600 s**, so an array task that starts while another is mid-build
waits out the window and raises. Dozens of tasks racing one cold build is the worst case: all of them
idle on a compute allocation for the duration.

The driver therefore does not build one. `main` checks `ppgs_venv_is_provisioned()` — over
`provisioned_venv_dirs("ppgs")`, so it counts only a tree carrying the `.senselab-installed`
completion marker and never a half-built one — and exits 2 naming the pre-build step:

```
uv run python -c \
    "from senselab.audio.tasks.features_extraction import ensure_ppgs_venv; ensure_ppgs_venv()"
```

Both `ppgs-cpu` and `ppgs-cu128` are pre-built on the cluster.

`run_triage` has no such gate, because a fresh run is not an array task: it builds the venv if it must
and pays the 810 s once per host. Worth stating: adding these two blocks means a triage run now
touches a sixth subprocess venv, and the first run on a new host is 810 s slower than its
predecessors were.

## Tests

`src/tests/audio/workflows/triage/nodes/preprocess_test.py::TestThePosteriorgramAndPraatBlocks`:

- the posteriorgram is a sidecar under `derivatives/` whose entity carries the path, a 64-hex
  SHA-256, a size and the shape — and does **not** carry the array;
- its agent is a model agent with no commit and a stated reason, and its `wasDerivedFrom` names the
  `enhanced` stream entity;
- Praat's scalars are attributes, with no `path`, and every value finite or `null`;
- both blocks read the stream back out of the store: the samples the block handed the model are the
  samples `ppg_input` returns when called on the finished store, which is all an extend pass has;
- a `PpgsPosteriorgramUnavailable` is recorded absent by name, writes no npz, does not raise out of
  PREPROCESS, and does not stop Praat;
- no `enhanced` stream at all makes both a cascading absence.

`src/tests/scripts/extend_ppg_praat_test.py`: the run root derived from the enhanced path and refused
for any other shape; `rows[i::n]` partitioning; the ragged final batch; both measurements merged into
an existing store; nothing written outside `run/` and `prov/`; the BEP028 files re-exported and naming
both new entities; the ppgs venv's own environment record beside the host's; a rerun converging on the
same `fingerprint()` with exactly one posteriorgram entity; a half-extended run gaining only what it
lacked; a typed absence, a missing store and a whole-batch failure each isolated to their own rows;
one ppgs call per batch rather than one per recording; and the missing-venv refusal naming
`ensure_ppgs_venv`.

Nothing in either file calls ppgs. `extract_ppgs_from_audios` is monkeypatched — on the node module
for the block tests, on the driver module for the driver tests — and the venv gate is stubbed. Praat
is real: it is fast and has no venv.

## What was added to the task modules rather than to the driver

- `ppg.PHONEME_LABELS` and `ppg.PPGS_SAMPLE_RATE`, public, resolved from the `ppgs` library when it is
  importable and falling back to the 0.0.9 inventory and 16 kHz when it is not (it normally is not:
  ppgs lives in the subprocess venv).
- `ppg.ensure_ppgs_venv()` and `ppg.ppgs_venv_is_provisioned()`.
- `ppg.PpgsPosteriorgramUnavailable` and `ppg.require_posteriorgram`, and the worker reporting a
  per-index error instead of writing a NaN array.
- A sampling-rate guard in `extract_ppgs_from_audios`, beside the existing mono guard. The worker
  passes `ppgs.SAMPLE_RATE` to `ppgs.from_audio` for whatever it read off disk, so a file at any other
  rate was analysed as if it were at 16 kHz and came back a plausible-looking wrong answer.

## `voice.f0_range_hz` was null on a false premise

Unchanged from `d45de3ba` and kept.

`voice.f0_range_hz` was `null`, and PREPROCESS reads it. A null there raised inside the
`phonation_tracks` block, so that block was recorded absent — which is why the phonation tracks are
missing from all 62,547 recordings of the completed run, and with them the F0 and formant tracks the
whole voice kind rests on.

The key's premise was that no single search range serves both a low adult male fundamental and an
infant voice, so the caller must state which population is being measured. The first half is true. The
second does not follow: no fixed corpus-wide range is needed, because the range is derivable per
recording, and a derived one serves the adult male and the infant in the same run.

What `extract_pitch_values` did at the time this was written was **not** that derivation, and the
citation it carried supported neither. It ran a wide 50–600 Hz autocorrelation search, trimmed the
outliers at |z| ≤ 2, and then picked one of two hardcoded pairs by where the trimmed mean fell —
(60, 250) below 170 Hz, (100, 500) above. The DOI it cited, `doi:10.3758/BRM.41.2.318`, is Vogel et
al. (2009), who recommends fixed sex-specific settings and rejects per-recording derivation as
impractical at scale; neither 60 Hz nor the narrowing appears in it. Both the bin and the trim are
gone as of 2026-09-14: the function now narrows off robust percentiles of the wide contour, the
two-pass structure and the ceiling's quartile term following Hirst 2011 and the rest being
senselab's own. See [`praat-instrument-audit.md`](../20260817-triage-workflow-dag/praat-instrument-audit.md)
step 2 for the rule and [`config-derivations.md`](../20260817-triage-workflow-dag/config-derivations.md)
under `praat_features` for each coefficient.

`voice.f0_range_hz` is **deleted**. `voice.f0_search_range_hz: [50.0, 600.0]` replaces it — the wide
search the narrowing starts from, a property of the method and not a population guess.
`derive_f0_range(audio, search_floor_hz=, search_ceiling_hz=)` in `senselab.audio.tasks.phonation.api`
wraps `extract_pitch_values` and returns the narrowed `(f0_min_hz, f0_max_hz)`. PREPROCESS's
`_phonation_tracks` and VOICE's `_f0_range` both call it, both off `plain`, so the two still cannot
hold ranges that drift.

`voice.f0_range_by_population` stays and still wins when the hint declares a population.
`voice.f0_range_ratio_max` bounds the resolved range, whichever way it was resolved.

Two consequences worth saying plainly:

- **PREPROCESS behaves differently and `config_hash` changes.** The phonation-tracks pass now runs
  under the packaged config instead of being recorded absent. The `praat_features` section added by
  this work changes `config_hash` again.
- **`extract_pitch_values` no longer guesses on silence.** With no voiced frame at all, `np.mean` over
  an empty array gave NaN, `NaN < 170` is `False`, and the function fell through to the "female and
  child" branch and returned (100, 500) — a guessed range that read as a derived one. It now returns
  NaN for both, and `derive_f0_range` raises on that, so an underivable range is an absence.

## `phonation.hnr_floor_interval_db` and `phonation.rms_floor_interval` stay null

Unchanged from `d45de3ba`. Checked for the same resolution and it does not apply. These two are the
near-edge intervals VOICE reads a span's onset gate values against (N22). They were measured in
normalised-autocorrelation units — (0.44, 0.933) and (0.0007, 0.0161) — and the implementation reads
Praat harmonicity in dB and RMS, so the numbers do not transfer.

Praat self-calibrates neither. Its harmonicity analysis has a `silence_threshold` relative to the
global peak, already a separate config key (`phonation.silence_threshold: 0.1`), and it exposes no dB
floor and no RMS interval at all. There is nothing to derive per recording the way the F0 range is
derived, so resolving these two requires a measurement in the implementation's own units that nobody
has taken. While null the near-edge row is inert and the verdict records
`gate_interval: "unmeasured"`.


## The worker timeout was flat, and four array tasks lost their whole batch to it

The first corpus pass extended 58,308 of 60,202 recordings. Every one of the 1,883 absences was the
same `TimeoutExpired` on the ppgs subprocess, and Praat succeeded on all of them, so the model was
never the problem. Each array task carried 471 rows and therefore one batch, and 1,883 / 471 is
exactly four: four whole batches exceeded the timeout and every recording in them was recorded
absent.

The timeout was `timeout=600` at the `subprocess.run` call, a constant that did not move with the
batch. A batch's cost is dominated by its total audio, and the corpus is skewed — median 7.22 s,
p95 60.02 s, longest 307.46 s — so a batch that happens to collect the long recordings costs several
times one that does not, against the same fixed allowance. The absent set's duration profile shows
it: p95 60.02 s against the corpus median of 7.22 s.

It is now `WORKER_STARTUP_S + WORKER_SECONDS_PER_AUDIO_SECOND * audio_seconds`. The startup term
covers what the worker pays before it reads any audio — interpreter, torch, checkpoint — and the
per-second term covers the work itself, at a multiple far above the 185x realtime measured
marginal, because the point is to catch a hung worker rather than to bound a slow one.

The failures are recoverable without recomputing anything: the driver skips a recording whose store
already holds both measurements, so re-running the same manifest touches only the 1,883.


## The manifest is not the corpus: 2,376 stores hold neither measurement

**Inside the manifest, coverage is total**, and the section above is the whole account of how:
58,308 extended on the first pass, 1,883 `TimeoutExpired` absences, 11 stores already complete and
skipped — 58,308 + 1,883 + 11 = 60,202 — and a 32-slice retry that recovered all 1,883. **Coverage
of the corpus is not**, and the two read as one unless the difference is written down.

Measured on ORCD, 2026-09-14:

| set | rows |
| --- | --- |
| `ppg_20260911/manifest.jsonl` — what this pass ran over | 60,202 |
| `clipfix_20260913/manifest_all.jsonl` — the full store set | 62,578 |
| in the full set, absent from the PPG manifest | **2,376** (3.8%) |

Every path in the PPG manifest is in the full one, so the difference is one-directional: the
manifest is a subset, not an overlapping set. And the 2,376 do not hold stale readings taken under
the retired sex-typed F0 range — **they hold neither measurement and never did**:

- **0 of 25** stores sampled from the gap carry `ppg_posteriorgram` or `praat_features`; **25 of 25**
  sampled from the manifest carry both.
- **97.5% of the gap stores have an `enhanced.flac` on disk** — 10 of 400 sampled were missing one —
  so this is not a missing-stream exclusion. Both blocks read that stream and nothing else.

Against the scored count the arithmetic differs and both statements are true of the same set:
62,578 − 60,202 = **2,376**, and 62,547 − 60,202 = **2,345**. Which denominator applies is
[`branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)'s three-numbers
question, not a second gap.

### The exclusion is biased toward the low-energy task families

No family is excluded outright — all 148 have some of both — but the rate varies more than
fourfold. The ten deepest:

| family | missing | present | rate |
| --- | --- | --- | --- |
| `respiration-and-cough-v2-breath` | 197 | 504 | 28% |
| `respiration-and-cough-v2-threebreathsnose` | 144 | 555 | 21% |
| `respiration-and-cough-fivebreaths-1` | 180 | 716 | 20% |
| `respiration-and-cough-fivebreaths-2` | 153 | 741 | 17% |
| `respiration-and-cough-fivebreaths-3` | 150 | 745 | 17% |
| `maximum-phonation-time-v2-1` | 106 | 594 | 15% |
| `respiration-and-cough-fivebreaths-4` | 121 | 774 | 14% |
| `respiration-and-cough-v2-threebreathsmouth` | 98 | 601 | 14% |
| `maximum-phonation-time-2` | 84 | 818 | 9% |
| `glides-high-to-low` | 96 | 1458 | 6% |

These totals are over the 62,578-store set, so they sit at or a little above the per-family counts in
[`branch-airway.md`](../20260817-triage-workflow-dag/branch-airway.md):17-25, which are over the
scored 62,547 — `v2-threebreathsnose` and `v2-threebreathsmouth` match at 699, `v2-breath` reads 701
against 699. That direction is the expected one and is not a third divergence.

**Where the loss falls is the part that matters.** It is deepest on breath and maximum-phonation-time
and thinnest on glides — that is, on the whole of AIRWAY's task content (every
`respiration-and-cough-*` family is AIRWAY's, [`branch-airway.md`](../20260817-triage-workflow-dag/branch-airway.md):17-25)
and on one of VOICE's core measures ([`branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md):34,
`:38`), and so on the quietest, lowest-energy material in the corpus.

**The cause was established on 2026-09-14, by probing the excluded stores on ORCD.** The manifest
is the set of recordings that produced **consensus ASR output**, and the 2,376 are the recordings
that produced none.

Measured on a random sample of 80 of the excluded stores:

- **78 hold a `consensus_transcript` entity whose `n_words` is 0.** Two hold no consensus entity at
  all. None holds a non-empty transcript.
- **78 show ASR hypothesis entities**, so recognition ran and returned nothing. These are wordless
  recordings, not unprocessed ones.
- The excluded stores are otherwise healthy: the clip-amplitude driver read and wrote all 2,376 of
  them with zero errors (2,087 already held a live `clip_amplitude`, 289 were newly written), so
  this is not a broken-store exclusion.

**The manifest's `lexical` field is not the consensus word count, and the difference is the
bracketing rule.** Two counts exist over the same transcript: `consensus_words`
(`../../src/senselab/audio/workflows/triage/nodes/common.py:300-308`) is every live `word` entity —
the transcript's `n_words` — while `lexical_words` (`:311-321`) keeps only those whose `bracketed`
attribute is False. A token is bracketed when the recognizer already bracketed it or when its
vocabulary key is in `words.onomatopoeic_tokens`, which `bracketed_form`
(`../../src/senselab/audio/workflows/triage/consensus.py:183-200`) rewrites to `[COUGH]` — so a
recognizer that writes the plain word *cough* and one that writes `[COUGH]` agree.

So the corpus is three populations, not two:

| population | rows | what it is |
| --- | ---: | --- |
| `n_words == 0` | 2,376 | the recognizer returned nothing; **excluded from this manifest** |
| `n_words > 0`, `lexical == 0` | 14,836 | every token bracketed — `[breath]`, `[cough]`, `[UH]` |
| `lexical > 0` | 45,366 | at least one plain word |

Measured: of 30 sampled `lexical == 0` stores, **33 of 33** word entities carried `bracketed: true`.
The manifest is the second and third populations together, which is exactly *has consensus ASR
output* — bracketed-only recordings included, since a bracketed marker is output even though no part
of it is a word.

**So the rule is right for the posteriorgram and wrong for the Praat scalars.** A phoneme
posteriorgram over a recording with no recognised speech classifies nothing, so scoping the PPG to
stores with consensus ASR output is the owner's stated rule and is what happened — the PPG needs no
catch-up pass and none is owed. The Praat scalars inherited that population only because one driver
and one manifest serve both blocks. F0, HNR, CPPS, jitter, shimmer and maximum phonation time do not
need a word; on a sustained vowel or a breath task **zero words is the expected result**, and those
are the families the exclusion falls hardest on — 28% of `respiration-and-cough-v2-breath` against 6%
of `glides-high-to-low`.

**What is owed is therefore a Praat-only first derivation over the 2,376, not a provenance
reconstruction.** It is CPU-only: the PPG must not run on these stores, by the same rule that
excluded them. The manifest already exists on ORCD as
`clipfix_20260913/manifest_amplitudes.jsonl`, whose row set is **exactly** this gap — verified as a
set equality, not a matching count. Driving it through `extend_ppg_praat.py` would run the PPG
block, because `pending()` reports a missing posteriorgram as pending regardless of transcript; a
Praat-only path for these stores is what the pass needs and does not yet have.

**One thing this does not settle.** The `ppg_20260911` manifest was built by an operator artifact,
not by any script in this tree — the only in-tree builder,
`../../scripts/analyze_routing_evidence.py:85-136`, emits a different key set. The *rule* is now
known from its output; the *code that applied it* is still not in the repository, so a future pass
that must reproduce this selection has to re-derive it from consensus output rather than re-run a
script.

### The consequence, which is one specific misreading

Any later per-family analysis of `praat_features` or `ppg_posteriorgram` carries a 3.8% hole that is
**more than four times deeper on breath tasks (28%) than on glides (6%)**. Uncorrected, that reads as
a finding about breath recordings when it is a finding about the manifest. It is the misreading this
section exists to prevent, and
[`branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)'s rule is what
prevents it: every aggregate reports the count it was computed over and the count that was
unavailable. For these two measurements that count is **60,202**, never the corpus.

### Two jobs, and they must not be folded into one

- **Re-deriving the 60,202 is CPU-only.** `--force` overrides `praat_pending` alone
  (`../../scripts/extend_ppg_praat.py:218`, `:269`), and every store the manifest names already holds
  a posteriorgram, so `ppg_pending` is false, the driver `continue`s at `:225-226` before
  `ppg_input`, and no batch reaches ppgs. Size it against a Praat-only pass.
- **A first derivation for the 2,376 is GPU-bearing.** None of them holds a posteriorgram, so every
  one of them takes the PPG path the 60,202 skip. It needs a manifest that does not exist yet and it
  has a different resource profile; folding it into the forced re-derivation puts the ppgs venv and
  a GPU allocation back into a pass sized as CPU-only.

**So "a forced pass does no PPG work" is a property of this manifest, not of the flag** — which
[`praat-instrument-audit.md`](../20260817-triage-workflow-dag/praat-instrument-audit.md) step 2
already says, and the 2,376 are exactly the stores that would make it false.


## Reading one range back: `plot_range_with_ppg`

The corpus pass writes a posteriorgram per recording and nothing that looks at one. `plot_range_with_ppg`
in `senselab.audio.tasks.plotting.plotting` is that reader: a start and an end in seconds, drawn as
waveform, spectrogram and PPG phoneme boundaries on one shared axis.

**The range is a pair of numbers and nothing else.** It comes from a consensus word's extent, a span
entity, a detector's firing region or a hand-typed pair, and the function cannot tell which, because
knowing would make it a different function per source. The three sources that already exist would
have been three signatures, and the fourth — the number someone types while looking at a waveform —
would have had none.

**It composes `plot_aligned_panels` rather than drawing panels itself.** That function already owns
the shared axis, the range check, the page-window spectrogram (a ten-second page does not allocate a
full-recording transform), and the token lane that measures every label against its own bar at draw
time and shrinks or drops it rather than overprinting. Three additions were needed and were made
there, not copied:

- `_resolved_time_range(duration, start, end, name=)` — the bound check `plot_aligned_panels` already
  did inline, lifted so both callers raise the same message. `name` is what the message calls the
  pair, so each caller's own parameters are named: `time_limits` for one, `(start_s, end_s)` for the
  other. Reversed, zero-length and out-of-range are one condition, `0 <= start < end <= duration`,
  and the message carries both received values.
- `"boundaries": True` on a tokens panel — a vertical rule at every visible token edge. A lane of
  abutting bars shows its boundaries only where two colours meet, and the staggered rows that keep
  short labels legible put neighbouring bars at different heights, so the edge that matters is the
  rule and not the bar.
- `load_ppg_posteriorgram(path)` in `features_extraction.ppg` — the npz reader that
  `routing_analysis.features._ppg_segments` had inline. One reader of the sidecar format now, so a
  change to what `write_ppg_posteriorgram` writes has one place to land.

**A segment straddling the range edge is clipped and kept.** `plot_aligned_panels` already clipped
token extents to the page; what matters here is that the label survives the clip, because the
phoneme crossing the boundary is usually the one the range was opened to look at. Dropping it would
leave a gap at exactly the position under examination.

**An absent posteriorgram is a third panel, not two panels.** The recording with no derivative, the
sidecar that will not open and the array with no frame all resolve to a text panel reading
`PPG phonemes: absent — <reason>`, naming the path where there is one. A figure with two panels
looks like a figure of a recording whose phonemes were not asked for; an empty third lane looks like
a recording with no phonemes in it. Neither is the claim. This is why an unreadable path is an
absence rather than a raise: on this corpus the overwhelmingly likely cause is a run that predates
the block, which is a fact about the recording and not an error by the caller.

**The clock comes from the posteriorgram's own sidecar when there is one.** A tensor handed in
directly is timed against the `audio` argument, as `extract_ppg_segments` has always done; a sidecar
is timed against its own `duration_s` and `sampling_rate`, through the same zero-waveform clock
`_ppg_segments` uses. The two agree for a posteriorgram measured on the recording being drawn, and
where they would not — a caller drawing a trimmed stream — the sidecar's own clock is the one that
placed the frames.

**The bars are full-height washed bands, because a bar of any other height is read as a pair of
probabilities.** The lane and the posterior curves share one panel and one right-hand `0`–`1` scale.
Bars used to fill `TOKEN_BAR_HEIGHT_FRACTION = 0.7` of the lane, centred: rendered, their edges sat
at 0.15 and 0.85 of the posterior scale, and a reader taking a value off that scale reads two
horizontal lines at 0.15 and 0.85 that mean nothing at all — they are where the bar was drawn. On
the crowded synthetic window the fault compounded: the label stagger puts neighbouring bars on
different rows, so the same lane showed *four* such edges, at four different heights, none of them a
probability. The band now runs the full height of the scale, so its only horizontal edges are 0 and
1, which are the two values the scale actually has.

The wash is `TOKEN_BAND_ALPHA = 0.30`. Rendered against a two-phoneme crossover in the light half of
`tab20` and the same window recoloured into the dark half, 0.14 and 0.18 left the band too faint to
say which phoneme won without reading the name, and 0.40 lost the worst case — a curve crossing its
*own* band, where the two are the same hue and only the curve's white casing separates them. 0.30
holds that case in both halves of the cycle and still registers as a colour.

**The names moved to the head strip of the lane, and onto the curve axis.** They need a home that no
curve destroys, and there is none: a posteriorgram's losers hug 0 and its winner hugs 1, so every
height is occupied somewhere. Two things resolve it. The strip is at the head
(`TOKEN_BAND_LABEL_STRIP = 0.26` of the lane) rather than the foot, because the foot is where `N-1`
competitors tangle and the head carries at most one curve — the winner's plateau, which is the band's
own phoneme and carries the least information in the window. Rendering the foot variant showed
exactly that: the names punched white holes through the competitor tangle, which is the part of a
posteriorgram worth looking at. And the names are added to the twin curve axis rather than to the
lane, because matplotlib draws a twin after its host, so a name left on the lane is drawn *under*
every curve and crossed out by any that passes through it; on the twin it is drawn last. A white
casing (`TOKEN_BAND_LABEL_HALO_PT`) carries it over the winner's plateau where the two do meet.

Only the labels stagger now. `_StaggeredTokenLane` still decides a row count against the renderer,
but in banded mode it applies it to the names inside the head strip and leaves every band at full
height, so the row count buys horizontal room for the names without putting bar edges at four
heights. The row ceiling is measured against the strip rather than the whole lane.

**The treatment follows the curves, not a flag.** A `tokens` panel draws bands exactly when it
carries a `curves` block, because that is exactly when there is a second scale for a bar edge to be
misread against. A lane of words with no probability axis keeps its bars at row height, where the
bar *is* the reading.
