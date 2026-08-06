# `senselab.audio.workflows.audio_analysis`

Uncertainty axes for analyze_audio runs. Reads the per-task pipeline outputs (diarization, ASR,
scene classification, alignment, speaker embeddings) and emits one `[0, 1]` uncertainty per bucket
on each of four axes:

- **speech_presence** — was there a speaker?
- **speaker** — who is speaking here?
- **asr** — what was said?
- **background_mask** — is this region free of *target* activity?

The set is declared once, in `axes.AXES`, with each axis's properties beside its name. Any list of
four axes written out by hand is one that can go stale: the mask was fused and written while being
absent from region proposal, convergence and the disagreements index, because three consumers
enumerated three axes in literal tuples.

## One grid

**Every axis is harvested on `axes.DEFAULT_TIME_GRID`** — a 0.1 s window at a 0.1 s hop — so row *i*
of one axis is row *i* of another and a cross-axis join needs no reconciliation.

This was measured, not assumed. With a per-axis grid the four axes carried 242 / 242 / 19 / 8 rows
on 0.1/0.02, 0.1/0.02, 0.25/0.25 and 1.0/0.5, and shared **zero** bucket keys: `fuse.project_axis_onto`
found nothing to project, every cross-axis coupling ran and did nothing, and each round came out
byte-identical to the last. Unit tests missed it because their fixtures put every axis on one
synthetic grid — the one thing real data never did.

Window equals hop deliberately. A 0.1 s window at a 0.02 s hop makes adjacent rows share 80% of
their audio, so 1070 rows are not 1070 independent measurements and nothing in the output said so.
Fine *resolution* is what a question justifies; five near-duplicate rows per window is a different
thing. `BucketGrid()` defaults to the declared constant, so there is one answer rather than a
constant and a default that can disagree.

## One fold

`fuse.fuse_axis` is the run's single fold. It reads each signal's own reading through
`fuse.per_signal_uncertainty`, weights it by two *measured* factors — perturbation stability
(`reliability.signal_stability`: does the signal agree with itself between the raw and enhanced
passes?) and physical support (`support.signal_support`: does the audio carry what it claimed?) —
and collapses the weighted readings via the run config's `uncertainty.aggregator` (default `min`,
"most-doubtful wins"). A signal absent from the weights carries full weight: a factor never
measured must not act as a discount.

Each row carries four quantities that are deliberately **not** collapsed into one: `uncertainty`
(normalised binary entropy), `epistemic_uncertainty` (its reducible part), `confidence` (a weighted
mean, so a probability), and `variability` (dispersion across signals). `triage_score` is the policy
fold — what the adaptive loop ranks by — and is the only one an aggregator choice touches.

Nothing here is per pass. An axis aggregates across signals *and* across passes, so a pass is an
input dimension to the fold and appears on the output only as each row's `contributing_passes`.

### The asr axis has one voter

`asr.harvest_asr_votes` emits `consensus_words` and nothing else: the recognizers' words are fused
once per pass (`fuse_word_streams`, grouped by sequence alignment and graded phonemically by
`asr.phoneme_similarity`), and each bucket takes the coverage-weighted mean of
`1 - existence_confidence` over the words reaching it. A bucket no word reaches carries **no vote**
rather than `0.0` — nothing was said there, which is not the same as nothing being in doubt.

Four things used to ride beside it and all four are gone: the per-bucket text (a reconstruction of
what `final/transcript.json` holds at word resolution, and the reason this axis needed a 1.0 s
window — a fully-contained text read returns nothing from a bucket narrower than a word), the
cross-ASR pairwise phoneme distance (already recorded-but-unscored, because its source closure is a
subset of the consensus fold's), and the per-bucket `avg_logprob` / `token_entropy` /
`alignment_ctc_score` reads.

The `speaker` axis measures **attribution**: how sure we are *who* is speaking, composed by
`attribution.py` from three voters — per-speaker presence doubt (`max` over the speakers present of
the entropy of the model share), ASR word-location doubt (`1 - temporal_confidence`,
coverage-weighted over the words reaching the bucket), and target-activity doubt (the mask region's
uncertainty, only where its `state` is not `target_active`). A bucket the mask confidently calls
`target_free` carries no vote at all: there is nobody to attribute.

It asked "was it the same speaker as before?" until 2026-08-05, scored per (diar × embedder) pair
against embedding cosine — which on a 0.1 s grid asks ten times a second against 0.5 s windows, and
read 0.666 on a clean two-speaker conversation whose count posterior was 2 at 0.978 and whose
per-speaker presence doubt averaged 0.168. The cosines, the calibrated readings, the change points
and the overlap distribution all survive as **L1 measurements**; they simply stopped being scored.

**Temporal agreement is excluded from this axis on purpose.** Two attempts proved a single number
cannot carry both accuracy and localisation and stay readable: bucketed pairwise WER reported 0.4266
on a pair of word-identical transcripts (timing jitter reading as textual disagreement), and a joint
of accuracy × localisation gave 0.788, which could mean either half. Localisation now lives on the
word, split per edge (`onset_confidence` / `offset_confidence`), where a figure can show *which*
boundary is in doubt.

## L1 measures, L2 decides

L1 reports what a tool produced, in that tool's units, at its own resolution: no thresholds, no
rescaling against an anchor, no reduction across a dimension the tool reported separately, no
selection among estimators. Every interpretation lives at L2, where it is named and can be changed
without re-running a model. `speech_presence.harvest_speech_presence_evidence` emits measurements
and `speech_presence_link.link_speech_presence` turns them into votes under a recorded policy;
`quality.harvest_quality_measurements` emits dB / hertz / proportion and `degradation` anchors them.

## Output

- `L1/<pass>/signals/<signal>.parquet` — one row per (signal, bucket), in the tool's own units.
- `L1/stability/<signal>.parquet` — cross-pass `|Δ|` per bucket; the run-level mean is on every
  fused row as `weight_basis[signal]["stability"]`.
- `L2/round<N>/uncertainty/<axis>.parquet` — the fused axes, all on one grid.
- `L2/round0/votes/<axis>.parquet` — the linked evidence at vote level; what the adaptive store
  ingests.
- `L2/disagreements.json` — top-N over the fused axes by `triage_score`, axis-priority tiebreak.
- `final/` — the deliverables: `transcript.json`, `diarization.json`, `speakers.json`,
  `estimates/<axis>.parquet`, `decisions.json`, `timeline.png`, the annotated LS bundle.

## Public API

```python
from senselab.audio.workflows.audio_analysis import (
    BucketGrid,
    compute_uncertainty_axes,
    build_disagreements_index,
    build_aligned_timeline_plot,
    attach_uncertainty_tracks_to_ls,
)
```

`compute_uncertainty_axes(passes, grid, params, *, audio, speaker_embedding_models, aggregator,
speech_presence_labels)` is the entry point — a thin wrapper over `harvest_pass` (expensive,
model-touching) + `votes.link_pass` (pure) + `fuse.fuse_axis` (pure). There is deliberately **no**
per-axis grid parameter; `grid_test` asserts the absence on the signature, because an override
coming back would restore the four-grid defect with every value assertion still passing.

## Configuration

One versioned file, `data/run_config/default.yaml`, with each value's derivation written beside it:
model ids, the grid, the aggregator, the task type, the triage and enhancement gates, which stages
run, and the adaptive loop's policy as its `adaptive:` section. `run_config.load_run_config` merges
an optional override over it and hashes the *merged* mapping, so `{name, version, config_hash,
sources}` identifies the run in every artifact's provenance.

`scripts/analyze_audio.py` therefore takes an audio file, `--out`, and `--config` — nothing else.
The seventy flags that preceded it differed in ways a reader had no basis to choose between, and the
shipped defaults of four of them are what put the axes on four grids.

Scene-quality calibration (`calibration.py`) supplies the dB→`[0, 1]` anchors `degradation` reads.
Its `temperature` and `token_entropy_reference_nats` fields currently reach **no fold** — their only
consumers were the uncalled per-axis aggregators — and are kept validated rather than dropped so
fitted values survive until `fuse_axis` takes them.

## Importable pipeline: stages, cache, adaptive loop (T051 / T040)

The per-task pipeline used to live only inside `scripts/analyze_audio.py`. It is
now library code, so the adaptive loop (and any other caller) can run it
in-process instead of shelling out to the CLI:

```python
from senselab.audio.workflows.audio_analysis import PassPlan, StageContext, run_pass

summary = run_pass(audio, StageContext(perturbation="raw", audio_signature=sig), PassPlan(
    diarization_models=("pyannote/speaker-diarization-3.1",),
    asr_models=("openai/whisper-large-v3-turbo",),
))
```

- **`stages.py`** — six `stage_*` functions plus `run_pass`. Each takes
  `(audio, ctx, *, knobs)` and *returns* the fragment it contributes to the pass
  summary; none mutates a shared dict. `stage_alignment` takes `asr_by_model`
  explicitly, so a caller can align a cached ASR block it did not produce.
- **`stage_context.py`** (light — no torch import) — `StageContext` carries the run
  environment and derives cache keys and provenance; `PassPlan` says what to run,
  with absence meaning skip (empty tuples, `None` model ids) rather than a
  CLI-shaped skip set. `out_dir=None` gives a headless mode that writes no
  sidecars.
- **`senselab.utils.tasks.cached_inference`** — the content-addressable cache:
  `audio_signature`, `cache_key`, `align_cache_key`, `cache_lookup`/`cache_store`,
  the `run_task_cached` / `run_alignment_cached` runners, and
  `sync_cache_with_schema_version`.

Cache invalidation is coarse and deliberate. `STAGE_VERSIONS` in
`stage_context.py` holds a per-stage integer surfaced in keys and provenance as
`"asr@1"`; **bump a stage's number when the stored shape of its outcome changes.**
This replaced a sha256 of the CLI script's source, which rotated on every comment
edit or reformat and invalidated every cached model result for nothing.
`CACHE_SCHEMA_VERSION` remains the global lever — bumping it makes
`sync_cache_with_schema_version` wipe stale entries automatically on every host.

The adaptive loop accepts either ingest path: `run_adaptive_loop(run_dir)` reads the linked
votes from `L2/round0/votes/<axis>.parquet`, while
`run_adaptive_loop(run_dir, harvests=..., summary=...)` consumes in-memory `PassHarvest`
objects (what `analyze_audio.py` now does via `compute_uncertainty_axes(harvests_out=...)`).
Both see the same evidence, and both run `replay_check`: every bucket is rebuilt from the
persisted votes plus the recorded decisions and compared against the store's own aggregation.
That replaces a parity check against `within_pass_uncertainty` on the L1 parquet — an oracle
that was a per-pass axis (a quantity that cannot exist) produced by a second implementation of
the fold, and that the in-process path could not run at all.

The adaptive surface is config, not flags: `rounds.max_rounds`, `stages.adaptive_outputs`, and the
whole `adaptive:` section (budgets, region caps, reserve pools, per-rule enables). Precedence is
packaged default < `--config` file < in-memory overrides, and `policy_hash` is recomputed after
merging so two runs differing only by one entry cannot claim the same provenance. A file with
`thresholds:` / `fusion:` / `rules:` at the *top* level is refused rather than merged into keys
nothing reads. `rounds.max_rounds: 1` is baseline-only: no interventions and no rounds ≥ 2, though
`final/` is still emitted from the round-1 belief.
