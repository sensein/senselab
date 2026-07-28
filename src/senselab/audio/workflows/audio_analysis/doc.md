# `senselab.audio.workflows.audio_analysis`

Three-axis uncertainty for analyze_audio runs. Reads the per-task pipeline outputs
(diarization, ASR, scene classification, alignment, PPG, speaker embeddings) and emits a
single `[0, 1]` uncertainty scalar per bucket on each of three axes:

- **presence_uncertainty** — was there a speaker?
- **identity_uncertainty** — was it the same speaker?
- **utterance_uncertainty** — what was said?

Every model whose output naturally encodes an axis votes (max-inclusive). The vote
collapse is per-axis: Shannon entropy for presence (binary votes); cross-model label
disagreement + cosine across-time for identity; pairwise mean WER + Whisper avg_logprob
+ PPG-vs-ASR phoneme-error-rate for utterance. Sub-signals within each axis fold via the
shared `--uncertainty-aggregator` flag (default `min` over confidences ≡ `max` over
uncertainties).

Output:

- 9 parquets (3 axes × 2 passes + 3 raw_vs_enhanced deltas) — see
  `contracts/uncertainty-row.parquet.md`.
- `disagreements.json` — top-N ranked across all parquets — see
  `contracts/disagreements.json.md`.
- `timeline.png` — 5-row figure (presence / identity / utterance overlaid raw-vs-enhanced
  + delta strip + reference) — see `contracts/ls-bundle.md` for the matching LS tracks.

See `specs/20260508-173136-compare-uncertainty/spec.md` for the full design and
`specs/20260508-173136-compare-uncertainty/quickstart.md` for reviewer recipes.

## Public API

```python
from senselab.audio.workflows.audio_analysis import (
    BucketGrid,
    compute_uncertainty_axes,
    build_disagreements_index,
    build_aligned_timeline_plot,
    attach_uncertainty_tracks_to_ls,
    write_axis_parquet,
)
```

`compute_uncertainty_axes(passes, grid, params, *, audio, speaker_embedding_models,
aggregator)` is the workflow entry point. It is a pure function: callers (such as
`scripts/analyze_audio.py`) handle the surrounding I/O (cache lookup, parquet writing,
disagreements.json + plot, LS bundle extension).

## Scene-aware presence + calibration (spec 20260722-175022, US1–US5)

The presence axis carries additive scene columns beside `aggregated_uncertainty`
(unchanged): `presence_confidence`/`presence_uncertainty` (decisiveness +
within-bucket temporal instability from frame posteriors), four `quality_*`
degradation scores in `[0, 1]` (SNR / clipping / reverb / bandwidth) plus
`quality_uncertainty` (estimator spread), and the `src_*` source-category masses
with `src_dominant`. The utterance axis couples to the scene per FR-019: the
reported `aggregated_uncertainty` is the per-vote value times a
`scene_quality_coupling` multiplier (≥ 1; pre-coupling value preserved on
`raw_aggregated_uncertainty`), and carries `token_entropy` when the Whisper
token-confidence capture ran. The Label Studio bundle adds
`<pass>__presence__quality` / `<pass>__presence__sources` tracks and the
disagreements index exposes the presence sub-signals (FR-024).

Calibration (US5): dB→`[0, 1]` anchors and per-axis aggregator temperatures live
in a versioned `CalibrationProfile` JSON (`calibration.py`; documented defaults
when absent). Fit one from synthetic sweeps with
`scripts/calibrate_scene_quality.py` and pass it to the pipeline via
`scripts/analyze_audio.py --calibration-profile <profile.json>`. Temperatures
default to 1.0 — fitting them requires labeled correctness (see the adaptive
loop's ground-truth evaluation harness).

## Importable pipeline: stages, cache, adaptive loop (T051 / T040)

The per-task pipeline used to live only inside `scripts/analyze_audio.py`. It is
now library code, so the adaptive loop (and any other caller) can run it
in-process instead of shelling out to the CLI:

```python
from senselab.audio.workflows.audio_analysis import PassPlan, StageContext, run_pass

summary = run_pass(audio, StageContext(pass_label="raw_16k", audio_signature=sig), PassPlan(
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

The adaptive loop accepts either ingest path: `run_adaptive_loop(run_dir)` reads a
finished run's parquets, while `run_adaptive_loop(run_dir, harvests=..., summary=...)`
consumes in-memory `PassHarvest` objects (what `analyze_audio.py` now does via
`compute_uncertainty_axes(harvests_out=...)`). The in-process path reports
`parity_check.status == "skipped"` rather than a passing check, because parity
compares against stored parquet values that don't exist yet — a vacuous
"0 mismatches" would look like proof and be none.

Adaptive CLI surface: `--max-rounds`, `--policy`, `--budget-medium/-heavy`,
`--max-region-rounds`, `--region-top-n`, `--reserve-asr-models`,
`--enable-overlap-separation`, `--no-adaptive-outputs`. Precedence is packaged
default < `--policy` file < CLI flags, and `policy_hash` is recomputed after
merging so two runs differing only by a flag don't claim the same provenance.
`--max-rounds 1` is the golden-compat mode: verified to leave the uncertainty
parquets, Label Studio bundle and pre-existing `summary.json` keys byte-identical.
