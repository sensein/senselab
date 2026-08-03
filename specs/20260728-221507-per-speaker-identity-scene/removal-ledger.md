# Removal ledger

What D-17 – D-22 make unnecessary, what is already unnecessary and nobody noticed, and the order
the removals have to happen in. A worklist, not a design record — the reasoning lives in
`layered-architecture.md`.

Baseline: **29,063 lines** across `audio_analysis/`.

## Finding: there is almost no dead code, and a lot of unwired code

An AST sweep for definitions referenced nowhere returns **4 results** out of the whole package. The
29k lines are reachable. So "what can be removed" is not a dead-code question.

Reachability from *non-test* code is a different question, and it has a much worse answer. Three
modules are built, documented, and tested, and **the pipeline never calls them**.

### `sources.py` — 526 lines, zero non-test callers

The corroborated **3 / 6 / 10 dB tier ladder** above the band floor, plus the four fabrication
guards. Fifteen exports; `grep` finds two hits in `src/senselab`, and both are false positives — a
same-named local variable in `quality_control/metrics.py:248` and a `noqa` comment in `io.py:306`.

**The pipeline detects background sources a different way.** `stages.stage_background_sources` (wired,
runs by default) calls `noise_floor.detect_stationary_sources` — ECMA-74 prominence, ≥9 dB. So there
are two implementations of one capability, one wired, and **CLAUDE.md documents the unwired one** as
the mechanism.

`io.write_background_sources(findings: Any, ...)` is annotated `Any` because the type it wants is
`SourceFinding`, from the module nothing calls.

*Decision needed.* D-21's derivative list has no tier ladder — background sources arrive as
`(background_source, project_labels/<map>, (scene_labels, T, p))`. So the ladder is not part of the
new design, and its four fabrication guards are the reasoning at risk of being lost with it.
Wire it, or delete it and move the guards' reasoning into the projection?

### `foreground.py` — 191 lines src + 201 test, test-only

`suppress_foreground` is not in `__all__`, is called by nothing, and **the documented CLI flag does
not exist**: `--foreground-suppression` appears in CLAUDE.md and zero times in
`scripts/analyze_audio.py`.

*Settled by D-23 — wire it, as a **pathway**, not a perturbation.* Foreground suppression is an
alternate route to enhancing a signal component (the background), and a perturbation is by definition
a transform that does not remove the primary information targets. Suppression removes the foreground
target on purpose, even ideally, so it is what makes a *different* component primary — the `background`
pathway's constitutive transform. Enhancement stays a perturbation: ideally it removes only non-target
content, so the primary target survives. `project_onto` / `suppression_depth_db` /
`leakage_margin_db` become what the pathway reports about itself in the route register.

### `level.py` — 389 lines, one live export

`apply_gain_db` is called by `scripts/probe_classifier_levels.py`. Everything else — `integrated_lufs`,
`true_peak_dbtp`, `loudness_range_lu`, `measure_variant`, `normalization_gain_db`,
`peak_limited_gain_db`, `AudioVariant`, `GainCapExceededError` — has no caller. `contracts.py:882`
already records the symptom as a deviation: *"write_level_json writes `<run>/level.json`, which no
stage declares and no run produces."*

*No decision needed — wire it.* D-20 asks for `(loudness, pyloudnorm, p)` and `(clipping, pcm, p)`
and this module already computes both. The signals the design wants exist and are not plugged in.

### Orphaned definitions

| site | note |
|---|---|
| `embeddings.window_embedding_at` | dies with the nine embedding signals |
| `embeddings.silhouette_voice_score` | ditto — silhouette becomes internal to clustering |
| `background_mask.MaskedRegionIntrospection` | never constructed |
| `foreground.suppress_foreground` | above |

## Removals, in dependency order

The removals are **not independent cleanups** — each is a consequence of a restructure step, and
doing it early breaks the pipeline. Ordered by what unblocks what.

### Step 1 — row types (no removals, unblocks most of them)

The six L1 output kinds. Nothing can be removed first, because every reduction that dies needs a
native-shaped artifact to be replaced *by*.

### Step 2 — StageIO replaces the guard

| removed | lines | condition |
|---|---|---|
| `contracts.py` static AST guard + `_PathResolver` | ~900 of 1,883 | StageIO exists and L1/L2/FINAL are converted |
| `KNOWN_DEVIATIONS` (47 entries) | ~350 | each entry closed or migrated |
| `stage_contract_test.py` static half | ~600 of 1,143 | ditto |

The declarations **stay** — `Artifact` patterns and their `key` tuples are what StageIO is built
from. What goes is inspection-after-the-fact of an undecidable property, which was defeated in ten
ways and whose real-run test skips-or-fails.

### Step 3 — signals removed at L1

| removed | condition |
|---|---|
| `pyannote/segmentation-3.0` — 12 src + 8 test files | J1/J4/C2 rebuilt on spans (D-19) |
| 9 embedding-derived signals (`speaker_distance` ×4, `speaker_change` ×2, `embedding_silhouette` ×3) | the embedder-plus-clusterer diarizer emits `speaker_spans` |
| `ppgs` signal + the PER sub-signal | none — the *signal* goes; `features_extraction/ppg.py` stays a senselab task |
| the `scene_quality` bundle (`units: "mixed"`) | the 8 per-target scene signals exist |

`ppgs` is the only one removable **now**, and it is the only one with no replacement to build.

### Step 4 — the harvest/link/vote layer dissolves into derivatives

| removed | replaced by |
|---|---|
| `L2/round0/votes/<axis>.parquet` | `project` derivatives, keyed |
| `speech_presence_link` (588 lines) | a set of `project` derivatives, policy in the operator tag |
| `PassHarvest.speech_presence_evidence`, `votes_for_harvest` | the merged input pool |
| `signal_stability(..., axis=...)`'s axis parameter | stability is per-derivative (D-21) |
| `votes.py` (470), parts of `support.py` / `aggregate.py` / `estimates.py` | `fuse_axis` over the pool |

### Step 5 — the second speaker lineage

| removed | condition |
|---|---|
| `adaptive/plot._fused_axis` — self-documented as *"scaffolding for a defect, should be deleted rather than maintained"* | the store stops persisting estimates (D-22) |
| the store's materialised estimate copy + its parity oracle | ditto |

## Live defects found while surveying, not removals

- `aleatoric_floor` reads a `quality_snr`-family name present in neither ingest path → takes `None` →
  floors at `0.0` on every bucket of every run. D-21 gives it a source:
  `(scene_score, anchor/<profile>, (snr, T, p))`.
- `background_sources.parquet` is written with `[]` when detection finds nothing and the finding type
  comes from an uncalled module — so an empty artifact and an unwired subsystem look identical from
  the run directory. The same absent-vs-empty confusion this design keeps hitting.
