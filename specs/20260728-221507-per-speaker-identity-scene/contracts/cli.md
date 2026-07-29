# Contract: CLI surface

**Script**: `scripts/analyze_audio.py` (plus two new standalone scripts)

All new flags default to **off or to the current behavior**, so a default invocation is
unchanged except where a measured defect is being fixed. Every threshold has a default
sourced from the policy profile, never a literal in argument parsing (constitution VIII).

## `scripts/analyze_audio.py` — new flags

### Background mask (US4)

| Flag | Default | Effect |
|---|---|---|
| `--background-mask / --no-background-mask` | `--background-mask` | Emit the mask and its per-region uncertainty |
| `--task-type NAME` | `None` | Target event type for the mask (`speech`, `breath`, `cough`, …). Absent ⟹ conservative fallback, recorded as such (FR-033b) |
| `--mask-guard-interval SECONDS` | from policy | Guard interval around target activity (FR-034) |
| `--mask-introspect` | off | Emit per-region introspection (FR-037) |

### Background characterization (US3)

| Flag | Default | Effect |
|---|---|---|
| `--foreground-suppression` | off | Produce the suppressed variant; opt-in per FR-030 |
| `--suppression-model ID` | policy default | Model used to extract foreground for subtraction |
| `--detection-margin-profile PATH\|NAME` | bundled default | Versioned margin profile (FR-023) |
| `--level-target-lufs FLOAT` | from policy (−23) | Normalization target (FR-019c) |
| `--gain-cap-db FLOAT` | from policy (10) | Hard ceiling; exceeding is an error, not a warning |
| `--scene-variant {unmodified,enhanced,suppressed}` | `unmodified` | Which variant feeds scene analysis (FR-016) |
| `--stationary-pass / --no-stationary-pass` | `--stationary-pass` | Parallel unsubtracted analysis (FR-021i) |

### Per-speaker identity (US1)

| Flag | Default | Effect |
|---|---|---|
| `--per-speaker-identity` | on | Emit count posterior + per-speaker tracks |
| `--influence-profile PATH\|NAME` | bundled default | Influence weights and gates (FR-011b/c) |
| `--max-influence-rounds N` | from policy | Bound on mutual-influence iteration |

**Exit codes**: `0` success; `2` argument error; existing non-zero codes unchanged.
A gain request exceeding `--gain-cap-db` exits `2` rather than silently clamping — a
silently clamped gain would make the recorded provenance wrong.

## `scripts/probe_classifier_levels.py` (new)

Standalone amplitude-invariance probe (US2). **Must not download models** — uses cached
checkpoints and skips with a clear message when a model is absent (constitution VI).

```bash
uv run python scripts/probe_classifier_levels.py \
    --input src/tests/data_for_testing/<clip>.wav \
    --gains-db -40 -20 -10 0 10 \
    --classifiers ast yamnet \
    --out artifacts/level_probe/
```

| Flag | Default | Effect |
|---|---|---|
| `--input PATH` | required | Probe clip |
| `--gains-db ...` | `-40 -20 -10 0 10` | Gain points; range must span ≥30 dB (SC-005) |
| `--classifiers ...` | both | Which classifiers to probe |
| `--out DIR` | `artifacts/level_probe/` | Verdict output directory |
| `--include-silence-probe / --no-...` | included | Also probe digital silence to capture floor signatures (FR-020d) |

**Output**: one `level-verdicts.json` per run (see `level-verdicts.md`).

## `scripts/calibrate_detection_margin.py` (new)

Produces a versioned `DetectionMarginProfile` with its `DerivationRecord` (FR-022).

```bash
uv run python scripts/calibrate_detection_margin.py \
    --level-verdicts artifacts/level_probe/level-verdicts.json \
    --out src/senselab/audio/workflows/audio_analysis/data/detection_margin/<version>.json
```

Refuses to emit a profile whose `derivation.verification_status` marks a figure as
`provisional` without an accompanying note — the spec requires the evidential basis to be
auditable, so an unmarked provisional figure is a hard error.

## Behavior compatibility

Per the project's pre-alpha position, **no backwards compatibility is maintained**. Two
default-behavior changes are intentional and are defects being fixed, not features:

1. Scene-analysis score aggregation changes because summing mutually-exclusive and
   independent scores was structurally suppressing background categories (FR-017c).
2. The YAMNet input path gains amplify-before-serialize ordering (FR-019b).

Both change existing outputs. `CACHE_SCHEMA_VERSION` is bumped so stale cache entries are
discarded rather than reused.
