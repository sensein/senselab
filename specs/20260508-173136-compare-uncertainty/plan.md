# Implementation Plan: Comparison & Uncertainty Stage

**Branch**: `20260508-173136-compare-uncertainty` | **Date**: 2026-05-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification at `specs/20260508-173136-compare-uncertainty/spec.md`

## Summary

The comparator stage emits three per-bucket uncertainty time series, one per question
reviewers care about:

- **presence_uncertainty** — was there a speaker?
- **identity_uncertainty** — was it the same speaker?
- **utterance_uncertainty** — what was said?

Each axis collapses every contributing model's signal into a single `[0, 1]` scalar per
bucket. The output is exactly nine parquets per default two-pass run (3 axes × 2 passes +
3 raw-vs-enhanced deltas), one ranked `disagreements.json`, one timeline plot, and matching
LS-bundle tracks. There are no per-model-pair sidecars; the pairwise WER / label-flip /
boundary-shift artefacts exist only as in-memory intermediate state during cross-model
aggregation.

## Technical Context

**Language/Version**: Python 3.11–3.14 (managed via uv), matches senselab's `requires-python`.

**Primary Dependencies**:
- `senselab` (the package being extended).
- `pandas` + `pyarrow` (already in the venv via the features pipeline).
- `jiwer` for WER (already in the `[nlp]` extra).
- `g2p_en` for grapheme-to-phoneme (already added in the PPG sub-feature).
- `senselab.audio.tasks.speaker_embeddings.extract_speaker_embeddings` for ECAPA / ResNet
  per-segment embedding extraction. No new external dependency.

**Storage**: file-based —
- `<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet` (per-pass axes)
- `<run_dir>/uncertainty/raw_vs_enhanced/{presence,identity,utterance}.parquet` (deltas)
- `<run_dir>/disagreements.json` (top-N ranked across all 9 parquets)
- `<run_dir>/timeline.png` (5-row figure)
- `<run_dir>/labelstudio_{config.xml,tasks.json}` (existing bundle, with new uncertainty tracks)

**Testing**: pytest under uv; smoke tests in `src/tests/scripts/analyze_audio_test.py` use
cached / synthetic upstream results so the comparator framework can be validated without
re-running heavy models. The integration test that exercises the full stage retains its
existing skipif gates for venv- or weight-bound backends.

**Target Platform**: macOS arm64 + Linux CI; CPU-only by default, GPU when available for
ECAPA / ResNet inference.

**Project Type**: Python library + CLI script (`scripts/analyze_audio.py`).

**Performance Goals**: comparator stage adds no more than 30 % wall-clock overhead on top of
the existing analyze_audio run on a 1-minute clip when running on already-cached upstream
task results (carry-over from SC-004). The new cost is one ECAPA + one ResNet inference per
diarization segment per pass (≈ 10 segments per minute at default settings).

**Constraints**:
- `_CACHE_SCHEMA_VERSION = 1`.
- Existing per-task outputs (diarization JSON, ASR JSON, etc.) MUST remain unchanged in
  shape (FR-011, SC-005). The comparator is purely additive on top of those.
- Default cross-stream grid is `0.5 s` non-overlapping (FR-009).

**Scale/Scope**: a typical 1-minute conversational clip ≈ 120 buckets at the 0.5 s default
grid; 9 parquets × 120 rows = ≈ 1080 rows total per run, ≈ 5–8 KB each.

## Constitution Check

| Principle | Compliance |
|---|---|
| **I. UV-Managed Python** (NON-NEGOTIABLE) | ✅ All commands under `uv run`. New tests under `uv run pytest`. No `pip` / bare `python`. |
| **II. Encapsulated Testing** | ✅ Smoke tests in uv-managed venv. Integration test inherits existing skipif gates. |
| **III. Commit Early and Often** | ✅ The comparator lands as a sequence of small commits — data-model + contracts, helper functions per axis, plot, tests, E2E validation note. |
| **IV. CI Must Stay Green** | ✅ Smoke tests stay green at every commit boundary; ruff + mypy + codespell pass. |
| **V. Memory-Driven Anti-Pattern Avoidance** | ✅ Memory records the design rule: uncertainty is aggregated cross-model per axis, not pairwise. |
| **VI. No Unnecessary API Calls** | ✅ Speaker-embedding extraction reuses the existing `extract_speaker_embeddings` API, batched per pass. Cross-stream grid is non-overlapping by default. |
| **VII. Simplicity First** | ✅ One cross-model aggregation function per axis (presence / identity / utterance), not N×(N−1)/2 pairwise differencers. |
| **VIII. No Hardcoded Parameters** | ✅ `--cross-stream-win-length`, `--cross-stream-hop-length`, `--uncertainty-aggregator`, `--phoneme-disagreement-threshold`, `--speech-presence-labels`, `--asr-reference-model`, `--diarization-boundary-shift-ms`, `--disagreements-top-n` all carry over. LS bin thresholds (`< 0.33` / `[0.33, 0.66)` / `≥ 0.66`) ship as a default; configurable in a follow-up if needed. |

**Gate result**: PASS. No violations to track in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/20260508-173136-compare-uncertainty/
├── plan.md              # This file
├── research.md          # Phase 0: per-axis math, embedding source, output layout
├── data-model.md        # UncertaintyRow shape, per-axis vote dicts
├── quickstart.md        # How to consume the 9 parquets + plot
├── contracts/
│   ├── cli.md
│   ├── uncertainty-row.parquet.md
│   ├── disagreements.json.md
│   └── ls-bundle.md
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit.tasks output
```

### Source Code (repository root)

The comparator lives wholly inside the existing `scripts/analyze_audio.py` and its test
file; no new modules are added.

```text
scripts/
└── analyze_audio.py
    # Comparator helpers:
    #   _presence_votes_for_bucket(...)
    #   _identity_votes_for_bucket(...)
    #   _utterance_votes_for_bucket(...)
    #   _aggregate_axis_uncertainty(votes, axis, aggregator)
    #   compute_uncertainty_axes(passes, args, run_dir, ...)
    #     → emits 9 parquets, returns metadata for disagreements.json + LS bundle.
    #   build_aligned_timeline_plot(...) → 5-row figure
    #     (presence / identity / utterance overlaid raw vs enhanced + delta + reference)
    #   _build_disagreements_index(...) → reads the 9 axis parquets, ranks by
    #     aggregated_uncertainty desc with axis-priority tiebreak
    #     (utterance > identity > presence)

src/senselab/
└── (no changes — every helper used by the comparator already exists in the package)

src/tests/scripts/
└── analyze_audio_test.py
    # ~12 axis-aggregation tests:
    #   • presence: entropy of binary votes (0, 1, 2, n contributing models)
    #   • identity: cross-model label-pair fraction + cosine sub-signals + raw_vs_enh delta
    #   • utterance: pairwise mean WER + Whisper avg_logprob + PPG PER
    #   • disagreements.json: ordering + axis-priority tiebreak
    #   • plot: smoke-test that build_aligned_timeline_plot produces a 5-row figure

artifacts/
└── compare_uncertainty_e2e_validation.md
    # E2E run on the 38.4 s twin-1 clip: per-axis disagreement-bucket counts,
    # plot screenshot, top-10 disagreements.json entries.
```

**Structure Decision**: Single-project layout (Python library + CLI script). The full
comparator lives in `scripts/analyze_audio.py` and its sibling test file; no new modules
or packages.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none — see Constitution Check) | — | — |

## Out of Scope

- Configurable LS bin thresholds (`--ls-low-threshold`, `--ls-high-threshold`).
- Per-axis `--{presence,identity,utterance}-models` overrides for advanced workflows.
- Ensemble / Monte-Carlo dropout uncertainty for individual models.
- Cross-language phoneme inventory unification beyond English ARPAbet + uroman.
