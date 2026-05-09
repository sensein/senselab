# Implementation Plan: Comparison & Uncertainty Stage for analyze_audio.py

**Branch**: `20260508-173136-compare-uncertainty` | **Date**: 2026-05-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/20260508-173136-compare-uncertainty/spec.md`

## Summary

Add a fourth stage to `scripts/analyze_audio.py` that consumes the existing per-task outputs (diarization, AST, YAMNet, ASR, alignment) and emits four classes of comparison artefacts: (1) raw vs enhanced mismatches per (task, model), (2) within-stream model disagreements per (task, model_pair), (3) cross-stream mismatches (ASR↔diar, AST/YAMNet↔diar speech-presence, ASR↔PPG phoneme sequence), (4) per-region uncertainty harvested from underlying models. Outputs are parquet sidecars under `<run_dir>/<pass>/comparisons/`, a top-level `disagreements.json` index, and additional Label Studio Labels tracks. The stage hooks into the existing content-addressable cache so cache-only runs replay quickly.

## Technical Context

**Language/Version**: Python 3.11–3.14 (managed via uv) — matches senselab's `requires-python`.
**Primary Dependencies**: senselab (the merged audio analysis module from PR #510), pandas + pyarrow (already in the active venv via the prior PR's features pipeline), `jiwer` for WER (already in the `[nlp]` extra), `g2p-en` or similar small G2P library for grapheme→phoneme on the ASR side (new, ~1 MB).
**Storage**: File-based — parquet under `<run_dir>/<pass>/comparisons/<task>.parquet` and `<run_dir>/<pass>/comparisons/cross_stream/<a>_vs_<b>.parquet`; JSON for `<run_dir>/disagreements.json`; XML/JSON appendage to the existing LS bundle.
**Testing**: pytest (existing infra; `src/tests/scripts/analyze_audio_test.py` is the integration anchor). New comparator units must run in <30 s without venv provisioning or model downloads (SC-006), so tests operate on synthetic / cached upstream results.
**Target Platform**: Same as the merged script — macOS arm64 + Linux x86_64; CUDA / MPS where applicable. The comparator stage itself runs on CPU (no model loads).
**Project Type**: CLI script extension. Backwards-compatible additive output.
**Performance Goals**: ≤30 % wall-clock overhead on top of cached upstream tasks for a 1-minute clip (SC-004). Cache replay ≥95 % hit rate (SC-003).
**Constraints**: MUST NOT change shape of any existing per-task output (FR-005, SC-005). All comparator outputs strictly additive under `<run_dir>/<pass>/comparisons/`. Wrapper-version-hash invalidation rules from PR #510 continue to apply.
**Scale/Scope**: One audio per run, two passes (raw + enhanced), 4 ASR + 2 diarization + 2 classification models in the default config. Comparison parquet rows: ~10 rows/sec for cross-stream at the chosen 0.1 s hop → ~600 rows/min/pair; ~2 rows/sec for raw_vs_enhanced/within-stream at the features grid → ~120 rows/min/pair. Manageable.

## Constitution Check

| Principle | Compliance | Notes |
|---|---|---|
| **I. UV-Managed Python** | ✅ | All commands run via `uv run`; no bare `python`. New deps go through `uv sync --extra nlp` (jiwer already there) plus an opt-in `g2p` / `phonemizer` package. |
| **II. Encapsulated Testing** | ✅ | Comparator tests run in the existing `src/tests/scripts/` virtualenv; integration tests use the same skipif guards introduced in PR #510 for venv-bound backends. |
| **III. Commit Early and Often** | ✅ | Plan partitions the work into US1 / US2 / US3 / US4 commits, each independently testable; further sub-task commits within each US. |
| **IV. CI Must Stay Green** | ✅ | Comparator path runs on cached upstream results in CI (no heavy models). The integration test that exercises the full stage skips when venv-bound backends aren't provisioned. |
| **V. Memory-Driven Anti-Pattern Avoidance** | ✅ | We carry forward the dict-vs-Pydantic tolerance pattern (`_seg_attr` from PR #510) for any LS-helper that touches comparison rows; no new unaddressed anti-patterns. |
| **VI. No Unnecessary API Calls** | ✅ | Comparator runs entirely on local data — no HF Hub calls. The G2P library is pure-Python with bundled dictionary; no model download. |
| **VII. Simplicity First** | ✅ | One new module (`scripts/comparisons.py` or in-script section), one new helper for each comparison kind, one new section in build_labelstudio_task. No new abstractions until tested. |
| **VIII. No Hardcoded Parameters** | ✅ | All thresholds and grid resolutions are CLI-configurable: `--cross-stream-win-length`, `--cross-stream-hop-length`, `--phoneme-disagreement-threshold`, `--uncertainty-aggregator`, `--disagreements-top-n`. Speech-presence allowlist for AST/YAMNet is a list constant with override flag `--speech-presence-labels`. |

**Verdict**: ✅ All gates pass. Proceed to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/20260508-173136-compare-uncertainty/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (CLI flags + parquet schemas)
│   ├── cli.md
│   ├── comparison-row.parquet.md
│   ├── disagreements.json.md
│   └── ls-bundle.md
├── checklists/
│   └── requirements.md  # Already created by /speckit.specify
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
scripts/
├── analyze_audio.py                    # extend: new comparison stage hook in run_pass(),
│                                       # new CLI flags, extended LS export
└── (comparisons helpers live inline in analyze_audio.py for simplicity-first;
   if the file grows past ~2000 lines we extract scripts/_compare.py — not before)

src/senselab/audio/tasks/
├── speech_to_text/
│   └── (no changes — comparator reads existing ScriptLine outputs)
├── classification/
│   └── (no changes — comparator reads existing windowed dicts)
├── speaker_diarization/
│   └── (no changes — comparator reads existing ScriptLine outputs)
└── (no new senselab task module — comparator lives in the script)

src/tests/scripts/
├── analyze_audio_test.py               # extend: tests for comparator pure-functions
│                                       # using synthetic ScriptLine / dict fixtures
└── (no new test file unless the comparator helpers grow large enough to
   warrant their own — start with the existing one)
```

**Structure Decision**: All comparator logic lives in `scripts/analyze_audio.py` initially. Senselab core is unchanged — the comparator is a *consumer* of existing senselab outputs, not a new senselab task. This keeps the feature surface small (Principle VII) and lets us iterate on the comparator without re-running the senselab test suite. If a follow-up benchmark or notebook needs the comparator standalone, we'll extract `scripts/_compare.py` then.

## Phases

### Phase 0 — Research

Output: `research.md`

Topics to settle:

1. **Token-timestamp inventory across ASR backends** — for FR-009 / Q4 (ASR-says-speech via overlapping token timestamps): catalogue what each of the four ASR backends (Whisper turbo, Granite, Canary-Qwen, Qwen3-ASR) currently emits and how the auto-align stage normalizes them. Confirm Whisper exposes word-level timestamps via `return_timestamps="word"`; Qwen3-ASR exposes ForcedAligner-derived word chunks; Granite/Canary-Qwen are text-only and rely on the MMS auto-align stage (already producing per-segment timestamps in the merged PR).
2. **Confidence signals per backend** — for FR-007: enumerate the native scalar each backend exposes today (Whisper `avg_logprob` / `no_speech_prob` from `chunks`; pyannote per-frame score; AST/YAMNet top-1 score and full distribution; Granite/Canary-Qwen no native — flag for future plumbing; MMS aligner per-segment trellis score). Decide which signals are wired in v1 vs deferred.
3. **G2P library choice** — for FR-008: pick a small pure-Python G2P with a CMU-style ARPAbet output that aligns with the senselab PPG backend's phoneme inventory. Candidates: `g2p_en` (English-only, ~1 MB), `phonemizer` (multi-language, requires espeak system dep), `epitran` (multi-language, no system dep). Decision criteria: (a) inventory match with PPG backend, (b) install footprint, (c) language coverage matching the existing senselab `[nlp]` extra.
4. **WER computation** — confirm `jiwer` already in `[nlp]` extra (yes, ≥3.0). Decide on per-window tokenization: `jiwer.wer(reference, hypothesis)` after splitting the transcript by the cross-stream grid bucket boundaries; document edge-case (empty bucket → WER = 0 if both empty, 1 if one is empty).
5. **AST/YAMNet speech-presence label allowlist** — for cross-stream US3 scenario 2: which AudioSet labels count as "speech-present"? Default: `{"Speech", "Conversation", "Narration, monologue", "Female speech, woman speaking", "Male speech, man speaking", "Child speech, kid speaking", "Speech synthesizer"}`. Override via `--speech-presence-labels` (comma-separated list).
6. **Cache-key composition for the comparator** — for FR-005: define the comparison cache key as `sha256(audio_signature, comparison_kind, task_or_pair, model_set, params, wrapper_hash, senselab_version, schema_version=1)` where `model_set` is the sorted tuple of upstream cache_keys feeding this comparison (so an upstream task re-running invalidates downstream comparisons automatically).
7. **LS Labels enumeration** — for FR-004: enumerate the `<Labels>` values once at config-build time. The set is `{"agree", "disagree", "incomparable", "one_sided"}` (per-pair labels are encoded in the track *name*, not the label values, so we don't combinatorially explode the XML).

### Phase 1 — Design & Contracts

Outputs:

- **`data-model.md`** — fields for ComparisonRow, DisagreementsIndex, UncertaintyAnnotation, ComparisonGrid; per-task variant schemas (ASR row carries `wer/a_text/b_text`; classification row carries `top1_a/top1_b/agree`; diarization row carries `speaks_a/speaks_b/agree`).
- **`contracts/cli.md`** — exact spelling of the new CLI flags: `--skip-comparisons {raw_vs_enhanced,within_stream,cross_stream,uncertainty}`, `--cross-stream-win-length`, `--cross-stream-hop-length`, `--uncertainty-aggregator`, `--phoneme-disagreement-threshold`, `--speech-presence-labels`, `--disagreements-top-n`. Defaults match the spec clarifications.
- **`contracts/comparison-row.parquet.md`** — column-by-column schema for each parquet file the comparator emits, with dtypes, nullability, and example rows.
- **`contracts/disagreements.json.md`** — JSON shape for the index (`{schema_version, generated_at, top_n, aggregator, entries: [{rank, region_id, pass, parquet_path, row_index, start, end, mismatch_type, combined_uncertainty, ls_track_name}]}`).
- **`contracts/ls-bundle.md`** — exact LS XML appendage: one `<Labels>` block per (pass, comparison_kind, task) actually used in the run; values from the fixed enumerated set; one prediction per disagreement row.
- **`quickstart.md`** — three reviewer-facing recipes: (a) "did enhancement help?" (read raw_vs_enhanced track), (b) "where do my ASR models disagree?" (read within_stream parquet), (c) "where do ASR and diarization conflict?" (read cross_stream parquet).
- **Agent context update**: run `.specify/scripts/bash/update-agent-context.sh claude` to fold the new comparator design into CLAUDE.md.

### Phase 2 — Tasks (output: tasks.md, generated by `/speckit.tasks`)

Not created by `/speckit.plan`. Sketch of the expected partition (for later use):

- **Phase A — Foundational comparator framework**: cache-key helpers, ComparisonRow dataclass / TypedDict, parquet writer, LS export hooks. Pure-functional; testable with synthetic inputs.
- **Phase B — US1 (P1) raw_vs_enhanced**: per-task differencer that pairs `summary["passes"]["raw_16k"][task]` with `summary["passes"]["enhanced_16k"][task]` on the features grid; one parquet per task; LS Labels track per (task, model). Smoke test on synthetic dual-pass dicts.
- **Phase C — US2 (P2) within-stream**: model-pair iterator per task per pass. ASR-vs-ASR uses jiwer WER; classification top-1 vs top-1; diarization speech-presence per cross-stream-grid-bucket. Reuses Phase A schema.
- **Phase D — US3 (P2) cross-stream**: ASR↔diar, AST/YAMNet↔diar, ASR↔PPG. The PPG path is gated on `ppgs` extra availability; degrade gracefully when missing. New `--speech-presence-labels` flag. New G2P dep added to `[nlp]` extra.
- **Phase E — US4 (P3) uncertainty**: per-backend confidence harvesters; `combined_uncertainty` populated on every comparison row; `disagreements.json` ranking sorted by configurable aggregator. Touches every existing per-task region (only adding columns; no shape changes).
- **Phase F — Polish**: lint sweep, mypy clean, run the senselab test suite to confirm SC-005 (no regression in existing outputs), end-to-end run on twin-1.wav, update CLAUDE.md, document the comparator in the script's docstring header.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| (none) | — | — |

No constitution violations to justify. The plan adds CLI flags rather than baked-in constants, scopes new code to a single script, and reuses the existing cache primitives.
