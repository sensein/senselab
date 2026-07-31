# Quickstart: Optimize Import Times

**Branch**: `20260501-154228-optimize-import-times` | **Date**: 2026-05-01

## What This Feature Does

Profiles the cold-start import time of every Python import used across senselab's tutorial notebooks. Produces a report identifying which imports are slow, why they're slow (internal vs third-party), and which tutorials are most affected.

## How to Use

### Run the profiling script

```bash
uv run python scripts/profile_imports.py
```

This will:
1. Parse all tutorial notebooks under `tutorials/`
2. Extract distinct import statements
3. Time each import in an isolated subprocess
4. For bottleneck imports (>2s), capture transitive dependency breakdown
5. Time each tutorial's full import block as a unit
6. Write a Markdown report to `artifacts/import_profile_report.md`

### Customize the threshold

```bash
uv run python scripts/profile_imports.py --threshold 3.0
```

### View the report

The report has three sections:
1. **Ranked Imports** — all imports sorted slowest to fastest, bottlenecks flagged
2. **Per-Tutorial Summary** — aggregate import time for each tutorial notebook
3. **Dependency Breakdowns** — for each bottleneck import, which child modules consume the most time

## Prerequisites

- All senselab extras installed: `uv sync --extra articulatory --extra text --extra video --extra senselab-ai --group dev`
- macOS ARM64 or Linux (Colab-specific imports are auto-skipped)

## Expected Output

A Markdown report like:

```
# Import Profile Report
Generated: 2026-05-01T15:42:28

## Ranked Imports (slowest first)

| # | Import | Time (s) | Category | Bottleneck |
|---|--------|----------|----------|------------|
| 1 | import torch | 4.23 | third_party | YES |
| 2 | from speechbrain.inference... | 3.87 | third_party | YES |
| ... | ... | ... | ... | ... |

## Per-Tutorial Summary

| Tutorial | Import Time (s) | # Imports |
|----------|----------------|-----------|
| audio_recording_and_acoustic_analysis | 12.4 | 25 |
| transcription_and_phonemic_analysis | 11.8 | 24 |
| ... | ... | ... |

## Dependency Breakdowns

### import torch (4.23s)
  torch._C: 2100ms (self)
  torch.nn: 450ms (self)
  ...
```
