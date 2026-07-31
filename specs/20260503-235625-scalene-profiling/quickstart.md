# Quickstart: Scalene-Based Profiling

**Branch**: `20260503-235625-scalene-profiling` | **Date**: 2026-05-04
**Verified against**: Scalene 2.2.1

## What This Adds

A one-command profiling tool for any Python script or Jupyter notebook in senselab. Produces interactive standalone HTML reports showing line-level CPU and memory usage. Available as an opt-in dependency; default `uv sync` is unchanged.

## Install

```bash
uv sync --group profiling
```

This adds Scalene and `nbconvert` to your local environment. The default `uv sync` (no flag) does not install either.

## Profile a Python Script

```bash
uv run python scripts/profile_with_scalene.py path/to/script.py
```

Output: `artifacts/scalene/<script>_<timestamp>.html`. Open it in any browser.

## Profile a Tutorial Notebook

```bash
uv run python scripts/profile_with_scalene.py tutorials/audio/speech_to_text.ipynb
```

The wrapper auto-converts the notebook to a temporary `.py` file via `nbconvert` before profiling.

## Profile a Specific Subset

`--scope` and `--no-thirdparty` filter by **file path substring** (matching Scalene's `--profile-only`). They do not match function names directly — they match any file whose path contains the substring.

To profile only files whose path contains "speech_to_text":

```bash
uv run python scripts/profile_with_scalene.py --scope speech_to_text examples/run_stt.py
```

To restrict to senselab code only:

```bash
uv run python scripts/profile_with_scalene.py --no-thirdparty examples/short_demo.py
```

To hide a single noisy module while keeping everything else:

```bash
uv run python scripts/profile_with_scalene.py --exclude transformers examples/short_demo.py
```

## Common Options

| Option | When to use |
|--------|-------------|
| `--cpu-only` | Faster runs; skip memory tracking |
| `--no-thirdparty` | Hide PyTorch / transformers internals; focus on senselab code |
| `--scope SUBSTR` | Profile only files whose path contains SUBSTR |
| `--exclude SUBSTR` | Hide files whose path contains SUBSTR |
| `--format json` | Machine-readable output (skips the HTML conversion step) |
| `--gpu` | Enable GPU profiling (no-op on macOS) |

See `specs/20260503-235625-scalene-profiling/contracts/cli.md` for the full CLI reference.

## Forward Args to the Target Script

Use `---` (three dashes; matches Scalene's separator):

```bash
uv run python scripts/profile_with_scalene.py examples/run.py --- --input my.wav --batch 4
```

## Verify the Installation

```bash
uv run python scripts/profile_with_scalene.py --help
```

If Scalene is not installed, you'll see:

```text
ERROR: Scalene is not installed in this environment.
To install: uv sync --group profiling
```

## Worked Example

Profile the speech-to-text tutorial end-to-end:

```bash
uv sync --group profiling
uv run python scripts/profile_with_scalene.py tutorials/audio/speech_to_text.ipynb
open artifacts/scalene/speech_to_text_*.html  # macOS
```

Look for the slowest senselab functions in the resulting HTML report. The `Memory` column shows per-line memory allocation; the `CPU %` column shows where time is spent.

## Caveats

- macOS ARM64: GPU profiling is not available (no CUDA). CPU and memory profiling work normally.
- Notebooks must be non-interactive (no `input()` calls; no `widgets` waiting on UI events).
- Scalene adds runtime overhead (single-digit percent in normal mode, more with `--memory`); profile representative inputs, not microbenchmarks.
- Scalene 2.2.1 does not expose a child-process profiling flag from the wrapper. To profile a subprocess workload, profile the subprocess script directly.

## Why a Separate Tool from `profile_imports.py`?

`scripts/profile_imports.py` measures cold-start import times (one-shot, subprocess-isolated). The new `profile_with_scalene.py` is a general-purpose profiler — it tells you what happens *after* imports complete: which lines spend time, which lines allocate memory. Use them together for a complete picture.
