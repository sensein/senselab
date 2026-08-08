# CLI Contract: profile_with_scalene.py

**Branch**: `20260503-235625-scalene-profiling` | **Date**: 2026-05-04
**Verified against**: Scalene 2.2.1 (CLI subcommand structure: `run`, `view`)

## Command

```bash
uv run python scripts/profile_with_scalene.py [OPTIONS] TARGET [--- TARGET_ARGS...]
```

## Positional Arguments

| Argument | Description |
|----------|-------------|
| TARGET | Path to a `.py` script or `.ipynb` notebook to profile (required) |
| TARGET_ARGS | Arguments forwarded to the target script after `---` (three dashes; matches Scalene's separator). Optional, variadic. |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` | `artifacts/scalene/` | Directory where the report is written |
| `--format {html,json}` | `html` | Report output format. JSON is Scalene's native run output; HTML is produced by an additional `scalene view --standalone` step. |
| `--cpu-only` | `false` | Skip memory profiling for a faster run (passes Scalene's `--cpu-only`) |
| `--no-thirdparty` | `false` | Restrict profiling to files containing `senselab` in their path (passes Scalene's `--profile-only senselab`). Mutually exclusive with `--scope`. |
| `--scope SUBSTR` | (none) | Restrict profiling to files whose path contains SUBSTR (substring match; passes Scalene's `--profile-only`). Mutually exclusive with `--no-thirdparty`. |
| `--exclude SUBSTR` | (none) | Exclude files whose path contains SUBSTR (substring match; passes Scalene's `--profile-exclude`). May be combined with the scope flags. |
| `--gpu` | `false` | Enable GPU profiling (only effective when CUDA is available; on macOS the GPU columns will be empty regardless) |
| `--keep-intermediate` | `false` | Retain the intermediate JSON profile (when `--format html`) and the converted `.py` file (when target is a notebook) next to the final report |
| `--help`, `-h` | — | Show help text |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — report written |
| 1 | Profiling failed (Scalene errored, target script raised, or HTML view step failed) |
| 3 | Scalene is not installed in the current environment |
| 4 | Invalid arguments (e.g., target path does not exist, mutually-exclusive flags combined) |
| 5 | `nbconvert` is not installed (only when target is `.ipynb`) |

## Output

On success, prints:

```text
Scalene profile written to: <absolute_output_path>
Open with: open <absolute_output_path>   (macOS)
            xdg-open <absolute_output_path>  (Linux)
```

On missing Scalene (exit 3):

```text
ERROR: Scalene is not installed in this environment.
To install: uv sync --group profiling
```

On missing nbconvert (exit 5):

```text
ERROR: nbconvert is not installed; required to profile notebooks.
To install: uv sync --group profiling
```

## Examples

### Profile a Python script (default options)

```bash
uv run python scripts/profile_with_scalene.py examples/short_demo.py
```

### Profile a tutorial notebook

```bash
uv run python scripts/profile_with_scalene.py tutorials/audio/00_getting_started.ipynb
```

### Scoped profile (only files with "speech_to_text" in path)

```bash
uv run python scripts/profile_with_scalene.py --scope speech_to_text examples/run_stt.py
```

### Profile with arguments forwarded to target

```bash
uv run python scripts/profile_with_scalene.py examples/run_with_input.py --- --input my.wav --batch 4
```

Note the `---` (three dashes) separator. This matches Scalene's own separator.

### CPU-only fast run

```bash
uv run python scripts/profile_with_scalene.py --cpu-only tutorials/audio/speech_to_text.ipynb
```

### Hide third-party libraries

```bash
uv run python scripts/profile_with_scalene.py --no-thirdparty examples/short_demo.py
```

### Hide a specific noisy module

```bash
uv run python scripts/profile_with_scalene.py --exclude transformers examples/short_demo.py
```

### JSON output for tooling integration

```bash
uv run python scripts/profile_with_scalene.py --format json examples/short_demo.py
```

## Implementation Notes

### Scalene invocation pattern (two-step)

The wrapper performs two subprocess calls:

1. **Profile** — produces a JSON profile file:

   ```text
   python -m scalene run -o <stem>.json [scalene-flags...] <target_or_converted_py> [--- <target_args>]
   ```

   Scalene-flag mapping:

   - `--cpu-only` → `--cpu-only`
   - `--gpu` → `--gpu`
   - `--scope X` → `--profile-only X`
   - `--no-thirdparty` → `--profile-only senselab`
   - `--exclude X` → `--profile-exclude X`

2. **View** — when `--format html` (default), convert JSON to a standalone HTML file:

   ```text
   python -m scalene view --standalone <stem>.json
   ```

   Produces `<stem>.html`. The wrapper deletes `<stem>.json` afterward unless `--keep-intermediate` is set.

   When `--format json`, the view step is skipped and the JSON path is reported as the result.

### Notebook handling

When TARGET ends in `.ipynb`, the wrapper:

1. Verifies `nbconvert` is importable (else exits 5 with install hint)
2. Creates a `tempfile.TemporaryDirectory()`
3. Runs `python -m jupyter nbconvert --to script --output-dir <tmpdir> <notebook>` (or equivalently `python -m nbconvert ...`)
4. Locates the resulting `<stem>.py` in tmpdir
5. Profiles the converted `.py`
6. Cleans up tmpdir unless `--keep-intermediate` is set (in which case the `.py` is copied next to the final report)

### Mutually exclusive flags

- `--scope` and `--no-thirdparty` cannot be combined (argparse mutually-exclusive group)
- `--exclude` may be combined with either `--scope` or `--no-thirdparty`

## Programmatic Usage (Optional)

For scoped profiling driven from inside a Python script, Scalene exposes module-level primitives. Consult Scalene's own documentation for the up-to-date API (`scalene_profiler` module). The wrapper does not depend on these primitives — they work whenever a script is run under `python -m scalene run`. They are mentioned here for awareness only; the supported and documented path for scoping is the `--scope` / `--no-thirdparty` / `--exclude` flags above.
