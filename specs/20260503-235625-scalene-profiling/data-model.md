# Data Model: Scalene-Based Profiling Tooling

**Branch**: `20260503-235625-scalene-profiling` | **Date**: 2026-05-04

## Entities

### ProfilingTarget

The Python script, notebook, or scoped function being measured.

| Field | Description |
|-------|-------------|
| path | Path to the `.py` or `.ipynb` file to profile |
| kind | One of: `python_script`, `jupyter_notebook` |
| script_args | Optional list of arguments forwarded to the target script |
| scope | Optional scope identifier — when set, Scalene reports only on functions matching this name |

### ProfileReport

The output artifact produced by a profiling run.

| Field | Description |
|-------|-------------|
| target | Reference to the ProfilingTarget |
| output_path | Absolute path to the generated report file |
| format | One of: `html`, `json` |
| generated_at | ISO-8601 timestamp |
| scalene_version | Version of Scalene used |
| platform | `macOS-arm64`, `linux-x86_64`, etc. |
| gpu_profiling_enabled | Boolean — true only when CUDA is available and `--gpu` was passed |
| target_exit_code | Exit code of the profiled script (Scalene's `run` produces a complete JSON or none; partial reports are not produced) |

### ProfilingConfiguration

Runtime options used for a profiling invocation.

| Field | Description |
|-------|-------------|
| cpu_only | If true, skip memory profiling for faster runs |
| profile_all | If true, include third-party library lines in report; else senselab-only |
| sampling_rate | Sampling interval in seconds (default: Scalene's default) |
| include_children | If true, profile child processes spawned by target script |
| profile_only | Optional substring matched against function names to scope reporting |

## Relationships

```
ProfilingTarget 1──1 ProfileReport
ProfilingTarget 1──1 ProfilingConfiguration  (one config per invocation)
```

## State Transitions

ProfileReport status flow:
```
not_started → running → success (run + view both succeeded)
                      → failed (run step errored, view step errored, or target script raised — no report or incomplete JSON)
```

Scalene 2.2.1 does not produce partial profiles on target-script crash; the wrapper does not attempt to recover one.

## Validation Rules

- `path` must exist and be readable
- `kind` is derived from extension: `.ipynb` → `jupyter_notebook`, `.py` → `python_script`
- `format` must be `html` (default) or `json`
- `output_path` must reside under `artifacts/scalene/` (FR-010)
- `gpu_profiling_enabled` is forced to `false` on platforms without CUDA, regardless of user request
- `scope` (if set) is a substring; case-insensitive match against function names

## Notes

This data model is conceptual only — it describes the wrapper script's behavior, not a persisted database. The wrapper passes derived values to the Scalene CLI as flags; nothing is serialized except the final report file written by Scalene itself.
