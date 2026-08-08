# Data Model: Optimize Import Times

**Branch**: `20260501-154228-optimize-import-times` | **Date**: 2026-05-01

## Entities

### ImportStatement

A single Python import line extracted from a tutorial notebook.

| Field | Description |
|-------|-------------|
| raw_line | The original import line as it appears in the notebook (e.g., `from senselab.audio.data_structures import Audio`) |
| module_path | The top-level module being imported (e.g., `senselab.audio.data_structures`) |
| imported_name | The specific name imported, if any (e.g., `Audio`) |
| source_notebooks | List of tutorial notebook paths that contain this import |
| wall_clock_seconds | Measured cold-start wall-clock time in seconds |
| status | One of: `success`, `failed`, `skipped` |
| error_message | Error text if status is `failed`, null otherwise |
| is_bottleneck | True if wall_clock_seconds exceeds the threshold |
| category | One of: `senselab`, `third_party`, `stdlib`, `platform_specific` |

### ImportBreakdown

Transitive dependency timing for a single import (from `-X importtime` output).

| Field | Description |
|-------|-------------|
| parent_import | Reference to the ImportStatement being profiled |
| child_module | The transitively imported module (e.g., `torch`, `numpy`) |
| self_time_us | Self time in microseconds (time spent in this module alone) |
| cumulative_time_us | Cumulative time including all sub-imports |
| depth | Nesting depth in the import tree |

### TutorialNotebook

A Jupyter notebook file containing import blocks.

| Field | Description |
|-------|-------------|
| file_path | Relative path from repo root (e.g., `tutorials/audio/speech_enhancement.ipynb`) |
| display_name | Human-readable name derived from filename |
| imports | Ordered list of ImportStatement references |
| total_cold_start_seconds | Aggregate cold-start time for all imports executed together in one process |
| unique_import_count | Number of distinct import lines in this notebook |

### BottleneckReport

The output artifact aggregating all profiling results.

| Field | Description |
|-------|-------------|
| generated_at | Timestamp of report generation |
| threshold_seconds | The bottleneck threshold used (default: 2.0) |
| ranked_imports | All ImportStatements sorted by wall_clock_seconds descending |
| tutorial_summaries | All TutorialNotebooks sorted by total_cold_start_seconds descending |
| bottleneck_breakdowns | ImportBreakdown lists for each bottleneck import |
| skipped_imports | ImportStatements with status `skipped` |
| failed_imports | ImportStatements with status `failed` |

## Relationships

```
TutorialNotebook 1──* ImportStatement
ImportStatement 1──* ImportBreakdown  (only for bottleneck imports)
BottleneckReport 1──* ImportStatement
BottleneckReport 1──* TutorialNotebook
```

## State Transitions

ImportStatement status flow:
```
pending → running → success
                  → failed (import raised exception)
                  → skipped (platform-specific, e.g., google.colab)
```

## Validation Rules

- `wall_clock_seconds` must be non-negative
- `status` must be one of the defined enum values
- `is_bottleneck` must be consistent with `wall_clock_seconds >= threshold`
- `source_notebooks` must contain at least one entry
- `total_cold_start_seconds` for a tutorial must be measured independently (not summed from individual import times, since sequential imports in one process share cached modules)
