# Research: Optimize Import Times

**Branch**: `20260501-154228-optimize-import-times` | **Date**: 2026-05-01

## Import Chain Analysis

### Decision: Senselab's import architecture is lazy-by-design but base deps are heavy

**Rationale**: Analysis of the package shows that intermediate `__init__.py` files (audio, audio/tasks, etc.) contain only docstrings and no eager imports. Task API modules import lightweight data structures and decorators. Heavy ML backends (speechbrain, pyannote, transformers pipelines) are deferred until function call time via `@requires_compatibility` decorators and try/except blocks.

However, the base dependency chain is inherently heavy:
- `senselab/__init__.py` imports `nest_asyncio` and sets up async event loop
- `audio/data_structures/audio.py` imports `torch`, `numpy`, `pydantic` at module level
- `utils/data_structures/__init__.py` eagerly imports all model classes, `DeviceType`, `Language`, etc.
- Any senselab import that touches `Audio` or model types transitively loads `torch`

**Alternatives considered**: None — this is observational research, not a design decision.

## Profiling Approach

### Decision: Use Python `-X importtime` for dependency tracing + `time.perf_counter` for wall-clock

**Rationale**: Python 3.7+ provides `-X importtime` which prints a tree of every `import` call with cumulative and self time in microseconds. This gives the transitive dependency breakdown (FR-004) for free. Combined with wall-clock `time.perf_counter` wrapping the actual import statement in a subprocess, we get both the total time and the internal breakdown.

**Alternatives considered**:
- `importlib` hooks: More complex, same information as `-X importtime`
- `cProfile`: Too much noise from non-import code
- `sys.settrace`: Overhead distorts measurements

## Subprocess Isolation Strategy

### Decision: Each import timed in its own subprocess via `subprocess.run`

**Rationale**: Python caches imported modules in `sys.modules`. Once `torch` is imported, all subsequent imports that depend on torch appear near-zero. To get true cold-start times (FR-001, FR-007), each import must run in a fresh Python process. The subprocess approach also naturally handles failed imports (FR-006) — a crash in one subprocess doesn't affect others.

**Alternatives considered**:
- `importlib.reload()`: Unreliable for C extensions, doesn't clear all state
- Forking: macOS ARM64 has issues with fork + torch (spawn is required)

## Tutorial Import Extraction

### Decision: Parse notebook JSON to extract import lines

**Rationale**: Jupyter notebooks are JSON files. Code cells can be extracted by filtering for `cell_type == "code"` and scanning source lines for `import` or `from ... import` patterns. This is more reliable than regex on raw JSON.

**Alternatives considered**:
- `nbformat` library: Adds a dependency; raw JSON parsing is sufficient
- Manual listing: Error-prone as tutorials change

## Colab-Specific Imports

### Decision: Skip `google.colab` imports, record as "skipped (platform-specific)"

**Rationale**: Colab imports (`google.colab.userdata`, `google.colab.output`) are unavailable outside Colab. Attempting to import them wastes time on expected failures. They should be excluded from timing and noted in the report.

**Alternatives considered**:
- Mock the google.colab module: Adds complexity, doesn't represent real import time
- Install google-colab package locally: Heavy, unnecessary

## Report Format

### Decision: Markdown report with three sections — ranked imports, per-tutorial summary, dependency breakdown

**Rationale**: Markdown is readable in terminals, GitHub, and editors. Three sections map directly to the three user stories (P1: ranked imports, P3: per-tutorial summary, P2: dependency breakdown for flagged bottlenecks).

**Alternatives considered**:
- JSON output: Machine-readable but harder to scan visually
- CSV: Good for spreadsheets but loses the hierarchical dependency tree
- Both JSON and Markdown could be produced; Markdown is the primary deliverable

## Key Import Chains Identified

From codebase analysis, the following import chains are expected bottlenecks:

1. **torch ecosystem**: `torch` → `torchvision` → `torchaudio` — typically 3-8 seconds combined cold start
2. **transformers**: `transformers` pulls in tokenizers, safetensors, huggingface_hub — typically 2-4 seconds
3. **speechbrain**: imports torch + its own ecosystem — typically 2-5 seconds
4. **pyannote-audio**: imports torch + pyannote.core + pyannote.pipeline — typically 2-4 seconds
5. **Audio data structure**: Any `from senselab.audio.data_structures import Audio` transitively loads torch, numpy, pydantic
6. **opensmile**: Native bindings may have initialization cost
7. **sklearn/umap**: Scientific computing stack, moderate import time

## Distinct Imports Across Tutorials (Deduplicated)

From analysis of 20 notebooks with imports, approximately 80+ distinct import lines exist, mapping to ~50 distinct top-level packages. The profiling script will extract these dynamically rather than hardcoding.
