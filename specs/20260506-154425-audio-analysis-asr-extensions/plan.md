# Implementation Plan: Audio Analysis Script + ASR Backend Extensions + Forced Alignment

**Branch**: `20260506-154425-audio-analysis-asr-extensions` | **Date**: 2026-05-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/20260506-154425-audio-analysis-asr-extensions/spec.md`

## Summary

Three deliverables on one branch:

1. **`scripts/analyze_audio.py`** — a self-contained ~1,100-LOC dev script that runs senselab's full task suite on an input audio file (with and without enhancement), supports multiple models per task, drives AST and YAMNet at their native temporal resolutions, persists a content-addressable cache keyed by `sha256(audio, task, model, params, wrapper, senselab_ver)` with full provenance, **auto-pairs every timestamp-less ASR with the new multilingual aligner**, and emits a hierarchical Label Studio import bundle.

2. **Multilingual forced-alignment extension** in `src/senselab/audio/tasks/forced_alignment/` — adds an MMS-based aligner backend (`facebook/mms-1b-all` with per-language adapters) under a new `model_type = "mms"` branch, plus a language→backend registry so `align_transcriptions(audio, transcript, language=...)` auto-picks the right backend for any of 1100+ languages. Optional `uroman` dep for ja/zh romanization.

3. **Three new ASR paths** under `src/senselab/audio/tasks/speech_to_text/`:
   - **IBM Granite Speech 3.3** (small): modify `huggingface.py` to accept `return_timestamps: bool` so timestamp-less HF ASR models can run without the pipeline's safety check rejecting them.
   - **NVIDIA Canary-Qwen 2.5B** (medium): new `canary_qwen.py` module + new `nemo-canary-qwen` subprocess venv that loads NeMo's `SALM` class.
   - **Alibaba Qwen3-ASR 1.7B** (medium): new `qwen.py` module + new `qwen-asr` subprocess venv using Alibaba's `qwen-asr` package with its bundled companion forced-aligner.

The dispatch in `audio/tasks/speech_to_text/api.py` gains two new prefix groups (Canary, Qwen-ASR) and a known-list of timestamp-less HF models that triggers `return_timestamps=False`.

The script's auto-align step runs AFTER the ASR family completes and is a no-op for ASR results that already include timestamps (Whisper, NeMo CTC, Qwen3-with-companion). Failed alignments preserve the ASR text and don't fail the parent ASR.

## Technical Context

**Language/Version**: Python 3.11–3.14 (managed via uv) — matches senselab's `requires-python`.

**Primary Dependencies**:
- Script: stdlib only (`hashlib`, `json`, `subprocess`, `argparse`, `tempfile`) plus `torch` (already a senselab base dep).
- senselab forced_alignment extension: existing `transformers` (already pinned `>=5.0`); new optional `uroman` (MIT, pure-Python) for ja/zh romanization, added to the `[nlp]` extra.
- MMS aligner weights: `facebook/mms-1b-all` (CC-BY-NC 4.0) downloaded lazily on first use via HF Hub.
- Canary-Qwen subprocess venv: `nemo_toolkit[asr,tts]` (NeMo trunk; pinned ref TBD at implementation time).
- Qwen3-ASR subprocess venv: `qwen-asr` (Alibaba) with its companion `Qwen/Qwen3-ForcedAligner-0.6B`.

No new top-level senselab deps in the default `uv sync`. Subprocess venvs are provisioned lazily.

**Storage**: File-based — JSON outputs under `artifacts/analyze_audio/`; persistent cache under `artifacts/analyze_audio_cache/`; subprocess venvs under `~/.cache/senselab/venvs/{nemo-canary-qwen,qwen-asr}/`.

**Testing**: pytest with `@pytest.mark.skipif(...)` guards.

**Target Platform**: macOS ARM64 (developer workstation), Linux (CI). MMS, Canary-Qwen, and Qwen3-ASR run best on CUDA; CPU/MPS fallback is supported but slower.

**Project Type**: Mixed — a developer script + senselab library extensions.

**Performance Goals**: re-runs hit the cache and complete in seconds; first-time provisioning of a new subprocess venv is bounded by network speed; alignment cache hits keep iteration cycles short even when the user is just tuning the aligner.

**Constraints**:
- Default `uv sync` and existing CI suites MUST be unaffected.
- senselab's existing dispatch order (HF pipeline → NeMo) MUST be preserved; new prefixes are checked BEFORE the catch-all HF default.
- MMS adapter loading MUST happen inside the existing `align_transcriptions` flow without breaking its public API (existing callers of `align_transcriptions` keep working).
- The ASR-step and alignment-step caches MUST be independent (FR-024).

**Scale/Scope**: ~1,100-LOC script + ~150-LOC senselab MMS extension + ~150-LOC each for the two new ASR backends + small `huggingface.py` patch + tests + docs.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. UV-Managed Python | PASS | Script invoked via `uv run`; subprocess venvs use senselab's existing `venv_python()` helper |
| II. Encapsulated Testing | PASS | All new tests run under `uv run pytest` with `skipif` guards for absent venvs/models |
| III. Commit Early and Often | PASS | Plan splits into discrete commits: script + cache; MMS extension; Granite-via-HF tweak; Canary-Qwen backend; Qwen3-ASR backend |
| IV. CI Must Stay Green | PASS | All new venvs are opt-in (provisioned lazily); MMS weights are downloaded lazily; CI unchanged |
| V. Memory-Driven Anti-Pattern Avoidance | PASS | Reuses existing subprocess-venv pattern; no mocking of caches; no print debugging in source code |
| VI. No Unnecessary API Calls | PASS | Cache layer minimizes redundant model loads; HF Hub model downloads governed by HF's own cache |
| VII. Simplicity First | PASS | MMS slots into the existing CTC code via a new `model_type` branch — no architectural rewrite. New ASR backends mirror the existing NeMo pattern |
| VIII. No Hardcoded Parameters | PASS | All script behavior is CLI-flag-driven with sensible defaults; aligner model id is configurable via `--aligner-model`; language can be auto-detected or passed explicitly |

**Post-Phase 1 Re-check**: All gates still pass. The design adds three thin senselab modules (one extension to `forced_alignment`, two new `speech_to_text` backends) plus one small patch to `huggingface.py` plus the script. Nothing modifies existing senselab runtime behavior for users who don't invoke the new code paths.

## Project Structure

### Documentation (this feature)

```text
specs/20260506-154425-audio-analysis-asr-extensions/
├── plan.md                          # This file
├── spec.md                          # Feature specification
├── research.md                      # Phase 0 research
├── data-model.md                    # Phase 1 data model
├── quickstart.md                    # Phase 1 quickstart
├── contracts/
│   └── cli.md                       # CLI contract for the script
└── checklists/
    └── requirements.md              # Spec quality checklist
```

### Source Code (repository root)

```text
scripts/
└── analyze_audio.py                 # NEW (mostly written; needs --no-align-asr, --aligner-model, --asr-language flags + auto-align stage)

src/senselab/audio/tasks/forced_alignment/
├── forced_alignment.py              # MODIFIED: add `mms` model_type branch in `_get_prediction_matrix` + adapter loading in `align_transcriptions`
├── constants.py                     # MODIFIED: extend DEFAULT_ALIGN_MODELS_HF with MMS entries; add ISO-2 → ISO-3 map; add language→backend registry
├── data_structures.py               # UNCHANGED
└── __init__.py                      # UNCHANGED (existing public API stays)

src/senselab/audio/tasks/speech_to_text/
├── api.py                           # MODIFIED: add `_CANARY_PREFIXES` and `_QWEN_ASR_PREFIXES`; add timestamp-less HF model known-list; new dispatch branches
├── huggingface.py                   # MODIFIED: accept `return_timestamps: bool` kwarg; pass through to pipeline()
├── nemo.py                          # UNCHANGED
├── canary_qwen.py                   # NEW: Canary-Qwen subprocess-venv backend
└── qwen.py                          # NEW: Qwen3-ASR subprocess-venv backend

src/tests/audio/tasks/forced_alignment/
└── mms_test.py                      # NEW: skipif-guarded smoke test

src/tests/audio/tasks/speech_to_text/
├── canary_qwen_test.py              # NEW: skipif-guarded smoke test
├── qwen_test.py                     # NEW: skipif-guarded smoke test
└── huggingface_no_timestamps_test.py # NEW: verifies the return_timestamps=False path

src/tests/scripts/
└── analyze_audio_test.py            # NEW: argparse + cache-key + auto-align skip-condition tests (no model loads)

pyproject.toml                       # MODIFIED: add `uroman` to the `nlp` extra
```

**Structure Decision**: senselab's three new code paths (MMS, Canary, Qwen) each live in their own module to keep imports lazy. The script lives entirely under `scripts/` and uses senselab's public APIs only.

## Implementation Design

### Component 1: senselab `forced_alignment` extension — MMS backend

`src/senselab/audio/tasks/forced_alignment/constants.py` gets:

- An ISO-639-1 → ISO-639-3 map (for MMS adapter selection).
- A new `MMS_MODEL_ID = "facebook/mms-1b-all"` constant.
- A small set of language-specific overrides (e.g., default `iso3="jpn"` for `language="ja"`, `romanize=True` for ja/zh).
- A language→backend registry: any language not in `DEFAULT_ALIGN_MODELS_HF` falls through to MMS.

`forced_alignment.py` changes:

- `align_transcriptions(audios, transcripts, language=...)` (existing signature) — extended internally to consult the registry. The public API stays unchanged.
- After `from_pretrained`, when `model_type == "mms"`, also call `processor.tokenizer.set_target_lang(iso3)` and `model.load_adapter(iso3)`. Cache the loaded model+adapter pair keyed by (model_id, iso3).
- `_preprocess_segments` (existing) gains an optional `romanize` path that calls `uroman` (lazily imported) for ja/zh transcripts before tokenization. Without `uroman` installed, alignment for those languages errors with a clear message.
- The CTC trellis/backtrack code (`_get_trellis`, `_backtrack`, `_merge_repeats`, `_assign_timestamps`) is reused unchanged — MMS produces compatible CTC posteriors.

Output shape stays exactly as today: `List[List[ScriptLine | None]]`.

### Component 2: senselab `huggingface.py` — `return_timestamps` parameter

In `src/senselab/audio/tasks/speech_to_text/huggingface.py`:

- `transcribe_with_huggingface(audios, model, ..., return_timestamps: bool = True)` — new keyword.
- When `return_timestamps=False`, the call to `transformers.pipeline(...)` omits the `return_timestamps` keyword (or passes `False`); the pipeline returns text-only.
- The senselab dispatcher (`api.py`) detects timestamp-less models via a known-list (e.g., `_TIMESTAMP_LESS_HF_MODELS = ("ibm-granite/granite-speech-",)`) and defaults `return_timestamps=False` for those. End users can override by passing `return_timestamps=True` explicitly (and accept the pipeline error).

This unblocks Granite Speech 3.3 without a new module.

### Component 3: senselab `canary_qwen.py` backend

New module mirroring `nemo.py`:

- Constants: `_VENV_NAME = "nemo-canary-qwen"`, requirements `nemo_toolkit[asr,tts]` from a NeMo trunk ref.
- `transcribe_with_canary_qwen(audios, model, language, device)` — serializes audios to 16 kHz WAVs in a tempdir, spawns the worker subprocess in the dedicated venv.
- Worker imports `from nemo.collections.speechlm2.models import SALM`, calls `model.generate(prompts=[...], audio=[paths])`, decodes via `model.tokenizer.ids_to_text(ids)`, returns text.
- senselab parent process reassembles into `[ScriptLine(text=..., start=None, end=None, speaker=None)]` per audio (text-only — model has no native timestamps).

The script's auto-align stage adds timestamps via MMS.

### Component 4: senselab `qwen.py` backend

New module:

- Constants: `_VENV_NAME = "qwen-asr"`, requirements include the `qwen-asr` package and pin to a known-good version.
- `transcribe_with_qwen(audios, model, language, device, *, return_timestamps: bool = True)`:
  - Worker imports `from qwen_asr import Qwen3ASRModel, AlignerModel`.
  - When `return_timestamps=True` (default), loads the companion `Qwen/Qwen3-ForcedAligner-0.6B` and calls `model.transcribe(..., forced_aligner=aligner, return_time_stamps=True)`.
  - Maps per-word `start_time/end_time` into `ScriptLine.chunks`; the parent `ScriptLine` carries the full transcript.
- The script's auto-align stage skips Qwen3-ASR output (already timestamped).

### Component 5: senselab `api.py` dispatch extension

```python
_CANARY_PREFIXES = ("nvidia/canary-",)
_QWEN_ASR_PREFIXES = ("Qwen/Qwen3-ASR",)
_TIMESTAMP_LESS_HF_MODELS = ("ibm-granite/granite-speech-",)
```

Dispatch order (existing → new):

1. NeMo ASR via `_NEMO_PREFIXES` → `nemo.py`.
2. Canary-Qwen via `_CANARY_PREFIXES` → `canary_qwen.py` (new).
3. Qwen-ASR via `_QWEN_ASR_PREFIXES` → `qwen.py` (new).
4. Default: HF pipeline. If model id matches `_TIMESTAMP_LESS_HF_MODELS`, default `return_timestamps=False`.

The four prefix groups are disjoint.

### Component 6: Script — `analyze_audio.py` updates

The script is largely written. New work:

- Add `--no-align-asr`, `--aligner-model`, `--asr-language`, `--qwen-asr-no-timestamps` flags.
- Add the auto-align stage AFTER the ASR family runs:
  - Iterate each ASR ModelRun.
  - Skip if `--no-align-asr` set.
  - Skip if the ASR result already has timestamps (`start` is not None on any ScriptLine, or `chunks` is non-empty).
  - Otherwise: compute the alignment cache key `sha256(audio_sig || "alignment" || aligner_model_id || aligner_params || transcript_sha || language || wrapper || senselab_ver)`. Cache lookup, then `senselab.audio.tasks.forced_alignment.align_transcriptions(audio, transcript, language)` on miss. Persist to its own JSON under `<pass>/alignment/<sanitized_model_id>.json`.
  - Failure: preserve ASR text, mark alignment failed in its own JSON, no entry written to alignment cache (so a future fix triggers re-run).
- Update the LS export so:
  - For ASR runs WITH native timestamps (or with a successful alignment): per-segment regions on the timeline.
  - For ASR runs without timestamps and no successful alignment: a single full-audio TextArea region.
- Bump `_CACHE_SCHEMA_VERSION` from 1 to 2.

### Component 7: Tests

- `analyze_audio_test.py`: parses defaults, audio_signature stability, cache_key changes when params change, auto-align skip-condition logic (mock ScriptLine with/without start/end), LS export for the alignment-failed case. No model loads.
- `mms_test.py` (forced_alignment): skipif when MMS weights aren't downloaded; else runs alignment on a synthetic en + ja transcript and asserts output shape.
- `canary_qwen_test.py`: skipif venv absent; else runs on synthetic audio, asserts `[ScriptLine(text=...)]` shape.
- `qwen_test.py`: skipif venv absent; else runs with and without `return_time_stamps`, asserts shapes.
- `huggingface_no_timestamps_test.py`: verifies `transcribe_with_huggingface(audios, model, return_timestamps=False)` runs Granite-style models without the pipeline's timestamp-safety error.

### Component 8: Documentation

- Update `CLAUDE.md` with a one-paragraph pointer to `scripts/analyze_audio.py` and the new aligner/ASR backends.
- Spec docs (this directory) are workspace-only.
- `quickstart.md` includes the runnable example using a tutorial audio file.

## Out of Scope

- Promotion of the analysis script to a `senselab.workflows.analyze_audio` API.
- Streaming ASR / live-microphone mode.
- Multi-file batch driver.
- Auto-resampling beyond 16 kHz mono.
- Cross-machine cache sharing.
- Replacing the existing English-aligner default with MMS (English keeps using the existing per-language wav2vec2; MMS is the fallback for non-English).
- Forced alignment for languages NOT covered by MMS (1100+ is enough for the in-scope ASR models).

## Complexity Tracking

No constitution violations to justify. Each new piece has a clear separation of concern:

| Component | Why a separate module rather than inline? |
|-----------|------------------------------------------|
| MMS extension to `forced_alignment` | Adds one new `model_type` branch; no new module; reuses CTC trellis code. Lowest possible scope expansion. |
| `canary_qwen.py` | Pinned NeMo trunk + speechlm2 imports must not leak into the senselab main process; separation makes the subprocess-only nature explicit. |
| `qwen.py` | Same rationale as Canary; `qwen-asr` and forced-aligner deps live in their own venv. |
| `huggingface.py` patch | One new keyword; the simplest possible change to unblock Granite. |
