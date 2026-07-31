# Feature Specification: Audio Analysis Script + ASR Backend Extensions

**Feature Branch**: `20260506-154425-audio-analysis-asr-extensions`
**Created**: 2026-05-06
**Status**: Draft
**Input**: User description: "Add a general-purpose audio analysis script that runs senselab's full task suite on an input file with and without speech enhancement, with multiple models per task, native temporal precision per scene-classification model, content-addressable cache + provenance, and a Label Studio export. Additionally extend senselab's ASR dispatch for two model families that today fail through the HuggingFace pipeline path: NVIDIA Canary-Qwen 2.5B (NeMo-native; needs NeMo subprocess-venv router extension for the nvidia/canary-* prefix) and Alibaba Qwen3-ASR 1.7B (model_type qwen3_asr unrecognized by the installed transformers; needs transformers upgrade or a custom backend). Additionally extend senselab's `forced_alignment` module with multilingual aligner backends so that timestamp-less ASR models (including IBM Granite Speech 3.3) can be brought back into scope by post-processing their text output through forced alignment; the script will route any timestamp-less ASR through the unified aligner."

## Clarifications

### Session 2026-05-06

- Q: Forced-aligner scope — extend the existing module with new backends, only wire the existing aligner, or both? → A: Option C — extend `forced_alignment` with multilingual aligner backend(s) (e.g., MMS, Qwen3-ForcedAligner) under a unified API and auto-pick per language; wire all timestamp-less ASR models through it. Brings IBM Granite Speech 3.3 back into the supported set.
- Q: For the analysis script, should timestamp-less ASR output be auto-aligned by default? → A: Option A — auto-align by default; provide a single `--no-align-asr` flag to disable. Keeps the Label Studio export consistent across all ASR models without surprising the user.
- Q: Alignment-failure fallback behavior? → A: Option A — preserve ASR text, record alignment failure separately, fall back to a single full-audio TextArea region in the LS export. Constraint: provenance and caching MUST be captured separately for ASR and alignment so that alignment can be re-run independently (e.g., after a senselab upgrade or aligner fix) without re-running the slow ASR call.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One-Command End-to-End Audio Analysis (Priority: P1)

As a senselab developer or researcher, I want to point a single command at any audio file and get a full multi-task analysis (diarization, audio scene classification, audio features, ASR, speaker embeddings) on both the original and the speech-enhanced versions, so that I can compare model outputs across the same recording without orchestrating each task by hand.

**Why this priority**: Today every multi-task investigation in senselab is bespoke — researchers wire up the same six task calls themselves and re-derive the same boilerplate. A single canonical entry point removes that friction and enables consistent comparisons across files and across analyses.

**Independent Test**: Run the new script on a sample audio file and verify all six task families produced output JSON files for both the resampled-original and the speech-enhanced versions (where each task's model is supported). Each output records which model produced it, which parameters were used, and how long it took.

**Acceptance Scenarios**:

1. **Given** a `.wav` file at any sampling rate, **When** the script runs end-to-end, **Then** the file is downmixed to mono, resampled to 16 kHz, and every task family produces at least one output JSON for both the resampled and the enhanced audio passes.
2. **Given** a per-model failure (e.g., a model id transformers can't load), **When** the script proceeds, **Then** that one task-model combination's output JSON contains a structured failure record (status, error message, traceback) but the rest of the run completes without aborting.
3. **Given** a successful run, **When** the resulting summary JSON is opened, **Then** it lists every task-model combination tried, their status, and the cache state (hit/miss/disabled) per combination.

---

### User Story 2 - Re-Run Without Re-Computation (Priority: P1)

As a researcher iterating on parameters, I want re-running the script on the same audio with the same settings to skip the model calls and replay prior outputs, so that incremental changes (adjusting one model, adding one task) don't force me to wait for the entire pipeline to re-run from scratch.

**Why this priority**: Models in the pipeline take minutes to load and run; without caching, every iteration restarts from cold and discourages exploration.

**Independent Test**: Run the script twice on the same audio with identical parameters; the second run reports cache hits for every task-model combination that succeeded the first time, and total wall-clock time on the second run is dominated by I/O rather than model inference.

**Acceptance Scenarios**:

1. **Given** the script has run successfully against an audio file, **When** the same script is re-invoked with the same arguments, **Then** every successful task reports a cache hit and skips the model call entirely.
2. **Given** a cached result, **When** the developer changes one parameter (e.g., a model id, the AST window length), **Then** only the affected task-model combinations re-run; unaffected ones still hit the cache.
3. **Given** a cached result, **When** the audio file is modified (a different file at the same path, or trimmed differently), **Then** the cache invalidates for that audio's entries and re-computes them.
4. **Given** any cached output, **When** a developer inspects it, **Then** the JSON includes a provenance block recording the audio signature, the task and model id, the parameter values, the device used, the wrapper-version hash, the senselab version, and the timestamp of the original computation.

---

### User Story 3 - Hierarchical Annotations Imported into Label Studio (Priority: P2)

As an annotator or analyst, I want the script to emit a Label Studio-compatible bundle of annotations so that I can import the analysis directly into a labeling project and compare model outputs visually on the audio timeline.

**Why this priority**: Label Studio is the project's primary annotation surface; producing import-ready JSON closes the loop between automated analysis and human review.

**Independent Test**: Import the script's `labelstudio_tasks.json` and the matching `labelstudio_config.xml` into a Label Studio project and verify each model's annotations appear as a separate timeline track.

**Acceptance Scenarios**:

1. **Given** a successful run, **When** the developer imports the produced LS tasks JSON, **Then** Label Studio shows one task per audio variant (e.g., raw and enhanced) with multiple parallel timeline tracks (one per analyzer-model combination).
2. **Given** the produced labeling-config XML, **When** it is pasted into the LS project's labeling configuration, **Then** the labels and tracks referenced by the import resolve correctly, with each model's output rendered as its own row.
3. **Given** scene-classification predictions from two models that share a window/hop grid, **When** they are visualized in LS, **Then** their region boundaries align exactly so disagreements are visible at a glance.

---

### User Story 4 - Use NVIDIA Canary-Qwen 2.5B as a Senselab ASR Model (Priority: P2)

As a senselab user evaluating top-tier ASR accuracy, I want to pass `nvidia/canary-qwen-2.5b` as a model id to senselab's transcribe function and have it run successfully without me orchestrating NeMo separately.

**Why this priority**: Canary-Qwen currently leads several public ASR benchmarks; senselab today fails to run it because the only NeMo-router prefixes recognized are `nvidia/stt_*` and `nvidia/conformer*`. Extending the router pulls a state-of-the-art model into senselab's normal API surface without forcing every consumer to write custom NeMo wrappers.

**Independent Test**: Pass `nvidia/canary-qwen-2.5b` to senselab's transcribe API on a short audio file and receive timed transcription output, with the underlying model executing through senselab's NeMo subprocess venv.

**Acceptance Scenarios**:

1. **Given** an audio file and senselab's transcribe API, **When** the model id `nvidia/canary-qwen-2.5b` is supplied, **Then** the dispatcher routes the call to the NeMo subprocess venv (rather than failing inside the HuggingFace pipeline path).
2. **Given** the NeMo subprocess venv is already prepared on the host, **When** Canary-Qwen runs, **Then** the user receives a transcription with start/end timestamps in the same shape as Whisper output (so downstream code that consumes ScriptLines does not need model-specific branches).
3. **Given** the NeMo venv has not been prepared yet, **When** Canary-Qwen is invoked for the first time, **Then** senselab transparently provisions the venv (mirroring the existing Sortformer/YAMNet pattern) and proceeds.

---

### User Story 6 - Forced-Alignment Backends Cover Multilingual ASR Output (Priority: P2)

As a senselab user running an ASR model that does not natively produce timestamps (e.g., IBM Granite Speech 3.3, NVIDIA Canary-Qwen), I want senselab's `forced_alignment` module to be able to add per-segment (and ideally per-word) timestamps to the text output across the languages those models produce, so that timestamp-less ASR can still drive timeline annotation in Label Studio.

**Why this priority**: Without multilingual forced alignment, IBM Granite Speech 3.3 and NVIDIA Canary-Qwen are unusable for the script's primary use case. Bringing them back into the supported set is the difference between offering 1 working ASR option (Whisper) and offering 3+ working options.

**Independent Test**: Pass an audio file plus a plain-text transcript (in any of a target set of languages) to senselab's `align_transcriptions` API and receive timed segments back. The same API call should pick the right aligner backend for the supplied language without the user having to choose a model id.

**Acceptance Scenarios**:

1. **Given** an audio file and an English plain-text transcript, **When** the user calls `align_transcriptions(audio, transcript)`, **Then** they receive a list of timed segments (and word chunks where the underlying aligner supports it).
2. **Given** an audio file and a non-English transcript in a language supported by the new multilingual aligner backend, **When** the user calls the same API, **Then** they receive timed segments in the same shape, without changing the API surface.
3. **Given** the analysis script runs with a timestamp-less ASR (Canary-Qwen or Granite), **When** the script post-processes that output, **Then** the resulting Label Studio export contains region-level transcription annotations on the timeline (not a single full-audio TextArea).
4. **Given** an unsupported language or a transcript the aligner cannot align, **When** the script encounters it, **Then** the failure is captured per-model with a clear error and the rest of the run continues.

---

### User Story 5 - Use Alibaba Qwen3-ASR 1.7B as a Senselab ASR Model (Priority: P3)

As a senselab user wanting multilingual ASR coverage beyond Whisper, I want to pass `Qwen/Qwen3-ASR-1.7B` (or its current canonical id) as a model id and have it run successfully.

**Why this priority**: Qwen3-ASR claims competitive multilingual accuracy at a smaller parameter count than Whisper Large V3. It currently fails because the installed `transformers` does not register the `qwen3_asr` model_type. Adding a path for it broadens senselab's multilingual ASR options.

**Independent Test**: Pass `Qwen/Qwen3-ASR-1.7B` to senselab's transcribe API and receive transcription output, either via an upgraded `transformers` version or via a dedicated backend.

**Acceptance Scenarios**:

1. **Given** an audio file and senselab's transcribe API, **When** the model id `Qwen/Qwen3-ASR-1.7B` is supplied, **Then** the model loads and produces a transcription without requiring the user to pre-install Qwen-specific support.
2. **Given** Qwen3-ASR's output, **When** consumed downstream, **Then** the timing and text fields match the same shape that Whisper produces (so the analysis script and other consumers do not need a Qwen-specific branch).

---

### Edge Cases

- A model id is unrecognized by every backend → senselab raises a clear, actionable error rather than silently falling back to the wrong path.
- Cache directory exists but a specific cache file is corrupt or partially written → the affected cache lookup misses and the task re-runs; corrupt entries do not crash the script.
- The audio file is a stereo / non-16 kHz file → preprocessing converts it to 16 kHz mono before any task receives it; the audio signature in the cache key is computed from the post-preprocessing waveform so two physically different files that produce the same 16 kHz mono signal share cache entries.
- The user provides multiple ids of the same model with different revisions → the cache key includes any revision-affecting parameter, so each variant gets its own cache entry.
- The wrapper script source itself changes (logic refactor) → the wrapper-version hash in the cache key changes, invalidating prior entries.
- A senselab upgrade changes the output shape of a task → the senselab version field in the cache key changes, invalidating prior entries.
- AST is asked to operate on a window shorter than its native input requires → the failure is captured per-model with a clear error, the rest of the run continues, and the documentation explains the native window expectation.
- The user passes a model that doesn't natively produce timestamps (e.g., Granite Speech) → that model is documented as out of scope for this script; the existing senselab transcribe API still rejects it through the underlying pipeline's safety check.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a single command that accepts a path to an audio file and runs the full senselab analysis suite on it.
- **FR-002**: System MUST resample input audio to 16 kHz mono before invoking any task; downstream tasks MUST observe the same waveform.
- **FR-003**: System MUST run every task once on the resampled-original audio and once on the same audio after senselab's speech enhancement, when enhancement is enabled.
- **FR-004**: System MUST allow a list of models per task, where the registry offers more than one, and run the task once per model in that list.
- **FR-005**: System MUST default to ≥ 2 models per task when ≥ 2 are listed in `model_registry.yaml` and at least one is expected to succeed today.
- **FR-006**: System MUST persist each task-model outcome to its own JSON file under a per-run output directory and aggregate them in a top-level summary.
- **FR-007**: System MUST capture per-model failures (model not found, architecture unsupported, etc.) as structured records in the task-model JSON without aborting the rest of the run.
- **FR-008**: System MUST drive each scene-classification model at its own native temporal precision by default (AST and YAMNet receive different default windows). Per-model overrides MUST be available.
- **FR-009**: System MUST produce a content-addressable cache of task-model outcomes. Cache hits MUST replay the prior outcome verbatim and report `cache: hit` in the output without invoking the underlying model.
- **FR-010**: The cache key MUST be derived from a stable hash of (audio waveform signature, task name, model id, task parameters, wrapper-version hash, senselab version, cache schema version). Changing any of these MUST cause a cache miss for that one task-model entry; unrelated entries MUST still hit.
- **FR-011**: Each non-cached outcome MUST include a provenance record (audio source path, audio signature, task, model id, parameters, device, wrapper-version hash, senselab version, timestamp, pass label).
- **FR-012**: System MUST allow disabling the cache via a single flag, in which case lookups always miss and outcomes are not stored.
- **FR-013**: System MUST emit a Label Studio import bundle: a tasks JSON file containing per-audio-variant tasks with predictions structured as parallel timeline tracks (one per analyzer-model combination), and a labeling-configuration XML file matching the tracks the run produced.
- **FR-014**: When two scene-classification models share an identical window/hop grid, System MUST additionally emit a side-by-side agreement file pairing top-1 labels per shared window.
- **FR-015**: System MUST let the user select a compute device (cpu, cuda, mps, or auto) and propagate "auto" as a "task chooses its own compatible device" signal so that tasks with restrictive compatibility (e.g., diarization on macOS) fall back gracefully.
- **FR-016**: senselab MUST recognize NVIDIA Canary-* model ids as NeMo-backed and route their calls through the existing NeMo subprocess-venv mechanism rather than the HuggingFace pipeline path.
- **FR-017**: senselab MUST be able to run NVIDIA Canary-Qwen 2.5B end-to-end on an audio file and return a list of timed transcription segments compatible with senselab's existing ScriptLine output shape.
- **FR-018**: senselab MUST be able to run Alibaba Qwen3-ASR 1.7B end-to-end and return a list of timed transcription segments compatible with the existing ScriptLine shape, either by upgrading transformers to a version that registers `qwen3_asr` or by providing a dedicated backend.
- **FR-019**: System MUST handle absent or untracked optional dependencies (NeMo subprocess venv not yet provisioned, transformers version pinned without `qwen3_asr`) by surfacing actionable errors rather than silent failures.
- **FR-020**: senselab's `forced_alignment` module MUST accept (audio, transcript, language) and return timed segments, without the caller having to select a specific aligner-model id. The module MUST internally pick a compatible aligner backend based on the language.
- **FR-021**: senselab's `forced_alignment` module MUST support at least one multilingual aligner backend whose language coverage is a superset of the languages produced by the in-scope timestamp-less ASR models (IBM Granite Speech 3.3 supports English plus 7 translation languages; the multilingual aligner MUST cover all of those).
- **FR-022**: When a timestamp-less ASR backend (e.g., Canary-Qwen, Granite) is used, the system MUST be able to post-process the resulting text through `forced_alignment` and emit the result as a list of timed segments compatible with the existing ScriptLine output shape.
- **FR-023**: The analysis script MUST automatically pair every timestamp-less ASR output with `forced_alignment` post-processing by default, so that the Label Studio export contains region-level transcription annotations on the timeline. The auto-alignment behavior MUST be toggleable via the single CLI flag `--no-align-asr` (or equivalent), which when present skips alignment and emits text-only ScriptLines (single full-audio TextArea region in the LS export).
- **FR-024**: ASR steps and alignment steps MUST be cached independently. The ASR step's cache entry contains only the text output (and any native timestamps the ASR itself produced); the alignment step's cache entry contains the aligner backend's input (audio signature, transcript text, language) keyed parameters and the resulting timed segments. Re-running the script MUST be able to hit the ASR cache while missing the alignment cache, and vice versa, so that fixing or upgrading just one of the two does not force the other to re-run.
- **FR-025**: When auto-alignment is enabled and the alignment step fails (no aligner available for the language, transcript-vs-audio mismatch, aligner backend error, etc.), the system MUST preserve the ASR text result, mark only the alignment as failed (its own cache entry remains absent so a future fix triggers a re-run), and fall back to a single full-audio TextArea region for that model in the Label Studio export. The overall pass MUST continue with the rest of the tasks.
- **FR-026**: Each alignment outcome JSON MUST include its own provenance block (audio signature, transcript text or its hash, language, aligner backend id, params, wrapper-version hash, senselab version, timestamp), separate from the ASR provenance that produced the input transcript. The two MUST be linkable via the parent ASR's cache key so a reader can trace which ASR run an alignment came from.

### Key Entities

- **Audio Variant**: One of {raw 16 kHz mono, enhanced 16 kHz mono}. Each variant has its own audio signature and its own pass directory.
- **Task**: One of {diarization, audio scene classification, multi-backend feature extraction, ASR, speaker embeddings}.
- **Model Run**: One concrete execution of (task × model id × audio variant). Each Model Run has a status, elapsed time, optional result, optional error, cache state, cache key, and provenance block.
- **Cache Entry**: A persisted Model Run keyed by `sha256(audio signature, task, model id, params, wrapper hash, senselab version, schema version)`, stored under the cache directory.
- **Label Studio Bundle**: The pair (tasks JSON, labeling-config XML) ready for import into a Label Studio project.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new user can analyze a short audio file end-to-end with one command in under 5 minutes from a clean checkout (excluding first-time model downloads).
- **SC-002**: Re-running the same command on the same input is at least 100× faster than a first run, by virtue of cache hits replacing model invocations.
- **SC-003**: Every task-model outcome JSON includes a provenance block sufficient to reproduce that outcome from scratch (audio signature, model id, parameters, software versions).
- **SC-004**: A Label Studio project configured with the produced labeling-config XML successfully imports the produced tasks JSON without manual editing.
- **SC-005**: Running senselab's transcribe API with `nvidia/canary-qwen-2.5b` on an audio file produces a non-empty transcription on supported hardware (text-only ScriptLines — Canary-Qwen has no native timestamp output). The analyze_audio script's auto-align stage subsequently adds per-segment timestamps via the multilingual aligner; verifiable separately via SC-010.
- **SC-006**: Running senselab's transcribe API with `Qwen/Qwen3-ASR-1.7B` on an audio file produces a non-empty, timestamped transcription on supported hardware.
- **SC-007**: After this feature ships, a user comparing Whisper Large V3 Turbo, Canary-Qwen 2.5B, and Qwen3-ASR 1.7B on the same audio sees three distinct transcripts in the output without needing to manually configure any of the backends.
- **SC-008**: When the user changes only the model list, only model entries that differ from the cache invalidate; previously-cached unchanged entries still hit. Verifiable by per-entry `cache: hit/miss` reporting.
- **SC-009**: After this feature ships, `senselab.audio.tasks.forced_alignment.align_transcriptions(audio, transcript, language=…)` returns timed segments for English and for every language IBM Granite Speech 3.3 produces (English + 7 translation languages), without the caller having to know which aligner-model id is appropriate.
- **SC-010**: When the analysis script runs with a timestamp-less ASR (Canary-Qwen or Granite Speech), its Label Studio export contains region-level transcription annotations on the timeline (one region per aligned segment), not a single full-audio TextArea. Verifiable by inspecting the `result` array in `labelstudio_tasks.json` for that pass × model.
- **SC-011**: After a successful run, re-invoking the script with `--no-align-asr` removed (or with a different aligner backend) re-uses the ASR cache (no ASR call) and re-runs only the alignment step. Verifiable by per-step `cache: hit/miss` reporting on the ASR vs alignment outputs.
- **SC-012**: When the alignment step fails for a particular ASR output, the resulting JSON for that pass × ASR model still contains the ASR text and explicitly records the alignment-step failure with an actionable error message; the script's overall exit code remains success.

## Assumptions

- The script is a developer/research tool, not a packaged senselab API surface for external users. End-user-friendly polish (progress bars, web UI, etc.) is out of scope.
- The cache is local to the developer machine; no cross-machine sharing or sync is required for this feature.
- The audio signature is derived from the post-preprocessing PCM waveform, not the original file bytes. Two distinct files that resample to identical 16 kHz mono waveforms are treated as the same audio for caching purposes (this is a feature, not a bug — it captures the actual model inputs).
- Cache invalidation on senselab upgrades happens automatically via the senselab-version field in the cache key. Users who manage multiple senselab versions in parallel should not share a single cache directory; the feature does not attempt to detect such mixing.
- IBM Granite Speech 3.3, NVIDIA Canary-Qwen 2.5B, and any other timestamp-less ASR model are brought into scope via the multilingual `forced_alignment` extension (Story 6). Their ASR backends produce text-only ScriptLines; the script automatically post-processes those through the unified aligner to add per-segment timestamps before the Label Studio export.
- For Canary-Qwen, the assumed integration path is senselab's existing NeMo subprocess-venv pattern; the alternative (a transformers-compatible release of the model) is not assumed available.
- For Qwen3-ASR, the assumed integration path is the simpler one available at implementation time: either an upgraded `transformers` version that registers `qwen3_asr`, or — if that version is not yet released — a custom subprocess-venv backend mirroring the NeMo pattern. The choice between the two is a planning-phase decision.
- AST cannot operate on chunks shorter than ~1 second of audio (its internal kaldi-fbank requirement). The script's defaults respect this; users who pass shorter windows should expect a captured per-model failure rather than silent corruption.
