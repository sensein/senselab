# Data Model: Audio Analysis Script + ASR Backend Extensions + Forced Alignment

**Branch**: `20260506-154425-audio-analysis-asr-extensions` | **Date**: 2026-05-06

## Entities

### AudioVariant

| Field | Description |
|-------|-------------|
| label | One of `raw_16k`, `enhanced_16k` |
| audio | Senselab `Audio` object (16 kHz mono after preprocessing) |
| audio_signature | sha256 of (sampling_rate, shape, PCM bytes) |
| duration_s | length in seconds |
| derivation | For `enhanced_16k`: the enhancement model id used; otherwise null |

### Task

| Field | Description |
|-------|-------------|
| name | One of `diarization`, `ast`, `yamnet`, `features`, `asr`, `embeddings`, `alignment` (new) |
| accepts_models_list | Bool: does this task vary by model? |

### ModelRun

The atomic unit produced by every task call.

| Field | Description |
|-------|-------------|
| pass_label | `raw_16k` or `enhanced_16k` |
| task | Reference to a Task |
| model_id | Model identifier string, or null for `features` |
| status | `ok`, `failed`, or `skipped` |
| elapsed_s | Wall-clock seconds spent on the call (0 on cache hit) |
| result | Serializable task output |
| error | Error repr() string when `status == "failed"` |
| traceback | Truncated Python traceback when `status == "failed"` |
| cache | One of `hit`, `miss`, `disabled` |
| cache_key | sha256 hex digest used as the cache key |
| provenance | Provenance block (see below) |

### AlignmentRun (new)

A specialization of ModelRun for the alignment step that always follows a timestamp-less ASR.

| Field | Description |
|-------|-------------|
| parent_asr_cache_key | The `cache_key` of the ASR ModelRun whose text this alignment is timing |
| transcript_sha | `sha256(asr_text)` — anchors the alignment to the exact text it aligned |
| language | ISO 639-1 / 639-3 language string used for aligner backend selection |
| aligner_model_id | The aligner backend's model id (e.g., `facebook/mms-1b-all`) |
| aligner_params | Dict of aligner-specific parameters (target_lang adapter id, romanize flag, etc.) |
| (all other ModelRun fields) | Apply unchanged |

The alignment cache entry is independent from the parent ASR's cache entry — re-running the script can hit the ASR cache while missing the alignment cache, and vice versa.

### Provenance

Embedded in every non-cached ModelRun output.

| Field | Description |
|-------|-------------|
| task | Task name (`asr`, `alignment`, `diarization`, ...) |
| model_id | Model id string, or null for `features` |
| params | Dict of task-specific parameters |
| audio_signature | sha256 of the post-preprocessing waveform |
| audio_source | Absolute filesystem path to the original input file |
| pass | `raw_16k` or `enhanced_16k` |
| device | Device label resolved at run time |
| wrapper_version_hash | sha256 of `scripts/analyze_audio.py` source bytes |
| senselab_version | Installed senselab version |
| cache_schema_version | `_CACHE_SCHEMA_VERSION = 2` |
| timestamp_utc | ISO-8601 timestamp of the original (non-cached) computation |

For an AlignmentRun, the provenance additionally includes:

| Field | Description |
|-------|-------------|
| transcript_sha | sha256 of the input transcript text |
| language | Language code used to select the aligner backend |
| parent_asr_cache_key | Backlink to the ASR run that produced the transcript |

### CacheEntry

Persisted on disk under `<cache-dir>/<cache_key>.json`. Equivalent to a serialized ModelRun (or AlignmentRun).

### LabelStudioBundle

| Field | Description |
|-------|-------------|
| tasks_json_path | `<run-dir>/labelstudio_tasks.json` |
| config_xml_path | `<run-dir>/labelstudio_config.xml` |
| tasks | List of `{data: {...}, predictions: [{result: [regions]}]}` |
| labeling_config_xml | Auto-generated XML defining the timeline tracks |

For ASR ModelRuns:
- If the run has timed segments (Whisper, NeMo Conformer CTC, Qwen3-ASR-with-companion-aligner, OR a successful AlignmentRun followed a text-only ASR): emit per-segment regions on the timeline.
- If the run has only text and no AlignmentRun (alignment skipped via `--no-align-asr`, or alignment failed): emit a single full-audio TextArea region with the full transcript.

### ASR Backend Registry (senselab-side)

| Backend | Prefix(es) / Trigger | Module | Venv name | Native timestamps | Status pre-feature |
|---------|----------------------|--------|-----------|-------------------|--------------------|
| HuggingFace pipeline (Whisper, CTC) | (default) | `huggingface.py` | senselab main | yes (Whisper); CTC native | works |
| HuggingFace pipeline (timestamp-less, e.g., Granite Speech) | known-list (`ibm-granite/granite-speech-*`) → `return_timestamps=False` | `huggingface.py` (modified) | senselab main | no — paired with auto-align | timestamp request rejected; FIXED by this feature |
| NeMo ASR | `nvidia/stt_*`, `nvidia/conformer*` | `nemo.py` | `nemo-diarization` (shared) | yes (CTC) | works |
| **NeMo Canary-Qwen** (new) | `nvidia/canary-*` | `canary_qwen.py` (new) | `nemo-canary-qwen` (new) | no — paired with auto-align | not supported |
| **Alibaba Qwen ASR** (new) | `Qwen/Qwen3-ASR*` | `qwen.py` (new) | `qwen-asr` (new) | yes (via internal companion aligner) | not supported |

### Aligner Backend Registry (senselab-side)

| Backend | model_type | Module location | Languages | Notes |
|---------|------------|-----------------|-----------|-------|
| WhisperX-style wav2vec2 CTC | `huggingface` | existing `forced_alignment.py` | ~30 (existing dict in `constants.py`) | per-language CTC model |
| torchaudio bundle | `torchaudio` | existing `forced_alignment.py` | English | unchanged |
| **MMS via HF adapters** (new) | `mms` | extension to existing module | 1100+ via `facebook/mms-1b-all`'s per-language adapters | default for any language not in the `huggingface` dict |

The dispatcher in `align_transcriptions(audio, transcript, language=...)` consults a language→backend registry. For ja/zh, an additional `romanize: bool` flag triggers `uroman` preprocessing before the aligner sees the transcript.

## Relationships

```
AudioVariant 1──* ModelRun
ModelRun (asr, no native timestamps) 1──0..1 AlignmentRun
ModelRun (asr, native timestamps) 1──0 AlignmentRun  (auto-align skipped)
AlignmentRun.parent_asr_cache_key references ModelRun.cache_key
ModelRun 1──1 CacheEntry  (when status==ok and cache enabled)
ModelRun 1──1 Provenance
LabelStudioBundle aggregates ModelRuns + AlignmentRuns from one run
```

## State Transitions

ASR step:

```
not_started → cache_lookup → cache_hit (status=ok, cache=hit, elapsed=0)
                          → cache_miss → running → ok      (cache=miss, store)
                                                 → failed  (cache=miss, NOT stored)
```

Alignment step (when ASR succeeded and the result has no native timestamps and `--no-align-asr` was not set):

```
asr_text_obtained → align_cache_lookup → align_cache_hit  (status=ok, cache=hit, elapsed=0)
                                       → align_cache_miss → running → ok      (cache=miss, store)
                                                                    → failed  (cache=miss, NOT stored)
```

Alignment failure does NOT mark the parent ASR as failed. The ASR status stays `ok`; the alignment status is `failed` independently.

## Validation Rules

- `audio_signature` is computed once per AudioVariant and reused for every ModelRun and AlignmentRun in that pass.
- `transcript_sha` for an AlignmentRun MUST be `sha256(parent_asr.result.text)`.
- `parent_asr_cache_key` MUST resolve to a `status=ok` ASR ModelRun in the same run; alignment is not attempted for failed ASR.
- An AlignmentRun's cache_key is independent of the ASR's cache_key — they share `audio_signature` but have different `task` strings ("asr" vs "alignment") and different `params` so the keys naturally diverge.
- LabelStudioBundle drops ModelRuns with `status != ok` from its regions but keeps a top-level summary entry; AlignmentRuns with `status == failed` cause the parent ASR's region to fall back to a single full-audio TextArea.
- For the script's auto-align step: it is a no-op for ASR results that already include timestamps (Whisper, CTC ASR, Qwen3-ASR with companion aligner). Detection: if `result[0].start is not None` (or `chunks` is non-empty), skip alignment.
- For ASR backends, dispatch order is: NeMo-ASR → Canary-Qwen → Qwen-ASR → HF-pipeline (default). The four prefix groups are disjoint.
