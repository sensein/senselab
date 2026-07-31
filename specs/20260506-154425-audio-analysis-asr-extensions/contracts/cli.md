# CLI Contract: scripts/analyze_audio.py

**Branch**: `20260506-154425-audio-analysis-asr-extensions` | **Date**: 2026-05-06

## Command

```bash
uv run python scripts/analyze_audio.py [OPTIONS] AUDIO
```

## Positional Arguments

| Argument | Description |
|----------|-------------|
| AUDIO | Path to a `.wav`, `.flac`, `.mp3`, or other librosa-readable audio file |

## Options (selected; full list in `--help`)

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir PATH` | `artifacts/analyze_audio/` | Per-run results directory |
| `--cache-dir PATH` | `artifacts/analyze_audio_cache/` | Persistent content-addressable cache |
| `--no-cache` | off | Disable both lookup and store |
| `--device {cpu,cuda,mps,auto}` | `auto` | Per-task compatible device selection |
| `--no-enhancement` | off | Skip the `enhanced_16k` pass |
| `--diarization-models …` | `[pyannote/speaker-diarization-community-1, nvidia/diar_sortformer_4spk-v1]` | List of diarization models |
| `--ast-model ID` | `MIT/ast-finetuned-audioset-10-10-0.4593` | AST model |
| `--yamnet-model ID` | `google/yamnet` | YAMNet model |
| `--asr-models …` | `[openai/whisper-large-v3-turbo, ibm-granite/granite-speech-3.3-8b, nvidia/canary-qwen-2.5b, Qwen/Qwen3-ASR-1.7B]` | List of ASR models. Granite/Canary are **back in defaults** — they go through auto-align after transcription. |
| `--embeddings-models …` | `[speechbrain/spkrec-ecapa-voxceleb, speechbrain/spkrec-resnet-voxceleb]` | Embedding models |
| `--enhancement-model ID` | `speechbrain/sepformer-wham16k-enhancement` | Enhancement model |
| `--ast-win-length` `--ast-hop-length` | `10.24, 10.24` | AST native window |
| `--yamnet-win-length` `--yamnet-hop-length` | `0.96, 0.48` | YAMNet native frame and hop |
| `--no-align-asr` (new) | off | Disable the auto-align step for timestamp-less ASR. Outputs become text-only ScriptLines (single full-audio TextArea region in LS export). |
| `--aligner-model ID` (new) | `facebook/mms-1b-all` | Multilingual aligner backend used for auto-align. Override to swap MMS for an alternative. |
| `--asr-language ISO` (new) | (auto from ASR model card / Whisper LID) | Force a specific language for the aligner. Use ISO 639-1 (`en`, `fr`, ...) or ISO 639-3 (`eng`, `fra`, ...). |
| `--qwen-asr-no-timestamps` (new) | off | Skip Qwen3's internal forced-aligner companion. Then the script's auto-align step takes over (unless `--no-align-asr` is also set). |
| `--skip TASK …` | `()` | Skip listed tasks. Choices: `diarization, ast, yamnet, features, asr, embeddings, alignment` |
| `--help`, `-h` | — | Show help text |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success — at least the summary, LS bundle, and per-task outputs were written |
| 2 | Audio file not found / unreadable |

Per-task and per-alignment failures do not change the exit code; they are captured per-step in their own JSON.

## Output Layout

```
<output-dir>/<stem>_<UTC-timestamp>/
├── summary.json              # top-level run summary + provenance
├── labelstudio_tasks.json    # importable into Label Studio
├── labelstudio_config.xml    # paste into LS project labeling config
├── raw_16k/
│   ├── diarization/
│   │   ├── pyannote_speaker_diarization_community_1.json
│   │   └── nvidia_diar_sortformer_4spk_v1.json
│   ├── ast.json
│   ├── yamnet.json
│   ├── features.json
│   ├── asr/
│   │   ├── openai_whisper_large_v3_turbo.json
│   │   ├── ibm_granite_granite_speech_3_3_8b.json
│   │   ├── nvidia_canary_qwen_2_5b.json
│   │   └── Qwen_Qwen3_ASR_1_7B.json
│   ├── alignment/
│   │   ├── ibm_granite_granite_speech_3_3_8b.json   # only for timestamp-less ASRs
│   │   └── nvidia_canary_qwen_2_5b.json
│   └── embeddings/
│       ├── speechbrain_spkrec_ecapa_voxceleb.json
│       └── speechbrain_spkrec_resnet_voxceleb.json
└── enhanced_16k/   (same shape if enhancement succeeded)
```

ASR-with-native-timestamps (Whisper, NeMo Conformer CTC, Qwen3-ASR with companion aligner) does NOT get a sibling `alignment/<model>.json` — alignment is a no-op for those.

## Cache Layout

```
<cache-dir>/<cache_key>.json
```

One file per (audio, task, model, params, wrapper, senselab) tuple. Two cache entries per timestamp-less ASR: one for the ASR step, one for the alignment step. Either can hit while the other misses.

## Per-task output JSON shape

```json
{
  "status": "ok" | "failed" | "skipped",
  "elapsed_s": <float>,
  "result": <task-specific>,
  "error": "<repr>",       // when failed
  "traceback": "<text>",   // when failed
  "cache": "hit" | "miss" | "disabled",
  "cache_key": "<sha256 hex>",
  "provenance": { ... }    // only on cache miss
}
```

For an alignment output, `provenance` additionally includes `transcript_sha`, `language`, `parent_asr_cache_key`.

## ASR backend dispatch (senselab-side, observable from this script)

| Model id matches | Backend | Native timestamps |
|------------------|---------|-------------------|
| `nvidia/stt_*` or `nvidia/conformer*` | NeMo subprocess venv `nemo-diarization` (shared) | yes |
| `nvidia/canary-*` (new) | NeMo subprocess venv `nemo-canary-qwen` (new) | no — script auto-aligns |
| `Qwen/Qwen3-ASR*` (new) | Qwen subprocess venv `qwen-asr` (new) | yes (companion aligner; toggle with `--qwen-asr-no-timestamps`) |
| `ibm-granite/granite-speech-*` | HF pipeline (modified to allow `return_timestamps=False`) | no — script auto-aligns |
| anything else | HuggingFace `automatic-speech-recognition` pipeline | depends on model (Whisper: yes; CTC: yes) |

## Auto-align stage (script-side)

After the ASR family completes, the script iterates each ASR ModelRun:

1. If `--no-align-asr` is set → skip alignment entirely.
2. If the ASR result already includes timestamps (`start`/`end` set, or `chunks` non-empty) → skip alignment for that one (no-op).
3. Otherwise → call `senselab.audio.tasks.forced_alignment.align_transcriptions(audio, transcript, language=…)` with the resolved language (from `--asr-language`, the ASR model's documented language, or auto-LID via Whisper). On success → emit per-segment regions in the LS export. On failure → preserve the ASR text, mark the alignment as failed in its own JSON, fall back to single-region TextArea in LS.

## Aligner backend dispatch (senselab-side)

`align_transcriptions(audio, transcript, language)` consults a language→backend registry:

| Language | Backend | Notes |
|----------|---------|-------|
| any in existing `DEFAULT_ALIGN_MODELS_HF` (en, fr, de, es, it, pt, …) | WhisperX-style per-language wav2vec2 (existing) | Backwards-compat default. |
| ja, zh | MMS via HF (`facebook/mms-1b-all` + iso3 adapter) + uroman romanization | New — covers Granite Speech 3.3's ja/zh outputs. |
| anything else | MMS via HF | New — fall-back default. |

Override per-language with a config dict if needed; the common case is zero-config.

## Examples

```bash
# Default — Whisper, Granite, Canary-Qwen, Qwen3-ASR all run; Granite + Canary auto-align.
uv run python scripts/analyze_audio.py path/to/audio.wav

# Same audio, fix the aligner — re-uses ASR cache, reruns just alignment.
uv run python scripts/analyze_audio.py path/to/audio.wav \
    --aligner-model facebook/mms-1b-fl102

# Skip alignment entirely; LS export becomes single-region for Granite/Canary.
uv run python scripts/analyze_audio.py --no-align-asr path/to/audio.wav

# Force a non-English language for the aligner (Granite outputs Japanese, say).
uv run python scripts/analyze_audio.py --asr-language ja path/to/audio.wav

# Skip the heavy ASR family entirely.
uv run python scripts/analyze_audio.py --skip asr alignment path/to/audio.wav
```

## Out-of-scope behaviors

- No automatic resampling to anything other than 16 kHz mono.
- No multi-file batch mode.
- No streaming / live-microphone mode.
- No automatic forced alignment of Whisper or other natively-timestamped output (auto-align is a no-op for them).
- No cross-machine cache sharing.
