# Quickstart: Audio Analysis Script + ASR Backend Extensions + Forced Alignment

**Branch**: `20260506-154425-audio-analysis-asr-extensions` | **Date**: 2026-05-06

## What this delivers

1. `scripts/analyze_audio.py`: one-command multi-task audio analysis with content-addressable cache + provenance + Label Studio export.
2. **Multilingual forced-alignment** in senselab via a new MMS backend, covering 1100+ languages (Granite's 8 languages and beyond).
3. **Three new ASR paths** that all work end-to-end through the script:
   - **IBM Granite Speech 3.3 8B** via the existing HF pipeline with `return_timestamps=False`, then auto-aligned by the script.
   - **NVIDIA Canary-Qwen 2.5B** via a new `nemo-canary-qwen` subprocess venv, then auto-aligned by the script.
   - **Alibaba Qwen3-ASR 1.7B** via a new `qwen-asr` subprocess venv with its bundled companion forced-aligner for native timestamps (auto-align step is a no-op for it).
4. Separable caching: ASR and alignment have independent cache entries, so re-running just the alignment doesn't re-run the (slow) ASR.

## Install

```bash
uv sync --extra articulatory --extra text --extra video --extra nlp --group dev
```

The new MMS aligner uses the existing `transformers` dependency. The optional `nlp` extra now also includes `uroman` for ja/zh romanization.

The two new subprocess venvs (`nemo-canary-qwen`, `qwen-asr`) are provisioned automatically the first time their model ids are requested. First-time provisioning takes minutes and downloads several GB.

## Run the script

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav
```

What happens (default flow):

1. Audio is downmixed to mono and resampled to 16 kHz.
2. Each task runs once per (model × audio variant). For each ASR model:
   - If the ASR has native timestamps (Whisper, Qwen3-ASR with companion): emit timestamped ScriptLines.
   - If text-only (Granite, Canary-Qwen): the ASR step caches text; the alignment step caches timestamps separately.
3. Caching: every step's outcome is stored under `artifacts/analyze_audio_cache/<key>.json`. Re-running with identical inputs replays prior outputs.
4. Label Studio export: every model gets its own timeline track.

## Try the new ASR backends

The default `--asr-models` already includes all four (Whisper turbo, Granite Speech, Canary-Qwen, Qwen3-ASR). Just run:

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav
```

First time:

- senselab detects `nvidia/canary-` → provisions `nemo-canary-qwen` venv → loads `SALM` → returns text-only ScriptLines → script auto-aligns via MMS.
- senselab detects `Qwen/Qwen3-ASR` → provisions `qwen-asr` venv → loads `Qwen3ASRModel` + companion aligner → returns timestamped ScriptLines (auto-align step skips this one).
- senselab detects `ibm-granite/granite-speech-` → uses the main venv's HF pipeline with `return_timestamps=False` → returns text-only ScriptLines → script auto-aligns via MMS.
- Whisper turbo runs as before, with native timestamps.

## Cache hits — re-run is fast

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav   # second time
```

Console will show:

```
[asr[openai/whisper-large-v3-turbo]] cache HIT (a3f2bc...)
[asr[ibm-granite/granite-speech-3.3-8b]] cache HIT (b7d1e9...)
[alignment[ibm-granite/granite-speech-3.3-8b]] cache HIT (c4e0a2...)
[asr[nvidia/canary-qwen-2.5b]] cache HIT (...)
[alignment[nvidia/canary-qwen-2.5b]] cache HIT (...)
[asr[Qwen/Qwen3-ASR-1.7B]] cache HIT (...)
```

Re-run only the alignment step (e.g., after a senselab MMS upgrade):

```bash
# Bump the senselab version → alignment cache invalidates automatically (senselab_ver in key).
# Or pass a different aligner explicitly:
uv run python scripts/analyze_audio.py path/to/audio.wav \
    --aligner-model facebook/mms-1b-fl102
```

The ASR step still hits the cache; only `alignment[granite]` and `alignment[canary]` re-run.

## Toggle alignment off

```bash
uv run python scripts/analyze_audio.py --no-align-asr path/to/audio.wav
```

Now Granite and Canary-Qwen produce text-only ScriptLines. Their LS export tracks become a single full-audio TextArea region with the full transcript — useful for fast text-only comparison.

## Force a specific language for the aligner

When the ASR's output language doesn't match the model's documented default (e.g., Granite Speech 3.3 in Japanese-translation mode):

```bash
uv run python scripts/analyze_audio.py --asr-language ja path/to/audio.wav
```

For ja/zh, the script automatically applies `uroman` romanization before MMS sees the transcript.

## Import into Label Studio

1. Settings → Labeling Interface: paste `<run-dir>/labelstudio_config.xml`.
2. Data Manager → Import: upload `<run-dir>/labelstudio_tasks.json`.

Each audio variant becomes one LS task. Within each task you'll see parallel timeline tracks:

- `raw_16k__diarization__pyannote_speaker_diarization_community_1` — speaker regions
- `raw_16k__diarization__nvidia_diar_sortformer_4spk_v1` — speaker regions
- `raw_16k__ast` — AST scene labels at 10.24-s windows
- `raw_16k__yamnet` — YAMNet scene labels at 0.96-s windows
- `raw_16k__asr__openai_whisper_large_v3_turbo` — TextArea regions per Whisper segment
- `raw_16k__asr__ibm_granite_granite_speech_3_3_8b` — TextArea regions per MMS-aligned segment
- `raw_16k__asr__nvidia_canary_qwen_2_5b` — TextArea regions per MMS-aligned segment
- `raw_16k__asr__Qwen_Qwen3_ASR_1_7B` — TextArea regions per Qwen3-companion-aligned word

## When alignment fails

If MMS can't align (audio doesn't match the transcript closely enough, language unsupported, etc.):

- The ASR text is preserved.
- `alignment/<model>.json` records `status: failed` with the error.
- The LS export for that one model falls back to a single full-audio TextArea region.
- The overall run continues normally.

Re-run after upgrading the aligner; only the failed alignment retries (the ASR cache hit means no expensive re-transcription).

## Caveats

- **AST window**: the script defaults to AST's native 10.24 s. Smaller windows fail inside AST's kaldi-fbank.
- **Diarization on macOS**: pyannote rejects MPS; `--device auto` falls back to CPU per task automatically.
- **Canary-Qwen on macOS**: NeMo's macOS support is uneven; expect CPU fallback (slow).
- **Granite Speech multilingual outputs**: the user controls the output language via prompting; pass `--asr-language` to tell the aligner what language to expect, or rely on the default model-card language (en).
- **MMS license**: `facebook/mms-1b-all` weights are CC-BY-NC 4.0 (research/non-commercial). Swap with `--aligner-model` for commercial use.

## Verify (developer test)

```bash
uv run pytest src/tests/scripts/analyze_audio_test.py
uv run pytest src/tests/audio/tasks/forced_alignment/   # MMS-aware tests skipif on missing model
uv run pytest src/tests/audio/tasks/speech_to_text/canary_qwen_test.py  # skipif venv absent
uv run pytest src/tests/audio/tasks/speech_to_text/qwen_test.py        # skipif venv absent
```

All new tests `skipif` automatically when their resources aren't available, so default CI passes unchanged.
