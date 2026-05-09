# Quickstart — Reviewer recipes for the comparator outputs

Every recipe assumes you have just run:

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav
```

…with the default config. The new comparator stage runs at the end of the pipeline.

## Recipe 1 — "Did enhancement help here?"

Open the run's Label Studio bundle and look at the `pass_pair__compare__*` tracks. Any region you see flagged is one where the same task / model gave a different answer between raw and enhanced. Specific tracks to scrub:

- `pass_pair__compare__asr__whisper` — Whisper edits between raw and enhanced.
- `pass_pair__compare__diarization__pyannote` — pyannote shifted boundaries.
- `pass_pair__compare__ast` / `pass_pair__compare__yamnet` — scene label flipped.

If the track is **empty**, nothing flipped → enhancement was a no-op for that task.

For a deeper read, open the parquet:

```python
import pandas as pd
df = pd.read_parquet("artifacts/analyze_audio/<run>/comparisons/raw_vs_enhanced/asr/whisper.parquet")
df[df["agree"] == False][["start", "end", "wer", "a_text", "b_text"]]
```

## Recipe 2 — "Where do my ASR models disagree on this clip?"

Scrub the within-stream tracks:

- `raw_16k__compare__asr__whisper_vs_granite`
- `raw_16k__compare__asr__whisper_vs_canary_qwen`
- `raw_16k__compare__asr__whisper_vs_qwen3_asr`
- `raw_16k__compare__asr__granite_vs_canary_qwen`
- `raw_16k__compare__asr__granite_vs_qwen3_asr`
- `raw_16k__compare__asr__canary_qwen_vs_qwen3_asr`

Each has a paired TextArea track showing the actual disagreement text and WER. Sort by uncertainty:

```python
import pandas as pd
df = pd.read_parquet("artifacts/analyze_audio/<run>/raw_16k/comparisons/asr/whisper_vs_granite.parquet")
df.sort_values("combined_uncertainty", ascending=False).head(10)
```

## Recipe 3 — "Where do ASR and diarization conflict?"

The cross-stream tracks flag regions where ASR returned text that diarization said was silent (or vice versa):

- `raw_16k__compare__asr_vs_diarization__whisper__pyannote`
- `raw_16k__compare__asr_vs_diarization__whisper__sortformer`

Open the parquet:

```python
import pandas as pd
df = pd.read_parquet(
    "artifacts/analyze_audio/<run>/raw_16k/comparisons/cross_stream/asr__whisper__vs__diarization__pyannote.parquet"
)
df[df["agree"] == False][["start", "end", "asr_says_speech", "diar_says_speech"]]
```

A typical pattern: `asr_says_speech=True, diar_says_speech=False` clusters around very short utterances or low-energy speech that diarization missed.

## Recipe 4 — "What were the most uncertain disagreements overall?"

Open the index:

```bash
jq '.entries[:10]' artifacts/analyze_audio/<run>/disagreements.json
```

The top entries are the regions where (a) something disagreed and (b) the contributing models had the lowest combined confidence. Each entry points at the parquet path + row index so you can drill in.

## Recipe 5 — Custom AudioSet speech-presence allowlist

If you want singing or laughter to count as "speech" for the AST/YAMNet ↔ diarization check:

```bash
uv run python scripts/analyze_audio.py audio.wav \
    --speech-presence-labels "Speech,Conversation,Singing,Laughter,Narration, monologue"
```

The allowlist is recorded in the parquet provenance, so each run is reproducible.

## Recipe 6 — Tighter cross-stream grid for boundary precision

Pyannote produces speech boundaries with sub-100ms precision. To match it:

```bash
uv run python scripts/analyze_audio.py audio.wav \
    --cross-stream-win-length 0.05 --cross-stream-hop-length 0.025
```

Parquet size scales linearly with hop; expect ~40 KB / second / pair at this resolution.

## Recipe 7 — Skip the comparator entirely

For a legacy run (bit-identical to current main):

```bash
uv run python scripts/analyze_audio.py audio.wav --skip comparisons
```
