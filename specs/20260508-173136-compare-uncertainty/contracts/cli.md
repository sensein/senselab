# CLI Contract — analyze_audio.py comparator additions

All flags are additive on top of the script's existing analyze_audio flags.

## New flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--cross-stream-win-length` | float (seconds) | `0.5` | Bucket length for the comparator. |
| `--cross-stream-hop-length` | float (seconds) | `0.5` | Hop length for the comparator (≤ win-length; default = non-overlapping). |
| `--uncertainty-aggregator` `{min,mean,harmonic_mean,disagreement_weighted}` | choice | `min` | Function used to collapse per-axis sub-signals into `aggregated_uncertainty`. |
| `--phoneme-disagreement-threshold` | float in `[0, 1]` | `0.50` | Threshold on `phoneme_per` above which `phoneme_disagreement = true` (utterance axis sub-signal). |
| `--speech-presence-labels` `LABEL [LABEL ...]` | space-separated list of strings (`nargs="+"`) | seven AudioSet "Speech" subtree labels | AudioSet labels that count as "speech-present" for AST/YAMNet contributions to the presence axis. AudioSet labels themselves contain commas (e.g. `"Narration, monologue"`) — that's why this is `nargs="+"` not a comma-string. |
| `--asr-reference-model` | string (HF model id) | `openai/whisper-large-v3-turbo` | Reserved for utterance-axis transcript-consensus tiebreaks (currently used to pick the soft reference for the LS TextArea sibling track). |
| `--diarization-boundary-shift-ms` | float ≥ 0 | `50.0` | Boundary-shift tolerance for treating two diar segments as describing the same speech region; per Constitution §VIII (No Hardcoded Parameters). |
| `--disagreements-top-n` | int ≥ 0 | `100` | Number of top-ranked rows in `disagreements.json`. `0` disables the index. |

## `--skip comparisons`

The script's existing `--skip` flag accepts `comparisons` as a new value to disable the
entire comparator stage. With `--skip comparisons`, the script's output is identical to a
run from before the comparator was added — no `<run_dir>/<pass>/uncertainty/` subtree, no
`<run_dir>/uncertainty/raw_vs_enhanced/` subtree, no `disagreements.json`, no comparator
tracks in the LS bundle. This satisfies SC-005.

## Defaults rationale

- `0.5 s` non-overlapping grid: matches the temporal resolution that's actually meaningful
  given the underlying signal granularities (Whisper word ≈ 20 ms, pyannote ≈ 62.5 ms,
  AST 10.24 s) without double-counting via overlap.
- `min` aggregator: surfaces the most-doubtful contributing signal — the right default for
  reviewers reading the disagreements index ("show me where *any* model is unsure").
- `0.50` phoneme PER threshold: half the phonemes in a transcript span had to be wrong
  before flagging — a strong signal, not a noisy one.

## Example invocations

```bash
# Default — runs the full pipeline including comparator
uv run python scripts/analyze_audio.py audio.wav

# Switch the aggregator from "max-doubtful-signal" to "average-doubt"
uv run python scripts/analyze_audio.py audio.wav --uncertainty-aggregator mean

# Tighter grid for sub-second analysis (slower, double-counts via overlap)
uv run python scripts/analyze_audio.py audio.wav \
    --cross-stream-win-length 0.2 --cross-stream-hop-length 0.1

# Treat singing/laughter as "speech" for AST/YAMNet contributions to presence
uv run python scripts/analyze_audio.py audio.wav \
    --speech-presence-labels Speech Conversation Singing Laughter \
    "Narration, monologue"

# Skip the entire comparator stage
uv run python scripts/analyze_audio.py audio.wav --skip comparisons

# Disable the disagreements index but keep the parquets + plot + LS bundle
uv run python scripts/analyze_audio.py audio.wav --disagreements-top-n 0
```

## Validation rules

The script rejects (via `argparse.error`):

- `--cross-stream-win-length <= 0`
- `--cross-stream-hop-length > --cross-stream-win-length`
- `--phoneme-disagreement-threshold` outside `[0, 1]`
- `--diarization-boundary-shift-ms < 0`
- `--disagreements-top-n < 0`
