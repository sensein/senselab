# CLI Contract — analyze_audio.py comparator additions

All flags are additive. The script's existing flags from PR #510 are unchanged.

## New flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--skip-comparisons {raw_vs_enhanced,within_stream,cross_stream,uncertainty}` | repeated | (none) | Skip a specific comparison axis. May be passed multiple times. |
| `--cross-stream-win-length` | float (seconds) | `0.2` | Window length for cross-stream comparisons. |
| `--cross-stream-hop-length` | float (seconds) | `0.1` | Hop length for cross-stream comparisons. |
| `--uncertainty-aggregator {min,mean,harmonic_mean,disagreement_weighted}` | choice | `min` | Function used to combine per-model confidences into the `combined_uncertainty` score that drives `disagreements.json` ranking. |
| `--phoneme-disagreement-threshold` | float in [0, 1] | `0.50` | Threshold on `phoneme_per` above which `phoneme_disagreement = true`. |
| `--speech-presence-labels` | comma-separated list of strings | the seven AudioSet "Speech" subtree labels (see research.md §5) | AudioSet labels that count as "speech-present" for AST/YAMNet ↔ diarization comparison. |
| `--asr-reference-model` | string (HF model id) | `openai/whisper-large-v3-turbo` | Which ASR model is the soft reference for ASR-vs-ASR WER computation. |
| `--diarization-boundary-shift-ms` | float ≥ 0 | `50.0` | Boundary-shift threshold (in milliseconds) used by the per-task differencer to decide when two diarization model outputs disagree on segment boundaries. Per Constitution §VIII (No Hardcoded Parameters), this is a CLI flag rather than a baked-in constant. |
| `--disagreements-top-n` | int ≥ 0 | `100` | Number of top-ranked rows to emit in `disagreements.json`. `0` disables the index. |

## Existing `--skip` flag

The existing `--skip` flag accepts `comparisons` as a new value to skip the entire comparator stage at once (equivalent to passing all four `--skip-comparisons` axes). Other existing values (`diarization`, `ast`, `yamnet`, `features`, `asr`, `embeddings`, `alignment`) are unchanged.

## Backwards compatibility

- A run with `--skip comparisons` MUST produce **bit-identical** output to a current-main run (SC-005). The comparator stage is the only new code; not running it MUST be a no-op.
- All comparator flags have defaults that produce useful behavior, so the common case is `uv run python scripts/analyze_audio.py path/to.wav` with no extra arguments.
- New flags appear in `--help` ordered after the existing comparable flags (e.g. `--cross-stream-*-length` next to `--features-*-length`).

## Example invocations

```bash
# Default — runs the full pipeline including comparisons
uv run python scripts/analyze_audio.py audio.wav

# Skip just the cross-stream comparisons; keep raw_vs_enhanced and within_stream
uv run python scripts/analyze_audio.py audio.wav --skip-comparisons cross_stream

# Switch the uncertainty aggregator and tighten the cross-stream grid
uv run python scripts/analyze_audio.py audio.wav \
    --uncertainty-aggregator harmonic_mean \
    --cross-stream-win-length 0.1 --cross-stream-hop-length 0.05

# Treat singing/laughter as "speech" for the AST/YAMNet vs diarization check
uv run python scripts/analyze_audio.py audio.wav \
    --speech-presence-labels "Speech,Conversation,Singing,Laughter"

# Quiet run that skips the entire comparator stage (legacy output only)
uv run python scripts/analyze_audio.py audio.wav --skip comparisons
```

## Validation

Argparse must reject:

- `--cross-stream-win-length 0` or negative.
- `--cross-stream-hop-length` greater than `--cross-stream-win-length`.
- `--phoneme-disagreement-threshold` outside `[0, 1]`.
- `--diarization-boundary-shift-ms` negative.
- `--disagreements-top-n` negative.
- Unknown `--uncertainty-aggregator` value (argparse `choices=` enforces).
- Unknown `--skip-comparisons` value (argparse `choices=` enforces).
