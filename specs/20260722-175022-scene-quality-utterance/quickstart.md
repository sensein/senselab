# Quickstart: Scene-aware presence axis + improved utterance uncertainty

**Feature**: `20260722-175022-scene-quality-utterance`

## Prerequisites

```bash
# Dev environment (adds librosa as an explicit dep during implementation)
uv sync --extra text --extra video --extra senselab-ai --extra nlp --group dev

# HuggingFace token for gated pyannote models (segmentation-3.0, brouhaha)
export HF_TOKEN=<your token>
# Accept model conditions once at:
#   https://hf.co/pyannote/segmentation-3.0  and  https://hf.co/pyannote/brouhaha
```

## Run the full workflow (new signals on by default)

```bash
uv run python scripts/analyze_audio.py --input tutorial_audio_files/<clip>.wav
```

Then inspect the presence parquet:

```python
import pandas as pd
df = pd.read_parquet("<run_dir>/raw_16k/uncertainty/presence.parquet")
df[["start","end","presence_confidence","presence_uncertainty",
    "quality_snr","quality_clip","quality_reverb","quality_bandwidth","quality_uncertainty",
    "src_speech","src_people","src_machine","src_environment","src_dominant"]].head()
```

## Common variations

```bash
# Keep the old 0.5 s presence resolution
uv run python scripts/analyze_audio.py --input clip.wav --presence-grid-win 0.5 --presence-grid-hop 0.5

# Skip the scene/quality model (quality columns null, faster)
uv run python scripts/analyze_audio.py --input clip.wav --no-scene-quality

# Only persist top-50 scene classes instead of the full distribution
uv run python scripts/analyze_audio.py --input clip.wav --scene-top-k 50
```

## Standalone API

```python
from senselab.audio.workflows.audio_analysis import BucketGrid, compute_uncertainty_axes

axis_results, incomparable = compute_uncertainty_axes(
    passes=passes_summary,
    grid=BucketGrid(),                                  # identity/shared 0.5 s
    presence_grid=BucketGrid(win_length=0.1, hop_length=0.02),
    utterance_grid=BucketGrid(win_length=1.0, hop_length=0.5),
    params={...},
    audio={"raw_16k": audio},
    scene_quality=True,
    sound_sources=True,
)
```

## Calibration (optional, P3)

```bash
uv run python scripts/calibrate_scene_quality.py \
    --clean tutorial_audio_files/<clip>.wav \
    --snr-sweep 30 20 10 5 0 --rt60-sweep 0.2 0.5 1.0 \
    --out src/senselab/audio/workflows/audio_analysis/data/scene_quality_calibration.json
# → writes the profile + a reported-vs-true validation plot under artifacts/
```

## Validate the acceptance criteria

```bash
# Full new test set
uv run pytest src/tests/audio/workflows/audio_analysis/ -v

# Category-map coverage (SC-003)
uv run pytest src/tests/audio/workflows/audio_analysis/sound_sources_test.py -v

# Regression: existing consumers unchanged (SC-008)
uv run pytest src/tests/audio/workflows/audio_analysis/{compute_uncertainty_axes_test,disagreements_test,labelstudio_test,plot_test}.py

# Lint / types / spelling before pushing
uv run ruff format && uv run ruff check --fix && uv run mypy . && uv run codespell
```

## Phase-by-phase demo (each independently testable)

| Phase | Command to see it work |
|---|---|
| 1 Quality | run workflow on a clip, confirm `quality_*` columns; noised region shows higher `quality_snr` |
| 2 Sources | run on a clip with background traffic, confirm `src_machine` elevated; coverage test green |
| 3 Temporal | `--presence-grid-win 0.1`, confirm finer buckets + `presence_confidence`/`presence_uncertainty` |
| 4 Utterance | run with a Whisper model, confirm `token_entropy` populated + `scene_quality_coupling` in utterance parquet |
| 5 Calibration | run the calibrate helper, confirm profile JSON + validation plot |

## Notes / gotchas (from research)

- Quality metrics use a fixed **0.5 s analysis window** broadcast onto the finer presence buckets — the reported quality resolution is 0.5 s even when presence buckets are 0.1 s (recorded in provenance).
- `--scene-top-k` must be large enough (default = full) for source masses to be meaningful; top-1 presence behavior is unchanged regardless.
- Token entropy only populates for the Whisper HF backend; other ASR backends leave it null (utterance falls back to today's signals).
- Gated pyannote models: first run downloads and caches; subsequent runs use `local_files_only` (no Hub calls).
