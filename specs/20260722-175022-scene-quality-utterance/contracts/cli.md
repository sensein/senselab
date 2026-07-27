# Contract: CLI additions (`scripts/analyze_audio.py`)

All new flags have defaults so the common case needs zero configuration (constitution VIII).

| Flag | Default | Effect |
|---|---|---|
| `--presence-grid-win` / `--presence-grid-hop` | `0.1` / `0.02` | presence reporting grid (new `presence_grid`) |
| `--scene-top-k` | full (527 AST / 521 YAMNet) | class count persisted from AST/YAMNet windowed classification (enables source masses) |
| `--scene-model` | `pyannote/brouhaha` | scene/quality model id |
| `--no-scene-quality` | off | skip Brouhaha + quality columns (columns emitted null) |
| `--no-sound-sources` | off | skip source categorization (columns null) |
| `--calibration-profile` | bundled default | path to a fitted `CalibrationProfile` JSON |
| `--utterance-scene-coupling-weights` | documented defaults `w_q`, `w_s` | scene→utterance coupling strength |

## `compute_uncertainty_axes` signature additions

```python
def compute_uncertainty_axes(
    *,
    passes, grid, params, audio,
    utterance_grid: BucketGrid | None = None,        # existing
    presence_grid: BucketGrid | None = None,         # NEW (default 0.1/0.02)
    scene_quality: bool = True,                       # NEW
    sound_sources: bool = True,                       # NEW
    calibration: CalibrationProfile | None = None,    # NEW
    ...
) -> tuple[list[AxisResult], list]:
    ...
```

- Defaults preserve current behavior for callers that don't pass the new kwargs, except that presence now reports on the finer default grid; a caller can pass `presence_grid=grid` to retain the old 0.5 s presence resolution (documented in quickstart).

## Calibration helper — `scripts/calibrate_scene_quality.py`

```
uv run python scripts/calibrate_scene_quality.py \
    --clean tutorial_audio_files/<clip>.wav \
    --snr-sweep 30 20 10 5 0 \
    --rt60-sweep 0.2 0.5 1.0 \
    --out src/senselab/audio/workflows/audio_analysis/data/scene_quality_calibration.json
```

- Synthesizes noise + decay-RIR mixtures (numpy, no new dep), runs the quality estimators, fits normalization + temperature, writes the profile, and emits a reported-vs-true validation plot under `artifacts/`.
