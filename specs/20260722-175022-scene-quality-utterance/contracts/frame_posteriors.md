# Contract: frame-level speech posteriors

**Module**: `src/senselab/audio/tasks/voice_activity_detection/frame_posteriors.py` (new)

## Public function

```python
def extract_speech_frame_posteriors(
    audios: list[Audio],
    model: PyannoteAudioModel | None = None,   # default pyannote/segmentation-3.0, pinned revision
    device: DeviceType | None = None,
) -> list[FramePosterior]:
    ...
```

Where `FramePosterior` is a small dataclass:

```python
@dataclass
class FramePosterior:
    probs: np.ndarray   # shape (num_frames,), P(speech) in [0,1]
    frame_hop_s: float  # seconds per frame (~0.0169 for segmentation-3.0)
    frame_win_s: float  # analysis window per frame
```

## Behavior

- Loads the model via `pyannote.audio.Model.from_pretrained(repo, revision=..., token=get_huggingface_token())` wrapped by `ensure_hf_model` + `local_files_only` when cached (constitution VI). Cached per `(repo, revision, device)`.
- Runs `pyannote.audio.Inference(model, ...)` → `SlidingWindowFeature`; collapses the class/powerset axis with `max` → continuous P(speech) per frame.
- **Does NOT** route through `VoiceActivityDetection`/`Pipeline` (that re-segments and smooths).
- Returns per-frame probability arrays, not `ScriptLine` segments.

## Bucket aggregation helper

```python
def mean_posterior_in_window(fp: FramePosterior, start_s: float, end_s: float) -> tuple[float, float]:
    """Return (mean P(speech), within-window frame std) over frames overlapping [start,end)."""
```

- Mean → presence confidence contribution; std → within-bucket temporal-instability contribution to `presence_uncertainty`.
- Empty overlap → `(nan, nan)` handled by caller as a missing voter.

## Error handling

- Model unavailable (no token / not cached / import failure `(ImportError, RuntimeError)`) → function returns posteriors from whatever backends loaded; a fully failed model contributes no voter (FR-023). Never aborts the workflow.

## Brouhaha variant

`scene_quality/brouhaha.py` exposes the analogous multitask extractor (see `contracts/quality.md`); its VAD head reuses `FramePosterior` so presence gets a second frame-posterior voter with identical bucket aggregation.
