# Scene description

Describes the environment visible in a recorded video call or video interview: the call layout, and
for each participant tile whether the camera is on, the setting, background, lighting, other people
and distractions.

```python
from senselab.video.tasks.scene_description import default_backend, describe_video_scenes

backend = default_backend()  # MLX Qwen3-VL-4B on Apple silicon, transformers Qwen3-VL-4B elsewhere
reports = describe_video_scenes(["interview1.mp4", "interview2.mp4"], backend=backend, interval_s=180)
summary = reports[0].summary
summary.categorical["primary_setting"].mode, summary.categorical["layout"].consistency
summary.change_points  # layout / tile-count changes, other people appearing or leaving
```

The pipeline has four stages, each usable alone:

1. `sample_frames` decodes the file once (PyAV) and keeps one frame per `interval_s` plus frames
   whose colour histogram jumps by more than `scene_threshold`, downscaled to `long_side`.
2. `frame_metrics` measures brightness, contrast, blur (variance of the Laplacian), colourfulness,
   the near-black area (camera-off tiles) and the Haar-cascade face count.
3. `describe_frames` asks the VLM for strict JSON matching `SceneDescription`, validates it, and
   sends an invalid completion back once with the error; `parse_ok` records the outcome.
4. `aggregate` reports per-field modes, means and consistency (fraction of frames agreeing with the
   mode), distraction rates and change points.

The MLX backend runs in the `mlx-vlm` subprocess venv, whose worker keeps the model resident across
frames and videos. Schema, model choice and measurements:
`specs/20261017-104000-video-scene-description/design.md`.
