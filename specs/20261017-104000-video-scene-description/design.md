# Video scene description: context evaluation for video-call recordings

Module: `src/senselab/video/tasks/scene_description/`. Goal: describe the surrounding environment
captured in Zoom-style interview recordings (about 27 h of video), with the VLM stage taking
roughly 1.5 h or less on an M1 Pro (32 GB).

## Pipeline

| Stage | Function | Cost |
|---|---|---|
| Sample | `sample_frames(video, interval_s=60, scene_threshold=0.3, max_frames=None, long_side=896)` | one sequential decode |
| Measure | `frame_metrics(image)` | milliseconds per frame |
| Describe | `describe_frames(images, backend)` | dominant: seconds per frame |
| Summarise | `aggregate(timestamps, descriptions, metrics)` | negligible |

`describe_video_scene` chains them in batches of `batch_size` frames, so memory holds one batch of
downscaled frames; `describe_video_scenes` runs many videos against one backend.

### Sampling

PyAV decodes the stream once. A frame is kept when it is the first at or after the next multiple of
`interval_s`, or when the histogram distance (half the L1 distance between 16-bin per-channel
histograms of a 64x36 thumbnail, averaged over channels; 0 = identical, 1 = disjoint) to the
previous compared frame exceeds `scene_threshold`. Comparison runs at `analysis_fps=2` and a cut
must be `min_scene_gap_s=2` after the previous kept frame, so a cross-fade yields one frame, not a
burst. A cut that lands on an interval boundary is labelled `interval`. Only the kept frames and one
thumbnail are held; the video is never loaded whole. `keyframes_only=True` skips non-key frames for
a ~10x faster decode at the cost of snapping samples to keyframes.

Why a histogram rather than ffmpeg's `select='gt(scene,...)'`: it needs no second ffmpeg process and
no filter-graph parsing, works on in-memory `Video` objects, and a layout switch (gallery to speaker
view, screen share) moves most of the colour mass, which is what the score measures. The 0.3 default
is the requested value, not a fitted one; on the synthetic cut video the cuts score > 0.8 and
within-segment noise < 0.01.

### Cheap metrics

Brightness and contrast are luma mean and standard deviation over 255. Blur is the variance of the
Laplacian. Colourfulness is Hasler and Suesstrunk (2003). `near_black_fraction` is the area of
16-pixel blocks with mean luma <= 40 and standard deviation <= 8: Zoom's camera-off tile is a flat
dark grey (about RGB 26-36) with a name or initials, so "dark and flat" separates it from a dim but
textured room, which "dark" alone does not. Faces use OpenCV's bundled frontal Haar cascade, chosen
over YuNet because it needs no model download; it misses small or turned faces, so `n_faces` is a
lower bound and the VLM's `n_people_visible` is the primary count. These levels are first choices,
not fitted thresholds; nothing downstream gates on them.

## Schema

`SceneDescription` (Pydantic, `extra="forbid"`):

```
layout: gallery | speaker_view | screen_share | single | other
n_tiles: int >= 0
tiles: [                                  # largest (main) tile first
  camera_on: bool
  n_people_visible: int >= 0
  setting: home | office | clinic | vehicle | outdoors | classroom | other | unknown
  background_type: real | virtual | blurred | unknown
  lighting: good | dim | backlit | uneven | unknown
  other_people_present: bool
  distractions: [pets | children | tv_screen | phone_use | eating | moving_background |
                 noise_source_visible] or [none]
  head_framing: centered | partial | off_frame
]
notes: str
```

`lighting` gained `unknown` beyond the requested four values: on a 2x2 gallery with one camera-off
tile the model had no valid lighting for that tile and both attempts failed validation. The prompt
now tells it to give camera-off tiles `n_people_visible 0`, `unknown` setting, background and
lighting, and `off_frame` framing.

Parsing strips code fences and surrounding prose, takes the outermost `{...}`, matches enum values
case-insensitively with spaces and hyphens read as underscores, drops `none` beside real
distractions, and turns `[]` into `["none"]`. Nothing else is coerced: an unknown enum value, an
extra key or a negative count fails. A failed completion is sent back once, text-only (no image),
with the error and the schema; `parse_ok`, `repaired`, both raw texts and the last error are kept.

## Aggregation

Per frame, `frame_fields` flattens a description to `layout`, `n_tiles`, `n_camera_on`,
`n_people_visible` (summed over tiles), `any_other_people_present`, and `primary_*` fields from the
first tile (tile order is not stable across frames, so only the main tile is tracked over time).
Categorical fields get mode, counts and consistency (fraction of described frames equal to the
mode); numeric fields get mean, median, range and consistency. Distractions are the fraction of
described frames in which any tile lists each item. Change points are reported between consecutive
described frames for `layout`, `n_tiles` and `any_other_people_present`; their time resolution is
the sampling interval, sharpened at layout switches by the scene-cut frames.

## Model

| Host | Backend | Model | Commit |
|---|---|---|---|
| Apple silicon | `MLXVLMBackend`, `mlx-vlm==0.7.8` in subprocess venv `mlx-vlm` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` (3.11 GB) | `2fd8dacbdb8f1e54b8c005f081ec5bf79c56376b` |
| Elsewhere | `TransformersVLMBackend`, in process | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` (resolved from `main` at run time; not downloaded here) |

`mlx-vlm` 0.7.8 needs `transformers>=5.14`, `opencv-python` (which clashes with senselab's
`opencv-python-headless`), `mlx-audio`, `datasets` and a FastAPI server stack, so it lives in a
subprocess venv with a committed lock (`src/senselab/utils/data/venv_locks/mlx-vlm.txt`,
torch-free). Both backends resolve the ref to a commit SHA first; the MLX worker loads the staged
`snapshots/<sha>` directory, the transformers backend loads with `revision=<sha>` through
`load_hf_resilient`. The MLX worker is a `serve_in_venv` worker keyed on (venv, model, SHA), so the
model loads once per Python process and serves every frame and every video after that.

## Measurements (M1 Pro, 32 GB)

Benchmark, re-runnable with one command (needs only the cached model and the repo's
`tutorial_images/`):

```bash
uv run python specs/20261017-104000-video-scene-description/bench_vlm.py --long-side 896 672 --repeats 2
```

It composes five 1920x1080 call frames from the public tutorial images (two-tile gallery, 2x2
gallery with a camera-off tile, speaker view with a thumbnail, single full frame, 3x3 gallery with a
camera-off tile), downsizes them to each `long_side`, and prints one JSON line per size.

**All figures below were taken with the machine shared by five other benchmarking agents, load
average 10-16 on 10 cores. They are upper bounds, not throughput figures**; an idle serial re-run
with the command above replaces them.

| long_side | load avg | wall s/frame | median gen s | max gen s | prompt tok | gen tok (median) | parse_ok |
|---|---|---|---|---|---|---|---|
| 896 | 10.1-14.4 | 17.2 | 15.5 | 29.9 | 864 | 194 | 10/10 |
| 672 | 10.1-14.4 | 13.1 | 11.3 | 22.1 | 668 | 117 | 10/10 |
| 512 (earlier prompt) | 11.4-12.3 | 11.3 | 9.8 | 22.9 | 560 | 239 | 8/8 |

Single-view frames took 4-5 s each; a 3x3 gallery took 24 s (554 generated tokens). Time is
dominated by generated tokens, roughly 40 ms each under this load, so it scales with the number of
tiles described, not with image size. The model writes indented JSON despite the single-line
instruction.

Memory: peak MLX (Metal) allocation 4.11 GB; peak worker RSS 1.5-3.0 GB; model load and first
frame about 17-38 s, once per process.

Correctness findings that changed the code:
- `max_tokens=512` truncated the 3x3 gallery at 896 px in both attempts (invalid JSON, `parse_ok`
  False); 1024 is the default now and the same frame parses in 554 tokens.
- Camera-off tiles had no valid `lighting`; see the schema section.
- The model counts a 3x3 gallery as 6-7 tiles; `n_tiles` for large galleries is approximate.

Decode (sampling), 120 s synthetic 1280x720 30 fps H.264, same load: full decode 1.9 s (62x real
time), `keyframes_only` 0.18 s (650x). For 27 h that is about 26 min and 2.5 min respectively,
on CPU, overlappable with the GPU-bound VLM.

## Recommended interval for 27 h

Budget 1.5 h = 5400 s of VLM time over 97,200 s of video.

| s/frame | frames in budget | interval |
|---|---|---|
| 17 (896, loaded machine) | 318 | 306 s |
| 13 (672, loaded machine) | 415 | 234 s |
| 8 (plausible idle) | 675 | 144 s |

Recommendation: `interval_s=240` with `long_side=672` (405 interval frames, about 88 min at the
loaded 13 s/frame), leaving the scene-cut frames to be paid for by the idle-machine speed-up. If the
idle re-run measures <= 8 s/frame, `interval_s=180` fits. Interview recordings are mostly one or two
tiles, which are the cheap cases (4-10 s), so these figures are conservative. If scene cuts are
frequent in the real recordings, cap the total with `max_frames` per video.

## Tests

`src/tests/video/tasks/scene_description/`:
- `scene_sampling_test.py`: an FFmpeg/libx264 synthetic video with cuts at 10 s and 20 s; cuts found
  between intervals, interval/cut coincidence, threshold off, `max_frames`, downscale, RGB order,
  in-memory `Video`.
- `scene_cv_metrics_test.py`: synthetic images with known brightness, contrast, blur ordering,
  colourfulness ordering, a camera-off quadrant, a dark textured region; Haar faces on the public
  tutorial photos.
- `scene_describe_test.py`: parsing and normalisation, rejection cases, repair-once logic with a
  fake backend, aggregation modes, consistency and change points.
- `scene_api_test.py`: end-to-end report on a synthetic video with a mocked VLM; real MLX VLM on the
  public `tutorial_images/three_people.jpg` (runs only on Apple silicon with the weights cached);
  transformers backend (CUDA hosts only).
