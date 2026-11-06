# Standardize a video's encoding

## Task overview

Re-encode (or, when it already conforms, remux) a video file into one fixed, seek-friendly H.264/AAC encoding
with a predictable GOP, and return the output path together with a provenance record. Every target
(resolution, frame rate, audio rate, channels) defaults to "keep the source", so the default call produces the
*archival standard*; a model name or explicit targets produce a model-ready input on the fly.

Adapted from the [BBQS encoding-helper](https://github.com/brain-bbqs/encoding-helper) "recommended" encode
(MIT licence): H.264 High, `-preset slow -crf 23`, `yuv420p`, AAC, `+faststart`, metadata stripped.

| Choice | encoding-helper | here | why |
|---|---|---|---|
| Frame rate | assumes ~30 fps | **native** (`-fps_mode passthrough`, no `-r`) | timestamps and frame counts are evidence; resample only on request |
| B-frames | High-profile default | **none** (`bframes=0`) | decode order equals display order; a cut at a keyframe is exact |
| GOP | `keyint=30` | **`round(fps)` frames**, `min-keyint` equal, scene-cut off | exactly one second of the native rate, keyframes at indices `0, K, 2K, ...` |
| Audio | AAC 128k | **rate and channels kept**; AAC copied, else 192 kbit/s | no resampling unless a target asks for it |
| Rotation | n/a | applied (upright pixels, no display matrix) | models ignore the matrix |

## Usage

```python
from senselab.video.tasks.standardize_encoding import standardize_video_encoding

# archival standard: everything native
result = standardize_video_encoding("in.mov", "out.mp4")
result.ok, result.provenance["mode"], result.provenance["check"]

# on the fly, for a model: 32-pixel grid, fast preset, GPU if present, cached by content
result = standardize_video_encoding("in.mov", model="qwen3-vl", profile="transient", backend="auto", cache_dir="/scratch/cache")

# explicit targets
standardize_video_encoding("in.mov", "out.mp4", resolution=(640, 640), resolution_mode="pad", fps=10, audio_rate=16000, audio_channels=1)
```

`ffmpeg` and `ffprobe` (with `libx264`; `h264_nvenc` for the GPU backend) are found on `PATH` or passed as `ffmpeg=`/`ffprobe=`.
`scripts/install-ffmpeg.sh` installs a build.

## Targets, profiles, backends

* **resolution**: `None` keeps the size (odd sizes are cropped to even); an `int` caps the longest side (never upscales);
  `(W, H)` with `resolution_mode` `fit` (inside, aspect kept, never upscaled), `pad` (fit then black bars to exactly `W x H`),
  `crop` (cover then centre-crop) or `stretch`. `resolution_multiple` floors sizes to a patch grid (28 for Qwen2.5-VL, 32 for Qwen3-VL).
* **fps**: `None` keeps the source timestamps. A number resamples to a constant rate by dropping or duplicating the nearest
  frame (`fps_method="drop-dup"`); the GOP becomes `round(fps)`. For a VFR source without a target the GOP is `round(average fps)`
  frames, which is not exactly one second everywhere; the provenance records `vfr`.
* **audio_rate / audio_channels**: `None` keeps them; otherwise the first audio stream is resampled/remixed to AAC.
* **profile**: `archival` (`-preset slow`) or `transient` (`-preset veryfast`, NVENC `p4`). `preset=` overrides.
* **backend**: `libx264` (reference, bit-reproducible for a given ffmpeg build and thread count), `h264_nvenc` (NVDEC decode,
  NVENC encode with `-cq`; its output differs from x264 at the same CRF, so quality should be checked before archival use),
  or `auto`. The backend is recorded in the provenance.

## Fast path and cache

A source that is already H.264 `yuv420p`, upright, B-frame free, even-sized and keyframed exactly every `round(fps)` frames
is **remuxed** (video stream copied, metadata stripped, `+faststart`): `mode="remux"`, or `"audio-only"` when just the audio
needs encoding. Remuxing keeps the source's own quality and bitrate. Long-GOP camera files never match and are re-encoded.
With `cache_dir`, results are content addressed by `sha256(source sha256, resolved parameters, ffmpeg version)`; a hit returns
without encoding. Entries are never evicted.

## Provenance

`provenance` holds: `mode`, `backend`, `profile`, `source` (size, sha256, xxh3 when installed), `output` (size, hashes, ratio),
`avg_frame_rate`, `r_frame_rate`, `fps`, `vfr`, `duration`, `frames_decoded`, geometry, `rotation`, `audio`, `settings` (every
encode parameter, including the targets), `ffmpeg_version`, timings, `cache`, and `check`: the output is decoded again and
tested for frame count, timestamps (1 ms), keyframes exactly at multiples of the GOP, no B-frames, geometry and audio rate,
channels and duration (50 ms). `ok` is true when all pass.

## Model input specs

`MODEL_INPUT_SPECS` declares, per model, whether it needs specific dimensions, the patch multiple, preferred frame rate and audio
format, and the source of each claim; `verified=False` marks entries not read from a processor config. `model="name"` derives the
targets; explicit arguments win. Frame sampling stays with each model's processor.

## Limits

Only the first video and first audio stream are kept (other streams, chapters and metadata are dropped). 10-bit and full-range
sources are converted to 8-bit limited-range `yuv420p`. No VMAF is computed here (the ffmpeg build may lack `libvmaf`).
