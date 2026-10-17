# Facial analysis of gallery-view recordings

## Task overview

Per-person facial geometry, nonverbal events and affect from video calls recorded in gallery view
(Zoom and similar): each participant occupies a fixed tile on a dark canvas. The recording is split
into tiles once, each tile is analysed as its own face stream, and tiles can be linked to diarized
speakers from mouth motion.

```python
from senselab.video.tasks.facial_analysis import (
    analyze_gallery_video, detect_gallery_tiles, extract_face_geometry, extract_facial_affect,
    detect_nonverbal_events, summarize_tiles, link_tiles_to_speakers, segments_to_activity,
)

layout = detect_gallery_tiles("call.mp4")                 # tiles T0..Tn + layout timeline
geometry = extract_face_geometry("call.mp4", layout)      # MediaPipe, default 8 fps
affect = extract_facial_affect("call.mp4", layout)        # py-feat Detectorv2, default 2 fps
events = detect_nonverbal_events(geometry)                # nods, shakes, blinks per tile
summaries = summarize_tiles(geometry, affect, events)

frame = geometry.tiles["T0"]                              # one pandas DataFrame per tile
geometry.to_long().to_parquet("geometry.parquet")         # all tiles, with a tile_id column

t = geometry.tiles["T0"]["time_s"].to_numpy()
activity = {spk: segments_to_activity(segs, t) for spk, segs in diarization.items()}
mouths = {tile: df["bs_jawOpen"].to_numpy() for tile, df in geometry.tiles.items()}
link = link_tiles_to_speakers(mouths, activity, fps=geometry.fps)
link.assignment, link.p_value
```

`analyze_gallery_video(path)` runs all of the above in order. Inputs are file paths or file-backed
`Video` objects; the video is always decoded from the file, front to back, inside the worker, so no
frames cross the process boundary and memory stays bounded for any length.

## Stages

| Stage | Function | Backend | Default rate |
|---|---|---|---|
| Tiles | `detect_gallery_tiles` | numpy, in process | 48 sampled frames |
| Geometry | `extract_face_geometry` | MediaPipe Face Landmarker (subprocess venv `mediapipe-face`) | 8 fps |
| Events | `detect_nonverbal_events` | scipy, in process | from geometry |
| Affect | `extract_facial_affect` | py-feat 2.1.5 `Detectorv2` (subprocess venv `pyfeat`) | 2 fps |
| Linking | `link_tiles_to_speakers` | scipy Hungarian, in process | geometry grid |

**Tiles.** Sampled frames are split recursively along rows and columns that are dark over nearly
their full extent (XY-cut), which handles one tile, side-by-side pairs, grids, and uneven
arrangements such as two over one. The most frequent layout is the reference; its extents are
re-estimated on the per-pixel maximum over the frames that share it and equalised across tiles, then
inset past the active-speaker border, with a mask over the bottom-left name label. Every sampled
frame is classified as `reference`, `subset` (a tile went dark, e.g. camera off) or `different`
(screen share, speaker view); `layout.layout_changes()` lists the `different` runs.

**Geometry columns.** `face_present`, `n_faces`, `head_pitch_deg` (positive up), `head_yaw_deg`
(positive toward the subject's right, i.e. image left), `head_roll_deg`, head translation, face box
in tile-normalised coordinates, iris offsets `eye_horizontal`/`eye_vertical` (eye widths), inner-lip
`mouth_aspect`, all 52 blendshapes as `bs_<name>`, tile `motion_energy` (mean absolute grey-level
frame difference, 0-1, label masked) and the gaze proxy `gaze_yaw_deg`, `gaze_pitch_deg`,
`toward_camera` (0-1). MediaPipe reports no face-confidence score; presence is binary.

**Affect columns.** `face_score` (0 when no face), `n_faces`, face box, head pose and gaze in
degrees (same sign conventions as geometry), seven emotion probabilities (`neutral`, `happy`, `sad`,
`surprise`, `fear`, `disgust`, `anger`), `valence` and `arousal` in [-1, 1], twenty AU probabilities
(`AU01` ... `AU43`). The largest face per tile is kept.

**Events.** Nods and shakes are runs of alternating extrema of band-passed (0.5-4 Hz) pitch or yaw
with a minimum swing and a dominance ratio over the other axis. Blinks are brief rises of the
eyeBlink blendshapes above a rolling-median baseline.

**Linking.** Mouth activity (rolling standard deviation of jawOpen) is correlated with each
speaker's smoothed activity; the Hungarian algorithm picks the assignment, and a circular-shift null
gives `p_value`. Use `p_value`, not `total_margin`, as the confidence.

## Configuration

Every threshold and rate is in `data/defaults.yaml` with its derivation; pass a partial mapping as
`config=` to any function to override it (deep-merged). `geometry.threads` / `affect.threads` cap a
worker's CPU threads on a shared host. Throughput is measured by `scripts/bench_facial_analysis.py`.

## Licences

- MediaPipe and the Face Landmarker `face_landmarker.task` (float16, v1): Apache-2.0. The file is
  downloaded from Google's model bucket and verified against a pinned SHA-256.
- py-feat code: MIT. **The `py-feat/face_multitask_v2` weights are released for research use
  only**; see that repository's LICENSE. RetinaFace weights come from `py-feat/retinaface_r34`.
  Both are fetched at pinned commit SHAs, recorded in `FrameTable.provenance`.

## Limitations

- The gaze proxy is uncalibrated (no screen or camera geometry).
- Blinks need about 16 fps; at the default 8 fps, rates are lower bounds. Blendshape blinks are weak
  behind glasses and on low-resolution video.
- Nod, shake and blink thresholds are unfitted priors.
- A tile showing two people is analysed for one face (MediaPipe `num_faces=1`; py-feat keeps the
  largest).
- Gutters narrower than about 3 px at the 960 px analysis width can merge tiles.
