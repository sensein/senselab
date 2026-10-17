# Video facial analysis for gallery-view recordings

Module: `src/senselab/video/tasks/facial_analysis/` (user docs in its `doc.md`, defaults with
derivations in `data/defaults.yaml`).

## Goal

Per-person video-based affect and nonverbal cues from two-person Zoom gallery recordings (1080p
16 fps with two side-by-side tiles, plus one 4K 30 fps and one 720p 25 fps file; 27 to 98 min each,
about 27 h in total), cheaply enough that the corpus finishes in a few hours on an M1 Pro (32 GB,
MPS).

## Public API

```python
detect_gallery_tiles(video, sample_frames=None, config=None) -> GalleryLayout
extract_face_geometry(video, layout=None, fps=None, config=None, timeout_s=None) -> FrameTable
extract_facial_affect(video, layout=None, fps=None, device=None, config=None, timeout_s=None) -> FrameTable
detect_nonverbal_events(geometry, config=None) -> Dict[tile_id, NonverbalEvents]
summarize_tiles(geometry=None, affect=None, events=None, config=None) -> List[TileSummary]
link_tiles_to_speakers(mouth_series_per_tile, speaker_activity_per_speaker, fps, window_s=None) -> SpeakerLink
segments_to_activity(segments, times) -> np.ndarray
analyze_gallery_video(video, run_affect=True, single_tile=False, config=None, device=None) -> FacialAnalysis
load_defaults(overrides=None) -> dict
```

`FrameTable.tiles` holds one pandas DataFrame per tile; `FrameTable.to_long()` is Parquet-ready.

## Throughput

Benchmark: `uv run python scripts/bench_facial_analysis.py --threads 3` (builds the recording below
if `--video` is not given; `--out results.json` keeps the rows). Recording: 120 s, 1920x1080,
16 fps, two 952x536 tiles composed from two public clips (NIOSH "Science Speaks ... Liz,
Epidemiologist", public domain; "Interview with Barbara Firth", CC BY 3.0). Host: M1 Pro 10-core,
32 GB, shared with five other agents' benchmarks.

**Every number below is load-contaminated** (1-minute load average beside each row on a 10-core
machine) and is an upper bound on cost, not a measurement of the hardware. Treat them as relative
comparisons; an idle-machine pass with the same command gives the absolute figures.

| Component | Rate | Sampled instants/s | Tile-frames/s | x real time | Hours for 27 h | Load 1m |
|---|---|---|---|---|---|---|
| Tiles (48 seeks) | - | - | - | about 2 s per file | negligible | 12.5 |
| Geometry, MediaPipe CPU, threads 3 | 8 fps | 52 | 104 | 6.5 | 4.1 | 12.5 |
| Geometry, MediaPipe CPU, threads 3 | 16 fps | 59 | 118 | 3.7 | 7.3 | 11.8 |
| Affect, py-feat MPS, batch 16 | 2 fps | 12.8 | 25.6 | 6.4 | 4.2 | 9.9 |
| Affect, py-feat MPS, batch 16 | 4 fps | 11.5 | 23.0 | 2.9 | 9.4 | 9.0 |
| Decode only (PyAV, scaled), 1080p16 | 8 / 16 fps | - | - | 46 / 25 | 0.6 / 1.1 | 8.1 |
| Decode only (PyAV, scaled), 4K30 (synthetic upscale) | 8 / 16 fps | - | - | 21 / 13 | - | 8.0 |

Earlier probes on the same clip (outside the module, load 7 to 16):

| Probe | Result |
|---|---|
| MediaPipe 16 fps, tile width 480 / 960 / 1920 px | 3.96 / 4.60 / 4.39 x RT: cost does not depend on tile resolution |
| MediaPipe, one thread per tile | 5.17 vs 4.60 x RT: little gain, XNNPACK already multithreads |
| MediaPipe GPU delegate (SRGBA input) | 6.40 x RT but tile T1 detections fell from 1809 to 765 of 1920 frames; rejected |
| py-feat 4 fps on MPS vs CPU | 3.82 vs 0.32 x RT (CPU 12x slower) |
| py-feat model forward on MPS | about 28 ms per face; RetinaFace about 6-10 ms per tile; tile width 320 vs 480 changes total little |
| py-feat bf16 autocast on MPS | fails in a torchvision MPS kernel; fp16 weights fail in `torch.bmm` (dtype mismatch) |
| Worker overhead before fixes | 1.7 x RT vs 4.6 for the bare probe; a per-frame median fill (10 s) and full-tile grey conversion (9 s) per 120 s were removed |

**Recommended defaults** (in `data/defaults.yaml`): geometry 8 fps, affect 2 fps. Geometry runs on
CPU and affect on the GPU, so the two passes can run concurrently on one machine: about 4-4.5 h
of wall time for 27 h under this load. Blink counts want 16 fps geometry (7.3 h under load).
Decoding is never the bottleneck, including the 4K 30 fps file.

## Decisions

- **D-1 Workers decode the video themselves.** Frames never cross a process boundary; the parent
  sends the tile layout and receives per-tile arrays in one `.npz`. Decoding is sequential
  (no seeking) and selects frames on presentation time, so variable-frame-rate Zoom files are
  sampled on a uniform grid; only selected frames are converted and scaled (swscale), which is
  where the per-frame cost is. Memory is bounded by the per-tile scalar arrays (about 70 floats per
  tile-frame; 98 min at 16 fps and 2 tiles is under 60 MB).
- **D-2 MediaPipe in a subprocess venv (`mediapipe-face`).** `mediapipe==1.1.1` requires
  `opencv-contrib-python`, which installs the same `cv2` module as senselab's
  `opencv-python-headless`; installed together one overwrites the other. It also ships a second
  copy of FFmpeg's libavdevice (an ObjC class clash with PyAV's on macOS).
- **D-3 py-feat in a subprocess venv (`pyfeat`).** It pulls xgboost, skops, timm, celluloid, plotly
  and `torchcodec>=0.11`, and its weights are research-only; isolating it keeps both the
  dependency set and the licence out of the main environment. `Detectorv2` is assembled from its
  parts with `identity_model=None` (ArcFace is not needed) so no unpinned download happens.
- **D-4 Pinning.** Face Landmarker: the versioned bucket URL (`float16/1`) plus a SHA-256 checked on
  download. py-feat: `hf_hub_download(..., revision=<40-hex SHA>)` for the single file each repo
  needs (`face_multitask_v28.safetensors` at `d710b6f4...`, `retinaface_r34/model.safetensors` at
  `15191e03...`). `resolve_model` is not used because it snapshots the whole multitask repository
  (five checkpoint generations, about 800 MB) for one 167 MB file; the file is on
  `RAW_LOAD_EXCEPTIONS` in `hf_load_coverage_test.py` with that reason.
- **D-5 Tile detection is XY-cut on canvas-dark lines**, not a single column profile, so two-over-one
  and grid layouts work. Reference extents come from the per-pixel maximum over the frames sharing
  the modal layout (a letterboxed tile is otherwise trimmed in every frame), then equalised to the
  largest tile within 10 %. Analysis width 960: a 4 px gutter between 1280x720 tiles merged at
  640 px. Timeline kinds `reference` / `subset` / `different` separate a camera-off or dark tile
  (subset) from screen share or speaker view (different): in the development clip, fades in one
  source made 13 of 48 samples lose a tile, which a single "matches" flag reported as layout
  changes.
- **D-6 torch is imported before feat in the py-feat worker.** With `OMP_NUM_THREADS` set (or
  `torch.set_num_threads`), importing feat first (it imports xgboost, which brings a second OpenMP
  runtime) and then building the model segfaulted in timm's `trunc_normal_` every time; importing torch first never did.
  Reproduced with `OMP_NUM_THREADS=3 python -c "import xgboost; import torch, timm;
  timm.create_model('convnextv2_tiny')"` (exit 139), and the same with torch imported first
  (exit 0). Importing scikit-learn alone first did not crash.
- **D-7 Pose sign conventions match py-feat** (pitch up positive, yaw toward the subject's right
  positive). Checked on frames: the MediaPipe yaw minimum (-43.7 before negation) is a face turned
  to image left, the maximum a near-frontal face; after the convention change both backends report
  positive median yaw for the same two subjects (MediaPipe +20.8 / +34.0, py-feat +13.4 / +27.0).
- **D-8 Eye vertical offset is normalised by eye width, not lid gap.** Lid-gap normalisation went
  to 160 eye-heights during blinks. The iris centre sits 0.10-0.17 eye widths above the corner line
  at rest in the clip; `eye_vertical_neutral` removes 0.10.
- **D-9 Blinks are rises over a rolling-median baseline.** A fixed 0.35 floor read the subject with
  glasses (resting eyeBlink about 0.37) as one long closure, so zero blinks; the baseline-relative
  rule finds 25 in the unobstructed face. The subject with glasses still yields none (largest rise
  0.27): a recorded limitation, not tuned away.
- **D-10 Linking confidence is a circular-shift p-value.** With two strictly alternating speakers
  the activity tracks are complements, so random mouth motion produced a total margin of 0.56 in
  the unit test; the shift null (200 shifts, at least 10 s) reports it as not significant while the
  true assignment gets p < 0.05.
- **D-11 Defaults 8 fps geometry and 2 fps affect** from the throughput table, with the 0.5-4 Hz
  gesture band clipped to 0.45 x fps so 8 fps stays valid.

## Merged tiles

Pilot (privacy-preserving, numbers only): a 1920x1080, 16 fps Zoom gallery recording with two
participants side by side, letterboxed (`near_black_fraction` median 0.432 per frame). The VLM saw
2 tiles in 7 of 8 frames and OpenCV counted 2 faces per frame (median), but `detect_gallery_tiles`
returned one tile, x=7 y=277 w=1906 h=526, `reference` over all 48 samples: the two ~953x526 tiles
with the gutter between them lost at the 960 px analysis width. 28 of the 30 corpus recordings
share this format.

`tiles.split` (in `data/defaults.yaml`) adds a second pass over each reference tile:

- **Candidate grids.** Equal `columns x rows` grids (up to `max_grid` = 4) whose cells are within
  `aspect_tolerance` = 0.05 of 16:9 or 4:3. The pilot region (1906/526 = 3.62) is 2x1 with cells
  1.81, 1.9 % off 16:9; a single letterboxed 16:9 speaker only admits 2x2, 3x3, 4x4. Zoom crops
  participant video to fill its tile, so 16:9 and 4:3 are the shapes that occur.
- **Seams.** 12 frames (`sample_frames`) are decoded at up to 1920 px (`seam_width`), the
  resolution at which a 1 px gutter in 1080p still exists. Each boundary is scored by the larger of
  two robust z-scores over the region's boundaries: the mean absolute row-wise difference across it,
  and the column-mean step over 1 and 3 columns. The median over frames suppresses moving content.
  `seam_min_z` = 8 is a floor; the binding threshold is `seam_dominance` = 3 times the 99th
  percentile outside the search window. Measured on an mpeg4-encoded 1920x1080 synthetic pair
  (noise tiles, 1 px gutter): macroblock edges at multiples of 16 px scored 13-16 (99th percentile
  13.6), the true boundary 193. Without the dominance term a 4x2 grid on block edges was accepted.
  The search half-window is 4 % of the region side (`seam_search_fraction`); dark columns up to
  8 px beside the seam (`seam_max_gutter_px`) belong to neither tile. A grid needs a seam at every
  internal boundary in every band of cells; the largest such grid wins.
- **Faces.** With no seam grid (identical backgrounds, no gutter), the Haar cascade runs on the 12
  frames (at most 960 px wide); a grid is accepted when every cell holds a face in at least half the
  frames (`face_min_fraction`) and at most 20 % of faces straddle a boundary (`face_max_straddle`).
  Both are unfitted priors. Two people sharing one 16:9 camera do not split: that region only admits
  2x2 and up, and its bottom cells hold no face.
- **Confidence.** `gutter` tiles: fraction of seam frames whose own score passes at the boundary
  (unsplit tiles: fraction of sampled frames whose estimate contains the tile). `aspect_split`
  tiles: minimum cell occupancy times one minus the straddle fraction.

Synthetic checks (all in `facial_analysis_test.py::TestMergedTiles` and the encoded-file test):
letterboxed 953x526 pairs with gutters 0, 1, 2, 4 px, distinct and identical mean colours, 3x3
binomial blur; 2x2 and 1x3 with 0-2 px gutters; the encoded pilot format. The only unsplit pair is
0 px with identical backgrounds and no faces, which carries no information; given one face per
half it splits as `aspect_split`.

## Licences

| Component | Licence |
|---|---|
| MediaPipe 1.1.1, `face_landmarker.task` float16 v1 | Apache-2.0 |
| py-feat 2.1.5 (code) | MIT |
| `py-feat/face_multitask_v2` weights | research-only (non-commercial), see the repo LICENSE |
| `py-feat/retinaface_r34` weights | see the repo LICENSE-NOTICE |
| Benchmark clips | NIOSH clip public domain; Barbara Firth clip CC BY 3.0 (not committed) |

EmotiEffLib (`enet_b0_8_va_mtl`, Apache-2.0) was scoped as a fallback affect backend and is not
implemented in this change; it would take face boxes from the geometry pass.

## Tests

- `src/tests/video/tasks/facial_analysis_test.py` (synthetic): 1, 2, 2x2 and two-over-one
  layouts, slivers, equalisation, timeline kinds, an encoded two-tile video with a screen-share
  window, resampled and chunked decoding, label masking, nods/shakes/noise/drift/gaps/8 fps,
  blinks over a high baseline, gaze proxy, worker pose and iris helpers, AV linking (assignment,
  unassigned extra speaker, unrelated motion not significant), config merge and summaries.
- `src/tests/video/tasks/facial_analysis_integration_test.py` (2 tests): a 12 s two-tile gallery
  composed from the CC BY 3.0 clip, through both workers. Skipped when the venv is not provisioned
  or the clip cannot be downloaded.

## Known limitations

- Merged-tile splitting is checked on synthetic noise tiles only; the pilot recording has not been
  re-run. A static vertical edge in a single speaker's background at exactly the 2x2 centre lines
  could read as a seam.

- Gaze proxy uncalibrated; nod, shake and blink thresholds are unfitted priors.
- One face per tile; a camera-off tile reads as no face, and its layout run as `subset`.
- The throughput table is load-contaminated (see above).
