"""Public API for per-person facial analysis of gallery-view video recordings."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from senselab.video.tasks.facial_analysis import backends
from senselab.video.tasks.facial_analysis.frame_stream import probe_video
from senselab.video.tasks.facial_analysis.results import (
    BlinkEvent,
    FrameTable,
    GalleryLayout,
    HeadGestureEvent,
    SpeakerLink,
    TileSummary,
)
from senselab.video.tasks.facial_analysis.signals import (
    detect_blinks,
    detect_head_gestures,
    gaze_proxy,
)
from senselab.video.tasks.facial_analysis.signals import (
    link_tiles_to_speakers as _link_tiles_to_speakers,
)
from senselab.video.tasks.facial_analysis.tiles import detect_gallery_tiles_in_file, single_tile_layout

VideoSource = Union[str, os.PathLike, Any]
"""A video file path, or a file-backed ``senselab.video.data_structures.Video``."""

_DEFAULTS_PATH = Path(__file__).parent / "data" / "defaults.yaml"


def load_defaults(overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The packaged defaults, with ``overrides`` deep-merged over them.

    Args:
        overrides: Partial mapping in the same shape as ``data/defaults.yaml``.

    Returns:
        The merged configuration.
    """
    config = yaml.safe_load(_DEFAULTS_PATH.read_text())

    def merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> None:
        for key, value in extra.items():
            if isinstance(value, Mapping) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    if overrides:
        merge(config, overrides)
    return config


def _video_path(video: VideoSource) -> str:
    path = getattr(video, "_file_path", None) if not isinstance(video, (str, os.PathLike)) else video
    if path is None:
        raise ValueError("facial analysis decodes from the file; pass a path or a file-backed Video")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    return str(path)


def detect_gallery_tiles(
    video: VideoSource, sample_frames: Optional[int] = None, config: Optional[Mapping[str, Any]] = None
) -> GalleryLayout:
    """Infer the fixed participant tiles of a gallery-view recording.

    Frames sampled uniformly across the recording are split recursively along rows and columns that
    are dark over almost their whole extent (the canvas gutters), which handles 1, 2 or N tiles in
    any grid. The most frequent per-frame layout is the reference; its tile extents are refined on
    the per-pixel maximum over the frames that share it, inset past the active-speaker border, and
    given a mask over the bottom-left name label. A tile whose aspect ratio is that of an equal grid
    of 16:9 or 4:3 cells is split at a seam found at full resolution (``derivation="gutter"``), or,
    without a seam, when faces occupy every cell (``derivation="aspect_split"``). Every sampled frame
    is classified against the reference, so screen share, speaker view and camera-off tiles show up
    in the timeline.

    Args:
        video: The recording.
        sample_frames: Frames to sample; ``tiles.sample_frames`` from the defaults when ``None``.
        config: Overrides of the defaults, in their shape.

    Returns:
        Tiles ``T0..Tn`` in reading order, with the layout timeline.
    """
    tiles_config = load_defaults(config)["tiles"]
    count = int(sample_frames if sample_frames is not None else tiles_config["sample_frames"])
    return detect_gallery_tiles_in_file(_video_path(video), count, tiles_config)


def _layout_for(path: str, layout: Optional[GalleryLayout], config: Dict[str, Any]) -> GalleryLayout:
    if layout is not None:
        return layout
    tiles_config = config["tiles"]
    return detect_gallery_tiles_in_file(path, int(tiles_config["sample_frames"]), tiles_config)


def _timing(status: Dict[str, Any]) -> Dict[str, float]:
    wall = float(status.get("process_wall_s", status.get("wall_s", 0.0)))
    video_s = float(status.get("video_s", 0.0))
    frames = float(status.get("frames", 0))
    return {
        "wall_s": wall,
        "video_s": video_s,
        "x_realtime": video_s / wall if wall > 0 else 0.0,
        "frames_per_s": frames / wall if wall > 0 else 0.0,
    }


def extract_face_geometry(
    video: VideoSource,
    layout: Optional[GalleryLayout] = None,
    fps: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    timeout_s: Optional[float] = None,
) -> FrameTable:
    """Per-tile face geometry with MediaPipe Face Landmarker.

    The video is decoded once, sequentially, at ``fps``; each tile (name label masked) goes through
    its own landmarker in video mode. Columns per frame: ``face_present``, ``n_faces``, head
    ``pitch/yaw/roll`` in degrees and translation from the facial transformation matrix, the face box
    in tile-normalised coordinates, iris offsets, mouth aspect ratio, all 52 blendshapes as
    ``bs_<name>``, tile ``motion_energy`` (mean absolute frame difference, 0-1), and the
    toward-camera gaze proxy (``gaze_yaw_deg``, ``gaze_pitch_deg``, ``toward_camera``).

    Args:
        video: The recording.
        layout: Tiles to analyse; detected when ``None``.
        fps: Sampling rate; ``geometry.fps`` from the defaults when ``None``.
        config: Overrides of the defaults.
        timeout_s: Wall-clock ceiling on the worker.

    Returns:
        One DataFrame per tile, with provenance and timing.
    """
    merged = load_defaults(config)
    path = _video_path(video)
    layout = _layout_for(path, layout, merged)
    geometry = dict(merged["geometry"])
    if fps is not None:
        geometry["fps"] = float(fps)
    status, arrays = backends.run_mediapipe(path, layout, geometry, timeout_s=timeout_s)
    names = status.get("blendshape_names") or []
    gaze = merged["gaze"]
    tables: Dict[str, pd.DataFrame] = {}
    for tile in layout.tiles:
        key = tile.tile_id
        if f"{key}/time_s" not in arrays:
            continue
        pose = arrays[f"{key}/pose"]
        eye = arrays[f"{key}/eye_offset"]
        box = arrays[f"{key}/face_box"]
        trans = arrays[f"{key}/translation"]
        frame = pd.DataFrame(
            {
                "time_s": arrays[f"{key}/time_s"],
                "face_present": arrays[f"{key}/face_present"],
                "n_faces": arrays[f"{key}/n_faces"].astype(np.int16),
                "head_pitch_deg": pose[:, 0],
                "head_yaw_deg": pose[:, 1],
                "head_roll_deg": pose[:, 2],
                "head_tx": trans[:, 0],
                "head_ty": trans[:, 1],
                "head_tz": trans[:, 2],
                "face_x": box[:, 0],
                "face_y": box[:, 1],
                "face_w": box[:, 2],
                "face_h": box[:, 3],
                "eye_horizontal": eye[:, 0],
                "eye_vertical": eye[:, 1],
                "mouth_aspect": arrays[f"{key}/mouth_aspect"],
                "motion_energy": arrays[f"{key}/motion_energy"],
            }
        )
        blends = arrays[f"{key}/blendshapes"]
        for i, name in enumerate(names):
            frame[f"bs_{name}"] = blends[:, i]
        gaze_yaw, gaze_pitch, score = gaze_proxy(
            frame["head_pitch_deg"].to_numpy(),
            frame["head_yaw_deg"].to_numpy(),
            frame["eye_horizontal"].to_numpy(),
            frame["eye_vertical"].to_numpy(),
            eye_gain_deg=gaze["eye_gain_deg"],
            sigma_deg=gaze["sigma_deg"],
            eye_vertical_neutral=gaze["eye_vertical_neutral"],
        )
        frame["gaze_yaw_deg"] = gaze_yaw.astype(np.float32)
        frame["gaze_pitch_deg"] = gaze_pitch.astype(np.float32)
        frame["toward_camera"] = score.astype(np.float32)
        tables[key] = frame
    return FrameTable(
        kind="geometry",
        fps=float(geometry["fps"]),
        layout=layout,
        tiles=tables,
        provenance={
            "backend": "mediapipe-face-landmarker",
            "mediapipe_version": status.get("mediapipe_version"),
            "model_url": backends.MEDIAPIPE_MODEL_URL,
            "model_sha256": backends.MEDIAPIPE_MODEL_SHA256,
            "decode_size": status.get("decode_size"),
            "config": {k: v for k, v in geometry.items() if k != "derivation"},
        },
        timing=_timing(status),
    )


def _default_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_PYFEAT_RENAMES = {
    "FaceRectX": "face_x",
    "FaceRectY": "face_y",
    "FaceRectWidth": "face_w",
    "FaceRectHeight": "face_h",
    "FaceScore": "face_score",
    "Pitch": "head_pitch_deg",
    "Roll": "head_roll_deg",
    "Yaw": "head_yaw_deg",
    "gaze_pitch": "gaze_pitch_deg",
    "gaze_yaw": "gaze_yaw_deg",
    "gaze_angle": "gaze_angle_deg",
}
_RADIAN_COLUMNS = ("Pitch", "Roll", "Yaw", "gaze_pitch", "gaze_yaw", "gaze_angle")


def extract_facial_affect(
    video: VideoSource,
    layout: Optional[GalleryLayout] = None,
    fps: Optional[float] = None,
    device: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    timeout_s: Optional[float] = None,
) -> FrameTable:
    """Per-tile facial affect with py-feat ``Detectorv2`` (research-only weights).

    Tile crops of each frame sampled at ``fps`` are batched through RetinaFace and the
    ``face_multitask_v2`` model; the largest face per tile is kept. Columns per frame: ``face_score``
    (0 without a face), ``n_faces``, the face box in tile pixels at the decode scale, head pose and
    gaze in degrees, seven emotion probabilities (lower-case names), ``valence`` and ``arousal`` in
    ``[-1, 1]``, and twenty AU probabilities (``AU01``..``AU43``).

    The model weights are released for research use only; see the module documentation.

    Args:
        video: The recording.
        layout: Tiles to analyse; detected when ``None``.
        fps: Sampling rate; ``affect.fps`` from the defaults when ``None``.
        device: ``"mps"``, ``"cuda"`` or ``"cpu"``; the best available when ``None``.
        config: Overrides of the defaults.
        timeout_s: Wall-clock ceiling on the worker.

    Returns:
        One DataFrame per tile, with provenance and timing.
    """
    merged = load_defaults(config)
    path = _video_path(video)
    layout = _layout_for(path, layout, merged)
    affect = dict(merged["affect"])
    if fps is not None:
        affect["fps"] = float(fps)
    status, arrays = backends.run_pyfeat(path, layout, affect, device or _default_device(), timeout_s=timeout_s)
    columns: List[str] = status["columns"]
    tables: Dict[str, pd.DataFrame] = {}
    for tile in layout.tiles:
        key = tile.tile_id
        if f"{key}/time_s" not in arrays:
            continue
        values = arrays[f"{key}/values"]
        frame = pd.DataFrame(values, columns=columns)
        for column in _RADIAN_COLUMNS:
            if column in frame:
                frame[column] = np.degrees(frame[column])
        frame = frame.rename(columns=_PYFEAT_RENAMES)
        frame = frame.rename(
            columns={c: c.lower() for c in frame.columns if c[:1].isupper() and not c.startswith("AU")}
        )
        frame.insert(0, "time_s", arrays[f"{key}/time_s"])
        frame["n_faces"] = frame["n_faces"].astype(np.int16)
        tables[key] = frame
    return FrameTable(
        kind="affect",
        fps=float(affect["fps"]),
        layout=layout,
        tiles=tables,
        provenance={
            "backend": "py-feat-detectorv2",
            "pyfeat_version": status.get("pyfeat_version"),
            "torch_version": status.get("torch_version"),
            "device": status.get("device"),
            "multitask": (
                f"{backends.PYFEAT_MULTITASK_REPO}@{backends.PYFEAT_MULTITASK_SHA}/{backends.PYFEAT_MULTITASK_FILE}"
            ),
            "retinaface": f"{backends.PYFEAT_RETINAFACE_REPO}@{backends.PYFEAT_RETINAFACE_SHA}",
            "weights_license": "research-only (py-feat face_multitask_v2 LICENSE)",
            "config": {k: v for k, v in affect.items() if k != "derivation"},
        },
        timing=_timing(status),
    )


class NonverbalEvents(BaseModel):
    """Head-gesture and blink events of one tile.

    Attributes:
        nods: Nod events.
        shakes: Shake events.
        blinks: Blink events.
    """

    nods: List[HeadGestureEvent] = Field(default_factory=list)
    shakes: List[HeadGestureEvent] = Field(default_factory=list)
    blinks: List[BlinkEvent] = Field(default_factory=list)


def detect_nonverbal_events(
    geometry: FrameTable, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, NonverbalEvents]:
    """Nods, shakes and blinks per tile from a geometry table.

    Args:
        geometry: Output of :func:`extract_face_geometry`.
        config: Overrides of the defaults (``gestures`` and ``blinks`` sections).

    Returns:
        ``tile_id -> events``.
    """
    merged = load_defaults(config)
    gestures = {k: v for k, v in merged["gestures"].items() if k != "derivation"}
    blinks = {k: v for k, v in merged["blinks"].items() if k != "derivation"}
    out: Dict[str, NonverbalEvents] = {}
    for tile_id, frame in geometry.tiles.items():
        head = detect_head_gestures(
            frame["time_s"].to_numpy(),
            frame["head_pitch_deg"].to_numpy(),
            frame["head_yaw_deg"].to_numpy(),
            geometry.fps,
            tile_id=tile_id,
            **gestures,
        )
        blink_events: List[BlinkEvent] = []
        if "bs_eyeBlinkLeft" in frame and "bs_eyeBlinkRight" in frame:
            blink_events = detect_blinks(
                frame["time_s"].to_numpy(),
                frame["bs_eyeBlinkLeft"].to_numpy(),
                frame["bs_eyeBlinkRight"].to_numpy(),
                tile_id=tile_id,
                **blinks,
            )
        out[tile_id] = NonverbalEvents(
            nods=[e for e in head if e.kind == "nod"],
            shakes=[e for e in head if e.kind == "shake"],
            blinks=blink_events,
        )
    return out


def summarize_tiles(
    geometry: Optional[FrameTable] = None,
    affect: Optional[FrameTable] = None,
    events: Optional[Mapping[str, NonverbalEvents]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> List[TileSummary]:
    """Per-tile summaries over the whole recording.

    Rates are per minute of face-present time (geometry frames with a face times the frame step).

    Args:
        geometry: Output of :func:`extract_face_geometry`.
        affect: Output of :func:`extract_facial_affect`.
        events: Output of :func:`detect_nonverbal_events`.
        config: Overrides of the defaults (``gaze.toward_threshold``).

    Returns:
        One summary per tile seen in either table.
    """
    threshold = float(load_defaults(config)["gaze"]["toward_threshold"])
    tile_ids = sorted(set(geometry.tiles if geometry else {}) | set(affect.tiles if affect else {}))
    out: List[TileSummary] = []
    for tile_id in tile_ids:
        summary: Dict[str, Any] = {"tile_id": tile_id, "duration_s": 0.0, "face_present_fraction": 0.0}
        if geometry is not None and tile_id in geometry.tiles:
            frame = geometry.tiles[tile_id]
            present = frame["face_present"].to_numpy(bool)
            times = frame["time_s"].to_numpy()
            summary["duration_s"] = float(times[-1] - times[0] + 1.0 / geometry.fps) if len(times) else 0.0
            summary["face_present_fraction"] = float(present.mean()) if len(present) else 0.0
            minutes = present.sum() / geometry.fps / 60.0
            if present.any():
                face = frame[present]
                if "bs_jawOpen" in face:
                    summary["mouth_open_mean"] = float(face["bs_jawOpen"].mean())
                summary["toward_camera_fraction"] = float((face["toward_camera"] >= threshold).mean())
            summary["motion_energy_mean"] = float(np.nanmean(frame["motion_energy"])) if len(frame) > 1 else None
            if events is not None and tile_id in events and minutes > 0:
                tile_events = events[tile_id]
                summary["blink_rate_per_min"] = len(tile_events.blinks) / minutes
                summary["nods_per_min"] = len(tile_events.nods) / minutes
                summary["shakes_per_min"] = len(tile_events.shakes) / minutes
        if affect is not None and tile_id in affect.tiles:
            frame = affect.tiles[tile_id]
            face = frame[frame["face_score"] > 0]
            skip = {"time_s", "n_faces", "face_x", "face_y", "face_w", "face_h"}
            summary["affect_means"] = {c: float(face[c].mean()) for c in face.columns if c not in skip and len(face)}
            if geometry is None or tile_id not in geometry.tiles:
                summary["face_present_fraction"] = float((frame["face_score"] > 0).mean()) if len(frame) else 0.0
        out.append(TileSummary(**summary))
    return out


def link_tiles_to_speakers(
    mouth_series_per_tile: Mapping[str, np.ndarray],
    speaker_activity_per_speaker: Mapping[str, np.ndarray],
    fps: float,
    window_s: Optional[float] = None,
    n_shifts: int = 200,
    min_shift_s: float = 10.0,
) -> SpeakerLink:
    """Assign tiles to diarized speakers by correlating mouth motion with speech activity.

    See :func:`senselab.video.tasks.facial_analysis.signals.link_tiles_to_speakers`; build the
    activity series with :func:`~senselab.video.tasks.facial_analysis.signals.segments_to_activity`
    on the geometry table's ``time_s``.

    Args:
        mouth_series_per_tile: ``tile_id -> jawOpen series`` on one time grid.
        speaker_activity_per_speaker: ``speaker -> 0/1 series`` on the same grid.
        fps: Sampling rate.
        window_s: Smoothing window; ``linking.window_s`` from the defaults when ``None``.
        n_shifts: Circular shifts drawn for the null distribution of the best total correlation.
        min_shift_s: Smallest shift, so the null excludes near-aligned series.

    Returns:
        The assignment with its correlations and margins.
    """
    window = float(window_s if window_s is not None else load_defaults()["linking"]["window_s"])
    return _link_tiles_to_speakers(
        mouth_series_per_tile,
        speaker_activity_per_speaker,
        fps,
        window_s=window,
        n_shifts=n_shifts,
        min_shift_s=min_shift_s,
    )


class FacialAnalysis(BaseModel):
    """Everything :func:`analyze_gallery_video` produces.

    Attributes:
        layout: The tile layout.
        geometry: MediaPipe per-frame table.
        affect: py-feat per-frame table, when run.
        events: Nods, shakes and blinks per tile.
        summaries: Per-tile summaries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layout: GalleryLayout
    geometry: FrameTable
    affect: Optional[FrameTable] = None
    events: Dict[str, NonverbalEvents]
    summaries: List[TileSummary]


def analyze_gallery_video(
    video: VideoSource,
    run_affect: bool = True,
    single_tile: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> FacialAnalysis:
    """Tiles, geometry, events, optional affect, and summaries for one recording.

    Args:
        video: The recording.
        run_affect: Also run the py-feat affect pass.
        single_tile: Treat the whole frame as one tile instead of detecting a gallery.
        config: Overrides of the defaults.
        device: Device for the affect pass.
        timeout_s: Wall-clock ceiling on each worker.

    Returns:
        The combined result.
    """
    path = _video_path(video)
    if single_tile:
        info = probe_video(path)
        layout = single_tile_layout(info["width"], info["height"])
    else:
        layout = detect_gallery_tiles(path, config=config)
    geometry = extract_face_geometry(path, layout=layout, config=config, timeout_s=timeout_s)
    affect = (
        extract_facial_affect(path, layout=layout, device=device, config=config, timeout_s=timeout_s)
        if run_affect
        else None
    )
    events = detect_nonverbal_events(geometry, config=config)
    summaries = summarize_tiles(geometry, affect, events, config=config)
    return FacialAnalysis(layout=layout, geometry=geometry, affect=affect, events=events, summaries=summaries)
