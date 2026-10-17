"""Sequential frame sampling: one frame per interval plus frames at scene cuts.

The decoder walks the file once with PyAV, holding only the current frame, a small thumbnail of
the last compared frame, and the frames already kept. A frame is kept when it is the first at or
after the next interval boundary, or when its colour histogram differs from the previous compared
frame by more than ``scene_threshold``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal, Optional, Union

import numpy as np

from senselab.video.data_structures.video import Video

FrameReason = Literal["interval", "scene_cut"]

_THUMB_SIZE = (64, 36)
_HIST_BINS = 16


@dataclass
class SampledFrame:
    """One frame kept by :func:`sample_frames`.

    Attributes:
        timestamp_s: Presentation time of the frame in seconds.
        frame_index: Index of the frame in decode order.
        reason: ``"interval"`` or ``"scene_cut"``.
        scene_score: Histogram distance to the previous compared frame (0 for the first frame).
        image: RGB ``uint8`` array ``(H, W, 3)``, downscaled so its long side is at most ``long_side``.
        source_size: ``(width, height)`` of the decoded frame before downscaling.
    """

    timestamp_s: float
    frame_index: int
    reason: FrameReason
    scene_score: float
    image: np.ndarray = field(repr=False)
    source_size: tuple = (0, 0)


def histogram_distance(a: np.ndarray, b: np.ndarray, bins: int = _HIST_BINS) -> float:
    """Distance between the per-channel colour histograms of two RGB images.

    Half the L1 distance between normalised histograms, averaged over channels: 0 for identical
    colour distributions, 1 for disjoint ones.

    Args:
        a: RGB ``uint8`` image.
        b: RGB ``uint8`` image.
        bins: Histogram bins per channel.

    Returns:
        The distance in ``[0, 1]``.
    """
    total = 0.0
    for c in range(3):
        ha = np.bincount((a[..., c].ravel() >> (8 - int(np.log2(bins)))), minlength=bins).astype(np.float64)
        hb = np.bincount((b[..., c].ravel() >> (8 - int(np.log2(bins)))), minlength=bins).astype(np.float64)
        total += 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum()
    return float(total / 3.0)


def _target_size(width: int, height: int, long_side: Optional[int]) -> tuple:
    if long_side is None or max(width, height) <= long_side:
        return width, height
    scale = long_side / max(width, height)
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def _resize(image: np.ndarray, size: tuple) -> np.ndarray:
    import cv2

    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _iter_decoded(source: Union[str, os.PathLike, Video], keyframes_only: bool) -> Iterator[tuple]:
    """Yield ``(index, timestamp_s, frame)`` where ``frame`` is a PyAV frame or an RGB array."""
    path: Optional[Union[str, os.PathLike]]
    if isinstance(source, Video):
        path = source._file_path
        if path is None:
            fps = float(source.frame_rate)
            for i, frame in enumerate(source.frames):
                yield i, i / fps, np.ascontiguousarray(frame.cpu().numpy().astype(np.uint8))
            return
    else:
        path = source
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and time_base:
                ts = frame.pts * time_base
            elif fps:
                ts = i / fps
            else:
                ts = float(frame.time or 0.0)
            yield i, ts, frame


def _to_rgb(frame: Any, size: Optional[tuple] = None) -> np.ndarray:  # noqa: ANN401
    """Convert a PyAV frame (scaled by swscale) or an RGB array to an RGB array of ``size``."""
    if isinstance(frame, np.ndarray):
        return frame if size is None else _resize(frame, size)
    if size is None:
        return frame.to_ndarray(format="rgb24")
    return frame.to_ndarray(format="rgb24", width=size[0], height=size[1])


def _frame_size(frame: Any) -> tuple:  # noqa: ANN401
    if isinstance(frame, np.ndarray):
        return int(frame.shape[1]), int(frame.shape[0])
    return int(frame.width), int(frame.height)


def iter_sampled_frames(
    video: Union[str, os.PathLike, Video],
    interval_s: float = 60.0,
    scene_threshold: Optional[float] = 0.3,
    max_frames: Optional[int] = None,
    long_side: Optional[int] = 896,
    analysis_fps: float = 2.0,
    min_scene_gap_s: float = 2.0,
    keyframes_only: bool = False,
) -> Iterator[SampledFrame]:
    """Decode ``video`` once and yield the frames :func:`sample_frames` keeps, as they are found.

    Args:
        video: A file path or a ``Video``.
        interval_s: Keep the first frame at or after every multiple of this many seconds.
        scene_threshold: Keep a frame whose :func:`histogram_distance` to the previous compared
            frame exceeds this; ``None`` disables scene-cut sampling.
        max_frames: Stop after this many frames.
        long_side: Downscale kept frames so the longer side is at most this; ``None`` keeps full size.
        analysis_fps: Rate at which frames are thumbnailed and compared for scene cuts.
        min_scene_gap_s: Minimum time between a scene-cut frame and the previous kept frame.
        keyframes_only: Decode only keyframes (much faster; scene cuts and interval frames then land
            on keyframes).

    Yields:
        ``SampledFrame`` objects in time order.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    next_interval = 0.0
    next_compare = 0.0
    last_kept = -np.inf
    prev_thumb: Optional[np.ndarray] = None
    n_kept = 0
    compare_step = 1.0 / analysis_fps if analysis_fps > 0 else 0.0
    for index, ts, frame in _iter_decoded(video, keyframes_only):
        reason: Optional[FrameReason] = None
        score = 0.0
        if ts + 1e-6 >= next_interval:
            reason = "interval"
            while next_interval <= ts + 1e-6:
                next_interval += interval_s
        if scene_threshold is not None and ts + 1e-6 >= next_compare:
            next_compare = ts + compare_step
            thumb = _to_rgb(frame, _THUMB_SIZE)
            if prev_thumb is not None:
                score = histogram_distance(prev_thumb, thumb)
                if reason is None and score > scene_threshold and ts - last_kept >= min_scene_gap_s:
                    reason = "scene_cut"
            prev_thumb = thumb
        if reason is None:
            continue
        width, height = _frame_size(frame)
        image = _to_rgb(frame, _target_size(width, height, long_side))
        yield SampledFrame(
            timestamp_s=float(ts),
            frame_index=int(index),
            reason=reason,
            scene_score=float(score),
            image=image,
            source_size=(width, height),
        )
        last_kept = ts
        n_kept += 1
        if max_frames is not None and n_kept >= max_frames:
            return


def sample_frames(
    video: Union[str, os.PathLike, Video],
    interval_s: float = 60.0,
    scene_threshold: Optional[float] = 0.3,
    max_frames: Optional[int] = None,
    long_side: Optional[int] = 896,
    analysis_fps: float = 2.0,
    min_scene_gap_s: float = 2.0,
    keyframes_only: bool = False,
) -> List[SampledFrame]:
    """Sample one frame per interval plus frames at scene cuts, decoding the video once.

    Only the kept frames (downscaled to ``long_side``) and one 64x36 thumbnail are held in memory.

    Args:
        video: A file path or a ``Video``.
        interval_s: Keep the first frame at or after every multiple of this many seconds.
        scene_threshold: Histogram distance above which a frame counts as a scene cut; ``None``
            disables scene-cut sampling.
        max_frames: Stop after this many frames.
        long_side: Downscale kept frames so the longer side is at most this; ``None`` keeps full size.
        analysis_fps: Rate at which frames are compared for scene cuts.
        min_scene_gap_s: Minimum time between a scene-cut frame and the previous kept frame.
        keyframes_only: Decode only keyframes.

    Returns:
        The kept frames in time order.
    """
    return list(
        iter_sampled_frames(
            video,
            interval_s=interval_s,
            scene_threshold=scene_threshold,
            max_frames=max_frames,
            long_side=long_side,
            analysis_fps=analysis_fps,
            min_scene_gap_s=min_scene_gap_s,
            keyframes_only=keyframes_only,
        )
    )
