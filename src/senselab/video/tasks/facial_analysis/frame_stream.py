"""Chunked, sequential frame decoding at a target rate with PyAV.

This module imports only ``numpy`` and ``av`` so that the subprocess-venv workers can import it
from the package directory without senselab installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]
"""A rectangle ``(x, y, width, height)`` in pixels."""


def probe_video(path: str) -> Dict[str, Any]:
    """Read a video's geometry and timing without decoding it.

    Args:
        path: The video file.

    Returns:
        ``width``, ``height``, ``fps`` (average frame rate), ``duration_s`` and ``n_frames``
        (``0`` when the container does not say).
    """
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or 0
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000
        return {
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": float(rate),
            "duration_s": duration,
            "n_frames": int(stream.frames or 0),
        }


def scaled_size(width: int, height: int, scale: float) -> Tuple[int, int]:
    """Return ``(width, height)`` scaled by ``scale`` and rounded to even numbers.

    Args:
        width: Source width.
        height: Source height.
        scale: Factor in ``(0, 1]``.

    Returns:
        The even-rounded output size.
    """
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def iter_frames(
    path: str,
    fps: Optional[float] = None,
    scale: float = 1.0,
    pixel_format: str = "rgb24",
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Decode a video once, front to back, yielding frames resampled to ``fps``.

    Every frame is decoded (inter-coded streams require it), but only the selected ones are
    converted and rescaled, which is where the per-frame cost lies. Selection follows presentation
    timestamps, so variable-frame-rate recordings are sampled on a uniform time grid: the first
    frame at or after each grid instant ``k / fps`` is kept, and a frame is never emitted twice.

    Args:
        path: The video file.
        fps: Target rate. ``None`` keeps every frame.
        scale: Spatial scale applied during colour conversion (swscale), in ``(0, 1]``.
        pixel_format: PyAV output format, ``"rgb24"`` or ``"gray"``.
        start_s: Seek here before decoding.
        end_s: Stop after this time.

    Yields:
        ``(time_s, frame)`` with ``frame`` shaped ``(H, W, 3)`` for RGB or ``(H, W)`` for gray.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        out_w, out_h = scaled_size(stream.codec_context.width, stream.codec_context.height, scale)
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        if start_s > 0:
            container.seek(int(start_s / time_base), stream=stream, backward=True, any_frame=False)
        step = 1.0 / fps if fps else 0.0
        next_t = start_s
        tolerance = 1e-4
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = frame.pts * time_base
            if t + tolerance < start_s:
                continue
            if end_s is not None and t > end_s:
                break
            if step:
                if t + tolerance < next_t:
                    continue
                next_t = (math.floor(t / step + tolerance) + 1) * step
            yield t, frame.to_ndarray(format=pixel_format, width=out_w, height=out_h)


def iter_chunks(
    path: str,
    chunk_size: int,
    fps: Optional[float] = None,
    scale: float = 1.0,
    pixel_format: str = "rgb24",
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Group :func:`iter_frames` into fixed-size chunks so memory stays bounded.

    Args:
        path: The video file.
        chunk_size: Frames per chunk (the last chunk may be shorter).
        fps: Target rate, as in :func:`iter_frames`.
        scale: Spatial scale, as in :func:`iter_frames`.
        pixel_format: As in :func:`iter_frames`.

    Yields:
        ``(times, frames)``: a ``(n,)`` float array and a ``(n, H, W[, 3])`` uint8 array.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    times: List[float] = []
    frames: List[np.ndarray] = []
    for t, frame in iter_frames(path, fps=fps, scale=scale, pixel_format=pixel_format):
        times.append(t)
        frames.append(frame)
        if len(frames) == chunk_size:
            yield np.asarray(times), np.stack(frames)
            times, frames = [], []
    if frames:
        yield np.asarray(times), np.stack(frames)


def sample_frames(
    path: str, times_s: Sequence[float], scale: float = 1.0, pixel_format: str = "rgb24"
) -> List[Tuple[float, np.ndarray]]:
    """Decode the first frame at or after each requested time, seeking between requests.

    Args:
        path: The video file.
        times_s: Requested times, in seconds; sorted internally.
        scale: Spatial scale.
        pixel_format: As in :func:`iter_frames`.

    Returns:
        ``(time_s, frame)`` per request that the stream could satisfy, in time order.
    """
    import av

    out: List[Tuple[float, np.ndarray]] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        out_w, out_h = scaled_size(stream.codec_context.width, stream.codec_context.height, scale)
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0
        for target in sorted(float(t) for t in times_s):
            container.seek(int(max(target, 0.0) / time_base), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = frame.pts * time_base
                if t + 1e-4 >= target:
                    out.append((t, frame.to_ndarray(format=pixel_format, width=out_w, height=out_h)))
                    break
    return out


def crop_tile(frame: np.ndarray, box: Box, scale: float = 1.0, masks: Sequence[Box] = ()) -> np.ndarray:
    """Cut one tile out of a (possibly rescaled) frame and blank its masked regions.

    Args:
        frame: ``(H, W[, C])`` frame decoded at ``scale``.
        box: The tile in *source* pixels.
        scale: The scale ``frame`` was decoded at.
        masks: Rectangles in *tile-relative source* pixels to fill with the tile's mean value.

    Returns:
        A contiguous copy of the tile.
    """
    x, y, w, h = (int(round(v * scale)) for v in box)
    tile = np.array(frame[y : y + h, x : x + w], copy=True)
    if masks:
        coarse = tile[::8, ::8]
        fill = coarse.reshape(-1, coarse.shape[-1]).mean(axis=0) if tile.ndim == 3 else coarse.mean()
        for mx, my, mw, mh in masks:
            sx, sy = int(round(mx * scale)), int(round(my * scale))
            tile[sy : sy + int(round(mh * scale)), sx : sx + int(round(mw * scale))] = fill.astype(tile.dtype)
    return tile


def tile_placement(tile: Dict[str, Any], t: float) -> Optional[Tuple[str, Box, List[Box]]]:
    """Where a worker tile is analysed at time ``t``.

    Args:
        tile: A worker tile: ``tile_id``, ``box``, ``masks`` and, for a person track, ``segments`` of
            ``[start_s, end_s, region_id, box, masks]`` in time order.
        t: Frame time in seconds.

    Returns:
        ``(region_key, box, masks)``; ``None`` when a person track is not shown at ``t``.
    """
    segments = tile.get("segments")
    if segments is None:
        return tile["tile_id"], tuple(tile["box"]), list(tile.get("masks", []))  # type: ignore[return-value]
    for start, end, region_id, box, masks in segments:
        if start - 1e-6 <= t <= end + 1e-6:
            return str(region_id), tuple(box), list(masks)  # type: ignore[return-value]
        if start > t:
            break
    return None


def widest_box(tiles: Sequence[Dict[str, Any]]) -> int:
    """The widest box among worker tiles and their segments, in source pixels."""
    widths = [int(t["box"][2]) for t in tiles]
    widths += [int(s[3][2]) for t in tiles for s in t.get("segments") or []]
    return max(widths)
