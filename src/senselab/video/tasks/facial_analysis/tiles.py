"""Gallery-tile detection from dark-gutter intensity profiles."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from senselab.video.tasks.facial_analysis.frame_stream import probe_video, sample_frames
from senselab.video.tasks.facial_analysis.results import GalleryLayout, LayoutSegment, Rect, TileBox


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` runs where ``mask`` is True."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def _xy_cut(
    dark: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    gutter_fraction: float,
    min_w: int,
    min_h: int,
    depth: int = 0,
) -> List[Rect]:
    """Recursively split a region along fully dark rows or columns.

    Args:
        dark: ``(H, W)`` boolean mask of canvas-dark pixels.
        x0: Region left (inclusive).
        y0: Region top (inclusive).
        x1: Region right (exclusive).
        y1: Region bottom (exclusive).
        gutter_fraction: Fraction of a line that must be dark for it to count as a gutter.
        min_w: Smallest tile width kept.
        min_h: Smallest tile height kept.
        depth: Recursion depth guard.

    Returns:
        Leaf rectangles ``(x, y, w, h)`` trimmed of dark margins.
    """
    region = dark[y0:y1, x0:x1]
    if region.size == 0 or depth > 12:
        return []
    row_gutter = region.mean(axis=1) >= gutter_fraction
    col_gutter = region.mean(axis=0) >= gutter_fraction
    rows = [(a, b) for a, b in _runs(~row_gutter) if b - a >= min_h]
    cols = [(a, b) for a, b in _runs(~col_gutter) if b - a >= min_w]
    if not rows or not cols:
        return []
    ty0, ty1 = rows[0][0], rows[-1][1]
    tx0, tx1 = cols[0][0], cols[-1][1]
    if len(rows) > 1:
        out: List[Rect] = []
        for a, b in rows:
            out += _xy_cut(dark, x0, y0 + a, x1, y0 + b, gutter_fraction, min_w, min_h, depth + 1)
        return out
    if len(cols) > 1:
        out = []
        for a, b in cols:
            out += _xy_cut(dark, x0 + a, y0, x0 + b, y1, gutter_fraction, min_w, min_h, depth + 1)
        return out
    return [(x0 + tx0, y0 + ty0, tx1 - tx0, ty1 - ty0)]


def _reading_order(rects: Sequence[Rect]) -> List[Rect]:
    """Sort rectangles top-to-bottom by row band, then left-to-right."""
    if not rects:
        return []
    heights = np.median([r[3] for r in rects])
    return sorted(rects, key=lambda r: (round((r[1] + r[3] / 2) / max(heights, 1.0)), r[0]))


def estimate_frame_layout(
    gray: np.ndarray,
    dark_level: float = 0.16,
    gutter_fraction: float = 0.97,
    min_tile_fraction: float = 0.12,
) -> List[Rect]:
    """Estimate tile rectangles in one grayscale frame.

    Args:
        gray: ``(H, W)`` frame, uint8 or float in ``[0, 1]``.
        dark_level: Pixels below this (0-1 scale) count as canvas.
        gutter_fraction: Fraction of a row/column that must be canvas for it to be a gutter.
        min_tile_fraction: Smallest tile side as a fraction of the frame side.

    Returns:
        Tile rectangles in this frame's pixels, in reading order.
    """
    g = gray.astype(np.float32)
    if gray.dtype == np.uint8:
        g /= 255.0
    h, w = g.shape
    dark = g < dark_level
    rects = _xy_cut(
        dark,
        0,
        0,
        w,
        h,
        gutter_fraction,
        max(2, int(min_tile_fraction * w)),
        max(2, int(min_tile_fraction * h)),
    )
    return _reading_order(rects)


def _iou(a: Rect, b: Rect) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _same_layout(a: Sequence[Rect], b: Sequence[Rect], match_iou: float) -> bool:
    return len(a) == len(b) and all(_iou(p, q) >= match_iou for p, q in zip(a, b))


def _signature(rects: Sequence[Rect], width: int, height: int) -> Tuple[Tuple[int, int, int, int], ...]:
    """Rectangles quantised to 2 % of the frame, for grouping near-identical estimates."""
    qx, qy = width / 50.0, height / 50.0
    return tuple((round(r[0] / qx), round(r[1] / qy), round(r[2] / qx), round(r[3] / qy)) for r in rects)


def _classify(rects: Sequence[Rect], reference: Sequence[Rect], match_iou: float) -> str:
    """``reference``, ``subset`` (only reference tiles, some missing) or ``different``."""
    if _same_layout(rects, reference, match_iou):
        return "reference"
    if len(rects) < len(reference) and all(any(_iou(r, q) >= match_iou for q in reference) for r in rects):
        return "subset"
    return "different"


def layout_from_estimates(
    estimates: Sequence[Tuple[float, List[Rect]]],
    frame_width: int,
    frame_height: int,
    match_iou: float = 0.90,
    label_mask_width_fraction: float = 0.30,
    label_mask_height_fraction: float = 0.08,
    border_inset_fraction: float = 0.012,
    reference: Optional[Sequence[Rect]] = None,
) -> GalleryLayout:
    """Fold per-frame estimates into a reference layout and a layout timeline.

    Args:
        estimates: ``(time_s, rects)`` per sampled frame, rects in source pixels.
        frame_width: Source width.
        frame_height: Source height.
        match_iou: Per-tile IoU at or above which two estimates are the same layout.
        label_mask_width_fraction: Name-label mask width, as a fraction of tile width.
        label_mask_height_fraction: Name-label mask height, as a fraction of tile height.
        border_inset_fraction: Inset applied on every side, as a fraction of the tile's short side.
        reference: Reference rectangles to use instead of the modal estimate's.

    Returns:
        The layout. With no non-empty estimate the whole frame is one tile.
    """
    if reference is None:
        reference = modal_layout(estimates, frame_width, frame_height, match_iou)
    if not reference:
        reference = [(0, 0, frame_width, frame_height)]

    tiles = []
    for index, (x, y, w, h) in enumerate(reference):
        inset = int(math.ceil(border_inset_fraction * min(w, h)))
        x, y, w, h = x + inset, y + inset, w - 2 * inset, h - 2 * inset
        mh = int(round(label_mask_height_fraction * h))
        mw = int(round(label_mask_width_fraction * w))
        masks = [(0, h - mh, mw, mh)] if mh > 0 and mw > 0 else []
        tiles.append(TileBox(tile_id=f"T{index}", x=x, y=y, width=w, height=h, masks=masks))

    timeline: List[LayoutSegment] = []
    matches = 0
    for t, rects in estimates:
        kind = _classify(rects, reference, match_iou)
        matches += kind == "reference"
        last = timeline[-1] if timeline else None
        if last is not None and last.kind == kind and _same_layout(rects, last.boxes, match_iou):
            last.end_s = t
            last.n_samples += 1
        else:
            timeline.append(LayoutSegment(start_s=t, end_s=t, n_samples=1, boxes=list(rects), kind=kind))  # type: ignore[arg-type]
    return GalleryLayout(
        frame_width=frame_width,
        frame_height=frame_height,
        tiles=tiles,
        timeline=timeline,
        reference_fraction=matches / len(estimates) if estimates else 0.0,
        n_samples=len(estimates),
    )


def modal_layout(
    estimates: Sequence[Tuple[float, List[Rect]]], frame_width: int, frame_height: int, match_iou: float
) -> List[Rect]:
    """The most frequent per-frame layout, with each tile's median rectangle over its members.

    Args:
        estimates: ``(time_s, rects)`` per sampled frame.
        frame_width: Source width.
        frame_height: Source height.
        match_iou: Per-tile IoU for membership.

    Returns:
        The modal rectangles; empty when no estimate found a tile.
    """
    nonempty = [rects for _, rects in estimates if rects]
    if not nonempty:
        return []
    counts = Counter(_signature(r, frame_width, frame_height) for r in nonempty)
    modal = counts.most_common(1)[0][0]
    exemplar = next(r for r in nonempty if _signature(r, frame_width, frame_height) == modal)
    members = [r for r in nonempty if _same_layout(r, exemplar, match_iou)]
    return [
        tuple(int(round(v)) for v in np.median([m[i] for m in members], axis=0))  # type: ignore[misc]
        for i in range(len(exemplar))
    ]


def refine_with_projection(frames: Sequence[np.ndarray], coarse: Sequence[Rect], config: Dict[str, Any]) -> List[Rect]:
    """Re-estimate tile extents on the per-pixel maximum over frames that share a layout.

    A tile whose content is dark at its edges in any one frame is trimmed by a single-frame
    estimate; the maximum projection keeps every pixel that was ever lit, so the canvas alone stays
    dark. Each refined rectangle is matched to the coarse tile it overlaps most.

    Args:
        frames: Grayscale frames, all sharing ``coarse``'s layout, in one pixel grid.
        coarse: The layout in that grid.
        config: The ``tiles`` section of the defaults.

    Returns:
        Refined rectangles, one per coarse tile (the coarse one where refinement found no match).
    """
    if not frames:
        return list(coarse)
    projection = np.max(np.stack(frames), axis=0)
    refined = estimate_frame_layout(
        projection,
        dark_level=config["dark_level"],
        gutter_fraction=config["gutter_fraction"],
        min_tile_fraction=config["min_tile_fraction"],
    )
    out: List[Rect] = []
    for box in coarse:
        best = max(refined, key=lambda r: _iou(r, box), default=None)
        out.append(best if best is not None and _iou(best, box) >= 0.5 else box)
    return out


def equalize_tiles(rects: Sequence[Rect], tolerance: float = 0.1) -> List[Rect]:
    """Grow tiles that are slightly smaller than the largest to its size, about their centres.

    Gallery tiles are drawn at one size; a tile whose own video is letterboxed is estimated a few
    pixels small. A tile more than ``tolerance`` smaller on either side is left alone.

    Args:
        rects: Tile rectangles.
        tolerance: Largest relative shortfall that is equalised.

    Returns:
        The rectangles, equalised where within tolerance.
    """
    if len(rects) < 2:
        return list(rects)
    big_w = max(r[2] for r in rects)
    big_h = max(r[3] for r in rects)
    out: List[Rect] = []
    for x, y, w, h in rects:
        if w >= (1 - tolerance) * big_w and h >= (1 - tolerance) * big_h:
            cx, cy = x + w / 2, y + h / 2
            out.append((int(round(cx - big_w / 2)), int(round(cy - big_h / 2)), big_w, big_h))
        else:
            out.append((x, y, w, h))
    return out


def detect_gallery_tiles_in_file(path: str, sample_frames_n: int, config: Dict[str, Any]) -> GalleryLayout:
    """Sample frames across a file and estimate its gallery layout.

    Args:
        path: The video file.
        sample_frames_n: Frames to sample, spread uniformly over the recording.
        config: The ``tiles`` section of the defaults.

    Returns:
        The layout.
    """
    info = probe_video(path)
    width, height, duration = info["width"], info["height"], info["duration_s"]
    scale = min(1.0, float(config["analysis_width"]) / width)
    if duration > 0:
        times = np.linspace(0.01 * duration, 0.99 * duration, max(1, sample_frames_n))
    else:
        times = np.zeros(1)
    samples = sample_frames(path, times.tolist(), scale=scale, pixel_format="gray")
    local: List[Tuple[float, List[Rect]]] = []
    for t, gray in samples:
        rects = estimate_frame_layout(
            gray,
            dark_level=config["dark_level"],
            gutter_fraction=config["gutter_fraction"],
            min_tile_fraction=config["min_tile_fraction"],
        )
        local.append((float(t), rects))
    if not samples:
        return layout_from_estimates([], width, height)
    gh, gw = samples[0][1].shape
    coarse = modal_layout(local, gw, gh, config["match_iou"])
    members = [
        gray
        for (_, gray), (_, rects) in zip(samples, local)
        if coarse and _same_layout(rects, coarse, config["match_iou"])
    ]
    refined = equalize_tiles(refine_with_projection(members, coarse, config)) if coarse else []
    sx, sy = width / gw, height / gh

    def to_source(rects: Sequence[Rect]) -> List[Rect]:
        return [
            (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))) for x, y, w, h in rects
        ]

    return layout_from_estimates(
        [(t, to_source(rects)) for t, rects in local],
        width,
        height,
        match_iou=config["match_iou"],
        label_mask_width_fraction=config["label_mask_width_fraction"],
        label_mask_height_fraction=config["label_mask_height_fraction"],
        border_inset_fraction=config["border_inset_fraction"],
        reference=to_source(refined) if refined else None,
    )


def single_tile_layout(width: int, height: int, tile_id: Optional[str] = None) -> GalleryLayout:
    """A layout that treats the whole frame as one tile (no masks).

    Args:
        width: Frame width.
        height: Frame height.
        tile_id: Tile id; ``T0`` by default.

    Returns:
        The layout.
    """
    tile = TileBox(tile_id=tile_id or "T0", x=0, y=0, width=width, height=height)
    return GalleryLayout(
        frame_width=width, frame_height=height, tiles=[tile], timeline=[], reference_fraction=1.0, n_samples=0
    )
