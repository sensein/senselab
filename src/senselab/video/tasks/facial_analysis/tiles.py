"""Gallery-tile detection from dark-gutter intensity profiles, image seams and tile aspect ratios."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from senselab.video.tasks.facial_analysis.frame_stream import probe_video, sample_frames
from senselab.video.tasks.facial_analysis.results import GalleryLayout, LayoutSegment, Rect, TileBox

Derived = Tuple[Rect, str, Optional[float]]
"""A tile rectangle with its derivation (``gutter`` or ``aspect_split``) and confidence."""

FaceDetector = Callable[[Sequence[np.ndarray], Rect], List[List[Rect]]]
"""``(gray frames, region) -> face boxes per frame`` in the frames' pixels."""


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
    derivations: Optional[Sequence[Tuple[str, Optional[float]]]] = None,
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
        derivations: ``(derivation, confidence)`` per reference rectangle. A ``None`` confidence, and
            every tile when this is ``None``, takes the fraction of estimates that contain the tile.

    Returns:
        The layout. With no non-empty estimate the whole frame is one ``whole_frame`` tile.
    """
    if reference is None:
        reference = modal_layout(estimates, frame_width, frame_height, match_iou)
    if not reference:
        reference = [(0, 0, frame_width, frame_height)]
        derivations = [("whole_frame", 0.0)]
    if derivations is None:
        derivations = [("gutter", None)] * len(reference)

    tiles = []
    for index, ((x, y, w, h), (derivation, confidence)) in enumerate(zip(reference, derivations)):
        if confidence is None:
            box = (x, y, w, h)
            seen = sum(any(_iou(r, box) >= match_iou for r in rects) for _, rects in estimates)
            confidence = seen / len(estimates) if estimates else 0.0
        inset = int(math.ceil(border_inset_fraction * min(w, h)))
        x, y, w, h = x + inset, y + inset, w - 2 * inset, h - 2 * inset
        mh = int(round(label_mask_height_fraction * h))
        mw = int(round(label_mask_width_fraction * w))
        masks = [(0, h - mh, mw, mh)] if mh > 0 and mw > 0 else []
        tiles.append(
            TileBox(
                tile_id=f"T{index}",
                x=x,
                y=y,
                width=w,
                height=h,
                masks=masks,
                derivation=derivation,  # type: ignore[arg-type]
                confidence=float(confidence),
            )
        )

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


def detect_gallery_tiles_in_file(
    path: str, sample_frames_n: int, config: Dict[str, Any], face_detector: Optional[FaceDetector] = None
) -> GalleryLayout:
    """Sample frames across a file and estimate its gallery layout.

    Args:
        path: The video file.
        sample_frames_n: Frames to sample, spread uniformly over the recording.
        config: The ``tiles`` section of the defaults.
        face_detector: Face boxes per frame for :func:`split_merged_tiles`; :func:`detect_faces` when ``None``.

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
    split_times = times[
        np.unique(np.linspace(0, len(times) - 1, int(config["split"]["sample_frames"])).round().astype(int))
    ]
    seam_scale = min(1.0, float(config["split"]["seam_width"]) / width)
    seam_frames = [g for _, g in sample_frames(path, split_times.tolist(), scale=seam_scale, pixel_format="gray")]
    return layout_from_frames(samples, seam_frames, width, height, config, face_detector=face_detector)


def layout_from_frames(
    samples: Sequence[Tuple[float, np.ndarray]],
    seam_frames: Sequence[np.ndarray],
    width: int,
    height: int,
    config: Dict[str, Any],
    face_detector: Optional[FaceDetector] = None,
) -> GalleryLayout:
    """Estimate a gallery layout from sampled frames at the analysis and seam resolutions.

    Args:
        samples: ``(time_s, gray)`` frames at the analysis resolution.
        seam_frames: Gray frames at the seam resolution, used to split merged tiles.
        width: Source width.
        height: Source height.
        config: The ``tiles`` section of the defaults.
        face_detector: As in :func:`split_merged_tiles`.

    Returns:
        The layout.
    """
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

    def rescale(rects: Sequence[Rect], sx: float, sy: float) -> List[Rect]:
        return [
            (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))) for x, y, w, h in rects
        ]

    estimates = [(t, rescale(rects, width / gw, height / gh)) for t, rects in local]
    reference = rescale(refined, width / gw, height / gh)
    derivations: Optional[List[Tuple[str, Optional[float]]]] = None
    if reference and seam_frames:
        sh, sw = seam_frames[0].shape
        children = split_merged_tiles(
            seam_frames, rescale(reference, sw / width, sh / height), config, face_detector=face_detector
        )
        split = [[(rescale([r], width / sw, height / sh)[0], d, c) for r, d, c in kids] for kids in children]
        estimates = [(t, _substitute(rects, reference, split, config["match_iou"])) for t, rects in estimates]
        flat = [kid for kids in split for kid in kids]
        band = max(float(np.median([r[3] for r, _, _ in flat])), 1.0)
        flat.sort(key=lambda k: (round((k[0][1] + k[0][3] / 2) / band), k[0][0]))
        reference = [r for r, _, _ in flat]
        derivations = [(d, c) for _, d, c in flat]

    return layout_from_estimates(
        estimates,
        width,
        height,
        match_iou=config["match_iou"],
        label_mask_width_fraction=config["label_mask_width_fraction"],
        label_mask_height_fraction=config["label_mask_height_fraction"],
        border_inset_fraction=config["border_inset_fraction"],
        reference=reference or None,
        derivations=derivations,
    )


def _substitute(
    rects: Sequence[Rect], parents: Sequence[Rect], children: Sequence[Sequence[Derived]], match_iou: float
) -> List[Rect]:
    """Replace each rectangle that matches a split parent by that parent's children, in reading order."""
    out: List[Rect] = []
    for rect in rects:
        for parent, kids in zip(parents, children):
            if len(kids) > 1 and _iou(rect, parent) >= match_iou:
                out += [k[0] for k in kids]
                break
        else:
            out.append(rect)
    return _reading_order(out)


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Values in robust standard units about their median (MAD scaled to a normal sigma)."""
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / (mad + 1e-6)


def seam_scores(gray: np.ndarray, lags: Sequence[int] = (1, 3)) -> np.ndarray:
    """Discontinuity score of every column boundary of one region.

    Boundary ``b`` lies between columns ``b - 1`` and ``b``. Two cues are scored in robust standard
    units over the region's boundaries and the larger is kept: the mean over rows of the absolute
    difference across the boundary, and the step in the column-mean profile between the ``lag``
    columns on either side.

    Args:
        gray: ``(H, W)`` region.
        lags: Profile step widths in columns.

    Returns:
        ``(W + 1,)`` scores; the two outer boundaries score 0.
    """
    g = gray.astype(np.float32)
    w = g.shape[1]
    out = np.zeros(w + 1, np.float32)
    if w < 3:
        return out
    cues = [_robust_z(np.abs(np.diff(g, axis=1)).mean(axis=0))]
    cumulative = np.concatenate(([0.0], np.cumsum(g.mean(axis=0))))
    b = np.arange(1, w)
    for lag in lags:
        lo, hi = np.maximum(b - lag, 0), np.minimum(b + lag, w)
        left = (cumulative[b] - cumulative[lo]) / (b - lo)
        right = (cumulative[hi] - cumulative[b]) / (hi - b)
        cues.append(_robust_z(np.abs(right - left)))
    out[1:w] = np.max(cues, axis=0)
    return out


def find_seam(
    frames: Sequence[np.ndarray], x0: int, x1: int, y0: int, y1: int, expected: float, config: Dict[str, Any]
) -> Optional[Tuple[int, int, float]]:
    """Locate a vertical tile boundary near ``expected`` inside ``[x0, x1) x [y0, y1)``.

    Per-frame :func:`seam_scores` are combined by their median over frames, so content edges that
    move between frames are suppressed and the fixed boundary between two tiles remains. Columns of
    dark canvas on either side of the boundary are assigned to neither tile.

    Args:
        frames: Gray frames, one pixel grid.
        x0: Region left.
        x1: Region right (exclusive).
        y0: Region top.
        y1: Region bottom (exclusive).
        expected: Expected boundary column, absolute.
        config: The ``tiles`` section of the defaults.

    Returns:
        ``(left_end, right_start, confidence)`` in absolute columns, with ``confidence`` the fraction
        of frames whose own score passes at the boundary; ``None`` when the median score is below
        ``split.seam_min_z`` or below ``split.seam_dominance`` times the 99th percentile of the
        scores outside the search window (the codec's block edges sit there).
    """
    split = config["split"]
    regions = [f[y0:y1, x0:x1] for f in frames]
    scores = np.stack([seam_scores(r) for r in regions])
    width = x1 - x0
    reach = max(2, int(round(split["seam_search_fraction"] * width)))
    centre = int(round(expected)) - x0
    lo, hi = max(1, centre - reach), min(width - 1, centre + reach)
    if hi <= lo:
        return None
    outside = np.ones(scores.shape[1], bool)
    outside[lo : hi + 1] = False
    outside[[0, -1]] = False
    floors = np.maximum(split["seam_min_z"], split["seam_dominance"] * np.percentile(scores[:, outside], 99, axis=1))
    median = np.median(scores, axis=0)
    b = lo + int(np.argmax(median[lo : hi + 1]))
    if median[b] < np.median(floors):
        return None
    near = scores[:, max(1, b - 1) : b + 2].max(axis=1)
    confidence = float(np.mean(near >= floors))
    level = config["dark_level"] * (255.0 if regions[0].dtype == np.uint8 else 1.0)
    dark = (np.median(np.stack(regions), axis=0) < level).mean(axis=0) >= config["gutter_fraction"]
    left, right = b, b
    limit = int(split["seam_max_gutter_px"])
    while left - 1 > 0 and b - left < limit and dark[left - 1]:
        left -= 1
    while right < width - 1 and right - b < limit and dark[right]:
        right += 1
    return x0 + left, x0 + right, confidence


def _grid_candidates(width: int, height: int, config: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Equal ``(columns, rows)`` grids, other than 1 x 1, whose cells have a tile aspect ratio."""
    split = config["split"]
    out: List[Tuple[int, int, float]] = []
    for columns in range(1, int(split["max_grid"]) + 1):
        for rows in range(1, int(split["max_grid"]) + 1):
            if columns == rows == 1:
                continue
            cell = (width / columns) / max(height / rows, 1e-6)
            error = min(abs(cell / aspect - 1.0) for aspect in split["aspects"])
            if error <= split["aspect_tolerance"]:
                out.append((columns, rows, error))
    out.sort(key=lambda c: (-c[0] * c[1], c[2]))
    return [(c, r) for c, r, _ in out]


def _seam_grid(
    frames: Sequence[np.ndarray], rect: Rect, columns: int, rows: int, config: Dict[str, Any]
) -> Optional[List[Derived]]:
    """Cells of a ``columns x rows`` grid bounded by seams, or ``None`` when any seam is missing."""
    x, y, w, h = rect
    transposed = [f.T for f in frames]
    edges = []
    for count, span, origin, other, other_span, other_count, grid in (
        (columns, w, x, y, h, rows, frames),
        (rows, h, y, x, w, columns, transposed),
    ):
        cuts = []
        for i in range(1, count):
            found = []
            for j in range(other_count):
                pad = int(round(0.05 * other_span / other_count))
                b0 = other + int(round(j * other_span / other_count)) + pad
                b1 = other + int(round((j + 1) * other_span / other_count)) - pad
                seam = find_seam(grid, origin, origin + span, b0, b1, origin + i * span / count, config)
                if seam is None:
                    return None
                found.append(seam)
            cuts.append(
                (int(np.median([s[0] for s in found])), int(np.median([s[1] for s in found])), min(s[2] for s in found))
            )
        bounds = [origin] + [v for left, right, _ in cuts for v in (left, right)] + [origin + span]
        edges.append(
            ([(bounds[2 * k], bounds[2 * k + 1]) for k in range(count)], min([c for _, _, c in cuts], default=1.0))
        )
    (col_spans, col_conf), (row_spans, row_conf) = edges
    confidence = min(col_conf, row_conf)
    return [((a, c, b - a, d - c), "gutter", confidence) for c, d in row_spans for a, b in col_spans]


def _face_grid(
    faces: Sequence[Sequence[Rect]], rect: Rect, columns: int, rows: int, config: Dict[str, Any]
) -> Optional[List[Derived]]:
    """Equal cells of a grid when faces occupy every cell and rarely straddle a boundary."""
    split = config["split"]
    x, y, w, h = rect
    xs = [x + int(round(i * w / columns)) for i in range(columns + 1)]
    ys = [y + int(round(j * h / rows)) for j in range(rows + 1)]
    occupied = np.zeros((rows, columns))
    total = straddling = 0
    for boxes in faces:
        seen = np.zeros((rows, columns), bool)
        for fx, fy, fw, fh in boxes:
            cx, cy = fx + fw / 2, fy + fh / 2
            if not (x <= cx < x + w and y <= cy < y + h):
                continue
            total += 1
            straddling += any(fx < b < fx + fw for b in xs[1:-1]) or any(fy < b < fy + fh for b in ys[1:-1])
            col = min(columns - 1, int((cx - x) * columns / w))
            row = min(rows - 1, int((cy - y) * rows / h))
            seen[row, col] = True
        occupied += seen
    if not faces or total == 0:
        return None
    occupancy = float(occupied.min() / len(faces))
    straddle = straddling / total
    if occupancy < split["face_min_fraction"] or straddle > split["face_max_straddle"]:
        return None
    confidence = occupancy * (1.0 - straddle)
    return [
        ((xs[i], ys[j], xs[i + 1] - xs[i], ys[j + 1] - ys[j]), "aspect_split", confidence)
        for j in range(rows)
        for i in range(columns)
    ]


def detect_faces(frames: Sequence[np.ndarray], rect: Rect, config: Optional[Dict[str, Any]] = None) -> List[List[Rect]]:
    """Frontal faces per frame inside ``rect`` with OpenCV's Haar cascade.

    Args:
        frames: Gray uint8 frames.
        rect: Region searched, in the frames' pixels.
        config: The ``tiles`` section of the defaults (uses ``split.face_detect_width``).

    Returns:
        Face boxes per frame in the frames' pixels; empty lists when OpenCV is unavailable.
    """
    try:
        import cv2
    except ImportError:
        return [[] for _ in frames]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    x, y, w, h = rect
    target = float((config or {}).get("split", {}).get("face_detect_width", 960))
    scale = min(1.0, target / max(w, 1))
    out: List[List[Rect]] = []
    for frame in frames:
        region = np.ascontiguousarray(frame[y : y + h, x : x + w])
        if scale < 1.0:
            region = cv2.resize(region, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        found = cascade.detectMultiScale(region, scaleFactor=1.1, minNeighbors=5)
        out.append(
            [
                (x + int(fx / scale), y + int(fy / scale), int(fw / scale), int(fh / scale))
                for fx, fy, fw, fh in (found if len(found) else [])
            ]
        )
    return out


def split_merged_tiles(
    frames: Sequence[np.ndarray],
    rects: Sequence[Rect],
    config: Dict[str, Any],
    face_detector: Optional[FaceDetector] = None,
) -> List[List[Derived]]:
    """Split tiles that are several gallery tiles merged across an undetected gutter.

    For each rectangle, equal grids whose cells have a tile aspect ratio (``split.aspects``) are
    tried, largest first. A grid is accepted when every internal boundary is a seam
    (:func:`find_seam`) in every band of cells (``gutter``); failing that, when faces occupy every
    cell and rarely straddle a boundary (``aspect_split``). A rectangle with no accepted grid is kept.

    Args:
        frames: Gray frames at the seam resolution.
        rects: Tile rectangles in those frames' pixels.
        config: The ``tiles`` section of the defaults.
        face_detector: Face boxes per frame within a region; :func:`detect_faces` when ``None``.

    Returns:
        Per input rectangle, its tiles with derivation and confidence (``None`` confidence for an
        unsplit rectangle, to be filled from the per-frame estimates).
    """
    out: List[List[Derived]] = []
    for rect in rects:
        candidates = _grid_candidates(rect[2], rect[3], config) if frames else []
        chosen: Optional[List[Derived]] = None
        for columns, rows in candidates:
            chosen = _seam_grid(frames, rect, columns, rows, config)
            if chosen is not None:
                break
        if chosen is None and candidates:
            faces = (face_detector or (lambda f, r: detect_faces(f, r, config)))(frames, rect)
            for columns, rows in candidates:
                chosen = _face_grid(faces, rect, columns, rows, config)
                if chosen is not None:
                    break
        out.append(chosen if chosen is not None else [(rect, "gutter", None)])
    return out


def single_tile_layout(width: int, height: int, tile_id: Optional[str] = None) -> GalleryLayout:
    """A layout that treats the whole frame as one tile (no masks).

    Args:
        width: Frame width.
        height: Frame height.
        tile_id: Tile id; ``T0`` by default.

    Returns:
        The layout.
    """
    tile = TileBox(tile_id=tile_id or "T0", x=0, y=0, width=width, height=height, derivation="whole_frame")
    return GalleryLayout(
        frame_width=width, frame_height=height, tiles=[tile], timeline=[], reference_fraction=1.0, n_samples=0
    )
