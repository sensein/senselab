"""Cheap image measurements for one sampled frame.

Every metric is computed on the downscaled RGB frame with OpenCV and NumPy; none needs a model
download. Face counting uses the Haar cascade bundled with OpenCV.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from pydantic import BaseModel, Field

Box = Tuple[int, int, int, int]
"""``(x, y, width, height)`` in pixels."""

_THRESHOLDS_PATH = Path(__file__).parent / "data" / "cv_metrics.yaml"


@lru_cache(maxsize=1)
def thresholds() -> Dict[str, Any]:
    """The packaged thresholds in ``data/cv_metrics.yaml``."""
    return yaml.safe_load(_THRESHOLDS_PATH.read_text())


class FrameMetrics(BaseModel):
    """Image measurements for one frame.

    Attributes:
        brightness: Mean luma, scaled to ``[0, 1]``.
        contrast: Standard deviation of luma, scaled to ``[0, 1]``.
        blur: Variance of the Laplacian of the luma image (higher is sharper).
        colourfulness: Hasler and Suesstrunk colourfulness of the RGB image.
        near_black_fraction: Fraction of the whole image covered by dark, flat blocks (camera-off
            tiles, canvas and letterbox or pillarbox bars alike).
        bar_fraction: Fraction of the image in uniform dark bars along its edges (letterbox and
            pillarbox), as found by :func:`find_bars`.
        camera_off_fraction: With tile boxes, the fraction of tiles that are camera-off; without,
            the near-black fraction of the image inside its edge bars.
        tiles_camera_off: Camera-off flag per tile box, in the order given; empty without tiles.
        n_faces: Frontal faces found by the Haar cascade.
    """

    brightness: float
    contrast: float
    blur: float
    colourfulness: float
    near_black_fraction: float
    bar_fraction: float
    camera_off_fraction: float
    tiles_camera_off: List[bool] = Field(default_factory=list)
    n_faces: int


def _luma(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def colourfulness(image: np.ndarray) -> float:
    """Hasler and Suesstrunk (2003) colourfulness of an RGB ``uint8`` image.

    Args:
        image: RGB ``uint8`` array ``(H, W, 3)``.

    Returns:
        ``sigma_rgyb + 0.3 * mu_rgyb``; 0 for a grey image.
    """
    rgb = image.astype(np.float64)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    sigma = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mu = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return sigma + 0.3 * mu


def near_black_fraction(
    luma: np.ndarray, block: Optional[int] = None, max_mean: Optional[float] = None, max_std: Optional[float] = None
) -> float:
    """Fraction of the image area in blocks that are both dark and flat.

    Args:
        luma: Luma image in ``[0, 255]``.
        block: Block side in pixels; ``near_black.block`` when ``None``.
        max_mean: A block's mean luma must be at most this; ``near_black.max_mean`` when ``None``.
        max_std: A block's luma standard deviation must be at most this; ``near_black.max_std`` when ``None``.

    Returns:
        The covered fraction in ``[0, 1]``; 0 for an image smaller than one block.
    """
    config = thresholds()["near_black"]
    block = int(block if block is not None else config["block"])
    max_mean = float(max_mean if max_mean is not None else config["max_mean"])
    max_std = float(max_std if max_std is not None else config["max_std"])
    h, w = luma.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return 0.0
    tiles = luma[: hb * block, : wb * block].reshape(hb, block, wb, block)
    dark = (tiles.mean(axis=(1, 3)) <= max_mean) & (tiles.std(axis=(1, 3)) <= max_std)
    return float(dark.mean())


def _edge_run(flags: np.ndarray) -> int:
    """Length of the run of True values at the start of ``flags``."""
    off = np.flatnonzero(~flags)
    return int(off[0]) if off.size else int(flags.size)


def find_bars(luma: np.ndarray) -> Box:
    """The image area inside uniform dark bars along its edges (letterbox and pillarbox).

    A bar is a run of full-width rows (top, bottom) or full-height columns (left, right), starting at
    the edge, each dark and flat over its whole length (``bars.max_mean``, ``bars.max_std``). Rows
    are found first and columns are measured within the remaining rows. Bars are removed on an axis
    only in centred pairs: both edges have one, their lengths differ by at most
    ``bars.symmetry_tolerance`` of the axis, and together they cover at most ``bars.max_fraction``
    of it. A camera-off tile at one edge, or an almost all-dark camera-off speaker view, therefore
    keeps its area.

    Args:
        luma: Luma image in ``[0, 255]``.

    Returns:
        The content box ``(x, y, width, height)``; the whole image when there are no bars.
    """
    config = thresholds()["bars"]
    h, w = luma.shape

    def uniform_dark(lines: np.ndarray) -> np.ndarray:
        return (lines.mean(axis=1) <= config["max_mean"]) & (lines.std(axis=1) <= config["max_std"])

    def centred(first: int, last: int, length: int) -> Tuple[int, int]:
        paired = first > 0 and last > 0 and abs(first - last) <= config["symmetry_tolerance"] * length
        return (first, last) if paired and first + last <= config["max_fraction"] * length else (0, 0)

    rows = uniform_dark(luma)
    top, bottom = centred(_edge_run(rows), _edge_run(rows[::-1]), h)
    inner = luma[top : h - bottom]
    cols = uniform_dark(inner.T) if inner.size else np.zeros(w, bool)
    left, right = centred(_edge_run(cols), _edge_run(cols[::-1]), w)
    return left, top, w - left - right, h - top - bottom


def camera_off_tiles(luma: np.ndarray, tiles: Sequence[Box]) -> List[bool]:
    """Whether each tile is camera-off: mostly dark and flat (``camera_off.tile_min_near_black``).

    Args:
        luma: Luma image in ``[0, 255]``.
        tiles: Tile boxes in the image's pixels; parts outside the image are clipped.

    Returns:
        One flag per tile; an empty or fully clipped tile is not camera-off.
    """
    h, w = luma.shape
    floor = thresholds()["camera_off"]["tile_min_near_black"]
    out: List[bool] = []
    for x, y, tw, th in tiles:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + tw), min(h, y + th)
        region = luma[y0:y1, x0:x1]
        out.append(bool(region.size and near_black_fraction(region) >= floor))
    return out


@lru_cache(maxsize=1)
def _face_cascade() -> Any:  # noqa: ANN401
    import cv2

    cascade = cv2.CascadeClassifier(getattr(cv2, "data").haarcascades + "haarcascade_frontalface_default.xml")  # noqa: B009
    if cascade.empty():
        raise RuntimeError("OpenCV's bundled Haar face cascade could not be loaded")
    return cascade


def count_faces(luma_u8: np.ndarray, min_face_fraction: float = 0.04) -> int:
    """Count frontal faces with OpenCV's bundled Haar cascade.

    Args:
        luma_u8: Grayscale ``uint8`` image.
        min_face_fraction: Smallest face side to look for, as a fraction of the image's short side.

    Returns:
        The number of detections.
    """
    min_side = max(16, int(min(luma_u8.shape) * min_face_fraction))
    faces = _face_cascade().detectMultiScale(luma_u8, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
    return int(len(faces))


def frame_metrics(image: np.ndarray, detect_faces: bool = True, tiles: Optional[Sequence[Box]] = None) -> FrameMetrics:
    """Compute the cheap image measurements of one RGB frame.

    Args:
        image: RGB ``uint8`` array ``(H, W, 3)``.
        detect_faces: Run the Haar face detector; when False ``n_faces`` is -1.
        tiles: Participant tile boxes in this image's pixels. With them, ``camera_off_fraction`` is
            the fraction of camera-off tiles; without, it is the near-black area inside edge bars.

    Returns:
        The frame's ``FrameMetrics``.
    """
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {image.shape}")
    luma = _luma(image)
    luma_u8 = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    h, w = luma.shape
    bx, by, bw, bh = find_bars(luma)
    flags = camera_off_tiles(luma, tiles) if tiles else []
    if tiles:
        camera_off = sum(flags) / len(flags)
    else:
        camera_off = near_black_fraction(luma[by : by + bh, bx : bx + bw]) if bw and bh else 0.0
    return FrameMetrics(
        brightness=float(luma.mean() / 255.0),
        contrast=float(luma.std() / 255.0),
        blur=float(cv2.Laplacian(luma_u8, cv2.CV_64F).var()),
        colourfulness=colourfulness(image),
        near_black_fraction=near_black_fraction(luma),
        bar_fraction=1.0 - (bw * bh) / float(h * w),
        camera_off_fraction=camera_off,
        tiles_camera_off=flags,
        n_faces=count_faces(luma_u8) if detect_faces else -1,
    )
