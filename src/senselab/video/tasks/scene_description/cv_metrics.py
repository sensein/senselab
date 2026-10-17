"""Cheap image measurements for one sampled frame.

Every metric is computed on the downscaled RGB frame with OpenCV and NumPy; none needs a model
download. Face counting uses the Haar cascade bundled with OpenCV.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel


class FrameMetrics(BaseModel):
    """Image measurements for one frame.

    Attributes:
        brightness: Mean luma, scaled to ``[0, 1]``.
        contrast: Standard deviation of luma, scaled to ``[0, 1]``.
        blur: Variance of the Laplacian of the luma image (higher is sharper).
        colourfulness: Hasler and Suesstrunk colourfulness of the RGB image.
        near_black_fraction: Fraction of the image covered by dark, flat blocks (camera-off tiles,
            black bars).
        n_faces: Frontal faces found by the Haar cascade.
    """

    brightness: float
    contrast: float
    blur: float
    colourfulness: float
    near_black_fraction: float
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


def near_black_fraction(luma: np.ndarray, block: int = 16, max_mean: float = 40.0, max_std: float = 8.0) -> float:
    """Fraction of the image area in blocks that are both dark and flat.

    Args:
        luma: Luma image in ``[0, 255]``.
        block: Block side in pixels.
        max_mean: A block's mean luma must be at most this.
        max_std: A block's luma standard deviation must be at most this.

    Returns:
        The covered fraction in ``[0, 1]``; 0 for an image smaller than one block.
    """
    h, w = luma.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return 0.0
    tiles = luma[: hb * block, : wb * block].reshape(hb, block, wb, block)
    dark = (tiles.mean(axis=(1, 3)) <= max_mean) & (tiles.std(axis=(1, 3)) <= max_std)
    return float(dark.mean())


@lru_cache(maxsize=1)
def _face_cascade() -> Any:  # noqa: ANN401
    import cv2

    cascade = cv2.CascadeClassifier(getattr(cv2, "data").haarcascades + "haarcascade_frontalface_default.xml")
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


def frame_metrics(image: np.ndarray, detect_faces: bool = True) -> FrameMetrics:
    """Compute the cheap image measurements of one RGB frame.

    Args:
        image: RGB ``uint8`` array ``(H, W, 3)``.
        detect_faces: Run the Haar face detector; when False ``n_faces`` is -1.

    Returns:
        The frame's ``FrameMetrics``.
    """
    import cv2

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {image.shape}")
    luma = _luma(image)
    luma_u8 = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    return FrameMetrics(
        brightness=float(luma.mean() / 255.0),
        contrast=float(luma.std() / 255.0),
        blur=float(cv2.Laplacian(luma_u8, cv2.CV_64F).var()),
        colourfulness=colourfulness(image),
        near_black_fraction=near_black_fraction(luma),
        n_faces=count_faces(luma_u8) if detect_faces else -1,
    )
