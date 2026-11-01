"""Face detection and identity embeddings with OpenCV's YuNet and SFace (opencv_zoo, pinned)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from senselab.video.tasks.facial_analysis.results import Rect

YUNET_REPO = "opencv/face_detection_yunet"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_COMMIT = "3cc26e7f1014a5ee5d74a42acee58bafc9d0a310"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_LICENSE = "MIT"

SFACE_REPO = "opencv/face_recognition_sface"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
SFACE_COMMIT = "3d7082438a6e4551e840c9b2bb60b71e8da4b524"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_LICENSE = "Apache-2.0"


def _url(repo: str, commit: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{commit}/{filename}"


def identity_model_paths() -> Dict[str, Path]:
    """Download (once) and verify the YuNet detector and SFace recogniser.

    Returns:
        ``detector`` and ``recognizer`` local ``.onnx`` paths.
    """
    from senselab.utils.model_revision import cache_root
    from senselab.video.tasks.facial_analysis.backends import download_pinned

    folder = cache_root() / "models" / "opencv"
    return {
        "detector": download_pinned(
            _url(YUNET_REPO, YUNET_COMMIT, YUNET_FILE), YUNET_SHA256, folder / f"{YUNET_SHA256[:12]}_{YUNET_FILE}"
        ),
        "recognizer": download_pinned(
            _url(SFACE_REPO, SFACE_COMMIT, SFACE_FILE), SFACE_SHA256, folder / f"{SFACE_SHA256[:12]}_{SFACE_FILE}"
        ),
    }


def identity_provenance(cosine_threshold: float) -> Dict[str, Any]:
    """Model identity of the face-identity pass, for :attr:`GalleryLayout.identity`."""
    return {
        "detector": {
            "url": _url(YUNET_REPO, YUNET_COMMIT, YUNET_FILE),
            "sha256": YUNET_SHA256,
            "license": YUNET_LICENSE,
        },
        "recognizer": {
            "url": _url(SFACE_REPO, SFACE_COMMIT, SFACE_FILE),
            "sha256": SFACE_SHA256,
            "license": SFACE_LICENSE,
        },
        "cosine_threshold": float(cosine_threshold),
    }


@dataclass
class Face:
    """One detected face.

    Attributes:
        box: ``(x, y, w, h)`` in the pixels of the frame it was found in.
        score: Detector score.
        embedding: L2-normalised identity embedding (128 values for SFace).
    """

    box: Rect
    score: float
    embedding: np.ndarray


class FaceIdentifier:
    """YuNet face detection followed by SFace embedding of each aligned face.

    Args:
        detector_path: YuNet ``.onnx``.
        recognizer_path: SFace ``.onnx``.
        score_threshold: Smallest detector score kept.
        nms_threshold: Detector non-maximum-suppression IoU.
        top_k: Most candidate boxes before suppression.
    """

    def __init__(
        self,
        detector_path: Path,
        recognizer_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
    ) -> None:
        """Load both models with OpenCV's DNN module."""
        import cv2

        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (320, 320), float(score_threshold), float(nms_threshold), int(top_k)
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def __call__(self, rgb: np.ndarray) -> List[Face]:
        """Faces in one RGB frame, each with its embedding.

        Args:
            rgb: ``(H, W, 3)`` uint8 frame.

        Returns:
            The faces, largest first.
        """
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(bgr)
        faces: List[Face] = []
        for row in rows if rows is not None else []:
            aligned = self._recognizer.alignCrop(bgr, row)
            vector = np.asarray(self._recognizer.feature(aligned), np.float32).ravel()
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            x, y, bw, bh = (int(round(v)) for v in row[:4])
            faces.append(Face(box=(x, y, bw, bh), score=float(row[14]), embedding=vector))
        faces.sort(key=lambda f: -f.box[2] * f.box[3])
        return faces


def default_identifier(config: Dict[str, Any]) -> FaceIdentifier:
    """A :class:`FaceIdentifier` on the pinned models, with the ``speaker_view`` detector settings."""
    paths = identity_model_paths()
    return FaceIdentifier(
        paths["detector"],
        paths["recognizer"],
        score_threshold=config["score_threshold"],
        nms_threshold=config["nms_threshold"],
        top_k=config["top_k"],
    )


def cluster_identities(
    embeddings: Sequence[np.ndarray],
    cosine_threshold: float,
    min_fraction: float = 0.0,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Group face embeddings into persons.

    Average-linkage agglomerative clustering on cosine distance, cut at ``1 - cosine_threshold``.
    With more than ``max_samples`` embeddings, an evenly spaced subset is clustered and every other
    embedding joins the cluster whose centroid it is most similar to, when that similarity reaches
    ``cosine_threshold``.

    Args:
        embeddings: L2-normalised vectors.
        cosine_threshold: Same-person cosine similarity.
        min_fraction: Clusters holding a smaller fraction of the embeddings are dropped (label -1).
        max_samples: Most embeddings clustered directly; all when ``None``.

    Returns:
        A label per embedding: ``0, 1, ...`` in order of first occurrence, ``-1`` for none.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embeddings)
    if n == 0:
        return np.zeros(0, int)
    vectors = np.stack([np.asarray(e, np.float64).ravel() for e in embeddings])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    if max_samples is not None and n > max_samples:
        chosen = np.unique(np.linspace(0, n - 1, max_samples).round().astype(int))
    else:
        chosen = np.arange(n)
    if len(chosen) == 1:
        raw = np.ones(1, int)
    else:
        raw = fcluster(linkage(vectors[chosen], method="average", metric="cosine"), 1.0 - cosine_threshold, "distance")
    labels = np.full(n, -1, int)
    labels[chosen] = raw
    rest = np.setdiff1d(np.arange(n), chosen)
    if len(rest):
        ids = np.unique(raw)
        centroids = np.stack([vectors[chosen][raw == k].mean(axis=0) for k in ids])
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
        similarity = vectors[rest] @ centroids.T
        best = similarity.argmax(axis=1)
        labels[rest] = np.where(similarity[np.arange(len(rest)), best] >= cosine_threshold, ids[best], -1)
    counts = {k: int((labels == k).sum()) for k in np.unique(labels) if k >= 0}
    keep = [k for k, c in counts.items() if c >= min_fraction * n]
    order = sorted(keep, key=lambda k: int(np.flatnonzero(labels == k)[0]))
    out = np.full(n, -1, int)
    for new, old in enumerate(order):
        out[labels == old] = new
    return out
