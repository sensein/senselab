"""Sparse phonetic posteriorgrams: the top phones per frame, in coordinate (COO) form.

A ppgs posteriorgram is 40 float32 values per 10 ms frame. :func:`sparsify_ppg` keeps, per frame,
the ``top_k`` most probable phones whose posterior is at least ``min_prob`` (the most probable phone
is always kept), and stores them as three parallel arrays ``frame``, ``phone``, ``prob``.
:func:`sparse_ppg_fidelity` reports what the reduction kept. Definitions and measured compression
are in ``specs/20261017-104500-sparse-ppg-boundaries/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, model_validator

from senselab.audio.tasks.features_extraction.ppg import (
    PHONEME_LABELS,
    PPGS_FRAME_RATE,
    to_frame_major_posteriorgram,
)


class SparsePPG(BaseModel):
    """A posteriorgram stored as its retained (frame, phone, probability) triples.

    Attributes:
        frame_rate: Frames per second; frame ``i`` is centred at ``(i + 0.5) / frame_rate`` seconds.
        phones: The phone vocabulary; ``phone`` indexes into it.
        num_frames: Frames in the dense posteriorgram, including any with no retained entry.
        frame: Frame index of each entry, ascending.
        phone: Phone index of each entry; within a frame, entries run from most to least probable.
        prob: Posterior of each entry.
        model_id: Repository of the model that produced the posteriorgram, when known.
        model_revision: 40-hex commit of that model, when known.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frame_rate: float
    phones: Tuple[str, ...]
    num_frames: int
    frame: np.ndarray
    phone: np.ndarray
    prob: np.ndarray
    model_id: Optional[str] = None
    model_revision: Optional[str] = None

    @model_validator(mode="after")
    def _check_arrays(self) -> "SparsePPG":
        if not (self.frame.ndim == self.phone.ndim == self.prob.ndim == 1):
            raise ValueError("frame, phone and prob must be 1-D")
        if not (len(self.frame) == len(self.phone) == len(self.prob)):
            raise ValueError("frame, phone and prob must have the same length")
        if self.frame_rate <= 0:
            raise ValueError(f"frame_rate must be positive, got {self.frame_rate}")
        if len(self.frame) and (int(self.frame.max()) >= self.num_frames or int(self.frame.min()) < 0):
            raise ValueError("frame index out of range")
        if len(self.phone) and int(self.phone.max()) >= len(self.phones):
            raise ValueError("phone index out of range")
        return self

    @property
    def nnz(self) -> int:
        """Number of retained entries."""
        return int(len(self.prob))

    @property
    def nbytes(self) -> int:
        """Bytes held by the three entry arrays, uncompressed."""
        return int(self.frame.nbytes + self.phone.nbytes + self.prob.nbytes)

    def frame_mass(self) -> np.ndarray:
        """Posterior mass retained in each frame.

        Returns:
            A ``(num_frames,)`` float64 array; ``1 - frame_mass()`` is what the reduction dropped.
        """
        return np.bincount(self.frame, weights=self.prob.astype(np.float64), minlength=self.num_frames)

    def to_dense(self, renormalize: bool = False) -> np.ndarray:
        """The frame-major dense posteriorgram, with zeros where nothing was retained.

        Args:
            renormalize: Divide each frame by its retained mass, so every non-empty frame sums to one.

        Returns:
            A ``(num_frames, len(phones))`` float32 array.
        """
        dense = np.zeros((self.num_frames, len(self.phones)), dtype=np.float32)
        dense[self.frame, self.phone] = self.prob.astype(np.float32)
        if renormalize:
            mass = dense.sum(axis=1, keepdims=True)
            np.divide(dense, mass, out=dense, where=mass > 0)
        return dense

    def save(self, path: Union[str, Path]) -> Path:
        """Write the sparse posteriorgram to a compressed ``.npz``.

        Args:
            path: Destination; ``.npz`` is appended by numpy when missing.

        Returns:
            The path written.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        np.savez_compressed(
            path,
            frame_rate=np.float64(self.frame_rate),
            phones=np.array(self.phones),
            num_frames=np.int64(self.num_frames),
            frame=self.frame,
            phone=self.phone,
            prob=self.prob,
            model_id=np.array(self.model_id or ""),
            model_revision=np.array(self.model_revision or ""),
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SparsePPG":
        """Read a sparse posteriorgram :meth:`save` wrote.

        Args:
            path: The ``.npz`` file.

        Returns:
            The sparse posteriorgram.
        """
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                frame_rate=float(archive["frame_rate"]),
                phones=tuple(str(p) for p in archive["phones"]),
                num_frames=int(archive["num_frames"]),
                frame=archive["frame"],
                phone=archive["phone"],
                prob=archive["prob"],
                model_id=_optional_text(archive, "model_id"),
                model_revision=_optional_text(archive, "model_revision"),
            )


def _optional_text(archive: Any, key: str) -> Optional[str]:  # noqa: ANN401
    value = str(archive[key]) if key in archive.files else ""
    return value or None


PosteriorgramLike = Union[torch.Tensor, np.ndarray, SparsePPG]
"""Anything the PPG analysis functions read: a ppgs tensor in any layout, a frame-major array, or a sparse PPG."""


def as_frame_major_array(ppg: PosteriorgramLike, num_phones: int = len(PHONEME_LABELS)) -> np.ndarray:
    """A posteriorgram as a frame-major float32 array.

    Args:
        ppg: A ppgs tensor (``(1, phones, frames)``, ``(phones, frames)`` or ``(frames, phones)``), a
            numpy array in one of those layouts, or a :class:`SparsePPG` (densified, not renormalized).
        num_phones: The phone-axis length used to tell the layouts apart.

    Returns:
        A ``(frames, phones)`` float32 array.
    """
    if isinstance(ppg, SparsePPG):
        return ppg.to_dense()
    tensor = ppg if isinstance(ppg, torch.Tensor) else torch.from_numpy(np.asarray(ppg))
    if tensor.ndim == 2 and tensor.shape[1] == num_phones:
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    return to_frame_major_posteriorgram(tensor).numpy().astype(np.float32, copy=False)


def sparsify_ppg(
    ppg: Union[torch.Tensor, np.ndarray],
    top_k: int = 3,
    min_prob: float = 0.05,
    dtype: Any = np.float16,  # noqa: ANN401
    frame_rate: float = PPGS_FRAME_RATE,
    phones: Tuple[str, ...] = PHONEME_LABELS,
    model_id: Optional[str] = None,
    model_revision: Optional[str] = None,
) -> SparsePPG:
    """Keep each frame's ``top_k`` most probable phones whose posterior is at least ``min_prob``.

    The most probable phone of every frame is kept whatever its posterior, so the dense argmax is
    recoverable except where storage rounding (``dtype``) ties it with the runner-up.

    Args:
        ppg: A ppgs posteriorgram in any layout :func:`as_frame_major_array` accepts.
        top_k: Most entries kept per frame.
        min_prob: Least posterior an entry other than the frame's most probable needs to be kept.
        dtype: Storage dtype of the probabilities.
        frame_rate: Frames per second of ``ppg``.
        phones: The phone vocabulary of ``ppg``'s phone axis.
        model_id: Repository of the model that produced ``ppg`` (e.g. :data:`PPGS_REPO_ID`).
        model_revision: Its commit (e.g. :func:`ppgs_model_revision`).

    Returns:
        The sparse posteriorgram.

    Raises:
        ValueError: If ``top_k`` is not positive, or ``phones`` does not match the phone axis.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    dense = as_frame_major_array(ppg, num_phones=len(phones))
    num_frames, num_phones = dense.shape
    if num_phones != len(phones):
        raise ValueError(f"posteriorgram has {num_phones} phones, vocabulary has {len(phones)}")
    k = min(top_k, num_phones)
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    values = np.take_along_axis(dense, order, axis=1)
    keep = values >= min_prob
    keep[:, 0] = True
    frame_index = np.broadcast_to(np.arange(num_frames)[:, None], order.shape)
    phone_dtype = np.uint8 if num_phones <= np.iinfo(np.uint8).max + 1 else np.uint16
    return SparsePPG(
        frame_rate=float(frame_rate),
        phones=tuple(phones),
        num_frames=int(num_frames),
        frame=frame_index[keep].astype(np.uint32),
        phone=order[keep].astype(phone_dtype),
        prob=values[keep].astype(dtype),
        model_id=model_id,
        model_revision=model_revision,
    )


def sparse_ppg_fidelity(ppg: Union[torch.Tensor, np.ndarray], sparse: SparsePPG) -> Dict[str, float]:
    """How much of a dense posteriorgram a sparse one keeps.

    Args:
        ppg: The dense posteriorgram ``sparse`` was made from.
        sparse: Its sparse form.

    Returns:
        A dict with ``mass_retained_mean`` and ``mass_retained_min`` (per-frame retained posterior
        mass), ``argmax_agreement`` (fraction of frames whose most probable phone is unchanged),
        ``max_abs_error`` and ``mean_abs_error`` (over every dense cell, sparse not renormalized),
        ``entries_per_frame``, ``dense_float32_bytes``, ``sparse_bytes`` and ``compression_ratio``.
    """
    dense = as_frame_major_array(ppg, num_phones=len(sparse.phones))
    approx = sparse.to_dense()
    mass = sparse.frame_mass()
    error = np.abs(dense - approx)
    num_frames = max(1, dense.shape[0])
    dense_bytes = int(dense.shape[0] * dense.shape[1] * 4)
    return {
        "mass_retained_mean": float(mass.mean()) if mass.size else 1.0,
        "mass_retained_min": float(mass.min()) if mass.size else 1.0,
        "argmax_agreement": float((dense.argmax(axis=1) == approx.argmax(axis=1)).mean()) if dense.size else 1.0,
        "max_abs_error": float(error.max()) if error.size else 0.0,
        "mean_abs_error": float(error.mean()) if error.size else 0.0,
        "entries_per_frame": sparse.nnz / num_frames,
        "dense_float32_bytes": float(dense_bytes),
        "sparse_bytes": float(sparse.nbytes),
        "compression_ratio": dense_bytes / max(1, sparse.nbytes),
    }
