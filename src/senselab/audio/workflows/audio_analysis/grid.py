"""BucketGrid — shared time grid that all model outputs project onto before voting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketGrid:
    """Time grid for per-bucket cross-model aggregation.

    Defaults match FR-010: 0.5 s non-overlapping. Every model's output is projected onto
    this grid before voting; the bucket boundaries appear verbatim on each parquet row's
    ``start`` / ``end`` columns.

    Attributes:
        win_length: Bucket length in seconds. Must be > 0.
        hop_length: Hop between consecutive bucket starts in seconds.
            Must satisfy 0 < hop_length <= win_length.
        name: Provenance label recorded in the parquet metadata.
    """

    win_length: float = 0.5
    hop_length: float = 0.5
    name: str = "comparator"

    def __post_init__(self) -> None:
        """Reject impossible grid configurations early."""
        if self.win_length <= 0:
            raise ValueError(f"win_length must be > 0, got {self.win_length}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {self.hop_length}")
        if self.hop_length > self.win_length:
            raise ValueError(f"hop_length ({self.hop_length}) must be <= win_length ({self.win_length})")

    def iter_buckets(self, duration_s: float) -> Iterator[tuple[float, float, int]]:
        """Yield ``(start, end, idx)`` covering ``[0, duration_s]``.

        The last bucket is included only when ``start + win_length <= duration_s`` so
        every bucket is fully inside the audio.
        """
        if duration_s <= 0:
            return
        idx = 0
        start = 0.0
        # Use a small epsilon to avoid float-rounding excluding a legitimate boundary bucket.
        eps = 1e-9
        while start + self.win_length <= duration_s + eps:
            yield round(start, 6), round(start + self.win_length, 6), idx
            idx += 1
            start = idx * self.hop_length
