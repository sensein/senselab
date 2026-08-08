"""BucketGrid — the one time grid every model output projects onto before voting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from senselab.audio.workflows.audio_analysis.axes import DEFAULT_TIME_GRID


@dataclass(frozen=True)
class BucketGrid:
    """Time grid for per-bucket cross-model aggregation.

    Defaults to :data:`~.axes.DEFAULT_TIME_GRID`, and that is the point rather than a convenience:
    **every axis is on this grid**, so row *i* of one axis is row *i* of another and a cross-axis
    join needs no reconciliation. Measured before it was: the four axes carried 242 / 242 / 19 / 8
    rows on 0.1/0.02, 0.1/0.02, 0.25/0.25 and 1.0/0.5 respectively, shared zero bucket keys, and
    the coupling between them therefore did nothing while reporting that it had run.

    A default that differed from the declared constant is how that happened — the constant was
    declared, nothing read it, and each caller supplied its own pair.

    Attributes:
        win_length: Bucket length in seconds. Must be > 0.
        hop_length: Hop between consecutive bucket starts in seconds.
            Must satisfy 0 < hop_length <= win_length. Equal to ``win_length`` in the default, so
            no two rows share a frame; see :data:`~.axes.DEFAULT_TIME_GRID` for why overlap is
            not "finer resolution".
        name: Provenance label recorded in the parquet metadata.
    """

    win_length: float = DEFAULT_TIME_GRID[0]
    hop_length: float = DEFAULT_TIME_GRID[1]
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
