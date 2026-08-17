"""BucketGrid — the one time grid every model output projects onto before voting.

Stdlib-only, and deliberately so: this module is imported by the extraction layer, which must not
reach the axis vocabulary that consumes its output.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

DEFAULT_TIME_GRID: Final[tuple[float, float]] = (0.1, 0.1)
"""``(win_length, hop_length)`` in seconds for every ``"time"``-gridded axis. **Configurable** — a
downstream need for finer or coarser buckets changes this, or the run's params override it.

**Window equals hop, so the buckets do not overlap**, and that is the point rather than a coincidence.
The run that motivated this used a 0.1 s window at a 0.02 s hop: adjacent rows shared 80% of their
audio, so 1070 rows were not 1070 independent measurements and nothing told a consumer so. A fine
*resolution* is what the question justifies; reporting five near-duplicate rows per window is not the
same thing, and the near-duplication was invisible in the output.

100 ms is sufficient for the downstream needs known today — speech and target-activity onsets are
resolved at it, and speaker turns and mask regions are much longer.

**It lives here, not in ``axes``, because it is a property of the grid rather than of the axes.**
It was declared in ``axes.py``, which inverted the arrow: an axis *reads* the grid it is estimated
on, so the grid does not belong to the axes — and the one line importing it back the other way
(``grid`` → ``axes``) was, at import time, the *only* path from the extraction layer to the
refiner's axis vocabulary. ``grid.py`` now has no intra-package import at all, which is what makes
that boundary checkable instead of incidental.
"""


@dataclass(frozen=True)
class BucketGrid:
    """Time grid for per-bucket cross-model aggregation.

    Defaults to :data:`DEFAULT_TIME_GRID`, and that is the point rather than a convenience:
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
            no two rows share a frame; see :data:`DEFAULT_TIME_GRID` for why overlap is
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
