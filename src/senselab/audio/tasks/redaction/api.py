"""Planning and applying redactions over audio."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from senselab.audio.data_structures import Audio


@dataclass(frozen=True)
class RedactionExtent:
    """A region to remove.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        category: What was found here. Never the matched text.
    """

    start: float
    end: float
    category: str


def plan_redactions(extents: Sequence[RedactionExtent], *, padding_ms: int) -> list[RedactionExtent]:
    """Pad every extent outward and merge those that then overlap.

    Args:
        extents: Regions to redact.
        padding_ms: Margin in milliseconds added to each side. Keyword-only with no default; supplied by
            the ``redaction.padding_ms`` config key (see ``specs/20260817-triage-workflow-dag/redact.md``).

    Returns:
        Padded, merged extents in time order. Categories of merged extents are joined with ``+``,
        deduplicated in first-seen order.

    Raises:
        ValueError: If an extent has a non-finite bound, a negative start, or an end before its start.
            The error names the extent's bounds and category, never any matched text.
    """
    for extent in extents:
        if (
            not (math.isfinite(extent.start) and math.isfinite(extent.end))
            or extent.start < 0
            or extent.end < extent.start
        ):
            raise ValueError(f"invalid extent: start={extent.start}, end={extent.end}, category={extent.category}")
    pad = padding_ms / 1000.0
    widened = sorted(
        (RedactionExtent(max(0.0, e.start - pad), e.end + pad, e.category) for e in extents),
        key=lambda e: e.start,
    )
    merged: list[RedactionExtent] = []
    for extent in widened:
        if merged and extent.start <= merged[-1].end:
            last = merged[-1]
            categories = last.category.split("+")
            for category in extent.category.split("+"):
                if category not in categories:
                    categories.append(category)
            merged[-1] = RedactionExtent(last.start, max(last.end, extent.end), "+".join(categories))
        else:
            merged.append(extent)
    return merged


def apply_redactions(audio: Audio, extents: Sequence[RedactionExtent]) -> Audio:
    """Silence every extent, preserving duration.

    Args:
        audio: The recording.
        extents: Regions to silence. Pass the output of :func:`plan_redactions`, not raw findings.

    Returns:
        A new ``Audio``. The input is not modified. Each extent's start rounds down to a sample index and
        its end rounds up, both clamped to the recording; an extent that selects no samples is a no-op.
    """
    x = np.array(np.asarray(audio.waveform, dtype=np.float32), copy=True)
    if x.ndim == 1:
        x = x[None, :]
    sr = audio.sampling_rate
    n = x.shape[-1]
    for extent in extents:
        lo = max(0, int(extent.start * sr))
        hi = max(lo, min(n, math.ceil(extent.end * sr)))
        x[:, lo:hi] = 0.0
    return Audio(waveform=x, sampling_rate=sr)
