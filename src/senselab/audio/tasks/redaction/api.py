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


def apply_redactions(
    audio: Audio,
    extents: Sequence[RedactionExtent],
    *,
    fill: str,
    bleep_hz: float | None = None,
) -> Audio:
    """Mask every extent with the named fill, preserving duration.

    Args:
        audio: The recording.
        extents: Regions to mask. Pass the output of :func:`plan_redactions`, not raw findings.
        fill: ``"silence"`` writes zeros; ``"bleep"`` writes a sine at ``bleep_hz`` scaled to the
            extent's own peak. **Required, with no default**: which fill is least damaging to
            downstream measurement is unmeasured, so a caller that does not say gets no answer rather
            than silently getting silence. Read it from ``redaction.fill``.
        bleep_hz: The bleep's frequency. Required when ``fill`` is ``"bleep"``. Read it from
            ``redaction.bleep_hz``.

    Returns:
        A new ``Audio``. The input is not modified. Each extent's start rounds down to a sample index
        and its end rounds up, both clamped to the recording; an extent that selects no samples is a
        no-op.

    Raises:
        NotImplementedError: If ``fill`` is ``"noise"``. Which fill is least damaging to the
            measurements taken downstream of a released artifact has not been measured, and
            "speech-shaped" names a shaping nobody has fitted.
        ValueError: If ``fill`` names no implemented fill, or if ``"bleep"`` is asked for without
            ``bleep_hz``.
    """
    if fill == "noise":
        raise NotImplementedError(
            "the 'noise' fill is deferred: which fill is least damaging to downstream measurement "
            "has not been measured, and a speech-shaped spectrum nobody fitted is not a default"
        )
    if fill not in ("silence", "bleep"):
        raise ValueError(f"fill must be 'silence' or 'bleep'; got {fill!r}")
    if fill == "bleep" and bleep_hz is None:
        raise ValueError("fill='bleep' needs bleep_hz; read it from redaction.bleep_hz")
    tone_hz = 0.0 if bleep_hz is None else float(bleep_hz)
    x = np.array(np.asarray(audio.waveform, dtype=np.float32), copy=True)
    if x.ndim == 1:
        x = x[None, :]
    sr = audio.sampling_rate
    n = x.shape[-1]
    for extent in extents:
        lo = max(0, int(extent.start * sr))
        hi = max(lo, min(n, math.ceil(extent.end * sr)))
        if hi <= lo:
            continue
        if fill == "silence":
            x[:, lo:hi] = 0.0
        else:
            level = float(np.abs(x[:, lo:hi]).max())
            t = np.arange(hi - lo, dtype=np.float32) / sr
            x[:, lo:hi] = (level * np.sin(2.0 * np.pi * tone_hz * t)).astype(np.float32)
    return Audio(waveform=x, sampling_rate=sr)
