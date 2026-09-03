"""Clip-event detection: ClipDaT (Hansen, Stauffer & Xia, Speech Communication 134 (2021) 20-31)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio

MINIMUM_EXTREME_RUN = 2
"""Consecutive samples at a *sub-full-scale* extreme required before a clip event opens.

Two, because a clipped waveform holds its extreme while an unclipped one only touches it. Measured
in ``benchmarks/preprocess-params.md``.
"""

FULL_SCALE_TOLERANCE = 1.0 / 32768.0
"""How far below ``1.0`` an extreme may sit and still count as digital full scale.

One int16 quantisation step, because int16 audio decodes to a positive full scale of
``32767 / 32768`` against a negative full scale of exactly ``-1.0``; an equality test against
``1.0`` would therefore catch negative saturation and miss positive.
"""


@dataclass(frozen=True)
class ClipEvent:
    """One run of samples pinned near the recording's own amplitude extreme.

    Attributes:
        start_sample: First sample of the run, inclusive.
        end_sample: Last sample counted as part of the run, inclusive — including any trailing
            samples tolerated by the leniency window.
        polarity: ``"positive"`` when the run tracks the file's global maximum, ``"negative"`` when
            it tracks the global minimum.
    """

    start_sample: int
    end_sample: int
    polarity: str


def _required_run(extreme: float) -> int:
    """How many consecutive samples at ``extreme`` must occur before a clip event may open.

    Args:
        extreme: The file's global maximum or minimum.

    Returns:
        ``1`` when ``extreme`` sits at digital full scale, ``MINIMUM_EXTREME_RUN`` otherwise.
    """
    return 1 if abs(extreme) >= 1.0 - FULL_SCALE_TOLERANCE else MINIMUM_EXTREME_RUN


def _held_extreme(x: np.ndarray, value: float, minimum_run: int) -> np.ndarray:
    """Mark samples belonging to a run of ``minimum_run`` or more consecutive ``value``s.

    Args:
        x: The signal.
        value: The exact value to look for runs of.
        minimum_run: Shortest run that qualifies.

    Returns:
        A boolean mask over ``x``, true across every qualifying run and false elsewhere.
    """
    equal = x == value
    if minimum_run <= 1:
        return equal
    held = np.zeros_like(equal)
    edges = np.diff(np.concatenate(([False], equal, [False])).astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= minimum_run:
            held[start:stop] = True
    return held


def detect_clip_events(
    audio: Audio,
    *,
    near_threshold: float,
    leniency_samples: int,
    minimum_extreme: float,
) -> list[ClipEvent]:
    """ClipDaT: tag runs of samples pinned near the recording's own amplitude extreme.

    Hansen, Stauffer & Xia, "Nonlinear waveform distortion: Assessment and detection of clipping on
    speech data and systems", Speech Communication 134 (2021) 20-31, section 3 ("ClipDaT"), Fig. 9.
    The threshold is the file's own max/min rather than a fixed full-scale value, because pre-amp
    saturation does not always produce a perfectly flat top — the paper's Fig. 10b/10c show wavering
    near-clip amplitudes — and a gain change applied after clipping still leaves a run pinned at
    whatever became this file's own extreme.

    Algorithm: find the file's global max and min once; then walk the signal, and wherever an
    extreme is *held* — see below — open a clip event and keep extending it while subsequent samples
    stay within ``near_threshold`` of that same extreme, tolerating up to ``leniency_samples``
    consecutive samples that dip below it before closing the event at the sample where the tolerance
    was exceeded — the paper's own mechanism for surviving the amplitude wobble Fig. 10 documents,
    rather than fragmenting one clipped burst into many.

    What counts as held depends on where the extreme sits, and is decided per extreme, so a file's
    max and min may be treated differently:

    * **Below digital full scale** — the extreme must repeat across at least
      ``MINIMUM_EXTREME_RUN`` consecutive samples. A lone sample at a merely relative maximum is
      where the waveform happened to peak and is no evidence of clipping, whereas a saturated
      waveform holds its extreme. The repeat must be *consecutive*: a periodic signal revisits the
      identical extreme value once per period, so a count across the whole file does not separate
      the two cases.
    * **At digital full scale** (within ``FULL_SCALE_TOLERANCE`` of ``1.0``) — a single sample opens
      an event, no repeat required. Reaching the representable ceiling is itself evidence of
      saturation, and a single-sample clip is only detectable at all in this case.

    Args:
        audio: The recording. A multi-channel input is averaged, matching
            ``disruptions.detect_disruptions``'s convention elsewhere in this graph.
        near_threshold: Fraction of the file's own max/min a sample must reach to still count as
            part of the run. The paper's own reported constant is ``0.995``.
        leniency_samples: How many consecutive samples may fall below ``near_threshold`` before the
            run closes. The paper's own reported constant is ``3``.
        minimum_extreme: Below this absolute sample value, the file's global max/min is treated as
            noise floor rather than a genuine peak, and no event is ever opened. Not part of the
            paper's algorithm, which assumes a real recording's peak is meaningfully above zero; a
            near-silent file would otherwise open a spurious event at every sample near 0.0, since
            ``near_threshold`` of an extreme near 0.0 excludes almost nothing.

    Returns:
        The tagged clip events, in sample order. Empty when neither extreme is ever held — for a
        sub-full-scale extreme that means it never repeats across ``MINIMUM_EXTREME_RUN``
        consecutive samples, since a waveform that only touches its extreme in passing is not
        clipped — or when the file's peak never clears ``minimum_extreme``.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    n = len(x)
    if n == 0:
        return []
    global_max = float(x.max())
    global_min = float(x.min())
    # Each extreme is guarded on its own magnitude, not the pair's: a one-sided clip against a
    # zero baseline has a genuine positive peak but global_min == 0.0, which every baseline sample
    # matches exactly — treating 0.0 as a real "negative extreme" would open a clip event on the
    # entire baseline.
    watch_max = abs(global_max) >= minimum_extreme
    watch_min = abs(global_min) >= minimum_extreme
    if not watch_max and not watch_min:
        return []

    held_max = _held_extreme(x, global_max, _required_run(global_max)) if watch_max else np.zeros(n, dtype=bool)
    held_min = _held_extreme(x, global_min, _required_run(global_min)) if watch_min else np.zeros(n, dtype=bool)

    events: list[ClipEvent] = []
    i = 0
    while i < n:
        at_max = bool(held_max[i])
        at_min = bool(held_min[i])
        if not at_max and not at_min:
            i += 1
            continue
        polarity = "positive" if at_max else "negative"
        extreme = global_max if polarity == "positive" else global_min
        band = near_threshold * extreme
        below_run = 0
        end = i
        j = i + 1
        while j < n:
            within = (x[j] >= band) if polarity == "positive" else (x[j] <= band)
            end = j
            if within:
                below_run = 0
            else:
                below_run += 1
                if below_run > leniency_samples:
                    break
            j += 1
        events.append(ClipEvent(start_sample=i, end_sample=end, polarity=polarity))
        i = end + 1
    return events
