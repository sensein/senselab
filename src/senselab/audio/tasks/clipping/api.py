"""Clip-event detection: ClipDaT (Hansen, Stauffer & Xia, Speech Communication 134 (2021) 20-31)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio


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

    Algorithm: find the file's global max and min once; then walk the signal, and whenever a sample
    equals the max or the min, open a clip event and keep extending it while subsequent samples stay
    within ``near_threshold`` of that same extreme, tolerating up to ``leniency_samples`` consecutive
    samples that dip below it before closing the event at the sample where the tolerance was
    exceeded — the paper's own mechanism for surviving the amplitude wobble Fig. 10 documents,
    rather than fragmenting one clipped burst into many.

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
        The tagged clip events, in sample order. Empty when the file's peak amplitude never repeats
        — nothing is "clipped" if the extreme sample value occurs only once — or when the file's
        peak never clears ``minimum_extreme``.
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

    events: list[ClipEvent] = []
    i = 0
    while i < n:
        at_max = watch_max and x[i] == global_max
        at_min = watch_min and x[i] == global_min
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
