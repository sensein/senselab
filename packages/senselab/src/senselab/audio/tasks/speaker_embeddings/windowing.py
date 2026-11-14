"""Uniform windows over mono audio and the speaker embedding of each, through the `speaker_embeddings` task."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from senselab_core.kernel.results import Embedding
from senselab_core.kernel.signals import Audio
from senselab_core.runtime.call import Result
from senselab_worker.units import NS_PER_S


@dataclass(frozen=True)
class WindowEmbedding:
    """One embedding of a fixed time window `[start_s, end_s)`."""

    start_s: float
    end_s: float
    embedding: Embedding


def window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    """Return window start times spanning `[0, duration_s]`; the last is pinned so it ends at `duration_s`.

    Audio no longer than one window gives the single start 0; a non-positive duration, window or hop gives none.
    """
    if duration_s <= 0 or window_s <= 0 or hop_s <= 0:
        return []
    if duration_s <= window_s:
        return [0.0]
    count = math.floor((duration_s - window_s) / hop_s + 1e-9) + 1
    starts = [i * hop_s for i in range(count)]
    last = duration_s - window_s
    if starts[-1] < last - 1e-6:
        starts.append(last)
    return starts


def slice_audio(audio: Audio, start_s: float, end_s: float) -> Audio:
    """Return the audio over `[start_s, end_s)`, clamped to its bounds."""
    return Audio.from_array(audio.read(round(start_s * NS_PER_S), round(end_s * NS_PER_S)), audio.sampling_rate)


def embed_windows(
    audio: Audio, task: Callable[[Audio], Result], window_s: float = 1.0, hop_s: float = 0.5
) -> list[WindowEmbedding]:
    """Return the embedding of each window of `audio` (mono), one `task` call per window, in window order.

    Raises:
        ValueError: if `audio` is shorter than one window.
    """
    duration_s = audio.frames / audio.sampling_rate
    if duration_s < window_s:
        raise ValueError(f"audio of {duration_s:.3f} s is shorter than the {window_s} s window")
    spans = [(s, s + window_s) for s in window_starts(duration_s, window_s, hop_s)]
    return [
        WindowEmbedding(s, e, task(slice_audio(audio, s, e)).outputs["embedding"])  # type: ignore[arg-type]
        for s, e in spans
    ]
