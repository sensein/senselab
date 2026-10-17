"""Probabilistic phone onsets and offsets from a phonetic posteriorgram, and timing helpers over them.

A phone event is a run of frames sharing one most-probable phone. Its boundaries are placed with
sub-frame precision where the outgoing and incoming phones' posteriors cross, and each boundary
carries the width of the transition around it and a logistic-equivalent standard deviation.
Definitions, their derivations and measurements are in ``specs/20261017-104500-sparse-ppg-boundaries/``.

Frame ``i`` is centred at ``(i + 0.5) / frame_rate`` seconds (ppgs' 160-sample hop with its 432-sample
reflect pad). Times here are therefore half a frame later than the frame-edge times
:func:`~senselab.audio.tasks.features_extraction.ppg.extract_ppg_segments` reports.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS, PPGS_FRAME_RATE
from senselab.audio.tasks.features_extraction.sparse_ppg import PosteriorgramLike, as_frame_major_array

SILENCE_LABEL = "<silent>"
"""The ppgs label for non-speech frames."""

PHONE_CLASSES: Dict[str, str] = {
    **{p: "vowel" for p in ("aa", "ae", "ah", "ao", "eh", "er", "ih", "iy", "uh", "uw")},
    **{p: "diphthong" for p in ("aw", "ay", "ey", "ow", "oy")},
    **{p: "stop" for p in ("b", "d", "g", "k", "p", "t")},
    **{p: "affricate" for p in ("ch", "jh")},
    **{p: "fricative" for p in ("dh", "f", "hh", "s", "sh", "th", "v", "z", "zh")},
    **{p: "nasal" for p in ("m", "n", "ng")},
    **{p: "liquid" for p in ("l", "r")},
    **{p: "glide" for p in ("w", "y")},
    SILENCE_LABEL: "silence",
}
"""ARPAbet manner class of every ppgs phone."""

FILLER_CLASSES: Tuple[str, ...] = ("vowel", "diphthong", "nasal")
"""Classes whose posterior mass supports a filled pause (``uh`` = ``ah``; ``um`` = ``ah`` + ``m``)."""


def logistic_sigma_from_width(width_s: Union[float, np.ndarray], level: float) -> np.ndarray:
    """Standard deviation of the logistic transition whose ``level`` crossings are ``width_s`` apart.

    A logistic ``p(t) = 1 / (1 + exp(-(t - t0) / s))`` crosses ``level`` and ``1 - level`` at
    ``t0 ± s ln(level / (1 - level))``; its standard deviation is ``s π / √3``.

    Args:
        width_s: Transition widths, seconds.
        level: The posterior level the widths were measured at, in ``(0.5, 1)``.

    Returns:
        The standard deviations, seconds.
    """
    return np.asarray(width_s) * math.pi / (2.0 * math.sqrt(3.0) * math.log(level / (1.0 - level)))


class PhoneEvents(BaseModel):
    """Columnar phone events, one row per run of frames with the same most probable phone.

    Boundary columns (``*_width_s``, ``*_sigma_s``, ``*_sharpness``) are NaN where the event touches
    the start or end of the posteriorgram, since no transition is observed there.

    Attributes:
        phones: The phone vocabulary ``phone_index`` indexes.
        frame_rate: Frames per second of the source posteriorgram.
        level: Posterior level the transition widths were measured at.
        phone_index: Phone of each event.
        start_frame: First frame of each event's run.
        end_frame: Last frame of each event's run, inclusive.
        onset_s: Sub-frame onset, seconds: where the incoming and outgoing posteriors cross.
        offset_s: Sub-frame offset, seconds.
        onset_width_s: Time from the outgoing phone falling below ``level`` to this phone rising above it.
        offset_width_s: The same width at the offset.
        onset_sigma_s: :func:`logistic_sigma_from_width` of ``onset_width_s``.
        offset_sigma_s: :func:`logistic_sigma_from_width` of ``offset_width_s``.
        onset_sharpness: Half the jump, across the crossing frame pair, in incoming minus outgoing
            posterior; 1 is a complete swap within one frame.
        offset_sharpness: The same at the offset.
        peak_prob: Highest posterior of the event's phone over its run.
        mean_prob: Mean posterior of the event's phone over its run.
        mean_entropy_bits: Mean over the run of each frame's posterior entropy, bits.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    phones: Tuple[str, ...]
    frame_rate: float
    level: float
    phone_index: np.ndarray
    start_frame: np.ndarray
    end_frame: np.ndarray
    onset_s: np.ndarray
    offset_s: np.ndarray
    onset_width_s: np.ndarray
    offset_width_s: np.ndarray
    onset_sigma_s: np.ndarray
    offset_sigma_s: np.ndarray
    onset_sharpness: np.ndarray
    offset_sharpness: np.ndarray
    peak_prob: np.ndarray
    mean_prob: np.ndarray
    mean_entropy_bits: np.ndarray

    def __len__(self) -> int:
        """Number of events."""
        return int(len(self.phone_index))

    @property
    def labels(self) -> np.ndarray:
        """Phone label of each event."""
        return np.asarray(self.phones, dtype=object)[self.phone_index]

    @property
    def classes(self) -> np.ndarray:
        """:data:`PHONE_CLASSES` class of each event; ``"other"`` for a label outside it."""
        return np.asarray([PHONE_CLASSES.get(label, "other") for label in self.labels], dtype=object)

    @property
    def durations_s(self) -> np.ndarray:
        """Offset minus onset of each event, seconds."""
        return self.offset_s - self.onset_s

    @property
    def is_silence(self) -> np.ndarray:
        """Whether each event is the silence label."""
        return self.labels == SILENCE_LABEL

    @property
    def is_interior(self) -> np.ndarray:
        """Whether each event has an observed transition at both ends."""
        return ~(np.isnan(self.onset_sigma_s) | np.isnan(self.offset_sigma_s))

    def to_records(self) -> List[Dict[str, object]]:
        """The events as one dict per row, with ``phone`` and ``phone_class`` added.

        Returns:
            The rows, in time order.
        """
        columns = [name for name in type(self).model_fields if name not in ("phones", "frame_rate", "level")]
        labels, classes = self.labels, self.classes
        return [
            {"phone": labels[i], "phone_class": classes[i], **{c: getattr(self, c)[i].item() for c in columns}}
            for i in range(len(self))
        ]


def _runs(argmax: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(argmax[1:] != argmax[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(argmax) - 1]))
    return starts, ends


def _fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(min(1.0, max(0.0, numerator / denominator)))


def _frame_entropy_bits(dense: np.ndarray) -> np.ndarray:
    p = np.clip(dense, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=1)


def ppg_phone_events(
    ppg: PosteriorgramLike,
    frame_rate: Optional[float] = None,
    phones: Tuple[str, ...] = PHONEME_LABELS,
    level: float = 0.8,
) -> PhoneEvents:
    """Phone events with sub-frame onsets and offsets and their transition uncertainty.

    Each boundary between run ``A`` (phone ``a``, last frame ``i``) and run ``B`` (phone ``b``) sits
    where ``d = p_b - p_a`` crosses zero, linearly interpolated between frames ``i`` and ``i + 1``.
    Its width runs from ``p_a`` last falling below ``level`` within ``A`` to ``p_b`` first rising above
    it within ``B``, each linearly interpolated; a run whose phone never reaches ``level`` is measured
    at its own peak posterior instead.

    Args:
        ppg: A posteriorgram in any form :func:`as_frame_major_array` accepts.
        frame_rate: Frames per second; None takes a :class:`SparsePPG`'s own, else ppgs' 100.
        phones: The phone vocabulary of the phone axis.
        level: Posterior level transition widths are measured at, in ``(0.5, 1)``.

    Returns:
        The events in time order; empty for an empty posteriorgram.

    Raises:
        ValueError: If ``level`` is outside ``(0.5, 1)`` or the posteriorgram contains NaN.
    """
    if not 0.5 < level < 1.0:
        raise ValueError(f"level must lie in (0.5, 1), got {level}")
    if frame_rate is None:
        frame_rate = float(getattr(ppg, "frame_rate", PPGS_FRAME_RATE))
    dense = as_frame_major_array(ppg, num_phones=len(phones)).astype(np.float64)
    if np.isnan(dense).any():
        raise ValueError("posteriorgram contains NaN")
    num_frames = dense.shape[0]
    dt = 1.0 / frame_rate

    if num_frames == 0:
        empty_f, empty_i = np.zeros(0), np.zeros(0, dtype=np.int64)
        return PhoneEvents(
            phones=tuple(phones),
            frame_rate=frame_rate,
            level=level,
            phone_index=empty_i,
            start_frame=empty_i,
            end_frame=empty_i,
            **{
                name: empty_f
                for name in (
                    "onset_s",
                    "offset_s",
                    "onset_width_s",
                    "offset_width_s",
                    "onset_sigma_s",
                    "offset_sigma_s",
                    "onset_sharpness",
                    "offset_sharpness",
                    "peak_prob",
                    "mean_prob",
                    "mean_entropy_bits",
                )
            },
        )

    argmax = dense.argmax(axis=1)
    starts, ends = _runs(argmax)
    run_phone = argmax[starts]
    num_runs = len(starts)
    centre = (np.arange(num_frames) + 0.5) * dt

    own = dense[np.arange(num_frames), argmax]
    cumulative = np.concatenate(([0.0], np.cumsum(own)))
    mean_prob = (cumulative[ends + 1] - cumulative[starts]) / (ends - starts + 1)
    peak_prob = np.maximum.reduceat(own, starts)
    entropy = _frame_entropy_bits(dense)
    cumulative_entropy = np.concatenate(([0.0], np.cumsum(entropy)))
    mean_entropy = (cumulative_entropy[ends + 1] - cumulative_entropy[starts]) / (ends - starts + 1)

    num_boundaries = num_runs - 1
    crossing = np.empty(num_boundaries)
    width = np.empty(num_boundaries)
    sharpness = np.empty(num_boundaries)
    for r in range(num_boundaries):
        a, b = run_phone[r], run_phone[r + 1]
        i = ends[r]
        d_i = dense[i, b] - dense[i, a]
        d_next = dense[i + 1, b] - dense[i + 1, a]
        jump = d_next - d_i
        fraction = -d_i / jump if jump > 0 else 0.5
        crossing[r] = centre[i] + fraction * dt
        sharpness[r] = jump / 2.0

        fall_level = min(level, peak_prob[r])
        p_a = dense[starts[r] : i + 2, a]
        above = np.flatnonzero(p_a[:-1] >= fall_level)
        j = int(above[-1])
        t_fall = centre[starts[r] + j] + _fraction(p_a[j] - fall_level, p_a[j] - p_a[j + 1]) * dt

        rise_level = min(level, peak_prob[r + 1])
        p_b = dense[i : ends[r + 1] + 1, b]
        reached = np.flatnonzero(p_b >= rise_level)
        k = int(reached[reached >= 1][0])
        t_rise = centre[i + k - 1] + _fraction(rise_level - p_b[k - 1], p_b[k] - p_b[k - 1]) * dt
        width[r] = max(0.0, t_rise - t_fall)

    sigma = logistic_sigma_from_width(width, level)
    nan = np.array([np.nan])
    return PhoneEvents(
        phones=tuple(phones),
        frame_rate=frame_rate,
        level=level,
        phone_index=run_phone.astype(np.int64),
        start_frame=starts.astype(np.int64),
        end_frame=ends.astype(np.int64),
        onset_s=np.concatenate(([0.0], crossing)),
        offset_s=np.concatenate((crossing, [num_frames * dt])),
        onset_width_s=np.concatenate((nan, width)),
        offset_width_s=np.concatenate((width, nan)),
        onset_sigma_s=np.concatenate((nan, sigma)),
        offset_sigma_s=np.concatenate((sigma, nan)),
        onset_sharpness=np.concatenate((nan, sharpness)),
        offset_sharpness=np.concatenate((sharpness, nan)),
        peak_prob=peak_prob,
        mean_prob=mean_prob,
        mean_entropy_bits=mean_entropy,
    )


def ppg_boundary_probability(ppg: PosteriorgramLike, num_phones: int = len(PHONEME_LABELS)) -> np.ndarray:
    """Per-frame total-variation change of the posterior, a continuous segmentation signal.

    ``b[t] = 0.5 * Σ_k |p_k(t) - p_k(t - 1)|``, in ``[0, 1]`` for normalized frames, located at
    ``t / frame_rate`` seconds (midway between the two frame centres); ``b[0] = 0``.

    Args:
        ppg: A posteriorgram in any form :func:`as_frame_major_array` accepts.
        num_phones: The phone-axis length.

    Returns:
        A ``(frames,)`` float32 array.
    """
    dense = as_frame_major_array(ppg, num_phones=num_phones)
    out = np.zeros(dense.shape[0], dtype=np.float32)
    if dense.shape[0] > 1:
        out[1:] = 0.5 * np.abs(np.diff(dense, axis=0)).sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# Coordination helpers
# ---------------------------------------------------------------------------

SpeakerSpan = Tuple[str, float, float]
"""``(speaker, start_s, end_s)``."""


def _merged(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals if e > s):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _spans_by_speaker(speaker_spans: Sequence[SpeakerSpan]) -> Dict[str, List[Tuple[float, float]]]:
    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for speaker, start, end in speaker_spans:
        grouped.setdefault(str(speaker), []).append((float(start), float(end)))
    return {speaker: _merged(spans) for speaker, spans in grouped.items()}


def _inside(times: np.ndarray, spans: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for start, end in spans:
        mask |= (times >= start) & (times < end)
    return mask


def _overlap_s(onsets: np.ndarray, offsets: np.ndarray, spans: List[Tuple[float, float]]) -> float:
    total = 0.0
    for start, end in spans:
        total += float(np.clip(np.minimum(offsets, end) - np.maximum(onsets, start), 0.0, None).sum())
    return total


def phone_rate_by_speaker(events: PhoneEvents, speaker_spans: Sequence[SpeakerSpan]) -> Dict[str, Dict[str, float]]:
    """Phones per second of each speaker's speech.

    A non-silent event belongs to a speaker when its midpoint lies in one of that speaker's spans.

    Args:
        events: Phone events of the recording.
        speaker_spans: ``(speaker, start_s, end_s)`` spans; a speaker's overlapping spans are merged.

    Returns:
        Per speaker: ``speech_s`` (merged span time), ``silence_s`` (silent-event time inside those
        spans), ``n_phones``, ``phone_rate`` (``n_phones / speech_s``) and ``articulation_rate``
        (``n_phones / (speech_s - silence_s)``); a rate is NaN when its denominator is zero.
    """
    midpoints = (events.onset_s + events.offset_s) / 2.0
    speech = ~events.is_silence
    out: Dict[str, Dict[str, float]] = {}
    for speaker, spans in _spans_by_speaker(speaker_spans).items():
        speech_s = sum(end - start for start, end in spans)
        silent = events.is_silence
        silence_s = _overlap_s(events.onset_s[silent], events.offset_s[silent], spans)
        n_phones = int((_inside(midpoints, spans) & speech).sum())
        articulating_s = speech_s - silence_s
        out[speaker] = {
            "speech_s": speech_s,
            "silence_s": silence_s,
            "n_phones": float(n_phones),
            "phone_rate": n_phones / speech_s if speech_s > 0 else float("nan"),
            "articulation_rate": n_phones / articulating_s if articulating_s > 0 else float("nan"),
        }
    return out


def _duration_summary(durations: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(len(durations)),
        "mean_s": float(durations.mean()),
        "median_s": float(np.median(durations)),
        "std_s": float(durations.std(ddof=1)) if len(durations) > 1 else float("nan"),
        "p10_s": float(np.percentile(durations, 10)),
        "p90_s": float(np.percentile(durations, 90)),
    }


def phone_class_duration_stats(
    events: PhoneEvents,
    speaker_spans: Optional[Sequence[SpeakerSpan]] = None,
    classes: Mapping[str, str] = PHONE_CLASSES,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Duration statistics of interior events, per phone class.

    Events touching the start or end of the posteriorgram are excluded, since their duration is cut.

    Args:
        events: Phone events of the recording.
        speaker_spans: When given, statistics are per speaker, assigning events by midpoint as
            :func:`phone_rate_by_speaker` does; otherwise everything is under the key ``"all"``.
        classes: Phone label to class.

    Returns:
        ``{speaker: {class: {count, mean_s, median_s, std_s, p10_s, p90_s}}}``; a class with no
        interior event is absent.
    """
    labels = events.labels
    event_class = np.asarray([classes.get(label, "other") for label in labels], dtype=object)
    durations = events.durations_s
    midpoints = (events.onset_s + events.offset_s) / 2.0
    interior = events.is_interior
    groups = (
        {"all": np.ones(len(events), dtype=bool)}
        if speaker_spans is None
        else {speaker: _inside(midpoints, spans) for speaker, spans in _spans_by_speaker(speaker_spans).items()}
    )
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for speaker, member in groups.items():
        selected = member & interior
        out[speaker] = {
            name: _duration_summary(durations[selected & (event_class == name)])
            for name in sorted(set(event_class[selected]))
        }
    return out


class TurnTransitionTiming(BaseModel):
    """The phone-level gap across one turn transition.

    Attributes:
        prev_end_s: The outgoing speaker's span end the transition was given with.
        next_start_s: The incoming speaker's span start.
        last_phone: Label of the outgoing speaker's last non-silent phone, or None if none was found.
        last_phone_offset_s: Its offset.
        last_phone_offset_sigma_s: Its offset's standard deviation.
        first_phone: Label of the incoming speaker's first non-silent phone, or None.
        first_phone_onset_s: Its onset.
        first_phone_onset_sigma_s: Its onset's standard deviation.
        gap_s: ``first_phone_onset_s - last_phone_offset_s``; negative is overlap.
        gap_sigma_s: ``sqrt(offset_sigma² + onset_sigma²)``, treating the two boundaries as independent.
    """

    prev_end_s: float
    next_start_s: float
    last_phone: Optional[str] = None
    last_phone_offset_s: Optional[float] = None
    last_phone_offset_sigma_s: Optional[float] = None
    first_phone: Optional[str] = None
    first_phone_onset_s: Optional[float] = None
    first_phone_onset_sigma_s: Optional[float] = None
    gap_s: Optional[float] = None
    gap_sigma_s: Optional[float] = None


def _nearest(times: np.ndarray, candidates: np.ndarray, target: float, search_s: float) -> Optional[int]:
    eligible = candidates & (np.abs(times - target) <= search_s)
    if not eligible.any():
        return None
    indices = np.flatnonzero(eligible)
    return int(indices[np.argmin(np.abs(times[indices] - target))])


def turn_transition_timing(
    events: PhoneEvents,
    transitions: Sequence[Tuple[float, float]],
    search_s: float = 0.5,
) -> List[TurnTransitionTiming]:
    """Last-phone offset to first-phone onset across each turn transition.

    For each ``(prev_end_s, next_start_s)``, the outgoing phone is the non-silent event whose offset is
    nearest ``prev_end_s`` and the incoming one the non-silent event whose onset is nearest
    ``next_start_s``, each within ``search_s``. A boundary at the posteriorgram edge has no observed
    transition; its sigma is reported as None.

    Args:
        events: Phone events of the recording.
        transitions: ``(prev_speaker_end_s, next_speaker_start_s)`` pairs, e.g. from diarization.
        search_s: Furthest a phone boundary may lie from the span edge it is matched to, seconds.

    Returns:
        One timing per transition, in input order; fields stay None where no phone was found.
    """
    speech = ~events.is_silence
    labels = events.labels
    out: List[TurnTransitionTiming] = []
    for prev_end, next_start in transitions:
        timing = TurnTransitionTiming(prev_end_s=float(prev_end), next_start_s=float(next_start))
        last = _nearest(events.offset_s, speech, float(prev_end), search_s)
        first = _nearest(events.onset_s, speech, float(next_start), search_s)
        updates: Dict[str, object] = {}
        if last is not None:
            sigma = float(events.offset_sigma_s[last])
            updates.update(
                last_phone=str(labels[last]),
                last_phone_offset_s=float(events.offset_s[last]),
                last_phone_offset_sigma_s=None if math.isnan(sigma) else sigma,
            )
        if first is not None:
            sigma = float(events.onset_sigma_s[first])
            updates.update(
                first_phone=str(labels[first]),
                first_phone_onset_s=float(events.onset_s[first]),
                first_phone_onset_sigma_s=None if math.isnan(sigma) else sigma,
            )
        if last is not None and first is not None:
            off_sigma, on_sigma = updates["last_phone_offset_sigma_s"], updates["first_phone_onset_sigma_s"]
            updates["gap_s"] = float(events.onset_s[first] - events.offset_s[last])
            if off_sigma is not None and on_sigma is not None:
                updates["gap_sigma_s"] = math.hypot(float(off_sigma), float(on_sigma))  # type: ignore[arg-type]
        out.append(timing.model_copy(update=updates))
    return out


class FillerSupport(BaseModel):
    """How much a posteriorgram span looks like a filled pause.

    Attributes:
        start_s: Span start the support was measured over.
        end_s: Span end.
        n_frames: Frames whose centre lies in the span.
        support: Mean per-frame posterior mass on :data:`FILLER_CLASSES` phones.
        vowel_mass: Mean per-frame mass on vowel and diphthong phones.
        nasal_mass: Mean per-frame mass on nasal phones.
        silence_mass: Mean per-frame mass on the silence label.
        stability: One minus the mean :func:`ppg_boundary_probability` over the span's internal frame
            steps; 1 is an unchanging posterior.
        dominant_phone: Phone with the highest mean posterior over the span.
        dominant_mean_prob: That mean posterior.
        mean_entropy_bits: Mean per-frame posterior entropy, bits.
    """

    start_s: float
    end_s: float
    n_frames: int
    support: float
    vowel_mass: float
    nasal_mass: float
    silence_mass: float
    stability: float
    dominant_phone: Optional[str]
    dominant_mean_prob: float
    mean_entropy_bits: float


def filler_support(
    ppg: PosteriorgramLike,
    start_s: float,
    end_s: float,
    frame_rate: Optional[float] = None,
    phones: Tuple[str, ...] = PHONEME_LABELS,
) -> FillerSupport:
    """Vowel/nasal posterior mass and its stability over a span, to verify an ASR-detected filler.

    Args:
        ppg: A posteriorgram in any form :func:`as_frame_major_array` accepts.
        start_s: Span start, seconds.
        end_s: Span end, seconds.
        frame_rate: Frames per second; None takes a :class:`SparsePPG`'s own, else ppgs' 100.
        phones: The phone vocabulary of the phone axis.

    Returns:
        The support; every mass is NaN and ``dominant_phone`` None when no frame centre lies in the span.
    """
    if frame_rate is None:
        frame_rate = float(getattr(ppg, "frame_rate", PPGS_FRAME_RATE))
    dense = as_frame_major_array(ppg, num_phones=len(phones))
    centre = (np.arange(dense.shape[0]) + 0.5) / frame_rate
    window = dense[(centre >= start_s) & (centre < end_s)].astype(np.float64)
    nan = float("nan")
    if window.shape[0] == 0:
        return FillerSupport(
            start_s=start_s,
            end_s=end_s,
            n_frames=0,
            support=nan,
            vowel_mass=nan,
            nasal_mass=nan,
            silence_mass=nan,
            stability=nan,
            dominant_phone=None,
            dominant_mean_prob=nan,
            mean_entropy_bits=nan,
        )
    phone_class = np.asarray([PHONE_CLASSES.get(p, "other") for p in phones], dtype=object)

    def mass(names: Tuple[str, ...]) -> float:
        return float(window[:, np.isin(phone_class, names)].sum(axis=1).mean())

    mean_posterior = window.mean(axis=0)
    dominant = int(mean_posterior.argmax())
    steps = 0.5 * np.abs(np.diff(window, axis=0)).sum(axis=1)
    return FillerSupport(
        start_s=start_s,
        end_s=end_s,
        n_frames=int(window.shape[0]),
        support=mass(FILLER_CLASSES),
        vowel_mass=mass(("vowel", "diphthong")),
        nasal_mass=mass(("nasal",)),
        silence_mass=mass(("silence",)),
        stability=float(1.0 - steps.mean()) if steps.size else 1.0,
        dominant_phone=str(phones[dominant]),
        dominant_mean_prob=float(mean_posterior[dominant]),
        mean_entropy_bits=float(_frame_entropy_bits(window).mean()),
    )
