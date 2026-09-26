"""The broadband amplitude envelope, in dBFS, and the recording's own global floor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import butter, filtfilt, hilbert, medfilt

from senselab.audio.data_structures import Audio


class EnvelopeSmoothing(Protocol):
    """A strategy for smoothing a rectified envelope, pluggable into :func:`hilbert_envelope_dbfs`.

    An implementation takes the envelope's linear magnitude and returns a same-length smoothed
    magnitude; the caller never sees which concrete strategy ran.
    """

    def apply(self, x: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Smooth ``x``, sampled at ``sampling_rate``."""
        ...


@dataclass(frozen=True)
class ButterworthSmoothing:
    """A zero-phase Butterworth lowpass (forward-and-backward ``filtfilt``).

    Being a resonant IIR design, its impulse response rings; a sharp onset in ``x`` can make it
    overshoot past a level the signal never took, including below zero on a rectified envelope, and
    zero-phase filtering spreads that ringing to both sides of the onset (a real, load-bearing
    property demonstrated in ``TestAnUnmeasurableSampleHasNoDecibelValue``, not a bug in this code).

    Attributes:
        cutoff_hz: Lowpass cutoff.
        order: Butterworth order; doubled in effect by the forward-and-backward pass.
    """

    cutoff_hz: float
    order: int

    def apply(self, x: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Zero-phase lowpass ``x`` at this instance's cutoff."""
        b, a = butter(self.order, self.cutoff_hz / (sampling_rate / 2), "low")
        return np.asarray(filtfilt(b, a, x), dtype=np.float64)


@dataclass(frozen=True)
class MedianSmoothing:
    """A sliding median.

    Its output is always one of the values already present in the window, so unlike
    :class:`ButterworthSmoothing` it cannot overshoot past a transient.

    Attributes:
        window_s: Width of the sliding window.
    """

    window_s: float

    def apply(self, x: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Median-filter ``x`` over this instance's window."""
        kernel = max(1, int(round(self.window_s * sampling_rate)) | 1)
        return np.asarray(medfilt(x, kernel_size=kernel), dtype=np.float64)


@dataclass(frozen=True)
class PercentileSmoothing:
    """A rolling percentile of the rectified magnitude.

    Generalizes :class:`MedianSmoothing` (its 50th-percentile case) toward the peak: a high
    percentile sits close to the local maximum within its window without being pinned to a single
    outlier sample the way a rolling maximum is — a fraction of the window (``100 - percentile``
    percent) may still exceed it, so one lone spike cannot drag the whole window's output up to its
    own height and hold it there after the spike has already passed.

    Attributes:
        window_s: Width of the sliding window.
        percentile: Which percentile within the window to report, in ``[0, 100]``.
    """

    window_s: float
    percentile: float

    def apply(self, x: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Percentile-filter ``x`` over this instance's window."""
        kernel = max(1, int(round(self.window_s * sampling_rate)))
        return np.asarray(
            percentile_filter(x, percentile=self.percentile, size=kernel, mode="reflect"), dtype=np.float64
        )


def analytic_magnitude(signal: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """``|hilbert(signal)|`` -- the unsmoothed rectified analytic magnitude.

    Args:
        signal: Real-valued samples. Any shape; the transform runs along ``axis``.
        axis: Axis the signal runs along.

    Returns:
        The analytic-signal magnitude, same shape as ``signal``.
    """
    return np.abs(hilbert(np.asarray(signal, dtype=np.float64), axis=axis))


@dataclass(frozen=True)
class AnalyticEnvelope:
    """One recording's rectified analytic magnitude, with what reading it in dB requires.

    Attributes:
        magnitude: ``|hilbert(x)|`` over the mono-averaged signal, one value per input sample.
        sampling_rate: Sampling rate of ``magnitude``, in Hz.
        resolution: Smallest magnitude the input representation can express; anything below it has
            no dB value.
    """

    magnitude: np.ndarray
    sampling_rate: int
    resolution: float


def analytic_envelope(audio: Audio) -> AnalyticEnvelope:
    """Transform once, so any number of smoothing strategies can be read off the same magnitude.

    Args:
        audio: Mono audio. A multi-channel input is averaged.

    Returns:
        The rectified analytic magnitude and what :func:`envelope_dbfs` needs to read it in dB.
    """
    source = np.asarray(audio.waveform)
    source_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float32)
    x = np.asarray(source, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    return AnalyticEnvelope(
        magnitude=analytic_magnitude(x),
        sampling_rate=int(audio.sampling_rate),
        resolution=float(np.finfo(source_dtype).eps),
    )


def envelope_dbfs(envelope: AnalyticEnvelope, *, smoothing: EnvelopeSmoothing) -> np.ndarray:
    """Smooth an already-computed analytic magnitude and read it in dBFS.

    Args:
        envelope: The magnitude to smooth, from :func:`analytic_envelope`.
        smoothing: Strategy applied to the magnitude before it is read in dB.

    Returns:
        One value per input sample, in dBFS, absolute and never normalised by the input's maximum.
        A sample whose smoothed envelope is non-positive or below the input representation's
        resolution has no dB value and reads ``nan``.
    """
    env = smoothing.apply(envelope.magnitude, envelope.sampling_rate)
    out = np.full(env.shape, np.nan)
    # A resonant smoothing (Butterworth) can ring through zero after a transient. Tiny positive
    # values on that crossing are numerical residue, not a measurable acoustic level; admitting them
    # into a local percentile creates implausible downward floor spikes and inflated span contrast.
    measurable = env >= envelope.resolution
    out[measurable] = 20.0 * np.log10(env[measurable])
    return out


def hilbert_envelope_dbfs(audio: Audio, *, smoothing: EnvelopeSmoothing) -> np.ndarray:
    """The analytic-signal magnitude, smoothed, in dBFS.

    A resonant smoothing strategy is offline-only if it is zero-phase; :class:`MedianSmoothing` has
    no phase response to speak of. Reading two smoothings off one recording costs two transforms
    through this entry point; call :func:`analytic_envelope` once and :func:`envelope_dbfs` per
    strategy instead.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        smoothing: Strategy applied to ``|hilbert(x)|`` before it is read in dB.

    Returns:
        One value per input sample, in dBFS, absolute and never normalised by the input's maximum.
        A sample whose smoothed envelope is non-positive or below the input representation's
        resolution has no dB value and reads ``nan``.
    """
    return envelope_dbfs(analytic_envelope(audio), smoothing=smoothing)


def global_floor_dbfs(envelope_db: np.ndarray, *, percentile: float) -> float:
    """A single low percentile of the envelope, over the whole recording.

    One number for the whole signal, not a rolling one: a rolling percentile of the envelope tracked
    whichever moment happened to be quietest inside whatever window currently surrounded the walk,
    which on continuous real speech with no genuine internal silence swings by 20+ dB across one
    uninterrupted utterance — "quietest recent speech" is not "background noise" when there is no
    pause long enough to expose one. It also fed a span's offset threshold from the floor at the
    peak's own sample, never re-read as the walk advanced away from it. A single global value removes
    both: there is no per-sample floor for a later sample to have drifted away from. Reading the
    percentile from the envelope rather than the raw waveform keeps it directly comparable to
    ``envelope_db`` in :func:`~senselab.audio.tasks.spans.api.propose_spans`'s ``rise``: a floor and
    an envelope are the same statistic of the same underlying quantity, not two different summaries
    of two different signals a fixed number of dB apart regardless of level.

    Args:
        envelope_db: Output of :func:`hilbert_envelope_dbfs`, ``nan`` where it had no dB value.
        percentile: Which percentile of the envelope is the floor. Read it from ``floor.percentile``.

    Returns:
        One dBFS value for the whole recording, over its measured samples. ``nan`` when nothing in
        the envelope was measurable at all — propagates cleanly to
        :class:`~senselab.audio.tasks.spans.api.NoContrast` rather than a fabricated number.
    """
    measured = envelope_db[np.isfinite(envelope_db)]
    return float(np.percentile(measured, percentile)) if measured.size else float("nan")


def dynamic_range_normalize(
    audio: Audio,
    *,
    macro_smoothing: EnvelopeSmoothing,
    micro_smoothing: EnvelopeSmoothing,
    target_dr_db: float,
    compression_ratio: float,
    macro_target_dbfs: float,
    gain_smoothing: EnvelopeSmoothing,
    floor_dbfs: float,
    ceiling: float,
) -> Audio:
    """Even out local dynamic range against the recording's own slow-moving loudness context.

    A quiet passage and a loud passage of the same recording are each brought toward
    ``macro_target_dbfs``, and within each, a passage whose *local* excursion above or below that
    passage's own macro level exceeds ``target_dr_db`` is compressed back toward it — so a loud
    transient in a quiet scene, or a quiet word in a loud scene, is not read as clipping or as
    silence by an instrument downstream that assumes one stable dynamic range for the whole file.

    Reads both envelopes off one :func:`analytic_envelope` through :func:`envelope_dbfs`, each with
    its own smoothing strategy: a slow one for the scene's macro loudness context and a fast one for
    the passage's own micro dynamics. The two are already in dB, so the local dynamic range is their
    plain difference rather than a ratio.

    Args:
        audio: The recording, already known to be clipping-free at the point this runs — this
            function only redistributes level, so a genuinely clipped sample stays clipped.
        macro_smoothing: Strategy for the slow, scene-level envelope. Construct from
            ``normalization.macro_smoothing``.
        micro_smoothing: Strategy for the fast, passage-level envelope. Construct from
            ``normalization.micro_smoothing``.
        target_dr_db: The local dynamic range, in dB, a passage may carry before its excess is
            compressed toward the macro level. Read it from ``normalization.target_dr_db``.
        compression_ratio: How much of the excess above ``target_dr_db`` is removed;
            ``compression_ratio=2`` removes half. Read it from ``normalization.compression_ratio``.
        macro_target_dbfs: The reference level every scene's macro envelope is brought toward. Read
            it from ``normalization.macro_target_dbfs``.
        gain_smoothing: Strategy that keeps the final gain curve continuous before it multiplies the
            waveform. A zero-phase Butterworth was tried here first and measured to fail on short
            events: its own resonance means it cannot settle to a brief event's correct gain within
            the event, injecting several-hundred-percent excess gain for most of a ~150 ms burst's
            duration rather than a short, localized ringing artifact at its edges — raising the
            cutoff did not fix it (still ~3.5x too high at the burst's centre with a 200 Hz cutoff),
            because the residual is the filter's own lagging response to the *macro* transition it
            is downstream of, not insufficient bandwidth. A short median settles to the correct
            plateau almost immediately regardless of window width, at the cost of a bounded rather
            than a ramped transition — construct from ``normalization.gain_smoothing``.
        floor_dbfs: The dB value substituted wherever an envelope has no measurable value (silence,
            or below the input's numeric resolution — see :func:`envelope_dbfs`), so the gain curve
            stays finite through a silent stretch instead of propagating ``nan``. Read it from
            ``normalization.floor_dbfs``.
        ceiling: The absolute peak the processed waveform is rescaled to if it would otherwise
            exceed it — a numeric safety bound on this operation's own output, not a perceptual
            choice. Read it from ``normalization.ceiling``.

    Returns:
        The gain-adjusted audio, same shape, sampling rate and channel count as the input (a
        multi-channel input has the same mono gain curve applied to every channel).
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    mono = x.mean(axis=0) if x.ndim > 1 else x
    mono_audio = Audio(waveform=mono[None, :], sampling_rate=audio.sampling_rate)

    envelope = analytic_envelope(mono_audio)
    macro_db = envelope_dbfs(envelope, smoothing=macro_smoothing)
    micro_db = envelope_dbfs(envelope, smoothing=micro_smoothing)
    macro_db = np.where(np.isfinite(macro_db), macro_db, floor_dbfs)
    micro_db = np.where(np.isfinite(micro_db), micro_db, floor_dbfs)

    local_dr_db = micro_db - macro_db
    gain_db = np.zeros_like(local_dr_db)
    over_target = local_dr_db > 0.0
    gain_db[over_target] = -local_dr_db[over_target] * (1.0 - 1.0 / compression_ratio)
    under_target = local_dr_db < -target_dr_db
    gain_db[under_target] = (-target_dr_db - local_dr_db[under_target]) * (1.0 - 1.0 / compression_ratio)

    combined_gain_db = gain_db + (macro_target_dbfs - macro_db)
    combined_gain = np.power(10.0, combined_gain_db / 20.0)
    smooth_gain = gain_smoothing.apply(combined_gain, int(audio.sampling_rate))

    processed = x * smooth_gain
    peak = float(np.abs(processed).max()) if processed.size else 0.0
    if peak > ceiling:
        processed = processed * (ceiling / peak)
    return Audio(waveform=processed.astype(np.float32), sampling_rate=audio.sampling_rate)
