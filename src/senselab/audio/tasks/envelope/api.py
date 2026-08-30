"""The broadband amplitude envelope, in dBFS, and a floor that tracks the recording."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from senselab.audio.data_structures import Audio


def hilbert_envelope_dbfs(audio: Audio, *, lowpass_hz: float, filter_order: int) -> np.ndarray:
    """The analytic-signal magnitude, lowpassed, in dBFS.

    The filter is zero-phase, so the envelope is offline-only.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        lowpass_hz: Cutoff of the zero-phase Butterworth lowpass. Read it from
            ``envelope.lowpass_hz`` in the triage config.
        filter_order: Order of the Butterworth design. Read it from ``envelope.filter_order``.

    Returns:
        One value per input sample, in dBFS, absolute and never normalised by the input's maximum.
        A sample whose filtered envelope is non-positive or below the input representation's
        resolution has no dB value and reads ``nan``.
    """
    source = np.asarray(audio.waveform)
    source_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float32)
    resolution = float(np.finfo(source_dtype).eps)
    x = np.asarray(source, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    b, a = butter(filter_order, lowpass_hz / (audio.sampling_rate / 2), "low")
    env = np.asarray(filtfilt(b, a, np.abs(hilbert(x))), dtype=np.float64)
    out = np.full(env.shape, np.nan)
    # A zero-phase lowpass can ring through zero after a transient. Tiny positive values on that
    # crossing are numerical residue, not a measurable acoustic level; admitting them into a local
    # percentile creates implausible downward floor spikes and inflated span contrast.
    measurable = env >= resolution
    out[measurable] = 20.0 * np.log10(env[measurable])
    return out


def rolling_floor_dbfs(
    envelope_db: np.ndarray,
    sampling_rate: int,
    *,
    window_s: float,
    percentile: float,
    eval_grid_s: float,
) -> np.ndarray:
    """A low percentile of the envelope over a sliding window.

    Args:
        envelope_db: Output of :func:`hilbert_envelope_dbfs`, ``nan`` where it had no dB value.
        sampling_rate: Samples per second of ``envelope_db``.
        window_s: Width of the sliding window. Read it from ``floor.window_s``.
        percentile: Which percentile within the window is the floor. Config `floor.percentile`.
        eval_grid_s: How often the percentile is evaluated before interpolation. Read it from
            ``floor.eval_grid_s``.

    Returns:
        One floor value per sample of ``envelope_db``, taken over that window's measured samples.
        A window holding none reads ``nan``, and so does every sample interpolating from it.
    """
    n = len(envelope_db)
    half = int(window_s * sampling_rate) // 2
    step = max(1, int(eval_grid_s * sampling_rate))
    centres = range(0, n, step)
    vals: list[float] = []
    for c in centres:
        window = envelope_db[max(0, c - half) : min(n, c + half)]
        measured = window[np.isfinite(window)]
        vals.append(float(np.percentile(measured, percentile)) if measured.size else float("nan"))
    return np.interp(np.arange(n), centres, vals)


def _zero_phase_lowpass(x: np.ndarray, cutoff_hz: float, sampling_rate: int, *, order: int) -> np.ndarray:
    """A forward-and-backward Butterworth lowpass, for smoothing a gain curve rather than audio."""
    b, a = butter(order, cutoff_hz / (sampling_rate / 2), "low")
    return np.asarray(filtfilt(b, a, x), dtype=np.float64)


def dynamic_range_normalize(
    audio: Audio,
    *,
    macro_lowpass_hz: float,
    micro_lowpass_hz: float,
    envelope_filter_order: int,
    target_dr_db: float,
    compression_ratio: float,
    macro_target_dbfs: float,
    gain_smooth_hz: float,
    gain_filter_order: int,
    floor_dbfs: float,
    ceiling: float,
) -> Audio:
    """Even out local dynamic range against the recording's own slow-moving loudness context.

    A quiet passage and a loud passage of the same recording are each brought toward
    ``macro_target_dbfs``, and within each, a passage whose *local* excursion above or below that
    passage's own macro level exceeds ``target_dr_db`` is compressed back toward it — so a loud
    transient in a quiet scene, or a quiet word in a loud scene, is not read as clipping or as
    silence by an instrument downstream that assumes one stable dynamic range for the whole file.
    The whole operation is one continuous, zero-phase gain curve applied to the original waveform, so
    it introduces no discontinuity a later disruption detector could mistake for one in the recording
    itself.

    Reuses :func:`hilbert_envelope_dbfs` for both envelopes, at two cutoffs: a slow one for the
    scene's macro loudness context and a fast one for the passage's own micro dynamics. The two are
    already in dB, so the local dynamic range is their plain difference rather than a ratio.

    Args:
        audio: The recording, already known to be clipping-free at the point this runs — this
            function only redistributes level, so a genuinely clipped sample stays clipped.
        macro_lowpass_hz: Cutoff for the slow, scene-level envelope. Read it from
            ``normalization.macro_lowpass_hz``.
        micro_lowpass_hz: Cutoff for the fast, passage-level envelope. Read it from
            ``normalization.micro_lowpass_hz``.
        envelope_filter_order: Butterworth order shared by both envelopes, forwarded to
            :func:`hilbert_envelope_dbfs`. Read it from ``normalization.envelope_filter_order``.
        target_dr_db: The local dynamic range, in dB, a passage may carry before its excess is
            compressed toward the macro level. Read it from ``normalization.target_dr_db``.
        compression_ratio: How much of the excess above ``target_dr_db`` is removed;
            ``compression_ratio=2`` removes half. Read it from ``normalization.compression_ratio``.
        macro_target_dbfs: The reference level every scene's macro envelope is brought toward. Read
            it from ``normalization.macro_target_dbfs``.
        gain_smooth_hz: Cutoff of the final zero-phase smoothing pass over the combined gain curve,
            the identity that keeps the applied gain itself free of any discontinuity. Read it from
            ``normalization.gain_smooth_hz``.
        gain_filter_order: Butterworth order for that smoothing pass. Read it from
            ``normalization.gain_filter_order``.
        floor_dbfs: The dB value substituted wherever an envelope has no measurable value (silence,
            or below the input's numeric resolution — see :func:`hilbert_envelope_dbfs`), so the gain
            curve stays finite through a silent stretch instead of propagating ``nan``. Read it from
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

    macro_db = hilbert_envelope_dbfs(mono_audio, lowpass_hz=macro_lowpass_hz, filter_order=envelope_filter_order)
    micro_db = hilbert_envelope_dbfs(mono_audio, lowpass_hz=micro_lowpass_hz, filter_order=envelope_filter_order)
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
    smooth_gain = _zero_phase_lowpass(combined_gain, gain_smooth_hz, int(audio.sampling_rate), order=gain_filter_order)

    processed = x * smooth_gain
    peak = float(np.abs(processed).max()) if processed.size else 0.0
    if peak > ceiling:
        processed = processed * (ceiling / peak)
    return Audio(waveform=processed.astype(np.float32), sampling_rate=audio.sampling_rate)
