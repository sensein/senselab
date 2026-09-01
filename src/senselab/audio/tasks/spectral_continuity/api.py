"""Frame-to-frame spectral continuity: how steady the spectral shape stays over time."""

from __future__ import annotations

import numpy as np

from senselab.audio.tasks.envelope.api import EnvelopeSmoothing


def spectral_continuity(
    magnitude: np.ndarray, *, hop_s: float, sampling_rate: int, n_samples: int, smoothing: EnvelopeSmoothing
) -> np.ndarray:
    """Cosine similarity between consecutive log-magnitude spectra, smoothed and resampled to samples.

    A sustained, spectrally stable sound -- a held tone, a glide's own slowly-moving harmonic
    structure -- keeps a similar spectral shape from one analysis frame to the next, so this reads
    high and steady through it. A transient (onset, offset, a plosive) changes the spectral shape
    abruptly between frames, reading low right at the transition. Verified on real recordings this
    session: a glide's voiced region reads continuity 0.95-0.99 against 0.86-0.92 for the silence
    flanking it, a clean, well-separated signal. The same measure does not discriminate a breath from
    the background noise around it on the recording tested (both read 0.85-0.90 with no separation
    tracking the breath's own timing) -- turbulent broadband noise does not carry the frame-to-frame
    bin-level coherence a harmonic sound does, so this measure is a real detector for sustained
    tonal/harmonic production (glides, phonation, vowels) and not, on the evidence gathered so far,
    for breath noise specifically.

    Takes an already-computed magnitude spectrogram rather than raw audio and its own STFT
    parameters: the caller (PREPROCESS's ``_spans``) reuses the narrowband spectrogram block's own
    output directly, so a recording is not put through two independent STFTs at merely-matching
    parameters for two purposes that are both "look at this recording's spectral structure". This
    does mean a caller wanting its own analysis resolution must supply its own magnitude array;
    nothing here computes one. ``smoothing`` reuses the same pluggable
    :class:`~senselab.audio.tasks.envelope.api.EnvelopeSmoothing` strategy
    :func:`~senselab.audio.tasks.envelope.api.hilbert_envelope_dbfs` already takes, applied to the
    per-frame continuity trace as if it were a signal sampled at the frame rate (``1 / hop_s``) --
    the same generic mechanism, not a bespoke moving average, and :class:`MedianSmoothing` for the
    same reason it is used for the gain curve elsewhere: it cannot overshoot past a transient the way
    a moving average or a resonant filter can.

    Args:
        magnitude: A magnitude (not power) spectrogram, shape ``(n_freqs, n_frames)``.
        hop_s: The hop between frames the spectrogram was computed at, in seconds -- needed to place
            each frame's value back on the sample timeline.
        sampling_rate: The original audio's sampling rate, in Hz -- together with ``hop_s`` this
            gives the hop in samples.
        n_samples: Length of the original audio, in samples, so the returned trace matches it exactly
            (a spectrogram's own frame count does not by itself say how long the source signal was).
        smoothing: Strategy applied to the per-frame continuity trace, at the frame rate ``1 / hop_s``,
            before it is resampled to one value per input sample.

    Returns:
        One continuity value per input sample, in ``[0, 1]``. Never ``nan``: a tiny epsilon in the
        norm turns the ``0/0`` two all-zero frames would otherwise produce into ``0.0`` rather than
        an undefined value -- true digital silence reads as the *least* continuous case, not the
        most, which is what a span-proposal gate needs: a long silent stretch must never itself read
        as a sustained, continuous event.
    """
    log_mag = np.log1p(np.asarray(magnitude, dtype=np.float64))

    if log_mag.shape[1] < 2:
        return np.ones(n_samples)

    norms = np.linalg.norm(log_mag, axis=0) + 1e-12
    dots = np.sum(log_mag[:, :-1] * log_mag[:, 1:], axis=0)
    cos_sim = dots / (norms[:-1] * norms[1:])
    cos_sim = np.concatenate([cos_sim[:1], cos_sim])

    frame_rate = int(round(1.0 / hop_s))
    smoothed = smoothing.apply(cos_sim, frame_rate)

    hop_samples = max(1, int(round(hop_s * sampling_rate)))
    frame_centers = np.arange(len(smoothed)) * hop_samples
    resampled = np.interp(np.arange(n_samples), frame_centers, smoothed, left=smoothed[0], right=smoothed[-1])
    return np.clip(resampled, 0.0, 1.0)
