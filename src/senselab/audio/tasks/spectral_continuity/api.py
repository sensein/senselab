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
    abruptly between frames, reading low right at the transition.

    Takes an already-computed magnitude spectrogram rather than raw audio and its own STFT
    parameters, so a recording is not put through two independent STFTs at merely-matching
    parameters. PREPROCESS's ``_spans`` passes the **narrowband** block's magnitude. A caller wanting
    its own analysis resolution must supply its own magnitude array; nothing here computes one.
    ``smoothing`` is the same pluggable
    :class:`~senselab.audio.tasks.envelope.api.EnvelopeSmoothing` strategy
    :func:`~senselab.audio.tasks.envelope.api.envelope_dbfs` takes, applied to the per-frame
    continuity trace as if it were a signal sampled at the frame rate (``1 / hop_s``).

    What this measure does and does not detect, which spectrogram to feed it, and the jitter it
    carries at a transition are measured in
    ``specs/20260817-triage-workflow-dag/benchmarks/preprocess-params.md``.

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
