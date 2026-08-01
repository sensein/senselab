"""L2 joint estimation — signals that exist only by combining others.

Each function here answers a question no single tool was asked. They are L2 by construction: the
inputs are L1 measurements, and the combining rule is a modelling choice that belongs where it can
be seen and changed.

**J1 — how many speakers are simultaneously active** (:func:`overlap_count_posterior`).

Available now, while J4 (per-speaker presence) still needs rounds, and the reason is worth stating
because it decides what else can be built on the activation channels. `segmentation-3.0` reports
one activation per speaker, but the channel ordering is arbitrary *within a window*: channel 1 in
one window and channel 1 in the next are not the same person. So any quantity that depends on which
channel is whom is ill-defined until the speaker↔channel assignment is resolved, which is the joint
space D-7 hands to L2 rounds. A **count** of active channels is invariant to that permutation, so it
is well-defined immediately — and it is precisely the signal the old noisy-or collapse destroyed,
since `1 − Π(1 − p_k)` answers "is anyone speaking" and discards how many.

**J2 — where the voice changes** (:func:`speaker_change_series`). Compares each embedding window
against the one a whole window-width later, so the two sides are disjoint spans meeting at a
boundary. Adjacent windows at the 50 ms hop share 97.5% of their audio and would measure phonetic
drift rather than speaker identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.statistics import entropy_uncertainty

if TYPE_CHECKING:
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

__all__ = ["overlap_count_posterior", "count_distribution", "speaker_change_series"]


def count_distribution(probs: np.ndarray) -> np.ndarray:
    """Distribution over "how many of these succeeded", for independent Bernoulli trials.

    The Poisson-binomial pmf, computed by the standard convolution: start certain that zero have
    succeeded, then fold in one channel at a time.

    Independence is an assumption, and it is the right one to be explicit about. In per-speaker
    mode the columns are *marginal* speaker probabilities, so reading them as independent is the
    model's own framing; but two channels tracking the same speaker through a permutation flip are
    obviously not independent, and this will then overstate the chance of overlap. Stating it here
    is what lets a later stage measure that error rather than inherit it silently.

    Args:
        probs: One probability per channel, in ``[0, 1]``.

    Returns:
        Array of length ``len(probs) + 1`` where entry ``j`` is ``P(exactly j active)``.
    """
    pmf = np.zeros(len(probs) + 1, dtype=np.float64)
    pmf[0] = 1.0
    for k, p in enumerate(probs):
        p = float(min(1.0, max(0.0, p)))
        # Walk downward so each update reads the previous iteration's values.
        pmf[1 : k + 2] = pmf[1 : k + 2] * (1.0 - p) + pmf[0 : k + 1] * p
        pmf[0] *= 1.0 - p
    return pmf


def overlap_count_posterior(
    posterior: "FramePosterior",
    start_s: float,
    end_s: float,
) -> dict[str, Any] | None:
    """J1: a distribution over the number of speakers active in ``[start_s, end_s)``.

    Built **per frame and then pooled**, never from the bucket's per-channel means. Two speakers
    taking turns within a bucket average to 0.5 on each channel, which as a per-bucket calculation
    would report a 25% chance of overlap that never occurred. Overlap is an instantaneous fact, so
    it has to be evaluated at frame resolution and only then reduced.

    Args:
        posterior: Frame posteriors with per-speaker channels intact (D-5).
        start_s: Bucket start, seconds.
        end_s: Bucket end, seconds.

    Returns:
        ``{"counts", "expected_count", "p_overlap", "uncertainty", "n_frames", "n_channels"}``, or
        ``None`` when the question cannot be answered from this input — a single collapsed speech
        probability has already discarded the count and must not be made to guess one, and a bucket
        containing no frames has nothing to count.

        ``counts`` maps speaker count → probability. ``uncertainty`` is the normalised Shannon
        entropy of that distribution per ``statistics.entropy_uncertainty``, so it is comparable
        with the other axes; ``p_overlap`` is the mass above one speaker.
    """
    if posterior is None or posterior.channel_format == "single":
        return None
    data = np.asarray(posterior.activations, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    window = posterior.frame_slice(start_s, end_s)
    if window is None:
        return None
    lo, hi = window
    frames = data[lo:hi]
    if frames.size == 0:
        return None

    # One count distribution per frame, averaged over the bucket. Averaging distributions (rather
    # than averaging the channel probabilities first) is what preserves within-bucket timing.
    pooled = np.mean([count_distribution(frame) for frame in frames], axis=0)
    total = float(pooled.sum())
    if total <= 0:
        return None
    pooled = pooled / total

    counts = {int(k): float(v) for k, v in enumerate(pooled)}
    return {
        "counts": counts,
        "expected_count": float(np.dot(np.arange(len(pooled)), pooled)),
        "p_overlap": float(pooled[2:].sum()) if len(pooled) > 2 else 0.0,
        "uncertainty": entropy_uncertainty({str(k): v for k, v in counts.items()}),
        "n_frames": int(hi - lo),
        "n_channels": int(data.shape[1]),
    }


def speaker_change_series(
    entries: Sequence[Any],
    *,
    same_speaker_floor: float = 0.30,
    diff_speaker_floor: float = 0.70,
) -> dict[str, Any] | None:
    """J2: where the voice changes, from windowed speaker embeddings.

    Compares each window against the one a **whole window-width later**, not the adjacent one. At
    the 50 ms hop D-2 chose, two adjacent 2 s windows share 97.5% of their audio, so their distance
    is dominated by the 2.5% that is new — phonetic content, not speaker identity. Lagging by the
    window width makes the two sides disjoint spans meeting at a boundary, which is the comparison
    a change point actually is. The fine hop then buys *localisation* of that boundary, roughly
    tenfold, which is exactly what D-2 said it buys and does not: it does not buy independent
    samples, and treating neighbouring scores as independent evidence would overcount badly.

    The distance is read through the calibration band the speaker axis already uses rather than a
    new anchor — a raw cosine of 0.2 is not evidence of anything, because same-speaker embeddings
    sit in a 0.1–0.3 noise floor from phonetic variation alone.

    Args:
        entries: ``WindowEmbedding`` list for one pass, ascending in time.
        same_speaker_floor: Distance at or below which the two spans are confidently one speaker.
        diff_speaker_floor: Distance at or above which they are confidently different.

    Returns:
        ``{"times", "distance", "p_change", "uncertainty", "lag_steps", "window_s", "hop_s"}``, all
        arrays aligned on boundary times, or ``None`` when the pass has fewer windows than one lag —
        with nothing disjoint to compare there is no claim to make.

        ``uncertainty`` is the binary entropy of ``{change, no change}``, so a confident change and
        a confident continuation are both certain and the doubt sits where the calibration band
        cannot resolve the distance.
    """
    items = list(entries or [])
    if len(items) < 2:
        return None
    starts = np.asarray([float(w.start_s) for w in items], dtype=np.float64)
    window_s = float(items[0].end_s) - float(items[0].start_s)
    hop_s = float(starts[1] - starts[0])
    if window_s <= 0 or hop_s <= 0:
        return None
    lag = max(1, int(round(window_s / hop_s)))
    if len(items) <= lag:
        return None

    from senselab.audio.workflows.audio_analysis.embeddings import calibrate_cosine_uncertainty

    times, distances, p_change = [], [], []
    for i in range(len(items) - lag):
        a = np.asarray(items[i].vector, dtype=np.float64).ravel()
        b = np.asarray(items[i + lag].vector, dtype=np.float64).ravel()
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 0 or nb <= 0:
            continue
        d = 1.0 - float(np.dot(a, b) / (na * nb))
        # ``direction="same"`` returns how far the audio contradicts a same-speaker claim, which is
        # the probability that a change occurred here.
        p = calibrate_cosine_uncertainty(
            d,
            same_speaker_floor=same_speaker_floor,
            diff_speaker_floor=diff_speaker_floor,
            direction="same",
        )
        times.append(float(items[i].end_s))  # the boundary the two disjoint spans meet at
        distances.append(d)
        p_change.append(float(min(1.0, max(0.0, p))))

    if not times:
        return None
    probs = np.asarray(p_change, dtype=np.float64)
    unc = np.asarray(
        [entropy_uncertainty({"change": float(p), "same": float(1.0 - p)}) or 0.0 for p in probs],
        dtype=np.float64,
    )
    return {
        "times": np.asarray(times, dtype=np.float64),
        "distance": np.asarray(distances, dtype=np.float64),
        "p_change": probs,
        "uncertainty": unc,
        "lag_steps": lag,
        "window_s": window_s,
        "hop_s": hop_s,
    }
