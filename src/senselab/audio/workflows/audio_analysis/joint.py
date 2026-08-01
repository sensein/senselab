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
boundary. Adjacent windows at the 50 ms hop share 97.5% of their audio, so the change is present
but low-amplitude and smeared across the window width rather than appearing as a step.

**J7 — which reading the acoustics support** (:func:`phoneme_transcript_agreement`). PPG posteriors
reach the audio without passing through a language model, so they can adjudicate between two ASR
readings of the same span without echoing a third transcriber's opinion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.statistics import entropy_uncertainty

if TYPE_CHECKING:
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

__all__ = ["overlap_count_posterior", "count_distribution", "speaker_change_series", "phoneme_transcript_agreement"]


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
    same_speaker_floor: float,
    diff_speaker_floor: float,
) -> dict[str, Any] | None:
    """J2: where the voice changes, from windowed speaker embeddings.

    Compares each window against the one a **whole window-width later**, not the adjacent one.

    The reason is contrast, not a difference in what is being measured. Two adjacent 2 s windows at
    a 50 ms hop share 97.5% of their audio, so swapping 50 ms of content moves the embedding only
    slightly — the distance is small everywhere, and a speaker change is spread across the window
    width as one voice is gradually exchanged for the other rather than appearing as a step. The
    change is still *there*; it is just low-amplitude and smeared, so a boundary is hard to place
    and easy to lose in noise. Lagging by the window width makes the two sides disjoint spans
    meeting at a boundary, which is the comparison a change point actually is and which puts the
    full between-speaker difference into a single score.

    The fine hop still earns its keep: it *localises* that boundary, roughly tenfold. What it does
    not buy is independent samples, so neighbouring scores are near-duplicates and must not be
    counted as separate evidence — the hop is reported alongside so a consumer can see that.

    The distance is read through the calibration band the speaker axis already uses rather than a
    new anchor — a raw cosine of 0.2 is not evidence of anything, because same-speaker embeddings
    sit in a 0.1–0.3 noise floor from phonetic variation alone.

    Args:
        entries: ``WindowEmbedding`` list for one pass, ascending in time.
        same_speaker_floor: Distance at or below which the two spans are confidently one speaker.
        diff_speaker_floor: Distance at or above which they are confidently different. Both are
            **required**: a pass whose embeddings were measured not to separate speakers has no
            usable band, and defaulting to library anchors there would let this signal vote
            confidently on exactly the evidence that was found wanting (FR-007).

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


def phoneme_transcript_agreement(
    slots: Sequence[Any],
    *,
    ppg_per_frame: Sequence[str],
    ppg_frame_hop: float,
) -> list[dict[str, Any]] | None:
    """J7: score each reading of a transcript slot against the phoneme evidence.

    PPG posteriors are an **independent witness**. They come from the audio without passing through
    a language model, so where two ASR models read the same span differently, the phoneme frames
    can favour one without merely echoing another transcriber's opinion. That is what this adds
    over the per-window PER the ASR axis already computes: that measure asks whether *a* model's
    transcript matches the audio, this one asks which of the readings actually on the table does.

    Each distinct reading in a slot is converted to phonemes and compared against the PPG's argmax
    run sequence over the slot's span by phoneme error rate. The candidate distribution is
    ``max(0, 1 − PER)`` normalised — a linear link with no free temperature, so nothing here is a
    tuned parameter: equally supported candidates give a uniform distribution and maximal doubt,
    and one candidate matching exactly while the others do not puts the mass on it.

    Args:
        slots: :class:`~.harmonize.TranscriptSlot` list from H3.
        ppg_per_frame: Argmax phoneme label per PPG frame.
        ppg_frame_hop: Seconds per PPG frame.

    Returns:
        One dict per slot that had both candidates and phoneme frames, carrying ``per`` per
        candidate, the ``acoustic_choice`` (``None`` when nothing separates them), whether it
        ``agrees_with_consensus``, and the selection ``uncertainty``. ``None`` when there is no PPG
        at all — an absent witness is not agreement, and must not be recorded as a verdict.
    """
    if not ppg_per_frame or ppg_frame_hop <= 0:
        return None

    from senselab.audio.workflows.audio_analysis.harvesters import (
        _levenshtein,
        arpabet_to_ppg_inventory,
        g2p_phonemes,
        ppg_argmax_runs_in_window,
    )

    out: list[dict[str, Any]] = []
    for slot in slots or []:
        runs = ppg_argmax_runs_in_window(
            list(ppg_per_frame), float(ppg_frame_hop), float(slot.start_s), float(slot.end_s)
        )
        observed = [p for _, _, p in runs if p != "<silent>"]
        candidates = sorted({str(w) for w in slot.words.values() if w})
        if not observed or not candidates:
            continue

        per: dict[str, float] = {}
        for candidate in candidates:
            expected = [arpabet_to_ppg_inventory(p) for p in g2p_phonemes(candidate)]
            expected = [p for p in expected if p]
            if not expected:
                continue
            per[candidate] = float(_levenshtein(expected, observed)) / max(1, len(expected))
        if not per:
            continue

        support = {c: max(0.0, 1.0 - v) for c, v in per.items()}
        total = sum(support.values())
        if total <= 0:
            # Every reading contradicted by the audio. That is a real finding, not a missing one:
            # the candidates are indistinguishable *because all are unsupported*, so the choice
            # carries full doubt rather than being dropped.
            distribution = {c: 1.0 / len(support) for c in support}
        else:
            distribution = {c: v / total for c, v in support.items()}

        best = max(distribution.values())
        winners = [c for c, v in distribution.items() if v == best]
        choice = winners[0] if len(winners) == 1 else None
        out.append(
            {
                "start_s": float(slot.start_s),
                "end_s": float(slot.end_s),
                "per": per,
                "candidate_support": distribution,
                "acoustic_choice": choice,
                "consensus": slot.consensus,
                "agrees_with_consensus": (
                    None if choice is None or slot.consensus is None else choice == slot.consensus
                ),
                "uncertainty": entropy_uncertainty(distribution) if len(distribution) > 1 else 0.0,
                "n_phoneme_runs": len(observed),
            }
        )
    return out
