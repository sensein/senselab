"""L2 joint estimation — signals that exist only by combining others.

Each function here answers a question no single tool was asked. They are L2 by construction: the
inputs are L1 measurements, and the combining rule is a modelling choice that belongs where it can
be seen and changed.

**J1 and J4 have moved.** The count posterior is now cross-diarizer spread
(:mod:`.occupancy`) and the speaker binding is over each tool's own labels
(:mod:`.identity_binding`), both from spans. What lived here was built on
``segmentation-3.0``'s per-speaker channels, whose independence the Poisson-binomial assumed
and which a powerset conversion does not have.

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

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.statistics import entropy_uncertainty

__all__ = [
    "speaker_change_series",
]






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




def speaker_spans_from_votes(
    speaker_votes: Sequence[Mapping[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
    """Harmonised speaker spans from the speaker axis's per-bucket cluster ids.

    The spans J4 binds must live in the **harmonised** space: a raw ``SPEAKER_00`` means different
    people to different diarizers, and the cluster id is exactly what H2 exists to produce. A bucket
    counts for a cluster when *any* diar model placed it there — the same union rule coverage uses,
    so two models agreeing cannot inflate a span.

    Contiguous buckets are merged, so a speaker that spoke across ten buckets is one span rather
    than ten, which is what the temporal-agreement match wants.
    """
    from senselab.audio.workflows.audio_analysis.speaker import SILENT_CLUSTER_ID

    per_cluster: dict[str, list[tuple[float, float]]] = {}
    for bucket in speaker_votes or []:
        start, end = _finite_pair(bucket.get("start"), bucket.get("end"))
        if start is None or end is None:
            continue
        seen: set[str] = set()
        for entry in (bucket.get("votes") or {}).values():
            if not isinstance(entry, Mapping):
                continue
            for cluster in (entry.get("cluster_ids") or {}).values():
                if cluster and str(cluster) != SILENT_CLUSTER_ID:
                    seen.add(str(cluster))
        for cluster in seen:
            per_cluster.setdefault(cluster, []).append((start, end))

    merged: dict[str, list[tuple[float, float]]] = {}
    for cluster, spans in sorted(per_cluster.items()):
        spans.sort()
        out: list[tuple[float, float]] = []
        for lo, hi in spans:
            if out and lo <= out[-1][1] + 1e-9:
                out[-1] = (out[-1][0], max(out[-1][1], hi))
            else:
                out.append((lo, hi))
        merged[cluster] = out
    return merged


def _finite_pair(a: Any, b: Any) -> tuple[float | None, float | None]:  # noqa: ANN401
    """Coerce a bucket's start/end to floats, or ``(None, None)`` if either is unusable."""
    try:
        return float(a), float(b)
    except (TypeError, ValueError):
        return None, None
