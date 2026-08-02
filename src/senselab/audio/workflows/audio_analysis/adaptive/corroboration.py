"""Independent presence evidence, derived from the run rather than configured.

One measurement — how far independent evidence supports a speech claim in a span — serving both
consumers that used to erase evidence instead: the belief store's uncorroborated-claim attenuation
and the word-stream ensemble's per-word weight. Sharing the derivation is the point: two
definitions of "corroborated" would drift, and the one that drifted would be the one deciding what
reaches the transcript.

**The pool must exclude claimants, and that is a correctness condition, not a refinement.** The
obvious quantity to reach for — the belief row's ``p_voice`` — is a weighted mean over *all*
presence voters including the ASR models themselves, and ``aggregate._weighted_p_voice`` maps a
voter carrying ``hallucinated: True`` to ``p = 0.1``. Measuring an ASR's claim against that number
is the model indicting itself, the exact failure ``adaptive.provenance.classify_resolution`` exists
to catch. ``support.evidence_signal_names`` excludes ASR and diarizer ids structurally, on the
ground that both infer presence from a decision that already presupposes a speaker.

**A signal that never reports absence makes the measure inert.** Corroboration only ever removes
weight, so it runs entirely on negative evidence; ``support.informative_evidence`` drops voters
that never say "no speech" (measured: ``acoustic_loudness`` median 0.897, ``ast`` 0.728 over 697
buckets — pooled with max they pin corroboration near 1.0). An empty pool is a legitimate outcome
and means the mechanism is inert on this run; it must be *reported*, never silently assumed away.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.support import (
    bucket_corroboration,
    evidence_signal_names,
    informative_evidence,
)

__all__ = [
    "independent_presence_pool",
    "corroboration_in_bucket",
    "corroboration_over_span",
    "make_corroboration_lookup",
    "apply_corroboration",
]


def independent_presence_pool(store: Any, stream: str) -> tuple[list[str], dict[str, str]]:  # noqa: ANN401
    """Presence voters usable as corroboration for a speech claim, derived from the run itself.

    Args:
        store: The vote store (duck-typed: ``buckets`` / ``active_votes``).
        stream: Pass label.

    Returns:
        ``(pool, rejected)`` — the admitted voter names, and every candidate that was dropped
        mapped to the reason. The rejection map is returned rather than discarded so an inert run
        is visible in the artifacts instead of looking like a run where nothing was doubtful.
    """
    buckets = [
        {"votes": store.active_votes(stream, "speech_presence", bk)} for bk in store.buckets(stream, "speech_presence")
    ]
    candidates = sorted(evidence_signal_names(buckets))
    kept = informative_evidence(buckets, candidates)
    rejected = {name: "never_reports_absence" for name in candidates if name not in kept}
    return sorted(kept), rejected


def corroboration_in_bucket(
    store: Any,  # noqa: ANN401
    stream: str,
    bucket: tuple[float, float],
    *,
    pool: Sequence[str],
) -> float | None:
    """Independent evidence for speech in one presence bucket, or ``None`` if unmeasured."""
    return bucket_corroboration(store.active_votes(stream, "speech_presence", bucket), evidence_signals=pool)


def corroboration_over_span(
    store: Any,  # noqa: ANN401
    stream: str,
    start: float,
    end: float,
    *,
    pool: Sequence[str],
) -> tuple[float | None, int, int]:
    """Independent evidence for speech anywhere in ``[start, end)``.

    Max over the overlapping buckets, for the same reason the pool is pooled with max and one step
    further: presence buckets (0.5 s) are coarser than the spans asked about (a word is ~0.2–0.4 s
    and straddles boundaries), and a coarse measurement must not confidently indict a finer one.

    Returns:
        ``(corroboration | None, n_buckets, n_measured)`` — the counts travel with the number so
        the coarseness of the underlying grid stays auditable per span.
    """
    overlapping = [bk for bk in store.buckets(stream, "speech_presence") if bk[0] < end and bk[1] > start]
    measured = [
        p
        for bk in overlapping
        if (p := bucket_corroboration(store.active_votes(stream, "speech_presence", bk), evidence_signals=pool))
        is not None
    ]
    return (max(measured) if measured else None), len(overlapping), len(measured)


def make_corroboration_lookup(store: Any, stream: str, *, pool: Sequence[str]) -> Any:  # noqa: ANN401 — callable
    """``(start, end) -> (p_independent | None, n_buckets, n_measured)`` over one stream."""

    def lookup(start: float, end: float) -> tuple[float | None, int, int]:
        return corroboration_over_span(store, stream, float(start), float(end), pool=pool)

    return lookup


def apply_corroboration(
    word_streams: dict[str, list[dict[str, Any]]],
    lookup: Any,  # noqa: ANN401 — callable
    *,
    exponent: float,
    min_corroboration: float,
    pool: Sequence[str],
    rejected: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Stamp each word with its measured corroboration weight; return ``(streams, provenance)``.

    ``max(floor, p ** exponent)`` is the same gate shape as ``influence.effective_weight``'s
    ``max(min_gate, (1 - u) ** exponent)`` — monotone, floored, and introducing no new anchor
    constant. ``p is None`` propagates as ``corroboration = None``, which the fusion task treats as
    1.0: a factor never measured never discounts.

    Every word is stamped, whether or not any intervention fired. That is the point: under the
    previous design a word survived or vanished according to whether an intervention had been
    *admitted within budget*, so budget accounting decided what reached the transcript.

    Args:
        word_streams: ``{model → [word]}``; mutated in place and returned.
        lookup: From :func:`make_corroboration_lookup`.
        exponent: Sharpness of the gate. 1.0 is the identity above the floor.
        min_corroboration: Floor on the stamped weight.
        pool: The evidence pool the measurement was taken against.
        rejected: Candidates dropped from the pool, mapped to the reason.

    Returns:
        ``(word_streams, provenance)`` where provenance is the document-level block written into
        ``final/transcript.json`` — including the pool and its rejections, so a run where the
        mechanism was inert says so instead of looking like a run where nothing was doubtful.
    """
    n_measured = 0
    n_unmeasured = 0
    for words in word_streams.values():
        for word in words:
            p, n_buckets, n_bucket_measured = lookup(word["start"], word["end"])
            if p is None:
                word["corroboration"] = None
                n_unmeasured += 1
            else:
                word["corroboration"] = max(float(min_corroboration), min(1.0, float(p) ** float(exponent)))
                n_measured += 1
            word["corroboration_evidence"] = {
                "p_independent": p,
                "n_buckets": n_buckets,
                "n_measured": n_bucket_measured,
            }
    provenance: dict[str, Any] = {
        "evidence_pool": list(pool),
        "evidence_pool_rejected": dict(rejected),
        "pool_derivation": "support.evidence_signal_names + support.informative_evidence",
        "exponent": float(exponent),
        "min_corroboration": float(min_corroboration),
        "n_words_measured": n_measured,
        "n_words_unmeasured": n_unmeasured,
    }
    return word_streams, provenance
