"""Reading a scene classifier's per-window label scores.

Windows carry ``label_scores``: a list of single-entry dicts, ``[{label: score}, ...]``, ordered by
descending score. The previous shape was two parallel arrays, ``labels`` and ``scores``, which is
one ``[:top_k]`` slice away from disagreeing — nothing in the shape requires them to stay the same
length or the same order, and a consumer that zips them cannot tell when they have drifted. Pairing
each score with its own label makes that failure unrepresentable.

Order is preserved rather than normalised: these arrive ranked, and the rank is information a
consumer uses (``top_label``, label-mass thresholds).
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Mapping, Optional

__all__ = ["label_scores", "top_label"]


def label_scores(window: Mapping[str, Any]) -> list[dict[str, float]]:
    """The window's ``[{label: score}, ...]`` pairs, in rank order.

    Args:
        window: A classifier window.

    Returns:
        One single-entry dict per label. Empty when the window carries no classification — which is
        not the same as a window classified with low scores, and callers that need to tell them
        apart get to.

        Entries that are not single-key dicts are skipped: a multi-key or empty dict is not a
        label/score pair, and guessing which key was meant would be worse than dropping it.
    """
    raw = window.get("label_scores") or []
    out: list[dict[str, float]] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            continue
        ((label, score),) = entry.items()
        # ``numbers.Real`` rather than ``(int, float)``: parquet round-trips hand back numpy
        # scalars, which are Real but not float, and an isinstance check on the concrete types
        # silently drops every score that has been through storage. ``bool`` is excluded because
        # it is an int in Python and a classifier score is not a flag.
        if isinstance(score, Real) and not isinstance(score, bool):
            out.append({str(label): float(score)})
    return out


def top_label(window: Mapping[str, Any]) -> Optional[tuple[str, float]]:
    """The highest-scoring ``(label, score)``, or ``None`` when nothing was classified.

    Asked once here rather than re-derived at each call site, where "the first element" and "the
    max" are the same thing only while the input stays sorted.

    Args:
        window: A classifier window.

    Returns:
        The top pair, or ``None``.
    """
    pairs = label_scores(window)
    if not pairs:
        return None
    best = max(pairs, key=lambda d: next(iter(d.values())))
    label, score = next(iter(best.items()))
    return label, score
