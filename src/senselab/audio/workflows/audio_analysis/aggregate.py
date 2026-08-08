"""Per-voter presence mapping — one bucket's votes read as a probability of voice.

**No per-axis aggregator lives here any more.** ``aggregate_speech_presence``, ``aggregate_speaker``
and ``aggregate_asr`` were three complete, tested folds with no production caller: the run's single
fold is ``fuse.fuse_axis``, which reads each voter through ``fuse.per_signal_uncertainty`` and
weights it by measured stability and support. Two implementations of "collapse a bucket to one
number", only one of which ran, is a second answer nobody was comparing against the first — and the
uncalled one still carried its own weighting rules, its own calibration temperature and its own
sub-signal list, which is exactly how a reader comes to believe the wrong thing about a number.
``mean_token_entropy`` and ``_axis_temperature`` went with ``aggregate_asr``, their only caller.

What survives is the part with real callers, and it is not a fold across an axis: the per-voter
``(speaks, native_confidence) → p_voice`` mapping, used by the belief store's ingest and by S1's
stream election.
"""

from __future__ import annotations

from typing import Any, Mapping

# Surface-level differences (case + punctuation + repeated whitespace) are stripped before any
# transcript comparison so a measure reflects *semantic* disagreement rather than surface noise. The
# canonical normalizer lives at the task layer (architecture-review T049) so task- and
# workflow-level WER share one definition; re-exported under the historical name for its importers.
from senselab.audio.tasks.speech_to_text_evaluation.utils import (
    normalize_transcript_for_wer as _normalize_transcript_for_wer,
)

__all__ = [
    "_normalize_transcript_for_wer",
    "per_source_voice",
    "speech_presence_p_voice",
]


# ── speech_presence ──────────────────────────────────────────────────────────


def _evidence_factor(weights: Mapping[str, float] | None, source: str) -> float:
    """Per-source evidence weight from ``weights``, or 1.0 when this source was never measured.

    Absent means *unmeasured*, so it must not act as a discount: mapping a missing entry to
    anything below 1.0 would let a factor nobody gathered decide the fold.
    """
    if not weights:
        return 1.0
    raw = weights.get(source)
    if raw is None:
        return 1.0
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0


def per_source_voice(
    votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None
) -> dict[str, tuple[float, float]]:
    """``{source → (p_voice, weight)}`` for one bucket, before any fold across voters.

    Exposed because the fold across *passes* has to happen at this level. A caller that folds
    ``_weighted_p_voice`` per pass and averages the results weights each pass by how many voters it
    happened to contain; averaging each voter's own reading first — the rule :func:`fuse.fuse_axis`
    applies to every other signal — does not. Same per-voter mapping either way, defined once.

    Voters with no ``speaks`` field, or a non-positive payload weight, are absent from the result:
    policy declaring a voter inapplicable on this grid is not a reading of zero.
    """
    out: dict[str, tuple[float, float]] = {}
    for source, v in votes.items():
        if not isinstance(v, dict) or "speaks" not in v:
            continue
        speak_val = v.get("speaks")
        if speak_val is None:
            continue
        try:
            weight = float(v.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        # The payload weight may legitimately be zero — that is policy declaring a voter
        # inapplicable on this grid. Attenuation cannot reach zero (its floor is > 0), so this
        # guard never erases a corroboration-weighted vote.
        if weight <= 0:
            continue
        weight *= _evidence_factor(weights, str(source))
        raw_nc = v.get("native_confidence")
        nc: float | None
        if raw_nc is None:
            nc = None
        else:
            try:
                nc = max(0.0, min(1.0, float(raw_nc)))
            except (TypeError, ValueError):
                nc = None
        if v.get("hallucinated"):
            p_voter = 0.1
        elif nc is None:
            p_voter = 1.0 if speak_val else 0.0
        else:
            p_voter = nc if speak_val else (1.0 - nc)
        out[str(source)] = (p_voter, weight)
    return out


def _weighted_p_voice(votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None) -> float | None:
    """Weighted mean per-voter probability of voice for one bucket, or ``None``.

    Each voter maps ``(speaks, native_confidence)`` to a per-voter voice
    probability, then contributes with its optional ``weight`` (default 1.0):

    - ``native_confidence`` ``c`` with ``speaks=True`` → ``p = c``;
      with ``speaks=False`` → ``p = 1 - c``.
    - No ``native_confidence`` → ``p = 1.0`` if ``speaks`` else ``0.0``.
    - ``hallucinated`` → ``p = 0.1`` (vote against voice).

    ``weight`` lets a caller demote coarse voters (whole-window scene tags,
    per-segment no-speech probability, sentence-level ASR) on fine reporting
    grids without dropping them (FR-014). When every weight is 1.0 (the
    default) this is the plain mean, so existing outputs are unchanged.

    ``weights`` is the *second*, independent factor: how far this source's claim was corroborated
    by evidence measured about it (``belief.VoteStore.evidence_weights``). The two multiply and
    stay separately recoverable — the payload keeps what the link layer decided about the voter's
    coarseness, the map keeps what a later round measured about its corroboration. A source absent
    from ``weights`` was not measured and keeps its payload weight untouched.
    """
    num = 0.0
    den = 0.0
    for p_voter, weight in per_source_voice(votes, weights=weights).values():
        num += weight * p_voter
        den += weight
    if den <= 0:
        return None
    return num / den


def speech_presence_p_voice(
    votes: dict[str, dict[str, Any]], *, weights: Mapping[str, float] | None = None
) -> float | None:
    """Return the calibrated probability of voice ``p_voice`` for one bucket.

    Same per-voter math as :func:`_weighted_p_voice` but returns the raw
    probability rather than the symmetric uncertainty. Used both as the
    speech_presence-axis ``speech_presence_confidence`` column and to MASK speaker /
    asr buckets where we are confident there is no speech.
    """
    return _weighted_p_voice(votes, weights=weights)
