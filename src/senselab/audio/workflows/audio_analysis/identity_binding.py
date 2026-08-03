"""Binding fused speaker ids to each tool's own labels, from spans (D-19's C2, replacing J4).

Three id namespaces stay distinct because all three once rendered as ``S0``: a model's own labels
(``SPEAKER_00``, ``spk0``), the pass-wide cluster that harmonises labels across diarizers (``C0``),
and the fused speaker id in ``final/speakers.json`` (``S0``). This module binds the third to the
first, and the binding is **evidence** rather than a preprocessing step — how well-determined it is
*is* part of the speaker uncertainty, which is what makes its stability a convergence criterion (C2).

**What changes from J4.** It bound ``S_k`` to ``segmentation-3.0``'s activation channels, which are
permutation-arbitrary within each inference: they carried timing but could not name anyone, so the
binding was the only thing supplying a name. With diarizers emitting spans there are no channels —
each tool has its own labels, carrying timing *and* its own identity — and the binding gains something
the channel version could not have: **a speaker bound by one diarizer and unbound by another is a
measurable disagreement.** That is the signature an off-target speaker leaves.

**Two properties carried over unchanged, because both are refusals to decide.** A speaker with no
overlapping label is left *unbound* rather than given the least-bad one; a tool label no speaker
claimed is *reported* rather than dropped, because that is the shape a missed speaker takes.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.shapes import Spans

__all__ = ["bind_labels", "per_speaker_presence"]

SpeakerSpans = Mapping[str, Sequence[tuple[float, float]]]


def bind_labels(speaker_spans: SpeakerSpans, spans: Spans) -> Optional[dict[str, Any]]:
    """Bind each fused speaker id to one of this tool's labels, by temporal agreement.

    Args:
        speaker_spans: ``{fused speaker id → [(start_s, end_s), …]}``.
        spans: One diarizer's span set, in its own label namespace.

    Returns:
        ``{"assignment", "margin", "uncertainty", "unassigned_speakers", "unassigned_labels"}``, or
        ``None`` when the question cannot be asked — no speakers means nothing to bind, and a tool
        that produced no spans has not *failed* to bind, it had nothing to bind to.

        ``margin`` is the bound label's overlap lead over the runner-up, normalised; ``uncertainty``
        is one minus the mean margin, so a tie between two equally-overlapping labels reads as doubt
        rather than as a decision the matcher happened to make.

    Matched **globally** (Hungarian) rather than greedily: a greedy pass takes the best pair first and
    can strand a better overall assignment, which for identity means confidently binding the wrong
    name.
    """
    from scipy.optimize import linear_sum_assignment

    speakers = sorted(s for s, sp in (speaker_spans or {}).items() if sp)
    labels = sorted({span.label for span in spans.spans})
    if not speakers or not labels:
        return None

    by_label = {label: [(s.start, s.end) for s in spans.spans if s.label == label] for label in labels}
    # Agreement in overlap-seconds: a label covering a whole turn agrees more than one clipping its
    # edge, so duration is the weight rather than a boolean "did they touch".
    score = np.array(
        [[_overlap(speaker_spans[speaker], by_label[label]) for label in labels] for speaker in speakers],
        dtype=np.float64,
    )

    rows, cols = linear_sum_assignment(-score)
    assignment: dict[str, str] = {}
    margin: dict[str, float] = {}
    for row, col in zip(rows, cols):
        if score[row, col] <= 0:
            continue  # no overlap at all: unbound, not least-bad
        assignment[speakers[row]] = labels[col]
        others = np.delete(score[row], col)
        runner_up = float(others.max()) if others.size else 0.0
        best = float(score[row, col])
        margin[speakers[row]] = (best - runner_up) / best if best > 0 else 0.0

    return {
        "assignment": assignment,
        "margin": margin,
        # One minus the mean margin over *bound* speakers. Unbound speakers are reported separately
        # rather than folded in as maximal doubt: not binding is a different statement from binding
        # ambiguously, and averaging them together would blur the two.
        "uncertainty": 1.0 - (sum(margin.values()) / len(margin)) if margin else 1.0,
        "unassigned_speakers": tuple(s for s in speakers if s not in assignment),
        "unassigned_labels": tuple(lab for lab in labels if lab not in set(assignment.values())),
    }


def _overlap(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    """Total overlap in seconds between two span sets."""
    return sum(max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo)) for a_lo, a_hi in a for b_lo, b_hi in b)


def per_speaker_presence(
    speaker_spans: SpeakerSpans,
    *,
    spans_by_tool: Mapping[str, Spans],
) -> dict[str, dict[str, Any]]:
    """Per fused speaker: where they are, and which diarizers agree they exist.

    Args:
        speaker_spans: ``{fused speaker id → [(start_s, end_s), …]}``.
        spans_by_tool: ``{tool → its span set}``, each carrying its own speaker capacity.

    Returns:
        ``{speaker id → {"spans", "bound_in", "unbound_in", "censored_in", "binding_agreement"}}``.

        ``binding_agreement`` is the fraction of **eligible** tools that bound this speaker, and
        eligibility is where D-19's censoring enters: a tool already at its capacity had no further
        label to offer, so its silence about an additional speaker is not dissent. Counting it as
        dissent would make every low-capacity tool an argument against every additional speaker —
        the same bias the count posterior corrects, one level up.

        A speaker no tool bound reports ``binding_agreement`` of ``0.0`` and empty spans rather than
        vanishing from the mapping: a fused id nothing supports is a finding, not an absence.
    """
    bindings = {tool: bind_labels(speaker_spans, spans) for tool, spans in spans_by_tool.items()}
    out: dict[str, dict[str, Any]] = {}
    for speaker in sorted(s for s, sp in (speaker_spans or {}).items() if sp):
        bound: list[str] = []
        unbound: list[str] = []
        censored: list[str] = []
        spans_for_speaker: list[tuple[float, float]] = []
        for tool, binding in sorted(bindings.items()):
            if binding is None:
                continue
            label = binding["assignment"].get(speaker)
            if label is not None:
                bound.append(tool)
                spans_for_speaker.extend((s.start, s.end) for s in spans_by_tool[tool].spans if s.label == label)
            elif spans_by_tool[tool].is_censored_at(len(set(binding["assignment"].values()))):
                censored.append(tool)
            else:
                unbound.append(tool)
        eligible = len(bound) + len(unbound)
        out[speaker] = {
            "spans": tuple(sorted(spans_for_speaker)),
            "bound_in": tuple(bound),
            "unbound_in": tuple(unbound),
            "censored_in": tuple(censored),
            "binding_agreement": (len(bound) / eligible) if eligible else 0.0,
        }
    return out
