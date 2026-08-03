"""Binding fused speaker ids to each tool's own labels, from spans (D-19's C2).

Three id namespaces stay distinct because all three once rendered as `S0`: a model's own labels
(`SPEAKER_00`, `spk0`), the pass-wide cluster that harmonises labels across diarizers (`C0`), and the
fused speaker id in `final/speakers.json` (`S0`). This module binds the third to the first.

J4 bound `S_k` to `segmentation-3.0`'s **activation channels**, which are permutation-arbitrary within
each inference: they carried timing but could not name anyone. With diarizers emitting spans there are
no channels — each tool has its own labels, which carry both timing and (its own) identity. So the
binding is `S_k` ↔ tool label, per tool, and it gains something the channel version could not have:
**a speaker bound by one diarizer and unbound by another is a measurable disagreement**, and it is the
signature an off-target speaker leaves.

Two properties carried over from J4 because they were right, and both are about refusing to decide:
a speaker with no overlapping label is left **unbound** rather than given the least-bad one, and a
tool label no speaker claimed is **reported** rather than dropped — that is the shape a missed
speaker takes.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.identity_binding import bind_labels, per_speaker_presence
from senselab.audio.workflows.audio_analysis.shapes import Capacity, Span, Spans


def _spans(*items: tuple[float, float, str], capacity: Capacity = None) -> Spans:
    return Spans(spans=tuple(Span(start=s, end=e, label=lab) for s, e, lab in items), capacity=capacity)


def _binding(speakers: dict[str, list[tuple[float, float]]], spans: Spans) -> dict[str, Any]:
    result = bind_labels(speakers, spans)
    assert result is not None, "expected a binding from a non-empty speaker set"
    return result


# ── bind_labels ────────────────────────────────────────────────────────


def test_a_speaker_binds_to_the_label_it_overlaps() -> None:
    """The only evidence linking a fused id to a tool's label is temporal agreement."""
    result = _binding(
        {"S0": [(0.0, 1.0)], "S1": [(2.0, 3.0)]},
        _spans((0.0, 1.0, "spk_a"), (2.0, 3.0, "spk_b")),
    )
    assert result["assignment"] == {"S0": "spk_a", "S1": "spk_b"}


def test_the_binding_is_matched_globally_not_greedily() -> None:
    """A greedy pass takes the best pair first and can strand a better overall assignment.

    Here S0 overlaps both labels, but only S1 can take spk_b — so S0 must yield it.
    """
    result = _binding(
        {"S0": [(0.0, 3.0)], "S1": [(2.0, 3.0)]},
        _spans((0.0, 1.0, "spk_a"), (2.0, 3.0, "spk_b")),
    )
    assert result["assignment"] == {"S0": "spk_a", "S1": "spk_b"}


def test_a_speaker_with_no_overlap_is_left_unbound() -> None:
    """Nothing is thresholded into a decision: the least-bad label is still not evidence."""
    result = _binding({"S0": [(0.0, 1.0)], "S1": [(9.0, 10.0)]}, _spans((0.0, 1.0, "spk_a")))
    assert result["assignment"] == {"S0": "spk_a"}
    assert result["unassigned_speakers"] == ("S1",)


def test_a_label_no_speaker_claimed_is_reported_rather_than_dropped() -> None:
    """That is the shape a missed speaker takes, so it must survive into the output."""
    result = _binding({"S0": [(0.0, 1.0)]}, _spans((0.0, 1.0, "spk_a"), (5.0, 6.0, "spk_b")))
    assert result["unassigned_labels"] == ("spk_b",)


def test_a_tie_between_labels_reads_as_doubt_rather_than_a_decision() -> None:
    """``uncertainty`` is one minus the mean margin, so an arbitrary pick is visibly arbitrary."""
    tied = _binding({"S0": [(0.0, 1.0)]}, _spans((0.0, 1.0, "spk_a"), (0.0, 1.0, "spk_b")))
    clear = _binding({"S0": [(0.0, 1.0)]}, _spans((0.0, 1.0, "spk_a"), (5.0, 6.0, "spk_b")))
    assert tied["uncertainty"] > clear["uncertainty"]
    assert clear["uncertainty"] == pytest.approx(0.0)


def test_no_speakers_yields_none() -> None:
    """Nothing to bind is not an empty binding — it is a question that cannot be asked."""
    assert bind_labels({}, _spans((0.0, 1.0, "spk_a"))) is None


def test_no_labels_yields_none() -> None:
    """A diarizer that produced nothing cannot be bound to, and did not fail to bind."""
    assert bind_labels({"S0": [(0.0, 1.0)]}, _spans()) is None


def test_agreement_is_weighted_by_overlap_duration() -> None:
    """A label overlapping a whole turn agrees more than one clipping its edge."""
    result = _binding(
        {"S0": [(0.0, 10.0)]},
        _spans((0.0, 10.0, "spk_long"), (9.9, 10.0, "spk_touch")),
    )
    assert result["assignment"] == {"S0": "spk_long"}
    assert result["margin"]["S0"] > 0.9


# ── per_speaker_presence ───────────────────────────────────────────────


def test_presence_comes_from_the_bound_labels_spans() -> None:
    """The timing the fused speaker space never had, taken from the tool that supplied it."""
    result = per_speaker_presence(
        {"S0": [(0.0, 1.0)]},
        spans_by_tool={"community": _spans((0.0, 1.0, "spk_a"), capacity="unbounded")},
    )
    assert result["S0"]["bound_in"] == ("community",)
    assert result["S0"]["spans"] == ((0.0, 1.0),)


def test_a_speaker_bound_by_one_tool_and_not_another_is_recorded_as_such() -> None:
    """The measurable disagreement the channel version could not express.

    This is the signature an off-target speaker leaves: one diarizer draws a span for them and
    another does not. Reporting only the union would hide which tool saw them.
    """
    result = per_speaker_presence(
        {"S0": [(0.0, 1.0)], "S1": [(5.0, 6.0)]},
        spans_by_tool={
            "community": _spans((0.0, 1.0, "a"), (5.0, 6.0, "b"), capacity="unbounded"),
            "sortformer": _spans((0.0, 1.0, "spk0"), capacity=4),
        },
    )
    assert set(result["S0"]["bound_in"]) == {"community", "sortformer"}
    assert result["S1"]["bound_in"] == ("community",)
    assert result["S1"]["unbound_in"] == ("sortformer",)


def test_binding_agreement_is_the_fraction_of_tools_that_bound_the_speaker() -> None:
    """A per-speaker quantity, which is what the identity axis needs per speaker."""
    result = per_speaker_presence(
        {"S0": [(0.0, 1.0)], "S1": [(5.0, 6.0)]},
        spans_by_tool={
            "community": _spans((0.0, 1.0, "a"), (5.0, 6.0, "b"), capacity="unbounded"),
            "sortformer": _spans((0.0, 1.0, "spk0"), capacity=4),
        },
    )
    assert result["S0"]["binding_agreement"] == pytest.approx(1.0)
    assert result["S1"]["binding_agreement"] == pytest.approx(0.5)


def test_a_tool_at_capacity_that_missed_a_speaker_is_not_counted_against_them() -> None:
    """D-19's censoring, applied to the binding rather than to the count.

    A 1-capacity tool that bound one speaker had no second label to offer, so its silence about a
    second speaker is not disagreement. Counting it as such would make every low-capacity tool an
    argument against every additional speaker.
    """
    result = per_speaker_presence(
        {"S0": [(0.0, 1.0)], "S1": [(5.0, 6.0)]},
        spans_by_tool={
            "community": _spans((0.0, 1.0, "a"), (5.0, 6.0, "b"), capacity="unbounded"),
            "tiny": _spans((0.0, 1.0, "only"), capacity=1),
        },
    )
    assert result["S1"]["unbound_in"] == (), "the tiny tool was at capacity, so it did not dissent"
    assert result["S1"]["censored_in"] == ("tiny",)
    assert result["S1"]["binding_agreement"] == pytest.approx(1.0), "one eligible tool, and it bound"


def test_a_speaker_no_tool_bound_reports_zero_agreement_rather_than_vanishing() -> None:
    """A fused id nothing supports is a finding, not an absence."""
    result = per_speaker_presence(
        {"S0": [(20.0, 21.0)]},
        spans_by_tool={"community": _spans((0.0, 1.0, "a"), capacity="unbounded")},
    )
    assert result["S0"]["bound_in"] == ()
    assert result["S0"]["binding_agreement"] == pytest.approx(0.0)
    assert result["S0"]["spans"] == ()
