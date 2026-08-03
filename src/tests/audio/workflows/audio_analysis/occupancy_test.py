"""Speaker occupancy and count, derived from spans across diarizers of differing capacity (D-19).

This replaces a Poisson-binomial over `segmentation-3.0`'s per-speaker channel probabilities. That
construction treated the channels as independent Bernoullis; they are a **powerset conversion**, whose
classes are mutually exclusive by construction and whose per-speaker columns are derived from them —
so the independence it assumed was never there. It was one model's internal confidence dressed as a
distribution over speaker count.

The honest uncertainty about "how many speakers are active here" is the same as for every other axis
in this design: **disagreement across models.** Each diarizer's spans give a count at time *t*, and
the spread across diarizers is the uncertainty — measured, not assumed.

And it composes with censoring. A tool at its capacity cannot report one more speaker and **does not
say so**, so its count is a *lower bound*. Treating it as a point makes a fused posterior biased
toward the smallest-capacity tool, which is invisible in the output when it bites.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.occupancy import (
    count_at,
    count_posterior,
    occupancy,
)
from senselab.audio.workflows.audio_analysis.shapes import Capacity, Span, Spans


def _spans(*items: tuple[float, float, str], capacity: Capacity = None) -> Spans:
    return Spans(spans=tuple(Span(start=s, end=e, label=lab) for s, e, lab in items), capacity=capacity)


def _posterior(counts: dict[str, int], capacities: dict[str, object]) -> dict[str, Any]:
    """The posterior, asserted present. ``None`` means no source reported, tested separately."""
    result = count_posterior(counts, capacities=capacities)  # type: ignore[arg-type]
    assert result is not None, "expected a posterior from a non-empty count set"
    return result


# ── occupancy: spans → per-bucket coverage, per speaker ────────────────


def test_occupancy_reports_coverage_per_speaker_not_a_single_fraction() -> None:
    """``covered_fraction`` collapsed a bucket that may contain two speakers into one number."""
    result = occupancy(_spans((0.0, 0.5, "spk0"), (0.25, 1.0, "spk1")), start=0.0, end=1.0)
    assert result == {"spk0": pytest.approx(0.5), "spk1": pytest.approx(0.75)}


def test_a_speaker_absent_from_a_bucket_is_absent_rather_than_zero() -> None:
    """Zero coverage is a claim the diarizer did not make; it simply drew no span here."""
    result = occupancy(_spans((0.0, 0.5, "spk0")), start=0.6, end=1.0)
    assert result == {}


def test_occupancy_clips_a_span_to_the_bucket() -> None:
    """A span running past the bucket contributes only its overlap, never more than 1.0."""
    result = occupancy(_spans((0.0, 10.0, "spk0")), start=0.0, end=1.0)
    assert result == {"spk0": pytest.approx(1.0)}


def test_two_spans_for_one_speaker_union_rather_than_summing() -> None:
    """Overlapping spans from one label are one speaker present, not 1.5 speakers."""
    result = occupancy(_spans((0.0, 0.6, "spk0"), (0.4, 1.0, "spk0")), start=0.0, end=1.0)
    assert result == {"spk0": pytest.approx(1.0)}


# ── count_at: per-tool count, with censoring ──────────────────────────


def test_the_count_is_how_many_distinct_speakers_cover_the_instant() -> None:
    """Overlap is an instantaneous fact, so the count is evaluated at a time, not over a span."""
    spans = _spans((0.0, 1.0, "spk0"), (0.5, 1.5, "spk1"))
    assert count_at(spans, 0.25) == 1
    assert count_at(spans, 0.75) == 2
    assert count_at(spans, 1.25) == 1


def test_two_speakers_taking_turns_do_not_read_as_overlap() -> None:
    """The defect the frame-level version existed to avoid, preserved here.

    Per-bucket channel means made two speakers alternating within a bucket average to 0.5 each,
    reporting an overlap that never occurred. Evaluating at an instant cannot produce that.
    """
    spans = _spans((0.0, 0.5, "spk0"), (0.5, 1.0, "spk1"))
    assert max(count_at(spans, t) for t in (0.1, 0.3, 0.6, 0.9)) == 1


def test_a_tool_at_its_capacity_is_reported_as_censored() -> None:
    """It cannot report one more speaker and does not say so — the count is a lower bound."""
    spans = _spans((0.0, 1.0, "a"), (0.0, 1.0, "b"), (0.0, 1.0, "c"), (0.0, 1.0, "d"), capacity=4)
    assert count_at(spans, 0.5) == 4
    assert spans.is_censored_at(4) is True


# ── count_posterior: the distribution comes from cross-tool spread ─────


def test_unanimous_tools_give_a_certain_count() -> None:
    """Agreement across independent tools is what certainty about a count looks like."""
    result = _posterior({"a": 2, "b": 2, "c": 2}, {"a": "unbounded", "b": 4, "c": 8})
    assert result["counts"] == {2: pytest.approx(1.0)}
    assert result["uncertainty"] == pytest.approx(0.0)
    assert result["expected_count"] == pytest.approx(2.0)


def test_disagreeing_tools_give_a_spread() -> None:
    """The measured quantity: two tools saying 2 and one saying 3 is genuine doubt."""
    result = _posterior({"a": 2, "b": 2, "c": 3}, {"a": "unbounded", "b": 8, "c": 8})
    assert result["counts"][2] == pytest.approx(2 / 3)
    assert result["counts"][3] == pytest.approx(1 / 3)
    assert result["uncertainty"] > 0.0


def test_a_censored_tool_is_not_evidence_against_a_higher_count() -> None:
    """The D-19 rule, and the reason the whole thing was rebuilt.

    A 4-capacity tool reporting 4 while an unbounded tool reports 5 has not contradicted it — it had
    no fifth column. Counting its 4 as a vote against 5 biases the posterior toward the
    smallest-capacity tool.
    """
    censored = _posterior({"small": 4, "big": 5}, {"small": 4, "big": "unbounded"})
    assert censored["censored_sources"] == ("small",)
    # The censored tool cannot discriminate 4 from 5, so it backs both equally; the tool that *could*
    # see a fifth speaker saw one. So 5 leads — but 4 keeps the mass its lower bound licenses.
    assert censored["counts"][5] > censored["counts"][4]
    uncensored = _posterior({"small": 4, "big": 5}, {"small": 8, "big": "unbounded"})
    assert uncensored["counts"][4] == pytest.approx(0.5), "with a column to spare, its 4 is a real vote"
    assert censored["counts"][4] < uncensored["counts"][4], "censoring must weaken it, not erase it"


def test_a_tool_below_its_capacity_is_full_evidence() -> None:
    """Censoring applies only at the ceiling. Below it, the tool had a column and did not use it."""
    result = _posterior({"small": 2, "big": 5}, {"small": 4, "big": "unbounded"})
    assert result["counts"][2] == pytest.approx(0.5)
    assert result["counts"][5] == pytest.approx(0.5)
    assert result["censored_sources"] == ()


def test_a_censored_tool_still_corroborates_counts_at_or_above_its_bound() -> None:
    """Its evidence is *at least this many*, which is real information, not an abstention."""
    result = _posterior({"small": 3, "mid": 3, "big": 4}, {"small": 3, "mid": "unbounded", "big": "unbounded"})
    # One tool said 3 outright, one said 4 outright, and the censored one said "at least 3" — which
    # backs both. So they tie, and the censored tool has neither abstained nor decided it.
    assert result["counts"][3] == pytest.approx(0.5)
    assert result["counts"][4] == pytest.approx(0.5)


def test_when_every_tool_is_censored_the_result_says_it_is_a_lower_bound() -> None:
    """Otherwise a ceiling reached by all tools reads as a confident count."""
    result = _posterior({"a": 3, "b": 4}, {"a": 3, "b": 4})
    assert result["lower_bounded"] is True


def test_a_posterior_from_uncensored_tools_is_not_lower_bounded() -> None:
    """A tool below its ceiling had a column to spare, so its count is a point not a floor."""
    result = _posterior({"a": 2, "b": 3}, {"a": "unbounded", "b": 8})
    assert result["lower_bounded"] is False


def test_no_tools_yields_none_rather_than_a_flat_distribution() -> None:
    """A guess with no evidence behind it is the failure this design keeps finding."""
    assert count_posterior({}, capacities={}) is None


def test_p_overlap_is_the_mass_above_one_speaker() -> None:
    """Kept from J1: the quantity consumers actually ask for."""
    result = _posterior({"a": 1, "b": 2}, {"a": "unbounded", "b": "unbounded"})
    assert result["p_overlap"] == pytest.approx(0.5)


def test_the_contributing_tools_are_recorded() -> None:
    """A posterior whose sources are unnamed cannot be re-derived or argued with."""
    result = _posterior({"a": 1, "b": 2}, {"a": "unbounded", "b": 4})
    assert result["contributing_sources"] == ("a", "b")


def test_a_missing_capacity_is_not_read_as_unbounded() -> None:
    """Absent and unbounded are different claims; guessing the permissive one hides the bias."""
    with pytest.raises(KeyError, match="capacity"):
        count_posterior({"a": 1}, capacities={})


def test_a_censored_tool_does_not_abstain() -> None:
    """A lower bound is information, and dropping the source entirely would discard it.

    The subtle half of the censoring rule: the source must not be evidence *against* a higher count,
    and must still be evidence *for* the range it does support. Abstention and full-weight are both
    wrong, in opposite directions.
    """
    alone = _posterior({"small": 3}, {"small": 3})
    assert alone["counts"] == {3: pytest.approx(1.0)}
    assert alone["lower_bounded"] is True, "and the reader is told the 3 is a floor"
