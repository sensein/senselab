"""Tests for assisted recalibration (FR-016/017)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from senselab.audio.workflows.ranking import io, recalibrate
from senselab.audio.workflows.ranking.types import Annotation, MetricDefinition, SignalTerm

from .conftest import quality_for_index


def _two_signal_table(make_signal_table: Callable[..., Path], n: int = 24) -> Path:
    """Signal `good` aligns with quality; `noise` is irrelevant."""
    good = [float(i) for i in range(n)]
    noise = [float((i * 7) % 5) for i in range(n)]
    return make_signal_table({"good": good, "noise": noise})


def _annotations(n: int) -> list[Annotation]:
    """Tertile ordinal annotations over n items."""
    return [
        Annotation(item_id=f"item{i:03d}", label=quality_for_index(i, n), score=None, unit="file") for i in range(n)
    ]


def test_recalibration_improves_agreement(make_signal_table: Callable[..., Path]) -> None:
    """Recalibration raises rank agreement and up-weights the informative signal."""
    n = 24
    table = io.load_signal_table(_two_signal_table(make_signal_table, n))
    base = MetricDefinition(name="bad", terms=[SignalTerm("good", 0.0), SignalTerm("noise", 1.0)])
    result = recalibrate.propose_recalibration(table, base, _annotations(n))
    assert result.status in ("proposed", "warned")
    assert result.proposed_definition is not None
    assert result.agreement_after is not None and result.agreement_before is not None
    assert result.agreement_after >= result.agreement_before
    weights = {t.signal: t.weight for t in result.proposed_definition.terms}
    assert weights["good"] > weights["noise"]


def test_refuse_too_few_annotations(make_signal_table: Callable[..., Path]) -> None:
    """Recalibration refuses below the minimum annotation count (FR-017)."""
    table = io.load_signal_table(_two_signal_table(make_signal_table, 24))
    base = MetricDefinition(name="m", terms=[SignalTerm("good", 1.0), SignalTerm("noise", 1.0)])
    result = recalibrate.propose_recalibration(table, base, _annotations(24)[:3])
    assert result.status == "refused"
    assert result.proposed_definition is None
    assert "annotations" in result.message


def test_refuse_single_quality_level(make_signal_table: Callable[..., Path]) -> None:
    """Recalibration refuses when there is only one distinct quality level."""
    n = 15
    table = io.load_signal_table(_two_signal_table(make_signal_table, n))
    base = MetricDefinition(name="m", terms=[SignalTerm("good", 1.0), SignalTerm("noise", 1.0)])
    anns = [Annotation(item_id=f"item{i:03d}", label="good", score=None, unit="file") for i in range(n)]
    result = recalibrate.propose_recalibration(table, base, anns)
    assert result.status == "refused"
    assert "distinct quality levels" in result.message
