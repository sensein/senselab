"""Reproducibility regression: identical inputs → byte-identical ranking (SC-003)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from senselab.audio.workflows.ranking.rank import rank_corpus
from senselab.audio.workflows.ranking.store import RankingStore
from senselab.audio.workflows.ranking.types import MetricDefinition, SignalTerm


def test_rerank_byte_identical(tmp_path: Path, make_signal_table: Callable[..., Path]) -> None:
    """Re-ranking identical inputs (with ties) produces byte-identical parquet."""
    cols = {"q": [3.0, 1.0, 3.0, 2.0, 3.0, 1.0]}
    path_a = make_signal_table(cols, name="a.parquet")
    path_b = make_signal_table(cols, name="b.parquet")
    defn = MetricDefinition(name="m", terms=[SignalTerm("q", 1.0)])

    store_a = RankingStore(tmp_path / "a")
    store_b = RankingStore(tmp_path / "b")
    rank_corpus(store_a, path_a, defn, created_at="fixed", as_version="v1")
    rank_corpus(store_b, path_b, defn, created_at="fixed", as_version="v1")

    assert store_a.ranking_path("v1").read_bytes() == store_b.ranking_path("v1").read_bytes()
