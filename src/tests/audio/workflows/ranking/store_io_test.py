"""Tests for store immutability and parquet/JSON round-trips (FR-007/018, SC-002)."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.rank import rank_corpus
from senselab.audio.workflows.ranking.store import RankingStore, metric_definition_hash
from senselab.audio.workflows.ranking.types import MetricDefinition, SignalTerm


def _defn(name: str = "m") -> MetricDefinition:
    """One-signal metric definition."""
    return MetricDefinition(name=name, terms=[SignalTerm("q", 1.0)])


def test_metric_version_immutable(store: RankingStore, aligned_signals: Path) -> None:
    """Re-writing an existing version id is refused (FR-018)."""
    rank_corpus(store, aligned_signals, _defn(), created_at="t0", as_version="v1")
    with pytest.raises(FileExistsError):
        rank_corpus(store, aligned_signals, _defn(), created_at="t1", as_version="v1")


def test_version_lineage_and_unit(store: RankingStore, aligned_signals: Path) -> None:
    """Versions record origin/lineage and the store fixes a single unit."""
    rank_corpus(store, aligned_signals, _defn(), created_at="t0")
    rank_corpus(store, aligned_signals, _defn("m2"), created_at="t1", origin="manual", parent_version_id="v1")
    assert store.list_versions() == ["v1", "v2"]
    assert store.unit() == "file"
    v2 = store.read_metric_version("v2")
    assert v2.origin == "manual" and v2.parent_version_id == "v1"


def test_ranking_parquet_round_trip(store: RankingStore, aligned_signals: Path) -> None:
    """A written ranking reloads with the same fields and provenance."""
    written = rank_corpus(store, aligned_signals, _defn(), created_at="t0")
    loaded = io.read_ranking(store.ranking_path("v1"))
    assert loaded.version_id == written.version_id
    assert loaded.n_scored == written.n_scored
    assert loaded.unit == "file"
    assert loaded.provenance["tie_break"] == "score_desc,item_id_asc"
    assert loaded.provenance["metric_definition_hash"] == metric_definition_hash(_defn())


def test_definition_hash_stable() -> None:
    """Definition hash is stable for equal definitions and differs otherwise."""
    assert metric_definition_hash(_defn()) == metric_definition_hash(_defn())
    assert metric_definition_hash(_defn("a")) != metric_definition_hash(_defn("b"))


def test_schema_version_round_trips(store: RankingStore, aligned_signals: Path) -> None:
    """Persisted metric versions carry a schema_version."""
    rank_corpus(store, aligned_signals, _defn(), created_at="t0")
    data = io.load_json(store.version_path("v1"))
    assert data["schema_version"] >= 1


def test_unscorable_reported_not_dropped(store: RankingStore, make_signal_table: Callable[..., Path]) -> None:
    """Unscorable items are persisted, not dropped (SC-002)."""
    path = make_signal_table({"q": [1.0, math.nan, 3.0, 4.0]})
    rank_corpus(store, path, _defn(), created_at="t0")
    loaded = io.read_ranking(store.ranking_path("v1"))
    assert len(loaded.items) == 4
    assert loaded.n_unscorable == 1
