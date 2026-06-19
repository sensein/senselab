"""Shared fixtures for ranking-workflow tests.

Builds deterministic synthetic signal tables (no model loads) with controllable
rank-vs-quality alignment, plus helpers to construct a ranking store.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.ranking.store import RankingStore
from senselab.audio.workflows.ranking.types import QualityLabel


@pytest.fixture
def make_signal_table(tmp_path: Path) -> Callable[..., Path]:
    """Factory: write a signal-table parquet and return its path.

    ``columns`` maps signal name → list of floats (use ``math.nan`` for missing).
    """

    def _make(
        columns: dict[str, list[float]],
        *,
        item_ids: list[str] | None = None,
        unit: str = "file",
        name: str = "signals.parquet",
        with_locators: bool = False,
    ) -> Path:
        n = len(next(iter(columns.values())))
        ids = item_ids or [f"item{i:03d}" for i in range(n)]
        data: dict[str, list] = {"item_id": ids, "unit": [unit] * n}
        if unit == "segment" and with_locators:
            data["source_audio"] = [f"rec_{i}" for i in range(n)]
            data["start"] = [float(i) for i in range(n)]
            data["end"] = [float(i) + 1.0 for i in range(n)]
        for sig, vals in columns.items():
            data[sig] = vals
        path = tmp_path / name
        pq.write_table(pa.table(data), path)
        return path

    return _make


@pytest.fixture
def store(tmp_path: Path) -> RankingStore:
    """A fresh ranking store under tmp_path."""
    return RankingStore(tmp_path / "ranking_store")


@pytest.fixture
def aligned_signals(make_signal_table: Callable[..., Path]) -> Path:
    """20 items where signal `q` increases with item index (clean rank↔quality)."""
    n = 20
    return make_signal_table({"q": [float(i) for i in range(n)]})


def quality_for_index(i: int, n: int) -> QualityLabel:
    """Map item index to an ordinal label by tertile (low index = poor)."""
    if i < n / 3:
        return "poor"
    if i < 2 * n / 3:
        return "acceptable"
    return "good"


NAN = math.nan
