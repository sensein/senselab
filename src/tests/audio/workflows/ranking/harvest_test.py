"""Smoke test for the audio_analysis → signal-table adapter (T032)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.ranking import io
from senselab.audio.workflows.ranking.harvest import harvest_from_axis_parquets


def _axis_parquet(path: Path, starts: list[float], ends: list[float], vals: list[float]) -> Path:
    """Write a minimal audio_analysis-shaped axis parquet."""
    pq.write_table(pa.table({"start": starts, "end": ends, "aggregated_uncertainty": vals}), path)
    return path


def test_harvest_pivots_axes_to_signal_table(tmp_path: Path) -> None:
    """Per-axis parquets pivot into one segment signal table with one column per axis."""
    presence = _axis_parquet(tmp_path / "presence.parquet", [0.0, 1.0], [1.0, 2.0], [0.1, 0.4])
    identity = _axis_parquet(tmp_path / "identity.parquet", [0.0, 1.0], [1.0, 2.0], [0.2, 0.9])
    out = harvest_from_axis_parquets(
        {"presence_unc": presence, "identity_unc": identity},
        source_audio="rec1",
        out=tmp_path / "signals.parquet",
    )
    table = io.load_signal_table(out)
    assert table.unit == "segment"
    assert set(table.signal_columns) == {"presence_unc", "identity_unc"}
    assert len(table.item_ids) == 2
    assert table.item_ids[0] == "rec1#0.00-1.00"
