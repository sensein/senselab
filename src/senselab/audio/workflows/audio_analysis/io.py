"""Parquet writer for ``UncertaintyRow``s.

Writes one parquet per ``AxisResult`` with a stable schema: ``start``, ``end``, ``axis``,
``aggregated_uncertainty``, ``contributing_models``, ``model_votes`` (JSON-encoded for
heterogeneous-shape robustness — Arrow's strict struct typing fights us when different
axes have different vote shapes), ``comparison_status``. The provenance dict goes into
``schema.metadata`` under the ``comparator_provenance`` key per FR-014.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.audio_analysis.types import AxisResult


def write_axis_parquet(
    axis_result: AxisResult,
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Serialize an ``AxisResult`` to parquet at ``dest``.

    Returns the destination path. Creates parent directories. Always writes the file —
    even when ``axis_result.rows`` is empty — so downstream consumers can rely on the
    9-parquet output shape per SC-002.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    starts = [r.start for r in axis_result.rows]
    ends = [r.end for r in axis_result.rows]
    axes = [r.axis for r in axis_result.rows]
    uncertainties = [r.aggregated_uncertainty for r in axis_result.rows]
    raw_uncertainties = [r.raw_aggregated_uncertainty for r in axis_result.rows]
    intensity_weights = [r.intensity_weight for r in axis_result.rows]
    contributing = [list(r.contributing_models) for r in axis_result.rows]
    votes_json = [json.dumps(r.model_votes, default=str, separators=(",", ":")) for r in axis_result.rows]
    statuses = [r.comparison_status for r in axis_result.rows]

    table = pa.table(
        {
            "start": pa.array(starts, type=pa.float64()),
            "end": pa.array(ends, type=pa.float64()),
            "axis": pa.array(axes, type=pa.string()),
            "aggregated_uncertainty": pa.array(uncertainties, type=pa.float64()),
            "raw_aggregated_uncertainty": pa.array(raw_uncertainties, type=pa.float64()),
            "intensity_weight": pa.array(intensity_weights, type=pa.float64()),
            "contributing_models": pa.array(contributing, type=pa.list_(pa.string())),
            "model_votes": pa.array(votes_json, type=pa.string()),
            "comparison_status": pa.array(statuses, type=pa.string()),
        }
    )

    metadata: dict[bytes, bytes] = {}
    if axis_result.provenance or provenance:
        merged = {**axis_result.provenance, **(provenance or {})}
        metadata[b"comparator_provenance"] = json.dumps(merged, default=str).encode("utf-8")

    if metadata:
        table = table.replace_schema_metadata(metadata)

    pq.write_table(table, dest)
    return dest
