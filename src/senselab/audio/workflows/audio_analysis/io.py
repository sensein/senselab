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

    # Scene-aware presence + utterance extension columns (feature 20260722-175022).
    # Additive and all-nullable: rows on axes that don't populate a given column
    # write null, keeping one uniform schema across the three parquets. Existing
    # column-projecting readers ignore these.
    float_extension_columns = (
        "presence_confidence",
        "presence_uncertainty",
        "quality_snr",
        "quality_clip",
        "quality_reverb",
        "quality_bandwidth",
        "quality_uncertainty",
        "src_speech",
        "src_people",
        "src_machine",
        "src_environment",
        "token_entropy",
        "scene_quality_coupling",
    )

    columns: dict[str, pa.Array] = {
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
    for col in float_extension_columns:
        columns[col] = pa.array([getattr(r, col) for r in axis_result.rows], type=pa.float64())
    columns["src_dominant"] = pa.array([r.src_dominant for r in axis_result.rows], type=pa.string())

    table = pa.table(columns)

    metadata: dict[bytes, bytes] = {}
    if axis_result.provenance or provenance:
        merged = {**axis_result.provenance, **(provenance or {})}
        metadata[b"comparator_provenance"] = json.dumps(merged, default=str).encode("utf-8")

    if metadata:
        table = table.replace_schema_metadata(metadata)

    pq.write_table(table, dest)
    return dest


def write_background_mask(
    mask: Any,  # noqa: ANN401 — BackgroundMask; typed loosely to keep io import-light
    dest_dir: Path,
) -> tuple[Path, Path]:
    """Write ``background_mask.parquet`` + ``background_mask.json`` for one pass.

    Both files are written even when the mask is empty. "No mask was produced" and "the
    mask was empty" are different facts about a recording, and omitting the files would
    make them indistinguishable — which matters because an empty mask means every
    background finding depends on suppression depth alone (FR-040).

    Args:
        mask: A ``BackgroundMask``.
        dest_dir: Pass output directory.

    Returns:
        ``(parquet_path, json_path)``.
    """
    import pandas as pd

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows = mask.to_rows()
    columns = [
        "region_id",
        "start",
        "end",
        "state",
        "uncertainty",
        "guard_trimmed_s",
        "contains_nontarget_speech",
        "supports_long_window",
        "target_event_types",
    ]
    frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame({c: [] for c in columns})
    parquet_path = dest_dir / "background_mask.parquet"
    frame.to_parquet(parquet_path, index=False)

    json_path = dest_dir / "background_mask.json"
    json_path.write_text(json.dumps(mask.to_json(), indent=2) + "\n")
    return parquet_path, json_path
