"""Parquet writers for the level-1 evidence artifacts.

``write_signal_parquet`` writes one file per ``(pass, signal)`` — long format, one row per
bucket, the measurement carried as JSON in the tool's own units. Units, window, hop, model and
revision travel in ``schema.metadata`` so a reader can interpret the number without knowing which
module produced it.

There is deliberately no axis writer here. An axis is a fold across signals *and* passes, so it
cannot be indexed by pass; the fused axes are written by
``fuse.write_final_uncertainty`` to ``L2/round<N>/uncertainty/<axis>.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.audio_analysis.types import SignalResult


def write_signal_parquet(
    signal_result: SignalResult,
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Serialize a ``SignalResult`` to parquet at ``dest``.

    Returns the destination path. Creates parent directories. Always writes the file — even when
    ``signal_result.rows`` is empty — so "the signal ran and found nothing" stays distinguishable
    from "the signal never ran".
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = signal_result.rows

    columns: dict[str, pa.Array] = {
        "start": pa.array([r.start for r in rows], type=pa.float64()),
        "end": pa.array([r.end for r in rows], type=pa.float64()),
        "signal": pa.array([r.signal for r in rows], type=pa.string()),
        # JSON rather than a struct: signals report different shapes, and Arrow's strict struct
        # typing would force a union of every signal's fields onto every signal's file.
        "measurement": pa.array(
            [json.dumps(r.measurement, default=str, separators=(",", ":")) for r in rows], type=pa.string()
        ),
        "units": pa.array([r.units for r in rows], type=pa.string()),
        "native_window_s": pa.array([r.native_window_s for r in rows], type=pa.float64()),
        "resolution_s": pa.array([r.resolution_s for r in rows], type=pa.float64()),
        "model_id": pa.array([r.model_id for r in rows], type=pa.string()),
        "revision": pa.array([r.revision for r in rows], type=pa.string()),
        "status": pa.array([r.status for r in rows], type=pa.string()),
    }
    table = pa.table(columns)

    merged = {**signal_result.provenance, **(provenance or {})}
    merged.setdefault("pass", signal_result.pass_label)
    merged.setdefault("signal", signal_result.signal)
    table = table.replace_schema_metadata({b"signal_provenance": json.dumps(merged, default=str).encode("utf-8")})

    pq.write_table(table, dest)
    return dest


def write_signal_stability(
    per_bucket: Sequence[Mapping[str, Any]],
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write ``L1/stability/<signal>.parquet`` — one signal's cross-pass disagreement per bucket.

    Perturbation stability is a property of a *signal*, not of an axis: the two passes are the
    same recording under a transform, so a signal that answers differently between them has not
    earned its weight. Keyed by signal for that reason, rather than by a third pseudo-pass.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, pa.Array] = {
        "start": pa.array([float(r["start"]) for r in per_bucket], type=pa.float64()),
        "end": pa.array([float(r["end"]) for r in per_bucket], type=pa.float64()),
        "signal": pa.array([str(r["signal"]) for r in per_bucket], type=pa.string()),
        "pass_a": pa.array([str(r["pass_a"]) for r in per_bucket], type=pa.string()),
        "pass_b": pa.array([str(r["pass_b"]) for r in per_bucket], type=pa.string()),
        "abs_delta": pa.array([float(r["abs_delta"]) for r in per_bucket], type=pa.float64()),
        "n_passes_present": pa.array([int(r["n_passes_present"]) for r in per_bucket], type=pa.int64()),
    }
    table = pa.table(columns)
    if provenance:
        table = table.replace_schema_metadata({b"stability_provenance": json.dumps(provenance, default=str).encode()})
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


def write_noise_floor(rows: Any, dest_dir: Path) -> Path:  # noqa: ANN401 — NoiseFloorEstimate sequence
    """Write ``noise_floor.parquet`` for one pass (T059)."""
    import pandas as pd

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "band_low_hz",
        "band_high_hz",
        "target_activity",
        "floor_db",
        "quantile",
        "bias_correction_db",
        "window_s",
        "iterations",
        "frames",
        "recorder_floor_db",
        "binding",
        "status",
    ]
    records = [r.to_row() for r in rows]
    frame = pd.DataFrame(records, columns=columns) if records else pd.DataFrame({c: [] for c in columns})
    out = dest_dir / "noise_floor.parquet"
    frame.to_parquet(out, index=False)
    return out


def write_background_sources(findings: Any, dest_dir: Path) -> Path:  # noqa: ANN401 — SourceFinding sequence
    """Write ``background_sources.parquet`` for one pass (T069).

    Written even when empty. Zero rows on amplified noise floor is the *expected* result
    (SC-018), and an absent file would make "nothing was found" indistinguishable from
    "the stage never ran".
    """
    import pandas as pd

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "start",
        "end",
        "category",
        "label",
        "classifier",
        "above_floor_db",
        "tier",
        "binding_floor",
        "variant",
        "gain_db",
        "computed_on",
        "padding_fraction",
        "from_mask_region",
        "mask_confidence",
        "leakage_margin_db",
        "suppression_depth_db",
        "flatness",
        "modulation_depth",
        "occupancy",
        "stationary_pass",
        "discounted_reason",
    ]
    records = [f.to_row() for f in findings]
    frame = pd.DataFrame(records, columns=columns) if records else pd.DataFrame({c: [] for c in columns})
    out = dest_dir / "background_sources.parquet"
    frame.to_parquet(out, index=False)
    return out


def write_suppression_json(suppression: Any, dest_dir: Path) -> Path:  # noqa: ANN401 — ForegroundSuppression
    """Write ``suppression.json`` for one pass.

    Always carries the achieved depth when suppression was requested (SC-016), so a null
    background result is attributable to insufficient suppression rather than to absence of
    background content.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / "suppression.json"
    out.write_text(json.dumps(suppression.to_json(), indent=2) + "\n")
    return out
