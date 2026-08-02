"""Parquet writers for the level-1 evidence artifacts.

``write_signal_parquet`` writes **one file per signal**, accumulating across raw and every
perturbation — long format, one row per ``(perturbation, bucket)``, the measurement carried as
JSON in the tool's own units. Units, window, hop, model and revision travel per row so a reader
can interpret the number without knowing which module produced it.

One file per ``(pass, signal)`` was the earlier form, and it made the perturbation an index on
the *location*. That is what let a consumer open one perturbation's directory and get an answer
that looked like the signal's, when the signal's answer is the whole set of perturbations it was
measured under. Here the perturbation is a column, so asking for one of them is something the
reader has to say out loud.

There is deliberately no axis writer here. An axis is a fold across signals *and* perturbations,
so it can be indexed by neither; the fused axes are written by ``fuse.write_final_uncertainty``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.audio_analysis.types import SignalResult


def write_signal_parquet(
    signal_results: Sequence[SignalResult],
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Serialize one signal's rows, across every perturbation, to parquet at ``dest``.

    Args:
        signal_results: Every ``SignalResult`` for one signal — one per perturbation that
            measured it. Order is irrelevant; rows are sorted by ``(perturbation, start, end)``
            so the file is byte-reproducible.
        dest: ``L1/signals/<signal>.parquet``.
        provenance: Run-level provenance merged onto each result's own.

    Returns:
        The destination path.

    Raises:
        ValueError: If the results do not all describe the same signal. One file per signal is
            the artifact's identity; silently writing a mixture would make the file's name a lie.

    Always writes the file — even with no rows — so "the signal ran and found nothing" stays
    distinguishable from "the signal never ran".
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    names = {result.signal for result in signal_results}
    if len(names) > 1:
        raise ValueError(f"one file per signal, but got {sorted(names)}")

    flat = sorted(
        ((result.perturbation, row) for result in signal_results for row in result.rows),
        key=lambda pair: (pair[0], pair[1].start, pair[1].end),
    )
    rows = [row for _, row in flat]

    columns: dict[str, pa.Array] = {
        # The perturbation is a *dimension of the measurement*: this file is what the signal
        # reported across every transform of the recording, and a row that could not say which
        # one it came from would make the set unusable as evidence.
        "perturbation": pa.array([name for name, _ in flat], type=pa.string()),
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

    merged: dict[str, Any] = {
        "signal": next(iter(names), None),
        # Per perturbation, because each measured under its own model revision and window.
        "per_perturbation": {result.perturbation: dict(result.provenance) for result in signal_results},
        **(provenance or {}),
    }
    table = table.replace_schema_metadata({b"signal_provenance": json.dumps(merged, default=str).encode("utf-8")})

    pq.write_table(table, dest)
    return dest


def write_linked_votes(
    buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    axis: str,
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write ``L2/round0/votes/<axis>.parquet`` — the linked evidence, at the vote level.

    A *vote* is legitimately keyed ``(axis, bucket, source, pass, scope)``: it is one source's
    statement about one bucket of one pass, and a signal measured on a pass is a per-pass
    measurement. What may not be keyed by pass is the **axis** — the fold across signals and
    passes — which is why this file sits next to ``uncertainty/<axis>.parquet`` rather than under
    ``L1/<pass>/``: the link that turned measurements into statements applied a policy, and every
    threshold is L2's.

    This is what the artifact-driven adaptive path ingests, so its beliefs come from the same
    linked evidence the in-process path uses rather than from a per-pass axis fold.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float, float, str, str, str]] = []
    for perturbation in sorted(buckets_by_pass):
        for bucket in buckets_by_pass[perturbation] or []:
            start, end = float(bucket.get("start", 0.0)), float(bucket.get("end", 0.0))
            for source, payload in (bucket.get("votes") or {}).items():
                rows.append(
                    (
                        axis,
                        start,
                        end,
                        str(source),
                        str(perturbation),
                        json.dumps(payload, default=str, separators=(",", ":")),
                    )
                )
            # Bucket-level measurements that belong to no single source travel alongside, so a
            # re-ingest sees the same context the in-process path does.
            for name in ("frame_dispersion",):
                if isinstance(bucket.get(name), (int, float)):
                    rows.append(
                        (axis, start, end, f"__{name}__", str(perturbation), json.dumps({"value": bucket[name]}))
                    )
    table = pa.table(
        {
            "axis": pa.array([r[0] for r in rows], type=pa.string()),
            "start": pa.array([r[1] for r in rows], type=pa.float64()),
            "end": pa.array([r[2] for r in rows], type=pa.float64()),
            "source": pa.array([r[3] for r in rows], type=pa.string()),
            "perturbation": pa.array([r[4] for r in rows], type=pa.string()),
            "payload": pa.array([r[5] for r in rows], type=pa.string()),
        }
    )
    if provenance:
        table = table.replace_schema_metadata({b"link_provenance": json.dumps(provenance, default=str).encode()})
    pq.write_table(table, dest)
    return dest


def write_signal_stability(
    per_bucket: Sequence[Mapping[str, Any]],
    dest: Path,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Write one signal's cross-perturbation disagreement per bucket, as a **round derivative**.

    Perturbation stability is a property of a *signal*: the perturbations are the same recording
    under a transform, so a signal that answers differently between them has not earned its
    weight. Keyed by signal for that reason, rather than by a pseudo-perturbation.

    It sits under the round rather than under ``L1/`` because relating two perturbations is a
    fold over an input dimension, which is L2's by exactly the argument that makes an axis L2's.
    Each row carries ``pass_a`` and ``pass_b`` — two values of one dimension, which is what a
    fold looks like and what no L1 artifact may be.

    Its run-level summary has no file at all. ``L1/stability/signals.json`` used to hold the mean
    that sets each signal's fusion weight, and that same number is on every fused row as
    ``weight_basis[signal]["stability"]``: one quantity in two places is one quantity that can
    disagree with itself.
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
