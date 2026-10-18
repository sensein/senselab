"""Persistence primitives + signal-table / ranking / movement (de)serialization.

All long-lived artifacts are written atomically (write a sibling ``.tmp`` then
``os.replace``) and stamped with ``schema_version`` so readers never see a
half-written file and never silently misinterpret a newer shape. Schemas are
defined in ``specs/20260604-173646-iterative-metric-ranking/contracts/``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.ranking.constants import SCHEMA_VERSION
from senselab.audio.workflows.ranking.types import MovementReport, Ranking, RankingItem, SignalTable


class RankingSchemaError(ValueError):
    """Raised when an on-disk artifact cannot be safely interpreted."""


# ── Generic atomic IO ──────────────────────────────────────────────────────


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (``.tmp`` then ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as JSON with a ``schema_version`` stamp."""
    payload = {"schema_version": SCHEMA_VERSION, **payload}
    atomic_write_text(Path(path), json.dumps(payload, indent=2, default=str))


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact written by :func:`save_json`.

    Refuses a ``schema_version`` greater than this reader's ``SCHEMA_VERSION``.
    """
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(data.get("schema_version", SCHEMA_VERSION))
    if version > SCHEMA_VERSION:
        raise RankingSchemaError(f"{path}: schema_version {version} is newer than supported {SCHEMA_VERSION}")
    return data


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    """Atomically write a pyarrow table to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _meta_bytes(meta: dict[str, Any]) -> dict[bytes, bytes]:
    """Encode a metadata dict as parquet key/value bytes (values JSON/str)."""
    out: dict[bytes, bytes] = {}
    for key, value in meta.items():
        encoded = value if isinstance(value, str) else json.dumps(value)
        out[key.encode("utf-8")] = encoded.encode("utf-8")
    return out


def _meta_str(table: pa.Table, key: str, default: str = "") -> str:
    """Read a string metadata value from a parquet table."""
    raw = (table.schema.metadata or {}).get(key.encode("utf-8"))
    return raw.decode("utf-8") if raw is not None else default


# ── Signal table (input) ───────────────────────────────────────────────────

_RESERVED_COLUMNS = {"item_id", "unit", "source_audio", "start", "end"}


def load_signal_table(path: Path) -> SignalTable:
    """Load a per-item signal table (``signal-table.parquet.md``).

    Validates unique ``item_id`` and a single ``unit``; every non-reserved
    column is treated as a signal (``NaN`` = missing). For segment unit,
    ``(source_audio, start, end)`` locators are captured per item.
    """
    table = pq.read_table(path)
    names = table.column_names
    if "item_id" not in names:
        raise RankingSchemaError(f"{path}: signal table missing required 'item_id' column")
    if "unit" not in names:
        raise RankingSchemaError(f"{path}: signal table missing required 'unit' column")

    item_ids = [str(x) for x in table.column("item_id").to_pylist()]
    if len(set(item_ids)) != len(item_ids):
        raise RankingSchemaError(f"{path}: duplicate item_id values in signal table")

    units = {str(u) for u in table.column("unit").to_pylist()}
    if len(units) != 1:
        raise RankingSchemaError(f"{path}: signal table mixes units {units!r}; one unit per table")
    unit = units.pop()
    if unit not in ("file", "segment"):
        raise RankingSchemaError(f"{path}: invalid unit {unit!r}")

    signal_columns = [n for n in names if n not in _RESERVED_COLUMNS]
    if not signal_columns:
        raise RankingSchemaError(f"{path}: signal table has no signal columns")

    columns: dict[str, Any] = {name: np.asarray(table.column(name).to_pylist(), dtype=float) for name in signal_columns}

    locators: dict[str, tuple[str, float, float]] = {}
    if unit == "segment" and {"source_audio", "start", "end"} <= set(names):
        src = table.column("source_audio").to_pylist()
        start = table.column("start").to_pylist()
        end = table.column("end").to_pylist()
        for iid, s, a, b in zip(item_ids, src, start, end, strict=True):
            locators[iid] = (str(s), float(a), float(b))

    return SignalTable(
        unit=unit,  # type: ignore[arg-type]
        item_ids=item_ids,
        columns=columns,
        signal_columns=signal_columns,
        locators=locators,
    )


# ── Ranking (output) ───────────────────────────────────────────────────────


def write_ranking(path: Path, ranking: Ranking) -> None:
    """Write a ranking to parquet (``ranking.parquet.md``)."""
    item_id, score, rank, percentile, band, status, reason = [], [], [], [], [], [], []
    for it in ranking.items:
        item_id.append(it.item_id)
        score.append(math.nan if it.score is None else float(it.score))
        rank.append(-1 if it.rank is None else int(it.rank))
        percentile.append(math.nan if it.percentile is None else float(it.percentile))
        band.append(it.band or "")
        status.append(it.status)
        reason.append(it.reason or "")

    table = pa.table(
        {
            "item_id": pa.array(item_id, pa.string()),
            "score": pa.array(score, pa.float64()),
            "rank": pa.array(rank, pa.int64()),
            "percentile": pa.array(percentile, pa.float64()),
            "band": pa.array(band, pa.string()),
            "status": pa.array(status, pa.string()),
            "reason": pa.array(reason, pa.string()),
        }
    )
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "version_id": ranking.version_id,
        "unit": ranking.unit,
        "band_fraction": str(ranking.band_fraction),
        "n_scored": str(ranking.n_scored),
        "n_unscorable": str(ranking.n_unscorable),
        **{k: ranking.provenance[k] for k in ranking.provenance},
    }
    table = table.replace_schema_metadata(_meta_bytes(meta))
    atomic_write_parquet(Path(path), table)


def read_ranking(path: Path) -> Ranking:
    """Reconstruct a :class:`Ranking` written by :func:`write_ranking`."""
    table = pq.read_table(path)
    version = int(_meta_str(table, "schema_version", str(SCHEMA_VERSION)))
    if version > SCHEMA_VERSION:
        raise RankingSchemaError(f"{path}: ranking schema_version {version} > supported {SCHEMA_VERSION}")

    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    items: list[RankingItem] = []
    for i in range(table.num_rows):
        status = cols["status"][i]
        scorable = status == "scored"
        items.append(
            RankingItem(
                item_id=str(cols["item_id"][i]),
                score=float(cols["score"][i]) if scorable else None,
                rank=int(cols["rank"][i]) if scorable else None,
                percentile=float(cols["percentile"][i]) if scorable else None,
                band=cols["band"][i] or None if scorable else None,
                status=status,
                reason=cols["reason"][i] or None,
            )
        )
    return Ranking(
        version_id=_meta_str(table, "version_id"),
        unit=_meta_str(table, "unit") or "file",  # type: ignore[arg-type]
        band_fraction=float(_meta_str(table, "band_fraction", "0.2")),
        items=items,
        n_scored=int(_meta_str(table, "n_scored", "0")),
        n_unscorable=int(_meta_str(table, "n_unscorable", "0")),
        provenance={
            k: _meta_str(table, k) for k in ("metric_definition_hash", "tie_break", "created_at", "signal_columns")
        },
    )


# ── Movement report ────────────────────────────────────────────────────────


def write_movement_report(path: Path, report: MovementReport) -> None:
    """Write a movement report to JSON (``movement-report.schema.md``)."""
    payload = {
        "from_version": report.from_version,
        "to_version": report.to_version,
        "unit": report.unit,
        "band_fraction": report.band_fraction,
        "band_summary": report.band_summary,
        "added": report.added,
        "removed": report.removed,
        "became_unscorable": report.became_unscorable,
        "entries": [asdict(e) for e in report.entries],
    }
    save_json(Path(path), payload)
