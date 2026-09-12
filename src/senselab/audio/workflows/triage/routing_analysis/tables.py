"""Writing and reading the generated tables of the routing analysis as parquet.

Every table this package generates is columnar, is read back by column rather than by row, and is
large enough that the encoding matters. ``specs/20260912-parquet-tables/design.md`` carries the
measurements behind the compression codec, the row-group size and the flattening.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

COMPRESSION = "zstd"
"""The codec every generated table is written with."""

ROW_GROUP_ROWS = 8192
"""How many rows share one row group, and so one set of per-column chunk statistics."""

METADATA_KEY = b"senselab.routing_analysis"
"""The parquet key-value metadata entry a table's non-tabular header is carried under."""


def write_table(table: pa.Table, path: Path, header: Mapping[str, Any] | None = None) -> None:
    """Write one table, with any scalar header carried in the file's key-value metadata.

    Args:
        table: The table.
        path: Where to write it. Overwritten.
        header: Scalars that describe the whole table rather than any row, JSON-encoded into the
            file metadata so the table stays one file and one shape.
    """
    if header is not None:
        schema = table.schema.with_metadata({METADATA_KEY: json.dumps(header, sort_keys=True).encode("utf-8")})
        table = table.replace_schema_metadata(schema.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression=COMPRESSION, row_group_size=ROW_GROUP_ROWS)


def read_header(table: pa.Table) -> dict[str, Any]:
    """The scalar header :func:`write_table` carried in a table's file metadata.

    Args:
        table: A table read back from parquet.

    Returns:
        The decoded header, or an empty mapping when the table carries none.
    """
    metadata = table.schema.metadata or {}
    raw = metadata.get(METADATA_KEY)
    if raw is None:
        return {}
    decoded: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return decoded


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path, header: Mapping[str, Any] | None = None) -> int:
    """Write a sequence of flat mappings as one table, unioning their keys into the schema.

    Args:
        rows: The rows. A key one row omits is null there, which is how an unmeasured quantity
            stays distinct from a measured zero.
        path: Where to write.
        header: Scalars describing the whole table, as in :func:`write_table`.

    Returns:
        How many rows were written.
    """
    write_table(pa.Table.from_pylist(list(rows)), path, header)
    return len(rows)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:  # noqa: ANN401
    """Fold one nested JSON-shaped value into ``parent.child`` columns of a flat row.

    Args:
        prefix: The column name the value sits under.
        value: A mapping, or a leaf.
        out: The row being built, updated in place.
    """
    if isinstance(value, Mapping):
        for key, nested in value.items():
            flatten(f"{prefix}.{key}", nested, out)
        return
    out[prefix] = value
