#!/usr/bin/env python3
r"""Write the replay manifest: every recording of the AIRWAY families, and every recording whose run is incomplete.

    uv run python scripts/select_replay_manifest.py RECORDING_VECTORS.parquet --out-root DIR OUT.jsonl \
        [--branch AIRWAY] [--no-incomplete]

``RECORDING_VECTORS.parquet`` is the corpus table ``recording_vectors.py`` writes; its ``run_dir``
is relative to ``--out-root``, the directory the corpus runs live under. Each selected row becomes
one JSONL object carrying ``stem``, ``enhanced`` (``<out-root>/<run_dir>/run/streams/enhanced.flac``,
the path the extend drivers derive a run root from), ``declared_family``, ``verdict``, ``selected_by``
(``family``, ``incomplete`` or both), and the run's recorded identity as the table carries it
(``participant``, ``session``, ``task`` and their ``*_from`` sources; all null for a run that records none).
A corpus pass groups the manifest's rows by those columns, so a table built before it carried them is
refused: rebuild the corpus table.

``--branch`` names the branch whose in-family rows are selected (``AIRWAY`` by default); a row is
also selected where its ``run_status`` is ``incomplete``, unless ``--no-incomplete``. The selection is in
``specs/20261006-airway-move/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from senselab.audio.workflows.triage.identity import ROW_PARTICIPANT, ROW_SESSION, ROW_SOURCES, ROW_TASK
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.vocabulary import RunStatus

IDENTITY_COLUMNS = (ROW_PARTICIPANT, ROW_SESSION, ROW_TASK, *ROW_SOURCES.values())
"""The corpus table's identity columns, copied onto every manifest row."""


def select(table: pd.DataFrame, *, branch: str, include_incomplete: bool) -> pd.DataFrame:
    """The rows to replay, each with why it was selected.

    Args:
        table: The recording-vectors table.
        branch: The branch whose declared families are selected.
        include_incomplete: Whether every row whose run is incomplete is selected too.

    Returns:
        The selected rows, with a ``selected_by`` column.
    """
    by_family = table["declared_family"].isin(sorted(EXPECTATIONS[branch]))
    by_status = table["run_status"] == RunStatus.INCOMPLETE.value
    by_incomplete = by_status if include_incomplete else pd.Series(False, index=table.index)
    chosen = table[by_family | by_incomplete].copy()
    chosen["selected_by"] = [
        "+".join(name for name, hit in (("family", f), ("incomplete", r)) if hit)
        for f, r in zip(by_family[by_family | by_incomplete], by_incomplete[by_family | by_incomplete])
    ]
    return chosen


def _null(value: object) -> object:
    """A table value as JSON writes it: a pandas missing value becomes None."""
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def main() -> int:
    """Write the manifest and print what it holds.

    Returns:
        0; 2 when the table carries no identity columns.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("table", type=Path, help="recording_vectors.parquet")
    parser.add_argument("out", type=Path, help="The manifest JSONL to write")
    parser.add_argument("--out-root", type=Path, required=True, help="The directory the corpus runs live under")
    parser.add_argument("--branch", default="AIRWAY", choices=sorted(EXPECTATIONS), help="Whose families to select")
    parser.add_argument(
        "--no-incomplete", action="store_true", help="Do not also select every recording whose run is incomplete"
    )
    args = parser.parse_args()

    absent = [name for name in IDENTITY_COLUMNS if name not in pq.read_schema(args.table).names]
    if absent:
        print(
            f"refusing: {args.table} carries no {', '.join(absent)} column; rebuild the corpus table",
            file=sys.stderr,
        )
        return 2
    columns = ["stem", "run_dir", "declared_family", "verdict", "run_status", *IDENTITY_COLUMNS]
    table = pd.read_parquet(args.table, columns=columns)
    chosen = select(table, branch=args.branch, include_incomplete=not args.no_incomplete)
    with args.out.open("w") as handle:
        for row in chosen.to_dict(orient="records"):
            handle.write(
                json.dumps(
                    {
                        "stem": row["stem"],
                        "enhanced": str(args.out_root / str(row["run_dir"]) / "run" / "streams" / "enhanced.flac"),
                        "declared_family": _null(row["declared_family"]),
                        "verdict": row["verdict"],
                        "selected_by": row["selected_by"],
                        **{name: _null(row[name]) for name in IDENTITY_COLUMNS},
                    }
                )
                + "\n"
            )
    print(f"{len(chosen)} recordings -> {args.out}")
    print(chosen["selected_by"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
