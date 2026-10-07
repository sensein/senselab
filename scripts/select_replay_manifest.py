#!/usr/bin/env python3
r"""Write the replay manifest: every recording of the AIRWAY families, and every recording held for a rerun.

    uv run python scripts/select_replay_manifest.py RECORDING_VECTORS.parquet --out-root DIR OUT.jsonl \
        [--branch AIRWAY] [--no-rerun]

``RECORDING_VECTORS.parquet`` is the corpus table ``recording_vectors.py`` writes; its ``run_dir``
is relative to ``--out-root``, the directory the corpus runs live under. Each selected row becomes
one JSONL object carrying ``stem``, ``enhanced`` (``<out-root>/<run_dir>/run/streams/enhanced.flac``,
the path the extend drivers derive a run root from), ``declared_family``, ``verdict`` and
``selected_by`` (``family``, ``rerun`` or both).

``--branch`` names the branch whose in-family rows are selected (``AIRWAY`` by default); a row is
also selected where its verdict is ``rerun``, unless ``--no-rerun``. The selection is in
``specs/20261006-airway-move/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.vocabulary import Triage


def select(table: pd.DataFrame, *, branch: str, include_rerun: bool) -> pd.DataFrame:
    """The rows to replay, each with why it was selected.

    Args:
        table: The recording-vectors table.
        branch: The branch whose declared families are selected.
        include_rerun: Whether every ``rerun`` row is selected too.

    Returns:
        The selected rows, with a ``selected_by`` column.
    """
    by_family = table["declared_family"].isin(sorted(EXPECTATIONS[branch]))
    by_rerun = (table["verdict"] == Triage.RERUN.value) if include_rerun else pd.Series(False, index=table.index)
    chosen = table[by_family | by_rerun].copy()
    chosen["selected_by"] = [
        "+".join(name for name, hit in (("family", f), ("rerun", r)) if hit)
        for f, r in zip(by_family[by_family | by_rerun], by_rerun[by_family | by_rerun])
    ]
    return chosen


def main() -> int:
    """Write the manifest and print what it holds.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("table", type=Path, help="recording_vectors.parquet")
    parser.add_argument("out", type=Path, help="The manifest JSONL to write")
    parser.add_argument("--out-root", type=Path, required=True, help="The directory the corpus runs live under")
    parser.add_argument("--branch", default="AIRWAY", choices=sorted(EXPECTATIONS), help="Whose families to select")
    parser.add_argument("--no-rerun", action="store_true", help="Do not also select every rerun recording")
    args = parser.parse_args()

    table = pd.read_parquet(args.table, columns=["stem", "run_dir", "declared_family", "verdict"])
    chosen = select(table, branch=args.branch, include_rerun=not args.no_rerun)
    with args.out.open("w") as handle:
        for row in chosen.itertuples(index=False):
            handle.write(
                json.dumps(
                    {
                        "stem": row.stem,
                        "enhanced": str(args.out_root / str(row.run_dir) / "run" / "streams" / "enhanced.flac"),
                        "declared_family": row.declared_family,
                        "verdict": row.verdict,
                        "selected_by": row.selected_by,
                    }
                )
                + "\n"
            )
    print(f"{len(chosen)} recordings -> {args.out}")
    print(chosen["selected_by"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
