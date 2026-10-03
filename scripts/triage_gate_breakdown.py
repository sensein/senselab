"""Break a recording-vectors parquet down by task and by gate.

    uv run python scripts/triage_gate_breakdown.py recording_vectors.parquet --out DIR

Writes ``by_task.md`` and ``by_gate.md``. Both are counts, categories and quantiles of numeric
readings only: no transcript text, no stem, no PII extent or category is read or emitted, so
unlike the parquet they are shareable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from senselab.audio.workflows.triage.recording_vectors import GATE_NAMES

TASK_COLUMNS = (
    "task",
    "declared_family",
    "verdict",
    "gate_group",
    "gate_applied_n",
    "gate_failed_n",
    "gate_undetermined_n",
    "gate_flagging_n",
)
OUTCOMES = ("true", "false", "undetermined")


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    """The five-number summary of a numeric reading.

    Args:
        values: Finite readings, in any order.

    Returns:
        min, p05, p50, p95 and max, each None when nothing was read.
    """
    if not values:
        return dict.fromkeys(("min", "p05", "p50", "p95", "max"))
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"min": ordered[0], "p05": at(0.05), "p50": at(0.5), "p95": at(0.95), "max": ordered[-1]}


def fmt(value: Any) -> str:  # noqa: ANN401 -- a cell is any type
    """One table cell.

    Args:
        value: What goes in it.

    Returns:
        The rendered cell; an em dash for None.
    """
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A markdown table.

    Args:
        header: The column names.
        rows: One sequence of cells per row.

    Returns:
        The rendered table.
    """
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def by_task(columns: dict[str, list[Any]], total: int) -> str:
    """How every declared task fared under its gates.

    Args:
        columns: The parquet columns this breakdown reads.
        total: How many recordings the file holds.

    Returns:
        The markdown page.
    """
    tasks: dict[str, dict[str, Any]] = {}
    for index in range(total):
        key = columns["declared_family"][index] or columns["task"][index] or "(none declared)"
        slot = tasks.setdefault(
            str(key),
            {
                "n": 0,
                "group": set(),
                "pass": 0,
                "flag": 0,
                "discard": 0,
                "applied": 0,
                "failed": 0,
                "undetermined": 0,
                "refused": 0,
                "flagged": 0,
            },
        )
        slot["n"] += 1
        if columns["gate_group"][index]:
            slot["group"].add(str(columns["gate_group"][index]))
        verdict = str(columns["verdict"][index] or "")
        if verdict in slot:
            slot[verdict] += 1
        applied = columns["gate_applied_n"][index] or 0
        failed = columns["gate_failed_n"][index] or 0
        slot["applied"] += applied
        slot["failed"] += failed
        slot["undetermined"] += columns["gate_undetermined_n"][index] or 0
        slot["flagged"] += columns["gate_flagging_n"][index] or 0
        if failed:
            slot["refused"] += 1

    rows = []
    for name, slot in sorted(tasks.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        rows.append(
            [
                name,
                "/".join(sorted(slot["group"])) or "—",
                slot["n"],
                slot["pass"],
                slot["flag"],
                slot["discard"],
                slot["applied"],
                slot["failed"],
                slot["undetermined"],
                slot["refused"],
                round(100.0 * slot["refused"] / slot["n"], 1) if slot["n"] else None,
            ]
        )
    header = [
        "declared family",
        "group",
        "n",
        "pass",
        "flag",
        "discard",
        "gates applied",
        "gates failed",
        "undetermined",
        "recordings refused",
        "% refused",
    ]
    return (
        "# Gates by task\n\n"
        f"One row per declared family over {total:,} recordings. *gates applied* counts gate "
        "applications, not recordings; *recordings refused* counts recordings on which at least "
        "one gate read false. A family whose group configures no gates shows zero throughout — "
        "that is the configuration, not a failure to measure.\n\n" + table(header, rows) + "\n"
    )


def by_gate(columns: dict[str, list[Any]], total: int) -> str:
    """How every gate fared, and what its reading looked like.

    Args:
        columns: The parquet columns this breakdown reads.
        total: How many recordings the file holds.

    Returns:
        The markdown page.
    """
    rows = []
    for gate in GATE_NAMES:
        outcomes = columns[f"gate_{gate}_passed"]
        readings = columns[f"gate_{gate}"]
        bounds = columns[f"gate_{gate}_bound"]
        counts = {name: 0 for name in OUTCOMES}
        values: list[float] = []
        seen_bounds: set[float] = set()
        applied = 0
        zeros = 0
        for index in range(total):
            outcome = outcomes[index]
            if outcome is None:
                continue
            applied += 1
            counts[str(outcome)] = counts.get(str(outcome), 0) + 1
            reading = readings[index]
            if reading is not None:
                values.append(float(reading))
                if float(reading) == 0.0:
                    zeros += 1
            if bounds[index] is not None:
                seen_bounds.add(float(bounds[index]))
        if not applied:
            rows.append([gate, 0, None, "—", 0, 0, 0, 0, None, None, None])
            continue
        q = quantiles(values)
        rows.append(
            [
                gate,
                applied,
                round(100.0 * applied / total, 1),
                "/".join(fmt(b) for b in sorted(seen_bounds)) or "—",
                counts["true"],
                counts["false"],
                counts["undetermined"],
                zeros,
                q["p05"],
                q["p50"],
                q["p95"],
            ]
        )
    header = [
        "gate",
        "applied",
        "% of file",
        "bound(s)",
        "passed",
        "failed",
        "undetermined",
        "readings of exactly 0",
        "reading p05",
        "p50",
        "p95",
    ]
    return (
        "# Gates by gate\n\n"
        f"One row per gate in the registry over {total:,} recordings. A gate is *applied* when the "
        "fold resolved a bound for this recording's task group; *undetermined* means the reading or "
        "the bound was absent, never that the gate failed. The zero column is there because a "
        "reading of exactly 0 is a measured value and must not be confused with an absent one.\n\n"
        + table(header, rows)
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write both breakdowns.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when the file held rows, 1 when it held none.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="A merged recording_vectors.parquet.")
    parser.add_argument("--out", type=Path, required=True, help="Where the two markdown files go.")
    args = parser.parse_args(argv)

    wanted = list(TASK_COLUMNS)
    for gate in GATE_NAMES:
        wanted += [f"gate_{gate}", f"gate_{gate}_bound", f"gate_{gate}_passed"]
    table_in = pq.read_table(args.parquet, columns=wanted)
    total = table_in.num_rows
    if not total:
        print(f"{args.parquet} holds no rows")
        return 1
    columns = {name: table_in.column(name).to_pylist() for name in wanted}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "by_task.md").write_text(by_task(columns, total))
    (args.out / "by_gate.md").write_text(by_gate(columns, total))
    print(f"{total:,} recordings -> {args.out / 'by_task.md'}, {args.out / 'by_gate.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
