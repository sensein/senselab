#!/usr/bin/env python3
r"""Compare the per-span SQUIM columns of one ``triage.tsv`` with the task-extent ones of another.

    uv run python scripts/verify_task_squim.py --old OLD/triage.tsv --new NEW/triage.tsv \
        --logs ROWS_DIR [--logs ROWS_DIR ...] --out verify.md

``--old`` is the table whose ``squim_*`` are the per-span medians (``triage.tsv`` schema 3);
``--new`` the one whose ``squim_*`` are the task-extent medians (schema 4). Rows are joined on
participant, session and task, inside this process only. ``--logs`` names directories holding the
``slices/task-squim-slice-*.jsonl`` logs of ``scripts/extend_task_squim.py``, read for the window
counts and seconds covered.

The report is aggregates only: no participant, session, stem, path or transcript is printed, and a
group under ``--min-group`` rows is not reported on its own.

See ``specs/20261027-task-extent-squim/design.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from senselab.audio.workflows.triage.recording_vectors import stem_of
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
from senselab.audio.workflows.triage.task_squim import branch_of_family

HEADS = ("squim_pesq", "squim_stoi", "squim_si_sdr")
LABEL = {"squim_pesq": "PESQ", "squim_stoi": "STOI", "squim_si_sdr": "SI-SDR (dB)"}
KEY = ("participant_id", "session_id", "task")
SNR_COLUMNS = ("snr_db", "speech_over_residual_db")
REPORTER = {"squim_pesq": 2.9, "squim_stoi": 0.98, "squim_si_sdr": 17.0}
"""The #614 reporter's 10 s window medians on clean read speech (two samples, 50 and 60 recordings)."""
READ_SPEECH_MARKERS = ("passage", "sentences")
"""A SPEECH family whose name carries one of these is read speech."""
csv.field_size_limit(sys.maxsize)


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a ``triage.tsv``.

    Args:
        path: The table.

    Returns:
        Its rows.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def number(value: str | None) -> float | None:
    """A cell as a float, or None where it is empty or not finite.

    Args:
        value: The cell.

    Returns:
        The number, or None.
    """
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if np.isfinite(out) else None


def read_logs(directories: Sequence[Path]) -> list[dict[str, Any]]:
    """The last record per stem across the driver's slice logs.

    Args:
        directories: Log directories, each holding ``slices/task-squim-slice-*.jsonl``.

    Returns:
        One record per stem, the latest seen.
    """
    by_stem: dict[str, dict[str, Any]] = {}
    for directory in directories:
        for path in sorted((directory / "slices").glob("task-squim-slice-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("stem") and record.get("status"):
                    by_stem[str(record["stem"])] = record
    return list(by_stem.values())


def describe(values: Iterable[float | None]) -> tuple[int, str, str]:
    """``n``, the median and the IQR of the values present.

    Args:
        values: The values; None is skipped.

    Returns:
        ``(n, median, "q1-q3")``, formatted.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return 0, "-", "-"
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return int(arr.size), f"{med:.3g}", f"{q1:.3g} to {q3:.3g}"


def spearman(pairs: Sequence[tuple[float | None, float | None]]) -> tuple[int, str]:
    """Spearman's rho over the pairs where both are present.

    Args:
        pairs: ``(x, y)`` pairs.

    Returns:
        ``(n, rho)``, rho formatted, ``-`` under 10 pairs.
    """
    kept = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(kept) < 10:
        return len(kept), "-"
    rho = spearmanr([x for x, _ in kept], [y for _, y in kept]).statistic
    return len(kept), f"{rho:+.3f}"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """A markdown table.

    Args:
        header: The column titles.
        rows: The cells.

    Returns:
        Its lines.
    """
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def by_group(
    joined: Sequence[Mapping[str, Any]], group: Callable[[Mapping[str, Any]], str | None], min_group: int
) -> list[list[Any]]:
    """Old and new median and IQR per head, per group.

    Args:
        joined: The joined rows.
        group: The group a row falls in, or None to leave it out.
        min_group: The fewest rows a group is reported with.

    Returns:
        One table row per group and head.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined:
        name = group(row)
        if name is not None:
            groups[name].append(row)
    out: list[list[Any]] = []
    small = 0
    for name in sorted(groups):
        members = groups[name]
        if len(members) < min_group:
            small += len(members)
            continue
        for head in HEADS:
            n_old, med_old, iqr_old = describe(r["old"][head] for r in members)
            n_new, med_new, iqr_new = describe(r["new"][head] for r in members)
            out.append([name, LABEL[head], n_old, med_old, iqr_old, n_new, med_new, iqr_new])
    if small:
        out.append([f"(groups under {min_group} rows, pooled out: {small} rows)", "", "", "", "", "", "", ""])
    return out


def deciles(joined: Sequence[Mapping[str, Any]], column: str) -> list[list[Any]]:
    """Median old and new value per decile of one SNR column.

    Args:
        joined: The joined rows.
        column: ``snr_db`` or ``speech_over_residual_db``.

    Returns:
        One table row per decile.
    """
    present = [r for r in joined if r[column] is not None]
    if len(present) < 10:
        return []
    edges = np.percentile([r[column] for r in present], np.linspace(0, 100, 11))
    out: list[list[Any]] = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        members = [r for r in present if (lo <= r[column] <= hi if i == 9 else lo <= r[column] < hi)]
        cells: list[Any] = [i + 1, f"{lo:.1f} to {hi:.1f}", len(members)]
        for head in HEADS:
            cells.append(describe(r["old"][head] for r in members)[1])
            cells.append(describe(r["new"][head] for r in members)[1])
        out.append(cells)
    return out


def absent_kind(reason: str) -> str:
    """The kind of reason a recording got no value, without its numbers.

    Args:
        reason: The driver's ``reason``.

    Returns:
        ``airway family``, ``no task extent``, ``no plain stream``, ``under floor`` or ``other``.
    """
    if "AIRWAY" in reason:
        return "airway family"
    if "floor" in reason:
        return "under floor"
    return reason if reason in ("no task extent", "no plain stream") else "other"


def join(old_rows: Sequence[Mapping[str, str]], new_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """The new table's rows, each with its old row's SQUIM values.

    Args:
        old_rows: The per-span table.
        new_rows: The task-extent table.

    Returns:
        One record per new row: branch, family, verdict, reason, the SNR columns, task bounds, and
        the old and new SQUIM values. The join key is not carried.
    """
    old = {tuple(r.get(k, "") for k in KEY): r for r in old_rows}
    out: list[dict[str, Any]] = []
    for row in new_rows:
        before = old.get(tuple(row.get(k, "") for k in KEY), {})
        out.append(
            {
                "branch": row.get("branch") or "none",
                "family": row.get("task_family") or "",
                "verdict": row.get("verdict") or "",
                "reason": row.get("reason") or "",
                "reasons": [r for r in (row.get("reasons") or "").split(";") if r],
                "matched": bool(before),
                "task_start_s": number(row.get("task_start_s")),
                "task_duration_s": number(row.get("task_duration_s")),
                **{c: number(row.get(c)) for c in SNR_COLUMNS},
                "old": {h: number(before.get(h)) for h in HEADS},
                "new": {h: number(row.get(h)) for h in HEADS},
            }
        )
    return out


def report(joined: Sequence[Mapping[str, Any]], logs: Sequence[Mapping[str, Any]], min_group: int) -> list[str]:
    """The verification report.

    Args:
        joined: :func:`join`'s rows.
        logs: :func:`read_logs`'s records.
        min_group: The fewest rows a group is reported with.

    Returns:
        The markdown lines.
    """
    lines = ["# Task-extent SQUIM: old (per-span) against new (task-extent windows)", ""]
    lines.append(
        f"{len(joined)} rows in the new table, {sum(r['matched'] for r in joined)} matched in the old. "
        "Aggregates only; groups under "
        f"{min_group} rows are pooled out."
    )
    lines.append("")

    head = ["group", "head", "old n", "old median", "old IQR", "new n", "new median", "new IQR"]
    lines += ["## Old against new", "", "### Overall", ""]
    lines += table(head, by_group(joined, lambda r: "all", 1))
    lines += ["### By branch", ""]
    lines += table(head, by_group(joined, lambda r: str(r["branch"]), min_group))
    lines += ["### By family", ""]
    lines += table(head, by_group(joined, lambda r: str(r["family"]), min_group))

    lines += ["## Against SNR", "", "Spearman's rho; the new values should rise with SNR and the old fall.", ""]
    rho_rows: list[list[Any]] = []
    for column in SNR_COLUMNS:
        for h in HEADS:
            n_old, rho_old = spearman([(r[column], r["old"][h]) for r in joined])
            n_new, rho_new = spearman([(r[column], r["new"][h]) for r in joined])
            rho_rows.append([column, LABEL[h], n_old, rho_old, n_new, rho_new])
    lines += table(["SNR column", "head", "old n", "old rho", "new n", "new rho"], rho_rows)
    for branch in ("SPEECH", "VOICE"):
        members = [r for r in joined if r["branch"] == branch]
        rows: list[list[Any]] = []
        for h in HEADS:
            rows.append(
                [
                    LABEL[h],
                    spearman([(r["snr_db"], r["old"][h]) for r in members])[1],
                    spearman([(r["snr_db"], r["new"][h]) for r in members])[1],
                ]
            )
        lines += [f"{branch} only, against snr_db:", ""]
        lines += table(["head", "old rho", "new rho"], rows)
    for column in SNR_COLUMNS:
        decile_rows = deciles([r for r in joined if r["branch"] != "AIRWAY"], column)
        if decile_rows:
            lines += [f"### Median by {column} decile (non-airway rows)", ""]
            lines += table(
                ["decile", "range", "n", *[f"{LABEL[h]} {w}" for h in HEADS for w in ("old", "new")]], decile_rows
            )

    lines += ["## Against verdict and reason", ""]
    lines += ["### By verdict", ""]
    lines += table(head, by_group([r for r in joined if r["branch"] != "AIRWAY"], lambda r: r["verdict"], min_group))
    lines += ["### By primary reason (non-airway rows)", ""]
    lines += table(
        head, by_group([r for r in joined if r["branch"] != "AIRWAY"], lambda r: r["reason"] or "(none)", min_group)
    )
    lines += ["### Weak-event and low-SNR reasons against the rest (non-airway rows)", ""]

    def flag(r: Mapping[str, Any]) -> str:
        keys = " ".join(r["reasons"]).lower()
        if "weak" in keys:
            return "a weak-event reason"
        if "snr" in keys or "noise" in keys:
            return "a low-SNR or noise reason"
        return "neither"

    lines += table(head, by_group([r for r in joined if r["branch"] != "AIRWAY"], flag, min_group))

    lines += ["## Against the reporter's numbers", ""]
    snr = [r["snr_db"] for r in joined if r["snr_db"] is not None]
    median_snr = float(np.median(snr)) if snr else None
    clean = [
        r
        for r in joined
        if r["branch"] == "SPEECH"
        and r["verdict"] == "pass"
        and any(m in r["family"] for m in READ_SPEECH_MARKERS)
        and median_snr is not None
        and r["snr_db"] is not None
        and r["snr_db"] >= median_snr
    ]
    lines.append(
        "Clean read speech: SPEECH branch, verdict pass, a family named *passage* or *sentences*, snr_db at or "
        f"over the corpus median ({median_snr:.1f} dB). #614 reported PESQ 2.9, STOI 0.98, SI-SDR 17 dB "
        "on 10 s windows."
        if median_snr is not None
        else "No snr_db present."
    )
    lines.append("")
    rows = []
    for h in HEADS:
        n_old, med_old, iqr_old = describe(r["old"][h] for r in clean)
        n_new, med_new, iqr_new = describe(r["new"][h] for r in clean)
        rows.append([LABEL[h], REPORTER[h], n_old, med_old, iqr_old, n_new, med_new, iqr_new])
    lines += table(["head", "#614", "old n", "old median", "old IQR", "new n", "new median", "new IQR"], rows)

    lines += ["## Coverage", "", "### Rows with a value, by branch", ""]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in joined:
        counts[r["branch"]]["rows"] += 1
        counts[r["branch"]]["old"] += r["old"]["squim_pesq"] is not None
        counts[r["branch"]]["new"] += r["new"]["squim_pesq"] is not None
    lines += table(
        ["branch", "rows", "old with value", "new with value"],
        [[b, c["rows"], c["old"], c["new"]] for b, c in sorted(counts.items())],
    )
    if logs:
        lines += ["### Driver outcomes, by branch", ""]
        outcome: dict[str, Counter[str]] = defaultdict(Counter)
        for record in logs:
            branch = branch_of_family(task_family(task_id_of(stem_of(Path(str(record["stem"])))))) or "none"
            status = str(record["status"])
            if status == "absent":
                status = "absent: " + absent_kind(str(record.get("reason") or ""))
            outcome[branch][status] += 1
        lines += table(
            ["branch", "status", "rows"],
            [[b, s, n] for b, c in sorted(outcome.items()) for s, n in sorted(c.items())],
        )
        measured = [r for r in logs if r.get("windows_n")]
        windows = Counter(int(r["windows_n"]) for r in measured)
        lines += ["### Windows per measured recording", ""]
        top = sorted(windows.items())
        shown = [[k, v] for k, v in top if k <= 10] + (
            [[">10", sum(v for k, v in top if k > 10)]] if any(k > 10 for k, _ in top) else []
        )
        lines += table(["windows", "recordings"], shown)
        ratio = [
            float(r["covered_s"]) / float(r["extent_s"])
            for r in measured
            if r.get("covered_s") is not None and r.get("extent_s")
        ]
        n, med, iqr = describe(ratio)
        extent_n, extent_med, extent_iqr = describe(float(r["extent_s"]) for r in measured if r.get("extent_s"))
        lines += table(
            ["quantity", "n", "median", "IQR"],
            [
                ["seconds covered / task extent (padded) seconds", n, med, iqr],
                ["task extent (padded) seconds", extent_n, extent_med, extent_iqr],
            ],
        )
        source = Counter(str(r.get("extent_source")) for r in measured)
        lines += table(["extent source", "recordings"], sorted(source.items()))
        n, med, iqr = describe(float(r["elapsed_s"]) for r in logs if r.get("elapsed_s") is not None)
        lines += table(["driver seconds per recording", "n", "median", "IQR"], [["", n, med, iqr]])

    lines += ["## Sanity", ""]
    airway_valued = sum(1 for r in joined if r["branch"] == "AIRWAY" and r["new"]["squim_pesq"] is not None)
    no_extent_valued = sum(1 for r in joined if r["task_start_s"] is None and r["new"]["squim_pesq"] is not None)
    lines += table(
        ["check", "rows violating", "expected"],
        [
            ["airway row carries a new value", airway_valued, 0],
            ["row with no task extent carries a new value", no_extent_valued, 0],
        ],
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Write the report.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0 when the sanity checks hold, 1 when either is violated.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("--old", type=Path, required=True, help="triage.tsv with the per-span squim_*")
    parser.add_argument("--new", type=Path, required=True, help="triage.tsv with the task-extent squim_*")
    parser.add_argument("--logs", type=Path, action="append", default=[], help="extend_task_squim log directory")
    parser.add_argument("--out", type=Path, required=True, help="The markdown report")
    parser.add_argument("--min-group", type=int, default=20, help="Fewest rows a group is reported with")
    args = parser.parse_args(argv)
    joined = join(read_table(args.old), read_table(args.new))
    lines = report(joined, read_logs(args.logs), args.min_group)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    violated = any(
        (r["branch"] == "AIRWAY" or r["task_start_s"] is None) and r["new"]["squim_pesq"] is not None for r in joined
    )
    print(f"wrote {args.out}; sanity {'VIOLATED' if violated else 'ok'}")
    return 1 if violated else 0


if __name__ == "__main__":
    raise SystemExit(main())
