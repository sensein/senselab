#!/usr/bin/env python3
r"""Check that a corpus's declared identities hold together, before a run or over its corpus table.

    uv run python scripts/check_identity.py --recordings LIST [--hints DIR] [--session-scope SCOPE] \
        [--config OVERRIDE.yaml] [--profile DIR] [--out REPORT.json]
    uv run python scripts/check_identity.py --table RECORDING_VECTORS.parquet [--session-scope SCOPE] \
        [--config OVERRIDE.yaml] [--profile DIR] [--out REPORT.json]

``--recordings`` checks a corpus before it is run. ``LIST`` names one recording per line: its path, or a JSON
object carrying ``source`` (the field the extend drivers' manifests carry it in); blank lines and lines starting
with ``#`` are skipped. Each recording's identity is the one ADMIT would record,
:func:`~senselab.audio.workflows.triage.identity.identity_of` over its stem and the hint ``--hints`` builds.
``--hints DIR`` names a directory holding a ``hints.py`` exposing ``build_hint(wav)``, as the replay driver takes;
without it every identity is read off the stem. No store is opened.

``--table`` checks a finished corpus: the ``recording_vectors`` parquet, of which only ``stem``, ``run_dir``,
``participant``, ``session``, ``task`` and their ``*_from`` columns are read. A row whose ``*_from`` columns are
all empty is a run that records no identity.

The checks and their levels are in :mod:`senselab.audio.workflows.triage.identity_check`. ``--session-scope``
says where the corpus's session identifiers are unique: within their participant (the default, as BIDS ``ses-1``) or
across the corpus. ``--out`` writes the
full report as JSON, mode 600: it names participants and sessions. A summary of counts per check is printed.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from senselab.audio.workflows.triage.extend import load_hint_builder
from senselab.audio.workflows.triage.identity import (
    ROW_PARTICIPANT,
    ROW_SESSION,
    ROW_SOURCES,
    ROW_TASK,
    IdentityLabelError,
    identity_from_stem,
    identity_of,
    row_identity,
)
from senselab.audio.workflows.triage.identity_check import (
    CHECK_LEVELS,
    REFUSE,
    SESSION_SCOPE_PARTICIPANT,
    SESSION_SCOPES,
    IdentityRecord,
    IdentityReport,
    check_identities,
    triage_families,
)
from senselab.audio.workflows.triage.nodes.common import describe_exception

SOURCE_KEY = "source"
"""The manifest field a recording's path is carried in."""

TABLE_COLUMNS = ("stem", "run_dir", ROW_PARTICIPANT, ROW_SESSION, ROW_TASK, *ROW_SOURCES.values())
"""The corpus-table columns the check reads; no other column is read."""

PRIVATE_FILE_MODE = 0o600
SHOWN_PER_CHECK = 5
"""How many refused findings the summary names per check."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI: the corpus to check, before or after its run, and where the report goes.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recordings", type=Path, default=None, help="Recording list: one path or JSON row per line")
    source.add_argument("--table", type=Path, default=None, help="A recording_vectors parquet from a finished corpus")
    parser.add_argument("--hints", type=Path, default=None, help="Directory holding a hints.py exposing build_hint")
    parser.add_argument(
        "--session-scope",
        choices=SESSION_SCOPES,
        default=SESSION_SCOPE_PARTICIPANT,
        help="Where session identifiers are unique: within their participant (default) or across the corpus",
    )
    parser.add_argument("--out", type=Path, default=None, help="Where the JSON report goes")
    return parser


def read_recording_list(path: Path) -> list[Path]:
    """The recordings a list names.

    Args:
        path: One recording per line: a path, or a JSON object carrying ``source``.

    Returns:
        Each recording's path, in file order.

    Raises:
        ValueError: If a JSON line is not an object or carries no ``source``.
    """
    recordings: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("{"):
                payload = json.loads(text)
                if not isinstance(payload, dict) or not payload.get(SOURCE_KEY):
                    raise ValueError(f"{path}:{number} carries no {SOURCE_KEY}")
                text = str(payload[SOURCE_KEY])
            recordings.append(Path(text))
    return recordings


def records_before_run(recordings: list[Path], build_hint: Callable[[Path], Any] | None) -> list[IdentityRecord]:
    """Each recording's identity as ADMIT would declare it.

    Args:
        recordings: The recordings' paths.
        build_hint: The hint populator, or None to read every identity off the stem.

    Returns:
        One record per recording, located at its path; a hint that raises, or a label that cannot name a folder,
        is the record's ``error``.
    """
    records: list[IdentityRecord] = []
    for wav in recordings:
        stem = wav.stem
        parsed = identity_from_stem(stem)
        try:
            hint = build_hint(wav)[0] if build_hint is not None else None
            identity = identity_of(stem, hint)
        except IdentityLabelError as error:
            records.append(IdentityRecord(stem, str(wav), None, parsed, str(error)))
            continue
        except (OSError, ValueError, LookupError, TypeError) as error:
            records.append(IdentityRecord(stem, str(wav), None, parsed, f"hint: {describe_exception(error)}"))
            continue
        records.append(IdentityRecord(stem, str(wav), identity, parsed))
    return records


def records_after_run(table: Path) -> list[IdentityRecord]:
    """Each finished run's recorded identity, from the corpus table.

    Args:
        table: The ``recording_vectors`` parquet.

    Returns:
        One record per row, located at its ``run_dir``; a row recording no identity carries None.

    Raises:
        ValueError: If the table lacks one of :data:`TABLE_COLUMNS`.
    """
    present = set(pq.read_schema(table).names)
    missing = [column for column in TABLE_COLUMNS if column not in present]
    if missing:
        raise ValueError(f"{table} carries no {', '.join(missing)}; rebuild it from a current corpus pass")
    rows = pq.read_table(table, columns=list(TABLE_COLUMNS)).to_pylist()
    records: list[IdentityRecord] = []
    for row in rows:
        stem = str(row.get("stem") or "")
        records.append(IdentityRecord(stem, str(row.get("run_dir") or ""), row_identity(row), identity_from_stem(stem)))
    return records


def write_report(report: IdentityReport, path: Path, extra: dict[str, Any]) -> None:
    """Write the report as JSON, readable by its owner only.

    Args:
        report: The report.
        path: Where it goes.
        extra: Fields naming the input, merged in front of the report's own.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
    os.fchmod(descriptor, PRIVATE_FILE_MODE)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({**extra, **report.as_dict()}, indent=1) + "\n")


def print_summary(report: IdentityReport) -> None:
    """Print the counts per check, and the first few refused findings.

    Args:
        report: The report.
    """
    print(f"Recordings:    {report.recordings}")
    print(f"Session scope: {report.session_scope}")
    involved = report.recordings_involved
    for check, number in report.counts.items():
        print(f"  {CHECK_LEVELS[check]:<7} {check:<31} {number:>7} findings  {involved[check]:>7} recordings")
    for check in CHECK_LEVELS:
        refused = [finding for finding in report.findings if finding.check == check and finding.level == REFUSE]
        for finding in refused[:SHOWN_PER_CHECK]:
            print(f"REFUSED {check} {finding.key}: {', '.join(finding.locations)}", file=sys.stderr)
        if len(refused) > SHOWN_PER_CHECK:
            print(f"REFUSED {check}: {len(refused) - SHOWN_PER_CHECK} more in the report", file=sys.stderr)
    print("Refused" if report.refused else "Passed")


def main(argv: list[str] | None = None) -> int:
    """Check the corpus and print what was found.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when no finding is refused, 1 when any is, and 2 when the arguments or the input could not be read and
        nothing was checked.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hints is not None and args.table is not None:
        parser.error("--hints applies to --recordings only; a finished run's identity is the one it recorded")
    given = args.recordings if args.recordings is not None else args.table
    if not given.exists():
        print(f"ERROR: not found: {given}", file=sys.stderr)
        return 2
    try:
        scope = args.session_scope
        if args.recordings is not None:
            build_hint = load_hint_builder(args.hints) if args.hints is not None else None
            records = records_before_run(read_recording_list(args.recordings), build_hint)
        else:
            records = records_after_run(args.table)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    report = check_identities(records, session_scope=scope, known_families=triage_families())
    if args.out is not None:
        extra = {
            "mode": "recordings" if args.recordings is not None else "table",
            "input": str(given),
            "hints": str(args.hints) if args.hints is not None else None,
        }
        write_report(report, args.out, extra)
        print(f"Report:        {args.out}")
    print_summary(report)
    return 1 if report.refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
