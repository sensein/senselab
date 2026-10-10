#!/usr/bin/env python3
r"""Run COHORT over finished triage runs: corpus quantiles, then session enrollment and readings.

    uv run python scripts/extend_cohort.py collect MANIFEST --slice-index I --slice-count N [--log-dir DIR]
    uv run python scripts/extend_cohort.py quantiles (--facts DIR | --table TRIAGE_TSV) --out FILE [--commit SHA]
    uv run python scripts/extend_cohort.py apply MANIFEST --quantiles FILE --slice-index I --slice-count N \
        [--sessions KEY ...] [--readings-out DIR] [--enrollments-out DIR] [--force] [--config OVERRIDE.yaml] \
        [--commit SHA] [--log-dir DIR] [--device cpu|cuda]

``MANIFEST`` is the JSONL every extend driver takes (``stem``, ``enhanced``).

- ``collect`` reads each store's duration, declared family and task extent into
  ``slices/cohort-facts-slice-I-of-N.jsonl``; nothing is written into a store.
- ``quantiles`` takes the per-family quantiles over every collected row (or over a ``triage.tsv``,
  whose ``task_family``, ``recording_duration_s`` and ``task_duration_s`` columns are the same facts)
  and writes the run's cohort artefact, ``cohort_quantiles.json``, with its provenance.
- ``apply`` groups the manifest by BIDS session and shards the sessions, never a session's members.
  Per session it builds the participant's enrollment, compares every member's speech runs with it,
  checks every member against the quantiles, and writes one ``cohort_reading`` into each store.
  With ``--readings-out`` nothing is written into a store: each session's readings go to
  ``<DIR>/<session>.jsonl`` instead.

``apply`` is resumable: a session whose every store already carries a live reading under the same
cohort key (parameters, quantile digest, model, cut, members) is skipped unless ``--force``; with
``--readings-out`` a session whose file exists is skipped. A repeated write of an identical reading
mints nothing.

Run ``scripts/extend_refold.py`` afterwards: VERDICT reads the reading.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from senselab.audio.workflows.triage.cohort_fold import COHORT_READING
from senselab.audio.workflows.triage.cohort_stage import (
    QUANTILES_FILE,
    EnrollmentConfig,
    RecordingFacts,
    SessionEnrollment,
    build_quantile_artefact,
    cohort_key,
    cohort_parameters,
    ecapa_embedder,
    enrollment_config,
    facts_of,
    plain_loader,
    read_quantile_artefact,
    read_session,
    write_cohort_reading,
    write_enrollment_entity,
    write_quantile_artefact,
)
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.background import session_key
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement
from senselab.audio.workflows.triage.routing_analysis.families import task_family
from senselab.utils.prov_store import ProvStore

FACTS_PREFIX = "cohort-facts-slice-"


def build_parser() -> argparse.ArgumentParser:
    """The CLI: three subcommands, one per phase of the cohort pass.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Read each store's duration, family and task extent")
    collect.add_argument("manifest", type=Path)
    collect.add_argument("--slice-index", type=int, required=True)
    collect.add_argument("--slice-count", type=int, required=True)
    collect.add_argument("--log-dir", type=Path, default=None, help=f"Where {SLICES_SUBDIR}/ goes")

    quantiles = sub.add_parser("quantiles", help="Write the run's cohort quantile artefact")
    source = quantiles.add_mutually_exclusive_group(required=True)
    source.add_argument("--facts", type=Path, help=f"Directory holding {FACTS_PREFIX}*.jsonl")
    source.add_argument("--table", type=Path, help="A triage.tsv carrying the same facts")
    quantiles.add_argument("--out", type=Path, required=True, help=f"The artefact path, e.g. DIR/{QUANTILES_FILE}")
    quantiles.add_argument("--commit", default=None)

    apply = sub.add_parser("apply", help="Enroll each session and write every member's cohort reading")
    apply.add_argument("manifest", type=Path)
    apply.add_argument("--quantiles", type=Path, default=None, help="The artefact; omitted, every check is unavailable")
    apply.add_argument("--slice-index", type=int, required=True)
    apply.add_argument("--slice-count", type=int, required=True)
    apply.add_argument("--sessions", nargs="*", default=None, help="Only these sub-<label>_ses-<label> keys")
    apply.add_argument("--readings-out", type=Path, default=None, help="Write readings here, never into a store")
    apply.add_argument("--enrollments-out", type=Path, default=None, help="Write each session's vector here, 0600")
    apply.add_argument("--force", action="store_true", help="Re-read sessions already carrying this cohort key")
    apply.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    apply.add_argument("--commit", default=None)
    apply.add_argument("--log-dir", type=Path, default=None, help=f"Where {SLICES_SUBDIR}/ goes")
    apply.add_argument("--device", default=None, help="cpu or cuda; default lets senselab pick")
    return parser


# ------------------------------------------------------------------------------ collect


def collect_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fact row of every recording named, reading only its store.

    Args:
        rows: Manifest rows.

    Returns:
        One fact row per readable store; an unreadable one carries ``error``.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            store = read_store(run_root_of(Path(row["enhanced"])))
        except (OSError, ValueError) as error:
            out.append({"stem": row["stem"], "error": describe_exception(error)})
            continue
        out.append(facts_of(store, None, stem=str(row["stem"])).fact_row())
    return out


def run_collect(manifest: Path, *, slice_index: int, slice_count: int, log_dir: Path) -> Path:
    """Collect one shard's fact rows into its slice file.

    Args:
        manifest: The manifest.
        slice_index: This task's index.
        slice_count: How many tasks.
        log_dir: Where ``slices/`` goes.

    Returns:
        The slice file.
    """
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    found = collect_rows(mine)
    path = log_dir / SLICES_SUBDIR / f"{FACTS_PREFIX}{slice_index}-of-{slice_count}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in found), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------ quantiles


def facts_from_dir(directory: Path) -> list[dict[str, Any]]:
    """Every fact row the collect slices wrote.

    Args:
        directory: The log directory or its ``slices/``.

    Returns:
        The rows that carry a duration.
    """
    base = directory / SLICES_SUBDIR if (directory / SLICES_SUBDIR).is_dir() else directory
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob(f"{FACTS_PREFIX}*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("duration_s"):
                rows.append(record)
    return rows


def facts_from_table(path: Path) -> list[dict[str, Any]]:
    """Fact rows from a ``triage.tsv``.

    Args:
        path: The table.

    Returns:
        One row per recording with a duration, with ``verdict`` carried where the table has it.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            if not record.get("recording_duration_s"):
                continue
            duration = float(record["recording_duration_s"])
            extent = float(record["task_duration_s"]) if record.get("task_duration_s") else None
            stem = f"{record.get('participant_id')}_{record.get('session_id')}_task-{record.get('task')}"
            rows.append(
                {
                    "stem": stem,
                    "family": record.get("task_family") or task_family(str(record.get("task") or "")),
                    "duration_s": duration,
                    "extent_s": extent,
                    "extent_fraction": round((extent or 0.0) / duration, 4) if duration else None,
                    "verdict": record.get("verdict"),
                }
            )
    return rows


def _digest_file(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def run_quantiles(*, facts: Path | None, table: Path | None, out: Path, commit: str | None) -> dict[str, Any]:
    """Write the artefact and summarise what each check would do over the same rows.

    Args:
        facts: The collect log directory, or None.
        table: A ``triage.tsv``, or None.
        out: The artefact path.
        commit: The code revision.

    Returns:
        A summary: the artefact's digest, rows per family and each check's review count.
    """
    from senselab.audio.workflows.triage.cohort_stage import run_checks  # noqa: PLC0415

    if table is not None:
        rows = facts_from_table(table)
        source = {"kind": "triage_table", "path": str(table), "sha256": _digest_file(table)}
    else:
        assert facts is not None
        rows = facts_from_dir(facts)
        source = {"kind": "collected_facts", "path": str(facts)}
    artefact = build_quantile_artefact(
        rows, source=source, commit=commit, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    digest = write_quantile_artefact(artefact, out)
    reviewed: dict[str, dict[str, int]] = {}
    for row in rows:
        duration = float(row["duration_s"])
        extent = row.get("extent_s")
        facts_row = RecordingFacts(
            stem=str(row["stem"]),
            session=session_key(str(row["stem"])),
            family=str(row["family"]),
            lexical_task=False,
            duration_s=duration,
            task_extent=None if extent is None else (0.0, float(extent)),
        )
        for name, block in run_checks(facts_row, artefact).items():
            if block.get("outcome") == "review":
                counts = reviewed.setdefault(name, {})
                for key in ("total", f"{facts_row.family}", f"verdict:{row.get('verdict')}"):
                    counts[key] = counts.get(key, 0) + 1
    return {"artefact": str(out), "sha256": digest, "rows": len(rows), "reviewed": reviewed}


# ------------------------------------------------------------------------------ apply


def sessions_of(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """The manifest's rows grouped by BIDS session, sessions in sorted order.

    Args:
        rows: Every manifest row.

    Returns:
        ``(session, rows)`` pairs; a stem naming no session is its own group, keyed by the stem.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(session_key(str(row["stem"])) or str(row["stem"]), []).append(row)
    return sorted(groups.items())


def _write_enrollment(enrollment: SessionEnrollment, directory: Path) -> None:
    if enrollment.vector is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{enrollment.session}.npy"
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as handle:
        np.save(handle, np.asarray(enrollment.vector, dtype=np.float32))


def process_session(
    session: str,
    rows: Sequence[dict[str, Any]],
    *,
    settings: EnrollmentConfig,
    embed: Any,  # noqa: ANN401 -- an Embedder, or None where the settings are unusable
    artefact: dict[str, Any] | None,
    quantiles_ref: dict[str, Any] | None,
    readings_out: Path | None = None,
    enrollments_out: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """COHORT over one session's members, written into their stores or into ``readings_out``.

    Args:
        session: The session key.
        rows: Its members' manifest rows.
        settings: The enrollment settings.
        embed: The span embedder.
        artefact: The quantile artefact, or None.
        quantiles_ref: Its ``{path, sha256, version}``, or None.
        readings_out: Where readings go instead of the stores, or None.
        enrollments_out: Where the session vector goes, or None.
        force: Re-read a session already carrying this cohort key.

    Returns:
        One outcome record per member: counts and ids only.
    """
    started = time.time()
    if readings_out is not None and (readings_out / f"{session}.jsonl").exists() and not force:
        return [{"stem": row["stem"], "session": session, "status": SKIPPED} for row in rows]
    opened: list[tuple[dict[str, Any], Path, ProvStore]] = []
    failed: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            opened.append((row, run_root, read_store(run_root)))
        except (OSError, ValueError) as error:
            failed.append(
                {"stem": row["stem"], "session": session, "status": ERROR, "error": describe_exception(error)}
            )
    stream = str(cohort_parameters()["runs"]["diarization_stream"])
    members = [
        (facts_of(store, run_root / RUN_SUBDIR, diarization_stream=stream, stem=str(row["stem"])), run_root, store)
        for row, run_root, store in opened
    ]
    key = cohort_key(
        session=session,
        members=[facts for facts, _, _ in members],
        quantiles_sha256=None if quantiles_ref is None else str(quantiles_ref.get("sha256")),
        enrollment_config=settings.record(),
    )
    if readings_out is None and not force and members:
        held = [find_measurement(store, COHORT_READING) for _, _, store in members]
        if all(h is not None and h.attributes.get("cohort_key") == key for h in held):
            return [{"stem": f.stem, "session": session, "status": SKIPPED} for f, _, _ in members] + failed
    enrollment, readings = read_session(
        session,
        [(facts, plain_loader(store, run_root / RUN_SUBDIR)) for facts, run_root, store in members],
        settings=settings,
        embed=embed,
        artefact=artefact,
        quantiles_ref=quantiles_ref,
    )
    if enrollment is not None and enrollments_out is not None:
        _write_enrollment(enrollment, enrollments_out)
    out: list[dict[str, Any]] = []
    for (facts, run_root, store), attributes in zip(members, readings, strict=True):
        if readings_out is None:
            reading_id = write_cohort_reading(store, attributes)
            if enrollment is not None and enrollment.vector is not None:
                write_enrollment_entity(store, enrollment, reading_id)
            capture_environments(store, {})
            write_store(store, run_root)
            export_prov(store, run_root)
        block = attributes["other_speaker"]
        checks = attributes["checks"]
        out.append(
            {
                "stem": facts.stem,
                "session": session,
                "family": facts.family,
                "status": OK,
                "other_speaker": block.get("status"),
                "runs_n": block.get("runs_n"),
                "nonmatch_n": block.get("nonmatch_n"),
                "cosines": [[r["source"], r["duration_s"], r["cosine"]] for r in block.get("runs") or ()],
                "checks": {name: c.get("outcome") or c.get("status") for name, c in checks.items()},
                "attributes": attributes if readings_out is not None else None,
            }
        )
    elapsed = time.time() - started
    if readings_out is not None:
        readings_out.mkdir(parents=True, exist_ok=True)
        partial = readings_out / f"{session}.jsonl.partial"
        partial.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out), encoding="utf-8")
        partial.replace(readings_out / f"{session}.jsonl")
    summary = {
        "enrollment": None if enrollment is None else enrollment.record(),
        "members_n": len(rows),
        "elapsed_s": round(elapsed, 2),
    }
    for record in out:
        record.pop("attributes", None)
        record["session_summary"] = summary
    return out + failed


def run_apply(args: argparse.Namespace) -> dict[str, Any]:
    """Apply COHORT over one shard of sessions.

    Args:
        args: The parsed ``apply`` arguments.

    Returns:
        The task's summary.
    """
    started = time.time()
    config = load_triage_config(args.config) if args.config else load_triage_config()
    settings = enrollment_config(config)
    artefact, quantiles_ref = None, None
    if args.quantiles is not None:
        artefact, digest = read_quantile_artefact(args.quantiles)
        quantiles_ref = {"path": str(args.quantiles), "sha256": digest, "version": artefact.get("version")}
    embed = None
    if settings.usable:
        params = cohort_parameters()["enrollment"]
        device = None
        if args.device:
            from senselab.utils.data_structures import DeviceType  # noqa: PLC0415

            device = DeviceType(args.device)
        embed = ecapa_embedder(
            str(settings.model_id),
            str(settings.commit),
            window_s=float(params["window_s"]),
            hop_s=float(params["hop_s"]),
            device=device,
        )
    sessions = sessions_of(read_manifest(args.manifest, required=("stem", "enhanced")))
    if args.sessions:
        wanted = set(args.sessions)
        sessions = [(s, rows) for s, rows in sessions if s in wanted]
    mine = take_slice([{"session": s, "rows": rows} for s, rows in sessions], args.slice_index, args.slice_count)
    print(f"[slice {args.slice_index}/{args.slice_count}] {len(mine)} sessions", flush=True)
    log: list[dict[str, Any]] = []
    for group in mine:
        records = process_session(
            group["session"],
            group["rows"],
            settings=settings,
            embed=embed,
            artefact=artefact,
            quantiles_ref=quantiles_ref,
            readings_out=args.readings_out,
            enrollments_out=args.enrollments_out,
            force=args.force,
        )
        log.extend(records)
        head = records[0].get("session_summary") if records else None
        if head:
            enrolled = head["enrollment"] or {}
            print(
                f"  {group['session']}: {head['members_n']} members, enrollment {enrolled.get('status')} "
                f"({enrolled.get('recordings_n')} recordings, {enrolled.get('spans_n')} spans), {head['elapsed_s']} s",
                flush=True,
            )
    counts: dict[str, int] = {}
    for record in log:
        key = f"{record['status']}:{record.get('other_speaker')}"
        counts[key] = counts.get(key, 0) + 1
    log_dir = args.log_dir if args.log_dir is not None else args.manifest.parent
    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"cohort-slice-{args.slice_index}-of-{args.slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in log), encoding="utf-8")
    summary = {
        "manifest": str(args.manifest),
        "slice_index": args.slice_index,
        "slice_count": args.slice_count,
        "sessions": len(mine),
        "rows": len(log),
        "counts": counts,
        "nonmatch_recordings": sum(1 for r in log if (r.get("nonmatch_n") or 0) > 0),
        "quantiles": quantiles_ref,
        "config_hash": config.config_hash,
        "commit": args.commit,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run one phase of the cohort pass.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 on success, 1 when any member is ``error``, 2 on bad arguments.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "collect":
            if not args.manifest.exists():
                print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
                return 2
            path = run_collect(
                args.manifest,
                slice_index=args.slice_index,
                slice_count=args.slice_count,
                log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
            )
            print(f"Facts: {path}")
            return 0
        if args.command == "quantiles":
            summary = run_quantiles(facts=args.facts, table=args.table, out=args.out, commit=args.commit)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if not args.manifest.exists():
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            return 2
        summary = run_apply(args)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Sessions: {summary['sessions']}  Rows: {summary['rows']}  Elapsed: {summary['elapsed_s']:.1f} s")
    print(f"Log:      {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<32} {number}")
    return 1 if any(key.startswith(ERROR) for key in summary["counts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
