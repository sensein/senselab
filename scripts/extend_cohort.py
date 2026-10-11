#!/usr/bin/env python3
r"""Run COHORT over finished triage runs: corpus quantiles, then speaker enrollment and readings.

    uv run python scripts/extend_cohort.py collect MANIFEST --slice-index I --slice-count N [--log-dir DIR]
    uv run python scripts/extend_cohort.py quantiles (--facts DIR | --table TRIAGE_TSV) --out FILE [--commit SHA]
    uv run python scripts/extend_cohort.py apply MANIFEST --quantiles FILE --slice-index I --slice-count N \
        [--participants SUB ...] [--readings-out DIR] [--enrollments-out DIR] [--force] [--config OVERRIDE.yaml] \
        [--commit SHA] [--log-dir DIR] [--device cpu|cuda]
    uv run python scripts/extend_cohort.py consistency LOG_DIR [--iqr-multiple K]

``MANIFEST`` is the JSONL every extend driver takes (``stem``, ``enhanced``).

- ``collect`` reads each store's duration, declared family and task extent into
  ``slices/cohort-facts-slice-I-of-N.jsonl``; nothing is written into a store.
- ``quantiles`` takes the per-family quantiles over every collected row (or over a ``triage.tsv``,
  whose ``task_family``, ``recording_duration_s`` and ``task_duration_s`` columns are the same facts)
  and writes the run's cohort artefact, ``cohort_quantiles.json``, with its provenance.
- ``apply`` groups the manifest by participant and shards the participants, never a participant's
  sessions. Per participant it builds each session's enrollment and, per session, the participant
  enrollment of the other sessions; compares every member's speech runs with its session's (or, where
  the session has none, the other sessions'); checks each session against the other sessions and
  every member against the quantiles; and writes one ``cohort_reading`` into each store. With
  ``--readings-out`` nothing is written into a store: each participant's readings go to
  ``<DIR>/<participant>.jsonl`` instead.
- ``consistency`` reads the apply logs' session-vs-other-sessions cosines and prints their
  distribution and lower fence, from which ``participant.consistency_cosine_min`` is set.

``apply`` is resumable: a participant whose every store already carries a live reading under the
same cohort key (parameters, quantile digest, model, cut, every member of every session) is skipped
unless ``--force``; with ``--readings-out`` a participant whose file exists is skipped. A repeated
write of an identical reading mints nothing.

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

from senselab.audio.workflows.triage.cohort_fold import COHORT_READING, POOLS
from senselab.audio.workflows.triage.cohort_stage import (
    CONSISTENCY_CHECK,
    QUANTILES_FILE,
    EnrollmentConfig,
    RecordingFacts,
    build_quantile_artefact,
    cohort_key,
    cohort_parameters,
    consistency_bound,
    ecapa_embedder,
    enrollment_config,
    facts_of,
    participant_key,
    plain_loader,
    read_participant,
    read_quantile_artefact,
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

    apply = sub.add_parser("apply", help="Enroll each participant and session and write every member's reading")
    apply.add_argument("manifest", type=Path)
    apply.add_argument("--quantiles", type=Path, default=None, help="The artefact; omitted, every check is unavailable")
    apply.add_argument("--slice-index", type=int, required=True)
    apply.add_argument("--slice-count", type=int, required=True)
    apply.add_argument("--participants", nargs="*", default=None, help="Only these sub-<label> participants")
    apply.add_argument("--readings-out", type=Path, default=None, help="Write readings here, never into a store")
    apply.add_argument(
        "--enrollments-out", type=Path, default=None, help="Write each session's and participant's vector here, 0600"
    )
    apply.add_argument("--force", action="store_true", help="Re-read participants already carrying this cohort key")
    apply.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    apply.add_argument("--commit", default=None)
    apply.add_argument("--log-dir", type=Path, default=None, help=f"Where {SLICES_SUBDIR}/ goes")
    apply.add_argument("--device", default=None, help="cpu or cuda; default lets senselab pick")

    consistency = sub.add_parser("consistency", help="The session-vs-other-sessions cosines and their lower fence")
    consistency.add_argument("logs", type=Path, help="The apply --log-dir, its slices/, or a --readings-out directory")
    consistency.add_argument("--iqr-multiple", type=float, default=1.5)
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


def participants_of(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[tuple[str, list[dict[str, Any]]]]]]:
    """The manifest's rows grouped by participant, then by BIDS session, both in sorted order.

    Args:
        rows: Every manifest row.

    Returns:
        ``(participant, [(session, rows), ...])`` pairs; a stem naming no session is its own session,
        keyed by the stem, under the participant its first underscore names.
    """
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        session = session_key(str(row["stem"])) or str(row["stem"])
        groups.setdefault(participant_key(session), {}).setdefault(session, []).append(row)
    return [(participant, sorted(sessions.items())) for participant, sessions in sorted(groups.items())]


def _write_vector(vector: np.ndarray | None, path: Path) -> None:
    if vector is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as handle:
        np.save(handle, np.asarray(vector, dtype=np.float32))


def _consistency_cosine(block: dict[str, Any]) -> float | None:
    for comparison in block.get("comparisons") or ():
        if comparison.get("name") == "session_vs_other_sessions_cosine":
            return comparison.get("value")
    return None


def process_participant(
    participant: str,
    sessions: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    *,
    settings: EnrollmentConfig,
    embed: Any,  # noqa: ANN401 -- an Embedder, or None where the settings are unusable
    artefact: dict[str, Any] | None,
    quantiles_ref: dict[str, Any] | None,
    readings_out: Path | None = None,
    enrollments_out: Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """COHORT over every session of one participant, written into the stores or into ``readings_out``.

    Args:
        participant: The participant label.
        sessions: ``(session, manifest rows)`` per session.
        settings: The enrollment settings.
        embed: The span embedder.
        artefact: The quantile artefact, or None.
        quantiles_ref: Its ``{path, sha256, version}``, or None.
        readings_out: Where readings go instead of the stores, or None.
        enrollments_out: Where the session and participant vectors go, or None.
        force: Re-read a participant already carrying this cohort key.

    Returns:
        One outcome record per member: ids, counts and cosines only.
    """
    started = time.time()
    rows_n = sum(len(rows) for _, rows in sessions)
    if readings_out is not None and (readings_out / f"{participant}.jsonl").exists() and not force:
        return [
            {"stem": row["stem"], "participant": participant, "session": session, "status": SKIPPED}
            for session, rows in sessions
            for row in rows
        ]
    failed: list[dict[str, Any]] = []
    stream = str(cohort_parameters()["comparison"]["diarization_stream"])
    opened: list[tuple[str, list[tuple[RecordingFacts, Path, ProvStore]]]] = []
    for session, rows in sessions:
        members: list[tuple[RecordingFacts, Path, ProvStore]] = []
        for row in rows:
            try:
                run_root = run_root_of(Path(row["enhanced"]))
                store = read_store(run_root)
            except (OSError, ValueError) as error:
                failed.append(
                    {
                        "stem": row["stem"],
                        "participant": participant,
                        "session": session,
                        "status": ERROR,
                        "error": describe_exception(error),
                    }
                )
                continue
            facts = facts_of(store, run_root / RUN_SUBDIR, diarization_stream=stream, stem=str(row["stem"]))
            members.append((facts, run_root, store))
        if members:
            opened.append((session, members))
    every = [member for _, members in opened for member in members]
    key = cohort_key(
        group=participant,
        members=[facts for facts, _, _ in every],
        quantiles_sha256=None if quantiles_ref is None else str(quantiles_ref.get("sha256")),
        enrollment_config=settings.record(),
    )
    if readings_out is None and not force and every:
        held = [find_measurement(store, COHORT_READING) for _, _, store in every]
        if all(h is not None and h.attributes.get("cohort_key") == key for h in held):
            return [
                {"stem": facts.stem, "participant": participant, "session": session, "status": SKIPPED}
                for session, members in opened
                for facts, _, _ in members
            ] + failed
    enrollments, readings = read_participant(
        participant,
        [
            (session, [(facts, plain_loader(store, run_root / RUN_SUBDIR)) for facts, run_root, store in members])
            for session, members in opened
        ],
        settings=settings,
        embed=embed,
        artefact=artefact,
        quantiles_ref=quantiles_ref,
    )
    if enrollments is not None and enrollments_out is not None:
        for session, enrollment in enrollments.sessions.items():
            _write_vector(enrollment.vector, enrollments_out / f"{session}.npy")
        _write_vector(enrollments.pooled.vector, enrollments_out / f"{participant}.npy")
    out: list[dict[str, Any]] = []
    for session, members in opened:
        compared = None if enrollments is None else enrollments.source_for(session)[1]
        for (facts, run_root, store), attributes in zip(members, readings[session], strict=True):
            if readings_out is None:
                reading_id = write_cohort_reading(store, attributes)
                if compared is not None and compared.vector is not None:
                    write_enrollment_entity(store, compared, reading_id)
                capture_environments(store, {})
                write_store(store, run_root)
                export_prov(store, run_root)
            block = attributes["other_speaker"]
            checks = attributes["checks"]
            consistency = checks.get(CONSISTENCY_CHECK) or {}
            out.append(
                {
                    "stem": facts.stem,
                    "participant": participant,
                    "session": session,
                    "family": facts.family,
                    "status": OK,
                    "other_speaker": block.get("status"),
                    "enrollment_source": block.get("enrollment_source"),
                    "pools": {
                        pool: {
                            key: (block.get(pool) or {}).get(key)
                            for key in (
                                "status",
                                "reason",
                                "pooled_s",
                                "spans_n",
                                "cosine",
                                "participant_cosine",
                                "match",
                            )
                        }
                        for pool in POOLS
                    },
                    "nonmatch": list(block.get("nonmatch") or ()),
                    "consistency": {
                        "status": consistency.get("status"),
                        "cosine": _consistency_cosine(consistency),
                        "outcome": consistency.get("outcome"),
                    },
                    "checks": {name: c.get("outcome") or c.get("status") for name, c in checks.items()},
                    "attributes": attributes if readings_out is not None else None,
                }
            )
    if readings_out is not None:
        readings_out.mkdir(parents=True, exist_ok=True)
        partial = readings_out / f"{participant}.jsonl.partial"
        partial.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out), encoding="utf-8")
        partial.replace(readings_out / f"{participant}.jsonl")
    summary = {
        "participant": participant,
        "sessions_n": len(sessions),
        "members_n": rows_n,
        "enrolled_sessions_n": None if enrollments is None else len(enrollments.enrolled_sessions),
        "sessions": {}
        if enrollments is None
        else {
            session: {
                "enrollment": enrollment.record(),
                "enrollment_source": enrollments.source_for(session)[0],
            }
            for session, enrollment in enrollments.sessions.items()
        },
        "pooled": None if enrollments is None else enrollments.pooled.record(),
        "elapsed_s": round(time.time() - started, 2),
    }
    for record in out:
        record.pop("attributes", None)
        record["participant_summary"] = summary
    return out + failed


def run_apply(args: argparse.Namespace) -> dict[str, Any]:
    """Apply COHORT over one shard of participants.

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
    participants = participants_of(read_manifest(args.manifest, required=("stem", "enhanced")))
    if args.participants:
        wanted = set(args.participants)
        participants = [(p, sessions) for p, sessions in participants if p in wanted]
    mine = take_slice(
        [{"participant": p, "sessions": sessions} for p, sessions in participants], args.slice_index, args.slice_count
    )
    print(f"[slice {args.slice_index}/{args.slice_count}] {len(mine)} participants", flush=True)
    log: list[dict[str, Any]] = []
    for group in mine:
        records = process_participant(
            group["participant"],
            group["sessions"],
            settings=settings,
            embed=embed,
            artefact=artefact,
            quantiles_ref=quantiles_ref,
            readings_out=args.readings_out,
            enrollments_out=args.enrollments_out,
            force=args.force,
        )
        log.extend(records)
        head = records[0].get("participant_summary") if records else None
        if head:
            sources = [s["enrollment_source"] for s in head["sessions"].values()]
            print(
                f"  {group['participant']}: {head['sessions_n']} sessions, {head['members_n']} members, "
                f"{head['enrolled_sessions_n']} enrolled, sources {sorted(sources)}, {head['elapsed_s']} s",
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
        "participants": len(mine),
        "rows": len(log),
        "counts": counts,
        "nonmatch_recordings": sum(1 for r in log if r.get("nonmatch")),
        "quantiles": quantiles_ref,
        "config_hash": config.config_hash,
        "commit": args.commit,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------ consistency


def lower_fence(values: Sequence[float], iqr_multiple: float) -> dict[str, Any]:
    """Tukey's lower fence over a sample: the first quartile less ``iqr_multiple`` interquartile ranges.

    Args:
        values: The sample.
        iqr_multiple: The multiple of the interquartile range.

    Returns:
        ``n``, the quartiles, the 1st/5th percentiles, the minimum, the fence and how many values fall under it.
    """
    array = np.asarray(values, dtype=np.float64)
    q1, q2, q3 = (float(np.quantile(array, q)) for q in (0.25, 0.5, 0.75))
    fence = q1 - iqr_multiple * (q3 - q1)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 4),
        "p01": round(float(np.quantile(array, 0.01)), 4),
        "p05": round(float(np.quantile(array, 0.05)), 4),
        "q1": round(q1, 4),
        "median": round(q2, 4),
        "q3": round(q3, 4),
        "iqr_multiple": iqr_multiple,
        "fence": round(fence, 4),
        "under_fence": int((array < fence).sum()),
    }


def run_consistency(logs: Path, *, iqr_multiple: float) -> dict[str, Any]:
    """The session-vs-other-sessions cosines the apply logs hold, and their lower fence.

    Args:
        logs: The apply log directory, its ``slices/``, or a ``--readings-out`` directory.
        iqr_multiple: The fence's multiple of the interquartile range.

    Returns:
        The fence summary and, at the packaged bound, how many sessions it flags.
    """
    base = logs / SLICES_SUBDIR if (logs / SLICES_SUBDIR).is_dir() else logs
    paths = sorted(base.glob("cohort-slice-*.jsonl")) or sorted(base.glob("sub-*.jsonl"))
    by_session: dict[str, float] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cosine = (record.get("consistency") or {}).get("cosine")
            if cosine is not None:
                by_session[str(record["session"])] = float(cosine)
    if not by_session:
        return {"sessions": 0}
    bound = consistency_bound()
    return {
        "sessions": len(by_session),
        "fence": lower_fence(list(by_session.values()), iqr_multiple),
        "packaged_bound": bound,
        "flagged_at_packaged_bound": None if bound is None else sum(1 for c in by_session.values() if c < bound),
    }


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
        if args.command == "consistency":
            print(json.dumps(run_consistency(args.logs, iqr_multiple=args.iqr_multiple), indent=2, sort_keys=True))
            return 0
        if not args.manifest.exists():
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            return 2
        summary = run_apply(args)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Participants: {summary['participants']}  Rows: {summary['rows']}  Elapsed: {summary['elapsed_s']:.1f} s")
    print(f"Log:      {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<32} {number}")
    return 1 if any(key.startswith(ERROR) for key in summary["counts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
