r"""Triage every recording of a corpus manifest, one slice per task, resumably, then list the finished runs.

    python -m senselab.audio.workflows.triage.drivers.triage_corpus run MANIFEST --out-root OUT \
        --slice-index I --slice-count N [--hints DIR] [--corpus-root ROOT] [--profile P] [--config C] [--commit SHA]
    python -m senselab.audio.workflows.triage.drivers.triage_corpus manifest OUT --out RUNS.jsonl [--expect MANIFEST]

``run`` is :func:`~senselab.audio.workflows.triage.run.run_triage` over the rows ``[I::N]`` of ``MANIFEST`` (one JSON
object per line, each with ``source``, the recording's path). Each recording gets a fresh run root under
``OUT/out/<participant>/<session>/`` (``triage_audio``'s layout) and, written last, one row of JSON under
``OUT/rows/<participant>/<session>/<stem>.row.json``: the resume sentinel. A recording whose row says it finished and
whose store and summary exist is skipped; anything less is run again, and the run roots an earlier attempt left for
it are removed first. A failure is recorded in the recording's row and never stops the slice.

``--hints DIR`` names a directory holding a dataset's ``hints.py``, whose ``build_hint(path)`` gives each recording
its hint, as the replay drivers read it; without it every recording runs with no hint. ``--corpus-root`` is the
corpus's root, the recording's path relative to which is its id where its hint declares none.

``manifest`` reads the rows under ``OUT/rows/`` and writes the finished runs as the manifest every ``extend_*``
driver reads: one line per recording with ``stem``, ``source``, ``enhanced`` (the run's
``run/streams/enhanced.flac``) and the identity columns the session and cohort passes group rows by, sorted by stem.
With ``--expect``, it also reports every recording of that manifest with no finished run, and exits 1 if there is one.

The row carries what the run concluded per stage, its errors, its paths, its wall time, and the persistent model
workers' events, load times and memory high-water marks; it carries no transcript or measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Sequence

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.extend import (
    add_config_arguments,
    driver_config,
    load_hint_builder,
    read_manifest,
    take_slice,
)
from senselab.audio.workflows.triage.identity import RECORDING_ID_COLUMN, identity_columns, identity_of
from senselab.audio.workflows.triage.run import corpus_path_of, entity_subdir, run_triage
from senselab.utils.portable_audio_io import audio_info
from senselab.utils.venv_worker import shutdown_venv_workers, venv_worker_events, venv_worker_stats

RUNS_SUBDIR = "out"
"""Where the run roots go under ``--out-root``."""

ROWS_SUBDIR = "rows"
"""Where the per-recording rows go under ``--out-root``."""

ROW_SUFFIX = ".row.json"

ENHANCED_RELATIVE = Path("run") / "streams" / "enhanced.flac"
"""The stream every ``extend_*`` driver's manifest names, relative to the run root."""

ISOLATE_LONGER_THAN_S = 150.0
"""A recording at least this long runs with no persistent model worker resident: they are ended before and after it,
so its enhancement never shares memory with the recognisers' weights and a task sized for the rest still fits."""

_RUN_ROOT = re.compile(r"_\d{8}-\d{6}(-\d+)?$")
"""A run root's name less its stem: ``_<utc stamp>``, then ``-<n>`` where two runs landed in one second."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``run`` one slice of a manifest, or list the finished runs as a ``manifest``.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Triage one slice of a manifest")
    run.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording, each with source")
    run.add_argument("--out-root", type=Path, required=True, help="Where out/ (run roots) and rows/ go")
    run.add_argument("--slice-index", type=int, required=True, help="This task's index, 0-based")
    run.add_argument("--slice-count", type=int, required=True, help="How many tasks the array has")
    run.add_argument("--hints", type=Path, default=None, help="Directory holding a hints.py exposing build_hint")
    run.add_argument("--corpus-root", type=Path, default=None, help="The corpus's root, for recording ids")
    run.add_argument("--commit", default=None, help="The code revision to record on each row")
    run.add_argument(
        "--isolate-longer-than",
        type=float,
        default=ISOLATE_LONGER_THAN_S,
        help="Seconds: a recording this long or longer runs with no persistent model worker resident",
    )
    add_config_arguments(run)

    listing = commands.add_parser("manifest", help="Write the finished runs as the passes' manifest")
    listing.add_argument("out_root", type=Path, help="The --out-root the run tasks wrote")
    listing.add_argument("--out", type=Path, required=True, help="The manifest JSONL to write")
    listing.add_argument("--expect", type=Path, default=None, help="The input manifest: report what did not finish")
    return parser


def row_path(out_root: Path, subdir: Path, stem: str) -> Path:
    """Where one recording's row goes.

    Args:
        out_root: The ``--out-root``.
        subdir: The recording's :func:`~senselab.audio.workflows.triage.run.entity_subdir`.
        stem: The recording's file stem.

    Returns:
        ``out_root/rows/<subdir>/<stem>.row.json``.
    """
    return out_root / ROWS_SUBDIR / subdir / f"{stem}{ROW_SUFFIX}"


def finished(row_file: Path) -> bool:
    """Whether a row records a run that finished and whose store and summary products are all still there.

    Args:
        row_file: The recording's row.

    Returns:
        True where the recording need not run again.
    """
    try:
        row = json.loads(row_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(row, dict) or not row.get("ok"):
        return False
    store = Path(str(row.get("store") or ""))
    if not row.get("store") or not store.is_file() or store.stat().st_size == 0:
        return False
    products = row.get("summary") or {}
    return bool(products) and all(Path(str(path)).is_file() for path in products.values())


def remove_earlier_attempts(runs_dir: Path, subdir: Path, stem: str) -> list[str]:
    """Remove the run roots an earlier, unfinished attempt left for a recording.

    Args:
        runs_dir: ``out_root/out``.
        subdir: The recording's entity subdirectory.
        stem: The recording's file stem.

    Returns:
        The names of the run roots removed.
    """
    parent = runs_dir / subdir
    removed: list[str] = []
    if not parent.is_dir():
        return removed
    for candidate in sorted(parent.glob(f"{glob_escape(stem)}_*")):
        if candidate.is_dir() and _RUN_ROOT.sub("", candidate.name) == stem:
            shutil.rmtree(candidate)
            removed.append(candidate.name)
    return removed


def glob_escape(text: str) -> str:
    """``text`` with the glob metacharacters bracketed, so it matches only itself."""
    return re.sub(r"([*?\[])", r"[\1]", text)


def _audio_seconds(path: Path) -> float | None:
    """The recording's duration from its header, or None where it cannot be read."""
    try:
        info = audio_info(path)
        return float(info.frames) / float(info.sampling_rate) if info.frames and info.sampling_rate else None
    except Exception:  # noqa: BLE001 -- a header that cannot be read only skips the isolation rule
        return None


def _high_water_gb(pid: int) -> float | None:
    """A process's resident-memory high-water mark in GB, where ``/proc`` reports one."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1048576, 3)
    except (OSError, ValueError, IndexError):
        return None
    return None


def triage_one(
    entry: dict[str, Any],
    config: TriageConfig,
    *,
    out_root: Path,
    build_hint: Callable[[Path], Any] | None,
    corpus_root: Path | None,
    isolate_longer_than: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Triage one manifest row, unless an earlier attempt finished it, and write its row.

    Args:
        entry: The manifest row; its ``source`` is the recording.
        config: The triage configuration.
        out_root: The ``--out-root``.
        build_hint: The dataset's hint populator, or None.
        corpus_root: The corpus's root, or None.
        isolate_longer_than: Seconds at or above which no persistent model worker stays resident.
        context: Fields every row of this task carries (slice, commit, host, job).

    Returns:
        The row written, or ``{"stem": ..., "skipped": True}`` where the recording had finished.
    """
    source = Path(str(entry["source"]))
    stem = source.stem
    row: dict[str, Any] = {"source": str(source), "stem": stem, **context}
    started = time.time()
    hint = None
    try:
        if not source.is_file():
            raise FileNotFoundError("input audio missing")
        hint = build_hint(source)[0] if build_hint is not None else None
        identity = identity_of(hint, stem=stem, corpus_path=corpus_path_of(source, corpus_root))
        subdir = entity_subdir(identity)
        row["identity"] = {**identity_columns(identity), RECORDING_ID_COLUMN: identity.id_of(stem)}
    except Exception as error:  # noqa: BLE001 -- recorded per row, never fatal to the slice
        subdir = Path("_unplaced")
        row.update(ok=False, error=f"{type(error).__name__}: {error}")
    row_file = row_path(out_root, subdir, stem)
    if "error" not in row:
        if finished(row_file):
            return {"stem": stem, "skipped": True}
        row["removed_attempts"] = remove_earlier_attempts(out_root / RUNS_SUBDIR, subdir, stem)
        seconds = _audio_seconds(source)
        isolated = seconds is not None and seconds >= isolate_longer_than
        row.update(duration_s=seconds, isolated_long_recording=isolated)
        if isolated:
            shutdown_venv_workers()
        try:
            result = run_triage(source, out_root / RUNS_SUBDIR, config, hint=hint, corpus_root=corpus_root)
            row.update(
                ok=True,
                run_root=str(result.run_dir.parent),
                run_dir=str(result.run_dir),
                store=str(result.store_path),
                summary={name: str(path) for name, path in result.summary.items()},
                released={name: str(path) for name, path in result.released.items()},
                states={node: outcome.state.value for node, outcome in result.nodes.items()},
                errors={node: outcome.error for node, outcome in result.nodes.items() if outcome.error},
            )
        except Exception as error:  # noqa: BLE001 -- recorded per row, never fatal to the slice
            row.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-3000:])
        finally:
            if isolated:
                shutdown_venv_workers()
    row["seconds"] = round(time.time() - started, 2)
    row["worker_events"] = venv_worker_events(clear=True)
    workers = venv_worker_stats()
    row["workers"] = [
        {key: worker.get(key) for key in ("label", "pid", "load_s", "served", "threads")} for worker in workers
    ]
    row["high_water_gb"] = {
        "driver": _high_water_gb(os.getpid()),
        **{str(worker.get("label")): _high_water_gb(int(worker["pid"])) for worker in workers if worker.get("pid")},
    }
    row_file.parent.mkdir(parents=True, exist_ok=True)
    partial = row_file.with_name(row_file.name + ".partial")
    partial.write_text(json.dumps(row, default=str, indent=1), encoding="utf-8")
    partial.replace(row_file)
    return row


def run_slice(
    manifest: Path,
    *,
    out_root: Path,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    build_hint: Callable[[Path], Any] | None = None,
    corpus_root: Path | None = None,
    commit: str | None = None,
    isolate_longer_than: float = ISOLATE_LONGER_THAN_S,
) -> dict[str, Any]:
    """Triage one array task's stride of a manifest.

    Args:
        manifest: The manifest JSONL, each row with ``source``.
        out_root: Where ``out/`` and ``rows/`` go.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        build_hint: The dataset's hint populator (``hints.py``'s ``build_hint``), or None to run with no hint.
        corpus_root: The corpus's root, or None.
        commit: The code revision to record on each row.
        isolate_longer_than: Seconds at or above which no persistent model worker stays resident.

    Returns:
        The task's counts: ``ok``, ``failed`` and ``skipped``, and the stems that failed.
    """
    mine = take_slice(read_manifest(manifest, required=("source",)), slice_index, slice_count)
    context = {
        "slice": slice_index,
        "n_slices": slice_count,
        "config_hash": config.config_hash,
        "profile": None if config.profile is None else f"{config.profile.name} {config.profile.version}",
        "commit": commit,
        "host": os.uname().nodename,
        "job": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} recordings, config {config.config_hash}", flush=True)
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    failed: list[str] = []
    for number, entry in enumerate(mine, start=1):
        row = triage_one(
            entry,
            config,
            out_root=out_root,
            build_hint=build_hint,
            corpus_root=corpus_root,
            isolate_longer_than=isolate_longer_than,
            context=context,
        )
        if row.get("skipped"):
            counts["skipped"] += 1
            continue
        if row.get("ok"):
            counts["ok"] += 1
        else:
            counts["failed"] += 1
            failed.append(str(row["stem"]))
            print(f"  FAILED {row['stem']}: {str(row.get('error')).split(':', 1)[0]}", flush=True)
        if row.get("errors"):
            print(f"  NODE-ERROR {row['stem']}: {sorted(row['errors'])}", flush=True)
        print(f"[{number}/{len(mine)}] {row['stem']} {row.get('seconds')}s", flush=True)
    return {"rows": len(mine), "counts": counts, "failed": failed}


def finished_runs(out_root: Path) -> list[dict[str, Any]]:
    """Every finished run under ``out_root/rows``, as the passes' manifest rows.

    Args:
        out_root: The ``--out-root`` the run tasks wrote.

    Returns:
        One row per recording with a finished run, sorted by stem: ``stem``, ``source``, ``enhanced`` and the
        identity columns the corpus passes group by (as the corpus table writes them).

    Raises:
        ValueError: If two rows name one stem.
    """
    rows: dict[str, dict[str, Any]] = {}
    for row_file in sorted((out_root / ROWS_SUBDIR).rglob(f"*{ROW_SUFFIX}")):
        if not finished(row_file):
            continue
        row = json.loads(row_file.read_text(encoding="utf-8"))
        stem = str(row["stem"])
        if stem in rows:
            raise ValueError(f"two finished rows name {stem}: {row_file} and an earlier one")
        rows[stem] = {
            "stem": stem,
            "source": str(row["source"]),
            "enhanced": str(Path(row["run_root"]) / ENHANCED_RELATIVE),
            **(row.get("identity") or {}),
        }
    return [rows[stem] for stem in sorted(rows)]


def write_manifest(out_root: Path, out: Path, *, expect: Path | None = None) -> dict[str, Any]:
    """Write the finished runs as the passes' manifest, and what of an expected manifest did not finish.

    Args:
        out_root: The ``--out-root`` the run tasks wrote.
        out: The manifest JSONL to write.
        expect: The input manifest, or None.

    Returns:
        ``written`` (how many rows), and with ``expect`` the ``missing`` stems.
    """
    rows = finished_runs(out_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    partial.replace(out)
    summary: dict[str, Any] = {"written": len(rows), "out": str(out)}
    if expect is not None:
        have = {row["stem"] for row in rows}
        wanted = [Path(str(entry["source"])).stem for entry in read_manifest(expect, required=("source",))]
        summary["missing"] = sorted(stem for stem in wanted if stem not in have)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one slice, or write the finished runs' manifest.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        ``run``: 0 where every recording finished or had finished, 1 where any failed. ``manifest``: 0, or 1 where
        ``--expect`` names a recording with no finished run. 2 where the arguments could not be resolved.
    """
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        try:
            summary = write_manifest(args.out_root, args.out, expect=args.expect)
        except (OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"Manifest: {summary['out']} ({summary['written']} runs)")
        missing = summary.get("missing") or []
        if missing:
            print(f"Missing:  {len(missing)} recordings with no finished run, e.g. {missing[:5]}")
        return 1 if missing else 0
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        config = driver_config(args)
        build_hint = load_hint_builder(args.hints) if args.hints is not None else None
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    try:
        summary = run_slice(
            args.manifest,
            out_root=args.out_root,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            config=config,
            build_hint=build_hint,
            corpus_root=args.corpus_root,
            commit=args.commit,
            isolate_longer_than=args.isolate_longer_than,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    counts = summary["counts"]
    done = f"{counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped"
    print(f"=== SLICE DONE: {done}, of {summary['rows']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
