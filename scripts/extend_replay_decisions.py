#!/usr/bin/env python3
r"""Re-decide a finished triage run from TAXONOMY, over the PREPROCESS output its store holds.

    uv run python scripts/extend_replay_decisions.py MANIFEST --slice-index I --slice-count N \
        [--hints DIR] [--out-root DIR] [--commit SHA] [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is the JSONL the other extend drivers take: one object per line carrying ``stem`` and
``enhanced`` (the absolute path of that recording's ``run/streams/enhanced.flac``), from which the
run root is derived.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

Unlike the drivers that add a measurement, this one re-decides: every live entity a node from
TAXONOMY on generated is retired with an invalidation edge, and TAXONOMY, routing, the selected
branches, QUALITY, REDACT, VERDICT and REPORT run again. Nothing is deleted, so a replayed store
reads as the new decision, the old one, and the edge between them.

``--hints DIR`` names a directory holding a ``hints.py`` exposing ``build_hint(wav)``. The hint is
the one input the run does not record, and routing, the branches, REDACT and VERDICT all read it;
without ``--hints`` the replay passes none, which is a different run and is recorded as such on
every row.

``--out-root DIR`` writes each replayed run into a fresh root under ``DIR`` instead of in place,
with the finished run's ``derivatives/`` and the streams the replay only reads reachable through
symlinks. A stream a replayed node writes -- REDACT's ``redacted``, SPEECH's ``separated_*`` -- is
never linked, so that write lands in the new root. Use it when the finished tree must not be
modified.

A store already carrying this configuration's replay marker is skipped and not rewritten.

The design is in ``specs/20260922-replay-decisions-over-a-finished-corpus/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    PRESENT,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    ReplayOutcome,
    export_prov,
    find_replay_marker,
    read_manifest,
    read_store,
    replay_decisions,
    replay_run_id,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import STREAM_SUFFIX, describe_exception
from senselab.audio.workflows.triage.nodes.redact import STREAM_NAME as REDACTED_STREAM
from senselab.audio.workflows.triage.nodes.speech import SEPARATED_PREFIX
from senselab.audio.workflows.triage.run import LOG_FILE, RELEASE_SUBDIR, SUMMARY_SUBDIR, entity_subdir

DERIVATION = "replay"
STREAMS_SUBDIR = "streams"
DERIVATIVES_SUBDIR = "derivatives"
WRITTEN_STREAM_STEMS = (REDACTED_STREAM, SEPARATED_PREFIX)
"""The stems a replayed node writes into ``streams/``: REDACT's, and SPEECH's per-source prefix."""


def written_by_a_replayed_node(name: str) -> bool:
    """Whether a stream file name is one a replayed node writes rather than only reads.

    Args:
        name: The file name inside ``streams/``.

    Returns:
        True when a replayed node writes this name.
    """
    stem = name[: -len(STREAM_SUFFIX)] if name.endswith(STREAM_SUFFIX) else name
    return any(stem == each or stem.startswith(each) for each in WRITTEN_STREAM_STEMS)


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, which shard of it this task takes, and where the replay writes.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording")
    parser.add_argument("--slice-index", type=int, required=True, help="This task's index, 0-based")
    parser.add_argument("--slice-count", type=int, required=True, help="How many tasks the array has")
    parser.add_argument("--hints", type=Path, default=None, help="Directory holding a hints.py exposing build_hint")
    parser.add_argument("--out-root", type=Path, default=None, help="Write replayed runs here instead of in place")
    parser.add_argument("--commit", default=None, help="The code revision to record on each replay marker")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    return parser


def load_hint_builder(directory: Path) -> Callable[[Path], Any]:
    """The hint populator from a directory holding ``hints.py``.

    Args:
        directory: The directory holding ``hints.py``.

    Returns:
        Its ``build_hint``, called with the recording's path.

    Raises:
        FileNotFoundError: If the directory holds no importable ``hints.py``.
    """
    spec = importlib.util.spec_from_file_location("replay_hints", directory / "hints.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no importable hints.py in {directory}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder: Callable[[Path], Any] = module.build_hint
    return builder


def source_of(run_root: Path) -> Path:
    """The recording a finished run was over, from its own ``run.json``.

    Args:
        run_root: The run root.

    Returns:
        The recording's path.

    Raises:
        FileNotFoundError: If the run holds no log.
        ValueError: If the log names no source.
    """
    log_path = run_root / RUN_SUBDIR / LOG_FILE
    if not log_path.is_file():
        raise FileNotFoundError(f"no run log at {log_path}")
    source = json.loads(log_path.read_text()).get("source")
    if not source:
        raise ValueError(f"{log_path} names no source")
    return Path(str(source))


def mirror_run_root(run_root: Path, out_root: Path, stem: str) -> Path:
    """A writable run root under ``out_root`` whose sidecars read through to the finished one.

    ``derivatives/`` is a symlink because no replayed node writes there. ``streams/`` is a real
    directory holding a symlink per stream the replay only *reads*; a stream a replayed node writes
    is never linked, so the write lands here and not in the finished tree.

    Args:
        run_root: The finished run root.
        out_root: The tree replayed roots are created under.
        stem: The recording's file stem, placing the root under its BIDS entity path.

    Returns:
        The replayed run root, created.
    """
    dest = out_root / entity_subdir(stem) / run_root.name
    run_dir = dest / RUN_SUBDIR
    (run_dir / STREAMS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (dest / RELEASE_SUBDIR).mkdir(parents=True, exist_ok=True)
    finished = run_root / RUN_SUBDIR
    derivatives = run_dir / DERIVATIVES_SUBDIR
    if not derivatives.is_symlink() and not derivatives.exists():
        derivatives.symlink_to(finished / DERIVATIVES_SUBDIR)
    for stream in sorted((finished / STREAMS_SUBDIR).glob("*")):
        if written_by_a_replayed_node(stream.name):
            continue
        link = run_dir / STREAMS_SUBDIR / stream.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(stream)
    return dest


def replay_one(
    run_root: Path,
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any] | None,
    out_root: Path | None,
    stem: str,
    commit: str | None,
) -> dict[str, Any]:
    """Replay one finished run, writing its store only once every replayed node has run.

    The store is read under :func:`replay_run_id`, so what the replay writes takes ids distinct
    from the decisions it retires. A run already carrying this configuration's marker is skipped
    and nothing is written; a run that raises before the write keeps the store it had, so a
    preempted task redoes that recording from the beginning and cannot stack two retirements.

    Args:
        run_root: The finished run root.
        config: The replaying configuration.
        build_hint: The hint populator, or None to replay with no hint.
        out_root: Where to write the replayed run, or None to replay in place.
        stem: The recording's file stem.
        commit: The code revision to record on the marker.

    Returns:
        ``{status, replay, ...}`` — ``ok`` when the replay landed, ``present`` when this
        configuration had already replayed the run, ``error`` when it could not.
    """
    target = mirror_run_root(run_root, out_root, stem) if out_root is not None else run_root
    read_from = target if (target / RUN_SUBDIR / "store.jsonl").is_file() else run_root
    try:
        store = read_store(read_from, run_id=replay_run_id(run_root, config.config_hash))
    except (OSError, ValueError) as error:
        return {"status": ERROR, DERIVATION: describe_exception(error)}
    if find_replay_marker(store, config.config_hash) is not None:
        return {"status": PRESENT, DERIVATION: "this configuration has already replayed this run"}
    try:
        hint = build_hint(source_of(run_root))[0] if build_hint is not None else None
    except (OSError, ValueError, LookupError, KeyError, IndexError) as error:
        return {"status": ERROR, DERIVATION: f"hint: {describe_exception(error)}"}
    try:
        outcome: ReplayOutcome = replay_decisions(
            store,
            config,
            hint,
            run_dir=target / RUN_SUBDIR,
            artifacts_dir=target / RELEASE_SUBDIR,
            summary_dir=target / SUMMARY_SUBDIR,
            commit=commit,
        )
    except (OSError, ValueError, LookupError) as error:
        return {"status": ERROR, DERIVATION: describe_exception(error)}
    write_store(store, target)
    export_prov(store, target)
    return {
        "status": OK,
        DERIVATION: "replayed",
        "run_root": str(target),
        "retired": outcome.retired,
        "hint": "built" if build_hint is not None else "none",
        "states": outcome.states,
        "errors": outcome.errors,
        "released": sorted(outcome.released),
        "summary": sorted(outcome.summary),
    }


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any] | None,
    out_root: Path | None,
    commit: str | None,
) -> list[dict[str, Any]]:
    """Replay every run named by these rows, one at a time.

    A recording whose store will not open, or whose run tree no longer holds its sidecars, gets an
    outcome record; its neighbours are unaffected.

    Args:
        rows: The manifest rows this task owns.
        config: The replaying configuration.
        build_hint: The hint populator, or None.
        out_root: Where to write replayed runs, or None to replay in place.
        commit: The code revision to record on each marker.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            out.append({**row, "status": ERROR, DERIVATION: describe_exception(error)})
            continue
        replayed = replay_one(
            run_root,
            config,
            build_hint=build_hint,
            out_root=out_root,
            stem=str(row["stem"]),
            commit=commit,
        )
        out.append({**row, **replayed})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
    hints: Path | None,
    out_root: Path | None,
    commit: str | None,
) -> dict[str, Any]:
    """Replay every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The replaying configuration.
        log_dir: Where this task's ``slices/`` log goes.
        hints: The directory holding ``hints.py``, or None to replay with no hint.
        out_root: Where to write replayed runs, or None to replay in place.
        commit: The code revision to record on each marker.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    build_hint = load_hint_builder(hints) if hints is not None else None
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    log = process(mine, config, build_hint=build_hint, out_root=out_root, commit=commit)

    counts: dict[str, int] = {}
    node_errors: dict[str, int] = {}
    retired = 0
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        retired += int(record.get("retired") or 0)
        for node in record.get("errors") or {}:
            node_errors[node] = node_errors.get(node, 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"replay-decisions-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "commit": commit,
        "hints": str(hints) if hints is not None else None,
        "out_root": str(out_root) if out_root is not None else None,
        "rows": len(mine),
        "counts": counts,
        "retired": retired,
        "node_errors": node_errors,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Replay every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard was replayed or already had been, 1 when any row is
        ``error`` — the other stores are written either way — and 2 when the arguments could not be
        resolved and nothing was read.
    """
    args = build_parser().parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            config=load_triage_config(args.config) if args.config else load_triage_config(),
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
            hints=args.hints,
            out_root=args.out_root,
            commit=args.commit,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    print(f"Retired: {summary['retired']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<9} {number}")
    for node, number in sorted(summary["node_errors"].items()):
        print(f"  node error {node:<10} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
