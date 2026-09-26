#!/usr/bin/env python3
r"""Decide every recording in a finished corpus again, reading nothing new.

    uv run python scripts/extend_refold.py MANIFEST --slice-index I --slice-count N \
        --hints DIR [--log-dir DIR] [--config OVERRIDE.yaml] [--commit SHA]

``MANIFEST`` is the JSONL every ``extend_*`` driver takes: ``stem`` and ``enhanced``, and
optionally ``source``, the recording the run was over, which is what the hint is built from and
which a **mirrored** tree needs because it carries no ``run.json``.

The fold is the cheap half of the graph. A change to the release vocabulary, to a flag ground, or
to a ``verdict.*`` policy key changes what the store's existing evidence *means* without changing
any of it, so bringing a corpus to such a change is a re-fold and never a replay: no model is
loaded, no audio is opened, no measurement is taken. Over 62,548 recordings that is the difference
between minutes of CPU and days of GPU.

It retires only VERDICT's own conclusion, runs VERDICT again over the store as it stands, re-renders
REPORT, and writes a ``REFOLD``/``verdict_refolded`` marker carrying the config hash and the commit.
Every branch report, every other node's verdict and every measurement -- REVIEW's annotation
included -- stands untouched.

**``--hints`` is required.** VERDICT scores each branch against what the recording was declared to
contain, and ``fold_file_verdict`` treats a declaration it cannot resolve as a flag ground of its
own, so re-folding blind would turn every recording's triage axis to ``flag``: a corpus-wide
corruption that reads as a finding.

Unlike the adding drivers this one is **not** resumable by skipping: a re-fold is idempotent, so a
task that is resubmitted simply folds again and reaches the same answer. The store is
content-addressed, so a fold that concludes what already stands mints the same entity and retires
nothing.

Nothing is written outside the recording's own run: ``run/store.jsonl`` is replaced atomically and
``prov/`` is re-exported.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    RUN_SUBDIR,
    SLICES_SUBDIR,
    attempt_derivation,
    export_prov,
    load_hint_builder,
    read_manifest,
    read_store,
    refold_verdict,
    run_root_of,
    source_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import describe_exception
from senselab.audio.workflows.triage.run import RELEASE_SUBDIR, SUMMARY_SUBDIR

NODE = "REFOLD"
UNCHANGED = "unchanged"
"""The fold reached what already stood, so the store was not rewritten."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, and which shard of it this task takes.

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
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    parser.add_argument(
        "--hints",
        type=Path,
        required=True,
        help="Directory holding hints.py; required, because VERDICT reads the declaration",
    )
    parser.add_argument("--commit", default=None, help="The code revision to record on the re-fold marker")
    return parser


def refold_one(
    run_root: Path,
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any],
    source: Path,
    commit: str | None,
) -> dict[str, str]:
    """Decide one finished run again and write the store only if the decision moved.

    Args:
        run_root: The run root.
        config: The triage configuration.
        build_hint: The hint populator.
        source: The recording the run was over.
        commit: The code revision to record on the marker.

    Returns:
        ``{status, REFOLD}`` -- ``ok`` with the two axes where the fold moved, ``unchanged`` where it
        reached what already stood, ``error`` where the store would not open or the fold refused it.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    try:
        hint = build_hint(source)[0]
    except (OSError, ValueError, LookupError, KeyError, IndexError) as error:
        return {"status": ERROR, NODE: f"hint: {describe_exception(error)}"}

    before = store.fingerprint()
    outcome = attempt_derivation(
        lambda: _fold(store, config, hint, run_root=run_root, commit=commit),
    )
    if outcome.failed:
        return {"status": ERROR, NODE: outcome.detail}
    if store.fingerprint() == before:
        return {"status": UNCHANGED, NODE: outcome.detail}
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, NODE: outcome.detail}


def _fold(store: Any, config: TriageConfig, hint: Any, *, run_root: Path, commit: str | None) -> str:  # noqa: ANN401
    """Run the re-fold and render what it decided.

    Args:
        store: The run's store.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_root: The run root.
        commit: The code revision to record on the marker.

    Returns:
        The two axes the fold reached, with a REPORT failure appended rather than conflated: the
        decision landed either way and only the rendering of it is missing.
    """
    result = refold_verdict(
        store,
        config,
        hint,
        run_dir=run_root / RUN_SUBDIR,
        artifacts_dir=run_root / RELEASE_SUBDIR,
        summary_dir=run_root / SUMMARY_SUBDIR,
        commit=commit,
    )
    if "VERDICT" in result.errors:
        raise RuntimeError(result.errors["VERDICT"])
    decided = f"{result.triage}/{result.release}"
    rendered = result.errors.get("REPORT")
    return decided if rendered is None else f"{decided} (report: {rendered})"


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any],
    commit: str | None = None,
) -> list[dict[str, Any]]:
    """Decide every run named by these rows again, one at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        build_hint: The hint populator.
        commit: The code revision to record on each marker.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            out.append({**row, "status": ERROR, NODE: describe_exception(error)})
            continue
        try:
            source = Path(str(row["source"])) if row.get("source") else source_of(run_root)
        except (OSError, ValueError, KeyError) as error:
            out.append({**row, "status": ERROR, NODE: f"source: {describe_exception(error)}"})
            continue
        out.append({**row, **refold_one(run_root, config, build_hint=build_hint, source=source, commit=commit)})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
    hints: Path,
    commit: str | None = None,
) -> dict[str, Any]:
    """Decide every run in one array task's stride of the manifest again.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.
        hints: The directory holding ``hints.py``.
        commit: The code revision to record on each marker.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    log = process(mine, config, build_hint=load_hint_builder(hints), commit=commit)

    counts: dict[str, int] = {}
    decided: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        decided[str(record[NODE])] = decided.get(str(record[NODE]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"refold-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "decided": decided,
        "hints": str(hints),
        "commit": commit,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Decide one array task's stride of a finished corpus again.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0 where every row reached a determinate outcome, 1 where any row is ``error`` (the other
        stores are written either way), 2 where the arguments could not be resolved.
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
            commit=args.commit,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<10} {number}")
    for what, number in sorted(summary["decided"].items(), key=lambda pair: -pair[1])[:10]:
        print(f"  decided {what:<32} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
