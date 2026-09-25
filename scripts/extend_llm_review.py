#!/usr/bin/env python3
r"""Run REVIEW over a finished triage corpus, in place, without replaying the graph.

    uv run python scripts/extend_llm_review.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml] [--out-root DIR] [--apply] [--force]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the manifest every ``extend_*`` driver
takes, and the run root is derived from ``enhanced`` the same way.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

REVIEW reads the store and nothing else: both transcripts come out of the consensus words and the
redaction spans, so this pass opens no audio and decodes no stream unless ``--apply`` is given. The
replay driver reaches the same step by re-running TAXONOMY through REPORT on every recording; over
this corpus that is the graph's cost paid to get at one node's.

``--apply`` additionally writes the recording's redacted stream from the reviewer's refined span
set, which is the one place this pass touches audio. It is CPU and I/O rather than GPU, so a run
with scarce cards should leave it off here and make a second CPU pass.

Resumability is the store's, as in the other adding drivers: a recording whose store already holds
a live ``redaction_llm_annotation`` written under this configuration is ``present`` and is not read
again, so a preempted array task is resumed by resubmitting it. ``--force`` reads again and retires
what stood only where the re-reading actually differs.

Nothing is written outside the recording's own run: ``run/store.jsonl`` is replaced atomically and
``prov/`` is re-exported. ``--out-root`` mirrors each run root under a directory of its own first,
for a corpus tree that must not be modified; the mirror seeds ``streams/`` with symlinks for the
streams nothing here writes and a real file for the one ``--apply`` does, because writing through a
symlink into a finished tree is how 17,924 corpus files were overwritten.

The design is in ``specs/20260924-reviewer-over-every-transcript/design.md``.

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
from typing import Any, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    PRESENT,
    PROV_SUBDIR,
    RUN_SUBDIR,
    SLICES_SUBDIR,
    attempt_derivation,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    supersede,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import (
    capture_environments,
    describe_exception,
    find_measurement,
    software_agent,
)
from senselab.audio.workflows.triage.nodes.redact import STREAM_NAME as REDACTED_STREAM
from senselab.audio.workflows.triage.nodes.review import NODE, apply_proposal, review
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION
from senselab.utils.prov_store import ProvStore
from senselab.utils.subprocess_venv import record_venv_use

STREAM_SUFFIX = ".flac"
SOURCE_STREAM = "enhanced"
ANNOTATION_SUPERSEDED = "llm_annotation_superseded"
_SUPERSEDED_REASON = "re-read under a later configuration"


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
    parser.add_argument("--out-root", type=Path, default=None, help="Mirror each run root here instead of writing it")
    parser.add_argument("--apply", action="store_true", help="Also write the redacted stream from the proposal")
    parser.add_argument("--force", action="store_true", help="Re-read where an annotation already stands")
    return parser


def mirror_run_root(run_root: Path, out_root: Path) -> Path:
    """A writable copy of one run root whose streams point back at the finished one.

    ``streams/`` is a real directory holding a symlink per stream this pass only reads. The one
    stream ``--apply`` writes is never seeded as a link: an audio write is a truncating open that
    follows one, which is how a mirrored run root destroyed 17,924 files in the corpus it mirrored.

    Args:
        run_root: The finished run root.
        out_root: Where the mirror goes.

    Returns:
        The mirror's path.
    """
    mirror = out_root / run_root.name
    (mirror / RUN_SUBDIR / "streams").mkdir(parents=True, exist_ok=True)
    source_run = run_root / RUN_SUBDIR
    store_path = source_run / "store.jsonl"
    target = mirror / RUN_SUBDIR / "store.jsonl"
    if store_path.is_file() and not target.exists():
        target.write_bytes(store_path.read_bytes())
    derivatives = source_run / "derivatives"
    link = mirror / RUN_SUBDIR / "derivatives"
    if derivatives.exists() and not link.is_symlink() and not link.exists():
        link.symlink_to(derivatives)
    for stream in sorted((source_run / "streams").glob(f"*{STREAM_SUFFIX}")):
        if stream.stem == REDACTED_STREAM:
            continue
        seeded = mirror / RUN_SUBDIR / "streams" / stream.name
        if not seeded.is_symlink() and not seeded.exists():
            seeded.symlink_to(stream)
    return mirror


def standing(store: ProvStore) -> str | None:
    """The status of the annotation this store already holds, if any.

    Args:
        store: The run's store.

    Returns:
        The status, or None where no live annotation stands.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    return None if annotation is None else str(annotation.attributes.get("status") or "")


def review_one(store: ProvStore, config: TriageConfig, *, run_dir: Path, apply: bool, held: str | None) -> str:
    """Read one finished run back, write the refined audio where asked to, and retire what it replaced.

    The retirement happens after the write and only where the new annotation's id differs from the
    one that stood, which is the ordering the adding drivers use. The store is content-addressed,
    so a re-read reaching the same conclusion under the same configuration mints the same entity;
    retiring first would invalidate the re-read along with what it replaced.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run directory, ``<run_root>/run``.
        apply: Whether to write the redacted stream from the proposal.
        held: The id of the annotation that stood before this read, or None.

    Returns:
        The reading's status, with ``+applied`` appended where a stream was written.
    """
    outcome = review(store, config)
    if held is not None and held != outcome.annotation_id:
        supersede(
            store,
            held,
            node=NODE,
            step=ANNOTATION_SUPERSEDED,
            reason=_SUPERSEDED_REASON,
            software=software_agent(store),
        )
    if not apply:
        return outcome.status
    written = apply_proposal(store, config, run_dir=run_dir, source=SOURCE_STREAM)
    return f"{outcome.status}+applied" if written is not None else outcome.status


def extend_one(run_root: Path, config: TriageConfig, *, apply: bool, force: bool) -> dict[str, str]:
    """Read one finished run back and write the store only if it changed.

    Args:
        run_root: The run root, already the writable one.
        config: The triage configuration.
        apply: Whether to write the redacted stream from the proposal.
        force: Whether to retire a standing annotation and read again.

    Returns:
        ``{status, REVIEW}`` -- ``ok`` when a reading landed, ``present`` when one already stood,
        ``error`` when the store would not open or the node refused it.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    held = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if held is not None and not force:
        return {"status": PRESENT, NODE: str(held.attributes.get("status") or "")}
    before = store.fingerprint()
    with record_venv_use() as used:
        outcome = attempt_derivation(
            lambda: review_one(
                store, config, run_dir=run_root / RUN_SUBDIR, apply=apply, held=None if held is None else held.id
            )
        )
    if outcome.failed:
        return {"status": ERROR, NODE: outcome.detail}
    if store.fingerprint() == before:
        return {"status": PRESENT, NODE: outcome.detail}
    capture_environments(store, used)
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, NODE: outcome.detail}


def process(
    rows: Sequence[dict[str, Any]], config: TriageConfig, *, out_root: Path | None, apply: bool, force: bool
) -> list[dict[str, Any]]:
    """Read back every run named by these rows, one at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        out_root: Where to mirror each run root, or None to write in place.
        apply: Whether to write the redacted stream from the proposal.
        force: Whether to retire a standing annotation and read again.

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
        if out_root is not None:
            run_root = mirror_run_root(run_root, out_root)
        out.append({**row, **extend_one(run_root, config, apply=apply, force=force)})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
    out_root: Path | None = None,
    apply: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Read back every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.
        out_root: Where to mirror each run root, or None to write in place.
        apply: Whether to write the redacted stream from the proposal.
        force: Whether to retire a standing annotation and read again.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    log = process(mine, config, out_root=out_root, apply=apply, force=force)

    counts: dict[str, int] = {}
    readings: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        readings[str(record[NODE])] = readings.get(str(record[NODE]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"llm-review-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "readings": readings,
        "apply": apply,
        "force": force,
        "out_root": str(out_root) if out_root is not None else None,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Read one array task's stride of a finished corpus back with the reviewer.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0 where every row reached a determinate outcome or already had one, 1 where any row is
        ``error`` (the other stores are written either way), 2 where the arguments could not be
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
            out_root=args.out_root,
            apply=args.apply,
            force=args.force,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<9} {number}")
    for reading, number in sorted(summary["readings"].items()):
        print(f"  read {reading:<16} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
