#!/usr/bin/env python3
r"""Run REVIEW over a finished triage corpus, in place, without replaying the graph.

    uv run python scripts/extend_llm_review.py MANIFEST --slice-index I --slice-count N \
        --hints DIR [--log-dir DIR] [--config OVERRIDE.yaml] [--out-root DIR] [--apply] \
        [--force] [--commit SHA] [--no-refold]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the manifest every ``extend_*`` driver
takes, and the run root is derived from ``enhanced`` the same way.

A row may also carry ``source``, the recording the run was over, which is what the hint is built
from. It is optional and it is how this driver runs over a **mirrored** tree: a replayed corpus
holds ``store.jsonl``, ``streams/`` and a ``derivatives`` symlink, and no ``run.json``, so
:func:`~senselab.audio.workflows.triage.extend.source_of` has nothing to read there. Where the row
names no source the run's own log is read, which is the in-place case.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

REVIEW reads the store and nothing else: both transcripts come out of the consensus words and the
redaction spans, so this pass opens no audio and decodes no stream unless ``--apply`` is given. The
replay driver reaches the same step by re-running TAXONOMY through REPORT on every recording; over
this corpus that is the graph's cost paid to get at one node's.

``--apply`` additionally writes the recording's redacted stream from the reviewer's refined span
set, which is the one place this pass touches audio. It is CPU and I/O rather than GPU, so a run
with scarce cards should leave it off here and make a second CPU pass.

**Every recording whose reading lands is decided again.** REVIEW writes an input VERDICT reads, so
a store that gained the reading without folding again would hold a judgement made blind to it.
``--hints`` is therefore required: VERDICT scores each branch against the recording's declaration,
and an unresolvable declaration is a flag ground of its own, so re-folding without a hint would turn
every recording's triage axis to ``flag``. Only VERDICT's own conclusion is retired; every branch
report and every other node's verdict stands, which is what separates this from the replay driver.
``--no-refold`` leaves the recorded decisions exactly where they were, for a pass whose only purpose
is to collect readings.

Resumability is the store's, as in the other adding drivers: a recording whose store already holds
a live ``redaction_llm_annotation`` **that a REVIEW activity generated** is ``present`` and is not
read again, so a preempted array task is resumed by resubmitting it. ``--force`` reads again and
retires what stood only where the re-reading actually differs.

The node is the whole of that test. A corpus replayed before the reviewer became its own node
carries a live annotation from REDACT's ``llm_check`` step, and over the r4 corpus every one of
them reads ``disabled``, because the packaged config leaves the reviewer off. Counting those as
standing readings turns the entire pass into a no-op that reports ``present`` on every row and
exits 0.

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
from typing import Any, Callable, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    PRESENT,
    PROV_SUBDIR,
    RUN_SUBDIR,
    SLICES_SUBDIR,
    VERDICT_NODE,
    attempt_derivation,
    export_prov,
    load_hint_builder,
    read_manifest,
    read_store,
    refold_verdict,
    run_root_of,
    source_of,
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
from senselab.audio.workflows.triage.run import REPORT_NODE, SUMMARY_SUBDIR
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION
from senselab.utils.prov_store import Entity, ProvStore
from senselab.utils.subprocess_venv import record_venv_use

STREAM_SUFFIX = ".flac"
SOURCE_STREAM = "enhanced"
ANNOTATION_SUPERSEDED = "llm_annotation_superseded"
_SUPERSEDED_REASON = "re-read under a later configuration"
REFOLD = "refold"
"""The outcome key carrying what the re-fold decided, or why it did not run."""

SKIPPED = "skipped"
"""The re-fold's outcome where the caller passed no hint and asked for none."""


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
    parser.add_argument(
        "--hints",
        type=Path,
        default=None,
        help="Directory holding hints.py; required unless --no-refold, because VERDICT reads the declaration",
    )
    parser.add_argument(
        "--no-refold",
        action="store_true",
        help="Leave the recorded verdict as it stands, having seen no reading",
    )
    parser.add_argument("--commit", default=None, help="The code revision to record on the re-fold marker")
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


def standing(store: ProvStore) -> Entity | None:
    """The reading REVIEW has already left here, if any.

    An annotation is this pass's work only where a ``REVIEW`` activity generated it. A store
    replayed before the reviewer became its own node carries a live annotation from REDACT's
    ``llm_check`` step -- ``disabled`` over the whole r4 corpus, because the packaged config leaves
    the reviewer off -- and counting that as a standing reading makes the whole pass a silent no-op
    that reports ``present`` on every row.

    Args:
        store: The run's store.

    Returns:
        The annotation entity, or None where REVIEW has left none.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if annotation is None:
        return None
    activity_id = store.generated_by(annotation.id)
    if activity_id is None:
        return None
    try:
        node = store.get_activity(activity_id).node
    except KeyError:
        return None
    return annotation if node == NODE else None


def live_annotations(store: ProvStore) -> list[str]:
    """Every live ``redaction_llm_annotation`` in the store, whichever node wrote it.

    Separate from :func:`standing`, and deliberately: whether to read again is a question about
    REVIEW's own work, while what to retire afterwards is a question about the name. A store
    replayed before the reviewer moved out of REDACT carries that node's annotation under the same
    name, and leaving it live puts two of them in the store.

    Every current consumer takes ``find_measurement``, which returns the latest live entity, so the
    stale one is not what any of them folds. What the retirement buys is the store's own discipline:
    one live record per name, a reader that does not depend on write order to be correct, and a
    ``find_measurements`` consumer that cannot double-count.

    Args:
        store: The run's store.

    Returns:
        The entity ids, in the store's own order.
    """
    return [
        entity.id
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == REDACTION_LLM_ANNOTATION and not store.is_invalidated(entity.id)
    ]


def review_one(store: ProvStore, config: TriageConfig, *, run_dir: Path, apply: bool, held: Sequence[str]) -> str:
    """Read one finished run back, write the refined audio where asked to, and retire what it replaced.

    The retirement happens after the write and only for ids the new annotation did not itself take,
    which is the ordering the adding drivers use. The store is content-addressed, so a re-read
    reaching the same conclusion under the same configuration mints the same entity; retiring first
    would invalidate the re-read along with what it replaced.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run directory, ``<run_root>/run``.
        apply: Whether to write the redacted stream from the proposal.
        held: The ids of every annotation that was live before this read.

    Returns:
        The reading's status, with ``+applied`` appended where a stream was written.
    """
    outcome = review(store, config)
    for entity_id in held:
        if entity_id == outcome.annotation_id:
            continue
        supersede(
            store,
            entity_id,
            node=NODE,
            step=ANNOTATION_SUPERSEDED,
            reason=_SUPERSEDED_REASON,
            software=software_agent(store),
        )
    if not apply:
        return outcome.status
    written = apply_proposal(store, config, run_dir=run_dir, source=SOURCE_STREAM)
    return f"{outcome.status}+applied" if written is not None else outcome.status


def extend_one(
    run_root: Path,
    config: TriageConfig,
    *,
    apply: bool,
    force: bool,
    build_hint: Callable[[Path], Any] | None,
    source: Path | None,
    commit: str | None = None,
) -> dict[str, str]:
    """Read one finished run back, re-fold its verdict over the reading, and write the store once.

    The re-fold is not optional decoration. REVIEW writes an input VERDICT reads, so a store that
    gains the reading without deciding again holds a judgement made blind to it. It runs only where
    a reading actually landed: a ``present`` or ``error`` row leaves the recorded decision alone.

    Args:
        run_root: The run root, already the writable one.
        config: The triage configuration.
        apply: Whether to write the redacted stream from the proposal.
        force: Whether to retire a standing annotation and read again.
        build_hint: The hint populator, or None to leave the recorded verdict as it stands.
        source: The recording the run was over, for the hint. None where no hint is built.
        commit: The code revision to record on the re-fold marker.

    Returns:
        ``{status, REVIEW}`` -- ``ok`` when a reading landed, ``present`` when one already stood,
        ``error`` when the store would not open or the node refused it. A re-fold adds ``refold``
        with the two axes it reached, or the reason it could not run.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    held = standing(store)
    if held is not None and not force:
        return {"status": PRESENT, NODE: str(held.attributes.get("status") or "")}
    replaced = live_annotations(store)
    before = store.fingerprint()
    with record_venv_use() as used:
        outcome = attempt_derivation(
            lambda: review_one(store, config, run_dir=run_root / RUN_SUBDIR, apply=apply, held=replaced)
        )
    if outcome.failed:
        return {"status": ERROR, NODE: outcome.detail}
    if store.fingerprint() == before:
        return {"status": PRESENT, NODE: outcome.detail}
    capture_environments(store, used)
    refolded = _refold(store, config, run_root=run_root, build_hint=build_hint, source=source, commit=commit)
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, NODE: outcome.detail, REFOLD: refolded}


def _refold(
    store: ProvStore,
    config: TriageConfig,
    *,
    run_root: Path,
    build_hint: Callable[[Path], Any] | None,
    source: Path | None,
    commit: str | None,
) -> str:
    """Decide the file again over the reading REVIEW just wrote.

    Args:
        store: The run's store, carrying the new annotation.
        config: The triage configuration.
        run_root: The run root, already the writable one.
        build_hint: The hint populator, or None to leave the recorded verdict as it stands.
        source: The recording the run was over, for the hint.
        commit: The code revision to record on the marker.

    Returns:
        The two axes the re-fold reached, or why it did not run. A REPORT that raised is appended
        rather than conflated with a VERDICT that did: the decision landed either way, and only
        the rendering of it is missing.
    """
    if build_hint is None or source is None:
        return SKIPPED
    try:
        hint = build_hint(source)[0]
    except (OSError, ValueError, LookupError, KeyError, IndexError) as error:
        return f"{ERROR}: hint: {describe_exception(error)}"
    try:
        outcome = refold_verdict(
            store,
            config,
            hint,
            run_dir=run_root / RUN_SUBDIR,
            summary_dir=run_root / SUMMARY_SUBDIR,
            commit=commit,
        )
    except (OSError, ValueError, LookupError) as error:
        return f"{ERROR}: {describe_exception(error)}"
    if VERDICT_NODE in outcome.errors:
        return f"{ERROR}: {outcome.errors[VERDICT_NODE]}"
    decided = f"{outcome.triage}/{outcome.release}"
    rendered = outcome.errors.get(REPORT_NODE)
    return decided if rendered is None else f"{decided} (report: {rendered})"


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    out_root: Path | None,
    apply: bool,
    force: bool,
    build_hint: Callable[[Path], Any] | None = None,
    commit: str | None = None,
) -> list[dict[str, Any]]:
    """Read back every run named by these rows, one at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        out_root: Where to mirror each run root, or None to write in place.
        apply: Whether to write the redacted stream from the proposal.
        force: Whether to retire a standing annotation and read again.
        build_hint: The hint populator, or None to leave each recorded verdict as it stands.
        commit: The code revision to record on each re-fold marker.

    A row's own ``source`` wins over the run's log, because a mirrored run root carries no log.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            finished = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            out.append({**row, "status": ERROR, NODE: describe_exception(error)})
            continue
        source: Path | None = None
        if build_hint is not None:
            try:
                source = Path(str(row["source"])) if row.get("source") else source_of(finished)
            except (OSError, ValueError, KeyError) as error:
                out.append({**row, "status": ERROR, NODE: f"source: {describe_exception(error)}"})
                continue
        run_root = mirror_run_root(finished, out_root) if out_root is not None else finished
        out.append(
            {
                **row,
                **extend_one(
                    run_root, config, apply=apply, force=force, build_hint=build_hint, source=source, commit=commit
                ),
            }
        )
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
    hints: Path | None = None,
    commit: str | None = None,
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
        hints: The directory holding ``hints.py``, or None to leave each recorded verdict alone.
        commit: The code revision to record on each re-fold marker.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    build_hint = load_hint_builder(hints) if hints is not None else None
    log = process(mine, config, out_root=out_root, apply=apply, force=force, build_hint=build_hint, commit=commit)

    counts: dict[str, int] = {}
    readings: dict[str, int] = {}
    refolds: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        readings[str(record[NODE])] = readings.get(str(record[NODE]), 0) + 1
        if REFOLD in record:
            refolds[str(record[REFOLD])] = refolds.get(str(record[REFOLD]), 0) + 1

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
        "refolds": refolds,
        "apply": apply,
        "force": force,
        "hints": str(hints) if hints is not None else None,
        "commit": commit,
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
    if args.hints is None and not args.no_refold:
        print(
            "ERROR: --hints is required. VERDICT reads the declaration, and re-folding without one "
            "flags every recording; pass --no-refold to leave the recorded verdicts untouched.",
            file=sys.stderr,
        )
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
            hints=None if args.no_refold else args.hints,
            commit=args.commit,
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
    for refold, number in sorted(summary["refolds"].items()):
        print(f"  refold {refold:<14} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
