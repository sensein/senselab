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

``--hints DIR`` names a directory holding a ``hints.py`` exposing ``build_hint(wav)``; routing, the branches,
REDACT and VERDICT all read the hint. Without ``--hints`` the replay runs with the hint the store records
(``hint: recorded`` on the row); a store that records none, or no task, is refused unless ``--no-hint`` is given,
which runs with no hint and is recorded as such. A hint that declares another task than the record, or a task list
that now describes the task otherwise, is recorded as the run's own, and the row's ``identity_change`` names what
moved. A hint that names another participant or session than the run records is refused (the row's
``identity_refused`` names what it would move; the summary counts them, and every refused row's reason is printed
to stderr) unless ``--allow-identity-change`` is given;
a run that records no identity yet records the hint's.

``--out-root DIR`` writes each replayed run into a fresh root under ``DIR`` instead of in place,
with the finished run's ``derivatives/``, its ``run.json`` and the streams the replay only reads reachable through
symlinks. A stream a replayed node writes -- REDACT's ``redacted``, SPEECH's ``separated_*`` -- is
never linked, so that write lands in the new root. Use it when the finished tree must not be
modified.

Every recording is re-hashed before its run is replayed: the row's ``audio`` is ``verified`` only where the digest
matched ADMIT's (the other passes take a recording whose size and modification time match as ``stat_matched``). A
recording that is no longer there cannot be hashed: its row's ``audio`` is ``unverified`` and the replay goes ahead.

A store already carrying this configuration's and revision's replay marker is skipped and not rewritten,
unless ``--force`` is given; its ``present`` row carries the ``review`` state that replay recorded. A row whose
store landed although a replayed node raised is ``errored``, not ``ok``, and names those nodes.

REVIEW is replayed too, under the replaying configuration. Where that configuration does not read
(the LLM check off), an answered reading from before the replay is carried forward when the replayed
store's texts and task context give the same result-cache key; otherwise the row's ``review`` is
``needs_reread``, the summary lists the stems, and the driver exits 3. Follow such a run with
``scripts/extend_llm_review.py`` over those stems and ``scripts/extend_refold.py``.

The design is in ``specs/20260922-replay-decisions-over-a-finished-corpus/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.extend import (
    ERROR,
    ERRORED,
    OK,
    PRESENT,
    REFUSED,
    REVIEW_KEY,
    REVIEW_NEEDS_REREAD,
    REVIEW_WHY_KEY,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    IdentityChangeRefused,
    ReplayOutcome,
    ReplayRefused,
    SliceLog,
    add_config_arguments,
    check_replay_inputs,
    corpus_inputs_digest,
    driver_config,
    export_prov,
    find_replay_marker,
    load_hint_builder,
    logged,
    read_manifest,
    read_store,
    replay_decisions,
    replay_run_id,
    replayed_identity,
    run_root_of,
    source_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.hint_record import hint_digest
from senselab.audio.workflows.triage.identity import (
    DECLARED_PARTICIPANT,
    DECLARED_SESSION,
    RecordingIdentity,
    identity_of,
    recorded_identity,
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
        "--force", action="store_true", help="Replay runs that already carry this configuration's and revision's marker"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument(
        "--no-hint", action="store_true", help="Replay with no hint, where --hints is not given and none is recorded"
    )
    parser.add_argument(
        "--allow-identity-change",
        action="store_true",
        help="Replay a run whose hint names another participant or session than the run records",
    )
    add_config_arguments(parser, allow_profile_change=True)
    return parser


def mirror_root(run_root: Path, out_root: Path, identity: RecordingIdentity) -> Path:
    """Where :func:`mirror_run_root` places a replayed run root, without creating it.

    Args:
        run_root: The finished run root.
        out_root: The tree replayed roots are created under.
        identity: The identity the replay leaves the run with.

    Returns:
        The replayed run root's path.
    """
    return out_root / entity_subdir(identity) / run_root.name


def mirror_run_root(run_root: Path, out_root: Path, identity: RecordingIdentity) -> Path:
    """A writable run root under ``out_root`` whose sidecars read through to the finished one.

    ``derivatives/`` is a real directory holding a symlink per entry of the finished one; the sidecar
    writers a replay runs remove the entry before writing, so a rewrite replaces the link here and never
    follows it. A ``derivatives`` that an earlier mirror made a link to the whole directory is replaced.
    ``streams/`` is a real directory holding a symlink per stream the replay only *reads*; a stream a
    replayed node writes is never linked. ``run.json`` is linked, so a later pass over the mirror checks the
    recording and the profile against the run's record as it would over the finished root.

    Args:
        run_root: The finished run root.
        out_root: The tree replayed roots are created under.
        identity: The identity the replay leaves the run with, placing the root under its entity path.

    Returns:
        The replayed run root, created.
    """
    dest = mirror_root(run_root, out_root, identity)
    run_dir = dest / RUN_SUBDIR
    (run_dir / STREAMS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (dest / RELEASE_SUBDIR).mkdir(parents=True, exist_ok=True)
    finished = run_root / RUN_SUBDIR
    derivatives = run_dir / DERIVATIVES_SUBDIR
    if derivatives.is_symlink():
        derivatives.unlink()
    derivatives.mkdir(parents=True, exist_ok=True)
    for entry in sorted((finished / DERIVATIVES_SUBDIR).glob("*")):
        link = derivatives / entry.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(entry)
    for stream in sorted((finished / STREAMS_SUBDIR).glob("*")):
        if written_by_a_replayed_node(stream.name):
            continue
        link = run_dir / STREAMS_SUBDIR / stream.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(stream)
    log = run_dir / LOG_FILE
    if (finished / LOG_FILE).is_file() and not log.is_symlink() and not log.exists():
        log.symlink_to(finished / LOG_FILE)
    return dest


def replay_identity(
    recorded: RecordingIdentity | None, stem: str, hint: AudioHints | None, *, hint_given: bool
) -> RecordingIdentity:
    """The identity a replay leaves its run with, which places an ``--out-root`` mirror.

    A replay without ``--hints`` into an out-root that a hint-moved replay already wrote places a second mirror under
    the finished run's identity; keep hinted and unhinted replays in separate out-roots.

    Args:
        recorded: The identity the finished run records (:func:`recorded_identity`), or None.
        stem: The recording's file stem.
        hint: The hint the replay was given, if any.
        hint_given: Whether a hint was given (``--hints``).

    Returns:
        The given hint's identity, keeping the recorded participant and session where it declares none
        (:func:`~senselab.audio.workflows.triage.extend.replayed_identity`); else the one the run records; else an
        identity declaring nothing (a run that records none; the replay refuses such a run unless its hint declares
        one).
    """
    if hint_given:
        return replayed_identity(hint, recorded)
    return recorded if recorded is not None else identity_of(None)


def _first_missing(target: Path, out_root: Path) -> Path | None:
    """The outermost directory from ``out_root`` down to ``target`` that does not exist yet, or None where all do.

    Creating ``target`` creates that directory and everything under it, so removing it undoes the creation and
    nothing a previous call made.
    """
    missing = None
    for path in (target, *target.parents):
        if path.exists() or path.is_symlink():
            break
        missing = path
        if path == out_root:
            break
    return missing


def replay_one(
    run_root: Path,
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any] | None,
    out_root: Path | None,
    stem: str,
    commit: str | None,
    force: bool = False,
    no_hint: bool = False,
    allow_profile_change: bool = False,
    allow_identity_change: bool = False,
) -> dict[str, Any]:
    """Replay one finished run, writing its store only once every replayed node has run.

    The store is read under :func:`replay_run_id`, so what the replay writes takes ids distinct
    from the decisions it retires. A run already carrying this configuration's and revision's marker is skipped
    and nothing is written; a run that raises before the write keeps the store it had, so a
    preempted task redoes that recording from the beginning and cannot stack two retirements. An ``out_root``
    mirror is created only once the run's recorded inputs are accepted and the replay will run, so a refused run, or
    one already replayed, leaves none; a replay that raises removes the part of the mirror this call created.

    Args:
        run_root: The finished run root.
        config: The replaying configuration.
        build_hint: The hint populator, or None to replay with no hint.
        out_root: Where to write the replayed run, or None to replay in place.
        stem: The recording's file stem.
        commit: The code revision to record on the marker.
        force: Replay even when this configuration's and revision's marker is already present.
        no_hint: Replay with no hint where none is given or recorded.
        allow_profile_change: Accept a profile whose name differs from the run's.
        allow_identity_change: Accept a hint naming another participant or session than the run records.

    Returns:
        ``{status, replay, ...}`` — ``ok`` when the replay landed, ``errored`` when it landed but a
        replayed node raised (``errored_nodes`` names them), ``present`` when this configuration, revision
        and hint had already replayed the run (with the ``review`` and ``review_why`` that replay's marker
        recorded, where it recorded them), ``refused`` when the run's recorded inputs refuse it
        (:func:`~senselab.audio.workflows.triage.extend.check_replay_inputs`; ``identity_refused`` names the
        participant or session change where that is why), ``error`` when it could not.
    """
    try:
        hint = build_hint(source_of(run_root))[0] if build_hint is not None else None
    except (OSError, ValueError, LookupError, KeyError, IndexError) as error:
        return {"status": ERROR, DERIVATION: f"hint: {describe_exception(error)}"}
    run_id = replay_run_id(run_root, config.config_hash, commit)
    placed: RecordingIdentity | None = None
    try:
        store = read_store(run_root, run_id=run_id)
        target = run_root
        if out_root is not None:
            placed = replay_identity(recorded_identity(store), stem, hint, hint_given=build_hint is not None)
            target = mirror_root(run_root, out_root, placed)
            if (target / RUN_SUBDIR / "store.jsonl").is_file():
                store = read_store(target, run_id=run_id)
    except (OSError, ValueError) as error:
        return {"status": ERROR, DERIVATION: describe_exception(error)}
    try:
        inputs = check_replay_inputs(
            store,
            run_root,
            config,
            hint=hint,
            hint_given=build_hint is not None,
            no_hint=no_hint,
            allow_profile_change=allow_profile_change,
            allow_identity_change=allow_identity_change,
        )
    except IdentityChangeRefused as refusal:
        return {"status": REFUSED, DERIVATION: str(refusal), "identity_refused": refusal.change}
    except ReplayRefused as refusal:
        return {"status": REFUSED, DERIVATION: str(refusal)}
    marker = (
        None
        if force
        else find_replay_marker(
            store, config.config_hashes, commit, hint_digest(inputs.hint), corpus_inputs=corpus_inputs_digest(store)
        )
    )
    if marker is not None:
        recorded = store.get_activity(marker).parameters
        return {
            "status": PRESENT,
            DERIVATION: "this configuration, revision and hint have already replayed this run",
            **(
                {"review": recorded[REVIEW_KEY], "review_why": recorded.get(REVIEW_WHY_KEY)}
                if recorded.get(REVIEW_KEY)
                else {}
            ),
        }
    created = _first_missing(target, out_root) if out_root is not None and placed is not None else None
    try:
        if out_root is not None and placed is not None:
            mirror_run_root(run_root, out_root, placed)
        outcome: ReplayOutcome = replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=target / RUN_SUBDIR,
            artifacts_dir=target / RELEASE_SUBDIR,
            summary_dir=target / SUMMARY_SUBDIR,
            commit=commit,
            inputs=inputs,
        )
    except BaseException as error:
        if created is not None:
            shutil.rmtree(created, ignore_errors=True)  # what this call created goes; links are unlinked, not followed
        if isinstance(error, (OSError, ValueError, LookupError)):
            return {"status": ERROR, DERIVATION: describe_exception(error)}
        raise
    write_store(store, target)
    export_prov(store, target)
    return {
        "status": ERRORED if outcome.errors else OK,
        DERIVATION: "replayed",
        "errored_nodes": sorted(outcome.errors),
        "run_root": str(target),
        "retired": outcome.retired,
        "hint": inputs.hint_from,
        "hint_changed": inputs.changed,
        "audio": inputs.audio,
        "states": outcome.states,
        "errors": outcome.errors,
        "released": sorted(outcome.released),
        "summary": sorted(outcome.summary),
        "review": outcome.review,
        "review_why": outcome.review_why,
        "identity_change": outcome.identity_change,
    }


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    build_hint: Callable[[Path], Any] | None,
    out_root: Path | None,
    commit: str | None,
    log: SliceLog | None = None,
    force: bool = False,
    no_hint: bool = False,
    allow_profile_change: bool = False,
    allow_identity_change: bool = False,
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
        log: Where each row's record is appended as it lands, or None.
        force: Replay runs that already carry this configuration's and revision's marker.
        no_hint: Replay with no hint where none is given or recorded.
        allow_profile_change: Accept a profile whose name differs from a run's.
        allow_identity_change: Accept a hint naming another participant or session than a run records.

    Returns:
        One outcome record per input row, in order.
    """

    def one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            return {**row, "status": ERROR, DERIVATION: describe_exception(error)}
        replayed = replay_one(
            run_root,
            config,
            build_hint=build_hint,
            out_root=out_root,
            stem=str(row["stem"]),
            commit=commit,
            force=force,
            no_hint=no_hint,
            allow_profile_change=allow_profile_change,
            allow_identity_change=allow_identity_change,
        )
        return {**row, **replayed}

    return logged(rows, one, log)


def _participants_of_change(change: dict[str, Any]) -> set[str]:
    """The participants a row's ``identity_change`` moves a run from or to, for the cohort pass to rerun.

    A change counts where the participant or the session the run now records differs from the one it recorded
    before, an empty ``from`` included: since the corpus passes refuse a store that records no identity, a first record
    makes the run visible to the cohort pass for the first time. It names the ``to`` participant, and the ``from`` one
    where there was one. A change of only the task, or of where the identity came from, moves no run, so it names
    nobody.
    """
    before, after = change.get("from") or {}, change.get("to") or {}
    named: set[str] = set()
    for key in (DECLARED_PARTICIPANT, DECLARED_SESSION):
        old, new = before.get(key), after.get(key)
        if new and old != new:
            named.update(str(side[DECLARED_PARTICIPANT]) for side in (before, after) if side.get(DECLARED_PARTICIPANT))
    return named


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
    force: bool = False,
    no_hint: bool = False,
    allow_profile_change: bool = False,
    allow_identity_change: bool = False,
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
        force: Replay runs that already carry this configuration's and revision's marker.
        no_hint: Replay with no hint where none is given or recorded.
        allow_profile_change: Accept a profile whose name differs from a run's.
        allow_identity_change: Accept a hint naming another participant or session than a run records.

    Returns:
        The task's summary: its counts, its parameters, and where its log went. ``identity_participants`` lists the
        participants a recorded participant or session change, a first record included, took a run from or to
        (:func:`_participants_of_change`), which ``extend_cohort.py apply --participants`` reruns. ``refused`` holds
        each refused row's stem and reason.
    """
    started = time.time()
    build_hint = load_hint_builder(hints) if hints is not None else None
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    slices_dir = log_dir / SLICES_SUBDIR
    label = f"replay-decisions-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    rows_log = SliceLog(log_path, slice_index=slice_index, slice_count=slice_count, total=len(mine))
    try:
        log = process(
            mine,
            config,
            build_hint=build_hint,
            out_root=out_root,
            commit=commit,
            log=rows_log,
            force=force,
            no_hint=no_hint,
            allow_profile_change=allow_profile_change,
            allow_identity_change=allow_identity_change,
        )
    finally:
        rows_log.close()

    counts: dict[str, int] = {}
    node_errors: dict[str, int] = {}
    reviews: dict[str, int] = {}
    needs_reread: list[str] = []
    identity_changed: list[str] = []
    identity_participants: set[str] = set()
    identity_refused: list[str] = []
    refused: list[dict[str, str]] = []
    retired = 0
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        retired += int(record.get("retired") or 0)
        if record.get("review"):
            reviews[str(record["review"])] = reviews.get(str(record["review"]), 0) + 1
        if record.get("review") == REVIEW_NEEDS_REREAD:
            needs_reread.append(str(record["stem"]))
        if record.get("identity_change"):
            identity_changed.append(str(record["stem"]))
            identity_participants.update(_participants_of_change(record["identity_change"]))
        if record.get("identity_refused"):
            identity_refused.append(str(record["stem"]))
        if record["status"] == REFUSED:
            refused.append({"stem": str(record["stem"]), "why": str(record.get(DERIVATION) or "")})
        for node in record.get("errors") or {}:
            node_errors[node] = node_errors.get(node, 0) + 1

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
        "reviews": reviews,
        "needs_reread": needs_reread,
        "identity_changed": identity_changed,
        "identity_participants": sorted(identity_participants),
        "identity_refused": identity_refused,
        "refused": refused,
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
        ``error``, ``errored`` or ``refused`` (an identity change refused among them) — the other stores are
        written either way — 2 when the arguments could not be resolved and nothing was read, and 3 when any
        replayed recording's REVIEW reading could not be carried forward and must be read again (the summary's
        ``needs_reread`` names them). 3 takes precedence over 1: a shard with both exits 3. Every refused row's stem
        and reason is printed to stderr either way, and named in the summary (``counts``, ``refused``, and
        ``identity_refused`` for an identity change).
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
            config=driver_config(args),
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
            hints=args.hints,
            out_root=args.out_root,
            commit=args.commit,
            force=args.force,
            no_hint=args.no_hint,
            allow_profile_change=args.allow_profile_change,
            allow_identity_change=args.allow_identity_change,
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
    for state, number in sorted(summary["reviews"].items()):
        print(f"  review {state:<12} {number}")
    if summary["identity_changed"]:
        print(f"  identity recorded anew on {len(summary['identity_changed'])} runs (each row's identity_change)")
        if summary["identity_participants"]:
            print(f"  participants affected: {' '.join(summary['identity_participants'])}")
    for row in summary["refused"]:
        print(f"REFUSED {row['stem']}: {row['why']}", file=sys.stderr)
    if summary["identity_refused"]:
        print(
            f"REFUSED: {len(summary['identity_refused'])} runs whose hint names another participant or session "
            "(each row's identity_refused); check --hints, or pass --allow-identity-change",
            file=sys.stderr,
        )
    if summary["needs_reread"]:
        print(f"REVIEW: {len(summary['needs_reread'])} recordings need the reviewer again", file=sys.stderr)
        return 3
    counts = summary["counts"]
    return 1 if counts.get(ERROR) or counts.get(ERRORED) or counts.get(REFUSED) else 0


if __name__ == "__main__":
    raise SystemExit(main())
