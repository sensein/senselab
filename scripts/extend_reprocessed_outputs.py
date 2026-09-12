#!/usr/bin/env python3
r"""Bring a finished triage run's derivable PREPROCESS and TAXONOMY outputs up to date, in place.

    uv run python scripts/extend_reprocessed_outputs.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the same manifest
``scripts/extend_ppg_praat.py`` and ``scripts/extend_clip_amplitudes.py`` take, and the run root is
derived from ``enhanced`` the same way: ``<run_root>/run/store.jsonl`` and ``<run_root>/prov/``.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

Two derivations, both of them functions of what the store and the run tree already hold:

* ``phonation_tracks`` -- F0 over the pre-emphasised stream and the first four formants over
  ``plain``, both read back out of the run's own ``run/streams/``. An append: a corpus run has no
  phonation tracks at all, because ``voice.f0_search_range_hz`` was null when it ran and that null
  suppressed the block.
* ``consensus_taxonomy`` -- reconsolidated from the ``span_yamnet`` and ``span_hear`` ``raw_scores``
  already in the store, now merged on AudioSet node identity rather than on the label string. A
  rewrite: the older reading is retired with a ``wasInvalidatedBy`` edge naming why, because the
  store is append-only and two live consolidations of one recording would contradict each other.

Neither runs a model and neither reads the source recording. **Re-bracketing the onomatopoeic words
is deliberately not here**; it is not derivable from stored state, and
``specs/20260912-extend-reprocessed-outputs/design.md`` says what stops it.

Nothing is written outside the recording's own run. A store the two derivations leave unchanged --
its fingerprint before and after is the test -- is not rewritten at all, so a completed slice re-run
is a pass over ``store.jsonl`` and nothing else, and a task that dies mid-slice restarts where it
stopped.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    RUN_SUBDIR,
    SLICES_SUBDIR,
    export_prov,
    read_manifest,
    read_store,
    rewrite_consensus_taxonomy,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement
from senselab.audio.workflows.triage.nodes.preprocess import PHONATION_TRACKS_MEASUREMENT, phonation_tracks
from senselab.utils.prov_store import ProvStore

_OK = "ok"
_ERROR = "error"
_SKIPPED = "skipped"
_PRESENT = "present"
_CURRENT = "current"
_REWRITTEN = "rewritten"
_ABSENT = "absent"


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
    return parser


def extend_phonation_tracks(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str:
    """Give one finished run the phonation tracks it has none of, or say why it gets none.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run directory the streams and the npz sidecar live under.

    Returns:
        ``ok``, ``present`` when the store already carries the measurement, or the reason no track
        could be measured.
    """
    if find_measurement(store, PHONATION_TRACKS_MEASUREMENT) is not None:
        return _PRESENT
    try:
        phonation_tracks(store, config, run_dir=run_dir)
    except (OSError, ValueError, LookupError) as error:
        return describe_exception(error)
    return _OK


def extend_consensus_taxonomy(store: ProvStore, config: TriageConfig) -> str:
    """Make one finished run's consolidated taxonomy current, or say why it is not touched.

    Args:
        store: The run's store.
        config: The triage configuration.

    Returns:
        ``rewritten``, ``current`` when the store already holds this consolidation, ``absent`` when
        it holds none to make current, or the reason the consolidation could not be recomputed.
    """
    try:
        written = rewrite_consensus_taxonomy(store, config)
    except (OSError, ValueError, LookupError) as error:
        return describe_exception(error)
    if written is not None:
        return _REWRITTEN
    return _CURRENT if find_measurement(store, "consensus_taxonomy") is not None else _ABSENT


def extend_one(run_root: Path, config: TriageConfig) -> dict[str, str]:
    """Apply both derivations to one finished run and write the store only if either changed.

    The fingerprint before and after is the whole convergence argument: a derivation that produces
    records the store already holds is a set-union no-op, which leaves the fingerprint where it was,
    and a store whose fingerprint did not move has nothing to write back.

    Args:
        run_root: The run root.
        config: The triage configuration.

    Returns:
        ``{status, phonation_tracks, consensus_taxonomy}`` — ``ok`` when both derivations reached a
        determinate outcome, ``skipped`` when neither changed the store, ``error`` otherwise.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        reason = describe_exception(error)
        return {"status": _ERROR, PHONATION_TRACKS_MEASUREMENT: reason, "consensus_taxonomy": reason}
    before = store.fingerprint()
    phonation = extend_phonation_tracks(store, config, run_dir=run_root / RUN_SUBDIR)
    taxonomy = extend_consensus_taxonomy(store, config)
    landed = [phonation in (_OK, _PRESENT), taxonomy in (_REWRITTEN, _CURRENT, _ABSENT)]
    if store.fingerprint() != before:
        capture_environments(store, {})
        write_store(store, run_root)
        export_prov(store, run_root)
        status = _OK if all(landed) else _ERROR
    else:
        status = _SKIPPED if all(landed) else _ERROR
    return {"status": status, PHONATION_TRACKS_MEASUREMENT: phonation, "consensus_taxonomy": taxonomy}


def process(rows: Sequence[dict[str, Any]], config: TriageConfig) -> list[dict[str, Any]]:
    """Extend every run named by these rows, one at a time.

    A recording whose store will not open, or whose streams the run tree no longer holds, gets an
    outcome record; its neighbours are unaffected.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            reason = describe_exception(error)
            out.append({**row, "status": _ERROR, PHONATION_TRACKS_MEASUREMENT: reason, "consensus_taxonomy": reason})
            continue
        out.append({**row, **extend_one(run_root, config)})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
) -> dict[str, Any]:
    """Extend every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    log = process(mine, config)

    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"reprocessed-outputs-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Extend every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard reached a determinate outcome, 1 when any row is
        ``error`` — the other stores are written either way — and 2 when the arguments could not be
        resolved and nothing was derived.
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
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<9} {number}")
    return 1 if summary["counts"].get(_ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
