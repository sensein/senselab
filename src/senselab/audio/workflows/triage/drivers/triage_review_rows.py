r"""Pick the recordings the reviewer must read again after a replay, and split them by whether they have words.

    python -m senselab.audio.workflows.triage.drivers.triage_review_rows rereads REPLAY_MANIFEST \
        --replay-logs LOG_DIR --out ROWS.jsonl
    python -m senselab.audio.workflows.triage.drivers.triage_review_rows split ROWS.jsonl [ROWS.jsonl ...] \
        --out-dir DIR --profile PROFILE [--workers N] [--config C]

``rereads`` reads the slice logs ``extend_replay_decisions`` wrote under ``LOG_DIR/slices/`` and writes the rows of
``REPLAY_MANIFEST`` the reviewer must read, with ``ROWS.counts.json`` beside it counting every review state. By default
those are the rows whose earlier reading could not be carried forward (``needs_reread``): a replay over a corpus that
was reviewed before. ``--all-unread`` takes every row the replay did not itself read (any state but ``read``, and a
row the logs do not name): a corpus whose runs were never reviewed, such as one fresh from a run with the reviewer
off, where every state is ``none``. ``extend_llm_review`` skips a row whose reading already stands. The picked rows
are what ``extend_llm_review`` and ``extend_second_opinion`` take next.

``split`` sorts rows by what REVIEW reads, with REVIEW's own test: a recording whose transcript has words is a
``words`` row, which needs the reviewer's model; any other is a ``nowords`` row, which REVIEW and SECOND_OPINION record
without loading a model, so it can run on CPU. Rows are de-duplicated by stem, first file first. A store that will not
open goes to ``errors.txt`` and to neither list. It writes ``rows_words.jsonl``, ``rows_nowords.jsonl``,
``errors.txt`` and ``split.counts.json`` under ``DIR``. It takes the run's ``--profile``: reading a transcript reads
the profile's values, in every worker it starts.

Only stems, paths and counts are written or printed; no transcript text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.extend import (
    REVIEW_NEEDS_REREAD,
    REVIEW_READ,
    SLICES_SUBDIR,
    activate_in_worker,
    add_config_arguments,
    driver_config,
    read_manifest,
    read_store,
    worker_arguments,
    worker_failure,
)

KEPT_KEYS = ("stem", "enhanced", "source")
"""The manifest fields a picked row keeps."""

WORDS = "words"
NOWORDS = "nowords"
NOT_IN_LOG = "not_in_replay_log"
UNRECORDED = "unrecorded"


def build_parser() -> argparse.ArgumentParser:
    """The CLI: ``rereads`` after a replay, or ``split`` a row list.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    rereads = commands.add_parser("rereads", help="The replayed rows whose review must be read again")
    rereads.add_argument("manifest", type=Path, help="The manifest the replay ran over")
    rereads.add_argument("--replay-logs", type=Path, required=True, help="The replay's --log-dir")
    rereads.add_argument("--out", type=Path, required=True, help="The rows JSONL to write")
    rereads.add_argument(
        "--all-unread", action="store_true", help="Every row the replay did not read, not only needs_reread"
    )
    split = commands.add_parser("split", help="Split rows into words and nowords")
    split.add_argument("rows", type=Path, nargs="+", help="Row JSONL files, each row with stem and enhanced")
    split.add_argument("--out-dir", type=Path, required=True, help="Where the two lists go")
    split.add_argument("--workers", type=int, default=None, help="Processes (default: SLURM_CPUS_PER_TASK, else 4)")
    add_config_arguments(split)
    return parser


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write rows as JSONL, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    partial.replace(path)


def replay_review_states(log_dir: Path) -> dict[str, str]:
    """Each replayed stem's review state, from the replay's slice logs; a later line for a stem wins.

    Args:
        log_dir: The replay's ``--log-dir``.

    Returns:
        ``{stem: review state}``.
    """
    states: dict[str, str] = {}
    for path in sorted((log_dir / SLICES_SUBDIR).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("{"):
                    continue
                record = json.loads(line)
                if "stem" in record:
                    states[str(record["stem"])] = str(record.get("review") or UNRECORDED)
    return states


def pick_rereads(manifest: Path, log_dir: Path, out: Path, *, all_unread: bool = False) -> dict[str, Any]:
    """Write the replayed rows the reviewer must read, and the counts of every review state.

    Args:
        manifest: The manifest the replay ran over.
        log_dir: The replay's ``--log-dir``.
        out: The rows JSONL to write; ``<out stem>.counts.json`` goes beside it.
        all_unread: Pick every row the replay did not read (any state but ``read``, or none logged), not only
            ``needs_reread``.

    Returns:
        The counts written.
    """
    states = replay_review_states(log_dir)
    rows = read_manifest(manifest, required=("stem", "enhanced"))
    counts = Counter(states.get(str(row["stem"]), NOT_IN_LOG) for row in rows)

    def wanted(state: str | None) -> bool:
        return state != REVIEW_READ if all_unread else state == REVIEW_NEEDS_REREAD

    picked = [
        {key: row[key] for key in KEPT_KEYS if key in row} for row in rows if wanted(states.get(str(row["stem"])))
    ]
    _write_rows(out, picked)
    summary = {
        "replay_rows": len(rows),
        "rule": "all_unread" if all_unread else REVIEW_NEEDS_REREAD,
        "picked": len(picked),
        "states": dict(sorted(counts.items())),
    }
    (out.parent / f"{out.stem}.counts.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def classify(row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Whether REVIEW would read words in a row's store.

    Args:
        row: A manifest row with ``enhanced``.

    Returns:
        The row and :data:`WORDS`, :data:`NOWORDS`, or ``error:<type>`` where the store would not open.
    """
    from senselab.audio.workflows.triage.nodes.redact import transcript_texts

    failed = worker_failure()
    if failed is not None:
        return row, f"error:{failed}"
    try:
        store = read_store(Path(str(row["enhanced"])).parents[2])
        original, _ = transcript_texts(store)
    except Exception as error:  # noqa: BLE001 -- a store that will not open is listed, never fatal
        return row, f"error:{type(error).__name__}"
    return row, WORDS if original.strip() else NOWORDS


def split_rows(
    paths: Sequence[Path],
    out_dir: Path,
    *,
    workers: int,
    worker_config: tuple[str | None, str | None] = (None, None),
) -> dict[str, Any]:
    """Split rows into the words and nowords lists.

    Reading a transcript reads the dataset profile's values, so the profile must be active here and in every worker
    (a spawned worker starts with none).

    Args:
        paths: Row JSONL files.
        out_dir: Where the lists go.
        workers: Processes to read stores with.
        worker_config: :func:`~senselab.audio.workflows.triage.extend.worker_arguments` of the driver's configuration,
            activated in each worker.

    Returns:
        The counts written to ``split.counts.json``.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_manifest(path, required=("stem", "enhanced")):
            if row["stem"] not in seen:
                seen.add(str(row["stem"]))
                rows.append(row)
    if workers > 1 and rows:
        with ProcessPoolExecutor(workers, initializer=activate_in_worker, initargs=worker_config) as pool:
            results = list(pool.map(classify, rows, chunksize=8))
    else:
        results = [classify(row) for row in rows]
    groups: dict[str, list[tuple[dict[str, Any], str]]] = {WORDS: [], NOWORDS: [], "error": []}
    for row, group in results:
        groups["error" if group.startswith("error") else group].append((row, group))
    for name in (WORDS, NOWORDS):
        _write_rows(out_dir / f"rows_{name}.jsonl", [row for row, _ in groups[name]])
    (out_dir / "errors.txt").write_text("".join(f"{row['stem']}\t{why}\n" for row, why in groups["error"]))
    counts = {"inputs": [str(path) for path in paths], "rows": len(rows), **{k: len(v) for k, v in groups.items()}}
    (out_dir / "split.counts.json").write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """Pick the rereads, or split rows.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0; 1 where ``split`` found a store that would not open; 2 where the arguments could not be resolved.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "rereads":
            summary = pick_rereads(args.manifest, args.replay_logs, args.out, all_unread=args.all_unread)
            print(json.dumps(summary))
            return 0
        workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
        config = driver_config(args)
        counts = split_rows(args.rows, args.out_dir, workers=workers, worker_config=worker_arguments(args, config))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(counts))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
