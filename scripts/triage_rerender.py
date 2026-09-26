"""Re-render the summaries for a tree of completed triage runs, from their stores.

REPORT reads the store and writes nothing back, so a renderer fix does not need the graph re-run:
the stores already hold every decision and every span the summary draws.

    uv run python scripts/triage_rerender.py RUN_ROOT [--config FILE] [--limit N] [--dry-run]

``RUN_ROOT`` is the directory holding the run directories — ``run/out`` for a corpus run. Each run
directory is one recording: its ``store.jsonl`` and the sidecars the panels are drawn over. The
summaries are rewritten in place, beside the store, exactly where the run put them.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.report import SUMMARY_STEM, report
from senselab.utils.prov_store import ProvStore

STORE_NAME = "store.jsonl"


def run_dirs(root: Path) -> list[Path]:
    """Every run directory under a tree, found by the store each one holds.

    Args:
        root: The directory to search.

    Returns:
        The run directories, in path order.
    """
    return sorted(path.parent for path in root.rglob(STORE_NAME))


def rerender(run_dir: Path, config: TriageConfig) -> dict[str, Path]:
    """Re-render one run's summary from its store.

    Args:
        run_dir: The run directory, holding ``store.jsonl`` and the sidecars. Its parent's name is
            the run id, which the file does not carry and which the summary's title is built from.
        config: The triage configuration, read for the report format.

    Returns:
        What REPORT wrote, keyed as it names its products.
    """
    store = ProvStore.read_jsonl(run_dir / STORE_NAME, run_id=run_dir.parent.name)
    return report(store, run_dir.parent / SUMMARY_STEM, config, run_dir=run_dir)


def main(argv: list[str] | None = None) -> int:
    """Re-render every summary under one run tree.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every run rendered, 1 when any raised.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path, help="Directory holding the run directories.")
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML merged over the packaged config.")
    parser.add_argument("--limit", type=int, default=None, help="Render at most this many, for a spot check.")
    parser.add_argument("--slice", type=int, default=0, help="This worker's index, for a sharded re-render.")
    parser.add_argument("--slices", type=int, default=1, help="How many workers share the tree.")
    parser.add_argument("--dry-run", action="store_true", help="Count the runs and render none.")
    args = parser.parse_args(argv)

    config = load_triage_config(args.config)
    found = run_dirs(args.run_root)
    mine = found[args.slice :: args.slices][: args.limit]
    print(f"{len(found)} run(s) under {args.run_root}; this worker takes {len(mine)}", flush=True)
    if args.dry_run:
        return 0

    done = failed = 0
    for index, run_dir in enumerate(mine, 1):
        try:
            rerender(run_dir, config)
            done += 1
        except Exception as error:  # noqa: BLE001 -- one unreadable store must not cost the rest
            failed += 1
            print(f"  FAILED {run_dir}: {type(error).__name__}: {error}", flush=True)
            print(traceback.format_exc()[-1500:], flush=True)
        if index % 100 == 0:
            print(f"=== {done} rendered, {failed} failed, {index}/{len(mine)} ===", flush=True)
    print(f"=== DONE: {done} rendered, {failed} failed, of {len(mine)} ===", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
