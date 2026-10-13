"""Write ``triage.tsv``, one flat row per recording, and its column dictionary beside it.

    uv run python scripts/triage_table.py --decisions DIR/triage_decisions.parquet
        --evidence DIR/triage_evidence.parquet --run-root OUT --bids-root BIDS --out DIR/triage.tsv
        [--manifest MANIFEST.jsonl] [--vectors DIR/recording_vectors.parquet]

The decision tables are the ones ``scripts/triage_recording_vectors.py --merge`` writes; ``--run-root``
is the tree they were built over. The columns are ``data/triage_table.yaml``; their BIDS sidecar is
written beside the TSV as ``triage.json``. The table carries no transcript text. Nothing is written,
and the exit status is 2, where the table would not be one row per recording
(``triage_table.coverage_problems``): a stem, run directory or source path held twice, a row with no
source path, or, with ``--manifest``, a run asked for and absent or present and not asked for. The
manifest also supplies the source path of a run whose store records none. ``--vectors`` is the
``recording_vectors.parquet`` built over the same tree, read for the whole-file SNR, noise floor and
SQUIM columns; without it they are empty. Every store is read, never written.

    uv run python scripts/triage_table.py --sidecar-only DIR/triage.json

writes only the sidecar, building no table.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from senselab.audio.workflows.triage import triage_table
from senselab.utils import fastio


def main(argv: list[str] | None = None) -> int:
    """Build the table.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when rows (or, with ``--sidecar-only``, the sidecar) were written, 1 when no rows were, 2 when
        coverage failed and nothing was written.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--sidecar-only"]:
        sidecar = argparse.ArgumentParser(description="Write only triage.json, the BIDS sidecar of triage.tsv.")
        sidecar.add_argument("--sidecar-only", type=Path, required=True, help="Where triage.json goes.")
        path = sidecar.parse_args(argv).sidecar_only
        triage_table.write_sidecar(path)
        print(f"{len(triage_table.sidecar())} columns -> {path}")
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", type=Path, required=True, help="triage_decisions.parquet")
    parser.add_argument("--evidence", type=Path, required=True, help="triage_evidence.parquet")
    parser.add_argument("--run-root", type=Path, required=True, help="The run tree the tables were built over.")
    parser.add_argument("--bids-root", type=Path, required=True, help="The BIDS root source paths are relative to.")
    parser.add_argument("--out", type=Path, required=True, help="Where triage.tsv goes.")
    parser.add_argument("--manifest", type=Path, help="The run manifest (JSONL of stem, source) to check against.")
    parser.add_argument("--vectors", type=Path, help="recording_vectors.parquet, for the whole-file quality columns.")
    parser.add_argument("--threads", type=int, default=fastio.DEFAULT_THREADS, help="Run-tree reads in flight.")
    args = parser.parse_args(argv)

    decisions = pq.read_table(args.decisions)
    evidence = pq.read_table(args.evidence)
    verdicts: Counter[str] = Counter()
    releases: Counter[str] = Counter()

    def counted(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for row in rows:
            verdicts[str(row["verdict"])] += 1
            releases[str(row["release"] or "")] += 1
            yield row

    sources = triage_table.manifest_sources(args.manifest) if args.manifest else None
    vectors = None
    if args.vectors is not None:
        wanted = ["stem", *triage_table.VECTOR_COLUMNS]
        present = set(pq.read_schema(args.vectors).names)
        vectors = pq.read_table(args.vectors, columns=[name for name in wanted if name in present])
    built = list(
        counted(
            triage_table.rows(
                decisions,
                evidence,
                args.run_root,
                args.bids_root,
                threads=args.threads,
                sources=sources,
                vectors=vectors,
            )
        )
    )
    problems = triage_table.coverage_problems(decisions, built, None if sources is None else sources.keys())
    if problems:
        for problem in problems:
            print(f"COVERAGE: {problem}", file=sys.stderr)
        return 2
    written = triage_table.write_tsv(built, args.out)
    print(f"{written} rows -> {args.out}")
    print(f"verdict {dict(sorted(verdicts.items()))}")
    print(f"release {dict(sorted(releases.items()))}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
