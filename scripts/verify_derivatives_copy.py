#!/usr/bin/env python3
"""Check that a derivatives copy holds what its source tree holds: one threaded pass per tree.

    uv run python scripts/verify_derivatives_copy.py --src SRC --dst DST \
        [--threads 64] [--needle /orcd/scratch] [--json OUT]

``SRC`` is walked following symlinks, as ``find -L`` would, so a run root whose streams link into
an earlier tree is counted at its targets. ``DST`` is walked without following them, and only the
``sub-*`` entries directly under it are counted, so the top-level files written beside the copy are
not compared. Each pass counts files, bytes, symlinks, ``store.jsonl`` files and the stores whose
text contains ``--needle``, and every file is stat-ed and every store read inside the walk's threads.

One line summarises the comparison::

    files src=… dst=… | bytes src=… dst=… | symlinks in dst=… | stores mentioning …: N of M | MATCH

The exit status is 0 only when the file and byte counts match, ``DST`` holds no symlink and
nothing in either tree failed to read. ``--json`` also writes both tallies and the match.

The design is in ``specs/20261002-threaded-walk/design.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from senselab.utils import fastio

STORE_NAME = "store.jsonl"
DST_PREFIX = "sub-"


@dataclass
class Tally:
    """What one pass over a tree counted.

    Attributes:
        files: Regular files, and under a followed walk the targets of file symlinks.
        bytes: The summed sizes of those files.
        symlinks: Symlinks met, to files or (when not followed) to directories.
        stores: ``store.jsonl`` files.
        stores_with_needle: Stores whose text contains the needle.
        errors: Files that could not be stat-ed or read, and directories that could not be listed.
        seconds: Wall time of the pass.
    """

    files: int = 0
    bytes: int = 0
    symlinks: int = 0
    stores: int = 0
    stores_with_needle: int = 0
    errors: int = 0
    seconds: float = 0.0


def _visitor(follow: bool, needle: bytes) -> Callable[[str], Tally]:
    """The per-file work run inside the walk's threads.

    Args:
        follow: Whether the walk follows symlinks; a followed symlink is counted at its target.
        needle: The bytes a store is searched for.

    Returns:
        A function from a path to that one entry's tally.
    """

    def visit(path: str) -> Tally:
        one = Tally()
        try:
            if os.path.islink(path):
                one.symlinks = 1
            if follow or not one.symlinks:
                one.bytes = os.stat(path, follow_symlinks=follow).st_size
                one.files = 1
            if os.path.basename(path) == STORE_NAME:
                one.stores = 1
                with open(path, "rb") as handle:
                    one.stores_with_needle = int(needle in handle.read())
        except OSError as error:
            one.errors = 1
            print(f"{path}: {error}", file=sys.stderr)
        return one

    return visit


def census(top: Path, *, threads: int, follow: bool, needle: str, only_prefix: str | None = None) -> Tally:
    """Count one tree in a single threaded pass.

    Args:
        top: The tree's root.
        threads: Size of the walk's thread pool.
        follow: Whether symlinks are followed.
        needle: The text a store is searched for.
        only_prefix: When given, only the entries directly under ``top`` whose names start with it.

    Returns:
        The tree's tally.
    """
    total = Tally()
    lock = threading.Lock()

    def listing_failed(path: str, error: OSError) -> None:
        print(f"{path}: {error}", file=sys.stderr)
        with lock:
            total.errors += 1

    keep = None if only_prefix is None else (lambda name: name.startswith(only_prefix))
    started = time.monotonic()
    for one in fastio.map_files(
        top,
        _visitor(follow, needle.encode()),
        threads=threads,
        follow_symlinks=follow,
        keep=keep,
        on_error=listing_failed,
        links_as_files=not follow,
    ):
        total.files += one.files
        total.bytes += one.bytes
        total.symlinks += one.symlinks
        total.stores += one.stores
        total.stores_with_needle += one.stores_with_needle
        total.errors += one.errors
    total.seconds = round(time.monotonic() - started, 1)
    return total


def summary(src: Tally, dst: Tally, needle: str) -> tuple[str, bool]:
    """The one-line comparison of two tallies and whether the counts match.

    Args:
        src: The source tree's tally.
        dst: The copy's tally.
        needle: The text the stores were searched for.

    Returns:
        The line and whether files and bytes agree.
    """
    same = src.files == dst.files and src.bytes == dst.bytes
    line = (
        f"files src={src.files} dst={dst.files} | bytes src={src.bytes} dst={dst.bytes} | "
        f"symlinks in dst={dst.symlinks} | stores mentioning {needle}: {dst.stores_with_needle} "
        f"of {dst.stores} | {'MATCH' if same else 'MISMATCH'}"
    )
    return line, same


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description="Check a derivatives copy against its source tree.")
    parser.add_argument("--src", type=Path, required=True, help="The source tree, walked following symlinks")
    parser.add_argument("--dst", type=Path, required=True, help="The copy; only its sub-* entries are counted")
    parser.add_argument("--threads", type=int, default=64, help="Size of each walk's thread pool")
    parser.add_argument("--needle", default="/orcd/scratch", help="Text whose presence in a store is counted")
    parser.add_argument("--json", type=Path, default=None, help="Also write both tallies and the match here")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both passes, print the comparison and return the exit status.

    Args:
        argv: The arguments; None reads them from the command line.

    Returns:
        0 when the counts match, the copy holds no symlink and nothing failed to read; 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    src = census(args.src, threads=args.threads, follow=True, needle=args.needle)
    print("src", json.dumps(asdict(src)), flush=True)
    dst = census(args.dst, threads=args.threads, follow=False, needle=args.needle, only_prefix=DST_PREFIX)
    print("dst", json.dumps(asdict(dst)), flush=True)
    line, same = summary(src, dst, args.needle)
    print(line, flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps({"src": asdict(src), "dst": asdict(dst), "match": same}, indent=1))
    return 0 if same and not dst.symlinks and not src.errors and not dst.errors else 1


if __name__ == "__main__":
    sys.exit(main())
