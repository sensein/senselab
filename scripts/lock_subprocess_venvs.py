#!/usr/bin/env python3
"""Compile the committed lock of every subprocess venv senselab builds.

    uv run python scripts/lock_subprocess_venvs.py [NAME ...] [--exclude-newer TIMESTAMP] [--check]

With no NAME, every venv found by :func:`senselab.utils.venv_lock.discover_backends` is locked.
``--exclude-newer`` (RFC 3339, default: now, rounded down to the day) bounds what the resolver
may pick. ``--check`` compiles nothing and exits non-zero when any lock is missing or stale.

Locks are written to ``src/senselab/utils/data/venv_locks/<name>.txt``. The design is in
``specs/20261002-subprocess-venv-locks/design.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from senselab.utils.venv_lock import (
    LOCK_DIR,
    VenvLockError,
    compile_lock,
    discover_backends,
    load_lock,
    lock_path,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="Venv names to lock; default all")
    parser.add_argument("--exclude-newer", default=None, help="RFC 3339 timestamp; default today, 00:00 UTC")
    parser.add_argument("--check", action="store_true", help="Only report missing or stale locks")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Lock the requested venvs, or check them."""
    args = build_parser().parse_args(argv)
    backends = discover_backends()
    names = args.names or sorted(backends)
    unknown = [name for name in names if name not in backends]
    if unknown:
        print(f"unknown venv names: {unknown}; known: {sorted(backends)}", file=sys.stderr)
        return 2
    if args.check:
        bad = 0
        for name in names:
            backend = backends[name]
            try:
                load_lock(name, backend.requirements, backend.python, backend.max_cuda_version)
                print(f"ok      {name}")
            except VenvLockError as error:
                bad += 1
                print(f"STALE   {name}: {error}")
        return 1 if bad else 0
    exclude_newer = args.exclude_newer or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        backend = backends[name]
        try:
            text = compile_lock(
                name,
                backend.requirements,
                backend.python,
                backend.max_cuda_version,
                exclude_newer=exclude_newer,
            )
        except VenvLockError as error:
            failed.append(name)
            print(f"FAILED  {name}: {error}", file=sys.stderr)
            continue
        lock_path(name).write_text(text)
        pins = [line for line in text.splitlines() if line.startswith("# torch:")]
        print(f"locked  {name}  {len(text.splitlines())} lines  {pins[0] if pins else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
