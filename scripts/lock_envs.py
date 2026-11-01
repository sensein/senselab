#!/usr/bin/env python3
"""Compile, check or re-tag the env locks under packages/senselab/src/senselab/data/env_locks/.

    uv run python scripts/lock_envs.py [NAME ...] [--exclude-newer TIMESTAMP] [--check | --retag]

With no NAME, every env in `inputs.yaml` is used. The default compiles each lock (needs network: submit it to ORCD).
`--check` compiles nothing and exits non-zero when a lock's `# identity-digest:` does not match its inputs.
`--retag` rewrites each lock's digest header from its inputs, with no resolution and no install.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml
from senselab_core.runtime.envs import (
    LOCK_DIGEST_TAG,
    EnvLockError,
    compile_lock,
    lock_digest,
    lock_identity_digest,
    retag_lock,
)

LOCK_DIR = Path(__file__).resolve().parents[1] / "packages/senselab/src/senselab/data/env_locks"


def _inputs() -> dict[str, dict]:
    loaded = yaml.safe_load((LOCK_DIR / "inputs.yaml").read_text("utf-8"))
    loaded.pop("format")
    return loaded


def _digest(spec: dict) -> str:
    return lock_digest(spec["requirements"], spec["python"], tuple(spec["max_cuda"]) if "max_cuda" in spec else None)


def main(argv: list[str] | None = None) -> int:
    """Lock, check or re-tag the requested envs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--exclude-newer", default=None, help="RFC 3339 timestamp; default today, 00:00 UTC")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--retag", action="store_true")
    args = parser.parse_args(argv)
    inputs = _inputs()
    names = args.names or sorted(inputs)
    unknown = [n for n in names if n not in inputs]
    if unknown:
        print(f"unknown env names: {unknown}; known: {sorted(inputs)}", file=sys.stderr)
        return 2
    failed = 0
    stamp = args.exclude_newer or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
    for name in names:
        spec, path = inputs[name], LOCK_DIR / f"{name}.txt"
        try:
            if args.check:
                tag, digest = lock_identity_digest(path)
                stale = (tag, digest) != (LOCK_DIGEST_TAG, _digest(spec))
                failed += stale
                print(f"{'STALE' if stale else 'ok':8}{name}")
            elif args.retag:
                path.write_text(retag_lock(path.read_text("utf-8"), _digest(spec)), "utf-8")
                print(f"retagged {name}")
            else:
                cuda = tuple(spec["max_cuda"]) if "max_cuda" in spec else None
                path.write_text(
                    compile_lock(name, spec["requirements"], spec["python"], cuda, exclude_newer=stamp), "utf-8"
                )
                print(f"locked  {name}")
        except (EnvLockError, RuntimeError, OSError) as error:
            failed += 1
            print(f"FAILED  {name}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
