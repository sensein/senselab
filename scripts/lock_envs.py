#!/usr/bin/env python3
"""Compile or check the env locks under packages/senselab/src/senselab/data/env_locks/.

    uv run python scripts/lock_envs.py [NAME ...] [--check]

With no NAME, every env in `inputs.yaml` is used. The default compiles each lock at the pinned `exclude_newer` (needs
network: submit it to ORCD). `--check` compiles nothing and exits non-zero when a lock's `# identity-digest:` is
missing or does not match its inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from senselab_core.runtime.envs import (
    LOCK_DIGEST_TAG,
    EnvLockError,
    compile_lock,
    lock_digest,
    lock_identity_digest,
    read_lock_inputs,
)

LOCK_DIR = Path(__file__).resolve().parents[1] / "packages/senselab/src/senselab/data/env_locks"


def main(argv: list[str] | None = None) -> int:
    """Lock or check the requested envs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    inputs = read_lock_inputs(LOCK_DIR / "inputs.yaml")
    names = args.names or sorted(inputs)
    unknown = [n for n in names if n not in inputs]
    if unknown:
        print(f"unknown env names: {unknown}; known: {sorted(inputs)}", file=sys.stderr)
        return 2
    failed = 0
    for name in names:
        path = LOCK_DIR / f"{name}.txt"
        try:
            if args.check:
                stale = lock_identity_digest(path) != (LOCK_DIGEST_TAG, lock_digest(inputs[name]))
                failed += stale
                print(f"{'STALE' if stale else 'ok':8}{name}")
            else:
                path.write_text(compile_lock(inputs[name]), "utf-8")
                print(f"locked  {name}")
        except (EnvLockError, RuntimeError, OSError) as error:
            failed += 1
            print(f"FAILED  {name}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
