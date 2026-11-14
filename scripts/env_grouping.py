#!/usr/bin/env python3
"""Propose which backend envs could share one env, and write ``data/env_grouping.yaml`` (``env-grouping@1``).

    uv run python scripts/env_grouping.py [NAME ...] [--out PATH]

Every pair of envs in ``env_locks/inputs.yaml`` is resolved as the union of their requirements (the same resolution a
lock uses, `compile_lock`), and a pair is compatible when the union resolves, both use one Python and one CUDA ceiling.
Groups are then built greedily in name order, each member added only when the whole group still resolves. The file
records the proposal; no env is merged by it. Needs network: submit it to ORCD.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from senselab_core.runtime.envs import LockInputs, compile_lock, lock_pins, read_lock_inputs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "packages/senselab/src/senselab/data"
FORMAT = "env-grouping@1"

Resolve = Callable[[LockInputs], str]
"""Lock text for the inputs, or a ``RuntimeError`` / ``ValueError`` naming why they do not resolve."""


def union(envs: Mapping[str, LockInputs], names: tuple[str, ...]) -> LockInputs | str:
    """The inputs of the envs `names` resolved together, or the reason they cannot share one."""
    members = [envs[n] for n in names]
    if len({e.python for e in members}) > 1:
        return f"python differs: {sorted({e.python for e in members})}"
    if len({e.max_cuda for e in members}) > 1:
        return f"max_cuda differs: {sorted({str(e.max_cuda) for e in members})}"
    requirements = tuple(dict.fromkeys(r for e in members for r in e.requirements))
    excluded = tuple(dict.fromkeys(x for e in members for x in e.excluded))
    return replace(members[0], name="+".join(names), requirements=requirements, excluded=excluded)


def resolve_group(envs: Mapping[str, LockInputs], names: tuple[str, ...], resolve: Resolve) -> dict[str, str] | str:
    """The ``package -> version`` pins of `names` resolved as one env, or why they do not resolve."""
    inputs = union(envs, names)
    if isinstance(inputs, str):
        return inputs
    try:
        return lock_pins(resolve(inputs))
    except (RuntimeError, ValueError) as error:
        return " ".join(str(error).split())[:300]


def propose(envs: Mapping[str, LockInputs], resolve: Resolve = compile_lock) -> dict[str, Any]:
    """Return the ``env-grouping`` document body: every pair's verdict and the greedy groups."""
    names = sorted(envs)
    separate = {n: lock_pins(resolve(envs[n])) for n in names}
    cache: dict[tuple[str, ...], dict[str, str] | str] = {}

    def group(members: tuple[str, ...]) -> dict[str, str] | str:
        if len(members) == 1:
            return separate[members[0]]
        return cache.setdefault(members, resolve_group(envs, members, resolve))

    pairs = []
    for a, b in itertools.combinations(names, 2):
        got = group((a, b))
        entry: dict[str, Any] = {"envs": [a, b], "compatible": not isinstance(got, str)}
        if isinstance(got, str):
            entry["conflict"] = got
        else:
            entry["packages"] = len(got)
            entry["saved_packages"] = len(separate[a]) + len(separate[b]) - len(got)
        pairs.append(entry)
    groups: list[list[str]] = []
    for name in names:
        for members in groups:
            if not isinstance(group((*members, name)), str):
                members.append(name)
                break
        else:
            groups.append([name])
    return {
        "pairs": pairs,
        "groups": [
            {
                "envs": members,
                "packages": len(group(tuple(members))),
                "separate_packages": sum(len(separate[m]) for m in members),
            }
            for members in groups
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Write the proposal for the named envs, or every env in `inputs.yaml`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*")
    parser.add_argument("--out", type=Path, default=DATA / "env_grouping.yaml")
    args = parser.parse_args(argv)
    inputs = read_lock_inputs(DATA / "env_locks" / "inputs.yaml")
    unknown = [n for n in args.names if n not in inputs]
    if unknown:
        print(f"unknown env names: {unknown}; known: {sorted(inputs)}", file=sys.stderr)
        return 2
    chosen = {n: inputs[n] for n in args.names or inputs}
    body = {"format": FORMAT, "generated": date.today().isoformat(), "adopted": False, **propose(chosen)}
    args.out.write_text(yaml.safe_dump(body, sort_keys=False, width=120))
    print(f"wrote {args.out}: {len(body['groups'])} groups over {len(chosen)} envs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
