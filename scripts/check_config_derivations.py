"""Fail when a config file carries an inline derivation, or a key of one has no section in config-derivations.md.

Each file in ``CONFIGS`` holds values only; why each top-level key has its value is a ``### <key>`` section under the
file's ``## <path>`` section of config-derivations.md (P-17). A ``derivation`` or ``derivations`` key at any depth is
refused. Without config-derivations.md in the tree (a code branch), only the inline keys are checked.

Usage::

    uv run python scripts/check_config_derivations.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "20261030-140022-senselab-2x-api" / "config-derivations.md"
CONFIGS = (
    "packages/senselab-core/src/senselab_core/runtime/data/equivalence.yaml",
    "packages/senselab-core/src/senselab_core/runtime/data/launch_environment.yaml",
    "packages/senselab-core/src/senselab_core/runtime/data/run_config.yaml",
    "packages/senselab-core/src/senselab_core/runtime/data/store_policy.yaml",
    "packages/senselab/src/senselab/data/env_locks/inputs.yaml",
    "packages/senselab/src/senselab/signal/data/resample.yaml",
    "scripts/data/default_python.yaml",
    "scripts/data/fixture_budget.yaml",
    "scripts/data/license_policy.yaml",
    "scripts/data/size_budget.yaml",
)
INLINE = frozenset({"derivation", "derivations"})


def inline(value: Any, where: str = "") -> list[str]:
    """The dotted path of every ``derivation`` or ``derivations`` key in `value`."""
    if isinstance(value, dict):
        found = [f"{where}{k}" for k in value if k in INLINE]
        return found + [p for k, v in value.items() for p in inline(v, f"{where}{k}.")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in inline(v, f"{where}{i}.")]
    return []


def sections(spec: str) -> dict[str, set[str]]:
    """The ``### `` headings under each ``## `` heading of `spec`."""
    out: dict[str, set[str]] = {}
    current = None
    for line in spec.splitlines():
        if line.startswith("## "):
            current = out.setdefault(line[3:].strip(), set())
        elif line.startswith("### ") and current is not None:
            current.add(line[4:].strip())
    return out


def problems(path: str, raw: Any, spec: dict[str, set[str]] | None) -> list[str]:
    """Why the config at `path` (parsed as `raw`) fails: inline derivations, and keys without a section."""
    out = [f"{path}: {key} is inline; move it to config-derivations.md under ## {path}" for key in inline(raw)]
    if spec is not None and isinstance(raw, dict):
        have = spec.get(path, set())
        out += [
            f"{path}: {key} has no '### {key}' under '## {path}'" for key in raw if key != "format" and key not in have
        ]
    return out


def main(root: Path = ROOT, spec_path: Path = SPEC) -> int:
    """Check every file in ``CONFIGS``; print the problems; return 1 if any."""
    spec = sections(spec_path.read_text()) if spec_path.is_file() else None
    if spec is None:
        print(f"{spec_path.name} is not in this tree: section check skipped")
    found = [p for path in CONFIGS for p in problems(path, yaml.safe_load((root / path).read_text()), spec)]
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
