"""Restricted steps: the declared ``restricted`` flag, checked at load against inputs, patterns and needs.

A step must declare ``restricted: true`` when it has a registered input with ``restricted: true``, an
``inputs`` or ``stage`` entry on a data root whose pattern can name a path in a restricted dir (followed
through symlinks in the pattern's fixed prefix), or a ``needs`` entry on a restricted step
(`check_restricted`). A restricted step needs a profile with a restricted results root. A public step's
sandbox masks every restricted dir (`mask_dirs`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from senselab.utils.reproduce import digest, inputs
from senselab.utils.reproduce.spec import RESERVED_ROOT_NAMES, Manifest, Profile, StepSpec

CODE = "E_SPEC_RESTRICTED"
PROFILE_FIELDS = ("restricted_dirs", "restricted_results_root", "released_metrics_root", "scan_hook")
_WILD = re.compile(r"[*?\[]")


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _spellings(path: str | os.PathLike[str]) -> set[str]:
    return {os.path.normpath(path), os.path.realpath(path)}


def restricted_dir_paths(profile: Profile) -> list[str]:
    """Every restricted dir of the profile, as written and resolved."""
    return sorted({s for d in profile.restricted_dirs for s in _spellings(d)})


def mask_dirs(profile: Profile) -> list[Path]:
    """The dirs a public step's sandbox masks; `sandbox.sandbox_script` masks each at its resolved path."""
    return list(profile.restricted_dirs)


def _reaches_below(parts: Sequence[str], rel: Sequence[str]) -> bool:
    """Whether pattern components ``parts`` can match a path that runs through every component of ``rel``."""
    if not rel:
        return bool(parts)
    if not parts:
        return False
    if parts[0] == "**":
        return True
    return digest.pattern_matches(rel[0], parts[0]) and _reaches_below(parts[1:], rel[1:])


def pattern_reaches(top: str | os.PathLike[str], pattern: str, dirs: Iterable[str]) -> bool:
    """Whether ``pattern`` under ``top`` can name a file in one of ``dirs``.

    The pattern's wildcard-free leading components are resolved as `digest` resolves them (a symlink there
    is followed); the rest is matched component by component against each dir below that prefix.
    """
    parts = pattern.split("/")
    fixed: list[str] = []
    for part in parts:
        if part == "**" or _WILD.search(part):
            break
        fixed.append(part)
    rest = parts[len(fixed) :]
    for base in _spellings(os.path.join(os.fspath(top), *fixed)):
        for d in dirs:
            if _under(base, d):
                return True
            if _under(d, base) and _reaches_below(rest, os.path.relpath(d, base).split("/")):
                return True
    return False


def _pattern_violation(step: StepSpec, profile: Profile, dirs: Sequence[str]) -> bool:
    for entry in [*step.inputs, *step.stage]:
        if entry.root.startswith("up:") or entry.root in RESERVED_ROOT_NAMES or entry.root not in profile.roots:
            continue
        if any(pattern_reaches(profile.roots[entry.root], p, dirs) for p in entry.patterns):
            return True
    return False


def violations(
    steps: Sequence[StepSpec], registry: Mapping[str, inputs.RegisteredInput], profile: Profile
) -> list[str]:
    """The names of steps whose ``restricted`` declaration is refused (`check_restricted`), sorted."""
    flagged = {s.name for s in steps if s.restricted}
    dirs = restricted_dir_paths(profile)
    bad = set()
    for step in steps:
        if step.restricted:
            if profile.restricted_results_root is None:
                bad.add(step.name)
            continue
        if (
            any(registry[i].restricted for i in step.lock_inputs if i in registry)
            or any(n in flagged for n in step.needs)
            or _pattern_violation(step, profile, dirs)
        ):
            bad.add(step.name)
    return sorted(bad)


def check_restricted(
    steps: Sequence[StepSpec], registry: Mapping[str, inputs.RegisteredInput], profile: Profile
) -> None:
    """Refuse a step set whose ``restricted`` declarations do not hold.

    Args:
        steps: Every step of the manifest set.
        registry: The registry entries of the steps' ``lock_inputs``, by id; an id missing here is refused
            later by the step's own lock checks.
        profile: The caller's profile.

    Raises:
        ValueError: ``E_SPEC_RESTRICTED: [<step names>]`` for a public step with a restricted registered
            input, a data-root pattern reaching a restricted dir or a need on a restricted step, and for a
            restricted step on a profile with no restricted results root.
    """
    bad = violations(steps, registry, profile)
    if bad:
        raise ValueError(f"{CODE}: {bad}")


def step_registry(steps: Iterable[StepSpec], base_dir: Path) -> dict[str, inputs.RegisteredInput]:
    """The registry entries of every ``lock_inputs`` id of ``steps`` that can be read; the others are left out."""
    out: dict[str, inputs.RegisteredInput] = {}
    for input_id in sorted({i for s in steps for i in s.lock_inputs}):
        try:
            out[input_id] = inputs.read_registry_entry(base_dir, input_id)[0]
        except inputs.InputError:
            continue
    return out


def check_manifest_set(manifests: Sequence[Manifest], base_dir: Path, profile: Profile) -> None:
    """`check_restricted` over every step of ``manifests``, with the registry read from ``base_dir``."""
    steps = [s for m in manifests for s in m.steps]
    check_restricted(steps, step_registry(steps, base_dir), profile)
