"""Restricted steps: the declared ``restricted`` flag, checked at load against inputs, patterns and needs.

A step must declare ``restricted: true`` when it has a registered input that is restricted (flagged, or its
tree, manifest or original in a restricted dir), an
``inputs`` or ``stage`` entry on a data root whose pattern can name a path in a restricted dir (followed
through symlinks in the pattern's fixed prefix), or a ``needs`` entry on a restricted step
(`check_restricted`). A restricted step needs a profile with a restricted results root. A public step's
sandbox masks every restricted dir (`mask_dirs`).

A done restricted step's ``metrics.json`` and ``summary.json`` are copied to the profile's release root only
when the scan hook passes both (`release`); the hook fails closed (`run_scan_hook`). A public step's summary
records its ``input_closure``; verifying a restricted input purges every public result whose closure holds it
(`purge_closure`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from senselab.utils.reproduce import cache, digest, inputs
from senselab.utils.reproduce.spec import RESERVED_ROOT_NAMES, InputSpec, Manifest, Profile, StageSpec, StepSpec

CODE = "E_SPEC_RESTRICTED"
HOOK_MISSING = "E_SCAN_HOOK_MISSING"
RELEASE_REFUSED = "W_RELEASE_REFUSED"
SCAN_TIMEOUT_S = 300.0
SCAN_LOG = "scan.log"
HOOK_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}
RELEASED_FILES = ("metrics.json", "summary.json")
RELEASE_MODE = 0o660
PURGE_CLAIM_TIMEOUT_S = cache.CLAIM_TIMEOUT_S
_KEY = re.compile(r"^[0-9a-f]{64}$")
PROFILE_FIELDS = ("restricted_dirs", "restricted_results_root", "released_metrics_root", "scan_hook")
_WILD = re.compile(r"[*?\[]")


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _spellings(path: str | os.PathLike[str]) -> set[str]:
    return {os.path.normpath(path), os.path.realpath(path)}


def restricted_dir_paths(profile: Profile) -> list[str]:
    """Every restricted dir of the profile, as written and resolved (`inputs.restricted_dir_paths`)."""
    return inputs.restricted_dir_paths(profile)


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


def input_is_restricted(entry: inputs.RegisteredInput, profile: Profile) -> bool:
    """Whether a registered input is flagged restricted, or its tree, manifest or original is in a restricted dir."""
    return entry.restricted or inputs.in_restricted_dir(entry, profile)


def _pattern_violation(step: StepSpec, profile: Profile, dirs: Sequence[str]) -> bool:
    entries: list[InputSpec | StageSpec] = [*step.inputs, *step.stage]
    for entry in entries:
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
            any(input_is_restricted(registry[i], profile) for i in step.lock_inputs if i in registry)
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


# --- the scan hook -----------------------------------------------------------------------------------------


def hook_missing(profile: Profile) -> bool:
    """Whether the scan hook cannot run: none named, `argv[0]` not executable, or an absolute argument absent."""
    hook = profile.scan_hook
    if not hook:
        return True
    program = hook[0]
    if not (os.path.isabs(program) and os.path.isfile(program) and os.access(program, os.X_OK)):
        return True
    return any(os.path.isabs(arg) and not os.path.exists(arg) for arg in hook[1:])


def _log_fd(log: Path | None) -> int:
    if log is None:
        return os.open(os.devnull, os.O_WRONLY)
    return os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o660)


def run_scan_hook(profile: Profile, path: str | os.PathLike[str], log: Path | None = None) -> bool:
    """Run ``scan_hook + [path]`` outside the sandbox; True only when it exits 0 within `SCAN_TIMEOUT_S`.

    The hook gets `HOOK_ENV` only, no stdin, and its own process group (killed on a timeout). Its output goes to
    ``log`` (appended), or nowhere. Fails closed: no hook, a non-zero exit, a timeout, a missing or
    non-executable program or any ``OSError`` is False.
    """
    if not profile.scan_hook:
        return False
    try:
        fd = _log_fd(log)
    except OSError:
        return False
    try:
        child = subprocess.Popen(
            [*profile.scan_hook, os.fspath(path)],
            env=dict(HOOK_ENV),
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=fd,
            start_new_session=True,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    finally:
        os.close(fd)
    try:
        return child.wait(timeout=SCAN_TIMEOUT_S) == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            child.kill()
        child.wait()
        return False


def scan_values(profile: Profile, values: object, directory: Path, log: Path | None = None) -> bool:
    """Write ``values`` as JSON to a temporary file in ``directory``, run the hook on it, remove it."""
    path = directory / f".scan-{uuid.uuid4().hex}.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(values, handle, sort_keys=True)
        return run_scan_hook(profile, path, log)
    except (OSError, TypeError, ValueError):
        return False
    finally:
        try:
            path.unlink()
        except OSError:
            pass


# --- the release copy --------------------------------------------------------------------------------------


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, RELEASE_MODE)
    try:
        try:
            os.write(fd, data)
            os.fchmod(fd, RELEASE_MODE)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def release(result: Path, key: str, profile: Profile, summary: dict[str, object], log: Path | None = None) -> bool:
    """Scan a done restricted step's ``metrics.json`` and its final ``summary`` and, when both are clean, copy them.

    Args:
        result: The step's result dir; ``R/metrics.json`` has passed the driver's validator.
        key: The step key, naming ``<released_metrics_root>/<key>/``.
        profile: The profile (its ``scan_hook`` and ``released_metrics_root``).
        summary: The summary the step will write, with ``"released": true``.
        log: Where the hook's output goes.

    Returns:
        True when both files were scanned clean and copied (atomically, mode ``0660``). False otherwise, and
        then an earlier copy for the key is removed (`withdraw`); a symlink in the key's place, or a
        ``metrics.json`` that is a symlink, is refused and left as it is.
    """
    root = profile.released_metrics_root
    if root is None or not _KEY.match(key):
        return False
    target = root / key
    if target.is_symlink():
        return False
    staged = result / f".release-{uuid.uuid4().hex}"
    released = False
    try:
        os.mkdir(staged, 0o700)
        fd = os.open(result / RELEASED_FILES[0], os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            metrics = handle.read()
        data = {RELEASED_FILES[0]: metrics, RELEASED_FILES[1]: _summary_bytes(summary)}
        for name, content in data.items():
            _write(staged / name, content)
        if all(run_scan_hook(profile, staged / name, log) for name in RELEASED_FILES):
            target.mkdir(parents=True, exist_ok=True)
            for name, content in data.items():
                _write(target / name, content)
            released = True
    except OSError:
        released = False
    finally:
        shutil.rmtree(staged, ignore_errors=True)
        if not released:
            withdraw(profile, key)
    return released


def withdraw(profile: Profile, key: str) -> None:
    """Remove ``<released_metrics_root>/<key>/`` when it is a directory (a symlink there is left alone)."""
    root = profile.released_metrics_root
    if root is None or not _KEY.match(key):
        return
    target = root / key
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)


def _summary_bytes(summary: dict[str, object]) -> bytes:
    """The bytes `cache.write_summary` writes for ``summary``."""
    return (json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


# --- the closure purge -------------------------------------------------------------------------------------


def _closure_of(out: Path) -> list[str] | None:
    """The summary's ``input_closure`` (empty when it has none); None when there is no readable summary."""
    try:
        fd = os.open(out / cache.SUMMARY, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            data = json.loads(handle.read())
    except (OSError, ValueError):
        return None
    closure = data.get("input_closure") if isinstance(data, dict) else None
    return [c for c in closure if isinstance(c, str)] if isinstance(closure, list) else []


def _run_links(runs_root: Path) -> list[Path]:
    """Every ``<runs_root>/<date>/<analysis>/<step>`` symlink."""
    found: list[Path] = []
    if not runs_root.is_dir():
        return found
    for day in (p for p in runs_root.iterdir() if p.is_dir() and not p.is_symlink()):
        for analysis in (p for p in day.iterdir() if p.is_dir() and not p.is_symlink()):
            found += [p for p in analysis.iterdir() if p.is_symlink()]
    return found


def purge_closure(profile: Profile, input_id: str) -> int:
    """Remove every public result whose summary's ``input_closure`` holds ``input_id``, and its run links.

    A result with no readable summary (a run in flight, or one that died) is taken under its claim too, which
    waits for a live run: it is removed unless its summary, read again there, has a closure without the id.
    A held claim (`PURGE_CLAIM_TIMEOUT_S`) skips that result and the purge goes on. Stateless: no earlier
    record is consulted, so a second call removes nothing.

    Returns:
        The number of result dirs removed.

    Raises:
        CacheError: ``E_CLAIM_HELD`` after the other results, when a claim stayed held.
        OSError: When a dir or link cannot be removed.
    """
    root = profile.results_root
    if not root.is_dir():
        return 0
    purged, held = 0, False
    links = None
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not _KEY.match(entry.name) or not entry.is_dir(follow_symlinks=False):
            continue
        out = Path(entry.path)
        closure = _closure_of(out)
        if closure is not None and input_id not in closure:
            continue
        try:
            with cache.claim(root, entry.name, timeout=PURGE_CLAIM_TIMEOUT_S):
                closure = _closure_of(out)
                if (closure is not None and input_id not in closure) or not out.is_dir():
                    continue
                real = os.path.realpath(out)
                links = _run_links(profile.runs_root) if links is None else links
                for link in links:
                    if link.is_symlink() and os.path.realpath(link) == real:
                        link.unlink()
                shutil.rmtree(out)
                purged += 1
        except cache.CacheError:
            held = True
    if held:
        raise cache.CacheError("E_CLAIM_HELD")
    return purged


def closure_now_restricted(closure: Iterable[str], base_dir: Path, profile: Profile) -> bool:
    """Whether any id of a closure is now restricted (`input_is_restricted`) or its registry entry unreadable."""
    for input_id in closure:
        try:
            entry = inputs.read_registry_entry(base_dir, input_id)[0]
        except inputs.InputError:
            return True
        if input_is_restricted(entry, profile):
            return True
    return False
