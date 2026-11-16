"""The ``python -m senselab.utils.reproduce`` command line.

Subcommands:

- ``lock``: run the lock builders for each env, in order (first resolve, torch, body, overlays), and
  write each lock's sha256 (and a uv-export body's command) into the manifest.
- ``prefetch``: lint, then check every declared Hugging Face and URL weights file against its sha256 once
  (`weights.verify_hf`, `weights.verify_url`; printed as ``weights …`` and ``url-weights …`` count lines),
  then check out every step (with the profile's deny patterns), build or reuse each env with the profile's
  checks, record a changed ``wheels.lock`` sha256 in the manifest, and install R libraries. CUDA sdists are
  built as batch jobs (``build-one``) first, unless ``--local-builds``.
  ``--sources-only`` checks out draft manifests with the empty patch set and nothing else (through the
  ``--profile``'s mirror map when one is given).
- ``build-one``: build or reuse one sdist's wheel and record the new ``wheels.lock`` sha256.
- ``run``: run steps in DAG order. ``--dry-run`` reports each step's key and ``will-execute``, ``skipped`` or
  ``blocked`` with its env and source state, executing and writing nothing (exit 1 when a step is blocked).
  ``--step`` runs just those (one batch job; ``--pin`` refuses a manifest changed since submission with
  ``E_MANIFEST_CHANGED``). ``--finalize --submission <id>`` cancels that submission's jobs (queued or running,
  found by recorded id and by their ``--comment`` tag) and settles its ``pending`` rows. A local run refuses
  (``E_SUBMIT_OVERLAP``) while a submission is live in its run dir.
- ``submit``: lint, then one held batch job per step chained with ``afterok``, then release
  (`slurm.submit`). When the inputs gate has a code for any step, nothing is submitted but one
  ``verify-inputs`` job per input whose pre-check found a mismatch or a missing file; each step prints as
  ``never-ran`` with its codes, and each such job as ``verify-submitted <id> <job id>``.
- ``lint``: check manifests and patches against the profile.
- ``status``: print each step's key, status, exit and codes from ``run.json`` and ``summary.json``.
- ``verify-inputs``: re-hash registered inputs and append a record each (`inputs.verify`); one line of counts
  per input id. ``--registry-only`` checks the registry files and their manifests and hashes nothing.

Exit codes, for every subcommand:

- ``0`` (`EXIT_OK`): success; for ``lint``, no findings; for ``run`` and ``status``, every step ran or was
  skipped with exit 0.
- ``1`` (`EXIT_FAILED`): ``lint`` findings (one line each on stdout), a step that did not succeed, or a
  failed lock, prefetch, build or submission (one ``error:`` line on stderr).
- ``2`` (`EXIT_USAGE`): bad arguments, or a named file or run dir that cannot be read.
- ``3`` (`EXIT_PENDING`): ``status`` only: no step failed, but at least one is still ``pending``.

``run`` and ``status`` print one line per step, tab-separated: name, the first 8 hex of the key (``-``
when there is none), status, exit (``-``), and the comma-joined codes (``-``). A code that is not an
``E_*``/``W_*`` name prints as ``?``; nothing else from a run reaches the terminal.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pydantic
import yaml

from senselab.utils.reproduce import (
    bands,
    build,
    cache,
    env,
    inputs,
    lock,
    lockgen,
    restricted,
    run,
    sandbox,
    slurm,
    source,
    weights,
)
from senselab.utils.reproduce.dag import DagError, toposort
from senselab.utils.reproduce.locks import TORCH_NAMES, ReproduceLockError, file_sha256, normalize_name
from senselab.utils.reproduce.spec import (
    UNDECLARED_ROOT,
    EnvSpec,
    Manifest,
    Profile,
    StepSpec,
    load_manifest,
    load_profile,
    load_sources,
    parse_manifest,
    rewrite_manifest,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PENDING = 3
WHEELHOUSE = "wheelhouse"
SDISTS = "sdists"
WHEELS_LOCK = "wheels.lock"
_SUBCOMMANDS = ("lock", "prefetch", "build-one", "run", "submit", "lint", "status", "verify-inputs")
_CODE = re.compile(r"^[EW]_[A-Z0-9_]{1,64}$")
_STEP = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_KEY = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = frozenset({"executed", "skipped", "failed", "never-ran", run.PENDING})


class UsageError(Exception):
    """A command line or file that cannot be used; exit `EXIT_USAGE`."""


# --- shared options ----------------------------------------------------------------------------------


def _date(text: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from None


def _site_options(parser: argparse.ArgumentParser, *, profile_required: bool = True) -> None:
    parser.add_argument("--profile", type=Path, required=profile_required, help="the site profile (YAML or JSON)")
    parser.add_argument("--base-dir", type=Path, help="the manifest repository (default: the manifest's git top level)")
    parser.add_argument("--cache", type=Path, help=f"the reproduce cache (default: ${slurm.CACHE_VARIABLE})")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser with its eight subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m senselab.utils.reproduce",
        description="Reproduce analyses: pinned code, locked venvs, sandboxed steps.",
        epilog="exit codes: 0 ok, 1 failed or findings, 2 usage or unreadable input, 3 status: still pending",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(_SUBCOMMANDS) + "}")

    lock = sub.add_parser("lock", help="compile each env's locks and record their sha256s")
    lock.add_argument("manifests", nargs="+", type=Path)
    _site_options(lock)

    prefetch = sub.add_parser("prefetch", help="mirrors, worktrees, venvs, wheels and R libraries")
    prefetch.add_argument("manifests", nargs="+", type=Path)
    _site_options(prefetch, profile_required=False)
    prefetch.add_argument("--sources-only", action="store_true", help="only check draft manifests out")
    prefetch.add_argument("--rebuild", action="store_true", help="rebuild every sdist wheel and venv")
    prefetch.add_argument("--local-builds", action="store_true", help="build CUDA sdists here, not as batch jobs")

    one = sub.add_parser("build-one", help="build or reuse one sdist wheel (a batch job's entry point)")
    one.add_argument("manifest", type=Path)
    _site_options(one)
    one.add_argument("--step", required=True)
    one.add_argument("--sdist", required=True)
    one.add_argument("--rebuild", action="store_true")

    run_cmd = sub.add_parser("run", help="run steps in DAG order")
    run_cmd.add_argument("manifests", nargs="+", type=Path)
    _site_options(run_cmd)
    run_cmd.add_argument("--date", type=_date, help="the run date (default: today, UTC)")
    run_cmd.add_argument("--step", action="append", help="run only this step (repeatable); upstream must be done")
    run_cmd.add_argument(
        "--finalize",
        action="store_true",
        help="settle pending rows as never-ran; with --submission, first cancel that submission's jobs, "
        "including any still queued or running",
    )
    run_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="print each step's key and whether it would execute, skip or be blocked; execute and write nothing",
    )
    run_cmd.add_argument("--shard", type=int, help="run only this shard of the one sharded --step (a work-pool item)")
    run_cmd.add_argument(
        "--host-probe",
        action="store_true",
        help="with --step and --shard: run the step's host_probe on this host, in its sandbox; exit 0 when it passes",
    )
    run_cmd.add_argument(
        "--pool",
        metavar="STEP",
        help="a sharded step's pool job: publish its shards, run the work-pool controller until they end, settle",
    )
    run_cmd.add_argument("--submission", help="the submission id (set by submit; printed by submit)")
    run_cmd.add_argument("--pin", action="append", default=[], help="a manifest's sha256 at submission, in order")
    run_cmd.add_argument("--timeout", type=float, help="seconds before a step is killed")

    submit = sub.add_parser("submit", help="one batch job per step, chained with afterok")
    submit.add_argument("manifests", nargs="+", type=Path)
    _site_options(submit)
    submit.add_argument("--date", type=_date, help="the run date (default: today, UTC)")

    lint = sub.add_parser("lint", help="check manifests, their sources locks and patches")
    lint.add_argument("manifests", nargs="*", type=Path)
    _site_options(lint, profile_required=False)
    lint.add_argument("--lock", type=Path, help="lint this sources lock alone")
    lint.add_argument("--skip-locks", action="store_true", help="skip the sources-lock checks (W_LOCKS_SKIPPED)")

    status = sub.add_parser("status", help="per step: key, status, exit, codes")
    status.add_argument("run_dir", type=Path, help="<runs_root>/<date>/<analysis>")

    verify = sub.add_parser("verify-inputs", help="re-hash registered inputs and record each result")
    verify.add_argument("ids", nargs="*", help="registered input ids")
    verify.add_argument("--profile", type=Path, required=True, help="the site profile (YAML or JSON)")
    verify.add_argument("--base-dir", type=Path, required=True, help="the manifest repository")
    verify.add_argument("--all", action="store_true", help="every registry file under <base-dir>/inputs")
    verify.add_argument("--registry-only", action="store_true", help="check registry files only; hash nothing")
    return parser


def subcommands() -> list[str]:
    """The subcommand names, in help order."""
    return list(_SUBCOMMANDS)


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.UTC).date()


def _cache_dir(args: argparse.Namespace) -> Path:
    value = args.cache or os.environ.get(slurm.CACHE_VARIABLE)
    if not value:
        raise UsageError(f"pass --cache or set {slurm.CACHE_VARIABLE}")
    return Path(value).absolute()


def _base_dir(args: argparse.Namespace, first: Path) -> Path:
    if args.base_dir is not None:
        return Path(args.base_dir).absolute()
    home = os.environ.get("HOME", "/")
    git_env = {"PATH": build.SYSTEM_PATH, "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1", "HOME": home}
    done = subprocess.run(
        ["git", "-C", str(first.absolute().parent), "rev-parse", "--show-toplevel"],
        env=git_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0 or not done.stdout.strip():
        raise UsageError(f"{first} is not in a git repository; pass --base-dir")
    return Path(done.stdout.strip())


def _require(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise UsageError(f"{path}: no such file")


def _profile(args: argparse.Namespace) -> Profile:
    if args.profile is None:
        raise UsageError("--profile is required")
    _require([args.profile])
    return load_profile(args.profile)


def _manifests(paths: Sequence[Path], profile: Profile) -> list[Manifest]:
    _require(paths)
    return [load_manifest(p, headline_rule=profile.policy.headline_rule) for p in paths]


def _deny(profile: Profile) -> list[str]:
    """The profile's deny patterns; none when its policy does not require a file and it names none."""
    return [] if profile.deny_patterns_file is None else source.load_deny_patterns(profile.deny_patterns_file)


def _uv_environ(cache_dir: Path) -> dict[str, str]:
    values = {key: os.environ.get(key, "") for key in slurm.UV_EXPORTS}
    if not all(values.values()):
        uv = lockgen.uv_executable()
        values["UV_CACHE_DIR"] = build._uv_location(uv, "UV_CACHE_DIR", ["cache", "dir"])
        values["UV_PYTHON_INSTALL_DIR"] = build._uv_location(uv, "UV_PYTHON_INSTALL_DIR", ["python", "dir"])
    return {**values, slurm.CACHE_VARIABLE: os.fspath(cache_dir)}


# --- output -------------------------------------------------------------------------------------------------


def status_line(step: str, key: str | None, status: str, exit_code: int | None, codes: Sequence[Any]) -> str:
    """One tab-separated status line; anything not of a known shape prints as ``?``."""
    name = step if _STEP.match(step) else "?"
    key8 = key[:8] if isinstance(key, str) and _KEY.match(key) else "-"
    shown = status if status in _STATUSES else "?"
    code_text = ",".join(c if isinstance(c, str) and _CODE.match(c) else "?" for c in codes) or "-"
    exit_text = "-" if exit_code is None else str(exit_code) if type(exit_code) is int else "?"
    return f"{name}\t{key8}\t{shown}\t{exit_text}\t{code_text}"


def plan_line(plan: run.Plan) -> str:
    """One dry-run line: name, key8 (``-``), status, ``env:<state>``, ``src:<state>``, codes (``-``)."""
    name = plan.step if _STEP.match(plan.step) else "?"
    key8 = plan.key[:8] if isinstance(plan.key, str) and _KEY.match(plan.key) else "-"
    codes = ",".join(c if _CODE.match(c) else "?" for c in plan.codes) or "-"
    return f"{name}\t{key8}\t{plan.status}\tenv:{plan.env}\tsrc:{plan.src}\t{codes}"


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


# --- lint -----------------------------------------------------------------------------------------------


def _validation_text(error: pydantic.ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in error.errors(include_input=False)]
    return "; ".join(parts)


def _patch_findings(where: str, step: StepSpec, base: Path, deny: Sequence[str]) -> list[str]:
    refs = [*step.patches, *(p for s in step.env.sdist_allow for p in s.patches)]
    out = []
    for ref in refs:
        path = base / ref.file
        try:
            data = path.read_bytes()
        except OSError:
            out.append(f"{where}: patch-missing {ref.file}")
            continue
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            out.append(f"{where}: patch-sha {ref.file}")
            continue
        try:
            hits = source.lint_patch(data, deny)
        except source.ReproduceSourceError as error:
            out.append(f"{where}: {error.code} {ref.file}")
            continue
        for hit in hits:
            pattern = f" pattern {hit.pattern}" if hit.pattern >= 0 else ""
            out.append(f"{where}: patch-lint {hit.code} line {hit.line}{pattern} in {ref.file}")
    return out


def _named_values(manifest: Manifest) -> list[tuple[int | None, str, str]]:
    """``(step index or None for the manifest, field, value)`` for each name that leaves the driver in clear.

    The analysis and step names name dirs and batch jobs; the ``metrics_schema`` labels, counts and positive class
    are copied into ``metrics.json``.
    """
    out: list[tuple[int | None, str, str]] = [(None, "name", manifest.name)]
    for index, step in enumerate(manifest.steps):
        out.append((index, "name", step.name))
        schema = step.metrics_schema
        for field in ("headline_labels", "counts", "per_class"):
            out += [(index, f"metrics_schema.{field}[{i}]", v) for i, v in enumerate(getattr(schema, field))]
        if schema.positive is not None:
            out.append((index, "metrics_schema.positive", schema.positive))
    return out


def _name_findings(path: Path, manifest: Manifest, deny: Sequence[str]) -> tuple[list[str], set[int]]:
    """``name-lint deny-pattern`` findings for the manifest's names, and the indices of steps whose name is denied.

    A finding names the field and the pattern index, never the value; a step whose own name is denied is
    named ``steps[<i>]``.
    """
    values = _named_values(manifest)
    try:
        hits = source.deny_value_hits([v for _, _, v in values], deny)
    except source.ReproduceSourceError as error:
        return [f"{path}: {error.code}"], set()
    hidden = {values[i][0] for i, _ in hits if values[i][1] == "name" and values[i][0] is not None}
    out = []
    for value_index, pattern in hits:
        step, field, _ = values[value_index]
        if step is None:
            where = str(path)
        else:
            where = f"{path}:steps[{step}]" if step in hidden else f"{path}:{manifest.steps[step].name}"
        out.append(f"{where}: name-lint deny-pattern {field} pattern {pattern}")
    return out, {i for i in hidden if i is not None}


def _checked_submodules(step: StepSpec, cache_dir: Path | None) -> dict[str, str] | None:
    """The submodule commits of the step's prefetched checkout, when one exists; None otherwise."""
    if cache_dir is None:
        return None
    try:
        co = source.prefetched_checkout(step, cache=cache_dir)
    except source.ReproduceSourceError:
        return None
    return co.submodules if step.submodules else {}


def _lock_findings(
    path: Path,
    manifest: Manifest,
    base: Path,
    cache_dir: Path | None,
    profile: Profile | None,
    shown: Mapping[str, str] | None = None,
) -> list[str]:
    """Each manifest's sources lock: its sha256, and every step's pins and ``lock_inputs`` against it.

    Submodule commits are compared when ``cache_dir`` holds the step's prefetched checkout. With a profile,
    each registered input's manifest is read for the link rule (`inputs.unregistered_links`). ``shown`` maps a
    step name to how a finding names it (``steps[<i>]`` for a denied name, `_name_findings`).
    """
    shown = shown or {}
    try:
        sources_lock, _ = lock.load_lock(base, manifest.sources)
    except lock.LockError as error:
        return [f"{path}: {error.code} {manifest.sources.lock}"]
    out = []
    for step in manifest.steps:
        where = f"{path}:{shown.get(step.name, step.name)}"
        found = lock.step_findings(step, sources_lock, _checked_submodules(step, cache_dir))
        out += [f"{where}: {code} {detail}" for code, detail in found]
        for input_id in step.lock_inputs:
            try:
                lock.resolve_inputs(base, sources_lock, [input_id])
            except lock.LockError as error:
                if error.code != "E_LOCK_REF":
                    out.append(f"{where}: {error.code} {input_id}")
        if profile is not None:
            out += _link_findings(where, step, sources_lock, base, profile)
    return out


def _link_findings(
    where: str, step: StepSpec, sources_lock: lock.SourcesLock, base: Path, profile: Profile
) -> list[str]:
    """``input-link-unregistered <id> <count>`` for each declared input with links outside the step's inputs."""
    try:
        entries = lock.resolve_inputs(base, sources_lock, step.lock_inputs)
    except lock.LockError:
        return []
    out = []
    for input_id, entry in entries.items():
        others = [e for i, e in entries.items() if i != input_id]
        try:
            count = inputs.unregistered_links(entry, profile, others)
        except inputs.InputError as error:
            out.append(f"{where}: {error.code} {input_id}")
            continue
        if count:
            out.append(f"{where}: input-link-unregistered {input_id} {count}")
    return out


def _mirror_findings(where: str, step: StepSpec, profile: Profile, label: str | None = None) -> list[str]:
    """``repo-unmapped`` for a step repository the mirror map lacks; under strict, ``submodule-unmapped``.

    ``label`` is how the findings name the step (its name by default; ``steps[<i>]`` for a denied name).
    """
    name = step.name if label is None else label
    table = profile.mirror_map
    if not (table or profile.mirror_map_strict):
        return []
    mapped = table.get(source.canonical_repo(step.repo))
    if mapped is None:
        return [f"{where}: repo-unmapped {name}"]
    if not source.mapped_has_commit(mapped, step.commit):
        return [f"{where}: {source.COMMIT_MISSING} {name}"]
    if not (profile.mirror_map_strict and step.submodules):
        return []
    try:
        urls = source.submodule_urls(mapped, step.commit, url=step.repo, mirror_map=table)
    except source.ReproduceSourceError as error:
        return [f"{where}: {error.code} {name}"]
    missing = [u for u in urls if source.canonical_repo(u) not in table]
    return [f"{where}: submodule-unmapped {name} {len(missing)}"] if missing else []


def lint_findings(
    paths: Sequence[Path],
    profile_path: Path,
    base: Path,
    *,
    skip_locks: bool = False,
    cache_dir: Path | None = None,
) -> list[str]:
    """Check manifests, their sources locks and their patches against a profile.

    Args:
        paths: The manifest files.
        profile_path: The profile file.
        base: The manifest repository.
        skip_locks: Leave out the sources-lock checks and report the warning ``W_LOCKS_SKIPPED`` instead.
        cache_dir: The reproduce cache; a step's submodule commits are checked against its lock entry when
            the cache holds its prefetched checkout.

    Returns:
        One line per finding, ``<where>: <code> <detail>``; empty when clean. Codes: ``profile-invalid``,
        ``deny-patterns-unreadable``, ``manifest-invalid``, ``dag-invalid``, the `slurm.step_findings`
        codes, ``adapter-missing``, ``patch-missing``, ``patch-sha``, ``patch-lint`` (with the lint code,
        line and pattern index, never patch text), ``name-lint deny-pattern <field> pattern <i>`` for an analysis
        or step name or a ``metrics_schema`` label, count or positive class that a deny pattern matches (the
        value is never printed, and a step whose name matches is shown as ``steps[<i>]``),
        ``deny-pattern-invalid``, and the sources-lock codes
        (``E_LOCK_SHA``/``E_LOCK_INVALID`` with the lock path, ``E_LOCK_REF``/``E_LOCK_MISMATCH`` with the
        step field, ``E_LOCK_STALE``/``E_INPUT_REGISTRY`` with the input id), ``input-link-unregistered`` with
        the input id and the number of its manifest's links whose target lies in no input the step declares,
        ``scan-hook-missing`` when the profile's scan hook cannot run (`restricted.hook_missing`),
        ``E_SPEC_RESTRICTED`` for a step whose ``restricted`` declaration is refused (`restricted.violations`);
        with a mirror map (or ``mirror_map_strict``), ``repo-unmapped <step>`` for a step repository it lacks,
        ``commit-missing <step>`` when the mapped repository lacks the step's commit,
        and under strict ``submodule-unmapped <step> <count>`` for the submodule URLs at the step's commit,
        read from the mapped repository, that it lacks (or the `source` code when that cannot be read);
        and the warning
        ``W_BAND_AMIN_WIDE <step> <metric>`` for a band whose ``a_min`` exceeds ``max_width_frac`` times its
        declared lift, which alone does not fail a command (`blocking`).

    Raises:
        UsageError: If a named file does not exist.
    """
    _require([*paths, profile_path])
    out: list[str] = []
    profile: Profile | None = None
    try:
        profile = load_profile(profile_path)
    except pydantic.ValidationError as error:
        out.append(f"{profile_path}: profile-invalid {_validation_text(error)}")
    except (ValueError, yaml.YAMLError) as error:
        out.append(f"{profile_path}: profile-invalid {error}")
    deny: list[str] = []
    if profile is not None:
        out += [f"{profile_path}: {f.code} {f.detail}" for f in slurm.profile_findings(profile)]
        if profile.scan_hook and restricted.hook_missing(profile):
            out.append(f"{profile_path}: scan-hook-missing")
        try:
            deny = _deny(profile)
        except OSError:
            out.append(f"{profile_path}: deny-patterns-unreadable {profile.deny_patterns_file}")
    loaded: list[tuple[Path, Manifest]] = []
    for path in paths:
        try:
            rule = "balanced_accuracy_only" if profile is None else profile.policy.headline_rule
            loaded.append((path, load_manifest(path, headline_rule=rule)))
        except pydantic.ValidationError as error:
            out.append(f"{path}: manifest-invalid {_validation_text(error)}")
        except (ValueError, yaml.YAMLError) as error:
            out.append(f"{path}: manifest-invalid {error}")
    named: dict[Path, tuple[list[str], set[int]]] = {path: _name_findings(path, m, deny) for path, m in loaded}
    shown = {
        step.name: f"steps[{index}]"
        for path, m in loaded
        for index, step in enumerate(m.steps)
        if index in named[path][1]
    }
    if len(loaded) == len(paths):
        try:
            toposort([s for _, m in loaded for s in m.steps])
        except DagError as error:
            text = f"{error.code}: {', '.join(shown.get(s, s) for s in error.steps)}" if shown else str(error)
            out.append(f"{', '.join(str(p) for p in paths)}: dag-invalid {text}")
    refused: set[str] = set()
    steps = [s for _, m in loaded for s in m.steps]
    registry = restricted.step_registry(steps, base)
    uncovered = restricted.undeclared_roots(steps, registry)
    unpurged: set[str] = set()
    if profile is not None:
        refused = set(restricted.violations(steps, registry, profile))
        unpurged = set(restricted.unpurged_inputs(steps, registry, profile))
    for path, manifest in loaded:
        out += named[path][0]
        for step in manifest.steps:
            label = shown.get(step.name, step.name)
            where = f"{path}:{label}"
            if step.name in refused:
                out.append(f"{where}: {restricted.CODE}")
            if step.name in uncovered:
                out.append(f"{where}: {UNDECLARED_ROOT} {' '.join(uncovered[step.name])}")
            out += [f"{where}: {restricted.UNPURGED} {i}" for i in sorted(unpurged & set(step.lock_inputs))]
            if profile is not None and step.extract and restricted.hook_missing(profile):
                out.append(f"{where}: scan-hook-missing")
            if profile is not None:
                out += [f"{where}: {f.code} {f.detail}" for f in slurm.step_findings(step, profile)]
                out += _mirror_findings(where, step, profile, label)
            if not (base / step.adapter).is_file():
                out.append(f"{where}: adapter-missing {step.adapter}")
            for expect in step.expect:
                band = expect.band
                if band is not None and band.floor is not None and bands.amin_too_wide(band, band.floor):
                    out.append(f"{where}: W_BAND_AMIN_WIDE {label} {expect.metric}")
            out += _patch_findings(where, step, base, deny)
    if skip_locks:
        out.append(f"{', '.join(str(p) for p in paths)}: W_LOCKS_SKIPPED")
    else:
        for path, manifest in loaded:
            out += _lock_findings(path, manifest, base, cache_dir, profile, shown)
    return out


def blocking(findings: Sequence[str]) -> bool:
    """Whether any lint finding is an error rather than a ``W_`` warning."""
    return any(not f.partition(": ")[2].startswith("W_") for f in findings)


# --- manifest updates -----------------------------------------------------------------------------------


def _edit_envs(path: Path, old: EnvSpec, change: Callable[[dict[str, Any]], None]) -> None:
    """Apply ``change`` to the raw ``env`` mapping of every step of a manifest file whose env equals ``old``.

    The file is matched structurally (its headline rule was checked when the caller loaded it for a profile).
    """
    manifest = load_manifest(path, headline_rule="kind_vocabulary")

    def edit(data: Any) -> None:  # noqa: ANN401
        for raw, step in zip(data["steps"], manifest.steps):
            if step.env == old:
                change(raw["env"])

    rewrite_manifest(path, edit)


def default_wheels_lock(env_spec: EnvSpec) -> str:
    """Where a first prefetch records wheels when the env names no ``wheels`` file: beside the body lock."""
    return str(PurePosixPath(env_spec.body.lock).with_name(WHEELS_LOCK))


def _record_wheels(path: Path, env_spec: EnvSpec, sha256: str) -> None:
    rel = env_spec.wheels.lock if env_spec.wheels is not None else default_wheels_lock(env_spec)

    def change(raw: dict[str, Any]) -> None:
        raw["wheels"] = {"lock": rel, "lock_sha256": sha256}

    _edit_envs(path, env_spec, change)


def _wheels_location(env_spec: EnvSpec, base: Path) -> tuple[Path, str | None]:
    if env_spec.wheels is not None:
        return base / env_spec.wheels.lock, env_spec.wheels.lock_sha256
    return base / default_wheels_lock(env_spec), None


# --- lock ----------------------------------------------------------------------------------------------


def _export_options(cmd: Sequence[str] | None) -> tuple[list[str], list[str]]:
    extras: list[str] = []
    groups: list[str] = []
    items = list(cmd or [])
    for index, item in enumerate(items):
        for flag, into in (("--extra", extras), ("--group", groups)):
            if item == flag and index + 1 < len(items):
                into.append(items[index + 1])
            elif item.startswith(flag + "="):
                into.append(item.split("=", 1)[1])
    return extras, groups


def _torch_roots(uv_lock: Path) -> list[str]:
    packages = tomllib.loads(uv_lock.read_text()).get("package", [])
    names = {normalize_name(p["name"]) for p in packages if isinstance(p, dict) and "name" in p}
    return sorted(TORCH_NAMES & names)


def _lock_env(step: StepSpec, base: Path, cache_dir: Path, deny: Sequence[str], profile: Profile) -> dict[str, Any]:
    """Run the builders for one env in order: first resolve, torch, body, overlays. Returns the field updates."""
    e = step.env
    worktree: list[Path] = []

    def project() -> Path:
        if not worktree:
            co = source.checkout_step(
                step,
                cache=cache_dir,
                base_dir=base,
                deny_patterns=deny,
                mirror_map=profile.mirror_map,
                strict=profile.mirror_map_strict,
            )
            worktree.append(co.worktree)
        return worktree[0]

    written: dict[str, Any] = {}
    if e.body.source == "requirements.in":
        if e.first_resolve is not None:
            lockgen.first_resolve(e, base)
            written["first_resolve"] = e.first_resolve.file
        if e.torch is not None:
            lockgen.torch_lock_from_index(e, base)
            written["torch"] = e.torch.lock
        lockgen.body_lock_from_requirements(e, base)
    else:
        if e.torch is not None:
            uv_lock = project() / "uv.lock"
            lockgen.torch_lock_from_uvlock(e, base, uv_lock, _torch_roots(uv_lock))
            written["torch"] = e.torch.lock
        extras, groups = _export_options(e.body.export_cmd)
        argv = lockgen.body_lock_from_uvlock_export(e, base, project(), extras=extras, groups=groups)
        written["export_cmd"] = ["uv", *argv[1:]]
    written["body"] = e.body.lock
    if e.test is not None:
        lockgen.overlay_lock("test", e, base, project_dir=project())
        written["test"] = e.test.lock
    if e.notebook_tools is not None:
        requirements = str(PurePosixPath(e.notebook_tools.lock).with_suffix(".in"))
        lockgen.overlay_lock("notebook_tools", e, base, requirements=requirements)
        written["notebook_tools"] = e.notebook_tools.lock
    return written


def _apply_lock_updates(raw: dict[str, Any], written: dict[str, Any], base: Path) -> None:
    for field, rel in written.items():
        if field == "export_cmd":
            raw["body"]["export_cmd"] = rel
        elif field == "first_resolve":
            raw["first_resolve"]["sha256"] = file_sha256(base / rel)
        else:
            raw[field]["lock_sha256"] = file_sha256(base / rel)


def _cmd_lock(args: argparse.Namespace) -> int:
    profile = _profile(args)
    paths = list(args.manifests)
    _require(paths)
    base, cache_dir = _base_dir(args, paths[0]), _cache_dir(args)
    deny = _deny(profile)
    for path in paths:
        done: list[EnvSpec] = []
        for step in load_manifest(path, headline_rule=profile.policy.headline_rule).steps:
            if any(step.env == e for e in done):
                continue
            done.append(step.env)
            written = _lock_env(step, base, cache_dir, deny, profile)

            def change(raw: dict[str, Any], written: dict[str, Any] = written) -> None:
                _apply_lock_updates(raw, written, base)

            _edit_envs(path, step.env, change)
            print(f"{step.name}\tlocked\t{','.join(sorted(written))}")
    return EXIT_OK


# --- prefetch and build-one -----------------------------------------------------------------------------


def _next_cuda_build(
    paths: Sequence[Path],
    base: Path,
    wheelhouse: Path,
    rebuild: bool,
    done: set[tuple[str, str, str]],
    profile: Profile,
) -> tuple[Path, StepSpec, Any] | None:
    for path in paths:
        for step in load_manifest(path, headline_rule=profile.policy.headline_rule).steps:
            for sdist in step.env.sdist_allow:
                tag = (str(path), step.name, sdist.name)
                if tag in done or not build.needs_nvcc(sdist):
                    continue
                if not rebuild:
                    lock_path, expected = _wheels_location(step.env, base)
                    found = (
                        build.reuse_wheel(
                            lock_path,
                            sdist,
                            wheelhouse,
                            step.env.torch,
                            python=step.env.python_exact,
                            expected_sha256=expected,
                        )
                        if lock_path.exists()
                        else None
                    )
                    if found is not None:
                        done.add(tag)
                        continue
                return path, step, sdist
    return None


def _batch_builds(
    args: argparse.Namespace, paths: Sequence[Path], profile: Profile, base: Path, cache_dir: Path
) -> None:
    """Build every CUDA sdist without a reusable wheel as a ``build-one`` batch job, one at a time, waiting."""
    done: set[tuple[str, str, str]] = set()
    environ = _uv_environ(cache_dir)
    date = _today()
    while True:
        todo = _next_cuda_build(paths, base, cache_dir / WHEELHOUSE, args.rebuild, done, profile)
        if todo is None:
            return
        path, step, sdist = todo
        script = build.render_build_job(
            profile,
            sdist,
            manifest=path.absolute(),
            profile_path=Path(args.profile).absolute(),
            step=step.name,
            rebuild=args.rebuild,
            base_dir=base,
            date=date,
            environ=environ,
        )
        slurm.prepare_log_dir(profile, date)
        slurm.submit_and_wait(
            script, name=f"reproduce-build-{sdist.name}-{uuid.uuid4().hex[:8]}", umask=slurm.submit_umask(profile)
        )
        done.add((str(path), step.name, sdist.name))


def _cmd_prefetch(args: argparse.Namespace) -> int:
    with sandbox.tool_masks(_tool_masks(args)):
        return _cmd_prefetch_masked(args)


def _tool_masks(args: argparse.Namespace) -> list[Path]:
    """The restricted dirs of the command's profile; none without a profile (``prefetch --sources-only``)."""
    return [] if args.profile is None else _profile(args).restricted_dirs


def _cmd_prefetch_masked(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    _require(paths)
    cache_dir = _cache_dir(args)
    if args.sources_only:
        site = None if args.profile is None else _profile(args)
        mirror_map = {} if site is None else site.mirror_map
        strict = site is not None and site.mirror_map_strict
        for path in paths:
            draft = load_sources(path)
            for name, co in source.checkout_sources(
                draft, cache=cache_dir, mirror_map=mirror_map, strict=strict
            ).items():
                print(f"{name}\t{co.commit[:8]}\tsources")
        return EXIT_OK
    profile = _profile(args)
    base = _base_dir(args, paths[0])
    findings = lint_findings(paths, Path(args.profile), base, cache_dir=cache_dir)
    if findings:
        print("\n".join(findings))
    if blocking(findings):
        return EXIT_FAILED
    status, detail = sandbox.sandbox_status()
    if status != "ok":
        code = "E_SANDBOX_UNAVAILABLE" if status == "sweep-refused" else "E_NO_USERNS"
        _error(f"{code}: {detail}")
        return EXIT_FAILED
    rule = profile.policy.headline_rule
    manifests = [load_manifest(p, headline_rule=rule) for p in paths]
    steps = [s for m in manifests for s in m.steps]
    if any(s.hf_weights or s.url_weights for s in steps):
        registry = _weights_registry(manifests, base)
        stamps = cache_dir / weights.STAMPS_DIR
        hf = any(s.hf_weights for s in steps)
        failed = weights.verify_hf(steps, profile, registry, stamps=stamps) if hf else []
        url = any(s.url_weights for s in steps)
        failed += weights.verify_url(steps, profile, registry, stamps=stamps) if url else []
        if failed:
            _error("; ".join(failed))
            return EXIT_FAILED
    if not args.local_builds:
        _batch_builds(args, paths, profile, base, cache_dir)
    deny = _deny(profile)
    for path in paths:
        manifest = load_manifest(path, headline_rule=rule)
        done: list[tuple[EnvSpec, env.PrefetchResult]] = []
        for step in manifest.steps:
            co = source.checkout_step(
                step,
                cache=cache_dir,
                base_dir=base,
                deny_patterns=deny,
                mirror_map=profile.mirror_map,
                strict=profile.mirror_map_strict,
            )
            sources_lock, _ = lock.load_lock(base, manifest.sources)
            submodules = co.submodules if step.submodules else {}
            found = [f"{c} {d}" for c, d in lock.step_findings(step, sources_lock, submodules)]
            if found:
                _error(f"{path}:{step.name}: {', '.join(found)}")
                return EXIT_FAILED
            result = next((r for e, r in done if e == step.env), None)
            if result is None:
                result = _prefetch_env(step, co.worktree, manifest.name, path, profile, base, cache_dir, args.rebuild)
                done.append((step.env, result))
            if step.kind in cache.R_KINDS:
                venv = result.derived["notebook_tools"].path if step.kind == "ir_notebook" else None
                env.prefetch_r(step.env, cache_dir, profile, kind=step.kind, venv=venv)
            print(f"{step.name}\tprefetched")
    return EXIT_OK


def _weights_registry(manifests: Sequence[Manifest], base: Path) -> dict[str, inputs.RegisteredInput]:
    """The entries ``HFWeights.input`` names, and the ``lock_inputs`` of ``url_weights`` steps, as the locks pin them.

    An id the lock cannot resolve is left out, so its files are hashed (lint reports the lock problem).
    """
    registry: dict[str, inputs.RegisteredInput] = {}
    for manifest in manifests:
        ids = sorted({w.input for s in manifest.steps for w in s.hf_weights if w.input is not None})
        ids = sorted({*ids, *(i for s in manifest.steps if s.url_weights for i in s.lock_inputs)})
        if not ids:
            continue
        try:
            sources_lock, _ = lock.load_lock(base, manifest.sources)
        except lock.LockError:
            continue
        for input_id in ids:
            try:
                registry |= lock.resolve_inputs(base, sources_lock, [input_id])
            except lock.LockError:
                continue
    return registry


def _prefetch_env(
    step: StepSpec,
    worktree: Path,
    analysis: str,
    path: Path,
    profile: Profile,
    base: Path,
    cache_dir: Path,
    rebuild: bool,
) -> env.PrefetchResult:
    e = step.env
    uv_lock = worktree / "uv.lock" if e.body.source == "uv-export" and e.torch is not None else None
    wheels_lock = base / default_wheels_lock(e) if e.sdist_allow and e.wheels is None else None
    result = env.prefetch_env(
        e,
        base,
        cache_dir,
        analysis=analysis,
        wheelhouse=cache_dir / WHEELHOUSE,
        uv_lock=uv_lock,
        profile=profile,
        wheels_lock=wheels_lock,
        rebuild=rebuild,
    )
    if result.record and result.wheels_lock_sha256 is not None:
        _record_wheels(path, e, result.wheels_lock_sha256)
    return result


def _cmd_build_one(args: argparse.Namespace) -> int:
    with sandbox.tool_masks(_tool_masks(args)):
        return _cmd_build_one_masked(args)


def _cmd_build_one_masked(args: argparse.Namespace) -> int:
    path = Path(args.manifest)
    profile = _profile(args)
    (manifest,) = _manifests([path], profile)
    base, cache_dir = _base_dir(args, path), _cache_dir(args)
    step = next((s for s in manifest.steps if s.name == args.step), None)
    if step is None:
        raise UsageError(f"no step {args.step!r} in {path}")
    sdist = next((s for s in step.env.sdist_allow if s.name == args.sdist), None)
    if sdist is None:
        raise UsageError(f"step {args.step!r} lists no sdist {args.sdist!r}")
    lock_path, expected = _wheels_location(step.env, base)
    if expected is None and lock_path.exists():
        raise ReproduceLockError(f"{lock_path} exists but the manifest records no wheels lock")
    result = build.ensure_wheel(
        sdist,
        base,
        cache_dir / WHEELHOUSE,
        lock_path,
        step.env.torch,
        python=step.env.python_exact,
        rebuild=args.rebuild,
        expected_sha256=expected,
        sdist_cache=cache_dir / SDISTS,
        modules_init=profile.modules_init,
    )
    if result.wheels_lock_sha256 != expected:
        _record_wheels(path, step.env, result.wheels_lock_sha256)
    print(f"{sdist.name}\t{'built' if result.built else 'reused'}\t{result.wheel_sha256[:12]}")
    return EXIT_OK


# --- run, submit, status --------------------------------------------------------------------------------


def _run_config(args: argparse.Namespace, paths: Sequence[Path], profile: Profile) -> run.RunConfig:
    return run.RunConfig(
        profile=profile,
        cache=_cache_dir(args),
        base_dir=_base_dir(args, paths[0]),
        date=args.date or _today(),
        timeout=args.timeout,
    )


def _read_bytes(paths: Sequence[Path]) -> tuple[list[bytes], list[str]]:
    """Read each manifest once; its sha256 is taken before anything parses it."""
    _require(paths)
    data = [p.read_bytes() for p in paths]
    return data, [hashlib.sha256(d).hexdigest() for d in data]


def _read_manifests(paths: Sequence[Path], profile: Profile) -> tuple[list[Manifest], list[str]]:
    """Each manifest's sha256 and the manifest parsed from the same bytes, under the profile's headline rule."""
    data, shas = _read_bytes(paths)
    rule = profile.policy.headline_rule
    return [parse_manifest(d, p.suffix, headline_rule=rule) for d, p in zip(data, paths)], shas


def _manifest_changed(args: argparse.Namespace, paths: Sequence[Path], profile: Profile, data: list[bytes]) -> int:
    """Refuse a job whose manifests changed since submission, without trusting the changed files.

    Its row becomes ``failed`` with ``E_MANIFEST_CHANGED``. When the changed manifests still parse, its
    dependents' pending rows become ``never-ran`` with the same code; when they do not, the row is found by
    step name and submission among the run dirs of the date. Row order is kept, and only rows of this job's
    submission (or untagged) are changed.
    """
    config = _run_config(args, paths, profile)
    code = ["E_MANIFEST_CHANGED"]
    steps = list(args.step or ([args.pool] if args.pool else []))
    if args.shard is not None:
        for step in steps:
            print(status_line(step, None, "failed", EXIT_FAILED, code))
        return EXIT_FAILED
    try:
        manifests = [
            parse_manifest(d, p.suffix, headline_rule=profile.policy.headline_rule) for d, p in zip(data, paths)
        ]
    except (ValueError, yaml.YAMLError, UnicodeDecodeError):
        manifests = None
    if manifests is not None:
        analysis = {s.name: m.name for m in manifests for s in m.steps}
        specs = {s.name: s for m in manifests for s in m.steps}
    else:
        analysis = _rows_by_step(profile, config.date, steps, args.submission)
        specs = {}

    def write(name: str, rows: list[dict[str, Any]], **filters: Any) -> None:  # noqa: ANN401
        record = cache.run_dir(profile.runs_root, config.date, name) / run.RUN_JSON
        with contextlib.suppress(OSError, ValueError, cache.CacheError):
            order = [r["step"] for r in run.read_run_record(record)]
            run.update_run_record(profile.runs_root, config.date, name, rows, order=order, **filters)

    for step in steps:
        print(status_line(step, None, "failed", EXIT_FAILED, code))
        if step in analysis:
            row = {"step": step, "key": None, "status": "failed", "exit": EXIT_FAILED, "codes": code}
            if step in specs:
                row["restricted"] = specs[step].restricted
            write(analysis[step], [row], submission=args.submission, allow_untagged=True)
    dependents = run.descendants(specs, {s for s in steps if s in specs}) - set(steps)
    for name in sorted({analysis[d] for d in dependents}):
        rows = [run.never_ran_row(d, code, specs[d].restricted) for d in sorted(dependents) if analysis[d] == name]
        write(name, rows, pending_only=True, submission=args.submission)
    return EXIT_FAILED


def _rows_by_step(
    profile: Profile, date: datetime.date, steps: Sequence[str], submission: str | None
) -> dict[str, str]:
    """``{step: analysis}`` for rows of these steps (of this submission, when given) in the date's run dirs."""
    found: dict[str, str] = {}
    day = profile.runs_root / date.isoformat()
    try:
        names = sorted(p.name for p in day.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return found
    for name in names:
        try:
            rows = run.read_run_record(day / name / run.RUN_JSON)
        except (OSError, ValueError):
            continue
        for row in rows:
            if row["step"] in steps and (submission is None or row.get("submission") == submission):
                found.setdefault(row["step"], name)
    return found


def _cmd_run(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    profile = _profile(args)
    data, shas = _read_bytes(paths)
    if args.pin and not args.finalize and list(args.pin) != shas:
        if args.host_probe:
            print("host-probe manifest-changed")
            return EXIT_OK
        return _manifest_changed(args, paths, profile, data)
    manifests = [parse_manifest(d, p.suffix, headline_rule=profile.policy.headline_rule) for d, p in zip(data, paths)]
    config = _run_config(args, paths, profile)
    ids = {m.name: run.manifest_id(p, d, config.base_dir) for m, p, d in zip(manifests, paths, data)}
    config = dataclasses.replace(config, manifest_ids=ids)
    if args.dry_run:
        plans = run.plan_steps(manifests, config)
        for p in plans:
            print(plan_line(p))
        return EXIT_FAILED if any(p.status == "blocked" for p in plans) else EXIT_OK
    if args.pool is not None:
        return _run_pool(args, paths, manifests, config)
    if args.host_probe:
        return _host_probe(args, manifests, config)
    if args.finalize:
        if args.submission is not None:
            cancelled = slurm.cancel_submission(args.submission, run.pending_jobs(manifests, config, args.submission))
            if cancelled.leftover:
                _error(f"E_SCANCEL_FAILED: job(s) {','.join(cancelled.leftover)} may still be queued")
            if not cancelled.listed:
                _error(f"squeue failed: jobs tagged {args.submission} with no recorded id may remain")
        run.finalize_run_records(manifests, config, submission=args.submission, state_of=slurm.job_state)
        return EXIT_OK
    outcomes = run.run_steps(manifests, config, only=args.step or None, submission=args.submission, shard=args.shard)
    for o in outcomes:
        print(status_line(o.step, o.key, o.status, o.exit, o.codes))
    return EXIT_OK if all(o.ok for o in outcomes) else EXIT_FAILED


def _run_pool(args: argparse.Namespace, paths: Sequence[Path], manifests: list[Manifest], config: run.RunConfig) -> int:
    """``run … --pool <step> --submission <id>``: `slurm.run_pool` with one `slurm.shard_command` per shard."""
    if args.submission is None or args.step or args.shard is not None:
        raise UsageError("--pool needs --submission and takes no --step or --shard")
    spec = next((s for m in manifests for s in m.steps if s.name == args.pool), None)
    if spec is None or spec.shards is None:
        raise UsageError(f"--pool {args.pool}: not a sharded step of these manifests")
    base = slurm.driver_run_argv(
        manifest_paths=[p.absolute() for p in paths],
        profile_path=Path(args.profile).absolute(),
        base_dir=config.base_dir,
        cache=config.cache,
        date=config.date,
        submission=args.submission,
        pins=args.pin,
    )
    commands = [slurm.shard_command(base, spec.name, i, sys.executable) for i in range(spec.shards.count)]
    probe = None if spec.host_probe is None else slurm.host_probe_command(base, spec.name, sys.executable)

    def recheck() -> tuple[str, ...]:
        changed = bool(args.pin) and _read_bytes(paths)[1] != list(args.pin)
        return ("E_MANIFEST_CHANGED",) if changed else ()

    outcome = slurm.run_pool(manifests, config, spec.name, args.submission, commands, recheck=recheck, host_probe=probe)
    print(status_line(outcome.step, outcome.key, outcome.status, outcome.exit, outcome.codes))
    return EXIT_OK if outcome.ok else EXIT_FAILED


def _host_probe(args: argparse.Namespace, manifests: list[Manifest], config: run.RunConfig) -> int:
    """``run … --step <s> --shard <i> --host-probe``: `run.host_probe`; prints ``host-probe <s> ok|<reason>``.

    Exits 0 when the probe passes, else with the reason's `run.HOST_PROBE_EXIT` code. A manifest changed since
    submission also exits 0, so the shard itself fails with ``E_MANIFEST_CHANGED``.
    """
    step = args.step[0] if args.step and len(args.step) == 1 else None
    spec = next((s for m in manifests for s in m.steps if s.name == step), None)
    if spec is None or spec.shards is None or spec.host_probe is None or args.shard is None:
        raise UsageError("--host-probe needs one sharded --step with a host_probe, and --shard")
    reason = run.host_probe(manifests, config, spec.name, args.shard)
    print(f"host-probe {spec.name} {reason or 'ok'}")
    return EXIT_OK if reason is None else run.HOST_PROBE_EXIT[reason]


def _cmd_submit(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    profile = _profile(args)
    _require(paths)
    base, cache_dir = _base_dir(args, paths[0]), _cache_dir(args)
    findings = lint_findings(paths, Path(args.profile), base, cache_dir=cache_dir)
    if findings:
        print("\n".join(findings))
    if blocking(findings):
        return EXIT_FAILED
    manifests, shas = _read_manifests(paths, profile)
    config = slurm.SubmitConfig(
        profile=profile,
        profile_path=Path(args.profile).absolute(),
        manifest_paths=[p.absolute() for p in paths],
        base_dir=base,
        cache=cache_dir,
        date=args.date or _today(),
        exports=slurm.batch_exports(profile, cache=cache_dir, environ=_uv_environ(cache_dir)),
        driver_python=sys.executable,
        manifest_sha256s=shas,
    )
    submission = slurm.submit(manifests, config)
    if submission.refused:
        for step, codes in submission.refused.items():
            print(status_line(step, None, "never-ran", None, codes))
        for input_id, job_id in submission.verify_jobs:
            print(f"verify-submitted\t{input_id}\t{job_id}")
        for input_id in submission.verify_failed:
            print(f"verify-not-submitted\t{input_id}")
        return EXIT_FAILED
    print(f"submission\t{submission.submission}")
    for job in submission.jobs:
        print(f"{job.step}\t{job.job_id}")
    if submission.finalize is not None:
        print(f"{slurm.FINALIZE_STEP}\t{submission.finalize}")
    return EXIT_OK


def _summary(run_dir: Path, step: str, key: str) -> dict[str, Any] | None:
    """``<run_dir>/<step>/summary.json`` when the link names the key's result dir, else None."""
    link = run_dir / step
    if not _STEP.match(step) or not link.is_symlink() or Path(os.path.realpath(link)).name != key:
        return None
    try:
        fd = os.open(link / cache.SUMMARY, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            data = json.loads(handle.read(1 << 20))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _cmd_status(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    try:
        rows = run.read_run_record(run_dir / run.RUN_JSON)
    except (OSError, ValueError) as error:
        raise UsageError(f"{run_dir}: no readable run.json ({type(error).__name__})") from None
    if not rows and not (run_dir / run.RUN_JSON).exists():
        raise UsageError(f"{run_dir}: no run.json")
    failed = pending = False
    for row in rows:
        key, status, exit_code = row.get("key"), row.get("status"), row.get("exit")
        codes = row.get("codes") if isinstance(row.get("codes"), list) else []
        if isinstance(key, str) and _KEY.match(key) and status in ("executed", "skipped", "failed"):
            summary = _summary(run_dir, row["step"], key)
            if summary is not None:
                exit_code = summary.get("exit", exit_code)
                codes = summary.get("codes") if isinstance(summary.get("codes"), list) else codes
        print(status_line(row["step"], key, str(status), exit_code, codes or []))
        pending |= status == run.PENDING
        failed |= status in ("failed", "never-ran") or (status != run.PENDING and exit_code != 0)
    if failed:
        return EXIT_FAILED
    return EXIT_PENDING if pending else EXIT_OK


def _cmd_lint(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    if not paths and args.lock is None:
        raise UsageError("give manifests, --lock or both")
    if not paths and args.skip_locks:
        raise UsageError("--skip-locks applies to manifests, not to --lock alone")
    _require([*paths, *([args.lock] if args.lock is not None else [])])
    findings: list[str] = []
    if args.lock is not None:
        base = _base_dir(args, args.lock)
        findings += [f"{args.lock}: {code} {detail}" for code, detail in lock.lock_findings(base, args.lock)]
    if paths:
        if args.profile is None:
            raise UsageError("--profile is required with manifests")
        cache_value = args.cache or os.environ.get(slurm.CACHE_VARIABLE)
        findings += lint_findings(
            paths,
            Path(args.profile),
            _base_dir(args, paths[0]),
            skip_locks=args.skip_locks,
            cache_dir=Path(cache_value).absolute() if cache_value else None,
        )
    if findings:
        print("\n".join(findings))
    return EXIT_FAILED if blocking(findings) else EXIT_OK


def _cmd_verify_inputs(args: argparse.Namespace) -> int:
    if bool(args.ids) == bool(args.all):
        raise UsageError("give input ids or --all")
    profile = _profile(args)
    with slurm._umask(slurm.submit_umask(profile)):
        return _verify_inputs(args, profile)


def _verify_inputs(args: argparse.Namespace, profile: Profile) -> int:
    """Run ``verify-inputs``; each id prints one line, and an ``OSError`` for one id prints ``E_DRIVER``.

    Verifying a restricted input (flagged, or in a restricted dir) first purges its public closure
    (`restricted.purge_closure`) and its public verify records, and adds ``purged=<n> skipped_inflight=<n>`` to
    its line (results a run may still own are left alone and counted); an unflagged one is then
    ``E_INPUT_REGISTRY``.
    """
    base = Path(args.base_dir).absolute()
    stems = sorted(p.stem for p in (base / inputs.REGISTRY_DIR).glob("*.yaml"))
    ids = stems if args.all else list(args.ids)
    failed = False
    if args.registry_only:
        for input_id in ids:
            codes = inputs.registry_codes(base, input_id, profile)
            failed |= bool(codes)
            print(f"{_id_text(input_id)}\t{codes[0] if codes else 'registry-ok'}")
        return EXIT_FAILED if failed else EXIT_OK
    registry: dict[str, inputs.RegisteredInput] = {}
    for stem in stems:
        with contextlib.suppress(inputs.InputError):
            registry[stem] = inputs.read_registry_entry(base, stem)[0]
    for input_id in ids:
        purge: restricted.Purge | None = None
        try:
            entry = inputs.read_registry_entry(base, input_id)[0]
            if restricted.input_is_restricted(entry, profile):
                purge = restricted.purge_closure(profile, input_id)
                inputs.remove_public_records(input_id, profile)
            inputs.check_flag(entry, profile)
            directory = inputs.verify_dir(entry, profile)
            others = [e for i, e in registry.items() if i != input_id]
            verification = inputs.verify(entry, profile, now=datetime.datetime.now(datetime.UTC), others=others)
            inputs.append_record(directory, input_id, verification)
        except (inputs.InputError, cache.CacheError) as error:
            failed = True
            print(f"{_id_text(input_id)}\t{error.code}")
            continue
        except OSError:
            failed = True
            print(f"{_id_text(input_id)}\tE_DRIVER")
            continue
        record = verification.record
        failed |= record.status != "ok"
        counts = " ".join(f"{name}={record.counts[name]}" for name in inputs.COUNT_NAMES)
        suffix = "" if purge is None else f" purged={purge.purged} skipped_inflight={purge.skipped_inflight}"
        print(f"{_id_text(input_id)}\t{record.status}\t{counts}{suffix}")
    return EXIT_FAILED if failed else EXIT_OK


def _id_text(input_id: str) -> str:
    return input_id if _STEP.match(input_id) else "?"


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "lock": _cmd_lock,
    "prefetch": _cmd_prefetch,
    "build-one": _cmd_build_one,
    "run": _cmd_run,
    "submit": _cmd_submit,
    "lint": _cmd_lint,
    "status": _cmd_status,
    "verify-inputs": _cmd_verify_inputs,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line.

    Args:
        argv: The arguments after the program name; defaults to ``sys.argv[1:]``.

    Returns:
        The exit code (see the module docstring).
    """
    args = build_parser().parse_args(argv)
    try:
        return _HANDLERS[args.command](args)
    except UsageError as error:
        _error(str(error))
        return EXIT_USAGE
    except run.SubmissionOverlap as error:
        _error(f"{error}; a local run waits until the submission ends or is settled with --finalize")
        return EXIT_FAILED
    except pydantic.ValidationError as error:
        _error(_validation_text(error))
        return EXIT_FAILED
    except (ValueError, OSError, yaml.YAMLError, ReproduceLockError, source.ReproduceSourceError) as error:
        _error(str(error))
        return EXIT_FAILED
    except (slurm.SlurmError, subprocess.SubprocessError) as error:
        _error(str(error))
        return EXIT_FAILED
    except (KeyboardInterrupt, slurm.Terminated):
        _error("interrupted")
        return EXIT_FAILED
    except Exception as error:  # noqa: BLE001
        code = getattr(error, "code", None)
        _error(code if isinstance(code, str) and _CODE.match(code) else f"E_DRIVER ({type(error).__name__})")
        return EXIT_FAILED
