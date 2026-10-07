"""The ``python -m senselab.utils.reproduce`` command line.

Subcommands:

- ``lock``: run the lock builders for each env, in order (first resolve, torch, body, overlays), and
  write each lock's sha256 (and a uv-export body's command) into the manifest.
- ``prefetch``: lint, then check out every step (with the profile's deny patterns), build or reuse each
  env with the profile's checks, record a changed ``wheels.lock`` sha256 in the manifest, and install R
  libraries. CUDA sdists are built as batch jobs (``build-one``) first, unless ``--local-builds``.
  ``--sources-only`` checks out draft manifests with the empty patch set and nothing else.
- ``build-one``: build or reuse one sdist's wheel and record the new ``wheels.lock`` sha256.
- ``run``: run steps in DAG order; ``--step`` runs just those (one batch job; ``--pin`` refuses a manifest
  changed since submission with ``E_MANIFEST_CHANGED``), ``--finalize --submission <id>`` cancels that
  submission's jobs (queued or running, found by recorded id and by their ``--comment`` tag) and settles its
  ``pending`` rows. A local run refuses (``E_SUBMIT_OVERLAP``) while a submission is live in its run dir.
- ``submit``: lint, then one held batch job per step chained with ``afterok``, then release
  (`slurm.submit`).
- ``lint``: check manifests and patches against the profile.
- ``status``: print each step's key, status, exit and codes from ``run.json`` and ``summary.json``.

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
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pydantic
import yaml

from senselab.utils.reproduce import build, cache, env, lockgen, run, slurm, source
from senselab.utils.reproduce.dag import DagError, toposort
from senselab.utils.reproduce.locks import TORCH_NAMES, ReproduceLockError, file_sha256, normalize_name
from senselab.utils.reproduce.spec import (
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
_SUBCOMMANDS = ("lock", "prefetch", "build-one", "run", "submit", "lint", "status")
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
    """The argument parser with its seven subcommands."""
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
    run_cmd.add_argument("--submission", help="the submission id (set by submit; printed by submit)")
    run_cmd.add_argument("--pin", action="append", default=[], help="a manifest's sha256 at submission, in order")
    run_cmd.add_argument("--min-cell", type=int, default=5)
    run_cmd.add_argument("--timeout", type=float, help="seconds before a step is killed")

    submit = sub.add_parser("submit", help="one batch job per step, chained with afterok")
    submit.add_argument("manifests", nargs="+", type=Path)
    _site_options(submit)
    submit.add_argument("--date", type=_date, help="the run date (default: today, UTC)")

    lint = sub.add_parser("lint", help="check manifests and patches")
    lint.add_argument("manifests", nargs="+", type=Path)
    _site_options(lint)

    status = sub.add_parser("status", help="per step: key, status, exit, codes")
    status.add_argument("run_dir", type=Path, help="<runs_root>/<date>/<analysis>")
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


def _manifests(paths: Sequence[Path]) -> list[Manifest]:
    _require(paths)
    return [load_manifest(p) for p in paths]


def _uv_environ(cache_dir: Path) -> dict[str, str]:
    values = {key: os.environ.get(key, "") for key in slurm.UV_EXPORTS}
    if not all(values.values()):
        uv = lockgen.find_uv()
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


def lint_findings(paths: Sequence[Path], profile_path: Path, base: Path) -> list[str]:
    """Check manifests and their patches against a profile.

    Args:
        paths: The manifest files.
        profile_path: The profile file.
        base: The manifest repository.

    Returns:
        One line per finding, ``<where>: <code> <detail>``; empty when clean. Codes: ``profile-invalid``,
        ``deny-patterns-unreadable``, ``manifest-invalid``, ``dag-invalid``, the `slurm.step_findings`
        codes, ``adapter-missing``, ``patch-missing``, ``patch-sha``, ``patch-lint`` (with the lint code,
        line and pattern index, never patch text) and ``deny-pattern-invalid``.

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
        try:
            deny = source.load_deny_patterns(profile.deny_patterns_file)
        except OSError:
            out.append(f"{profile_path}: deny-patterns-unreadable {profile.deny_patterns_file}")
    loaded: list[tuple[Path, Manifest]] = []
    for path in paths:
        try:
            loaded.append((path, load_manifest(path)))
        except pydantic.ValidationError as error:
            out.append(f"{path}: manifest-invalid {_validation_text(error)}")
        except (ValueError, yaml.YAMLError) as error:
            out.append(f"{path}: manifest-invalid {error}")
    if len(loaded) == len(paths):
        try:
            toposort([s for _, m in loaded for s in m.steps])
        except DagError as error:
            out.append(f"{', '.join(str(p) for p in paths)}: dag-invalid {error}")
    for path, manifest in loaded:
        for step in manifest.steps:
            where = f"{path}:{step.name}"
            if profile is not None:
                out += [f"{where}: {f.code} {f.detail}" for f in slurm.step_findings(step, profile)]
            if not (base / step.adapter).is_file():
                out.append(f"{where}: adapter-missing {step.adapter}")
            out += _patch_findings(where, step, base, deny)
    return out


# --- manifest updates -----------------------------------------------------------------------------------


def _edit_envs(path: Path, old: EnvSpec, change: Callable[[dict[str, Any]], None]) -> None:
    """Apply ``change`` to the raw ``env`` mapping of every step of a manifest file whose env equals ``old``."""
    manifest = load_manifest(path)

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


def _lock_env(step: StepSpec, base: Path, cache_dir: Path, deny: Sequence[str]) -> dict[str, Any]:
    """Run the builders for one env in order: first resolve, torch, body, overlays. Returns the field updates."""
    e = step.env
    worktree: list[Path] = []

    def project() -> Path:
        if not worktree:
            worktree.append(source.checkout_step(step, cache=cache_dir, base_dir=base, deny_patterns=deny).worktree)
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
    deny = source.load_deny_patterns(profile.deny_patterns_file)
    for path in paths:
        done: list[EnvSpec] = []
        for step in load_manifest(path).steps:
            if any(step.env == e for e in done):
                continue
            done.append(step.env)
            written = _lock_env(step, base, cache_dir, deny)

            def change(raw: dict[str, Any], written: dict[str, Any] = written) -> None:
                _apply_lock_updates(raw, written, base)

            _edit_envs(path, step.env, change)
            print(f"{step.name}\tlocked\t{','.join(sorted(written))}")
    return EXIT_OK


# --- prefetch and build-one -----------------------------------------------------------------------------


def _next_cuda_build(
    paths: Sequence[Path], base: Path, wheelhouse: Path, rebuild: bool, done: set[tuple[str, str, str]]
) -> tuple[Path, StepSpec, Any] | None:
    for path in paths:
        for step in load_manifest(path).steps:
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
        todo = _next_cuda_build(paths, base, cache_dir / WHEELHOUSE, args.rebuild, done)
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
        slurm.submit_and_wait(script, name=f"reproduce-build-{sdist.name}-{uuid.uuid4().hex[:8]}")
        done.add((str(path), step.name, sdist.name))


def _cmd_prefetch(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    _require(paths)
    cache_dir = _cache_dir(args)
    if args.sources_only:
        for path in paths:
            for name, co in source.checkout_sources(load_sources(path), cache=cache_dir).items():
                print(f"{name}\t{co.commit[:8]}\tsources")
        return EXIT_OK
    profile = _profile(args)
    base = _base_dir(args, paths[0])
    findings = lint_findings(paths, Path(args.profile), base)
    if findings:
        print("\n".join(findings))
        return EXIT_FAILED
    weighted = [s.name for p in paths for s in load_manifest(p).steps if s.hf_weights]
    if weighted:
        _error(f"steps declaring hf_weights cannot be prefetched yet: {', '.join(weighted)}")
        return EXIT_FAILED
    if not args.local_builds:
        _batch_builds(args, paths, profile, base, cache_dir)
    deny = source.load_deny_patterns(profile.deny_patterns_file)
    for path in paths:
        manifest = load_manifest(path)
        done: list[tuple[EnvSpec, env.PrefetchResult]] = []
        for step in manifest.steps:
            co = source.checkout_step(step, cache=cache_dir, base_dir=base, deny_patterns=deny)
            result = next((r for e, r in done if e == step.env), None)
            if result is None:
                result = _prefetch_env(step, co.worktree, manifest.name, path, profile, base, cache_dir, args.rebuild)
                done.append((step.env, result))
            if step.kind in cache.R_KINDS:
                venv = result.derived["notebook_tools"].path if step.kind == "ir_notebook" else None
                env.prefetch_r(step.env, cache_dir, profile, kind=step.kind, venv=venv)
            print(f"{step.name}\tprefetched")
    return EXIT_OK


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
    path = Path(args.manifest)
    _profile(args)
    (manifest,) = _manifests([path])
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
        min_cell=args.min_cell,
        timeout=args.timeout,
    )


def _read_manifests(paths: Sequence[Path]) -> tuple[list[Manifest], list[str]]:
    """Read each manifest once: its sha256 and the manifest parsed from the same bytes."""
    _require(paths)
    data = [p.read_bytes() for p in paths]
    return [parse_manifest(d, p.suffix) for d, p in zip(data, paths)], [hashlib.sha256(d).hexdigest() for d in data]


def _manifest_changed(args: argparse.Namespace, paths: Sequence[Path], profile: Profile) -> int:
    """Refuse a job whose manifests changed since submission.

    Its row becomes ``failed`` with ``E_MANIFEST_CHANGED`` and its dependents' pending rows ``never-ran`` with
    the same code, where the changed manifests still parse; row order is kept, and only rows of this job's
    submission (or untagged) are changed.
    """
    try:
        manifests = _manifests(paths)
    except (ValueError, OSError, yaml.YAMLError):
        manifests = []
    config = _run_config(args, paths, profile)
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    specs = {s.name: s for m in manifests for s in m.steps}
    code = ["E_MANIFEST_CHANGED"]

    def write(name: str, rows: list[dict[str, Any]], **filters: Any) -> None:  # noqa: ANN401
        record = cache.run_dir(profile.runs_root, config.date, name) / run.RUN_JSON
        with contextlib.suppress(OSError, ValueError, cache.CacheError):
            order = [r["step"] for r in run.read_run_record(record)]
            run.update_run_record(profile.runs_root, config.date, name, rows, order=order, **filters)

    steps = list(args.step or [])
    for step in steps:
        print(status_line(step, None, "failed", EXIT_FAILED, code))
        if step in analysis:
            row = {"step": step, "key": None, "status": "failed", "exit": EXIT_FAILED, "codes": code}
            write(analysis[step], [row], submission=args.submission, allow_untagged=True)
    dependents = run.descendants(specs, {s for s in steps if s in specs}) - set(steps)
    for name in sorted({analysis[d] for d in dependents}):
        rows = [run.never_ran_row(d, code) for d in sorted(dependents) if analysis[d] == name]
        write(name, rows, pending_only=True, submission=args.submission)
    return EXIT_FAILED


def _cmd_run(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    profile = _profile(args)
    manifests, shas = _read_manifests(paths)
    if args.pin and not args.finalize and list(args.pin) != shas:
        return _manifest_changed(args, paths, profile)
    config = _run_config(args, paths, profile)
    if args.finalize:
        if args.submission is not None:
            cancelled = slurm.cancel_submission(args.submission, run.pending_jobs(manifests, config, args.submission))
            if cancelled.leftover:
                _error(f"E_SCANCEL_FAILED: job(s) {','.join(cancelled.leftover)} may still be queued")
            if not cancelled.listed:
                _error(f"squeue failed: jobs tagged {args.submission} with no recorded id may remain")
        run.finalize_run_records(manifests, config, submission=args.submission, state_of=slurm.job_state)
        return EXIT_OK
    outcomes = run.run_steps(manifests, config, only=args.step or None, submission=args.submission)
    for o in outcomes:
        print(status_line(o.step, o.key, o.status, o.exit, o.codes))
    return EXIT_OK if all(o.ok for o in outcomes) else EXIT_FAILED


def _cmd_submit(args: argparse.Namespace) -> int:
    paths = list(args.manifests)
    profile = _profile(args)
    _require(paths)
    base, cache_dir = _base_dir(args, paths[0]), _cache_dir(args)
    findings = lint_findings(paths, Path(args.profile), base)
    if findings:
        print("\n".join(findings))
        return EXIT_FAILED
    manifests, shas = _read_manifests(paths)
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
    _require(paths)
    findings = lint_findings(paths, Path(args.profile), _base_dir(args, paths[0]))
    if findings:
        print("\n".join(findings))
    return EXIT_FAILED if findings else EXIT_OK


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "lock": _cmd_lock,
    "prefetch": _cmd_prefetch,
    "build-one": _cmd_build_one,
    "run": _cmd_run,
    "submit": _cmd_submit,
    "lint": _cmd_lint,
    "status": _cmd_status,
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
