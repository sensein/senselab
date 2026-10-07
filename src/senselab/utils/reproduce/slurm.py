"""Batch jobs for reproduce steps: rendering, the preemption probe, and ``afterok`` submission chains.

Each step becomes one batch job that runs ``python -m senselab.utils.reproduce run … --step <name>``.
A job does not get its step key at submission: an ``up:`` input is digested from the upstream result
dir, which exists only once the upstream job has finished, so each job keys its own step when it starts
and resolves its upstream steps' done results (`run.run_steps` with ``only``). Jobs are chained with
``--dependency=afterok:<ids>`` and ``--kill-on-invalid-dep=yes``, so a failed step cancels every job
that depends on it.

``submit`` writes a ``pending`` row per step into each analysis's ``run.json`` before the first job is
submitted, then records each job id. A job rewrites its own row when it ends, and a job whose step did
not succeed turns its dependants' rows into ``never-ran``. A last ``--finalize`` job, ``afterany`` every
step job, turns any row still ``pending`` (a job Slurm cancelled or killed) into ``never-ran``, which is
terminal.

A job's ``--output`` and ``--error`` go to ``<slurm_logs_root>/<date>/<jobid>.{out,err}``: a job is
submitted before its driver knows whether the step will skip. The job starts from an empty environment
(``--export=NONE``) and exports only ``PATH``, the variables of `batch_exports`, the driver interpreter
and a per-job ``TMPDIR`` under ``/dev/shm``.
"""

from __future__ import annotations

import datetime
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from senselab.utils.reproduce import run
from senselab.utils.reproduce.build import SYSTEM_PATH, _safe_word
from senselab.utils.reproduce.dag import toposort
from senselab.utils.reproduce.spec import RESERVED_ROOT_NAMES, Manifest, Profile, StepSpec, placeholders

SBATCH = "sbatch"
DRIVER_MODULE = "senselab.utils.reproduce"
LINK_MODE = "hardlink"
UV_EXPORTS = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")
CACHE_VARIABLE = "SENSELAB_REPRO_CACHE"
FINALIZE_STEP = "finalize"
FINALIZE_CPUS = 1
FINALIZE_MEM = "1G"
FINALIZE_TIME = "00:10:00"
SUBMIT_TIMEOUT = 120
_SCRIPT_OWNED = frozenset({"PATH", "TMPDIR", "REPRO_DRIVER_PYTHON"})
_EXPORT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_ID = re.compile(r"^[0-9]+$")
_PREEMPTS = re.compile(r"\bPreempts:")

Runner = Callable[[list[str], str], "subprocess.CompletedProcess[str]"]
"""Runs ``(argv, script)``: an sbatch command line, with the batch script on its standard input."""


class SlurmError(RuntimeError):
    """A submission that failed or was refused."""


class Finding(NamedTuple):
    """One cross-check failure of a step against the profile: the step, a fixed code, and a detail."""

    step: str
    code: str
    detail: str


class Job(NamedTuple):
    """One submitted step job."""

    step: str
    analysis: str
    job_id: str
    partition: str


class Submission(NamedTuple):
    """The step jobs of one submission, in DAG order, and the ``--finalize`` job's id (None if none)."""

    jobs: list[Job]
    finalize: str | None


@dataclass
class SubmitConfig:
    """What every job of one submission shares.

    Attributes:
        profile: The site profile.
        profile_path: The profile file each job reads.
        manifest_paths: Every manifest of the set, as each job is given them (absolute paths).
        base_dir: The manifest repository.
        cache: The reproduce cache.
        date: The run date, fixed at submission, so a job that starts after midnight writes the same run dir.
        exports: The variables each job exports (`batch_exports`).
        driver_python: The interpreter with senselab installed.
    """

    profile: Profile
    profile_path: Path
    manifest_paths: list[Path]
    base_dir: Path
    cache: Path
    date: datetime.date
    exports: Mapping[str, str]
    driver_python: str = field(default_factory=lambda: sys.executable)


def default_runner(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
    """Run an sbatch command with ``script`` on its standard input.

    Args:
        argv: The command; its first element is looked up on ``PATH``.
        script: The batch script.

    Returns:
        The completed process, output captured.

    Raises:
        SlurmError: If the command is not on ``PATH``.
    """
    found = shutil.which(argv[0])
    if found is None:
        raise SlurmError(f"{argv[0]} is not on PATH")
    return subprocess.run(
        [found, *argv[1:]], input=script, capture_output=True, text=True, check=False, timeout=SUBMIT_TIMEOUT
    )


def log_dir(profile: Profile, date: datetime.date) -> Path:
    """``<slurm_logs_root>/<YYYY-MM-DD>``, where a job's ``<jobid>.out`` and ``<jobid>.err`` go."""
    return profile.slurm_logs_root / date.isoformat()


def batch_exports(profile: Profile, *, cache: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """The variables a step job exports, since a batch job inherits nothing.

    Args:
        profile: The site profile; the lib dirs of its ``native`` entries become ``LD_LIBRARY_PATH``.
        cache: The reproduce cache (``SENSELAB_REPRO_CACHE``).
        environ: Where ``UV_CACHE_DIR`` and ``UV_PYTHON_INSTALL_DIR`` come from.

    Returns:
        ``UV_CACHE_DIR``, ``UV_LINK_MODE=hardlink``, ``UV_PYTHON_INSTALL_DIR``, ``SENSELAB_REPRO_CACHE`` and,
        when the profile declares native lib dirs, ``LD_LIBRARY_PATH``.

    Raises:
        ValueError: If ``environ`` lacks a uv variable.
    """
    missing = [key for key in UV_EXPORTS if not environ.get(key)]
    if missing:
        raise ValueError(f"set {', '.join(missing)} for the batch jobs")
    exports = {
        "UV_CACHE_DIR": environ["UV_CACHE_DIR"],
        "UV_LINK_MODE": LINK_MODE,
        "UV_PYTHON_INSTALL_DIR": environ["UV_PYTHON_INSTALL_DIR"],
        CACHE_VARIABLE: os.fspath(cache),
    }
    libs = list(dict.fromkeys(os.fspath(d) for n in profile.native.values() for d in n.lib_dirs))
    if libs:
        exports["LD_LIBRARY_PATH"] = ":".join(libs)
    return exports


# --- cross-checks ----------------------------------------------------------------------------------------


def _root_names(step: StepSpec) -> list[str]:
    roots = [i.root for i in step.inputs] + [s.root for s in step.stage]
    names = [r for r in roots if not r.startswith("up:") and r not in RESERVED_ROOT_NAMES]
    templates = [*step.argv, step.cwd, *step.pythonpath, *(s.dest for s in step.stage)]
    names += [p.split(":", 1)[1] for t in templates for p in placeholders(t) if p.startswith("root:")]
    return list(dict.fromkeys(names))


def step_findings(step: StepSpec, profile: Profile) -> list[Finding]:
    """Cross-check what a step asks of the profile.

    Args:
        step: The step.
        profile: The site profile.

    Returns:
        ``partition-unknown`` (``resources.profile`` is no partition key), ``fallback-invalid`` (the
        partition's fallback is itself or not a plain word), ``gpu-no-gres`` (``gpu_arch`` or
        ``min_driver`` on a partition with no ``gres``), ``native-unknown`` and ``root-unknown`` (an input,
        stage or ``{root:<name>}`` naming no profile root) findings, in that order.
    """
    found: list[Finding] = []
    part = profile.partitions.get(step.resources.profile)
    if part is None:
        found.append(Finding(step.name, "partition-unknown", step.resources.profile))
    else:
        if part.fallback is not None:
            try:
                _safe_word(part.fallback, "fallback")
                if part.fallback == part.partition:
                    raise ValueError("fallback is the partition itself")
            except ValueError as error:
                found.append(Finding(step.name, "fallback-invalid", str(error)))
        if (step.resources.gpu_arch or step.resources.min_driver) and part.gres is None:
            found.append(Finding(step.name, "gpu-no-gres", step.resources.profile))
    native = sorted(set(step.resources.native) - set(profile.native))
    if native:
        found.append(Finding(step.name, "native-unknown", ", ".join(native)))
    roots = [r for r in _root_names(step) if r not in profile.roots]
    if roots:
        found.append(Finding(step.name, "root-unknown", ", ".join(roots)))
    return found


# --- rendering -------------------------------------------------------------------------------------------


def _header(name: str, partition: str, cpus: int, mem: str, time: str, gres: str | None, logs: Path) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={_safe_word(name, 'job name')}",
        f"#SBATCH --partition={_safe_word(partition, 'partition')}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={_safe_word(mem, 'mem')}",
        f"#SBATCH --time={_safe_word(time, 'time')}",
    ]
    if gres is not None:
        lines.append(f"#SBATCH --gres={_safe_word(gres, 'gres')}")
    logs_text = _safe_word(os.fspath(logs), "log dir")
    lines += [f"#SBATCH --output={logs_text}/%j.out", f"#SBATCH --error={logs_text}/%j.err", "#SBATCH --export=NONE"]
    return lines


def _dependency(kind: str, ids: Sequence[str]) -> list[str]:
    bad = [i for i in ids if not _JOB_ID.match(i)]
    if bad:
        raise ValueError(f"dependency job ids must be numbers: {bad}")
    return [f"#SBATCH --dependency={kind}:{':'.join(ids)}"] if ids else []


def _body(exports: Mapping[str, str], driver_python: str, command: Sequence[str]) -> list[str]:
    bad = sorted(k for k in exports if not _EXPORT_NAME.match(k) or k in _SCRIPT_OWNED)
    if bad:
        raise ValueError(f"export names refused: {bad}")
    lines = ["set -euo pipefail", "umask 007", f"export PATH={SYSTEM_PATH}"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
    lines += [
        f"export REPRO_DRIVER_PYTHON={shlex.quote(driver_python)}",
        'export TMPDIR="/dev/shm/reproduce-${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
        '"$REPRO_DRIVER_PYTHON" ' + shlex.join(["-m", DRIVER_MODULE, *command]),
    ]
    return lines


def render_step_job(
    step: StepSpec,
    profile: Profile,
    *,
    partition: str,
    command: Sequence[str],
    date: datetime.date,
    exports: Mapping[str, str],
    dependencies: Sequence[str] = (),
    driver_python: str = sys.executable,
) -> str:
    """Render the batch script of one step job.

    Args:
        step: The step; ``resources.profile`` picks the partition entry giving CPUs, memory, time and gres.
        profile: The site profile.
        partition: The partition to submit to (the entry's own, or its fallback; `choose_partition`).
        command: The driver arguments after ``-m senselab.utils.reproduce`` (`step_command`).
        date: The log date.
        exports: The variables to export (`batch_exports`).
        dependencies: The job ids of the step's needs; rendered as ``--dependency=afterok:<ids>``.
        driver_python: The interpreter with senselab installed.

    Returns:
        The script text; every job carries ``--kill-on-invalid-dep=yes``.

    Raises:
        ValueError: If a value would need quoting the script cannot express safely.
    """
    part = profile.partitions[step.resources.profile]
    logs = log_dir(profile, date)
    lines = _header(f"reproduce-{step.name}", partition, part.cpus, part.mem, part.time, part.gres, logs)
    lines += _dependency("afterok", dependencies)
    lines.append("#SBATCH --kill-on-invalid-dep=yes")
    lines += _body(exports, driver_python, command)
    return "\n".join(lines) + "\n"


def render_finalize_job(
    profile: Profile,
    *,
    command: Sequence[str],
    date: datetime.date,
    exports: Mapping[str, str],
    dependencies: Sequence[str],
    driver_python: str = sys.executable,
) -> str:
    """Render the job that settles ``pending`` rows once every step job has ended (``afterany``).

    It runs on the profile's ``build`` partition with one CPU, ``1G`` and ten minutes.

    Args:
        profile: The site profile.
        command: The driver arguments (`step_command` with ``finalize=True``).
        date: The log date.
        exports: The variables to export.
        dependencies: Every step job id.
        driver_python: The interpreter with senselab installed.

    Returns:
        The script text.
    """
    part = profile.partitions[profile.build]
    logs = log_dir(profile, date)
    lines = _header("reproduce-finalize", part.partition, FINALIZE_CPUS, FINALIZE_MEM, FINALIZE_TIME, None, logs)
    lines += _dependency("afterany", dependencies)
    lines += _body(exports, driver_python, command)
    return "\n".join(lines) + "\n"


def step_command(config: SubmitConfig, step: str | None, *, finalize: bool = False) -> list[str]:
    """The driver arguments of a job: ``run <manifests…> --profile … --date … --step <step>`` (or ``--finalize``)."""
    argv = ["run", *(os.fspath(p) for p in config.manifest_paths)]
    argv += ["--profile", os.fspath(config.profile_path), "--base-dir", os.fspath(config.base_dir)]
    argv += ["--cache", os.fspath(config.cache), "--date", config.date.isoformat()]
    if finalize:
        return [*argv, "--finalize"]
    assert step is not None
    return [*argv, "--step", step]


# --- the probe and submission --------------------------------------------------------------------------


def probe_preempts(output: str) -> bool:
    """Whether an ``sbatch --test-only`` answer reports preemption (a ``Preempts:`` line)."""
    return _PREEMPTS.search(output) is not None


def choose_partition(step: StepSpec, profile: Profile, script_for: Callable[[str], str], runner: Runner) -> str:
    """The partition a step job goes to.

    A partition entry with a ``fallback`` is probed with ``sbatch --test-only`` first; when the probe
    reports preemption or fails, the job goes to the fallback. An entry without one is used as is.

    Args:
        step: The step.
        profile: The site profile.
        script_for: Renders the step's script for a partition (without dependencies).
        runner: Runs sbatch.

    Returns:
        The partition name.
    """
    part = profile.partitions[step.resources.profile]
    if part.fallback is None:
        return part.partition
    result = runner([SBATCH, "--test-only"], script_for(part.partition))
    if result.returncode != 0 or probe_preempts((result.stdout or "") + (result.stderr or "")):
        return part.fallback
    return part.partition


def _submit_one(runner: Runner, script: str) -> str:
    result = runner([SBATCH, "--parsable"], script)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
        raise SlurmError(f"sbatch exited {result.returncode}: {tail[0]}")
    job_id = (result.stdout or "").strip().split(";", 1)[0]
    if not _JOB_ID.match(job_id):
        raise SlurmError("sbatch --parsable printed no job id")
    return job_id


def submit(manifests: Sequence[Manifest], config: SubmitConfig, runner: Runner | None = None) -> Submission:
    """Submit one job per step in DAG order, each ``afterok`` its needs' jobs, then a ``--finalize`` job.

    Every analysis's ``run.json`` gets a ``pending`` row per step before the first job is submitted,
    and each row then gets its ``job`` id unless the job already rewrote it. When a submission fails,
    that step and every later one are marked ``never-ran``, a finalize job is still submitted for the
    jobs already queued, and `SlurmError` is raised.

    Args:
        manifests: The manifests; step names are unique across them.
        config: The submission's shared settings.
        runner: Runs sbatch; defaults to `default_runner`.

    Returns:
        The submission.

    Raises:
        SlurmError: If a step fails the profile cross-checks (`step_findings`) or a submission fails.
    """
    runner = default_runner if runner is None else runner
    profile = config.profile
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    ordered = toposort([s for m in manifests for s in m.steps])
    findings = [f for s in ordered for f in step_findings(s, profile)]
    if findings:
        raise SlurmError(f"{len(findings)} profile cross-check finding(s); run lint")
    orders = {m.name: [s.name for s in ordered if analysis[s.name] == m.name] for m in manifests}
    log_dir(profile, config.date).mkdir(parents=True, exist_ok=True)
    for name, order in orders.items():
        pending = [{**run.never_ran_row(s), "status": run.PENDING, "job": None} for s in order]
        run.update_run_record(profile.runs_root, config.date, name, pending, order=order)
    jobs: list[Job] = []
    by_step: dict[str, str] = {}
    for index, step in enumerate(ordered):

        def script_for(partition: str, needs: Sequence[str] = (), step: StepSpec = step) -> str:
            return render_step_job(
                step,
                profile,
                partition=partition,
                command=step_command(config, step.name),
                date=config.date,
                exports=config.exports,
                dependencies=needs,
                driver_python=config.driver_python,
            )

        try:
            partition = choose_partition(step, profile, script_for, runner)
            job_id = _submit_one(runner, script_for(partition, [by_step[n] for n in step.needs]))
        except (SlurmError, OSError, subprocess.SubprocessError) as error:
            rest = ordered[index:]
            for name, order in orders.items():
                gone = [run.never_ran_row(s.name) for s in rest if analysis[s.name] == name]
                if gone:
                    run.update_run_record(profile.runs_root, config.date, name, gone, order=order, pending_only=True)
            if by_step:
                try:
                    _submit_finalize(config, runner, list(by_step.values()))
                except (SlurmError, OSError, subprocess.SubprocessError):
                    pass
            raise SlurmError(f"submitting {step.name} failed ({error}); {len(jobs)} job(s) already queued") from None
        by_step[step.name] = job_id
        name = analysis[step.name]
        row = {"step": step.name, "job": job_id}
        run.update_run_record(profile.runs_root, config.date, name, [row], order=orders[name], pending_only=True)
        jobs.append(Job(step.name, name, job_id, partition))
    finalize = _submit_finalize(config, runner, [j.job_id for j in jobs]) if jobs else None
    return Submission(jobs, finalize)


def _submit_finalize(config: SubmitConfig, runner: Runner, ids: list[str]) -> str:
    script = render_finalize_job(
        config.profile,
        command=step_command(config, None, finalize=True),
        date=config.date,
        exports=config.exports,
        dependencies=ids,
        driver_python=config.driver_python,
    )
    return _submit_one(runner, script)
