"""Batch jobs for reproduce steps: rendering, the feasibility probe, and held ``afterok`` submission chains.

Each step becomes one batch job that runs ``python -P -m senselab.utils.reproduce run … --step <name>``.
No step key is computed at submission (an ``up:`` input exists only once its upstream job finished);
each job keys its own step when it starts and resolves its upstream steps' done results
(`run.run_steps` with ``only``).

`submit` renders every script first (both partitions of a step with a fallback, and the finalize job),
so a render error fails with no side effect. It then submits every job ``--hold``, chained with
``--dependency=afterok:<ids>`` and ``--kill-on-invalid-dep=yes``, writes a ``pending`` row per step
tagged with the submission id and job id, submits a ``--finalize`` job ``afterany`` every step job, and
only then releases them all with ``scontrol release``. Any exception or interrupt before the release
completes cancels every job submitted so far and marks the steps ``never-ran`` with ``E_SUBMIT_ABORTED``.

Every job runs in ``<slurm_logs_root>/<date>`` (``--chdir``), which the driver creates with mode
``2777`` less the profile policy's ``umask`` (`log_dir_mode`: ``2770`` for ``007``); sbatch and the job
script run under that umask (`submit_umask`). ``--output`` and ``--error`` are ``<jobid>.{out,err}`` there.
The job starts from an empty environment (``--export=NONE``) and exports only ``PATH``, the variables of
`batch_exports`, the driver interpreter and, when the profile names a ``job_tmp_root``, a per-job ``TMPDIR``
under it.
"""

from __future__ import annotations

import contextlib
import datetime
import getpass
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, NamedTuple

from senselab.utils.reproduce import run
from senselab.utils.reproduce.build import SYSTEM_PATH
from senselab.utils.reproduce.dag import toposort
from senselab.utils.reproduce.spec import RESERVED_ROOT_NAMES, Manifest, PartitionSpec, Profile, StepSpec, placeholders

SBATCH = "sbatch"
SCANCEL = "scancel"
SCONTROL = "scontrol"
SQUEUE = "squeue"
DRIVER_MODULE = "senselab.utils.reproduce"
LINK_MODE = "hardlink"
UV_EXPORTS = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")
CACHE_VARIABLE = "SENSELAB_REPRO_CACHE"
FINALIZE_STEP = "finalize"
FINALIZE_CPUS = 1
FINALIZE_MEM = "1G"
FINALIZE_TIME = "00:10:00"
SUBMIT_TIMEOUT = 120
CODES = frozenset({"E_SUBMIT_ABORTED", "E_SUBMIT_OVERLAP", "E_MANIFEST_CHANGED", "E_SCANCEL_FAILED"})
"""The submission's codes, written into ``run.json`` rows."""
_SCRIPT_OWNED = frozenset({"PATH", "TMPDIR", "REPRO_DRIVER_PYTHON"})
_EXPORT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_ID = re.compile(r"^[0-9]+$")
_SLURM_WORD = re.compile(r"^[A-Za-z0-9_.+/@:=-]+(,[A-Za-z0-9_.+/@:=-]+)*$")
_PATH_WORD = re.compile(r"^[A-Za-z0-9_.+/@:=-]+$")
_STATE = re.compile(r"^[A-Z_]{1,32}$")

Runner = Callable[[list[str], str], "subprocess.CompletedProcess[str]"]
"""Runs ``(argv, stdin)``: an sbatch, scontrol or scancel command line, with a batch script (or "") on stdin."""


class SlurmError(RuntimeError):
    """A submission that failed or was refused; the message starts with a code when one applies."""


class Terminated(BaseException):
    """SIGTERM arrived while a submission or a waited build job was in flight."""


class Finding(NamedTuple):
    """One cross-check failure against the profile: the step (or ``<profile>``), a fixed code, a detail."""

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
    """The step jobs of one submission, in DAG order, the ``--finalize`` job's id, and the submission id."""

    jobs: list[Job]
    finalize: str | None
    submission: str


def _new_submission() -> str:
    return uuid.uuid4().hex[:12]


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
        manifest_sha256s: The sha256 of each manifest file as submitted, in ``manifest_paths`` order; each job
            refuses to run (``E_MANIFEST_CHANGED``) when a file no longer has it. Empty pins nothing.
        submission: The submission id that tags every row.
    """

    profile: Profile
    profile_path: Path
    manifest_paths: list[Path]
    base_dir: Path
    cache: Path
    date: datetime.date
    exports: Mapping[str, str]
    driver_python: str = field(default_factory=lambda: sys.executable)
    manifest_sha256s: list[str] = field(default_factory=list)
    submission: str = field(default_factory=_new_submission)


def default_runner(
    argv: list[str], script: str, timeout: float | None = SUBMIT_TIMEOUT
) -> subprocess.CompletedProcess[str]:
    """Run a Slurm command with ``script`` on its standard input.

    Args:
        argv: The command; its first element is looked up on ``PATH``.
        script: Standard input (a batch script, or "").
        timeout: Seconds before the command is killed; None waits for ever.

    Returns:
        The completed process, output captured.

    Raises:
        SlurmError: If the command is not on ``PATH``.
    """
    found = shutil.which(argv[0])
    if found is None:
        raise SlurmError(f"{argv[0]} is not on PATH")
    return subprocess.run(
        [found, *argv[1:]], input=script, capture_output=True, text=True, check=False, timeout=timeout
    )


def wait_runner(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
    """`default_runner` with no timeout, for ``sbatch --wait``."""
    return default_runner(argv, script, timeout=None)


@contextlib.contextmanager
def _umask(mask: int) -> Iterator[None]:
    old = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old)


TERMINATING = (signal.SIGTERM, signal.SIGHUP)
INTERRUPTING = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@contextlib.contextmanager
def terminate_raises() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP into `Terminated` for the duration (main thread only; elsewhere a no-op)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: FrameType | None) -> None:
        raise Terminated()

    old = {sig: signal.signal(sig, handler) for sig in TERMINATING}
    try:
        yield
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


@contextlib.contextmanager
def signals_ignored() -> Iterator[None]:
    """Ignore SIGINT, SIGTERM and SIGHUP for the duration, so a second signal cannot interrupt a cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = {sig: signal.signal(sig, signal.SIG_IGN) for sig in INTERRUPTING}
    try:
        yield
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


def cancel_jobs(ids: Sequence[str], runner: Runner | None = None) -> list[str]:
    """``scancel`` the given job ids; returns the ids left uncancelled (all of them if scancel failed)."""
    if not ids:
        return []
    runner = default_runner if runner is None else runner
    try:
        result = runner([SCANCEL, *ids], "")
    except Exception:  # noqa: BLE001
        return list(ids)
    return [] if result.returncode == 0 else list(ids)


class Cancelled(NamedTuple):
    """What a submission-wide cancel did.

    ``jobs`` is the union of the recorded ids and the ids squeue lists under the submission tag;
    ``leftover`` those scancel did not confirm; ``listed`` whether squeue answered.
    """

    jobs: list[str]
    leftover: list[str]
    listed: bool


def tagged_jobs(submission: str, runner: Runner | None = None) -> list[str] | None:
    """The user's queued or running job ids whose ``--comment`` is ``submission`` (None if squeue failed).

    The calling job itself (``SLURM_JOB_ID``) is left out, so a finalize job never cancels itself.
    """
    runner = default_runner if runner is None else runner
    try:
        result = runner([SQUEUE, "-h", f"--user={getpass.getuser()}", "-o", "%i %k"], "")
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    own = os.environ.get("SLURM_JOB_ID")
    ids = []
    for line in (result.stdout or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == submission and _JOB_ID.match(parts[0]) and parts[0] != own:
            ids.append(parts[0])
    return ids


def cancel_submission(submission: str, recorded: Sequence[str], runner: Runner | None = None) -> Cancelled:
    """Cancel a submission's jobs: the recorded ids plus every job squeue lists under its ``--comment`` tag.

    The tag catches a job slurmctld accepted whose id never reached the caller (an interrupt between the
    accept and the reply).
    """
    listed = tagged_jobs(submission, runner)
    own = os.environ.get("SLURM_JOB_ID")
    union = sorted({*recorded, *(listed or [])} - {own}, key=int)
    return Cancelled(union, cancel_jobs(union, runner), listed is not None)


def job_state(job_id: str, runner: Runner | None = None) -> str | None:
    """A job's final state word from ``sacct`` (``TIMEOUT``, ``NODE_FAIL``, ``CANCELLED``…), or None."""
    if not _JOB_ID.match(job_id):
        return None
    runner = default_runner if runner is None else runner
    try:
        result = runner(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State"], "")
    except Exception:  # noqa: BLE001
        return None
    words = (result.stdout or "").split()
    if result.returncode != 0 or not words or not _STATE.match(words[0]):
        return None
    return words[0]


def submit_umask(profile: Profile) -> int:
    """The umask sbatch and each job script run under: the profile policy's ``umask``."""
    return int(profile.policy.umask, 8)


def log_dir_mode(profile: Profile) -> int:
    """The dated log dir's mode: setgid ``2777`` less the policy's umask (``2770`` for ``007``)."""
    return 0o2777 & ~submit_umask(profile)


def log_dir(profile: Profile, date: datetime.date) -> Path:
    """``<slurm_logs_root>/<YYYY-MM-DD>``: every job's working dir and where its ``<jobid>.out``/``.err`` go."""
    return profile.slurm_logs_root / date.isoformat()


def prepare_log_dir(profile: Profile, date: datetime.date) -> Path:
    """Create the dated log dir with `log_dir_mode` (setgid), or set that mode on an existing one.

    Raises:
        SlurmError: If a symlink or a non-directory is in its place.
    """
    logs = log_dir(profile, date)
    logs.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileExistsError):
        os.mkdir(logs, log_dir_mode(profile))
    if logs.is_symlink() or not logs.is_dir():
        raise SlurmError("the dated Slurm log dir is not a directory")
    os.chmod(logs, log_dir_mode(profile))
    return logs


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


def slurm_word(value: str, what: str) -> str:
    """Return ``value`` if it is a plain word or a comma-separated list of them (``gpu:a100:1,shard:1``).

    Raises:
        ValueError: Otherwise.
    """
    if not _SLURM_WORD.match(value):
        raise ValueError(f"{what} {value!r} is not a plain word or comma list")
    return value


def _path_word(value: str, what: str) -> str:
    if not _PATH_WORD.match(value):
        raise ValueError(f"{what} {value!r} is not a plain path")
    return value


def _partition_findings(owner: str, key: str, part: PartitionSpec) -> list[Finding]:
    found = []
    fields = {"partition": part.partition, "mem": part.mem, "time": part.time, "gres": part.gres}
    fields["fallback"] = part.fallback
    for name, value in fields.items():
        if value is not None and not _SLURM_WORD.match(value):
            found.append(Finding(owner, "resource-invalid", f"{key}.{name}"))
    if part.fallback is not None and part.fallback == part.partition:
        found.append(Finding(owner, "fallback-invalid", f"{key}.fallback is the partition itself"))
    return found


def profile_findings(profile: Profile) -> list[Finding]:
    """Cross-check what every job's rendering reads from the profile itself.

    Returns:
        ``logs-invalid`` (``slurm_logs_root`` is not a plain path) and the ``build`` partition's
        ``resource-invalid``/``fallback-invalid`` findings (the finalize and build jobs run there).
    """
    found = []
    if not _PATH_WORD.match(os.fspath(profile.slurm_logs_root)):
        found.append(Finding("<profile>", "logs-invalid", "slurm_logs_root"))
    found += _partition_findings("<profile>", profile.build, profile.partitions[profile.build])
    return found


def _root_names(step: StepSpec) -> list[str]:
    roots = [i.root for i in step.inputs] + [s.root for s in step.stage]
    names = [r for r in roots if not r.startswith("up:") and r not in RESERVED_ROOT_NAMES]
    templates = [*step.argv, step.cwd, *step.pythonpath, *(s.dest for s in step.stage)]
    names += [p.split(":", 1)[1] for t in templates for p in placeholders(t) if p.startswith("root:")]
    return list(dict.fromkeys(names))


def step_findings(step: StepSpec, profile: Profile) -> list[Finding]:
    """Cross-check what a step asks of the profile, so lint catches what rendering would refuse.

    Args:
        step: The step.
        profile: The site profile.

    Returns:
        ``partition-unknown`` (``resources.profile`` is no partition key); ``resource-invalid`` (a
        partition, mem, time, gres or fallback value that is not a plain word or comma list);
        ``fallback-invalid`` (the fallback is the partition itself); ``gpu-no-gres`` (``gpu_arch`` or
        ``min_driver`` on a partition with no ``gres``); ``native-unknown``; and ``root-unknown`` (an
        input, stage or ``{root:<name>}`` naming no profile root).
    """
    found: list[Finding] = []
    key = step.resources.profile
    part = profile.partitions.get(key)
    if part is None:
        found.append(Finding(step.name, "partition-unknown", key))
    else:
        found += _partition_findings(step.name, key, part)
        if (step.resources.gpu_arch or step.resources.min_driver) and part.gres is None:
            found.append(Finding(step.name, "gpu-no-gres", key))
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
        f"#SBATCH --job-name={_path_word(name, 'job name')}",
        f"#SBATCH --partition={slurm_word(partition, 'partition')}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={slurm_word(mem, 'mem')}",
        f"#SBATCH --time={slurm_word(time, 'time')}",
    ]
    if gres is not None:
        lines.append(f"#SBATCH --gres={slurm_word(gres, 'gres')}")
    logs_text = _path_word(os.fspath(logs), "log dir")
    lines += [
        f"#SBATCH --output={logs_text}/%j.out",
        f"#SBATCH --error={logs_text}/%j.err",
        f"#SBATCH --chdir={logs_text}",
        "#SBATCH --export=NONE",
    ]
    return lines


def _dependency(kind: str, ids: Sequence[str]) -> list[str]:
    bad = [i for i in ids if not _JOB_ID.match(i)]
    if bad:
        raise ValueError(f"dependency job ids must be numbers: {bad}")
    return [f"#SBATCH --dependency={kind}:{':'.join(ids)}"] if ids else []


def _body(exports: Mapping[str, str], driver_python: str, command: Sequence[str], profile: Profile) -> list[str]:
    bad = sorted(k for k in exports if not _EXPORT_NAME.match(k) or k in _SCRIPT_OWNED)
    if bad:
        raise ValueError(f"export names refused: {bad}")
    lines = ["set -euo pipefail", f"umask {profile.policy.umask}", f"export PATH={SYSTEM_PATH}"]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in exports.items()]
    lines.append(f"export REPRO_DRIVER_PYTHON={shlex.quote(driver_python)}")
    if profile.job_tmp_root is not None:
        root = _path_word(os.fspath(profile.job_tmp_root), "job_tmp_root")
        lines += [f'export TMPDIR="{root}/reproduce-${{SLURM_JOB_ID}}"', 'mkdir -p "$TMPDIR"']
        lines.append("trap 'rm -rf \"$TMPDIR\"' EXIT")
    lines.append('"$REPRO_DRIVER_PYTHON" ' + shlex.join(["-P", "-m", DRIVER_MODULE, *command]))
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
        command: The driver arguments after ``-P -m senselab.utils.reproduce`` (`step_command`).
        date: The log date; the job runs in that log dir.
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
    lines += _body(exports, driver_python, command, profile)
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
    lines += _body(exports, driver_python, command, profile)
    return "\n".join(lines) + "\n"


def step_command(config: SubmitConfig, step: str | None, *, finalize: bool = False) -> list[str]:
    """A job's driver arguments: ``run <manifests…> --profile … --date … --submission <id> [--pin <sha>…]``.

    Ends with ``--step <step>``, or ``--finalize``.
    """
    argv = ["run", *(os.fspath(p) for p in config.manifest_paths)]
    argv += ["--profile", os.fspath(config.profile_path), "--base-dir", os.fspath(config.base_dir)]
    argv += ["--cache", os.fspath(config.cache), "--date", config.date.isoformat()]
    argv += ["--submission", config.submission]
    for sha in config.manifest_sha256s:
        argv += ["--pin", sha]
    if finalize:
        return [*argv, "--finalize"]
    assert step is not None
    return [*argv, "--step", step]


# --- the probe and submission --------------------------------------------------------------------------


def choose_partition(step: StepSpec, profile: Profile, script_for: Callable[[str], str], runner: Runner) -> str:
    """The partition a step job goes to: a feasibility probe of the primary partition.

    A partition entry with a ``fallback`` is probed with ``sbatch --test-only``. The job goes to the
    fallback only when the probe rejects the primary partition (non-zero exit: unavailable, or the request
    cannot be satisfied there) or the probe itself errors or times out. A ``Preempts:`` line, which lists
    other jobs ours would preempt, is not a reason to fall back. An entry without a fallback is not probed.

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
    try:
        result = runner([SBATCH, "--test-only"], script_for(part.partition))
    except (SlurmError, OSError, subprocess.SubprocessError):
        return part.fallback
    return part.partition if result.returncode == 0 else part.fallback


def _submit_held(runner: Runner, script: str, submission: str) -> str:
    result = runner([SBATCH, "--parsable", "--hold", f"--comment={submission}"], script)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-1:] or [""]
        raise SlurmError(f"sbatch exited {result.returncode}: {tail[0]}")
    job_id = (result.stdout or "").strip().split(";", 1)[0]
    if not _JOB_ID.match(job_id):
        raise SlurmError("sbatch --parsable printed no job id")
    return job_id


def submit(manifests: Sequence[Manifest], config: SubmitConfig, runner: Runner | None = None) -> Submission:
    """Reserve the run dirs, submit held jobs in DAG order (``afterok`` their needs) and a finalize job, release.

    Order:

    1. cross-checks (`profile_findings`, `step_findings`);
    2. render every script: each step's primary and fallback partition, and the finalize job (a render
       error raises here, before anything is written or submitted);
    3. reserve: for each analysis in sorted order, under its ``run.json`` lock, refuse another submission's
       ``pending`` rows (``E_SUBMIT_OVERLAP``) and write placeholder ``pending`` rows tagged with this
       submission (job null); an overlap undoes the reservations already made;
    4. under `submit_umask`, with SIGTERM and SIGHUP raising `Terminated`: create the dated log dir
       (`log_dir_mode`); probe partitions; submit every step job ``--hold``; fill in the job ids under the lock
       (this submission's rows only); submit the finalize job ``--hold``; ``scontrol release`` them all.

    Any exception, Ctrl-C, SIGTERM or SIGHUP in step 4 runs the abort with those signals ignored: every
    submitted job is cancelled (a failing ``scancel`` is reported as ``E_SCANCEL_FAILED`` with the leftover
    job ids), and this submission's still-pending rows get back the row each replaced, or become
    ``never-ran`` with ``E_SUBMIT_ABORTED``. Rows of other submissions are never touched.

    Args:
        manifests: The manifests; step names are unique across them.
        config: The submission's shared settings.
        runner: Runs sbatch, scontrol and scancel; defaults to `default_runner`.

    Returns:
        The submission.

    Raises:
        SlurmError: On a cross-check finding, an overlap, or an aborted submission.
        ValueError: If a script cannot be rendered (nothing was written or submitted).
    """
    runner = default_runner if runner is None else runner
    profile, submission = config.profile, config.submission
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    ordered = toposort([s for m in manifests for s in m.steps])
    findings = profile_findings(profile) + [f for s in ordered for f in step_findings(s, profile)]
    if findings:
        raise SlurmError(f"{len(findings)} profile cross-check finding(s); run lint")
    orders = {name: [s.name for s in ordered if analysis[s.name] == name] for name in sorted(set(analysis.values()))}

    def script_for(step: StepSpec, partition: str, needs: Sequence[str] = ()) -> str:
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

    def finalize_script(ids: Sequence[str]) -> str:
        return render_finalize_job(
            profile,
            command=step_command(config, None, finalize=True),
            date=config.date,
            exports=config.exports,
            dependencies=ids,
            driver_python=config.driver_python,
        )

    for step in ordered:
        part = profile.partitions[step.resources.profile]
        for partition in (part.partition, part.fallback):
            if partition is not None:
                script_for(step, partition)
    finalize_script([])

    snapshots: dict[str, dict[str, dict[str, Any] | None]] = {}
    submitted: list[str] = []
    jobs: list[Job] = []
    with _umask(submit_umask(profile)), terminate_raises():
        try:
            for name, order in orders.items():
                snapshots[name] = run.reserve_run_record(profile.runs_root, config.date, name, order, submission)
        except BaseException as error:
            with signals_ignored():
                for name, snapshot in snapshots.items():
                    with contextlib.suppress(Exception):
                        run.release_reservation(
                            profile.runs_root, config.date, name, orders[name], submission, snapshot, None
                        )
            if isinstance(error, run.SubmissionOverlap):
                raise SlurmError(
                    f"E_SUBMIT_OVERLAP: {error.analysis} has pending rows from submission(s) "
                    f"{', '.join(error.others)}; wait for them, or settle them with `run --finalize --submission <id>`"
                ) from None
            _interrupted(submission, error)
            raise
        try:
            prepare_log_dir(profile, config.date)
            by_step: dict[str, str] = {}
            for step in ordered:

                def probe_script(partition: str, step: StepSpec = step) -> str:
                    return script_for(step, partition)

                partition = choose_partition(step, profile, probe_script, runner)
                job_id = _submit_held(runner, script_for(step, partition, [by_step[n] for n in step.needs]), submission)
                submitted.append(job_id)
                by_step[step.name] = job_id
                jobs.append(Job(step.name, analysis[step.name], job_id, partition))
            for name, order in orders.items():
                ids = [{"step": s, "job": by_step[s]} for s in order]
                run.update_run_record(
                    profile.runs_root, config.date, name, ids, order=order, pending_only=True, submission=submission
                )
            finalize = _submit_held(runner, finalize_script(submitted), submission) if submitted else None
            if finalize is not None:
                submitted.append(finalize)
            if submitted:
                _check(runner([SCONTROL, "release", ",".join(submitted)], ""), "scontrol release")
        except BaseException as error:
            cancelled = _abort(runner, config, orders, snapshots, submitted)
            _interrupted(submission, error)
            reason = type(error).__name__ if not isinstance(error, (SlurmError, ValueError)) else str(error)
            message = f"E_SUBMIT_ABORTED: {reason}; cancelled job(s) {','.join(cancelled.jobs) or 'none'}"
            if cancelled.leftover:
                message += f"; E_SCANCEL_FAILED: job(s) {','.join(cancelled.leftover)} may still be queued"
            if not cancelled.listed:
                message += "; squeue failed, so jobs with an unrecorded id may remain (tag " + submission + ")"
            raise SlurmError(message) from None
    return Submission(jobs, finalize, submission)


def _check(result: subprocess.CompletedProcess[str], what: str) -> None:
    if result.returncode != 0:
        raise SlurmError(f"{what} exited {result.returncode}")


def _interrupted(submission: str, error: BaseException) -> None:
    """On an interrupt, name the submission on stderr (its id only), so it can be settled by hand."""
    if isinstance(error, (KeyboardInterrupt, Terminated)):
        print(f"interrupted: submission {submission}", file=sys.stderr)


def _abort(
    runner: Runner,
    config: SubmitConfig,
    orders: Mapping[str, list[str]],
    snapshots: Mapping[str, Mapping[str, Mapping[str, Any] | None]],
    submitted: list[str],
) -> Cancelled:
    """With INT/TERM/HUP ignored: cancel the submission's jobs (recorded and tagged), undo its pending rows."""
    with signals_ignored():
        cancelled = cancel_submission(config.submission, submitted, runner)
        for name, order in orders.items():
            with contextlib.suppress(Exception):
                run.release_reservation(
                    config.profile.runs_root,
                    config.date,
                    name,
                    order,
                    config.submission,
                    snapshots.get(name, {}),
                    ("E_SUBMIT_ABORTED",),
                )
    return cancelled


def submit_and_wait(script: str, *, name: str, umask: int, runner: Runner | None = None) -> None:
    """Submit a batch job with ``sbatch --wait`` and no timeout; cancel it if the wait is interrupted.

    The job is named ``name`` on the command line (overriding the script's), so an interrupted wait
    (Ctrl-C, SIGTERM, any exception) can ``scancel --name`` it: no orphan job finishes later.

    Args:
        script: The batch script.
        name: A job name unique to this wait.
        umask: The umask the sbatch runs under (`submit_umask`).
        runner: Runs the waiting sbatch; defaults to `wait_runner`. ``scancel`` uses `default_runner`.

    Raises:
        SlurmError: If the job failed.
    """
    runner = wait_runner if runner is None else runner
    argv = [SBATCH, "--wait", "--parsable", f"--job-name={_path_word(name, 'job name')}"]
    try:
        with _umask(umask), terminate_raises():
            result = runner(argv, script)
    except BaseException:
        with signals_ignored():
            cancelled = False
            with contextlib.suppress(Exception):
                done = default_runner([SCANCEL, f"--name={name}", f"--user={getpass.getuser()}"], "")
                cancelled = done is not None and done.returncode == 0
            if not cancelled:
                print(f"error: E_SCANCEL_FAILED: batch job {name} may still be queued", file=sys.stderr)
        raise
    if result.returncode != 0:
        raise SlurmError(f"batch job {name} failed (sbatch exit {result.returncode})")
