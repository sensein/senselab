"""Substrates: `scale` within the caps, the `sbatch` line of a resource class, excluded nodes, queue names, defaults."""

import contextlib
import errno
import math
import os
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog, read_events
from senselab_core.runtime.execution.executor import WorkersBroken, follow
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.pool import Pool, ScalerFailed
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import Run, _close_owned, default_executor, prepare
from senselab_core.runtime.execution.substrates import (
    EXIT_UNREACHABLE,
    ExternalWorkers,
    Fleet,
    Local,
    LocalError,
    Slurm,
    SlurmError,
    Substrate,
    WorkerHandle,
    class_of,
    profile_argument,
    scale,
    substrate_of,
)
from senselab_core.runtime.execution.supervisor import Supervisor
from senselab_core.runtime.execution_settings import LeaseBoundError, PoolSettings, SlurmCommands
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile, QosCapTooSmall
from senselab_core.runtime.storage import StorageCapabilities, StorageUnsupported
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.measures import CEILED_SIZING, FixedMeasures
from syscall_faults_fixture import refuse_call

SCORE = scoring("substrates_test.score")
REGISTRY = registry(SCORE)
HASH = "ab" * 32
MODEL = ResolvedModel(ref=ModelRef(source="hf", id="org/name", ref="main"), identity="c" * 40)
CLASSES: dict[str, Any] = {
    "cpu": {
        "device": "cpu",
        "cpus": 4,
        "mem_bytes": 2**33,
        "time_s": 3600,
        "partition": "${CPU_PARTITION}",
        "max_workers": 32,
    },
    "gpu:h100": {
        "device": "cuda",
        "gpu_type": "h100",
        "cpus": 8,
        "mem_bytes": 2**36,
        "time_s": 7200,
        "partition": "${GPU_PARTITION}",
        "gres": "gpu:h100:1",
        "max_workers": 4,
    },
}

SBATCH = """#!/bin/sh
d="$FAKE_SLURM"
[ -f "$d/fail-sbatch" ] && { echo "Socket timed out" >&2; exit 1; }
n=$(cat "$d/next" 2>/dev/null || echo 100)
echo $((n + 1)) > "$d/next"
for a in "$@"; do printf '%s\\n' "$a" >> "$d/sbatch.log"; done
echo "--" >> "$d/sbatch.log"
comment=""; name=""
for a in "$@"; do
  case "$a" in
    --comment=*) comment="${a#--comment=}";;
    --job-name=*) name="${a#--job-name=}";;
  esac
done
mkdir -p "$d/jobs"
echo "$name|$comment|RUNNING" > "$d/jobs/$n"
echo "$n"
"""
SQUEUE = """#!/bin/sh
d="$FAKE_SLURM"
[ -f "$d/fail-squeue" ] && { echo "Socket timed out" >&2; exit 1; }
echo "$@" >> "$d/squeue.log"
for j in "$d"/jobs/*; do
  [ -f "$j" ] || continue
  IFS='|' read -r name comment state < "$j"
  echo "$(basename "$j")|$state|$comment|$name"
done
exit 0
"""
SCANCEL = """#!/bin/sh
for j in "$@"; do rm -f "$FAKE_SLURM/jobs/$j"; echo "$j" >> "$FAKE_SLURM/cancelled"; done
"""


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry and the cache under `tmp_path`, with a fake `sbatch`, `squeue` and `scancel` on `PATH`."""
    bin_dir, state = tmp_path / "bin", tmp_path / "slurm"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in (("sbatch", SBATCH), ("squeue", SQUEUE), ("scancel", SCANCEL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(state))
    monkeypatch.setenv("CPU_PARTITION", "cpuq")
    monkeypatch.setenv("GPU_PARTITION", "gpuq")
    with packaged(REGISTRY, tmp_path / "cache"):
        yield


def _submitted(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "slurm" / "sbatch.log"
    calls: list[list[str]] = [[]]
    for line in log.read_text().splitlines() if log.exists() else []:
        calls.append([]) if line == "--" else calls[-1].append(line)
    return [c for c in calls if c]


def _profile(**substrate: Any) -> ExecutionProfile:  # noqa: ANN401 - substrate keys of any type
    kind = substrate.pop("kind", "slurm")
    override = {
        "devices": ["cpu"],
        "resource_classes": CLASSES,
        "substrate": {"kind": kind, "commands": {"retry_first_s": 0.001}, **substrate},
        "pool": {"quantum_items": 1, "first_poll_s": 0.01, "poll_s": 0.2},
        "sizing": CEILED_SIZING,
    }
    return ExecutionProfile.load("orcd", override)


def _commands(**update: float) -> SlurmCommands:
    commands = _profile().substrate.commands
    assert commands is not None
    return commands.model_copy(update=update)


def _fleet(substrate: Substrate, profile: ExecutionProfile, workers_dir: Path | None = None) -> Fleet:
    return Fleet(
        substrate,
        workers_dir=workers_dir,
        budget=profile.faults.infrastructure_replacements,
        first_s=profile.substrate.start_backoff_first_s,
        max_s=profile.substrate.start_backoff_max_s,
    )


class Recording:
    """A `Substrate` that records each start and holds its workers, each pending when started."""

    def __init__(self) -> None:
        """Start with no worker."""
        self.handles: list[WorkerHandle] = []
        self.starts: list[tuple[str, int]] = []
        self.stops: list[WorkerHandle] = []
        self._next = 0

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Record `count` pending workers of the queue."""
        self.starts.append((resource_class, count))
        made = [WorkerHandle(resource_class, f"w{self._next + i}", pending=True) for i in range(count)]
        self._next += count
        self.handles.extend(made)
        return made

    def live(self) -> list[WorkerHandle]:
        """Return the workers held."""
        return list(self.handles)

    def running(self, resource_class: str) -> int:
        """Return the workers held for the queue."""
        return sum(1 for h in self.handles if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Forget `handles`."""
        self.stops.extend(handles)
        gone = {h.id for h in handles}
        self.handles = [h for h in self.handles if h.id not in gone]


def _queued(  # noqa: PLR0913 - the plan's shape, then where it is queued
    profile: ExecutionProfile,
    tmp_path: Path,
    items: int,
    model: bool = False,
    substrate: Any = None,  # noqa: ANN401
    scaled: bool = False,
) -> tuple[Plan, Pool, Recording]:
    """Return a plan of `items` jobs, one fragment per job, and a pool over `substrate` (a recording one by default).

    The fragments are submitted through a closed pool over `ExternalWorkers`, so the pool returned has not scaled;
    with `scaled` they are submitted through the pool returned, once its scaler has started workers.
    """
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(items)])
    if model:
        models = {n: MODEL for n in placed.logical.models}
        placed = replace(placed, logical=placed.logical.model_copy(update={"models": models}))
    run_dir = tmp_path / "run"
    recording = substrate if substrate is not None else Recording()
    epoch = Epoch(run_dir).take()

    def pool_over(workers: Any) -> Pool:  # noqa: ANN401
        return Pool(
            run_dir,
            queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
            store=ResultStore(tmp_path / "store"),
            events=EventLog(run_dir, "a1", epoch, epoch=epoch),
            profile=profile,
            substrate=workers,
            epoch=epoch,
            measures=FixedMeasures(),
        )

    pool = pool_over(recording)
    feeder = pool if scaled else pool_over(ExternalWorkers())
    admission = feeder.admit(placed, values)
    for element in range(items):
        feeder.submit(node_fragment(placed, admission, elements=[(element,)]))
    if scaled:
        deadline = time.monotonic() + 10
        while not recording.starts and time.monotonic() < deadline:
            time.sleep(0.01)
    else:
        feeder.close()
    return placed, pool, recording


@pytest.mark.parametrize(
    ("cap", "depth", "expected"),
    [(10, 20, 2), (None, 20, 20), (None, 40, 32), (100, 20, 20)],
    ids=["qos-cap", "depth", "class-max", "cap-above-depth"],
)
@pytest.mark.logic
def test_scale_bounded_by_qos_cap(tmp_path: Path, cap: int | None, depth: int, expected: int) -> None:
    """Workers started are the depth over the quantum, within the class's `max_workers` and cap ÷ CPUs per worker."""
    profile = _profile(qos_cpu_cap=cap)
    _, pool, recording = _queued(profile, tmp_path, depth)
    recording.starts.clear()
    recording.handles.clear()
    scale(pool.queues, recording, profile, workers=_per_quantum(profile))
    assert len(recording.handles) == expected
    scale(pool.queues, recording, profile, workers=_per_quantum(profile))
    assert len(recording.handles) == expected


@pytest.mark.logic
def test_scale_counts_workers_alive_and_a_quantum_of_items(tmp_path: Path) -> None:
    """Five items over a quantum of 2 want three workers; one alive, two are started; none once three are alive."""
    profile = ExecutionProfile.load("orcd", {**_override(), "pool": {"quantum_items": 2, "first_poll_s": 0.01}})
    _, pool, recording = _queued(profile, tmp_path, 5)
    recording.starts.clear()
    recording.handles = [WorkerHandle("cpu", "x")]
    scale(pool.queues, recording, profile, workers=_per_quantum(profile))
    assert recording.starts == [("cpu", 2)]
    scale(pool.queues, recording, profile, workers=_per_quantum(profile))
    assert recording.starts == [("cpu", 2)]


def _per_quantum(profile: ExecutionProfile) -> Callable[[str, int], int]:
    """A queue wants a worker per `pool.quantum_items` of its waiting jobs."""
    return lambda _, work: math.ceil(work / profile.pool.quantum_items)


def _override() -> dict[str, Any]:
    return {"devices": ["cpu"], "resource_classes": CLASSES, "substrate": {"kind": "slurm"}, "sizing": CEILED_SIZING}


@pytest.mark.logic
def test_slurm_args_from_resource_class(tmp_path: Path) -> None:
    """One `sbatch` per worker: the class's partition, gres, CPUs, memory and time, named by plan hash and attempt."""
    slurm = Slurm(
        _profile().resource_classes,
        None,
        command=["senselab", "execute"],
        plan_hash=HASH,
        run_dir=tmp_path,
        attempt="a7",
        commands=_commands(),
    )
    handles = slurm.start("gpu:h100#" + "c" * 40, 2)
    assert [h.id for h in handles] == ["100", "101"]
    [first, second] = _submitted(tmp_path)
    assert first[0] == "--parsable" and first[2:4] == [f"--comment=gpu:h100#{'c' * 40}", "--partition=gpuq"]
    names = [call[1].removeprefix("--job-name=") for call in (first, second)]
    assert all(n.startswith(f"{slurm.prefix}a7-") and n.endswith("-e0") for n in names) and names[0] != names[1]
    assert {"--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=65536M", "--time=120"} <= set(first)
    assert first[-2] == "--wrap" and first[-1] == f"senselab execute --classes 'gpu:h100#{'c' * 40}'"
    assert slurm.running("cpu") == 0 and slurm.running(f"gpu:h100#{'c' * 40}") == 2
    slurm.stop(handles[:1])
    assert slurm.running(f"gpu:h100#{'c' * 40}") == 1


@pytest.mark.logic
def test_slurm_refuses_an_unset_placeholder_and_an_unknown_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partition naming an unset variable, or a queue of no class, submits nothing."""
    slurm = Slurm(
        _profile().resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path, commands=_commands()
    )
    monkeypatch.delenv("CPU_PARTITION")
    with pytest.raises(ValueError, match="CPU_PARTITION"):
        slurm.start("cpu", 1)
    with pytest.raises(ValueError, match="no resource class"):
        slurm.start("gpu:a100", 1)
    assert _submitted(tmp_path) == []


@pytest.mark.logic
def test_slurm_names_a_failed_submission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failing `sbatch` raises `SlurmError` with its message."""
    (tmp_path / "bin" / "sbatch").write_text("#!/bin/sh\necho 'invalid partition' >&2\nexit 1\n")
    slurm = Slurm(
        _profile().resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path, commands=_commands()
    )
    with pytest.raises(SlurmError, match="invalid partition"):
        slurm.start("cpu", 1)


@pytest.mark.logic
def test_excluded_node_never_targeted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every submission excludes the profile's nodes, `${VAR}` expanded, and names no node to run on."""
    monkeypatch.setenv("BAD_NODE", "node-b")
    profile = _profile(exclude_nodes=["node-a", "${BAD_NODE}"])
    slurm = Slurm(
        profile.resource_classes,
        None,
        command=["x"],
        plan_hash=HASH,
        run_dir=tmp_path,
        exclude_nodes=profile.substrate.exclude_nodes,
        commands=_commands(),
    )
    slurm.start("cpu", 1)
    slurm.start("gpu:h100", 1)
    calls = _submitted(tmp_path)
    assert len(calls) == 2 and all("--exclude=node-a,node-b" in c for c in calls)
    assert not any(a.startswith(("--nodelist", "-w")) for c in calls for a in c)


@pytest.mark.logic
def test_slurm_stays_within_its_qos_cap(tmp_path: Path) -> None:
    """With a cap of 10 CPUs and 4 per worker, a request for 5 submits 2, and none while they are alive."""
    slurm = Slurm(
        _profile().resource_classes, 10, command=["x"], plan_hash=HASH, run_dir=tmp_path, commands=_commands()
    )
    assert len(slurm.start("cpu", 5)) == 2
    assert slurm.start("cpu", 5) == []
    assert len(_submitted(tmp_path)) == 2


@pytest.mark.logic
def test_unsupported_store_refused_at_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan whose run directory refuses `mkdir` as unsupported is `StorageUnsupported`; nothing is planned."""
    profile = _profile(kind="local")
    refuse_call(monkeypatch, "mkdir", ".senselab-probe", OSError(errno.ENOTSUP, "Operation not supported"))
    with pytest.raises(StorageUnsupported, match="atomic mkdir"):
        planned(SCORE, profile, ["one"])


@pytest.mark.logic
def test_a_store_missing_a_primitive_or_delaying_too_long_is_refused(tmp_path: Path) -> None:
    """`check_storage` names the missing primitive, and refuses a measured delay that breaks the lease bound."""
    profile = _profile()
    lacking = StorageCapabilities(True, True, True, False, True, 0.0)
    with pytest.raises(StorageUnsupported, match="atomic_rename"):
        check_storage(profile, tmp_path, probe=lambda root: lacking)
    slow = StorageCapabilities(True, True, True, True, True, profile.lease.stale_after_s)
    with pytest.raises(ValueError, match="attribute_delay_s"):
        check_storage(profile, tmp_path, probe=lambda root: slow)
    check_storage(profile, tmp_path, probe=lambda root: StorageCapabilities(True, True, True, True, True, 1.0))


@pytest.mark.logic
def test_a_queue_is_named_by_class_and_resident_model(tmp_path: Path) -> None:
    """A node with a resident model queues on `<class>#<identity>`; scaling starts workers of that queue."""
    profile = _profile()
    _, pool, recording = _queued(profile, tmp_path, 3, model=True, scaled=True)
    assert pool.queues.names() == [f"cpu#{'c' * 40}"]
    assert recording.starts and {name for name, _ in recording.starts} == {f"cpu#{'c' * 40}"}
    assert class_of(f"cpu#{'c' * 40}") == "cpu" == class_of("cpu")
    _, other, _ = _queued(profile, tmp_path / "plain", 1)
    assert other.queues.names() == ["cpu"]


@pytest.mark.logic
def test_a_slurm_profile_defaults_to_a_pool_over_slurm_and_a_local_one_to_in_process(tmp_path: Path) -> None:
    """With no executor, a slurm profile runs on `Pool` over `Slurm`, a local one on `InProcess`."""
    store = ResultStore(tmp_path / "store")
    slurm, _ = planned(SCORE, _profile(), ["one"])
    found = default_executor(slurm, None, store)
    assert isinstance(found, Pool) and isinstance(found.substrate, Slurm) and found.substrate.plan_hash == slurm.hash
    assert found.substrate.qos_cpu_cap is None and found.substrate.classes == dict(_profile().resource_classes)
    local, _ = planned(SCORE, _profile(kind="local"), ["one"])
    assert isinstance(default_executor(local, None, store), InProcess)


@pytest.mark.logic
def test_a_pool_names_its_slurm_jobs_by_its_attempt(tmp_path: Path) -> None:
    """The pool binds its attempt to the substrate, so jobs are named by the plan hash and that attempt."""
    profile = _profile()
    run_dir = tmp_path / "run"
    slurm = Slurm(profile.resource_classes, None, command=["x"], plan_hash=HASH, run_dir=run_dir, commands=_commands())
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=slurm,
        epoch=0,
        measures=FixedMeasures(),
    )
    pool.bind_epoch(3, EventLog(run_dir, "attempt-3", 3, epoch=3))
    assert slurm.job_name("t") == f"{slurm.prefix}attempt-3-t-e3"


@pytest.mark.logic
def test_a_custom_profile_reaches_its_workers_as_a_file(tmp_path: Path) -> None:
    """A packaged profile is named; a changed one is written whole under the run's `private/`, to the same digest."""
    assert profile_argument(ExecutionProfile.load("orcd"), tmp_path) == "orcd"
    custom = _profile(qos_cpu_cap=12, exclude_nodes=["n1"])
    written = Path(profile_argument(custom, tmp_path))
    loaded = ExecutionProfile.model_validate_json(written.read_text())
    assert loaded.canonical_digest() == custom.canonical_digest()
    assert written.parent == tmp_path / "private" and written.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.logic
def test_slurm_worker_logs_go_under_the_runs_private_directory(tmp_path: Path) -> None:
    """A worker's `--output` is under `<run>/private/slurm`, made owner-only before the first submission."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    slurm = substrate_of(_profile(), run_dir, HASH)
    slurm.start("cpu", 1)
    [args] = _submitted(tmp_path)
    logs = run_dir / "private" / "slurm"
    assert f"--output={logs}/%x-%j.out" in args and logs.stat().st_mode & 0o777 == 0o700


@pytest.mark.logic
def test_local_runs_at_most_max_workers_and_stops_them(tmp_path: Path) -> None:
    """A local substrate starts `senselab execute`-shaped processes up to its bound and stops them on request."""
    local = Local(1, command=[sys.executable, "-c", "import time; time.sleep(60)"])
    handles = local.start("cpu", 2)
    try:
        assert len(handles) == 1 and local.running("cpu") == 1 and local.running("gpu") == 0
        assert local.start("cpu", 1) == []
    finally:
        local.stop(handles)
    assert local.running("cpu") == 0
    fleet = _fleet(local, _profile())
    assert len(fleet.start("cpu", 1)) == 1 and local.running("cpu") == 1
    fleet.close()
    assert local.running("cpu") == 0


DEVICE_CHILD = (
    "import os, pathlib, sys, time; "
    "pathlib.Path(sys.argv[1], str(os.getpid())).write_text(os.environ.get('CUDA_VISIBLE_DEVICES', '-')); "
    "time.sleep(60)"
)


def _devices_seen(directory: Path, count: int, waits: PoolSettings) -> dict[str, str]:
    """Wait, within `waits.poll_s`, until `count` workers wrote the device they were given; return it by process id."""
    deadline = time.monotonic() + waits.poll_s
    while len(written := list(directory.iterdir())) < count:
        assert time.monotonic() < deadline, f"{len(written)} of {count} workers wrote their device"
        time.sleep(waits.first_poll_s)
    return {p.name: p.read_text() for p in written}


@pytest.mark.logic
def test_local_runs_one_cuda_worker_per_visible_device(tmp_path: Path, slack: dict[str, Any]) -> None:
    """A cuda class's workers each get a free device as `CUDA_VISIBLE_DEVICES`; no more start than there are devices."""
    waits = ExecutionProfile.load("orcd", slack).pool
    seen = tmp_path / "seen"
    seen.mkdir()
    local = Local(8, command=[sys.executable, "-c", DEVICE_CHILD, str(seen)], cuda=["gpu"], devices=["G-0", "G-1"])
    handles = local.start("gpu#m", 3)
    try:
        assert len(handles) == 2 and local.start("gpu#n", 1) == []
        assert sorted(_devices_seen(seen, 2, waits).values()) == ["G-0", "G-1"]
        local.stop(handles[:1])
        handles = [*handles[1:], *local.start("gpu#n", 2)]
        assert len(handles) == 2 and set(_devices_seen(seen, 3, waits).values()) == {"G-0", "G-1"}
        handles += local.start("cpu", 2)
        assert local.running("cpu") == 2 and "-" in _devices_seen(seen, 5, waits).values()
    finally:
        local.stop(handles)
    with pytest.raises(LocalError, match="no device is visible"):
        Local(1, command=[sys.executable], cuda=["gpu"]).start("gpu", 1)


@pytest.mark.logic
def test_external_workers_start_nothing() -> None:
    """The substrate for workers started on their own starts none and counts none."""
    external = ExternalWorkers()
    assert external.start("cpu", 3) == [] and external.running("cpu") == 0


CHILD = """
import sys
from pathlib import Path
from pool_stack_fixture import packaged, registry, scoring
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution_settings import LeaseBoundError
from senselab_core.runtime.profiles import ExecutionProfile, QosCapTooSmall
from senselab_core.runtime.store import ResultStore

pool_dir, cache, store = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
classes = sys.argv[sys.argv.index("--classes") + 1].split(",")
profile = ExecutionProfile.model_validate_json(Path(sys.argv[4]).read_text())
with packaged(registry(scoring("substrates_test.score")), cache):
    sys.exit(serve_pool(pool_dir, classes, profile=profile, store=ResultStore(store)))
"""


@pytest.mark.limits
def test_local_pool_runs_toy_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool of two local worker processes runs a six-job plan: every job done once, within ten seconds."""
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": {"cpu": {"device": "cpu", "max_workers": 2}},
            "pool": {"quantum_items": 1, "poll_s": 0.2, "first_poll_s": 0.02},
        },
    )
    placed, _ = planned(SCORE, profile, [f"t{i}" for i in range(6)])
    run_dir, store = tmp_path / "run", ResultStore(tmp_path / "store")
    prepare(placed, run_dir, store)
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(profile.model_dump_json())
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
    command = [sys.executable, "-c", CHILD, str(run_dir / RUN_QUEUES), str(tmp_path / "cache"), str(tmp_path / "store")]
    local = Local(2, command=[*command, str(profile_file)])
    began = time.monotonic()
    outcome = coordinate(run_dir, substrate=local, profile=profile, store=store)
    assert time.monotonic() - began < 10
    assert outcome.done and len(outcome.results["a"]) == 6 and all(r.failure is None for r in outcome.results["a"])
    status = Run(run_dir).wait(None)
    assert status.done and status.counts["done"] == 6
    assert sum(e.kind == "claimed" for e in read_events(run_dir)) == 6
    assert local.running("cpu") == 0


def _worker_profile(**faults: int) -> ExecutionProfile:
    return ExecutionProfile.load(
        "laptop",
        {
            "devices": ["cpu"],
            "resource_classes": {"cpu": {"device": "cpu", "max_workers": 2}},
            "pool": {"quantum_items": 1, "first_poll_s": 0.01, "poll_s": 0.2},
            "faults": {"infrastructure_replacements": 2, **faults},
        },
    )


@pytest.mark.logic
def test_failed_starts_back_off_and_stop_at_the_budget() -> None:
    """Workers that vanish before marking themselves ready are restarted after 1, 2, 4 s, then the class is broken."""
    now = [0.0]
    substrate = Recording()
    fleet = Fleet(substrate, workers_dir=None, budget=3, first_s=1.0, max_s=4.0, clock=lambda: now[0])
    starts: list[float] = []
    for step in range(60):
        now[0] = step * 0.5
        substrate.handles.clear()
        fleet.live()
        if fleet.start("cpu", 1):
            starts.append(now[0])
    assert starts == [0.0, 1.5, 4.0, 8.5] and fleet.broken["cpu"].failures == 4
    assert fleet.start("cpu#abc", 1) == [] and fleet.broken_for("cpu#abc") is fleet.broken["cpu"]
    fleet.reset()
    assert fleet.broken == {} and len(fleet.start("cpu", 1)) == 1


@pytest.mark.logic
def test_a_worker_that_exits_at_once_makes_a_bounded_number_of_starts(tmp_path: Path) -> None:
    """A command that exits at once gives at most budget + 1 starts through `scale`, then the class is broken."""
    profile = _worker_profile()
    _, pool, _ = _queued(profile, tmp_path, 3)
    local = Local(2, command=[sys.executable, "-c", "raise SystemExit(1)"])
    fleet = Fleet(local, workers_dir=tmp_path / "run" / "workers", budget=2, first_s=0.0, max_s=0.0)
    for _ in range(40):
        scale(pool.queues, fleet, profile, workers=_per_quantum(profile))
        time.sleep(0.1)
    assert fleet.starts["cpu"] == 3 and fleet.broken["cpu"].failures == 3


@pytest.mark.logic
def test_a_pool_whose_workers_never_start_holds_its_jobs(tmp_path: Path) -> None:
    """The pool raises `WorkersBroken` for a class that cannot start workers and keeps the item; a new epoch resets."""
    profile = _worker_profile(infrastructure_replacements=1)
    local = Local(2, command=[sys.executable, "-c", "raise SystemExit(1)"])
    _, pool, _ = _queued(profile, tmp_path, 0, substrate=local)
    placed, values = planned(SCORE, profile, ["one"])
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
    with pytest.raises(WorkersBroken, match="failed to start"):
        list(follow(pool, handle))
    assert "cpu" in pool.broken_workers and pool.queues.queue("cpu").depth() == 1
    pool.bind_epoch(pool.epoch + 1, EventLog(tmp_path / "run", "a2", pool.epoch + 1, epoch=pool.epoch + 1))
    assert pool._fleet.broken == {}  # noqa: SLF001 - the budget a new epoch resets


@pytest.mark.logic
def test_a_worker_that_reaches_the_run_marks_itself_and_one_that_cannot_exits_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve_pool` marks the worker ready; with the run directory unreachable it exits `EXIT_UNREACHABLE`, unmarked."""
    profile = _worker_profile()
    store = ResultStore(tmp_path / "store")
    run_dir = tmp_path / "run"
    assert serve_pool(run_dir / RUN_QUEUES, ["cpu"], profile=profile, store=store) == 0
    assert (run_dir / "workers" / str(os.getpid())).exists()
    other = tmp_path / "other"
    refuse_call(monkeypatch, "mkdir", ".senselab-probe", OSError(errno.ENOTSUP, "Operation not supported"))
    assert serve_pool(other / RUN_QUEUES, ["cpu"], profile=profile, store=store) == EXIT_UNREACHABLE
    assert not (other / "workers").exists()


def _slurm(tmp_path: Path, epoch: int = 1, plan_hash: str = HASH, commands: SlurmCommands | None = None) -> Slurm:
    return Slurm(
        _profile().resource_classes,
        None,
        command=["x"],
        plan_hash=plan_hash,
        run_dir=tmp_path / "run",
        attempt="a1",
        epoch=epoch,
        commands=commands or _commands(),
    )


@pytest.mark.logic
def test_slurm_jobs_are_scoped_to_the_user_run_and_epoch(tmp_path: Path) -> None:
    """`squeue --me` lists the run's jobs only; a later epoch cancels earlier epochs' jobs and ignores later ones."""
    first = _slurm(tmp_path, epoch=1)
    mine = first.start("cpu", 2)
    _slurm(tmp_path, plan_hash="cd" * 32).start("cpu", 1)
    assert [h.id for h in first.live()] == [h.id for h in mine] and "--me" in (
        tmp_path / "slurm" / "squeue.log"
    ).read_text()
    second = _slurm(tmp_path, epoch=2)
    assert second.live() == []
    assert (tmp_path / "slurm" / "cancelled").read_text().split() == [h.id for h in mine]
    later = second.start("cpu", 1)
    assert [h.id for h in first.live()] == [] and [h.id for h in second.live()] == [later[0].id]
    assert first.job_name != second.job_name


@pytest.mark.logic
def test_pending_surplus_is_cancelled_first_and_running_ones_only_when_drained(tmp_path: Path) -> None:
    """One item wants one worker: pending extras go once the surplus lasts; an empty queue stops every worker."""
    profile = _profile()
    _, pool, recording = _queued(profile, tmp_path, 1)
    recording.handles = [WorkerHandle("cpu", "p1", True), WorkerHandle("cpu", "r1"), WorkerHandle("cpu", "p2", True)]
    since: dict[str, float] = {}
    scale(pool.queues, recording, profile, workers=_per_quantum(profile), surplus_since=since, now=0.0)
    assert recording.stops == [] and since == {"cpu": 0.0}
    scale(pool.queues, recording, profile, workers=_per_quantum(profile), surplus_since=since, now=10.0)
    assert {h.id for h in recording.stops} == {"p1", "p2"} and [h.id for h in recording.handles] == ["r1"]
    empty = QueueSet(tmp_path / "empty" / RUN_QUEUES, profile)
    empty.queue("cpu")
    recording.handles = [WorkerHandle("cpu", "r1"), WorkerHandle("cpu", "p3", True)]
    recording.stops.clear()
    scale(empty, recording, profile, workers=_per_quantum(profile))
    assert [h.id for h in recording.stops] == ["p3", "r1"] and recording.handles == []


@pytest.mark.logic
def test_an_empty_service_queue_never_stops_its_running_worker(tmp_path: Path) -> None:
    """A `svc` queue with nothing waiting stops its pending workers only; the running one exits on its own."""
    classes = {
        **CLASSES,
        "svc:pii": {"device": "cpu", "cpus": 1, "partition": "${CPU_PARTITION}", "max_workers": 1},
    }
    profile = ExecutionProfile.load("orcd", {"devices": ["cpu"], "resource_classes": classes})
    empty = QueueSet(tmp_path / "empty" / RUN_QUEUES, profile)
    empty.queue("svc:pii")
    recording = Recording()
    recording.handles = [WorkerHandle("svc:pii", "r1"), WorkerHandle("svc:pii", "p1", True)]
    scale(empty, recording, profile, workers=_per_quantum(profile))
    assert [h.id for h in recording.stops] == ["p1"] and [h.id for h in recording.handles] == ["r1"]


@pytest.mark.logic
def test_a_transient_slurm_failure_is_retried_and_a_lasting_one_holds_the_run(tmp_path: Path) -> None:
    """A failing `sbatch` is tried three times; a pool over a failing `squeue` raises no error until it is held."""
    (tmp_path / "slurm" / "fail-sbatch").write_text("")
    with pytest.raises(SlurmError, match="Socket timed out"):
        _slurm(tmp_path).start("cpu", 1)
    (tmp_path / "slurm" / "fail-sbatch").unlink()
    (tmp_path / "slurm" / "fail-squeue").write_text("")
    profile = _worker_profile(infrastructure_replacements=1)
    slurm = Slurm(
        profile.resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path / "run", commands=_commands()
    )
    _, pool, _ = _queued(profile, tmp_path, 0, substrate=slurm)
    placed, values = planned(SCORE, profile, ["one"])
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
    with pytest.raises(WorkersBroken) as held:
        list(follow(pool, handle))
    assert held.value.resource_class == "*" and pool.queues.queue("cpu").depth() == 1


@pytest.mark.logic
def test_one_squeue_serves_a_scale_round(tmp_path: Path) -> None:
    """A round that lists and then starts workers makes one `squeue` call."""
    profile = _profile()
    slurm = _slurm(tmp_path, epoch=0)
    _, pool, _ = _queued(profile, tmp_path, 3, substrate=_fleet(slurm, profile))
    log = tmp_path / "slurm" / "squeue.log"
    log.unlink(missing_ok=True)
    scale(pool.queues, _fleet(slurm, profile), profile, workers=_per_quantum(profile))
    assert len(log.read_text().splitlines()) == 1 and slurm.running("cpu") >= 1
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.logic
def test_a_placed_class_needing_more_cpus_than_the_qos_cap_is_refused_at_plan(tmp_path: Path) -> None:
    """A plan placed on a class with more CPUs per worker than the cap names the node, class, CPUs and cap."""
    profile = _profile(qos_cpu_cap=3)
    with pytest.raises(QosCapTooSmall, match=r"'cpu' \(node 'a'\).*4 CPUs.*cap is 3") as raised:
        planned(SCORE, profile, ["one"])
    assert (raised.value.resource_class, raised.value.cpus, raised.value.cap) == ("cpu", 4, 3)


@pytest.mark.logic
def test_unplaced_classes_do_not_count_against_the_qos_cap(tmp_path: Path) -> None:
    """The packaged orcd profile loads under a cap of 2, and a CPU plan under a cap of 4 ignores the 8-CPU classes."""
    assert ExecutionProfile.load("orcd", {"substrate": {"qos_cpu_cap": 2}}).substrate.qos_cpu_cap == 2
    placed, _ = planned(SCORE, _profile(qos_cpu_cap=4), ["one"])
    assert placed.placement.nodes["a"].resource_class == "cpu"


@pytest.mark.logic
def test_pending_jobs_scale_to_max_workers_and_a_failure_switches_the_cap_on(tmp_path: Path) -> None:
    """Pending jobs never use the start budget: a 32-worker class reaches 32; after one failed start it is capped."""
    profile = _profile()
    _, pool, recording = _queued(profile, tmp_path, 40)
    recording.handles.clear()
    fleet = Fleet(recording, workers_dir=None, budget=2, first_s=0.0, max_s=0.0)
    for _ in range(8):
        scale(pool.queues, fleet, profile, workers=_per_quantum(profile))
    assert len(recording.handles) == 32
    capped = Fleet(Recording(), workers_dir=None, budget=2, first_s=0.0, max_s=0.0)
    capped.substrate.live = lambda: []  # type: ignore[method-assign]
    assert len(capped.start("cpu", 2)) == 2
    capped.live()
    assert len(capped.start("cpu", 5)) == 1


class _Failing:
    """An executor whose `close` raises."""

    def close(self) -> None:
        """Raise."""
        raise SlurmError("scancel failed")


class _Handed:
    """A run that records what it is told failed."""

    run_dir = Path("run")

    def __init__(self) -> None:
        """Start with no error."""
        self.errors: list[Exception] = []

    def fail(self, err: Exception) -> None:
        """Record `err`."""
        self.errors.append(err)


@pytest.mark.logic
def test_a_failing_close_is_logged_and_handed_to_the_run_only_when_nothing_failed_before() -> None:
    """The close error never escapes; it fails the run only when the run had not failed already."""
    clean, failed = _Handed(), _Handed()
    _close_owned(_Failing(), True, clean, False)  # type: ignore[arg-type]
    _close_owned(_Failing(), True, failed, True)  # type: ignore[arg-type]
    _close_owned(_Failing(), False, failed, False)  # type: ignore[arg-type]
    assert [str(e) for e in clean.errors] == ["scancel failed"] and failed.errors == []


@pytest.mark.logic
def test_the_profiles_attribute_delay_is_a_floor(tmp_path: Path) -> None:
    """The effective delay is the larger of the profile's and the measured one, and it is the bound's."""
    profile = ExecutionProfile.load("laptop", {"lease": {"attribute_delay_s": 100}})
    ok = StorageCapabilities(True, True, True, True, True, 3.0)
    assert check_storage(profile, tmp_path, probe=lambda root: ok) == 100.0
    worse = StorageCapabilities(True, True, True, True, True, 150.0)
    with pytest.raises(LeaseBoundError):
        check_storage(profile, tmp_path, probe=lambda root: worse)


@pytest.mark.logic
def test_a_local_worker_is_in_the_supervisors_table(tmp_path: Path) -> None:
    """A worker started by `Local` is a live group of the supervisor in force, and not once stopped."""
    local = Local(1, command=[sys.executable, "-c", "import time; time.sleep(60)"])
    with Supervisor(subreaper=False) as supervisor:
        [handle] = local.start("cpu", 1)
        assert supervisor.group_live(int(handle.id))
        local.stop([handle])
        assert not supervisor.group_live(int(handle.id))


@pytest.mark.contract
def test_a_default_slurm_run_cancels_its_workers_when_it_ends(tmp_path: Path) -> None:
    """`Plan.run` over Slurm closes its pool when the run ends (held, the substrate failing): jobs are cancelled."""
    pytest.importorskip("pydra")
    profile = ExecutionProfile.load(
        "orcd",
        {**_override(), "pool": {"first_poll_s": 0.01, "poll_s": 0.2}, "faults": {"infrastructure_replacements": 1}},
    )
    placed, _ = planned(SCORE, profile, ["one", "two"])
    run = placed.run(store=ResultStore(tmp_path / "store"))
    deadline = time.monotonic() + 30
    while not _submitted(tmp_path) and time.monotonic() < deadline:
        time.sleep(0.05)
    (tmp_path / "slurm" / "fail-squeue").write_text("")
    with contextlib.suppress(Exception):
        run.wait(120)
    cancelled = tmp_path / "slurm" / "cancelled"
    assert _submitted(tmp_path) and cancelled.exists() and cancelled.read_text().split()


@pytest.mark.contract
def test_start_refuses_a_result_store_without_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The store a run writes is probed at start, not only the default roots."""
    pytest.importorskip("pydra")
    placed, _ = planned(SCORE, _profile(kind="local"), ["one"])
    refuse_call(monkeypatch, "mkdir", ".senselab-probe", OSError(errno.ENOTSUP, "Operation not supported"))
    with pytest.raises(StorageUnsupported, match="atomic mkdir"):
        placed.run(store=ResultStore(tmp_path / "elsewhere"))


HANGS_AFTER_QUEUEING = """#!{python}
import os, signal, sys
from pathlib import Path

d = Path(os.environ["FAKE_SLURM"])
if (d / "hung").exists():
    os.execv("{queues}", ["{queues}", *sys.argv[1:]])
(d / "hung").touch()
n = int((d / "next").read_text()) if (d / "next").exists() else 100
(d / "next").write_text(str(n + 1))
given = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
(d / "jobs").mkdir(exist_ok=True)
staged = d / f".job-{{n}}"
staged.write_text(f"{{given['job-name']}}|{{given['comment']}}|RUNNING\\n")
staged.rename(d / "jobs" / str(n))
signal.pause()
"""
"""An `sbatch` whose first call queues its job in one rename and then never answers; later calls queue normally."""

FAILS_AFTER_ONE = """#!/bin/sh
[ -f "$FAKE_SLURM/submitted-one" ] && {{ echo "Socket timed out" >&2; exit 1; }}
: > "$FAKE_SLURM/submitted-one"
exec "{queues}" "$@"
"""
"""An `sbatch` that queues one job and fails every later call."""


def _replace_sbatch(tmp_path: Path, template: str) -> None:
    bin_dir = tmp_path / "bin"
    queues = bin_dir / "sbatch-queues"
    (bin_dir / "sbatch").rename(queues)
    (bin_dir / "sbatch").write_text(template.format(python=sys.executable, queues=queues))
    (bin_dir / "sbatch").chmod(stat.S_IRWXU)


def _jobs(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "slurm" / "jobs").iterdir())


@pytest.mark.logic
def test_a_submission_that_times_out_after_queueing_its_job_leaves_one_job(
    tmp_path: Path, slack: dict[str, Any]
) -> None:
    """An `sbatch` that queues its job and then hangs past the timeout is found by name, never submitted again.

    A duplicate the controller registers late under the same name is cancelled when the fleet closes.
    """
    _replace_sbatch(tmp_path, HANGS_AFTER_QUEUEING)
    profile = _profile()
    commands = ExecutionProfile.load("orcd", slack).substrate.commands
    assert commands is not None
    fleet = _fleet(_slurm(tmp_path, commands=_commands(timeout_s=commands.timeout_s)), profile)
    [handle] = fleet.start("cpu", 1)
    assert _jobs(tmp_path) == [handle.id] and "--name=" in (tmp_path / "slurm" / "squeue.log").read_text()
    late = tmp_path / "slurm" / "jobs" / "late"
    late.write_text((tmp_path / "slurm" / "jobs" / handle.id).read_text())
    fleet.close()
    assert _jobs(tmp_path) == [] and sorted((tmp_path / "slurm" / "cancelled").read_text().split()) == [
        handle.id,
        "late",
    ]


@pytest.mark.logic
def test_a_start_aborted_by_a_failed_submission_keeps_the_jobs_it_submitted(tmp_path: Path) -> None:
    """The job submitted before a failing `sbatch` is recorded, so a close with `squeue` down still cancels it."""
    _replace_sbatch(tmp_path, FAILS_AFTER_ONE)
    fleet = _fleet(_slurm(tmp_path), _profile())
    with pytest.raises(SlurmError, match="Socket timed out") as failed:
        fleet.start("cpu", 2)
    [first] = failed.value.started
    (tmp_path / "slurm" / "fail-squeue").write_text("")
    fleet.close()
    assert (tmp_path / "slurm" / "cancelled").read_text().split() == [first.id]


@pytest.mark.logic
def test_a_substrate_command_that_cannot_be_launched_is_a_substrate_failure(tmp_path: Path) -> None:
    """An `sbatch` that cannot be executed is a `SlurmError`; a local command that does not exist is a `LocalError`."""
    (tmp_path / "bin" / "sbatch").chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(SlurmError, match="could not run"):
        _slurm(tmp_path).start("cpu", 1)
    with pytest.raises(LocalError, match="could not launch"):
        Local(1, command=[str(tmp_path / "absent")]).start("cpu", 1)


@pytest.mark.logic
def test_substrate_failures_back_off_and_break_it_after_the_budget() -> None:
    """Each failure of the substrate's commands delays its next use, doubling; past the budget it is broken once."""
    now = [0.0]
    fleet = Fleet(Recording(), workers_dir=None, budget=2, first_s=1.0, max_s=4.0, clock=lambda: now[0])
    due: list[float] = []
    for step in range(40):
        now[0] = step * 0.5
        if fleet.substrate_due():
            due.append(now[0])
            fleet.fail_substrate(SlurmError("squeue timed out"))
    assert due == [0.0, 1.0, 3.0] and fleet.broken["*"].failures == 3
    assert fleet.broken_for("cpu") is fleet.broken["*"]


class _Down(Recording):
    """A substrate whose listing raises `error` the first `failures` times (always when None), counting its calls."""

    def __init__(self, error: Exception, failures: int | None = None) -> None:
        """Raise `error` from the first `failures` calls of `live`."""
        super().__init__()
        self.error, self.failures, self.calls, self.called = error, failures, 0, threading.Event()

    def live(self) -> list[WorkerHandle]:
        """Count the call; raise while failures remain."""
        self.calls += 1
        self.called.set()
        if self.failures is None or self.calls <= self.failures:
            raise self.error
        return super().live()


class _Bug(Exception):
    """An error no substrate raises."""


@pytest.mark.logic
def test_a_lasting_substrate_failure_is_reported_once_and_ends_scaling(tmp_path: Path) -> None:
    """A substrate failing past the budget holds the run, is reported broken once, and is not called again."""
    profile = _worker_profile(infrastructure_replacements=1)
    down = _Down(SlurmError("squeue timed out"))
    _, pool, _ = _queued(profile, tmp_path, 0, substrate=down)
    placed, values = planned(SCORE, profile, ["one", "two"])
    admission = pool.admit(placed, values)
    handle = pool.submit(node_fragment(placed, admission, elements=[(0,)]))
    with pytest.raises(WorkersBroken) as held:
        list(follow(pool, handle))
    pool.submit(node_fragment(placed, admission, elements=[(1,)]))
    time.sleep(2 * profile.pool.poll_s)
    calls = down.calls
    pool.close()
    assert held.value.resource_class == "*" and calls == 2
    assert list(pool.broken_workers) == ["*"] and pool.broken_workers["*"].failures == 2


@pytest.mark.logic
def test_an_unexpected_scaler_error_is_raised_by_poll(tmp_path: Path) -> None:
    """An error that is not a substrate failure ends the scaler and is raised to the caller, never retried."""
    profile = _worker_profile()
    down = _Down(_Bug("no such class"), failures=1)
    _, pool, _ = _queued(profile, tmp_path, 0, substrate=down)
    placed, values = planned(SCORE, profile, ["one"])
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
    assert down.called.wait(_commands().timeout_s)
    pool.close()
    with pytest.raises(ScalerFailed) as failed:
        pool.poll(handle)
    assert isinstance(failed.value.__cause__, _Bug) and down.calls == 2
