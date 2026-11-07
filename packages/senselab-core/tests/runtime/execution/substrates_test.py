"""Substrates: `scale` within the caps, the `sbatch` line of a resource class, excluded nodes, queue names, defaults."""

import contextlib
import errno
import os
import stat
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog, read_events
from senselab_core.runtime.execution.executor import WorkersBroken
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.pool_worker import serve_pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import Run, default_executor, prepare
from senselab_core.runtime.execution.substrates import (
    EXIT_UNREACHABLE,
    ExternalWorkers,
    Fleet,
    Local,
    Slurm,
    SlurmError,
    WorkerHandle,
    class_of,
    profile_argument,
    scale,
)
from senselab_core.runtime.execution.supervisor import Supervisor
from senselab_core.runtime.execution_settings import LeaseBoundError
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile, QosCapTooSmall
from senselab_core.runtime.storage import StorageCapabilities, StorageUnsupported
from senselab_core.runtime.store import ResultStore
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
        "substrate": {"kind": kind, **substrate},
        "pool": {"quantum_items": 1, "first_poll_s": 0.01, "poll_s": 0.2},
    }
    return ExecutionProfile.load("orcd", override)


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


def _queued(
    profile: ExecutionProfile,
    tmp_path: Path,
    items: int,
    model: bool = False,
    substrate: Any = None,  # noqa: ANN401
) -> tuple[Plan, Pool, Recording]:
    """Return a plan of `items` jobs, one fragment per job on a pool over `substrate` (a recording one by default)."""
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(items)])
    if model:
        models = {n: MODEL for n in placed.logical.models}
        placed = replace(placed, logical=placed.logical.model_copy(update={"models": models}))
    run_dir = tmp_path / "run"
    recording = substrate if substrate is not None else Recording()
    epoch = Epoch(run_dir).take()
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, "a1", epoch, epoch=epoch),
        profile=profile,
        substrate=recording,
        epoch=epoch,
    )
    admission = pool.admit(placed, values)
    for element in range(items):
        pool.submit(node_fragment(placed, admission, elements=[(element,)]))
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
    scale(pool.queues, recording, profile)
    assert len(recording.handles) == expected
    scale(pool.queues, recording, profile)
    assert len(recording.handles) == expected


@pytest.mark.logic
def test_scale_counts_workers_alive_and_a_quantum_of_items(tmp_path: Path) -> None:
    """Five items over a quantum of 2 want three workers; one alive, two are started; none once three are alive."""
    profile = ExecutionProfile.load("orcd", {**_override(), "pool": {"quantum_items": 2, "first_poll_s": 0.01}})
    _, pool, recording = _queued(profile, tmp_path, 5)
    recording.starts.clear()
    recording.handles = [WorkerHandle("cpu", "x")]
    scale(pool.queues, recording, profile)
    assert recording.starts == [("cpu", 2)]
    scale(pool.queues, recording, profile)
    assert recording.starts == [("cpu", 2)]


def _override() -> dict[str, Any]:
    return {"devices": ["cpu"], "resource_classes": CLASSES, "substrate": {"kind": "slurm"}}


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
    )
    handles = slurm.start("gpu:h100#" + "c" * 40, 2)
    assert [h.id for h in handles] == ["100", "101"]
    [first, _] = _submitted(tmp_path)
    assert first[:4] == [
        "--parsable",
        f"--job-name={slurm.job_name}",
        f"--comment=gpu:h100#{'c' * 40}",
        "--partition=gpuq",
    ]
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
    slurm = Slurm(_profile().resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path)
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
    slurm = Slurm(_profile().resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path)
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
    )
    slurm.start("cpu", 1)
    slurm.start("gpu:h100", 1)
    calls = _submitted(tmp_path)
    assert len(calls) == 2 and all("--exclude=node-a,node-b" in c for c in calls)
    assert not any(a.startswith(("--nodelist", "-w")) for c in calls for a in c)


@pytest.mark.logic
def test_slurm_stays_within_its_qos_cap(tmp_path: Path) -> None:
    """With a cap of 10 CPUs and 4 per worker, a request for 5 submits 2, and none while they are alive."""
    slurm = Slurm(_profile().resource_classes, 10, command=["x"], plan_hash=HASH, run_dir=tmp_path)
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
    _, pool, recording = _queued(profile, tmp_path, 3, model=True)
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
    slurm = Slurm(profile.resource_classes, None, command=["x"], plan_hash=HASH, run_dir=run_dir)
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile),
        store=ResultStore(tmp_path / "store"),
        events=EventLog(run_dir, "a0", 1, epoch=0),
        profile=profile,
        substrate=slurm,
        epoch=0,
    )
    pool.bind_epoch(3, EventLog(run_dir, "attempt-3", 3, epoch=3))
    assert slurm.job_name.startswith(slurm.prefix) and slurm.job_name.endswith("-attempt-3-e3")


@pytest.mark.logic
def test_a_custom_profile_reaches_its_workers_as_a_file(tmp_path: Path) -> None:
    """A packaged profile is named; a changed one is written whole beside the run, and loads to the same digest."""
    assert profile_argument(ExecutionProfile.load("orcd"), tmp_path) == "orcd"
    custom = _profile(qos_cpu_cap=12, exclude_nodes=["n1"])
    written = profile_argument(custom, tmp_path)
    loaded = ExecutionProfile.model_validate_json(Path(written).read_text())
    assert loaded.canonical_digest() == custom.canonical_digest()


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
    fleet = Fleet(local)
    assert len(fleet.start("cpu", 1)) == 1 and local.running("cpu") == 1
    fleet.close()
    assert local.running("cpu") == 0


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
        scale(pool.queues, fleet, profile)
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
        list(pool.results(handle))
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


def _slurm(tmp_path: Path, epoch: int = 1, plan_hash: str = HASH, **kw: Any) -> Slurm:  # noqa: ANN401
    return Slurm(
        _profile().resource_classes,
        None,
        command=["x"],
        plan_hash=plan_hash,
        run_dir=tmp_path / "run",
        attempt="a1",
        epoch=epoch,
        retry_s=0.01,
        **kw,
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
    scale(pool.queues, recording, profile, surplus_since=since, now=0.0)
    assert recording.stops == [] and since == {"cpu": 0.0}
    scale(pool.queues, recording, profile, surplus_since=since, now=10.0)
    assert {h.id for h in recording.stops} == {"p1", "p2"} and [h.id for h in recording.handles] == ["r1"]
    empty = QueueSet(tmp_path / "empty" / RUN_QUEUES, profile)
    empty.queue("cpu")
    recording.handles = [WorkerHandle("cpu", "r1"), WorkerHandle("cpu", "p3", True)]
    recording.stops.clear()
    scale(empty, recording, profile)
    assert [h.id for h in recording.stops] == ["p3", "r1"] and recording.handles == []


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
        profile.resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path / "run", retry_s=0.001
    )
    _, pool, _ = _queued(profile, tmp_path, 0, substrate=slurm)
    placed, values = planned(SCORE, profile, ["one"])
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
    with pytest.raises(WorkersBroken) as held:
        list(pool.results(handle))
    assert held.value.resource_class == "*" and pool.queues.queue("cpu").depth() == 1


@pytest.mark.logic
def test_one_squeue_serves_a_scale_round(tmp_path: Path) -> None:
    """A round that lists and then starts workers makes one `squeue` call."""
    profile = _profile()
    slurm = Slurm(profile.resource_classes, None, command=["x"], plan_hash=HASH, run_dir=tmp_path / "run")
    _, pool, _ = _queued(profile, tmp_path, 3, substrate=Fleet(slurm))
    log = tmp_path / "slurm" / "squeue.log"
    log.unlink(missing_ok=True)
    scale(pool.queues, Fleet(slurm), profile)
    assert len(log.read_text().splitlines()) == 1 and slurm.running("cpu") >= 1
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.logic
def test_a_class_needing_more_cpus_than_the_qos_cap_is_refused() -> None:
    """A profile whose class has more CPUs per worker than the cap names the class, its CPUs and the cap."""
    with pytest.raises(QosCapTooSmall, match=r"'cpu'.*4 CPUs.*cap is 3") as raised:
        _profile(qos_cpu_cap=3)
    assert (raised.value.resource_class, raised.value.cpus, raised.value.cap) == ("cpu", 4, 3)


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
