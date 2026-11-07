"""Substrates: `scale` within the caps, the `sbatch` line of a resource class, excluded nodes, queue names, defaults."""

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
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import Run, default_executor, prepare
from senselab_core.runtime.execution.substrates import (
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
from senselab_core.runtime.planner import Plan, check_storage
from senselab_core.runtime.profiles import ExecutionProfile
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
n=$(cat "$d/next" 2>/dev/null || echo 100)
echo $((n + 1)) > "$d/next"
for a in "$@"; do printf '%s\\n' "$a" >> "$d/sbatch.log"; done
echo "--" >> "$d/sbatch.log"
comment=""
for a in "$@"; do case "$a" in --comment=*) comment="${a#--comment=}";; esac; done
mkdir -p "$d/jobs"
echo "$comment" > "$d/jobs/$n"
echo "$n"
"""
SQUEUE = """#!/bin/sh
d="$FAKE_SLURM"
for j in "$d"/jobs/*; do [ -f "$j" ] && echo "$(basename "$j")|$(cat "$j")"; done
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
    """A `Substrate` that records each start and counts its workers per queue."""

    def __init__(self) -> None:
        """Start with no worker."""
        self.alive: dict[str, int] = {}
        self.starts: list[tuple[str, int]] = []

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        """Record `count` workers of the queue."""
        self.starts.append((resource_class, count))
        self.alive[resource_class] = self.alive.get(resource_class, 0) + count
        return [WorkerHandle(resource_class, str(i)) for i in range(count)]

    def running(self, resource_class: str) -> int:
        """Return the workers recorded for the queue."""
        return self.alive.get(resource_class, 0)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        """Forget `handles`."""
        for h in handles:
            self.alive[h.queue] -= 1


def _queued(profile: ExecutionProfile, tmp_path: Path, items: int, model: bool = False) -> tuple[Plan, Pool, Recording]:
    """Return a plan of `items` jobs, one fragment per job on a pool over a recording substrate, and that substrate."""
    placed, values = planned(SCORE, profile, [f"t{i}" for i in range(items)])
    if model:
        models = {n: MODEL for n in placed.logical.models}
        placed = replace(placed, logical=placed.logical.model_copy(update={"models": models}))
    run_dir, recording = tmp_path / "run", Recording()
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
    [(10, 20, 2), (None, 20, 20), (None, 40, 32), (100, 20, 20), (3, 20, 0)],
    ids=["qos-cap", "depth", "class-max", "cap-above-depth", "cap-below-one-worker"],
)
@pytest.mark.logic
def test_scale_bounded_by_qos_cap(tmp_path: Path, cap: int | None, depth: int, expected: int) -> None:
    """Workers started are the depth over the quantum, within the class's `max_workers` and cap ÷ CPUs per worker."""
    profile = _profile(qos_cpu_cap=cap)
    _, pool, recording = _queued(profile, tmp_path, depth)
    recording.starts.clear()
    recording.alive.clear()
    scale(pool.queues, recording, profile)
    assert sum(recording.alive.values()) == expected
    scale(pool.queues, recording, profile)
    assert sum(recording.alive.values()) == expected


@pytest.mark.logic
def test_scale_counts_workers_alive_and_a_quantum_of_items(tmp_path: Path) -> None:
    """Five items over a quantum of 2 want three workers; one alive, two are started; none once three are alive."""
    profile = ExecutionProfile.load("orcd", {**_override(), "pool": {"quantum_items": 2, "first_poll_s": 0.01}})
    _, pool, recording = _queued(profile, tmp_path, 5)
    recording.starts.clear()
    recording.alive = {"cpu": 1}
    scale(pool.queues, recording, profile)
    assert recording.starts == [("cpu", 2)]
    scale(pool.queues, recording, profile)
    assert recording.starts == [("cpu", 2)]


def _override() -> dict[str, Any]:
    return {"devices": ["cpu"], "resource_classes": CLASSES, "substrate": {"kind": "slurm"}}


@pytest.mark.logic
def test_slurm_args_from_resource_class(tmp_path: Path) -> None:
    """One `sbatch` per worker: the class's partition, gres, CPUs, memory and time, named by plan hash and attempt."""
    slurm = Slurm(_profile().resource_classes, None, command=["senselab", "execute"], plan_hash=HASH, attempt="a7")
    handles = slurm.start("gpu:h100#" + "c" * 40, 2)
    assert [h.id for h in handles] == ["100", "101"]
    [first, _] = _submitted(tmp_path)
    assert first[:4] == [
        "--parsable",
        f"--job-name=senselab-{HASH[:12]}-a7",
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
    slurm = Slurm(_profile().resource_classes, None, command=["x"], plan_hash=HASH)
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
    slurm = Slurm(_profile().resource_classes, None, command=["x"], plan_hash=HASH)
    with pytest.raises(SlurmError, match="invalid partition"):
        slurm.start("cpu", 1)


@pytest.mark.logic
def test_excluded_node_never_targeted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every submission excludes the profile's nodes, `${VAR}` expanded, and names no node to run on."""
    monkeypatch.setenv("BAD_NODE", "node-b")
    profile = _profile(exclude_nodes=["node-a", "${BAD_NODE}"])
    slurm = Slurm(
        profile.resource_classes, None, command=["x"], plan_hash=HASH, exclude_nodes=profile.substrate.exclude_nodes
    )
    slurm.start("cpu", 1)
    slurm.start("gpu:h100", 1)
    calls = _submitted(tmp_path)
    assert len(calls) == 2 and all("--exclude=node-a,node-b" in c for c in calls)
    assert not any(a.startswith(("--nodelist", "-w")) for c in calls for a in c)


@pytest.mark.logic
def test_slurm_stays_within_its_qos_cap(tmp_path: Path) -> None:
    """With a cap of 10 CPUs and 4 per worker, a request for 5 submits 2, and none while they are alive."""
    slurm = Slurm(_profile().resource_classes, 10, command=["x"], plan_hash=HASH)
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
    lacking = StorageCapabilities(True, True, True, False, 0.0)
    with pytest.raises(StorageUnsupported, match="atomic_rename"):
        check_storage(profile, tmp_path, probe=lambda root: lacking)
    slow = StorageCapabilities(True, True, True, True, profile.lease.stale_after_s)
    with pytest.raises(ValueError, match="attribute_delay_s"):
        check_storage(profile, tmp_path, probe=lambda root: slow)
    check_storage(profile, tmp_path, probe=lambda root: StorageCapabilities(True, True, True, True, 1.0))


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
    slurm = Slurm(profile.resource_classes, None, command=["x"], plan_hash=HASH)
    run_dir = tmp_path / "run"
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
    assert slurm.job_name == f"senselab-{HASH[:12]}-attempt-3"


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
from senselab_core.runtime.profiles import ExecutionProfile
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
