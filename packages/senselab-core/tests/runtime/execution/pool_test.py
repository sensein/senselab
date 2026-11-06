"""`Pool`: fragments put on a run's queues, served by a fake substrate's worker thread through `execute`."""

import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy
import pytest
import soundfile
from senselab_core.cli import main as cli_main
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.audio_io import open_audio
from senselab_core.runtime.call import Packaged, RunConfig, TaskCall, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.epoch import Epoch, StaleCoordinator
from senselab_core.runtime.execution.events import EventLog, read_events, status_from
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment, QueueHeld, ResultRef
from senselab_core.runtime.execution.in_process import InProcess, job_key, node_fragment
from senselab_core.runtime.execution.job import Failed, JobOutcome
from senselab_core.runtime.execution.pool import Pool, report
from senselab_core.runtime.execution.queue import RUN_QUEUES, Claim, QueueSet
from senselab_core.runtime.execution.run import INPUTS_FILE, Run, prepare, record_sources
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.interpreter import source_tree_python

SCORE = TaskSpec(
    name="pool_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=TaskParams,
)
HEAR = SCORE.model_copy(update={"name": "pool_test.hear", "inputs": {"audio": "Audio"}})
REGISTRY = Registry()
for _spec in (SCORE, HEAR):
    REGISTRY.add_task(_spec)
    REGISTRY.add_backend(
        BackendSpec(
            task=_spec.name,
            name="dsp",
            version="1",
            stack="dsp",
            env=Env.host(),
            devices=("cpu",),
            batch_invariant=True,
            worker="backends.sizes_backend_fixture",
            load_stub={},
        )
    )
CLASSES = {
    "cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1},
    "cpu:large": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 1},
}
WORKERS = WorkerSet(4)
CONTENT = {"content_attempts": 2, "infrastructure_replacements": 6, "resource_replacements": 1}
GAP = FaultRecord(fault="infrastructure", type="RuntimeError", message="unclassified", gap=True, charged=False)
LOST = FaultRecord(fault="infrastructure", type="WorkerLost", message="gone", gap=False, charged=False)
OOM = FaultRecord(fault="resource", type="ResourceFault", message="out of memory", gap=False, charged=False)


def _profile(**faults: int) -> ExecutionProfile:
    override = {"devices": ["cpu"], "resource_classes": CLASSES, "pool": {"poll_s": 0.02}, "faults": faults}
    return ExecutionProfile.load("laptop", override)


def _host(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="3" * 64,
        wheel_set="4" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    mit = license_record("MIT", source="lock", verified={"date": "2026-11-06", "how": "pep639"}, gated=False)
    previous = use_packaged(
        Packaged(Catalogue.load("format: catalogue@2.0\nentries: []\n"), _host, REGISTRY, lambda _: {"p": mit})
    )
    yield
    use_packaged(previous)
    WORKERS.close()


def _planned(profile: ExecutionProfile, texts: Sequence[str]) -> tuple[Plan, dict[InputIdentity, object]]:
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=SCORE.name, task_version="1"),),
        edges=(Edge(source="input.text", target="a.text", kind="map"),),
        outputs={"value": "a.value"},
    )
    rows = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(texts)]
    placed = plan(graph, RecordSet(rows), config=RunConfig(), execution=profile)
    return placed, {bind_input("x", "json", t)[0]: t for t in texts}


Act = Callable[[PlanFragment], "FaultRecord | ResultStore | None"]


class FakeWorkers:
    """A `Substrate` whose thread claims each queue it serves, logs each job `claimed` and runs it through `execute`.

    `act` decides per fragment: a `FaultRecord` fails every job with it, a `ResultStore` executes into that store,
    None executes into `store`. `served` holds each fragment claimed, in order.
    """

    def __init__(self, run_dir: Path, store: ResultStore, placed: Plan, values: Mapping[InputIdentity, object]) -> None:
        """Serve the queues of `run_dir` for `placed`, its graph inputs `values`."""
        self.queues = QueueSet(run_dir / RUN_QUEUES, placed.execution)
        self.run_dir = run_dir
        self.store, self.placed, self.values = store, placed, values
        self.act: Act = lambda fragment: None
        self.served: list[PlanFragment] = []
        self.names: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def ensure(self, queue: str) -> None:
        """Serve `queue` from the worker thread, started on the first call."""
        self.names.add(queue)
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(30)

    def _loop(self) -> None:
        while not self._stop.is_set():
            claims = [c for name in sorted(self.names) if (c := self.queues.queue(name).claim("fake")) is not None]
            for claim in claims:
                self._serve(claim)
            if not claims:
                time.sleep(0.01)

    def _log(self) -> EventLog:
        """Return a log of the run's latest claimed attempt, `a1` when none was claimed."""
        claimed = sorted((int(p.name), p.read_text()) for p in self.run_dir.glob("attempts/*"))
        ordinal, attempt = claimed[-1] if claimed else (1, "a1")
        return EventLog(self.run_dir, attempt, ordinal)

    def _serve(self, claim: Claim) -> None:
        log = self._log()
        for digest, fragment in zip(claim.items, claim.fragments, strict=True):
            for job in fragment.jobs:
                log.record("claimed", job.node, job=job)
            self.served.append(fragment)
            report(claim, digest, self._outcomes(fragment))

    def _outcomes(self, fragment: PlanFragment) -> list[JobOutcome]:
        act = self.act(fragment)
        if isinstance(act, FaultRecord):
            keys = [job_key(self.placed, fragment.node, inputs).digest() for inputs in fragment.inputs]
            return [Failed(job, key, act) for job, key in zip(fragment.jobs, keys, strict=True)]
        executor = InProcess(store=act or self.store, profile=self.placed.execution, run=RunConfig(), workers=WORKERS)
        admission = executor.admit(self.placed, self.values)
        handle = executor.submit(fragment.model_copy(update={"admission": admission.id}))
        return executor.outcomes(handle)


def _stack(
    tmp_path: Path, profile: ExecutionProfile, texts: Sequence[str]
) -> tuple[Plan, dict[InputIdentity, object], ResultStore, FakeWorkers]:
    placed, values = _planned(profile, texts)
    store = ResultStore(tmp_path / "store")
    return placed, values, store, FakeWorkers(tmp_path / "run", store, placed, values)


def _pool(tmp_path: Path, store: ResultStore, workers: FakeWorkers | Any, profile: ExecutionProfile) -> Pool:  # noqa: ANN401
    run_dir = tmp_path / "run"
    return Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile),
        store=store,
        events=EventLog(run_dir, "a1", 1),
        profile=profile,
        substrate=workers,
        epoch=Epoch(run_dir).take(),
    )


def _all(pool: Pool, placed: Plan, values: Mapping[InputIdentity, object]) -> list[ResultRef]:
    fragment = node_fragment(placed, pool.admit(placed, values))
    return list(pool.results(pool.submit(fragment)))


def _kinds(run_dir: Path) -> list[str]:
    return [e.kind for e in read_events(run_dir) if e.kind not in ("claimed",)]


@pytest.mark.contract
def test_two_coordinators_one_result_per_key(tmp_path: Path) -> None:
    """Two coordinators put one plan: one item per fragment, executed once, both read one stored result per key."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two"])
    first = _pool(tmp_path, store, workers, profile)
    fragment = node_fragment(placed, first.admit(placed, values))
    early = first.submit(fragment)
    second = _pool(tmp_path, store, workers, profile)
    late = second.submit(node_fragment(placed, second.admit(placed, values)))
    try:
        both = [list(second.results(late)), list(first.results(early))]
    finally:
        workers.stop()
    assert both[0] == both[1] and all(r.failure is None for r in both[0]) and len(workers.served) == 1
    for ref in both[0]:
        entries = list((store.root).rglob(f"{ref.key_digest}/*"))
        assert len([e for e in entries if e.parent.name == ref.key_digest]) == 1


@pytest.mark.logic
def test_stale_coordinator_stops_after_new_epoch(tmp_path: Path) -> None:
    """Once a newer epoch is taken, the older coordinator's submit raises `StaleCoordinator` and puts nothing."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    stale = _pool(tmp_path, store, workers, profile)
    assert Epoch(tmp_path / "run").take() == stale.epoch + 1
    with pytest.raises(StaleCoordinator) as raised:
        stale.submit(node_fragment(placed, stale.admit(placed, values)))
    assert (raised.value.mine, raised.value.current) == (1, 2)
    assert stale.queues.queue("cpu").depth() == 0 and not workers.names


@pytest.mark.contract
def test_content_fault_dead_after_budget(tmp_path: Path) -> None:
    """A content fault is charged: retried once, then dead at `content_attempts` 2, its failure read from `dead/`."""
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["fail"])
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.failure is not None and ref.failure.fault == "content" and ref.failure.charged
    assert [f.attempt for f in workers.served] == [1, 2]
    dead = json.loads((tmp_path / "run" / "dead" / f"{ref.key_digest}.json").read_text())
    assert FaultRecord.model_validate(dead["failure"]) == ref.failure
    assert _kinds(tmp_path / "run") == ["failed", "queued"]


@pytest.mark.contract
def test_gap_fault_never_dead(tmp_path: Path) -> None:
    """A gap fault is never charged: with one content attempt, two gaps are retried and the third attempt completes."""
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: GAP if fragment.attempt < 3 else None
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.failure is None and store.outputs_of(ref.key_digest, str(ref.output_identity)) is not None
    assert [g.type for _, g in pool.gaps] == ["RuntimeError", "RuntimeError"] and not pool.broken
    assert not any((tmp_path / "run").glob("dead/*"))
    assert _kinds(tmp_path / "run") == ["failed", "queued", "failed", "queued"]


@pytest.mark.contract
def test_resource_fault_moves_to_larger_class_uncharged(tmp_path: Path) -> None:
    """A resource fault moves the job to `cpu:large` uncharged; with no move left it is a resource failure."""
    profile = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=1)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    workers.act = lambda fragment: OOM if fragment.placement.resource_class == "cpu" else None
    try:
        [ref] = _all(_pool(tmp_path, store, workers, profile), placed, values)
    finally:
        workers.stop()
    assert ref.failure is None
    assert [(f.placement.resource_class, f.attempt) for f in workers.served] == [("cpu", 1), ("cpu:large", 2)]

    stuck = _profile(content_attempts=1, infrastructure_replacements=6, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path / "again", stuck, ["one"])
    workers.act = lambda fragment: OOM
    try:
        [failed] = _all(_pool(tmp_path / "again", store, workers, stuck), placed, values)
    finally:
        workers.stop()
    assert failed.failure == OOM and len(workers.served) == 1
    assert (tmp_path / "again" / "run" / "dead" / f"{failed.key_digest}.json").is_file()


@pytest.mark.contract
def test_broken_env_reported_once(tmp_path: Path) -> None:
    """Two infrastructure faults of one env break it: reported once, its queue held, no job failed or dead."""
    profile = _profile(content_attempts=2, infrastructure_replacements=2, resource_replacements=0)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "two", "three"])
    workers.act = lambda fragment: LOST
    pool = _pool(tmp_path, store, workers, profile)
    admission = pool.admit(placed, values)
    handles = [pool.submit(node_fragment(placed, admission, elements=[(i,)])) for i in range(3)]
    try:
        for handle in handles:
            with pytest.raises(QueueHeld) as raised:
                list(pool.results(handle))
            assert (raised.value.queue, raised.value.failures) == ("cpu", 2)
    finally:
        workers.stop()
    [broken] = pool.broken.values()
    assert broken.env == "host" and broken.failures == 2 and broken.fault == LOST
    events = read_events(tmp_path / "run")
    assert [e.kind for e in events if e.kind in ("failed", "queued")] == ["failed", "queued", "failed"]
    assert [b.env for b in status_from(events, broken_after=2).broken] == ["host"]
    queue = pool.queues.queue("cpu")
    assert queue.held() == pool.epoch and queue.claim("w") is None
    assert not any((tmp_path / "run").glob("dead/*"))
    with pytest.raises(QueueHeld):
        pool.submit(node_fragment(placed, admission, elements=[(0,)]))


class _Idle:
    """A `Substrate` that starts no worker."""

    def ensure(self, queue: str) -> None:
        """Start nothing."""

    def stop(self) -> None:
        """Stop nothing."""


@pytest.mark.contract
def test_results_read_from_store_only(tmp_path: Path) -> None:
    """A result whose outputs the store lacks is a gap and retried; a restarted coordinator reads it from disk alone.

    The first attempt's worker stores into another store; once the retry stored the outputs, a restarted coordinator
    with no worker reads the first attempt's outcome and finds them in the store.
    """
    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one"])
    elsewhere = ResultStore(tmp_path / "elsewhere")
    workers.act = lambda fragment: elsewhere if fragment.attempt == 1 else None
    pool = _pool(tmp_path, store, workers, profile)
    try:
        [ref] = _all(pool, placed, values)
    finally:
        workers.stop()
    assert ref.output_identity is not None and store.outputs_of(ref.key_digest, ref.output_identity) is not None
    assert [f.attempt for f in workers.served] == [1, 2] and [g.type for _, g in pool.gaps] == ["MissingOutput"]
    restarted = _pool(tmp_path, store, _Idle(), profile)
    assert restarted.epoch == 2 and _all(restarted, placed, values) == [ref]
    assert not restarted.gaps and [f.attempt for f in workers.served] == [1, 2]


def _wav(path: Path) -> Path:
    soundfile.write(path, numpy.zeros(1600, dtype="float32"), 16000)
    return path


def _heard(source: object, profile: ExecutionProfile) -> Plan:
    call = TaskCall.model_validate({"task": HEAR.name, "inputs": {"audio": source}}, context={"registry": REGISTRY})
    return plan(call, config=RunConfig(), execution=profile)


@pytest.mark.contract
def test_two_starters_keep_every_durable_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A starter holding audio in memory, its write delayed, and one giving the path merge one after the other.

    The memory starter takes the lease first and stalls before its rename; the path starter waits for the lease, reads
    the memory marker and keeps its path, so the durable source survives whichever finishes last.
    """
    profile = _profile(**CONTENT)
    path = _wav(tmp_path / "a.wav")
    memory, durable = _heard(open_audio(path), profile), _heard(path, profile)
    run_dir = tmp_path / "run"
    real, entered = os.replace, threading.Event()

    def delayed(src: Any, dst: Any, **kw: Any) -> None:  # noqa: ANN401
        if threading.current_thread().name == "memory" and os.fspath(dst).endswith(INPUTS_FILE):
            entered.set()
            time.sleep(1.0)
        real(src, dst, **kw)

    monkeypatch.setattr(os, "replace", delayed)
    first = threading.Thread(target=record_sources, args=(memory, run_dir), name="memory")
    first.start()
    assert entered.wait(30)
    record_sources(durable, run_dir)
    first.join(30)
    assert json.loads((run_dir / INPUTS_FILE).read_text())["rows"][0]["audio"] == str(path)


CRASH = """
import os, sys
from pathlib import Path
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution_settings import JobLease
epochs = Epoch(Path(sys.argv[1]))
with epochs.held(epochs.take(), JobLease(stale_after_s=1.0, renew_fraction=0.2, attribute_delay_s=0)):
    os._exit(9)
"""


@pytest.mark.contract
def test_wait_on_a_run_whose_coordinator_died_ends_interrupted(tmp_path: Path) -> None:
    """A coordinator killed while holding its epoch: `wait(None)` returns once its lease is stale, interrupted."""
    run_dir = tmp_path / "run"
    log = EventLog(run_dir, "a1", 1)
    job = node_fragment(*_admitted(tmp_path)).jobs[0]
    log.record("queued", job.node, job=job)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    crashed = subprocess.run([sys.executable, "-c", CRASH, str(run_dir)], env=env, check=False)
    assert crashed.returncode == 9 and not Epoch(run_dir).stopped()
    started = time.monotonic()
    status = Run(run_dir).wait(None, poll=0.1)
    assert status.interrupted and not status.done and status.counts["queued"] == 1
    assert time.monotonic() - started < 30


def _admitted(tmp_path: Path) -> tuple[Plan, Any]:
    profile = _profile(**CONTENT)
    placed, values = _planned(profile, ["one"])
    pool = _pool(tmp_path / "admitting", ResultStore(tmp_path / "store"), _Idle(), profile)
    return placed, pool.admit(placed, values)


@pytest.mark.contract
def test_coordinate_runs_a_plan_on_the_pool_as_the_next_epoch(tmp_path: Path) -> None:
    """`coordinate` takes epoch 1, runs every job through the pool, and ends its epoch; status reads claims, not starts.

    The `fail` record is dead after its two content attempts and its descendants are none; the others are done.
    """
    pytest.importorskip("pydra")
    from senselab_core.runtime.execution.pool import coordinate  # noqa: PLC0415 - after the pydra check

    profile = _profile(**CONTENT)
    placed, values, store, workers = _stack(tmp_path, profile, ["one", "fail", "two"])
    run_dir = tmp_path / "run"
    prepare(placed, run_dir, store)
    try:
        outcome = coordinate(run_dir, substrate=workers, profile=profile, store=store)
    finally:
        workers.stop()
    failed = [r.job.element for r in outcome.results["a"] if r.failure is not None]
    assert failed == [(1,)] and len(outcome.results["a"]) == 3
    events = read_events(run_dir)
    assert "started" not in {e.kind for e in events} and {e.ordinal for e in events} == {2}
    status = Run(run_dir).wait(None)
    assert status.counts["done"] == 2 and status.counts["failed"] == 1 and status.done and not status.interrupted
    assert Epoch(run_dir).current() == 1 and Epoch(run_dir).stopped()
    assert cli_main(["coordinate", str(tmp_path / "absent")]) == 1
