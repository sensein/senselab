"""What a request reports of its model state: requests answered before it and the load it carried."""

from pathlib import Path

import numpy
import pytest
from pool_stack_fixture import host_env, registry, scoring
from senselab_core.runtime import request_reports
from senselab_core.runtime.execution.workers import host_model_key
from senselab_core.runtime.host_worker import HostWorker
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.model_cache import ModelCache, ModelKey

COUNTING = "backends.counting_backend_fixture"
FAKE = "senselab_worker.fake"
FIRST: ModelKey = ("fake", "c" * 40, "cpu", "float32")
SECOND: ModelKey = ("fake", "d" * 40, "cpu", "float32")
ONE = [{"id": "a", "inputs": {}}]


def _size(_: object) -> int:
    return 1


def _asks(worker: HostWorker, times: int = 1) -> list[tuple[int | None, bool]]:
    """Run `times` one-item requests; return each one's `since_load` and whether it carried a load."""
    with request_reports.collected() as found:
        for _ in range(times):
            worker.run(ONE, {})
    return [(r.since_load, r.load_s is not None) for r in found]


@pytest.mark.logic
def test_a_reloaded_model_reports_a_load() -> None:
    """A budget that holds one model: a second model evicts the first, and the first's next request reports a load."""
    cache = ModelCache(1)
    first = HostWorker(COUNTING, init={}, key=FIRST, cache=cache, size_of=_size)
    assert _asks(first, 3) == [(0, True), (1, False), (2, False)]
    second = HostWorker(COUNTING, init={}, key=SECOND, cache=cache, size_of=_size)
    assert _asks(second) == [(0, True)]
    assert _asks(first, 2) == [(0, True), (1, False)]


@pytest.mark.logic
def test_a_second_backend_on_one_queue_reports_its_load() -> None:
    """Two backends in one process cache each count their own requests, so the second reports its own first one."""
    cache = ModelCache(10)
    first = HostWorker(COUNTING, init={}, key=FIRST, cache=cache, size_of=_size)
    assert _asks(first, 2) == [(0, True), (1, False)]
    other = HostWorker(COUNTING, init={}, key=("other", "c" * 40, "cpu", "float32"), cache=cache, size_of=_size)
    assert _asks(other) == [(0, True)] and _asks(first) == [(2, False)]


@pytest.mark.logic
def test_a_second_model_less_unit_on_cpu_reports_a_load() -> None:
    """Two model-less units on cpu have keys of their own, so the second one's first request reports a load."""
    task, profile = scoring("host_reports_test.score"), ExecutionProfile.load("laptop")
    specs = [registry(task, name=name).backends(task.name)[0] for name in ("dsp", "other")]
    keys = [host_model_key(spec, host_env(spec, profile), None, "cpu") for spec in specs]
    assert keys[0] != keys[1]
    cache = ModelCache(10)
    workers = [HostWorker(COUNTING, init={}, key=k, cache=cache, size_of=_size) for k in keys]
    assert [_asks(w, 2) for w in workers] == [[(0, True), (1, False)]] * 2


@pytest.mark.logic
def test_a_worker_process_reports_each_request_it_answered_and_its_load_on_the_first(tmp_path: Path) -> None:
    """A worker process counts the requests its state answered, and reports the load's seconds on the first only."""
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    with WorkerClient(source_tree_python(), FAKE, init={}, array_parent=arrays) as client:
        with request_reports.collected() as found:
            for _ in range(3):
                client.run([{"id": "a", "inputs": {"payload": 1, "signal": numpy.arange(4.0)}}], {})
    assert [(r.since_load, r.load_s is not None) for r in found] == [(0, True), (1, False), (2, False)]
