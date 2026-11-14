"""`WorkerSet`: a bounded pool that closes idle workers first, and server starts leased per env, backend and model."""

import dataclasses
import platform
import threading
from typing import Any

import backends.release_backend_fixture as release_fixture
import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import Attestation, ModelRef, ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.workers import PRECISION, WorkerLoadFailed, WorkerSet, host_model_key
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import VllmOutOfMemory, VllmStartError
from senselab_core.runtime.worker_client import WorkerClient, WorkerLost
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import FaultClass, classify

ENV = EnvHandle(source_tree_python(), platform.python_version(), "1" * 64, "2" * 64, None)
PROFILE = ExecutionProfile.load()


def _host(name: str) -> BackendSpec:
    return BackendSpec(
        task=fake.ECHO.name,
        name=name,
        version="1",
        stack=name,
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker="backends.sizes_backend_fixture",
        load_stub={},
    )


def _model(identity: str) -> ResolvedModel:
    return ResolvedModel(ref=ModelRef.parse(f"org/m@{identity}"), identity=identity)


@pytest.mark.logic
def test_worker_set_bounded_and_evicts_idle() -> None:
    """Beyond the bound the least recently used idle worker is dropped; a held worker never is."""
    workers = WorkerSet(1)
    a, b = _host("a"), _host("b")
    with workers.checkout(a, ENV, None, device="cpu", profile=PROFILE) as held_a:
        with workers.checkout(b, ENV, None, device="cpu", profile=PROFILE) as held_b:
            assert workers.pooled() == [held_a, held_b]
        assert workers.pooled() == [held_a]
    with workers.checkout(b, ENV, None, device="cpu", profile=PROFILE) as again_b:
        assert again_b is not held_b
    assert workers.pooled() == [again_b]
    with pytest.raises(ValueError, match="at least one"):
        WorkerSet(0)


@pytest.mark.logic
def test_least_recently_used_idle_worker_goes_first_and_a_tighter_bound_evicts() -> None:
    """Under a bound of two the least recently used idle worker is evicted; tightening evicts down to the new bound."""
    workers = WorkerSet(2)
    a, b, c = _host("a"), _host("b"), _host("c")
    held = {}
    for backend in (a, b, a, c):
        with workers.checkout(backend, ENV, None, device="cpu", profile=PROFILE) as worker:
            held[backend.name] = worker
    assert workers.pooled() == [held["a"], held["c"]]
    workers.tighten(3)
    assert workers.bound == 2
    workers.tighten(1)
    assert workers.bound == 1 and workers.pooled() == [held["c"]]


@pytest.mark.contract
def test_evicted_worker_process_is_closed() -> None:
    """An isolated worker evicted beyond the bound is closed; the one kept is still alive."""
    workers = WorkerSet(1)
    isolated = fake.FAKE.model_copy(update={"name": "fake-isolated", "env": Env.isolated("fake")})
    first, second = (_model(c * 40) for c in "ab")
    try:
        with workers.checkout(isolated, ENV, first, device="cpu", profile=PROFILE) as old:
            assert isinstance(old, WorkerClient) and old.alive
        with workers.checkout(isolated, ENV, second, device="cpu", profile=PROFILE) as new:
            assert isinstance(new, WorkerClient)
        assert not old.alive and new.alive and workers.pooled() == [new]
    finally:
        workers.close()


@pytest.mark.contract
def test_a_client_that_failed_a_request_is_replaced_by_the_next_checkout() -> None:
    """A worker lost on a request is not alive, and the next checkout hands out another process, as a retry needs."""
    workers = WorkerSet(1)
    isolated = fake.FAKE.model_copy(update={"name": "fake-isolated", "env": Env.isolated("fake")})
    model = _model("c" * 40)
    items = [{"id": "a", "inputs": {}}]
    try:
        with workers.checkout(isolated, ENV, model, device="cpu", profile=PROFILE) as first:
            assert isinstance(first, WorkerClient)
            with pytest.raises(WorkerLost):
                first.run(items, fake.fault_param(FakeFaults(crash_after=0)))
            assert not first.alive
        with workers.checkout(isolated, ENV, model, device="cpu", profile=PROFILE) as second:
            assert isinstance(second, WorkerClient) and second is not first and second.alive
            assert second.hello["pid"] != first.hello["pid"]
            assert second.run(items, {})[0]["id"] == "a"
    finally:
        workers.close()


class _Server:
    """A started server that attests what it was given."""

    url: str | None = None

    def __init__(self, attestation: Attestation) -> None:
        self.label = attestation.model_identity
        self._attestation: Attestation | None = attestation

    @property
    def attestation(self) -> Attestation | None:
        return self._attestation

    def context_fields(self) -> dict[str, Any]:
        return {}

    def stop(self) -> None:
        self._attestation = None


def _attestation(model: ResolvedModel) -> Attestation:
    return Attestation(engine_version="0", served_model=model.ref.id, model_identity=model.identity, config_digest="c")


@pytest.mark.logic
def test_server_start_lock_per_env_and_model() -> None:
    """A server's start leases its env, backend and model alone: another model starts while it starts, once."""
    first, second = _model("a" * 40), _model("b" * 40)
    second_started = threading.Event()
    starts: list[str] = []

    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> _Server:  # noqa: ARG001 - a ServerStart
        starts.append(model.identity)
        if model is first:
            assert second_started.wait(5), "the second server's start waited for the first's"
        else:
            second_started.set()
        return _Server(_attestation(model))

    backend = fake.FAKE.model_copy(
        update={"name": "served", "env": Env.server("s"), "server": ServerConfig(settings={})}
    )
    workers = WorkerSet(1, start_server=start)
    failures: list[BaseException] = []

    def serve(model: ResolvedModel) -> None:
        try:
            workers.server(backend, ENV, model, PROFILE, _attestation(model))
        except BaseException as exc:  # noqa: BLE001 - every failure is reported by the assertion
            failures.append(exc)

    threads = [threading.Thread(target=serve, args=(m,)) for m in (first, second, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == [] and sorted(starts) == sorted([first.identity, second.identity])
    served = workers.server(backend, ENV, first, PROFILE, _attestation(first))
    assert served.attestation == _attestation(first) and len(starts) == 2
    workers.close()
    assert served.attestation is None


@pytest.mark.logic
def test_same_named_host_backends_of_two_tasks_never_share_a_loaded_state() -> None:
    """Two host backends named `dsp`, of two tasks, load their own state; a model's key is its identity as before."""
    first = _host("dsp").model_copy(update={"worker": "backends.load_counter_backend_fixture"})
    second = first.model_copy(update={"task": "worker_set_test.other"})
    workers, states = WorkerSet(4), []
    for backend in (first, second, first):
        with workers.checkout(backend, ENV, None, device="cpu", profile=PROFILE) as worker:
            states.append(worker.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["state"])
    workers.close()
    assert states[0] != states[1] and states[2] == states[0]
    keys = {
        host_model_key(b, e, None, "cpu")
        for b in (first, second)
        for e in (ENV, dataclasses.replace(ENV, python="other"))
    }
    assert len(keys) == 4


@pytest.mark.logic
def test_a_model_is_shared_by_one_backend_code_across_tasks_and_never_by_two() -> None:
    """Two tasks on one backend code and model share its loaded state; a same-named backend of other code never does."""
    model = _model("a" * 40)
    first = _host("dsp").model_copy(update={"worker": "backends.load_counter_backend_fixture"})
    same_code = first.model_copy(update={"task": "worker_set_test.other"})
    other_code = first.model_copy(update={"worker": "backends.sizes_backend_fixture"})
    key = host_model_key(first, ENV, model, "cpu")
    assert key == ("dsp:backends.load_counter_backend_fixture@1", "a" * 40, "cpu", PRECISION)
    assert host_model_key(same_code, ENV, model, "cpu") == key != host_model_key(other_code, ENV, model, "cpu")
    assert host_model_key(first.model_copy(update={"version": "2"}), ENV, model, "cpu") != key
    workers, states = WorkerSet(4), []
    for backend in (first, same_code):
        with workers.checkout(backend, ENV, model, device="cpu", profile=PROFILE) as worker:
            states.append(worker.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["state"])
    workers.close()
    assert states[0] == states[1]


@pytest.mark.logic
def test_release_frees_the_cached_models_workers_and_servers() -> None:
    """`release` frees a host backend's cached model and stops the set's servers; the next checkout loads again."""
    stopped: list[str] = []

    class Served:
        label, url, attestation = "served", "http://127.0.0.1:1", _attestation(_model("d" * 40))

        def context_fields(self) -> dict[str, Any]:
            return {}

        def stop(self) -> None:
            stopped.append(self.label)

    model = _model("d" * 40)
    backend = _host("released").model_copy(update={"worker": "backends.release_backend_fixture"})
    workers = WorkerSet(2, start_server=lambda *_: Served())
    workers.server(backend, ENV, model, PROFILE, Served.attestation)

    def load() -> int:
        with workers.checkout(backend, ENV, model, device="cpu", profile=PROFILE) as worker:
            return int(worker.run([{"id": "a", "inputs": {}}], {})[0]["outputs"]["state"])

    first = load()
    assert release_fixture.RELEASED == [] and load() == first
    workers.release()
    assert release_fixture.RELEASED == [first] and stopped == ["served"] and workers.pooled() == []
    assert load() != first
    workers.close()


@pytest.mark.logic
def test_a_host_backend_serving_models_must_size_its_state() -> None:
    """A host backend declaring `models` with no `state_bytes` is refused; one declaring none counts its state 0."""
    serving = _host("serving").model_copy(update={"models": ("org/m",)})
    with pytest.raises(ValueError, match="defines no state_bytes"):
        with WorkerSet(1).checkout(serving, ENV, None, device="cpu", profile=PROFILE):
            pass
    with WorkerSet(1).checkout(_host("plain"), ENV, None, device="cpu", profile=PROFILE) as worker:
        assert worker.run([{"id": "a", "inputs": {"x": "v"}}], {})[0]["outputs"] == {"value": "v"}


@pytest.mark.logic
def test_a_busy_server_start_lease_is_not_a_load_failure() -> None:
    """A server start whose lease is busy raises `LeaseTimeout` as it is: a busy resource, never a broken env."""

    def start(*_: Any) -> Any:  # noqa: ANN401 - a ServerStart
        raise LeaseTimeout("another holder is starting this server")

    backend = _host("served").model_copy(update={"env": Env.server("s"), "server": ServerConfig(settings={})})
    model = _model("c" * 40)
    with pytest.raises(LeaseTimeout):
        with WorkerSet(1, start_server=start).checkout(
            backend, ENV, model, device="cpu", profile=PROFILE, expected=_attestation(model)
        ):
            pass


@pytest.mark.logic
def test_a_server_that_fails_to_start_is_a_load_failure() -> None:
    """A server that fails to start reaches a checkout as the cause of `WorkerLoadFailed`, an infrastructure fault."""

    def start(*_: Any) -> Any:  # noqa: ANN401 - a ServerStart
        raise VllmStartError("timeout", "the server did not come up")

    backend = _host("served").model_copy(update={"env": Env.server("s"), "server": ServerConfig(settings={})})
    model = _model("b" * 40)
    workers = WorkerSet(1, start_server=start)
    with pytest.raises(WorkerLoadFailed, match="VllmStartError") as raised:
        with workers.checkout(backend, ENV, model, device="cpu", profile=PROFILE, expected=_attestation(model)):
            pass
    assert isinstance(raised.value.__cause__, VllmStartError) and classify(raised.value) is FaultClass.INFRASTRUCTURE


@pytest.mark.logic
def test_a_server_short_of_device_memory_is_a_resource_fault() -> None:
    """A server that fails to start for lack of device memory is the cause of a resource `WorkerLoadFailed`."""

    def start(*_: Any) -> Any:  # noqa: ANN401 - a ServerStart
        raise VllmOutOfMemory("vllm serve exited with 1 before it was healthy: ValueError: Free memory on device")

    backend = _host("served").model_copy(update={"env": Env.server("s"), "server": ServerConfig(settings={})})
    model = _model("e" * 40)
    with pytest.raises(WorkerLoadFailed) as raised:
        with WorkerSet(1, start_server=start).checkout(
            backend, ENV, model, device="cpu", profile=PROFILE, expected=_attestation(model)
        ):
            pass
    cause = raised.value.__cause__
    assert isinstance(cause, VllmOutOfMemory) and cause.kind == "memory"
    assert classify(raised.value) is FaultClass.RESOURCE
