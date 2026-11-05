"""`WorkerSet`: a bounded pool that closes idle workers first, and server starts leased per env, backend and model."""

import platform
import threading
from typing import Any

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import Attestation
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python

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


class _Server:
    """A started server that attests what it was given."""

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
    """A server's start leases its env, backend and model alone: another model starts while it starts."""
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
        update={"name": "served", "env": Env.server("s"), "server": ServerConfig(settings={}, startup_timeout_s=1.0)}
    )
    workers = WorkerSet(1, start_server=start)
    failures: list[BaseException] = []

    def serve(model: ResolvedModel) -> None:
        try:
            workers.server(backend, ENV, model, PROFILE, _attestation(model))
        except BaseException as exc:  # noqa: BLE001 - every failure is reported by the assertion
            failures.append(exc)

    threads = [threading.Thread(target=serve, args=(m,)) for m in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == [] and sorted(starts) == sorted([first.identity, second.identity])
    served = workers.server(backend, ENV, first, PROFILE, _attestation(first))
    assert served.attestation == _attestation(first) and len(starts) == 2
    workers.close()
    assert served.attestation is None
