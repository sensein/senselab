"""`run_call`: placement by profile, store reuse, one activity per call, model source and profile digest recorded."""

import os
import platform
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import senselab_core.runtime.call as call_module
import senselab_core.runtime.fake_backend as fake
import soundfile
from senselab_core.kernel import prov
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, OutputIdentity, ReusePolicy
from senselab_core.runtime.audio_io import DECODER
from senselab_core.runtime.call import (
    CallFailed,
    Packaged,
    RunConfig,
    TaskCall,
    run_call,
    use_packaged,
)
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerClient

TOY = "toyorg/toy"
REVISION = "a" * 40
TASK = TaskSpec(
    name="call_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=fake.EchoParams,
)
AUDIO_TASK = TaskSpec(
    name="call_test.identify",
    version="1",
    kind="transform",
    inputs={"audio": "Audio"},
    outputs={"value": "sha256"},
    params=fake.EchoParams,
)


def _backend(name: str, stack: str, env: Env, task: TaskSpec = TASK, worker: str = fake.WORKER_MODULE) -> BackendSpec:
    return BackendSpec(
        task=task.name,
        name=name,
        version="1",
        stack=stack,
        models=(TOY,) if task is TASK else (),
        env=env,
        devices=("cpu",),
        batch_invariant=True,
        worker=worker,
        load_stub={},
    )


TEST_REGISTRY = Registry()
for _task in (TASK, AUDIO_TASK, fake.ECHO):
    TEST_REGISTRY.add_task(_task)
TEST_REGISTRY.add_backend(_backend("fake-alpha", "alpha", Env.host()))
TEST_REGISTRY.add_backend(_backend("fake-beta", "beta", Env.host()))
TEST_REGISTRY.add_backend(_backend("fake-isolated", "isolated", Env.isolated("fake")))
TEST_REGISTRY.add_backend(fake.FAKE)
TEST_REGISTRY.add_backend(_backend("identity", "dsp", Env.host(), AUDIO_TASK, "backends.identity_backend_fixture"))

CATALOGUE = Catalogue.load(
    f"""
format: catalogue@1.0
entries:
- id: {TOY}
  revision: {REVISION}
  tasks: [{TASK.name}]
  backends: [fake-alpha, fake-beta, fake-isolated]
  stacks: [alpha, beta, isolated]
  default_for: [{TASK.name}]
  license:
    expression: MIT
    source: toy card
    verified: {{date: 2026-11-01, how: hf-card}}
    gated: false
"""
)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    """Every env is the running interpreter: the declared test seam of the fake stack; nothing is built."""
    return EnvHandle(
        python=sys.executable,
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """`run_call` reads the toy catalogue and registry and opens every env as this interpreter."""
    previous = use_packaged(Packaged(CATALOGUE, _this_interpreter, TEST_REGISTRY))
    yield
    use_packaged(previous)


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    """A fresh result store."""
    return ResultStore(tmp_path / "store")


def _profile(*stacks: str, **override: object) -> ExecutionProfile:
    return ExecutionProfile.load(
        "laptop", {"devices": ["cpu"], "stack_preferences": {TASK.name: list(stacks)}, **override}
    )


def _call(text: object = "hello", **params: object) -> TaskCall:
    return TaskCall.model_validate({"task": TASK.name, "inputs": {"text": text}, "params": params})


def _activities(store: ResultStore) -> list[prov.Record]:
    return store.provenance.activities()


def _executions(store: ResultStore) -> list[prov.Record]:
    return [a for a in _activities(store) if "senselab:Reuse" not in a.types()]


@pytest.mark.logic
def test_stack_switch_by_profile(store: ResultStore) -> None:
    """Changing `stack_preferences` alone changes the backend and the recorded key, never the call or output ports."""
    call = _call()
    alpha = run_call(call, execution=_profile("alpha"), store=store)
    beta = run_call(call, execution=_profile("beta"), store=store)
    assert (alpha.placement.backend.stack, beta.placement.backend.stack) == ("alpha", "beta")
    assert (alpha.key.backend, beta.key.backend) == ("fake-alpha", "fake-beta")
    assert alpha.outputs.keys() == beta.outputs.keys() == set(TASK.outputs)
    keys = [prov.literal_value(a.node()["senselab:logicalKey"]) for a in _executions(store)]
    assert sorted(k["backend"] for k in keys if isinstance(k, dict)) == ["fake-alpha", "fake-beta"]


@pytest.mark.logic
def test_second_call_hits_store(store: ResultStore) -> None:
    """The same call again is served from the store, with the outputs and identity the first one stored."""
    first = run_call(_call(salt="s"), execution=_profile("alpha"), store=store)
    second = run_call(_call(salt="s"), execution=_profile("alpha"), store=store)
    assert (first.reused, second.reused) == (None, ReusePolicy.EXACT)
    assert second.output_identity == first.output_identity and second.outputs == first.outputs
    assert run_call(_call(salt="t"), execution=_profile("alpha"), store=store).reused is None


@pytest.mark.logic
def test_one_activity_per_call(store: ResultStore) -> None:
    """An execution writes one activity and a store hit only the store's reuse activity, never a second execution."""
    run_call(_call("one"), execution=_profile("alpha"), store=store)
    [execution] = _activities(store)
    assert "senselab:Reuse" not in execution.types()
    assert execution.node()["prov:generated"] == [{"@id": _result_id(store)}]
    run_call(_call("one"), execution=_profile("alpha"), store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    assert len(_activities(store)) == 2
    assert _executions(store) == [execution]


def _result_id(store: ResultStore) -> str:
    [entity] = [e for e in store.provenance.entities() if e.id.startswith("output:")]
    return entity.id


@pytest.mark.logic
def test_model_source_recorded(store: ResultStore) -> None:
    """The context's `model_source` is how the weights were reached; the agent names the model at its pin."""
    result = run_call(_call(), execution=_profile("alpha"), store=store)
    assert result.context.model_source == "hf"
    assert result.key.model is not None and (result.key.model.id, result.key.model.identity) == (TOY, REVISION)
    [agent] = store.provenance.agents()
    assert agent.node()["senselab:modelSource"] == f"hf:{TOY}@{REVISION}"
    [execution] = _executions(store)
    assert execution.node()["prov:wasAssociatedWith"] == [{"@id": agent.id}]
    modelless = run_call(
        TaskCall.model_validate({"task": fake.ECHO.name, "inputs": {"payload": {"x": 1}}}),
        execution=_profile(),
        store=store,
    )
    assert modelless.context.model_source is None and modelless.key.model is None


@pytest.mark.logic
def test_profile_digest_recorded_not_keyed(store: ResultStore) -> None:
    """Each execution records its profile's digest; a profile that changes no placement changes no key."""
    plain, retrying = _profile("alpha"), _profile("alpha", retry={"attempts": 5})
    first = run_call(_call(), execution=plain, store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    second = run_call(_call(), execution=retrying, store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    assert first.key == second.key and second.reused is ReusePolicy.EXACT
    [execution] = _executions(store)
    assert prov.literal_value(execution.node()["senselab:executionProfile"]) == plain.canonical_digest()


@pytest.mark.logic
def test_backend_params_reach_the_key_and_are_validated(store: ResultStore) -> None:
    """Backend params are validated against the chosen backend; an unknown one is refused before anything runs."""
    call = TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "x"}, "backend_params": {"beam": 2}})
    with pytest.raises(ValueError, match="beam"):
        run_call(call, execution=_profile("alpha"), store=store)
    assert _activities(store) == []


@pytest.mark.logic
def test_backend_fault_raises_call_failed(store: ResultStore) -> None:
    """An item the backend fails raises `CallFailed` with its fault class; nothing is stored or recorded."""
    call = TaskCall.model_validate({"task": fake.ECHO.name, "inputs": {"payload": float("nan")}})
    with pytest.raises(CallFailed) as info:
        run_call(call, execution=_profile(), store=store)
    assert info.value.error["fault"].value == "content"
    assert _activities(store) == [] and store.provenance.entities() == []


@pytest.mark.contract
def test_isolated_worker_equals_host(store: ResultStore) -> None:
    """An isolated backend runs in a pooled worker process and gives the host backend's outputs."""
    host = run_call(_call("same"), execution=_profile("alpha"), store=store)
    isolated = run_call(_call("same"), execution=_profile("isolated"), store=store)
    again = run_call(_call("other"), execution=_profile("isolated"), store=store)
    assert isolated.outputs == host.outputs and isolated.key != host.key
    assert again.reused is None and again.outputs != isolated.outputs


@pytest.mark.logic
def test_call_shape_validates_against_the_declaration() -> None:
    """A call from the one call shape validates its params against the task and parses its model reference."""
    call = TaskCall.model_validate({"task": TASK.name, "inputs": {"text": 1}, "model": f"{TOY}@{REVISION}"})
    assert call.task_version == "1" and type(call.params) is fake.EchoParams
    assert call.model is not None and call.model.ref == REVISION
    with pytest.raises(ValueError, match="salty"):
        TaskCall.model_validate({"task": TASK.name, "inputs": {"text": 1}, "params": {"salty": 1}})


@pytest.mark.contract
def test_dead_pooled_worker_is_closed_and_replaced(store: ResultStore) -> None:
    """A pooled worker that died is closed, its array directory removed, and a new one runs the next call."""
    run_call(_call("first"), execution=_profile("isolated"), store=store)
    [old] = [w for w in call_module._POOL.values() if isinstance(w, WorkerClient) and w.alive]
    os.kill(int(old.hello["pid"]), signal.SIGKILL)
    deadline = time.monotonic() + 10
    while old.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not old.alive
    result = run_call(_call("second"), execution=_profile("isolated"), store=store)
    assert result.reused is None
    assert not old.array_dir.exists() and old not in call_module._POOL.values()


@pytest.mark.logic
def test_budget_enters_the_host_pool(store: ResultStore) -> None:
    """A host worker pooled under one cache budget is not reused under another, whose sizing it lacks."""
    run_call(_call("budget"), execution=_profile("alpha"), store=store)
    with pytest.raises(ValueError, match="state_bytes"):
        run_call(_call("budget-2"), execution=_profile("alpha", model_cache_budget_bytes=10), store=store)


@pytest.mark.logic
def test_decoder_recorded_when_a_path_is_decoded(store: ResultStore, tmp_path: Path) -> None:
    """An input read from a path records `open_audio`'s decoder; an in-memory input records none."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, numpy.zeros(160, dtype=numpy.float32), 16000)
    call = TaskCall.model_validate({"task": AUDIO_TASK.name, "inputs": {"audio": str(path)}})
    result = run_call(call, execution=_profile(), store=store)
    assert result.context.decoder == DECODER
    assert run_call(_call(), execution=_profile("alpha"), store=store).context.decoder is None


class _BeatenToCommit(ResultStore):
    """A store where another writer commits different outputs for the key just before each `put`."""

    def put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> OutputIdentity:
        super().put(key, {"value": "the other writer's"}, context)
        return super().put(key, outputs, context)


@pytest.mark.logic
def test_lost_commit_race_serves_the_committed_entry(tmp_path: Path) -> None:
    """When another writer committed first, the call returns that entry as a reuse and writes no execution."""
    store = _BeatenToCommit(tmp_path / "store")
    result = run_call(_call("race"), execution=_profile("alpha"), store=store)
    assert result.outputs == {"value": "the other writer's"} and result.reused is ReusePolicy.EXACT
    assert _executions(store) == [] and len(_activities(store)) == 1
