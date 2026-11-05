"""`InProcess`: placement by profile, store reuse, one activity per call, model source and profile digest recorded."""

import dataclasses
import importlib
import json
import os
import platform
import signal
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import senselab_core.runtime.call as call_module
import senselab_core.runtime.fake_backend as fake
import soundfile
from pydantic import ValidationError
from senselab_core.kernel import prov
from senselab_core.kernel._base import Frozen
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import (
    NO_EQUIVALENCE,
    Equivalence,
    ExecutionContext,
    LogicalKey,
    ModelRef,
    OutputIdentity,
    ReusePolicy,
    member_key,
    port_output_identity,
)
from senselab_core.kernel.license import License
from senselab_core.kernel.results import Intervals, Metric, Transcript, TranscriptNode, Trials
from senselab_core.kernel.signals import Audio, PlainText
from senselab_core.kernel.time import Span, TimeAnchor
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime.audio_io import DECODER
from senselab_core.runtime.call import Packaged, Result, RunConfig, TaskCall, bind_input, use_packaged
from senselab_core.runtime.call import packaged as packaged_now
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.compare import Thresholds
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.equivalence import SAMPLING
from senselab_core.runtime.execution.executor import JobRef, PlanFragment, Status
from senselab_core.runtime.execution.in_process import CallFailed, InProcess, node_fragment, process_workers
from senselab_core.runtime.execution.job import Completed, Job, lost_race
from senselab_core.runtime.execution.outputs import AmbiguousClock, OutputTypeMismatch
from senselab_core.runtime.execution.records import verify_attributes
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.keys import Call
from senselab_core.runtime.licenses import LicensePolicy, LicenseRefused, license_record
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import ConcurrentUse, WorkerClient
from senselab_core.testing.interpreter import source_tree_python
from senselab_worker.faults import InfrastructureFault

PYTHON = source_tree_python()

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


class ReturnParams(TaskParams):
    """Params of the transcribe test task: the outputs the fake returns, as JSON text."""

    outputs_json: str = ""


TRANSCRIBE_TASK = TaskSpec(
    name="call_test.transcribe",
    version="1",
    kind="transform",
    inputs={"text": "Text"},
    outputs={"transcript": "Transcript"},
    params=ReturnParams,
)

PAIR_TASK = TaskSpec(
    name="call_test.pair_audio",
    version="1",
    kind="transform",
    inputs={"first": "Audio", "second": "Audio"},
    outputs={"value": "sha256"},
    params=fake.EchoParams,
)

CLOCKED_TASK = TaskSpec(
    name="call_test.detect",
    version="1",
    kind="transform",
    inputs={"audio": "Audio"},
    outputs={"spans": "Intervals"},
    params=fake.EchoParams,
)

TRIALS_TASK = TaskSpec(
    name="call_test.pair",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"trials": "Trials"},
    params=fake.EchoParams,
)
OVERLAP_TASK = TRIALS_TASK.model_copy(update={"name": "call_test.overlap", "inputs": {"label": "json"}})
CHAIN_TASK = TASK.model_copy(update={"name": "call_test.chain", "inputs": {"value": "sha256"}})
TWO_PORTS_TASK = TASK.model_copy(update={"name": "call_test.two_ports", "outputs": {"x": "sha256", "y": "sha256"}})


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
TASKS = (
    TASK,
    AUDIO_TASK,
    PAIR_TASK,
    CLOCKED_TASK,
    TRANSCRIBE_TASK,
    TRIALS_TASK,
    OVERLAP_TASK,
    CHAIN_TASK,
    TWO_PORTS_TASK,
    fake.ECHO,
)
for _task in TASKS:
    TEST_REGISTRY.add_task(_task)
TEST_REGISTRY.add_backend(_backend("fake-alpha", "alpha", Env.host()))
TEST_REGISTRY.add_backend(
    BackendSpec(
        **{
            **_backend("fake-batched", "batched", Env.host()).model_dump(),
            "batch_invariant": False,
            "output_affecting": ("batch_formation", "chunk_grid"),
            "chunk_grid": "whole-item",
        }
    )
)
TEST_REGISTRY.add_backend(_backend("fake-beta", "beta", Env.host()))
TEST_REGISTRY.add_backend(_backend("fake-isolated", "isolated", Env.isolated("fake")))
TEST_REGISTRY.add_backend(fake.FAKE)
TEST_REGISTRY.add_backend(_backend("identity", "dsp", Env.host(), AUDIO_TASK, "backends.identity_backend_fixture"))
TEST_REGISTRY.add_backend(_backend("fake-audio", "isolated", Env.isolated("fake"), AUDIO_TASK))
TEST_REGISTRY.add_backend(_backend("fake-transcribe", "isolated", Env.isolated("fake"), TRANSCRIBE_TASK))
TEST_REGISTRY.add_backend(_backend("pair", "dsp", Env.host(), PAIR_TASK, "backends.identity_backend_fixture"))
TEST_REGISTRY.add_backend(_backend("clocked", "dsp", Env.host(), CLOCKED_TASK, "backends.clocked_backend_fixture"))
TEST_REGISTRY.add_backend(_backend("trials", "dsp", Env.host(), TRIALS_TASK, "backends.trials_backend_fixture"))
TEST_REGISTRY.add_backend(_backend("overlap", "dsp", Env.host(), OVERLAP_TASK, "backends.overlap_backend_fixture"))
TEST_REGISTRY.add_backend(_backend("chain", "dsp", Env.host(), CHAIN_TASK, "backends.sizes_backend_fixture"))
TEST_REGISTRY.add_backend(
    _backend("two-ports", "dsp", Env.host(), TWO_PORTS_TASK, "backends.two_ports_backend_fixture")
)
TEST_REGISTRY.add_backend(_backend("counting", "counting", Env.host(), fake.ECHO, "backends.counting_backend_fixture"))

CATALOGUE = Catalogue.load(
    f"""
format: catalogue@2.0
entries:
- id: {TOY}
  revision: {REVISION}
  tasks: [{TASK.name}]
  backends: [fake-alpha, fake-batched, fake-beta, fake-isolated]
  stacks: [alpha, batched, beta, isolated]
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
        python=PYTHON,
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


WORKERS = WorkerSet(4)
"""The worker set every call of this module runs on; closed after each test."""


def _in_process(
    call: TaskCall,
    *,
    store: ResultStore,
    execution: ExecutionProfile | None = None,
    run: RunConfig | None = None,
) -> Result:
    """Run `call` on an `InProcess` executor over `store`, under `execution` and `run`, on `WORKERS`."""
    return InProcess(store=store, profile=execution, run=run, workers=WORKERS).call(call)


@pytest.fixture(autouse=True)
def workers() -> Iterator[None]:
    """Every worker and server a test starts is stopped after it."""
    yield
    WORKERS.close()


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """The executor reads the toy catalogue and registry, opens every env as this interpreter, locked to MIT code."""
    records = {"pkg": _record("MIT")}
    previous = use_packaged(Packaged(CATALOGUE, _this_interpreter, TEST_REGISTRY, lambda _: records))
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
def test_catalogue_metadata_binds_to_the_pinned_revision() -> None:
    """A model resolved to its pin carries the entry's license and placement; another commit carries neither."""
    model, entry = call_module._resolve_model(TASK, ModelRef.parse(f"{TOY}@{REVISION}"), _profile("alpha"), CATALOGUE)
    assert model is not None and model.license is not None and entry is not None
    other = "b" * 40
    model, entry = call_module._resolve_model(TASK, ModelRef.parse(f"{TOY}@{other}"), _profile("alpha"), CATALOGUE)
    assert model is not None and (model.identity, model.license, entry) == (other, None, None)


@pytest.mark.logic
def test_declared_batch_formation_and_chunk_grid_are_produced(store: ResultStore) -> None:
    """A backend declaring batch formation and chunk grid runs, keyed by one-item requests on its chunk grid."""
    result = _in_process(_call(), execution=_profile("batched"), store=store)
    assert (result.key.batch_formation, result.key.chunk_grid) == ("single", "whole-item")


@pytest.mark.logic
def test_stack_switch_by_profile(store: ResultStore) -> None:
    """Changing `stack_preferences` alone changes the backend and the recorded key, never the call or output ports."""
    call = _call()
    alpha = _in_process(call, execution=_profile("alpha"), store=store)
    beta = _in_process(call, execution=_profile("beta"), store=store)
    assert (alpha.placement.backend.stack, beta.placement.backend.stack) == ("alpha", "beta")
    assert (alpha.key.backend, beta.key.backend) == ("fake-alpha", "fake-beta")
    assert alpha.outputs.keys() == beta.outputs.keys() == set(TASK.outputs)
    keys = [prov.literal_value(a.node()["senselab:logicalKey"]) for a in _executions(store)]
    assert sorted(k["backend"] for k in keys if isinstance(k, dict)) == ["fake-alpha", "fake-beta"]


@pytest.mark.logic
def test_second_call_hits_store(store: ResultStore) -> None:
    """The same call again is served from the store, with the outputs and identity the first one stored."""
    first = _in_process(_call(salt="s"), execution=_profile("alpha"), store=store)
    second = _in_process(_call(salt="s"), execution=_profile("alpha"), store=store)
    assert (first.reused, second.reused) == (None, ReusePolicy.EXACT)
    assert second.output_identity == first.output_identity and second.outputs == first.outputs
    assert _in_process(_call(salt="t"), execution=_profile("alpha"), store=store).reused is None


@pytest.mark.logic
def test_one_activity_per_call(store: ResultStore) -> None:
    """An execution writes one activity and a store hit only the store's reuse activity, never a second execution."""
    _in_process(_call("one"), execution=_profile("alpha"), store=store)
    [execution] = _activities(store)
    assert "senselab:Reuse" not in execution.types()
    assert execution.node()["prov:generated"] == [{"@id": _result_id(store)}]
    _in_process(_call("one"), execution=_profile("alpha"), store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    assert len(_activities(store)) == 2
    assert _executions(store) == [execution]


def _result_id(store: ResultStore) -> str:
    [entity] = [e for e in store.provenance.entities() if e.id.startswith("output:")]
    return entity.id


@pytest.mark.logic
def test_model_source_recorded(store: ResultStore) -> None:
    """The context's `model_source` is how the weights were reached; the agent names the model at its pin."""
    result = _in_process(_call(), execution=_profile("alpha"), store=store)
    assert result.context.model_source == "hf"
    assert result.key.model is not None and (result.key.model.id, result.key.model.identity) == (TOY, REVISION)
    [agent] = store.provenance.agents()
    assert agent.node()["senselab:modelSource"] == f"hf:{TOY}@{REVISION}"
    assert prov.literal_value(agent.node()["senselab:envLock"]) == result.context.env_lock
    [execution] = _executions(store)
    assert execution.node()["prov:wasAssociatedWith"] == [{"@id": agent.id}]
    modelless = _in_process(
        TaskCall.model_validate({"task": fake.ECHO.name, "inputs": {"payload": {"x": 1}}}),
        execution=_profile(),
        store=store,
    )
    assert modelless.context.model_source is None and modelless.key.model is None


@pytest.mark.logic
def test_profile_digest_recorded_not_keyed(store: ResultStore) -> None:
    """Each execution records its profile's digest; a profile that changes no placement changes no key."""
    plain, retrying = _profile("alpha"), _profile("alpha", retry={"attempts": 5})
    first = _in_process(_call(), execution=plain, store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    second = _in_process(_call(), execution=retrying, store=store, run=RunConfig(reuse=ReusePolicy.EXACT))
    assert first.key == second.key and second.reused is ReusePolicy.EXACT
    [execution] = _executions(store)
    assert prov.literal_value(execution.node()["senselab:executionProfile"]) == plain.canonical_digest()


@pytest.mark.logic
def test_backend_params_reach_the_key_and_are_validated(store: ResultStore) -> None:
    """Only the chosen backend's entry is validated against it; an unknown param is refused before anything runs."""
    call = TaskCall.model_validate(
        {"task": TASK.name, "inputs": {"text": "x"}, "backend_params": {"fake-alpha": {"beam": 2}}}
    )
    with pytest.raises(ValueError, match="beam"):
        _in_process(call, execution=_profile("alpha"), store=store)
    assert _activities(store) == []
    with pytest.raises(ValidationError):
        TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "x"}, "backend_params": {"beam": 2}})


@pytest.mark.logic
def test_backend_fault_raises_call_failed(store: ResultStore) -> None:
    """An item the backend fails raises `CallFailed` with its fault class; nothing is stored or recorded."""
    call = TaskCall.model_validate({"task": fake.ECHO.name, "inputs": {"payload": float("nan")}})
    with pytest.raises(CallFailed) as info:
        _in_process(call, execution=_profile(), store=store)
    assert info.value.failure.fault == "content"
    assert _activities(store) == [] and store.provenance.entities() == []


@pytest.mark.contract
def test_isolated_worker_equals_host(store: ResultStore) -> None:
    """An isolated backend runs in a pooled worker process and gives the host backend's outputs."""
    host = _in_process(_call("same"), execution=_profile("alpha"), store=store)
    isolated = _in_process(_call("same"), execution=_profile("isolated"), store=store)
    again = _in_process(_call("other"), execution=_profile("isolated"), store=store)
    assert isolated.outputs == host.outputs and isolated.key != host.key
    assert again.reused is None and again.outputs != isolated.outputs


@pytest.mark.contract
def test_isolated_worker_reads_audio_by_its_wire_form(store: ResultStore, tmp_path: Path) -> None:
    """An isolated backend is sent audio, in memory or read from a path, as its wire form."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, numpy.array([0.25, -0.5], dtype=numpy.float32), 16000, subtype="FLOAT")
    profile = _profile(stack_preferences={AUDIO_TASK.name: ["isolated"]})
    calls = [
        TaskCall.model_validate({"task": AUDIO_TASK.name, "inputs": {"audio": a}})
        for a in (str(path), Audio.from_array(numpy.array([0.25, -0.5], dtype=numpy.float32), 16000))
    ]
    results = [_in_process(c, execution=profile, store=store) for c in calls]
    WORKERS.close()
    assert all(r.placement.backend.name == "fake-audio" and r.reused is None for r in results)
    assert results[0].outputs != results[1].outputs


@pytest.mark.contract
def test_isolated_outputs_are_read_as_their_declared_types(store: ResultStore) -> None:
    """An isolated backend's wire-form output comes back as its declared type, also from the store; another is refused.

    A refused output is not stored.
    """
    span = Span(start_ns=0, end_ns=400_000_000, clock_id="c")
    transcript = Transcript(segments=(TranscriptNode(text="hi", span=span, confidence=0.5),), extent=span)
    profile = _profile(stack_preferences={TRANSCRIBE_TASK.name: ["isolated"]})

    def transcribe(output: Frozen, text: str) -> Result:
        outputs_json = json.dumps({"transcript": wire_form(output)})
        params = {"outputs_json": outputs_json}
        call = TaskCall.model_validate({"task": TRANSCRIBE_TASK.name, "inputs": {"text": text}, "params": params})
        return _in_process(call, execution=profile, store=store)

    fresh, hit = transcribe(transcript, "a"), transcribe(transcript, "a")
    assert fresh.outputs == hit.outputs == {"transcript": transcript} and hit.reused is not None
    assert fresh.key.inputs == {"text": PlainText(content="a").identity}
    with pytest.raises(OutputTypeMismatch, match="metric@2.0"):
        transcribe(Metric(name="wer", value=0.1, unit="1", items=1), "b")
    WORKERS.close()
    assert len(_executions(store)) == 1


@pytest.mark.logic
def test_array_outputs_are_stored_and_read_back(store: ResultStore) -> None:
    """A host backend's `Trials` output is stored with its arrays and read back equal from the store."""
    call = TaskCall.model_validate({"task": TRIALS_TASK.name, "inputs": {"text": "pairs"}})
    fresh, hit = (_in_process(call, execution=_profile(), store=store) for _ in range(2))
    assert isinstance(fresh.outputs["trials"], Trials) and fresh.reused is None and hit.reused is not None
    assert hit.outputs == fresh.outputs and hit.output_identity == fresh.output_identity


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
    _in_process(_call("first"), execution=_profile("isolated"), store=store)
    [old] = [w for w in WORKERS.pooled() if isinstance(w, WorkerClient) and w.alive]
    os.kill(int(old.hello["pid"]), signal.SIGKILL)
    deadline = time.monotonic() + 10
    while old.alive and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not old.alive
    result = _in_process(_call("second"), execution=_profile("isolated"), store=store)
    assert result.reused is None
    assert not old.array_dir.exists() and old not in WORKERS.pooled()


@pytest.mark.logic
def test_concurrent_calls_take_turns_on_a_pooled_worker(store: ResultStore) -> None:
    """Two threads calling one backend and model each hold the pooled worker for their call, never at once."""
    overlap = importlib.import_module("backends.overlap_backend_fixture")
    overlap.BARRIER.reset()
    overlap.MET.clear()
    start = threading.Barrier(2)
    failures: list[BaseException] = []

    def call(text: str) -> None:
        start.wait()
        try:
            _in_process(TaskCall.model_validate({"task": OVERLAP_TASK.name, "inputs": {"label": text}}), store=store)
        except Exception as exc:  # noqa: BLE001 - every failure is reported by the assertion
            failures.append(exc)

    threads = [threading.Thread(target=call, args=(text,)) for text in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == [] and overlap.MET == [False, False]


@pytest.mark.logic
def test_concurrent_use_of_a_worker_client_is_an_infrastructure_fault() -> None:
    """A second `run` in flight on one client is an infrastructure fault, typed as such."""
    assert issubclass(ConcurrentUse, InfrastructureFault)


@pytest.mark.logic
def test_budget_enters_the_host_pool(store: ResultStore) -> None:
    """A host worker pooled under one cache budget is not reused under another, whose sizing it lacks."""
    _in_process(_call("budget"), execution=_profile("alpha"), store=store)
    with pytest.raises(ValueError, match="state_bytes"):
        _in_process(_call("budget-2"), execution=_profile("alpha", model_cache_budget_bytes=10), store=store)


@pytest.mark.logic
def test_decoder_recorded_when_a_path_is_decoded(store: ResultStore, tmp_path: Path) -> None:
    """An input read from a path records `open_audio`'s decoder; an in-memory input records none."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, numpy.zeros(160, dtype=numpy.float32), 16000)
    call = TaskCall.model_validate({"task": AUDIO_TASK.name, "inputs": {"audio": str(path)}})
    result = _in_process(call, execution=_profile(), store=store)
    assert result.context.decoder == DECODER
    assert _in_process(_call(), execution=_profile("alpha"), store=store).context.decoder is None


class _BeatenToCommit(ResultStore):
    """A store where another writer commits different outputs for the key just before each `put`."""

    def put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> OutputIdentity:
        super().put(key, {"value": "the other writer's"}, context)
        return super().put(key, outputs, context)


@pytest.mark.logic
def test_lost_commit_race_serves_the_committed_entry(tmp_path: Path) -> None:
    """When another writer committed first, the call returns that entry as a reuse and writes no execution."""
    store = _BeatenToCommit(tmp_path / "store")
    result = _in_process(_call("race"), execution=_profile("alpha"), store=store)
    assert result.outputs == {"value": "the other writer's"} and result.reused is ReusePolicy.EXACT
    assert _executions(store) == [] and len(_activities(store)) == 1


@pytest.mark.logic
def test_rebound_equals_reclocked_compute(store: ResultStore, tmp_path: Path) -> None:
    """One content under a session anchor, then standalone, gets spans on each reader's clock from one computation.

    Each served result equals the one computed afresh on that reader's recording.
    """
    standalone = Audio.from_array(numpy.zeros(1600, dtype=numpy.float32), 16000)
    session = Audio(groups=standalone.groups, anchor=TimeAnchor(clock_id="session-7", absolute_start_ns=5))

    def detect(audio: Audio, into: ResultStore = store) -> Result:
        call = TaskCall.model_validate({"task": CLOCKED_TASK.name, "inputs": {"audio": audio}})
        return _in_process(call, execution=_profile(), store=into)

    under_session, alone, session_again = detect(session), detect(standalone), detect(session)
    assert (under_session.reused, alone.reused is not None, session_again.reused is not None) == (None, True, True)
    assert len(_executions(store)) == 1 and alone.key == under_session.key
    assert alone.output_identity == under_session.output_identity
    for served, audio, name in ((alone, standalone, "a"), (session_again, session, "b")):
        assert served.outputs == detect(audio, ResultStore(tmp_path / name)).outputs
    spans = [r.outputs["spans"] for r in (under_session, alone, session_again)]
    assert [s.clock_id if isinstance(s, Intervals) else s for s in spans] == [
        "session-7",
        standalone.identity,
        "session-7",
    ]


@pytest.mark.logic
def test_one_content_under_two_anchors_is_refused(store: ResultStore) -> None:
    """Two recording inputs of one content on two clocks raise `AmbiguousClock` before anything runs."""
    audio = Audio.from_array(numpy.zeros(160, dtype=numpy.float32), 16000)
    session = Audio(groups=audio.groups, anchor=TimeAnchor(clock_id="session-7", absolute_start_ns=None))
    call = TaskCall.model_validate({"task": PAIR_TASK.name, "inputs": {"first": audio, "second": session}})
    with pytest.raises(AmbiguousClock, match=audio.identity):
        _in_process(call, execution=_profile(), store=store)
    assert _activities(store) == []


GPU = {"device_kind": "cuda", "precision": "default", "gpu_architecture": "sm_90"}


def _elsewhere(tmp_path: Path, value: object, equivalence: Equivalence = NO_EQUIVALENCE) -> tuple[Result, ResultStore]:
    """Return a call's result here, and a store holding `value` for its key computed on a GPU instead."""
    here = _in_process(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    gpu = ExecutionContext.model_validate({**here.context.model_dump(), "numerics": GPU})
    store = ResultStore(tmp_path / "there", equivalence=equivalence)
    store.put(here.key, {"value": value}, gpu)
    return here, store


def _class(here: Result) -> Equivalence:
    return {
        "fake-alpha": {
            "c": frozenset(
                {member_key(here.context.numerics, here.context.env_lock), member_key(GPU, here.context.env_lock)}
            )
        }
    }


@pytest.mark.logic
def test_unmeasured_equivalent_is_exact(tmp_path: Path) -> None:
    """With no class, `equivalent` serves no other context's result; a measured class of the two numerics does."""
    here, store = _elsewhere(tmp_path, "from the gpu")
    assert _in_process(_call("v"), execution=_profile("alpha"), store=store).reused is None
    _, measured = _elsewhere(tmp_path / "m", "from the gpu", _class(here))
    served = _in_process(_call("v"), execution=_profile("alpha"), store=measured)
    assert served.reused is ReusePolicy.EQUIVALENT and served.outputs == {"value": "from the gpu"}


@pytest.mark.logic
def test_another_env_lock_is_served_only_when_the_run_asks_for_what_is_available(tmp_path: Path) -> None:
    """`exact` and `equivalent` refuse another env lock's unmeasured result; `tolerant` serves it, naming both agents.

    The `equivalent` store holds a class measured within the caller's lock, which does not reach the other lock.
    """
    here = _in_process(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    other = ExecutionContext.model_validate({**here.context.model_dump(), "env_lock": "another-lock"})
    for policy in ReusePolicy:
        store = ResultStore(tmp_path / policy.value, equivalence=_class(here))
        store.put(here.key, {"value": "from another lock"}, other)
        served = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(policy))
        assert served.reused is (ReusePolicy.TOLERANT if policy is ReusePolicy.TOLERANT else None), policy
    assert served.context.env_lock == "another-lock" and served.outputs == {"value": "from another lock"}
    locks = {a.id: prov.literal_value(a.node()["senselab:envLock"]) for a in store.provenance.agents()}
    [reuse] = [a.node() for a in _activities(store) if "senselab:Reuse" in a.types()]
    assert locks[str(prov.literal_value(reuse["senselab:storedAgent"]))] == "another-lock"
    assert reuse["prov:wasAssociatedWith"] == [{"@id": next(i for i, lock in locks.items() if lock != "another-lock")}]
    assert set(locks.values()) == {"another-lock", here.context.env_lock}
    assert prov.literal_value(reuse["senselab:reusePolicy"]) == "tolerant"


@pytest.mark.logic
def test_verify_agreement_adds_evidence(tmp_path: Path) -> None:
    """A sampled cross-context hit is recomputed and, agreeing, adds evidence with its thresholds; `verify: 0` none."""
    here, _ = _elsewhere(tmp_path, None)
    _, store = _elsewhere(tmp_path / "s", here.outputs["value"])
    served = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT))
    assert served.reused is ReusePolicy.TOLERANT and store.evidence() == []
    _, store = _elsewhere(tmp_path / "t", here.outputs["value"])
    run = RunConfig(ReusePolicy.TOLERANT, 1.0)
    verified = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=run)
    assert verified.reused is None and verified.outputs == here.outputs
    [evidence] = store.evidence()
    assert evidence.agree and (evidence.backend, evidence.key) == ("fake-alpha", here.key.digest())
    assert (evidence.stored.numerics, evidence.current.numerics) == (GPU, here.context.numerics)
    assert evidence.thresholds == RunConfig().thresholds.canonical_digest()


@pytest.mark.logic
def test_verify_disagreement_splits_class(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A sampled hit that differs is recorded as disagreeing, splits the store's class and is logged."""
    here, _ = _elsewhere(tmp_path, None)
    _, store = _elsewhere(tmp_path / "s", "not what the cpu computes", _class(here))
    result = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(verify=1.0))
    assert result.reused is None and result.outputs == here.outputs
    [evidence] = store.evidence()
    assert not evidence.agree and [d.path for d in evidence.mismatched] == ["value"]
    assert store.equivalence == {"fake-alpha": {}}
    assert ResultStore(store.root, equivalence=_class(here)).equivalence == {"fake-alpha": {}}
    assert "fake-alpha" in caplog.text and "value" in caplog.text and "not what" not in caplog.text


@pytest.mark.logic
def test_verify_compares_under_the_run_configs_thresholds(tmp_path: Path) -> None:
    """A recomputation within the packaged `embedding` bound agrees; under a run config's tighter bound it differs."""
    call = TaskCall.model_validate({"task": TRIALS_TASK.name, "inputs": {"text": "pairs"}})
    here_store = ResultStore(tmp_path / "here")
    here = _in_process(call, execution=_profile(), store=here_store)
    stored = here_store.get(here.key, here.context, ReusePolicy.EXACT, record=False)
    assert stored is not None
    form = dict(stored.outputs["trials"])  # type: ignore[call-overload]
    nudged: dict[str, object] = {"trials": {**form, "first": numpy.asarray(form["first"]) + 0.05}}
    gpu = ExecutionContext.model_validate({**here.context.model_dump(), "numerics": GPU})
    tight = Thresholds.load({"thresholds": {"embedding": {"atol": 0.01}}})
    found = {}
    for name, run in (
        ("packaged", RunConfig(ReusePolicy.TOLERANT, 1.0)),
        ("tight", RunConfig(ReusePolicy.TOLERANT, 1.0, thresholds=tight)),
    ):
        store = ResultStore(tmp_path / name)
        store.put(here.key, nudged, gpu)
        assert _in_process(call, execution=_profile(), store=store, run=run).reused is None
        [found[name]] = store.evidence()
        [execution] = _activities(store)
        assert prov.literal_value(execution.node()["senselab:verifyThresholds"]) == run.thresholds.canonical_digest()
    assert found["packaged"].agree and found["packaged"].thresholds == Thresholds.load().canonical_digest()
    assert found["tight"].thresholds == tight.canonical_digest()
    [difference] = found["tight"].mismatched
    assert (difference.path, difference.threshold, difference.bound) == (
        "trials.first",
        "embedding",
        tight.by_name["embedding"],
    )
    assert difference.absolute == pytest.approx(0.05) and difference.relative is not None


@pytest.mark.logic
def test_sampled_hit_records_an_execution_not_a_reuse(tmp_path: Path) -> None:
    """A sampled hit records one execution naming its evidence; a served hit one reuse; both record the fraction."""
    here, store = _elsewhere(tmp_path, None)
    _, store = _elsewhere(tmp_path / "s", here.outputs["value"])
    _in_process(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT, 1.0))
    [execution] = _activities(store)
    [evidence] = store.evidence()
    node = execution.node()
    assert "senselab:Reuse" not in execution.types()
    assert prov.literal_value(node["senselab:verifies"]) == evidence.canonical_digest()
    assert prov.literal_value(node["senselab:verifyFraction"]) == 1.0
    assert prov.literal_value(node["senselab:verifySampling"]) == SAMPLING
    _, store = _elsewhere(tmp_path / "t", here.outputs["value"])
    _in_process(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT, 0.0))
    [reuse] = _activities(store)
    assert "senselab:Reuse" in reuse.types() and "senselab:verifies" not in reuse.node()
    assert prov.literal_value(reuse.node()["senselab:verifyFraction"]) == 0.0


@pytest.mark.logic
@pytest.mark.parametrize("k", [-0.1, 1.5])
def test_verify_is_a_fraction(k: float) -> None:
    """`verify` outside [0, 1] is refused."""
    with pytest.raises(ValueError, match="verify"):
        RunConfig(verify=k)


def _record(expression: str) -> License:
    return license_record(expression, source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)


@pytest.fixture
def licensed() -> Iterator[None]:
    """Every env of the fake stack is locked to packages licensed `Apache-2.0 AND MPL-2.0`; the toy model is MIT."""
    records = {"pkg": _record("Apache-2.0 AND MPL-2.0")}
    previous = use_packaged(dataclasses.replace(packaged_now(), lock_records=lambda _: records))
    yield
    use_packaged(previous)


@pytest.mark.logic
@pytest.mark.usefixtures("licensed")
def test_the_env_license_is_recorded_on_the_context_and_its_agent(store: ResultStore) -> None:
    """A result's context and agent carry its env's license; a reuse names that agent as the one that computed it."""
    result = _in_process(_call(), execution=_profile("alpha"), store=store)
    assert result.context.env_license is not None and result.context.env_license.expression == "Apache-2.0 AND MPL-2.0"
    [agent] = store.provenance.agents()
    assert prov.literal_value(agent.node()["senselab:license"]) == result.context.env_license.model_dump(mode="json")
    assert _in_process(_call(), execution=_profile("alpha"), store=store).reused is ReusePolicy.EXACT
    [reuse] = [a.node() for a in _activities(store) if "senselab:Reuse" in a.types()]
    assert prov.literal_value(reuse["senselab:storedAgent"]) == agent.id


@pytest.mark.logic
@pytest.mark.usefixtures("licensed")
def test_only_a_users_license_policy_refuses_a_call(store: ResultStore) -> None:
    """With no policy a call runs; a policy refusing an id of the env's or the model's license refuses it first."""
    for policy, subject in [
        (LicensePolicy(refuse=frozenset({"MPL-2.0"})), "fake-alpha"),
        (LicensePolicy(allow=frozenset({"MIT"})), "fake-alpha"),
        (LicensePolicy(allow=frozenset({"apache-2.0", "mpl-2.0"})), "model"),
    ]:
        with pytest.raises(LicenseRefused, match=subject):
            _in_process(_call(), execution=_profile("alpha"), store=store, run=RunConfig(licenses=policy))
    assert _activities(store) == []
    allowed = RunConfig(licenses=LicensePolicy(allow=frozenset({"MIT", "Apache-2.0", "MPL-2.0"})))
    assert _in_process(_call(), execution=_profile("alpha"), store=store, run=allowed).reused is None


@pytest.mark.logic
@pytest.mark.usefixtures("licensed")
def test_a_policy_never_serves_a_result_from_an_env_it_refuses(tmp_path: Path) -> None:
    """Under `tolerant` a stored result from a GPL env is served, unless the run's policy refuses GPL."""
    here = _in_process(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    gpl = ExecutionContext.model_validate(
        {**here.context.model_dump(), "numerics": GPU, "env_license": _record("GPL-3.0-only")}
    )
    no_gpl = LicensePolicy(refuse=frozenset({"GPL-3.0-only"}))
    for policy, reused in [(None, ReusePolicy.TOLERANT), (no_gpl, None)]:
        store = ResultStore(tmp_path / str(policy))
        store.put(here.key, {"value": "from a GPL env"}, gpl)
        run = RunConfig(ReusePolicy.TOLERANT, licenses=policy)
        assert _in_process(_call("v"), execution=_profile("alpha"), store=store, run=run).reused is reused


def _stored_from(tmp_path: Path, name: str, env_license: License | None) -> tuple[Result, ResultStore]:
    """A store holding the toy call's result computed in this context but with `env_license` recorded."""
    here = _in_process(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    stored = ExecutionContext.model_validate({**here.context.model_dump(), "env_license": env_license})
    store = ResultStore(tmp_path / name)
    store.put(here.key, {"value": f"stored {name}"}, stored)
    return here, store


@pytest.mark.logic
def test_policy_refuses_tainted_hit(tmp_path: Path) -> None:
    """A stored result whose env is licensed as the policy refuses is recomputed, not served; others are served."""
    _, store = _stored_from(tmp_path, "nc", _record("CC-BY-NC-4.0"))
    assert _in_process(_call("v"), execution=_profile("alpha"), store=store).reused is ReusePolicy.EXACT
    _, store = _stored_from(tmp_path, "nc-refused", _record("CC-BY-NC-4.0"))
    run = RunConfig(licenses=LicensePolicy(refuse=frozenset({"CC-BY-NC-4.0"})))
    result = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=run)
    assert result.reused is None and result.outputs != {"value": "stored nc-refused"}


@pytest.mark.logic
def test_hit_without_env_license_refused_by_default(tmp_path: Path) -> None:
    """A stored result whose context records no env license is recomputed under no policy and under a plain one."""
    for i, run in enumerate((RunConfig(), RunConfig(licenses=LicensePolicy()))):
        _, store = _stored_from(tmp_path / str(i), "unlicensed", None)
        result = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=run)
        assert result.reused is None and result.context.env_license is not None
        assert [a for a in _activities(store) if "senselab:Reuse" in a.types()] == []


@pytest.mark.logic
def test_policy_allowing_unlicensed_hit_is_recorded(tmp_path: Path) -> None:
    """A policy allowing unlicensed hits serves one and its reuse records the allowance; a licensed hit records none."""
    _, store = _stored_from(tmp_path, "unlicensed", None)
    run = RunConfig(ReusePolicy.TOLERANT, licenses=LicensePolicy(allow_unlicensed_hits=True))
    served = _in_process(_call("v"), execution=_profile("alpha"), store=store, run=run)
    assert served.reused is ReusePolicy.TOLERANT and served.outputs == {"value": "stored unlicensed"}
    [reuse] = [a.node() for a in _activities(store) if "senselab:Reuse" in a.types()]
    assert prov.literal_value(reuse["senselab:unlicensedReuse"]) is True
    _, licensed = _stored_from(tmp_path / "l", "licensed", _record("MIT"))
    assert _in_process(_call("v"), execution=_profile("alpha"), store=licensed, run=run).reused is ReusePolicy.EXACT
    [reuse] = [a.node() for a in _activities(licensed) if "senselab:Reuse" in a.types()]
    assert "senselab:unlicensedReuse" not in reuse


@pytest.mark.logic
def test_recomputed_unlicensed_hit_is_served_next_time(tmp_path: Path) -> None:
    """A result with no recorded env license is recomputed once; the licensed recomputation serves the next call."""
    _, store = _stored_from(tmp_path, "unlicensed", None)
    first = _in_process(_call("v"), execution=_profile("alpha"), store=store)
    assert first.reused is None and first.outputs != {"value": "stored unlicensed"}
    second = _in_process(_call("v"), execution=_profile("alpha"), store=store)
    assert second.reused is ReusePolicy.EXACT and second.outputs == first.outputs


@pytest.mark.logic
def test_lost_race_records_the_unlicensed_allowance(tmp_path: Path) -> None:
    """An entry with no env license another writer committed is served only when allowed, and then flagged."""
    here, _ = _stored_from(tmp_path, "unused", None)
    unlicensed = ExecutionContext.model_validate({**here.context.model_dump(), "env_license": None})
    store = ResultStore(tmp_path / "race")
    store.put(here.key, {"value": "the other writer's"}, unlicensed)
    allowed = RunConfig(licenses=LicensePolicy(allow_unlicensed_hits=True))
    job = Job(JobRef(plan_hash="0" * 64, node="call"), Call(TASK, fake.EchoParams(), {}), {})
    args = (store, here.key, unlicensed)
    served = lost_race(*args, allowed, verify_attributes(allowed), job, here.placement)
    assert served is not None and served.outputs == {"value": "the other writer's"}
    [reuse] = [a.node() for a in _activities(store) if "senselab:Reuse" in a.types()]
    assert prov.literal_value(reuse["senselab:unlicensedReuse"]) is True
    assert lost_race(*args, RunConfig(), verify_attributes(RunConfig()), job, here.placement) is None
    use_packaged(dataclasses.replace(packaged_now(), lock_records=None))
    computed = _in_process(_call("race"), execution=_profile("alpha"), store=_BeatenToCommit(tmp_path / "refused"))
    assert computed.reused is None and computed.outputs != {"value": "the other writer's"}


@pytest.mark.logic
def test_fragment_is_a_document_and_results_reference_each_job(store: ResultStore) -> None:
    """A one-node plan's fragment round-trips as JSON; under its admission, each job's result is referenced."""
    executor = InProcess(store=store, profile=_profile("alpha"), workers=WORKERS)
    placed = plan(_call("doc"), config=executor.run, execution=executor.profile)
    admission = executor.admit(placed, {bind_input("text", "json", "doc")[0]: "doc"})
    fragment = node_fragment(placed, admission)
    assert PlanFragment.model_validate_json(fragment.model_dump_json()) == fragment
    assert fragment.jobs == (JobRef(plan_hash=placed.hash, node="call", element=(0,)),)
    handle = executor.submit(fragment)
    assert executor.status(handle) is Status.DONE
    [ref] = executor.results(handle)
    [outcome] = executor.outcomes(handle)
    assert isinstance(outcome, Completed) and ref.job == fragment.jobs[0] and ref.failure is None
    assert ref.output_identity == outcome.result.output_identity
    assert ref.key_digest == outcome.result.key.digest()
    executor.release(admission)
    with pytest.raises(ValueError, match="not held"):
        executor.submit(fragment)


@pytest.mark.logic
def test_two_admissions_of_one_plan_keep_their_own_values(store: ResultStore) -> None:
    """One content admitted twice under two anchors: each admission's job is served on its own anchor's clock."""
    standalone = Audio.from_array(numpy.zeros(1600, dtype=numpy.float32), 16000)
    session = Audio(groups=standalone.groups, anchor=TimeAnchor(clock_id="session-7", absolute_start_ns=5))
    executor = InProcess(store=store, profile=_profile(), workers=WORKERS)
    call = TaskCall.model_validate({"task": CLOCKED_TASK.name, "inputs": {"audio": standalone}})
    placed = plan(call, config=executor.run, execution=executor.profile)
    first = executor.admit(placed, {standalone.identity: session})
    second = executor.admit(placed, {standalone.identity: standalone})
    clocks = []
    for admission in (first, second):
        [outcome] = executor.outcomes(executor.submit(node_fragment(placed, admission)))
        assert isinstance(outcome, Completed)
        spans = outcome.result.outputs["spans"]
        clocks.append(spans.clock_id if isinstance(spans, Intervals) else None)
    assert clocks == ["session-7", standalone.identity]
    executor.release(first)
    [outcome] = executor.outcomes(executor.submit(node_fragment(placed, second)))
    assert isinstance(outcome, Completed)


@pytest.mark.logic
def test_two_node_graph_run_by_hand_reads_upstream_outputs_from_the_store(store: ResultStore) -> None:
    """Node A is submitted, then node B with A's output identities, which `execute` reads from the store."""
    executor = InProcess(store=store, profile=_profile("alpha"), workers=WORKERS)
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=TASK.name, task_version="1"), Node(id="b", task=CHAIN_TASK.name, task_version="1")),
        edges=(
            Edge(source="input.text", target="a.text", kind="map"),
            Edge(source="a.value", target="b.value", kind="plain"),
        ),
        outputs={"value": "b.value"},
    )
    records = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate(("one", "two"))]
    placed = plan(graph, RecordSet(records), config=executor.run, execution=executor.profile)
    admission = executor.admit(placed, {bind_input("text", "json", t)[0]: t for t in ("one", "two")})
    with pytest.raises(ValueError, match="give its results"):
        node_fragment(placed, admission, "b")
    upstream = executor.submit(node_fragment(placed, admission, "a"))
    refs = list(executor.results(upstream))
    fragment = node_fragment(placed, admission, "b", upstream={"a": refs})
    assert PlanFragment.model_validate_json(fragment.model_dump_json()) == fragment
    stored = {port_output_identity(r.key_digest, "value"): (r.key_digest, r.output_identity, "value") for r in refs}
    assert {i: (u.key_digest, u.outputs, u.port) for i, u in fragment.upstream.items()} == stored
    downstream = executor.outcomes(executor.submit(fragment))
    computed = [o.result for o in executor.outcomes(upstream) if isinstance(o, Completed)]
    served = [o.result for o in downstream if isinstance(o, Completed)]
    assert [r.outputs["value"] for r in served] == [str(r.outputs["value"]) for r in computed]
    assert [r.key.inputs["value"] for r in served] == [port_output_identity(r.key.digest(), "value") for r in computed]
    executor.release(admission)


@pytest.mark.logic
def test_two_profiles_in_one_process_load_one_model_once(store: ResultStore) -> None:
    """Profiles of two bounds share the process's worker set and model cache: the model is loaded once."""
    loads = importlib.import_module("backends.counting_backend_fixture").LOADS
    loads.clear()
    small = _profile(stack_preferences={fake.ECHO.name: ["counting"]})
    large = _profile(
        stack_preferences={fake.ECHO.name: ["counting"]}, resource_classes={"gpu": {"device": "cuda", "max_workers": 9}}
    )
    workers = process_workers(small)
    assert process_workers(large) is workers
    assert workers.bound <= sum(c.max_workers for c in small.resource_classes.values())
    for i, (profile, held) in enumerate(((small, None), (large, None), (large, WorkerSet(1)))):
        call = TaskCall.model_validate({"task": fake.ECHO.name, "inputs": {"payload": {"i": i}}})
        result = InProcess(store=store, profile=profile, workers=held).call(call)
        assert result.placement.backend.name == "counting" and result.reused is None
    assert len(loads) == 1


@pytest.mark.logic
def test_an_output_identity_names_one_port(store: ResultStore) -> None:
    """Two graphs feed one downstream port from the two ports of one upstream job: two keys, each port's own value."""
    executor = InProcess(store=store, profile=_profile(), workers=WORKERS)
    records = RecordSet([Record(id="r", inputs={"text": ValueInput(value="v")})])
    served = {}
    for port in ("x", "y"):
        graph = Graph(
            inputs={"text": "list[json]"},
            nodes=(
                Node(id="a", task=TWO_PORTS_TASK.name, task_version="1"),
                Node(id="b", task=CHAIN_TASK.name, task_version="1"),
            ),
            edges=(
                Edge(source="input.text", target="a.text", kind="map"),
                Edge(source=f"a.{port}", target="b.value", kind="plain"),
            ),
            outputs={"value": "b.value"},
        )
        placed = plan(graph, records, config=executor.run, execution=executor.profile)
        admission = executor.admit(placed, {bind_input("text", "json", "v")[0]: "v"})
        [upstream] = executor.results(executor.submit(node_fragment(placed, admission, "a")))
        [outcome] = executor.outcomes(
            executor.submit(node_fragment(placed, admission, "b", upstream={"a": [upstream]}))
        )
        assert isinstance(outcome, Completed)
        assert outcome.result.key.inputs["value"] == port_output_identity(upstream.key_digest, port)
        served[port] = outcome.result
        executor.release(admission)
    assert served["x"].key != served["y"].key
    assert (served["x"].outputs["value"], served["y"].outputs["value"]) == ("x:v", "y:v")
