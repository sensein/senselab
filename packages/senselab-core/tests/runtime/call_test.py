"""`run_call`: placement by profile, store reuse, one activity per call, model source and profile digest recorded."""

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
from senselab_core.kernel import prov
from senselab_core.kernel._base import Frozen
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.identity import (
    NO_EQUIVALENCE,
    Equivalence,
    ExecutionContext,
    LogicalKey,
    ModelRef,
    OutputIdentity,
    ReusePolicy,
    member_key,
)
from senselab_core.kernel.license import License
from senselab_core.kernel.results import Intervals, Metric, Transcript, TranscriptNode, Trials
from senselab_core.kernel.signals import Audio, PlainText
from senselab_core.kernel.time import Span, TimeAnchor
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime.audio_io import DECODER
from senselab_core.runtime.call import (
    AmbiguousClock,
    CallFailed,
    OutputTypeMismatch,
    Packaged,
    Result,
    RunConfig,
    TaskCall,
    run_call,
    use_packaged,
)
from senselab_core.runtime.call import packaged as packaged_now
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.compare import Thresholds
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.equivalence import SAMPLING
from senselab_core.runtime.licenses import LicensePolicy, LicenseRefused, license_record
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
for _task in (TASK, AUDIO_TASK, PAIR_TASK, CLOCKED_TASK, TRANSCRIBE_TASK, TRIALS_TASK, OVERLAP_TASK, fake.ECHO):
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

CATALOGUE = Catalogue.load(
    f"""
format: catalogue@1.0
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
    result = run_call(_call(), execution=_profile("batched"), store=store)
    assert (result.key.batch_formation, result.key.chunk_grid) == ("single", "whole-item")


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
    assert prov.literal_value(agent.node()["senselab:envLock"]) == result.context.env_lock
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
    results = [run_call(c, execution=profile, store=store) for c in calls]
    call_module.close_workers()
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
        return run_call(call, execution=profile, store=store)

    fresh, hit = transcribe(transcript, "a"), transcribe(transcript, "a")
    assert fresh.outputs == hit.outputs == {"transcript": transcript} and hit.reused is not None
    assert fresh.key.inputs == {"text": PlainText(content="a").identity}
    with pytest.raises(OutputTypeMismatch, match="metric@2.0"):
        transcribe(Metric(name="wer", value=0.1, unit="1", items=1), "b")
    call_module.close_workers()
    assert len(_executions(store)) == 1


@pytest.mark.logic
def test_array_outputs_are_stored_and_read_back(store: ResultStore) -> None:
    """A host backend's `Trials` output is stored with its arrays and read back equal from the store."""
    call = TaskCall.model_validate({"task": TRIALS_TASK.name, "inputs": {"text": "pairs"}})
    fresh, hit = (run_call(call, execution=_profile(), store=store) for _ in range(2))
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
            run_call(TaskCall.model_validate({"task": OVERLAP_TASK.name, "inputs": {"label": text}}), store=store)
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


@pytest.mark.logic
def test_rebound_equals_reclocked_compute(store: ResultStore, tmp_path: Path) -> None:
    """One content under a session anchor, then standalone, gets spans on each reader's clock from one computation.

    Each served result equals the one computed afresh on that reader's recording.
    """
    standalone = Audio.from_array(numpy.zeros(1600, dtype=numpy.float32), 16000)
    session = Audio(groups=standalone.groups, anchor=TimeAnchor(clock_id="session-7", absolute_start_ns=5))

    def detect(audio: Audio, into: ResultStore = store) -> Result:
        call = TaskCall.model_validate({"task": CLOCKED_TASK.name, "inputs": {"audio": audio}})
        return run_call(call, execution=_profile(), store=into)

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
        run_call(call, execution=_profile(), store=store)
    assert _activities(store) == []


GPU = {"device_kind": "cuda", "precision": "default", "gpu_architecture": "sm_90"}


def _elsewhere(tmp_path: Path, value: object, equivalence: Equivalence = NO_EQUIVALENCE) -> tuple[Result, ResultStore]:
    """Return a call's result here, and a store holding `value` for its key computed on a GPU instead."""
    here = run_call(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
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
    assert run_call(_call("v"), execution=_profile("alpha"), store=store).reused is None
    _, measured = _elsewhere(tmp_path / "m", "from the gpu", _class(here))
    served = run_call(_call("v"), execution=_profile("alpha"), store=measured)
    assert served.reused is ReusePolicy.EQUIVALENT and served.outputs == {"value": "from the gpu"}


@pytest.mark.logic
def test_another_env_lock_is_served_only_when_the_run_asks_for_what_is_available(tmp_path: Path) -> None:
    """`exact` and `equivalent` refuse another env lock's unmeasured result; `tolerant` serves it, naming both agents.

    The `equivalent` store holds a class measured within the caller's lock, which does not reach the other lock.
    """
    here = run_call(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    other = ExecutionContext.model_validate({**here.context.model_dump(), "env_lock": "another-lock"})
    for policy in ReusePolicy:
        store = ResultStore(tmp_path / policy.value, equivalence=_class(here))
        store.put(here.key, {"value": "from another lock"}, other)
        served = run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(policy))
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
    served = run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT))
    assert served.reused is ReusePolicy.TOLERANT and store.evidence() == []
    _, store = _elsewhere(tmp_path / "t", here.outputs["value"])
    verified = run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT, 1.0))
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
    result = run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(verify=1.0))
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
    here = run_call(call, execution=_profile(), store=here_store)
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
        assert run_call(call, execution=_profile(), store=store, run=run).reused is None
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
    run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT, 1.0))
    [execution] = _activities(store)
    [evidence] = store.evidence()
    node = execution.node()
    assert "senselab:Reuse" not in execution.types()
    assert prov.literal_value(node["senselab:verifies"]) == evidence.canonical_digest()
    assert prov.literal_value(node["senselab:verifyFraction"]) == 1.0
    assert prov.literal_value(node["senselab:verifySampling"]) == SAMPLING
    _, store = _elsewhere(tmp_path / "t", here.outputs["value"])
    run_call(_call("v"), execution=_profile("alpha"), store=store, run=RunConfig(ReusePolicy.TOLERANT, 0.0))
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
    result = run_call(_call(), execution=_profile("alpha"), store=store)
    assert result.context.env_license is not None and result.context.env_license.expression == "Apache-2.0 AND MPL-2.0"
    [agent] = store.provenance.agents()
    assert prov.literal_value(agent.node()["senselab:license"]) == result.context.env_license.model_dump(mode="json")
    assert run_call(_call(), execution=_profile("alpha"), store=store).reused is ReusePolicy.EXACT
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
            run_call(_call(), execution=_profile("alpha"), store=store, run=RunConfig(licenses=policy))
    assert _activities(store) == []
    allowed = RunConfig(licenses=LicensePolicy(allow=frozenset({"MIT", "Apache-2.0", "MPL-2.0"})))
    assert run_call(_call(), execution=_profile("alpha"), store=store, run=allowed).reused is None


@pytest.mark.logic
@pytest.mark.usefixtures("licensed")
def test_a_policy_never_serves_a_result_from_an_env_it_refuses(tmp_path: Path) -> None:
    """Under `tolerant` a stored result from a GPL env is served, unless the run's policy refuses GPL."""
    here = run_call(_call("v"), execution=_profile("alpha"), store=ResultStore(tmp_path / "here"))
    gpl = ExecutionContext.model_validate(
        {**here.context.model_dump(), "numerics": GPU, "env_license": _record("GPL-3.0-only")}
    )
    no_gpl = LicensePolicy(refuse=frozenset({"GPL-3.0-only"}))
    for policy, reused in [(None, ReusePolicy.TOLERANT), (no_gpl, None)]:
        store = ResultStore(tmp_path / str(policy))
        store.put(here.key, {"value": "from a GPL env"}, gpl)
        run = RunConfig(ReusePolicy.TOLERANT, licenses=policy)
        assert run_call(_call("v"), execution=_profile("alpha"), store=store, run=run).reused is reused
