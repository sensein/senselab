"""`execute`: requests as the backend's batch invariance allows, keys of the request sent, failures never stored."""

import importlib
import platform
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import pytest
import soundfile
from pydantic import ValidationError
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ReusePolicy
from senselab_core.runtime.audio_io import open_audio
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, UpstreamRef
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.job import Completed, Failed, Job, JobOutcome, Upstream, execute
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.fake_backend import EchoParams
from senselab_core.runtime.keys import Call, Placement, request_size
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import Batched, ClassMeasure, Measured, ResourceRow, ResourceTable
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.workers import mit_lock_records
from senselab_worker.faults import ContentFault, FaultClass, InfrastructureFault, ResourceFault, error_record

TASK = TaskSpec(
    name="engine_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=EchoParams,
)
CHAIN_TASK = TASK.model_copy(update={"name": "engine_test.chain", "inputs": {"value": "sha256"}})
AUDIO_TASK = TASK.model_copy(update={"name": "engine_test.identify", "inputs": {"audio": "Audio"}})
SIZES = "backends.sizes_backend_fixture"
_TYPED = {
    FaultClass.CONTENT: ContentFault,
    FaultClass.RESOURCE: ResourceFault,
    FaultClass.INFRASTRUCTURE: InfrastructureFault,
}


def _backend(name: str, task: TaskSpec = TASK, **fields: Any) -> BackendSpec:  # noqa: ANN401 - BackendSpec fields
    return BackendSpec(
        **{
            "task": task.name,
            "name": name,
            "version": "1",
            "stack": name,
            "env": Env.host(),
            "devices": ("cpu",),
            "batch_invariant": True,
            "worker": SIZES,
            "load_stub": {},
            **fields,
        }
    )


INVARIANT = _backend("invariant")
BATCHED = _backend("batched", batch_invariant=False, output_affecting=("batch_formation",))
IDENTIFY = _backend("identify", AUDIO_TASK)
CHAIN = _backend("chain", CHAIN_TASK)
REGISTRY = Registry()
for _task in (TASK, AUDIO_TASK, CHAIN_TASK):
    REGISTRY.add_task(_task)
for _backend_spec in (INVARIANT, BATCHED, IDENTIFY, CHAIN):
    REGISTRY.add_backend(_backend_spec)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(source_tree_python(), platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """The engine reads this module's registry, opens every env as this interpreter, locked to MIT code."""
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    yield
    use_packaged(previous)


@pytest.fixture
def sizes() -> list[int]:
    """The sizes of the requests the fixture backend is sent in this test."""
    recorded: list[int] = importlib.import_module(SIZES).SIZES
    recorded.clear()
    return recorded


def _jobs(*texts: str, task: TaskSpec = TASK) -> list[Job]:
    jobs = []
    for i, text in enumerate(texts):
        port = next(iter(task.inputs))
        identity, value, _ = bind_input(port, task.inputs[port], text)
        ref = JobRef(plan_hash="0" * 64, node="n", element=(i,))
        jobs.append(Job(ref, Call(task, EchoParams(), {port: identity}), {port: value}))
    return jobs


def _execute(jobs: list[Job], backend: BackendSpec, store: ResultStore, batch_size: int = 1) -> list[JobOutcome]:
    placement = Placement(backend=backend, backend_params=backend.params(), device="cpu", batch_size=batch_size)
    workers = WorkerSet(1)
    try:
        return execute(jobs, placement, store=store, run=RunConfig(), profile=ExecutionProfile.load(), workers=workers)
    finally:
        workers.close()


def _completed(outcomes: list[JobOutcome]) -> list[Completed]:
    assert all(isinstance(o, Completed) for o in outcomes), outcomes
    return [o for o in outcomes if isinstance(o, Completed)]


@pytest.mark.logic
def test_request_components_match_request_sent(tmp_path: Path, sizes: list[int]) -> None:
    """An invariant backend is sent requests of the batch size; each result names the formation it was sent in."""
    outcomes = _completed(_execute(_jobs("a", "b", "c"), INVARIANT, ResultStore(tmp_path), batch_size=2))
    assert sizes == [2, 1]
    assert {o.result.placement.batch_formation for o in outcomes} == {"mixed"}
    assert {o.result.key.batch_formation for o in outcomes} == {None}
    assert (request_size(INVARIANT, 5), request_size(BATCHED, 5)) == (5, 5)
    with pytest.raises(ValueError, match="at least one item"):
        request_size(INVARIANT, 0)


@pytest.mark.logic
def test_failure_never_stored(tmp_path: Path, sizes: list[int]) -> None:
    """A failed item is `Failed` beside its request's completed items; nothing of it is stored or recorded."""
    store = ResultStore(tmp_path)
    ok, failed = _execute(_jobs("a", "fail"), INVARIANT, store, batch_size=2)
    assert sizes == [2] and isinstance(ok, Completed) and isinstance(failed, Failed)
    assert (failed.failure.fault, failed.failure.type, failed.failure.charged) == ("content", "ContentFault", True)
    [activity] = store.provenance.activities()
    assert activity.node()["prov:generated"] == [{"@id": ok.result.output_identity}]
    again = _execute(_jobs("a", "fail"), INVARIANT, store, batch_size=2)
    assert isinstance(again[0], Completed) and again[0].result.reused is ReusePolicy.EXACT
    assert isinstance(again[1], Failed) and sizes == [2, 1]


@pytest.mark.logic
@pytest.mark.parametrize("fault", list(FaultClass))
def test_gap_fault_record_never_charged(fault: FaultClass) -> None:
    """Only a typed content fault is charged: a resource, an infrastructure or a gap fault never is."""
    typed = FaultRecord.of(error_record(_TYPED[fault]("typed")))
    gap = FaultRecord.of({"type": "ValueError", "fault": fault.value, "message": "untyped", "gap": True})
    assert (typed.fault_class, typed.gap, typed.charged) == (fault, False, fault is FaultClass.CONTENT)
    assert (gap.gap, gap.charged) == (True, False)
    with pytest.raises(ValidationError, match="only a content fault"):
        FaultRecord(fault=fault, type="X", message="m", gap=True, charged=True)
    if fault is not FaultClass.CONTENT:
        with pytest.raises(ValidationError, match="only a content fault"):
            FaultRecord(fault=fault, type="X", message="m", gap=False, charged=True)


@pytest.mark.logic
def test_an_untyped_failure_is_a_gap_and_not_charged(tmp_path: Path, sizes: list[int]) -> None:
    """An item that fails with an untyped exception is an infrastructure gap, never charged."""
    [failed] = _execute(_jobs("gap"), INVARIANT, ResultStore(tmp_path))
    assert isinstance(failed, Failed) and (failed.failure.fault, failed.failure.gap) == ("infrastructure", True)
    assert not failed.failure.charged and sizes == [1]


@pytest.mark.logic
def test_rewritten_input_is_stale_at_run(tmp_path: Path, sizes: list[int]) -> None:
    """A file-backed input rewritten after it was opened fails its job before anything is sent or stored."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, numpy.zeros(160, dtype=numpy.float32), 16000)
    audio = open_audio(path)
    ref = JobRef(plan_hash="0" * 64, node="n", element=(0,))
    jobs = [Job(ref, Call(AUDIO_TASK, EchoParams(), {"audio": audio.identity}), {"audio": audio})]
    soundfile.write(path, numpy.ones(160, dtype=numpy.float32) * 0.5, 16000)
    store = ResultStore(tmp_path / "store")
    [failed] = _execute(jobs, IDENTIFY, store)
    assert isinstance(failed, Failed) and failed.failure.type == "StaleSource"
    assert (failed.failure.fault_class, failed.failure.gap, failed.failure.charged) == (FaultClass.CONTENT, False, True)
    assert sizes == [] and store.provenance.activities() == []
    path.unlink()
    [gone] = _execute(jobs, IDENTIFY, store)
    assert isinstance(gone, Failed) and (gone.failure.type, gone.failure.fault) == ("StaleSource", "content")


@pytest.mark.logic
def test_a_bad_upstream_fails_only_its_element(tmp_path: Path, sizes: list[int]) -> None:
    """Of three siblings, one reads a port its entry lacks and one an entry the store lacks: only those two fail."""
    store = ResultStore(tmp_path)
    [upstream] = _completed(_execute(_jobs("a"), INVARIANT, store))
    key, outputs = upstream.result.key.digest(), upstream.result.output_identity
    refs = [
        UpstreamRef(key_digest=key, outputs=outputs, port="value"),
        UpstreamRef(key_digest=key, outputs=outputs, port="missing"),
        UpstreamRef(key_digest=key, outputs="output:" + "0" * 64, port="value"),
    ]
    jobs = [
        Job(
            JobRef(plan_hash="0" * 64, node="b", element=(i,)),
            Call(CHAIN_TASK, EchoParams(), {"value": ref.identity()}),
            {},
            upstream={"value": Upstream((ref,), TASK.name, "sha256")},
        )
        for i, ref in enumerate(refs)
    ]
    sizes.clear()
    good, wrong, missing = _execute(jobs, CHAIN, store, batch_size=3)
    assert isinstance(good, Completed) and good.result.outputs["value"] == upstream.result.outputs["value"]
    assert isinstance(wrong, Failed) and (wrong.failure.fault, wrong.failure.gap, wrong.failure.charged) == (
        "content",
        False,
        True,
    )
    assert isinstance(missing, Failed) and missing.failure.type == "UpstreamMissing"
    assert (missing.failure.fault, missing.failure.gap, missing.failure.charged) == ("infrastructure", True, False)
    assert sizes == [1]


@pytest.mark.logic
def test_a_retired_upstream_entry_fails_only_its_element_as_a_gap(tmp_path: Path, sizes: list[int]) -> None:
    """Of two siblings, one reads an upstream entry stored in a retired wire format: it fails as a gap, uncharged."""
    store = ResultStore(tmp_path)
    [upstream] = _completed(_execute(_jobs("a"), INVARIANT, store))
    key = LogicalKey(
        task=TASK.name,
        task_version="1",
        backend="b",
        backend_version="1",
        model=None,
        params={},
        backend_params={},
        inputs={"text": "content:" + "1" * 64},
    )
    context = ExecutionContext(
        env_lock="lock",
        env_license=None,
        platform="linux",
        wheel_set="wheels",
        numerics={"device": "cpu", "precision": "fp32"},
        decoder=None,
        driver={},
        host="node1",
        protocol="p1",
        model_source="hf",
    )
    old = store.put(key, {"value": {"format": "metric@1.0"}}, context)
    refs = [
        UpstreamRef(key_digest=upstream.result.key.digest(), outputs=upstream.result.output_identity, port="value"),
        UpstreamRef(key_digest=key.digest(), outputs=old, port="value"),
    ]
    jobs = [
        Job(
            JobRef(plan_hash="0" * 64, node="b", element=(i,)),
            Call(CHAIN_TASK, EchoParams(), {"value": ref.identity()}),
            {},
            upstream={"value": Upstream((ref,), TASK.name, "sha256" if i == 0 else "Metric")},
        )
        for i, ref in enumerate(refs)
    ]
    sizes.clear()
    good, retired = _execute(jobs, CHAIN, store, batch_size=2)
    assert isinstance(good, Completed)
    assert isinstance(retired, Failed) and retired.failure.type == "WireFormatRetired"
    assert (retired.failure.fault, retired.failure.gap, retired.failure.charged) == ("infrastructure", True, False)
    assert sizes == [1]


@pytest.mark.logic
def test_in_process_runs_a_unit_with_packaged_rows_in_batches(tmp_path: Path, sizes: list[int]) -> None:
    """Jobs of a unit with a packaged resource row are sent in the batch `size` gives, not one to a request."""
    measure = ClassMeasure(
        load_s=1.0,
        claim_s=0.0,
        resident_bytes=1024,
        peak_memory_bytes=1280,
        s_per_input_s=0.1,
        input_s_per_item=1.0,
        batched=Batched(batch_size=4, s_per_input_s=0.05),
        measured=Measured(sha="1" * 40, jobs=("1",)),
    )
    row = ResourceRow(id=None, revision=None, download_bytes=None, classes={"cpu": measure})
    table = ResourceTable(task=AUDIO_TASK.name, backend="identify", rows=(row,))
    use_packaged(
        Packaged(Catalogue(entries=(), resource_tables=(table,)), _this_interpreter, REGISTRY, mit_lock_records)
    )
    jobs = []
    for i in range(6):
        path = tmp_path / f"tone{i}.wav"
        soundfile.write(path, numpy.zeros(16000, dtype=numpy.float32), 16000)
        audio = open_audio(path)
        ref = JobRef(plan_hash="0" * 64, node="n", element=(i,))
        jobs.append(Job(ref, Call(AUDIO_TASK, EchoParams(), {"audio": audio.identity}), {"audio": audio}))
    profile = ExecutionProfile.load("laptop", {"resource_classes": {"cpu": {"mem_bytes": 10_000}}})
    placement = Placement(backend=IDENTIFY, backend_params=IDENTIFY.params(), device="cpu")
    workers = WorkerSet(1)
    try:
        executor = InProcess(store=ResultStore(tmp_path / "store"), profile=profile, run=RunConfig(), workers=workers)
        outcomes = executor._run(jobs, placement)  # noqa: SLF001 - the run `submit` makes of a fragment's jobs
    finally:
        workers.close()
    assert len(_completed(outcomes)) == 6 and sizes == [4, 2]
