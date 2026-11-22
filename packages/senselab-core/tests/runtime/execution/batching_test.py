"""Cross-record batching: a job's formation and key are its own, whatever jobs share its batch."""

import importlib
import platform
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy
import pytest
import soundfile
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import ExecutionContext, OutputIdentity
from senselab_core.kernel.signals import Audio, PlainText
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.batching import (
    BucketGrid,
    form_batches,
    formation,
    job_length_ns,
    job_text_chars,
)
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.fragments import job_key, plan_keys
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.fake_backend import EchoParams
from senselab_core.runtime.keys import MIXED, SINGLE, Call, Placement, build_key, request_components
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.contracts import BASE_CONTEXT
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.workers import mit_lock_records
from senselab_worker.units import NS_PER_S

AUDIO_TASK = TaskSpec(
    name="batching_test.identify",
    version="1",
    kind="transform",
    inputs={"audio": "Audio"},
    outputs={"value": "sha256"},
    params=EchoParams,
)
TEXT_TASK = AUDIO_TASK.model_copy(update={"name": "batching_test.score", "inputs": {"text": "json"}})
PASS_TASK = AUDIO_TASK.model_copy(update={"name": "batching_test.pass", "outputs": {"audio": "Audio"}})
SIZES = "backends.sizes_backend_fixture"
RATE = 100
GRID = BucketGrid.load_default()


def _backend(name: str, task: TaskSpec = AUDIO_TASK, **fields: Any) -> BackendSpec:  # noqa: ANN401 - BackendSpec fields
    defaults = {"version": "1", "stack": name, "env": Env.host(), "devices": ("cpu",), "worker": SIZES}
    return BackendSpec(
        **{"task": task.name, "name": name, "batch_invariant": True, "load_stub": {}, **defaults, **fields}
    )


INVARIANT = _backend("invariant")
BUCKETED = _backend("bucketed", batch_invariant=False, output_affecting=("batch_formation",))
UNKEYED = _backend("unkeyed", batch_invariant=False)
PASS = _backend("pass", PASS_TASK, worker="backends.echo_inputs_backend_fixture")
TEXT_BUCKETED = _backend("text_bucketed", TEXT_TASK, batch_invariant=False, output_affecting=("batch_formation",))
REGISTRY = Registry()
for _task in (AUDIO_TASK, TEXT_TASK, PASS_TASK):
    REGISTRY.add_task(_task)
for _spec in (INVARIANT, BUCKETED, UNKEYED, TEXT_BUCKETED, PASS):
    REGISTRY.add_backend(_spec)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile, device: str) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
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


def _audio(length_ns: int, seed: int) -> Audio:
    frames = max(1, length_ns * RATE // NS_PER_S)
    return Audio.from_array(numpy.full(frames, seed / (seed + 1), dtype=numpy.float32), RATE)


def _jobs(lengths: list[int], node: str = "n", plan: str = "0" * 64, task: TaskSpec = AUDIO_TASK) -> list[Job]:
    jobs = []
    for i, length in enumerate(lengths):
        audio = _audio(length, i)
        ref = JobRef(plan_hash=plan, node=node, element=(i,))
        jobs.append(Job(ref, Call(task, EchoParams(), {"audio": audio.identity}), {"audio": audio}))
    return jobs


def _text_jobs(*texts: str) -> list[Job]:
    jobs = []
    for i, text in enumerate(texts):
        identity, value, _ = bind_input("text", "json", text)
        ref = JobRef(plan_hash="0" * 64, node="n", element=(i,))
        jobs.append(Job(ref, Call(TEXT_TASK, EchoParams(), {"text": identity}), {"text": value}))
    return jobs


SHORT, LONG = GRID.edges_ns[0] // 2, GRID.edges_ns[-1]


class _CountingStore(ResultStore):
    """A result store that counts the stored outputs read through it."""

    reads = 0

    def outputs_of(self, key_digest: str, identity: OutputIdentity) -> dict[str, object] | None:
        """Count the read, then read."""
        self.reads += 1
        return super().outputs_of(key_digest, identity)


@pytest.mark.logic
def test_plan_keys_a_bucketed_node_on_graph_inputs_only(tmp_path: Path) -> None:
    """Before a run, a bucketed node on graph inputs is keyed by their lengths; one on upstream audio is not."""
    paths = []
    for i, length in enumerate((SHORT, LONG)):
        paths.append(tmp_path / f"r{i}.wav")
        soundfile.write(paths[-1], _audio(length, i).read()[0], RATE)
    graph = Graph(
        inputs={"audio": "list[Audio]"},
        nodes=(
            Node(id="pass", task=PASS_TASK.name, task_version="1"),
            Node(id="direct", task=AUDIO_TASK.name, task_version="1"),
            Node(id="after", task=AUDIO_TASK.name, task_version="1"),
        ),
        edges=(
            Edge(source="input.audio", target="pass.audio", kind="map"),
            Edge(source="input.audio", target="direct.audio", kind="map"),
            Edge(source="pass.audio", target="after.audio"),
        ),
        outputs={"direct": "direct.value", "after": "after.value"},
    )
    rows = [Record(id=f"r{i}", inputs={"audio": ValueInput(value=str(p))}) for i, p in enumerate(paths)]
    profile = ExecutionProfile.load("laptop", {"stack_preferences": {AUDIO_TASK.name: ["bucketed"]}})
    placed = plan(graph, RecordSet(rows), config=RunConfig(), execution=profile)
    values = {identity: value for identity, value, _ in (bind_input("audio", "Audio", str(p)) for p in paths)}
    keys = plan_keys(placed, values)
    assert {n for n, _ in keys} == {"pass", "direct"}
    calls = {(0,): SHORT, (1,): LONG}
    for element, length in calls.items():
        inputs = {"audio": placed.manifest.rows[element[0]].inputs["audio"]}
        expected = job_key(placed, "direct", inputs, formation=GRID.formation(length)).digest()
        assert keys[("direct", element)] == expected


@pytest.mark.logic
def test_bucket_in_key_changes_key() -> None:
    """Another bucket, or another grid, is another key; a formation the backend is not sent in is refused."""
    job = _jobs([SHORT])[0]
    context = ExecutionContext.model_validate(BASE_CONTEXT)
    base = Placement(backend=BUCKETED, backend_params=BUCKETED.params(), device="cpu")

    def key(formation: str) -> str:
        return build_key(job.call, replace(base, **request_components(BUCKETED, formation)), context).digest()

    other_grid = GRID.model_copy(update={"id": "other"})
    keys = {key(GRID.formation(SHORT)), key(GRID.formation(LONG)), key(other_grid.formation(SHORT)), key(SINGLE)}
    assert len(keys) == 4
    with pytest.raises(ValueError, match="formation 'mixed'"):
        request_components(BUCKETED, MIXED)
    with pytest.raises(ValueError, match="formation 'bucket:"):
        request_components(UNKEYED, GRID.formation(SHORT))
    with pytest.raises(ValueError, match="formation 'single'"):
        request_components(INVARIANT, SINGLE)


@pytest.mark.logic
def test_batches_never_cross_nodes() -> None:
    """Jobs of two nodes, or of two plans, are refused; a batch size below 1 is refused."""
    with pytest.raises(ValueError, match="one node of one plan"):
        form_batches(_jobs([SHORT]) + _jobs([SHORT], node="m"), INVARIANT, batch_size=2, grid=GRID)
    with pytest.raises(ValueError, match="one node of one plan"):
        form_batches(_jobs([SHORT]) + _jobs([SHORT], plan="1" * 64), INVARIANT, batch_size=2, grid=GRID)
    with pytest.raises(ValueError, match="at least one job"):
        form_batches(_jobs([SHORT]), INVARIANT, batch_size=0, grid=GRID)


@pytest.mark.logic
@settings(max_examples=30, deadline=None)
@given(
    lengths=st.lists(st.integers(min_value=1, max_value=2 * GRID.edges_ns[-1]), min_size=1, max_size=12),
    batch_size=st.integers(min_value=1, max_value=5),
    backend=st.sampled_from([INVARIANT, BUCKETED, UNKEYED]),
)
def test_batches_hold_each_job_once_under_its_own_formation(
    lengths: list[int], batch_size: int, backend: BackendSpec
) -> None:
    """Every job is in one batch of at most the batch size, whose formation is the job's own; `single` holds one."""
    jobs = _jobs(lengths)
    batches = form_batches(jobs, backend, batch_size=batch_size, grid=GRID)
    placed = [(job.ref, b.formation) for b in batches for job in b.jobs]
    assert sorted(r.element for r, _ in placed) == [j.ref.element for j in jobs]
    assert all(f == formation(j, backend, GRID) for j in jobs for r, f in placed if r == j.ref)
    assert all(len(b.jobs) <= (1 if b.formation == SINGLE else batch_size) for b in batches)


@pytest.mark.logic
def test_grid_buckets_by_edges_and_refuses_bad_edges() -> None:
    """A length at an edge opens the next bucket; edges that are empty, not positive or not increasing are refused."""
    first, second = GRID.edges_ns[:2]
    assert [GRID.formation(n) for n in (first - 1, first, second - 1, second)] == [
        f"bucket:{GRID.id}:{i}" for i in (0, 1, 1, 2)
    ]
    for edges in ((), (0,), (-first,), (second, first), (first, first)):
        with pytest.raises(ValidationError):
            BucketGrid(id="bad", edges_ns=edges)
    with pytest.raises(ValidationError):
        BucketGrid(id="", edges_ns=(first,))


@pytest.mark.logic
def test_a_held_texts_characters_are_its_input_size() -> None:
    """A job holding text gives its code points and no recording length; a job holding audio gives no characters."""
    text = PlainText(content="na\u00efve caf\u00e9")
    job = Job(
        JobRef(plan_hash="0" * 64, node="n", element=(0,)),
        Call(TEXT_TASK, EchoParams(), {"text": text.identity}),
        {"text": text},
    )
    assert (job_text_chars(job), job_length_ns(job)) == (10, None)
    [audio] = _jobs([SHORT])
    assert job_text_chars(audio) is None
