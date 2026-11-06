"""`resume`: a run started again by its plan hash under a fresh placement, and the plan its activities name."""

import ast
import dataclasses
import inspect
import json
import platform
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy
import pytest
import soundfile
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.signals import StaleSource
from senselab_core.runtime import planner
from senselab_core.runtime.audio_io import open_audio
from senselab_core.runtime.call import Packaged, RunConfig, TaskCall, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.events import Event, EventLog, read_events, run_dir_of
from senselab_core.runtime.execution.executor import Admission, Handle, PlanFragment, ResultRef, Status
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.run import ConflictingSources, InputChanged, Output, Run, load_plan
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import NoRun, Plan, plan, resume
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.revision import Resolver
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.interpreter import source_tree_python

pytest.importorskip("pydra")
pytestmark = pytest.mark.contract

SCORE = TaskSpec(
    name="resume_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=TaskParams,
)
HEAR = SCORE.model_copy(update={"name": "resume_test.hear", "inputs": {"audio": "Audio"}})
PIN, MOVED = "a" * 40, "c" * 40


def _backend(task: TaskSpec, stack: str) -> BackendSpec:
    return BackendSpec(
        task=task.name,
        name=stack,
        version="1",
        stack=stack,
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=True,
        worker="backends.sizes_backend_fixture",
        load_stub={},
    )


REGISTRY = Registry()
for _task in (SCORE, HEAR):
    REGISTRY.add_task(_task)
    REGISTRY.add_backend(_backend(_task, "dsp"))
REGISTRY.add_backend(_backend(SCORE, "alt"))
CATALOGUE = Catalogue.load("format: catalogue@2.0\nentries: []\n")
BASE = {"devices": ["cpu"], "model_sources": ["toyorg"]}
PROFILE = ExecutionProfile.load("laptop", BASE)
SAME_BACKEND = ExecutionProfile.load("laptop", {**BASE, "model_cache_budget_bytes": 2**30})
OFFLINE = ExecutionProfile.load("laptop", {**BASE, "network": "offline"})
OTHER_BACKEND = ExecutionProfile.load("laptop", {**BASE, "stack_preferences": {SCORE.name: ["alt"]}})
WORKERS = WorkerSet(4)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry, every env this interpreter and MIT-locked; runs under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    mit = license_record("MIT", source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)
    previous = use_packaged(Packaged(CATALOGUE, _this_interpreter, REGISTRY, lambda _: {"pkg": mit}))
    yield
    use_packaged(previous)
    WORKERS.close()


class Stopping(InProcess):
    """An `InProcess` executor that runs the first `k` fragments it is given and raises on every later one."""

    def __init__(self, store: ResultStore, k: int) -> None:
        """Run `k` fragments over `store` under `PROFILE`."""
        super().__init__(store=store, profile=PROFILE, run=RunConfig(), workers=WORKERS)
        self.left = k
        self.guard = threading.Lock()

    def submit(self, fragment: PlanFragment) -> Handle:
        """Submit while fragments are left; then raise, as a process stopped at the job boundary would."""
        with self.guard:
            self.left -= 1
            go = self.left >= 0
        if not go:
            raise RuntimeError("planted stop at the job boundary")
        return super().submit(fragment)


def _executor(store: ResultStore, profile: ExecutionProfile = PROFILE) -> InProcess:
    return InProcess(store=store, profile=profile, run=RunConfig(), workers=WORKERS)


def _graph(task: TaskSpec = SCORE, port: str = "text", model: str | None = None) -> Graph:
    return Graph(
        inputs={port: f"list[{task.inputs[port]}]"},
        nodes=(Node(id="a", task=task.name, task_version="1", model=model),),
        edges=(Edge(source=f"input.{port}", target=f"a.{port}", kind="map"),),
        outputs={"value": "a.value"},
    )


def _plan(n: int, **kw: Any) -> Plan:  # noqa: ANN401 - `plan` keywords
    rows = from_records({"id": f"r{i}", "inputs": {"text": f"t{i}"}} for i in range(n))
    placed = plan(kw.pop("graph", _graph()), rows, config=RunConfig(), execution=PROFILE, **kw)
    nodes = {k: v.model_copy(update={"batch_size": 1}) for k, v in placed.placement.nodes.items()}
    return dataclasses.replace(placed, placement=placed.placement.model_copy(update={"nodes": nodes}))


def _ran(placed: Plan, store: ResultStore) -> Run:
    run = placed.run(_executor(store), store=store)
    assert run.wait(60).done
    return run


def _latest(run: Run) -> list[Event]:
    events = read_events(run.run_dir)
    last = max(e.ordinal for e in events)
    return [e for e in events if e.ordinal == last]


def _kinds(events: Sequence[Event]) -> list[str]:
    return sorted(e.kind for e in events if e.kind in {"finished", "cache_hit", "failed", "blocked", "cancelled"})


def _placement(run: Run) -> dict[str, Any]:
    attempt = _latest(run)[0].attempt
    found: dict[str, Any] = json.loads((run.run_dir / "placements" / f"{attempt}.json").read_text())
    return found


def test_resume_executes_only_missing(tmp_path: Path) -> None:
    """Stopped after 2 of 4 jobs, the resumed attempt reuses those 2 and executes the other 2, under one hash."""
    store = ResultStore(tmp_path / "store")
    placed = _plan(4)
    stopped = placed.run(Stopping(store, 2), store=store)
    with pytest.raises(RuntimeError, match="planted stop"):
        stopped.wait(60)
    assert _kinds(_latest(stopped)) == ["failed", "failed", "finished", "finished"]
    resumed = resume(placed.hash, execution=PROFILE, store=store, executor=_executor(store))
    assert resumed.wait(60).done and resumed.run_dir == stopped.run_dir
    assert _kinds(_latest(resumed)) == ["cache_hit", "cache_hit", "finished", "finished"]
    assert all(isinstance(o, Output) for _, o in resumed.results())
    assert sorted(p.name for p in (resumed.run_dir / "attempts").iterdir()) == ["1", "2"]
    with pytest.raises(NoRun, match="0" * 64):
        resume("0" * 64)


def _hub(answers: dict[str, str], seen: list[str]) -> Callable[[str, str, str | None], str]:
    def hub(repo: str, ref: str, token: str | None) -> str:  # noqa: ARG001 - a hub lookup
        seen.append(f"{repo}@{ref}")
        return answers[ref]

    return hub


def test_resume_never_reresolves_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planned through a ref, the run resumes offline with an empty hub cache: any resolution would raise."""
    store, seen = ResultStore(tmp_path / "store"), list[str]()
    resolver = Resolver(cache_dir=tmp_path / "hf-hub", hub=_hub({"main": PIN}, seen), cache_root=tmp_path / "pins")
    placed = _plan(2, graph=_graph(model="toyorg/m@main"), resolver=resolver)
    assert seen == ["toyorg/m@main"]
    _ran(placed, store)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    resumed = resume(placed.hash, execution=OFFLINE, store=store, executor=_executor(store, OFFLINE))
    assert resumed.wait(60).done and all(isinstance(o, Output) for _, o in resumed.results())
    model = resumed.plan.logical.models["a"]
    assert model is not None and model.identity == PIN and model.ref.ref == PIN and resumed.plan.hash == placed.hash
    assert _placement(resumed)["profile_digest"] == OFFLINE.canonical_digest()


def test_resume_and_place_name_no_resolver() -> None:
    """Neither `resume` nor `place` names a resolver, `pin_model` or `resolve`, so neither can resolve a ref."""
    tree = ast.parse(inspect.getsource(planner))
    bodies = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"resume", "place"}]
    nodes = [n for body in bodies for n in ast.walk(body)]
    named = {n.id for n in nodes if isinstance(n, ast.Name)} | {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    named |= {
        name
        for n in nodes
        if isinstance(n, ast.Import | ast.ImportFrom)
        for alias in n.names
        for name in (alias.name.rpartition(".")[2], alias.asname)
        if name is not None
    }
    assert len(bodies) == 2 and "place" in named
    assert not named & {"pin_model", "resolve", "resolve_bound", "PROCESS_RESOLVER", "Resolver", "_pinned"}


def test_resume_with_other_profile_same_backend_reuses(tmp_path: Path) -> None:
    """Placed afresh under another profile that picks the same backend, every job is served from the store."""
    store = ResultStore(tmp_path / "store")
    placed = _plan(3)
    _ran(placed, store)
    resumed = resume(placed.hash, execution=SAME_BACKEND, store=store, executor=_executor(store, SAME_BACKEND))
    assert resumed.wait(60).done and resumed.run_dir.name == placed.hash
    assert _kinds(_latest(resumed)) == ["cache_hit"] * 3
    fresh = _placement(resumed)
    assert fresh["profile_digest"] == SAME_BACKEND.canonical_digest() != PROFILE.canonical_digest()
    assert fresh["nodes"]["a"]["backend"] == "dsp"


def test_resume_with_other_backend_recomputes(tmp_path: Path) -> None:
    """A fresh placement on another backend keeps the plan hash; the backend is keyed, so the node is recomputed."""
    store = ResultStore(tmp_path / "store")
    placed = _plan(3)
    run = _ran(placed, store)
    written = (run.run_dir / "plan.json").read_bytes()
    assert "placement" not in json.loads(written)
    resumed = resume(placed.hash, execution=OTHER_BACKEND, store=store, executor=_executor(store, OTHER_BACKEND))
    assert resumed.wait(60).done and resumed.run_dir.name == placed.hash
    assert _kinds(_latest(resumed)) == ["finished"] * 3
    assert _placement(resumed)["nodes"]["a"]["backend"] == "alt" == load_plan(run.run_dir).placement.nodes["a"].backend
    assert placed.placement.nodes["a"].backend == "dsp"
    assert (run.run_dir / "plan.json").read_bytes() == written


def _wav(path: Path, value: float) -> Path:
    soundfile.write(path, numpy.full((800, 1), value, dtype=numpy.float32), 8000, subtype="FLOAT")
    return path


def _tree(run_dir: Path) -> list[tuple[str, bytes]]:
    return sorted((str(p.relative_to(run_dir)), p.read_bytes()) for p in run_dir.rglob("*") if p.is_file())


def test_resume_refuses_stale_manifest(tmp_path: Path) -> None:
    """An input file changed or gone since planning refuses the whole resume, naming its records; nothing is written."""
    store = ResultStore(tmp_path / "store")
    paths = [_wav(tmp_path / f"r{i}.wav", 0.1 * (i + 1)) for i in range(3)]
    rows = from_records({"id": f"r{i}", "inputs": {"audio": p}} for i, p in enumerate(paths))
    placed = plan(_graph(HEAR, "audio"), rows, config=RunConfig(), execution=PROFILE)
    run = _ran(placed, store)
    _wav(paths[0], 0.9)
    paths[2].unlink()
    before = _tree(run.run_dir)
    with pytest.raises(StaleSource) as raised:
        resume(placed.hash, execution=PROFILE, store=store, executor=_executor(store))
    assert isinstance(raised.value, InputChanged) and set(raised.value.changed) == {("r0", "audio"), ("r2", "audio")}
    assert "r0" in str(raised.value) and "r2" in str(raised.value)
    assert _tree(run.run_dir) == before


def test_activity_had_plan(tmp_path: Path) -> None:
    """Execution and reuse activities are associated through `prov:hadPlan` with the run's one `senselab:Plan`."""
    store = ResultStore(tmp_path / "store")
    placed = _plan(2)
    run = _ran(placed, store)
    resume(placed.hash, execution=PROFILE, store=store, executor=_executor(store)).wait(60)
    read = ProvStore.read(run.run_dir)
    [entity] = read.plans()
    assert entity.id == f"plan:{placed.hash}" and "senselab:Plan" in entity.types()
    activities = read.activities()
    assert {("senselab:Reuse" in a.types()) for a in activities} == {True, False}
    for activity in activities:
        [association] = cast("list[dict[str, Any]]", activity.node()["prov:qualifiedAssociation"])
        assert association["prov:hadPlan"] == {"@id": entity.id}
        assert association["prov:hadRole"]["senselab:nodeId"] == "a"


class Delegating:
    """An `Executor` that is not `InProcess`: it hands every call to one it holds."""

    logs_claims: ClassVar[bool] = False

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Bind through `inner`."""
        self.inner.bind_epoch(epoch, events)

    def __init__(self, inner: InProcess) -> None:
        """Hold `inner`."""
        self.inner = inner

    def admit(self, placed: Plan, values: Mapping[str, object]) -> Admission:
        """Admit through `inner`."""
        return self.inner.admit(placed, values)

    def release(self, admission: Admission) -> None:
        """Release through `inner`."""
        self.inner.release(admission)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Submit through `inner`."""
        return self.inner.submit(fragment)

    def status(self, handle: Handle) -> Status:
        """Return the status `inner` gives."""
        return self.inner.status(handle)

    def cancel(self, handle: Handle) -> None:
        """Cancel through `inner`."""
        self.inner.cancel(handle)

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield the results `inner` gives."""
        return self.inner.results(handle)


def _linked_to_one_plan(provenance: ProvStore, plan_hash: str) -> None:
    [entity] = provenance.plans()
    assert entity.id == f"plan:{plan_hash}" and provenance.activities()
    for activity in provenance.activities():
        [association] = cast("list[dict[str, Any]]", activity.node()["prov:qualifiedAssociation"])
        assert association["prov:hadPlan"] == {"@id": entity.id}


def test_the_run_writes_its_plan_entity_for_any_executor(tmp_path: Path) -> None:
    """Two attempts on an executor that is not `InProcess` give one plan entity, which every activity names."""
    store = ResultStore(tmp_path / "store")
    placed = _plan(2)
    executor = Delegating(_executor(store))
    executor.admit(placed, {})
    assert store.provenance.plans() == []
    for _ in range(2):
        assert placed.run(executor, store=store).wait(60).done
    _linked_to_one_plan(ProvStore.read(run_dir_of(placed.hash)), placed.hash)


def test_a_single_call_writes_its_plan_entity(tmp_path: Path) -> None:
    """`InProcess.call` plans its call itself, and adds that plan's entity for its activity to name."""
    store = ResultStore(tmp_path / "store")
    call = TaskCall.model_validate({"task": SCORE.name, "inputs": {"text": "x"}}, context={"registry": REGISTRY})
    result = _executor(store).call(call)
    assert result.output_identity is not None
    _linked_to_one_plan(store.provenance, store.provenance.plans()[0].id.partition(":")[2])


FRESH = """
import sys
from senselab_core.runtime.execution.run import bind_inputs
print(len(bind_inputs(sys.argv[1])))
"""


def _heard(source: object) -> Plan:
    call = TaskCall.model_validate({"task": HEAR.name, "inputs": {"audio": source}}, context={"registry": REGISTRY})
    return plan(call, config=RunConfig(), execution=PROFILE)


def _sources(run: Run) -> object:
    return json.loads((run.run_dir / "inputs.json").read_text())["rows"][0]["audio"]


def test_a_rerun_in_memory_keeps_the_path_and_another_process_binds_from_it(tmp_path: Path) -> None:
    """Run from a path, then from the same audio held in memory: `inputs.json` keeps the path, read elsewhere."""
    store, path = ResultStore(tmp_path / "store"), _wav(tmp_path / "a.wav", 0.25)
    first = _ran(_heard(path), store)
    second = _ran(_heard(open_audio(path)), store)
    assert second.run_dir == first.run_dir and _sources(second) == str(path)
    found = subprocess.run(
        [sys.executable, "-c", FRESH, str(first.run_dir)], capture_output=True, text=True, check=True
    )
    assert found.stdout.strip() == "1"


def test_a_path_run_upgrades_a_memory_source(tmp_path: Path) -> None:
    """Run first from audio held in memory, then from its path: the recorded source becomes the path."""
    store, path = ResultStore(tmp_path / "store"), _wav(tmp_path / "a.wav", 0.25)
    first = _ran(_heard(open_audio(path)), store)
    assert set(cast("dict[str, Any]", _sources(first))) == {"memory"}
    assert _sources(_ran(_heard(path), store)) == str(path)


def test_conflicting_durable_sources_are_refused_before_any_write(tmp_path: Path) -> None:
    """A recorded path that now reads other audio than a new path of the plan's audio is refused; nothing is written."""
    store, recorded = ResultStore(tmp_path / "store"), _wav(tmp_path / "a.wav", 0.25)
    run = _ran(_heard(recorded), store)
    moved = tmp_path / "b.wav"
    moved.write_bytes(recorded.read_bytes())
    _wav(recorded, 0.5)
    placed = _heard(moved)
    assert placed.hash == run.run_dir.name
    before = _tree(run.run_dir)
    with pytest.raises(ConflictingSources) as raised:
        placed.run(_executor(store), store=store)
    assert raised.value.conflicts == (("call", "audio"),)
    assert _tree(run.run_dir) == before
