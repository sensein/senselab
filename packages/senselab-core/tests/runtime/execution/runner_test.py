"""`PydraRunner`: a placed plan run through an `Executor`, batched, per element, keyed by senselab (fake stack)."""

import dataclasses
import platform
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.kernel.identity import InputIdentity, ReusePolicy, gather_identity, port_output_identity
from senselab_core.kernel.license import License
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.executor import Admission, Handle, PlanFragment, ResultRef, Status
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.interpreter import source_tree_python

runner = pytest.importorskip("senselab_core.runtime.execution.runner", exc_type=ImportError)

SCORE = TaskSpec(
    name="runner_test.score",
    version="1",
    kind="transform",
    inputs={"text": "json"},
    outputs={"value": "sha256"},
    params=TaskParams,
)
CHAIN = SCORE.model_copy(update={"name": "runner_test.chain", "inputs": {"value": "sha256"}})
GATHER = SCORE.model_copy(update={"name": "runner_test.gather", "inputs": {"values": "list[sha256]"}})
PAIR = SCORE.model_copy(update={"name": "runner_test.pair", "inputs": {"left": "sha256", "right": "json"}})

REGISTRY = Registry()
for _task in (SCORE, CHAIN, GATHER, PAIR):
    REGISTRY.add_task(_task)
    REGISTRY.add_backend(
        BackendSpec(
            task=_task.name,
            name=_task.name.split(".")[1],
            version="1",
            stack="dsp",
            env=Env.host(),
            devices=("cpu",),
            batch_invariant=True,
            worker="backends.sizes_backend_fixture",
            load_stub={},
        )
    )
CATALOGUE = Catalogue.load("format: catalogue@2.0\nentries: []\n")
PROFILE = ExecutionProfile.load("laptop", {"devices": ["cpu"]})
WORKERS = WorkerSet(4)


def _license() -> License:
    return license_record("MIT", source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def packaged() -> Iterator[None]:
    """The runner's executor reads this module's registry and opens every env as this interpreter, MIT-locked."""
    records = {"pkg": _license()}
    previous = use_packaged(Packaged(CATALOGUE, _this_interpreter, REGISTRY, lambda _: records))
    yield
    use_packaged(previous)
    WORKERS.close()


class Recording:
    """An `Executor` over `InProcess` that records each fragment it is given, in order."""

    def __init__(self, store: ResultStore) -> None:
        """Run on an `InProcess` executor over `store`."""
        self.inner = InProcess(store=store, profile=PROFILE, run=RunConfig(), workers=WORKERS)
        self.fragments: list[PlanFragment] = []

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:
        """Admit through `InProcess`."""
        return self.inner.admit(placed, values)

    def release(self, admission: Admission) -> None:
        """Release through `InProcess`."""
        self.inner.release(admission)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Record `fragment` and submit it."""
        self.fragments.append(fragment)
        return self.inner.submit(fragment)

    def status(self, handle: Handle) -> Status:
        """Return the status `InProcess` gives."""
        return self.inner.status(handle)

    def cancel(self, handle: Handle) -> None:
        """Cancel through `InProcess`."""
        self.inner.cancel(handle)

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield the results `InProcess` gives."""
        return self.inner.results(handle)

    def sizes(self, node: str) -> list[int]:
        """Return the number of jobs of each fragment of `node`, in submission order."""
        return [len(f.jobs) for f in self.fragments if f.node == node]


def _graph(nodes: Sequence[tuple[str, TaskSpec]], edges: Sequence[tuple[str, str, str]], **inputs: str) -> Graph:
    return Graph(
        inputs=inputs,
        nodes=tuple(Node(id=n, task=t.name, task_version="1") for n, t in nodes),
        edges=tuple(Edge(source=s, target=t, kind=k) for s, t, k in edges),
    )


def _plan(graph: Graph, rows: Sequence[Mapping[str, str]], **broadcast: str) -> tuple[Plan, dict[str, object]]:
    """Return `graph` planned over records of `rows`, and the admitted value of each input identity."""
    records = [Record(id=f"r{i}", inputs={k: ValueInput(value=v) for k, v in r.items()}) for i, r in enumerate(rows)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=PROFILE, inputs=broadcast or None)
    given = [*(v for r in rows for v in r.values()), *broadcast.values()]
    return placed, {bind_input("x", "json", v)[0]: v for v in given}


def _batched(placed: Plan, size: int) -> Plan:
    nodes = {n: p.model_copy(update={"batch_size": size}) for n, p in placed.placement.nodes.items()}
    return dataclasses.replace(placed, placement=placed.placement.model_copy(update={"nodes": nodes}))


def _run(placed: Plan, values: Mapping[str, object], tmp_path: Path, **kw: object) -> tuple[Any, Recording, list[Any]]:
    executor = Recording(ResultStore(tmp_path / "store"))
    events: list[Any] = []
    work = tmp_path / "pydra"
    status = runner.PydraRunner(profile=PROFILE, work_dir=work).run_plan(
        placed, executor, values=values, events=events, **kw
    )
    return status, executor, events


CHAINED = [("a", SCORE), ("b", CHAIN)]
GATHERED = [("a", SCORE), ("g", GATHER)]
GATHER_EDGES = [("input.text", "a.text", "map"), ("a.value", "g.values", "gather")]
CHAIN_EDGES = [("input.text", "a.text", "map"), ("a.value", "b.value", "plain")]


@pytest.mark.contract
def test_map_node_submits_one_fragment_per_batch(tmp_path: Path) -> None:
    """Five records at batch size 2: each mapped node is sent as fragments of 2, 2 and 1 jobs, every element once."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": f"t{i}"} for i in range(5)])
    status, executor, _ = _run(_batched(placed, 2), values, tmp_path)
    assert executor.sizes("a") == [2, 2, 1]
    assert sum(executor.sizes("b")) == 5 and max(executor.sizes("b")) <= 2
    assert status.done and [r.job.element for r in status.results["b"]] == [(i,) for i in range(5)]


@pytest.mark.contract
def test_errored_element_fails_only_its_descendants(tmp_path: Path) -> None:
    """Element 1 of `a` fails: its descendants in `b`, `d` and the gather are blocked; `c` and the rest run."""
    graph = _graph(
        [*CHAINED, ("c", SCORE), ("d", PAIR), ("g", GATHER)],
        [
            *CHAIN_EDGES,
            ("input.other", "c.text", "map"),
            ("a.value", "d.left", "plain"),
            ("input.other", "d.right", "map"),
            ("a.value", "g.values", "gather"),
        ],
        text="list[json]",
        other="list[json]",
    )
    rows = [{"text": t, "other": o} for t, o in (("one", "x"), ("fail", "y"), ("two", "z"))]
    placed, values = _plan(graph, rows)
    status, _, events = _run(placed, values, tmp_path)
    [failed] = [r for r in status.results["a"] if r.failure is not None]
    assert failed.job.element == (1,) and failed.failure is not None and failed.failure.charged
    assert [r.job.element for r in status.results["b"]] == [(0,), (2,)] and status.blocked["b"] == ((1,),)
    assert len(status.results["c"]) == 3 and status.blocked["c"] == ()
    assert status.blocked["d"] == ((1,),) and status.blocked["g"] == ((),) and not status.done
    blocked = {(e.job.node, e.job.element) for e in events if e.kind == "blocked"}
    assert blocked == {("b", (1,)), ("d", (1,)), ("g", ())}


@pytest.mark.contract
def test_record_bound_inputs_zip_and_a_broadcast_input_reaches_every_element(tmp_path: Path) -> None:
    """`d` reads its record's `other` beside `a`'s same element; the broadcast `both` is one identity everywhere."""
    graph = _graph(
        [("a", SCORE), ("d", PAIR), ("e", PAIR)],
        [
            ("input.text", "a.text", "map"),
            ("a.value", "d.left", "plain"),
            ("input.other", "d.right", "map"),
            ("a.value", "e.left", "plain"),
            ("input.both", "e.right", "plain"),
        ],
        text="list[json]",
        other="list[json]",
        both="json",
    )
    placed, values = _plan(graph, [{"text": f"t{i}", "other": f"o{i}"} for i in range(3)], both="shared")
    status, executor, _ = _run(placed, values, tmp_path)
    assert status.done and len(status.results["d"]) == 3
    inputs = {(f.node, j.element): i for f in executor.fragments for j, i in zip(f.jobs, f.inputs, strict=True)}
    rows = placed.manifest.rows
    assert all(inputs[("d", (i,))]["right"] == rows[i].inputs["other"] for i in range(3))
    assert {inputs[("e", (i,))]["right"] for i in range(3)} == {rows[0].inputs["both"]}


@pytest.mark.contract
def test_gather_waits_for_every_element(tmp_path: Path) -> None:
    """The gather's one fragment is sent after every element of `a`, and reads their ports in element order."""
    graph = _graph(GATHERED, GATHER_EDGES, text="list[json]")
    placed, values = _plan(graph, [{"text": t} for t in ("one", "two", "three")])
    status, executor, _ = _run(placed, values, tmp_path)
    assert [f.node for f in executor.fragments][-1] == "g" and executor.sizes("g") == [1]
    ports = [port_output_identity(r.key_digest, "value") for r in status.results["a"]]
    [fragment] = [f for f in executor.fragments if f.node == "g"]
    assert fragment.inputs[0]["values"] == gather_identity(ports)
    assert len([f for f in executor.fragments if f.node == "a"]) == 3


@pytest.mark.contract
def test_an_empty_map_gathers_to_the_empty_list(tmp_path: Path) -> None:
    """No records: the mapped node runs no job, and the gathering node runs once over `gather:` of no element."""
    graph = _graph(GATHERED, GATHER_EDGES, text="list[json]")
    placed, values = _plan(graph, [])
    status, executor, _ = _run(placed, values, tmp_path)
    assert status.results["a"] == () and executor.sizes("a") == []
    [fragment] = executor.fragments
    assert fragment.inputs[0]["values"] == gather_identity([]) and status.done


@pytest.mark.contract
def test_pydra_checksum_equals_logical_key(tmp_path: Path) -> None:
    """Every leaf job pydra recorded is checksummed by the key digest the executor computed for it."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    store = runner.SenselabStore(tmp_path / "records", runner.SenselabReuse(ReusePolicy(placed.logical.reuse)))
    status, _, _ = _run(placed, values, tmp_path, store=store)
    keys = {r.key_digest for refs in status.results.values() for r in refs}
    leaves = {c for c in store.checksums() if not c.startswith(("records-", "plan-"))}
    assert len(keys) == 4 and leaves == keys


@pytest.mark.contract
def test_no_result_pickle_written(tmp_path: Path) -> None:
    """A run writes no `_result.pklz`: every result is in senselab's store and pydra's records are in memory."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    status, _, _ = _run(placed, values, tmp_path)
    assert status.done and list(tmp_path.rglob("_result.pklz")) == []


@pytest.mark.contract
def test_inprocess_and_pool_give_same_store(tmp_path: Path) -> None:
    """The runner over `InProcess` stores the keys and outputs a node-by-node run by hand stores."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    status, _, _ = _run(placed, values, tmp_path / "run")
    by_hand = InProcess(store=ResultStore(tmp_path / "hand"), profile=PROFILE, run=RunConfig(), workers=WORKERS)
    admission = by_hand.admit(placed, values)
    upstream = list(by_hand.results(by_hand.submit(node_fragment(placed, admission, "a"))))
    downstream = list(by_hand.results(by_hand.submit(node_fragment(placed, admission, "b", upstream={"a": upstream}))))
    by_hand.release(admission)
    ran = {(r.key_digest, r.output_identity) for refs in status.results.values() for r in refs}
    assert ran == {(r.key_digest, r.output_identity) for r in (*upstream, *downstream)}


class Refusing(Recording):
    """A `Recording` executor whose `submit` raises for node `b`."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Raise for node `b`; submit any other fragment."""
        if fragment.node == "b":
            raise LookupError("b is refused")
        return super().submit(fragment)


@pytest.mark.contract
def test_an_executor_error_is_raised_after_the_run(tmp_path: Path) -> None:
    """An error the executor raises is not an element's failure: the run raises it once every other job ran."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    executor = Refusing(ResultStore(tmp_path / "store"))
    with pytest.raises(LookupError, match="refused"):
        runner.PydraRunner(profile=PROFILE, work_dir=tmp_path / "pydra").run_plan(
            placed, executor, values=values, events=[]
        )
    assert executor.sizes("a") == [1, 1]
