"""`PydraRunner`: a placed plan run through an `Executor`, batched, per element, keyed by senselab (fake stack)."""

import asyncio
import concurrent.futures
import dataclasses
import os
import platform
import stat
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pytest
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node, fan_out_issues
from senselab_core.kernel.identity import InputIdentity, gather_identity, port_output_identity
from senselab_core.kernel.license import License
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.events import TERMINAL, EventLog, read_events, status_from
from senselab_core.runtime.execution.executor import Admission, Handle, PlanFragment, ResultRef, Status, follow
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.licenses import license_record
from senselab_core.runtime.planner import Plan, UnmappedOutput, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.interpreter import source_tree_python

T = TypeVar("T")

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
TWIN = SCORE.model_copy(update={"name": "runner_test.twin", "inputs": {"first": "json", "second": "json"}})

REGISTRY = Registry()
for _task in (SCORE, CHAIN, GATHER, PAIR, TWIN):
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

    logs_claims: ClassVar[bool] = False
    splits_fragments: ClassVar[bool] = False

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Bind through `InProcess`."""
        self.inner.bind_epoch(epoch, events)

    def __init__(self, store: ResultStore, profile: ExecutionProfile = PROFILE) -> None:
        """Run on an `InProcess` executor over `store` under `profile`."""
        self.store = store
        self.inner = InProcess(store=store, profile=profile, run=RunConfig(), workers=WORKERS)
        self.fragments: list[PlanFragment] = []
        self.telemetry: list[str | None] = []

    def admit(self, placed: Plan, values: Mapping[InputIdentity, object]) -> Admission:
        """Admit through `InProcess`."""
        return self.inner.admit(placed, values)

    def release(self, admission: Admission) -> None:
        """Release through `InProcess`."""
        self.inner.release(admission)

    def submit(self, fragment: PlanFragment) -> Handle:
        """Record `fragment` and the telemetry opt-out it is submitted under, and submit it."""
        self.fragments.append(fragment)
        self.telemetry.append(os.environ.get("NO_ET"))
        return self.inner.submit(fragment)

    def status(self, handle: Handle) -> Status:
        """Return the status `InProcess` gives."""
        return self.inner.status(handle)

    def cancel(self, handle: Handle) -> None:
        """Cancel through `InProcess`."""
        self.inner.cancel(handle)

    def poll(self, handle: Handle) -> tuple[list[ResultRef], bool]:
        """Poll through `InProcess`."""
        return self.inner.poll(handle)

    def sizes(self, node: str) -> list[int]:
        """Return the number of jobs of each fragment of `node`, in submission order."""
        return [len(f.jobs) for f in self.fragments if f.node == node]


def _graph(nodes: Sequence[tuple[str, TaskSpec]], edges: Sequence[tuple[str, str, str]], **inputs: str) -> Graph:
    return Graph(
        inputs=inputs,
        nodes=tuple(Node(id=n, task=t.name, task_version="1") for n, t in nodes),
        edges=tuple(Edge(source=s, target=t, kind=k) for s, t, k in edges),
    )


def _plan(
    graph: Graph,
    rows: Sequence[Mapping[str, str]],
    broadcast: Mapping[str, object] | None = None,
    profile: ExecutionProfile = PROFILE,
) -> tuple[Plan, dict[str, object]]:
    """Return `graph` planned over records of `rows` under `profile`, and the admitted value of each input identity."""
    records = [Record(id=f"r{i}", inputs={k: ValueInput(value=v) for k, v in r.items()}) for i, r in enumerate(rows)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=profile, inputs=broadcast or None)
    given = [*(v for r in rows for v in r.values()), *(broadcast or {}).values()]
    return placed, {bind_input("x", "json", v)[0]: v for v in given}


def _batched(placed: Plan, size: int) -> Plan:
    nodes = {n: p.model_copy(update={"batch_size": size}) for n, p in placed.placement.nodes.items()}
    return dataclasses.replace(placed, placement=placed.placement.model_copy(update={"nodes": nodes}))


def _run(
    placed: Plan, values: Mapping[str, object], tmp_path: Path, profile: ExecutionProfile = PROFILE, **kw: object
) -> tuple[Any, Recording, list[Any]]:
    executor = Recording(ResultStore(tmp_path / "store"), profile)
    events = EventLog(tmp_path / "run", "a1", 1)
    work = tmp_path / "pydra"
    status = runner.PydraRunner(profile=profile, work_dir=work).run_plan(
        placed, executor, values=values, events=events, **kw
    )
    return status, executor, read_events(tmp_path / "run")


CHAINED = [("a", SCORE), ("b", CHAIN)]
GATHERED = [("a", SCORE), ("g", GATHER)]
GATHER_EDGES = [("input.text", "a.text", "map"), ("a.value", "g.values", "gather")]
CHAIN_EDGES = [("input.text", "a.text", "map"), ("a.value", "b.value", "plain")]
ZIPPED = _graph(
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
ZIPPED_ROWS = [{"text": f"t{i}", "other": f"o{i}"} for i in range(3)]


@pytest.mark.contract
def test_map_node_submits_one_fragment_per_batch(tmp_path: Path) -> None:
    """Five records at batch size 2: each mapped node is sent as fragments of 2, 2 and 1 jobs, every element once."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": f"t{i}"} for i in range(5)])
    status, executor, _ = _run(_batched(placed, 2), values, tmp_path)
    assert sorted(executor.sizes("a")) == [1, 2, 2]
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
    assert {(e.node, e.job.element) for e in events if e.kind == "failed" and e.job} == {("a", (1,))}
    assert {(n, e) for n, elements in status.blocked.items() for e in elements} == {("b", (1,)), ("d", (1,)), ("g", ())}


class Finishing(asyncio.SelectorEventLoop):
    """An event loop whose `run_in_executor` runs the call at once and returns its future already done."""

    def run_in_executor(
        self, executor: concurrent.futures.Executor | None, func: Callable[..., T], *args: object
    ) -> asyncio.Future[T]:
        """Run `func` in this thread and return its finished future, as a pool that finished before it was awaited."""
        future: asyncio.Future[T] = self.create_future()
        try:
            future.set_result(func(*args))
        except BaseException as error:  # noqa: BLE001 - held by the future, as a pool thread holds it
            future.set_exception(error)
        return future


@pytest.mark.contract
def test_every_result_settles_its_own_element_when_the_fragment_ends_before_it_is_awaited(tmp_path: Path) -> None:
    """A fragment whose call has ended before the loop awaits it still gives each element its result, failure too."""
    placed, values = _plan(
        _graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "fail", "two")]
    )
    loop = Finishing()
    asyncio.set_event_loop(loop)
    try:
        status, _, events = _run(placed, values, tmp_path)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    assert [(r.job.element, r.failure is None) for r in status.results["a"]] == [
        ((0,), True),
        ((1,), False),
        ((2,), True),
    ]
    assert status.blocked["b"] == ((1,),)
    terminal = [(e.node, e.job.element) for e in events if e.kind in ("finished", "failed") and e.job]
    assert sorted(terminal) == [("a", (0,)), ("a", (1,)), ("a", (2,)), ("b", (0,)), ("b", (2,))]


@pytest.mark.contract
def test_record_bound_inputs_zip_and_a_broadcast_input_reaches_every_element(tmp_path: Path) -> None:
    """`d` reads its record's `other` beside `a`'s same element; the broadcast `both` is one identity everywhere."""
    placed, values = _plan(ZIPPED, ZIPPED_ROWS, {"both": "shared"})
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
@pytest.mark.parametrize(
    ("graph", "rows", "broadcast", "jobs"),
    [
        (_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}, {"text": "two"}], {}, 4),
        (_graph(GATHERED, GATHER_EDGES, text="list[json]"), [{"text": "one"}, {"text": "two"}], {}, 3),
        (ZIPPED, ZIPPED_ROWS, {"both": "shared"}, 9),
    ],
    ids=["plain", "gather", "zip_and_broadcast"],
)
def test_pydra_checksum_equals_logical_key(
    graph: Graph, rows: list[dict[str, str]], broadcast: dict[str, str], jobs: int, tmp_path: Path
) -> None:
    """Every leaf job pydra recorded, plain, gathered, zipped or broadcast, is keyed as the executor keyed it."""
    placed, values = _plan(graph, rows, broadcast)
    store = runner.SenselabStore(tmp_path / "records")
    status, _, _ = _run(placed, values, tmp_path, store=store)
    keys = {r.key_digest for refs in status.results.values() for r in refs}
    leaves = {c for c in store.checksums() if not c.startswith(("records-", "plan-"))}
    assert status.done and len(keys) == jobs and leaves == keys


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
    status, executor, _ = _run(placed, values, tmp_path / "run")
    by_hand = InProcess(store=ResultStore(tmp_path / "hand"), profile=PROFILE, run=RunConfig(), workers=WORKERS)
    admission = by_hand.admit(placed, values)
    upstream = list(follow(by_hand, by_hand.submit(node_fragment(placed, admission, "a"))))
    downstream = list(follow(by_hand, by_hand.submit(node_fragment(placed, admission, "b", upstream={"a": upstream}))))
    by_hand.release(admission)
    ran = {(r.key_digest, r.output_identity) for refs in status.results.values() for r in refs}
    assert ran == {(r.key_digest, r.output_identity) for r in (*upstream, *downstream)} and status.done
    for key, identity in ran:
        assert identity is not None
        stored = executor.inner.store.outputs_of(key, identity)
        assert stored is not None and stored == by_hand.store.outputs_of(key, identity)
    assert len(_entries(executor.inner.store.root)) == 4
    assert _entries(executor.inner.store.root) == _entries(by_hand.store.root)


def _entries(root: Path) -> set[str]:
    """Return the committed entries under a result store's root, as `<key digest>/<context fingerprint>`."""
    return {f"{p.parent.name}/{p.name}" for p in root.glob("*/*/*/*") if p.is_dir() and len(p.parent.name) == 64}


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
            placed, executor, values=values, events=EventLog(tmp_path / "events", "a1", 1)
        )
    assert executor.sizes("a") == [1, 1]


@pytest.mark.contract
def test_two_record_lists_zip_within_one_node(tmp_path: Path) -> None:
    """One node mapped over two record lists runs one job per record, each reading its own row of both."""
    graph = _graph(
        [("t", TWIN)],
        [("input.first", "t.first", "map"), ("input.second", "t.second", "map")],
        first="list[json]",
        second="list[json]",
    )
    placed, values = _plan(graph, [{"first": f"f{i}", "second": f"s{i}"} for i in range(3)])
    status, executor, _ = _run(placed, values, tmp_path)
    assert status.done and executor.sizes("t") == [1, 1, 1]
    inputs = {j.element: i for f in executor.fragments for j, i in zip(f.jobs, f.inputs, strict=True)}
    rows = placed.manifest.rows
    assert all(inputs[(i,)] == {"first": rows[i].inputs["first"], "second": rows[i].inputs["second"]} for i in range(3))


_STORE: Any = runner.SenselabStore


class LosingC(_STORE):  # type: ignore[misc]
    """A completion store that cannot record the results of node `c`."""

    def record(self, checksum: str, result: Any) -> None:  # noqa: ANN401 - a pydra result
        """Raise for a result of node `c`; record any other."""
        ref = getattr(result.outputs, "ref", None)
        if isinstance(ref, Mapping) and ref.get("job", {}).get("node") == "c":
            raise OSError("the record of c is lost")
        super().record(checksum, result)


@pytest.mark.contract
def test_an_unrelated_pydra_error_raises_beside_a_failed_element(tmp_path: Path) -> None:
    """Element 1 of `a` fails and, unrelated, every job of `c` errors in pydra: the run raises, not reports."""
    graph = _graph(
        [("a", SCORE), ("c", SCORE)],
        [("input.text", "a.text", "map"), ("input.other", "c.text", "map")],
        text="list[json]",
        other="list[json]",
    )
    placed, values = _plan(graph, [{"text": t, "other": f"o-{t}"} for t in ("one", "fail")])
    with pytest.raises(Exception, match="n_c") as raised:
        _run(placed, values, tmp_path, store=LosingC(tmp_path / "records"))
    assert type(raised.value).__name__ == "WorkflowError"
    status, _, _ = _run(placed, values, tmp_path / "again")
    assert status.blocked == {"a": (), "c": ()} and not status.done


@pytest.mark.contract
def test_a_run_makes_no_telemetry_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under an offline profile, pydra's version check is opted out for the run, and the opt-out is gone after."""
    monkeypatch.delenv("NO_ET", raising=False)
    offline = ExecutionProfile.load("laptop", {"devices": ["cpu"], "network": "offline"})
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}], profile=offline)
    status, executor, _ = _run(placed, values, tmp_path, profile=offline)
    assert status.done and executor.telemetry == ["1", "1"] and "NO_ET" not in os.environ


@pytest.mark.logic
@pytest.mark.parametrize(
    "extra", [Edge(source="input.text", target="g.extra", kind="map"), Edge(source="a.value", target="g.extra")]
)
def test_compile_plan_refuses_a_gather_into_a_mapped_node(extra: Edge) -> None:
    """A gather into a node mapped directly or through a plain edge: `compile_plan` refuses it as `validate` does."""
    placed, _ = _plan(_graph(GATHERED, GATHER_EDGES, text="list[json]"), [{"text": "one"}])
    nested = placed.logical.graph.model_copy(update={"edges": (*placed.logical.graph.edges, extra)})
    [issue] = fan_out_issues(nested)
    assert (issue.code, issue.node, issue.port) == ("nested_fan_out", "g", "values")
    unplanned = dataclasses.replace(placed, logical=placed.logical.model_copy(update={"graph": nested}))
    with pytest.raises(ValueError, match="runs once per element"):
        runner.compile_plan(unplanned)


@pytest.mark.logic
def test_the_pinned_pydra_takes_no_lease_settings() -> None:
    """The profile's job lease is held, not passed: the pinned pydra has no `LeaseSettings`; this fails once it does."""
    store = pytest.importorskip("pydra.engine.store")
    assert not hasattr(store, "LeaseSettings")
    assert runner.PydraRunner(profile=PROFILE).lease == PROFILE.lease


class Cancelling(Recording):
    """A `Recording` executor whose `submit` raises `asyncio.CancelledError` for node `b`."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Raise a cancellation for node `b`; submit any other fragment."""
        if fragment.node == "b":
            raise asyncio.CancelledError
        return super().submit(fragment)


@pytest.mark.contract
def test_a_cancellation_beside_a_failed_element_propagates(tmp_path: Path) -> None:
    """Element 1 of `a` fails and `b`'s submission is cancelled: the cancellation propagates, not a run status."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "fail")])
    executor = Cancelling(ResultStore(tmp_path / "store"))
    with pytest.raises(asyncio.CancelledError) as raised:
        runner.PydraRunner(profile=PROFILE, work_dir=tmp_path / "pydra").run_plan(
            placed, executor, values=values, events=EventLog(tmp_path / "events", "a1", 1)
        )
    assert type(raised.value.__cause__).__name__ == "WorkflowError" and executor.sizes("a") == [1, 1]


@pytest.mark.logic
def test_compile_plan_refuses_a_map_over_a_node_output() -> None:
    """A map over a node's list output is refused before pydra sees it, naming the node and the output."""
    placed, _ = _plan(_graph(GATHERED, GATHER_EDGES, text="list[json]"), [{"text": "one"}])
    graph = placed.logical.graph
    mapped = graph.model_copy(update={"edges": (*graph.edges, Edge(source="g.value", target="a.extra", kind="map"))})
    unplanned = dataclasses.replace(placed, logical=placed.logical.model_copy(update={"graph": mapped}))
    with pytest.raises(UnmappedOutput, match="has no identity yet") as raised:
        runner.compile_plan(unplanned)
    assert (raised.value.node, raised.value.source) == ("a", "g.value")


@pytest.mark.contract
def test_a_run_logs_each_job_queued_started_and_finished(tmp_path: Path) -> None:
    """Every job of a two-node chain is logged `queued`, `started` and `finished` with its key and placement."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    status, _, events = _run(placed, values, tmp_path)
    for node in ("a", "b"):
        for element in ((0,), (1,)):
            kinds = [e.kind for e in events if e.node == node and e.job is not None and e.job.element == element]
            assert kinds == ["queued", "started", "finished"]
    finished = [e for e in events if e.kind == "finished"]
    assert {e.key for e in finished} == {r.key_digest for refs in status.results.values() for r in refs}
    assert all(e.output is not None and e.placement_digest is not None and e.env == "host" for e in finished)


@pytest.mark.contract
def test_a_set_cancel_sends_no_further_fragment(tmp_path: Path) -> None:
    """Cancelled before the run, no fragment reaches the executor and the run raises `RunCancelled`."""
    import threading  # noqa: PLC0415

    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    executor = Recording(ResultStore(tmp_path / "store"))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(runner.RunCancelled):
        runner.PydraRunner(profile=PROFILE, work_dir=tmp_path / "pydra").run_plan(
            placed, executor, values=values, events=EventLog(tmp_path / "run", "a1", 1), cancel=cancel
        )
    assert executor.fragments == []


@pytest.mark.contract
def test_a_run_completes_with_pydra_pickle_loading_refused_and_leaves_no_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every cloudpickle load fails; the run completes and writes no result, error or return-value pickle."""

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pydra unpickled")

    monkeypatch.setattr("cloudpickle.load", refuse)
    monkeypatch.setattr("cloudpickle.loads", refuse)
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    status, _, _ = _run(placed, values, tmp_path)
    assert status.done and {p.name for p in (tmp_path / "pydra").rglob("*.pklz")} <= {"_job.pklz"}


@pytest.mark.contract
def test_the_pydra_work_dir_is_private_under_a_permissive_umask(tmp_path: Path) -> None:
    """The work dir the runner creates is mode `0o700` under umask `0o022`, and it is kept after the run."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}])
    before = os.umask(0o022)
    try:
        status, _, _ = _run(placed, values, tmp_path)
    finally:
        os.umask(before)
    assert status.done and stat.S_IMODE((tmp_path / "pydra").stat().st_mode) == 0o700


@pytest.mark.logic
def test_the_submitter_is_built_over_a_senselab_store_with_no_readonly_caches(tmp_path: Path) -> None:
    """`PydraRunner` submits through `SenselabStore` and passes no `readonly_caches`, so no cache is read."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}])
    store = runner.SenselabStore(tmp_path / "pydra")
    executor, cancel = Recording(ResultStore(tmp_path / "store"), PROFILE), threading.Event()
    run = runner._Run(
        placed,
        executor.admit(placed, values),
        values,
        executor,
        store,
        EventLog(tmp_path / "run", "a1", 1),
        cancel,
        concurrent.futures.ThreadPoolExecutor(1),
        runner.Follower(executor, cancel, first_s=0.01, max_s=0.1),
    )
    submitter = runner._submitter(run, store, ["cpu"])
    assert type(submitter.store) is runner.SenselabStore
    assert submitter.readonly_caches is None and not hasattr(submitter.store, "readonly_caches")


def _terminal(events: Sequence[Any], node: str) -> list[Any]:
    """Return the terminal events of `node`'s jobs in element order."""
    ended = [e for e in events if e.node == node and e.job is not None and e.kind in TERMINAL]
    return sorted(ended, key=lambda e: e.job.element)


@pytest.mark.logic
def test_records_of_one_content_are_one_execution_and_exact_reuses(tmp_path: Path) -> None:
    """Two records of one text: per node one job runs and the other is served it, exact, with its own output."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "same"}, {"text": "same"}])
    status, executor, events = _run(placed, values, tmp_path)
    store = executor.inner.store
    assert status.done and status_from(events).counts["done"] == 4
    for node in ("a", "b"):
        first, second = status.results[node]
        assert (first.job.element, second.job.element) == ((0,), (1,))
        assert (first.key_digest, first.output_identity) == (second.key_digest, second.output_identity)
        assert [(e.kind, e.reuse) for e in _terminal(events, node)] == [("finished", None), ("cache_hit", "exact")]
        outputs = [store.outputs_of(e.key, e.output) for e in _terminal(events, node)]
        assert outputs[0] is not None and outputs[0] == outputs[1]
    activities = store.provenance.activities()
    assert sorted("senselab:Reuse" in a.types() for a in activities) == [False, False, True, True]


@pytest.mark.logic
def test_records_of_one_failing_content_both_fail_and_only_descendants_are_blocked(tmp_path: Path) -> None:
    """Two records of one failing text: both fail with the one failure; their `b` jobs are blocked by it."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "fail"}, {"text": "fail"}])
    status, _, events = _run(placed, values, tmp_path)
    first, second = status.results["a"]
    assert first.failure is not None and first.failure == second.failure and status.blocked["a"] == ()
    assert [e.kind for e in _terminal(events, "a")] == ["failed", "failed"]
    assert [e.kind for e in _terminal(events, "b")] == ["blocked", "blocked"] and status.blocked["b"] == ((0,), (1,))
    assert status_from(events).counts["blocked"] == 2 and status_from(events).counts["failed"] == 2


@pytest.mark.logic
def test_a_job_reading_a_shared_result_runs_under_its_own_element(tmp_path: Path) -> None:
    """`a` is one job for both records; `d` reads each record's own `other` and runs once per record, as itself."""
    rows = [{"text": "same", "other": f"o{i}"} for i in range(2)]
    placed, values = _plan(ZIPPED, rows, {"both": "shared"})
    status, executor, events = _run(placed, values, tmp_path)
    assert status.done and status_from(events).counts["done"] == 6
    sent = {(f.node, j.element): i for f in executor.fragments for j, i in zip(f.jobs, f.inputs, strict=True)}
    assert [sent[("d", (i,))]["right"] for i in range(2)] == [placed.manifest.rows[i].inputs["other"] for i in range(2)]
    assert [e.kind for e in _terminal(events, "d")] == ["finished", "finished"]
    assert [e.kind for e in _terminal(events, "e")] == ["finished", "cache_hit"]
