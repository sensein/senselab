"""`PydraRunner`: a placed plan's jobs run on a warm pool session, per element, keyed by senselab (fake stack)."""

import dataclasses
import os
import platform
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel import prov
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.declare import BackendSpec, Env, Registry, TaskParams, TaskSpec
from senselab_core.kernel.graph import Edge, Graph, Node, fan_out_issues
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.kernel.license import License
from senselab_core.runtime.call import Packaged, RunConfig, bind_input, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution import portable, runner
from senselab_core.runtime.execution.events import TERMINAL, EventLog, read_events, run_dir_of, status_from
from senselab_core.runtime.execution.executor import RunCancelled
from senselab_core.runtime.execution.portable import read_outputs, stored_evidence
from senselab_core.runtime.execution.records import add_plan_entity
from senselab_core.runtime.licenses import LicensePolicy, license_record
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.testing.interpreter import source_tree_python
from senselab_core.testing.pools import POOL_PROFILE, served

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
PROFILE = POOL_PROFILE


def _license() -> License:
    return license_record("MIT", source="lock", verified={"date": "2026-11-02", "how": "pep639"}, gated=False)


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile, device: str) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(
        python=source_tree_python(),
        python_version=platform.python_version(),
        lock_sha="1" * 64,
        wheel_set="2" * 64,
        torch_index=None,
    )


@pytest.fixture(autouse=True)
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The runner reads this module's registry and opens every env as this interpreter, MIT-locked; cache here."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    records = {"pkg": _license()}
    previous = use_packaged(Packaged(CATALOGUE, _this_interpreter, REGISTRY, lambda _: records))
    yield
    use_packaged(previous)


def _graph(nodes: Sequence[tuple[str, TaskSpec]], edges: Sequence[tuple[str, str, str]], **inputs: str) -> Graph:
    return Graph(
        inputs=inputs,
        nodes=tuple(Node(id=n, task=t.name, task_version="1") for n, t in nodes),
        edges=tuple(Edge(source=s, target=t, kind=k) for s, t, k in edges),
    )


def _plan(
    graph: Graph, rows: Sequence[Mapping[str, str]], broadcast: Mapping[str, object] | None = None
) -> tuple[Plan, dict[str, object]]:
    """Return `graph` planned over records of `rows`, and the value of each input identity."""
    records = [Record(id=f"r{i}", inputs={k: ValueInput(value=v) for k, v in r.items()}) for i, r in enumerate(rows)]
    placed = plan(graph, RecordSet(records), config=RunConfig(), execution=PROFILE, inputs=broadcast or None)
    given = [*(v for r in rows for v in r.values()), *(broadcast or {}).values()]
    return placed, {bind_input("x", "json", v)[0]: v for v in given}


def _run(placed: Plan, values: Mapping[str, object], tmp_path: Path, **kw: Any) -> tuple[Any, Any, list[Any]]:  # noqa: ANN401
    """Run `placed` on a pool served here; return its outcome, the session's store and its events."""
    events = EventLog(tmp_path / "run", "a1", 1)
    with served(tmp_path / "pool") as (session, _):
        outcome = runner.PydraRunner(profile=PROFILE, work_dir=tmp_path / "pydra").run_plan(
            placed, session, values=values, events=events, **kw
        )
        store = session.store
    return outcome, store, read_events(tmp_path / "run")


def _terminal(events: Sequence[Any], node: str) -> list[Any]:
    """Return the terminal events of `node`'s jobs in element order."""
    ended = [e for e in events if e.node == node and e.job is not None and e.kind in TERMINAL]
    return sorted(ended, key=lambda e: e.job.element)


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


@pytest.mark.contract
def test_record_bound_inputs_zip_and_a_broadcast_input_reaches_every_element(tmp_path: Path) -> None:
    """`d` reads its record's `other` beside `a`'s same element; the broadcast `both` is one value everywhere."""
    placed, values = _plan(ZIPPED, ZIPPED_ROWS, {"both": "shared"})
    status, _, _ = _run(placed, values, tmp_path)
    assert status.done and len(status.results["d"]) == 3 and len(status.results["e"]) == 3
    assert len({r.key_digest for r in status.results["d"]}) == 3


@pytest.mark.contract
def test_gather_waits_for_every_element_and_an_empty_map_gathers_to_the_empty_list(tmp_path: Path) -> None:
    """The gather starts after every element of `a` ended; over no record it runs once, on no element."""
    placed, values = _plan(_graph(GATHERED, GATHER_EDGES, text="list[json]"), [{"text": t} for t in ("x", "y", "z")])
    status, _, events = _run(placed, values, tmp_path / "three")
    ended = max(e.seq for e in _terminal(events, "a"))
    [started] = [e for e in events if e.node == "g" and e.kind == "started"]
    assert status.done and started.seq > ended
    placed, values = _plan(_graph(GATHERED, GATHER_EDGES, text="list[json]"), [])
    status, _, _ = _run(placed, values, tmp_path / "none")
    assert status.results["a"] == () and len(status.results["g"]) == 1 and status.done


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
    """Every leaf job pydra recorded, plain, gathered, zipped or broadcast, is keyed as its pool job is keyed."""
    placed, values = _plan(graph, rows, broadcast)
    records = runner.SenselabStore(tmp_path / "records")
    status, store, _ = _run(placed, values, tmp_path, store=records)
    keys = {r.key_digest for refs in status.results.values() for r in refs}
    leaves = {c for c in records.checksums() if not c.startswith(("records-", "plan-"))}
    assert status.done and len(keys) == jobs and leaves == keys
    assert all(store.stored(k) for k in keys)


@pytest.mark.contract
def test_no_result_pickle_written(tmp_path: Path) -> None:
    """A run writes no `_result.pklz`: every result is in senselab's store and pydra's records are in memory."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    status, _, _ = _run(placed, values, tmp_path)
    assert status.done and list(tmp_path.rglob("_result.pklz")) == []


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
def test_a_set_cancel_sends_no_job_and_raises_run_cancelled(tmp_path: Path) -> None:
    """Cancelled before the run, no job reaches the pool and the run raises `RunCancelled`; each job ends cancelled."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RunCancelled):
        _run(placed, values, tmp_path, cancel=cancel)
    kinds = [e.kind for e in read_events(tmp_path / "run")]
    assert "started" not in kinds and kinds.count("cancelled") == 4


@pytest.mark.contract
def test_a_license_refusal_fails_every_job_of_its_node_and_is_not_raised(tmp_path: Path) -> None:
    """A run config refusing the env's license fails `a`'s jobs with `LicenseRefused`; `b` is blocked; no raise."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": t} for t in ("one", "two")])
    refusing = dataclasses.replace(placed, config=RunConfig(licenses=LicensePolicy(refuse=frozenset({"MIT"}))))
    status, _, events = _run(refusing, values, tmp_path)
    assert [r.failure.type for r in status.results["a"] if r.failure] == ["LicenseRefused"] * 2
    assert status.blocked["b"] == ((0,), (1,)) and not [e for e in events if e.kind == "started"]


@pytest.mark.contract
def test_records_of_one_content_are_one_execution_and_exact_reuses(tmp_path: Path) -> None:
    """Two records of one text: per node one job runs and the other is served it, exact, with its own output."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "same"}, {"text": "same"}])
    status, store, events = _run(placed, values, tmp_path)
    assert status.done and status_from(events).counts["done"] == 4
    for node in ("a", "b"):
        first, second = status.results[node]
        assert (first.key_digest, first.output_identity) == (second.key_digest, second.output_identity)
        assert [(e.kind, e.reuse) for e in _terminal(events, node)] == [("finished", None), ("cache_hit", "exact")]
        outputs = [read_outputs(store, e.key, e.output) for e in _terminal(events, node)]
        assert outputs[0] is not None and outputs[0] == outputs[1]
    plan_entity = ProvStore()
    add_plan_entity(plan_entity, placed.hash)
    plan_entity.write(run_dir_of(placed.hash))
    activities = ProvStore.read(run_dir_of(placed.hash)).activities()
    assert sorted("senselab:Reuse" in a.types() for a in activities) == [False, False, True, True]


@pytest.mark.contract
def test_a_verified_hit_records_an_execution_naming_its_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sampled cross-context hit recomputes: its executed activity's `senselab:verifies` is the evidence's digest."""
    placed, values = _plan(
        _graph([("a", SCORE)], [("input.text", "a.text", "map")], text="list[json]"), [{"text": "v"}]
    )
    status, store, _ = _run(placed, values, tmp_path)
    assert status.done
    elsewhere = dataclasses.replace(
        _this_interpreter(REGISTRY.backends(SCORE.name)[0], PROFILE, "cpu"), wheel_set="5" * 64
    )
    records = {"pkg": _license()}
    use_packaged(Packaged(CATALOGUE, lambda _b, _p, _d: elsewhere, REGISTRY, lambda _: records))
    sampling = dataclasses.replace(placed, config=RunConfig(ReusePolicy.TOLERANT, 1.0))
    events = EventLog(tmp_path / "again", "a2", 1)
    with served(tmp_path / "again-pool", store=tmp_path / "pool" / "store") as (session, _):
        again = runner.PydraRunner(profile=PROFILE, work_dir=tmp_path / "pydra2").run_plan(
            sampling, session, values=values, events=events
        )
        [evidence] = stored_evidence(session.store)
    assert again.done and evidence.agree
    executed = [a for a in ProvStore.read(run_dir_of(placed.hash)).activities() if "senselab:Reuse" not in a.types()]
    verifying = [a for a in executed if "senselab:verifies" in a.node()]
    assert len(executed) == 2 and len(verifying) == 1
    assert prov.literal_value(verifying[0].node()["senselab:verifies"]) == evidence.canonical_digest()
    assert portable._VERIFIED == {} and portable._RAN == {}


@pytest.mark.contract
def test_records_of_one_failing_content_both_fail_and_only_descendants_are_blocked(tmp_path: Path) -> None:
    """Two records of one failing text: both fail with the one failure; their `b` jobs are blocked by it."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "fail"}, {"text": "fail"}])
    status, _, events = _run(placed, values, tmp_path)
    first, second = status.results["a"]
    assert first.failure is not None and first.failure == second.failure and status.blocked["a"] == ()
    assert [e.kind for e in _terminal(events, "a")] == ["failed", "failed"]
    assert [e.kind for e in _terminal(events, "b")] == ["blocked", "blocked"] and status.blocked["b"] == ((0,), (1,))


@pytest.mark.parametrize(("text", "kinds"), [("one", ["finished", "cache_hit"]), ("fail", ["failed", "failed"])])
@pytest.mark.contract
def test_two_nodes_of_one_job_each_hold_their_own_result(text: str, kinds: list[str], tmp_path: Path) -> None:
    """Nodes `a` and `a2` run one task on one record: one job runs, and each node's result names that node."""
    graph = _graph(
        [("a", SCORE), ("a2", SCORE)],
        [("input.text", "a.text", "map"), ("input.text", "a2.text", "map")],
        text="list[json]",
    )
    placed, values = _plan(graph, [{"text": text}])
    status, _, events = _run(placed, values, tmp_path)
    assert [r.job.node for n in ("a", "a2") for r in status.results[n]] == ["a", "a2"]
    assert sorted(e.kind for n in ("a", "a2") for e in _terminal(events, n)) == sorted(kinds)


@pytest.mark.contract
def test_a_run_makes_no_telemetry_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pydra's version check is never called, and the environment is left as it was."""
    checks: list[object] = []
    monkeypatch.setattr("pydra.engine.job.Job.check_etelemetry", staticmethod(lambda: checks.append(1)))
    monkeypatch.delenv("NO_ET", raising=False)
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}])
    status, _, _ = _run(placed, values, tmp_path)
    assert status.done and checks == [] and "NO_ET" not in os.environ


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
def test_the_runner_holds_pydras_lease_protocol_but_not_its_directory_lease() -> None:
    """The job lease of the run's records satisfies pydra's `Lease` Protocol and is not a `DirectoryLease`."""
    lease_module = pytest.importorskip("pydra.engine.lease")
    held = runner.SenselabStore(Path("root")).acquire("c", {})
    assert isinstance(held, lease_module.Lease) and not isinstance(held, lease_module.DirectoryLease)
    assert held.try_take() is None
    with held as entered:
        assert entered is held


@pytest.mark.contract
def test_a_plan_of_another_profile_is_refused(tmp_path: Path) -> None:
    """A session runs only plans placed under its own profile."""
    placed, values = _plan(_graph(CHAINED, CHAIN_EDGES, text="list[json]"), [{"text": "one"}])
    other = PROFILE.model_copy(update={"pool": PROFILE.pool.model_copy(update={"idle_polls": 7})})
    events = EventLog(tmp_path / "run", "a1", 1)
    with served(tmp_path / "pool", other) as (session, _), pytest.raises(ValueError, match="another execution profile"):
        runner.PydraRunner(profile=other).run_plan(placed, session, values=values, events=events)
