"""A run's lifecycle over the fake stack: its directory, status, wait, results, cancel and the CLI that reads it."""

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from runner_test import (  # noqa: F401 - `packaged` is the autouse fixture of the fake stack
    CHAIN_EDGES,
    CHAINED,
    Recording,
    _graph,
    _plan,
    packaged,
)
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.events import read_events
from senselab_core.runtime.execution.executor import Handle, PlanFragment
from senselab_core.runtime.execution.run import Output, Run
from senselab_core.runtime.planner import ExecutorUnavailable
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import attempt_id
from senselab_core.runtime.store import ResultStore

pytest.importorskip("pydra")

GRAPH = _graph(CHAINED, CHAIN_EDGES, text="list[json]")
ROWS = [{"text": t} for t in ("one", "fail", "two")]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The run directory and the default result store are under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    yield tmp_path / "cache"


def _started(tmp_path: Path, rows: list[dict[str, str]] = ROWS) -> tuple[Run, Recording]:
    placed, values = _plan(GRAPH, rows)
    executor = Recording(ResultStore(tmp_path / "store"))
    run = placed.run(executor)
    assert values and run.wait(60).done
    return run, executor


@pytest.mark.contract
def test_run_writes_its_directory_and_derives_status_from_events(tmp_path: Path) -> None:
    """A run's directory holds the plan, manifest, placement, pool directories and one event file per writer."""
    run, _ = _started(tmp_path)
    names = {p.name for p in run.run_dir.iterdir()}
    assert {"plan.json", "manifest.json", "placements", "events", "dead", "prov", "queues", "epoch"} <= names
    status = run.status()
    assert status.counts == {"queued": 0, "running": 0, "done": 4, "failed": 1} and status.done
    assert status.faults == {"content": 1} and status.broken == ()
    assert status == Run(run.run_dir).status()
    assert run.manifest()["state"] == "failed" and len(list((run.run_dir / "events").iterdir())) == 1


@pytest.mark.contract
def test_attempt_id_distinct_from_run_id(tmp_path: Path) -> None:
    """The run id is the plan hash, which names the directory; the attempt id names the placement file."""
    run, _ = _started(tmp_path)
    manifest = run.manifest()
    assert run.plan is not None and run.run_dir.name == manifest["plan_hash"] == run.plan.hash
    assert manifest["attempt"] == attempt_id() != manifest["plan_hash"]
    assert (run.run_dir / "placements" / f"{attempt_id()}.json").is_file()
    assert {e.job.plan_hash for e in read_events(run.run_dir) if e.job} == {manifest["plan_hash"]}


@pytest.mark.contract
def test_results_come_from_events_and_the_store_in_record_order(tmp_path: Path) -> None:
    """A failed record gives its fault; the others give an output; a run opened over the directory reads the same."""
    run, _ = _started(tmp_path)
    items = list(run.results())
    assert [r.id for r, _ in items] == ["r0", "r1", "r2"]
    assert isinstance(items[0][1], Output) and isinstance(items[2][1], Output)
    assert items[1][1].fault == "content" and items[1][1].type == "ContentFault"  # type: ignore[union-attr]
    again = list(Run(run.run_dir, store=ResultStore(tmp_path / "store")).results())
    assert [(r.id, type(o).__name__) for r, o in again] == [(r.id, type(o).__name__) for r, o in items]


class Gated(Recording):
    """A `Recording` executor whose first fragment waits until `release` is set."""

    def __init__(self, store: ResultStore) -> None:
        """Hold the gate."""
        super().__init__(store)
        self.entered = threading.Event()
        self.gate = threading.Event()

    def submit(self, fragment: PlanFragment) -> Handle:
        """Wait at the gate before submitting a fragment of node `a`."""
        if fragment.node == "a":
            self.entered.set()
            assert self.gate.wait(30)
        return super().submit(fragment)


@pytest.mark.contract
def test_cancel_stops_release(tmp_path: Path) -> None:
    """Cancelled while node `a` is in flight, `a` finishes and no fragment of `b` is ever sent."""
    placed, _ = _plan(GRAPH, [{"text": t} for t in ("one", "two")])
    executor = Gated(ResultStore(tmp_path / "store"))
    run = placed.run(executor)
    assert executor.entered.wait(30)
    run.cancel()
    executor.gate.set()
    status = run.wait(60)
    assert executor.sizes("b") == [] and sum(executor.sizes("a")) == 2
    assert run.manifest()["state"] == "cancelled" and status.done
    assert status.counts["done"] == 2 and status.counts["failed"] == 0
    assert all(not isinstance(o, Output) for _, o in run.results())


@pytest.mark.contract
def test_a_slurm_substrate_without_an_executor_names_pool(tmp_path: Path) -> None:
    """With no executor given, a slurm substrate is not run on `InProcess`: it raises, naming `Pool`."""
    slurm = ExecutionProfile.load("orcd")
    placed, _ = _plan(GRAPH, ROWS)
    placed = type(placed)(**{**placed.__dict__, "execution": slurm})
    with pytest.raises(ExecutorUnavailable, match="Pool") as raised:
        placed.run()
    assert raised.value.needs == "Pool" and not (tmp_path / "cache").exists()


@pytest.mark.contract
def test_explain_names_fault_and_placement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`explain <run> <node>` prints the node's placement and reason, its events and its fault with the message."""
    run, _ = _started(tmp_path)
    assert cli_main(["explain", str(run.run_dir), "a"]) == 0
    out = capsys.readouterr().out
    assert "node a (runner_test.score): backend 'score'" in out and "device cpu" in out and "resource class" in out
    assert "failed a[1]" in out and "fault content" in out and "ContentFault" in out and "asked to fail" in out
    key = next(e.key for e in run.events() if e.kind == "failed")
    assert key is not None
    assert cli_main(["explain", run.run_dir.name[:10], key[:12]]) == 0
    assert "fault content: ContentFault" in capsys.readouterr().out
    assert cli_main(["explain", str(run.run_dir), "nope"]) == 1


@pytest.mark.contract
def test_status_command_reports_jobs_faults_and_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`status <run>` prints the attempt's state, the job counts, the faults by class and each node."""
    run, _ = _started(tmp_path)
    assert cli_main(["status", str(run.run_dir)]) == 0
    out = capsys.readouterr().out
    assert "failed" in out.splitlines()[0] and "jobs: queued 0, running 0, done 4, failed 1" in out
    assert "faults: content 1" in out and "node a: done 2, failed 1" in out
    assert cli_main(["status", str(tmp_path / "nowhere")]) == 1
    assert run.manifest()["plan_hash"] in out
