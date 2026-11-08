"""A run's lifecycle over the fake stack: its directory, status, wait, results, cancel and the CLI that reads it."""

import errno
import json
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from planted_objects_fixture import identities_in, identity_leaks, real_names
from queue_race_fixture import fragment
from runner_test import (  # noqa: F401 - `packaged` is the autouse fixture of the fake stack
    CHAIN_EDGES,
    CHAINED,
    ZIPPED,
    ZIPPED_ROWS,
    Recording,
    _graph,
    _plan,
    packaged,
)
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.events import read_events
from senselab_core.runtime.execution.executor import FaultRecord, Handle, PlanFragment
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import (
    InputChanged,
    InputsUnavailable,
    NotRun,
    Output,
    Run,
    bind_inputs,
    load_plan,
)
from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.revision import attempt_id, attempt_scope, process_attempt_id
from senselab_core.runtime.storage import owner_remedy
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerLost
from syscall_faults_fixture import refuse_call

pytest.importorskip("pydra")

GRAPH = _graph(CHAINED, CHAIN_EDGES, text="list[json]")
ROWS = [{"text": t} for t in ("one", "fail", "two")]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The run directory and the default result store are under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    yield tmp_path / "cache"


def _started(tmp_path: Path, rows: list[dict[str, str]] = ROWS) -> tuple[Run, Recording]:
    placed, _ = _plan(GRAPH, rows)
    executor = Recording(ResultStore(tmp_path / "store"))
    run = placed.run(executor, store=executor.store)
    assert run.wait(60).done
    return run, executor


@pytest.mark.contract
def test_a_failed_run_leaves_no_path_or_account_in_its_readable_files(tmp_path: Path) -> None:
    """A fault message that names a path is stored with the path rendered; `private/` holds the paths and is 0700."""
    run, _ = _started(tmp_path, [{"text": "leak"}, {"text": "one"}])
    failed = [e.failure for e in run.events() if e.failure is not None]
    assert failed and all("<path>" in f.message for f in failed)
    assert identity_leaks(run.run_dir, tmp_path) == []
    assert str(tmp_path / "store") in (run.run_dir / "private" / "manifest.json").read_text()


@pytest.mark.logic
def test_the_sweep_matches_a_name_as_a_token_and_a_path_anywhere(tmp_path: Path) -> None:
    """A name inside a longer word (`runner` in `runner_test`) is no leak; alone it is, and a path is anywhere."""
    for name in real_names():
        assert identities_in(f"{name}_test x{name} {name}-1".encode(), tmp_path) == []
        assert name in identities_in(f"by {name}.".encode(), tmp_path)
    assert identities_in(f"at {tmp_path}x".encode(), tmp_path) == [str(tmp_path)]


@pytest.mark.logic
def test_a_stored_fault_message_names_no_host(tmp_path: Path) -> None:
    """This host's name and its first label, as a peer address or a node name in a fault, are rendered to `<host>`."""
    host = socket.gethostname()
    text = f"ray: connect to {host}:6379 refused; node {host.split('.')[0]} is down; job x{host} kept"
    stored = FaultRecord.of({"fault": "infrastructure", "type": "T", "message": text}).message
    assert stored.count("<host>") == 2 and f"x{host}" in stored, stored


@pytest.mark.logic
def test_a_stored_fault_message_names_no_account_or_path(tmp_path: Path, slack_tests: dict[str, Any]) -> None:
    """A lease holder's user, an owner remedy's account and any absolute path are rendered to labels."""
    times = slack_tests["held_lease"]
    held = Lease(tmp_path / "lock", stale_after=times["stale_after_s"])
    with held, pytest.raises(LeaseTimeout) as timeout:
        Lease(tmp_path / "lock", stale_after=times["stale_after_s"], timeout=times["timeout_s"]).__enter__()
    remedy = owner_remedy(tmp_path)
    for text in (str(timeout.value), remedy, f"open {tmp_path / 'x.wav'}: denied", f"in ~/data and {Path.home()}"):
        stored = FaultRecord.of({"fault": "infrastructure", "type": "T", "message": text}).message
        assert identities_in(stored.encode(), tmp_path) == [], stored
        assert "<path>" in stored
    assert FaultRecord.of({"fault": "content", "type": "T", "message": "a/b and 3/4 http://h/x"}).message == (
        "a/b and 3/4 http://h/x"
    )


@pytest.mark.contract
def test_status_comes_from_events_alone_and_counts_blocked_jobs(tmp_path: Path) -> None:
    """The run directory holds no status file; blocked descendants of the failed element are counted, and done."""
    run, _ = _started(tmp_path)
    names = {p.name for p in run.run_dir.iterdir()}
    assert {
        "plan.json",
        "private",
        "events",
        "dead",
        "prov",
        "queues",
        "epoch",
    } <= names
    assert {p.name for p in (run.run_dir / "private").iterdir()} >= {"inputs.json", "manifest.json", "placements"}
    assert set(run.manifest()) == {"plan_hash", "store"}
    status = run.status()
    assert status.counts == {"queued": 0, "running": 0, "done": 4, "failed": 1, "blocked": 1, "cancelled": 0}
    assert status.done and not status.cancelled and status.faults == {"content": 1} and status.broken == ()
    assert status == Run(run.run_dir).status() and status.last_event_ns is not None


@pytest.mark.contract
def test_every_run_is_a_new_attempt_with_its_own_placement_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of one plan in one process have two attempt ids, two placement files and tagged events."""
    placed, _ = _plan(GRAPH, ROWS)
    executor = Recording(ResultStore(tmp_path / "store"))
    first, second = placed.run(executor, store=executor.store), None
    first.wait(60)
    second = placed.run(executor, store=executor.store)
    second.wait(60)
    attempts = {e.attempt for e in read_events(first.run_dir)}
    assert len(attempts) == 2 and {p.stem for p in (first.run_dir / "private" / "placements").iterdir()} == attempts
    assert first.run_dir.name == placed.hash and placed.hash not in attempts
    monkeypatch.setenv("SENSELAB_RUN_ID", "pinned")
    third, fourth = placed.run(executor, store=executor.store), None
    third.wait(60)
    fourth = placed.run(executor, store=executor.store)
    fourth.wait(60)
    pinned = {p.stem for p in (first.run_dir / "private" / "placements").iterdir()} - attempts
    assert pinned == {"pinned", "pinned-1"}
    claimed = {p.name: p.read_text() for p in (first.run_dir / "attempts").iterdir()}
    assert sorted(claimed) == ["1", "2", "3", "4"] and claimed["3"] == "pinned" and claimed["4"] == "pinned-1"
    assert {(e.attempt, e.ordinal) for e in read_events(first.run_dir)} == {(a, int(n)) for n, a in claimed.items()}


class Attempted(Recording):
    """A `Recording` executor that notes `attempt_id()` as each fragment is submitted."""

    def __init__(self, store: ResultStore) -> None:
        """Hold the notes."""
        super().__init__(store)
        self.seen: list[str] = []

    def submit(self, fragment: PlanFragment) -> Handle:
        """Note the attempt id in force, then submit."""
        self.seen.append(attempt_id())
        return super().submit(fragment)


@pytest.mark.contract
def test_attempt_id_is_the_runs_attempt_while_it_runs_and_the_process_id_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evidence and the revision manifest read `attempt_id()`: the running attempt's id, else the process's."""
    monkeypatch.delenv("SENSELAB_RUN_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    placed, _ = _plan(GRAPH, [{"text": "one"}])
    executor = Attempted(ResultStore(tmp_path / "store"))
    run = placed.run(executor, store=executor.store)
    assert run.wait(60).done
    [attempt] = {e.attempt for e in read_events(run.run_dir)}
    assert executor.seen and set(executor.seen) == {attempt} and attempt != process_attempt_id()
    assert attempt_id() == process_attempt_id()
    with attempt_scope("scoped"):
        assert attempt_id() == "scoped"
    assert attempt_id() == process_attempt_id()


def _live(tmp_path: Path) -> tuple[Run, FaultRecord]:
    """Run `ZIPPED` with its broadcast input given as an object that is neither a path nor JSON."""
    held = FaultRecord(fault="content", type="Held", message="in memory", gap=False, charged=True)
    placed, _ = _plan(ZIPPED, ZIPPED_ROWS, broadcast={"both": held})
    executor = Recording(ResultStore(tmp_path / "store"))
    run = placed.run(executor, store=executor.store)
    assert run.wait(60).done
    return run, held


@pytest.mark.contract
def test_an_input_given_as_an_object_runs_in_process_and_is_recorded_as_memory(tmp_path: Path) -> None:
    """The object is admitted live; `inputs.json` records `{"memory": <identity>}` for it."""
    run, _ = _live(tmp_path)
    assert run.status().counts["done"] == 9 and run.status().faults == {}
    assert all(isinstance(o, Output) for _, o in run.results())
    shared = json.loads((run.run_dir / "private" / "inputs.json").read_text())["shared"]
    [identity] = {i for row in load_plan(run.run_dir).manifest.rows for p, i in row.inputs.items() if p == "both"}
    assert shared == {"both": {"memory": identity}}


@pytest.mark.contract
def test_a_run_from_disk_without_its_live_inputs_raises_before_writing(tmp_path: Path) -> None:
    """Re-run from the directory, the memory input is `InputsUnavailable`, and no attempt or file is written."""
    run, _ = _live(tmp_path)
    with pytest.raises(InputsUnavailable, match="both") as raised:
        bind_inputs(run.run_dir)
    assert {port for _, port in raised.value.missing} == {"both"} and len(raised.value.missing) == 3
    before = sorted(str(p.relative_to(run.run_dir)) for p in run.run_dir.rglob("*") if "pydra" not in p.parts)
    with pytest.raises(InputsUnavailable):
        Run(run.run_dir).plan.run(Recording(ResultStore(tmp_path / "store")), store=ResultStore(tmp_path / "store"))
    after = sorted(str(p.relative_to(run.run_dir)) for p in run.run_dir.rglob("*") if "pydra" not in p.parts)
    assert after == before
    run.run_dir.rename(tmp_path / "moved")
    with pytest.raises(InputsUnavailable):
        load_plan(tmp_path / "moved").run(Recording(ResultStore(tmp_path / "store")))
    assert not run.run_dir.exists()


@pytest.mark.contract
def test_a_second_attempt_is_served_from_the_store_and_logged_as_cache_hits(tmp_path: Path) -> None:
    """Resumed from the directory alone, the jobs the store holds are `cache_hit` with their reuse policy."""
    run, executor = _started(tmp_path)
    fresh = Run(run.run_dir, store=executor.store)
    again = fresh.plan.run(executor, store=executor.store)
    assert again.wait(60).done
    latest = max(read_events(run.run_dir), key=lambda e: e.t_ns).attempt
    hits = [e for e in read_events(run.run_dir) if e.kind == "cache_hit"]
    assert hits and {e.attempt for e in hits} == {latest} and {e.reuse for e in hits} == {"exact"}
    assert not [e for e in read_events(run.run_dir) if e.kind == "finished" and e.attempt == latest]


@pytest.mark.contract
def test_results_come_lazily_from_the_run_in_record_order(tmp_path: Path) -> None:
    """A failed record gives its fault; a run opened over the directory, with no plan in memory, reads the same."""
    run, _ = _started(tmp_path)
    results = run.results()
    assert results.run is run and [r.id for r, _ in results] == ["r0", "r1", "r2"]
    items = list(results)
    assert isinstance(items[0][1], Output) and isinstance(items[2][1], Output)
    assert isinstance(items[1][1], FaultRecord) and items[1][1].type == "ContentFault"
    again = list(Run(run.run_dir, store=ResultStore(tmp_path / "store")).results())
    assert [(r.id, type(o).__name__) for r, o in again] == [(r.id, type(o).__name__) for r, o in items]


@pytest.mark.contract
def test_inputs_are_bound_from_the_run_directory_and_a_changed_one_is_refused(tmp_path: Path) -> None:
    """`bind_inputs` re-binds every source and refuses one whose identity differs from the manifest's."""
    run, _ = _started(tmp_path)
    placed = load_plan(run.run_dir)
    values = bind_inputs(run.run_dir)
    assert set(values) == {i for row in placed.manifest.rows for i in row.inputs.values()}
    document = json.loads((run.run_dir / "private" / "inputs.json").read_text())
    document["rows"][0]["text"] = "changed"
    (run.run_dir / "private" / "inputs.json").write_text(json.dumps(document))
    with pytest.raises(InputChanged, match="r0"):
        bind_inputs(run.run_dir)


_FRESH = """
import json, sys
from senselab_core.runtime.execution.run import Output, Run, bind_inputs, load_plan
run = Run(sys.argv[1])
items = list(run.results())
print(json.dumps({
    "ids": [r.id for r, _ in items],
    "kinds": [type(o).__name__ for _, o in items],
    "done": run.status().done,
    "values": len(bind_inputs(run.run_dir)),
    "hash": load_plan(run.run_dir).hash,
}))
"""


@pytest.mark.contract
def test_a_fresh_process_reads_and_binds_a_run_from_its_directory(tmp_path: Path) -> None:
    """With no plan in memory, a new process reads the results and status and binds the inputs of the run."""
    run, _ = _started(tmp_path)
    done = subprocess.run(
        [sys.executable, "-c", _FRESH, str(run.run_dir)], capture_output=True, text=True, check=True, timeout=120
    )
    found = json.loads(done.stdout)
    assert found["ids"] == ["r0", "r1", "r2"] and found["kinds"] == ["Output", "FaultRecord", "Output"]
    assert found["done"] is True and found["values"] == 3 and found["hash"] == run.run_dir.name


class Gated(Recording):
    """A `Recording` executor whose fragments of node `a` wait until `gate` is set.

    `entered` is set once fragments holding `jobs` jobs of `a` are waiting at the gate.
    """

    def __init__(self, store: ResultStore, jobs: int) -> None:
        """Hold the gate."""
        super().__init__(store)
        self.entered = threading.Event()
        self.gate = threading.Event()
        self._jobs = jobs
        self._held = 0
        self._lock = threading.Lock()

    def submit(self, fragment: PlanFragment) -> Handle:
        """Wait at the gate before submitting a fragment of node `a`."""
        if fragment.node == "a":
            with self._lock:
                self._held += len(fragment.jobs)
                if self._held >= self._jobs:
                    self.entered.set()
            assert self.gate.wait(30)
        return super().submit(fragment)


@pytest.mark.contract
def test_cancel_stops_release(tmp_path: Path) -> None:
    """Cancelled while every job of `a` is in flight, `a` finishes, no fragment of `b` is sent, and status says so."""
    placed, _ = _plan(GRAPH, [{"text": t} for t in ("one", "two")])
    executor = Gated(ResultStore(tmp_path / "store"), jobs=2)
    run = placed.run(executor, store=executor.store)
    assert executor.entered.wait(30)
    run.cancel()
    executor.gate.set()
    status = run.wait(60)
    assert executor.sizes("b") == [] and sum(executor.sizes("a")) == 2
    assert status.done and status.cancelled and status.faults == {}
    assert status.counts["done"] == 2 and status.counts["cancelled"] == 2 and status.counts["queued"] == 0
    assert all(isinstance(o, NotRun) and o.kind == "cancelled" for _, o in run.results())


@pytest.mark.contract
def test_wait_polls_the_events_of_a_run_another_object_drives(tmp_path: Path) -> None:
    """`Run(run_dir).wait(timeout)` returns not done at the timeout, and done once the other run's events say so."""
    placed, _ = _plan(GRAPH, [{"text": "one"}])
    executor = Gated(ResultStore(tmp_path / "store"), jobs=1)
    driver = placed.run(executor, store=executor.store)
    assert executor.entered.wait(30)
    watcher = Run(driver.run_dir)
    assert not watcher.wait(0.2, poll=0.05).done
    executor.gate.set()
    assert watcher.wait(60, poll=0.05).done and driver.wait(60).done


class Crashing(Recording):
    """A `Recording` executor whose worker dies on node `b`, leaving the stderr tail as a note."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Raise `WorkerLost` with a stderr note for node `b`."""
        if fragment.node == "b":
            error = WorkerLost("worker closed its output", returncode=-11)
            error.add_note("worker exit code -11; stderr, last 2 lines:\nloading model\nSegmentation fault: 11")
            raise error
        return super().submit(fragment)


@pytest.mark.contract
def test_a_lost_worker_fails_every_job_it_held_and_explain_shows_its_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The jobs of the batch the worker held are `failed` (infrastructure) with its stderr tail, shown by `explain`."""
    placed, _ = _plan(GRAPH, [{"text": "one"}, {"text": "two"}])
    crashing = Crashing(ResultStore(tmp_path / "store"))
    run = placed.run(crashing, store=crashing.store)
    with pytest.raises(WorkerLost):
        run.wait(60)
    status = run.status()
    assert status.done and status.counts["failed"] == 2 and status.faults == {"infrastructure": 2}
    assert cli_main(["explain", str(run.run_dir), "b"]) == 0
    out = capsys.readouterr().out
    assert "fault infrastructure: WorkerLost" in out and "Segmentation fault: 11" in out and "exit code -11" in out


class FailingProvenance(ProvStore):
    """A provenance store whose `write` fails as a full disk does."""

    def write(self, run: str | Path) -> None:
        """Raise `OSError`."""
        raise OSError(28, "No space left on device")


@pytest.mark.contract
def test_a_failed_provenance_write_fails_the_run(tmp_path: Path) -> None:
    """The run does not report done and clean when its provenance could not be written: `wait` raises."""
    placed, _ = _plan(GRAPH, [{"text": "one"}])
    store = ResultStore(tmp_path / "store", provenance=FailingProvenance())
    run = placed.run(Recording(store), store=store)
    with pytest.raises(OSError, match="No space"):
        run.wait(60)


@pytest.mark.contract
def test_explain_names_fault_and_placement(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`explain <run> <node>` prints the placement of the attempt that ran it, its events and its fault message."""
    run, _ = _started(tmp_path)
    assert cli_main(["explain", str(run.run_dir), "a"]) == 0
    out = capsys.readouterr().out
    assert "node a (runner_test.score): backend 'score'" in out and "device cpu" in out and "resource class" in out
    assert "failed a[1]" in out and "fault content" in out and "ContentFault" in out and "asked to fail" in out
    assert cli_main(["explain", str(run.run_dir), "b"]) == 0
    assert "blocked b[1]: an upstream job failed" in capsys.readouterr().out
    key = next(e.key for e in run.events() if e.kind == "failed")
    assert key is not None
    assert cli_main(["explain", run.run_dir.name[:10], key[:12]]) == 0
    assert "fault content: ContentFault" in capsys.readouterr().out
    assert cli_main(["explain", str(run.run_dir), "nope"]) == 1


@pytest.mark.contract
def test_status_command_reports_jobs_faults_and_state(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`status <run>` prints done or cancelled, the job counts, the faults by class and each node."""
    run, _ = _started(tmp_path)
    assert cli_main(["status", str(run.run_dir)]) == 0
    out = capsys.readouterr().out
    assert ": done;" in out.splitlines()[0] and "last event" in out.splitlines()[0]
    assert "done 4, failed 1, blocked 1, cancelled 0" in out
    assert "faults: content 1" in out and "node a: done 2, failed 1" in out and "node b: done 2, blocked 1" in out
    assert cli_main(["status", str(tmp_path / "nowhere")]) == 1
    assert run.manifest()["plan_hash"] in out


@pytest.mark.contract
def test_a_new_attempt_puts_the_runs_unindexed_queue_items_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An item the run's queue holds without a slot (an interrupted put) has a slot once the run starts again."""
    run, executor = _started(tmp_path)
    queue = QueueSet(run.run_dir / RUN_QUEUES).queue("cpu")
    with monkeypatch.context() as patch:
        refuse_call(patch, "symlink", "slot-", OSError(errno.EIO, "interrupted"))
        with pytest.raises(OSError, match="interrupted"):
            queue.put(fragment("interrupted"))
    assert queue.depth() == 0
    again = Run(run.run_dir, store=executor.store).plan.run(executor, store=executor.store)
    assert again.wait(60).done
    assert queue.depth() == 1
