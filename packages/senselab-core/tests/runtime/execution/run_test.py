"""A run's lifecycle over the fake stack: its directory, status, wait, results, cancel and the CLI that reads it."""

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from planted_objects_fixture import identities_in, identity_leaks, real_names
from runner_test import (  # noqa: F401 - `packaged` is the autouse fixture of the fake stack
    CHAIN_EDGES,
    CHAINED,
    ZIPPED,
    ZIPPED_ROWS,
    _graph,
    _plan,
    packaged,
)
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.events import placements, read_events
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.run import (
    InputChanged,
    InputsUnavailable,
    Output,
    PlacementFormatRetired,
    Run,
    RunPrivate,
    bind_inputs,
    load_plan,
    read_placement,
)
from senselab_core.runtime.execution.warm import PoolSession
from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.storage import owner_remedy
from senselab_core.testing.pools import served

pytest.importorskip("pydra")

GRAPH = _graph(CHAINED, CHAIN_EDGES, text="list[json]")
ROWS = [{"text": t} for t in ("one", "fail", "two")]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The run directory and the default result store are under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    yield tmp_path / "cache"


@pytest.fixture
def session(tmp_path: Path) -> Iterator[PoolSession]:
    """A pool session whose workers run here, its store at `tmp_path/store`."""
    with served(tmp_path / "pool") as (opened, _):
        yield opened


def _started(session: PoolSession, rows: list[dict[str, str]] = ROWS) -> tuple[Run, PoolSession]:
    placed, _ = _plan(GRAPH, rows)
    run = placed.run(session)
    assert run.wait(60).done
    return run, session


@pytest.mark.logic
def test_a_run_of_an_executor_with_no_workers_closes_its_placement_with_actuals(
    tmp_path: Path, session: PoolSession
) -> None:
    """A laptop-shaped run records each class's placement and, at its end, the same record with its wall time."""
    run, _ = _started(session)
    found = [e.placement for e in placements(run.events()) if e.placement is not None]
    opened = [p for p in found if p.closing is None]
    closed = [p for p in found if p.closing is not None]
    assert opened and [p.resource_class for p in closed] == [p.resource_class for p in opened]
    assert all(p.closing is not None and p.closing.wall_s is not None and p.closing.gpu_hours is None for p in closed)
    assert all(p.request is not None for p in closed)


@pytest.mark.contract
def test_a_failed_run_leaves_no_path_or_account_in_its_readable_files(tmp_path: Path, session: PoolSession) -> None:
    """A fault message that names a path is stored with the path rendered; `private/` holds the paths and is 0700."""
    run, _ = _started(session, [{"text": "leak"}, {"text": "one"}])
    failed = [e.failure for e in run.events() if e.failure is not None]
    assert failed and all("<path>" in f.message for f in failed)
    assert identity_leaks(run.run_dir, tmp_path) == []
    assert str(session.store.path) in (run.run_dir / "private" / "manifest.json").read_text()


@pytest.mark.contract
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_another_accounts_run_is_refused_by_a_typed_error_naming_the_store(
    tmp_path: Path, session: PoolSession
) -> None:
    """A run whose `private/` this account cannot read raises `RunPrivate` naming `store=`; its events still read."""
    run, _ = _started(session)
    private = run.run_dir / "private"
    private.chmod(0)
    try:
        for read in (Run(run.run_dir).manifest, Run(run.run_dir).results):
            with pytest.raises(RunPrivate, match=r"store=<its result store's root>"):
                read()
        assert Run(run.run_dir).status().done
    finally:
        private.chmod(0o700)


@pytest.mark.logic
def test_the_sweep_matches_a_name_as_a_token_and_a_path_anywhere(tmp_path: Path) -> None:
    """A name inside a longer word (`runner` in `runner_test`) is no leak; alone it is, and a path is anywhere."""
    for name in real_names():
        assert identities_in(f"{name}_test x{name} {name}-1".encode(), tmp_path) == []
        assert name in identities_in(f"by {name}.".encode(), tmp_path)
    assert identities_in(f"at {tmp_path}x".encode(), tmp_path) == [str(tmp_path)]


@pytest.mark.logic
def test_a_name_inside_a_reported_path_is_reported_once_as_the_path(tmp_path: Path) -> None:
    """A base that runs through a directory named for the account is one finding, the path, and not the name too."""
    account = sorted(real_names())[0]
    base = tmp_path / account / "work"
    base.mkdir(parents=True)
    assert identities_in(f"at {base}/x".encode(), base) == [str(base)]
    assert identities_in(f"{account} at {base}/x".encode(), base) == [account, str(base)]


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
def test_status_comes_from_events_alone_and_counts_blocked_jobs(tmp_path: Path, session: PoolSession) -> None:
    """The run directory holds no status file; blocked descendants of the failed element are counted, and done."""
    run, _ = _started(session)
    names = {p.name for p in run.run_dir.iterdir()}
    assert {"plan.json", "private", "events", "prov"} <= names
    assert {p.name for p in (run.run_dir / "private").iterdir()} >= {"inputs.json", "manifest.json", "placements"}
    assert set(run.manifest()) == {"plan_hash", "store", "pool"}
    status = run.status()
    assert status.counts == {"queued": 0, "running": 0, "done": 4, "failed": 1, "blocked": 1, "cancelled": 0}
    assert status.done and not status.cancelled and status.faults == {"content": 1} and status.broken == ()
    assert status == Run(run.run_dir).status() and status.last_event_ns is not None


@pytest.mark.contract
def test_every_run_is_a_new_attempt_with_its_own_placement_file(
    tmp_path: Path, session: PoolSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of one plan in one process have two attempt ids, two placement files and tagged events."""
    placed, _ = _plan(GRAPH, ROWS)
    first, second = placed.run(session), None
    first.wait(60)
    second = placed.run(session)
    second.wait(60)
    attempts = {e.attempt for e in read_events(first.run_dir)}
    assert len(attempts) == 2 and {p.stem for p in (first.run_dir / "private" / "placements").iterdir()} == attempts
    assert first.run_dir.name == placed.hash and placed.hash not in attempts
    monkeypatch.setenv("SENSELAB_RUN_ID", "pinned")
    third, fourth = placed.run(session), None
    third.wait(60)
    fourth = placed.run(session)
    fourth.wait(60)
    pinned = {p.stem for p in (first.run_dir / "private" / "placements").iterdir()} - attempts
    assert pinned == {"pinned", "pinned-1"}
    claimed = {p.name: p.read_text() for p in (first.run_dir / "attempts").iterdir()}
    assert sorted(claimed) == ["1", "2", "3", "4"] and claimed["3"] == "pinned" and claimed["4"] == "pinned-1"
    assert {(e.attempt, e.ordinal) for e in read_events(first.run_dir)} == {(a, int(n)) for n, a in claimed.items()}


def _live(session: PoolSession) -> tuple[Run, FaultRecord]:
    """Run `ZIPPED` with its broadcast input given as an object that is neither a path nor JSON."""
    held = FaultRecord(fault="content", type="Held", message="in memory", gap=False, charged=True)
    placed, _ = _plan(ZIPPED, ZIPPED_ROWS, broadcast={"both": held})
    run = placed.run(session)
    assert run.wait(60).done
    return run, held


@pytest.mark.contract
def test_an_input_given_as_an_object_runs_in_process_and_is_recorded_as_memory(
    tmp_path: Path, session: PoolSession
) -> None:
    """The object is admitted live; `inputs.json` records `{"memory": <identity>}` for it."""
    run, _ = _live(session)
    assert run.status().counts["done"] == 9 and run.status().faults == {}
    assert all(isinstance(o, Output) for _, o in run.results())
    shared = json.loads((run.run_dir / "private" / "inputs.json").read_text())["shared"]
    [identity] = {i for row in load_plan(run.run_dir).manifest.rows for p, i in row.inputs.items() if p == "both"}
    assert shared == {"both": {"memory": identity}}


@pytest.mark.contract
def test_a_run_from_disk_without_its_live_inputs_raises_before_writing(tmp_path: Path, session: PoolSession) -> None:
    """Re-run from the directory, the memory input is `InputsUnavailable`, and no attempt or file is written."""
    run, _ = _live(session)
    with pytest.raises(InputsUnavailable, match="both") as raised:
        bind_inputs(run.run_dir)
    assert {port for _, port in raised.value.missing} == {"both"} and len(raised.value.missing) == 3
    before = sorted(str(p.relative_to(run.run_dir)) for p in run.run_dir.rglob("*") if "pydra" not in p.parts)
    with pytest.raises(InputsUnavailable):
        Run(run.run_dir).plan.run(session)
    after = sorted(str(p.relative_to(run.run_dir)) for p in run.run_dir.rglob("*") if "pydra" not in p.parts)
    assert after == before
    run.run_dir.rename(tmp_path / "moved")
    with pytest.raises(InputsUnavailable):
        load_plan(tmp_path / "moved").run(session)
    assert not run.run_dir.exists()


@pytest.mark.contract
def test_a_second_attempt_is_served_from_the_store_and_logged_as_cache_hits(
    tmp_path: Path, session: PoolSession
) -> None:
    """Resumed from the directory alone, the jobs the store holds are `cache_hit` with their reuse policy."""
    run, executor = _started(session)
    fresh = Run(run.run_dir, store=session.store.path)
    again = fresh.plan.run(session)
    assert again.wait(60).done
    latest = max(read_events(run.run_dir), key=lambda e: e.t_ns).attempt
    hits = [e for e in read_events(run.run_dir) if e.kind == "cache_hit"]
    assert hits and {e.attempt for e in hits} == {latest} and {e.reuse for e in hits} == {"exact"}
    assert not [e for e in read_events(run.run_dir) if e.kind == "finished" and e.attempt == latest]


@pytest.mark.contract
def test_results_come_lazily_from_the_run_in_record_order(tmp_path: Path, session: PoolSession) -> None:
    """A failed record gives its fault; a run opened over the directory, with no plan in memory, reads the same."""
    run, _ = _started(session)
    results = run.results()
    assert results.run is run and [r.id for r, _ in results] == ["r0", "r1", "r2"]
    items = list(results)
    assert isinstance(items[0][1], Output) and isinstance(items[2][1], Output)
    assert isinstance(items[1][1], FaultRecord) and items[1][1].type == "ContentFault"
    again = list(Run(run.run_dir, store=session.store.path).results())
    assert [(r.id, type(o).__name__) for r, o in again] == [(r.id, type(o).__name__) for r, o in items]


@pytest.mark.contract
def test_inputs_are_bound_from_the_run_directory_and_a_changed_one_is_refused(
    tmp_path: Path, session: PoolSession
) -> None:
    """`bind_inputs` re-binds every source and refuses one whose identity differs from the manifest's."""
    run, _ = _started(session)
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
def test_a_fresh_process_reads_and_binds_a_run_from_its_directory(tmp_path: Path, session: PoolSession) -> None:
    """With no plan in memory, a new process reads the results and status and binds the inputs of the run."""
    run, _ = _started(session)
    done = subprocess.run(
        [sys.executable, "-c", _FRESH, str(run.run_dir)], capture_output=True, text=True, check=True, timeout=120
    )
    found = json.loads(done.stdout)
    assert found["ids"] == ["r0", "r1", "r2"] and found["kinds"] == ["Output", "FaultRecord", "Output"]
    assert found["done"] is True and found["values"] == 3 and found["hash"] == run.run_dir.name


@pytest.mark.contract
def test_explain_names_fault_and_placement(
    tmp_path: Path, session: PoolSession, capsys: pytest.CaptureFixture[str]
) -> None:
    """`explain <run> <node>` prints the placement of the attempt that ran it, its events and its fault message."""
    run, _ = _started(session)
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
def test_status_command_reports_jobs_faults_and_state(
    tmp_path: Path, session: PoolSession, capsys: pytest.CaptureFixture[str]
) -> None:
    """`status <run>` prints done or cancelled, the job counts, the faults by class and each node."""
    run, _ = _started(session)
    assert cli_main(["status", str(run.run_dir)]) == 0
    out = capsys.readouterr().out
    assert ": done;" in out.splitlines()[0] and "last event" in out.splitlines()[0]
    assert "done 4, failed 1, blocked 1, cancelled 0" in out
    assert "faults: content 1" in out and "node a: done 2, failed 1" in out and "node b: done 2, blocked 1" in out
    assert cli_main(["status", str(tmp_path / "nowhere")]) == 1
    assert run.manifest()["plan_hash"] in out


@pytest.mark.logic
def test_a_placement_of_a_retired_format_names_the_run_as_one_of_an_older_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run directory written at `placement@2.0` says so, by name, to `read_placement` and to `senselab explain`."""
    run_dir = tmp_path / "abc123"
    (run_dir / "placements").mkdir(parents=True)
    path = run_dir / "placements" / "a1.json"
    path.write_text(json.dumps({"format": "placement@2.0", "cost": {}}))
    with pytest.raises(PlacementFormatRetired, match=r"run abc123.*placement@2\.0.*older build"):
        read_placement(path)
    path.write_text(json.dumps({"format": "placement@9.0"}))
    with pytest.raises(ValueError) as unknown:
        read_placement(path)
    assert not isinstance(unknown.value, PlacementFormatRetired)
