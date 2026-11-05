"""The event log and the status and results derived from it: no runner, no executor."""

import sys
from pathlib import Path

import pytest
from senselab_core.kernel.dataset import Record
from senselab_core.runtime.execution.events import Event, EventLog, read_events, status_from
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.execution.run import MissingExtra, Output, ResultSet

PLAN = "ab" * 32
CONTENT = FaultRecord(fault="content", type="ContentFault", message="bad input", gap=False, charged=True)
INFRA = FaultRecord(fault="infrastructure", type="WorkerLost", message="gone", gap=False, charged=False)


def _event(kind: str, element: tuple[int, ...] = (), node: str = "a", t_ns: int = 1, **fields: object) -> Event:
    job = JobRef(plan_hash=PLAN, node=node, element=element)
    return Event(kind=kind, node=node, job=job, t_ns=t_ns, **fields)


@pytest.mark.logic
def test_status_derived_from_events_only() -> None:
    """Counts, faults, broken envs and `done` come from the events alone; a job is in its latest event's state."""
    events = [
        _event("queued", (0,), t_ns=1),
        _event("queued", (1,), t_ns=2),
        _event("queued", (2,), t_ns=3),
        _event("started", (0,), t_ns=4),
        _event("finished", (0,), t_ns=5),
        _event("failed", (1,), t_ns=6, fault="content", failure=CONTENT, env="host"),
        _event("verify", (0,), t_ns=7),
    ]
    mid = status_from(events, broken_after=2)
    assert mid.counts == {"queued": 1, "running": 0, "done": 1, "failed": 1}
    assert mid.faults == {"content": 1} and mid.broken == () and not mid.done
    events += [
        _event("started", (2,), t_ns=8),
        _event("failed", (2,), t_ns=9, fault="infrastructure", failure=INFRA, env="venv:x"),
        _event("queued", (2,), t_ns=10),
        _event("failed", (2,), t_ns=11, fault="infrastructure", failure=INFRA, env="venv:x"),
    ]
    end = status_from(events, broken_after=2)
    assert end.counts == {"queued": 0, "running": 0, "done": 1, "failed": 2} and end.done
    assert end.faults == {"content": 1, "infrastructure": 2}
    [broken] = end.broken
    assert (broken.env, broken.failures, broken.fault) == ("venv:x", 2, INFRA)
    assert status_from(events, broken_after=3).broken == ()
    assert end.nodes["a"] == end.counts and status_from([]).done is False


@pytest.mark.logic
def test_yielded_job_is_queued_again_and_a_cache_hit_is_done() -> None:
    """A `yielded` job goes back to the queue; a `cache_hit` counts as done."""
    events = [
        _event("claimed", (0,), t_ns=1),
        _event("yielded", (0,), t_ns=2),
        _event("cache_hit", (1,), t_ns=3, reuse="exact"),
    ]
    assert status_from(events).counts == {"queued": 1, "running": 0, "done": 1, "failed": 0}


@pytest.mark.contract
def test_two_writers_one_event_log(tmp_path: Path) -> None:
    """Two writers on one run append to their own files; reading merges them by time, and neither sees the other."""
    first, second = EventLog(tmp_path, clock=iter([1, 3, 5]).__next__), EventLog(tmp_path, clock=iter([2, 4]).__next__)
    assert first.writer != second.writer
    jobs = [JobRef(plan_hash=PLAN, node="a", element=(i,)) for i in range(5)]
    first.record("queued", "a", job=jobs[0])
    second.record("queued", "a", job=jobs[1])
    first.record("started", "a", job=jobs[0])
    second.record("started", "a", job=jobs[1])
    first.record("finished", "a", job=jobs[0], key=PLAN)
    files = sorted(p.name for p in (tmp_path / "events").iterdir())
    assert files == sorted([f"{first.writer}.jsonl", f"{second.writer}.jsonl"])
    assert [
        len((tmp_path / "events" / f).read_text().splitlines())
        for f in (f"{first.writer}.jsonl", f"{second.writer}.jsonl")
    ] == [3, 2]
    merged = read_events(tmp_path)
    assert [e.t_ns for e in merged] == [1, 2, 3, 4, 5]
    assert [e.kind for e in merged] == ["queued", "queued", "started", "started", "finished"]
    assert status_from(merged).counts == {"queued": 0, "running": 1, "done": 1, "failed": 0}


@pytest.mark.contract
def test_a_torn_last_line_is_skipped_and_a_corrupt_middle_line_raises(tmp_path: Path) -> None:
    """A crash mid-write leaves the writer's last line cut short, which is skipped; damage elsewhere is an error."""
    log = EventLog(tmp_path, clock=iter([1, 2]).__next__)
    log.record("queued", "a")
    log.record("started", "a")
    with log.path.open("a") as handle:
        handle.write('{"format":"run-event@1.0","kind":"fini')
    assert [e.kind for e in read_events(tmp_path)] == ["queued", "started"]
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[0], "not json", lines[1]]) + "\n")
    with pytest.raises(ValueError):
        read_events(tmp_path)


def _results() -> ResultSet:
    records = [
        Record(id="sub-01/a.wav", entities={"subject": "01", "task": "x"}),
        Record(id="sub-02/a.wav", entities={"subject": "02", "task": "x"}),
        Record(id="sub-03/a.wav", entities={"subject": "03", "task": "x"}),
    ]
    return ResultSet(
        [
            (records[0], Output({"score": 0.5, "label": "a"})),
            (records[1], CONTENT),
            (records[2], Output({"score": 0.25, "label": "b"})),
        ]
    )


@pytest.mark.logic
def test_result_set_iterates_in_record_order_and_gives_outputs_by_port() -> None:
    """Items come in record order; `outputs(port)` puts a failed record's fault in its place."""
    results = _results()
    assert [r.id for r, _ in results] == ["sub-01/a.wav", "sub-02/a.wav", "sub-03/a.wav"] and len(results) == 3
    assert results.outputs("score") == [0.5, CONTENT, 0.25]
    assert [r.id for r, _ in results.failures] == ["sub-02/a.wav"]


@pytest.mark.logic
def test_to_frame_indexed_by_entities() -> None:
    """The frame is indexed by the records' entities, one column per output, and a failed row holds its fault class."""
    pandas = pytest.importorskip("pandas")
    frame = _results().to_frame()
    assert list(frame.index.names) == ["subject", "task"]
    assert list(frame.index) == [("01", "x"), ("02", "x"), ("03", "x")]
    assert list(frame.columns) == ["id", "label", "score", "fault"]
    assert frame.loc[("01", "x"), "score"] == 0.5 and frame.loc[("03", "x"), "label"] == "b"
    assert frame.loc[("02", "x"), "fault"] == "content" and pandas.isna(frame.loc[("02", "x"), "score"])


@pytest.mark.logic
def test_to_frame_without_entities_is_indexed_by_id() -> None:
    """Records with no entities are indexed by their ids."""
    pytest.importorskip("pandas")
    frame = ResultSet([(Record(id="r0"), Output({"v": 1}))]).to_frame()
    assert list(frame.index) == ["r0"] and frame.index.name == "id"


@pytest.mark.logic
def test_to_frame_without_pandas_raises_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """With pandas absent, `to_frame` raises `MissingExtra("pandas")`."""
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(MissingExtra) as raised:
        _results().to_frame()
    assert raised.value.extra == "pandas"
