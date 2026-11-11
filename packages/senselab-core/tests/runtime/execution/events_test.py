"""The event log and the status and results derived from it: no runner, no executor."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from senselab_core.kernel.dataset import Record
from senselab_core.runtime.execution.events import (
    Event,
    EventLog,
    JobState,
    read_events,
    scan_events,
    status_from,
    terminal_events,
)
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.execution.run import MissingExtra, NotRun, Output, ResultSet

PLAN = "ab" * 32
CONTENT = FaultRecord(fault="content", type="ContentFault", message="bad input", gap=False, charged=True)
INFRA = FaultRecord(fault="infrastructure", type="WorkerLost", message="gone", gap=False, charged=False)


def _event(  # noqa: PLR0913 - the fields of an event
    kind: str,
    element: tuple[int, ...] = (),
    node: str = "a",
    t_ns: int = 1,
    seq: int = 0,
    attempt: str = "a1",
    writer: str = "w",
    **fields: object,
) -> Event:
    job = JobRef(plan_hash=PLAN, node=node, element=element)
    ordinal = int(attempt[1:])
    return Event(
        kind=kind, node=node, job=job, t_ns=t_ns, seq=seq, attempt=attempt, ordinal=ordinal, writer=writer, **fields
    )


@pytest.mark.logic
def test_status_derived_from_events_only() -> None:
    """Counts, faults, broken envs and `done` come from the events alone; a job is in its last terminal state."""
    events = [
        _event("queued", (0,), t_ns=1, seq=0),
        _event("queued", (1,), t_ns=2, seq=1),
        _event("queued", (2,), t_ns=3, seq=2),
        _event("started", (0,), t_ns=4, seq=3),
        _event("finished", (0,), t_ns=5, seq=4),
        _event("failed", (1,), t_ns=6, seq=5, fault="content", failure=CONTENT, env="host"),
        _event("verify", (0,), t_ns=7, seq=6),
    ]
    mid = status_from(events, broken_after=2)
    assert mid.counts == {"queued": 1, "running": 0, "done": 1, "failed": 1, "blocked": 0, "cancelled": 0}
    assert mid.faults == {"content": 1} and mid.broken == () and not mid.done and mid.last_event_ns == 7
    events += [
        _event("started", (2,), t_ns=8, seq=7),
        _event("failed", (2,), t_ns=9, seq=8, fault="infrastructure", failure=INFRA, env="venv:x"),
        _event("queued", (2,), t_ns=10, seq=9, attempt="a2"),
        _event("failed", (2,), t_ns=11, seq=10, attempt="a2", fault="infrastructure", failure=INFRA, env="venv:x"),
    ]
    end = status_from(events, broken_after=2)
    assert end.counts["done"] == 1 and end.counts["failed"] == 2 and end.done and not end.cancelled
    assert end.faults == {"content": 1, "infrastructure": 2}
    [broken] = end.broken
    assert (broken.env, broken.failures, broken.fault) == ("venv:x", 2, INFRA)
    assert status_from(events, broken_after=3).broken == ()
    assert end.nodes["a"] == end.counts and status_from([]).done is False and status_from([]).last_event_ns is None


@pytest.mark.logic
def test_cancelled_and_blocked_jobs_end_the_run_without_faults() -> None:
    """A cancelled job is terminal and not a fault, but the run says it was cancelled; blocked jobs are terminal."""
    events = [
        _event("queued", (0,), seq=0),
        _event("queued", (1,), seq=1),
        _event("queued", (2,), node="b", seq=2),
        _event("finished", (0,), seq=3),
        _event("cancelled", (1,), seq=4),
        _event("blocked", (2,), node="b", seq=5),
    ]
    status = status_from(events)
    assert status.done and status.cancelled and status.faults == {}
    assert status.counts["cancelled"] == 1 and status.counts["blocked"] == 1 and status.counts["done"] == 1
    assert not status_from(events[:3]).done and status_from(events[:5]).cancelled


@pytest.mark.logic
def test_yielded_job_is_queued_again_and_a_cache_hit_is_done() -> None:
    """A `yielded` job goes back to the queue; a `cache_hit` counts as done."""
    events = [
        _event("claimed", (0,), seq=0),
        _event("yielded", (0,), seq=1),
        _event("cache_hit", (1,), seq=2, reuse="exact"),
    ]
    assert status_from(events).counts["queued"] == 1 and status_from(events).counts["done"] == 1


@pytest.mark.logic
def test_a_later_attempt_overrides_an_earlier_one_and_a_crashed_run_keeps_its_last_state() -> None:
    """A job failed in attempt one and finished in attempt two is done; a run whose writer died is not done."""
    first = [
        _event("queued", seq=0),
        _event("started", seq=1, t_ns=2),
        _event("failed", seq=2, t_ns=3, fault="content", failure=CONTENT),
    ]
    second = [_event("queued", seq=0, attempt="a2", t_ns=10), _event("finished", seq=1, attempt="a2", t_ns=11)]
    assert status_from([*first, *second]).counts["done"] == 1
    crashed = status_from([*first[:2]])
    assert crashed.counts["running"] == 1 and not crashed.done and crashed.last_event_ns == 2


@pytest.mark.logic
def test_a_result_is_never_hidden_by_another_writers_not_run_event() -> None:
    """A worker's `finished` at seq 1 and the controller's later `cancelled` at seq 6 for one job give `finished`."""
    worker = [_event("started", seq=0, writer="w"), _event("finished", seq=1, writer="w", t_ns=2)]
    controller = [_event("queued", seq=0, writer="c"), _event("cancelled", seq=6, writer="c", t_ns=9)]
    for events in ([*controller, *worker], [*worker, *controller]):
        status = status_from(events)
        assert status.counts["done"] == 1 and status.counts["cancelled"] == 0 and not status.cancelled
    failed = _event("failed", seq=1, writer="w", fault="content", failure=CONTENT)
    blocked = _event("blocked", seq=9, writer="c")
    assert status_from([failed, blocked]).counts["failed"] == 1 and status_from([blocked, failed]).counts["failed"] == 1


@pytest.mark.logic
def test_a_later_attempt_wins_by_its_claim_whatever_its_clock() -> None:
    """Attempt two, claimed second, overrides attempt one even though its clock runs behind."""
    first = [_event("queued", t_ns=100), _event("failed", seq=1, t_ns=140, fault="content", failure=CONTENT)]
    second = [_event("queued", attempt="a2", t_ns=60), _event("finished", seq=1, attempt="a2", t_ns=70)]
    for events in ([*first, *second], [*second, *first]):
        assert status_from(events).counts == {**status_from(second).counts}
        assert status_from(events).counts["done"] == 1 and status_from(events).faults == {"content": 1}


@pytest.mark.logic
def test_queued_or_running_across_writers_reads_claims_not_clocks() -> None:
    """A coordinator queues and a pool worker claims: running from the claim, whatever either writer's clock says.

    A retry the coordinator logs `failed` then `queued` reads as queued, not failed, until a worker claims it again.
    """
    queued = _event("queued", writer="c", seq=0, t_ns=50)
    claimed = _event("claimed", writer="w", seq=0, t_ns=10)
    retried = [
        _event("failed", writer="c", seq=1, t_ns=60, fault="infrastructure", failure=INFRA, env="host"),
        _event("queued", writer="c", seq=2, t_ns=61),
    ]
    again = _event("claimed", writer="v", seq=0, t_ns=5)
    finished = _event("finished", writer="c", seq=3, t_ns=70)
    cases: list[tuple[list[Event], JobState]] = [
        ([queued], "queued"),
        ([claimed, queued], "running"),
        ([*retried, claimed, queued], "queued"),
        ([again, *retried, claimed, queued], "running"),
        ([finished, again, *retried, claimed, queued], "done"),
    ]
    for events, state in cases:
        assert status_from(events).counts[state] == 1, state
    assert status_from([*retried, claimed, queued]).faults == {"infrastructure": 1}


@pytest.mark.logic
def test_a_stale_epoch_never_overrides_the_live_one_whatever_its_ordinal() -> None:
    """A stale coordinator of epoch 1 that claimed ordinal 2 cannot fail or block what epoch 2 (ordinal 1) finished."""
    stale = [
        _event("queued", writer="s", attempt="a2", epoch=1),
        _event("failed", writer="s", seq=1, attempt="a2", epoch=1, fault="infrastructure", failure=INFRA),
        _event("blocked", (0,), node="b", writer="s", seq=2, attempt="a2", epoch=1),
    ]
    live = [
        _event("queued", writer="l", attempt="a1", epoch=2),
        _event("finished", writer="l", seq=1, attempt="a1", epoch=2),
        _event("queued", (0,), node="b", writer="l", seq=2, attempt="a1", epoch=2),
    ]
    for events in ([*stale, *live], [*live, *stale]):
        status = status_from(events)
        assert status.counts["done"] == 1 and status.counts["queued"] == 1 and not status.done
        assert status.counts["failed"] == 0 and status.counts["blocked"] == 0


@pytest.mark.logic
def test_a_resume_in_progress_shows_an_earlier_result_done_only_under_its_key() -> None:
    """A job finished in epoch 1 is done while epoch 2 has queued it under the same key; under another key it is not.

    The current epoch's own terminal state wins, and a job with no event of the run's current epoch is queued.
    """
    key, other = "1" * 64, "2" * 64
    first = [_event("queued", key=key, epoch=1), _event("finished", seq=1, key=key, epoch=1)]
    same = _event("queued", attempt="a2", writer="v", key=key, epoch=2)
    moved = _event("queued", attempt="a2", writer="v", key=other, epoch=2)
    assert status_from([*first, same]).counts == status_from(first).counts and status_from([*first, same]).done
    replaced = status_from([*first, moved])
    assert replaced.counts["queued"] == 1 and replaced.counts["done"] == 0 and not replaced.done
    assert terminal_events([*first, moved]) == {} and len(terminal_events([*first, same])) == 1
    unkeyed = _event("queued", attempt="a2", writer="v", epoch=2)
    assert status_from([*first, unkeyed]).counts["queued"] == 1
    blocked = _event("blocked", attempt="a2", writer="v", seq=1, epoch=2)
    assert status_from([*first, same, blocked]).counts["blocked"] == 1
    assert status_from(first, epoch=2).counts["queued"] == 1 and terminal_events(first, epoch=2) == {}


@pytest.mark.contract
def test_two_writers_one_event_log(tmp_path: Path) -> None:
    """Two writers on one run append to their own files; reading keeps each writer's order and merges by time."""
    first = EventLog(tmp_path, "a1", 1, clock=iter([1, 3, 5]).__next__)
    second = EventLog(tmp_path, "a1", 1, clock=iter([2, 4]).__next__)
    assert first.writer != second.writer
    jobs = [JobRef(plan_hash=PLAN, node="a", element=(i,)) for i in range(2)]
    first.record("queued", "a", job=jobs[0])
    second.record("queued", "a", job=jobs[1])
    first.record("started", "a", job=jobs[0])
    second.record("started", "a", job=jobs[1])
    first.record("finished", "a", job=jobs[0], key=PLAN)
    files = sorted(p.name for p in (tmp_path / "events").iterdir())
    assert files == sorted([f"{first.writer}.jsonl", f"{second.writer}.jsonl"])
    merged = read_events(tmp_path)
    assert [e.t_ns for e in merged] == [1, 2, 3, 4, 5]
    assert [(e.kind, e.seq) for e in merged if e.job == jobs[0]] == [("queued", 0), ("started", 1), ("finished", 2)]
    assert status_from(merged).counts["running"] == 1 and status_from(merged).counts["done"] == 1


@pytest.mark.contract
def test_a_clock_stepping_back_or_skewed_across_writers_changes_no_status(tmp_path: Path) -> None:
    """A writer's own order is kept whatever its clock did, and a skewed second writer cannot undo a terminal event."""
    job = JobRef(plan_hash=PLAN, node="a", element=(0,))
    controller = EventLog(tmp_path, "a1", 1, clock=iter([100, 50]).__next__)
    worker = EventLog(tmp_path, "a1", 1, clock=iter([60, 70]).__next__)
    controller.record("queued", "a", job=job)
    worker.record("started", "a", job=job)
    worker.record("finished", "a", job=job, key=PLAN, output="output:" + "0" * 64)
    controller.record("queued", "a", job=job)
    merged = read_events(tmp_path)
    assert [e.seq for e in merged if e.t_ns in (100, 50)] == [0, 1]
    assert status_from(merged).counts["done"] == 1


@pytest.mark.contract
def test_a_torn_last_line_and_a_corrupt_middle_line_are_quarantined(tmp_path: Path) -> None:
    """A crash mid-write leaves the writer's last line cut short; damage elsewhere is skipped the same way."""
    log = EventLog(tmp_path, "a1", 1, clock=iter([1, 2]).__next__)
    log.record("queued", "a")
    log.record("started", "a")
    with log.path.open("a") as handle:
        handle.write('{"format":"run-event@2.0","kind":"fini')
    assert [e.kind for e in read_events(tmp_path)] == ["queued", "started"]
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[0], "not json", lines[1]]) + "\n")
    assert [e.kind for e in read_events(tmp_path)] == ["queued", "started"]
    assert [(q.writer, q.line) for q in scan_events(tmp_path).quarantined] == [(log.writer, 1)]


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
def test_to_frame_without_pandas_raises_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """With pandas absent, `to_frame` raises `MissingExtra("pandas")`."""
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(MissingExtra) as raised:
        _results().to_frame()
    assert raised.value.extra == "pandas"


@pytest.mark.logic
def test_result_set_reads_an_entry_only_when_it_is_asked_for_and_once() -> None:
    """An entry given as a callable is not called by building, `len` or `failures` of others, and is called once."""
    calls: list[str] = []

    def entry(name: str) -> Callable[[], Output]:
        def read() -> Output:
            calls.append(name)
            return Output({"v": name})

        return read

    results = ResultSet([(Record(id=n), entry(n)) for n in ("a", "b")], {"s": entry("shared")})
    assert len(results) == 2 and calls == []
    assert results[1][1] == Output({"v": "b"}) and results[1][1] == Output({"v": "b"}) and calls == ["b"]
    assert results.outputs("v") == ["a", "b"] and calls == ["b", "a"]
    assert results.shared["s"] == Output({"v": "shared"}) and calls[-1] == "shared"


@pytest.mark.logic
def test_not_run_records_are_neither_outputs_nor_faults() -> None:
    """A blocked or cancelled record stands in place in `outputs` and is not among the failures."""
    cancelled = NotRun("cancelled")
    results = ResultSet([(Record(id="a"), Output({"v": 1})), (Record(id="b"), cancelled)])
    assert results.outputs("v") == [1, cancelled] and results.failures == []
