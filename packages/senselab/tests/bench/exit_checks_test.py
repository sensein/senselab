"""The Phase 3 exit-check harness: plan comparison, identities, bounds, resume accounting and the agent's call."""

import importlib.util
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from senselab_core.kernel.declare import REGISTRY, BackendSpec, TaskSpec
from senselab_core.runtime import fake_backend
from senselab_core.runtime.call import Packaged, TaskCall, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.events import Event, EventKind, EventLog, read_events
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import mit_lock_records

ROOT = Path(__file__).resolve().parents[4]
EXIT = ROOT / "benchmarks" / "exit"
HASH = "a" * 64


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exitlib = _module("exitlib", EXIT / "exitlib.py")
agent_plan = _module("agent_plan_under_test", EXIT / "agent_plan.py")
interrupt_resume = _module("interrupt_resume_under_test", EXIT / "interrupt_resume.py")


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.fixture
def fake_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TaskSpec]:
    """The fake echo task as the packaged stack, over a cache root of its own."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    yield fake_backend.ECHO
    use_packaged(previous)


def _plan(salt: str, profile: ExecutionProfile) -> Any:  # noqa: ANN401 - a Plan
    call = TaskCall(task="fake.echo", task_version="1", inputs={}, params={"salt": salt})
    return plan(
        call, from_records({"id": f"r{i}", "inputs": {"payload": {"n": i}}} for i in range(3)), execution=profile
    )


@pytest.mark.logic
def test_field_diff_lists_leaves_by_path_and_absent_fields() -> None:
    """A changed leaf, a changed list element and a field only one side has are each a difference."""
    left = {"a": {"b": 1, "c": [1, 2]}, "only": 1, "same": "x"}
    right = {"a": {"b": 2, "c": [1, 3]}, "same": "x", "extra": 0}
    found = {d.path: (d.left, d.right) for d in exitlib.field_diff(left, right)}
    assert found == {
        "a.b": (1, 2),
        "a.c[1]": (2, 3),
        "extra": (exitlib.ABSENT, 0),
        "only": (1, exitlib.ABSENT),
    }
    assert exitlib.field_diff(left, left) == []


@pytest.mark.logic
def test_plans_over_two_profiles_differ_only_in_placement(fake_stack: TaskSpec) -> None:  # noqa: ARG001
    """One logical plan under two profiles has one hash; a changed param is a difference outside placement."""
    laptop = ExecutionProfile.load("laptop")
    other = exitlib.profile_for("pool-local", laptop, SimpleNamespace(pool_local_workers=2))
    found = exitlib.compare_plans(_plan("x", laptop), _plan("x", other))
    assert found and all(d.path.startswith("placement") for d in found)
    with pytest.raises(exitlib.PlanIdentityError, match="logical") as refused:
        exitlib.compare_plans(_plan("x", laptop), _plan("y", laptop))
    assert any(d.path.startswith("logical") for d in refused.value.outside)


def _record(log: EventLog, kind: EventKind, element: int, key: str, output: str | None = None) -> None:
    job = JobRef(plan_hash=HASH, node="call", element=(element,))
    log.record(kind, "call", job=job, key=key * 64, output=output)


def _output(index: str) -> str:
    return "output:" + index * 64


@pytest.mark.logic
def test_output_identities_are_each_records_result_and_compare_names_the_difference(tmp_path: Path) -> None:
    """Terminal result events give record id to output; two maps differing in one id, one absent, are named."""
    log = EventLog(tmp_path, "a0", 1)
    for i in range(3):
        _record(log, "queued", i, "e")
        _record(log, "finished", i, str(i), _output(str(i)))
    rows = [SimpleNamespace(id=f"sub-{i}") for i in range(3)]
    found = exitlib.output_identities(read_events(tmp_path), rows)
    assert found == {f"sub-{i}": _output(str(i)) for i in range(3)}
    other = {**found, "sub-1": _output("f")}
    del other["sub-2"]
    assert exitlib.compare_identities(found, found) == []
    assert exitlib.compare_identities(found, other) == [
        "sub-2: only on the left",
        f"sub-1: {_output('1')} != {_output('f')}",
    ]


def _history(
    tmp_path: Path, committed: list[int], resumed: dict[EventKind, list[int]], keys: list[str]
) -> tuple[list[Event], list[Event]]:
    """Return events before a resume (the `committed` items finished) and after it (`resumed` kind to items)."""
    log = EventLog(tmp_path, "a0", 1, epoch=1)
    for i in committed:
        _record(log, "queued", i, keys[i])
        _record(log, "finished", i, keys[i], _output(keys[i]))
    before = read_events(tmp_path)
    again = EventLog(tmp_path, "a1", 2, epoch=2)
    for kind, items in resumed.items():
        for i in items:
            _record(again, kind, i, keys[i], _output(keys[i]))
    return before, read_events(tmp_path)


KEYS = ["0", "1", "2", "3", "4", "4"]
"""Six items; the last two share a key, an in-run duplicate."""


@pytest.mark.logic
def test_resume_executions_are_the_uncommitted_keys_and_the_rest_are_reuses(tmp_path: Path) -> None:
    """Two committed items are reused, three distinct uncommitted keys execute, the duplicate is a reuse."""
    before, after = _history(tmp_path, [0, 1], {"cache_hit": [0, 1, 5], "finished": [2, 3, 4]}, KEYS)
    found = exitlib.account_resume(before, after, 6)
    assert (found.committed, found.executions, found.late, found.reuses) == (2, 3, 0, 3)


@pytest.mark.logic
def test_an_item_finished_between_the_last_event_and_the_interrupt_is_a_late_hit(tmp_path: Path) -> None:
    """A key the resume finds in the store, not committed in events, is counted late, not as an execution."""
    before, after = _history(tmp_path, [0, 1], {"cache_hit": [0, 1, 2, 5], "finished": [3, 4]}, KEYS)
    found = exitlib.account_resume(before, after, 6)
    assert (found.executions, found.late, found.reuses) == (2, 1, 4)


@pytest.mark.logic
@pytest.mark.parametrize(
    ("committed", "resumed", "message"),
    [
        ([0, 1], {"finished": [0, 1, 2, 3, 4], "cache_hit": [5]}, "executed again"),
        ([0, 1], {"cache_hit": [0, 1, 5], "finished": [2, 3, 4, 5]}, "executed 2 times"),
        ([0, 1], {"cache_hit": [0, 1], "finished": [2, 3]}, "have no result"),
        ([], {"finished": [0, 1, 2, 3, 4], "cache_hit": [5]}, "did not follow an interrupt"),
        ([0, 1, 2, 3, 4, 5], {"cache_hit": [0, 1, 2, 3, 4, 5]}, "did not follow an interrupt"),
    ],
)
def test_resume_accounting_refuses_what_contradicts_the_check(
    tmp_path: Path, committed: list[int], resumed: dict[EventKind, list[int]], message: str
) -> None:
    """Re-executing a committed item or key, executing a key twice, a missing result and no interrupt each fail."""
    before, after = _history(tmp_path, committed, resumed, KEYS)
    with pytest.raises(exitlib.ResumeMismatch, match=message):
        exitlib.account_resume(before, after, 6)


@pytest.mark.logic
def test_a_resume_that_resolved_a_ref_is_refused(tmp_path: Path) -> None:
    """Manifests of any attempt under the cache root count: an unchanged snapshot passes, a new entry or file fails."""
    manifest = tmp_path / "2x" / "runs" / "100" / "resolutions@2.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"old": "' + "a" * 40 + '"}', encoding="utf-8")
    before = exitlib.resolution_snapshot(tmp_path)
    exitlib.assert_no_resolution(before, exitlib.resolution_snapshot(tmp_path))
    manifest.write_text('{"old": "' + "a" * 40 + '", "org/model": "' + "b" * 40 + '"}', encoding="utf-8")
    with pytest.raises(exitlib.ResumeMismatch, match="org/model"):
        exitlib.assert_no_resolution(before, exitlib.resolution_snapshot(tmp_path))
    manifest.write_text('{"old": "' + "a" * 40 + '"}', encoding="utf-8")
    other = tmp_path / "2x" / "runs" / "200" / "resolutions@2.json"
    other.parent.mkdir(parents=True)
    other.write_text('{"worker/model": "' + "c" * 40 + '"}', encoding="utf-8")
    with pytest.raises(exitlib.ResumeMismatch, match="worker/model"):
        exitlib.assert_no_resolution(before, exitlib.resolution_snapshot(tmp_path))


@pytest.mark.logic
def test_events_of_a_surviving_worker_of_the_interrupted_epoch_are_not_the_resumes(tmp_path: Path) -> None:
    """A `finished` the old epoch logs after the snapshot is not an execution of the resume."""
    before, _ = _history(tmp_path, [0, 1], {}, KEYS)
    stray = EventLog(tmp_path, "a0", 1, epoch=1)
    _record(stray, "finished", 2, KEYS[2], _output(KEYS[2]))
    resumed = EventLog(tmp_path, "a1", 2, epoch=2)
    served: dict[EventKind, list[int]] = {"cache_hit": [0, 1, 2, 5], "finished": [3, 4]}
    for kind, items in served.items():
        for i in items:
            _record(resumed, kind, i, KEYS[i], _output(KEYS[i]))
    found = exitlib.account_resume(before, read_events(tmp_path), 6)
    assert (found.executions, found.late) == (2, 1)


@pytest.mark.logic
def test_resident_bytes_sum_a_process_and_its_descendants() -> None:
    """The tree under the root counts, an unrelated process and the root's parent do not; rows are KiB."""
    text = "1 0 500\n10 1 100\n11 10 50\n12 1 25\n20 0 9999\n\n"
    assert exitlib.tree_resident(text, 10) == exitlib.KIB * 150
    assert exitlib.tree_resident(text, 1) == exitlib.KIB * 675
    assert exitlib.tree_resident(text, 99) == 0


@pytest.mark.logic
def test_the_watch_records_peaks_and_the_first_passed_resident_bound() -> None:
    """Peaks are the largest samples; a sample over the bound is named once and later samples do not replace it."""
    samples = iter([10, 30, 20, 50])
    watch = exitlib.Watch(exitlib.Bounds(None, 25), 1.0, resident=lambda: next(samples), load=lambda: 0.5)
    for _ in range(3):
        watch.sample()
    first = watch.exceeded
    watch.sample()
    assert (watch.peak_resident, watch.peak_load, watch.exceeded) == (50, 0.5, first)
    assert first is not None and "30" in first
    assert exitlib.Watch(exitlib.Bounds(None, None), 1.0, resident=lambda: 10**12, load=lambda: 0.0).exceeded is None


class _FakeRun:
    """A run whose `wait` answers from a script of statuses; `cancel` is recorded."""

    def __init__(self, statuses: list[SimpleNamespace]) -> None:
        self.statuses, self.cancelled = statuses, False

    def wait(self, timeout: float | None = None) -> SimpleNamespace:  # noqa: ARG002
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def cancel(self) -> None:
        self.cancelled = True


def _status(done: bool = False, interrupted: bool = False) -> SimpleNamespace:
    return SimpleNamespace(done=done, interrupted=interrupted)


def _drive(run: _FakeRun, bounds: Any, exceeded: str | None = None) -> Any:  # noqa: ANN401 - a status
    watch = exitlib.Watch(exitlib.Bounds(None, None), 1.0, resident=lambda: 0, load=lambda: 0.0)
    watch.exceeded = exceeded
    return exitlib.drive(run, watch, bounds, 0.001)


@pytest.mark.logic
def test_drive_returns_a_done_status_and_cancels_on_each_bound() -> None:
    """A run that finishes returns its status; a passed resident bound, wall bound or stopped coordinator cancels."""
    assert _drive(_FakeRun([_status(), _status(done=True)]), exitlib.Bounds(None, None)).done
    for run, bounds, exceeded, message in [
        (_FakeRun([_status()]), exitlib.Bounds(None, 1), "resident 9 B", "resident 9"),
        (_FakeRun([_status()]), exitlib.Bounds(0.0, None), None, "wall passed"),
        (_FakeRun([_status(interrupted=True)]), exitlib.Bounds(None, None), None, "coordinator stopped"),
    ]:
        with pytest.raises(exitlib.BoundExceeded, match=message):
            _drive(run, bounds, exceeded)
        assert run.cancelled


@pytest.mark.logic
def test_executors_fit_their_substrates_and_pool_local_has_the_configured_workers() -> None:
    """In-process and pool-local need a local profile, pool-slurm a slurm one; pool-local sets cpu workers only."""
    config = SimpleNamespace(pool_local_workers=2)
    laptop, orcd = ExecutionProfile.load("laptop"), ExecutionProfile.load("orcd")
    assert exitlib.profile_for("in-process", laptop, config) is laptop
    assert exitlib.profile_for("pool-slurm", orcd, config) is orcd
    pooled = exitlib.profile_for("pool-local", laptop, config)
    assert pooled.resource_classes["cpu"].max_workers == 2 and pooled.resource_classes["gpu"].max_workers == 1
    for executor, profile in [("pool-slurm", laptop), ("in-process", orcd), ("pool-local", orcd), ("threads", laptop)]:
        with pytest.raises(ValueError, match=executor):
            exitlib.profile_for(executor, profile, config)


@pytest.mark.logic
def test_the_config_loads_and_refuses_a_bad_share_or_executor_set(tmp_path: Path) -> None:
    """The shipped config names every executor and a proper fraction; one without either is refused."""
    config = exitlib.load_config()
    assert set(config.bounds) == set(exitlib.EXECUTORS) and 0 < config.interrupt_share[0] < config.interrupt_share[1]
    assert exitlib.interrupt_target(1000, (1, 3)) == 334
    text = (EXIT / "exit.yaml").read_text(encoding="utf-8")
    for name, edited, message in [
        ("share", text.replace("[1, 3]", "[3, 3]"), "proper fraction"),
        ("executors", text.replace("  pool-local: *laptop\n", ""), "bounds the executors"),
        ("profiles", text.replace("[laptop, orcd]", "[orcd, orcd]"), "at least two distinct"),
    ]:
        path = tmp_path / f"{name}.yaml"
        path.write_text(edited, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            exitlib.load_config(path)


@pytest.mark.logic
def test_slurm_job_ids_are_those_of_the_run_prefix() -> None:
    """Only live jobs whose name starts with the run's prefix are listed; malformed rows are ignored."""
    rows = (
        "11|RUNNING|cpu|senselab-abc-1-a0-t-e1\n12|PENDING|gpu:h100|senselab-abc-1-a0-u-e1\n13|RUNNING|x|other\nbad\n"
    )
    asked: list[list[str]] = []

    def run(argv: list[str], timeout_s: float) -> str:  # noqa: ARG001
        asked.append(argv)
        return rows

    assert exitlib.slurm_job_ids("senselab-abc-1-", 5.0, run) == ["11", "12"]
    assert asked[0][:2] == ["squeue", "--me"]


@pytest.mark.logic
def test_compare_command_exits_nonzero_on_a_difference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two files of one hash and identities compare equal; another identity or hash exits 1 and is named."""
    base = {"step": "resample", "hash": HASH, "identities": {"a": _output("1"), "b": _output("2")}}
    paths = []
    for name, document in [
        ("one", base),
        ("two", base),
        ("three", {**base, "identities": {"a": _output("1"), "b": _output("3")}}),
        ("four", {**base, "hash": "b" * 64}),
    ]:
        paths.append(tmp_path / f"{name}.json")
        paths[-1].write_text(json.dumps(document), encoding="utf-8")
    run: Callable[[Path, Path], int] = lambda a, b: exitlib.identity_cli("resample", ["compare", str(a), str(b)])  # noqa: E731
    assert run(paths[0], paths[1]) == 0
    assert run(paths[0], paths[2]) == 1 and "b:" in capsys.readouterr().out
    assert run(paths[0], paths[3]) == 1 and "hash:" in capsys.readouterr().out


@pytest.mark.logic
@pytest.mark.parametrize(
    ("call", "message"),
    [
        ([], "not a agent-plan@1.0 object"),
        ({"format": "agent-plan@1.0", "task": "no.such.task"}, "none of"),
        ({"format": "agent-plan@1.0", "task": "signal.resample", "dataset": "/no/such/dir"}, "directory"),
        ({"format": "agent-plan@1.0", "task": "signal.resample", "params": [], "dataset": "."}, "params is an object"),
    ],
)
def test_the_agents_call_is_refused_with_a_reason(call: object, message: str) -> None:
    """A malformed call, an unknown task, a missing dataset or non-object params is refused naming why."""
    with pytest.raises(agent_plan.BadCall, match=message):
        agent_plan.configured(call)


@pytest.mark.logic
def test_a_refused_call_is_a_result_not_a_crash(tmp_path: Path) -> None:
    """A call file the agent got wrong gives a result with the error and `ok` false, without running anything."""
    call = tmp_path / "call.json"
    call.write_text("not json", encoding="utf-8")
    result = agent_plan.run_call(call, "laptop", "in-process", tmp_path / "work", exitlib.load_config())
    assert result["ok"] is False and result["error"]["type"] == "JSONDecodeError"


@pytest.mark.logic
def test_a_tree_is_laid_once_and_the_resident_probe_reads_this_process(tmp_path: Path) -> None:
    """A step's tree is written under the work directory and found again; `ps` gives this process a size."""
    step = exitlib.Step("Resample", {"target_rate": 8000}, "synthetic", 3, 0)
    root = exitlib.build_tree(step, tmp_path)
    marker = (root / "dataset_description.json").stat().st_mtime_ns
    assert (
        exitlib.build_tree(step, tmp_path) == root and (root / "dataset_description.json").stat().st_mtime_ns == marker
    )
    assert len(list(root.rglob("*_audio.wav"))) == 3
    with pytest.raises(ValueError, match="not one this harness lays"):
        exitlib.build_tree(exitlib.Step("X", {}, "unknown", 1), tmp_path)
    assert exitlib.process_resident(exitlib.load_config().watch_poll_s) > 0


@pytest.mark.logic
def test_a_skipped_resident_sample_is_counted_not_read_as_zero() -> None:
    """A `ps` too slow to sample is counted in the report and moves no peak."""
    samples = iter([None, 40, None])
    watch = exitlib.Watch(exitlib.Bounds(None, None), 1.0, resident=lambda: next(samples), load=lambda: 0.0)
    for _ in range(3):
        watch.sample()
    assert (watch.peak_resident, watch.report()["skipped_samples"]) == (40, 2)


def _child(done: bool) -> Callable[[Path, str], subprocess.Popen[bytes]]:
    """A spawner of a process that ends at once when `done`, else sleeps until its group is killed."""
    code = "pass" if done else "import time; time.sleep(600)"
    return lambda work, profile: subprocess.Popen([sys.executable, "-c", code], start_new_session=True)  # noqa: ARG005


def _started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, committed: int) -> Any:  # noqa: ANN401
    """Set `start` over three items with `committed` finished; return the harness config."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    log = EventLog(tmp_path / "events", "a0", 1, epoch=1)
    for i in range(committed):
        _record(log, "finished", i, KEYS[i], _output(KEYS[i]))
    placed = SimpleNamespace(hash=HASH, manifest=SimpleNamespace(rows=[SimpleNamespace(id=f"r{i}") for i in range(3)]))
    monkeypatch.setattr(interrupt_resume, "run_dir_of", lambda plan_hash: tmp_path / "events")
    monkeypatch.setattr(exitlib, "build_tree", lambda step, work: work)
    monkeypatch.setattr(exitlib, "planned", lambda step, root, profile: placed)
    monkeypatch.setattr(exitlib, "slurm_prefix", lambda profile, run_dir, plan_hash: "senselab-x-")
    listings = iter([["77", "78"], []])
    monkeypatch.setattr(exitlib, "slurm_job_ids", lambda prefix, timeout_s, run: next(listings, []))
    monkeypatch.setattr(interrupt_resume, "scancel", lambda jobs: None)
    return exitlib.load_config()


@pytest.mark.logic
def test_start_refuses_a_run_that_is_done_before_its_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run done with fewer items committed than the target has nothing to interrupt and fails."""
    config = _started(tmp_path, monkeypatch, committed=0)
    with pytest.raises(RuntimeError, match="nothing to interrupt"):
        interrupt_resume.start(tmp_path, tmp_path / "out.json", "orcd", config, _child(True))


@pytest.mark.logic
def test_interrupt_cancels_the_prefix_kills_the_process_then_sweeps_what_it_submitted() -> None:
    """Jobs are cancelled before the kill; one the killed process submitted meanwhile is cancelled after it."""
    order: list[str] = []
    listings = iter([["1", "2"], ["3"], []])

    def jobs(prefix: str, timeout_s: float) -> list[str]:
        assert prefix == "senselab-x-" and timeout_s == 5.0
        return next(listings)

    cancelled = interrupt_resume.interrupt(
        "senselab-x-",
        99,
        5.0,
        0.0,
        None,
        jobs,
        lambda each: order.append("cancel " + " ".join(each)),
        lambda pid: order.append(f"kill {pid}"),
    )
    assert order == ["cancel 1 2", "kill 99", "cancel 3"] and cancelled == ["1", "2", "3"]


@pytest.mark.logic
def test_interrupt_fails_when_jobs_of_the_prefix_outlive_the_bound() -> None:
    """Jobs still listed after the deadline raise instead of reporting an interrupt."""
    with pytest.raises(exitlib.BoundExceeded, match="still live"):
        interrupt_resume.interrupt(
            "p-", 1, 1.0, 0.0, 0.0, lambda prefix, seconds: ["5"], lambda each: None, lambda pid: None
        )


@pytest.mark.logic
def test_start_names_the_failures_of_a_run_that_ended_short_of_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose items all failed is refused with the failure message, not as a bare 'nothing to interrupt'."""
    config = _started(tmp_path, monkeypatch, committed=0)
    log = EventLog(tmp_path / "events", "a1", 1, epoch=1)
    fault = FaultRecord(
        type="ScalerFailed", fault="infrastructure", message="'${GPU_PARTITION}' is not set", gap=False, charged=False
    )
    for i in range(3):
        log.record(
            "failed", "call", job=JobRef(plan_hash=HASH, node="call", element=(i,)), key=KEYS[i] * 64, failure=fault
        )
    with pytest.raises(RuntimeError, match=r"3 x '\$\{GPU_PARTITION\}' is not set"):
        interrupt_resume.start(tmp_path, tmp_path / "out.json", "orcd", config, _child(True))


_WRITER = """
import sys, time
from pathlib import Path
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import JobRef
log = EventLog(Path(sys.argv[1]), "a2", 1, epoch=1)
for i in range(3):
    log.record("finished", "call", job=JobRef(plan_hash=sys.argv[2], node="call", element=(i,)), key=str(i) * 64,
               output="output:" + str(i) * 64)
    time.sleep(0.3)
time.sleep(600)
"""


@pytest.mark.logic
def test_start_sees_a_separate_process_commit_items_and_interrupts_it_once_the_target_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count is read from the events on disk that another process writes, and that process is then killed."""
    config = _started(tmp_path, monkeypatch, committed=0)
    children: list[subprocess.Popen[bytes]] = []

    def spawn(work: Path, profile: str) -> subprocess.Popen[bytes]:  # noqa: ARG001
        argv = [sys.executable, "-c", _WRITER, str(tmp_path / "events"), HASH]
        children.append(subprocess.Popen(argv, start_new_session=True))
        return children[0]

    assert interrupt_resume.start(tmp_path, tmp_path / "out.json", "orcd", config, spawn) == 0
    committed = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["committed"]
    assert 1 <= committed <= 3 and children[0].returncode == -9


@pytest.mark.logic
def test_start_interrupts_its_own_run_at_the_target_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The child run is SIGKILLed at the target, the report names the cancelled jobs, and `start` returns 0."""
    config = _started(tmp_path, monkeypatch, committed=1)
    assert interrupt_resume.start(tmp_path, tmp_path / "out.json", "orcd", config) == 0
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert report["workers"] == ["77", "78"] and report["committed"] == 1 and report["hash"] == HASH
    assert f"INTERRUPTED hash={HASH} committed=1/3" in capsys.readouterr().out


@pytest.mark.logic
@pytest.mark.parametrize(("have", "counts"), [(1, {}), (3, {"failed": 1})])
def test_a_run_with_an_incomplete_or_failed_result_set_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, have: int, counts: dict[str, int]
) -> None:
    """`run_identity` raises when fewer records have a result than items, or any job failed."""
    log = EventLog(tmp_path / "events", "a0", 1, epoch=1)
    for i in range(have):
        _record(log, "finished", i, KEYS[i], _output(KEYS[i]))
    nodes = {"call": SimpleNamespace(backend="b", resource_class="cpu")}
    placed = SimpleNamespace(
        hash=HASH,
        manifest=SimpleNamespace(rows=[SimpleNamespace(id=f"r{i}") for i in range(3)]),
        placement=SimpleNamespace(nodes=nodes),
    )
    done = exitlib.Execution(placed, None, read_events(tmp_path / "events"), {}, 0.0, counts)
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(interrupt_resume, "run_dir_of", lambda plan_hash: tmp_path / "events")
    monkeypatch.setattr(exitlib, "build_tree", lambda step, work: work)
    monkeypatch.setattr(exitlib, "planned", lambda step, root, profile: placed)
    monkeypatch.setattr(exitlib, "execute", lambda placed, executor, config: done)
    with pytest.raises(RuntimeError, match="items have a result"):
        exitlib.run_identity("resample", "laptop", "in-process", tmp_path, exitlib.load_config())


@pytest.mark.logic
def test_the_agents_input_holds_the_descriptions_and_the_dataset_path_only(tmp_path: Path) -> None:
    """Nothing but the format tag, the prompt, the dataset path and `describe` of each task reaches the agent."""
    document = agent_plan.agent_input(tmp_path)
    assert set(document) == {"format", "instructions", "dataset", "describe"}
    assert document["dataset"] == str(tmp_path)
    with pytest.raises(ValueError, match="not a directory"):
        agent_plan.agent_input(tmp_path / "absent")
