"""The cluster evidence harness (`benchmarks/exit/`) as product code: its watchdog, join helper, diagnostics and driver.

The driver's commands each run against a tiny fake run made by the product itself (the fake echo task, in process,
twice: the second run is the resume), so an API change in the runtime breaks this test and not a queued job.
"""

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy
import pytest
from exit_fake_stack_fixture import fake_stack  # noqa: F401 - a fixture of every test here
from senselab_core.kernel.declare import TaskSpec
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelIdentity
from senselab_core.kernel.wire import WIRE_FORMATS, wire_form
from senselab_core.runtime.audio_io import READER
from senselab_core.runtime.call import TaskCall
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.envs import read_lock_inputs
from senselab_core.runtime.execution.events import read_events, run_dir_of
from senselab_core.runtime.execution.run import RESULTS_DIR
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.values import samples_of

ROOT = Path(__file__).resolve().parents[4]
EXIT = ROOT / "benchmarks" / "exit"
STEPS = EXIT / "orcd"


def _module(name: str) -> ModuleType:
    """Load `benchmarks/exit/<name>.py` as module `name`, as the scripts import each other."""
    spec = importlib.util.spec_from_file_location(name, EXIT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exitlib, keyjoin, stall, reuse_diag, orcd_driver = (
    _module(n) for n in ("exitlib", "keyjoin", "stall", "reuse_diag", "orcd_driver")
)

ITEMS = 3
CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)


def _drive(*argv: str) -> int:
    return int(orcd_driver.main(list(argv)))


def _verdict(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return document


@dataclass(frozen=True)
class FakeRun:
    """A run of three items executed, then run again (served from the store as a resume would be)."""

    hash: str
    run_dir: Path
    tmp: Path

    def start_json(self, **over: Any) -> Path:  # noqa: ANN401 - any report field
        """Write a Step 4 start report and return its path."""
        return self._write("start.json", {"hash": self.hash, "items": ITEMS, "committed": ITEMS, "workers": [], **over})

    def resume_json(self, **over: Any) -> Path:  # noqa: ANN401 - any accounting field
        """Write a Step 4 resume accounting and return its path."""
        accounting = {
            "hash": self.hash,
            "items": ITEMS,
            "committed": ITEMS,
            "executions": 0,
            "late": 0,
            "reuses": ITEMS,
        }
        return self._write("resume.json", {**accounting, **over})

    def _write(self, name: str, document: dict[str, Any]) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path


@pytest.fixture
def fake_run(fake_stack: TaskSpec, tmp_path: Path) -> FakeRun:  # noqa: ARG001, F811
    """Run the fake echo plan twice in process: epoch 1 executes it, epoch 2 reuses every result."""
    call = TaskCall(task="fake.echo", task_version="1", inputs={}, params={"salt": "a"})
    placed = None
    for _ in range(2):
        placed = plan(
            call,
            from_records({"id": f"r{i}", "inputs": {"payload": {"n": i}}} for i in range(ITEMS)),
            execution=ExecutionProfile.load("laptop"),
        )
        assert placed.run().wait(timeout=120).done
    assert placed is not None
    return FakeRun(placed.hash, run_dir_of(placed.hash), tmp_path)


# --- the watchdog -----------------------------------------------------------------------------------------------


@pytest.mark.logic
def test_the_last_progress_line_of_a_log_is_the_committed_count(tmp_path: Path) -> None:
    """Only the last `progress: N` line counts; a log without one, or no log, has none."""
    log = tmp_path / "run.log"
    assert stall.last_committed(log) is None
    log.write_text("starting\n")
    assert stall.last_committed(log) is None
    log.write_text(exitlib.progress_line(3, 10, 5) + "\nnoise\n" + exitlib.progress_line(7, 10, 65) + "\n")
    assert stall.last_committed(log) == 7


@pytest.mark.logic
@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param([(0, 1, False), (50, 1, False)], None, id="idle-for-less-than-the-bound"),
        pytest.param([(0, 1, False), (100, 1, False)], "unchanged at 1 for 100 s", id="idle-and-no-job"),
        pytest.param([(0, 1, True), (500, 1, True)], None, id="idle-with-a-job-queued"),
        pytest.param([(0, 1, None), (500, 1, None)], None, id="idle-with-squeue-unreadable"),
        pytest.param([(0, 1, False), (60, 2, False), (120, 2, False)], None, id="progress-resets-the-clock"),
        pytest.param([(0, None, False), (100, None, False)], "unchanged at none for 100 s", id="no-progress-line-yet"),
    ],
)
def test_a_run_stalls_only_when_idle_and_its_queue_is_read_and_empty(
    steps: list[tuple[float, int | None, bool | None]], expected: str | None
) -> None:
    """The rule: count unchanged for the bound and no run job in a readable queue; a failed `squeue` is no evidence."""
    rule = stall.Stall(100.0, "senselab-", 0.0)
    verdict = None
    for now, committed, live in steps:
        verdict = rule.observe(now, committed, lambda live=live: live)
    assert (verdict is None) if expected is None else (verdict is not None and expected in verdict)


@pytest.mark.logic
def test_a_failed_squeue_is_none_not_an_empty_queue() -> None:
    """`has_live_jobs` reads True for a listed job, False for an empty queue and None for a failed command."""
    row = "77|RUNNING|x|senselab-abc-1\n"

    def failing(argv: list[str], timeout_s: float) -> str:
        raise subprocess.CalledProcessError(1, argv)

    def missing(argv: list[str], timeout_s: float) -> str:
        raise FileNotFoundError(argv[0])

    assert stall.has_live_jobs("senselab-", 1.0, lambda argv, timeout_s: row) is True
    assert stall.has_live_jobs("senselab-", 1.0, lambda argv, timeout_s: "") is False
    assert stall.has_live_jobs("other-", 1.0, lambda argv, timeout_s: row) is False
    assert stall.has_live_jobs("senselab-", 1.0, failing) is None
    assert stall.has_live_jobs("senselab-", 1.0, missing) is None


def _command(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _watch(tmp_path: Path, code: str, live: Callable[[str, float], bool | None]) -> Any:  # noqa: ANN401 - Watched
    return stall.watched(
        _command(code), tmp_path / "w.log", stall_s=0.4, poll_s=0.05, prefix="senselab-", timeout_s=1.0, live=live,
        grace_s=5.0,
    )  # fmt: skip


_SILENT = "import time; print('progress: 1 items committed', flush=True); time.sleep(60)"
_BRIEF = "import time; print('progress: 1 items committed', flush=True); time.sleep(1.2)"
_ADVANCING = (
    "import time\nfor i in range(1, 13):\n    print(f'progress: {i} items committed', flush=True); time.sleep(0.1)"
)


@pytest.mark.logic
def test_a_command_without_progress_and_without_jobs_is_stopped_with_a_verdict(tmp_path: Path) -> None:
    """Silent past the bound with an empty queue: killed, status 3, the reason names the count and the prefix."""
    done = _watch(tmp_path, _SILENT, lambda prefix, timeout_s: False)
    assert done.status == stall.EXIT_STALLED and done.stalled is not None
    assert "unchanged at 1" in done.stalled and "senselab-*" in done.stalled
    assert "progress: 1" in (tmp_path / "w.log").read_text()
    stall.write_stall_verdict(tmp_path / "verdict.json", done.stalled)
    verdict = _verdict(tmp_path / "verdict.json")
    assert verdict["pass"] is False and verdict["failed"][0].startswith("no progress: ")


@pytest.mark.logic
@pytest.mark.parametrize("live", [True, None], ids=["a-job-is-queued", "squeue-failed"])
def test_a_command_is_left_running_when_a_job_is_queued_or_the_queue_is_unreadable(
    tmp_path: Path, live: bool | None
) -> None:
    """Idle past the bound but with a queued job, or a `squeue` that failed: not stalled; its own status stands."""
    done = _watch(tmp_path, _BRIEF, lambda prefix, timeout_s: live)
    assert (done.status, done.stalled) == (0, None)


@pytest.mark.logic
def test_a_command_that_keeps_committing_is_not_stalled(tmp_path: Path) -> None:
    """The count changes more often than the bound: never stalled even with an empty queue."""
    done = _watch(tmp_path, _ADVANCING, lambda prefix, timeout_s: False)
    assert (done.status, done.stalled) == (0, None)


@pytest.mark.logic
def test_the_watch_command_writes_the_verdict_and_exits_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the driver: a stalled command leaves a failing verdict file; a finished one leaves none."""
    monkeypatch.setattr(stall, "has_live_jobs", lambda prefix, timeout_s, run=None: False)
    common = ["watch", "--log", str(tmp_path / "w.log"), "--verdict", str(tmp_path / "v.json")]
    common += ["--stall-s", "0.4", "--poll-s", "0.05", "--"]
    assert _drive(*common, *_command("print('done')")) == 0
    assert not (tmp_path / "v.json").exists()
    assert _drive(*common, *_command(_SILENT)) == stall.EXIT_STALLED
    assert _verdict(tmp_path / "v.json")["pass"] is False
    assert _drive("watch", "--log", str(tmp_path / "x.log"), "--verdict", str(tmp_path / "x.json")) == 2


def _pooled(run_dir: Path) -> None:
    """Give `run_dir` the header of a pool of its own, whose id names its jobs."""
    (run_dir / "pool").mkdir(parents=True)
    (run_dir / "pool" / "pool.json").write_text(json.dumps({"pool_id": "d" * 12}))


@pytest.mark.logic
def test_another_runs_jobs_do_not_hide_a_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Watched by run directory: only another run's `pydra-` jobs in the queue is a stall; the run's own is not."""
    runs = tmp_path / "cache" / "ns" / "runs"
    _pooled(runs / ("d" * 64))
    own = exitlib.slurm_prefix(runs / ("d" * 64))
    other = "pydra-ffffffffffff-"
    listing = {"jobs": f"99|RUNNING|x|{other}0-tok-e1\n"}
    monkeypatch.setattr(exitlib, "squeue", lambda argv, timeout_s: listing["jobs"])
    common = ["watch", "--log", str(tmp_path / "w.log"), "--verdict", str(tmp_path / "v.json")]
    common += ["--runs-glob", str(runs / "*"), "--profile", str(CLUSTER), "--stall-s", "0.4", "--poll-s", "0.05", "--"]
    assert _drive(*common, *_command(_SILENT)) == stall.EXIT_STALLED
    assert own in _verdict(tmp_path / "v.json")["failed"][0]
    listing["jobs"] = f"98|PENDING|x|{own}0-tok-e1\n"
    (tmp_path / "v.json").unlink()
    assert _drive(*common, *_command(_BRIEF)) == 0 and not (tmp_path / "v.json").exists()
    for each in runs.iterdir():  # before the run directory exists any `pydra-` job counts, as it did
        shutil.rmtree(each)
    listing["jobs"] = f"99|RUNNING|x|{other}0-tok-e1\n"
    assert _drive(*common, *_command(_BRIEF)) == 0


@pytest.mark.logic
def test_a_failed_prefix_computation_is_logged_and_retried_and_the_watch_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`slurm_prefix` raises OSError once: the watch logs it, counts any `pydra-` job, then stalls on its own."""
    runs = tmp_path / "cache" / "ns" / "runs"
    _pooled(runs / ("d" * 64))
    real = exitlib.slurm_prefix
    calls: list[int] = []

    def flaky(run_dir: Path) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("stale file handle")
        return str(real(run_dir))

    monkeypatch.setattr(exitlib, "slurm_prefix", flaky)
    other = "99|RUNNING|x|pydra-ffffffffffff-e1-t\n"
    monkeypatch.setattr(exitlib, "squeue", lambda argv, timeout_s: other)
    common = ["watch", "--log", str(tmp_path / "w.log"), "--verdict", str(tmp_path / "v.json")]
    common += ["--runs-glob", str(runs / "*"), "--profile", str(CLUSTER), "--stall-s", "0.4", "--poll-s", "0.05", "--"]
    assert _drive(*common, *_command(_SILENT)) == stall.EXIT_STALLED
    assert len(calls) == 2  # computed again after the failure, then kept
    assert "stale file handle" in capsys.readouterr().err
    assert "pydra-dddddddddddd-" in _verdict(tmp_path / "v.json")["failed"][0]


@pytest.mark.logic
def test_an_unexpected_error_in_the_watch_loop_stops_the_child_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop raises: the child's group is stopped first, so no coordinator is left running unwatched."""
    pid_file = tmp_path / "pid"
    code = "import os, pathlib, sys, time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"

    def failing(log: Path) -> int | None:
        while not pid_file.exists() or not pid_file.read_text():
            time.sleep(0.01)  # the child is up before the loop fails
        raise RuntimeError("the log could not be read")

    monkeypatch.setattr(stall, "last_committed", failing)
    with pytest.raises(RuntimeError, match="could not be read"):
        stall.watched(
            [sys.executable, "-c", code, str(pid_file)],
            tmp_path / "w.log",
            stall_s=60.0,
            poll_s=0.05,
            prefix="senselab-",
            timeout_s=1.0,
            grace_s=5.0,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


# --- the join helper and the reuse diagnostic -------------------------------------------------------------------


@pytest.mark.logic
def test_a_join_prints_one_key_of_each_side_and_fails_when_nothing_matches() -> None:
    """Sample lines carry each side's key shape; a JSON-text side against a digest side fails naming both shapes."""
    digest = "ab" * 32
    lines: list[str] = []
    assert keyjoin.join("prov", {digest, "c" * 64}, "events", {digest}, lines.append) == {digest}
    assert any("64-hex digest" in line for line in lines) and any("1 keys match" in line for line in lines)
    with pytest.raises(keyjoin.JoinFailure, match="share no key.*shapes differ"):
        keyjoin.join("prov", {'"' + digest + '"'}, "events", {digest}, lines.append)
    with pytest.raises(keyjoin.JoinFailure, match="share no key"):
        keyjoin.join("prov", set[str](), "events", {digest}, lines.append)


@pytest.mark.contract
def test_the_reuse_diagnostic_finds_nothing_in_a_run_whose_executions_were_each_reused(fake_run: FakeRun) -> None:
    """Three start-phase executions, each with the resume's Reuse; the join matched their keys to the events'."""
    lines: list[str] = []
    assert reuse_diag.diagnose(fake_run.run_dir, lines.append) == []
    assert any("3 keys match" in line for line in lines)


@pytest.mark.contract
def test_the_reuse_diagnostic_names_an_execution_whose_reuse_is_missing(fake_run: FakeRun) -> None:
    """Remove the resume's documents: each start-phase key has no Reuse, and its events say it was committed."""
    boundary = reuse_diag.boundary_of(read_events(fake_run.run_dir))
    documents = {r.document for r in reuse_diag.activity_rows(fake_run.run_dir, boundary) if r.phase == "resume"}
    for document in documents:
        (fake_run.run_dir / "prov" / document).unlink()
    found = reuse_diag.diagnose(fake_run.run_dir, lambda line: None)
    assert len(found) == ITEMS and {f.status for f in found} == {"committed"}


@pytest.mark.contract
def test_the_reuse_diagnostic_fails_when_its_two_sources_spell_keys_differently(
    fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The T24-10 defect: keys read as JSON text against digests match nothing, and the diagnostic stops there."""
    monkeypatch.setattr(reuse_diag, "activity_key", lambda record: json.dumps("x" * 64))
    with pytest.raises(keyjoin.JoinFailure, match="shapes differ"):
        reuse_diag.diagnose(fake_run.run_dir, lambda line: None)


@pytest.mark.logic
def test_a_reuse_is_settled_by_its_shape_not_its_document() -> None:
    """The coordinator's settled Reuse has neither an execution context nor an associated agent; a worker's has both."""
    assert reuse_diag.is_settled_reuse({"senselab:storedContext": {}})
    assert not reuse_diag.is_settled_reuse({"senselab:context": {}, "prov:wasAssociatedWith": "agent"})
    assert not reuse_diag.is_settled_reuse({"prov:wasAssociatedWith": "agent"})


# --- the driver, one command at a time, against the fake run ----------------------------------------------------


@pytest.mark.contract
def test_prov_passes_a_run_accounted_by_the_runtime_check_and_names_what_it_lacks(
    fake_run: FakeRun, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`prov` calls `check_provenance`; Step 4's own accounting (reuses, executions, keys) fails on a wrong figure."""
    out = tmp_path / "verdict.json"
    args = [
        "prov",
        "--out",
        str(out),
        "--run-dir",
        str(fake_run.run_dir),
        "--hash",
        fake_run.hash,
        "--quantum-items",
        "1",
    ]
    assert _drive(*args, "--start-json", str(fake_run.start_json()), "--resume-json", str(fake_run.resume_json())) == 0
    verdict = _verdict(out)
    assert verdict["pass"] is True and verdict["reuses"] == ITEMS and verdict["executions"] == ITEMS
    assert verdict["reuses_worker"] == ITEMS and verdict["reuses_coordinator"] == 0
    assert "PASS provenance complete (ADR-010" in capsys.readouterr().out
    wrong = fake_run.resume_json(reuses=ITEMS - 1)
    assert _drive(*args, "--start-json", str(fake_run.start_json()), "--resume-json", str(wrong)) == 1
    assert _verdict(out)["failed"] == [f"senselab:Reuse activities = the resume's reuses ({ITEMS - 1})"]
    fewer = fake_run.start_json(items=ITEMS + 1)
    assert _drive(*args, "--start-json", str(fewer), "--resume-json", str(fake_run.resume_json(items=ITEMS + 1))) == 1
    assert any(name.startswith("the run committed 4 keys") for name in _verdict(out)["failed"])


@pytest.mark.contract
def test_prov_fails_when_a_committed_key_has_neither_an_execution_nor_a_reuse(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    """Remove every document with an activity: the runtime's check names the keys and the verdict fails on it."""
    for document in (fake_run.run_dir / "prov").iterdir():
        if '"senselab:logicalKey"' in document.read_text(encoding="utf-8"):
            document.unlink()
    out = tmp_path / "verdict.json"
    args = [
        "prov",
        "--out",
        str(out),
        "--run-dir",
        str(fake_run.run_dir),
        "--hash",
        fake_run.hash,
        "--quantum-items",
        "1",
    ]
    assert _drive(*args, "--start-json", str(fake_run.start_json()), "--resume-json", str(fake_run.resume_json())) == 1
    assert any(name.startswith("provenance complete") for name in _verdict(out)["failed"])


@pytest.mark.contract
def test_prov_expects_resume_executions_from_the_resumes_own_committed_count(fake_run: FakeRun, tmp_path: Path) -> None:
    """An item committed between the interrupt snapshot and the resume's start is not a resume execution."""
    out = tmp_path / "verdict.json"
    args = ["prov", "--out", str(out), "--run-dir", str(fake_run.run_dir), "--hash", fake_run.hash]
    start = fake_run.start_json(committed=ITEMS - 1)
    assert (
        _drive(*args, "--quantum-items", "3", "--start-json", str(start), "--resume-json", str(fake_run.resume_json()))
        == 0
    )
    assert _verdict(out)["failed"] == []


@pytest.mark.contract
def test_prov_bounds_the_keys_a_kill_lost_by_one_claim_per_cancelled_worker(fake_run: FakeRun, tmp_path: Path) -> None:
    """Keys without an execution activity are at most quantum_items x cancelled workers (the three here), not twice."""
    boundary = reuse_diag.boundary_of(read_events(fake_run.run_dir))
    rows = reuse_diag.activity_rows(fake_run.run_dir, boundary)
    lost = next(r for r in rows if r.kind == "execution")
    (fake_run.run_dir / "prov" / lost.document).unlink()
    out = tmp_path / "verdict.json"
    args = [
        "prov",
        "--out",
        str(out),
        "--run-dir",
        str(fake_run.run_dir),
        "--hash",
        fake_run.hash,
        "--quantum-items",
        "1",
    ]
    start, resume = fake_run.start_json(workers=["1", "2", "3"]), fake_run.resume_json()
    assert _drive(*args, "--start-json", str(start), "--resume-json", str(resume)) == 0
    assert _drive(*args, "--start-json", str(fake_run.start_json(workers=[])), "--resume-json", str(resume)) == 1
    message = next(n for n in _verdict(out)["failed"] if n.startswith("keys without an execution activity"))
    assert "<= 0 (ADR-010: a kill loses the claim in flight, at most quantum_items 1 per cancelled worker)" in message


@pytest.mark.logic
def test_interrupt_and_resume_verdicts_follow_the_accounting(tmp_path: Path) -> None:
    """Step 4's two verdicts: committed in [a third, all); executions + late = items - committed; reuses the rest."""
    out = tmp_path / "v.json"
    start = tmp_path / "start.json"
    h = "a" * 64
    start.write_text(json.dumps({"hash": h, "items": 9, "committed": 3, "workers": ["1"]}))
    assert _drive("interrupt", "--out", str(out), "--start-json", str(start), "--status", "0") == 0
    start.write_text(json.dumps({"hash": h, "items": 9, "committed": 9, "workers": []}))
    assert _drive("interrupt", "--out", str(out), "--start-json", str(start), "--status", "0") == 1
    start.write_text(json.dumps({"hash": h, "items": 9, "committed": 3, "workers": ["1"]}))
    ok = {"hash": h, "items": 9, "committed": 3, "executions": 5, "late": 1, "reuses": 4}
    resume = tmp_path / "resume.json"
    resume.write_text(json.dumps(ok))
    args = ["resume", "--out", str(out), "--start-json", str(start), "--resume-json", str(resume), "--status", "0"]
    assert _drive(*args) == 0
    resume.write_text(json.dumps({**ok, "executions": 4}))
    assert _drive(*args) == 1 and len(_verdict(out)["failed"]) == 2


@pytest.mark.contract
def test_identity_checks_the_run_against_its_plan_and_the_laptops_outputs(
    fake_run: FakeRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal identities pass; an identity the laptop's store cannot give is read through the run's own events."""
    events = read_events(fake_run.run_dir)
    from senselab_core.runtime.execution.run import load_plan

    rows = load_plan(fake_run.run_dir).manifest.rows
    identities = exitlib.output_identities(events, rows)
    assert len(identities) == ITEMS
    document = {
        "step": "fake", "hash": fake_run.hash, "items": ITEMS, "counts": {"done": ITEMS}, "identities": identities,
        "placement": {"call": {"resource_class": "gpu:h100"}}, "wall_s": 1.0, "watch": {},
    }  # fmt: skip
    run_json, laptop = tmp_path / "run.json", tmp_path / "laptop.json"
    run_json.write_text(json.dumps(document))
    laptop.write_text(json.dumps(document))
    plan_log = tmp_path / "plan.log"
    plan_log.write_text(f"fake: one plan hash {fake_run.hash} under laptop, cluster; 3 differences, all in placement\n")
    out = tmp_path / "v.json"
    args = [
        "identity",
        "--out",
        str(out),
        "--plan-log",
        str(plan_log),
        "--plan-status",
        "0",
        "--run-json",
        str(run_json),
    ]
    args += ["--run-status", "0", "--items", str(ITEMS), "--expect-hash", fake_run.hash, "--require-class", "gpu:h100"]
    assert _drive(*args, "--laptop", str(laptop)) == 0
    plan_log.write_text("no hash here\n")
    assert _drive(*args) == 1


def _series(scale: float) -> dict[str, object]:
    """The output of one resampled record: a regular group, its samples scaled by `scale`."""
    group = wire_form(samples_of(WIRE_FORMATS["regular-group@3.0"])[0], READER)
    return {"series": {**group, "data": numpy.asarray(group["data"]) * scale}}


KEY = LogicalKey(
    task="resample", task_version="1", backend="b", backend_version="1", params={}, backend_params={},
    model=ModelIdentity(id="o/m", identity="0" * 40), inputs={"series": "content:" + "a" * 64},
)  # fmt: skip
CONTEXT = ExecutionContext(
    env_lock="lock", platform="linux", wheel_set="w", numerics={}, decoder=None, driver={}, host="h", protocol="p",
    model_source="local", env_license=None,
)  # fmt: skip


def _compare_step2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_scale: float) -> list[str]:
    """Return what `_array_differences` reports for a laptop output against one scaled by `run_scale`.

    Both outputs are committed to real result stores, one for the laptop and one for this run; r1 has equal identities.
    """
    stores = {name: ResultStore(tmp_path / name / "cache" / RESULTS_DIR) for name in ("laptop", "run")}
    identity = {
        "laptop": stores["laptop"].put(KEY, _series(1.0), CONTEXT),
        "run": stores["run"].put(KEY, _series(run_scale), CONTEXT),
    }
    monkeypatch.setattr(orcd_driver, "_keyed_outputs", lambda cache, plan_hash: {"r0": (KEY.digest(), identity["run"])})
    laptop = {"step": "resample", "hash": "h", "identities": {"r0": identity["laptop"], "r1": "output:c"}}
    run = {"step": "resample", "hash": "h", "identities": {"r0": identity["run"], "r1": "output:c"}}
    cache = {name: tmp_path / name / "cache" for name in stores}
    return orcd_driver._array_differences(laptop, run, cache["laptop"], cache["run"])  # type: ignore[no-any-return]  # noqa: SLF001


@pytest.mark.logic
def test_suite_combine_and_crossnode_verdicts(tmp_path: Path) -> None:
    """A tier's junit file, the combination of tier verdicts, and the lease probe's verdict over three hosts."""
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite tests="4" failures="0" errors="0" skipped="1"/></testsuites>')
    out = tmp_path / "logic-verdict.json"
    base = ["suite", "--out", str(out), "--tier", "logic", "--junit", str(junit), "--status"]
    assert _drive(*base, "0") == 0 and _drive(*base, "1") == 1
    missing = ["suite", "--out", str(out), "--tier", "logic", "--junit", str(tmp_path / "none.xml"), "--status", "1"]
    assert _drive(*missing) == 1 and "logic: pytest wrote its junit file" in json.loads(out.read_text())["failed"]
    junit.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="0"/></testsuites>')
    assert _drive(*base, "0") == 1
    junit.write_text('<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0"/></testsuites>')
    assert _drive(*base, "0") == 0
    combined = tmp_path / "combined.json"
    assert _drive("combine", "--out", str(combined), "--dir", str(tmp_path)) == 0
    (tmp_path / "queue-verdict.json").write_text(json.dumps({"pass": False}))
    assert _drive("combine", "--out", str(combined), "--dir", str(tmp_path)) == 1
    envs = tmp_path / "envs.json"
    with_envs = ["suite", "--out", str(tmp_path / "gpu.json"), "--tier", "gpu", "--junit", str(junit), "--status", "0"]
    assert _drive(*with_envs, "--envs", str(envs)) == 1
    envs.write_text(json.dumps({"pass": False, "failed": ["env vllm-server (cuda) builds"]}))
    assert _drive(*with_envs, "--envs", str(envs)) == 1
    envs.write_text(json.dumps({"pass": True}))
    assert _drive(*with_envs, "--envs", str(envs)) == 0
    probes = tmp_path / "probes"
    probes.mkdir()
    (probes / "lease-verdict.json").write_text(json.dumps({"pass": True, "hosts": ["a", "b", "c"]}))
    assert _drive("crossnode", "--out", str(combined), "--dir", str(probes)) == 0
    (probes / "lease-verdict.json").write_text(json.dumps({"pass": True, "hosts": ["a", "b"]}))
    assert _drive("crossnode", "--out", str(combined), "--dir", str(probes)) == 1


@pytest.mark.logic
def test_profile_refuses_a_class_whose_time_passes_its_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a stand-in `sinfo` on the path: a class within the partition's limit is copied, one over it is refused."""
    sinfo = tmp_path / "bin" / "sinfo"
    sinfo.parent.mkdir()
    sinfo.write_text("#!/bin/sh\necho 01:00:00\n")
    sinfo.chmod(sinfo.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{sinfo.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("TEST_PARTITION", "p")
    src = tmp_path / "site.yaml"
    src.write_text("resource_classes:\n  cpu:\n    partition: $TEST_PARTITION\n    time_s: 7200\n")
    out = tmp_path / "profile.yaml"
    base = ["profile", "--src", str(src), "--out", str(out)]
    assert _drive(*base) == 2 and not out.exists()
    assert _drive(*base, "--worker-time-s", "3600") == 0 and out.is_file()


@pytest.mark.logic
def test_profile_refuses_an_unset_variable_naming_its_key_and_writes_the_copy_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A partition naming an unset variable refuses the setup at once, naming key and variable; a set one is written."""
    monkeypatch.delenv("TEST_UNSET_PARTITION", raising=False)
    src = tmp_path / "site.yaml"
    src.write_text("resource_classes:\n  gpu:h100:\n    partition: ${TEST_UNSET_PARTITION}\n    time_s: 60\n")
    out = tmp_path / "profile.yaml"
    assert _drive("profile", "--src", str(src), "--out", str(out), "--classes", "-") == 2 and not out.exists()
    assert "resource_classes.gpu:h100.partition" in capsys.readouterr().err
    monkeypatch.setenv("TEST_UNSET_PARTITION", "p")
    assert _drive("profile", "--src", str(src), "--out", str(out), "--classes", "-") == 0
    assert orcd_driver.load_yaml(out, "profile")["resource_classes"]["gpu:h100"]["partition"] == "p"


@pytest.mark.logic
def test_profile_makes_a_local_substrate_copy_with_its_own_no_progress_bound(tmp_path: Path) -> None:
    """`--substrate local` turns the site's Slurm substrate into a local one; `--no-progress-s` sets the run bound."""
    src = tmp_path / "site.yaml"
    src.write_text(
        "resource_classes: {}\npool:\n  no_progress_s: 7200\nsubstrate:\n  kind: slurm\n  qos_cpu_cap: 64\n"
        "  exclude_nodes: [n1]\n  placement_hook:\n  placement_hook_s: 1\n"
        "  commands: {timeout_s: 60, attempts: 3, retry_first_s: 0.5}\n"
    )
    out = tmp_path / "profile.yaml"
    assert (
        _drive("profile", "--src", str(src), "--out", str(out), "--substrate", "local", "--no-progress-s", "600") == 0
    )
    written = orcd_driver.load_yaml(out, "profile")
    assert written["substrate"] == {"kind": "local", "placement_hook": None, "placement_hook_s": 1}
    assert written["pool"]["no_progress_s"] == 600  # noqa: PLR2004 - the value passed
    assert _drive("profile", "--src", str(src), "--out", str(out)) == 0
    assert orcd_driver.load_yaml(out, "profile")["substrate"]["kind"] == "slurm"


@pytest.mark.logic
def test_every_test_tier_keeps_its_files_on_scratch_and_streams_warnings() -> None:
    """A pytest of a tier script puts its temporary tree under $WORK and logs WARNING live; the cache is the job's."""
    seen = 0
    for script in sorted(STEPS.glob("step-*.sh")):
        commands = _commands(script)
        for command in commands:
            if "uv run --no-sync pytest " in command and "--collect-only" not in command:
                seen += 1
                assert '--basetemp="$WORK/pytest-' in command, f"{script.name}: {command[:80]}"
                assert "-o log_cli=true --log-cli-level=WARNING" in command, f"{script.name}: {command[:80]}"
                assert 'export SENSELAB_CACHE="$WORK/cache"' in commands, script.name
    assert seen >= 1


@pytest.mark.logic
def test_the_test_tiers_run_on_a_local_substrate_and_only_the_pool_step_on_slurm() -> None:
    """The suite's pool jobs run inside the job's own allocation; step-pool alone may submit its worker to Slurm."""
    setup = re.compile(r"^orcd_setup (\S+) (\S+)(?: (\S+))?")
    tiers = [c for c in _commands(STEPS / "step-5.sh") if c.startswith("orcd_setup ")]
    assert [m.group(3) for c in tiers if (m := setup.match(c))] == ["local"]
    assert any('NO_PROGRESS_S="${NO_PROGRESS_S:-240}"' in c for c in _commands(STEPS / "step-5.sh"))
    tier = [c for c in _commands(STEPS / "step-tier.sh") if c.startswith("orcd_setup ")]
    assert [m.group(3) for c in tier if (m := setup.match(c))] == ["local", "local"]
    pool = [c for c in _commands(STEPS / "step-pool.sh") if c.startswith("orcd_setup ")]
    assert [m.group(3) for c in pool if (m := setup.match(c))] == [None]


@pytest.mark.logic
def test_get_prints_a_dotted_value_and_prefix_names_the_runs_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`get` reads JSON or YAML by key path; `prefix` is the Slurm name prefix of the jobs of the run's own pool."""
    document = tmp_path / "d.yaml"
    document.write_text("pool:\n  quantum_items: 4\n")
    assert _drive("get", "--file", str(document), "--key", "pool.quantum_items") == 0
    assert capsys.readouterr().out.strip() == "4"
    assert _drive("get", "--file", str(document), "--key", "pool.absent") == 1
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    h = "c" * 64
    _pooled(tmp_path / "work" / "cache" / "stt" / "2x" / "runs" / h)
    assert _drive("prefix", "--work", str(tmp_path / "work"), "--profile", str(CLUSTER), "--hash", h) == 0
    assert capsys.readouterr().out.strip() == f"pydra-{'d' * 12}-"


@pytest.mark.logic
def test_tree_lays_a_synthetic_bids_tree(tmp_path: Path) -> None:
    """`tree` writes the requested number of recordings."""
    assert _drive("tree", "--out", str(tmp_path / "t"), "--items", "2") == 0
    assert len(list((tmp_path / "t").rglob("*.wav"))) == 2


@pytest.mark.logic
def test_agent_verdicts_read_the_agents_files(tmp_path: Path) -> None:
    """Step 6: the prepared input must describe every task; a refused call is a failed run."""
    out, source = tmp_path / "v.json", tmp_path / "in.json"
    source.write_text(json.dumps({"describe": {f"t{i}": {} for i in range(10)}}))
    assert _drive("agent-prepared", "--out", str(out), "--input", str(source), "--status", "0") == 0
    source.write_text(json.dumps({"describe": {"t": {}}}))
    assert _drive("agent-prepared", "--out", str(out), "--input", str(source), "--status", "0") == 1
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"ok": True, "call": {"task": "x"}}))
    assert _drive("agent-ran", "--out", str(out), "--result", str(result), "--status", "0") == 0
    result.write_text(json.dumps({"ok": False, "error": "refused"}))
    assert _drive("agent-ran", "--out", str(out), "--result", str(result), "--status", "1") == 1


@pytest.mark.contract
def test_the_envs_command_covers_every_shipped_lock() -> None:
    """The driver names an import probe for each env lock the product ships: a new lock fails here, not on ORCD."""
    lock_dir = ROOT / "packages/senselab/src/senselab/data/env_locks"
    assert set(read_lock_inputs(lock_dir / "inputs.yaml")) == set(orcd_driver.ENV_IMPORTS)


# --- the step scripts -------------------------------------------------------------------------------------------


def _commands(script: Path) -> list[str]:
    """Return each shell command of `script`, continuation lines joined, comments dropped."""
    text = re.sub(r"\\\n\s*", " ", script.read_text(encoding="utf-8"))
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


@pytest.mark.logic
@pytest.mark.parametrize("script", sorted(STEPS.glob("*.sh")), ids=lambda p: p.name)
def test_each_step_script_parses_and_names_only_driver_commands_that_exist(script: Path) -> None:
    """`bash -n` accepts it, and every `orcd_driver <command>` it calls is a command of the driver's parser."""
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True, check=False).returncode == 0
    parser = orcd_driver.build_parser()
    known = set(next(a for a in parser._actions if a.dest == "command").choices or {})  # noqa: SLF001
    called = {m.group(1) for c in _commands(script) for m in re.finditer(r"orcd_driver ([a-z0-9-]+)", c)}
    assert called <= known, called - known
    if script.name != "lib.sh":
        assert 'source "$SCRIPTS_DIR/lib.sh"' in script.read_text(encoding="utf-8")


@pytest.mark.logic
def test_every_coordinator_step_runs_under_the_watchdog() -> None:
    """A command that runs a Pool on Slurm, or interrupts or resumes one, goes through `orcd_watched`."""
    coordinators = re.compile(r"--executor pool-slurm|interrupt_resume\.py (start|resume)")
    seen = 0
    for script in sorted(STEPS.glob("step-*.sh")):
        for command in _commands(script):
            if coordinators.search(command):
                seen += 1
                assert command.startswith("orcd_watched "), f"{script.name}: {command[:80]}"
    assert seen == 5  # noqa: PLR2004 - steps 2, 3 and 6, and the start and resume of step 4


@pytest.mark.logic
def test_the_step_scripts_carry_no_site_values() -> None:
    """P-28: partitions, accounts and paths come from the environment, never from a tracked script."""
    text = "\n".join(p.read_text(encoding="utf-8") for p in STEPS.glob("*.sh"))
    assert not re.search(r"/(?:home|Users|orcd)/|ou_\w+|--partition=[a-z]", text)


@pytest.mark.logic
def test_every_tier_pytest_bounds_each_test_and_the_gpu_tier_folds_in_its_env_builds() -> None:
    """Each tier pytest passes a per-test `--timeout`: 120 s on CPU, 300 s on GPU; the GPU suite verdict reads envs."""
    tier = "\n".join(_commands(STEPS / "step-tier.sh"))
    assert re.search(r"cpu\).*?TIMEOUT=120.*?gpu\).*?TIMEOUT=300", tier, re.S)
    assert '--timeout="$TIMEOUT"' in tier
    assert re.search(r"orcd_driver envs .*?ENVS=\(--envs .*?orcd_driver suite .*\$\{ENVS\[@\]\}", tier, re.S)
    five = "\n".join(_commands(STEPS / "step-5.sh"))
    assert "logic | contract) TIMEOUT=120" in five and '[ "$MODE" = gpu ] && TIMEOUT=300' in five
    for script in sorted(STEPS.glob("step-*.sh")):
        for command in _commands(script):
            if "uv run --no-sync pytest " in command and "--collect-only" not in command:
                assert "--timeout" in command, f"{script.name}: {command[:80]}"


_COLLECT = """
import json, sys
import pytest

class Bounds:
    def pytest_collection_finish(self, session):
        found = {}
        for item in session.items:
            marker = item.get_closest_marker("timeout")
            names = vars(getattr(item, "module", None) or object())
            starts = any(n in names for n in ("start_vllm", "VllmServer", "vllm_backend"))
            found[item.nodeid] = [marker.args[0] if marker else None, starts]
        with open(sys.argv[1], "w") as out:
            json.dump(found, out)

sys.exit(pytest.main(sys.argv[2:], plugins=[Bounds()]))
"""

GPU_TIER_TESTS = {
    "packages/senselab-core/tests/runtime/execution/services_gpu_test.py::"
    "test_two_plans_share_one_pii_server_and_a_new_config_is_another_service",
    "packages/senselab/tests/text/tasks/pii_spans/pii_spans_transformers_test.py::"
    "test_pii_spans_transformers_finds_the_identifying_span_with_real_weights",
    "packages/senselab/tests/text/tasks/pii_spans/pii_spans_transformers_test.py::"
    "test_pii_spans_transformers_edge_inputs_with_real_weights",
    "packages/senselab/tests/text/tasks/pii_spans/pii_spans_vllm_test.py::"
    "test_pii_spans_vllm_on_the_h100_finds_spans_and_bounds_edge_inputs",
}
"""The GPU tests the GPU tier ran at S1b (ORCD, H100); the tier's selection must keep collecting them."""


@pytest.mark.logic
def test_the_gpu_tier_selection_collects_the_gpu_tests_and_bounds_each_server_start(tmp_path: Path) -> None:
    """The GPU tier's selection, collected only, holds the known GPU tests and bounds every vLLM server start.

    step-tier.sh's GPU suites, marks and `--timeout` are read from the script. Each test of a module that starts a
    vLLM server runs under its own bound, at least the cluster profile's `vllm.startup_s`.
    """
    tier = "\n".join(_commands(STEPS / "step-tier.sh"))
    block = re.search(r"\bgpu\)(.*?);;", tier, re.S)
    assert block is not None
    suites = re.search(r"SUITES=\((.*?)\)", block.group(1), re.S)
    marks = re.search(r'MARKS="(.*?)"', block.group(1))
    timeout = re.search(r"TIMEOUT=(\d+)", block.group(1))
    assert suites and marks and timeout
    report = tmp_path / "collected.json"
    argv = [*suites.group(1).split(), "-m", marks.group(1), f"--timeout={timeout.group(1)}", "--collect-only", "-q"]
    argv += ["-p", "no:cacheprovider"]
    env = {**os.environ, "SENSELAB_TEST_PROFILE": str(CLUSTER)}
    done = subprocess.run(
        [sys.executable, "-c", _COLLECT, str(report), *argv],
        cwd=ROOT,
        env=env,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stdout.decode()[-2000:] + done.stderr.decode()[-2000:]
    collected: dict[str, list[Any]] = json.loads(report.read_text())
    assert GPU_TIER_TESTS <= set(collected), GPU_TIER_TESTS - set(collected)
    startup_s = ExecutionProfile.load(CLUSTER).vllm.startup_s
    unbounded = {n: b for n, (b, starts) in collected.items() if starts and (b is None or b < startup_s)}
    assert unbounded == {}, f"a vLLM server start needs its own timeout mark of at least {startup_s} s"


@pytest.mark.logic
def test_jobs_share_one_warm_uv_cache_that_the_prepare_step_fills() -> None:
    """Every job syncs from the group's cache or the account's one under WORK_ROOT; step-prepare syncs it first."""
    assert 'export UV_CACHE_DIR="${GROUP_UV_CACHE:-$WORK_ROOT/uv-cache}"' in (STEPS / "lib.sh").read_text()
    assert [c for c in _commands(STEPS / "step-prepare.sh") if c.startswith("orcd_setup ")] == ['orcd_setup prepare ""']
    for script in ("step-tier.sh", "step-5.sh", "step-pool.sh"):
        assert "--dependency=afterok:<prepare job>" in (STEPS / script).read_text(), script
