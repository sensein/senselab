"""Runs on senselab's warm pool session: results, warm reuse, a load out of memory, a hold, a license, a cancel."""

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.runtime.call import RunConfig, TaskCall
from senselab_core.runtime.execution import warm
from senselab_core.runtime.execution.calls import call
from senselab_core.runtime.execution.events import read_events
from senselab_core.runtime.execution.executor import NoProgress, QueueHeld
from senselab_core.runtime.execution.run import start
from senselab_core.runtime.execution.warm import PoolSession, claim_leases, newest_beat
from senselab_core.runtime.licenses import LicensePolicy
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.site_placement import PlacementAnswer, PlacementRequest, PlacementValues, load_overrides
from senselab_core.testing.pools import POOL_PROFILE as PROFILE
from senselab_core.testing.pools import served

pytestmark = pytest.mark.contract

TASK = scoring("warm_pool_test.score")
BACKEND = "backends.pool_backend_fixture"

TWO_CLASSES = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {
            "cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1},
            "cpu:big": {"device": "cpu", "mem_bytes": 2**31, "max_workers": 1},
        },
        "pool": {"poll_s": 0.05, "quantum_items": 8, "quantum_s": 60, "idle_polls": 60},
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
        "faults": {"infrastructure_replacements": 2},
    },
)
"""A small and a big CPU class, the small one first in the ladder."""

RESIDENT = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
    },
)
"""One CPU class, the packaged laptop profile's pool settings: its polls and its idle window."""

BOUND_S = 2.0
BOUNDED = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "quantum_items": 8, "quantum_s": 60, "idle_polls": 40, "no_progress_s": BOUND_S},
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
        "faults": {"infrastructure_replacements": 1000},
    },
)
"""One CPU class whose runs fail after `BOUND_S` without progress; a refused start never breaks the class first."""

DEADLINE = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {
            "poll_s": 0.05,
            "quantum_items": 8,
            "quantum_s": 60,
            "idle_polls": 40,
            "start_deadline_s": 0.2,
            "start_deadline_failures": 2,
        },
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
        "faults": {"infrastructure_replacements": 1000},
    },
)
"""One CPU class whose refused starts break it after two failures over 0.2 s, well inside its no-progress bound."""

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)


@dataclass(frozen=True)
class _Pending:
    """A worker handle of a submitted job that stays queued: listed live, never ready."""

    token: str
    worker_class: str
    cause: str | None = None


class _Queued:
    """A substrate whose every start is a job that stays queued, as a Slurm worker job waiting for a GPU does."""

    def __init__(self) -> None:
        """No job submitted yet."""
        self.handles: list[_Pending] = []

    def attach(self, attachment: object) -> None:
        """Nothing to keep."""

    def start(self, cls: Any, n: int) -> list[_Pending]:  # noqa: ANN401 - a pydra worker class
        made = [_Pending(f"{len(self.handles) + i:032x}", cls.name) for i in range(n)]
        self.handles += made
        return made

    def live(self) -> list[_Pending]:
        return list(self.handles)

    def stop(self, workers: object) -> None:
        """A queued job is left to the test's end."""

    def can_start(self, cls: object) -> bool:
        return True


class _Outage:
    """A substrate whose starts are refused while `down` is set, as a restarting scheduler refuses; else local."""

    def __init__(self) -> None:
        """Down, over a local substrate for when it is back."""
        self.down = True
        self.local = warm.substrate_of(RESIDENT)

    def attach(self, attachment: object) -> None:
        self.local.attach(attachment)

    def start(self, cls: Any, n: int) -> list[Any]:  # noqa: ANN401 - a pydra worker class
        if self.down:
            raise RuntimeError("sbatch: error: Slurm controller not responding")
        return list(self.local.start(cls, n))

    def live(self) -> list[Any]:
        return list(self.local.live())

    def stop(self, workers: Any) -> None:  # noqa: ANN401 - pydra worker handles
        self.local.stop(workers)

    def can_start(self, cls: Any) -> Any:  # noqa: ANN401 - a pydra worker class
        return self.local.can_start(cls)


class _Refusing:
    """A substrate whose every start is refused, as a rejected `sbatch` is; it lists no worker."""

    def attach(self, attachment: object) -> None:
        """Nothing to keep."""

    def start(self, cls: object, n: int) -> list[object]:
        raise RuntimeError("sbatch: error: invalid partition specified")

    def live(self) -> list[object]:
        return []

    def stop(self, workers: object) -> None:
        """Nothing was started."""

    def can_start(self, cls: object) -> bool:
        return True


class _Unlisted(_Refusing):
    """A substrate whose listing fails, as an `squeue` that times out does."""

    def live(self) -> list[object]:
        raise RuntimeError("could not run squeue: timed out after 60 s")


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the fixture backend's load counter and planted faults at files under `tmp_path`, for every process."""
    monkeypatch.setenv("SENSELAB_TEST_LOADS", str(tmp_path / "loads"))
    monkeypatch.setenv("SENSELAB_TEST_OOM_LOADS", str(tmp_path / "oom"))
    monkeypatch.setenv("SENSELAB_TEST_BROKEN", str(tmp_path / "broken"))
    backends = str(Path(__file__).parents[1])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([backends, *filter(None, [os.environ.get("PYTHONPATH")])]))
    return tmp_path


def _loads(tmp_path: Path) -> int:
    path = tmp_path / "loads"
    return len(path.read_text().split()) if path.exists() else 0


def _kinds(run_dir: Path) -> list[str]:
    return sorted(e.kind for e in read_events(run_dir) if e.kind in {"finished", "cache_hit", "failed"})


def test_a_run_on_a_session_gives_each_records_result(tmp_path: Path, planted: Path) -> None:
    """Every record's job runs on the pool worker, its result is read back from the store, its events logged."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path) as (session, _):
        placed, values = planned(TASK, PROFILE, ["a", "b"])
        run = start(placed, session, values=values)
        status = run.wait(timeout=120)
        given = list(run.results())
    assert status.done
    assert [g.outputs["value"] for _, g in given] == ["a", "b"]  # type: ignore[union-attr]
    assert _kinds(run.run_dir) == ["finished", "finished"]


def test_two_runs_on_a_caller_owned_session_load_the_model_once(tmp_path: Path, planted: Path) -> None:
    """Two runs of two plans on one session's worker process load the backend once; a repeated key is a reuse."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        session = PoolSession(RESIDENT, tmp_path / "session", tmp_path / "store")
        try:
            first, values = planned(TASK, RESIDENT, ["a"])
            assert start(first, session, values=values).wait(timeout=180).done
            second, values = planned(TASK, RESIDENT, ["a", "b", "c"])
            run = start(second, session, values=values)
            assert run.wait(timeout=180).done
        finally:
            session.close()
    assert _loads(tmp_path) == 1
    assert _kinds(run.run_dir) == ["cache_hit", "finished", "finished"]


def test_two_calls_without_a_session_load_the_model_once(tmp_path: Path, planted: Path) -> None:
    """Two calls given no session run on the process's default session: one worker process, one load."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        first = call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "a"}}), profile=RESIDENT)
        second = call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "b"}}), profile=RESIDENT)
        assert len(warm._DEFAULTS) == 1
    assert (first.outputs["value"], second.outputs["value"]) == ("a", "b")
    assert _loads(tmp_path) == 1


def test_a_run_whose_every_job_is_a_store_hit_sends_nothing_to_the_pool(tmp_path: Path, planted: Path) -> None:
    """A second run of done jobs, on a session with no worker, is served by its coordinator: each a `cache_hit`."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        with served(tmp_path / "first", store=tmp_path / "store") as (session, _):
            placed, values = planned(TASK, PROFILE, ["a", "b", "b"])
            assert start(placed, session, values=values).wait(timeout=120).done
        with served(tmp_path / "second", store=tmp_path / "store", workers=0) as (session, _):
            again, values = planned(TASK, PROFILE, ["b", "a"])
            run = start(again, session, values=values)
            assert run.wait(timeout=60).done
    assert _kinds(run.run_dir) == ["cache_hit", "cache_hit"] and _loads(tmp_path) == 1
    assert not [e for e in read_events(run.run_dir) if e.kind == "started"]


def test_explicit_values_and_the_site_hook_are_the_worker_classes_a_run_opens(tmp_path: Path, planted: Path) -> None:
    """A run's explicit time and its hook's worker cap place its default session's class; another session is refused."""
    two = RESIDENT.model_copy(
        update={"resource_classes": {"cpu": RESIDENT.resource_classes["cpu"].model_copy(update={"max_workers": 2})}}
    )

    def hook(request: PlacementRequest) -> PlacementAnswer:  # noqa: ARG001 - answers every request alike
        return PlacementAnswer(classes={"cpu": PlacementValues(max_workers=1)}, why="one worker")

    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        placed, values = planned(TASK, two, ["a"])
        placed = replace(placed, overrides=load_overrides({"cpu": {"time_s": 600}}), hook=hook)
        with PoolSession(two, tmp_path / "session", tmp_path / "cache" / "2x" / "results") as other:
            with pytest.raises(ValueError, match="other placement values"):
                start(placed, other, values=values)
        run = start(placed, values=values)
        assert run.wait(timeout=180).done
        [opened] = warm._DEFAULTS.values()
    [cpu] = opened.classes
    assert (cpu.time, cpu.max_workers) == (600, 1)
    [placement, closing] = [e.placement for e in read_events(run.run_dir) if e.kind == "placement"]
    assert placement is not None and closing is not None
    assert (placement.used.time_s, placement.used.max_workers, placement.verdict) == (600, 1, "accepted")


def test_a_process_holds_one_default_session_and_closes_the_one_it_replaces(tmp_path: Path, planted: Path) -> None:
    """A call under other worker classes closes the default session the process holds and opens one for them."""
    two = RESIDENT.model_copy(
        update={"resource_classes": {"cpu": RESIDENT.resource_classes["cpu"].model_copy(update={"max_workers": 2})}}
    )
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "a"}}), profile=RESIDENT)
        [first] = warm._DEFAULTS.values()
        call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "b"}}), profile=two)
        [second] = warm._DEFAULTS.values()
    assert second is not first and first.pool._closed and not second.pool._closed


def test_a_failed_open_of_a_new_default_session_closes_the_one_it_replaced(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next default session's open raises (a Slurm pool under the temporary directory): the one it replaced closes.

    No session is held after; the cache of the second call is under the interpreter's temporary directory whatever
    pytest's own base directory is.
    """
    local = Path(tempfile.mkdtemp(prefix="senselab-node-local-"))
    try:
        with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
            call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "a"}}), profile=RESIDENT)
            [first] = warm._DEFAULTS.values()
            monkeypatch.setenv("SENSELAB_CACHE", str(local))
            with pytest.raises(ValueError, match="every node") as refused:
                call(
                    TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "b"}}),
                    profile=ExecutionProfile.load(CLUSTER),
                )
    finally:
        shutil.rmtree(local, ignore_errors=True)
    assert type(refused.value).__name__ == "NodeLocalStorage"
    assert first.pool._closed and not warm._DEFAULTS and not warm._RETIRED


def test_a_replaced_default_session_is_closed_only_when_its_live_run_ends(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call under other worker classes retires the session a live run is on; the run ends done, then it closes."""
    monkeypatch.setenv("SENSELAB_TEST_SLEEP_S", "1")
    two = RESIDENT.model_copy(
        update={"resource_classes": {"cpu": RESIDENT.resource_classes["cpu"].model_copy(update={"max_workers": 2})}}
    )
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        placed, values = planned(TASK, RESIDENT, ["a", "b", "c"])
        live = start(placed, values=values)
        [first] = warm._DEFAULTS.values()
        given = call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "d"}}), profile=two)
        assert given.outputs["value"] == "d" and not first.pool._closed
        assert live.wait(timeout=100).done
        deadline = time.monotonic() + 10
        while not first.pool._closed and time.monotonic() < deadline:
            time.sleep(0.05)
    assert first.pool._closed and _kinds(live.run_dir) == ["finished"] * 3


def test_callers_that_first_use_a_key_at_once_share_one_opened_session(tmp_path: Path, planted: Path) -> None:
    """Threads making their first call at once open one default session between them, and no other pool root."""
    gate = threading.Barrier(4)
    given: list[object] = []

    def first(text: str) -> None:
        gate.wait()
        given.append(call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": text}}), profile=RESIDENT))

    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        threads = [threading.Thread(target=first, args=(t,)) for t in "abcd"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(100)
        held = list(warm._DEFAULTS.values())
    assert len(given) == 4 and len(held) == 1
    assert [p.name for p in (tmp_path / "cache" / warm.POOLS_DIR).iterdir()] == [held[0].root.name]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX")
def test_a_forked_child_opens_its_own_default_session_and_leaves_the_parents_workers(
    tmp_path: Path, planted: Path
) -> None:
    """A call in a forked child runs on a session of its own; the child's exit stops none of the parent's workers."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "a"}}), profile=RESIDENT)
        [parents] = warm._DEFAULTS.values()
        pid = os.fork()
        if pid == 0:  # the child: one call, then what its exit runs
            code = 1
            try:
                given = call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "b"}}), profile=RESIDENT)
                [own] = warm._DEFAULTS.values()
                code = 0 if given.outputs["value"] == "b" and own is not parents else 2
                warm.close_default_sessions()
            finally:
                os._exit(code)
        deadline = time.monotonic() + 100
        while (ended := os.waitpid(pid, os.WNOHANG)) == (0, 0) and time.monotonic() < deadline:
            time.sleep(0.2)
        if ended == (0, 0):
            os.kill(pid, 9)
            os.waitpid(pid, 0)
        assert ended[0] == pid and os.waitstatus_to_exitcode(ended[1]) == 0
        third = call(TaskCall.model_validate({"task": TASK.name, "inputs": {"text": "c"}}), profile=RESIDENT)
        assert list(warm._DEFAULTS.values()) == [parents] and not parents.pool._closed
    assert third.outputs["value"] == "c" and _loads(tmp_path) == 2


def test_a_load_out_of_memory_moves_the_job_to_the_larger_class(tmp_path: Path, planted: Path) -> None:
    """A load that runs out of memory on the small class is retried on the big one, uncharged, and gives the result."""
    (tmp_path / "oom").write_text("1")
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path, TWO_CLASSES) as (session, pool):
        pool.serve("cpu:big")
        placed, values = planned(TASK, TWO_CLASSES, ["a"])
        assert placed.placement.nodes["a"].resource_class == "cpu"
        run = start(placed, session, values=values)
        assert run.wait(timeout=180).done
        [(_, given)] = list(run.results())
        stored = session.store.stored(next(iter(given.keys.values())))  # type: ignore[union-attr]
    assert given.outputs["value"] == "a"  # type: ignore[union-attr]
    assert [context["pool"]["class"] for _, context, _ in stored] == ["cpu:big"]
    assert _loads(tmp_path) == 2


def test_a_later_session_reads_an_earlier_ones_unfit_record_and_starts_on_the_larger_class(
    tmp_path: Path, planted: Path
) -> None:
    """A load's unfit record is kept under the measures root: a session over another root starts on the big class."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        (tmp_path / "oom").write_text("1")
        with served(tmp_path / "first", TWO_CLASSES, store=tmp_path / "store") as (session, pool):
            pool.serve("cpu:big")
            placed, values = planned(TASK, TWO_CLASSES, ["a"])
            assert start(placed, session, values=values).wait(timeout=180).done
            first = session.calibration.root
        with served(tmp_path / "second", TWO_CLASSES, store=tmp_path / "store") as (session, pool):
            pool.serve("cpu:big")
            assert session.calibration.root == first and first.is_relative_to(tmp_path / "cache")
            placed, values = planned(TASK, TWO_CLASSES, ["b"])
            run = start(placed, session, values=values)
            assert run.wait(timeout=180).done
            [(_, given)] = list(run.results())
            stored = session.store.stored(next(iter(given.keys.values())))  # type: ignore[union-attr]
    assert [context["pool"]["class"] for _, context, _ in stored] == ["cpu:big"]
    assert _loads(tmp_path) == 3


def test_under_own_trust_another_users_unfit_record_does_not_steer_placement(tmp_path: Path, planted: Path) -> None:
    """A group-trusted session's unfit record is not read by an own-trusted one, whose calibration is the uid's."""
    group = TWO_CLASSES.model_copy(
        update={"calibration": TWO_CLASSES.calibration.model_copy(update={"trust": "group"})}
    )
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        (tmp_path / "oom").write_text("1")
        with served(tmp_path / "first", group, store=tmp_path / "store") as (session, pool):
            pool.serve("cpu:big")
            placed, values = planned(TASK, group, ["a"])
            assert start(placed, session, values=values).wait(timeout=100).done
            shared = session.calibration.root
        with served(tmp_path / "second", TWO_CLASSES, store=tmp_path / "store") as (session, pool):
            pool.serve("cpu:big")
            assert session.calibration.root == shared / str(os.geteuid())
            placed, values = planned(TASK, TWO_CLASSES, ["b"])
            run = start(placed, session, values=values)
            assert run.wait(timeout=100).done
            [(_, given)] = list(run.results())
            stored = session.store.stored(next(iter(given.keys.values())))  # type: ignore[union-attr]
    assert [context["pool"]["class"] for _, context, _ in stored] == ["cpu"]


def test_off_host_workers_get_only_tolerant_unsampled_hits_served_by_the_coordinator(
    tmp_path: Path, planted: Path
) -> None:
    """Off-host workers: the coordinator serves a tolerant, unsampled run's hit; an exact run's goes to the pool."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"):
        with served(tmp_path / "seed", store=tmp_path / "store") as (session, _):
            placed, values = planned(TASK, PROFILE, ["a"])
            assert start(placed, session, values=values).wait(timeout=100).done
        tolerant = RunConfig(ReusePolicy.TOLERANT, 0.0)
        with served(tmp_path / "remote", store=tmp_path / "store", workers=0, here=False) as (session, _):
            again, values = planned(TASK, PROFILE, ["a"], tolerant)
            served_here = start(again, session, values=values)
            assert served_here.wait(timeout=60).done
        with served(tmp_path / "exact", store=tmp_path / "store", here=False) as (session, _):
            exact, values = planned(TASK, PROFILE, ["a"], RunConfig(ReusePolicy.EXACT, 0.0))
            sent = start(exact, session, values=values)
            assert sent.wait(timeout=100).done
    assert [e.kind for e in read_events(served_here.run_dir) if e.kind in {"started", "cache_hit"}] == ["cache_hit"]
    assert [e.kind for e in read_events(sent.run_dir) if e.kind in {"started", "cache_hit"}] == ["started", "cache_hit"]
    assert _loads(tmp_path) == 2


def test_a_residency_broken_past_its_budget_holds_its_jobs_and_the_run(tmp_path: Path, planted: Path) -> None:
    """A load that fails for no resource reason, past the infrastructure budget, holds the job: no terminal event."""
    (tmp_path / "broken").write_text("")
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path, TWO_CLASSES) as (session, _):
        placed, values = planned(TASK, TWO_CLASSES, ["a"])
        run = start(placed, session, values=values)
        with pytest.raises(QueueHeld):
            run.wait(timeout=180)
    assert _kinds(run.run_dir) == []
    assert not run.status().done


def test_a_license_refusal_fails_every_job_and_is_not_raised(tmp_path: Path, planted: Path) -> None:
    """A run whose policy refuses the backend env's license fails each job with `LicenseRefused`; nothing is sent."""
    refusing = RunConfig(licenses=LicensePolicy(refuse=frozenset({"MIT"})))
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path) as (session, _):
        placed, values = planned(TASK, PROFILE, ["a", "b"])
        run = start(replace(placed, config=refusing), session, values=values)
        status = run.wait(timeout=120)
    failed = [e.failure for e in read_events(run.run_dir) if e.kind == "failed"]
    assert status.done
    assert [f.type for f in failed if f is not None] == ["LicenseRefused", "LicenseRefused"]
    assert _loads(tmp_path) == 0


def test_a_cancelled_run_withdraws_its_unstarted_jobs(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel withdraws the jobs the pool has not started; they end cancelled, and the run is cancelled."""
    monkeypatch.setenv("SENSELAB_TEST_SLEEP_S", "0.5")
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path) as (session, _):
        placed, values = planned(TASK, PROFILE, [str(i) for i in range(12)])
        run = start(placed, session, values=values)
        deadline = time.monotonic() + 60
        while not any(e.kind == "finished" for e in read_events(run.run_dir)) and time.monotonic() < deadline:
            time.sleep(0.05)
        run.cancel()
        status = run.wait(timeout=120)
    kinds = [e.kind for e in read_events(run.run_dir)]
    assert status.cancelled
    assert "cancelled" in kinds and kinds.count("finished") < 12


def test_the_store_entry_records_the_pool_facts_beside_a_context_it_never_splits_on(
    tmp_path: Path, planted: Path
) -> None:
    """An entry's context records where it ran (`pool`) and the run job (`senselab`); neither enters its slot."""
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path) as (session, _):
        placed, values = planned(TASK, PROFILE, ["a"])
        run = start(placed, session, values=values)
        run.wait(timeout=120)
        [(_, given)] = list(run.results())
        [(entry, context, _)] = session.store.stored(next(iter(given.keys.values())))  # type: ignore[union-attr]
        fingerprint = session.store.identity.fingerprint(context)
    assert context["pool"]["class"] == "cpu" and context["senselab"]["node"] == "a"
    assert entry.fingerprint == fingerprint
    assert json.loads(json.dumps(context["senselab"]))["plan"] == placed.hash


def test_a_run_whose_pool_starts_no_worker_fails_within_its_no_progress_bound(tmp_path: Path, planted: Path) -> None:
    """No worker ever attaches: the run raises `NoProgress` naming the waiting class, within the profile's bound."""
    with (
        packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"),
        served(tmp_path, BOUNDED, workers=0) as (session, _),
    ):
        placed, values = planned(TASK, BOUNDED, ["a"])
        began = time.monotonic()
        run = start(placed, session, values=values)
        with pytest.raises(NoProgress, match="cpu") as caught:
            run.wait(timeout=60)
        took = time.monotonic() - began
    assert caught.value.classes == ("cpu",)
    assert caught.value.bound_s == BOUND_S
    assert BOUND_S <= took < BOUND_S + 10


def test_a_job_longer_than_the_bound_completes_while_its_worker_beats_its_claim(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One job runs past `no_progress_s` on a worker whose claim lease beats: the run ends done, not `NoProgress`."""
    monkeypatch.setenv("SENSELAB_TEST_SLEEP_S", str(3 * BOUND_S))
    with (
        packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"),
        served(tmp_path, BOUNDED) as (session, _),
    ):
        placed, values = planned(TASK, BOUNDED, ["a"])
        run = start(placed, session, values=values)
        status = run.wait(timeout=60)
    assert status.done and _kinds(run.run_dir) == ["finished"]


def test_a_class_whose_worker_is_still_starting_is_not_stalled(tmp_path: Path, planted: Path) -> None:
    """A worker job that stays queued past the bound keeps the run waiting; the run is then cancelled, not stalled."""
    with (
        packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"),
        PoolSession(BOUNDED, tmp_path / "session", tmp_path / "store", substrate=_Queued(), here=True) as session,
    ):
        placed, values = planned(TASK, BOUNDED, ["a"])
        run = start(placed, session, values=values)
        time.sleep(3 * BOUND_S)
        assert not run.status().done
        run.cancel()
        status = run.wait(timeout=60)
    assert status.cancelled


def test_a_class_whose_starts_are_refused_holds_its_jobs_with_the_start_error(tmp_path: Path, planted: Path) -> None:
    """Every start is refused: past its start deadline, and inside the no-progress bound, the pool breaks the class.

    The run raises the hold, naming the class and the refusal; the job has no terminal event.
    """
    with (
        packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"),
        PoolSession(DEADLINE, tmp_path / "session", tmp_path / "store", substrate=_Refusing(), here=True) as session,
    ):
        placed, values = planned(TASK, DEADLINE, ["a"])
        run = start(placed, session, values=values)
        with pytest.raises(QueueHeld, match="invalid partition") as caught:
            run.wait(timeout=60)
    assert caught.value.resource_class == "cpu" and caught.value.jobs == 1
    assert _kinds(run.run_dir) == []


def test_a_slurm_session_refuses_a_pool_root_on_node_local_storage(tmp_path: Path) -> None:
    """A Slurm substrate's workers run on other nodes: pydra refuses a root under the temporary directory, unwritten."""
    profile = ExecutionProfile.load(CLUSTER)
    root = Path(tempfile.gettempdir()) / f"senselab-node-local-{os.getpid()}" / "session"
    with pytest.raises(ValueError, match="every node") as refused:
        PoolSession(profile, root, tmp_path / "store")
    assert type(refused.value).__name__ == "NodeLocalStorage" and not root.exists()


def test_a_running_jobs_claim_lease_is_where_the_pool_layout_puts_it_and_beats(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a job runs, `claim_leases` finds its ticket's claim in pydra's pool layout and `newest_beat` moves."""
    monkeypatch.setenv("SENSELAB_TEST_SLEEP_S", "4")
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path) as (session, _):
        placed, values = planned(TASK, PROFILE, ["a"])
        run = start(placed, session, values=values)
        pool = session.root / "pool"
        deadline = time.monotonic() + 30
        while not claim_leases(pool, ["cpu"]) and time.monotonic() < deadline:
            time.sleep(0.05)
        [lease] = claim_leases(pool, ["cpu"])
        first = newest_beat(lease)
        time.sleep(2)
        later = newest_beat(lease)
        assert claim_leases(pool, ["cpu:other"]) == []
        assert run.wait(timeout=60).done
    assert first is not None and later is not None and later > first


def test_a_class_broken_in_one_run_is_attempted_afresh_by_the_next_run_on_the_session(
    tmp_path: Path, planted: Path
) -> None:
    """Run 1 is held while the substrate refuses; once it starts workers again, run 2 on the same session completes."""
    outage = _Outage()
    with (
        packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"),
        PoolSession(DEADLINE, tmp_path / "session", tmp_path / "store", substrate=outage, here=True) as session,
    ):
        placed, values = planned(TASK, DEADLINE, ["a"])
        with pytest.raises(QueueHeld, match="controller not responding"):
            start(placed, session, values=values).wait(timeout=60)
        outage.down = False
        again, values = planned(TASK, DEADLINE, ["b"])
        run = start(again, session, values=values)
        status = run.wait(timeout=100)
    assert status.done and _kinds(run.run_dir) == ["finished"]


def test_a_failed_listing_at_the_no_progress_check_counts_as_working(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unknown state is not a stall: a substrate whose listing raises leaves the session working, logged."""
    with PoolSession(DEADLINE, tmp_path / "session", tmp_path / "store", substrate=_Unlisted(), here=True) as session:
        assert session.working(["cpu"]) is True
    assert "could not run squeue" in caplog.text


def test_a_plan_placed_in_one_shell_runs_in_another_whose_env_root_differs(
    tmp_path: Path, planted: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A profile's identity is its values as written: another `SENSELAB_ENV_ROOT` changes no plan's profile."""
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**30, "max_workers": 1}},
        "pool": {"poll_s": 0.05, "quantum_items": 8, "quantum_s": 60, "idle_polls": 40},
        "lease": {"stale_after_s": 4.0, "renew_fraction": 0.2, "attribute_delay_s": 0, "wait_s": 10.0},
        "env_roots": {"isolated": "${SENSELAB_TEST_ENV_ROOT}/v", "server": "${SENSELAB_TEST_ENV_ROOT}/s"},
    }
    monkeypatch.setenv("SENSELAB_TEST_ENV_ROOT", str(tmp_path / "first"))
    first = ExecutionProfile.load("laptop", override)
    monkeypatch.setenv("SENSELAB_TEST_ENV_ROOT", str(tmp_path / "second"))
    second = ExecutionProfile.load("laptop", override)
    assert first.env_root("isolated") != second.env_root("isolated")
    assert first.canonical_digest() != second.canonical_digest()
    with packaged(registry(TASK, worker=BACKEND), tmp_path / "cache"), served(tmp_path, second) as (session, _):
        placed, values = planned(TASK, first, ["a"])
        assert start(placed, session, values=values).wait(timeout=100).done
