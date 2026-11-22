"""Runs on senselab's warm pool session: results, warm reuse, a load out of memory, a hold, a license, a cancel."""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.runtime.call import RunConfig, TaskCall
from senselab_core.runtime.execution import warm
from senselab_core.runtime.execution.calls import call
from senselab_core.runtime.execution.events import read_events
from senselab_core.runtime.execution.executor import QueueHeld
from senselab_core.runtime.execution.run import start
from senselab_core.runtime.execution.warm import PoolSession
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
