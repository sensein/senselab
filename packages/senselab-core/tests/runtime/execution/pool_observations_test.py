"""What a pool worker reports per request: walls, outcomes, co-tenancy and hardware, and how it serves a probe queue."""

import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from planted_objects_fixture import identity_leaks
from pool_stack_fixture import packaged, registry, scoring
from pool_worker_test import WORKERS, _outcomes, _profile, _put, _run, _sender, _text
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome
from senselab_core.runtime.execution.observing import hardware_of
from senselab_core.runtime.execution.pool import split
from senselab_core.runtime.execution.pool_worker import PoolWorker, Send
from senselab_core.runtime.execution.queue import QueueSet, probe_queue
from senselab_core.runtime.execution.substrates import mark_ready
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerLost

TIMED = scoring("pool_observations_test.timed")
TIMER = "backends.timed_backend_fixture"
HARDWARE = HardwareClass(device="cpu", name="a cpu", cpus=2)
SHARED = Cotenancy(device=2, host=3)


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """A registry whose one task is served by the timed backend; every env this interpreter; cache under `tmp_path`."""
    with packaged(registry(TIMED, worker=TIMER), tmp_path / "cache"):
        yield
    WORKERS.close()


def _observations(pool_dir: Path, profile: ExecutionProfile, queue: str = "cpu") -> list[Observation]:
    return [
        Observation.model_validate(o)
        for out in _outcomes(pool_dir, profile, queue)
        for o in out.get("observations", [])
    ]


def _worker(
    pool_dir: Path, queues: Sequence[str], profile: ExecutionProfile, store: ResultStore, send: Send
) -> PoolWorker:
    return PoolWorker(
        pool_dir,
        queues,
        profile=profile,
        store=store,
        send=send,
        device_memory=lambda: None,
        hardware=lambda _: HARDWARE,
        cotenancy=lambda _: SHARED,
    )


@pytest.fixture
def obs(slack_tests: dict[str, Any]) -> dict[str, Any]:
    """The worker-observation test slack."""
    section: dict[str, Any] = slack_tests["observations"]
    return section


def _items(profile: ExecutionProfile, tmp_path: Path, texts: Sequence[str]) -> tuple[Any, ResultStore, Path]:
    return _run(tmp_path, profile, texts, TIMED)


def _one_job_items(placed: Any, elements: range) -> list[Any]:  # noqa: ANN401 - a plan
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    return split(node_fragment(placed, admission, elements=[(i,) for i in elements]), 1)


def _near(want: Sequence[float], got: Sequence[float], slack: float) -> bool:
    return all(w <= g < w + slack for w, g in zip(want, got, strict=True))


@pytest.mark.logic
def test_gathered_items_report_their_own_request_walls(tmp_path: Path, obs: dict[str, Any]) -> None:
    """Four items gathered into a claim, each its own request of 0.05 s more than the last, are timed one by one."""
    profile = _profile(quantum_items=4)
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0", "0.05", "0.1", "0.15", "0.2"])
    _put(placed, pool_dir, 0)
    QueueSet(pool_dir, profile).queue("cpu").put_all(_one_job_items(placed, range(1, 5)))
    _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    gathered = sorted(
        (o for out in _outcomes(pool_dir, profile) for o in out["observations"] if o["since_load"] > 0),
        key=lambda o: o["wall_s"],
    )
    assert [o["jobs"] for o in gathered] == [1, 1, 1, 1]
    wants = (0.05, 0.1, 0.15, 0.2)
    assert _near(wants, [o["wall_s"] for o in gathered], obs["wall_slack_s"])
    samples = sorted(out["measure"]["wall_s"] for out in _outcomes(pool_dir, profile) if out["measure"]["jobs"] == 1)
    assert _near(wants, samples[1:], obs["wall_slack_s"])


@pytest.mark.logic
def test_a_bisected_request_is_reported_bisected(tmp_path: Path) -> None:
    """A call that fails whole and is split is `bisected`, a lone job that fails `failed`, a request that ran `ok`."""
    profile = _profile(quantum_items=8)
    texts = [f"0.0{i}" for i in range(8)]
    placed, store, pool_dir = _items(profile, tmp_path, texts)
    _put(placed, pool_dir, range(8))
    send = _sender(store, profile)

    def poisoned(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        if any(_text(j) == "0.05" for j in jobs):
            raise WorkerLost("the worker died on its request", returncode=-11)
        return send(jobs, placement)

    _worker(pool_dir, ["cpu"], profile, store, poisoned).serve()
    found = _observations(pool_dir, profile)
    assert Counter(o.outcome for o in found) == {"bisected": 3, "failed": 1, "ok": 7}
    assert sorted(o.jobs for o in found if o.outcome == "bisected") == [2, 4, 8]
    assert [o.jobs for o in found if o.outcome == "failed"] == [1]
    assert all(o.since_load is None for o in found if o.outcome != "ok")


@pytest.mark.logic
def test_observations_carry_device_cotenancy(tmp_path: Path) -> None:
    """Each request's observation carries the co-tenancy and the hardware class read for its claim."""
    profile = _profile()
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0", "0.00"])
    for element in range(2):
        _put(placed, pool_dir, element)
    _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    found = _observations(pool_dir, profile)
    assert len(found) == 2 and {(o.cotenancy, o.hardware) for o in found} == {(SHARED, HARDWARE)}
    assert [o.probe for o in found] == [None, None] and all(o.unit_kind == "none" and o.units == 0 for o in found)


@pytest.mark.logic
def test_an_unread_device_keeps_the_observation_with_no_device_count(tmp_path: Path) -> None:
    """A device count that could not be read is recorded as None; the claim's observations are still reported."""
    profile = _profile()
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0"])
    _put(placed, pool_dir, 0)
    worker = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    worker._cotenancy = lambda _: Cotenancy(device=None, host=1)  # noqa: SLF001 - the seam under test, set after build
    worker.serve()
    [found] = _observations(pool_dir, profile)
    assert found.cotenancy == Cotenancy(device=None, host=1) and found.outcome == "ok"


@pytest.mark.logic
def test_no_run_record_names_the_host_the_account_or_a_home_path(tmp_path: Path) -> None:
    """A worker left to name itself leaves no host name, account name or home path in any file of the run."""
    profile = _profile()
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0", "0.00"])
    for element in range(2):
        _put(placed, pool_dir, element)
    PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        device_memory=lambda: None,
        hardware=lambda _: HARDWARE,
        cotenancy=lambda _: SHARED,
    ).serve()
    mark_ready(tmp_path / "run", {"cpu": HARDWARE})
    assert [p for p in (tmp_path / "run").rglob("*") if p.is_file()]
    assert identity_leaks(tmp_path / "run", tmp_path) == []


@pytest.mark.logic
def test_a_worker_reads_its_own_hardware_and_alone_on_its_host_it_counts_one(tmp_path: Path) -> None:
    """Without seams a cpu class reads this host's hardware, and a pool of one worker is co-tenancy 1 on 1."""
    profile = _profile()
    try:
        expected = hardware_of(profile, "cpu")
    except ValueError:
        pytest.skip("this host reports no CPU model")
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0"])
    _put(placed, pool_dir, 0)
    PoolWorker(pool_dir, ["cpu"], profile=profile, store=store, send=_sender(store, profile)).serve()
    [found] = _observations(pool_dir, profile)
    assert (found.hardware, found.cotenancy) == (expected, Cotenancy(device=1, host=1))


def _probe_items(
    tmp_path: Path, count: int, obs: dict[str, Any]
) -> tuple[Any, ResultStore, Path, ExecutionProfile, Any]:
    profile = _profile(
        quantum_items=4, poll_s=obs["poll_s"], first_poll_s=obs["first_poll_s"], presence_refresh_s=obs["poll_s"]
    )
    placed, store, pool_dir = _items(profile, tmp_path, [f"0.00{i}" for i in range(count + 1)])
    queue = QueueSet(pool_dir, profile).queue(probe_queue("cpu", "p1"))
    for item in _one_job_items(placed, range(count)):
        queue.put_probe(item, "SINGLE", 1)
    return placed, store, pool_dir, profile, queue


@pytest.mark.logic
def test_probe_items_are_claimed_one_per_claim_by_the_serve_lease_holder(tmp_path: Path, obs: dict[str, Any]) -> None:
    """Another worker holding the lease keeps this one off the probe queue; once released, items go one per claim."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 4, obs)
    held = queue.serve_lease()
    assert held is not None
    sizes: list[int] = []
    send = _sender(store, profile)

    def recorded(jobs: Sequence[Job], placement: Placement) -> list[JobOutcome]:
        sizes.append(len(jobs))
        return send(jobs, placement)

    name = probe_queue("cpu", "p1")
    skipped = _worker(pool_dir, ["cpu"], profile, store, recorded)
    skipped.serve()
    assert skipped.served == [] and sizes == []
    held.release()
    taker = _worker(pool_dir, ["cpu"], profile, store, recorded)
    taker.serve()
    assert [q for q, _ in taker.served] == [name] * 4 and sizes == [1, 1, 1, 1]
    found = _observations(pool_dir, profile, name)
    assert len(found) == 4 and {(o.probe.id, o.probe.phase, o.probe.stratum) for o in found if o.probe} == {
        ("p1", "SINGLE", 1)
    }
    assert queue.serve_lease() is not None


@pytest.mark.logic
def test_an_unmarked_probe_item_reports_no_observation_and_is_never_telemetry(
    tmp_path: Path, obs: dict[str, Any]
) -> None:
    """A probe item whose mark is gone commits its result with no observation, not one without a probe."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 2, obs)
    for mark in (queue.path / "probe").iterdir():
        mark.unlink()
    _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    done = _outcomes(pool_dir, profile, probe_queue("cpu", "p1"))
    assert len(done) == 2 and not [o for out in done for o in out.get("observations", [])]


@pytest.mark.logic
def test_a_worker_lists_probe_queues_once_per_refresh(tmp_path: Path, obs: dict[str, Any]) -> None:
    """A probe queue made after the worker listed is not served until the refresh interval has passed."""
    profile = _profile(poll_s=obs["poll_s"], presence_refresh_s=obs["presence_refresh_s"])
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0", "0.00"])
    now = [0.0]
    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        device_memory=lambda: None,
        hardware=lambda _: HARDWARE,
        cotenancy=lambda _: SHARED,
        clock=lambda: now[0],
    )
    assert worker._next(None, 4) is None  # noqa: SLF001 - the claim step, with nothing queued yet
    QueueSet(pool_dir, profile).queue(probe_queue("cpu", "p1")).put_probe(
        _one_job_items(placed, range(1))[0], "SINGLE", 0
    )
    assert worker._next(None, 4) is None  # noqa: SLF001 - listed within the refresh: not seen
    now[0] = obs["presence_refresh_s"] + 1
    claim = worker._next(None, 4)  # noqa: SLF001 - listed again: seen and taken
    assert claim is not None and claim.queue.name == probe_queue("cpu", "p1")
    claim.release()


@pytest.mark.logic
def test_a_pinned_worker_waits_on_its_probe_queue_from_the_first_poll(tmp_path: Path, obs: dict[str, Any]) -> None:
    """With a long poll, an item put just after the last is claimed within the first poll, not after the long one."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 3, obs)
    later = _one_job_items(placed, range(3, 4))[0]
    profile = _profile(quantum_items=4, poll_s=obs["long_poll_s"], first_poll_s=obs["first_poll_s"])
    send = _sender(store, profile)

    def put_later() -> None:
        time.sleep(obs["put_later_s"])
        queue.put_probe(later, "SINGLE", 1)

    def stop_after_four() -> None:
        while len(worker.served) < 4:
            time.sleep(obs["watch_poll_s"])
        reached.append(time.monotonic())
        worker.halt()

    worker = _worker(pool_dir, ["cpu"], profile, store, send)
    reached: list[float] = []
    threads = [threading.Thread(target=put_later), threading.Thread(target=stop_after_four, daemon=True)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    worker.serve()
    threads[0].join()
    assert len(worker.served) == 4 and reached[0] - began < obs["first_bound_s"]
