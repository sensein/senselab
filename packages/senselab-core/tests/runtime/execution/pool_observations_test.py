"""What a pool worker reports per request: walls, outcomes, co-tenancy and hardware, and how it serves a probe queue."""

import threading
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from backends import fake_time_backend_fixture
from faulty_workers_fixture import FOREVER, Faulty, crashes, put_batched, sender
from lease_clock_fixture import ManualClock
from planted_objects_fixture import identity_leaks
from pool_stack_fixture import packaged, registry, scoring
from pool_worker_test import OVERRIDE, WORKERS, _outcomes, _profile, _put, _run, _sender, _served
from senselab_core.kernel.declare import Env
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Job, JobOutcome
from senselab_core.runtime.execution.observing import hardware_of
from senselab_core.runtime.execution.pool import split
from senselab_core.runtime.execution.pool_worker import PoolWorker, Send
from senselab_core.runtime.execution.queue import QueueSet, probe_queue
from senselab_core.runtime.execution.substrates import QueueReady, mark_ready
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_worker.fake import OutOfMemoryError

TIMED = scoring("pool_observations_test.timed")
TIMER = "backends.timed_backend_fixture"
FAKE_TIME = "backends.fake_time_backend_fixture"
CROWDED = "backends.crowding_backend_fixture"
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


def _worker(  # noqa: PLR0913 - the queues and what the worker serves them with, then its clock
    pool_dir: Path,
    queues: Sequence[str],
    profile: ExecutionProfile,
    store: ResultStore,
    send: Send,
    clock: ManualClock | None = None,
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
        **({} if clock is None else {"clock": clock}),
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


@pytest.mark.logic
def test_gathered_items_report_their_own_request_walls(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """Four items gathered into a claim, each its own request, are each charged their own request's span, no mean.

    The worker's clock is a fake one that the backend advances, by the seconds each item names, so a request's
    span is exactly its item's seconds, in binary fractions that subtract exactly. The test advances it only while the
    worker waits for work. Claiming takes none, and an item's claim-to-commit wall is that of the one call its
    claim's items were sent in: element 0 spends one of the quantum's four items, so elements 1 to 3 are the
    next claim and element 4 the last.
    """
    profile = _profile(quantum_items=4)
    texts = ["0", "0.25", "0.5", "0.75", "1"]
    fake_time_backend_fixture.NOW[0] = 0.0
    clock = lease_clock(cell=fake_time_backend_fixture.NOW)
    with packaged(registry(TIMED, worker=FAKE_TIME), tmp_path / "cache"):
        placed, store, pool_dir = _items(profile, tmp_path, texts)
        _put(placed, pool_dir, 0)
        QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu").put_all(_one_job_items(placed, range(1, 5)))
        worker = PoolWorker(
            pool_dir,
            ["cpu"],
            profile=profile,
            store=store,
            send=_sender(store, profile),
            device_memory=lambda: None,
            hardware=lambda _: HARDWARE,
            cotenancy=lambda _: SHARED,
            clock=clock,
        )
        _served(worker, clock)
    outcomes = {out["jobs"][0]["job"]["element"][0]: out for out in _outcomes(pool_dir, profile)}
    assert sorted(outcomes) == [0, 1, 2, 3, 4] and all(len(out["jobs"]) == 1 for out in outcomes.values())
    assert sorted(outcomes[e]["observations"][0]["since_load"] for e in range(1, 5)) == [1, 2, 3, 4]
    for element, out in outcomes.items():
        (observed,) = out["observations"]
        assert (observed["jobs"], out["measure"]["jobs"]) == (1, 1)
        assert observed["wall_s"] == out["measure"]["wall_s"] == float(texts[element])
        assert out["measure"]["claim_s"] == 0.0
    for claim in ([0], [1, 2, 3], [4]):
        spent = sum(float(texts[e]) for e in claim)
        assert [outcomes[e]["observations"][0]["item_wall_s"] for e in claim] == [spent] * len(claim)


@pytest.mark.logic
def test_a_bisected_request_is_reported_bisected(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A request that is split is `bisected`, a lone job lost twice `failed` twice, the rest `ok`."""
    profile = _profile(quantum_items=8)
    texts = [f"0.0{i}" for i in range(8)]
    placed, store, pool_dir = _items(profile, tmp_path, texts)
    put_batched(placed, pool_dir, profile, 8)
    workers = crashes({"0.05": FOREVER})
    clock = lease_clock()
    _served(_worker(pool_dir, ["cpu"], profile, store, sender(store, profile, workers), clock), clock)
    found = _observations(pool_dir, profile)
    assert Counter(o.outcome for o in found) == {"bisected": 3, "failed": 2, "ok": 3}
    assert sorted(o.jobs for o in found if o.outcome == "bisected") == [2, 4, 8]
    assert "resource" not in {o.fault for o in found}
    assert [o.jobs for o in found if o.outcome == "failed"] == [1, 1]
    assert all(o.since_load is None for o in found if o.outcome != "ok")


@pytest.mark.logic
def test_a_request_reports_the_class_of_the_fault_it_raised(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """Requests of over two jobs run out of memory and split; each says `resource`, the ones that ran say nothing."""
    profile = _profile(quantum_items=8)
    texts = [f"0.0{i}" for i in range(8)]
    placed, store, pool_dir = _items(profile, tmp_path, texts)
    put_batched(placed, pool_dir, profile, 8)
    workers = Faulty(lambda seen: OutOfMemoryError("out of memory") if len(seen) > 2 else None)
    clock = lease_clock()
    _served(_worker(pool_dir, ["cpu"], profile, store, sender(store, profile, workers), clock), clock)
    found = _observations(pool_dir, profile)
    assert {(o.batch, o.outcome, o.fault) for o in found if o.fault is not None} == {
        (8, "bisected", "resource"),
        (4, "bisected", "resource"),
    }
    assert all(o.fault is None for o in found if o.outcome == "ok")


@pytest.mark.logic
def test_a_request_names_the_one_class_all_its_items_failed_with(tmp_path: Path) -> None:
    """Nothing raises: two items out of memory say `resource` and split; one beside a content fault, none."""
    profile = _profile(quantum_items=2)
    with packaged(registry(TIMED, worker=CROWDED), tmp_path / "cache"):
        placed, store, pool_dir = _items(profile, tmp_path, ["oom", "oom", "oom", "fail"])
        admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
        for pair in ((0, 1), (2, 3)):
            fragment = node_fragment(placed, admission, elements=[(e,) for e in pair])
            paired = fragment.placement.model_copy(update={"batch_size": 2})
            QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu").put(
                fragment.model_copy(update={"placement": paired})
            )
        _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    found = _observations(pool_dir, profile)
    assert sorted((o.batch, o.outcome, str(o.fault)) for o in found) == [
        (1, "failed", "resource"),
        (1, "failed", "resource"),
        (2, "bisected", "resource"),
        (2, "failed", "None"),
    ]


@pytest.mark.logic
def test_a_request_names_no_class_when_its_items_failed_with_two(tmp_path: Path) -> None:
    """An item out of memory beside a content fault in one request of two: no class, and nothing is split."""
    profile = _profile(quantum_items=2)
    with packaged(registry(TIMED, worker=CROWDED), tmp_path / "cache"):
        placed, store, pool_dir = _items(profile, tmp_path, ["oom", "fail"])
        admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
        fragment = node_fragment(placed, admission, elements=[(e,) for e in range(2)])
        whole = fragment.placement.model_copy(update={"batch_size": 2})
        QueueSet(pool_dir, profile, lease=profile.lease).queue("cpu").put(
            fragment.model_copy(update={"placement": whole})
        )
        _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    found = _observations(pool_dir, profile)
    assert [(o.batch, o.outcome, str(o.fault)) for o in found] == [(2, "failed", "None")]


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
    mark_ready(tmp_path / "run", {"cpu": QueueReady(HARDWARE, {})})
    assert [p for p in (tmp_path / "run").rglob("*") if p.is_file()]
    assert identity_leaks(tmp_path / "run", tmp_path) == []


@pytest.mark.logic
def test_a_run_of_an_isolated_env_names_no_env_root_in_its_readable_files(tmp_path: Path) -> None:
    """The env root, a path the profile expands, stays with the worker: no queue item or outcome of the run holds it."""
    envs = tmp_path / "envs"
    profile = ExecutionProfile.load("laptop", {**OVERRIDE, "env_roots": {"isolated": str(envs), "server": str(envs)}})
    with packaged(registry(TIMED, worker=TIMER, env=Env.isolated("timed")), tmp_path / "cache"):
        placed, store, pool_dir = _items(profile, tmp_path, ["0.0"])
        _put(placed, pool_dir, 0)
        _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile)).serve()
    assert [o.outcome for o in _observations(pool_dir, profile)] == ["ok"]
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


def _unit_env(pool_dir: Path, profile: ExecutionProfile, store: ResultStore) -> tuple[str, dict[str, Any]]:
    """Return the backend and env entry a worker of this test's stack has: what a probe queue records for its unit."""
    worker = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    return f"{TIMED.name}~dsp", worker._env("cpu", TIMED.name, "dsp")  # noqa: SLF001 - the worker's own env read


def _probe_items(
    tmp_path: Path, count: int, obs: dict[str, Any]
) -> tuple[Any, ResultStore, Path, ExecutionProfile, Any]:
    profile = _profile(
        quantum_items=4, poll_s=obs["poll_s"], first_poll_s=obs["first_poll_s"], presence_refresh_s=obs["poll_s"]
    )
    placed, store, pool_dir = _items(profile, tmp_path, [f"0.00{i}" for i in range(count + 1)])
    queue = QueueSet(pool_dir, profile, lease=profile.lease).queue(probe_queue("cpu", "p1"))
    queue.set_probe_unit(HARDWARE, *_unit_env(pool_dir, profile, store))
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
def test_a_worker_lists_probe_queues_once_per_refresh(
    tmp_path: Path, obs: dict[str, Any], lease_clock: Callable[..., ManualClock]
) -> None:
    """A probe queue made after the worker listed is not served until the refresh interval has passed."""
    profile = _profile(poll_s=obs["poll_s"], presence_refresh_s=obs["presence_refresh_s"])
    placed, store, pool_dir = _items(profile, tmp_path, ["0.0", "0.00"])
    clock = lease_clock()
    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        device_memory=lambda: None,
        hardware=lambda _: HARDWARE,
        cotenancy=lambda _: SHARED,
        clock=clock,
    )
    assert worker._next(None, 4) is None  # noqa: SLF001 - the claim step, with nothing queued yet
    later = QueueSet(pool_dir, profile, lease=profile.lease).queue(probe_queue("cpu", "p1"))
    later.set_probe_unit(HARDWARE, *_unit_env(pool_dir, profile, store))
    later.put_probe(_one_job_items(placed, range(1))[0], "SINGLE", 0)
    assert worker._next(None, 4) is None  # noqa: SLF001 - listed within the refresh: not seen
    clock.advance(obs["presence_refresh_s"] + 1)
    claim = worker._next(None, 4)  # noqa: SLF001 - listed again: seen and taken
    assert claim is not None and claim.queue.name == probe_queue("cpu", "p1")
    claim.release()


@pytest.mark.logic
def test_a_pinned_worker_waits_on_its_probe_queue_from_the_first_poll(
    tmp_path: Path, obs: dict[str, Any], lease_clock: Callable[..., ManualClock]
) -> None:
    """With a long poll, an item put just after the last is claimed within the first poll, not after the long one."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 3, obs)
    later = _one_job_items(placed, range(3, 4))[0]
    profile = _profile(quantum_items=4, poll_s=obs["long_poll_s"], first_poll_s=obs["first_poll_s"])
    clock = lease_clock()
    worker = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        device_memory=lambda: None,
        hardware=lambda _: HARDWARE,
        cotenancy=lambda _: SHARED,
        clock=clock,
    )
    exits: list[int] = []
    thread = threading.Thread(target=lambda: exits.append(worker.serve()), daemon=True)
    thread.start()
    try:
        assert clock.parked(thread) and len(worker.served) == 3
        queue.put_probe(later, "SINGLE", 1)
        clock.advance(obs["first_poll_s"])
        assert clock.parked(thread) and len(worker.served) == 4
    finally:
        worker.halt()
        thread.join(clock.hang_s)
    assert exits == [0]


@pytest.mark.logic
def test_only_a_worker_of_the_recorded_class_takes_a_probe_queues_lease(tmp_path: Path, obs: dict[str, Any]) -> None:
    """A worker of another class than the probe queue's recorded one skips it, and so does any while none is recorded.

    The lease is never taken, so nothing runs the probe on the wrong card.
    """
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 2, obs)
    other = HardwareClass(device="cpu", name="another cpu", cpus=2)
    elsewhere = PoolWorker(
        pool_dir,
        ["cpu"],
        profile=profile,
        store=store,
        send=_sender(store, profile),
        device_memory=lambda: None,
        hardware=lambda _: other,
        cotenancy=lambda _: SHARED,
    )
    elsewhere.serve()
    assert elsewhere.served == []
    unrecorded = QueueSet(pool_dir, profile, lease=profile.lease).queue(probe_queue("cpu", "p2"))
    unrecorded.put_probe(_one_job_items(placed, range(2, 3))[0], "SINGLE", 0)
    assert queue.probe_hardware() == HARDWARE and unrecorded.probe_hardware() is None
    matching = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    matching.serve()
    assert {q for q, _ in matching.served} == {probe_queue("cpu", "p1")} and unrecorded.serve_lease() is not None


@pytest.mark.logic
def test_a_closed_probe_queue_is_served_by_no_one(tmp_path: Path, obs: dict[str, Any]) -> None:
    """Once its queue is closed a worker neither takes the serve lease nor claims, and one that held it lets it go."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 3, obs)
    worker = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    claim = worker._next(None, 4)  # noqa: SLF001 - the claim step
    assert claim is not None and claim.queue.name == queue.name
    claim.release()
    queue.close_probe()
    assert queue.probe_closed() and worker._next(None, 4) is None  # noqa: SLF001 - drops its lease, claims nothing
    assert queue.serve_lease() is not None
    other = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    other.serve()
    assert other.served == []


@pytest.mark.logic
def test_a_worker_whose_env_differs_from_the_units_takes_no_lease(tmp_path: Path, obs: dict[str, Any]) -> None:
    """The class is the unit's but the env's `wheel_set` is not: no lease, nothing claimed; the same env serves it."""
    placed, store, pool_dir, profile, queue = _probe_items(tmp_path, 2, obs)
    key, entry = _unit_env(pool_dir, profile, store)
    other = QueueSet(pool_dir, profile, lease=profile.lease).queue(probe_queue("cpu", "p2"))
    other.set_probe_unit(HARDWARE, key, {**entry, "wheel_set": "9" * 64})
    for item in _one_job_items(placed, range(2, 3)):
        other.put_probe(item, "SINGLE", 0)
    worker = _worker(pool_dir, ["cpu"], profile, store, _sender(store, profile))
    worker.serve()
    assert {q for q, _ in worker.served} == {queue.name} and other.serve_lease() is not None
