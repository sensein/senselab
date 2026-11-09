"""`ProbeMachine` on its own: a host of fake queues, the injected clock, the real resolver, roots and `Probe`."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pool_probe_test import HOST, JOBS, SEEN, WORKERS, Clock, _plan, _profile, _seen, stack  # noqa: F401 - the stack
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.calibration import Cotenancy, HardwareClass, Observation
from senselab_core.runtime.execution.executor import Admission, PlanFragment
from senselab_core.runtime.execution.in_process import fragment_jobs, fragment_keys, node_fragment
from senselab_core.runtime.execution.job import Job
from senselab_core.runtime.execution.observing import request_units
from senselab_core.runtime.execution.pool_parts import Followed, Held, Release
from senselab_core.runtime.execution.probe_machine import ProbeMachine
from senselab_core.runtime.execution.queue import item_digest
from senselab_core.runtime.execution.substrates import QueueReady, ReadyMark, WorkerHandle, backend_key
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.sizing import Grant
from senselab_core.runtime.units import UnitResolver, default_measures, mark_entry, measure_roots

pytestmark = pytest.mark.logic


class FakeQueue:
    """A probe queue in memory: what was put, and the outcomes and claims a test gives."""

    def __init__(self) -> None:
        """Start empty and open."""
        self.unit: tuple[HardwareClass, str, Mapping[str, object]] | None = None
        self.items: dict[str, PlanFragment] = {}
        self.outcomes: dict[str, dict[str, object]] = {}
        self.live: set[str] = set()
        self.closed = False

    def set_probe_unit(self, hardware: HardwareClass, backend: str, env: Mapping[str, object]) -> None:
        """Record the unit."""
        self.unit = (hardware, backend, env)

    def put_probe(self, fragment: PlanFragment, phase: str, stratum: int) -> None:  # noqa: ARG002
        """Keep the item."""
        self.items[item_digest(fragment)] = fragment

    def outcome(self, digest: str) -> dict[str, object] | None:
        """Return the outcome a test gave."""
        return self.outcomes.get(digest)

    def claimed(self, digest: str) -> bool:
        """Return whether a test holds a claim on `digest`."""
        return digest in self.live

    def close_probe(self) -> None:
        """Close."""
        self.closed = True


class FakeHost:
    """Queues in memory, marks and live workers a test sets, and the releases the machine makes."""

    def __init__(self, placed: Plan, values: Mapping[InputIdentity, object], run_name: str = "run") -> None:
        """Host the plan `placed`."""
        self.placed, self.values, self.run_name = placed, values, run_name
        self.queues: dict[str, FakeQueue] = {}
        self.new_marks: list[tuple[str, ReadyMark | None]] = []
        self.workers: list[WorkerHandle] = []
        self.released: list[Release] = []
        self.woken = 0

    def queue(self, name: str) -> FakeQueue:
        """Return the queue `name`."""
        return self.queues.setdefault(name, FakeQueue())

    def marks(self) -> list[tuple[str, ReadyMark | None]]:
        """Return the marks a test has made."""
        return list(self.new_marks)

    def live_workers(self) -> list[WorkerHandle]:
        """Return the workers a test says are alive."""
        return list(self.workers)

    def keys(self, part: PlanFragment) -> list[str]:
        """Return each job's key digest."""
        return fragment_keys(part, self.placed, self.values)

    def plan_of(self, part: PlanFragment) -> tuple[Plan, Mapping[InputIdentity, object]]:  # noqa: ARG002
        """Return the plan."""
        return self.placed, self.values

    def grant(self, queue: str) -> Grant:  # noqa: ARG002
        """Return a grant of two workers."""
        return Grant(max_workers=2, memory_bytes=2**33, quantum_items=10, quantum_s=60.0)

    def produce(self, release: Release, jobs: Sequence[Job], placement: Placement) -> None:  # noqa: ARG002
        """Keep the release."""
        self.released.append(release)

    def wake(self) -> None:
        """Count."""
        self.woken += 1


def _machine(tmp_path: Path, seen: bool, clock: Clock) -> tuple[ProbeMachine, FakeHost, Held, Any]:
    profile = _profile(tmp_path)
    if seen:
        _seen(profile, SEEN)
    placed, values = _plan(profile, range(1, JOBS + 1))
    roots = measure_roots(profile)
    host = FakeHost(placed, values)
    machine = ProbeMachine(
        profile, roots, default_measures(profile, roots), UnitResolver(profile, roots, hardware=HOST), clock, host
    )
    fragment = node_fragment(placed, Admission(id=f"run-{placed.hash}", plan_hash=placed.hash))
    return machine, host, Held(Followed([]), "cpu", fragment), profile


def _observations(host: FakeHost, queue: FakeQueue) -> None:
    """Give every item of `queue` that has no outcome the outcome a worker would: one request per job."""
    since = sum(len(o["observations"]) for o in queue.outcomes.values())  # type: ignore[arg-type, misc]
    for digest, item in queue.items.items():
        if digest in queue.outcomes:
            continue
        jobs, _ = fragment_jobs(item, host.placed, host.values)
        found = []
        for index, job in enumerate(jobs):
            kind, units, padded = request_units([job])
            found.append(
                Observation.model_validate(
                    {
                        "probe": {"id": "p", "phase": "SINGLE", "stratum": 0},
                        "item": digest,
                        "request_index": index,
                        "jobs": 1,
                        "batch": 1,
                        "unit_kind": kind,
                        "units": units,
                        "padded_units": padded,
                        "wall_s": 0.05 + 0.01 * units,
                        "item_wall_s": 0.1 if index == len(jobs) - 1 else None,
                        "since_load": since,
                        "load_s": 0.5 if since == 0 else None,
                        "device_peak": None,
                        "device_after": None,
                        "host_peak": None,
                        "host_after": None,
                        "cotenancy": Cotenancy(device=1, host=1),
                        "outcome": "ok",
                        "hardware": SEEN,
                    }
                )
            )
            since += 1
        queue.outcomes[digest] = {"outcome": "done", "observations": [o.model_dump(mode="json") for o in found]}


def test_a_unit_with_its_hardware_seen_probes_then_releases_its_held_jobs_sized(tmp_path: Path) -> None:
    """Place -> probing (items on a queue that records the unit) -> ticks -> measured and the rest released."""
    machine, host, held, _ = _machine(tmp_path, True, Clock())
    machine.place(held)
    [name] = host.queues
    queue = host.queues[name]
    assert queue.unit is not None and queue.unit[0] == SEEN and queue.items and not host.released
    assert machine.counts("cpu") == (1, False) and machine.probes
    _observations(host, queue)
    machine.advance()
    assert machine.counts("cpu") == (0, False) and [r.estimate is not None for r in host.released] == [True]
    rest = host.released[0].held.part
    probed = {j for i in queue.items.values() for j in i.jobs}
    assert not probed & set(rest.jobs) and len(probed) + len(rest.jobs) == JOBS


def test_a_unit_whose_hardware_is_unseen_waits_for_a_mark_then_probes(tmp_path: Path) -> None:
    """Without the class in the index the part is held and a worker wanted; a mark with its env resolves it."""
    machine, host, held, profile = _machine(tmp_path, False, Clock())
    machine.place(held)
    assert not host.queues and machine.counts("cpu") == (0, True) and held.followed.waiting
    _, placement = fragment_jobs(held.part, host.placed, host.values)
    from senselab_core.runtime.call import packaged  # noqa: PLC0415 - the stack's opener

    entry = mark_entry(profile, placement.backend, packaged().open_env(placement.backend, profile), HOST, "cpu")
    ready = QueueReady(SEEN, {backend_key(placement.backend): entry})
    host.new_marks = [("w1", ReadyMark("h", 1, None, {"cpu": ready}))]
    machine.advance()
    assert len(host.queues) == 1 and machine.counts("cpu") == (1, False)


def test_a_unit_resolving_past_resolve_s_is_released_unmeasured(tmp_path: Path) -> None:
    """From the queue's first live worker, `resolve_s` later the held part is released with no estimate."""
    clock = Clock()
    machine, host, held, profile = _machine(tmp_path, False, clock)
    machine.place(held)
    host.workers = [WorkerHandle("cpu", "w0")]
    machine.advance()
    clock.advance(profile.calibration.resolve_s - 1)
    machine.advance()
    assert not host.released
    clock.advance(2)
    machine.advance()
    assert [r.estimate for r in host.released] == [None] and machine.counts("cpu") == (0, False)


def test_a_probe_with_no_progress_for_the_stale_window_closes_its_queue_and_gives_its_jobs_back(tmp_path: Path) -> None:
    """Window from the last claim: a live claim renews it; past it the queue is closed and the jobs are released."""
    clock = Clock()
    machine, host, held, profile = _machine(tmp_path, True, clock)
    machine.place(held)
    [queue] = host.queues.values()
    window = profile.lease.stale_after_s
    digest = next(iter(queue.items))
    queue.live.add(digest)
    clock.advance(window - 1)
    machine.advance()
    clock.advance(window - 1)
    machine.advance()
    assert not queue.closed and not host.released
    queue.live.clear()
    clock.advance(window + 1)
    machine.advance()
    assert queue.closed and host.released and host.released[0].estimate is None
