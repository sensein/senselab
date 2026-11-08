"""`Probe.plan` and `Probe.estimate`: what counts, what is planned, and the figures the old `calibrated` got wrong."""

import itertools
import random
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from senselab_core.runtime.calibration import (
    ESTIMATE_REASONS,
    Candidate,
    Cotenancy,
    Estimate,
    Observation,
    OomRecord,
    Prior,
    Probe,
    ProbeItem,
    ProbePlan,
    ProbeRef,
    RegimeFactor,
    memory_batch,
)
from senselab_core.runtime.execution_settings import CalibrationSettings
from senselab_core.runtime.measures import MeasureSource
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Grant, Sample, calibrated, measure_of
from senselab_core.testing.calibration_samples import UNIT

CONFIG: CalibrationSettings = ExecutionProfile.load("laptop").calibration
GRANT = Grant(max_workers=1, memory_bytes=10**10, quantum_items=64, quantum_s=3600)
HEADROOM = 0.9
FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
_ids = itertools.count()


def _obs(units: float, wall: float, stratum: int = 0, batch: int = 1, **update: Any) -> Observation:  # noqa: ANN401
    fields: dict[str, Any] = {
        "probe": ProbeRef(id="p1", phase="SINGLE", stratum=stratum),
        "item": f"i{next(_ids)}",
        "request_index": 0,
        "jobs": batch,
        "batch": batch,
        "units": units * batch,
        "padded_units": units * batch,
        "wall_s": wall,
        "item_wall_s": None,
        "since_load": CONFIG.warmup_requests,
        "load_s": None,
        "device_peak": None,
        "device_after": None,
        "host_peak": None,
        "host_after": None,
        "cotenancy": Cotenancy(device=1, host=1),
        "outcome": "ok",
        "hardware": UNIT.hardware,
    }
    return Observation.model_validate({**fields, **update})


def _line(r: float, u: float, lengths: Sequence[float] = (2.0, 8.0), batch: int = 1) -> list[Observation]:
    """`min_requests` warm requests in each of two strata on the line `r + u·padded`."""
    return [
        _obs(x, r + u * x * batch, stratum=i, batch=batch)
        for i, x in enumerate(lengths)
        for _ in range(CONFIG.min_requests)
    ]


def _estimate(observations: Sequence[Observation], records: Sequence[Any] = ()) -> Estimate:
    return Probe.estimate(observations, records, CONFIG)


def _candidates(units: Sequence[float | None], keys: Sequence[int] | None = None) -> list[Candidate]:
    keys = keys if keys is not None else range(len(units))
    return [
        Candidate(fragment="f", index=i, key=f"k{k}", units=u) for i, (u, k) in enumerate(zip(units, keys, strict=True))
    ]


def _plan(
    candidates: Sequence[Candidate],
    prior: Prior = Prior(),
    config: CalibrationSettings = CONFIG,
    formation: str = "mixed",
) -> ProbePlan:  # noqa: B008
    return Probe.plan(UNIT, candidates, prior, GRANT, config, formation, 90.0)


def _with_device(m0: float, m1: float, resident: int = 10**9) -> list[Observation]:
    out = []
    for i, x in enumerate((2.0, 8.0)):
        for _ in range(CONFIG.min_requests):
            out.append(
                _obs(x, 0.1 + 0.01 * x, stratum=i, device_after=resident, device_peak=resident + int(m0 + m1 * x))
            )
    return out


# --- what counts -----------------------------------------------------------------------------------------------


@pytest.mark.logic
@given(
    wall=st.floats(0.0, 1e4),
    since=st.integers(0, CONFIG.warmup_requests - 1),
    batch=st.sampled_from([1, 4]),
)
@FAST
def test_cold_and_reloaded_requests_never_move_r_u_or_the_batch(wall: float, since: int, batch: int) -> None:
    """A request before warmup, or one that loaded, gives a load and nothing else."""
    base = _line(0.2, 0.05) + _line(0.3, 0.04, batch=4)
    extra = [
        _obs(5.0, wall, since_load=since, batch=batch),
        _obs(5.0, wall, since_load=0, load_s=wall, batch=batch),
    ]
    a, b = _estimate(base), _estimate(base + extra)
    assert (b.single, b.batched, b.batch, b.device) == (a.single, a.batched, a.batch, a.device)
    assert b.load_n == 1 and a.load_n == 0


@pytest.mark.logic
@given(wall=st.floats(0.0, 1e4), outcome=st.sampled_from(["failed", "bisected", "served"]))
@FAST
def test_a_failed_bisected_or_served_request_never_enters_any_figure(wall: float, outcome: str) -> None:
    """Such a request enters nothing, and no other request of a bisected item does."""
    base = _line(0.2, 0.05) + _line(0.3, 0.04, batch=4)
    bad = _obs(5.0, wall, outcome=outcome, device_after=1, device_peak=10, item_wall_s=wall)
    extra = [bad]
    if outcome == "bisected":
        extra.append(_obs(5.0, wall, item=bad.item, probe=bad.probe, since_load=5))
    a, b = _estimate(base), _estimate(base + extra)
    assert (b.single, b.batched, b.item_s, b.device, b.source) == (a.single, a.batched, a.item_s, a.device, a.source)


@pytest.mark.logic
@given(wall=st.floats(0.0, 1e4), device=st.integers(2, 4), host=st.integers(2, 4))
@FAST
def test_another_regime_never_enters_r_or_u(wall: float, device: int, host: int) -> None:
    """A request under other co-tenancy gives no fit; a regime record gives `regimes`."""
    base = _line(0.2, 0.05)
    other = [_obs(x, wall, stratum=0, cotenancy=Cotenancy(device=device, host=host)) for x in (2.0, 8.0)]
    a = _estimate(base)
    b = _estimate(base + other, [RegimeFactor(k=device, rho=1.5, n=6), RegimeFactor(k=device, rho=2.0, n=9)])
    assert b.single == a.single
    assert [(r.k, r.rho) for r in b.regimes] == [(device, 2.0)]


@pytest.mark.logic
@given(
    sizes=st.lists(st.sampled_from([2, 3, 5, 8, 16]), unique=True, min_size=0, max_size=3),
    r_b=st.floats(0.01, 1.0),
    u_b=st.floats(0.0, 0.05),
)
@FAST
def test_no_batch_above_every_measured_one_is_chosen(sizes: list[int], r_b: float, u_b: float) -> None:
    """`B*` is 1 or a measured batch; a batch never measured is never chosen."""
    obs = _line(0.5, 0.2)
    for b in sizes:
        obs += _line(r_b, u_b, batch=b)
    got = _estimate(obs)
    assert got.single is not None
    assert {s.batch for s in got.batch} <= {1, *sizes}


@pytest.mark.logic
def test_a_clearly_faster_batch_is_chosen_and_a_slower_one_is_not() -> None:
    """Per-job time 0.5 + 0.2·L at one against (0.5 + 0.2·B·L)/B at B: B = 8 wins by a wide margin."""
    obs = _line(0.5, 0.2) + _line(0.5, 0.2, batch=8)
    # per-job time at batch 8 is 0.0625 + 0.2·L against 0.5 + 0.2·L: faster, but only with spread known.
    spread = _estimate(obs)
    assert spread.single is not None and spread.single.spread is not None
    assert {s.batch for s in spread.batch} == {8}
    slow = _estimate(_line(0.5, 0.2) + _line(5.0, 0.2, batch=8))
    assert {s.batch for s in slow.batch} == {1}


@pytest.mark.logic
@given(
    r=st.floats(0.01, 2.0),
    u=st.floats(0.001, 0.5),
    outlier=st.floats(-1e3, 1e3),
    position=st.integers(0, 5),
)
@FAST
def test_theil_sen_tolerates_one_outlier_in_three(r: float, u: float, outlier: float, position: int) -> None:
    """One wild request among six (one in three of its stratum) leaves the fit at the exact line."""
    obs = _line(r, u)
    wild = obs[position]
    obs[position] = _obs(wild.units, max(0.0, wild.wall_s + outlier), stratum=wild.probe.stratum)  # type: ignore[union-attr]
    got = _estimate(obs).single
    assert got is not None and got.u is not None
    assert got.u == pytest.approx(u, rel=1e-6) and got.r == pytest.approx(r, rel=1e-6, abs=1e-9)


@pytest.mark.logic
@given(seed=st.integers(0, 10**6))
@FAST
def test_the_estimate_is_invariant_to_the_order_of_observations(seed: int) -> None:
    """Shuffling the observations (and their records) changes nothing."""
    obs = _line(0.2, 0.05) + _line(0.3, 0.04, batch=4) + _with_device(1e6, 1e8)
    a = _estimate(obs)
    shuffled = list(obs)
    random.Random(seed).shuffle(shuffled)
    b = _estimate(shuffled)
    assert a.model_dump() == b.model_dump()


@pytest.mark.logic
def test_every_reason_is_from_the_declared_vocabulary() -> None:
    """No estimate, however thin its evidence, carries a reason outside the vocabulary or one with an identifier."""
    for obs in ([], _line(0.2, 0.05)[:2], _line(0.2, 0.05), _line(0.2, 0.05) + _line(0.3, 0.04, batch=4)[:1]):
        got = _estimate(obs)
        assert set(got.reasons) <= ESTIMATE_REASONS
    assert _estimate([]).status == "unmeasured"


# --- the figures the old path got wrong -------------------------------------------------------------------------


def _sample(wall: float, jobs: int = 1, cold: bool = False) -> Sample:
    return Sample(
        wall_s=wall, claim_s=0.01, jobs=jobs, batch_size=1, input_s=None, baseline_bytes=1, resident_bytes=1000,
        peak_memory_bytes=1100, cold=cold, job="j", ended_s=0,
    )  # fmt: skip


@pytest.mark.logic
def test_a_failed_request_never_enters_the_per_item_time() -> None:
    """`calibrated` takes a failed 9 s request into its median (2 s per item); the estimate keeps 0.1 + 0.02·L."""
    good = _line(0.1, 0.02)
    failed = [_obs(5.0, 9.0, outcome="failed") for _ in range(4)]
    samples = [_sample(o.wall_s) for o in good[:3]] + [_sample(9.0) for _ in range(4)]
    assert calibrated(samples) is not None
    assert calibrated(samples).s_per_input_s > 1.0  # type: ignore[union-attr]
    got = _estimate(good + failed).single
    assert got is not None and got.u == pytest.approx(0.02) and got.r == pytest.approx(0.1)


@pytest.mark.logic
def test_fixed_cost_is_fitted_not_shared() -> None:
    """With item wall 0.040 + 0.0366 per job the estimate gives `u` 0.0366 and `a` 0.040; `calibrated` gives 0.075."""
    fixed, per_job = 0.040, 0.0366
    obs = []
    for stratum, length in enumerate((1.0, 4.0)):
        for _ in range(CONFIG.min_requests):
            obs.append(_obs(length, per_job * length, stratum=stratum, item_wall_s=fixed + per_job * length))
    got = _estimate(obs)
    assert got.single is not None and got.single.u == pytest.approx(per_job)
    assert got.item_s == pytest.approx(fixed)
    shared = calibrated([_sample(fixed + per_job * 1.0, jobs=1), _sample(fixed + per_job * 1.0, jobs=1)] * 2)
    assert shared is not None and shared.s_per_input_s == pytest.approx(0.0766, abs=0.002)
    assert got.single.u is not None and shared.s_per_input_s - got.single.u > 0.039


@pytest.mark.logic
def test_device_bytes_scale_with_length() -> None:
    """Peak beyond resident is 5 000 + 50 per unit: the estimate gives `m1` 50; `measure_of` gives one constant."""
    got = _estimate(_with_device(5000, 50))
    assert got.device is not None and got.device.m1 == pytest.approx(50.0) and got.device.m0 == pytest.approx(5000.0)
    assert got.device.resident == 10**9
    short = _sample(1.0).model_copy(update={"device_item_bytes": 5100, "device_allocated_bytes": 10**9})
    longer = _sample(1.0).model_copy(update={"device_item_bytes": 5400, "device_allocated_bytes": 10**9})
    measure = measure_of(calibrated([short, longer]), on_device=True)  # type: ignore[arg-type]
    assert measure.bytes_per_item == 5400  # the one constant, at every length
    assert got.device.m1 * 2.0 + got.device.m0 != got.device.m1 * 8.0 + got.device.m0


# --- the OOM raise ----------------------------------------------------------------------------------------------


@pytest.mark.logic
def test_an_oom_raises_m1_without_a_constant() -> None:
    """`m1′ = (budget − resident − m0)/((B − 1)·L)` for B = 8, L = 40: the faulting batch becomes infeasible."""
    base = _estimate(_with_device(5000, 50))
    budget, batch, length = 2 * 10**9, 8, 40.0
    record = OomRecord(resource="device", batch=batch, length=length, cotenancy=1, budget_bytes=budget)
    raised = _estimate(_with_device(5000, 50), [record])
    assert base.device is not None and raised.device is not None
    expected = (budget - base.device.resident - base.device.m0) / ((batch - 1) * length)
    assert raised.device.m1 == pytest.approx(expected)
    assert raised.device.m1 > base.device.m1
    assert raised.oom == (record,)


@pytest.mark.logic
def test_an_oom_under_shared_device_or_at_batch_one_does_not_raise() -> None:
    """A record of another co-tenancy applies to that regime and above; a batch of one has none to shrink."""
    base = _estimate(_with_device(5000, 50))
    shared = OomRecord(resource="device", batch=8, length=40.0, cotenancy=2, budget_bytes=2 * 10**9)
    single = OomRecord(resource="device", batch=1, length=40.0, cotenancy=1, budget_bytes=2 * 10**9)
    got = _estimate(_with_device(5000, 50), [shared, single])
    assert got.device == base.device and got.oom == (shared, single)


@pytest.mark.logic
@given(
    m1=st.floats(1.0, 1e6),
    batch=st.integers(2, 64),
    length=st.floats(0.5, 200.0),
    above=st.floats(1.0, 4.0),
    budget=st.integers(10**9, 10**11),
)
@FAST
def test_after_an_oom_the_memory_batch_never_reaches_the_faulting_one(
    m1: float, batch: int, length: float, above: float, budget: int
) -> None:
    """At any length at or above the faulting one, the memory batch is below the faulting batch."""
    obs = _with_device(5000, m1)
    record = OomRecord(resource="device", batch=batch, length=length, cotenancy=1, budget_bytes=budget)
    fit = _estimate(obs, [record]).device
    assert fit is not None
    assert memory_batch(fit, budget, length * above, ceiling=10**6) < batch


# --- plan -------------------------------------------------------------------------------------------------------


_UNITS = st.lists(st.floats(0.5, 60.0), min_size=0, max_size=60)


@pytest.mark.logic
@given(units=_UNITS, keys=st.data(), probe_items=st.integers(1, 80))
@FAST
def test_plan_never_chooses_two_jobs_of_one_key_and_stays_within_probe_items(
    units: list[float], keys: st.DataObject, probe_items: int
) -> None:
    """A key is planned at most once, a job is planned once, and the jobs planned stay within `probe_items`."""
    ks = keys.draw(st.lists(st.integers(0, max(len(units) // 2, 1)), min_size=len(units), max_size=len(units)))
    config = CONFIG.model_copy(update={"probe_items": probe_items})
    plan = _plan(_candidates(units, ks), config=config)
    jobs = [j for item in plan.items for j in item.jobs]
    assert len({j.key for j in jobs}) == len(jobs) == len({(j.fragment, j.index) for j in jobs})
    assert plan.used_jobs == len(jobs) <= probe_items
    left = {j.key for pool in plan.remaining for j in pool}
    assert left.isdisjoint({j.key for j in jobs})


@pytest.mark.logic
def test_the_plan_cuts_strata_of_equal_count_and_takes_the_shortest() -> None:
    """Forty lengths, two strata: warmups and singles come from the shortest of each stratum."""
    plan = _plan(_candidates([float(i + 1) for i in range(40)]))
    assert [(s.lo, s.hi) for s in plan.strata] == [(1.0, 20.0), (21.0, 40.0)]
    warm = [i for i in plan.items if i.phase == "WARMUP"]
    assert len(warm) == CONFIG.warmup_requests and all(i.stratum == 0 for i in warm)
    singles = [i for i in plan.items if i.phase == "SINGLE"]
    assert sorted({i.stratum for i in singles}) == [0, 1]
    assert [j.units for i in singles if i.stratum == 1 and len(i.jobs) == 1 for j in i.jobs] == [21.0, 22.0, 23.0]
    many = [i for i in singles if len(i.jobs) == CONFIG.single_item_jobs]
    assert len(many) == CONFIG.min_requests and all(
        i.stratum == len(plan.strata) // 2 and i.request_jobs == 1 for i in many
    )


def _covering(lengths: Sequence[float], batch: int = 1) -> list[Observation]:
    out = []
    for stratum, x in enumerate(lengths):
        out += [
            _obs(x, 0.2 + 0.05 * x * batch, stratum=stratum, batch=batch, item_wall_s=1.0)
            for _ in range(CONFIG.min_requests)
        ]
    return out


@pytest.mark.logic
@given(extra=st.integers(0, 5))
@FAST
def test_covered_strata_are_not_replanned_until_an_oom_record_reopens_them(extra: int) -> None:
    """Strata prior observations cover plan nothing; an `oom` record at batch 2 or more re-opens its stratum's batch."""
    candidates = _candidates([float(i + 1) for i in range(20 + extra)])
    prior = Prior(observations=tuple(_covering([5.0, 15.0]) + _covering([5.0, 15.0], batch=4)))
    plan = _plan(candidates, prior)
    assert plan.items == () and plan.batched == ()
    record = OomRecord(resource="device", batch=4, length=15.0, cotenancy=1, budget_bytes=10**9)
    reopened = _plan(candidates, Prior(observations=prior.observations, records=(record,)))
    assert reopened.items == () and reopened.batched == (1,)
    assert (
        _plan(
            candidates, Prior(observations=prior.observations, records=(record.model_copy(update={"batch": 1}),))
        ).batched
        == ()
    )


@pytest.mark.logic
def test_a_partly_covered_plan_plans_only_what_is_missing() -> None:
    """Stratum 0 covered, stratum 1 not: only stratum 1's singles are planned, behind the warmups."""
    plan = _plan(_candidates([float(i + 1) for i in range(20)]), Prior(observations=tuple(_covering([5.0]))))
    assert {i.stratum for i in plan.items if i.phase == "SINGLE"} == {1}
    assert any(u.reason == "covered" and u.stratum == 0 for u in plan.unplanned)


@pytest.mark.logic
def test_too_few_candidates_leave_a_phase_unplanned() -> None:
    """With fewer candidates than a phase needs the phase is not planned and the plan says why."""
    plan = _plan(_candidates([1.0, 2.0]))
    assert plan.items == ()
    assert {u.reason for u in plan.unplanned} <= {"too_few_candidates"} and plan.unplanned


@pytest.mark.logic
def test_lengthless_jobs_form_one_stratum_and_a_single_formation_plans_no_batch() -> None:
    """Jobs with no length are one stratum; a `single` formation names no batched stratum."""
    plan = _plan(_candidates([None] * 20), formation="single")
    assert len(plan.strata) == 1 and plan.strata[0].lo is None and plan.batched == ()
    assert (
        Probe.plan_batched(plan, _measured_single(), GRANT, CONFIG, 90.0, memory_headroom=HEADROOM).unplanned[-1].reason
        == "single_formation"
    )


def _measured_single(m1: float = 1e6) -> Estimate:
    return _estimate(_line(0.2, 0.05) + _with_device_and_time(m1))


def _with_device_and_time(m1: float) -> list[Observation]:
    out = []
    for i, x in enumerate((2.0, 8.0)):
        out += [
            _obs(
                x,
                0.2 + 0.05 * x,
                stratum=i,
                item_wall_s=0.2 + 0.05 * x + 0.04,
                device_after=10**9,
                device_peak=10**9 + int(5000 + m1 * x),
            )
            for _ in range(CONFIG.min_requests)
        ]
    return out


@pytest.mark.logic
@given(
    seconds_left=st.floats(0.0, 500.0),
    probe_items=st.integers(20, 400),
    m1=st.floats(1e3, 1e9),
    sizes=st.sampled_from([1, 2]),
)
@FAST
def test_batched_items_fit_the_seconds_left_at_the_upper_prediction(
    seconds_left: float, probe_items: int, m1: float, sizes: int
) -> None:
    """Whatever `B₁` is planned, its items' upper-predicted time fits `seconds_left` and its jobs fit `probe_items`."""
    config = CONFIG.model_copy(update={"probe_items": probe_items, "batched_sizes": sizes})
    plan = _plan(_candidates([float(i + 1) for i in range(400)]), config=config)
    single = _estimate(_line(0.2, 0.05) + _with_device_and_time(m1))
    assert single.single is not None and single.device is not None
    batched = Probe.plan_batched(plan, single, GRANT, config, seconds_left, memory_headroom=HEADROOM)
    assert batched.used_jobs <= probe_items
    fit = single.single
    wall = 0.0
    for item in batched.items:
        assert item.phase == "BATCHED" and item.request_jobs == len(item.jobs) >= 2
        top = batched.strata[item.stratum].hi
        wall += len(item.jobs) * (fit.r + (fit.u or 0.0) * (top or 0.0)) * (1 + config.confidence * (fit.spread or 0.0))
        wall += single.item_s or 0.0
        budget = HEADROOM * (GRANT.memory_bytes or 0)
        assert len(item.jobs) <= memory_batch(single.device, budget, top or 0.0, GRANT.quantum_items)
    assert wall <= seconds_left + 1e-9
    keys = [j.key for i in batched.items for j in i.jobs]
    assert len(keys) == len(set(keys))


@pytest.mark.logic
def test_batched_items_are_planned_when_budget_memory_and_seconds_allow() -> None:
    """Ample seconds and memory: both strata get `min_requests` items of one `B₁`-job request."""
    plan = _plan(_candidates([float(i + 1) for i in range(400)]))
    batched = Probe.plan_batched(plan, _measured_single(1e3), GRANT, CONFIG, 1e6, memory_headroom=HEADROOM)
    assert [i.stratum for i in batched.items] == [0] * CONFIG.min_requests + [1] * CONFIG.min_requests
    assert all(2 <= len(i.jobs) <= GRANT.quantum_items for i in batched.items)
    assert len({len(i.jobs) for i in batched.items if i.stratum == 1}) == 1


@pytest.mark.logic
def test_a_second_batch_size_is_the_integer_nearest_the_root_of_the_first() -> None:
    """`batched_sizes` 2 adds `min_requests` items of the integer nearest √B₁ jobs, when it differs from B₁."""
    config = CONFIG.model_copy(update={"batched_sizes": 2})
    plan = _plan(_candidates([float(i + 1) for i in range(400)]), config=config)
    batched = Probe.plan_batched(plan, _measured_single(1e3), GRANT, config, 1e6, memory_headroom=HEADROOM)
    widths = sorted({len(i.jobs) for i in batched.items if i.stratum == 0})
    assert len(widths) == 2 and widths[0] == int(widths[1] ** 0.5 + 0.5)


@pytest.mark.logic
def test_batched_phase_names_what_blocks_it() -> None:
    """Unmeasured single time, unmeasured device memory and an unknown grant each block, with a reason."""
    plan = _plan(_candidates([float(i + 1) for i in range(40)]))
    none = _estimate([])
    assert (
        Probe.plan_batched(plan, none, GRANT, CONFIG, 90.0, memory_headroom=HEADROOM).unplanned[-1].reason
        == "single_unmeasured"
    )
    nodev = _estimate(_line(0.2, 0.05))
    assert (
        Probe.plan_batched(plan, nodev, GRANT, CONFIG, 90.0, memory_headroom=HEADROOM).unplanned[-1].reason
        == "device_unmeasured"
    )
    nogrant = GRANT.model_copy(update={"memory_bytes": None})
    assert Probe.plan_batched(plan, _measured_single(), nogrant, CONFIG, 90.0, memory_headroom=HEADROOM).items == ()


@pytest.mark.logic
def test_replace_served_swaps_the_item_for_the_next_candidates_of_its_stratum() -> None:
    """A served single item is replaced by the next shortest remaining candidate; an unknown item is refused."""
    plan = _plan(_candidates([float(i + 1) for i in range(20)]))
    served = next(i for i in plan.items if i.phase == "SINGLE" and i.stratum == 1 and len(i.jobs) == 1)
    new = Probe.replace_served(plan, served)
    swapped = new.items[plan.items.index(served)]
    assert swapped.phase == served.phase and swapped.stratum == served.stratum
    assert swapped.jobs[0].key not in {j.key for i in plan.items for j in i.jobs}
    assert len(new.items) == len(plan.items) and new.used_jobs == plan.used_jobs
    with pytest.raises(ValueError, match="not an item"):
        Probe.replace_served(plan, ProbeItem(phase="SINGLE", stratum=0, request_jobs=1, jobs=(_candidates([99.0])[0],)))


@pytest.mark.logic
def test_replace_served_drops_the_item_when_its_stratum_has_no_candidates_left() -> None:
    """With nothing left to take, the item is dropped and the plan records it."""
    plan = _plan(_candidates([float(i + 1) for i in range(14)]))
    item = plan.items[-1]
    drained = plan.model_copy(update={"remaining": tuple(() for _ in plan.remaining)})
    new = Probe.replace_served(drained, item)
    assert item not in new.items and new.unplanned[-1].reason == "too_few_candidates"


# --- the MeasureSource seam -------------------------------------------------------------------------------------


class _Root:
    def __init__(self, observations: Sequence[Observation]) -> None:
        self._o = observations

    def observations(self, unit: Any) -> Sequence[Observation]:  # noqa: ANN401
        return self._o


class _NoTable:
    def get(self, unit: Any) -> None:  # noqa: ANN401
        return None


@pytest.mark.contract
def test_estimator_is_the_callable_measure_source_takes() -> None:
    """`Probe.estimator` has `MeasureSource`'s callable shape and filters observations by the unit's hardware."""
    other = UNIT.hardware.model_copy(update={"cpus": 99})
    pooled = _line(0.2, 0.05) + [_obs(5.0, 500.0, hardware=other)]
    source = MeasureSource([_Root(pooled)], _NoTable(), estimate=Probe.estimator(CONFIG, unit_kind="audio_s"))
    got = source.get(UNIT)
    assert got is not None and got.unit_kind == "audio_s" and got.estimator == "estimate@1"
    assert got.single is not None and got.single.u == pytest.approx(0.05)
