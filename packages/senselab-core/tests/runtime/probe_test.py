"""`Probe.plan` and `Probe.estimate`: what counts, what is planned."""

import itertools
import math
import random
import statistics
import sys
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from senselab_core.kernel.formats import ESTIMATE_FORMAT
from senselab_core.runtime.calibration import (
    Candidate,
    Cotenancy,
    Estimate,
    Observation,
    OomRecord,
    Prior,
    ProbeItem,
    ProbePlan,
    ProbeRef,
    RegimeFactor,
    Unplanned,
)
from senselab_core.runtime.execution_settings import CalibrationSettings
from senselab_core.runtime.measures import MeasureSource
from senselab_core.runtime.probe import Probe, Work, _rank_bound, _theil_sen, memory_batch
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Grant
from senselab_core.testing.calibration_samples import UNIT

CONFIG: CalibrationSettings = ExecutionProfile.load("laptop").calibration
GRANT = Grant(max_workers=1, memory_bytes=10**10, quantum_items=64, quantum_s=3600)
PROFILE = ExecutionProfile.load("laptop")
HEADROOM = PROFILE.sizing.memory_headroom
PROBE_S = PROFILE.probe_s
FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
_ids = itertools.count()


def _obs(units: float, wall: float, stratum: int = 0, batch: int = 1, **update: Any) -> Observation:  # noqa: ANN401
    fields: dict[str, Any] = {
        "probe": ProbeRef(id="p1", phase="SINGLE", stratum=stratum),
        "item": f"i{next(_ids)}",
        "request_index": 0,
        "jobs": batch,
        "batch": batch,
        "unit_kind": "audio_s",
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
    return Probe.estimate(observations, records, CONFIG, hardware=UNIT.hardware, memory_headroom=HEADROOM)


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
    edges: Sequence[float] = (),
) -> ProbePlan:  # noqa: B008
    return Probe.plan(UNIT, candidates, prior, config, formation, PROBE_S, bucket_edges=edges)


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
    base = _line(0.2, 0.05) + _line(0.3, 0.04, batch=4) + _with_device(5000, 50)
    wild = {"device_after": 1, "device_peak": 10**12}
    extra = [
        _obs(5.0, wall, since_load=since, batch=batch, **wild),
        _obs(5.0, wall, since_load=0, load_s=wall, batch=batch, **wild),
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
def test_reasons_are_a_declared_type() -> None:
    """An estimate refuses a reason outside the declared type; thin evidence gives only declared ones."""
    with pytest.raises(ValidationError):
        Estimate(status="unmeasured", unit_kind="none", reasons=("lazy",))
    for obs in ([], _line(0.2, 0.05)[:2], _line(0.2, 0.05), _line(0.2, 0.05) + _line(0.3, 0.04, batch=4)[:1]):
        got = Estimate.model_validate(_estimate(obs).model_dump())
        assert got == _estimate(obs)
    assert _estimate([]).status == "unmeasured"


@pytest.mark.contract
def test_the_estimator_tag_is_registered() -> None:
    """`estimate@1` is a registered format; an estimate names it and refuses another."""
    assert ESTIMATE_FORMAT == "estimate@1" == _estimate(_line(0.2, 0.05)).estimator
    with pytest.raises(ValidationError):
        Estimate(status="unmeasured", unit_kind="none", reasons=("no_usable_observations",), estimator="estimate@2")


@pytest.mark.contract
def test_a_regime_factor_exists_only_for_shared_co_tenancy() -> None:
    """`k` 1 is the probe regime, which only a confirming probe may change."""
    with pytest.raises(ValidationError):
        RegimeFactor(k=1, rho=3.0, n=6)
    assert RegimeFactor(k=2, rho=3.0, n=6).k == 2


@pytest.mark.logic
def test_hardware_and_headroom_are_required() -> None:
    """No caller inherits a guessed hardware or headroom."""
    for missing in ("hardware", "memory_headroom"):
        given = {"hardware": UNIT.hardware, "memory_headroom": HEADROOM}
        del given[missing]
        with pytest.raises(TypeError):
            Probe.estimate([], [], CONFIG, **given)  # type: ignore[arg-type]


@pytest.mark.logic
def test_a_peer_request_at_another_host_load_does_not_change_a_probes_regime() -> None:
    """A probe at host co-tenancy 4 stays measured when one peer request at host co-tenancy 1 joins the pool."""
    shared = Cotenancy(device=1, host=4)
    probe = [o.model_copy(update={"cotenancy": shared}) for o in _line(0.2, 0.05)]
    peer = _obs(5.0, 0.45, probe=ProbeRef(id="peer", phase="SINGLE", stratum=0), cotenancy=Cotenancy(device=1, host=1))
    alone, pooled = _estimate(probe), _estimate([*probe, peer])
    assert alone.status == "measured" and pooled.status == "measured"
    assert pooled.single is not None and alone.single is not None
    assert (pooled.single.r, pooled.single.u) == pytest.approx((alone.single.r, alone.single.u))


@pytest.mark.logic
def test_a_single_stratum_median_fit_carries_its_spread_so_a_faster_batch_is_chosen() -> None:
    """One stratum, batch 8 eight times faster per job: the median fits have spreads and the batch is chosen."""
    jitter = [0.98, 1.0, 1.02]
    one = [_obs(5.0, 1.0 * j) for j in jitter]
    eight = [_obs(5.0, 1.0 * j, batch=8) for j in jitter]
    got = _estimate(one + eight)
    assert got.single is not None and got.single.u is None and got.single.spread is not None
    assert [b.batch for b in got.batch] == [8]


# --- OOM records lift by probe id --------------------------------------------------------------------------------


def _fault(basis: tuple[str, ...] = ("p1",)) -> OomRecord:
    return OomRecord(resource="device", batch=8, length=5.0, cotenancy=1, grant_bytes=2 * 10**9, basis=basis)


def _probe_of(name: str, batch: int = 4, count: int | None = None) -> list[Observation]:
    ref = ProbeRef(id=name, phase="BATCHED", stratum=0)
    return [_obs(5.0, 0.5, batch=batch, probe=ref) for _ in range(count or CONFIG.min_requests)]


@pytest.mark.logic
def test_an_oom_record_is_lifted_by_a_later_probe_not_in_its_basis() -> None:
    """A probe outside the basis with `min_requests` batched requests in the stratum lifts the record; others do not."""
    base = _with_device(5000, 50)
    raised = _estimate(base, [_fault()])
    plain = _estimate(base)
    assert raised.device is not None and plain.device is not None and raised.device.m1 > plain.device.m1
    own = _estimate(base + _probe_of("p1"), [_fault()])
    assert own.device is not None and own.device.m1 == raised.device.m1
    thin = _estimate(base + _probe_of("p2", count=CONFIG.min_requests - 1), [_fault()])
    assert thin.device is not None and thin.device.m1 == raised.device.m1
    later = _estimate(base + _probe_of("p2"), [_fault()])
    assert later.device == plain.device and later.oom == ()


@pytest.mark.logic
def test_plan_reopens_only_unlifted_records_by_the_same_rule() -> None:
    """The record that raises `m1` is the record that re-opens its stratum, and a lifted one does neither."""
    candidates = _candidates([float(i + 1) for i in range(40)])
    obs = _covering([5.0, 30.0]) + _covering([5.0, 30.0], batch=4)
    held = _plan(candidates, Prior(observations=tuple(obs), records=(_fault(("p1",)),)))
    assert held.batched == (0,) and held.items and all(i.phase == "WARMUP" for i in held.items)
    lifted = _plan(candidates, Prior(observations=tuple(obs + _probe_of("p2")), records=(_fault(("p1",)),)))
    assert lifted.batched == () and lifted.items == ()


@pytest.mark.logic
def test_the_oom_raise_follows_the_headroom_given() -> None:
    """The record holds the grant; the headroom in force is the caller's, not frozen into the record."""
    record = _fault()
    low = Probe.estimate(_with_device(5000, 50), [record], CONFIG, hardware=UNIT.hardware, memory_headroom=0.5)
    high = Probe.estimate(_with_device(5000, 50), [record], CONFIG, hardware=UNIT.hardware, memory_headroom=1.0)
    assert low.device is not None and high.device is not None and high.device.m1 > low.device.m1


@pytest.mark.logic
def test_a_probes_own_batched_successes_do_not_lift_its_own_fault() -> None:
    """A probe whose warmups never entered the sizing estimate is in its fault's basis, so its B₂ cannot lift its B₁."""
    warmups = [
        _obs(5.0, 0.5, since_load=i, probe=ProbeRef(id="p9", phase="WARMUP", stratum=0))
        for i in range(CONFIG.warmup_requests)
    ]
    sizing = _estimate(_with_device(5000, 50) + warmups)
    assert sizing.source.probe_ids == ("p1",) and sizing.fault_basis("p9") == ("p1", "p9")
    fault = _fault(sizing.fault_basis("p9"))
    own = _estimate(_with_device(5000, 50) + warmups + _probe_of("p9", batch=3), [fault])
    raised = _estimate(_with_device(5000, 50), [fault])
    assert own.device is not None and raised.device is not None and own.device.m1 == raised.device.m1
    assert own.oom == (fault,) and sizing.fault_basis(None) == ("p1",)


@pytest.mark.logic
def test_a_shared_co_tenancy_record_stays_in_force_and_reopens_nothing() -> None:
    """A probe runs alone: a co-tenancy 2 fault neither re-opens a stratum nor raises its fit, and stays in force."""
    candidates = _candidates([float(i + 1) for i in range(40)])
    obs = _covering([5.0, 30.0]) + _covering([5.0, 30.0], batch=4)
    shared = _fault().model_copy(update={"cotenancy": 2})
    plan = _plan(candidates, Prior(observations=tuple(obs), records=(shared,)))
    assert plan.batched == () and plan.items == ()
    got, plain = _estimate(obs + _with_device(5000, 50), [shared]), _estimate(obs + _with_device(5000, 50))
    assert got.device is not None and got.device == plain.device and got.oom == (shared,)
    lifted = _estimate(obs + _probe_of("p2"), [shared])
    assert lifted.oom == (shared,)


@pytest.mark.logic
def test_strata_are_length_ranges_whatever_the_probes_label_them() -> None:
    """Two probes at lengths 1-1.2 and 40-42 give one estimate whether both say stratum 0 or each its own."""

    def probe(name: str, lengths: Sequence[float], stratum: int) -> list[Observation]:
        out = []
        for x in lengths:
            for batch, r in ((1, 0.2), (8, 0.3)):
                ref = ProbeRef(id=name, phase="SINGLE" if batch == 1 else "BATCHED", stratum=stratum)
                out.append(_obs(x, r + 0.05 * x * batch, batch=batch, probe=ref))
        return out

    low = probe("pA", (1.0, 1.1, 1.2), 0)
    same, apart = low + probe("pB", (40.0, 41.0, 42.0), 0), low + probe("pB", (40.0, 41.0, 42.0), 1)
    a, b = _estimate(same), _estimate(apart)
    assert a.model_dump() == b.model_dump()
    assert a.single is not None and a.single.u == pytest.approx(0.05)
    assert [(s.lo, s.hi) for s in a.batch] == [(1.0, 1.2), (40.0, 42.0)]


# --- the strata ------------------------------------------------------------------------------------------------


@pytest.mark.logic
def test_lengthless_jobs_are_one_extra_stratum_beside_the_sized_ones() -> None:
    """One job with no length leaves the sized jobs in their two strata and gets a stratum of its own."""
    plan = _plan(_candidates([*(float(i + 1) for i in range(40)), None]))
    assert [(s.lo, s.hi) for s in plan.strata] == [(1.0, 20.0), (21.0, 40.0), (None, None)]


@pytest.mark.logic
def test_bucket_edges_never_move_a_stratum_cut() -> None:
    """Edge 39.5 over lengths 1..40 leaves the equal-count strata, and both strata get their singles."""
    plan = _plan(_candidates([float(i + 1) for i in range(40)]), formation="bucket:g", edges=(39.5,))
    assert [(s.lo, s.hi) for s in plan.strata] == [(1.0, 20.0), (21.0, 40.0)]
    assert {i.stratum for i in plan.items if i.phase == "SINGLE"} == {0, 1}
    with pytest.raises(ValueError, match="bucket edges"):
        _plan(_candidates([1.0, 2.0]), formation="bucket:g")


@pytest.mark.logic
def test_narrow_buckets_still_plan_a_batched_phase_one_bucket_per_item() -> None:
    """Width-2 buckets over 400 lengths: each item is two jobs of one bucket, drawn from many buckets."""
    edges = tuple(float(e) for e in range(1, 402, 2))
    plan = _plan(_candidates([float(i + 1) for i in range(400)]), formation="bucket:g", edges=edges)
    batched = Probe.plan_batched(plan, _measured_single(1e3), GRANT, CONFIG, 1e6, memory_headroom=HEADROOM)
    assert {i.stratum for i in batched.items} == {0, 1}
    for item in batched.items:
        assert len(item.jobs) == 2 and len({bisect_right(edges, j.units or 0.0) for j in item.jobs}) == 1


@pytest.mark.logic
def test_a_batched_item_never_spans_two_buckets() -> None:
    """With edges (15, 30) every batched item and every replacement draws its jobs from one bucket."""
    edges = (15.0, 30.0)
    plan = _plan(_candidates([float(i + 1) for i in range(400)]), formation="bucket:g", edges=edges)
    batched = Probe.plan_batched(plan, _measured_single(1e3), GRANT, CONFIG, 1e6, memory_headroom=HEADROOM)
    assert batched.items
    for item in batched.items:
        assert len({bisect_right(edges, j.units or 0.0) for j in item.jobs}) == 1
    swapped = Probe.replace_served(batched, batched.items[0]).items[0]
    assert len({bisect_right(edges, j.units or 0.0) for j in swapped.jobs}) == 1


@pytest.mark.logic
def test_a_batched_phase_names_a_memory_batch_below_two_and_an_unbounded_grant() -> None:
    """A memory batch of one everywhere, and a grant with no memory, each have their own reason."""
    plan = _plan(_candidates([float(i + 1) for i in range(40)]))
    tight = Probe.plan_batched(plan, _measured_single(1e12), GRANT, CONFIG, PROBE_S, memory_headroom=HEADROOM)
    assert tight.unplanned[-1].reason == "memory_below_two"
    open_ = GRANT.model_copy(update={"memory_bytes": None})
    unbounded = Probe.plan_batched(plan, _measured_single(), open_, CONFIG, PROBE_S, memory_headroom=HEADROOM)
    assert unbounded.unplanned[-1].reason == "grant_unbounded"


# --- records, kind and hardware are resolved for the unit asked about ------------------------------------------


@pytest.mark.contract
def test_the_estimator_resolves_records_per_unit() -> None:
    """Another unit's `oom` record never raises this unit's `m1`."""
    other = UNIT.model_copy(update={"backend": "other"})
    records = {UNIT.digest(): [_fault(())], other.digest(): []}
    estimate = Probe.estimator(CONFIG, records_of=lambda u: records[u.digest()], memory_headroom=HEADROOM)
    obs = _with_device(5000, 50)
    raised, plain = estimate(UNIT, obs), estimate(other, obs)
    assert raised.device is not None and plain.device is not None and raised.device.m1 > plain.device.m1


@pytest.mark.logic
def test_the_unit_kind_is_its_observations_and_mixed_kinds_leave_the_slope_unmeasured() -> None:
    """Observations of one kind give that kind; audio seconds beside characters give `none`, `r` and no `u`."""
    chars = [o.model_copy(update={"unit_kind": "chars"}) for o in _line(0.2, 0.05)]
    assert _estimate(chars).unit_kind == "chars" and _estimate(chars).single.u == pytest.approx(0.05)  # type: ignore[union-attr]
    mixed = _estimate(_line(0.2, 0.05) + chars)
    assert mixed.unit_kind == "none" and mixed.single is not None
    assert mixed.single.u is None and 0.2 <= mixed.single.r <= 0.6 and "single_slope_unmeasured" in mixed.reasons


# --- the figures the old path got wrong -------------------------------------------------------------------------


@pytest.mark.logic
def test_a_failed_request_never_enters_the_per_item_time() -> None:
    """A failed 9 s request does not move the estimate, which keeps 0.1 + 0.02·L."""
    good = _line(0.1, 0.02)
    failed = [_obs(5.0, 9.0, outcome="failed") for _ in range(4)]
    got = _estimate(good + failed).single
    assert got is not None and got.u == pytest.approx(0.02) and got.r == pytest.approx(0.1)


@pytest.mark.logic
def test_fixed_cost_is_fitted_not_shared() -> None:
    """Item wall 0.040 + 0.0366 per job over 1-job and 4-job items: `u` 0.0366 and `a` 0.040."""
    fixed, per_job = 0.040, 0.0366
    obs = []
    for stratum, length in enumerate((1.0, 4.0)):
        for _ in range(CONFIG.min_requests):
            obs.append(_obs(length, per_job * length, stratum=stratum, item_wall_s=fixed + per_job * length))
    got = _estimate(obs)
    assert got.single is not None and got.single.u == pytest.approx(per_job)
    assert got.item_s == pytest.approx(fixed)


@pytest.mark.logic
def test_device_bytes_scale_with_length() -> None:
    """Peak beyond resident is 5 000 + 50 per unit: the estimate gives `m1` 50."""
    got = _estimate(_with_device(5000, 50))
    assert got.device is not None and got.device.m1 == pytest.approx(50.0) and got.device.m0 == pytest.approx(5000.0)
    assert got.device.resident == 10**9


# --- the OOM raise ----------------------------------------------------------------------------------------------


@pytest.mark.logic
def test_an_oom_raises_m1_without_a_constant() -> None:
    """`m1′ = (headroom·grant − resident − m0)/((B − 1)·L)`: B = 8 at L = 40 leaves a memory batch of 7, not 199 998."""
    base = _estimate(_with_device(5000, 50))
    grant, batch, length = 2 * 10**9, 8, 40.0
    record = OomRecord(resource="device", batch=batch, length=length, cotenancy=1, grant_bytes=grant, basis=())
    raised = _estimate(_with_device(5000, 50), [record])
    assert base.device is not None and raised.device is not None
    expected = (HEADROOM * grant - base.device.resident - base.device.m0) / ((batch - 1) * length)
    assert raised.device.m1 == pytest.approx(expected)
    assert raised.oom == (record,)
    budget = HEADROOM * grant
    assert memory_batch(raised.device, budget, length, ceiling=10**9) == 7
    doubled = base.device.model_copy(update={"m1": base.device.m1 * 2})  # what `Pool._grow` did: double per-item bytes
    assert memory_batch(doubled, budget, length, ceiling=10**9) == 199_998 >= batch


@pytest.mark.logic
def test_an_oom_under_shared_device_or_at_batch_one_does_not_raise() -> None:
    """A record of another co-tenancy applies to that regime and above; a batch of one has none to shrink."""
    base = _estimate(_with_device(5000, 50))
    shared = OomRecord(resource="device", batch=8, length=40.0, cotenancy=2, grant_bytes=2 * 10**9, basis=())
    single = OomRecord(resource="device", batch=1, length=40.0, cotenancy=1, grant_bytes=2 * 10**9, basis=())
    got = _estimate(_with_device(5000, 50), [shared, single])
    assert got.device == base.device and got.oom == (shared, single)


@pytest.mark.logic
@given(
    m1=st.floats(1.0, 1e6),
    batch=st.integers(2, 64),
    length=st.floats(0.5, 200.0),
    above=st.floats(1.0, 4.0),
    grant=st.integers(10**9, 10**11),
)
@FAST
def test_after_an_oom_the_memory_batch_never_reaches_the_faulting_one(
    m1: float, batch: int, length: float, above: float, grant: int
) -> None:
    """At any length at or above the faulting one, the memory batch is below the faulting batch."""
    obs = _with_device(5000, m1)
    record = OomRecord(resource="device", batch=batch, length=length, cotenancy=1, grant_bytes=grant, basis=())
    fit = _estimate(obs, [record]).device
    assert fit is not None
    assert memory_batch(fit, HEADROOM * grant, length * above, ceiling=10**6) < batch


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
    record = OomRecord(resource="device", batch=4, length=15.0, cotenancy=1, grant_bytes=10**9, basis=("p1",))
    reopened = _plan(candidates, Prior(observations=prior.observations, records=(record,)))
    assert reopened.batched == (1,) and all(i.phase == "WARMUP" for i in reopened.items) and reopened.items
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
        Probe.plan_batched(plan, _measured_single(), GRANT, CONFIG, PROBE_S, memory_headroom=HEADROOM)
        .unplanned[-1]
        .reason
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
        Probe.plan_batched(plan, none, GRANT, CONFIG, PROBE_S, memory_headroom=HEADROOM).unplanned[-1].reason
        == "single_unmeasured"
    )
    nodev = _estimate(_line(0.2, 0.05))
    assert (
        Probe.plan_batched(plan, nodev, GRANT, CONFIG, PROBE_S, memory_headroom=HEADROOM).unplanned[-1].reason
        == "device_unmeasured"
    )
    nogrant = GRANT.model_copy(update={"memory_bytes": None})
    assert Probe.plan_batched(plan, _measured_single(), nogrant, CONFIG, PROBE_S, memory_headroom=HEADROOM).items == ()


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
    estimate = Probe.estimator(CONFIG, records_of=lambda _: (), memory_headroom=HEADROOM)
    got = MeasureSource([_Root(pooled)], _NoTable(), estimate=estimate).get(UNIT)
    assert got is not None and got.unit_kind == "audio_s" and got.estimator == "estimate@1"
    assert got.single is not None and got.single.u == pytest.approx(0.05)


@pytest.mark.logic
def test_a_gap_tie_lifts_by_the_range_above() -> None:
    """A record at L = 5.0 between ranges (2, 2) and (8, 8) is held by the range at or above L: only 8 lifts it."""
    base = _with_device(5000, 50)

    def probe_at(x: float) -> list[Observation]:
        ref = ProbeRef(id="p2", phase="BATCHED", stratum=1)
        return [_obs(x, 0.5, batch=4, probe=ref) for _ in range(CONFIG.min_requests)]

    fault = _fault()
    assert _estimate(base + probe_at(8.0), [fault]).oom == ()
    assert _estimate(base + probe_at(2.0), [fault]).oom == (fault,)


@pytest.mark.logic
def test_a_covered_plan_reports_unplanned_batched() -> None:
    """A plan whose strata the prior covers in both phases says so for the batched phase, not nothing."""
    obs = _covering([5.0, 30.0]) + _covering([5.0, 30.0], batch=4)
    plan = _plan(_candidates([float(i + 1) for i in range(40)]), Prior(observations=tuple(obs)))
    assert plan.batched == ()
    batched = Probe.plan_batched(plan, _measured_single(1e3), GRANT, CONFIG, 1e6, memory_headroom=HEADROOM)
    assert batched.items == () and batched.unplanned[-1] == Unplanned(phase="BATCHED", stratum=None, reason="covered")


def _reference_theil_sen(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """The definition: the median of every pairwise slope, and the median intercept at that slope."""
    ordered = sorted(points)
    slopes = [(y2 - y1) / (x2 - x1) for (x1, y1), (x2, y2) in itertools.combinations(ordered, 2) if x1 != x2]
    if not slopes:
        return None
    slope = statistics.median(sorted(slopes))
    return statistics.median(sorted(y - slope * x for x, y in ordered)), slope


@pytest.mark.logic
@given(points=st.lists(st.tuples(st.integers(0, 12), st.integers(-30, 30)), max_size=40))
@FAST
def test_bounded_fit_equals_reference(points: list[tuple[int, int]]) -> None:
    """The bounded fit returns the reference's intercept and slope exactly, ties in `x` and collinear data included."""
    floats = [(float(x), float(y)) for x, y in points]
    assert _theil_sen(floats) == _reference_theil_sen(floats)


def _agrees(points: Sequence[tuple[float, float]]) -> bool:
    """Whether the fit is within `_rank_bound` of the reference.

    The slope is within the bound plus one rounding of the mean of two ranks. Each intercept `y − s·x` moves by the
    slope's error times `|x|` and rounds twice in each fit; their median adds one rounding of the mean.
    """
    fit, reference = _theil_sen(points), _reference_theil_sen(points)
    assert fit is not None and reference is not None
    eps, (intercept, slope) = sys.float_info.epsilon, reference
    slope_tol = _rank_bound(points, slope) + eps * abs(slope)
    terms = max(abs(y) + 2 * abs(slope * x) for x, y in points)
    intercept_tol = slope_tol * max(abs(x) for x, _ in points) + 2 * eps * terms + eps * abs(intercept)
    return abs(fit[1] - slope) <= slope_tol and abs(fit[0] - intercept) <= intercept_tol


@pytest.mark.logic
@given(
    points=st.lists(
        st.tuples(st.floats(-1e3, 1e3), st.floats(-1e6, 1e6)), min_size=3, max_size=60, unique_by=lambda p: p[0]
    )
)
@FAST
def test_bounded_fit_equals_reference_on_real_data(points: list[tuple[float, float]]) -> None:
    """On arbitrary floats the fit is within the documented bound of the reference."""
    assert _agrees(points)


@pytest.mark.logic
def test_a_line_through_the_median_x_is_within_the_rounding_of_the_product() -> None:
    """On `y = s·(x − median x)`, `y − t·x̃` is near zero and the rounding of `t·x̃` alone bounds the fit."""
    xs = [-309, -44, 249, 258, 294, 332, 435, 912]
    assert _agrees([(float(x), -7.513524149434952 * (x - 276.0)) for x in xs])


@pytest.mark.logic
def test_x_a_few_ulps_apart_does_not_misrank_slopes() -> None:
    """Eight lengths one ulp apart near 476 with walls of either sign: centring makes the fit the reference's."""
    xs = [476.4309283301684]
    for _ in range(7):
        xs.append(math.nextafter(xs[-1], math.inf))
    ys = [-827064.8205435034, 327515.6096879614, -784137.4758118002, -672603.4057000517]
    ys += [679903.0809589487, -258954.66670850653, 465531.61784373154, -61359.71779395221]
    points = list(zip(xs, ys, strict=True))
    assert _theil_sen(points) == _reference_theil_sen(points)


@pytest.mark.logic
def test_a_covered_batched_phase_is_reported_without_a_device_measurement() -> None:
    """Coverage is decided before the single and device estimates are asked for."""
    obs = _covering([5.0, 30.0]) + _covering([5.0, 30.0], batch=4)
    plan = _plan(_candidates([float(i + 1) for i in range(40)]), Prior(observations=tuple(obs)))
    batched = Probe.plan_batched(plan, _estimate(_line(0.2, 0.05)), GRANT, CONFIG, 1e6, memory_headroom=HEADROOM)
    assert batched.unplanned[-1] == Unplanned(phase="BATCHED", stratum=None, reason="covered")


def _pooled(n: int) -> list[tuple[float, float]]:
    rng = random.Random(n)
    return [(x, 0.2 + 0.05 * x + rng.gauss(0, 0.05)) for x in (rng.uniform(1.0, 60.0) for _ in range(n))]


def _fit_steps(points: Sequence[tuple[float, float]]) -> int:
    work = Work()
    _theil_sen(points, work)
    return work.steps


@pytest.mark.logic
def test_fit_work_is_subquadratic(slack_tests: dict[str, Any]) -> None:
    """The slope fit's step count at 10^4 pooled requests is under `tests.probe.fit_ops_ratio` times that at 10^3."""
    ratio = _fit_steps(_pooled(10_000)) / _fit_steps(_pooled(1_000))
    assert ratio < slack_tests["probe"]["fit_ops_ratio"], f"{ratio:.1f}x the operations for 10x the requests"
