"""Samples of the placement, license-set, resource-table and execution-settings types, for the value contracts."""

from collections.abc import Callable, Mapping, Sequence

from senselab_core.kernel._base import Frozen
from senselab_core.kernel.license import License, LicenseSet
from senselab_core.kernel.placement import Cost, NodePlacement, PlanPlacement
from senselab_core.runtime.execution.events import ClassActuals, PlacementRecord
from senselab_core.runtime.execution_settings import (
    FaultBudgets,
    JobLease,
    PoolSettings,
    ResourceClass,
    ServiceSettings,
    SizingSettings,
    SlurmCommands,
    SubstrateSpec,
    SupervisorSettings,
    Timeouts,
)
from senselab_core.runtime.resources import (
    Batched,
    ClassMeasure,
    Measured,
    ResourceMeasure,
    ResourceRow,
    ResourceTable,
)
from senselab_core.runtime.site_placement import (
    ClassNeeds,
    ClassPlacement,
    CurrentValues,
    HookRef,
    PlacementAnswer,
    PlacementRequest,
    PlacementValues,
    ResolvedPlacement,
    UnmeasuredFigure,
)
from senselab_core.runtime.sizing import Basis, Grant, Measure, RowSizing, Sample, Sizing

Register = Callable[..., None]

_COMMIT = "0" * 40
_MEASURED = Measured(sha="1" * 40, jobs=("100", "101"))
_MEASURE = ResourceMeasure(
    load_s=2.5, claim_s=0.01, resident_bytes=1024, peak_memory_bytes=4096, s_per_input_s=0.1, input_s_per_item=8.0
)
_CLASS = ClassMeasure(**_MEASURE.model_dump(), measured=_MEASURED)


def resource_table(task: str, backend: str, model_id: str | None, revision: str | None = _COMMIT) -> ResourceTable:
    """Return a one-row table of `task` and `backend` measuring `model_id` at `revision` on `cpu`."""
    row = ResourceRow(
        id=model_id,
        revision=None if model_id is None else revision,
        download_bytes=1000,
        classes={"cpu": _CLASS},
    )
    return ResourceTable(task=task, backend=backend, rows=(row,))


def _node(device: str = "cpu", resource_class: str = "cpu") -> NodePlacement:
    return NodePlacement(
        backend="b",
        backend_version="1",
        backend_params={"beam": 2},
        stack="s",
        env="isolated:x",
        device=device,
        batch_size=1,
        resource_class=resource_class,
        partition="${GPU_PARTITION}" if device == "cuda" else None,
        why="first declared",
    )


def register_placement_samples(register: Register, license_: License) -> None:
    """Register the samples of every type here through `register` (`values.register_samples`)."""
    _register_kernel(register, license_)
    _register_resources(register)
    _register_settings(register)
    _register_sizing(register)
    _register_sizings(register)
    _register_site(register)


def _register_kernel(register: Register, license_: License) -> None:
    sets = (LicenseSet(records={"model org/m": license_, "env of backend t/b": None}), LicenseSet())
    register(LicenseSet, lambda: list(sets), invalid_updates=[{"records": {"": None}}, {"records": {"m": 1}}])
    register(
        NodePlacement,
        lambda: [_node(), _node("cuda", "gpu:h100")],
        invalid_updates=[{"batch_size": 0}, {"resource_class": "tpu"}, {"backend": ""}],
    )
    costs = (Cost(wall_s=12.5, gpu_hours=0.25, download_bytes=10), Cost(unmeasured=("call", "org/m@" + _COMMIT)))
    register(
        Cost,
        lambda: list(costs),
        invalid_updates=[{"wall_s": -1.0}, {"download_bytes": -1}, {"unmeasured": ("b", "a")}],
    )
    register(
        PlanPlacement,
        lambda: [
            PlanPlacement(
                profile_digest="ab" * 32,
                nodes={"n": _node()},
                env_builds=("isolated:x",),
                licenses={"n": sets[0]},
                cost=costs[1],
            ),
            PlanPlacement(profile_digest="cd" * 32, nodes={}, licenses={}, cost=Cost()),
        ],
        invalid_updates=[{"format": "placement@3.0"}, {"licenses": {"z": sets[1]}}, {"profile_digest": "x"}],
    )


def _register_resources(register: Register) -> None:
    register(
        ResourceMeasure,
        lambda: [
            _MEASURE,
            ResourceMeasure(
                load_s=0,
                claim_s=0,
                resident_bytes=0,
                peak_memory_bytes=1,
                device_resident_bytes=2**30,
                device_peak_bytes=2**31,
                s_per_input_s=0,
                input_s_per_item=None,
                batched=Batched(batch_size=8, s_per_input_s=0.05),
            ),
        ],
        invalid_updates=[
            {"peak_memory_bytes": 0},
            {"load_s": -1.0},
            {"claim_s": -1.0},
            {"resident_bytes": 8192},
            {"input_s_per_item": 0.0},
            {"device_resident_bytes": 1, "device_peak_bytes": None},
            {"device_resident_bytes": 2**32, "device_peak_bytes": 2**31},
        ],
    )
    register(
        Batched,
        lambda: [Batched(batch_size=8, s_per_input_s=0.05), Batched(batch_size=2, s_per_input_s=0)],
        invalid_updates=[{"batch_size": 1}, {"s_per_input_s": -1.0}],
    )
    register(
        ClassMeasure,
        lambda: [_CLASS, _CLASS.model_copy(update={"measured": Measured(sha=None, run="r", jobs=("7",))})],
        invalid_updates=[{"measured": None}, {"load_s": -1.0}],
    )
    register(
        Measured,
        lambda: [_MEASURED, Measured(sha=None, run="0a" * 32, jobs=("host:1",))],
        invalid_updates=[{"sha": "main"}, {"jobs": ("",)}, {"jobs": ()}],
    )
    rows: Sequence[ResourceRow] = (
        resource_table("t", "b", "org/m").rows[0],
        resource_table("t", "b", None).rows[0],
    )
    register(
        ResourceRow,
        lambda: list(rows),
        invalid_updates=[{"revision": "main"}, {"classes": {}}, {"classes": {"tpu": _CLASS}}],
    )
    register(
        ResourceTable,
        lambda: [resource_table("t", "b", "org/m"), resource_table("u", "c", None)],
        invalid_updates=[{"format": "resource-table@2.0"}, {"rows": (rows[0], rows[0])}],
    )


def _register_settings(register: Register) -> None:
    classes: Mapping[type[Frozen], tuple[Sequence[Frozen], Sequence[Mapping[str, object]]]] = {
        ResourceClass: (
            [
                ResourceClass(device="cpu", max_workers=1),
                ResourceClass(
                    device="cuda",
                    gpu_type="h100",
                    cpus=8,
                    mem_bytes=2**36,
                    time_s=3600.0,
                    partition="${GPU_PARTITION}",
                    gres="gpu:h100:1",
                    max_workers=4,
                    memory_cap_bytes=2**36,
                ),
            ],
            [{"max_workers": 0}, {"gpu_type": "h100", "device": "cpu"}, {"device": "tpu"}],
        ),
        SubstrateSpec: (
            [
                SubstrateSpec(kind="local", start_backoff_first_s=0.05, start_backoff_max_s=30, placement_hook_s=0.05),
                SubstrateSpec(
                    kind="slurm",
                    qos_cpu_cap=64,
                    exclude_nodes=("n1",),
                    start_backoff_first_s=1,
                    start_backoff_max_s=30,
                    placement_hook="site.hooks:place",
                    placement_hook_s=1,
                    commands=SlurmCommands(timeout_s=60, attempts=3, retry_first_s=0.5),
                ),
            ],
            [
                {"kind": "k8s"},
                {"kind": "local", "qos_cpu_cap": 4},
                {"kind": "slurm", "commands": None},
                {"start_backoff_first_s": 0},
                {"placement_hook_s": 0},
                {"placement_hook": "not a path"},
            ],
        ),
        SlurmCommands: (
            [
                SlurmCommands(timeout_s=60, attempts=3, retry_first_s=0.5),
                SlurmCommands(timeout_s=0.5, attempts=1, retry_first_s=0.01),
            ],
            [{"timeout_s": 0}, {"attempts": 0}, {"retry_first_s": -1}],
        ),
        PoolSettings: (
            [
                PoolSettings(
                    quantum_items=100,
                    quantum_s=300,
                    poll_s=30,
                    first_poll_s=0.05,
                    presence_refresh_s=1,
                    segment_items=10,
                    admit_threads=2,
                ),
                PoolSettings(
                    quantum_items=1,
                    quantum_s=1,
                    poll_s=0.5,
                    first_poll_s=0.5,
                    presence_refresh_s=0.5,
                    segment_items=1,
                    admit_threads=1,
                ),
            ],
            [{"poll_s": 0}, {"quantum_items": 0}],
        ),
        JobLease: (
            [
                JobLease(stale_after_s=300, renew_fraction=0.33, attribute_delay_s=60),
                JobLease(stale_after_s=0.5, renew_fraction=0.9, attribute_delay_s=0),
            ],
            [{"stale_after_s": 0}, {"renew_fraction": 1.0}, {"renew_fraction": 0}, {"attribute_delay_s": -1}],
        ),
        FaultBudgets: (
            [
                FaultBudgets(content_attempts=2, infrastructure_replacements=6, resource_replacements=1),
                FaultBudgets(content_attempts=1, infrastructure_replacements=0, resource_replacements=0),
            ],
            [{"content_attempts": 0}, {"resource_replacements": -1}],
        ),
        Timeouts: ([Timeouts(), Timeouts(load_s=60.0, request_s=600.0)], [{"load_s": 0}, {"request_s": -1.0}]),
        SupervisorSettings: (
            [SupervisorSettings(start_time_probe_s=5), SupervisorSettings(start_time_probe_s=0.5)],
            [{"start_time_probe_s": 0}, {"start_time_probe_s": float("inf")}],
        ),
        ServiceSettings: ([ServiceSettings(idle_exit_s=600), ServiceSettings(idle_exit_s=1.5)], [{"idle_exit_s": 0}]),
    }
    for cls, (samples, invalid) in classes.items():
        register(cls, lambda samples=samples: list(samples), invalid_updates=invalid)


_BASIS = Basis(time=True, memory=False, batch=False, workers=True)


def _register_sizings(register: Register) -> None:
    classes: Mapping[type[Frozen], tuple[Sequence[Frozen], Sequence[Mapping[str, object]]]] = {
        Basis: (
            [
                Basis(time=True, memory=True, batch=True, workers=True),
                Basis(time=False, memory=False, batch=False, workers=False),
            ],
            [{"time": "maybe"}],
        ),
        Sizing: (
            [
                Sizing(cuts=((0, 3), (3, 4)), workers=2, predicted_s=(1.0, None, 2.5, 0.5), measured=_BASIS),
                Sizing(cuts=(), workers=0, measured=Basis(time=False, memory=False, batch=False, workers=False)),
            ],
            [{"cuts": ((0,),)}, {"workers": -1}, {"ceiling": 0}],
        ),
    }
    for cls, (samples, invalid) in classes.items():
        register(cls, lambda samples=samples: list(samples), invalid_updates=invalid)


def _register_sizing(register: Register) -> None:
    classes: Mapping[type[Frozen], tuple[Sequence[Frozen], Sequence[Mapping[str, object]]]] = {
        SizingSettings: (
            [
                SizingSettings(
                    load_share=0.05,
                    memory_headroom=0.9,
                    target_wall_s=1800,
                    calibration_items=4,
                    calibration_workers=1,
                    device_query_s=30,
                ),
                SizingSettings(
                    load_share=0.5,
                    memory_headroom=1.0,
                    target_wall_s=0.5,
                    calibration_items=2,
                    calibration_workers=3,
                    device_query_s=0.5,
                ),
            ],
            [
                {"load_share": 1.0},
                {"memory_headroom": 0},
                {"target_wall_s": 0},
                {"calibration_items": 1},
                {"calibration_workers": 0},
                {"device_query_s": 0},
            ],
        ),
        Measure: (
            [
                Measure(load_s=2.5, claim_s=0.01, resident_bytes=1024, bytes_per_item=256, s_per_item=0.5),
                Measure(load_s=0, claim_s=0, resident_bytes=0, bytes_per_item=0, s_per_item=0, on_device=True),
            ],
            [{"load_s": -1.0}, {"bytes_per_item": -1}],
        ),
        Grant: (
            [
                Grant(max_workers=4, memory_bytes=2**36, quantum_items=100, quantum_s=300),
                Grant(max_workers=1, memory_bytes=None, quantum_items=1, quantum_s=0.5),
            ],
            [{"max_workers": 0}, {"memory_bytes": 0}, {"quantum_s": 0}],
        ),
        RowSizing: (
            [
                RowSizing(batch_size=8, memory_batch=8, quantum_items=50, quantum_s=27.5, workers=4),
                RowSizing(batch_size=1, memory_batch=4, quantum_items=1, quantum_s=0, workers=0),
            ],
            [{"batch_size": 0}, {"memory_batch": 0}, {"workers": -1}],
        ),
        Sample: (
            [
                Sample(
                    wall_s=3.0,
                    claim_s=0.01,
                    jobs=1,
                    batch_size=1,
                    input_s=8.0,
                    baseline_bytes=1024,
                    resident_bytes=2048,
                    peak_memory_bytes=4096,
                    device_baseline_bytes=0,
                    device_bytes=2**30,
                    device_item_bytes=2**20,
                    device_allocated_bytes=2**29,
                    cold=True,
                    job="7",
                    ended_s=1_800_000_000.0,
                ),
                Sample(
                    wall_s=0,
                    claim_s=0,
                    jobs=2,
                    batch_size=2,
                    input_s=None,
                    baseline_bytes=0,
                    resident_bytes=None,
                    peak_memory_bytes=0,
                    cold=False,
                    job="h:1",
                    ended_s=0,
                ),
            ],
            [{"jobs": 0}, {"peak_memory_bytes": -1}, {"job": ""}, {"claim_s": -1.0}, {"ended_s": -1.0}],
        ),
    }
    for cls, (samples, invalid) in classes.items():
        register(cls, lambda samples=samples: list(samples), invalid_updates=invalid)


_CURRENT = CurrentValues(partition="gpuq", time_s=21600.0, max_workers=4, layers={"partition": "profile"})
_NEEDS = ClassNeeds(
    resource_class="gpu:h100",
    device="cuda",
    gpu_type="h100",
    device_memory_bytes=2**36,
    cpus=8,
    mem_bytes=2**36,
    workers=2,
    workers_max=4,
    time_per_worker_s=3600.0,
    wall_s=7200.0,
    gpu_hours=4.0,
    unmeasured=(UnmeasuredFigure(figure="time_floor_s", why="the load is unmeasured"),),
    current=_CURRENT,
)
_PLACED = ClassPlacement(
    values=_CURRENT,
    exclude=("n1",),
    verdict="accepted",
    answer=PlacementAnswer(why="idle"),
    names={"q": "<partition>", "n1": "<node>"},
)


def _register_site(register: Register) -> None:
    classes: Mapping[type[Frozen], tuple[Sequence[Frozen], Sequence[Mapping[str, object]]]] = {
        PlacementValues: (
            [PlacementValues(), PlacementValues(partition="q", time_s=60.0, max_workers=2, exclude=("n1",))],
            [{"partition": ""}, {"exclude": ("",)}, {"max_workers": "many"}],
        ),
        CurrentValues: (
            [_CURRENT, CurrentValues(max_workers=1, layers={})],
            [{"max_workers": 0}, {"time_s": 0.0}, {"layers": {"time_s": "nobody"}}],
        ),
        UnmeasuredFigure: (
            [UnmeasuredFigure(figure="wall_s", why="unmeasured")],
            [{"why": ""}, {"figure": ""}],
        ),
        ClassNeeds: (
            [_NEEDS, ClassNeeds(resource_class="cpu", device="cpu", workers=1, workers_max=1, current=_CURRENT)],
            [{"workers": -1}, {"workers_max": 0}, {"device": "tpu"}],
        ),
        PlacementRequest: (
            [PlacementRequest(classes=(_NEEDS,)), PlacementRequest(classes=())],
            [{"format": "placement-request@2.0"}, {"classes": 3}],
        ),
        PlacementAnswer: (
            [PlacementAnswer(), PlacementAnswer(classes={"cpu": PlacementValues(time_s=60.0)}, why="idle")],
            [{"format": "placement-answer@2.0"}, {"classes": 3}],
        ),
        HookRef: (
            [HookRef(path="site.hooks:place"), HookRef(path="a:b", version="1.0", importable=False)],
            [{"path": ""}],
        ),
        ClassPlacement: (
            [_PLACED, ClassPlacement(values=_CURRENT)],
            [{"verdict": "maybe"}, {"exclude": ("",)}, {"names": {"q": "partition"}}],
        ),
        ResolvedPlacement: (
            [ResolvedPlacement(classes={"cpu": _PLACED}), ResolvedPlacement(classes={})],
            [{"classes": 3}],
        ),
        PlacementRecord: (
            [
                PlacementRecord(resource_class="cpu", verdict="no-hook", used=_CURRENT),
                PlacementRecord(
                    resource_class="gpu:h100",
                    request=_NEEDS,
                    answer=PlacementAnswer(why="idle"),
                    hook=HookRef(path="a:b"),
                    verdict="accepted",
                    used=_CURRENT,
                    exclude=("<node>",),
                    closing=ClassActuals(wall_s=12.5, gpu_hours=0.25, device_peak_bytes=2**30),
                ),
            ],
            [{"verdict": "maybe"}, {"resource_class": ""}, {"closing": {"wall_s": -1.0}}],
        ),
        ClassActuals: (
            [ClassActuals(), ClassActuals(wall_s=0.0, gpu_hours=1.5, device_peak_bytes=0)],
            [{"wall_s": -1.0}, {"gpu_hours": float("nan")}, {"device_peak_bytes": -1}],
        ),
    }
    for cls, (samples, invalid) in classes.items():
        register(cls, lambda samples=samples: list(samples), invalid_updates=invalid)
