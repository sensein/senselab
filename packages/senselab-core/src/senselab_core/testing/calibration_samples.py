"""Samples of the calibration types, for the value contracts."""

from collections.abc import Callable

from senselab_core.runtime.calibration import (
    Cotenancy,
    Estimate,
    EstimateSource,
    HardwareClass,
    LengthProfile,
    MemoryFit,
    Observation,
    OomRecord,
    ProbeRef,
    RegimeFactor,
    RequestFit,
    RunCalibration,
    StratumBatch,
)
from senselab_core.runtime.execution_settings import CalibrationSettings

Register = Callable[..., None]

_CUDA = HardwareClass(device="cuda", name="NVIDIA H100 80GB HBM3", memory_mib=81559, compute_capability="9.0", cpus=8)
_CPU = HardwareClass(device="cpu", name="CPU model", cpus=1)
_FIT = RequestFit(batch=1, r=0.2, u=0.05, n=6, spread=0.1, lo=2.0, hi=9.0)


def _observation(**update: object) -> Observation:
    fields: dict[str, object] = {
        "probe": ProbeRef(id="p1", phase="SINGLE", stratum=0),
        "item": "i1",
        "request_index": 0,
        "jobs": 1,
        "batch": 1,
        "units": 4.0,
        "padded_units": 4.0,
        "wall_s": 0.5,
        "item_wall_s": None,
        "since_load": 3,
        "load_s": None,
        "device_peak": None,
        "device_after": None,
        "host_peak": None,
        "host_after": None,
        "cotenancy": Cotenancy(device=1, host=1),
        "outcome": "ok",
        "hardware": _CUDA,
    }
    return Observation.model_validate({**fields, **update})


def register_calibration_samples(register: Register, packaged: CalibrationSettings) -> None:
    """Register the samples of every calibration type through `register` (`values.register_samples`)."""
    register(CalibrationSettings, lambda: [packaged], invalid_updates=[{"trust": "all"}, {"measures_roots": ()}])
    register(RunCalibration, lambda: [RunCalibration(), RunCalibration(trust="own")], invalid_updates=[{"trust": "x"}])
    register(
        HardwareClass,
        lambda: [_CUDA, _CPU],
        invalid_updates=[{"cpus": 0}, {"device": "mps"}, {"name": ""}],
    )
    register(ProbeRef, lambda: [ProbeRef(id="p", phase="WARMUP", stratum=0)], invalid_updates=[{"phase": "IDLE"}])
    register(Cotenancy, lambda: [Cotenancy(device=1, host=3)], invalid_updates=[{"device": 0}])
    register(
        Observation,
        lambda: [
            _observation(),
            _observation(probe=None, device_peak=10, device_after=4, item_wall_s=1.0, load_s=2.0, hardware=_CPU),
        ],
        invalid_updates=[
            {"batch": 0},
            {"padded_units": 1.0},
            {"outcome": "lost"},
            {"device_peak": 1, "device_after": 2},
        ],
    )
    register(
        RequestFit,
        lambda: [_FIT, RequestFit(batch=4, r=0.3, u=None, n=3, spread=None, lo=1.0, hi=1.0)],
        invalid_updates=[{"lo": 99.0}, {"n": 0}],
    )
    register(
        MemoryFit,
        lambda: [MemoryFit(resident=1024, m0=0.0, m1=2.5, n=3)],
        invalid_updates=[{"resident": -1}, {"n": 0}],
    )
    register(StratumBatch, lambda: [StratumBatch(stratum=0, batch=8)], invalid_updates=[{"batch": 0}])
    register(RegimeFactor, lambda: [RegimeFactor(k=2, rho=1.4, n=6)], invalid_updates=[{"rho": 0}, {"k": 0}])
    register(
        OomRecord,
        lambda: [OomRecord(resource="device", batch=16, length=30.0, cotenancy=1)],
        invalid_updates=[{"resource": "disk"}, {"batch": 0}],
    )
    register(
        EstimateSource,
        lambda: [EstimateSource(), EstimateSource(probe_ids=("p1",), observations=9, commit="a" * 40)],
        invalid_updates=[{"commit": "abc"}, {"probe_ids": ("",)}],
    )
    register(
        Estimate,
        lambda: [
            Estimate(status="unmeasured", unit_kind="none", reasons=("no warm request",)),
            Estimate(
                status="measured",
                unit_kind="audio_s",
                load_s=3.0,
                load_n=1,
                item_s=0.04,
                item_n=3,
                single=_FIT,
                batched=(RequestFit(batch=4, r=0.3, u=0.04, n=3, spread=0.1, lo=2.0, hi=9.0),),
                batch=(StratumBatch(stratum=0, batch=4),),
                device=MemoryFit(resident=1024, m0=0.0, m1=2.5, n=3),
                regimes=(RegimeFactor(k=2, rho=1.4, n=6),),
                oom=(OomRecord(resource="device", batch=16, length=30.0, cotenancy=1),),
                source=EstimateSource(probe_ids=("p1",), observations=9),
            ),
        ],
        invalid_updates=[
            {"status": "measured", "single": None},
            {"reasons": (), "status": "unmeasured"},
            {"batch": (StratumBatch(stratum=0, batch=4), StratumBatch(stratum=0, batch=8))},
            {"regimes": (RegimeFactor(k=2, rho=1.4, n=6),) * 2},
        ],
    )
    register(
        LengthProfile,
        lambda: [LengthProfile(units=(3.0, None, 1.0)), LengthProfile(units=())],
        invalid_updates=[{"units": (-1.0,)}],
    )
