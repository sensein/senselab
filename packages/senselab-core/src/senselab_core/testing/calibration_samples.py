"""Samples of the calibration types, for the value contracts."""

from collections.abc import Callable

from senselab_core.kernel.identity import ModelIdentity
from senselab_core.runtime.calibration import (
    Candidate,
    Cotenancy,
    Estimate,
    EstimateSource,
    HardwareClass,
    HardwareGrant,
    LengthProfile,
    MemoryFit,
    Observation,
    OomRecord,
    Prior,
    ProbeItem,
    ProbePlan,
    ProbeRef,
    RegimeFactor,
    RequestFit,
    RunCalibration,
    Stratum,
    StratumBatch,
    Unplanned,
)
from senselab_core.runtime.execution_settings import CalibrationSettings
from senselab_core.runtime.measures import MeasureEnv, MeasureUnit
from senselab_core.runtime.resources import Derivation, MeasureRow, MeasureTable

Register = Callable[..., None]

_CUDA = HardwareClass(device="cuda", name="NVIDIA H100 80GB HBM3", memory_mib=81559, compute_capability="9.0", cpus=8)
_CPU = HardwareClass(device="cpu", name="CPU model", cpus=1)
ENV = MeasureEnv(
    env_lock="a" * 64,
    wheel_set="b" * 64,
    driver={"torch_build": "cu128"},
    platform="linux-x86_64",
    decoder=None,
    numerics={"device": "cuda"},
    threads={"OMP_NUM_THREADS": 8},
)
UNIT = MeasureUnit(
    task="speech_to_text",
    task_version="1",
    backend="whisper",
    backend_version="3",
    params={"word_timestamps": False},
    backend_params={"dtype": "float16"},
    formation="mixed",
    chunk_grid=None,
    model=ModelIdentity(id="openai/whisper-tiny", identity="c" * 40),
    env=ENV,
    service=None,
    hardware=_CUDA,
)
_ITEM = ProbeItem(
    phase="SINGLE", stratum=0, request_jobs=1, jobs=(Candidate(fragment="f1", index=0, key="k", units=3.0),)
)
_PLAN = ProbePlan(
    items=(_ITEM,),
    strata=(Stratum(index=0, lo=3.0, hi=3.0),),
    remaining=((),),
    batched=(0,),
    unplanned=(),
    formation="mixed",
    seconds=90.0,
    probe_items=50,
    used_jobs=1,
)
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
            _observation(
                probe=None, device_peak=10, device_after=4, item_wall_s=1.0, since_load=0, load_s=2.0, hardware=_CPU
            ),
        ],
        invalid_updates=[
            {"batch": 0},
            {"padded_units": 1.0},
            {"outcome": "lost"},
            {"device_peak": 1, "device_after": 2},
            {"load_s": 1.0, "since_load": 2},
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
    register(RegimeFactor, lambda: [RegimeFactor(k=2, rho=1.4, n=6)], invalid_updates=[{"rho": 0}, {"k": 0}, {"k": 1}])
    register(
        OomRecord,
        lambda: [OomRecord(resource="device", batch=16, length=30.0, cotenancy=1, grant_bytes=2**36, basis=("p1",))],
        invalid_updates=[{"resource": "disk"}, {"batch": 0}, {"grant_bytes": 0}],
    )
    register(
        EstimateSource,
        lambda: [EstimateSource(), EstimateSource(probe_ids=("p1",), observations=9, commit="a" * 40)],
        invalid_updates=[{"commit": "abc"}, {"probe_ids": ("",)}],
    )
    register(
        Estimate,
        lambda: [
            Estimate(status="unmeasured", unit_kind="none", reasons=("no_usable_observations",)),
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
                oom=(
                    OomRecord(resource="device", batch=16, length=30.0, cotenancy=1, grant_bytes=2**36, basis=("p1",)),
                ),
                source=EstimateSource(probe_ids=("p1",), observations=9),
            ),
        ],
        invalid_updates=[
            {"status": "measured", "single": None},
            {"reasons": (), "status": "unmeasured"},
            {"batch": (StratumBatch(stratum=0, batch=4), StratumBatch(stratum=0, batch=8))},
            {"regimes": (RegimeFactor(k=2, rho=1.4, n=6),) * 2},
            {"reasons": ("lazy",)},
            {"estimator": "estimate@2"},
        ],
    )
    register(
        LengthProfile,
        lambda: [LengthProfile(units=(3.0, None, 1.0)), LengthProfile(units=())],
        invalid_updates=[{"units": (-1.0,)}],
    )
    register_probe_samples(register)
    register_measure_samples(register)


def register_probe_samples(register: Register) -> None:
    """Register the samples of the probe plan types."""
    register(
        Candidate,
        lambda: [
            Candidate(fragment="f1", index=0, key="k", units=3.0),
            Candidate(fragment="f1", index=1, key="j", units=None),
        ],
        invalid_updates=[{"fragment": ""}, {"units": -1.0}],
    )
    register(
        ProbeItem,
        lambda: [_ITEM, _ITEM.model_copy(update={"phase": "BATCHED", "request_jobs": 1})],
        invalid_updates=[{"jobs": ()}, {"request_jobs": 2}],
    )
    register(
        Stratum,
        lambda: [Stratum(index=0, lo=1.0, hi=4.0), Stratum(index=1, lo=None, hi=None)],
        invalid_updates=[{"index": -1}, {"lo": 9.0, "hi": 2.0}, {"lo": 1.0, "hi": None}],
    )
    register(
        Unplanned,
        lambda: [
            Unplanned(phase="SINGLE", stratum=0, reason="covered"),
            Unplanned(phase="WARMUP", stratum=None, reason="too_few_candidates"),
        ],
        invalid_updates=[{"reason": "lazy"}],
    )
    register(
        Prior,
        lambda: [Prior(), Prior(observations=(_observation(),), records=(RegimeFactor(k=2, rho=1.4, n=6),))],
        invalid_updates=[{"observations": (1,)}],
    )
    register(
        ProbePlan,
        lambda: [_PLAN],
        invalid_updates=[
            {"probe_items": 0},
            {"seconds": -1.0},
            {"formation": ""},
            {"remaining": ()},
            {"batched": (3,)},
            {"used_jobs": 51},
        ],
    )


def register_measure_samples(register: Register) -> None:
    """Register the samples of the measure identity types and the packaged table."""
    register(
        HardwareGrant,
        lambda: [HardwareGrant(device="cuda", cpus=4, gpu=1), HardwareGrant(device="cpu")],
        invalid_updates=[{"cpus": 0}, {"device": "tpu"}, {"gpu": -1}],
    )
    register(
        MeasureEnv,
        lambda: [ENV],
        invalid_updates=[
            {"env_lock": ""},
            {"platform": ""},
            {"driver": {"torch_index": "https://u:p@mirror.internal/whl/cu128"}},
            {"driver": {"launch_env": "9" * 64}},
            {"threads": {"OMP_NUM_THREADS": "abc"}},
            {"threads": {"OMP_NUM_THREADS": 0}},
        ],
    )
    register(
        MeasureUnit,
        lambda: [UNIT, UNIT.model_copy(update={"model": None, "formation": "bucket:g1", "hardware": _CPU})],
        invalid_updates=[{"formation": "pooled"}, {"task": ""}, {"chunk_grid": ""}, {"service": ""}],
    )
    derivation = Derivation(commit="d" * 40, jobs=("j1",), probe_ids=("p1",))
    row = MeasureRow(
        unit=UNIT,
        estimate=Estimate(status="unmeasured", unit_kind="none", reasons=("no_usable_observations",)),
        derivation=derivation,
    )
    register(Derivation, lambda: [derivation], invalid_updates=[{"commit": "abc"}, {"jobs": ()}])
    register(MeasureRow, lambda: [row], invalid_updates=[{"derivation": None}])
    register(
        MeasureTable,
        lambda: [MeasureTable(task="speech_to_text", backend="whisper", rows=(row,))],
        invalid_updates=[{"backend": "other"}, {"rows": (row, row)}],
    )
