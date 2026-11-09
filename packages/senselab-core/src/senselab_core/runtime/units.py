"""The plan-time `MeasureUnit` resolver: a call's unit from the env record, the hardware index and the placement.

`explain` and the in-process executor resolve a unit here before any worker starts, so a measure the roots or the
packaged table hold is read at plan time. A field that is not knowable yet gives `Unresolved`, whose reason names
what is true of the state.
"""

import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from senselab_core.kernel.declare import BackendSpec, Env
from senselab_core.runtime.calibration import (
    Estimate,
    HardwareClass,
    HardwareGrant,
)
from senselab_core.runtime.call import EnvUnreadable, host_hardware, packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.batching import BucketGrid
from senselab_core.runtime.execution.job import Job, env_fields, job_decoder
from senselab_core.runtime.execution.workers import PRECISION, service_of
from senselab_core.runtime.hardware import HardwareInfo, numerics_class
from senselab_core.runtime.keys import Call, Placement, call_key, fixed_formation, request_components
from senselab_core.runtime.launch import launch
from senselab_core.runtime.measures import MeasureRoots, MeasureSource, MeasureUnit
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.storage import StorePolicy
from senselab_core.runtime.vllm_server import server_variables

_NO_INPUT = "content:" + "0" * 64
"""The identity every input port holds while a key is built only to be read for its unit, which holds no inputs."""


@dataclass(frozen=True)
class Unresolved:
    """A unit that cannot be built yet, with the reason, true of the state, that it cannot."""

    reason: str


@dataclass(frozen=True)
class HardwareUnseen(Unresolved):
    """A unit whose only gap is that no worker of its class has reported the hardware it runs on.

    Attributes:
        seen: The class this host reports for the resource class: the one a first hello would confirm.
    """

    seen: HardwareClass | None = None


class _NoPackaged:
    """The packaged measures until `resource-table@2.0` ships: none."""

    def get(self, unit: MeasureUnit) -> Estimate | None:  # noqa: ARG002 - a PackagedMeasures
        return None


def measure_roots(profile: ExecutionProfile) -> MeasureRoots:
    """Return the measures roots `profile` names, read as this process's user."""
    return MeasureRoots.from_profile(
        profile.calibration,
        StorePolicy.load_default(),
        stale_after_s=profile.lease.stale_after_s,
        cache=cache_root(),
    )


def default_measures(profile: ExecutionProfile, roots: MeasureRoots | None = None) -> MeasureSource:
    """Return the source over `roots` (`measure_roots` when None) and the packaged table, estimated by `Probe`."""
    held = measure_roots(profile) if roots is None else roots
    estimate = Probe.estimator(
        profile.calibration, records_of=held.regime_records, memory_headroom=profile.sizing.memory_headroom
    )
    return MeasureSource([held], _NoPackaged(), estimate=estimate)


EnvIdentity = tuple[Env, tuple[tuple[str, str], ...], tuple[str, ...]]
"""What an env read builds a handle from: the env, the worker overlay placed in it and the requirements it is checked
against."""


def env_identity(backend: BackendSpec) -> EnvIdentity:
    """Return the identity of the env `backend` runs in: its env, its sorted worker sources and its requirements."""
    return backend.env, tuple(sorted(backend.worker_sources.items())), backend.requirements


class EnvStates:
    """What each backend's env is under a profile, read once while built: its handle, or why it is not known."""

    def __init__(self, profile: ExecutionProfile) -> None:
        """Read envs under `profile`."""
        self.profile = profile
        self._built: dict[EnvIdentity, EnvHandle] = {}

    def get(self, backend: BackendSpec) -> EnvHandle | Unresolved:
        """Return the handle of `backend`'s built env, else the true reason it is not known before a worker starts.

        A handle is kept under its `env_identity`; an unbuilt env is read again, since the run builds it.
        """
        if self.profile.substrate.kind != "local":
            return Unresolved("the node a slurm worker runs on is chosen at submission, so its platform is not known")
        key = env_identity(backend)
        if key in self._built:
            return self._built[key]
        try:
            handle = packaged().env_read(backend, self.profile)
        except EnvUnreadable as err:
            return Unresolved(f"the env of backend {backend.name!r} cannot be read: {err}")
        if handle is None:
            return Unresolved(f"the env of backend {backend.name!r} is not built")
        self._built[key] = handle
        return handle


class UnitResolver:
    """Builds the `MeasureUnit` of a call placed under a profile, reading the env record and the hardware index only."""

    def __init__(
        self, profile: ExecutionProfile, roots: MeasureRoots | None = None, hardware: HardwareInfo | None = None
    ) -> None:
        """Resolve under `profile` over `roots` (`measure_roots`) on `hardware` (this host's probe)."""
        self.profile = profile
        self.roots = measure_roots(profile) if roots is None else roots
        self.envs = EnvStates(profile)
        self._hardware_info = hardware

    @property
    def hardware(self) -> HardwareInfo:
        """The hardware the resolver reads: the given, else this host's probe (once per process)."""
        return self._hardware_info if self._hardware_info is not None else host_hardware()

    def unit(
        self,
        call: Call,
        placement: Placement,
        decoder: str | None = None,
        hardware: HardwareClass | None = None,
    ) -> MeasureUnit | Unresolved:
        """Return the unit of `call` run as `placement`, whose file-backed inputs are read by `decoder` (None: none).

        A bucketed backend's formation is the grid's, whichever bucket a job falls in. `hardware` is the class a
        worker's hello reported; it replaces this host's, which then need not be in the index.
        """
        backend = placement.backend
        handle = self.envs.get(backend)
        if isinstance(handle, Unresolved):
            return handle
        found = hardware if hardware is not None else self._hardware(placement)
        if isinstance(found, Unresolved):
            return found
        hardware = found
        try:
            expected = service_of(backend, call.model)
        except (LookupError, ValueError) as err:
            return Unresolved(f"the attestation of backend {backend.name!r}'s server is unreadable: {err}")
        formation = fixed_formation(backend) or BucketGrid.load_default().unindexed
        keyed = replace(placement, **request_components(backend, formation))
        fields = {
            **env_fields(keyed, handle, expected, network_variables(self.profile.network)),
            "platform": sysconfig.get_platform(),
            "numerics": numerics_class(self.hardware, placement.device, PRECISION),
            "decoder": decoder,
            "threads": self._threads(backend, handle, expected is not None, call),
        }
        blank = dict.fromkeys(call.task.inputs, _NO_INPUT)
        return MeasureUnit.of(keyed, call_key(replace(call, service=expected, inputs=blank), keyed), fields, hardware)

    def _threads(self, backend: BackendSpec, handle: EnvHandle, served: bool, call: Call) -> dict[str, str]:
        if backend.env.kind == "host":
            return {}
        declared = network_variables(self.profile.network)
        if served and backend.server is not None:
            declared = {**server_variables(backend.server.settings), **declared}
        return dict(launch(handle.python, declared).measured)

    def _hardware(self, placement: Placement) -> HardwareClass | Unresolved:
        """Return this host's class for the placement's resource class, which the hardware index must hold."""
        resource = self.profile.resource_classes[placement.resource_class]
        try:
            here = HardwareClass.from_report(self.hardware, HardwareGrant(device=resource.device, cpus=resource.cpus))
        except ValueError as err:
            return Unresolved(f"this host reports no hardware class for {placement.resource_class!r}: {err}")
        if here not in self.roots.seen_classes(resource):
            return HardwareUnseen(
                f"hardware not yet seen on this host: no worker of class {placement.resource_class!r} reported it",
                here,
            )
        return here


def decoder_of(jobs: Sequence[Job]) -> str | None | Unresolved:
    """Return the decoder the file-backed inputs of `jobs` are read by (None: none is), or why they are two units."""
    found = {job_decoder(j) for j in jobs}
    if len(found) > 1:
        return Unresolved("its jobs read both files and in-memory inputs, which are two units")
    return next(iter(found), None)


class UnitReads(Protocol):
    """What a pool reads to place a call: the env states and the unit of a call."""

    envs: EnvStates

    def unit(
        self,
        call: Call,
        placement: Placement,
        decoder: str | None = None,
        hardware: HardwareClass | None = None,
    ) -> MeasureUnit | Unresolved:
        """Return the unit of `call` run as `placement`, `hardware` the class a worker's hello reported."""


UnitOf = Callable[[Call, Placement, str | None], MeasureUnit | Unresolved]
"""What the in-process executor resolves a unit by: the call (its model set), its placement and its decoder."""

__all__ = [
    "HardwareUnseen",
    "UnitOf",
    "UnitReads",
    "UnitResolver",
    "Unresolved",
    "decoder_of",
    "default_measures",
    "measure_roots",
]
