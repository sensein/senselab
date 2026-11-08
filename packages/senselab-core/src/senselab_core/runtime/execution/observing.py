"""What a pool worker reports per request of the items it ran: `Observation`s, from the requests its claims sent.

`Sent` is one call of a claim's `send`; `RequestRecord` one request of it, as `request_reports` collected it, or the
call itself where it failed. `observations` turns an item's records into its `Observation`s; `hardware_of` and
`cotenancy_of` read what a worker runs on and beside.
"""

import functools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from senselab_worker.units import NS_PER_S

from senselab_core.runtime.calibration import (
    Cotenancy,
    HardwareClass,
    HardwareGrant,
    Observation,
    ProbeRef,
    UnitKind,
)
from senselab_core.runtime.execution.batching import LengthUnrecorded, job_length_ns, job_text_chars
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.job import Failed, Job, JobOutcome
from senselab_core.runtime.execution.substrates import workers_on_host
from senselab_core.runtime.execution_settings import is_service_class
from senselab_core.runtime.hardware import HardwareInfo, device_cotenants, probe_hardware, process_tree, visible_gpus
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.request_reports import RequestReport

logger = logging.getLogger("senselab")

Failure = Literal["bisected", "failed"]


@dataclass(frozen=True)
class Sent:
    """One call of a claim's `send` for `refs`: its wall seconds, the requests it made, and how it failed if it did.

    `failure` is `bisected` where the call raised and was split in halves, `failed` where it raised and was not;
    a call that raised made no request of its own to report.
    """

    refs: tuple[JobRef, ...]
    wall_s: float
    reports: tuple[RequestReport, ...]
    failure: Failure | None


@dataclass(frozen=True)
class RequestRecord:
    """One request of a claim: the jobs it carried, its wall seconds, what its model state reported, how it fared."""

    refs: tuple[JobRef, ...]
    wall_s: float
    items: int
    since_load: int | None
    load_s: float | None
    device: tuple[int, int] | None
    failure: Failure | None


def records_of(sent: Sent) -> list[RequestRecord]:
    """Return the requests of one `send` call: its reports, else the call itself where it raised.

    A report tagged with nothing belongs to every job of the call.
    """
    if sent.failure is not None:
        return [RequestRecord(sent.refs, sent.wall_s, len(sent.refs), 0, None, None, sent.failure)]
    return [
        RequestRecord(
            tuple(r for r in report.tags if isinstance(r, JobRef)) or sent.refs,
            sent.wall_s if report.wall_s is None else report.wall_s,
            report.items,
            report.since_load,
            report.load_s,
            report.device,
            None,
        )
        for report in sent.reports
    ]


def job_units(job: Job) -> tuple[UnitKind, float]:
    """Return the kind and amount of `job`'s input: recording seconds, else held text characters, else none."""
    try:
        length = job_length_ns(job)
    except LengthUnrecorded:
        length = None
    if length is not None:
        return "audio_s", length / NS_PER_S
    chars = job_text_chars(job)
    return ("chars", float(chars)) if chars is not None else ("none", 0.0)


def request_units(jobs: Sequence[Job]) -> tuple[UnitKind, float, float]:
    """Return the kind of input a request's jobs share, their units' sum and the padded units: batch × the longest.

    Jobs of mixed kinds are of kind `none`, with no units.
    """
    found = [job_units(j) for j in jobs]
    kinds = {k for k, _ in found}
    if len(kinds) != 1 or kinds == {"none"}:
        return "none", 0.0, 0.0
    amounts = [u for _, u in found]
    return next(iter(kinds)), sum(amounts), len(amounts) * max(amounts)


@dataclass(frozen=True)
class Seen:
    """What one claim saw of where it ran: the hardware class, its co-tenancy and the probe it belongs to."""

    hardware: HardwareClass
    cotenancy: Cotenancy
    probe: tuple[str, str] | None = None
    served: bool = False


def observations(  # noqa: PLR0913 - an item's requests, then what they ran on and what became of its jobs
    digest: str,
    records: Sequence[RequestRecord],
    jobs: Mapping[JobRef, Job],
    outcomes: Sequence[JobOutcome],
    seen: Seen,
    *,
    mark: tuple[str, int] | None,
    item_wall_s: float,
) -> list[Observation]:
    """Return one `Observation` per request of item `digest` that reported what its model state had answered.

    `records` are the requests the item's jobs were in, in the order they finished; `jobs` holds every job of the
    claim by ref. A request is `ok` unless it raised, its class is a service class (`served`), or one of the item's
    jobs in it failed. The last request carries `item_wall_s`. A probe's phase and stratum are `mark`.
    """
    failed = {o.ref for o in outcomes if isinstance(o, Failed)}
    heard = [r for r in records if r.since_load is not None]
    found: list[Observation] = []
    for index, record in enumerate(heard):
        kind, units, padded = request_units([jobs[ref] for ref in record.refs])
        outcome = record.failure or ("served" if seen.served else "failed" if failed & set(record.refs) else "ok")
        probe = None
        if seen.probe is not None and mark is not None:
            probe = ProbeRef.model_validate({"id": seen.probe[1], "phase": mark[0], "stratum": mark[1]})
        found.append(
            Observation(
                probe=probe,
                item=digest,
                request_index=index,
                jobs=len(record.refs),
                batch=len(record.refs),
                unit_kind=kind,
                units=units,
                padded_units=padded,
                wall_s=record.wall_s,
                item_wall_s=item_wall_s if index == len(heard) - 1 else None,
                since_load=record.since_load or 0,
                load_s=record.load_s,
                device_peak=None if record.device is None else record.device[0],
                device_after=None if record.device is None else record.device[1],
                host_peak=None,
                host_after=None,
                cotenancy=seen.cotenancy,
                outcome=outcome,
                hardware=seen.hardware,
            )
        )
    return found


@functools.cache
def host_report() -> HardwareInfo:
    """Return this host's hardware, probed once."""
    return probe_hardware()


def hardware_of(profile: ExecutionProfile, resource_class: str) -> HardwareClass:
    """Return the `HardwareClass` a worker of `resource_class` runs on, as the profile's class grants it.

    A service class names no grant: it is read as the host's first GPU, else its CPUs.

    Raises:
        ValueError: the grant names a device or GPU this host does not show.
    """
    grant = profile.resource_classes.get(resource_class)
    if grant is not None:
        given = HardwareGrant(device=grant.device, cpus=grant.cpus)
    elif is_service_class(resource_class):
        given = HardwareGrant(device="cuda" if visible_gpus(host_report()) else "cpu")
    else:
        raise ValueError(f"{resource_class!r} is not a resource class of the profile")
    return HardwareClass.from_report(host_report(), given)


def cotenancy_of(profile: ExecutionProfile, resource_class: str, run_dir: Path) -> Cotenancy | None:
    """Return the compute processes on this worker's device and the pool workers on its host, itself counted.

    The device count is read from `nvidia-smi` for a `cuda` class and is 1 elsewhere; None where it cannot be read.
    """
    host = workers_on_host(run_dir)
    grant = profile.resource_classes.get(resource_class)
    if grant is None or grant.device != "cuda":
        return Cotenancy(device=1, host=host)
    gpus = visible_gpus(host_report())
    uuid = gpus[0].uuid if gpus else None
    shared = None if uuid is None else device_cotenants(uuid, process_tree(), timeout_s=profile.sizing.device_query_s)
    if shared is None:
        logger.warning("the compute processes on the device of class %s could not be read", resource_class)
        return None
    return Cotenancy(device=shared, host=host)
