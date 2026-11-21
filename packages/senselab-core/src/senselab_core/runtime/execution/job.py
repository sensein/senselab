"""Job formation: one job's call, inputs and anchors; its code license, staleness, resolved inputs and context.

A key holds the batch formation of the request a job is sent in (`request_components`), which the job alone picks
(`batching.formation`). A failure is never stored.
"""

import sysconfig
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from senselab_worker import WIRE_VERSION

from senselab_core.kernel.declare import InconsistentInputs
from senselab_core.kernel.identity import Attestation, ExecutionContext
from senselab_core.kernel.license import License
from senselab_core.kernel.signals import Recording, StaleSource
from senselab_core.kernel.wire import WireFormatRetired
from senselab_core.runtime import audio_io
from senselab_core.runtime.call import RunConfig, env_license, host_hardware
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.batching import one_node
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, UpstreamRef
from senselab_core.runtime.execution.outputs import AmbiguousClock, Anchors, OutputTypeMismatch, anchors_of, read_port
from senselab_core.runtime.execution.workers import PRECISION
from senselab_core.runtime.hardware import host_token, numerics_class
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.launch import launch
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import attested_fields, server_launch, server_variables

Reader = Callable[[str, str], Mapping[str, object] | None]
"""Reads the stored outputs of an upstream entry by its key digest and output identity; None when absent."""


@dataclass(frozen=True)
class Upstream:
    """An input read from upstream jobs' outputs: where each element is stored, the source task and port type.

    A gathered input reads every ref, in order, as one list; any other reads its one ref.
    """

    refs: tuple[UpstreamRef, ...]
    task: str
    declared: str
    gathered: bool = False


@dataclass(frozen=True)
class Job:
    """One job: its reference, the call keyed for it, its input values by port and the anchors its outputs re-bind to.

    `call.inputs` holds the identity of each input; `values` holds the value of each graph input, and `upstream`
    names each input read from the result store, which `execute` resolves.
    """

    ref: JobRef
    call: Call
    values: Mapping[str, object]
    anchors: Anchors = ()
    upstream: Mapping[str, Upstream] = field(default_factory=dict)


def one_call(jobs: Sequence[Job]) -> Call:
    """Return the call every job shares but for its inputs.

    Raises:
        ValueError: two jobs differ in node, task, params, model or profile slices.
    """
    one_node(jobs)
    head = jobs[0].call
    for job in jobs[1:]:
        c = job.call
        if (c.task, c.params, c.model, dict(c.profile_slices)) != (
            head.task,
            head.params,
            head.model,
            dict(head.profile_slices),
        ):
            raise ValueError(f"the jobs of one execution are of one call; {job.ref} differs from {jobs[0].ref}")
    return head


def code_license(run: RunConfig, placement: Placement, call: Call) -> License | None:
    """Return the code license of the backend's env, once `run.licenses`, when given, accepts it and the model's.

    Raises:
        LicenseRefused: the policy refuses the env's or the model's license.
    """
    backend = placement.backend
    code = env_license(backend)
    if run.licenses is not None:
        run.licenses.check(f"the env of backend {backend.name!r}", code)
        if call.model is not None:
            run.licenses.check(f"model {call.model.ref}", call.model.license)
    return code


def stale(job: Job) -> FaultRecord | None:
    """Return the content fault of a job one of whose file-backed inputs no longer matches its stamp, else None."""
    for value in job.values.values():
        for group in value.groups.values() if isinstance(value, Recording) else ():
            source, stamp = getattr(group, "source", None), getattr(group, "source_stamp", None)
            if source is None or stamp is None:
                continue
            try:
                audio_io.check(stamp, source)
            except StaleSource as exc:
                return FaultRecord.from_exception(exc)
            except OSError as exc:
                return FaultRecord.from_exception(StaleSource(f"{source} cannot be read: {exc.strerror or exc}"))
    return None


def resolved(job: Job, read: Reader) -> tuple[Job, FaultRecord | None]:
    """Return `job` with each upstream input read through `read` and its constraints checked, or the job's failure.

    An upstream entry missing from the store, or stored in a retired wire format, is an infrastructure gap; an entry
    that lacks its port, holds a value not of its declared type, or gives inputs that break the task's constraints
    is a content fault of this job.

    Raises:
        InconsistentInputs, AmbiguousClock: a job with no upstream input breaks a constraint or carries one content
            on two clocks.
    """
    values = dict(job.values)
    try:
        for port, up in job.upstream.items():
            found = []
            for ref in up.refs:
                outputs = read(ref.key_digest, ref.outputs)
                if outputs is None:
                    return job, FaultRecord.gap_of(
                        "UpstreamMissing", f"input {port!r} reads {ref.outputs}, which the store does not hold"
                    )
                if ref.port not in outputs:
                    raise OutputTypeMismatch(f"input {port!r} reads port {ref.port!r}, which {ref.outputs} lacks")
                try:
                    found.append(read_port(up.task, ref.port, up.declared, outputs[ref.port]))
                except WireFormatRetired as exc:
                    return job, FaultRecord.gap_of("WireFormatRetired", f"input {port!r} reads {ref.outputs}: {exc}")
            values[port] = found if up.gathered else found[0]
        job.call.task.check_inputs(values)
        anchors = anchors_of(values.values())
    except (OutputTypeMismatch, InconsistentInputs, AmbiguousClock) as exc:
        if not job.upstream:
            raise
        return job, FaultRecord.from_exception(exc)
    return replace(job, values=values, anchors=anchors), None


def used(job: Job) -> tuple[tuple[str, str], ...]:
    """Return what a job's execution uses, by port: each graph input, and each upstream entry it reads."""
    found: list[tuple[str, str]] = []
    for port, identity in job.call.inputs.items():
        up = job.upstream.get(port)
        found += [(port, identity)] if up is None else [(port, ref.outputs) for ref in up.refs]
    return tuple(found)


def job_decoder(job: Job) -> str | None:
    """Return the decoder a job's file-backed inputs are read by, or None when none is file-backed."""
    return held_decoder(job.values.values())


def held_decoder(values: Iterable[object]) -> str | None:
    """Return the decoder the file-backed recordings among `values` are read by, or None when none is."""
    for value in values:
        if isinstance(value, Recording) and any(getattr(g, "source", None) is not None for g in value.groups.values()):
            return audio_io.DECODER
    return None


def job_context(  # noqa: PLR0913 - every input a context records
    placement: Placement,
    env: EnvHandle,
    expected: Attestation | None,
    profile: ExecutionProfile,
    job: Job,
    code_license: License | None,
    *,
    declared: bool = False,
) -> ExecutionContext:
    """Return the context a job runs in: its env, attestation, network, host token, decoder, model source, license.

    The numerics are this host's for the placement's device; `declared` gives the device kind and precision alone,
    for a context built where the job is submitted rather than run.
    """
    model = job.call.model
    numerics = (
        {"device_kind": placement.device.partition(":")[0], "precision": PRECISION}
        if declared
        else numerics_class(host_hardware(profile.sizing.device_query_s), placement.device, PRECISION)
    )
    return ExecutionContext(
        **env_fields(placement, env, expected, network_variables(profile.network)),
        platform=sysconfig.get_platform(),
        numerics=numerics,
        decoder=job_decoder(job),
        host=host_token(),
        protocol=f"wire@{WIRE_VERSION}",
        model_source=None if model is None else model.ref.source,
        env_license=code_license,
    )


def env_fields(
    placement: Placement, env: EnvHandle, expected: Attestation | None, declared: Mapping[str, str]
) -> dict[str, Any]:
    """Return the env's context fields: with the attestation of a server, the launch digest of any worker process.

    `declared` are the variables the profile sets on every process it starts.
    """
    backend = placement.backend
    if expected is not None:
        assert backend.server is not None
        variables = {**server_variables(backend.server.settings), **declared}
        return attested_fields(env, expected, server_launch(env, variables))
    fields = env.context_fields()
    if backend.env.kind == "host":
        return fields
    return {**fields, "driver": {**fields["driver"], "launch_env": launch(env.python, declared).digest}}
