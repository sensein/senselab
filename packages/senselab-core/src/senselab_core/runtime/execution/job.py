"""The one job engine: jobs keyed, served from the result store or sent to a worker, stored and recorded (`execute`).

A key holds the batch formation of the request a job is sent in (`request_components`), which the job alone picks
(`batching.formation`). A stored result is served under the run's reuse policy unless its env's license is refused,
and a sampled cross-context hit is recomputed and compared instead. A failure is never stored.
"""

import sysconfig
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from senselab_worker import WIRE_VERSION
from senselab_worker.faults import FaultClass, classify, error_record

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.declare import InconsistentInputs
from senselab_core.kernel.identity import (
    Attestation,
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    OutputIdentity,
    ReusePolicy,
)
from senselab_core.kernel.license import License
from senselab_core.kernel.signals import Recording, StaleSource
from senselab_core.kernel.wire import WireFormatRetired
from senselab_core.runtime import audio_io, request_reports
from senselab_core.runtime.call import Result, RunConfig, env_license, host_hardware, packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.batching import BucketGrid, form_batches, keyed_formation, one_node
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, UpstreamRef
from senselab_core.runtime.execution.outputs import (
    AmbiguousClock,
    Anchors,
    OutputTypeMismatch,
    anchors_of,
    as_stored,
    as_wire,
    decoded,
    read_port,
    rebound,
)
from senselab_core.runtime.execution.records import (
    execution_records,
    plan_link,
    reuse_attributes,
    unlicensed,
    verify,
)
from senselab_core.runtime.execution.records import verify_attributes as verify_settings
from senselab_core.runtime.execution.workers import PRECISION, WorkerSet, service_of
from senselab_core.runtime.hardware import host_token, numerics_class
from senselab_core.runtime.keys import Call, Placement, build_key, request_components
from senselab_core.runtime.launch import launch
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.network import network_variables
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.servable import Servable
from senselab_core.runtime.store import ResultStore, StoredResult
from senselab_core.runtime.store import output_identity as store_output_identity
from senselab_core.runtime.vllm_server import attested_fields, server_launch, server_variables
from senselab_core.runtime.vocabulary import core_vocabulary


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


@dataclass(frozen=True)
class Completed:
    """A job that gave a result, executed or served."""

    ref: JobRef
    result: Result


@dataclass(frozen=True)
class Failed:
    """A job that failed, with its key's digest; nothing of it is stored."""

    ref: JobRef
    key_digest: str
    failure: FaultRecord


JobOutcome = Completed | Failed
"""What one job gave."""


def splits(exc: BaseException) -> bool:
    """Whether `exc`, raised by a request to a loaded worker, is a resource fault that splits the request.

    A lease wait (`LeaseTimeout`) is a `TimeoutError` and so a resource class, but it is nobody's request failing.
    """
    return classify(exc) is FaultClass.RESOURCE and not isinstance(exc, LeaseTimeout)


def ran_out(outcomes: Sequence[JobOutcome]) -> bool:
    """Whether `outcomes` are those of a request of several jobs that every one ran out: each a resource fault."""
    return len(outcomes) > 1 and all(
        isinstance(o, Failed) and o.failure.fault == FaultClass.RESOURCE.value for o in outcomes
    )


@dataclass(frozen=True)
class _Pending:
    index: int
    job: Job
    context: ExecutionContext
    hit: StoredResult | None


def execute(  # noqa: PLR0913 - the jobs and their placement, then keyword-only run inputs
    jobs: Sequence[Job],
    placement: Placement,
    *,
    store: ResultStore,
    run: RunConfig,
    profile: ExecutionProfile,
    workers: WorkerSet,
    bisect_resource: bool = False,
) -> list[JobOutcome]:
    """Run `jobs`, all of one call but their inputs, under `placement`, and return one outcome per job, in order.

    Each job is keyed (`build_key`) under the formation of the request it is sent in and served from `store` under
    `run.reuse`, unless its env's license is refused or it is a sampled cross-context hit (`run.verify`), which is
    recomputed and compared before anything is stored. An upstream input is read from `store` by its output
    identity, and the inputs are checked against the task's constraints. A file-backed input is checked against its
    stamp (`audio_io.check`) before its job is keyed; a stale or missing one fails the job, as does an upstream
    recording whose length was not recorded (`LengthUnrecorded`, a gap). The jobs not served are
    sent in batches of at most `placement.batch_size` items (`form_batches`) through `workers`; each item's outputs
    are stored, and each execution writes one activity associated with its agent (the model's source included),
    naming its job's node and plan.
    An item the backend fails gives `Failed`. With `bisect_resource`, a request of several jobs each of which
    gave a resource fault, or which raised one, is sent again as two requests of its first and last jobs, each split
    in turn, so that the jobs that fit complete and a job that does not fit alone is the one that fails; without it
    the request's outcomes are returned as they came, for a caller that splits them itself (the pool worker).

    Raises:
        ValueError: the jobs are not of one call or of one node; a server backend has no model.
        InconsistentInputs: a job's inputs contradict a constraint the task declares.
        AmbiguousClock: two recording inputs of one job's content carry different anchors.
        LicenseRefused: `run.licenses` refuses the license of the backend's env or of the model.
        LeaseTimeout, LeaseUnavailable: a server backend's start lease is busy or cannot be read.
        WorkerLoadFailed: the backend's worker or server did not start or load.
        OutputTypeMismatch: an output of a kernel-typed port is not a wire form of its declared type.
        WireFormatRetired: a stored output is in a retired wire format.
        WorkerFatal, WorkerLost, ProtocolError: as the worker reports them.
    """
    if not jobs:
        return []
    call = _one_call(jobs)
    backend = placement.backend
    code_license = _code_license(run, placement, call)
    env = packaged().open_env(backend, profile)
    expected = service_of(backend, call.model)
    jobs = [replace(j, call=replace(j.call, service=expected)) for j in jobs]
    grid = BucketGrid.load_default()
    outcomes: dict[int, JobOutcome] = {}
    pending: list[_Pending] = []
    for index, given in enumerate(jobs):
        job, unread = _resolved(given, store)
        picked, unrecorded = keyed_formation(job, backend, grid)
        keyed = replace(placement, **request_components(backend, picked))
        context = job_context(keyed, env, expected, profile, job, code_license)
        gap = None if unrecorded is None else _upstream_gap(type(unrecorded).__name__, str(unrecorded))
        stale = unread or gap or _stale(job)
        key = build_key(job.call, keyed, context)
        if stale is not None:
            outcomes[index] = Failed(job.ref, key.digest(), stale)
            continue
        served, hit = _serve(store, key, context, run, job, keyed)
        if served is not None:
            outcomes[index] = Completed(job.ref, served)
        else:
            pending.append(_Pending(index, job, context, hit))
    by_job = {id(p.job): p for p in pending}
    for batch in form_batches([p.job for p in pending], backend, batch_size=placement.batch_size, grid=grid):
        chunk = [by_job[id(j)] for j in batch.jobs]
        sent = replace(placement, **request_components(backend, batch.formation))
        sending = _Sending(sent, env, profile, workers, expected, store, run, bisect_resource)
        for item, outcome in zip(chunk, _sent_in_halves(chunk, sending), strict=True):
            outcomes[item.index] = outcome
    return [outcomes[i] for i in range(len(jobs))]


class _Raised(Exception):
    """What `_send` raises for an exception its request to a loaded worker raised, as opposed to checkout or start."""

    def __init__(self, cause: Exception) -> None:
        super().__init__()
        self.cause = cause


@dataclass(frozen=True)
class _Sending:
    """What sending a request of one batch needs: its placement, env, profile, workers, attestation, store and run."""

    placement: Placement
    env: EnvHandle
    profile: ExecutionProfile
    workers: WorkerSet
    expected: Attestation | None
    store: ResultStore
    run: RunConfig
    bisect_resource: bool


def _sent_in_halves(chunk: Sequence[_Pending], sending: _Sending) -> list[JobOutcome]:
    """Return one outcome per item of `chunk`, sent as one request; with `bisect_resource`, one that ran out is split.

    A job alone whose request raised a splittable resource fault (`splits`) is that job's failure. What checkout,
    start or load raised is never split.
    """
    began = datetime.now(timezone.utc)
    jobs = [p.job for p in chunk]
    try:
        results = _send(jobs, sending.placement, sending.env, sending.profile, sending.workers, sending.expected)
    except _Raised as raised:
        exc = raised.cause
        if not sending.bisect_resource or not splits(exc):
            raise exc from None
        if len(chunk) > 1:
            return _halves(chunk, sending)
        key = build_key(chunk[0].job.call, sending.placement, chunk[0].context)
        return [Failed(chunk[0].job.ref, key.digest(), FaultRecord.of(error_record(exc)))]
    outcomes = [
        _record(item, result, sending.placement, sending.store, sending.run, sending.profile, began)
        for item, result in zip(chunk, results, strict=True)
    ]
    return _halves(chunk, sending) if sending.bisect_resource and ran_out(outcomes) else outcomes


def _halves(chunk: Sequence[_Pending], sending: _Sending) -> list[JobOutcome]:
    """Return the outcomes of `chunk` sent as two requests, each through `_sent_in_halves`."""
    half = len(chunk) // 2
    return _sent_in_halves(chunk[:half], sending) + _sent_in_halves(chunk[half:], sending)


def _one_call(jobs: Sequence[Job]) -> Call:
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


def _code_license(run: RunConfig, placement: Placement, call: Call) -> License | None:
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


def _upstream_gap(kind: str, message: str) -> FaultRecord:
    return FaultRecord(fault="infrastructure", type=kind, message=message, gap=True, charged=False)


def _resolved(job: Job, store: ResultStore) -> tuple[Job, FaultRecord | None]:
    """Return `job` with each upstream input read from `store` and its constraints checked, or the job's failure.

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
            read = []
            for ref in up.refs:
                outputs = store.outputs_of(ref.key_digest, ref.outputs)
                if outputs is None:
                    return job, _upstream_gap(
                        "UpstreamMissing", f"input {port!r} reads {ref.outputs}, which the store does not hold"
                    )
                if ref.port not in outputs:
                    raise OutputTypeMismatch(f"input {port!r} reads port {ref.port!r}, which {ref.outputs} lacks")
                try:
                    read.append(read_port(up.task, ref.port, up.declared, outputs[ref.port]))
                except WireFormatRetired as exc:
                    if not job.upstream:
                        raise
                    return job, _upstream_gap("WireFormatRetired", f"input {port!r} reads {ref.outputs}: {exc}")
            values[port] = read if up.gathered else read[0]
        job.call.task.check_inputs(values)
        anchors = anchors_of(values.values())
    except (OutputTypeMismatch, InconsistentInputs, AmbiguousClock) as exc:
        if not job.upstream:
            raise
        return job, FaultRecord(fault="content", type=type(exc).__name__, message=str(exc), gap=False, charged=True)
    return replace(job, values=values, anchors=anchors), None


def _stale(job: Job) -> FaultRecord | None:
    """Return the content fault of a job one of whose file-backed inputs no longer matches its stamp, else None."""
    for value in job.values.values():
        for group in value.groups.values() if isinstance(value, Recording) else ():
            source, stamp = getattr(group, "source", None), getattr(group, "source_stamp", None)
            if source is None or stamp is None:
                continue
            try:
                audio_io.check(stamp, source)
            except StaleSource as exc:
                return FaultRecord.of(error_record(exc))
            except OSError as exc:
                return FaultRecord.of(error_record(StaleSource(f"{source} cannot be read: {exc.strerror or exc}")))
    return None


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
) -> ExecutionContext:
    """Return the context a job runs in: its env, attestation, network, host token, decoder, model source, license."""
    model = job.call.model
    return ExecutionContext(
        **env_fields(placement, env, expected, network_variables(profile.network)),
        platform=sysconfig.get_platform(),
        numerics=numerics_class(host_hardware(), placement.device, PRECISION),
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


def _serve(  # noqa: PLR0913 - the job's store, key, context and run, and what serving needs
    store: ResultStore,
    key: LogicalKey,
    context: ExecutionContext,
    run: RunConfig,
    job: Job,
    placement: Placement,
) -> tuple[Result | None, StoredResult | None]:
    """Return the job's result served from `store`, recording the reuse, or None and the sampled hit to verify."""
    looked = datetime.now(timezone.utc)
    hit, verdict = store.fetch(key, context, run)
    if hit is None or verdict is Servable.REFUSE:
        return None, None
    if verdict is Servable.VERIFY:
        return None, hit
    store.record_reuse(key, context, hit, looked, reuse_attributes(run, hit), _linked(job))
    return served(job, hit, key, placement), None


def _linked(job: Job) -> Callable[[kprov.Record], kprov.Association]:
    """Return the association of a reuse activity of `job` with an agent (`plan_link`)."""
    return lambda agent: plan_link(agent, job.ref)


def served(job: Job, hit: StoredResult, key: LogicalKey, placement: Placement) -> Result:
    """Return a stored result as the job's result, its outputs decoded and re-bound to the job's anchors."""
    outputs = rebound(decoded(job.call.task, hit.outputs), job.anchors)
    return Result(outputs, hit.output_identity, key, hit.context, placement, hit.policy_used)


def lost_race(  # noqa: PLR0913 - the job's store, key, context and run, and what serving needs
    store: ResultStore,
    key: LogicalKey,
    context: ExecutionContext,
    run: RunConfig,
    attributes: Mapping[str, object],
    job: Job,
    placement: Placement,
) -> Result | None:
    """Serve the entry another writer committed first, recording the reuse; None when its license refuses it.

    Raises:
        RuntimeError: the committed entry cannot be read back.
    """
    committed, verdict = store.fetch(key, context, replace(run, reuse=ReusePolicy.EXACT))
    if verdict is Servable.REFUSE:
        return None
    if committed is None:
        raise RuntimeError(f"the entry another writer committed for {key.task!r} could not be read back")
    store.record_reuse(
        key, context, committed, datetime.now(timezone.utc), {**attributes, **unlicensed(committed)}, _linked(job)
    )
    return served(job, committed, key, placement)


def _send(  # noqa: PLR0913 - the request's jobs and everything placing it
    jobs: Sequence[Job],
    placement: Placement,
    env: EnvHandle,
    profile: ExecutionProfile,
    workers: WorkerSet,
    expected: Attestation | None,
) -> list[dict[str, Any]]:
    """Send `jobs` as one request to the backend's worker, with the task params and `backend_params` beside them.

    A host worker is handed kernel values as they are; an isolated or server one, their wire forms. The request is
    reported (`request_reports`) tagged with its jobs' refs.
    """
    backend, call = placement.backend, jobs[0].call
    params = {**call.params.model_dump(mode="json"), "backend_params": placement.backend_params.model_dump(mode="json")}
    host = backend.env.kind == "host"
    items = [{"id": str(i), "inputs": dict(j.values) if host else as_wire(j.values)} for i, j in enumerate(jobs)]
    held = workers.checkout(backend, env, call.model, device=placement.device, profile=profile, expected=expected)
    with held as worker, request_reports.request(tuple(j.ref for j in jobs)):
        try:
            return worker.run(items, params)
        except Exception as exc:
            raise _Raised(exc) from exc


def _record(  # noqa: PLR0913 - one sent item and where its result goes
    item: _Pending,
    result: Mapping[str, Any],
    placement: Placement,
    store: ResultStore,
    run: RunConfig,
    profile: ExecutionProfile,
    started: datetime,
) -> JobOutcome:
    """Return a sent item's outcome: its outputs stored and recorded, a lost race served, or its failure."""
    job, context = item.job, item.context
    key = build_key(job.call, placement, context)
    if "error" in result:
        return Failed(job.ref, key.digest(), FaultRecord.of(result["error"]))
    read, outputs = as_stored(job.call.task, as_wire(result["outputs"]), job.anchors)
    attributes = verify_settings(run)
    if item.hit is not None:
        attributes["senselab:verifies"] = verify(store, key, item.hit, context, outputs, run.thresholds)
    output_identity = store.put(key, outputs, context)
    if output_identity != store_output_identity(outputs):
        lost = lost_race(store, key, context, run, attributes, job, placement)
        if lost is not None:
            return Completed(job.ref, lost)
        output_identity = store_output_identity(outputs)
    _provenance(store, key, context, profile, _used(job), output_identity, started, attributes, job.ref)
    return Completed(job.ref, Result(read, output_identity, key, context, placement, None))


def _provenance(  # noqa: PLR0913 - one execution's records
    store: ResultStore,
    key: LogicalKey,
    context: ExecutionContext,
    profile: ExecutionProfile,
    used: Sequence[tuple[str, InputIdentity]],
    output: OutputIdentity,
    started: datetime,
    attributes: Mapping[str, object],
    ref: JobRef,
) -> None:
    store.provenance.add(
        *(kprov.entity(i, core_vocabulary()) for i in dict.fromkeys(i for _, i in used)),
        *execution_records(key, context, profile, used, output, started, attributes, ref),
    )


def _used(job: Job) -> list[tuple[str, InputIdentity]]:
    """Return what a job's execution used, by port: each graph input, and each upstream entry it read."""
    used: list[tuple[str, InputIdentity]] = []
    for port, identity in job.call.inputs.items():
        up = job.upstream.get(port)
        used += [(port, identity)] if up is None else [(port, ref.outputs) for ref in up.refs]
    return used
