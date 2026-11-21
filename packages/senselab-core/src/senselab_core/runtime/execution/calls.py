"""One task call, run as a one-node plan on a warm pool: `call`."""

from typing import Any

from senselab_core.kernel.declare import TaskSpec
from senselab_core.runtime.call import Result, RunConfig, TaskCall, bind_input, packaged
from senselab_core.runtime.execution.executor import Admission, FaultRecord
from senselab_core.runtime.execution.fragments import fragment_jobs, fragment_logical_keys, node_fragment
from senselab_core.runtime.execution.outputs import anchors_of, decoded, rebound
from senselab_core.runtime.execution.run import NotRun, Output, start
from senselab_core.runtime.planner import plan
from senselab_core.runtime.profiles import ExecutionProfile


class CallFailed(RuntimeError):
    """The backend failed the call's item; `failure` is its record, with its fault class."""

    def __init__(self, failure: FaultRecord) -> None:
        """Hold the failure and name it in the message."""
        super().__init__(f"{failure.type} ({failure.fault}): {failure.message}")
        self.failure = failure


def _checked(task: TaskCall) -> TaskSpec:
    spec = packaged().registry.spec(task.task)
    if task.task_version != spec.version:
        raise ValueError(f"{task.task!r} is declared at version {spec.version!r}, not {task.task_version!r}")
    if set(task.inputs) != set(spec.inputs):
        raise ValueError(f"{task.task!r} takes inputs {sorted(spec.inputs)}; the call gives {sorted(task.inputs)}")
    return spec


def call(
    task: TaskCall,
    *,
    config: RunConfig | None = None,
    profile: ExecutionProfile | None = None,
    session: Any = None,  # noqa: ANN401 - a PoolSession
) -> Result:
    """Run `task`: plan it as one node under `profile` and `config`, run it on a warm pool and return its result.

    Each input is bound once (`bind_input`) and its constraints checked. `session` is a `PoolSession` its caller owns,
    whose workers and models stay resident; with none, the run makes and closes its own. The result's outputs are
    decoded and re-bound to the inputs' clocks.

    Raises:
        UnknownTask, InconsistentInputs: the task is not declared; the inputs contradict a constraint it declares.
        AmbiguousClock: two recording inputs of one content carry different anchors.
        UnlistedSource, NoBackend, NoDevice, BackendParamsMismatch, LicenseRefused, LicenseConflict,
            MissingProfileSlice, ModelResolutionError: as `plan`.
        ValueError: the call names another task version or other inputs than the task's.
        CallFailed: the backend failed the item, or a hold or cancellation kept it from running.
    """
    spec = _checked(task)
    bound = {port: bind_input(port, spec.inputs[port], value)[:2] for port, value in task.inputs.items()}
    values = {port: value for port, (_, value) in bound.items()}
    spec.check_inputs(values)
    anchors = anchors_of(values.values())
    bound_call = TaskCall(
        task=task.task,
        task_version=task.task_version,
        inputs=values,
        params=task.params,
        model=task.model,
        backend_params=task.backend_params,
    )
    placed = plan(bound_call, config=config or RunConfig(), execution=profile or ExecutionProfile.load())
    held = dict(bound.values())
    run = start(placed, session, values=held)
    run.wait()
    [(_, given)] = list(run.results())
    if isinstance(given, FaultRecord):
        raise CallFailed(given)
    if isinstance(given, NotRun):
        raise CallFailed(FaultRecord.gap_of("NotRun", f"the call was {given.kind}"))
    assert isinstance(given, Output)
    fragment = node_fragment(placed, Admission(id="call", plan_hash=placed.hash))
    [key] = fragment_logical_keys(fragment, placed, held)
    placement = fragment_jobs(fragment, placed, held)[1]
    identity = next(iter(given.identities.values()))
    found = run.entry(key.digest(), identity)
    if found is None:
        raise CallFailed(FaultRecord.gap_of("MissingOutput", f"the result of {task.task!r} is not in the store"))
    outputs = rebound(decoded(spec, found[0]), anchors)
    return Result(outputs, identity, key, found[1], placement, run.reused(key.digest()))
