"""The facade: `plan`, `run`, `resume`, `use`, `Dataset` and the `.map` of a configured task, over the runtime."""

import contextvars
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypedDict, Unpack

from senselab_core.kernel.dataset import Dataset as DatasetProtocol
from senselab_core.kernel.graph import Graph
from senselab_core.runtime.bids import bids_dataset
from senselab_core.runtime.call import RunConfig, TaskCall
from senselab_core.runtime.datasets import from_records
from senselab_core.runtime.execution.executor import Executor
from senselab_core.runtime.execution.run import ResultSet, Run
from senselab_core.runtime.hardware import HardwareInfo
from senselab_core.runtime.planner import Plan, one_node_graph
from senselab_core.runtime.planner import plan as plan_runtime
from senselab_core.runtime.planner import resume as resume_runtime
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

from senselab._api.call import ConfiguredTask, reraise

__all__ = ["Dataset", "PlanKwargs", "Session", "map_over", "plan", "resume", "run", "session", "use"]


@dataclass(frozen=True)
class Session:
    """The defaults of `plan` and `run`: the execution profile, the run config, the executor and its result store."""

    execution: ExecutionProfile | None = None
    config: RunConfig | None = None
    executor: Executor | None = None
    store: ResultStore | None = None


_SESSION: contextvars.ContextVar[Session] = contextvars.ContextVar("senselab_session", default=Session())


def session() -> Session:
    """Return the session defaults in force in this context."""
    return _SESSION.get()


class use:  # noqa: N801 - a context manager used as a function
    """Set the session defaults for the context it is used in; as a `with` block, restore them on exit.

    `execution` is a profile or the name of a packaged one; a default not given keeps the one in force. An
    `executor` is given with the result `store` it writes to, which runs read their results from.

    Raises:
        ValueError: the session would have an executor and no store.
    """

    def __init__(
        self,
        *,
        execution: ExecutionProfile | str | None = None,
        config: RunConfig | None = None,
        executor: Executor | None = None,
        store: ResultStore | None = None,
    ) -> None:
        """Set the given defaults now."""
        current = session()
        profile = ExecutionProfile.load(execution) if isinstance(execution, str) else execution
        self.session = Session(
            execution=profile if profile is not None else current.execution,
            config=config if config is not None else current.config,
            executor=executor if executor is not None else current.executor,
            store=store if store is not None else None if executor is not None else current.store,
        )
        if self.session.executor is not None and self.session.store is None:
            raise ValueError("an executor is used with the result store it writes to; give `store`")
        self._token = _SESSION.set(self.session)

    def __enter__(self) -> Session:
        """Return the session in force."""
        return self.session

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        """Restore the defaults in force before `use`."""
        _SESSION.reset(self._token)


class Dataset:
    """The dataset constructors: `Dataset.bids(root, ...)` and `Dataset.from_records(rows, ...)`."""

    bids = staticmethod(bids_dataset)
    from_records = staticmethod(from_records)

    def __init__(self) -> None:
        """Refuse an instance: this is a namespace."""
        raise TypeError("Dataset is a namespace of constructors; call Dataset.bids or Dataset.from_records")


def _target(target: Graph | TaskCall | ConfiguredTask) -> Graph | TaskCall:
    return target._task_call() if isinstance(target, ConfiguredTask) else target  # noqa: SLF001


class PlanKwargs(TypedDict, total=False):
    """The keyword arguments of `plan` and `run`, those of `senselab_core.runtime.planner.plan`."""

    config: RunConfig | None
    dataset_profile: Mapping[str, Any] | None
    execution: ExecutionProfile | None
    bind: Mapping[str, str] | None
    inputs: Mapping[str, object] | None
    hardware: HardwareInfo | None


def plan(
    target: Graph | TaskCall | ConfiguredTask, dataset: DatasetProtocol | None = None, **plan_kwargs: Unpack[PlanKwargs]
) -> Plan:
    """Return `target` over `dataset` planned under the session's execution profile and run config.

    `plan_kwargs` are those of `senselab_core.runtime.planner.plan` (`bind`, `inputs`, `execution`, `config`, ...) and
    override the session's. A configured task is planned as its call, over the dataset's records.
    """
    current = session()
    plan_kwargs.setdefault("execution", current.execution)
    plan_kwargs.setdefault("config", current.config)
    return plan_runtime(_target(target), dataset, **plan_kwargs)


def run(
    target: Graph | TaskCall | ConfiguredTask, dataset: DatasetProtocol | None = None, **plan_kwargs: Unpack[PlanKwargs]
) -> ResultSet:
    """Plan `target` over `dataset` (`plan`), run it on the session's executor, wait and return its results."""
    current = session()
    started = plan(target, dataset, **plan_kwargs).run(current.executor, store=current.store)
    started.wait()
    return started.results()


def resume(target: Plan | Run | str) -> ResultSet:
    """Resume a run by its plan hash (`senselab_core.runtime.planner.resume`), wait and return its results.

    The run is read from its directory: its plan, its input sources and its result store, so jobs the store serves
    are not recomputed. It is placed afresh under the session's execution profile and run on the session's executor
    and store.

    Raises:
        NoRun, ValueError, InputChanged, InputsUnavailable, ExecutorUnavailable: as `resume`; a run whose inputs were
            held in memory raises `InputsUnavailable`.
    """
    plan_hash = target if isinstance(target, str) else target.run_dir.name if isinstance(target, Run) else target.hash
    current = session()
    started = resume_runtime(
        plan_hash, execution=current.execution, store=current.store, executor=current.executor, config=current.config
    )
    started.wait()
    return started.results()


def map_over(task: ConfiguredTask, dataset: DatasetProtocol, bind: Mapping[str, str] | None) -> ResultSet:
    """Run `task` on every record of `dataset`, each input port bound from the record input of its name or `bind`'s."""
    graph = one_node_graph(task._task_call(), task.registry, every_port=True)  # noqa: SLF001
    try:
        return run(graph, dataset, bind=dict(bind or {}))
    except Exception as error:
        reraise(error)
