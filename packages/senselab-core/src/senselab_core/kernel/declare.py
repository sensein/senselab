"""Task and backend declarations, the registry that holds them, and the logical key of a declared call.

Declarations only: no graph, planner or execution (ADR-001).
"""

import difflib
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr, StrMap
from senselab_core.kernel.identity import LogicalKey
from senselab_core.kernel.license import License

TaskKind = Literal["measure", "transform", "decide", "fit"]

VERDICT_TYPES: frozenset[str] = frozenset({"Verdict"})
"""Output types that carry a decision; only `decide` tasks may declare them (P-20)."""

OUTPUT_AFFECTING: frozenset[str] = frozenset({"batch_formation", "chunk_grid"})
"""Logical-key components a backend may declare output-affecting (ADR-002)."""


class DuplicateDeclaration(ValueError):
    """A declaration repeats an existing one; `existing` is that one."""

    def __init__(self, message: str, existing: "TaskSpec | BackendSpec") -> None:
        """Hold the existing declaration."""
        super().__init__(message)
        self.existing = existing


class UnknownTask(LookupError):
    """A task name that no declaration in the registry carries."""


class TaskParams(BaseModel):
    """Base for a task's params: immutable, unknown keys refused with the closest known key suggested."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _refuse_unknown(cls, data: object) -> object:
        if isinstance(data, Mapping):
            known = sorted(cls.model_fields)
            for key in sorted(set(data) - set(known), key=str):
                close = difflib.get_close_matches(str(key), known, n=1)
                hint = f"; did you mean {close[0]!r}?" if close else f"; known params: {known}"
                raise ValueError(f"unknown param {key!r} for {cls.__name__}{hint}")
        return data


class TaskSpec(Frozen):
    """A declared task: its name and version, kind, typed ports and params model.

    `inputs` and `outputs` map port names to type names.
    """

    name: NonEmptyStr
    version: NonEmptyStr
    kind: TaskKind
    inputs: StrMap
    outputs: StrMap
    params: type[BaseModel]

    def signature(self) -> tuple[str, frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
        """Return the declared operation: kind, input ports and output ports with their types."""
        return self.kind, frozenset(self.inputs.items()), frozenset(self.outputs.items())


class BackendSpec(Frozen):
    """A declared implementation of a task: its stack, environment, devices and license record.

    `output_affecting` names the components of `OUTPUT_AFFECTING` that change this backend's outputs
    and so enter its logical key.
    """

    task: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    stack: NonEmptyStr
    env: NonEmptyStr
    devices: tuple[NonEmptyStr, ...] = Field(min_length=1)
    license: License
    output_affecting: frozenset[str] = frozenset()

    @field_validator("output_affecting")
    @classmethod
    def _check_output_affecting(cls, v: frozenset[str]) -> frozenset[str]:
        if not v <= OUTPUT_AFFECTING:
            raise ValueError(f"output_affecting may name only {sorted(OUTPUT_AFFECTING)}, got {sorted(v)}")
        return v


class Registry:
    """Declared tasks and their backends, in declaration order.

    Refuses a repeated task name, a repeated backend name within a task, a backend of an undeclared
    task, and a second task with the same kind and ports (P-37).
    """

    def __init__(self) -> None:
        """Start empty."""
        self._tasks: dict[str, TaskSpec] = {}
        self._backends: dict[str, dict[str, BackendSpec]] = {}

    def add_task(self, spec: TaskSpec) -> None:
        """Declare `spec`.

        Raises:
            DuplicateDeclaration: if its name or its signature is already declared.
        """
        if spec.name in self._tasks:
            raise DuplicateDeclaration(f"task {spec.name!r} is already declared", self._tasks[spec.name])
        for existing in self._tasks.values():
            if existing.signature() == spec.signature():
                raise DuplicateDeclaration(
                    f"task {spec.name!r} declares the same kind and ports as {existing.name!r}; "
                    "make it a param, strategy or backend of that task",
                    existing,
                )
        self._tasks[spec.name] = spec
        self._backends[spec.name] = {}

    def add_backend(self, spec: BackendSpec) -> None:
        """Declare `spec` for its task.

        Raises:
            UnknownTask: if its task is not declared.
            DuplicateDeclaration: if the task already has a backend of that name.
        """
        backends = self._task_backends(spec.task)
        if spec.name in backends:
            raise DuplicateDeclaration(
                f"backend {spec.name!r} of task {spec.task!r} is already declared", backends[spec.name]
            )
        backends[spec.name] = spec

    def tasks(self) -> list[TaskSpec]:
        """Return the declared tasks."""
        return list(self._tasks.values())

    def backends(self, task: str) -> list[BackendSpec]:
        """Return the backends of `task`, or raise `UnknownTask`."""
        return list(self._task_backends(task).values())

    def _task_backends(self, task: str) -> dict[str, BackendSpec]:
        try:
            return self._backends[task]
        except KeyError:
            raise UnknownTask(f"no task {task!r} is declared; declared: {sorted(self._tasks)}") from None


REGISTRY = Registry()
"""The process-wide registry; modules declaring tasks and backends add to it on import."""


def key_for(
    task: TaskSpec,
    backend: BackendSpec,
    *,
    params: BaseModel,
    inputs: Mapping[str, str],
    model: str | None = None,
    profile_slices: Mapping[str, str] | None = None,
) -> LogicalKey:
    """Return the logical key of calling `backend` for `task` (ADR-002).

    `params` is an instance of the task's params model; `inputs` maps each declared input port, and
    no other, to its content or upstream output identity; `model` is the weight identity;
    `profile_slices` maps each profile slice the call reads to its identity. Batch formation and
    chunk grid are not set: `BackendSpec.output_affecting` does not reach the key yet.

    Raises:
        ValueError: if the backend implements another task, or params or inputs do not match the task.
    """
    if backend.task != task.name:
        raise ValueError(f"backend {backend.name!r} implements {backend.task!r}, not {task.name!r}")
    if type(params) is not task.params:
        raise ValueError(f"params of {task.name!r} are {task.params.__name__}, got {type(params).__name__}")
    if set(inputs) != set(task.inputs):
        raise ValueError(f"{task.name!r} takes inputs {sorted(task.inputs)}, got {sorted(inputs)}")
    return LogicalKey(
        task=task.name,
        task_version=task.version,
        backend=backend.name,
        backend_version=backend.version,
        model=model,
        params=params.model_dump(mode="json"),
        inputs=inputs,
        profile_slices=profile_slices or {},
    )
