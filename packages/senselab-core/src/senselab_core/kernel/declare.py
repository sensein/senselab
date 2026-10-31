"""Task and backend declarations, the registry that holds them, and the logical key of a declared call.

Declarations only: no graph, planner or execution (ADR-001).
"""

import difflib
import types
import typing
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from senselab_core.kernel._base import Frozen, NonEmptyStr, StrMap
from senselab_core.kernel.identity import InputIdentity, LogicalKey, ProfileSliceIdentity, ResolvedModel
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


class NonCanonicalParams(TypeError):
    """A params model declares a field whose type has no canonical JSON form."""


_SCALARS: tuple[type, ...] = (type(None), bool, int, float, str)
_SEQUENCES: frozenset[object] = frozenset({list, tuple})
_MAPPINGS: frozenset[object] = frozenset({dict, Mapping, typing.Mapping})
_UNIONS: frozenset[object] = frozenset({Union, types.UnionType})


def check_canonical_params(model: type[BaseModel]) -> None:
    """Check that every field of `model` has a type whose values have one canonical JSON form.

    Allowed, recursively: `None`, `bool`, `int`, `float`, `str`; `Literal` of those; an `Enum` whose
    values are all `str`; `list[X]`, `tuple[X, ...]` and `tuple[X, Y, ...]`; `dict[str, X]` and
    `Mapping[str, X]`; `Optional` and unions of allowed types; a nested `TaskParams`; and
    `Annotated` constraints over any of these.

    Raises:
        NonCanonicalParams: naming the first field and type outside that set, such as `set`,
            `frozenset`, `Decimal`, `datetime`, `Path`, `bytes`, `Any` or `object`.
    """
    _check_model(model, model.__name__, set())


def _check_model(model: type[BaseModel], where: str, seen: set[type[BaseModel]]) -> None:
    if model in seen:
        return
    seen.add(model)
    for name, field in model.model_fields.items():
        _check_type(field.annotation, f"{where}.{name}", seen)


def _check_type(t: object, where: str, seen: set[type[BaseModel]]) -> None:
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin is Annotated:
        _check_type(args[0], where, seen)
    elif origin is Literal:
        bad = [a for a in args if type(a) not in _SCALARS]
        if bad:
            raise NonCanonicalParams(f"{where}: Literal values must be JSON scalars, got {bad!r}")
    elif origin in _UNIONS:
        for arg in args:
            _check_type(arg, where, seen)
    elif origin in _SEQUENCES and args:
        for arg in args:
            if arg is not Ellipsis:
                _check_type(arg, where, seen)
    elif origin in _MAPPINGS and len(args) == 2:
        if args[0] is not str:
            raise NonCanonicalParams(f"{where}: mapping keys must be str, got {args[0]!r}")
        _check_type(args[1], where, seen)
    elif origin is None and t in _SCALARS:
        return
    elif origin is None and isinstance(t, type) and issubclass(t, Enum):
        if not all(type(m.value) is str for m in t):
            raise NonCanonicalParams(f"{where}: an Enum param must have str values, {t.__qualname__} does not")
    elif origin is None and isinstance(t, type) and issubclass(t, TaskParams):
        _check_model(t, where, seen)
    else:
        raise NonCanonicalParams(f"{where}: {t!r} has no canonical JSON form; declare a canonical type")


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
            NonCanonicalParams: if its params model has a field with no canonical JSON form.
            DuplicateDeclaration: if its name or its signature is already declared.
        """
        check_canonical_params(spec.params)
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
    inputs: Mapping[str, InputIdentity],
    model: ResolvedModel | None = None,
    profile_slices: Mapping[str, ProfileSliceIdentity] | None = None,
) -> LogicalKey:
    """Return the logical key of calling `backend` for `task` (ADR-002).

    `params` is an instance of the task's params model; `inputs` maps each declared input port, and
    no other, to its `content:` or `output:` identity; `model` is the resolved model, which enters
    the key as its `key_identity()`; `profile_slices` maps each profile slice the call reads to its
    `profile:` identity. Batch formation and
    chunk grid are not set: `BackendSpec.output_affecting` does not reach the key yet.

    Raises:
        ValueError: if the backend implements another task, or params or inputs do not match the task.
        UnpinnedModel: if `model` is not pinned.
        pydantic.ValidationError: if an identity is not of its typed form.
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
        model=None if model is None else model.key_identity(),
        params=params.model_dump(mode="json"),
        inputs=inputs,
        profile_slices=profile_slices or {},
    )
