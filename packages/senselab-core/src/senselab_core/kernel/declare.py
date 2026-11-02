"""Task and backend declarations, the registry that holds them, their JSON Schema, and the logical key of a call.

Declarations only: no graph, planner or execution (ADR-001).
"""

import difflib
import re
import types
import typing
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Annotated, Any, Literal, TypeVar, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.json_schema import GenerateJsonSchema

import senselab_core.kernel.results  # noqa: F401 - importing it defines the result types
import senselab_core.kernel.time  # noqa: F401 - importing it defines the time types
from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, StrMap
from senselab_core.kernel.formats import Formatted
from senselab_core.kernel.identity import InputIdentity, LogicalKey, Params, ProfileSliceIdentity, ResolvedModel
from senselab_core.kernel.signals import Recording, RegularSeries, Text

TaskKind = Literal["transform", "fit"]


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


class InconsistentInputs(ValueError):
    """A call's inputs or params contradict a constraint its task declares."""


class DeclaredParams(BaseModel):
    """Base for declared params: immutable, unknown keys refused with the closest known key suggested."""

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


class TaskParams(DeclaredParams):
    """Base for a task's params."""


class BackendParams(DeclaredParams):
    """Base for a backend's params, validated against the chosen backend only."""


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
    `Mapping[str, X]`; `Optional` and unions of allowed types; nested `DeclaredParams`; and
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
    origin = typing.get_origin(t)
    if origin is None:
        _check_plain(t, where, seen)
    else:
        _check_generic(origin, typing.get_args(t), t, where, seen)


def _check_generic(origin: object, args: tuple[object, ...], t: object, where: str, seen: set[type[BaseModel]]) -> None:
    if origin is Annotated:
        _check_type(args[0], where, seen)
    elif origin is Literal:
        bad = [a for a in args if type(a) not in _SCALARS]
        if bad:
            raise NonCanonicalParams(f"{where}: Literal values must be JSON scalars, got {bad!r}")
    elif origin in _UNIONS or (origin in _SEQUENCES and args):
        for arg in args:
            if arg is not Ellipsis:
                _check_type(arg, where, seen)
    elif origin in _MAPPINGS and len(args) == 2:  # noqa: PLR2004 - a mapping type has a key and a value parameter
        if args[0] is not str:
            raise NonCanonicalParams(f"{where}: mapping keys must be str, got {args[0]!r}")
        _check_type(args[1], where, seen)
    else:
        raise NonCanonicalParams(f"{where}: {t!r} has no canonical JSON form; declare a canonical type")


def _check_plain(t: object, where: str, seen: set[type[BaseModel]]) -> None:
    if t in _SCALARS:
        return
    if isinstance(t, type) and issubclass(t, Enum):
        if not all(type(m.value) is str for m in t):
            raise NonCanonicalParams(f"{where}: an Enum param must have str values, {t.__qualname__} does not")
    elif isinstance(t, type) and issubclass(t, DeclaredParams):
        _check_model(t, where, seen)
    else:
        raise NonCanonicalParams(f"{where}: {t!r} has no canonical JSON form; declare a canonical type")


class InputsAgree(Frozen):
    """The two input ports `ports` hold values with equal `property`."""

    kind: Literal["inputs_agree"] = "inputs_agree"
    ports: tuple[str, str]
    property: Literal["clock_id", "text", "sampling_rate", "channels", "frames"]


class ParamsOrdered(Frozen):
    """The param `low` is at most the param `high`, whenever both are given."""

    kind: Literal["params_ordered"] = "params_ordered"
    low: NonEmptyStr
    high: NonEmptyStr


Constraint = Annotated[InputsAgree | ParamsOrdered, Field(discriminator="kind")]
"""A relation between a task's inputs or between its params, checked before a call is keyed."""


class TaskSpec(Frozen):
    """A declared task: its name and version, kind, typed ports, params model and constraints.

    `inputs` and `outputs` map port names to kernel type names. `constraints` name only declared ports and params.
    """

    name: NonEmptyStr
    version: NonEmptyStr
    kind: TaskKind
    inputs: StrMap
    outputs: StrMap
    params: type[TaskParams]
    constraints: tuple[Constraint, ...] = ()

    @model_validator(mode="after")
    def _constraints_name_declared_things(self) -> "TaskSpec":
        for c in self.constraints:
            names, declared = (
                (c.ports, self.inputs) if isinstance(c, InputsAgree) else ((c.low, c.high), self.params.model_fields)
            )
            unknown = [n for n in names if n not in declared]
            if unknown:
                raise ValueError(f"{self.name}: constraint {c.kind} names {unknown}, which the task does not declare")
        return self

    def signature(self) -> tuple[str, frozenset[tuple[str, str]], frozenset[tuple[str, str]]]:
        """Return the declared operation: kind, input ports and output ports with their types."""
        return self.kind, frozenset(self.inputs.items()), frozenset(self.outputs.items())

    def check_inputs(self, values: Mapping[str, object]) -> None:
        """Check each `InputsAgree` constraint against the input values by port.

        Raises:
            InconsistentInputs: two inputs differ in a property the task declares they share.
        """
        for c in self.constraints:
            if isinstance(c, InputsAgree):
                first, second = (getattr(values[p], c.property) for p in c.ports)
                if first != second:
                    raise InconsistentInputs(
                        f"inputs {c.ports[0]!r} and {c.ports[1]!r} of {self.name!r} differ in {c.property}: "
                        f"{first!r} and {second!r}"
                    )

    def check_params(self, params: BaseModel) -> None:
        """Check each `ParamsOrdered` constraint against the validated params.

        Raises:
            InconsistentInputs: a `low` param exceeds its `high` param.
        """
        for c in self.constraints:
            if isinstance(c, ParamsOrdered):
                low, high = getattr(params, c.low), getattr(params, c.high)
                if low is not None and high is not None and low > high:
                    raise InconsistentInputs(f"param {c.low!r} ({low}) exceeds param {c.high!r} ({high})")


EnvName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
"""The name of an env and its lock: lower-case letters, digits and hyphens."""


class Env(Frozen):
    """Where a backend runs: `host` in the senselab venv, `isolated` in its own locked env, `server` in an engine's env.

    Build with `Env.host()`, `Env.isolated(name)` or `Env.server(name)`; `name` names the env's lock.
    """

    kind: Literal["host", "isolated", "server"]
    name: EnvName | None = None

    @model_validator(mode="after")
    def _check_name(self) -> "Env":
        if (self.kind == "host") != (self.name is None):
            raise ValueError(f"an {self.kind} env {'takes no' if self.kind == 'host' else 'needs a'} name")
        return self

    @classmethod
    def host(cls) -> "Env":
        """Return the in-process env, the locked senselab venv."""
        return cls(kind="host")

    @classmethod
    def isolated(cls, name: str) -> "Env":
        """Return the locked env `name` that a worker subprocess runs in."""
        return cls(kind="isolated", name=name)

    @classmethod
    def server(cls, name: str) -> "Env":
        """Return the env `name` of a long-lived engine process."""
        return cls(kind="server", name=name)


class ServerConfig(Frozen):
    """How a server backend's engine runs.

    `settings` are the keys `vllm_server.server_args` reads; `startup_timeout_s` is the seconds the engine has to
    become healthy.
    """

    settings: Params
    startup_timeout_s: float = Field(gt=0)


TunableValue = bool | int | float | str
Tunables = Annotated[
    Mapping[str, tuple[TunableValue, ...]],
    AfterValidator(lambda v: FrozenMapping({k: _non_empty(k, x) for k, x in v.items()})),
    PlainSerializer(dict),
]
"""A read-only mapping of tunable name to its declared values, at least one each."""


def _non_empty(name: str, values: tuple[TunableValue, ...]) -> tuple[TunableValue, ...]:
    if not values:
        raise ValueError(f"tunable {name!r} declares no values")
    return values


ModuleName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")]


class BackendSpec(Frozen):
    """A declared implementation of a task.

    `models` are the catalogue models it serves; `env` where it runs; `batch_invariant` whether an
    item's outputs are independent of its batch-mates; `tunables` the declared values of each
    tunable; `worker` the module serving `load(init)` and `handle(state, items, params)`; `worker_sources` the
    source file of each module of an isolated backend's worker, by dotted name, `worker` among them (empty
    for a host or server backend and for a worker the runtime itself provides), which `ensure_env` places in the env;
    `load_stub` the `init` under which `load` builds the fake stack's model in place of weights,
    taking `faults` (the fields of `senselab_worker.fake.FakeFaults`) beside it; `params` its
    `BackendParams` model; `server` the engine configuration of an `Env.server` backend, whose `worker` module runs in
    this process and is given the running server (`load` takes it in `init["server"]`); `requirements` the
    requirement strings a host backend needs from the senselab lock (`ensure_env` refuses an `Env.host()` backend
    unless that lock covers each). `output_affecting` names the components of `OUTPUT_AFFECTING` that change this
    backend's outputs and so enter its logical key, held sorted and once each; `chunk_grid` names the grid its worker
    cuts an item into when that grid is not a backend param.
    """

    task: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    stack: NonEmptyStr
    models: tuple[NonEmptyStr, ...] = ()
    env: Env
    devices: tuple[NonEmptyStr, ...] = Field(min_length=1)
    batch_invariant: bool
    tunables: Tunables = FrozenMapping({})
    output_affecting: tuple[str, ...] = ()
    worker: ModuleName
    worker_sources: StrMap = FrozenMapping({})
    load_stub: Params
    params: type[BackendParams] = BackendParams
    server: ServerConfig | None = None
    requirements: tuple[NonEmptyStr, ...] = ()
    chunk_grid: NonEmptyStr | None = None

    @field_validator("output_affecting")
    @classmethod
    def _check_output_affecting(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not set(v) <= OUTPUT_AFFECTING:
            raise ValueError(f"output_affecting may name only {sorted(OUTPUT_AFFECTING)}, got {sorted(v)}")
        return tuple(sorted(set(v)))

    @field_validator("worker_sources")
    @classmethod
    def _check_worker_sources(cls, v: Mapping[str, str]) -> Mapping[str, str]:
        bad = [m for m in v if not re.fullmatch(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", m)]
        if bad:
            raise ValueError(f"worker_sources names {bad}, not dotted module names")
        return v

    @model_validator(mode="after")
    def _check_worker_placement(self) -> "BackendSpec":
        if self.worker_sources and self.worker not in self.worker_sources:
            raise ValueError(f"worker_sources must hold the worker module {self.worker!r}")
        if self.worker_sources and self.env.kind == "host":
            raise ValueError("a host backend runs in the senselab venv and places no worker sources")
        return self

    @model_validator(mode="after")
    def _check_server(self) -> "BackendSpec":
        if (self.env.kind == "server") != (self.server is not None):
            raise ValueError("a server backend declares `server`, and no other backend does")
        if self.env.kind == "server" and self.worker_sources:
            raise ValueError("a server backend's worker runs in the senselab venv and places no worker sources")
        return self

    @model_validator(mode="after")
    def _check_batching(self) -> "BackendSpec":
        if self.batch_invariant and "batch_formation" in self.output_affecting:
            raise ValueError("a batch-invariant backend cannot declare batch_formation output-affecting")
        return self


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
            NonCanonicalParams: if its params model has a field with no canonical JSON form.
            UnknownTask: if its task is not declared.
            DuplicateDeclaration: if the task already has a backend of that name.
        """
        check_canonical_params(spec.params)
        backends = self._task_backends(spec.task)
        if spec.name in backends:
            raise DuplicateDeclaration(
                f"backend {spec.name!r} of task {spec.task!r} is already declared", backends[spec.name]
            )
        backends[spec.name] = spec

    def snapshot(self) -> tuple[dict[str, TaskSpec], dict[str, dict[str, BackendSpec]]]:
        """Return a copy of the declarations, for `restore`."""
        return dict(self._tasks), {t: dict(b) for t, b in self._backends.items()}

    def restore(self, snapshot: tuple[dict[str, TaskSpec], dict[str, dict[str, BackendSpec]]]) -> None:
        """Make the declarations those `snapshot` took, dropping any declared since."""
        tasks, backends = snapshot
        self._tasks = dict(tasks)
        self._backends = {t: dict(b) for t, b in backends.items()}

    def tasks(self) -> list[TaskSpec]:
        """Return the declared tasks."""
        return list(self._tasks.values())

    def spec(self, task: str) -> TaskSpec:
        """Return the declared task `task`, or raise `UnknownTask`."""
        self._task_backends(task)
        return self._tasks[task]

    def backends(self, task: str) -> list[BackendSpec]:
        """Return the backends of `task`, or raise `UnknownTask`."""
        return list(self._task_backends(task).values())

    def _task_backends(self, task: str) -> dict[str, BackendSpec]:
        try:
            return self._backends[task]
        except KeyError:
            raise UnknownTask(f"no task {task!r} is declared; declared: {sorted(self._tasks)}") from None


REGISTRY = Registry()
"""The process-wide registry: what discovery merges the declaring packages' `register` calls into."""


PORT_PROTOCOLS: tuple[type, ...] = (RegularSeries, Text)
"""The protocols a port may name (ADR-005); a value of any type implementing one is accepted."""


def kernel_types() -> dict[str, type]:
    """Return the kernel value types a port may name, by name: every `Recording` and `Formatted` subclass and protocol.

    The protocols are `PORT_PROTOCOLS`.
    """
    found: dict[str, type] = {Recording.__name__: Recording} | {p.__name__: p for p in PORT_PROTOCOLS}
    pending: list[type] = [Recording, Formatted]
    while pending:
        for sub in pending.pop().__subclasses__():
            found[sub.__name__] = sub
            pending.append(sub)
    return found


C = TypeVar("C", bound=type)


def task(  # noqa: PLR0913 - a task declares its name, version, kind, constraints and registry
    *,
    name: str,
    version: str,
    kind: TaskKind,
    constraints: tuple[Constraint, ...] = (),
    registry: Registry | None = None,
) -> Callable[[C], C]:
    """Set the decorated class's `spec` to its `TaskSpec`, and declare it in `registry` when one is given.

    A package declares such classes in the registry its `register` is given, with `declare_module`.

    The class holds `inputs` and `outputs`, each mapping port names to kernel value types (a port protocol, a
    `Recording` or a `Formatted` result), and may hold `Params`, a `TaskParams` subclass (`TaskParams` when absent).

    Raises:
        TypeError: if `inputs` or `outputs` is missing or maps a port to anything but a kernel value type.
    """

    def declare(cls: C) -> C:
        spec = TaskSpec(
            name=name,
            version=version,
            kind=kind,
            inputs=_port_types(cls, "inputs"),
            outputs=_port_types(cls, "outputs"),
            params=getattr(cls, "Params", TaskParams),
            constraints=constraints,
        )
        if registry is not None:
            registry.add_task(spec)
        cls.spec = spec  # type: ignore[attr-defined]
        return cls

    return declare


def _port_types(cls: type, ports: str) -> dict[str, str]:
    declared = getattr(cls, ports, None)
    if not isinstance(declared, Mapping):
        raise TypeError(f"task class {cls.__qualname__} must map {ports} port names to kernel value types")
    bad = {
        p: t
        for p, t in declared.items()
        if not (isinstance(t, type) and (t in PORT_PROTOCOLS or issubclass(t, (Recording, Formatted))))
    }
    if bad:
        raise TypeError(f"{cls.__qualname__}.{ports}: ports {sorted(bad)} are not kernel value types")
    return {p: t.__name__ for p, t in declared.items()}


def backend(*, name: str, version: str, registry: Registry | None = None, **fields: Any) -> Callable[[C], C]:  # noqa: ANN401
    """Set the decorated class's `spec` to backend `name`'s `BackendSpec`, and declare it in `registry` when given.

    `fields` are the other `BackendSpec` fields.

    Raises:
        pydantic.ValidationError: if a field is missing or not of its declared form.
        NonCanonicalParams, UnknownTask, DuplicateDeclaration: as `Registry.add_backend`.
    """
    spec = BackendSpec(name=name, version=version, **fields)

    def declare(cls: C) -> C:
        if registry is not None:
            registry.add_backend(spec)
        cls.spec = spec  # type: ignore[attr-defined]
        return cls

    return declare


def declare_module(registry: Registry, module: types.ModuleType) -> None:
    """Declare in `registry` every task, then every backend, that a class defined in `module` was decorated as.

    Raises:
        NonCanonicalParams, UnknownTask, DuplicateDeclaration: as `Registry.add_task` and `Registry.add_backend`.
    """
    defined = [o for o in vars(module).values() if isinstance(o, type) and o.__module__ == module.__name__]
    specs = [o.__dict__.get("spec") for o in defined]
    for spec in specs:
        if isinstance(spec, TaskSpec):
            registry.add_task(spec)
    for spec in specs:
        if isinstance(spec, BackendSpec):
            registry.add_backend(spec)


SCHEMA_DIALECT: str = GenerateJsonSchema.schema_dialect
"""The JSON Schema dialect of every generated schema: 2020-12."""


def json_schema(spec: TaskSpec, registry: Registry = REGISTRY) -> dict[str, Any]:
    """Return the JSON Schemas of `spec`'s inputs, params, outputs and each of its backends' params in `registry`.

    The result maps `inputs`, `params` and `outputs` to a schema each, and `backend_params` to a
    mapping of backend name to schema. Every schema names `SCHEMA_DIALECT` in `$schema`. A port's
    schema is titled by its kernel type name; its value's encoding is that type's format.

    Raises:
        UnknownTask: if `spec` is not declared in `registry`.
    """
    return {
        "inputs": _ports_schema(f"{spec.name} inputs", spec.inputs),
        "params": _model_schema(spec.params),
        "backend_params": {b.name: _model_schema(b.params) for b in registry.backends(spec.name)},
        "outputs": _ports_schema(f"{spec.name} outputs", spec.outputs),
    }


def _model_schema(model: type[BaseModel]) -> dict[str, Any]:
    return {"$schema": SCHEMA_DIALECT, **model.model_json_schema()}


def _ports_schema(title: str, ports: Mapping[str, str]) -> dict[str, Any]:
    return {
        "$schema": SCHEMA_DIALECT,
        "title": title,
        "type": "object",
        "properties": {port: {"title": t} for port, t in ports.items()},
        "required": sorted(ports),
        "additionalProperties": False,
    }


def key_for(  # noqa: PLR0913 - keyword-only, one argument per key component
    task: TaskSpec,
    backend: BackendSpec,
    *,
    params: BaseModel,
    backend_params: BaseModel,
    inputs: Mapping[str, InputIdentity],
    model: ResolvedModel | None = None,
    profile_slices: Mapping[str, ProfileSliceIdentity] | None = None,
    batch_formation: str | None = None,
    chunk_grid: str | None = None,
) -> LogicalKey:
    """Return the logical key of calling `backend` for `task` (ADR-002).

    `params` is an instance of the task's params model and `backend_params` of the backend's;
    `inputs` maps each declared input port, and no other, to its `content:` or
    `output:` identity; `model` is the resolved model, which enters the key as its `key_identity()`;
    `profile_slices` maps each profile slice the call reads to its `profile:` identity.
    `batch_formation` and `chunk_grid` are given exactly when the backend declares them
    output-affecting. The runtime builds keys with `senselab_core.runtime.keys.build_key`.

    Raises:
        ValueError: if the backend implements another task, params, backend params or inputs do not
            match the declarations, or an output-affecting component is given undeclared or missing.
        UnpinnedModel: if `model` is not pinned.
        pydantic.ValidationError: if an identity is not of its typed form.
    """
    if backend.task != task.name:
        raise ValueError(f"backend {backend.name!r} implements {backend.task!r}, not {task.name!r}")
    for model_type, value, whose in ((task.params, params, task.name), (backend.params, backend_params, backend.name)):
        if type(value) is not model_type:
            raise ValueError(f"params of {whose!r} are {model_type.__name__}, got {type(value).__name__}")
    if set(inputs) != set(task.inputs):
        raise ValueError(f"{task.name!r} takes inputs {sorted(task.inputs)}, got {sorted(inputs)}")
    components = {"batch_formation": batch_formation, "chunk_grid": chunk_grid}
    for name, component in components.items():
        if (name in backend.output_affecting) != (component is not None):
            state = "declares" if name in backend.output_affecting else "does not declare"
            raise ValueError(f"backend {backend.name!r} {state} {name} output-affecting, got {component!r}")
    return LogicalKey(
        task=task.name,
        task_version=task.version,
        backend=backend.name,
        backend_version=backend.version,
        model=None if model is None else model.key_identity(),
        params=params.model_dump(mode="json"),
        backend_params=backend_params.model_dump(mode="json"),
        inputs=inputs,
        profile_slices=profile_slices or {},
        **components,
    )
