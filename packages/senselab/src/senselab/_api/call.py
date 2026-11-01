"""The base of every generated task class and the error envelope; a configured task runs through `run_call`."""

import difflib
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationError
from senselab_core.kernel._base import FrozenMapping, NonEmptyStr
from senselab_core.kernel.declare import REGISTRY, DeclaredParams, Registry, TaskParams, TaskSpec
from senselab_core.kernel.identity import ModelRef
from senselab_core.runtime.call import Result, TaskCall, kernel_types, run_call


class _Unset:
    """The value of a parameter the caller did not give."""

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Any = _Unset()
"""The default of a parameter whose declared default is not a literal: the declared default applies."""


class ErrorEnvelope(BaseModel):
    """The one error shape of every surface: `{code, message, fix, field, suggestions}`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: NonEmptyStr
    message: NonEmptyStr
    fix: NonEmptyStr
    field: str | None = None
    suggestions: tuple[str, ...] = ()


class ApiError(Exception):
    """An error of the task API, carrying its `ErrorEnvelope`."""

    def __init__(
        self, code: str, message: str, fix: str, field: str | None = None, suggestions: Iterable[str] = ()
    ) -> None:
        """Build the envelope; `str` of the error is its message."""
        super().__init__(message)
        self.envelope = ErrorEnvelope(code=code, message=message, fix=fix, field=field, suggestions=tuple(suggestions))

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as a JSON-ready mapping."""
        return self.envelope.model_dump(mode="json")


def closest(word: str, known: Iterable[str], n: int = 3) -> list[str]:
    """Return up to `n` of `known` closest to `word`."""
    return difflib.get_close_matches(word, sorted(known), n=n)


def callable_names(spec: TaskSpec) -> tuple[str, str]:
    """Return the generated class name and function name of `spec`: its last dotted segment, in CamelCase and as is."""
    segment = spec.name.rsplit(".", 1)[-1]
    return "".join(w.capitalize() for w in segment.split("_")), segment


def task_spec(name: str, registry: Registry = REGISTRY) -> TaskSpec:
    """Return the declared task `name`.

    Raises:
        ApiError: `unknown_task`, with the closest declared names.
    """
    for spec in registry.tasks():
        if spec.name == name:
            return spec
    declared = [s.name for s in registry.tasks()]
    raise ApiError(
        "unknown_task",
        f"no task {name!r} is declared",
        "name a declared task; `tasks()` lists them",
        field="task",
        suggestions=closest(name, declared),
    )


def _is_data(value: object, declared: type | None) -> bool:
    return isinstance(value, str | PathLike) or declared is None or isinstance(value, declared)


class ConfiguredTask:
    """Base of the generated task classes: immutable, configured with a model and params.

    A subclass sets `name` to its declared task name and gives `__init__` and `call` real signatures.
    """

    name: ClassVar[str]
    registry: ClassVar[Registry] = REGISTRY
    __slots__ = ("_backend_params", "_given", "_model", "_params")
    _backend_params: Mapping[str, Any]
    _given: dict[str, Any]
    _model: ModelRef | None
    _params: TaskParams

    def __init__(
        self,
        model: ModelRef | str | None,
        params: Mapping[str, Any],
        backend_params: Mapping[str, Any] | None,
    ) -> None:
        """Validate the configuration; `UNSET` params take their declared default."""
        given = {k: v for k, v in params.items() if v is not UNSET}
        object.__setattr__(self, "_given", given)
        object.__setattr__(self, "_params", self._validate_params(given))
        object.__setattr__(self, "_model", _model_ref(model))
        object.__setattr__(self, "_backend_params", FrozenMapping(dict(backend_params or {})))

    @classmethod
    def from_mapping(
        cls,
        model: ModelRef | str | None = None,
        params: Mapping[str, Any] | None = None,
        backend_params: Mapping[str, Any] | None = None,
    ) -> Self:
        """Configure from the params mapping of the one call shape, as the CLI and JSON give it."""
        task = object.__new__(cls)
        ConfiguredTask.__init__(task, model, params or {}, backend_params)
        return task

    @property
    def spec(self) -> TaskSpec:
        """The declaration of this task."""
        return task_spec(self.name, self.registry)

    @property
    def model(self) -> ModelRef | None:
        """The model reference, if one was given."""
        return self._model

    @property
    def params(self) -> TaskParams:
        """The validated params."""
        return self._params

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; use with_params")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __repr__(self) -> str:
        """Return the class with its model and given params."""
        return f"{type(self).__name__}(model={None if self._model is None else str(self._model)!r}, **{self._given!r})"

    def _validate_params(self, given: Mapping[str, Any]) -> TaskParams:
        model = self.spec.params
        try:
            return model(**given)
        except (ValidationError, ValueError) as error:
            raise _params_error(self.name, model, given, error) from None

    def _replace(self, changes: Mapping[str, Any]) -> Any:  # noqa: ANN401 - the generated subclass names its own type
        merged = {**self._given, **{k: v for k, v in changes.items() if v is not UNSET}}
        return type(self)(self._model, **merged, backend_params=self._backend_params)

    def _run(self, inputs: Mapping[str, Any]) -> Result:
        return run_call(self._call(inputs))

    def _call(self, inputs: Mapping[str, Any]) -> TaskCall:
        spec = self.spec
        if set(inputs) != set(spec.inputs):
            raise ApiError(
                "wrong_inputs",
                f"task {spec.name!r} takes inputs {sorted(spec.inputs)}, got {sorted(inputs)}",
                f"give exactly the inputs {sorted(spec.inputs)}",
                field=", ".join(sorted(set(inputs) ^ set(spec.inputs))),
            )
        types = kernel_types()
        for port, value in inputs.items():
            if not _is_data(value, types.get(spec.inputs[port])):
                raise ApiError(
                    "wrong_input_type",
                    f"input {port!r} of {spec.name!r} is a {spec.inputs[port]}, got {type(value).__name__}",
                    f"pass a {spec.inputs[port]}, a path or a URI",
                    field=port,
                )
        return TaskCall(
            task=spec.name,
            task_version=spec.version,
            inputs=inputs,
            params=self._params,
            model=self._model,
            backend_params=self._backend_params,
        )


def _model_ref(model: ModelRef | str | None) -> ModelRef | None:
    if model is None or isinstance(model, ModelRef):
        return model
    try:
        return ModelRef.parse(model)
    except ValueError as error:
        raise ApiError(
            "invalid_model", str(error), "write the model as `org/name@revision`, `url:...` or `local:...`", "model"
        ) from None


def _params_error(task: str, model: type[DeclaredParams], given: Mapping[str, Any], error: Exception) -> ApiError:
    known = sorted(model.model_fields)
    unknown = sorted(set(given) - set(known))
    if unknown:
        return ApiError(
            "unknown_param",
            f"task {task!r} has no param {unknown[0]!r}",
            f"use one of the declared params: {known}",
            field=unknown[0],
            suggestions=closest(unknown[0], known),
        )
    first = error.errors()[0] if isinstance(error, ValidationError) and error.errors() else None
    field = str(first["loc"][0]) if first and first["loc"] else None
    return ApiError(
        "invalid_param",
        f"param {field!r} of task {task!r} is invalid: {first['msg'] if first else error}",
        f"see `describe({task!r})` for the type of each param",
        field=field,
    )
