"""The base of every generated task class and the error envelope; a configured task runs on `InProcess`."""

import difflib
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, ClassVar, NoReturn, Self

from pydantic import BaseModel, ConfigDict, ValidationError
from senselab_core.kernel._base import FrozenMapping, NonEmptyStr
from senselab_core.kernel.dataset import Dataset
from senselab_core.kernel.declare import (
    REGISTRY,
    DeclaredParams,
    InconsistentInputs,
    Registry,
    TaskParams,
    TaskSpec,
    UnknownTask,
    kernel_types,
)
from senselab_core.kernel.identity import ModelRef
from senselab_core.kernel.license import LicenseConflict
from senselab_core.kernel.wire import WireFormatRetired
from senselab_core.runtime.call import Result, TaskCall
from senselab_core.runtime.execution.in_process import CallFailed, InProcess
from senselab_core.runtime.execution.outputs import AmbiguousClock, OutputTypeMismatch
from senselab_core.runtime.execution.run import ResultSet
from senselab_core.runtime.execution.workers import WorkerLoadFailed
from senselab_core.runtime.keys import MissingProfileSlice
from senselab_core.runtime.licenses import LicenseRefused
from senselab_core.runtime.locks import LeaseTimeout, LeaseUnavailable
from senselab_core.runtime.models import ModelResolutionError
from senselab_core.runtime.placement import BackendParamsMismatch, MemoryCapExceeded
from senselab_core.runtime.profiles import NoBackend, NoDevice, UnlistedSource
from senselab_core.runtime.vllm_server import AttestationMismatch, VllmStartError
from senselab_core.runtime.worker_client import WorkerFatal, WorkerLost
from senselab_worker.faults import FaultClass, ProtocolError


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


_LOAD_FIX = "the backend's worker did not start or load; this is a fault of the host or its env, not of the input"
"""The fix of a `WorkerLoadFailed` envelope (`worker_load_failed`)."""

ERROR_CODES: tuple[tuple[type[Exception], str, str], ...] = (
    (InconsistentInputs, "inconsistent_inputs", "give inputs and params that agree; `describe` lists the constraints"),
    (UnknownTask, "unknown_task", "name a declared task; `tasks()` lists them"),
    (AmbiguousClock, "ambiguous_clock", "give the recordings of one content the same anchor"),
    (OutputTypeMismatch, "output_type_mismatch", "a backend returned a value of the wrong type; report the backend"),
    (WireFormatRetired, "wire_format_retired", "recompute the call; its stored output is in a retired format"),
    (UnlistedSource, "unlisted_source", "list the model source in the execution profile's allowlist"),
    (NoBackend, "no_backend", "choose a model with a backend, or one the execution profile allows"),
    (NoDevice, "no_device", "make a device the backend supports available, or change the execution profile"),
    (ModelResolutionError, "model_resolution_failed", "check the model reference and the network, then retry"),
    (BackendParamsMismatch, "invalid_backend_params", "give params only for a declared backend, as it declares them"),
    (LicenseConflict, "license_conflict", "one subject has two license records; report the catalogue"),
    (MemoryCapExceeded, "memory_cap_exceeded", "raise the resource class's memory cap, or choose a smaller model"),
    (MissingProfileSlice, "missing_profile_slice", "set the run settings or dataset profile value the task reads"),
    (VllmStartError, "server_start_failed", "see the server log; retry once the host has the resources it needs"),
    (AttestationMismatch, "attestation_mismatch", "start the server as the backend declares it"),
    (LeaseTimeout, "resource_busy", "retry when the other holder of the resource has finished"),
    (LeaseUnavailable, "resource_busy", "retry when the other holder of the resource has finished"),
    (WorkerLoadFailed, "worker_load_failed", _LOAD_FIX),
    (WorkerFatal, "worker_failed", "the worker stopped; this is a fault of the host, not of the input"),
    (WorkerLost, "worker_failed", "the worker stopped; this is a fault of the host, not of the input"),
    (ProtocolError, "worker_failed", "the worker broke its protocol; report the backend"),
    (
        LicenseRefused,
        "license_refused",
        "the run config's license policy refuses this backend's licenses; change the policy or choose another backend",
    ),
    (ValueError, "invalid_call", "see the message for the value to correct"),
)
"""Each error `InProcess.call` documents, with its envelope code and fix; the first row an instance matches wins."""

_FAULT_FIX = {
    FaultClass.CONTENT: "the backend rejected the input; correct or replace it",
    FaultClass.RESOURCE: "the call ran out of memory or time; give it a smaller input or more resources",
    FaultClass.INFRASTRUCTURE: "the backend failed for a reason of its own, not of the input; report it",
}


def api_error(error: Exception) -> ApiError | None:
    """Return the `ApiError` for an error `InProcess.call` documents, or None when it documents none such.

    A `WorkerLoadFailed` is `worker_load_failed` with its cause's type and message, or the envelope of its cause
    when that is a server that did not start (`server_start_failed`) or attests another model (`attestation_mismatch`).
    """
    if isinstance(error, WorkerLoadFailed):
        cause = error.__cause__
        if isinstance(cause, (VllmStartError, AttestationMismatch)):
            return api_error(cause)
        detail = f"{type(cause).__name__}: {cause}" if cause is not None else str(error)
        return ApiError("worker_load_failed", " ".join(detail.split()), _LOAD_FIX)
    if isinstance(error, CallFailed):
        fault = error.failure.fault_class
        return ApiError(f"{fault.value}_fault", str(error), _FAULT_FIX[fault])
    for kind, code, fix in ERROR_CODES:
        if isinstance(error, kind):
            return ApiError(code, " ".join(str(error).split()) or type(error).__name__, fix)
    return None


def reraise(error: Exception) -> NoReturn:
    """Raise the `ApiError` envelope for `error` from it; an error with no envelope is raised as it is."""
    envelope = api_error(error)
    if envelope is None:
        raise error
    raise envelope from error


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
        """Configure from the params mapping of the one call shape, parsed as `TaskCall` parses it."""
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
        try:
            return TaskCall.parse_params(self.name, given, self.registry)
        except InconsistentInputs as error:
            raise api_error(error) from None  # type: ignore[misc]
        except (ValidationError, ValueError) as error:
            raise _params_error(self.name, self.spec.params, given, error) from None

    def _replace(self, changes: Mapping[str, Any]) -> Self:
        merged = {**self._given, **{k: v for k, v in changes.items() if v is not UNSET}}
        return type(self)(self._model, **merged, backend_params=self._backend_params)

    def _run(self, inputs: Mapping[str, Any]) -> Result:
        call = self._call(inputs)
        try:
            return InProcess().call(call)
        except Exception as error:
            reraise(error)

    def _map(self, dataset: Dataset, bind: Mapping[str, str] | None) -> ResultSet:
        from senselab._api.facade import map_over  # noqa: PLC0415 - the facade imports this module

        return map_over(self, dataset, bind)

    def _task_call(self) -> TaskCall:
        """Return the call of this task with no inputs, to be bound from a dataset's records."""
        spec = self.spec
        return TaskCall(
            task=spec.name,
            task_version=spec.version,
            inputs={},
            params=self._params,
            model=self._model,
            backend_params=self._backend_params,
        )

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
    try:
        return TaskCall.parse_model(model)
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
