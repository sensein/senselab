"""Validated forwarding of backend-specific parameters, with a record of what was forwarded.

A task entry point takes ``audios``, a ``model`` and a ``device``; anything only one backend
understands — DriftSE's ``variant``, unasdiff's ``diffusion_steps``, a worker's ``timeout_s`` — has no
channel through that signature. :func:`resolve_backend_parameters` is that channel:

* keys are validated against the **selected** backend's own signature, so a typo raises instead of
  silently running the default (:func:`declared_parameters`);
* only the caller's explicit keys are forwarded, so a backend default stays the backend's to change;
* the effective set, defaults included, comes back as a :class:`ParameterRecord` for
  :func:`record_parameters_on` to stamp onto each result's metadata.

The defect this fixes, and why a permissive dictionary would be worse than no pathway:
``specs/20260819-clearvoice-integration/design.md`` §6.
"""

from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

# Parameters a dispatcher passes itself; routing one of these through the mapping would give one
# value two sources.
DISPATCHER_OWNED = ("audios", "audio", "videos", "model", "device", "self", "cls")


@dataclass(frozen=True)
class ParameterRecord:
    """What a dispatcher actually forwarded, fit to record as provenance.

    Attributes:
        backend: Name of the backend the parameters were validated against.
        effective: Every declared parameter and the value that ran, explicit or default, rendered
            JSON-friendly so it can go straight into ``Audio.metadata``.
        explicit: The subset the caller named, distinguishing a deliberate choice from a default.
    """

    backend: str
    effective: Dict[str, Any] = field(default_factory=dict)
    explicit: Tuple[str, ...] = ()

    def as_metadata(self) -> Dict[str, Any]:
        """Return the record as a plain dict, for a metadata or artifact field."""
        return {"backend": self.backend, "parameters": dict(self.effective), "explicit": list(self.explicit)}


def _jsonable(value: Any) -> Any:  # noqa: ANN401 -- deliberately accepts any parameter value
    """Render a parameter value for provenance without pretending it is structured data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def declared_parameters(backend: Callable[..., Any], owned: Sequence[str] = DISPATCHER_OWNED) -> Dict[str, Any]:
    """Return the backend's forwardable parameters and their defaults.

    Args:
        backend: The backend callable a dispatcher would forward to.
        owned: Parameter names the dispatcher passes itself and a caller may not override.

    Returns:
        ``{name: default}`` for every keyword-accepting parameter that is not dispatcher-owned. A
        parameter with no default maps to :data:`inspect.Parameter.empty`.
    """
    declared: Dict[str, Any] = {}
    for name, parameter in inspect.signature(backend).parameters.items():
        if name in owned:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            # A **kwargs backend declares nothing checkable, and is treated as declaring nothing.
            continue
        declared[name] = parameter.default
    return declared


def resolve_backend_parameters(
    backend: Callable[..., Any],
    parameters: Optional[Mapping[str, Any]],
    *,
    backend_name: str,
    owned: Sequence[str] = DISPATCHER_OWNED,
) -> Tuple[Dict[str, Any], ParameterRecord]:
    """Validate a caller's parameter mapping against a backend, and record what will run.

    Args:
        backend: The backend callable the dispatcher has selected.
        parameters: The caller's mapping, or ``None`` for "backend defaults".
        backend_name: Name used in error messages and in the record.
        owned: Parameter names the dispatcher passes itself.

    Returns:
        ``(kwargs, record)`` — the keyword arguments to forward, and the :class:`ParameterRecord`
        naming every effective value.

    Raises:
        TypeError: If ``parameters`` is not a mapping, or has a non-string key.
        ValueError: If any key is not declared by ``backend``, or names a dispatcher-owned parameter.
            The message names the offending key and the closest declared names.
    """
    declared = declared_parameters(backend, owned)

    given: Dict[str, Any] = {}
    if parameters is not None:
        if not isinstance(parameters, Mapping):
            raise TypeError(
                f"parameters for {backend_name} must be a mapping of parameter name to value, got "
                f"{type(parameters).__name__}"
            )
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise TypeError(f"parameters for {backend_name} must be keyed by parameter name, got key {key!r}")
            given[key] = value

    owned_keys = sorted(set(given) & set(owned))
    if owned_keys:
        raise ValueError(
            f"{', '.join(repr(k) for k in owned_keys)} cannot be passed through parameters for "
            f"{backend_name}: it is an argument of the task function itself. Pass it directly, so "
            "there is one source for the value rather than two."
        )

    unknown = sorted(set(given) - set(declared))
    if unknown:
        details = []
        for key in unknown:
            close = difflib.get_close_matches(key, sorted(declared), n=3, cutoff=0.6)
            details.append(f"{key!r}" + (f" (did you mean {', '.join(repr(c) for c in close)}?)" if close else ""))
        declared_list = ", ".join(sorted(declared)) if declared else "no tunable parameters"
        raise ValueError(
            f"Unknown parameter(s) for {backend_name}: {'; '.join(details)}. It declares: "
            f"{declared_list}. Refusing rather than ignoring them: an ignored key would run the "
            "default while the caller believed otherwise."
        )

    missing_required = sorted(
        name for name, default in declared.items() if default is inspect.Parameter.empty and name not in given
    )
    if missing_required:
        raise ValueError(
            f"{backend_name} requires parameter(s) with no default: {', '.join(missing_required)}. "
            "Pass them through parameters."
        )

    effective = {
        name: _jsonable(given.get(name, default))
        for name, default in declared.items()
        if default is not inspect.Parameter.empty or name in given
    }
    return given, ParameterRecord(backend=backend_name, effective=effective, explicit=tuple(sorted(given)))


# The metadata key a parameter record lands under, following this repository's flat convention for
# provenance on a data object (``metadata["vad"]``, ``metadata["unasdiff_alignment_margins"]``).
PARAMETER_RECORD_KEY = "backend_parameters"


def record_parameters_on(items: Any, record: ParameterRecord, key: str = PARAMETER_RECORD_KEY) -> None:  # noqa: ANN401
    """Stamp a parameter record onto the ``metadata`` of every returned object.

    Duck-typed on ``.metadata`` rather than typed against ``Audio``, so this module imports nothing
    from ``senselab.audio``. Nested lists — a separator's sources per input — are walked.

    Args:
        items: An object with ``.metadata``, or an arbitrarily nested sequence of them.
        record: The record to stamp.
        key: Metadata key to write under.
    """
    if isinstance(items, (list, tuple)):
        for item in items:
            record_parameters_on(item, record, key)
        return
    metadata = getattr(items, "metadata", None)
    if isinstance(metadata, dict):
        metadata[key] = record.as_metadata()
