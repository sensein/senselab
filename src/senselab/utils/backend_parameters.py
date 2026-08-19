"""The one way a dispatcher forwards backend-specific parameters, and records what it forwarded.

A task-level entry point takes ``audios``, a ``model``, and a ``device``, and dispatches to one of
several backends. Anything a backend alone understands — DriftSE's ``variant``, unasdiff's
``diffusion_steps``, a worker's ``timeout_s`` — has nowhere to travel through that signature, and
the measured consequence is that it does not travel: ``enhance_audios`` never forwarded ``variant``,
so only one of DriftSE's two released checkpoints was reachable through the public API, and it is
the one that suppresses a verified breath by 14.2 dB. The other checkpoint existed, was documented,
and could not be selected.

The obvious fix — a ``**kwargs`` or a free-form ``dict`` handed to the backend — is worse than the
defect. A misspelled key would be dropped by ``**kwargs`` or ignored by a permissive backend, the
default would run, and the run would report the parameter the caller thought they set. That is a
confidently wrong result, which this repository treats as worse than no result (see
``utils/model_revision.py``'s ``RevisionResolutionError`` for the same judgement about provenance).

So a parameter mapping is validated against the selected backend's *own signature* before anything
runs, and the effective set is returned for recording:

* **Declared, not documented.** :func:`declared_parameters` reads the backend callable's signature.
  A hand-maintained table of allowed keys is a second source of truth that goes stale the first
  time a backend gains a parameter; a signature cannot.
* **Validated against the selected backend.** A DriftSE key passed with a SpeechBrain model raises,
  rather than being accepted by a dispatcher that has not yet decided where it is going.
* **Unknown keys raise, with the near misses named.** ``difflib`` supplies the suggestion, because
  the failure this exists to prevent is a typo, and a caller who typed ``varient`` needs to be told
  ``variant``, not handed the whole declared set to search.
* **The effective set is recorded, defaults included.** A record of only the explicit values cannot
  answer "what ran"; a record of the defaults too can. :func:`ParameterRecord.explicit` keeps the
  distinction available for a reader who needs it.

Reasoning: ``specs/20260819-clearvoice-integration/design.md``.
"""

from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

# Parameters a task-level dispatcher owns and passes itself. A caller who tries to route one of
# these through the parameter mapping is not tuning a backend, they are trying to override the
# dispatcher's own argument from inside its payload -- two sources for one value, which is how a
# device gets selected twice and the loser silently wins.
DISPATCHER_OWNED = ("audios", "audio", "videos", "model", "device", "self", "cls")


@dataclass(frozen=True)
class ParameterRecord:
    """What a dispatcher actually forwarded, fit to record as provenance.

    Attributes:
        backend: Name of the backend the parameters were validated against.
        effective: Every declared parameter and the value that ran, explicit or default. Values are
            rendered JSON-friendly, so this can go straight into ``Audio.metadata``.
        explicit: The subset the caller named, so a reader can tell a deliberate choice from a
            default that happened to be recorded.
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
        ``{name: default}`` for every keyword-accepting parameter that is not dispatcher-owned.
        A parameter with no default maps to :data:`inspect.Parameter.empty`, which
        :func:`resolve_backend_parameters` reports as required rather than inventing a value for.
    """
    declared: Dict[str, Any] = {}
    for name, parameter in inspect.signature(backend).parameters.items():
        if name in owned:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            # A **kwargs backend declares nothing checkable, and treating it as "anything goes"
            # would reintroduce exactly the silent-typo failure this module exists to prevent.
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
        ``(kwargs, record)`` — the keyword arguments to forward (only the caller's explicit keys, so
        a backend default stays the backend's to change), and the :class:`ParameterRecord` naming
        every effective value.

    Raises:
        TypeError: If ``parameters`` is not a mapping, or has a non-string key. A list of pairs or a
            stray positional would otherwise be silently unpackable into something plausible.
        ValueError: If any key is not declared by ``backend``, or names a dispatcher-owned
            parameter. The message names the offending key, the closest declared names, and — when
            the backend declares none — says so, which is the answer for a caller who passed a
            DriftSE parameter to the SpeechBrain enhancer.
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
