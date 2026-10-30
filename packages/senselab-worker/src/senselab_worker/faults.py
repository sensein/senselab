"""Fault classes and the error records that carry them across the wire (ADR-009)."""

import traceback
from enum import Enum
from typing import Any

TRACEBACK_FRAMES = 8


class FaultClass(str, Enum):
    """What a failure says about its cause: the input, the resources, or the machinery."""

    CONTENT = "content"
    RESOURCE = "resource"
    INFRASTRUCTURE = "infrastructure"


class InfrastructureFault(RuntimeError):
    """Base of faults that are `infrastructure` by definition, decided before the fault table."""


class ProtocolError(InfrastructureFault):
    """A message, or a backend's result, breaks the wire protocol."""


class ArrayIntegrityError(InfrastructureFault):
    """An array reference is malformed, or its file is missing, unreadable or mismatched."""


_CONTENT = (ValueError, TypeError, KeyError)
_RESOURCE = (MemoryError, TimeoutError)
_OOM_CLASS_NAME = "OutOfMemoryError"


def _classify(exc: BaseException) -> tuple[FaultClass, bool]:
    if isinstance(exc, InfrastructureFault):
        return FaultClass.INFRASTRUCTURE, False
    if isinstance(exc, _CONTENT):
        return FaultClass.CONTENT, False
    if isinstance(exc, _RESOURCE) or any(c.__name__ == _OOM_CLASS_NAME for c in type(exc).__mro__):
        return FaultClass.RESOURCE, False
    return FaultClass.INFRASTRUCTURE, True


def classify(exc: BaseException) -> FaultClass:
    """Return the fault class of an exception by the fault table, first match wins."""
    return _classify(exc)[0]


def error_record(exc: BaseException) -> dict[str, Any]:
    """Return the wire error object for an exception.

    Keys: `type` (class name), `message`, `fault`, `traceback` (last 8 frames) and `gap` (true when
    the table had no entry and `infrastructure` was assumed).
    """
    fault, gap = _classify(exc)
    frames = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=-TRACEBACK_FRAMES)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "fault": fault.value,
        "traceback": "".join(frames),
        "gap": gap,
    }
