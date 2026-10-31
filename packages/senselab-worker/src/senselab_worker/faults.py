"""Fault classes and the error records that carry them across the wire (ADR-009).

A backend states a fault's class by raising `ContentFault`, `ResourceFault` or `InfrastructureFault`.
Any other exception is classified by the fallback table, and its record is marked as a gap.
"""

import traceback
from enum import Enum
from typing import Any

TRACEBACK_FRAMES = 8
MAX_TEXT_CHARS = 2**16


class FaultClass(str, Enum):
    """What a failure says about its cause: the input, the resources, or the machinery."""

    CONTENT = "content"
    RESOURCE = "resource"
    INFRASTRUCTURE = "infrastructure"


class ContentFault(Exception):
    """Raised by a backend for a failure caused by this item's input, or by the model on this item."""


class ResourceFault(Exception):
    """Raised by a backend for a failure caused by a lack of resources: memory, time, a device."""


class InfrastructureFault(RuntimeError):
    """Base of faults caused by the machinery: environment, handshake, network, lost service."""


class ProtocolError(InfrastructureFault):
    """A message, or a backend's result, breaks the wire protocol."""


class ArrayIntegrityError(InfrastructureFault):
    """An array reference is malformed, or its file is missing, unreadable or mismatched."""


class WireValueError(ContentFault, ValueError):
    """A value with no wire form: a type off the whitelist, a non-finite float, invalid Unicode, a cycle, too deep."""


class ResponseTooLarge(ResourceFault):
    """An item's result line would exceed the line budget."""


_CONTENT = (ValueError, TypeError, KeyError)
_RESOURCE = (MemoryError, TimeoutError)
_OOM_CLASS_NAME = "OutOfMemoryError"
_PASS_THROUGH = (KeyboardInterrupt, SystemExit)


def _classify(exc: BaseException) -> tuple[FaultClass, bool]:
    if isinstance(exc, InfrastructureFault):
        return FaultClass.INFRASTRUCTURE, False
    if isinstance(exc, ResourceFault):
        return FaultClass.RESOURCE, False
    if isinstance(exc, ContentFault):
        return FaultClass.CONTENT, False
    if isinstance(exc, _CONTENT):
        return FaultClass.CONTENT, True
    if isinstance(exc, _RESOURCE) or any(c.__name__ == _OOM_CLASS_NAME for c in type(exc).__mro__):
        return FaultClass.RESOURCE, True
    return FaultClass.INFRASTRUCTURE, True


def classify(exc: BaseException) -> FaultClass:
    """Return the fault class of an exception: by its typed fault base, else by the fallback table."""
    return _classify(exc)[0]


def wire_text(text: str) -> str:
    """Return `text` with every code point UTF-8 cannot encode backslash-escaped, cut to `MAX_TEXT_CHARS`."""
    return str.encode(text, "utf-8", "backslashreplace").decode("utf-8")[:MAX_TEXT_CHARS]


def _safe_text(exc: BaseException) -> str:
    """Return `str(exc)`, else `repr(exc)`, else `<unprintable TypeName>`, as wire text."""
    for show in (str, repr):
        try:
            text = show(exc)
            if isinstance(text, str):
                return wire_text(text)
        except _PASS_THROUGH:
            raise
        except BaseException:  # noqa: BLE001, S112 - a failing __str__ falls through to the next form
            continue
    return f"<unprintable {wire_text(type(exc).__name__)}>"


def _safe_traceback(exc: BaseException) -> str:
    try:
        frames = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=-TRACEBACK_FRAMES)
        return wire_text("".join(frames))
    except _PASS_THROUGH:
        raise
    except BaseException as failure:  # noqa: BLE001 - the record is total; a broken traceback is named, not raised
        return f"<traceback could not be formatted: {wire_text(type(failure).__name__)}>"


def error_record(exc: BaseException) -> dict[str, Any]:
    """Return the wire error object for an exception; total over every exception object.

    Keys: `type` (class name), `message` (`str(exc)`, else `repr(exc)`, else `<unprintable Type>`),
    `fault`, `traceback` (the last 8 frames) and `gap`, which is false only for a `ContentFault`,
    `ResourceFault` or `InfrastructureFault`, and true whenever the fallback table decided. Text
    fields are cut to `MAX_TEXT_CHARS`, and code points UTF-8 cannot encode, such as lone
    surrogates, are backslash-escaped.
    """
    fault, gap = _classify(exc)
    return {
        "type": wire_text(type(exc).__name__),
        "message": _safe_text(exc),
        "fault": fault.value,
        "traceback": _safe_traceback(exc),
        "gap": gap,
    }
