"""Fault classes and the error records that carry them across the wire (ADR-009).

A backend states a fault's class by raising `ContentFault`, `ResourceFault` or `InfrastructureFault`.
Any other exception is classified by the fallback table, and its record is marked as a gap, except an
out-of-memory error (`MemoryError`, or a class named `OutOfMemoryError`), which the table knows.
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


class ArrayReadError(ArrayIntegrityError, OSError):
    """An array file could not be opened or read: an `OSError` carrying the failed call's errno."""

    def __init__(self, errno: int | None, message: str) -> None:
        """Carry `errno` (None when the failure had none) and the whole `message`."""
        if errno is None:
            OSError.__init__(self, message)
        else:
            OSError.__init__(self, errno, message)


class WireValueError(ContentFault, ValueError):
    """A value with no wire form: a type off the whitelist, a non-finite float, invalid Unicode, a cycle, too deep."""


class ResponseTooLarge(ResourceFault):
    """An item's result line would exceed the line budget."""


class ValueTooLarge(ResourceFault, ValueError):
    """A value's wire form cannot fit one line: a lower bound on its encoded size exceeds the line budget."""


_CONTENT = (ValueError, TypeError, KeyError)
_RESOURCE = (MemoryError, TimeoutError)
_OOM_CLASS_NAME = "OutOfMemoryError"
PASS_THROUGH = (KeyboardInterrupt, SystemExit)
"""Exceptions no total function in the worker turns into a record; they end the process as usual."""


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
        return FaultClass.RESOURCE, not _is_out_of_memory(exc)
    return FaultClass.INFRASTRUCTURE, True


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError) or any(c.__name__ == _OOM_CLASS_NAME for c in type(exc).__mro__)


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
        except PASS_THROUGH:
            raise
        except BaseException:  # noqa: BLE001, S112 - a failing __str__ falls through to the next form
            continue
    return f"<unprintable {wire_text(type(exc).__name__)}>"


def _chain_limited(exc: BaseException) -> traceback.TracebackException:
    """Return `exc`'s formatter holding its most recent `TRACEBACK_FRAMES` frames in total across its chain.

    The frames are spent from `exc` outward along its causes and contexts, most recent first.
    """
    formatter = traceback.TracebackException(type(exc), exc, exc.__traceback__, compact=True)
    left = TRACEBACK_FRAMES
    node: traceback.TracebackException | None = formatter
    while node is not None:
        kept = list(node.stack)[-left:] if left else []
        node.stack = traceback.StackSummary.from_list(kept)
        left -= len(kept)
        node = node.__cause__ or (None if node.__suppress_context__ else node.__context__)
    return formatter


def _safe_traceback(exc: BaseException) -> str:
    try:
        return wire_text("".join(_chain_limited(exc).format()))
    except PASS_THROUGH:
        raise
    except BaseException as failure:  # noqa: BLE001 - the record is total; a broken traceback is named, not raised
        return f"<traceback could not be formatted: {wire_text(type(failure).__name__)}>"


def error_record(exc: BaseException) -> dict[str, Any]:
    """Return the wire error object for an exception; total over every exception object.

    Keys: `type` (class name), `message` (`str(exc)`, else `repr(exc)`, else `<unprintable Type>`),
    `fault`, `traceback` (the last 8 frames in total across the cause chain) and `gap`, which is false
    for a `ContentFault`, `ResourceFault` or `InfrastructureFault` and for an out-of-memory error, and true
    whenever the rest of the fallback table decided. Text
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
