"""Fault classification and error records."""

import pytest
from senselab_worker.faults import (
    ArrayIntegrityError,
    FaultClass,
    InfrastructureFault,
    ProtocolError,
    classify,
    error_record,
)


class OutOfMemoryError(RuntimeError):
    """Stands in for a framework's out-of-memory error, matched by class name."""


class DeviceOutOfMemory(OutOfMemoryError):
    """A subclass, found through the MRO."""


class IntegrityValueError(ArrayIntegrityError, ValueError):
    """Both an integrity error and a content error; precedence decides."""


@pytest.mark.logic
@pytest.mark.parametrize(
    ("exc", "fault", "gap"),
    [
        (ValueError("x"), FaultClass.CONTENT, False),
        (TypeError("x"), FaultClass.CONTENT, False),
        (KeyError("x"), FaultClass.CONTENT, False),
        (UnicodeEncodeError("utf-8", "x", 0, 1, "x"), FaultClass.CONTENT, False),
        (MemoryError(), FaultClass.RESOURCE, False),
        (TimeoutError(), FaultClass.RESOURCE, False),
        (OutOfMemoryError(), FaultClass.RESOURCE, False),
        (DeviceOutOfMemory(), FaultClass.RESOURCE, False),
        (ArrayIntegrityError("x"), FaultClass.INFRASTRUCTURE, False),
        (ProtocolError("x"), FaultClass.INFRASTRUCTURE, False),
        (IntegrityValueError("x"), FaultClass.INFRASTRUCTURE, False),
        (InfrastructureFault("x"), FaultClass.INFRASTRUCTURE, False),
        (RuntimeError("x"), FaultClass.INFRASTRUCTURE, True),
        (OSError("x"), FaultClass.INFRASTRUCTURE, True),
        (KeyboardInterrupt(), FaultClass.INFRASTRUCTURE, True),
    ],
)
def test_fault_table(exc: BaseException, fault: FaultClass, gap: bool) -> None:
    """Exceptions map to faults by the table, with precedence for infrastructure faults."""
    assert classify(exc) is fault
    record = error_record(exc)
    assert record["fault"] == fault.value and record["gap"] is gap
    assert record["type"] == type(exc).__name__


def _ping(n: int) -> None:
    if n == 0:
        raise ValueError("deep")
    _pong(n - 1)


def _pong(n: int) -> None:
    _ping(n)


@pytest.mark.logic
def test_error_record_keeps_last_eight_frames() -> None:
    """The traceback holds at most the last 8 frames."""
    try:
        _ping(20)
    except ValueError as exc:
        record = error_record(exc)
    assert record["message"] == "deep"
    assert record["traceback"].count('File "') == 8
    assert "raise ValueError" in record["traceback"]
