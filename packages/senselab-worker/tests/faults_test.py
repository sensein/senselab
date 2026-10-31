"""Fault classification and error records."""

import builtins
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from senselab_worker.faults import (
    MAX_TEXT_CHARS,
    ArrayIntegrityError,
    ContentFault,
    FaultClass,
    InfrastructureFault,
    ProtocolError,
    ResourceFault,
    ResponseTooLarge,
    classify,
    error_record,
)


class OutOfMemoryError(RuntimeError):
    """Stands in for a framework's out-of-memory error, matched by class name."""


class DeviceOutOfMemory(OutOfMemoryError):
    """A subclass, found through the MRO."""


class IntegrityValueError(ArrayIntegrityError, ValueError):
    """Both an integrity error and a content error; precedence decides."""


class TypedOutOfMemory(ResourceFault):
    """A backend's out-of-memory error, typed at the source."""


class BadInput(ContentFault, KeyError):
    """A backend's content fault that is also a `KeyError`; the type decides, not the table."""


@pytest.mark.logic
@pytest.mark.parametrize(
    ("exc", "fault", "gap"),
    [
        (ContentFault("x"), FaultClass.CONTENT, False),
        (BadInput("x"), FaultClass.CONTENT, False),
        (ResourceFault("x"), FaultClass.RESOURCE, False),
        (TypedOutOfMemory("x"), FaultClass.RESOURCE, False),
        (ResponseTooLarge("x"), FaultClass.RESOURCE, False),
        (ArrayIntegrityError("x"), FaultClass.INFRASTRUCTURE, False),
        (ProtocolError("x"), FaultClass.INFRASTRUCTURE, False),
        (IntegrityValueError("x"), FaultClass.INFRASTRUCTURE, False),
        (InfrastructureFault("x"), FaultClass.INFRASTRUCTURE, False),
        (ValueError("x"), FaultClass.CONTENT, True),
        (TypeError("x"), FaultClass.CONTENT, True),
        (KeyError("x"), FaultClass.CONTENT, True),
        (UnicodeEncodeError("utf-8", "x", 0, 1, "x"), FaultClass.CONTENT, True),
        (MemoryError(), FaultClass.RESOURCE, True),
        (TimeoutError(), FaultClass.RESOURCE, True),
        (OutOfMemoryError(), FaultClass.RESOURCE, True),
        (DeviceOutOfMemory(), FaultClass.RESOURCE, True),
        (RuntimeError("x"), FaultClass.INFRASTRUCTURE, True),
        (OSError("x"), FaultClass.INFRASTRUCTURE, True),
        (KeyboardInterrupt(), FaultClass.INFRASTRUCTURE, True),
    ],
)
def test_fault_table(exc: BaseException, fault: FaultClass, gap: bool) -> None:
    """Typed faults decide their class with no gap; the table is a fallback and always marks a gap."""
    assert classify(exc) is fault
    record = error_record(exc)
    assert record["fault"] == fault.value and record["gap"] is gap
    assert record["type"] == type(exc).__name__


_GROUP = getattr(builtins, "BaseExceptionGroup", None)


def _builtin_exception_types() -> list[type[BaseException]]:
    return [t for t in vars(builtins).values() if isinstance(t, type) and issubclass(t, BaseException)]


def _instance(t: type[BaseException]) -> BaseException:
    if issubclass(t, UnicodeTranslateError):
        return t("x", 0, 1, "r")
    if issubclass(t, (UnicodeEncodeError, UnicodeDecodeError)):
        return t("utf-8", "x" if issubclass(t, UnicodeEncodeError) else b"x", 0, 1, "r")  # type: ignore[arg-type]
    if _GROUP is not None and issubclass(t, _GROUP):
        return t("g", [ValueError("x")])  # type: ignore[call-arg]
    return t("x")


@pytest.mark.logic
def test_fallback_classification_marks_gap() -> None:
    """Every built-in exception is classified by the fallback table, so every one is a gap.

    The same exception mixed with a typed fault is classified by the type, with no gap.
    """
    assert error_record(KeyError("k"))["gap"] is True
    assert error_record(ContentFault("bad input"))["gap"] is False
    builtin_types = _builtin_exception_types()
    mixed_count = 0
    for t in builtin_types:
        assert error_record(_instance(t))["gap"] is True, t
        for typed in (ContentFault, ResourceFault, InfrastructureFault):
            try:
                mixed = _instance(type(f"Typed{t.__name__}", (typed, t), {}))
            except TypeError:
                continue
            mixed_count += 1
            assert error_record(mixed)["gap"] is False, (typed, t)
    assert len(builtin_types) > 60 and mixed_count > 100


@pytest.mark.logic
def test_a25_bug_key_error_is_not_a_content_verdict() -> None:
    """The probe: a backend bug raising `KeyError` was a plain `content` fault; it is now marked a gap."""
    record = error_record(KeyError("missing"))
    assert (record["fault"], record["gap"]) == ("content", True)


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


class _Raises(Exception):
    """Raises from `__str__`, `__repr__`, or both."""

    def __init__(self, raise_str: BaseException | None, raise_repr: BaseException | None) -> None:
        super().__init__("never shown")
        self.raise_str = raise_str
        self.raise_repr = raise_repr

    def __str__(self) -> str:
        if self.raise_str is not None:
            raise self.raise_str
        return "shown by str"

    def __repr__(self) -> str:
        if self.raise_repr is not None:
            raise self.raise_repr
        return "shown by repr"


class _NotStr(Exception):
    def __str__(self) -> str:
        return 42  # type: ignore[return-value]


def _check_record(record: dict) -> None:
    assert set(record) == {"type", "message", "fault", "traceback", "gap"}
    for key in ("type", "message", "traceback"):
        assert type(record[key]) is str
        record[key].encode("utf-8")
        assert len(record[key]) <= MAX_TEXT_CHARS
    assert record["fault"] in {f.value for f in FaultClass} and type(record["gap"]) is bool
    json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")


_FAILURES = st.sampled_from([None, ValueError("v"), RuntimeError("r"), GeneratorExit(), RecursionError()])


@pytest.mark.logic
@settings(max_examples=200, deadline=None)
@given(
    raise_str=_FAILURES,
    raise_repr=_FAILURES,
    message=st.text(alphabet=st.characters(blacklist_categories=()), max_size=50)
    | st.sampled_from(["\ud800", "a\udcffb", "x" * (MAX_TEXT_CHARS * 3)]),
    raised=st.booleans(),
)
def test_error_record_is_total(
    raise_str: BaseException | None, raise_repr: BaseException | None, message: str, raised: bool
) -> None:
    """Raising `__str__` and `__repr__`, lone surrogates and huge messages all give a valid, bounded record."""
    cases: list[BaseException] = [_Raises(raise_str, raise_repr), ValueError(message), ContentFault(message), _NotStr()]
    for exc in cases:
        if raised:
            try:
                raise exc
            except BaseException as caught:  # noqa: BLE001 - the record is built from whatever was raised
                exc = caught
        _check_record(error_record(exc))


@pytest.mark.logic
def test_error_record_message_falls_back_from_str_to_repr_to_type() -> None:
    """`message` is `str(exc)`, else `repr(exc)`, else names the type; surrogates are backslash-escaped."""
    assert error_record(_Raises(None, None))["message"] == "shown by str"
    assert error_record(_Raises(ValueError(), None))["message"] == "shown by repr"
    assert error_record(_Raises(ValueError(), ValueError()))["message"] == "<unprintable _Raises>"
    assert error_record(ValueError("a\udcffb"))["message"] == "a\\udcffb"
