"""Phase 1 adversarial findings on the worker, rebuilt from the report's break column.

Each probe uses only what existed at 94e81479 and imports it inside the test.
"""

import io
import json
from enum import Enum
from pathlib import Path

import pytest


def _lines(
    handle: object, messages: list[dict[str, object]], array_dir: Path
) -> tuple[int, dict[str, dict[str, object]]]:
    """Serve ``messages`` through ``handle``; return the exit code and every item record by item id."""
    from senselab_worker.serve import serve

    rx = io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in messages))
    tx = io.BytesIO()
    code = serve(lambda init: "state", handle, rx, tx)  # type: ignore[arg-type]
    records: dict[str, dict[str, object]] = {}
    for line in tx.getvalue().splitlines():
        message = json.loads(line)
        for result in message.get("results", []):
            records[result["id"]] = result
        if "item" in message:
            records[message["item"]] = message
    return code, records


def _session(handle: object, tmp_path: Path, first: list[str]) -> tuple[int, dict[str, dict[str, object]]]:
    def request(rid: str, ids: list[str]) -> dict[str, object]:
        return {"id": rid, "items": [{"id": i, "inputs": {"k": 1}} for i in ids], "params": {}}

    return _lines(
        handle,
        [{"init": {"array_dir": str(tmp_path)}}, request("req-1", first), request("req-2", ["after"]), {"stop": True}],
        tmp_path,
    )


class _Unprintable(Exception):
    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        raise RuntimeError("no repr")


@pytest.mark.logic
def test_a10_probe_unprintable_or_surrogate_error_fails_only_its_item(tmp_path: Path) -> None:
    """An error whose `__str__` raises, or whose message holds a lone surrogate, fails its item and not the worker."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:  # type: ignore[type-arg]
        errors = {"str": _Unprintable(), "surrogate": OSError("cannot open /data/\udcff.wav")}
        return [{"error": errors[i["id"]]} if i["id"] in errors else {"outputs": {"ok": 1}} for i in items]

    code, records = _session(handle, tmp_path, ["str", "ok", "surrogate"])
    assert code == 0
    assert records["ok"]["outputs"] == {"ok": 1}
    assert "error" in records["str"] and "error" in records["surrogate"]
    assert records["after"]["outputs"] == {"ok": 1}


@pytest.mark.logic
def test_a10_probe_aggregate_reply_over_the_line_budget_fails_no_item(tmp_path: Path) -> None:
    """Twenty items of 1 MB each do not kill the worker; every item still gets an answer."""

    def handle(state: object, items: list[dict], params: dict) -> list[dict]:  # type: ignore[type-arg]
        return [{"outputs": {"blob": "x" * 10**6}} for _ in items]

    ids = [str(i) for i in range(20)]
    code, records = _session(handle, tmp_path, ids)
    assert code == 0
    assert all(i in records and "outputs" in records[i] for i in ids)


@pytest.mark.logic
def test_a25_probe_bug_key_error_is_marked_a_gap() -> None:
    """A backend bug raising `KeyError` is not a plain content verdict: its record is marked a gap."""
    from senselab_worker.faults import error_record

    assert error_record(KeyError("missing"))["gap"] is True


@pytest.mark.logic
@pytest.mark.parametrize("line", [b'{"x": 1e999}\n', b'{"id": "a", "id": "b"}\n'])
def test_a34_probe_overflow_and_duplicate_keys_are_refused(line: bytes) -> None:
    """A number that overflows to infinity and a key repeated in one object are refused by `parse_line`."""
    from senselab_worker.codec import parse_line

    with pytest.raises(Exception):  # noqa: B017, PT011
        parse_line(line)


class _IntEnum(int, Enum):
    A = 1


def _lossy() -> dict[str, object]:
    import numpy

    return {
        "nat": numpy.datetime64("NaT"),
        "timedelta-ns": numpy.timedelta64(1, "ns"),
        "timedelta-ps": numpy.timedelta64(1, "ps"),
        "structured-void": numpy.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f8")])[0],
        "int-enum": _IntEnum.A,
    }


@pytest.mark.logic
@pytest.mark.parametrize("name", sorted(_lossy()))
def test_a4_probe_lossy_values_are_refused_by_the_canonical_encoder(name: str) -> None:
    """A value the canonical encoder would flatten into another value's bytes raises instead."""
    from senselab_worker.canonical import jcs_dumps

    with pytest.raises((TypeError, ValueError)):
        jcs_dumps({"a": _lossy()[name]})


@pytest.mark.logic
def test_a31_probe_content_root_covers_chunk_size(tmp_path: Path) -> None:
    """The same bytes read in different chunk sizes have different roots."""
    from senselab_worker.hashing import content_identity

    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert len({content_identity(path, chunk_size=n)["root"] for n in (3, 4, 2**20)}) == 3


@pytest.mark.logic
def test_a4_probe_masked_array_does_not_collide_with_the_plain_array() -> None:
    """A masked array is refused rather than encoded as its fill value, which equals a plain array's bytes."""
    import numpy
    from senselab_worker.canonical import canonical_npy_bytes

    masked = numpy.ma.masked_array([1, 2, 999999], mask=[False, False, True])
    with pytest.raises((TypeError, ValueError)):
        canonical_npy_bytes(masked)
