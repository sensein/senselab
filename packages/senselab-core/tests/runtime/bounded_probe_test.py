"""The shared probe limiter (H-R37): a finite source and a child under file-size, address-space and time limits."""

import errno
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from bounded_probe_fixture import MIB, FiniteSource, ProbeFailed, ProbeOverrun, run_bounded


def probe_echo(*args: str) -> list[str]:
    """Return the arguments, as a well-behaved probe does."""
    return list(args)


def probe_write_forever(path: str) -> dict[str, int]:
    """Append 1 MiB chunks to `path` until a write fails; report the errno and the size reached."""
    chunk = b"\0" * MIB
    with open(path, "wb") as out:
        try:
            while True:
                out.write(chunk)
                out.flush()
        except OSError as err:
            return {"errno": err.errno or 0, "size": os.path.getsize(path)}


def probe_sleep_forever() -> None:
    """Never return."""
    while True:
        time.sleep(1)


def probe_allocate(size: str) -> str:
    """Allocate `size` bytes; report whether it was refused."""
    try:
        bytearray(int(size))
    except MemoryError:
        return "refused"
    return "allocated"


def probe_raise() -> None:
    """Raise, so the child ends without a report."""
    raise RuntimeError("probe broke")


@pytest.mark.limits
@pytest.mark.parametrize("expected", [0, 1, 100, 3 * MIB + 5])
@pytest.mark.parametrize("ask", [-1, None, 1, 7, MIB, 1 << 40])
def test_a_finite_source_never_serves_past_its_bound(expected: int, ask: int | None) -> None:
    """Whatever the reader asks for, at most `expected + 1 MiB` bytes come out, then the source raises."""
    source = FiniteSource(expected)
    total = 0
    with pytest.raises(ProbeOverrun):
        for _ in range(10 * MIB + 10):
            total += len(source.read(ask))
    assert total == source.served == expected + MIB


@pytest.mark.limits
def test_a_probe_writing_without_end_is_stopped_at_the_file_size_limit(tmp_path: Path) -> None:
    """A child that writes forever gets EFBIG at the limit, and its file is no larger than the limit."""
    target = tmp_path / "grows"
    report = run_bounded("bounded_probe_test:probe_write_forever", str(target), fsize=4 * MIB, timeout=60)
    assert report["errno"] == errno.EFBIG and report["size"] <= 4 * MIB
    assert target.stat().st_size <= 4 * MIB


@pytest.mark.limits
def test_a_probe_that_never_returns_is_killed_at_the_timeout() -> None:
    """A child that never returns is killed once the timeout passes."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded("bounded_probe_test:probe_sleep_forever", timeout=2)
    assert time.monotonic() - started < 30


@pytest.mark.limits
@pytest.mark.skipif(sys.platform == "darwin", reason="macOS refuses an address-space limit below the current mapping")
def test_an_allocation_past_the_address_space_limit_is_refused_in_the_probe() -> None:
    """Under the address-space limit, an allocation larger than it fails inside the child."""
    assert run_bounded("bounded_probe_test:probe_allocate", str(8 << 30), address_space=2 << 30) == "refused"


@pytest.mark.limits
def test_a_probe_returns_its_report_and_a_broken_one_fails() -> None:
    """A probe's JSON value comes back; one that raises is `ProbeFailed` with its stderr."""
    assert run_bounded("bounded_probe_test:probe_echo", "a", "b") == ["a", "b"]
    with pytest.raises(ProbeFailed, match="probe broke"):
        run_bounded("bounded_probe_test:probe_raise")
