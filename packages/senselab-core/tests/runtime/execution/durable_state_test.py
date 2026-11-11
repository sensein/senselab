"""Run state is written durably through `StorageDir`: every writer survives a kill at every step (ADR-003).

A fault is injected through the `FileOps` seam of `StorageDir`, at the write, the fsync of the file, the link
and the fsync of the directory. After each, the final name is absent or complete and its readers do not raise.
"""

import errno
import itertools
import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from senselab_core.runtime import hardware
from senselab_core.runtime.execution import run as run_module
from senselab_core.runtime.execution import substrates
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog, EventWriteFailed, read_events, scan_events
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import FileOps, StorageDir, using_file_ops

PLAN = "ab" * 32


class Killed(BaseException):
    """The process died at this step: no `except Exception` and no cleanup in the code under test sees it as a fault."""


@dataclass
class Faulty(FileOps):
    """A `FileOps` that records its calls, and fails the `n`-th (from 0) call of `step`.

    `short` makes a failing write take half its bytes first; `err` is raised, `Killed` when None.
    """

    step: str = ""
    n: int = 0
    short: bool = False
    err: OSError | None = None
    calls: list[str] = field(default_factory=list)
    fired: bool = False

    def _due(self, step: str) -> bool:
        index = self.calls.count(step)
        self.calls.append(step)
        return step == self.step and index == self.n

    def _fail(self) -> None:
        self.fired = True
        raise self.err or Killed()

    def write(self, fd: int, data: bytes) -> int:
        """Fail or pass this write."""
        if self._due("write"):
            if self.short and len(data) > 1:
                super().write(fd, data[: len(data) // 2])
            self._fail()
        return super().write(fd, data)

    def fsync(self, fd: int) -> None:
        """Fail or pass this fsync."""
        if self._due("fsync"):
            self._fail()
        super().fsync(fd)

    def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
        """Fail or pass this link."""
        if self._due("link"):
            self._fail()
        super().link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)


@dataclass
class Site:
    """One writer: it writes `final` under a directory, and `read` is what a reader does with it."""

    name: str
    write: Callable[[Path], object]
    final: Callable[[Path], Path]
    read: Callable[[Path], object]
    links: bool
    setup: Callable[[Path], object] | None = None


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


class _Placement:
    def model_dump(self, mode: str = "python") -> dict[str, str]:
        return {"nodes": "x"}


def _lease_section() -> Any:  # noqa: ANN401
    return ExecutionProfile.load().lease


def _attempt_file(run_dir: Path) -> Path:
    return run_dir / "epoch" / "1" / "attempt.json"


def _epoch_take(run_dir: Path) -> None:
    Epoch(run_dir).take(_lease_section())


def _epoch_bind(run_dir: Path) -> None:
    Epoch(run_dir).bind(1, "att-1", 1)


def _custom_profile() -> ExecutionProfile:
    return ExecutionProfile.load().model_copy(update={"name": "t24-custom-profile"})


def _dead(run_dir: Path) -> None:
    fault = FaultRecord(fault="content", type="X", message="m", gap=False, charged=True)
    job = JobRef(plan_hash=PLAN, node="a", element=())
    Pool._dead(SimpleNamespace(run_dir=run_dir), job, "c" * 64, fault)  # type: ignore[arg-type]


def _secret(run_dir: Path) -> None:
    def no_boot() -> str:
        raise OSError("no boot id")

    hardware.host_secret(run=lambda _: "", boot_id=no_boot)


SITES = [
    Site(
        "run-write", lambda d: run_module._write(d / "inputs.json", {"a": 1}), lambda d: d / "inputs.json", _json, False
    ),
    Site(
        "claim-ordinal",
        lambda d: run_module._claim_ordinal(d, "att-1"),
        lambda d: d / "attempts" / "1",
        lambda p: p.read_text(),
        True,
    ),
    Site(
        "claim-attempt",
        lambda d: run_module._claim_attempt(d, _Placement()),  # type: ignore[arg-type]
        lambda d: next((d / "private" / "placements").glob("*.json"), d / "none"),
        _json,
        True,
    ),
    Site(
        "write-once",
        lambda d: run_module._write_once(d / "plan.json", {"plan_hash": PLAN}),
        lambda d: d / "plan.json",
        _json,
        True,
    ),
    Site(
        "epoch-lease",
        lambda d: Epoch(d).take(_lease_section()),
        lambda d: d / "epoch" / "1" / "lease.json",
        _json,
        True,
    ),
    Site("epoch-attempt", _epoch_bind, _attempt_file, _json, True, setup=_epoch_take),
    Site("dead", _dead, lambda d: d / "dead" / f"{'c' * 64}.json", _json, True),
    Site(
        "profile",
        lambda d: substrates.profile_argument(_custom_profile(), d),
        lambda d: next((d / "private").glob("profile-*.json"), d / "none"),
        _json,
        False,
    ),
    Site(
        "worker-mark",
        substrates.mark_ready,
        lambda d: d / "workers" / substrates.worker_id(),
        lambda p: p.read_bytes(),
        False,
    ),
    Site("host-secret", _secret, lambda d: d / f"senselab-host-{os.getuid()}", lambda p: p.read_text(), True),
]


def _run(site: Site, directory: Path, ops: Faulty, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    final = site.final(directory)
    if site.setup is not None:
        site.setup(directory)
    with using_file_ops(ops):
        try:
            site.write(directory)
        except (Killed, OSError):
            pass
    return final


def _assert_complete_or_absent(path: Path, read: Callable[[Path], object]) -> None:
    if path.exists():
        assert path.stat().st_size > 0 or path.name == "ended"
        read(path)


@pytest.mark.logic
@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
@pytest.mark.parametrize("step", ["write", "fsync", "link", "fsync-dir"])
def test_a_kill_at_any_step_leaves_the_name_absent_or_complete(
    site: Site, step: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write, both fsyncs and the link are each killed in turn; the final name is never empty or torn."""
    if step == "link" and not site.links:
        pytest.skip("this writer renames; it has no link")
    ops = Faulty(step="fsync" if step.startswith("fsync") else step, n=1 if step == "fsync-dir" else 0)
    final = _run(site, tmp_path, ops, monkeypatch)
    assert ops.fired, f"{site.name} never reached {step} through StorageDir"
    _assert_complete_or_absent(final, site.read)
    if site.setup is not None:
        assert ops.calls.count("link") <= 1, "the fault ran under the setup, not the site"
        _assert_complete_or_absent(tmp_path / "epoch" / "1" / "lease.json", _json)
        assert (tmp_path / "epoch" / "1" / "lease.json").exists()


@pytest.mark.logic
@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_a_short_write_is_continued(site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write that takes half the bytes is followed by another for the rest: the file is whole."""

    class Halving(FileOps):
        def write(self, fd: int, data: bytes) -> int:
            return super().write(fd, data[: max(1, len(data) // 2)])

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    if site.setup is not None:
        site.setup(tmp_path)
    with using_file_ops(Halving()):
        site.write(tmp_path)
    final = site.final(tmp_path)
    assert final.exists()
    site.read(final)


@pytest.mark.logic
@pytest.mark.parametrize("site", [s for s in SITES if s.links], ids=lambda s: s.name)
def test_the_file_is_fsynced_before_it_is_visible_and_the_directory_after(
    site: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order of calls is write, fsync of the file, link, fsync of the directory."""
    ops = Faulty()
    _run(site, tmp_path, ops, monkeypatch)
    calls = ops.calls
    link = calls.index("link")
    assert "fsync" in calls[:link] and calls[link:].count("fsync") >= 1
    assert calls.index("write") < calls.index("fsync") < link


@pytest.mark.logic
def test_two_claims_of_one_name_leave_one_whole_file(tmp_path: Path) -> None:
    """A claim that loses keeps the winner's bytes, and no staging file is left behind."""
    directory = StorageDir.open_root(tmp_path)
    try:
        grammar = re.compile(r"^claim$")
        assert directory.publish("claim", b"first", grammar=grammar) is True
        assert directory.publish("claim", b"second", grammar=grammar) is False
        with pytest.raises(FileExistsError):
            directory.publish("claim", b"third", grammar=grammar, on_exists="raise")
    finally:
        directory.close()
    assert [p.name for p in tmp_path.iterdir()] == ["claim"] and (tmp_path / "claim").read_bytes() == b"first"


def _log(run_dir: Path) -> EventLog:
    ticks = itertools.count(1)
    return EventLog(run_dir, "a1", 1, clock=lambda: next(ticks))


def _record(log: EventLog, node: str = "a") -> None:
    log.record("queued", node, job=JobRef(plan_hash=PLAN, node=node, element=()))


@pytest.mark.logic
@pytest.mark.parametrize("appends_before", [0, 1, 2])
@pytest.mark.parametrize("short", [False, True])
def test_a_failed_append_leaves_the_complete_lines_and_raises(appends_before: int, short: bool, tmp_path: Path) -> None:
    """A write that fails, after taking half its bytes or none, is cut back; every reader sees whole lines only."""
    log = _log(tmp_path)
    for _ in range(appends_before):
        _record(log)
    ops = Faulty(step="write", n=0, short=short, err=OSError(errno.ENOSPC, "no space"))
    with using_file_ops(ops), pytest.raises(EventWriteFailed) as raised:
        _record(log)
    assert ops.fired and raised.value.errno == errno.ENOSPC and raised.value.seq == appends_before
    scan = scan_events(tmp_path)
    assert [e.seq for e in scan.events] == list(range(appends_before)) and scan.quarantined == ()
    _record(log)
    assert len(read_events(tmp_path)) == appends_before + 1


@pytest.mark.logic
@pytest.mark.parametrize("appends_before", [0, 1, 2])
def test_a_kill_mid_append_leaves_a_log_every_reader_reads(appends_before: int, tmp_path: Path) -> None:
    """A kill leaves half a line (nothing cuts it back): readers get the whole lines and one quarantined line."""
    log = _log(tmp_path)
    for _ in range(appends_before):
        _record(log)

    class Dead(Faulty):
        def ftruncate(self, fd: int, length: int) -> None:
            raise Killed

    with using_file_ops(Dead(step="write", short=True)), pytest.raises(Killed):
        _record(log)
    scan = scan_events(tmp_path)
    assert [e.seq for e in scan.events] == list(range(appends_before))
    assert [(q.writer, q.seq) for q in scan.quarantined] == [(log.writer, appends_before)]
    assert len(read_events(tmp_path)) == appends_before


@pytest.mark.logic
def test_an_append_after_a_torn_tail_starts_its_own_line(tmp_path: Path) -> None:
    """When the cut-back itself fails, the next append does not run into the torn line: one line is lost, no more."""
    log = _log(tmp_path)
    _record(log)

    class Stuck(Faulty):
        def ftruncate(self, fd: int, length: int) -> None:
            raise OSError(errno.EIO, "no truncate")

    with (
        using_file_ops(Stuck(step="write", short=True, err=OSError(errno.ENOSPC, "full"))),
        pytest.raises(EventWriteFailed),
    ):
        _record(log)
    _record(log)
    _record(log)
    scan = scan_events(tmp_path)
    assert [e.seq for e in scan.events] == [0, 2, 3] and [q.line for q in scan.quarantined] == [1]


@pytest.mark.logic
def test_every_reader_quarantines_bad_lines_wherever_they_sit(tmp_path: Path) -> None:
    """Junk in the first, middle and last position is skipped and counted by writer and line; no reader raises."""
    log = _log(tmp_path)
    for _ in range(3):
        _record(log)
    good = log.path.read_text().splitlines()
    log.path.write_text("\n".join(["{not json", good[0], '{"seq": 7}', good[1], "\u0000\u0000", good[2], "[1,"]) + "\n")
    scan = scan_events(tmp_path)
    assert [e.seq for e in scan.events] == [0, 1, 2]
    assert [(q.writer, q.seq, q.line) for q in scan.quarantined] == [
        (log.writer, 0, 0),
        (log.writer, 7, 2),
        (log.writer, 4, 4),
        (log.writer, 6, 6),
    ]
    assert [e.seq for e in read_events(tmp_path)] == [0, 1, 2]
    log.path.write_bytes(good[0].encode() + b"\n\xff\xfe\n")
    assert len(scan_events(tmp_path).quarantined) == 1


@pytest.mark.logic
def test_a_link_whose_reply_was_lost_is_this_callers_claim(tmp_path: Path) -> None:
    """The link is made but the call says EEXIST (NFS retransmit): the claim is won, and the bytes are this caller's."""

    class LostReply(FileOps):
        def link(self, src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
            super().link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            raise FileExistsError(errno.EEXIST, "File exists")

    with StorageDir.open_root(tmp_path) as directory, using_file_ops(LostReply()):
        grammar = re.compile(r"^claim$")
        assert directory.publish("claim", b"mine", grammar=grammar) is True
    assert (tmp_path / "claim").read_bytes() == b"mine"
    assert [p.name for p in tmp_path.iterdir()] == ["claim"]


@pytest.mark.logic
def test_a_name_another_writer_linked_is_still_lost(tmp_path: Path) -> None:
    """EEXIST over a file that is not this caller's temporary stays a lost claim."""
    (tmp_path / "claim").write_bytes(b"rival")
    with StorageDir.open_root(tmp_path) as directory:
        assert directory.publish("claim", b"mine", grammar=re.compile(r"^claim$")) is False
    assert (tmp_path / "claim").read_bytes() == b"rival"
