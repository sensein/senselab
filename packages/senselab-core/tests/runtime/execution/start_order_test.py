"""The order of `start`: an epoch is current only once its coordinator is bound, so a failed start changes nothing."""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from runner_test import (  # noqa: F401 - `packaged` is the autouse fixture of the fake stack
    CHAIN_EDGES,
    CHAINED,
    Recording,
    _graph,
    _plan,
    packaged,
)
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog, read_events, run_dir_of
from senselab_core.runtime.execution.run import Run
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.storage import FileOps, using_file_ops
from senselab_core.runtime.store import ResultStore

pytest.importorskip("pydra")
pytestmark = pytest.mark.contract

GRAPH = _graph(CHAINED, CHAIN_EDGES, text="list[json]")
ROWS = [{"text": "one"}]


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The run directory and the default result store are under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    yield tmp_path / "cache"


class Refusing(FileOps):
    """Refuses, once, the write of a file whose bytes contain `marker`."""

    def __init__(self, marker: bytes) -> None:
        """Fail the first write of `marker`."""
        self.marker, self.spent = marker, False

    def write(self, fd: int, data: bytes) -> int:
        """Raise `OSError` the first time `data` holds the marker."""
        if self.marker in data and not self.spent:
            self.spent = True
            raise OSError(28, "no space left on device")
        return super().write(fd, data)


def _finished(tmp_path: Path) -> tuple[Plan, ResultStore]:
    """Run the one-job plan to its end under epoch 1; return the plan and the store."""
    placed, _ = _plan(GRAPH, ROWS)
    store = ResultStore(tmp_path / "store")
    assert placed.run(Recording(store), store=store).wait(60).done
    return placed, store


class Unbindable(Recording):
    """An executor that refuses to bind an epoch."""

    def bind_epoch(self, epoch: int, events: EventLog) -> None:
        """Raise as a coordinator that cannot take the epoch."""
        raise ValueError("this executor cannot bind an epoch")


def _another_plan(run_dir: Path) -> None:
    path = run_dir / "plan.json"
    document = json.loads(path.read_text())
    document["plan_hash"] = "f" * 64
    path.write_text(json.dumps(document))


def _unreadable_queues(run_dir: Path) -> None:
    (run_dir / "queues").chmod(0)


STEPS = {
    "plan-written-once": (_another_plan, None, ValueError, False),
    "attempt-claimed": (None, b"placement@3", OSError, False),
    "epoch-lease-published": (None, b"run-lease@1", OSError, True),
    "queues-reconciled": (_unreadable_queues, None, PermissionError, True),
    "epoch-bound-to-executor": (None, None, ValueError, True),
    "epoch-bound": (None, b'{"attempt": ', OSError, True),
}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
@pytest.mark.parametrize("step", sorted(STEPS))
def test_a_start_that_fails_at_a_step_leaves_the_runs_epoch_and_status_as_they_were(tmp_path: Path, step: str) -> None:
    """Whichever step fails, epoch 1 stays current, the finished run stays done and its stop reading is unchanged."""
    placed, store = _finished(tmp_path)
    run_dir = run_dir_of(placed.hash)
    before = (Epoch(run_dir).current(), Epoch(run_dir).stopped(), Run(run_dir).status().done)
    prepare, marker, error, ended = STEPS[step]
    if prepare is not None:
        prepare(run_dir)
    executor = Unbindable(store) if step == "epoch-bound-to-executor" else Recording(store)
    refusing = Refusing(marker) if marker is not None else FileOps()
    try:
        with using_file_ops(refusing), pytest.raises(error):
            placed.run(executor, store=store).wait(60)
    finally:
        (run_dir / "queues").chmod(0o770)
    assert before == (1, True, True)
    assert (Epoch(run_dir).current(), Epoch(run_dir).stopped(), Run(run_dir).status().done) == before
    assert (run_dir / "epoch" / "2" / "ended").exists() == ended


def test_two_coordinators_taking_epochs_fence_only_the_older_whichever_binds_first(tmp_path: Path) -> None:
    """A takes 1 and B takes 2: with either bind order exactly one is fenced, and none while only one has bound."""
    for order in ((0, 1), (1, 0)):
        epochs = Epoch(tmp_path / f"run{order[0]}")
        taken = [epochs.take(), epochs.take()]
        assert taken == [1, 2] and epochs.current() == 0
        epochs.bind(taken[order[0]], "a", 1)
        assert not any(epochs.current() > n for n in taken[order[0] : order[0] + 1])
        epochs.bind(taken[order[1]], "b", 2)
        assert [epochs.current() > n for n in taken] == [True, False]


def test_an_epoch_taken_and_never_bound_is_not_current_and_ages_out(tmp_path: Path) -> None:
    """A first start that died after `mkdir epoch/1`: no epoch is current, and it is judged by its directory."""
    epochs = Epoch(tmp_path / "run")
    n = epochs.take()
    assert n == 1 and epochs.current() == 0 and not epochs.stopped()
    (epochs.root / "1" / "ended").write_bytes(b"")
    assert epochs.stopped() and not read_events(tmp_path / "run")
