"""``logged`` and the worker hand-off of :mod:`~senselab.audio.workflows.triage.extend`.

``logged``: every row once, records in row order, the log complete, serially or across workers. A worker process
activates the parent's profile.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.triage.dataset_profile import active_profile
from senselab.audio.workflows.triage.extend import (
    SliceLog,
    activate_in_worker,
    config_for_workers,
    logged,
    pool_initargs,
    worker_failure,
)


def _rows(count: int) -> list[dict[str, Any]]:
    return [{"stem": f"row{index}"} for index in range(count)]


@pytest.mark.parametrize("workers", [1, 4])
def test_every_row_runs_once_and_records_keep_row_order(tmp_path: Path, workers: int) -> None:
    """Later rows finishing first still come back in the rows' order, and the log holds one line per row."""
    seen: list[str] = []
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        time.sleep(0.01 * (5 - int(row["stem"][3:]) % 5))
        with lock:
            seen.append(row["stem"])
        return {**row, "status": "ok"}

    rows = _rows(10)
    log = SliceLog(tmp_path / "slice.jsonl", slice_index=0, slice_count=1, total=len(rows), workers=workers)
    try:
        records = logged(rows, one, log, workers=workers)
    finally:
        log.close()
    assert [record["stem"] for record in records] == [row["stem"] for row in rows]
    assert sorted(seen) == sorted(row["stem"] for row in rows)
    lines = [json.loads(line) for line in (tmp_path / "slice.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(line["stem"] for line in lines) == sorted(row["stem"] for row in rows)


def test_workers_run_rows_concurrently() -> None:
    """With four workers, four rows are in flight at once."""
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return row

    logged(_rows(8), one, None, workers=4)
    assert peak == 4


class _Stop(Exception):
    pass


@pytest.mark.parametrize("workers", [1, 4])
def test_a_raising_row_stops_the_rows_not_yet_started(workers: int) -> None:
    """The exception reaches the caller, and queued rows are cancelled rather than run."""
    ran: list[str] = []
    lock = threading.Lock()

    def one(row: dict[str, Any]) -> dict[str, Any]:
        with lock:
            ran.append(row["stem"])
        if row["stem"] == "row0":
            raise _Stop("server gone")
        time.sleep(0.05)
        return {**row, "status": "ok"}

    with pytest.raises(_Stop):
        logged(_rows(40), one, None, workers=workers)
    assert len(ran) <= 2 * workers < 40


def test_a_worker_process_activates_the_parents_profile() -> None:
    """A spawned worker starts with no profile; the pool hands it the parent's, which it activates (#609-2)."""
    parent = active_profile()
    assert parent is not None
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn, initializer=activate_in_worker, initargs=pool_initargs()) as pool:
        failure = pool.submit(worker_failure).result()
        seen = pool.submit(active_profile).result()
    assert failure is None
    assert seen is not None and (seen.name, seen.root, seen.digest) == (parent.name, parent.root, parent.digest)


def test_a_worker_refuses_a_profile_whose_files_changed_after_the_parent_loaded_it() -> None:
    """The worker loads the profile again; a digest other than the parent's is held as the worker's failure."""
    override, root, _ = config_for_workers(None, active_profile())
    spawn = multiprocessing.get_context("spawn")
    stale = (override, root, "0" * 16)
    with ProcessPoolExecutor(1, mp_context=spawn, initializer=activate_in_worker, initargs=stale) as pool:
        failure = pool.submit(worker_failure).result()
    assert failure is not None and "not the parent's" in failure
