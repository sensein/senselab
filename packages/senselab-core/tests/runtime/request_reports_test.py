"""A worker's reported request device figures are recorded for the thread that ran the request."""

import json
import sys
from pathlib import Path

import pytest
from lifecycle_probes_fixture import interpreter
from senselab_core.runtime import request_reports
from senselab_core.runtime.worker_client import WorkerClient

SCRIPTED = Path(__file__).parent / "scripted_worker_fixture.py"


def _scripted(tmp_path: Path, replies: list[list[str]]) -> WorkerClient:
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    python = interpreter(tmp_path, f'exec "{sys.executable}" "{SCRIPTED}"')
    return WorkerClient(python, "unused", init={"replies": replies}, array_parent=arrays)


@pytest.mark.logic
def test_a_requests_figures_are_reported_for_the_thread(tmp_path: Path) -> None:
    """A `done` reporting a device peak, allocation and load gives one report per request, as the worker said it."""
    first = {"id": "$ID", "done": 2, "device_peak_bytes": 3000, "device_allocated_bytes": 1000, "since_load": 0}
    second = {"id": "$ID", "done": 1, "since_load": 1}
    replies = [
        [json.dumps({"id": "$ID", "item": i, "outputs": {}}) for i in ("a", "b")]
        + [json.dumps(first | {"load_s": 2.5})],
        [json.dumps({"id": "$ID", "item": "c", "outputs": {}}), json.dumps(second)],
    ]
    with _scripted(tmp_path, replies) as client, request_reports.collected() as found:
        client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
        client.run([{"id": "c", "inputs": {}}], {})
    assert [(r.items, r.since_load, r.load_s, r.device) for r in found] == [
        (2, 0, 2.5, (3000, 1000)),
        (1, 1, None, None),
    ]


@pytest.mark.logic
def test_an_opened_request_carries_its_tags_and_wall_and_a_failed_one_leaves_none() -> None:
    """A recorded request keeps its tags and a wall; a block that raises, or records nothing, leaves none."""
    with request_reports.collected() as found:
        with request_reports.request(("a", "b")):
            request_reports.record(2, since_load=3)
        with request_reports.request(("c",)):
            pass
        with pytest.raises(RuntimeError), request_reports.request(("d",)):
            request_reports.record(1)
            raise RuntimeError("the worker was lost")
        request_reports.record(1)
    assert [(r.tags, r.items) for r in found] == [(("a", "b"), 2), ((), 1)]
    assert found[0].wall_s is not None and found[0].wall_s >= 0 and found[1].wall_s is None
