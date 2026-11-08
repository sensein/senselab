"""A worker's reported request device figures are recorded for the thread that ran the request."""

import json
import sys
from pathlib import Path

import pytest
from lifecycle_probes_fixture import interpreter
from senselab_core.runtime import request_reports
from senselab_core.runtime.worker_client import WorkerClient

SCRIPTED = Path(__file__).parent / "scripted_worker_fixture.py"


@pytest.mark.logic
def test_a_requests_device_figures_are_recorded_for_the_thread(tmp_path: Path) -> None:
    """A `done` reporting a device peak and allocation is recorded with the item count; one without is not."""
    measured = {"id": "$ID", "done": 2, "device_peak_bytes": 3000, "device_allocated_bytes": 1000}
    replies = [
        [json.dumps({"id": "$ID", "item": i, "outputs": {}}) for i in ("a", "b")] + [json.dumps(measured)],
        [json.dumps({"id": "$ID", "item": "c", "outputs": {}}), json.dumps({"id": "$ID", "done": 1})],
    ]
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    python = interpreter(tmp_path, f'exec "{sys.executable}" "{SCRIPTED}"')
    with WorkerClient(python, "unused", init={"replies": replies}, array_parent=arrays) as client:
        with request_reports.collected() as found:
            client.run([{"id": "a", "inputs": {}}, {"id": "b", "inputs": {}}], {})
            client.run([{"id": "c", "inputs": {}}], {})
    assert found == [(3000, 1000, 2)]
