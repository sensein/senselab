"""The `fake` stack: its registration and `run_fake` over the real fake worker."""

from pathlib import Path

import pytest
from senselab_core.kernel.declare import REGISTRY
from senselab_core.runtime.fake_backend import ECHO, FAKE, EchoParams, fake_worker, run_fake
from senselab_worker.canonical import jcs_dumps
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import FaultClass
from senselab_worker.hashing import sha256_bytes


@pytest.mark.logic
def test_import_registers_echo_and_its_backend() -> None:
    """Importing the module declares `fake.echo` and its `fake` backend in the global registry."""
    assert ECHO in REGISTRY.tasks()
    assert REGISTRY.backends("fake.echo") == [FAKE]
    assert (FAKE.stack, FAKE.env, FAKE.license.expression) == ("fake", "host", "Apache-2.0")


@pytest.mark.logic
def test_run_fake_hashes_inputs_and_params(tmp_path: Path) -> None:
    """Each value is the sha256 of the JCS of the item's inputs and the params; the array dir is removed."""
    items = [{"id": "a", "inputs": {"payload": "x"}}, {"id": "b", "inputs": {"payload": "y"}}]
    params = EchoParams(salt="s").model_dump()
    results = run_fake(items, params, FakeFaults(), array_parent=tmp_path)
    expected = [sha256_bytes(jcs_dumps({"inputs": i["inputs"], "params": params})) for i in items]
    assert [r["outputs"]["value"] for r in results] == expected
    assert list(tmp_path.iterdir()) == []


@pytest.mark.logic
def test_run_fake_injects_declared_faults() -> None:
    """Faults reach the worker: a listed item fails with a content fault, an OOM item with a resource fault."""
    items = [{"id": i, "inputs": {"payload": i}} for i in "abc"]
    results = run_fake(items, {}, FakeFaults(fail_items=frozenset({"a"}), oom_items=frozenset({"b"})))
    assert [r.get("error", {}).get("fault") for r in results] == [FaultClass.CONTENT, FaultClass.RESOURCE, None]


@pytest.mark.logic
def test_fake_worker_process_has_exited_on_exit(tmp_path: Path) -> None:
    """Leaving `fake_worker`, which `run_fake` runs inside, reaps the worker process and removes its array dir."""
    with fake_worker(FakeFaults(), array_parent=tmp_path) as client:
        client.run([{"id": "a", "inputs": {"payload": "x"}}], {})
        assert client.returncode is None
    assert client.returncode is not None
    assert list(tmp_path.iterdir()) == []
