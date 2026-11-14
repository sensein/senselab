"""The `crisperwhisper` lock built into a real env, and the wire protocol round-tripped through its worker (ORCD)."""

from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.kernel.declare import Env
from senselab_core.runtime.envs import Uv, ensure_env
from senselab_core.runtime.profiles import Envs, ExecutionProfile
from senselab_core.runtime.worker_client import WorkerClient
from senselab_worker import fake

pytestmark = [pytest.mark.env, pytest.mark.model, pytest.mark.network]

LOCK = Path(__file__).parents[4] / "packages/senselab/src/senselab/data/env_locks/crisperwhisper.txt"


def _items() -> list[dict[str, Any]]:
    arrays = {
        f"a_{dtype}": numpy.arange(6).astype(dtype).reshape(2, 3)
        for dtype in ("<f4", "<f8", "<i4", "<i8", "|u1", "|b1")
    }
    scalars = {
        "none": None,
        "bool": True,
        "int": 7,
        "float": 1.5,
        "str": "héllo",
        "list": [1, "a", None],
        "map": {"k": [2]},
    }
    return [
        {"id": "arrays", "inputs": arrays},
        {"id": "scalars", "inputs": scalars},
        {"id": "mixed", "inputs": {**scalars, **arrays}},
    ]


def test_envs_crisperwhisper_lock_builds_and_round_trips_every_wire_type(tmp_path: Path) -> None:
    """The lock builds under `ensure_env`; scalars, containers and every array dtype survive the env's worker."""
    handle = ensure_env(
        Env.isolated("crisperwhisper"),
        LOCK,
        tmp_path / "envs",
        settings=Envs(server_start_margin_s=60),
        lease=ExecutionProfile.load().lease,
        uv=Uv.find(),
    )
    assert handle.python_version and handle.lock_sha and handle.wheel_set
    items = _items()
    state = fake.load({})
    expected = fake.handle(state, items, {})
    arrays = tmp_path / "arrays"
    arrays.mkdir()
    client = WorkerClient(handle.python, "senselab_worker.fake", init={}, array_parent=arrays)
    try:
        got = client.run(items, {})
    finally:
        client.close()
    assert [r["id"] for r in got] == [i["id"] for i in items]
    assert [r["outputs"] for r in got] == [r["outputs"] for r in expected]
