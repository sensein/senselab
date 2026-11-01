"""The `fake` stack: task `fake.echo` and its backend `fake`, run by the `senselab_worker.fake` worker.

Importing this module declares both in `REGISTRY`. Each output `value` is the sha256 of the JCS of the
item's inputs and the params; `FakeFaults` injects the declared fault model.
"""

import contextlib
import dataclasses
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from senselab_worker.fake import FakeFaults

from senselab_core.kernel.declare import REGISTRY, BackendSpec, TaskParams, TaskSpec
from senselab_core.kernel.license import License
from senselab_core.runtime.worker_client import WorkerClient

WORKER_MODULE = "senselab_worker.fake"


class EchoParams(TaskParams):
    """Params of `fake.echo`; `salt` enters every value."""

    salt: str = ""


ECHO = TaskSpec(
    name="fake.echo",
    version="1",
    kind="measure",
    inputs={"payload": "json"},
    outputs={"value": "sha256"},
    params=EchoParams,
)

FAKE = BackendSpec(
    task="fake.echo",
    name="fake",
    version="1",
    stack="fake",
    env="host",
    devices=("cpu",),
    license=License(
        expression="Apache-2.0",
        commercial_use="yes",
        copyleft="none",
        attribution=True,
        gated=False,
        source="senselab-worker pyproject.toml license",
        verified={"date": "2026-10-30", "how": "pep639"},
    ),
)

REGISTRY.add_task(ECHO)
REGISTRY.add_backend(FAKE)


def faults_json(faults: FakeFaults) -> dict[str, Any]:
    """Return `faults` as the JSON object `FakeFaults.from_json` reads."""
    return {k: sorted(v) if isinstance(v, frozenset) else v for k, v in dataclasses.asdict(faults).items()}


@contextlib.contextmanager
def fake_worker(faults: FakeFaults, *, array_parent: Path | None = None) -> Iterator[WorkerClient]:
    """Yield a client over a fresh fake worker injecting `faults`; stop it on exit.

    The client's array directory is made under `array_parent` (the system temporary directory if
    `None`) and removed on exit.
    """
    init = {"faults": faults_json(faults)}
    with WorkerClient(sys.executable, WORKER_MODULE, init=init, array_parent=array_parent) as w:
        yield w


def run_fake(
    items: list[dict[str, Any]],
    params: dict[str, Any],
    faults: FakeFaults,
    *,
    array_parent: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `items` through a fresh fake worker with `faults` and return one result per item.

    Results are as `WorkerClient.run` returns them. The worker is stopped and its array directory,
    made under `array_parent`, removed before returning.

    Raises:
        ProtocolError, WorkerLost, WorkerFatal: infrastructure faults, such as a `crash_after` crash.
    """
    with fake_worker(faults, array_parent=array_parent) as w:
        return w.run(items, params)
