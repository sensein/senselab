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

from senselab_worker.fake import FAULT_KEY, FakeFaults

from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, TaskParams, TaskSpec
from senselab_core.runtime.worker_client import WorkerClient

WORKER_MODULE = "senselab_worker.fake"


class EchoParams(TaskParams):
    """Params of `fake.echo`; `salt` enters every value."""

    salt: str = ""


ECHO = TaskSpec(
    name="fake.echo",
    version="1",
    kind="transform",
    inputs={"payload": "json"},
    outputs={"value": "sha256"},
    params=EchoParams,
)

FAKE = BackendSpec(
    task="fake.echo",
    name="fake",
    version="1",
    stack="fake",
    env=Env.host(),
    devices=("cpu",),
    batch_invariant=True,
    worker=WORKER_MODULE,
    load_stub={},
)

REGISTRY.add_task(ECHO)
REGISTRY.add_backend(FAKE)


def fault_param(faults: FakeFaults) -> dict[str, Any]:
    """Return the request param that makes the fake inject `faults`."""
    return {FAULT_KEY: {k: sorted(v) if isinstance(v, frozenset) else v for k, v in dataclasses.asdict(faults).items()}}


@contextlib.contextmanager
def fake_worker(*, array_parent: Path | None = None) -> Iterator[WorkerClient]:
    """Yield a client over a fresh fake worker; stop it on exit.

    The client's array directory is made under `array_parent` (the system temporary directory if
    `None`) and removed on exit.
    """
    with WorkerClient(sys.executable, WORKER_MODULE, init={}, array_parent=array_parent) as w:
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
    with fake_worker(array_parent=array_parent) as w:
        return w.run(items, {**params, **fault_param(faults)})
