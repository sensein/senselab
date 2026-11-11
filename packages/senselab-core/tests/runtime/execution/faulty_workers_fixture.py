"""Fake workers over the real host worker: a request is made to fail by what it carries, and each start is counted."""

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import Admission
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.job import Emit, Job, JobOutcome, execute
from senselab_core.runtime.execution.queue import QueueSet
from senselab_core.runtime.execution.workers import WorkerLoadFailed, WorkerSet
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.worker_client import WorkerFatal, WorkerLost
from senselab_worker.faults import ProtocolError

FOREVER = 10**6
"""A poisoned text that fails every request it is sent in."""


def text_of(item: Mapping[str, Any]) -> str:
    """Return the text an item of a request carries, with the padding of `_texts` removed."""
    value = next(iter(item["inputs"].values()))
    return (value if isinstance(value, str) else str(getattr(value, "content", ""))).rstrip("x")


class Faulty(WorkerSet):
    """Workers whose request raises what `fail(texts)` returns, when it returns one; each start is counted.

    A `WorkerLost`, `WorkerFatal` or `ProtocolError` discards the worker, so the next checkout starts another:
    `starts` counts those. `sent` is the size of every request, `failed` the size of each that raised, and
    `checkouts` the number of checkouts.
    """

    def __init__(self, fail: Callable[[Sequence[str]], Exception | None]) -> None:
        """Raise what `fail` returns for the texts of a request."""
        super().__init__(1)
        self.fail = fail
        self.starts = self.checkouts = 0
        self.sent: list[int] = []
        self.failed: list[int] = []
        self.live = False

    @contextmanager
    def checkout(self, *args: Any, **kwargs: Any) -> Iterator[Any]:  # noqa: ANN401 - `checkout`'s own arguments
        """Hold the real worker, wrapped; one that was lost is replaced, and counted as a start."""
        self.checkouts += 1
        with super().checkout(*args, **kwargs) as worker:
            if not self.live:
                self.live, self.starts = True, self.starts + 1
            yield _Wrapped(worker, self)


class _Wrapped:
    def __init__(self, worker: Any, owner: Faulty) -> None:  # noqa: ANN401 - the real worker
        self.worker, self.owner = worker, owner

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        owner = self.owner
        owner.sent.append(len(items))
        error = owner.fail([text_of(i) for i in items])
        if error is None:
            return self.worker.run(items, params)  # type: ignore[no-any-return]
        owner.failed.append(len(items))
        if isinstance(error, (WorkerLost, WorkerFatal, ProtocolError)):
            owner.live = False
        raise error


def crashes(
    left: Mapping[str, int], error: Callable[[], Exception] = lambda: WorkerLost("died on its request")
) -> Faulty:
    """Return workers lost on a request carrying a text of `left`, that many times for each."""
    remaining = dict(left)

    def fail(texts: Sequence[str]) -> Exception | None:
        hit = [t for t in texts if remaining.get(t, 0) > 0]
        for t in hit:
            remaining[t] -= 1
        return error() if hit else None

    return Faulty(fail)


def sender(store: ResultStore, profile: ExecutionProfile, workers: WorkerSet) -> Any:  # noqa: ANN401 - a `Send`
    """Return a pool worker's `send` over `workers`: `execute`, taking an emit and the cancel flag."""

    def send(
        jobs: Sequence[Job],
        placement: Placement,
        emit: Emit | None = None,
        cancelled: threading.Event | None = None,
    ) -> list[JobOutcome]:
        return execute(
            jobs,
            placement,
            store=store,
            run=RunConfig(),
            profile=profile,
            workers=workers,
            emit=emit,
            cancelled=cancelled,
        )

    return send


def put_batched(placed: Plan, pool_dir: Path, profile: ExecutionProfile, count: int, queue: str = "cpu") -> None:
    """Put one item of the jobs `0..count-1` on `queue`, its placement batching them all in one request."""
    admission = Admission(id=f"run-{placed.hash}", plan_hash=placed.hash)
    fragment = node_fragment(placed, admission, elements=[(e,) for e in range(count)])
    update = {"resource_class": queue, "batch_size": count}
    fragment = fragment.model_copy(update={"placement": fragment.placement.model_copy(update=update)})
    QueueSet(pool_dir, profile, lease=profile.lease).queue(queue).put(fragment)


class Unloadable(WorkerSet):
    """Workers of `inner` whose `at`th checkout (counting from 1) raises `WorkerLoadFailed`, as a start that failed."""

    def __init__(self, inner: WorkerSet, *, at: int) -> None:
        """Wrap `inner`, failing checkout number `at`."""
        super().__init__(1)
        self.inner, self.at, self.checkouts = inner, at, 0

    @contextmanager
    def checkout(self, *args: Any, **kwargs: Any) -> Iterator[Any]:  # noqa: ANN401 - `checkout`'s own arguments
        """Raise on checkout number `at`, else hold `inner`'s worker."""
        self.checkouts += 1
        if self.checkouts == self.at:
            raise WorkerLoadFailed("WorkerFatal: the replacement worker did not load")
        with self.inner.checkout(*args, **kwargs) as worker:
            yield worker

    def close(self) -> None:
        """Close `inner` too."""
        self.inner.close()
        super().close()
