"""Cross-record batches of one node's jobs: a job's formation, and so its key, is its own, never its batch-mates'."""

import bisect
import functools
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, model_validator
from senselab_worker.units import NS_PER_S

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.signals import IrregularGroup, Recording, RegularGroup
from senselab_core.runtime.keys import BUCKET, SINGLE, fixed_formation
from senselab_core.runtime.storage import load_policy_values

if TYPE_CHECKING:
    from senselab_core.runtime.execution.job import Job

BATCHING_FORMAT = "batching@1.0"
"""The format tag of `data/batching.yaml`."""


class BucketGrid(Frozen):
    """Canonical length buckets: bucket `i` holds lengths in `[edges_ns[i-1], edges_ns[i])`, the last one unbounded."""

    id: NonEmptyStr
    edges_ns: tuple[Annotated[int, Field(strict=True, gt=0)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _increasing(self) -> "BucketGrid":
        if any(b <= a for a, b in zip(self.edges_ns, self.edges_ns[1:])):
            raise ValueError(f"bucket edges must be strictly increasing, got {self.edges_ns}")
        return self

    @classmethod
    @functools.cache
    def load_default(cls) -> "BucketGrid":
        """Return the grid in `data/batching.yaml`.

        Raises:
            ValueError: the file lacks the format or does not validate.
        """
        return cls.model_validate(load_policy_values("senselab_core.runtime", "batching.yaml", BATCHING_FORMAT))

    def formation(self, length_ns: int) -> str:
        """Return the formation `bucket:<id>:<index>` of the bucket holding `length_ns`."""
        return f"{BUCKET}:{self.id}:{bisect.bisect_right(self.edges_ns, length_ns)}"


@dataclass(frozen=True)
class Batch:
    """Jobs sent as one request, all of one formation."""

    jobs: tuple["Job", ...]
    formation: str


def _span_ns(group: RegularGroup | IrregularGroup) -> int:
    if isinstance(group, IrregularGroup):
        times = group.timestamps_ns
        return int(times[-1] - times[0]) if len(times) else 0
    return int(Fraction(group.frames) * NS_PER_S / Fraction(group.sampling_rate))


def recording_span_ns(value: object) -> int | None:
    """Return the longest span of `value`'s groups in ns when it is a recording, else None."""
    return max(_span_ns(g) for g in value.groups.values()) if isinstance(value, Recording) else None


def job_length_ns(job: "Job") -> int | None:
    """Return the longest span of `job`'s recording inputs in ns, None when it has none.

    A held input is measured; an upstream input is its ref's recorded `length_ns`, so no stored value is read. A
    gathered (list) input does not count.
    """
    held = [recording_span_ns(v) for port, v in job.values.items() if port not in job.upstream]
    read = [r.length_ns for up in job.upstream.values() if not up.gathered for r in up.refs]
    lengths = [n for n in (*held, *read) if n is not None]
    return max(lengths) if lengths else None


def formation(job: "Job", backend: BackendSpec, grid: BucketGrid) -> str:
    """Return the formation `job` is sent and keyed in: the backend's fixed one, else its length's bucket in `grid`.

    A job with no recording input (`job_length_ns`) to a backend that keys a bucket is `single`.
    """
    fixed = fixed_formation(backend)
    if fixed is not None:
        return fixed
    length = job_length_ns(job)
    return SINGLE if length is None else grid.formation(length)


def one_node(jobs: Sequence["Job"]) -> None:
    """Check that `jobs` are of one node of one plan.

    Raises:
        ValueError: they are not.
    """
    nodes = {(j.ref.plan_hash, j.ref.node) for j in jobs}
    if len(nodes) > 1:
        raise ValueError(f"a batch forms within one node of one plan, got {sorted(nodes)}")


def form_batches(jobs: Sequence["Job"], backend: BackendSpec, *, batch_size: int, grid: BucketGrid) -> list[Batch]:
    """Return `jobs` as batches of at most `batch_size` jobs of one formation each (`formation`).

    Formations come in the order of their first job, and jobs in their given order within one. A `single` batch
    holds one job.

    Raises:
        ValueError: `batch_size` is below 1, or the jobs are not of one plan and node.
    """
    if batch_size < 1:
        raise ValueError(f"a batch holds at least one job, got batch size {batch_size}")
    one_node(jobs)
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(formation(job, backend, grid), []).append(job)
    batches = []
    for kind, members in groups.items():
        size = 1 if kind == SINGLE else batch_size
        batches += [Batch(tuple(members[i : i + size]), kind) for i in range(0, len(members), size)]
    return batches
