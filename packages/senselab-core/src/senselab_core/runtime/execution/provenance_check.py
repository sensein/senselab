"""The provenance completeness check of a run: `check_provenance`, `committed_keys` and `ProvenanceReport`.

A run's committed items each have an execution or a Reuse in its `prov/` (ADR-010), and no item has two executions.
`check_provenance` reads the run directory alone, so a test, `senselab status --check-provenance` and a cluster driver
call the one function.
"""

import json
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.identity import LogicalKey
from senselab_core.runtime.execution.events import RESULTS, read_events
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.provenance import PROV_DIR, ProvStore


@dataclass(frozen=True)
class ProvenanceReport:
    """What `check_provenance` found: the counts it compared and each problem as one sentence."""

    committed: int
    executions: int
    reuses: int
    problems: tuple[str, ...] = field(default=())

    @property
    def complete(self) -> bool:
        """Whether nothing is wrong."""
        return not self.problems

    def require(self) -> None:
        """Raise `ProvenanceIncomplete` naming every problem, unless there is none."""
        if self.problems:
            raise ProvenanceIncomplete(self)


class ProvenanceIncomplete(AssertionError):
    """A run's provenance does not account for what it committed."""

    def __init__(self, report: ProvenanceReport) -> None:
        """Name `report`'s problems and its counts."""
        super().__init__(
            f"provenance incomplete ({report.committed} committed keys, {report.executions} executions, "
            f"{report.reuses} reuses): " + "; ".join(report.problems)
        )
        self.report = report


def committed_keys(run_dir: Path) -> set[str]:
    """Return the key digest of every job a result stands for in `run_dir`: pool outcomes and logged results.

    A pool item's `done` outcome record names its jobs, each with the failure it ended in or none; a failed job stored
    nothing and an item that ended `dead` has no jobs. The event log's `finished` and `cache_hit` lines name the jobs an
    in-process attempt gave.
    """
    keys: set[str] = set()
    for outcome in sorted((run_dir / RUN_QUEUES).glob("*/outcome/*")):
        try:
            record = json.loads(outcome.read_bytes())
        except (OSError, ValueError):
            continue
        jobs = record.get("jobs") if isinstance(record, dict) else None
        keys.update(
            str(j["key_digest"])
            for j in jobs or []
            if isinstance(j, dict) and "key_digest" in j and not j.get("failure")
        )
    keys.update(e.key for e in read_events(run_dir) if e.kind in RESULTS and e.key is not None)
    return keys


def _key_of(record: kprov.Record) -> str:
    return LogicalKey.model_validate(kprov.literal_value(record.node()["senselab:logicalKey"])).digest()


def check_provenance(
    run_dir: Path, *, committed: Collection[str] | None = None, reuses: int | None = None
) -> ProvenanceReport:
    """Compare what `run_dir` committed with the activities in its `prov/`.

    Every committed key must have an execution or a Reuse, and at most one execution. With `reuses`, the run's
    `senselab:Reuse` activities must number `reuses`, the Reuse its resumes counted.

    Args:
        run_dir: The run directory.
        committed: The key digests to account for; by default `committed_keys(run_dir)`.
        reuses: The number of Reuse activities the run's resumes reported, or None to leave the count unchecked.
    """
    wanted = committed_keys(run_dir) if committed is None else set(committed)
    problems: list[str] = []
    found = ProvStore.read(run_dir).activities() if (run_dir / PROV_DIR).is_dir() else []
    executions = [a for a in found if "senselab:Reuse" not in a.types()]
    reused = [a for a in found if "senselab:Reuse" in a.types()]
    ran = Counter(_key_of(a) for a in executions)
    served = {_key_of(a) for a in reused}
    problems += [f"key {k[:12]} has no execution and no Reuse" for k in sorted(wanted - ran.keys() - served)]
    problems += [f"key {k[:12]} has {n} executions" for k, n in sorted(ran.items()) if n > 1]
    if reuses is not None and len(reused) != reuses:
        problems.append(f"{len(reused)} Reuse activities, the resume reported {reuses}")
    return ProvenanceReport(len(wanted), len(executions), len(reused), tuple(problems))
