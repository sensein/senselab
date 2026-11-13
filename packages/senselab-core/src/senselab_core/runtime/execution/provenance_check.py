"""The provenance completeness check of a run: `check_provenance`, `committed_keys` and `ProvenanceReport`.

ADR-010 states two rules, and the function checks exactly them. Every committed item has at least one execution or
one `senselab:Reuse`. A writer records at most one `senselab:Reuse` per item per attempt: a writer's document holds no
more Reuse of one key at one graph node than the run has jobs with that key and node. Several executions of one key
are allowed (one per execution context, and one per job of equal inputs), so they never fail it. `check_provenance`
reads the run directory alone, so a test, `senselab status --check-provenance` and a cluster driver call the one
function.
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


def _outcome_jobs(run_dir: Path) -> list[dict[str, object]]:
    """Return every job record of the `done` outcomes under `run_dir`'s queues, failed jobs excluded.

    A staged outcome (a dot-named temporary of an interrupted commit) is no outcome.
    """
    found: list[dict[str, object]] = []
    for outcome in sorted(p for p in (run_dir / RUN_QUEUES).glob("*/outcome/*") if not p.name.startswith(".")):
        try:
            record = json.loads(outcome.read_bytes())
        except (OSError, ValueError):
            continue
        jobs = record.get("jobs") if isinstance(record, dict) else None
        found += [j for j in jobs or [] if isinstance(j, dict) and "key_digest" in j and not j.get("failure")]
    return found


def committed_keys(run_dir: Path) -> set[str]:
    """Return the key digest of every job a result stands for in `run_dir`: pool outcomes and logged results.

    A pool item's `done` outcome record names its jobs, each with the failure it ended in or none; a failed job stored
    nothing and an item that ended `dead` has no jobs. The event log's `finished` and `cache_hit` lines name the jobs an
    in-process attempt gave.
    """
    keys = {str(j["key_digest"]) for j in _outcome_jobs(run_dir)}
    return keys | {e.key for e in read_events(run_dir) if e.kind in RESULTS and e.key is not None}


def _jobs_per_item(run_dir: Path) -> Counter[tuple[str, str]]:
    """Return how many committed jobs the run has per (key digest, graph node)."""
    pool: Counter[tuple[str, str]] = Counter()
    for job in _outcome_jobs(run_dir):
        ref = job.get("job")
        pool[(str(job["key_digest"]), str(ref.get("node", "")) if isinstance(ref, dict) else "")] += 1
    logged = Counter((e.key, e.node or "") for e in read_events(run_dir) if e.kind in RESULTS and e.key is not None)
    return pool | logged


def _reuse_items(data: bytes) -> list[tuple[str, str]]:
    """Return the (key digest, graph node) of each `senselab:Reuse` in one writer's document."""
    graph = json.loads(data).get("@graph", [])
    found: list[tuple[str, str]] = []
    for node in graph:
        if "senselab:Reuse" not in node.get("@type", []):
            continue
        key = LogicalKey.model_validate(kprov.literal_value(node["senselab:logicalKey"])).digest()
        roles = [a.get("prov:hadRole", {}) for a in node.get("prov:qualifiedAssociation", [])]
        found.append((key, str(next((r.get("senselab:nodeId") for r in roles if r.get("senselab:nodeId")), ""))))
    return found


def _key_of(record: kprov.Record) -> str:
    return LogicalKey.model_validate(kprov.literal_value(record.node()["senselab:logicalKey"])).digest()


def check_provenance(
    run_dir: Path, *, committed: Collection[str] | None = None, reuses: int | None = None
) -> ProvenanceReport:
    """Compare what `run_dir` committed with the activities in its `prov/`, by the two rules of ADR-010.

    Every committed key must have an execution or a Reuse, and no writer's document may hold more Reuse of one key at
    one graph node than the run has jobs with that key and node. With `reuses`, the run's `senselab:Reuse` activities
    must number `reuses`, the Reuse its resumes counted. Executions are never counted against a key.

    Args:
        run_dir: The run directory.
        committed: The key digests to account for; by default `committed_keys(run_dir)`.
        reuses: The number of Reuse activities the run's resumes reported, or None to leave the count unchecked.
    """
    wanted = committed_keys(run_dir) if committed is None else set(committed)
    problems: list[str] = []
    documents = ProvStore.read_documents(run_dir) if (run_dir / PROV_DIR).is_dir() else {}
    found = ProvStore.from_documents(documents).activities() if documents else []
    executions = [a for a in found if "senselab:Reuse" not in a.types()]
    reused = [a for a in found if "senselab:Reuse" in a.types()]
    served = {_key_of(a) for a in reused}
    problems += [
        f"key {k[:12]} has no execution and no Reuse"
        for k in sorted(wanted - {_key_of(a) for a in executions} - served)
    ]
    jobs = _jobs_per_item(run_dir)
    for name, data in documents.items():
        for (key, node), n in sorted(Counter(_reuse_items(data)).items()):
            if n > max(jobs[(key, node)], 1):
                problems.append(
                    f"writer {name[:8]} records {n} Reuse of key {key[:12]} at node {node!r}, jobs: {jobs[(key, node)]}"
                )
    if reuses is not None and len(reused) != reuses:
        problems.append(f"{len(reused)} Reuse activities, the resume reported {reuses}")
    return ProvenanceReport(len(wanted), len(executions), len(reused), tuple(problems))
