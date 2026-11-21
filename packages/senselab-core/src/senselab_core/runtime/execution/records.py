"""What a job records: its execution activity and agent, and the verify settings every activity carries."""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.identity import ExecutionContext, InputIdentity, LogicalKey, OutputIdentity
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.equivalence import SAMPLING
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.store import (
    agent_record,
    context_record,
    key_record,
)
from senselab_core.runtime.vocabulary import core_vocabulary

logger = logging.getLogger("senselab")


def add_plan_entity(provenance: ProvStore, plan_hash: str) -> None:
    """Add the `senselab:Plan` entity `plan:<plan_hash>` that the activities of the plan's jobs name."""
    provenance.add(kprov.plan(plan_hash, core_vocabulary()))


def plan_link(agent: kprov.Record, job: JobRef) -> kprov.Association:
    """Return the association of `job`'s activity with `agent`: its role the job's node, its plan `plan:<hash>`."""
    return kprov.Association(agent.id, node_id=job.node, plan=f"plan:{job.plan_hash}")


def verify_attributes(run: RunConfig) -> dict[str, object]:
    """Return the verify settings every activity records: the fraction, how hits are sampled, the thresholds."""
    return {
        "senselab:verifyFraction": float(run.verify),
        "senselab:verifySampling": SAMPLING,
        "senselab:verifyThresholds": run.thresholds.canonical_digest(),
    }


def execution_records(  # noqa: PLR0913 - one execution's records
    key: LogicalKey,
    context: ExecutionContext,
    profile_digest: str,
    used: Sequence[tuple[str, InputIdentity]],
    output: OutputIdentity,
    started: datetime,
    attributes: Mapping[str, object],
    job: JobRef,
) -> tuple[kprov.Record, kprov.Record]:
    """Return the execution's agent and its activity, which used each `(port, entity)` and generated the output.

    `attributes` are recorded on the activity beside the key, the context and the execution profile's digest. The
    activity's association names `job`'s node and its plan (`plan_link`).
    """
    agent = agent_record(key, context)
    activity = kprov.activity(
        core_vocabulary(),
        attributes={
            **attributes,
            "senselab:logicalKey": key_record(key),
            "senselab:context": context_record(context),
            "senselab:executionProfile": profile_digest,
        },
        used=[kprov.Usage(identity, port=port) for port, identity in used],
        generated=[kprov.Generation(output)],
        associations=[plan_link(agent, job)],
        started=started,
        ended=datetime.now(timezone.utc),
    )
    return agent, activity
