"""What a job records: its execution activity and agent, a reuse's attributes, and verify-sampling evidence."""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.identity import ExecutionContext, InputIdentity, LogicalKey, OutputIdentity
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.compare import Thresholds, compare
from senselab_core.runtime.equivalence import SAMPLING, EquivalenceEvidence, EvidenceSide
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import attempt_id
from senselab_core.runtime.store import ResultStore, StoredResult, agent_record, context_record, key_record
from senselab_core.runtime.vocabulary import core_vocabulary

logger = logging.getLogger("senselab")


def verify_attributes(run: RunConfig) -> dict[str, object]:
    """Return the verify settings every activity records: the fraction, how hits are sampled, the thresholds."""
    return {
        "senselab:verifyFraction": float(run.verify),
        "senselab:verifySampling": SAMPLING,
        "senselab:verifyThresholds": run.thresholds.canonical_digest(),
    }


def unlicensed(hit: StoredResult) -> dict[str, object]:
    """Return `senselab:unlicensedReuse` for a served result whose context records no env license, else nothing."""
    return {"senselab:unlicensedReuse": True} if hit.context.env_license is None else {}


def reuse_attributes(run: RunConfig, hit: StoredResult) -> dict[str, object]:
    """Return what a reuse activity records: the verify settings and, for a hit with no env license, its allowance."""
    return {**verify_attributes(run), **unlicensed(hit)}


def verify(  # noqa: PLR0913 - the hit, its recomputation and the thresholds judging them
    store: ResultStore,
    key: LogicalKey,
    hit: StoredResult,
    context: ExecutionContext,
    outputs: dict[str, object],
    thresholds: Thresholds,
) -> str:
    """Compare a sampled hit with its recomputation under `thresholds`, record the evidence and return its digest.

    A disagreement splits the store's class of the two members (`ResultStore.add_evidence`) and is logged with the
    paths that differ, never their values.
    """
    comparison = compare(hit.outputs, outputs, thresholds)
    stored, current = EvidenceSide.of(hit.context), EvidenceSide.of(context)
    evidence = EquivalenceEvidence(
        task=key.task,
        backend=key.backend,
        key=key.digest(),
        stored=stored,
        current=current,
        mismatched=comparison.differences,
        thresholds=comparison.thresholds,
        run=attempt_id(),
    )
    store.add_evidence(evidence)
    if not comparison.agree:
        logger.warning(
            "verify: %s on %s disagrees with its result from %s in %s",
            key.backend,
            current.member(),
            stored.member(),
            ", ".join(comparison.paths),
        )
    return evidence.canonical_digest()


def execution_records(  # noqa: PLR0913 - one execution's records
    key: LogicalKey,
    context: ExecutionContext,
    profile: ExecutionProfile,
    used: Sequence[tuple[str, InputIdentity]],
    output: OutputIdentity,
    started: datetime,
    attributes: Mapping[str, object],
) -> tuple[kprov.Record, kprov.Record]:
    """Return the execution's agent and its activity, which used each `(port, entity)` and generated the output.

    `attributes` are recorded on the activity beside the key, the context and the profile's digest.
    """
    agent = agent_record(key, context)
    activity = kprov.activity(
        core_vocabulary(),
        attributes={
            **attributes,
            "senselab:logicalKey": key_record(key),
            "senselab:context": context_record(context),
            "senselab:executionProfile": profile.canonical_digest(),
        },
        used=[kprov.Usage(identity, port=port) for port, identity in used],
        generated=[kprov.Generation(output)],
        associations=[kprov.Association(agent.id, node_id=key.task)],
        started=started,
        ended=datetime.now(timezone.utc),
    )
    return agent, activity
