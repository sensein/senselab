"""senselab's results on a pydra `PortableStore`: the job a store entry is of, its identity, reuse rule and records.

`open_store` opens a fresh store at a root every node sees: entries keyed by the senselab logical key's digest
(`logical-key@7`), slotted by the execution context's fingerprint (`context@5`), the `pool` and `senselab` sections
of a context recorded and never fingerprinted. `SenselabReuse` admits stored entries by the run's reuse policy and
measured equivalence classes, refuses a result whose env license the run's policy refuses, and samples
cross-context hits for verification. Each executed entry's activity is written to its run's provenance before the
entry is renamed into place (`record_execution`, the store's `on_commit`), and each reuse when it is served.
"""

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydra.engine.portable.reuse import Candidate, Verdict
from pydra.engine.portable.store import PortableStore
from senselab_worker.canonical import jcs_dumps

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.identity import (
    NO_EQUIVALENCE,
    Attestation,
    Equivalence,
    ExecutionContext,
    InputIdentity,
    LogicalKey,
    ReusePolicy,
    acceptable,
)
from senselab_core.kernel.license import License
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.compare import compare
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.equivalence import (
    EquivalenceEvidence,
    EvidenceSide,
    equivalence_digest,
    read_evidence,
)
from senselab_core.runtime.equivalence import split as split_class
from senselab_core.runtime.execution.events import run_dir_of
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.job import Job, job_context
from senselab_core.runtime.execution.records import execution_records, plan_link, verify_attributes
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.provenance import ProvStore
from senselab_core.runtime.servable import Servable, StoredContext, servable
from senselab_core.runtime.store import agent_record, context_record, key_record, output_identity
from senselab_core.runtime.vocabulary import core_vocabulary

if TYPE_CHECKING:
    from pydra.engine.job import Job as PydraJob
    from pydra.engine.portable.entry import StoredEntry

logger = logging.getLogger("senselab")

JOB = "senselab"
"""The recorded-only context section naming the run job a context is of: plan, node, element, attempt, inputs used."""
RECORDED_ONLY = frozenset({"pool", JOB})
"""The context sections an entry records and its slot never splits on."""
EVIDENCE = "equivalence-evidence"
"""The store sidecar holding one verify-sampling comparison per file, named by its digest."""
OUTPUTS = "outputs"
"""The one output of a pool job: the stored outputs mapping (`as_stored`)."""


@dataclass(frozen=True)
class JobDoc:
    """One job as a pool worker runs it: the job and its resolved inputs, where it runs, its key and its context.

    `context` is the lookup context, built where the job is submitted; the execute context is built where it runs
    (`execute_context`). `used` names each input entity by port, for the execution's activity; `attempt` is the run
    attempt that submitted the job.
    """

    job: Job
    placement: Placement
    env: EnvHandle
    expected: Attestation | None
    profile: ExecutionProfile
    code_license: License | None
    key: LogicalKey
    context: ExecutionContext
    run: RunConfig
    used: tuple[tuple[str, InputIdentity], ...]
    attempt: str
    equivalence: Equivalence = field(default_factory=lambda: NO_EQUIVALENCE)

    @property
    def ref(self) -> JobRef:
        """The job's reference in its plan."""
        return self.job.ref

    @property
    def host(self) -> bool:
        """Whether the backend runs in the pool worker's own process."""
        return self.placement.backend.env.kind == "host"

    def execute_context(self) -> ExecutionContext:
        """Return the context of running the job in this process: its env, host, numerics and decoder."""
        return job_context(self.placement, self.env, self.expected, self.profile, self.job, self.code_license)

    def section(self) -> dict[str, Any]:
        """Return the recorded-only `senselab` section: the job's plan, node, element, attempt and inputs used."""
        ref = self.ref
        return {
            "plan": ref.plan_hash,
            "node": ref.node,
            "element": list(ref.element),
            "attempt": self.attempt,
            "used": [[port, identity] for port, identity in self.used],
        }


def doc_of(task: object) -> JobDoc:
    """Return the `JobDoc` a pool task carries as its `job` input.

    Raises:
        TypeError: the task is no senselab pool task.
    """
    doc = getattr(task, "job", None)
    if not isinstance(doc, JobDoc):
        raise TypeError(f"a senselab store holds senselab pool jobs only, got {type(task).__name__}")
    return doc


def stored_context(context: Mapping[str, Any]) -> ExecutionContext:
    """Return a stored or carried context as an `ExecutionContext`, its recorded-only sections left out."""
    return ExecutionContext.model_validate({k: v for k, v in context.items() if k not in RECORDED_ONLY})


_RAN: dict[str, dict[str, str | None]] = {}
_VERIFIED: dict[str, str] = {}
_RAN_LOCK = threading.Lock()


def ran(checksum: str, started: datetime, output: str) -> None:
    """Note, in the process that ran job `checksum`, when its call started and ended and its output identity."""
    with _RAN_LOCK:
        _RAN[checksum] = {"started": started.isoformat(), "ended": _now(), "output": output}
        _VERIFIED.pop(checksum, None)


def _verified(checksum: str, evidence: str) -> None:
    with _RAN_LOCK:
        _VERIFIED[checksum] = evidence


def _verifies(checksum: str) -> str | None:
    with _RAN_LOCK:
        return _VERIFIED.pop(checksum, None)


def _ran(checksum: str) -> dict[str, str | None]:
    with _RAN_LOCK:
        found = _RAN.pop(checksum, None)
    return found if found is not None else {"started": _now(), "ended": _now(), "output": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SenselabIdentity:
    """`senselab-identity@1`: the key is the job's `LogicalKey`, the slot its context's `context@5` fingerprint."""

    recorded_only: frozenset[str] = RECORDED_ONLY
    name: ClassVar[str] = "senselab-identity@1"
    fingerprint_name: ClassVar[str] = "context@5"

    def key(self, job: "PydraJob") -> dict[str, Any]:
        """Return the record the job's checksum digests: its logical key's fields."""
        record = key_record(doc_of(job.task).key)
        assert isinstance(record, dict)
        return record

    def check_key(self, checksum: str, key: Mapping[str, Any]) -> bool:
        """Whether `key` is a logical key whose digest is `checksum`."""
        try:
            return LogicalKey.model_validate(dict(key)).digest() == checksum
        except ValueError:
            return False

    def context(self, job: "PydraJob", phase: str) -> dict[str, Any]:
        """Return the lookup context the job carries, or the context of running it here, with its `senselab` section.

        The execute context also records when the call ran and its output identity (`ran`).
        """
        doc = doc_of(job.task)
        section = doc.section()
        if phase == "lookup":
            return {**doc.context.model_dump(mode="json"), JOB: section}
        section.update(_ran(job.checksum), profile=doc.profile.identity_digest())
        return {**doc.execute_context().model_dump(mode="json"), JOB: section}

    def fingerprint(self, context: Mapping[str, Any]) -> str:
        """Return the context's `context@5` fingerprint, its recorded-only sections left out."""
        return stored_context(context).fingerprint()


def _member_name(evidence: EquivalenceEvidence) -> str:
    return f"{evidence.canonical_digest()}.json"


def record_evidence(store: PortableStore, evidence: EquivalenceEvidence) -> None:
    """Write one verify-sampling comparison to the store's sidecar, named by its digest."""
    store.write_sidecar(EVIDENCE, _member_name(evidence), jcs_dumps(evidence.model_dump(mode="json")))


def stored_evidence(store: PortableStore) -> list[EquivalenceEvidence]:
    """Return every verify-sampling comparison the store holds, by digest; an unreadable one is logged and skipped."""
    found = []
    with store.sidecar(EVIDENCE) as side:
        for name in sorted(side.names()):
            try:
                with side.open_read(name) as member:
                    found.append(read_evidence(name, member.read()))
            except (OSError, ValueError) as err:
                logger.warning("an equivalence record was skipped: %s", err)
    return found


def ranked(
    stored: Sequence[tuple[Any, Mapping[str, Any], str]],
    here: ExecutionContext,
    policy: ReusePolicy,
    backend: str,
    equivalence: Equivalence,
) -> list[Candidate]:
    """Return the stored entries admitted for `here`: the equal fingerprint first, then those `policy` accepts."""
    mine = here.fingerprint()
    found: list[Candidate] = []
    for entry, context, digest in stored:
        try:
            theirs = stored_context(context)
        except ValueError as err:
            logger.warning("a stored context of backend %s was skipped: %s", backend, err)
            continue
        if entry.fingerprint == mine:
            found.insert(0, Candidate(entry, context, digest, 0, ReusePolicy.EXACT.value))
        elif acceptable(theirs, here, policy, backend=backend, equivalence=equivalence):
            found.append(Candidate(entry, context, digest, 1, policy.value))
    return found


def serving(  # noqa: PLR0913 - the store and key, then the job's context, run, backend and classes
    store: PortableStore,
    key: str,
    here: ExecutionContext,
    run: RunConfig,
    backend: str,
    equivalence: Equivalence = NO_EQUIVALENCE,
) -> Candidate | None:
    """Return the entry `store` would serve the job of key digest `key` in `here` under `run`, or None.

    The first admitted entry is the one a lookup takes; it is served unless the run refuses its license or samples it
    for verification. Only each entry's `context.json` is read.
    """
    found = ranked(store.stored(key), here, ReusePolicy(run.reuse), backend, store.reuse.equivalence(equivalence))
    if not found:
        return None
    first = found[0]
    entry = StoredContext(first.entry.fingerprint, stored_context(first.context), ReusePolicy(first.admitted_by))
    return first if servable(run, key, here, entry) is Servable.SERVE else None


def admits(store: PortableStore, key: str, here: ExecutionContext, run: RunConfig, backend: str) -> bool:
    """Whether `store` would serve the job of key digest `key` in `here` under `run` (`serving`)."""
    return serving(store, key, here, run, backend) is not None


def entry_outputs(store: PortableStore, entry: Any) -> dict[str, object]:  # noqa: ANN401 - a pydra EntryRef
    """Return the stored outputs of `entry`, as a job's one output holds them.

    Raises:
        OSError, ValueError: the entry cannot be read.
    """
    outputs = store.read(entry, None).outputs
    return dict(outputs[OUTPUTS] if isinstance(outputs, Mapping) else getattr(outputs, OUTPUTS))


class SenselabReuse:
    """The reuse rule of a senselab store: the run's policy and classes admit, its license and verify rules decide.

    Entries are ranked in the process that looks up, against the context of running the job there: the equal
    fingerprint first, then each entry the job's reuse policy accepts under the measured classes (split by the store's
    disagreeing evidence). A result whose env license the run refuses is refused; a sampled cross-context hit is
    verified. A served entry's reuse activity is written to its run's provenance.
    """

    def __init__(self) -> None:
        """Bind to no store yet."""
        self._store: PortableStore | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Return the state a pickled store carries: no bound store."""
        return {"_store": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore an unbound hook."""
        self.__dict__.update(state)

    def bind(self, store: PortableStore) -> None:
        """Bind to `store`, whose sidecar holds the evidence."""
        self._store = store

    def equivalence(self, given: Equivalence) -> Equivalence:
        """Return `given` split by every disagreeing comparison the store holds."""
        if self._store is None or not any(given.values()):
            return given
        table = given
        for evidence in self._evidence():
            if not evidence.agree:
                table = split_class(table, evidence.backend, evidence.stored.member(), evidence.current.member())
        return table

    def _evidence(self) -> list[EquivalenceEvidence]:
        assert self._store is not None
        return stored_evidence(self._store)

    def rank(
        self, current: Mapping[str, Any], stored: Sequence[tuple[Any, Mapping[str, Any], str]], task: object
    ) -> list[Candidate]:
        """Return the admitted entries, the equal fingerprint first, then those the job's policy accepts."""
        doc = doc_of(task)
        table = self.equivalence(doc.equivalence)
        return ranked(stored, doc.execute_context(), ReusePolicy(doc.run.reuse), doc.key.backend, table)

    def verdict(self, candidate: Candidate, current: Mapping[str, Any], task: object) -> Verdict:  # noqa: ARG002
        """Return what the run does with `candidate`: refuse a refused license, verify a sampled hit, else serve."""
        doc = doc_of(task)
        entry = StoredContext(
            candidate.entry.fingerprint, stored_context(candidate.context), ReusePolicy(candidate.admitted_by)
        )
        said = servable(doc.run, doc.key.digest(), doc.execute_context(), entry)
        return {Servable.SERVE: Verdict.SERVE, Servable.VERIFY: Verdict.VERIFY, Servable.REFUSE: Verdict.REFUSE}[said]

    def served(
        self,
        candidate: Candidate,
        current: Mapping[str, Any],
        task: object,
        entry: "StoredEntry",  # noqa: ARG002
    ) -> None:
        """Write the reuse activity of serving `entry` for the job to its run's provenance, before it is served."""
        doc = doc_of(task)
        record_reuse(doc, stored_context(candidate.context), entry.outputs[OUTPUTS], ReusePolicy(candidate.admitted_by))

    def compare(
        self,
        candidate: Candidate,
        stored: "StoredEntry",
        fresh: Mapping[str, Any],
        fresh_digest: str,
        task: object,  # noqa: ARG002
    ) -> None:
        """Compare a verified job's fresh outputs with the stored ones under the run's thresholds; record the evidence.

        A disagreement splits the class of the two contexts for every later lookup.
        """
        doc = doc_of(task)
        comparison = compare(stored.outputs[OUTPUTS], fresh[OUTPUTS], doc.run.thresholds)
        evidence = EquivalenceEvidence(
            task=doc.key.task,
            backend=doc.key.backend,
            key=doc.key.digest(),
            stored=EvidenceSide.of(stored_context(candidate.context)),
            current=EvidenceSide.of(doc.execute_context()),
            mismatched=comparison.differences,
            thresholds=comparison.thresholds,
            run=doc.attempt,
        )
        if self._store is not None:
            record_evidence(self._store, evidence)
        if not comparison.agree:
            logger.warning(
                "verify: %s disagrees with its stored result in %s", doc.key.backend, ", ".join(comparison.paths)
            )
        _verified(doc.key.digest(), evidence.canonical_digest())


def _run_dir(plan_hash: str) -> Path:
    found = run_dir_of(plan_hash)
    found.mkdir(parents=True, exist_ok=True)
    return found


def record_execution(
    staged: object, checksum: str, key: Mapping[str, Any], context: Mapping[str, Any], digest: str
) -> None:  # noqa: ARG001
    """The store's `on_commit`: write the executed activity of an entry to its run's provenance, before its rename.

    The activity names the job's node and plan, the inputs it used, the output identity of its outputs and, for a
    verified job, the evidence its comparison recorded (noted by `SenselabReuse.compare`, which runs before this).
    """
    section = context[JOB]
    logical = LogicalKey.model_validate(dict(key))
    here = stored_context(context)
    ref = JobRef(plan_hash=section["plan"], node=section["node"], element=tuple(section["element"]))
    used = [(port, identity) for port, identity in section["used"]]
    attributes: dict[str, object] = {}
    verifies = _verifies(checksum)
    if verifies is not None:
        attributes["senselab:verifies"] = verifies
    started = datetime.fromisoformat(section["started"])
    output = section["output"]
    if output is None:
        raise RuntimeError(f"job {ref} published outputs its call never noted (ran)")
    provenance = ProvStore()
    agent, activity = execution_records(logical, here, section["profile"], used, output, started, attributes, ref)
    entities = (kprov.entity(i, core_vocabulary()) for i in dict.fromkeys([*(i for _, i in used), output]))
    provenance.add(kprov.plan(ref.plan_hash, core_vocabulary()), *entities, agent, activity)
    provenance.write(_run_dir(ref.plan_hash))


def record_reuse(
    doc: JobDoc,
    stored: ExecutionContext,
    outputs: Mapping[str, object],
    policy: ReusePolicy,
    current: ExecutionContext | None = None,
) -> None:
    """Write the `senselab:Reuse` of serving a stored entry for `doc`'s job to its run's provenance.

    `current` is the context the entry was admitted for: where the job would run (`execute_context`) when None.
    """
    key = doc.key
    current = current if current is not None else doc.execute_context()
    identity = output_identity(dict(outputs))
    agent, stored_agent = agent_record(key, current), agent_record(key, stored)
    own: dict[str, object] = {
        "senselab:storedAgent": stored_agent.id,
        **verify_attributes(doc.run),
        "senselab:logicalKey": key_record(key),
        "senselab:context": context_record(current),
        "senselab:storedContext": context_record(stored),
        "senselab:reusePolicy": policy.value,
    }
    if stored.env_license is None:
        own["senselab:unlicensedReuse"] = True
    if policy is ReusePolicy.EQUIVALENT:
        own["senselab:equivalence"] = equivalence_digest(doc.equivalence)
    now = datetime.now(timezone.utc)
    reuse = kprov.activity(
        core_vocabulary(),
        types=["senselab:Reuse"],
        attributes=own,
        used=[kprov.Usage(identity)],
        associations=[plan_link(agent, doc.ref)],
        started=now,
        ended=now,
    )
    provenance = ProvStore()
    provenance.add(
        kprov.plan(doc.ref.plan_hash, core_vocabulary()),
        kprov.entity(identity, core_vocabulary()),
        agent,
        stored_agent,
        reuse,
    )
    provenance.write(_run_dir(doc.ref.plan_hash))


def open_store(root: Path, work: Path) -> PortableStore:
    """Open the senselab store at `root`, every job running in a directory of its own under `work`.

    Both are on storage every node sees; `work` is the run's private directory.
    """
    work = Path(work).absolute()

    def work_dir(checksum: str, fingerprint: str, attempt: str) -> Path:
        return work / checksum / fingerprint / attempt

    return PortableStore(
        Path(root),
        trust="shared",
        identity=SenselabIdentity(),
        reuse=SenselabReuse(),
        on_commit=record_execution,
        work_dir=work_dir,
    )


def read_entry(store: PortableStore, checksum: str, identity: str) -> tuple[dict[str, object], ExecutionContext] | None:
    """Return the stored outputs and context of the entry of `checksum` whose output identity is `identity`, or None."""
    for entry, context, _ in store.stored(checksum):
        try:
            outputs = store.read(entry, None).outputs
        except (OSError, ValueError) as err:
            logger.warning("an entry of %s was skipped: %s", checksum, err)
            continue
        found = dict(outputs[OUTPUTS] if isinstance(outputs, Mapping) else getattr(outputs, OUTPUTS))
        if output_identity(found) == identity:
            return found, stored_context(context)
    return None


def read_outputs(store: PortableStore, checksum: str, identity: str) -> dict[str, object] | None:
    """Return the stored outputs of the entry of `checksum` whose output identity is `identity`, or None."""
    found = read_entry(store, checksum, identity)
    return None if found is None else found[0]
