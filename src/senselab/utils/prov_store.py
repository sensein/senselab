"""An append-only provenance store, modelled on W3C PROV.

Entities are what the graph believes exists, activities are node executions, agents are what acted.
Relations are PROV's own. Nothing is modified after it is added, so merging two stores is a set union
and is order-independent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence, cast, get_args

PROV_TYPE = Literal[
    "span",
    "word",
    "speaker",
    "interval",
    "measurement",
    "kind",
    "stream",
    "pii",
    "verdict",
    "assertion",
    "target_match",
    "branch_decision",
    "enrollment",
]
AGENT_TYPE = Literal["model", "software"]
RELATION = Literal[
    "wasGeneratedBy", "used", "wasAssociatedWith", "wasAttributedTo", "wasDerivedFrom", "wasInvalidatedBy"
]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_ALGORITHM = "sha256"
CHECKSUM_KEY = "checksum_sha256"
CHECKSUM_UNRESOLVED_KEY = "checksum_unresolved_reason"
SIZE_KEY = "size_bytes"
MTIME_KEY = "mtime_ns"
PATH_KEY = "path"
_READ_CHUNK = 1 << 20
_PROV_TYPES = frozenset(get_args(PROV_TYPE))
_AGENT_TYPES = frozenset(get_args(AGENT_TYPE))
_RELATIONS = frozenset(get_args(RELATION))
_RECORD_KEYS: dict[str, frozenset[str]] = {
    "entity": frozenset({"id", "prov_type", "extent", "attributes"}),
    "activity": frozenset({"id", "node", "step", "started", "ended", "parameters"}),
    "agent": frozenset({"id", "agent_type", "model_id", "commit_sha", "unresolved_reason", "version"}),
    "relation": frozenset({"relation", "source", "target"}),
}


@dataclass(frozen=True)
class Entity:
    """Something the graph believes exists."""

    id: str
    prov_type: PROV_TYPE
    extent: tuple[float, float] | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Activity:
    """One node execution, or one step of one.

    Attributes:
        id: The activity's id. ``started`` and ``ended`` are not part of it.
        node: The node executing.
        step: Which step of it, when a node has several.
        started: When it began, ISO 8601, or None when unrecorded.
        ended: When it finished, ISO 8601, or None when unrecorded.
        parameters: The values it ran with.
    """

    id: str
    node: str
    step: str | None
    started: str | None = None
    ended: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Agent:
    """What acted: a model at a resolved commit, or the software itself.

    Attributes:
        id: The agent's id.
        agent_type: ``"model"`` or ``"software"``.
        model_id: The model's identifier, for a model agent.
        commit_sha: A resolved 40-hex commit, when resolution succeeded.
        unresolved_reason: Why the commit is unknown, when it is. A provenance model that cannot say
            "I could not resolve this" forces either a lie or a crash.
        version: Software version, for a software agent.
    """

    id: str
    agent_type: AGENT_TYPE
    model_id: str | None = None
    commit_sha: str | None = None
    unresolved_reason: str | None = None
    version: str | None = None


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_digest(path: str | Path) -> tuple[str | None, str | None]:
    """Read a file and return its SHA-256, or the reason it has none.

    Args:
        path: The file to digest.

    Returns:
        ``(digest, reason)`` — a 64-character lowercase hex digest and None, or None and a reason
        the digest could not be taken.
    """
    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
    except FileNotFoundError:
        return None, "file not found"
    except IsADirectoryError:
        return None, "path is a directory"
    except PermissionError:
        return None, "permission denied"
    except OSError as err:
        return None, f"read failed: {type(err).__name__}"
    return digest.hexdigest(), None


def file_attributes(path: str | Path, *, digest: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Entity attributes describing a file's bytes as they are now.

    Args:
        path: The file, used for ``size_bytes`` and ``mtime_ns`` and, unless a digest or a reason is
            supplied, read to compute the digest.
        digest: An already-computed SHA-256 to record instead of reading the file again.
        reason: Why no digest is available, recorded instead of reading the file again.

    Returns:
        ``checksum_sha256`` or ``checksum_unresolved_reason``, and ``size_bytes`` and ``mtime_ns``
        when the file can be stat'd.

    Raises:
        ValueError: If both ``digest`` and ``reason`` are supplied.
    """
    if digest is not None and reason is not None:
        raise ValueError("digest and reason contradict each other; supply at most one")
    if digest is None and reason is None:
        digest, reason = file_digest(path)
    out: dict[str, Any] = {CHECKSUM_KEY: digest} if digest is not None else {CHECKSUM_UNRESOLVED_KEY: reason}
    try:
        stat = Path(path).stat()
    except OSError:
        return out
    out[SIZE_KEY] = int(stat.st_size)
    out[MTIME_KEY] = int(stat.st_mtime_ns)
    return out


def _check_entity_attributes(attributes: dict[str, Any]) -> None:
    """Refuse checksum attributes the store never accepts, at write time and at read-back alike.

    Raises:
        ValueError: If a digest and a reason it is missing are supplied together, if the digest is
            not 64 lowercase hex characters, if the reason is empty, or if either is present without
            a ``path`` saying which file it describes.
    """
    digest, reason = attributes.get(CHECKSUM_KEY), attributes.get(CHECKSUM_UNRESOLVED_KEY)
    if digest is not None and reason is not None:
        raise ValueError(f"{CHECKSUM_KEY} and {CHECKSUM_UNRESOLVED_KEY} contradict each other; supply exactly one")
    if digest is not None and not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
        raise ValueError(f"{CHECKSUM_KEY} must be 64 lowercase hex characters, got {digest!r}")
    if reason is not None and not (isinstance(reason, str) and reason.strip()):
        raise ValueError(f"{CHECKSUM_UNRESOLVED_KEY} must not be empty; an empty reason says nothing")
    if (digest is not None or reason is not None) and not attributes.get(PATH_KEY):
        raise ValueError(f"a checksum needs a {PATH_KEY!r} naming the file it describes")


def _check_agent_fields(
    agent_type: str, model_id: str | None, commit_sha: str | None, unresolved_reason: str | None
) -> None:
    """Refuse agent field combinations the store never accepts, at write time and at read-back alike.

    Raises:
        ValueError: If ``commit_sha`` is not exactly 40 hex characters, if ``commit_sha`` and
            ``unresolved_reason`` are supplied together, if ``unresolved_reason`` is empty, or if a
            model agent is missing ``model_id`` or supplies neither a commit nor a reason it is
            missing.
    """
    if commit_sha is not None and not _SHA.fullmatch(commit_sha):
        raise ValueError(
            f"commit_sha must be a resolved 40-hex commit, got {commit_sha!r}. A ref recorded as a "
            "commit makes the provenance confidently wrong."
        )
    if commit_sha is not None and unresolved_reason is not None:
        raise ValueError("commit_sha and unresolved_reason contradict each other; supply exactly one")
    if unresolved_reason is not None and not unresolved_reason.strip():
        raise ValueError("unresolved_reason must not be empty; an empty reason says nothing")
    if agent_type == "model" and model_id is None:
        raise ValueError("a model agent needs model_id")
    if agent_type == "model" and commit_sha is None and unresolved_reason is None:
        raise ValueError("a model agent needs commit_sha or unresolved_reason; silence is not a third option")


class ProvStore:
    """An append-only PROV document.

    Args:
        run_id: Mixed into every id so two runs never collide.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty store."""
        self.run_id = run_id
        self._entities: dict[str, Entity] = {}
        self._activities: dict[str, Activity] = {}
        self._agents: dict[str, Agent] = {}
        self._relations: list[tuple[RELATION, str, str]] = []

    def entity(self, *, prov_type: PROV_TYPE, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """Add an entity.

        Args:
            prov_type: What sort of thing it is.
            extent: ``(start, end)`` in seconds, or None for something without one.
            attributes: Whatever describes it.

        Returns:
            Its id.

        Raises:
            ValueError: If the attributes carry both a checksum and a reason it is missing, a
                malformed digest, an empty reason, or either without a ``path``.
        """
        _check_entity_attributes(attributes)
        eid = f"{prov_type}-{_digest([self.run_id, prov_type, extent, attributes])}"
        self._entities[eid] = Entity(id=eid, prov_type=prov_type, extent=extent, attributes=dict(attributes))
        return eid

    def activity(
        self,
        *,
        node: str,
        step: str | None,
        parameters: dict[str, Any],
        started: str | None = None,
        ended: str | None = None,
    ) -> str:
        """Add an activity.

        Args:
            node: The node executing.
            step: Which step of it, when a node has several.
            parameters: The values it ran with.
            started: When it began, ISO 8601. Excluded from the id digest.
            ended: When it finished, ISO 8601. Excluded from the id digest.

        Returns:
            Its id — the same for two executions that differ only in ``started``/``ended``.
        """
        aid = f"act-{_digest([self.run_id, node, step, parameters])}"
        self._activities[aid] = Activity(
            id=aid, node=node, step=step, started=started, ended=ended, parameters=dict(parameters)
        )
        return aid

    def agent(
        self,
        *,
        agent_type: AGENT_TYPE,
        model_id: str | None = None,
        commit_sha: str | None = None,
        unresolved_reason: str | None = None,
        version: str | None = None,
    ) -> str:
        """Add an agent.

        Args:
            agent_type: ``"model"`` or ``"software"``.
            model_id: Required for a model agent.
            commit_sha: A resolved 40-hex commit.
            unresolved_reason: Why the commit is unknown, if it is.
            version: Software version, for a software agent.

        Returns:
            Its id.

        Raises:
            ValueError: If ``commit_sha`` is not exactly 40 hex characters, if ``commit_sha`` and
                ``unresolved_reason`` are supplied together, if ``unresolved_reason`` is empty, or if a
                model agent is missing ``model_id`` or supplies neither a commit nor a reason it is
                missing.
        """
        _check_agent_fields(agent_type, model_id, commit_sha, unresolved_reason)
        gid = f"agent-{_digest([self.run_id, agent_type, model_id, commit_sha, unresolved_reason, version])}"
        self._agents[gid] = Agent(
            id=gid,
            agent_type=agent_type,
            model_id=model_id,
            commit_sha=commit_sha,
            unresolved_reason=unresolved_reason,
            version=version,
        )
        return gid

    def _relate(self, relation: RELATION, source: str, target: str) -> None:
        triple = (relation, source, target)
        if triple not in self._relations:
            self._relations.append(triple)

    def was_generated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an activity produced an entity."""
        self._relate("wasGeneratedBy", entity_id, activity_id)

    def used(self, activity_id: str, entity_id: str) -> None:
        """Record that an activity read an entity."""
        self._relate("used", activity_id, entity_id)

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        """Record which agent ran an activity."""
        self._relate("wasAssociatedWith", activity_id, agent_id)

    def was_attributed_to(self, entity_id: str, agent_id: str) -> None:
        """Record which agent is answerable for an entity."""
        self._relate("wasAttributedTo", entity_id, agent_id)

    def was_derived_from(self, entity_id: str, source_entity_id: str) -> None:
        """Record that an entity refines or answers another, keeping both."""
        self._relate("wasDerivedFrom", entity_id, source_entity_id)

    def was_invalidated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an entity should no longer be read as what it was. It is not removed."""
        self._relate("wasInvalidatedBy", entity_id, activity_id)

    def get_entity(self, entity_id: str) -> Entity:
        """Return one entity."""
        return self._entities[entity_id]

    def get_activity(self, activity_id: str) -> Activity:
        """Return one activity."""
        return self._activities[activity_id]

    def get_agent(self, agent_id: str) -> Agent:
        """Return one agent."""
        return self._agents[agent_id]

    def entities(self, prov_type: PROV_TYPE | None = None) -> list[Entity]:
        """Return entities, optionally of one type."""
        return [e for e in self._entities.values() if prov_type is None or e.prov_type == prov_type]

    def activities(self, node: str | None = None) -> list[Activity]:
        """Return activities, optionally of one node."""
        return [a for a in self._activities.values() if node is None or a.node == node]

    def agents(self, agent_type: AGENT_TYPE | None = None) -> list[Agent]:
        """Return agents, optionally of one type.

        Args:
            agent_type: ``"model"`` or ``"software"``, or None for every agent.

        Returns:
            The agents, in write order.
        """
        return [g for g in self._agents.values() if agent_type is None or g.agent_type == agent_type]

    def relations(self) -> list[tuple[RELATION, str, str]]:
        """Return every relation as ``(relation, source, target)``, in write order."""
        return list(self._relations)

    def associations_of(self, agent_id: str) -> list[str]:
        """Return the activities an agent was associated with.

        The inverse of :meth:`associated_with`, for a reader asking what one agent did rather than
        who ran one activity.

        Args:
            agent_id: The agent's id.

        Returns:
            The activity ids, in the order the relations were recorded.
        """
        return [s for r, s, t in self._relations if r == "wasAssociatedWith" and t == agent_id]

    def _targets(self, relation: RELATION, source: str) -> list[str]:
        return [t for r, s, t in self._relations if r == relation and s == source]

    def generated_by(self, entity_id: str) -> str | None:
        """Return the activity that generated an entity, or None."""
        found = self._targets("wasGeneratedBy", entity_id)
        return found[0] if found else None

    def uses_of(self, activity_id: str) -> list[str]:
        """Return the entities an activity read."""
        return self._targets("used", activity_id)

    def associated_with(self, activity_id: str) -> list[str]:
        """Return the agents associated with an activity."""
        return self._targets("wasAssociatedWith", activity_id)

    def derived_from(self, entity_id: str) -> list[str]:
        """Return the entities an entity was derived from."""
        return self._targets("wasDerivedFrom", entity_id)

    def is_invalidated(self, entity_id: str) -> bool:
        """Whether an entity has been invalidated."""
        return bool(self._targets("wasInvalidatedBy", entity_id))

    def write_jsonl(self, path: str | Path) -> None:
        """Write the store as one PROV-JSON-shaped record per line."""
        lines = [
            json.dumps({"record": "entity", **asdict(e)}, sort_keys=True, default=str) for e in self._entities.values()
        ]
        lines += [
            json.dumps({"record": "activity", **asdict(a)}, sort_keys=True, default=str)
            for a in self._activities.values()
        ]
        lines += [
            json.dumps({"record": "agent", **asdict(g)}, sort_keys=True, default=str) for g in self._agents.values()
        ]
        lines += [
            json.dumps({"record": "relation", "relation": r, "source": s, "target": t}, sort_keys=True)
            for r, s, t in self._relations
        ]
        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, run_id: str = "read") -> "ProvStore":
        """Read a store back, holding every record to the write-time invariants.

        Relations pass through the same membership check as the relation methods, so a file carrying
        the same relation line twice reads back as one triple. Agent fields pass through the same
        checks as :meth:`agent`. A store written by :meth:`write_jsonl` always reads back unchanged.

        Args:
            path: The JSONL file to read.
            run_id: The run id of the returned store.

        Returns:
            The reconstructed store.

        Raises:
            ValueError: If a line is not an object carrying a ``record`` key, carries an unrecognised
                record kind, is missing required keys or carrying unexpected ones, names an unknown
                ``prov_type``, ``agent_type`` or relation, or holds agent fields that :meth:`agent`
                refuses. The error names the file, the line and the offending record.
        """
        store = cls(run_id=run_id)
        for line_no, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            where = f"{path}, line {line_no}"
            if not isinstance(rec, dict) or "record" not in rec:
                raise ValueError(f"{where}: not an object carrying a 'record' key: {line!r}")
            kind = rec.pop("record")
            expected = _RECORD_KEYS.get(kind)
            if expected is None:
                raise ValueError(f"{where}: unrecognised record kind {kind!r}")
            missing, unexpected = sorted(expected - rec.keys()), sorted(rec.keys() - expected)
            if missing or unexpected:
                raise ValueError(
                    f"{where}: {kind} record missing keys {missing}, carrying unexpected keys {unexpected}: {line!r}"
                )
            if kind == "entity":
                if rec["prov_type"] not in _PROV_TYPES:
                    raise ValueError(f"{where}: entity {rec['id']!r} has unknown prov_type {rec['prov_type']!r}")
                try:
                    _check_entity_attributes(rec["attributes"])
                except ValueError as err:
                    raise ValueError(f"{where}: entity {rec['id']!r}: {err}") from err
                extent = rec.pop("extent")
                store._entities[rec["id"]] = Entity(extent=tuple(extent) if extent else None, **rec)
            elif kind == "activity":
                store._activities[rec["id"]] = Activity(**rec)
            elif kind == "agent":
                if rec["agent_type"] not in _AGENT_TYPES:
                    raise ValueError(f"{where}: agent {rec['id']!r} has unknown agent_type {rec['agent_type']!r}")
                try:
                    _check_agent_fields(rec["agent_type"], rec["model_id"], rec["commit_sha"], rec["unresolved_reason"])
                except ValueError as err:
                    raise ValueError(f"{where}: agent {rec['id']!r}: {err}") from err
                store._agents[rec["id"]] = Agent(**rec)
            else:
                if rec["relation"] not in _RELATIONS:
                    raise ValueError(f"{where}: unknown relation {rec['relation']!r}")
                store._relate(cast(RELATION, rec["relation"]), rec["source"], rec["target"])
        return store

    @classmethod
    def merge(cls, stores: Sequence["ProvStore"]) -> "ProvStore":
        """Union several stores. Append-only makes this order-independent."""
        out = cls(run_id="merged")
        for s in stores:
            out._entities.update(s._entities)
            out._activities.update(s._activities)
            out._agents.update(s._agents)
        seen: set[tuple[RELATION, str, str]] = set()
        for s in stores:
            for rel in s._relations:
                if rel not in seen:
                    seen.add(rel)
                    out._relations.append(rel)
        return out

    def fingerprint(self) -> str:
        """A content hash that ignores insertion order."""
        return _digest(
            {
                "e": sorted(self._entities),
                "act": sorted(self._activities),
                "ag": sorted(self._agents),
                "r": sorted(f"{r}:{s}:{t}" for r, s, t in self._relations),
            }
        )
