"""A PROV-O JSON-LD serializer for :class:`~senselab.utils.prov_store.ProvStore`, following BIDS BEP028.

The store is the single source of truth; this module only reads it, so a run directory written before
this module existed converts without re-running the pipeline. The design, the mapping table and every
place BEP028 and the store disagree are in ``specs/20260908-triage-prov-bep028/design.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.utils.prov_store import (
    CHECKSUM_KEY,
    CHECKSUM_UNRESOLVED_KEY,
    MTIME_KEY,
    PATH_KEY,
    SIZE_KEY,
    Activity,
    Agent,
    Entity,
    Environment,
    ProvStore,
)

BEP028_CONTEXT_SOURCE = "https://github.com/bids-standard/bids-specification/pull/2099 — src/provenance-context.json"
"""Where :data:`BEP028_CONTEXT` was copied from, verbatim."""

BEP028_CONTEXT: dict[str, Any] = {
    "@version": 1.1,
    "Records": {"@container": "@type", "@id": "@graph"},
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "spdx": "http://spdx.org/rdf/terms#",
    "RRID": "http://scicrunch.org/resolver/",
    "Id": "@id",
    "Type": "@type",
    "Label": "rdfs:label",
    "Description": "rdfs:comment",
    "StartedAtTime": {"@id": "prov:startedAtTime", "@type": "xsd:dateTime"},
    "EndedAtTime": {"@id": "prov:endedAtTime", "@type": "xsd:dateTime"},
    "GeneratedBy": {"@id": "prov:wasGeneratedBy", "@type": "@id"},
    "AttributedTo": {"@id": "prov:wasAttributedTo", "@type": "@id"},
    "AssociatedWith": {"@id": "prov:wasAssociatedWith", "@type": "@id"},
    "InformedBy": {"@id": "prov:wasInformedBy", "@type": "@id"},
    "DerivedFrom": {"@id": "prov:wasDerivedFrom", "@type": "@id"},
    "Used": {"@id": "prov:used", "@type": "@id"},
    "ActedOnBehalfOf": {"@id": "prov:actedOnBehalfOf", "@type": "@id"},
    "Files": "prov:Entity",
    "Datasets": "prov:Collection",
    "Environments": "prov:Entity",
    "Activities": "prov:Activity",
    "Software": "prov:Agent",
    "Atlocation": "prov:atLocation",
    "Checksum": "spdx:Checksum",
    "ChecksumAlgorithm": "spdx:ChecksumAlgorithm",
    "ChecksumValue": "spdx:ChecksumValue",
}
"""BEP028's own JSON-LD context, embedded so a graph resolves with no network."""

SENSELAB_VOCABULARY = "https://senselab.sensein.group/prov#"
"""Namespace every store term BEP028 does not define expands into."""

SENSELAB_CONTEXT: dict[str, Any] = {
    "@vocab": SENSELAB_VOCABULARY,
    "sl": SENSELAB_VOCABULARY,
    "AtLocation": {"@id": "prov:atLocation"},
    "InvalidatedBy": {"@id": "prov:wasInvalidatedBy", "@type": "@id"},
}
"""The additions BEP028's context needs: ``prov:wasInvalidatedBy``, the cased ``AtLocation`` its own
examples use, and a vocabulary so no store attribute is silently dropped."""

SHA256_ALGORITHM = "spdx:checksumAlgorithm_sha256"
"""The SPDX term BEP028 requires for a SHA-256 checksum."""

ID_PREFIX = "prov#"
ENTITY_INFIX = "entity-"
RELATION_TERMS: dict[str, str] = {
    "wasGeneratedBy": "GeneratedBy",
    "used": "Used",
    "wasAssociatedWith": "AssociatedWith",
    "wasAttributedTo": "AttributedTo",
    "wasDerivedFrom": "DerivedFrom",
    "wasInvalidatedBy": "InvalidatedBy",
}
"""Each store relation and the JSON-LD term it is emitted under."""

_JSONLD_KEYWORDS = frozenset({"@context", "@id", "@type", "@graph", "@version", "@container", "@vocab"})


def bids_uri(store_id: str, *, dataset: str = "") -> str:
    """The BIDS URI for a store id.

    Args:
        store_id: An entity, activity or agent id as the store holds it.
        dataset: The BIDS dataset name, or ``""`` for the current dataset.

    Returns:
        ``bids:<dataset>:prov#entity-<store_id>`` for an entity, ``bids:<dataset>:prov#<store_id>``
        for an activity or agent.
    """
    infix = "" if store_id.startswith(("act-", "agent-")) else ENTITY_INFIX
    return f"bids:{dataset}:{ID_PREFIX}{infix}{store_id}"


def store_id(uri: str) -> str:
    """The store id a BIDS URI came from — the inverse of :func:`bids_uri`.

    Args:
        uri: A BIDS URI produced by :func:`bids_uri`.

    Returns:
        The store id.

    Raises:
        ValueError: If the URI is not one :func:`bids_uri` produces.
    """
    head, _, fragment = uri.partition(f":{ID_PREFIX}")
    if not fragment or not head.startswith("bids:"):
        raise ValueError(f"not a BIDS provenance URI: {uri!r}")
    return fragment.removeprefix(ENTITY_INFIX)


def _checksum(attributes: dict[str, Any]) -> dict[str, Any]:
    """The ``Checksum`` array for a file, or the reason it has none."""
    digest = attributes.get(CHECKSUM_KEY)
    if digest is not None:
        return {"Checksum": [{"ChecksumAlgorithm": SHA256_ALGORITHM, "ChecksumValue": digest}]}
    reason = attributes.get(CHECKSUM_UNRESOLVED_KEY)
    return {"ChecksumUnresolvedReason": reason} if reason is not None else {}


def _extras(attributes: dict[str, Any]) -> dict[str, Any]:
    """Store attributes with the ones that became BEP028 fields removed."""
    consumed = {PATH_KEY, CHECKSUM_KEY, CHECKSUM_UNRESOLVED_KEY, SIZE_KEY, MTIME_KEY}
    return {key: value for key, value in attributes.items() if key not in consumed and value is not None}


def _entity_record(entity: Entity, relations: dict[str, list[str]], *, dataset: str) -> dict[str, Any]:
    """One ``Files`` or ``prov:Entity`` object."""
    path = entity.attributes.get(PATH_KEY)
    record: dict[str, Any] = {
        "Id": bids_uri(entity.id, dataset=dataset),
        "Label": Path(str(path)).name if path else entity.id,
        "Type": [f"sl:{entity.prov_type}"],
    }
    if path:
        record["AtLocation"] = str(path)
        for key in (SIZE_KEY, MTIME_KEY):
            if entity.attributes.get(key) is not None:
                record[_term(key)] = entity.attributes[key]
    record.update(_checksum(entity.attributes))
    if entity.extent is not None:
        record["ExtentStartSeconds"], record["ExtentEndSeconds"] = entity.extent
    record.update(relations)
    extras = _extras(entity.attributes)
    if extras:
        record["Attributes"] = extras
    return record


def _term(key: str) -> str:
    """A store attribute name as an upper-camel JSON-LD term in the senselab vocabulary."""
    return "".join(part.capitalize() for part in key.split("_"))


def _activity_record(activity: Activity, relations: dict[str, list[str]], *, dataset: str) -> dict[str, Any]:
    """One ``Activities`` object."""
    label = activity.node if activity.step is None else f"{activity.node} / {activity.step}"
    record: dict[str, Any] = {
        "Id": bids_uri(activity.id, dataset=dataset),
        "Label": label,
        "Command": None,
        "Description": f"in-process senselab triage step, not a shell command: {label}",
        "Type": ["sl:workflow-node"],
        "Node": activity.node,
    }
    if activity.step is not None:
        record["Step"] = activity.step
    if activity.started is not None:
        record["StartedAtTime"] = activity.started
    if activity.ended is not None:
        record["EndedAtTime"] = activity.ended
    record.update(relations)
    if activity.parameters:
        record["Parameters"] = activity.parameters
    return record


def _agent_record(agent: Agent, *, dataset: str) -> dict[str, Any]:
    """One ``Software`` object."""
    record: dict[str, Any] = {"Id": bids_uri(agent.id, dataset=dataset)}
    if agent.agent_type == "model":
        record["Label"] = str(agent.model_id)
        record["Type"] = ["sl:model"]
        if agent.commit_sha is not None:
            record["Version"] = agent.commit_sha
        else:
            record["VersionUnresolvedReason"] = agent.unresolved_reason
    else:
        name, _, number = str(agent.version or "").partition(" ")
        record["Label"] = name or "senselab"
        record["Type"] = ["sl:software"]
        if number:
            record["Version"] = number
    return record


def _environment_record(environment: Environment, *, dataset: str) -> dict[str, Any]:
    """One ``Environments`` object."""
    record: dict[str, Any] = {
        "Id": bids_uri(environment.id, dataset=dataset),
        "Label": environment.label,
        _term("python_version"): environment.python_version,
    }
    if environment.operating_system is not None:
        record["OperatingSystem"] = environment.operating_system
    if environment.dependencies:
        record["Dependencies"] = environment.dependencies
    if environment.dependencies_digest is not None:
        record[_term("dependencies_digest")] = environment.dependencies_digest
    if environment.senselab_version is not None:
        record[_term("senselab_version")] = environment.senselab_version
    return record


def to_bep028_graph(store: ProvStore, *, dataset: str = "", label: str = "triage") -> dict[str, Any]:
    """Serialize a store as one aggregated BEP028 provenance graph.

    Args:
        store: The store to read. It is not modified.
        dataset: The BIDS dataset name for every identifier, or ``""`` for the current dataset.
        label: The ``prov-<label>`` entity the graph belongs to.

    Returns:
        A JSON-LD document: an ``@context`` and a ``Records`` object holding ``Files``,
        ``prov:Entity``, ``Activities``, ``Software`` and ``Environments`` arrays. ``Environments``
        is present only when the store recorded one — never invented at conversion time.
    """
    outgoing: dict[str, dict[str, list[str]]] = {}
    for relation, source, target in store.relations():
        term = RELATION_TERMS[relation]
        outgoing.setdefault(source, {}).setdefault(term, []).append(bids_uri(target, dataset=dataset))

    files: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for entity in store.entities():
        record = _entity_record(entity, outgoing.get(entity.id, {}), dataset=dataset)
        (files if entity.attributes.get(PATH_KEY) else others).append(record)

    records: dict[str, Any] = {}
    if files:
        records["Files"] = files
    if others:
        records["prov:Entity"] = others
    activities = [
        _activity_record(activity, outgoing.get(activity.id, {}), dataset=dataset) for activity in store.activities()
    ]
    if activities:
        records["Activities"] = activities
    software = [_agent_record(agent, dataset=dataset) for agent in store.agents()]
    if software:
        records["Software"] = software
    environments = [_environment_record(environment, dataset=dataset) for environment in store.environments()]
    if environments:
        records["Environments"] = environments
    return {
        "@context": [BEP028_CONTEXT, SENSELAB_CONTEXT],
        "Records": records,
        "ProvenanceLabel": f"prov-{label}",
        "StoreFingerprint": store.fingerprint(),
    }


def convert_store_file(store_path: str | Path, *, dataset: str = "", label: str = "triage") -> dict[str, Any]:
    """Read a written ``store.jsonl`` and serialize it as a BEP028 graph.

    Args:
        store_path: The ``store.jsonl`` to read.
        dataset: The BIDS dataset name for every identifier, or ``""`` for the current dataset.
        label: The ``prov-<label>`` entity the graph belongs to.

    Returns:
        The JSON-LD document.
    """
    return to_bep028_graph(ProvStore.read_jsonl(store_path), dataset=dataset, label=label)


def write_bep028_files(graph: dict[str, Any], prov_dir: str | Path, *, label: str = "triage") -> list[Path]:
    """Split an aggregated graph into the BEP028-named provenance files.

    Args:
        graph: A document from :func:`to_bep028_graph`.
        prov_dir: The ``prov/`` directory to write into. Created if absent.
        label: The ``prov-<label>`` entity the filenames carry.

    Returns:
        The files written, in the order ``_io``, ``_act``, ``_soft``, ``_env``.
    """
    records = graph["Records"]
    grouped: dict[str, dict[str, Any]] = {
        "io": {key: records[key] for key in ("Files", "Datasets", "prov:Entity") if key in records},
        "act": {key: records[key] for key in ("Activities",) if key in records},
        "soft": {key: records[key] for key in ("Software",) if key in records},
        "env": {key: records[key] for key in ("Environments",) if key in records},
    }
    target = Path(prov_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, body in grouped.items():
        if not body:
            continue
        path = target / f"prov-{label}_{suffix}.json"
        path.write_text(json.dumps({"@context": graph["@context"], **body}, indent=2, sort_keys=False) + "\n")
        written.append(path)
    return written


def _declared_terms(context: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """The term names and namespace prefixes a merged context declares."""
    terms: set[str] = set()
    prefixes: set[str] = set()
    for part in context:
        for key, value in part.items():
            if key.startswith("@"):
                continue
            terms.add(key)
            if isinstance(value, str) and (value.startswith("http") or value.endswith(":")):
                prefixes.add(key)
    return terms, prefixes


def _walk(node: object, prefixes: set[str], problems: list[str], where: str) -> None:
    """Report every key JSON-LD would drop: an unknown keyword, or an undeclared prefix."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key.startswith("@") and key not in _JSONLD_KEYWORDS:
                problems.append(f"{where}: {key!r} is not a JSON-LD keyword and would be dropped")
            elif ":" in key and not key.startswith("@") and key.split(":", 1)[0] not in prefixes:
                problems.append(f"{where}: term {key!r} has no declared prefix and would be dropped")
            _walk(value, prefixes, problems, f"{where}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk(value, prefixes, problems, f"{where}[{index}]")


def check_graph(graph: dict[str, Any]) -> list[str]:
    """Check a graph is JSON-LD its own context resolves, and that it refers only to itself.

    Args:
        graph: A document from :func:`to_bep028_graph`.

    Returns:
        Every problem found, empty when there are none. A graph is a problem when its context is
        missing a vocabulary (so unmapped terms would be dropped rather than expanded), when two
        nodes share an ``Id``, when an ``Id`` does not reverse to a store id, when a relation names
        an ``Id`` no node in the document declares, or when a checksum is not the SPDX SHA-256 term
        and 64 lowercase hex characters.
    """
    problems: list[str] = []
    context = graph.get("@context")
    if not isinstance(context, list) or not context or context[0] != BEP028_CONTEXT:
        problems.append("@context is not [BEP028_CONTEXT, ...]")
        return problems
    if not any(part.get("@vocab") for part in context):
        problems.append("@context declares no @vocab; unmapped terms would be dropped, not expanded")
    terms, prefixes = _declared_terms(context)

    records = graph.get("Records")
    if not isinstance(records, dict) or not records:
        problems.append("Records is missing or empty")
        return problems
    for key in records:
        if key not in terms and not (":" in key and key.split(":", 1)[0] in prefixes):
            problems.append(f"Records key {key!r} is not a context term")

    declared: set[str] = set()
    for kind, nodes in records.items():
        for index, node in enumerate(nodes):
            where = f"Records.{kind}[{index}]"
            identifier = node.get("Id")
            if not identifier:
                problems.append(f"{where}: no Id")
                continue
            if identifier in declared:
                problems.append(f"{where}: Id {identifier!r} is declared twice")
            declared.add(identifier)
            try:
                store_id(identifier)
            except ValueError as err:
                problems.append(f"{where}: {err}")
            for entry in node.get("Checksum") or []:
                value = entry.get("ChecksumValue", "")
                if entry.get("ChecksumAlgorithm") != SHA256_ALGORITHM:
                    problems.append(f"{where}: unexpected ChecksumAlgorithm {entry.get('ChecksumAlgorithm')!r}")
                if len(value) != 64 or value != value.lower() or not all(c in "0123456789abcdef" for c in value):
                    problems.append(f"{where}: ChecksumValue is not 64 lowercase hex characters")

    for kind, nodes in records.items():
        for index, node in enumerate(nodes):
            for term in RELATION_TERMS.values():
                for reference in node.get(term) or []:
                    if reference not in declared:
                        problems.append(f"Records.{kind}[{index}].{term}: {reference!r} is not a node in the graph")
    _walk({key: value for key, value in graph.items() if key != "@context"}, prefixes, problems, "graph")
    return problems
