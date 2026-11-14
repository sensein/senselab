"""W3C PROV records written as compact PROV-O JSON-LD, checked against the `senselab:` vocabulary (ADR-010).

A record is one JSON-LD node: an entity, activity or agent, or a PROV subtype. Its keys are compact IRIs under
the prefixes of the published context (`CONTEXT_URL`). Builders check every type and attribute against a
`Vocabulary` and encode every value as a JSON-LD literal; what they build is immutable.

Identifiers:
- an entity's `@id` is its content address, `content:<hex>` or `output:<hex>`;
- an activity's is `activity:` + `identity_digest("prov-activity@1", <node without @id>)`;
- an agent's is `agent:` and an environment's `environment:` + `identity_digest("prov-node@1", <node without @id>)`;
- a plan's is `plan:` + its plan hash.
"""

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import cast

import numpy
from senselab_worker.canonical import (
    MAX_SAFE_INT,
    InvalidText,
    canonical_scalar,
    canonical_text,
    jcs_dumps,
    jcs_loads,
)
from senselab_worker.digest import identity_digest, npy_digest

from senselab_core.kernel.formats import (
    PROV_ACTIVITY_TAG,
    PROV_NODE_TAG,
    PROVENANCE_FORMAT,
    SENSELAB_VOCAB_FORMAT,
    readable,
)

SENSELAB_NS = "https://w3id.org/senselab/ns#"
"""The IRI the `senselab:` prefix stands for."""
CONTEXT_URL = "https://w3id.org/senselab/ns/context.jsonld"
"""The published `@context` every document and vocabulary file names."""

PREFIXES: Mapping[str, str] = {
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "senselab": SENSELAB_NS,
}
"""The prefixes the published context defines."""

PROV_CLASSES: Mapping[str, str | None] = {
    "prov:Entity": None,
    "prov:Activity": None,
    "prov:Agent": None,
    "prov:Role": None,
    "prov:Plan": "prov:Entity",
    "prov:Collection": "prov:Entity",
    "prov:SoftwareAgent": "prov:Agent",
}
"""The PROV classes a vocabulary class may specialise, each with its PROV superclass."""

LITERAL_RANGES = frozenset(
    {"xsd:string", "xsd:boolean", "xsd:integer", "xsd:double", "xsd:decimal", "xsd:dateTime", "rdf:JSON"}
)
"""The literal ranges a vocabulary property may declare; any other range is a class, whose values are records."""

_RANGES_OF: Mapping[type, str] = {bool: "xsd:boolean", str: "xsd:string", datetime: "xsd:dateTime"}
_NUMERIC_RANGES = frozenset({"xsd:integer", "xsd:double", "xsd:decimal"})
ENTITY_ID = re.compile(r"(content|output):[0-9a-f]{64}")
"""The grammar of an entity `@id`."""
AGENT_ID = re.compile(r"agent:[0-9a-f]{64}")
"""The grammar of an agent `@id`."""
PLAN_ID = re.compile(r"plan:[0-9a-f]{64}")
"""The grammar of a plan `@id`."""
NODE_ID = re.compile(r"(content|output|activity|agent|environment|plan):[0-9a-f]{64}")
"""The grammar of every node `@id` a document holds."""
_TERM = re.compile(r"senselab:[A-Za-z][A-Za-z0-9_]*")
_VOCAB_NODE_KEYS = frozenset({"@id", "@type", "rdfs:subClassOf", "rdfs:domain", "rdfs:range", "rdfs:comment"})
_XSD_DOUBLE = {math.inf: "INF", -math.inf: "-INF"}


class VocabularyError(ValueError):
    """A vocabulary document that is malformed, of another format, or redefines a term."""


class UnknownTerm(ValueError):
    """A type or attribute the vocabulary does not define, or defines for another class.

    Attributes:
        path: The attribute or type that was refused.
    """

    def __init__(self, path: str, message: str) -> None:
        """Name the refused term."""
        super().__init__(f"{path}: {message}")
        self.path = path


class ValueOutOfRange(ValueError):
    """An attribute value outside the range its property declares.

    Attributes:
        path: The attribute path of the value, as `term`, `term[i]` or `term{}`.
    """

    def __init__(self, path: str, message: str) -> None:
        """Name the attribute path."""
        super().__init__(f"{path}: {message}")
        self.path = path


class UnrecordableValue(ValueError):
    """A value with no JSON-LD literal form: bytes, a path, a Python object, None, or a nested container.

    Attributes:
        path: The attribute path of the value, as `term`, `term[i]` or `term{}`.
    """

    def __init__(self, path: str, message: str) -> None:
        """Name the attribute path."""
        super().__init__(f"{path}: {message}")
        self.path = path


@dataclass(frozen=True)
class Property:
    """A vocabulary property: the class it applies to and the range of its values."""

    domain: str
    range: str


class Vocabulary:
    """The classes and properties records are checked against: the PROV classes and `senselab:` terms.

    Built from vocabulary documents (`from_document`), each extending the vocabulary it is read over.
    """

    def __init__(self, classes: Mapping[str, str | None], properties: Mapping[str, Property]) -> None:
        """Hold `classes` (term to superclass) and `properties`; use `from_document` to build one."""
        self._classes = dict(classes)
        self._properties = dict(properties)

    @classmethod
    def from_document(cls, document: object, base: "Vocabulary | None" = None) -> "Vocabulary":
        """Return `base` (or the PROV classes alone) extended by the terms of a `senselab-vocab@1` document.

        The document is `{"@context": CONTEXT_URL, "@graph": [...]}`. One node is the ontology,
        `{"@id": "senselab:", "@type": "owl:Ontology", "owl:versionInfo": "senselab-vocab@1.<minor>"}`;
        each other node defines one `senselab:` term, either a class
        (`"@type": "rdfs:Class"`, `"rdfs:subClassOf": {"@id": <known class>}`) or a property
        (`"@type": "rdf:Property"`, `"rdfs:domain"` a known class, `"rdfs:range"` one of `LITERAL_RANGES` or a
        known class), with an optional `rdfs:comment`.

        Raises:
            VocabularyError: on any other shape, another format, an unknown superclass, domain or range, or a
                term `base` or the document already defines.
        """
        if not isinstance(document, Mapping) or document.get("@context") != CONTEXT_URL:
            raise VocabularyError(f"a vocabulary document is a mapping whose @context is {CONTEXT_URL}")
        graph = document.get("@graph")
        if set(document) != {"@context", "@graph"} or not isinstance(graph, list):
            raise VocabularyError("a vocabulary document holds exactly @context and an @graph list")
        classes = dict(base._classes if base else PROV_CLASSES)
        properties = dict(base._properties if base else {})
        ontologies = [n for n in graph if isinstance(n, Mapping) and n.get("@type") == "owl:Ontology"]
        if len(ontologies) != 1 or not _readable_format(ontologies[0]):
            raise VocabularyError(f"a vocabulary document carries one ontology node of {SENSELAB_VOCAB_FORMAT}")
        for node in graph:
            if node is not ontologies[0]:
                _add_term(node, classes, properties)
        return cls(classes, properties)

    def is_a(self, term: str, ancestor: str) -> bool:
        """Whether class `term` is `ancestor` or one of its subclasses."""
        current: str | None = term
        while current is not None:
            if current == ancestor:
                return True
            current = self._classes.get(current)
        return False

    def has_class(self, term: str) -> bool:
        """Whether `term` is a class of this vocabulary."""
        return term in self._classes

    def property(self, term: str) -> Property | None:
        """Return the property `term`, or None if this vocabulary does not define it."""
        return self._properties.get(term)


def _readable_format(node: Mapping[str, object]) -> bool:
    return node.get("@id") == "senselab:" and readable(SENSELAB_VOCAB_FORMAT, node.get("owl:versionInfo"))


def _ref(node: Mapping[str, object], key: str) -> str:
    value = node.get(key)
    if not (isinstance(value, Mapping) and set(value) == {"@id"} and isinstance(value["@id"], str)):
        raise VocabularyError(f"{node.get('@id')!r}: {key} is {{'@id': <term>}}")
    return value["@id"]


def _add_term(node: object, classes: dict[str, str | None], properties: dict[str, Property]) -> None:
    if not isinstance(node, Mapping) or not set(node) <= _VOCAB_NODE_KEYS:
        raise VocabularyError(f"a term node holds only {sorted(_VOCAB_NODE_KEYS)}, got {node!r}")
    term = node.get("@id")
    if not (isinstance(term, str) and _TERM.fullmatch(term)):
        raise VocabularyError(f"a vocabulary term is senselab:<name>, got {term!r}")
    if term in classes or term in properties:
        raise VocabularyError(f"{term} is already defined")
    kind = node.get("@type")
    if kind == "rdfs:Class" and set(node) - {"rdfs:comment"} == {"@id", "@type", "rdfs:subClassOf"}:
        parent = _ref(node, "rdfs:subClassOf")
        if parent not in classes:
            raise VocabularyError(f"{term}: superclass {parent} is not a known class")
        classes[term] = parent
    elif kind == "rdf:Property" and set(node) - {"rdfs:comment"} == {"@id", "@type", "rdfs:domain", "rdfs:range"}:
        domain, rng = _ref(node, "rdfs:domain"), _ref(node, "rdfs:range")
        if domain not in classes:
            raise VocabularyError(f"{term}: domain {domain} is not a known class")
        if rng not in LITERAL_RANGES and rng not in classes:
            raise VocabularyError(f"{term}: range {rng} is neither one of {sorted(LITERAL_RANGES)} nor a known class")
        properties[term] = Property(domain=domain, range=rng)
    else:
        raise VocabularyError(
            f"{term}: a term is an rdfs:Class with rdfs:subClassOf or an rdf:Property with rdfs:domain and rdfs:range"
        )


_SEAL = object()


class Record:
    """One compact JSON-LD node and the `senselab:File` entities its attribute values name.

    Built only by `entity`, `plan`, `agent`, `environment` and `activity`. It holds the node's JCS bytes (`jcs`);
    `node()` returns a fresh copy. Records compare and hash by those bytes and their files.
    """

    __slots__ = ("_files", "_jcs")
    _jcs: bytes
    _files: tuple["Record", ...]

    def __init__(self, jcs: bytes, files: tuple["Record", ...], *, seal: object = None) -> None:
        """Hold a node the builders made; any other caller is refused with `TypeError`."""
        if seal is not _SEAL:
            raise TypeError("a Record is built by entity, plan, agent, environment or activity")
        object.__setattr__(self, "_jcs", jcs)
        object.__setattr__(self, "_files", files)

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse every assignment."""
        raise AttributeError(f"a Record is read-only; cannot set {name!r}")

    def __eq__(self, other: object) -> bool:
        """Equal when the nodes and files are byte-identical."""
        return isinstance(other, Record) and (self._jcs, self._files) == (other._jcs, other._files)

    def __hash__(self) -> int:
        """Hash of the node and files bytes."""
        return hash((self._jcs, self._files))

    def __repr__(self) -> str:
        """The node's JCS text."""
        return f"Record({self._jcs.decode()})"

    @property
    def jcs(self) -> bytes:
        """The node's JCS bytes."""
        return self._jcs

    @property
    def files(self) -> tuple["Record", ...]:
        """The `senselab:File` entities the node's attribute values name, in `@id` order."""
        return self._files

    @property
    def id(self) -> str:
        """The node's `@id`."""
        return str(self.node()["@id"])

    def node(self) -> dict[str, object]:
        """Return the compact JSON-LD node."""
        return cast("dict[str, object]", jcs_loads(self.jcs))

    def types(self) -> tuple[str, ...]:
        """Return the node's `@type` list."""
        return tuple(cast("list[str]", self.node()["@type"]))


@dataclass(frozen=True)
class Usage:
    """An activity's use of an entity, with the port it was read on and when, if known."""

    entity: str
    port: str | None = None
    time: datetime | None = None


@dataclass(frozen=True)
class Generation:
    """An activity's generation of an entity, with the port it was written on and when, if known."""

    entity: str
    port: str | None = None
    time: datetime | None = None


@dataclass(frozen=True)
class Association:
    """An activity's association with an agent, within a plan: its role carries the node id, `plan` the plan's `@id`.

    Outside a plan both are None and the association names the agent alone. With no `agent` the association is
    qualified only, naming the node and the plan (PROV-DM: the agent of an association is optional given a plan).
    """

    agent: str | None = None
    node_id: str | None = None
    plan: str | None = None


@dataclass
class _Builder:
    vocabulary: Vocabulary
    files: dict[str, Record] = field(default_factory=dict)

    def types(self, base: str, extra: Iterable[str]) -> list[str]:
        out = {base}
        for t in extra:
            if not self.vocabulary.has_class(t):
                raise UnknownTerm(t, "the vocabulary defines no such class")
            if not self.vocabulary.is_a(t, base):
                raise UnknownTerm(t, f"is not a subclass of {base}")
            out.add(t)
        return sorted(out)

    def attributes(self, types: Sequence[str], attributes: Mapping[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for term, value in attributes.items():
            prop = self.vocabulary.property(term)
            if prop is None:
                raise UnknownTerm(term, "the vocabulary defines no such property")
            if not any(self.vocabulary.is_a(t, prop.domain) for t in types):
                raise UnknownTerm(term, f"applies to {prop.domain}, not to {list(types)}")
            out[term] = self.value(value, prop.range, term)
        return out

    def value(self, value: object, rng: str, path: str) -> object:
        if rng == "rdf:JSON":
            return _json_literal(value, path)
        if isinstance(value, (set, frozenset)):
            items = {jcs_dumps(v): v for v in (self.scalar(x, rng, f"{path}{{}}") for x in value)}
            return {"@set": [items[k] for k in sorted(items)]}
        if type(value) is list or type(value) is tuple:
            return {"@list": [self.scalar(x, rng, f"{path}[{i}]") for i, x in enumerate(value)]}
        return self.scalar(value, rng, path)

    def scalar(self, value: object, rng: str, path: str) -> object:
        if rng not in LITERAL_RANGES:
            return self.reference(value, rng, path)
        if isinstance(value, Enum):
            if type(value.value) is not str:
                raise UnrecordableValue(path, f"an enum is recorded by its str value; {value!r} has none")
            value = value.value
        if rng in _NUMERIC_RANGES:
            return _number(value if isinstance(value, Decimal) else _plain(value, path), rng, path)
        if isinstance(value, Decimal):
            raise ValueOutOfRange(path, f"a Decimal is outside the range {rng}")
        if isinstance(value, datetime):
            _check_range(type(value), rng, path)
            return _datetime(value, path)
        return _json_scalar(value, rng, path)

    def reference(self, value: object, rng: str, path: str) -> object:
        if isinstance(value, Record):
            if not any(self.vocabulary.is_a(t, rng) for t in value.types()):
                raise ValueOutOfRange(path, f"names a {list(value.types())}, not a {rng}")
            self.files.update({f.id: f for f in value.files})
            return {"@id": value.id}
        if isinstance(value, numpy.ndarray):
            if not self.vocabulary.is_a("senselab:File", rng):
                raise ValueOutOfRange(path, f"an array is recorded as a senselab:File, not a {rng}")
            try:
                file_id = "content:" + npy_digest("npy@1", value)
            except (TypeError, ValueError) as err:
                raise UnrecordableValue(path, f"an array with no canonical .npy form: {err}") from None
            self.files[file_id] = _sealed({"@id": file_id, "@type": ["prov:Entity", "senselab:File"]})
            return {"@id": file_id}
        raise UnrecordableValue(path, f"a {rng} is recorded from a record or an array, not {type(value).__qualname__}")

    def record(self, node: dict[str, object]) -> Record:
        return _sealed(node, tuple(self.files[k] for k in sorted(self.files)))


def _plain(value: object, path: str) -> object:
    try:
        scalar = canonical_scalar(value)
    except TypeError:
        raise UnrecordableValue(path, f"{type(value).__qualname__} has no JSON-LD literal form") from None
    if scalar is None:
        raise UnrecordableValue(path, "JSON-LD drops null; leave the attribute out")
    return scalar


def _json_scalar(value: object, rng: str, path: str) -> object:
    scalar = _plain(value, path)
    _check_range(type(scalar), rng, path)
    if type(scalar) is str:
        try:
            return canonical_text(scalar)
        except InvalidText as err:
            raise UnrecordableValue(path, str(err)) from None
    return scalar


def _check_range(kind: type, rng: str, path: str) -> None:
    if _RANGES_OF.get(kind) != rng:
        raise ValueOutOfRange(path, f"a {kind.__qualname__} is outside the range {rng}")


def _number(value: object, rng: str, path: str) -> object:
    """Write a number in the form its range gives it, whatever its Python type; refuse one the range cannot hold."""
    if type(value) not in (int, float, Decimal):
        raise ValueOutOfRange(path, f"a {type(value).__qualname__} is outside the range {rng}")
    if isinstance(value, Decimal) and not value.is_finite():
        raise UnrecordableValue(path, f"a Decimal {value} has no {rng} form")
    if rng == "xsd:double":
        return _double(value, path)  # type: ignore[arg-type]
    if type(value) is float and (rng == "xsd:decimal" or not value.is_integer()):
        raise ValueOutOfRange(path, f"the float {value} is not exactly an {rng}; pass an int or a Decimal")
    exact = Decimal(value)  # type: ignore[arg-type]
    if rng == "xsd:decimal":
        text = format(exact.normalize(), "f")
        return {"@value": "0" if text in {"-0", "0"} else text, "@type": "xsd:decimal"}
    if exact != exact.to_integral_value():
        raise ValueOutOfRange(path, f"{exact} is not an xsd:integer")
    n = int(exact)
    return n if abs(n) <= MAX_SAFE_INT else {"@value": str(n), "@type": "xsd:integer"}


def _double(value: float | Decimal, path: str) -> object:
    if type(value) is float:
        f = value
    else:
        try:
            f = float(value)
        except OverflowError:
            raise ValueOutOfRange(path, "the number is outside xsd:double") from None
        if Decimal(f) != Decimal(value):
            raise ValueOutOfRange(path, "the number is not exactly an xsd:double")
    if not math.isfinite(f):
        return {"@value": _XSD_DOUBLE.get(f, "NaN"), "@type": "xsd:double"}
    return f


def _datetime(value: datetime, path: str) -> object:
    if value.utcoffset() is None:
        raise UnrecordableValue(path, "a time is recorded only with its UTC offset")
    text = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"@value": text, "@type": "xsd:dateTime"}


def _json_literal(value: object, path: str) -> object:
    try:
        return {"@value": jcs_loads(jcs_dumps(value)), "@type": "@json"}
    except (TypeError, ValueError) as err:
        raise UnrecordableValue(path, f"an rdf:JSON value must be canonical JSON: {err}") from None


def _sealed(node: Mapping[str, object], files: tuple[Record, ...] = ()) -> Record:
    return Record(jcs_dumps(dict(node)), files, seal=_SEAL)


def _addressed(prefix: str, tag: str, node: dict[str, object]) -> dict[str, object]:
    return {"@id": prefix + identity_digest(tag, node), **node}


def _sorted_unique(values: Iterable[object]) -> list[object]:
    unique = {jcs_dumps(v): v for v in values}
    return [unique[k] for k in sorted(unique)]


def entity(
    address: str,
    vocabulary: Vocabulary,
    *,
    types: Iterable[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> Record:
    """Return a `prov:Entity` whose `@id` is its content address (`content:<hex>` or `output:<hex>`).

    Raises:
        ValueError: if `address` is not a content address.
        UnknownTerm, ValueOutOfRange, UnrecordableValue: on a type or attribute the vocabulary refuses.
    """
    b = _Builder(vocabulary)
    t = b.types("prov:Entity", types)
    return b.record({"@id": _checked(ENTITY_ID, address, "address"), "@type": t, **b.attributes(t, attributes or {})})


def plan(plan_hash: str, vocabulary: Vocabulary) -> Record:
    """Return the `senselab:Plan` entity of a logical plan, `plan:<plan hash>`, with no attributes.

    Raises:
        ValueError: if `plan_hash` is not 64 lowercase hex digits.
    """
    b = _Builder(vocabulary)
    address = _checked(PLAN_ID, f"plan:{plan_hash}", "plan_hash")
    return b.record({"@id": address, "@type": b.types("prov:Entity", ["prov:Plan", "senselab:Plan"])})


def environment(vocabulary: Vocabulary, attributes: Mapping[str, object]) -> Record:
    """Return a `senselab:Environment` entity, content-addressed over its node under `prov-node@1`.

    Raises:
        UnknownTerm, ValueOutOfRange, UnrecordableValue: on an attribute the vocabulary refuses.
    """
    b = _Builder(vocabulary)
    t = b.types("prov:Entity", ["senselab:Environment"])
    return b.record(_addressed("environment:", PROV_NODE_TAG, {"@type": t, **b.attributes(t, attributes)}))


def agent(
    vocabulary: Vocabulary, *, types: Iterable[str] = (), attributes: Mapping[str, object] | None = None
) -> Record:
    """Return a `prov:Agent` (`types` adds subclasses), content-addressed over its node under `prov-node@1`.

    Raises:
        UnknownTerm, ValueOutOfRange, UnrecordableValue: on a type or attribute the vocabulary refuses.
    """
    b = _Builder(vocabulary)
    t = b.types("prov:Agent", types)
    return b.record(_addressed("agent:", PROV_NODE_TAG, {"@type": t, **b.attributes(t, attributes or {})}))


def activity(  # noqa: PLR0913 - keyword-only, one argument per PROV relation
    vocabulary: Vocabulary,
    *,
    types: Iterable[str] = (),
    attributes: Mapping[str, object] | None = None,
    used: Iterable[Usage] = (),
    generated: Iterable[Generation] = (),
    associations: Iterable[Association] = (),
    started: datetime | None = None,
    ended: datetime | None = None,
) -> Record:
    """Return a `prov:Activity` whose `@id` is `activity:` + `identity_digest("prov-activity@1", <node without @id>)`.

    Every relation is written as its unqualified PROV property (`prov:used`, `prov:generated`,
    `prov:wasAssociatedWith`) and, where a port, time or node id attaches, its qualified form: `prov:qualifiedUsage`,
    `prov:qualifiedAssociation` (whose `prov:hadRole` carries `senselab:nodeId` and `prov:hadPlan` the plan's `@id`,
    when the association names them), and `prov:qualifiedGeneration` of
    the entity, written inside this node through `@reverse`. Relations are sorted, so their order does not change
    the `@id`. The logical key of an execution is its `senselab:logicalKey` attribute.

    Raises:
        UnknownTerm, ValueOutOfRange, UnrecordableValue: on a type, attribute or time the vocabulary refuses.
    """
    b = _Builder(vocabulary)
    t = b.types("prov:Activity", types)
    node: dict[str, object] = {"@type": t, **b.attributes(t, attributes or {})}
    times = {"prov:startedAtTime": started, "prov:endedAtTime": ended}
    node.update({k: b.scalar(v, "xsd:dateTime", k) for k, v in times.items() if v is not None})
    usages, generations, assocs = list(used), list(generated), list(associations)
    _put(node, "prov:used", [{"@id": _checked(ENTITY_ID, u.entity, "used")} for u in usages])
    _put(node, "prov:qualifiedUsage", [_qualified(b, "prov:Usage", u) for u in usages if u.port or u.time])
    _put(node, "prov:generated", [{"@id": _checked(ENTITY_ID, g.entity, "generated")} for g in generations])
    reverse = [_qualified(b, "prov:Generation", g) for g in generations if g.port or g.time]
    if reverse:
        node["@reverse"] = {"prov:activity": _sorted_unique(reverse)}
    _put(
        node,
        "prov:wasAssociatedWith",
        [{"@id": _checked(AGENT_ID, a.agent, "associations")} for a in assocs if a.agent is not None],
    )
    _put(node, "prov:qualifiedAssociation", [_association(b, a) for a in assocs])
    return b.record(_addressed("activity:", PROV_ACTIVITY_TAG, node))


def _put(node: dict[str, object], key: str, values: list[object]) -> None:
    if values:
        node[key] = _sorted_unique(values)


def _checked(pattern: "re.Pattern[str]", node_id: str, what: str) -> str:
    if not pattern.fullmatch(node_id):
        raise ValueError(f"{what} must be an @id of the form {pattern.pattern}")
    return node_id


def _role(b: _Builder, term: str, value: str) -> dict[str, object]:
    return {"@type": "prov:Role", **b.attributes(["prov:Role"], {term: value})}


def _qualified(b: _Builder, kind: str, rel: Usage | Generation) -> dict[str, object]:
    q: dict[str, object] = {"@type": kind}
    if rel.port is not None:
        q["prov:hadRole"] = _role(b, "senselab:port", rel.port)
    if rel.time is not None:
        q["prov:atTime"] = b.scalar(rel.time, "xsd:dateTime", "prov:atTime")
    if kind == "prov:Usage":
        q["prov:entity"] = {"@id": rel.entity}
    else:
        q["@reverse"] = {"prov:qualifiedGeneration": {"@id": rel.entity}}
    return q


def _association(b: _Builder, a: Association) -> dict[str, object]:
    if a.agent is None and a.plan is None:
        raise ValueError("associations: an association names an agent, a plan or both")
    q: dict[str, object] = {"@type": "prov:Association"}
    if a.agent is not None:
        q["prov:agent"] = {"@id": a.agent}
    if a.node_id is not None:
        q["prov:hadRole"] = _role(b, "senselab:nodeId", a.node_id)
    if a.plan is not None:
        q["prov:hadPlan"] = {"@id": _checked(PLAN_ID, a.plan, "associations")}
    return q


def literal_value(literal: object) -> object:
    """Return the Python value a JSON-LD literal written by these builders stands for.

    JSON scalars are themselves; `xsd:double` gives a float (`INF`, `-INF`, `NaN` included), `xsd:integer` an int,
    `xsd:decimal` a `Decimal`, `xsd:dateTime` an aware `datetime`, `@json` its JSON value, `@set` a frozenset,
    `@list` a tuple and `{"@id": ...}` the id.

    Raises:
        ValueError: on a literal these builders do not write; the message does not echo it.
    """
    if not isinstance(literal, Mapping):
        return literal
    if set(literal) == {"@id"}:
        return literal["@id"]
    if set(literal) == {"@set"}:
        return frozenset(literal_value(v) for v in literal["@set"])
    if set(literal) == {"@list"}:
        return tuple(literal_value(v) for v in literal["@list"])
    if set(literal) == {"@value", "@type"}:
        kind, text = literal["@type"], literal["@value"]
        if kind == "@json":
            return text
        if isinstance(kind, str) and kind in _READERS and isinstance(text, str):
            try:
                return _READERS[kind](text)
            except (ArithmeticError, ValueError):
                pass
    raise ValueError("not a literal these builders write")


def _read_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


_READERS: Mapping[str, Callable[[str], object]] = {
    "xsd:double": float,
    "xsd:integer": int,
    "xsd:decimal": Decimal,
    "xsd:dateTime": _read_time,
}


def document(records: Iterable[Record]) -> dict[str, object]:
    """Return the `provenance@3.0` document of `records` and the files they name.

    It is `{"@context": CONTEXT_URL, "format": PROVENANCE_FORMAT, "@graph": [...]}`; the published context maps
    `format` to null, so JSON-LD processors ignore it. Nodes are in `@id` order. A node whose `@id` is not an
    address, or one `@id` held by two different nodes, is refused with `ValueError`.
    """
    nodes: dict[str, bytes] = {}
    for record in records:
        for r in (record, *record.files):
            if nodes.setdefault(_checked(NODE_ID, r.id, "a node"), r.jcs) != r.jcs:
                raise ValueError(f"two different nodes have the @id {r.id}")
    return {
        "@context": CONTEXT_URL,
        "format": PROVENANCE_FORMAT,
        "@graph": [jcs_loads(nodes[k]) for k in sorted(nodes)],
    }
