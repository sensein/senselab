"""PROV records in compact PROV-O JSON-LD and the `senselab:` vocabulary."""

import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

import numpy
import pytest
from senselab_core.kernel import prov
from senselab_core.kernel.prov import (
    CONTEXT_URL,
    PREFIXES,
    Association,
    Generation,
    UnknownTerm,
    UnrecordableValue,
    Usage,
    ValueOutOfRange,
    Vocabulary,
    VocabularyError,
    literal_value,
)
from senselab_core.runtime.vocabulary import context_document, core_vocabulary, read_prov_file
from senselab_worker.canonical import jcs_dumps, jcs_loads
from senselab_worker.digest import identity_digest, npy_digest

CORE = core_vocabulary()
E1 = "content:" + "1" * 64
E2 = "output:" + "2" * 64
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def _vocab_doc(*nodes: dict[str, Any], version: str = "senselab-vocab@1.0") -> dict[str, Any]:
    ontology = {"@id": "senselab:", "@type": "owl:Ontology", "owl:versionInfo": version}
    return {"@context": CONTEXT_URL, "@graph": [ontology, *nodes]}


def _prop(term: str, domain: str, rng: str) -> dict[str, Any]:
    return {"@id": term, "@type": "rdf:Property", "rdfs:domain": {"@id": domain}, "rdfs:range": {"@id": rng}}


EXTENDED = Vocabulary.from_document(
    _vocab_doc(
        {"@id": "senselab:Recording", "@type": "rdfs:Class", "rdfs:subClassOf": {"@id": "senselab:File"}},
        _prop("senselab:level", "prov:Entity", "xsd:double"),
        _prop("senselab:count", "prov:Entity", "xsd:integer"),
        _prop("senselab:amount", "prov:Entity", "xsd:decimal"),
        _prop("senselab:label", "prov:Entity", "xsd:string"),
        _prop("senselab:recordedAt", "prov:Entity", "xsd:dateTime"),
        _prop("senselab:waveform", "prov:Entity", "senselab:File"),
        _prop("senselab:recording", "prov:Entity", "senselab:Recording"),
    ),
    base=CORE,
)


def _entity(**attributes: object) -> dict[str, object]:
    return prov.entity(E1, EXTENDED, attributes={f"senselab:{k}": v for k, v in attributes.items()}).node()


class Mode(Enum):
    """An enum with str values."""

    FAST = "fast"


class Level(Enum):
    """An enum with int values."""

    LOW = 1


@pytest.mark.logic
@pytest.mark.parametrize(
    "build",
    [
        lambda: prov.entity(E1, CORE, attributes={"senselab:notATerm": 1}),
        lambda: prov.entity(E1, CORE, attributes={"foaf:name": "x"}),
        lambda: prov.entity(E1, CORE, types=["senselab:Recording"]),
        lambda: prov.entity(E1, CORE, types=["senselab:Run"]),
        lambda: prov.entity(E1, CORE, attributes={"senselab:modelSource": "hf"}),
        lambda: prov.agent(CORE, types=["prov:Plan"]),
    ],
    ids=["unknown-property", "foreign-prefix", "unknown-class", "class-of-another-kind", "wrong-domain", "agent-plan"],
)
def test_unknown_term_refused(build: Callable[[], object]) -> None:
    """A type or attribute outside the loaded vocabulary, or defined for another class, is refused."""
    with pytest.raises(UnknownTerm):
        build()


@pytest.mark.logic
@pytest.mark.parametrize(
    ("term", "value", "path"),
    [
        ("senselab:modelSource", 3, "senselab:modelSource"),
        ("senselab:modelSource", True, "senselab:modelSource"),
        ("senselab:modelSource", ["hf", 1], "senselab:modelSource[1]"),
    ],
)
def test_value_out_of_range_refused(term: str, value: object, path: str) -> None:
    """A value outside its property's declared range is refused, naming its attribute path."""
    with pytest.raises(ValueOutOfRange) as err:
        prov.agent(CORE, attributes={term: value})
    assert err.value.path == path


@pytest.mark.logic
@pytest.mark.parametrize(
    ("attributes", "path"),
    [
        ({"count": 1.5}, "senselab:count"),
        ({"level": "loud"}, "senselab:level"),
        ({"label": Decimal("1.5")}, "senselab:label"),
        ({"recordedAt": "2026-11-01"}, "senselab:recordedAt"),
        ({"recording": numpy.zeros(2)}, "senselab:recording"),
    ],
)
def test_value_out_of_range_refused_for_each_range(attributes: dict[str, object], path: str) -> None:
    """Each literal range and a class range refuse a value of another kind."""
    with pytest.raises(ValueOutOfRange) as err:
        _entity(**attributes)
    assert err.value.path == path


@pytest.mark.logic
def test_minus_inf_round_trips_as_xsd_double() -> None:
    """-inf is an xsd:double typed literal, and reads back as -inf through its canonical bytes."""
    node = _entity(level=-math.inf)
    assert node["senselab:level"] == {"@value": "-INF", "@type": "xsd:double"}
    back = jcs_loads(jcs_dumps(node))
    assert isinstance(back, dict)
    assert literal_value(back["senselab:level"]) == -math.inf


@pytest.mark.logic
@pytest.mark.parametrize(
    ("attributes", "literal", "value"),
    [
        ({"level": math.inf}, {"@value": "INF", "@type": "xsd:double"}, math.inf),
        ({"level": numpy.float32(0.5)}, 0.5, 0.5),
        ({"level": 3}, 3, 3),
        ({"count": 2**60}, {"@value": str(2**60), "@type": "xsd:integer"}, 2**60),
        ({"count": numpy.int64(-7)}, -7, -7),
        ({"amount": Decimal("1.50")}, {"@value": "1.5", "@type": "xsd:decimal"}, Decimal("1.5")),
        ({"amount": Decimal("-0")}, {"@value": "0", "@type": "xsd:decimal"}, Decimal(0)),
        ({"label": Mode.FAST}, "fast", "fast"),
        ({"label": "é"}, "é", "é"),
        ({"label": {"b", "a", "b"}}, {"@set": ["a", "b"]}, frozenset({"a", "b"})),
        ({"label": ("b", "a")}, {"@list": ["b", "a"]}, ("b", "a")),
        (
            {"recordedAt": datetime(2026, 11, 1, 14, tzinfo=timezone(timedelta(hours=2)))},
            {"@value": "2026-11-01T12:00:00Z", "@type": "xsd:dateTime"},
            T0,
        ),
    ],
)
def test_values_are_json_ld_literals(attributes: dict[str, object], literal: object, value: object) -> None:
    """JSON scalars are themselves; other values are typed literals, sets sorted @set, enums their str value."""
    node = _entity(**attributes)
    (term,) = attributes
    assert node[f"senselab:{term}"] == literal
    assert literal_value(node[f"senselab:{term}"]) == value


@pytest.mark.logic
def test_nan_is_one_xsd_double() -> None:
    """Every NaN is the xsd:double NaN."""
    assert _entity(level=float("nan"))["senselab:level"] == {"@value": "NaN", "@type": "xsd:double"}
    assert math.isnan(literal_value({"@value": "NaN", "@type": "xsd:double"}))  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize(
    ("attributes", "path"),
    [
        ({"label": b"x"}, "senselab:label"),
        ({"label": bytearray(b"x")}, "senselab:label"),
        ({"label": PurePosixPath("/data/sub-01.wav")}, "senselab:label"),
        ({"label": object()}, "senselab:label"),
        ({"label": None}, "senselab:label"),
        ({"label": ["a", ["b"]]}, "senselab:label[1]"),
        ({"label": ["a", b"x"]}, "senselab:label[1]"),
        ({"label": {b"x"}}, "senselab:label{}"),
        ({"level": Level.LOW}, "senselab:level"),
        ({"amount": Decimal("NaN")}, "senselab:amount"),
        ({"recordedAt": datetime(2026, 11, 1)}, "senselab:recordedAt"),  # noqa: DTZ001 - the naive time under test
        ({"waveform": numpy.array([1 + 2j])}, "senselab:waveform"),
        ({"waveform": "content:" + "0" * 64}, "senselab:waveform"),
    ],
)
def test_unrecordable_value_names_attribute(attributes: dict[str, object], path: str) -> None:
    """Bytes, paths, Python objects, None, nested containers and formless values are refused by attribute path."""
    with pytest.raises(UnrecordableValue) as err:
        _entity(**attributes)
    assert err.value.path == path


@pytest.mark.logic
def test_unrecordable_json_names_attribute() -> None:
    """An rdf:JSON value that is not canonical JSON is refused naming its attribute."""
    with pytest.raises(UnrecordableValue) as err:
        prov.activity(CORE, attributes={"senselab:logicalKey": {"x": math.inf}})
    assert err.value.path == "senselab:logicalKey"


@pytest.mark.logic
def test_array_is_a_content_addressed_file_entity() -> None:
    """An array attribute names a senselab:File entity whose @id is the digest of its canonical .npy."""
    a = numpy.arange(4, dtype="<f4")
    record = prov.entity(E1, EXTENDED, attributes={"senselab:waveform": a})
    file_id = "content:" + npy_digest("npy@1", a)
    assert record.node()["senselab:waveform"] == {"@id": file_id}
    assert [f.node() for f in record.files] == [{"@id": file_id, "@type": ["prov:Entity", "senselab:File"]}]
    assert prov.entity(E1, EXTENDED, attributes={"senselab:waveform": a.astype(">f4")}) == record


@pytest.mark.logic
def test_a_record_value_is_referenced_by_its_id_and_range_checked() -> None:
    """A record whose types fall in a class range is referenced by @id; one outside it is refused."""
    recording = prov.entity(E2, EXTENDED, types=["senselab:Recording"])
    node = prov.entity(E1, EXTENDED, attributes={"senselab:recording": recording}).node()
    assert node["senselab:recording"] == {"@id": E2}
    with pytest.raises(ValueOutOfRange):
        prov.entity(E1, EXTENDED, attributes={"senselab:recording": prov.entity(E2, EXTENDED)})


def _execution(**changes: object) -> prov.Record:
    software = prov.agent(CORE, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": "hf"})
    fields: dict[str, Any] = {
        "types": [],
        "attributes": {"senselab:logicalKey": {"task": "t", "inputs": {"x": E1}}},
        "used": [Usage(E1, port="x", time=T0)],
        "generated": [Generation(E2, port="y")],
        "associations": [Association(software.id, node_id="node-1")],
        "started": T0,
        "ended": T0 + timedelta(seconds=1),
    }
    return prov.activity(CORE, **{**fields, **changes})


@pytest.mark.logic
def test_activity_id_is_the_digest_of_its_node_without_id() -> None:
    """An activity's @id is the prov-activity@1 digest of its node without @id; the logical key is an attribute."""
    node = _execution().node()
    rest = {k: v for k, v in node.items() if k != "@id"}
    assert node["@id"] == "activity:" + identity_digest("prov-activity@1", rest)
    assert node["senselab:logicalKey"] == {"@value": {"inputs": {"x": E1}, "task": "t"}, "@type": "@json"}


@pytest.mark.logic
def test_one_key_two_executions_two_activities() -> None:
    """Two executions of one logical key (another context, another time) are two activities."""
    first = _execution()
    assert _execution(ended=T0 + timedelta(seconds=2)).id != first.id
    key = {"task": "t", "inputs": {"x": E1}}
    other_context = _execution(attributes={"senselab:logicalKey": key}, types=["senselab:Run"])
    assert other_context.id != first.id
    assert other_context.node()["senselab:logicalKey"] == first.node()["senselab:logicalKey"]


@pytest.mark.logic
def test_relation_order_does_not_change_the_activity() -> None:
    """Relations are sorted, so the order they are given in does not change the node or its @id."""
    used = [Usage(E1, port="x"), Usage(E2, port="z")]
    assert _execution(used=used) == _execution(used=used[::-1])


@pytest.mark.logic
def test_activity_relations_are_prov_with_qualified_forms() -> None:
    """Unqualified relations always; qualified usage, generation and association where a role or time attaches."""
    node = _execution().node()
    associations = node["prov:qualifiedAssociation"]
    assert isinstance(associations, list) and len(associations) == 1
    association = associations[0]
    assert association["prov:hadRole"] == {"@type": "prov:Role", "senselab:nodeId": "node-1"}
    assert node["prov:used"] == [{"@id": E1}]
    assert node["prov:qualifiedUsage"] == [
        {
            "@type": "prov:Usage",
            "prov:entity": {"@id": E1},
            "prov:hadRole": {"@type": "prov:Role", "senselab:port": "x"},
            "prov:atTime": {"@value": "2026-11-01T12:00:00Z", "@type": "xsd:dateTime"},
        }
    ]
    assert node["prov:generated"] == [{"@id": E2}]
    assert node["@reverse"] == {
        "prov:activity": [
            {
                "@type": "prov:Generation",
                "prov:hadRole": {"@type": "prov:Role", "senselab:port": "y"},
                "@reverse": {"prov:qualifiedGeneration": {"@id": E2}},
            }
        ]
    }
    assert "prov:hadPlan" not in jcs_dumps(node).decode()
    bare = _execution(used=[Usage(E1)], generated=[Generation(E2)]).node()
    assert "prov:qualifiedUsage" not in bare
    assert "@reverse" not in bare


@pytest.mark.logic
def test_agents_and_environments_are_content_addressed() -> None:
    """An agent's or environment's @id is the prov-node@1 digest of its node; equal fields, equal id."""
    attributes = {"senselab:modelSource": "url"}
    a = prov.agent(CORE, types=["prov:SoftwareAgent"], attributes=attributes)
    node = a.node()
    assert node["@id"] == "agent:" + identity_digest("prov-node@1", {k: v for k, v in node.items() if k != "@id"})
    assert node["@type"] == ["prov:Agent", "prov:SoftwareAgent"]
    assert prov.agent(CORE, types=["prov:SoftwareAgent"], attributes=attributes) == a
    assert prov.agent(CORE, types=["prov:SoftwareAgent"], attributes={"senselab:modelSource": "hf"}) != a
    env = prov.environment(CORE, {})
    assert env.id.startswith("environment:")
    assert env.types() == ("prov:Entity", "senselab:Environment")


@pytest.mark.logic
@pytest.mark.parametrize(
    "build",
    [
        lambda: prov.entity("sub-01.wav", CORE),
        lambda: prov.entity("activity:" + "0" * 64, CORE),
        lambda: prov.activity(CORE, used=[Usage("/data/x.wav")]),
        lambda: prov.activity(CORE, associations=[Association("someone", node_id="n")]),
    ],
)
def test_ids_are_addresses(build: Callable[[], object]) -> None:
    """An entity is named by its content address and an association by an agent's address, never a name."""
    with pytest.raises(ValueError, match="@id"):
        build()


@pytest.mark.logic
def test_document_holds_each_node_once_in_id_order() -> None:
    """A document lists records and the files they name by @id; two different nodes under one @id are refused."""
    a = numpy.arange(3)
    first = prov.entity(E1, EXTENDED, attributes={"senselab:waveform": a})
    doc = prov.document([first, prov.entity(E2, EXTENDED, attributes={"senselab:waveform": a}), first])
    ids = [n["@id"] for n in doc["@graph"]]  # type: ignore[attr-defined]
    assert doc["@context"] == CONTEXT_URL
    assert ids == sorted({E1, E2, "content:" + npy_digest("npy@1", a)})
    with pytest.raises(ValueError, match="two different nodes"):
        prov.document([first, prov.entity(E1, EXTENDED)])


@pytest.mark.logic
@pytest.mark.parametrize(
    "doc",
    [
        _vocab_doc(version="senselab-vocab@2.0"),
        {"@context": "https://example.org/other.jsonld", "@graph": []},
        _vocab_doc({"@id": "senselab:Run", "@type": "rdfs:Class", "rdfs:subClassOf": {"@id": "prov:Activity"}}),
        _vocab_doc({"@id": "ex:Thing", "@type": "rdfs:Class", "rdfs:subClassOf": {"@id": "prov:Entity"}}),
        _vocab_doc({"@id": "senselab:Thing", "@type": "rdfs:Class", "rdfs:subClassOf": {"@id": "foaf:Person"}}),
        _vocab_doc(_prop("senselab:x", "prov:Entity", "xsd:float")),
        _vocab_doc(_prop("senselab:x", "senselab:Nothing", "xsd:string")),
        _vocab_doc({**_prop("senselab:x", "prov:Entity", "xsd:string"), "rdfs:subClassOf": {"@id": "prov:Entity"}}),
    ],
    ids=[
        "other-major",
        "other-context",
        "redefines",
        "outside-namespace",
        "unknown-superclass",
        "unknown-range",
        "unknown-domain",
        "mixed-node",
    ],
)
def test_malformed_vocabulary_refused(doc: dict[str, Any]) -> None:
    """A vocabulary of another format, outside the namespace, or redefining or misdeclaring a term is refused."""
    with pytest.raises(VocabularyError):
        Vocabulary.from_document(doc, base=CORE)


@pytest.mark.logic
def test_vocabulary_reads_a_later_minor() -> None:
    """A vocabulary of a later minor of senselab-vocab@1 is read."""
    vocab = Vocabulary.from_document(
        _vocab_doc(_prop("senselab:x", "prov:Entity", "xsd:string"), version="senselab-vocab@1.7"), base=CORE
    )
    assert vocab.property("senselab:x") == prov.Property(domain="prov:Entity", range="xsd:string")


@pytest.mark.contract
def test_core_vocabulary_holds_the_core_terms() -> None:
    """The packaged vocabulary defines exactly the core terms, and the packaged context exactly the prefixes."""
    for term, parent in {
        "senselab:Run": "prov:Activity",
        "senselab:SoftwareAgent": "prov:SoftwareAgent",
        "senselab:Environment": "prov:Entity",
        "senselab:File": "prov:Entity",
        "senselab:Plan": "prov:Plan",
    }.items():
        assert CORE.has_class(term) and CORE.is_a(term, parent)
    assert CORE.property("senselab:modelSource") == prov.Property(domain="prov:Agent", range="xsd:string")
    assert CORE.property("senselab:logicalKey") == prov.Property(domain="prov:Activity", range="rdf:JSON")
    context = context_document()
    assert isinstance(context, dict)
    assert {k: v for k, v in context["@context"].items() if k != "@version"} == dict(PREFIXES)


def _offline_loader(url: str, options: object = None) -> dict[str, object]:
    if url != CONTEXT_URL:
        raise AssertionError(f"the expansion asked for {url}; only the packaged context is served")
    return {"contextUrl": None, "documentUrl": url, "document": context_document()}


def _expanded_iris(value: object, keys: set[str], ids: set[str]) -> None:
    if isinstance(value, list):
        for v in value:
            _expanded_iris(v, keys, ids)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k == "@value":
                continue
            if k in {"@id", "@type"}:
                ids.update(v if isinstance(v, list) else [v])
            else:
                keys.add(k)
                _expanded_iris(v, keys, ids)


def _compact_keys(value: object) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_compact_keys(v) for v in value))
    if isinstance(value, dict):
        out = {k for k in value if not k.startswith("@")}
        return out.union(*(_compact_keys(v) for k, v in value.items() if k != "@value"))
    return set()


@pytest.mark.contract
def test_document_expands_with_pyld() -> None:
    """A document and the vocabulary expand offline to prov: and senselab: IRIs; no term dropped, none relative."""
    from pyld import jsonld  # noqa: PLC0415 - pyld is in the core test group only; local runs deselect this test

    software = prov.agent(CORE, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": "hf"})
    recording = prov.entity(E2, EXTENDED, types=["senselab:Recording"], attributes={"senselab:level": -math.inf})
    run = _execution(
        types=["senselab:Run"], associations=[Association(software.id, node_id="n")], used=[Usage(E1, port="x")]
    )
    vocabulary_doc = jsonld.expand(
        read_prov_file("senselab_core.runtime", "senselab.jsonld"), {"documentLoader": _offline_loader}
    )
    record_doc = prov.document([software, recording, run, prov.environment(CORE, {})])
    expanded = jsonld.expand(record_doc, {"documentLoader": _offline_loader})
    vocab_ns, prov_ns = PREFIXES["senselab"], PREFIXES["prov"]
    keys: set[str] = set()
    ids: set[str] = set()
    _expanded_iris(expanded, keys, ids)
    assert all(k.startswith(("@", prov_ns, vocab_ns)) for k in keys)
    expected = {PREFIXES[k.partition(":")[0]] + k.partition(":")[2] for k in _compact_keys(record_doc["@graph"])}
    assert {k for k in keys if not k.startswith("@")} == expected
    schemes = ("content:", "output:", "activity:", "agent:", "environment:", "_:", "@json")
    for iri in ids:
        assert iri.startswith((prov_ns, vocab_ns, PREFIXES["xsd"], *schemes)), iri
    value_types = {v["@type"] for v in _values(expanded) if "@type" in v}
    assert value_types == {PREFIXES["xsd"] + "double", PREFIXES["xsd"] + "dateTime", "@json"}
    vocab_keys: set[str] = set()
    vocab_ids: set[str] = set()
    _expanded_iris(vocabulary_doc, vocab_keys, vocab_ids)
    namespaces = tuple(PREFIXES.values())
    assert all(k.startswith(namespaces) for k in vocab_keys if not k.startswith("@"))
    assert all(i.startswith(namespaces) for i in vocab_ids)


def _values(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [x for v in value for x in _values(v)]
    if isinstance(value, dict):
        if "@value" in value:
            return [value]
        return [x for v in value.values() for x in _values(v)]
    return []
