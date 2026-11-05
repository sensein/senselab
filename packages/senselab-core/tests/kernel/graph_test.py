"""The graph document: its checks against a registry, its order, its JSON round trip, digest and published schema."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError
from senselab_core.kernel.declare import Registry, TaskParams, TaskSpec, kernel_types, task
from senselab_core.kernel.graph import Graph, canonical, graph_digest, graph_schema, order, validate

ROOT = Path(__file__).resolve().parents[4]
_spec = importlib.util.spec_from_file_location("generate_graph_schema", ROOT / "scripts" / "generate_graph_schema.py")
assert _spec and _spec.loader
generator = importlib.util.module_from_spec(_spec)
sys.modules["generate_graph_schema"] = generator
_spec.loader.exec_module(generator)


def _registry() -> Registry:
    registry = Registry()
    for name, inputs, outputs in (
        ("speech.diarize", {"audio": "Audio"}, {"segments": "Intervals"}),
        ("speech.embed", {"audio": "Audio"}, {"embedding": "Embedding"}),
        ("speech.pool", {"embeddings": "list[Embedding]"}, {"embedding": "Embedding"}),
        ("signal.describe", {"series": "RegularSeries"}, {"note": "Annotation"}),
        ("speech.fit_space", {"first": "Audio", "second": "Audio"}, {"events": "Events"}),
        ("speech.score", {"audio": "Audio", "pool": "list[Embedding]"}, {"note": "Annotation"}),
    ):
        registry.add_task(
            TaskSpec(name=name, version="1", kind="transform", inputs=inputs, outputs=outputs, params=TaskParams)
        )
    return registry


def _node(node_id: str, task: str, version: str = "1") -> dict[str, Any]:
    return {"id": node_id, "task": task, "task_version": version}


def _graph(inputs: dict[str, str], nodes: list[dict[str, Any]], edges: list[tuple[str, str, str]]) -> Graph:
    return Graph.model_validate(
        {
            "inputs": inputs,
            "nodes": nodes,
            "edges": [{"source": s, "target": t, "kind": k} for s, t, k in edges],
        }
    )


def _codes(graph: Graph) -> list[tuple[str, str | None, str | None]]:
    return [(i.code, i.node, i.port) for i in validate(graph, _registry())]


@pytest.mark.logic
def test_list_into_item_port_without_map_is_unmapped_list() -> None:
    """A `list[Audio]` wired plainly into an `Audio` port is `unmapped_list`; the same edge as `map` is valid."""
    plain = _graph({"clips": "list[Audio]"}, [_node("e", "speech.embed")], [("input.clips", "e.audio", "plain")])
    assert _codes(plain) == [("unmapped_list", "e", "audio")]
    mapped = _graph({"clips": "list[Audio]"}, [_node("e", "speech.embed")], [("input.clips", "e.audio", "map")])
    assert _codes(mapped) == []


@pytest.mark.logic
def test_map_then_gather_is_valid_and_gather_needs_a_map() -> None:
    """A mapped node's output gathers into a `list[T]` port; a gather from an unmapped node is refused."""
    nodes = [_node("e", "speech.embed"), _node("p", "speech.pool")]
    gathered = _graph(
        {"clips": "list[Audio]"}, nodes, [("input.clips", "e.audio", "map"), ("e.embedding", "p.embeddings", "gather")]
    )
    assert _codes(gathered) == []
    unmapped = _graph(
        {"clip": "Audio"}, nodes, [("input.clip", "e.audio", "plain"), ("e.embedding", "p.embeddings", "gather")]
    )
    assert _codes(unmapped) == [("gather_without_map", "p", "embeddings")]
    not_a_list = _graph({"clip": "Audio"}, [_node("e", "speech.embed")], [("input.clip", "e.audio", "map")])
    assert _codes(not_a_list) == [("type_mismatch", "e", "audio")]


@pytest.mark.logic
def test_a_gather_into_a_mapped_node_is_nested_fan_out() -> None:
    """A gather into a node a map edge also targets is `nested_fan_out`; into an unmapped node it is valid."""
    edges = [("input.clips", "e.audio", "map"), ("e.embedding", "s.pool", "gather")]
    nested = _graph(
        {"clips": "list[Audio]"},
        [_node("e", "speech.embed"), _node("s", "speech.score")],
        [*edges, ("input.clips", "s.audio", "map")],
    )
    assert _codes(nested) == [("nested_fan_out", "s", "pool")]
    once = _graph(
        {"clips": "list[Audio]", "clip": "Audio"},
        [_node("e", "speech.embed"), _node("s", "speech.score")],
        [*edges, ("input.clip", "s.audio", "plain")],
    )
    assert _codes(once) == []


@pytest.mark.logic
def test_ports_types_tasks_and_bindings_are_checked() -> None:
    """Unknown tasks, versions and ports, a mismatched type and an unbound input are each reported once."""
    graph = Graph.model_validate(
        {
            "inputs": {"clip": "Audio"},
            "nodes": [_node("a", "speech.missing"), _node("d", "speech.diarize", "2"), _node("e", "speech.embed")],
            "edges": [
                {"source": "input.clip", "target": "d.audio"},
                {"source": "d.segments", "target": "e.audio"},
                {"source": "input.other", "target": "a.x"},
            ],
            "outputs": {"spk": "d.speakers", "seg": "d.segments"},
        }
    )
    assert _codes(graph) == [
        ("unknown_task", "a", None),
        ("version_mismatch", "d", None),
        ("type_mismatch", "e", "audio"),
        ("unknown_port", None, "other"),
        ("unknown_port", "d", "speakers"),
    ]
    unbound = _graph({}, [_node("e", "speech.embed")], [])
    assert _codes(unbound) == [("unbound_input", "e", "audio")]


@pytest.mark.logic
def test_a_protocol_port_takes_an_implementing_type() -> None:
    """An `Audio` source satisfies a `RegularSeries` port; an `Embedding` does not."""
    audio = _graph({"clip": "Audio"}, [_node("s", "signal.describe")], [("input.clip", "s.series", "plain")])
    assert _codes(audio) == []
    embedding = _graph({"v": "Embedding"}, [_node("s", "signal.describe")], [("input.v", "s.series", "plain")])
    assert _codes(embedding) == [("type_mismatch", "s", "series")]


@pytest.mark.logic
def test_cycle_reported_with_nodes() -> None:
    """Each node of a cycle is reported with the cycle's nodes; `order` refuses the graph."""
    nodes = [_node("x", "speech.diarize"), _node("b", "speech.embed"), _node("a", "speech.pool")]
    graph = _graph(
        {"clips": "list[Audio]"},
        nodes,
        [
            ("input.clips", "x.audio", "map"),
            ("a.embedding", "b.audio", "plain"),
            ("b.embedding", "a.embeddings", "gather"),
        ],
    )
    cycles = [i for i in validate(graph, _registry()) if i.code == "cycle"]
    assert [i.node for i in cycles] == ["a", "b"]
    assert all("['a', 'b']" in i.message for i in cycles)
    with pytest.raises(ValueError, match=r"cycles \[\['a', 'b'\]\]"):
        order(graph)


@pytest.mark.logic
def test_order_is_lexical_among_ready() -> None:
    """Among the nodes whose inputs are ready, the lexically smallest id runs first, whatever the declaration order."""
    nodes = [_node(i, "speech.embed") for i in ("z", "m", "c", "a")]
    graph = _graph({}, nodes, [("z.embedding", "c.audio", "plain"), ("m.embedding", "a.audio", "plain")])
    assert order(graph) == ("m", "a", "z", "c")
    assert order(_graph({}, nodes, [])) == ("a", "c", "m", "z")


@pytest.mark.logic
def test_graph_round_trips_json() -> None:
    """A graph with params, backend params, a model and outputs survives its JSON form unchanged."""
    graph = Graph.model_validate(
        {
            "inputs": {"clips": "list[Audio]"},
            "nodes": [
                {
                    **_node("e", "speech.embed"),
                    "params": {"window": 2, "names": ["a", "b"], "nested": {"k": None}},
                    "backend_params": {"speechbrain": {"batch": 4}},
                    "model": "speechbrain/spkrec@main",
                }
            ],
            "edges": [{"source": "input.clips", "target": "e.audio", "kind": "map"}],
            "outputs": {"embedding": "e.embedding"},
        }
    )
    back = Graph.model_validate_json(graph.model_dump_json())
    assert back == graph and back.model_dump() == graph.model_dump()
    assert json.loads(graph.model_dump_json())["format"] == "graph@1.0"
    assert graph_digest(back) == graph_digest(graph)


@pytest.mark.logic
def test_graph_digest_ignores_key_order() -> None:
    """Key order and node or edge declaration order leave the digest alone; a changed param does not."""
    first = {
        "nodes": [{**_node("a", "speech.embed"), "params": {"x": 1, "y": 2}}, _node("b", "speech.embed")],
        "edges": [{"source": "a.embedding", "target": "b.audio"}],
    }
    second = {
        "edges": [{"target": "b.audio", "source": "a.embedding"}],
        "nodes": [_node("b", "speech.embed"), {"params": {"y": 2, "x": 1}, **_node("a", "speech.embed")}],
    }
    assert graph_digest(Graph.model_validate(first)) == graph_digest(Graph.model_validate(second))
    changed = {
        **first,
        "nodes": [{**_node("a", "speech.embed"), "params": {"x": 2, "y": 2}}, _node("b", "speech.embed")],
    }
    assert graph_digest(Graph.model_validate(changed)) != graph_digest(Graph.model_validate(first))


@pytest.mark.logic
@pytest.mark.parametrize(
    "document",
    [
        {"nodes": [_node("a", "t"), _node("a", "t")]},
        {"nodes": [_node("input", "t")]},
        {"edges": [{"source": "a.x", "target": "input.y"}]},
        {"edges": [{"source": "a.x", "target": "b.y"}, {"source": "c.x", "target": "b.y"}]},
        {"edges": [{"source": "a.x", "target": "b.y", "kind": "route"}]},
        {"inputs": {"x": "list[list[Audio]]"}},
        {"inputs": {"a.b": "Audio"}},
        {"outputs": {"a.b": "e.v"}},
        {"format": "graph@2.0"},
    ],
)
def test_malformed_documents_are_refused(document: dict[str, Any]) -> None:
    """Repeated or reserved ids, an edge into an input, a doubly bound port, an unknown kind or major are refused."""
    with pytest.raises(ValidationError):
        Graph.model_validate(document)


@pytest.mark.contract
def test_published_graph_schema_has_no_drift(tmp_path: Path) -> None:
    """The committed schema is what the generator writes now; an edit to it, or its absence, is drift."""
    assert generator.main(["--check"]) == 0
    edited = tmp_path / "graph.schema.json"
    edited.write_text(generator.OUTPUT.read_text().replace('"Graph"', '"Other"'))
    assert generator.main(["--check", "--output", str(edited)]) == 1
    assert generator.main(["--check", "--output", str(tmp_path / "missing.json")]) == 1


@pytest.mark.contract
def test_graph_schema_validates_2020_12() -> None:
    """The schema is valid 2020-12, accepts a graph's JSON form and refuses a document without a `graph@1` format."""
    schema = graph_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)
    checker = jsonschema.Draft202012Validator(schema)
    graph = _graph({"clips": "list[Audio]"}, [_node("e", "speech.embed")], [("input.clips", "e.audio", "map")])
    assert checker.is_valid(json.loads(graph.model_dump_json()))
    assert checker.is_valid({"format": "graph@1.7", "later_field": 1})
    assert not checker.is_valid({"format": "graph@2.0"})
    assert not checker.is_valid({"nodes": []})
    assert not checker.is_valid({"format": "graph@1.0", "inputs": {"a.b": "Audio"}})
    assert not checker.is_valid({"format": "graph@1.0", "edges": [{"source": "a.x", "target": "b.y", "kind": "loop"}]})


@pytest.mark.logic
def test_graph_digest_leaves_format_out_and_refuses_a_later_minor() -> None:
    """The canonical form carries no `format`; a later-minor graph, whose extra fields were dropped, has no digest."""
    content = {"nodes": [_node("a", "speech.embed")]}
    current = Graph.model_validate({"format": "graph@1.0", **content})
    later = Graph.model_validate({"format": "graph@1.3", "added_in_1_3": {"x": 1}, **content})
    assert "format" not in canonical(current) and canonical(later) == canonical(current)
    assert graph_digest(current) == graph_digest(Graph.model_validate(content))
    with pytest.raises(ValueError, match=r"graph@1\.3"):
        graph_digest(later)


@pytest.mark.logic
def test_an_input_type_must_name_a_port_type() -> None:
    """A graph input whose type, or list element type, names no kernel or declared port type is a `type_mismatch`."""
    graph = _graph({"x": "Bogus", "ys": "list[Bogus]", "ok": "list[Audio]"}, [], [])
    assert _codes(graph) == [("type_mismatch", None, "x"), ("type_mismatch", None, "ys")]


@pytest.mark.logic
def test_two_map_edges_into_one_node_are_refused() -> None:
    """A node with two `map` edges has no defined fan-out and is a `type_mismatch` on the node."""
    nodes = [_node("s", "speech.fit_space")]
    graph = _graph(
        {"a": "list[Audio]", "b": "list[Audio]"}, nodes, [("input.a", "s.first", "map"), ("input.b", "s.second", "map")]
    )
    assert _codes(graph) == [("type_mismatch", "s", None)]


@pytest.mark.logic
def test_a_graph_is_not_a_port_type() -> None:
    """A graph is a document, not a port value: no port may name it, and a task declaring one is refused."""
    assert "Graph" not in kernel_types()

    class Planner:
        inputs = {"graph": Graph}
        outputs: dict[str, type] = {}

    with pytest.raises(TypeError, match="not kernel value types"):
        task(name="t.plan", version="1", kind="transform")(Planner)
