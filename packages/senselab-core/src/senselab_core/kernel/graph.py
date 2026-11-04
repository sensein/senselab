"""The graph document `graph@1.0`: task nodes joined by plain, `map` and `gather` edges, and its checks.

A graph is data: it is built from JSON (`Graph.model_validate_json`), checked against a registry with `validate`,
ordered with `order` and identified by `graph_digest`. `graph_schema` is its published JSON Schema.
"""

import heapq
import re
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AfterValidator, Field, PlainSerializer, StringConstraints, model_validator
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.declare import PORT_PROTOCOLS, SCHEMA_DIALECT, Registry, TaskSpec, UnknownTask, kernel_types
from senselab_core.kernel.formats import GRAPH_FORMAT, FormatTag, Formatted, later_minor
from senselab_core.kernel.identity import Params

GRAPH_DIGEST_TAG = "graph@1"
"""The identity tag of a graph's canonical JSON."""

INPUT = "input"
"""The node part of a reference to a graph input, `input.<name>`; no node takes this id."""

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_LIST = re.compile(rf"^list\[({_NAME})\]$")

NodeId = Annotated[str, StringConstraints(pattern=rf"^{_NAME}$")]
"""A node's id: letters, digits and underscores, not starting with a digit."""
PortName = Annotated[str, StringConstraints(pattern=rf"^{_NAME}$")]
"""A graph input's or output's name, of the same form as a node id."""
PortRef = Annotated[str, StringConstraints(pattern=rf"^{_NAME}\.{_NAME}$")]
"""`<node>.<port>`, or `input.<name>` for a graph input."""
PortType = Annotated[str, StringConstraints(pattern=rf"^({_NAME}|list\[{_NAME}\])$")]
"""A kernel port type name, or `list[T]` for a list of them."""


def _empty() -> Any:  # noqa: ANN401 - a pydantic field default
    return Field(default_factory=lambda: FrozenMapping({}))


EdgeKind = Literal["plain", "map", "gather"]
IssueCode = Literal[
    "unknown_task",
    "version_mismatch",
    "unknown_port",
    "type_mismatch",
    "unmapped_list",
    "gather_without_map",
    "cycle",
    "unbound_input",
]


def split(ref: str) -> tuple[str, str]:
    """Return the node and port of a `PortRef`."""
    node, port = ref.split(".")
    return node, port


class Node(Frozen):
    """One call of a declared task: its version, params, per-backend params and model reference."""

    id: NodeId
    task: NonEmptyStr
    task_version: NonEmptyStr
    params: Params = _empty()
    backend_params: Annotated[Mapping[str, Params], AfterValidator(FrozenMapping), PlainSerializer(thawed)] = _empty()
    model: NonEmptyStr | None = None


class Edge(Frozen):
    """A wire from an output or graph input to a node's input port.

    `map` runs the target once per element of a `list[T]` source; `gather` collects a mapped node's outputs into a
    list.
    """

    source: PortRef
    target: PortRef
    kind: EdgeKind = "plain"


class Graph(Formatted):
    """A `graph@1.0` document: typed inputs, task nodes, edges and named outputs.

    Node ids are distinct and none is `input`; an edge targets a node's input port, and binds a port no other edge
    binds.
    """

    format: FormatTag = GRAPH_FORMAT
    port_value: ClassVar[bool] = False
    inputs: Annotated[Mapping[PortName, PortType], AfterValidator(FrozenMapping), PlainSerializer(dict)] = _empty()
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    outputs: Annotated[Mapping[PortName, PortRef], AfterValidator(FrozenMapping), PlainSerializer(dict)] = _empty()

    @model_validator(mode="after")
    def _check_structure(self) -> "Graph":
        ids = [n.id for n in self.nodes]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"node ids {repeated} are repeated")
        if INPUT in ids:
            raise ValueError(f"no node may be named {INPUT!r}")
        targets = [e.target for e in self.edges]
        if any(split(t)[0] == INPUT for t in targets):
            raise ValueError("an edge cannot target a graph input")
        doubled = sorted({t for t in targets if targets.count(t) > 1})
        if doubled:
            raise ValueError(f"ports {doubled} are each bound by more than one edge")
        return self


class GraphIssue(Frozen):
    """One problem `validate` found: its code, the node and port it concerns, and what is wrong."""

    code: IssueCode
    node: NodeId | None
    port: NonEmptyStr | None
    message: NonEmptyStr


def _accepts(source: str, target: str) -> bool:
    """Whether a value of port type `source` may be wired into a port of type `target`."""
    if source == target:
        return True
    source_list, target_list = _LIST.match(source), _LIST.match(target)
    if source_list or target_list:
        return bool(source_list and target_list and _accepts(source_list[1], target_list[1]))
    types = kernel_types()
    if source not in types or target not in types:
        return False
    found, wanted = types[source], types[target]
    if wanted in PORT_PROTOCOLS:
        members = {n for n in vars(wanted) if not n.startswith("_")} | set(getattr(wanted, "__annotations__", {}))
        fields = getattr(found, "model_fields", {})
        return all(n in fields or hasattr(found, n) for n in members)
    return issubclass(found, wanted)


class _Check:
    """The state of one `validate` run: the declared specs of the nodes and the issues found."""

    def __init__(self, graph: Graph, registry: Registry) -> None:
        self.graph = graph
        self.issues: list[GraphIssue] = []
        self.specs: dict[str, TaskSpec] = {}
        for node in graph.nodes:
            try:
                spec = registry.spec(node.task)
            except UnknownTask:
                self.issue("unknown_task", node.id, None, f"node {node.id!r}: no task {node.task!r} is declared")
                continue
            self.specs[node.id] = spec
            if spec.version != node.task_version:
                self.issue(
                    "version_mismatch",
                    node.id,
                    None,
                    f"node {node.id!r} names {node.task} v{node.task_version}; v{spec.version} is declared",
                )

    def issue(self, code: IssueCode, node: str | None, port: str | None, message: str) -> None:
        self.issues.append(GraphIssue(code=code, node=node, port=port, message=message))

    def lookup(self, ref: str, side: Literal["inputs", "outputs"], context: str) -> str | None:
        """Return the type of port `ref` on `side` of its node (a graph input for `input.<name>`).

        None when the node's task is unknown, or after reporting `unknown_port` prefixed by `context`.
        """
        node, port = split(ref)
        if node == INPUT:
            declared: Mapping[str, str] = self.graph.inputs
            missing = f"the graph declares no input {port!r}"
        elif node in self.specs:
            declared = getattr(self.specs[node], side)
            missing = f"task {self.specs[node].name!r} has no {side[:-1]} {port!r}"
        elif node in {n.id for n in self.graph.nodes}:
            return None
        else:
            declared, missing = {}, f"the graph has no node {node!r}"
        if port not in declared:
            self.issue("unknown_port", None if node == INPUT else node, port, f"{context}: {missing}")
            return None
        return declared[port]


def _mapped(graph: Graph) -> set[str]:
    """The nodes that run once per element: a `map` edge's targets and every node a plain edge from one reaches."""
    mapped = {split(e.target)[0] for e in graph.edges if e.kind == "map"}
    grown = True
    while grown:
        grown = False
        for e in graph.edges:
            source, target = split(e.source)[0], split(e.target)[0]
            if e.kind == "plain" and source in mapped and target not in mapped:
                mapped.add(target)
                grown = True
    return mapped


def _check_edge(check: _Check, edge: Edge, mapped: set[str]) -> None:
    where = f"{edge.source} -> {edge.target}"
    source, target = check.lookup(edge.source, "outputs", where), check.lookup(edge.target, "inputs", where)
    node, port = split(edge.target)
    if edge.kind == "gather" and split(edge.source)[0] not in mapped:
        check.issue("gather_without_map", node, port, f"{where}: {split(edge.source)[0]!r} is not mapped")
        return
    if source is None or target is None:
        return
    listed = _LIST.match(source)
    if edge.kind == "map":
        if listed is None or not _accepts(listed[1], target):
            check.issue("type_mismatch", node, port, f"{where}: map needs a list of {target}, got {source}")
        return
    carried = f"list[{source}]" if edge.kind == "gather" else source
    if _accepts(carried, target):
        return
    if edge.kind == "plain" and listed is not None and _accepts(listed[1], target):
        check.issue("unmapped_list", node, port, f"{where}: a {source} into a {target} port needs a map edge")
    else:
        check.issue("type_mismatch", node, port, f"{where}: {carried} is not {target}")


def _successors(graph: Graph) -> dict[str, set[str]]:
    """Each node's id mapped to the ids of the nodes its outputs feed."""
    successors: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        source, target = split(e.source)[0], split(e.target)[0]
        if source in successors and target in successors:
            successors[source].add(target)
    return successors


def _topological(successors: Mapping[str, set[str]]) -> list[str]:
    """The ids in topological order, the lexically smallest ready id first; ids on or after a cycle are left out."""
    waiting = dict.fromkeys(successors, 0)
    for targets in successors.values():
        for t in targets:
            waiting[t] += 1
    ready = [v for v, n in waiting.items() if n == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        v = heapq.heappop(ready)
        ordered.append(v)
        for t in successors[v]:
            waiting[t] -= 1
            if waiting[t] == 0:
                heapq.heappush(ready, t)
    return ordered


def _reached(successors: Mapping[str, set[str]], start: str) -> set[str]:
    reached: set[str] = set()
    pending = list(successors[start])
    while pending:
        v = pending.pop()
        if v not in reached:
            reached.add(v)
            pending.extend(successors[v])
    return reached


def _cycles(graph: Graph) -> list[list[str]]:
    """The node sets of the graph's cycles, each sorted, in order of their smallest id."""
    successors = _successors(graph)
    left = sorted(set(successors) - set(_topological(successors)))
    reached = {v: _reached(successors, v) for v in left}
    cycles: list[list[str]] = []
    placed: set[str] = set()
    for v in left:
        if v in reached[v] and v not in placed:
            cycle = [w for w in left if w in reached[v] and v in reached[w]]
            placed.update(cycle)
            cycles.append(cycle)
    return cycles


def _check_declarations(check: _Check, registry: Registry) -> None:
    """Report graph inputs of no known port type, and nodes more than one `map` edge targets."""
    known = set(kernel_types()) | {t for s in registry.tasks() for t in (*s.inputs.values(), *s.outputs.values())}
    for name, declared in check.graph.inputs.items():
        listed = _LIST.match(declared)
        if (listed[1] if listed else declared) not in known:
            message = f"input {name!r}: {declared} names no kernel or declared port type"
            check.issue("type_mismatch", None, name, message)
    maps = [split(e.target)[0] for e in check.graph.edges if e.kind == "map"]
    for node_id in sorted({n for n in maps if maps.count(n) > 1}):
        check.issue("type_mismatch", node_id, None, f"node {node_id!r}: more than one map edge targets it")


def validate(graph: Graph, registry: Registry) -> list[GraphIssue]:
    """Return every issue of `graph` against the tasks declared in `registry`; an empty list is a valid graph."""
    check = _Check(graph, registry)
    _check_declarations(check, registry)
    mapped = _mapped(graph)
    for edge in graph.edges:
        _check_edge(check, edge, mapped)
    bound = {e.target for e in graph.edges}
    for node_id, spec in check.specs.items():
        for port in spec.inputs:
            if f"{node_id}.{port}" not in bound:
                check.issue("unbound_input", node_id, port, f"{node_id}.{port}: no edge binds this input")
    for name, ref in graph.outputs.items():
        check.lookup(ref, "outputs", f"output {name!r} ({ref})")
    for cycle in _cycles(graph):
        for node_id in cycle:
            check.issue("cycle", node_id, None, f"nodes {cycle} form a cycle")
    return check.issues


def order(graph: Graph) -> tuple[str, ...]:
    """Return the node ids in topological order, the lexically smallest ready id first.

    Raises:
        ValueError: if the graph has a cycle.
    """
    successors = _successors(graph)
    ordered = _topological(successors)
    if len(ordered) < len(successors):
        raise ValueError(f"the graph has cycles {_cycles(graph)}")
    return tuple(ordered)


def canonical(graph: Graph) -> dict[str, Any]:
    """Return the graph's JSON form without `format`, nodes sorted by id and edges by source, target and kind."""
    form: dict[str, Any] = graph.model_dump(mode="json", exclude={"format"})
    form["nodes"] = sorted(form["nodes"], key=lambda n: n["id"])
    form["edges"] = sorted(form["edges"], key=lambda e: (e["source"], e["target"], e["kind"]))
    return form


def graph_digest(graph: Graph) -> str:
    """Return `identity_digest("graph@1", canonical(graph))`.

    Raises:
        ValueError: if the graph declares a later minor than `graph@1.0`, whose fields this reader may not hold.
    """
    if later_minor(GRAPH_FORMAT, graph.format):
        raise ValueError(
            f"a {graph.format} graph has fields a {GRAPH_FORMAT} reader does not hold; it has no digest here"
        )
    return identity_digest(GRAPH_DIGEST_TAG, canonical(graph))


def graph_schema() -> dict[str, Any]:
    """Return the JSON Schema (2020-12) of a `graph@1` document; `format` is required and any minor of `graph@1`."""
    schema = Graph.model_json_schema(ref_template="#/$defs/{model}")
    name, major = GRAPH_FORMAT.split(".")[0].split("@")
    schema["properties"]["format"] = {
        "type": "string",
        "pattern": rf"^{name}@{major}\.\d+$",
        "description": f"Required in a published document; a reader given none reads the document as {GRAPH_FORMAT}.",
    }
    schema["required"] = sorted({"format", *schema.get("required", ())})
    schema["properties"]["inputs"]["additionalProperties"] = False
    schema["properties"]["outputs"]["additionalProperties"] = False
    return {"$schema": SCHEMA_DIALECT, "x-senselab-format": GRAPH_FORMAT, **schema}
