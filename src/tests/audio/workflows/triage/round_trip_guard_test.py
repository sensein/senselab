"""What a triage node writes and what a reader reads back, held together.

Two guards over one defect class. The class and the four cases that produced it are
``specs/20260817-triage-workflow-dag/round-trip-guard.md``.

:class:`TestEverySelectorNamesSomethingWritten` is a static sweep: a reader that selects entities by
a literal value of a keying attribute must select a value some writer in the tree produces, and a
reader that additionally qualifies by a node name must name a node whose own module writes it.

:class:`TestEveryReportFieldSurvivesTheStore` is a round trip: every field of ``BranchReport`` goes
through ``write_report`` into a store and back out through VERDICT's own reader, under two values
that differ, so neither a dropped field nor a constant survives.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.triage import nodes as nodes_package
from senselab.audio.workflows.triage.nodes.branches import BRANCH_FAMILY, PROPOSERS
from senselab.audio.workflows.triage.nodes.common import software_agent, write_report
from senselab.audio.workflows.triage.nodes.verdict import _branch_report_from_entity
from senselab.audio.workflows.triage.vocabulary import STORE_ASSERTIONS, TASK, UNDETERMINED, BranchReport
from senselab.utils.prov_store import ProvStore

TRIAGE_ROOT = Path(nodes_package.__file__ or "").parent.parent

KEYED_ATTRIBUTES = ("name", "family", "role", "verb")
"""The entity attributes a reader selects on. Each is the ``==`` in a store scan."""

MINIMUM_SELECTOR_HELPERS = 4
"""How many selector helpers the sweep must discover before it is reading the tree at all."""

UNWRITTEN_ON_PURPOSE: dict[tuple[str, str], str] = {}
"""Selections knowingly matching no writer. Each entry is a reader whose absent result is the
design, not a defect; anything else here is the defect this file exists to stop."""

_FANOUT_CAP = 8
"""How many values one f-string placeholder may resolve to before the write is treated as dynamic."""


def _sources() -> dict[Path, ast.Module]:
    """Every triage module, parsed.

    Returns:
        The parsed module per path, in path order.
    """
    return {path: ast.parse(path.read_text()) for path in sorted(TRIAGE_ROOT.rglob("*.py"))}


TREES = _sources()


def _module_constants() -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings across the tree, minus any bound two ways.

    Returns:
        The constant name to its string value.
    """
    found: dict[str, str] = {}
    conflicting: set[str] = set()
    for tree in TREES.values():
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    if found.get(target.id, value.value) != value.value:
                        conflicting.add(target.id)
                    found[target.id] = value.value
    for name in conflicting:
        found.pop(name, None)
    return found


CONSTANTS = _module_constants()


def _signatures() -> dict[str, list[str]]:
    """Every function defined in the tree, by name, to its positional parameter names.

    Returns:
        The function name to its parameters; the longest signature wins a name collision.
    """
    found: dict[str, list[str]] = {}
    for tree in TREES.values():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
                if len(params) >= len(found.get(node.name, [])):
                    found[node.name] = params
    return found


SIGNATURES = _signatures()


def _called(call: ast.Call) -> str:
    """The bare name a call names.

    Args:
        call: The call node.

    Returns:
        The attribute or identifier called, or ``""`` for anything else.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _plain(node: ast.AST | None) -> list[str]:
    """The strings an expression is without any knowledge of its enclosing scope.

    Args:
        node: The expression, or None.

    Returns:
        Every string it can be, empty when it is not statically a string.
    """
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return [CONSTANTS[node.id]]
    if isinstance(node, ast.IfExp):
        return _plain(node.body) + _plain(node.orelse)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [value for element in node.elts for value in _plain(element)]
    if isinstance(node, ast.Dict):
        return [value for element in node.values for value in _plain(element)]
    return []


def _parameter_literals() -> dict[tuple[str, str], set[str]]:
    """The literal strings each ``(function, parameter)`` is called with anywhere in the tree.

    Returns:
        The parameter to every literal reaching it, for resolving an f-string inside that function.
    """
    found: dict[tuple[str, str], set[str]] = {}
    for tree in TREES.values():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _called(node)
            params = SIGNATURES.get(name)
            if not params:
                continue
            for index, argument in enumerate(node.args):
                if index < len(params):
                    found.setdefault((name, params[index]), set()).update(_plain(argument))
            for keyword in node.keywords:
                if keyword.arg in params:
                    found.setdefault((name, str(keyword.arg)), set()).update(_plain(keyword.value))
    return found


PARAMETER_LITERALS = _parameter_literals()


def _written_values(node: ast.AST | None, scope: str) -> list[str]:
    """The strings a written expression can be, resolving f-strings through the scope's parameters.

    Args:
        node: The expression, or None.
        scope: The enclosing function's name, for resolving its parameters.

    Returns:
        Every string it can be, empty when it cannot be pinned down.
    """
    direct = _plain(node)
    if direct:
        return direct
    if isinstance(node, ast.Name):
        return sorted(PARAMETER_LITERALS.get((scope, node.id), set()))
    if not isinstance(node, ast.JoinedStr):
        return []
    parts: list[list[str]] = []
    for piece in node.values:
        if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
            parts.append([piece.value])
            continue
        if not isinstance(piece, ast.FormattedValue):
            return []
        resolved = sorted(set(_written_values(piece.value, scope)))
        if not resolved or len(resolved) > _FANOUT_CAP:
            return []
        parts.append(resolved)
    return ["".join(combination) for combination in itertools.product(*parts)] if parts else []


def _selected_key(node: ast.AST) -> str | None:
    """The attribute an expression selects: ``x.attributes.get("k")`` or ``x.attributes["k"]``.

    Args:
        node: The expression being compared.

    Returns:
        The attribute name, or None when the expression is not an attribute lookup.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        base = node.func.value
        if isinstance(base, ast.Attribute) and base.attr == "attributes" and node.args:
            literal = _plain(node.args[0])
            return literal[0] if len(literal) == 1 else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "attributes":
        literal = _plain(node.slice)
        return literal[0] if len(literal) == 1 else None
    return None


def _selector_helpers() -> dict[str, dict[int, str]]:
    """Functions that select on a keyed attribute given as an argument, and which argument that is.

    A function qualifies when its body compares an entity's keyed attribute against one of its own
    parameters: ``find_measurement``, ``find_measurements``, ``resolve_stream``, ``_spans_of_family``.

    Returns:
        The function name to its selecting parameter positions, each with the attribute it selects.
    """
    found: dict[str, dict[int, str]] = {}
    for tree in TREES.values():
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [argument.arg for argument in (*function.args.posonlyargs, *function.args.args)]
            for node in ast.walk(function):
                if not isinstance(node, ast.Compare):
                    continue
                key = _selected_key(node.left)
                if key not in KEYED_ATTRIBUTES:
                    continue
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Name) and comparator.id in params:
                        found.setdefault(function.name, {})[params.index(comparator.id)] = str(key)
    return found


SELECTOR_HELPERS = _selector_helpers()


class _Writes(ast.NodeVisitor):
    """Every keyed-attribute value written in one module, by key, with the node answerable for it."""

    def __init__(self, node_name: str | None) -> None:
        self.node_name = node_name
        self.found: dict[str, set[str]] = {key: set() for key in KEYED_ATTRIBUTES}
        self._scope: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 — the visitor protocol
        """Track the enclosing function, whose parameters resolve an f-string written inside it."""
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802 — the visitor protocol
        """An attributes mapping written as a display: ``{"family": "voice", ...}``."""
        for key, value in zip(node.keys, node.values):
            literal = _plain(key)
            if len(literal) == 1 and literal[0] in KEYED_ATTRIBUTES:
                self.found[literal[0]].update(_written_values(value, self._scope[-1] if self._scope else ""))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 — the visitor protocol
        """An argument bound to a parameter a keyed attribute is named after."""
        scope = self._scope[-1] if self._scope else ""
        for keyword in node.keywords:
            if keyword.arg in KEYED_ATTRIBUTES:
                self.found[str(keyword.arg)].update(_written_values(keyword.value, scope))
        selects = SELECTOR_HELPERS.get(_called(node), {})
        params = SIGNATURES.get(_called(node), [])
        for index, argument in enumerate(node.args):
            if index not in selects and index < len(params) and params[index] in KEYED_ATTRIBUTES:
                self.found[params[index]].update(_written_values(argument, scope))
        self.generic_visit(node)


def _declared_node(tree: ast.Module) -> str | None:
    """The node a module writes on behalf of, from its own ``NODE`` binding.

    Args:
        tree: The parsed module.

    Returns:
        The node name, or None for a shared module that writes on behalf of whoever calls it.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "NODE" and isinstance(node.value.value, str):
                    return node.value.value
    return None


def _write_index() -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Every keyed-attribute value written in the tree, overall and per answerable node.

    Returns:
        ``(anywhere, by_node)`` — the values written by any module keyed by attribute, and the
        values each node's own module writes. A value written by a shared module reaches every
        node, because a shared writer writes on behalf of whoever calls it.
    """
    anywhere: dict[str, set[str]] = {key: set() for key in KEYED_ATTRIBUTES}
    per_node: dict[str, dict[str, set[str]]] = {}
    shared: dict[str, set[str]] = {key: set() for key in KEYED_ATTRIBUTES}
    for tree in TREES.values():
        node_name = _declared_node(tree)
        scan = _Writes(node_name)
        scan.visit(tree)
        for key, values in scan.found.items():
            anywhere[key] |= values
            if node_name is None:
                shared[key] |= values
            else:
                per_node.setdefault(node_name, {key: set() for key in KEYED_ATTRIBUTES})[key] |= values
    declared = set(BRANCH_FAMILY.values())
    anywhere["family"] |= declared
    for node_name, family in BRANCH_FAMILY.items():
        per_node.setdefault(node_name, {key: set() for key in KEYED_ATTRIBUTES})["family"].add(family)
    for buckets in per_node.values():
        for key in KEYED_ATTRIBUTES:
            buckets[key] |= shared[key]
    return anywhere, per_node


WRITTEN_ANYWHERE, WRITTEN_BY_NODE = _write_index()


def _node_qualifiers(scope: ast.AST) -> set[str]:
    """The node names a reader body compares an activity's ``node`` against.

    Args:
        scope: The function body being read.

    Returns:
        Every literal compared against ``<...>.node``. Empty for an unqualified reader.
    """
    found: set[str] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "node"):
            continue
        for operator, comparator in zip(node.ops, node.comparators):
            if isinstance(operator, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                found.update(_plain(comparator))
    return found


def _reads() -> list[tuple[str, str, str, set[str]]]:
    """Every literal selection a triage reader makes.

    Returns:
        ``(site, key, value, node_qualifiers)`` per selection, where the qualifiers are the nodes
        the enclosing function restricts the selection to.
    """
    found: list[tuple[str, str, str, set[str]]] = []
    for path, tree in TREES.items():
        where = str(path.relative_to(TRIAGE_ROOT))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualifiers = _node_qualifiers(function)
            for node in ast.walk(function):
                site = f"{where}:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.Call):
                    for index, key in SELECTOR_HELPERS.get(_called(node), {}).items():
                        if len(node.args) > index:
                            found += [(key, value, site, qualifiers) for value in _plain(node.args[index])]
                elif isinstance(node, ast.Compare):
                    selected = _selected_key(node.left)
                    if selected not in KEYED_ATTRIBUTES:
                        continue
                    for operator, comparator in zip(node.ops, node.comparators):
                        if isinstance(operator, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)):
                            found += [(str(selected), value, site, qualifiers) for value in _plain(comparator)]
    return [(site, key, value, qualifiers) for key, value, site, qualifiers in found]


class TestEverySelectorNamesSomethingWritten:
    """A reader keyed to a value nobody writes renders nothing, forever, and passes every test.

    Four such readers shipped. The sweep reads both sides out of the tree, so the two cannot drift:
    the written set widens whenever a writer does, and a selector that no longer matches it fails
    here rather than going quiet.
    """

    def test_every_selected_value_is_written_somewhere(self) -> None:
        """A selector matching no writer at all — the ``phonation`` family, ``hear_span_window``."""
        unwritten = [
            f"{site} selects {key}={value!r}, which no triage writer produces"
            for site, key, value, _ in _reads()
            if value not in WRITTEN_ANYWHERE[key] and (key, value) not in UNWRITTEN_ON_PURPOSE
        ]
        assert unwritten == [], "\n".join(unwritten)

    def test_every_node_qualified_selection_is_written_by_that_node(self) -> None:
        """A selector matching a writer, but not the node it restricts itself to — AIRWAY's labels."""
        wrong_node = [
            f"{site} selects {key}={value!r} from {node}, which writes no such {key}"
            for site, key, value, qualifiers in _reads()
            for node in sorted(qualifiers)
            if node in WRITTEN_BY_NODE and value not in WRITTEN_BY_NODE[node][key]
        ]
        assert wrong_node == [], "\n".join(wrong_node)

    def test_every_exemption_is_still_needed(self) -> None:
        """An entry outliving the reader it excuses is an allowlist that has started lying."""
        selected = {(key, value) for _, key, value, _ in _reads()}
        stale = sorted(entry for entry in UNWRITTEN_ON_PURPOSE if entry not in selected)
        assert stale == [], f"exempted but no longer selected anywhere: {stale}"
        written = sorted(entry for entry in UNWRITTEN_ON_PURPOSE if entry[1] in WRITTEN_ANYWHERE[entry[0]])
        assert written == [], f"exempted but a writer now produces it; drop the entry: {written}"

    def test_the_sweep_sees_both_sides(self) -> None:
        """A sweep that parses nothing passes everything; these are the floors it must clear."""
        assert len(TREES) > 15, f"only {len(TREES)} triage modules parsed"
        assert len(_reads()) > 25, "the read side found almost no selectors"
        assert len(SELECTOR_HELPERS) >= MINIMUM_SELECTOR_HELPERS, f"only {sorted(SELECTOR_HELPERS)} discovered"
        for key in KEYED_ATTRIBUTES:
            assert WRITTEN_ANYWHERE[key], f"no writer of {key!r} found; the write side is not reading the tree"
        assert {proposer("x", (0.0, 1.0), "e").family for proposer in PROPOSERS.values()} == set(BRANCH_FAMILY.values())
        for node in ("AIRWAY", "SPEECH", "VOICE", "PREPROCESS"):
            assert node in WRITTEN_BY_NODE, f"{node} contributes no writes; node attribution is broken"


ROUND_TRIP_VALUES: dict[str, tuple[Any, Any]] = {
    "node": ("SPEECH", "VOICE"),
    "kind": ("speech", None),
    "conformance": (False, UNDETERMINED),
    "conformance_of": (TASK, STORE_ASSERTIONS),
    "deviations": (("omission",), ()),
    "unmeasured": (("branch.target_match_cosine",), ()),
    "in_family": (True, False),
}
"""Two distinguishable values per ``BranchReport`` field. A field stored as a constant fails on one
of its two, so neither a dropped field nor a frozen one survives."""


def _round_trip(**written: Any) -> BranchReport:  # noqa: ANN401 — the writer's own keyword arguments
    """One report through ``write_report`` into a store and back out through VERDICT's reader.

    Args:
        **written: The writer's keyword arguments.

    Returns:
        The report VERDICT reads back off the entity.
    """
    store = ProvStore(run_id="round-trip-guard")
    agent = software_agent(store)
    activity = store.activity(node=str(written["node"]), step="branch", parameters={})
    entity_id, _ = write_report(store, activity, agent, detail={}, **written)
    return _branch_report_from_entity(store.get_entity(entity_id))


class TestEveryReportFieldSurvivesTheStore:
    """``write_report`` stored ``unmeasured``; VERDICT's reader dropped it, and the flag was dead.

    Every other test builds a ``BranchReport`` directly, which cannot see a field the writer stores
    and the reader never reads. This drives both values off the dataclass, so a field added to
    ``BranchReport`` is covered the moment it exists.
    """

    def test_the_table_covers_every_field(self) -> None:
        """A new field with no declared pair would otherwise be round-tripped by nothing."""
        declared = {field.name for field in fields(BranchReport)}
        assert declared == set(ROUND_TRIP_VALUES), (
            f"BranchReport fields without a round-trip pair: {sorted(declared - set(ROUND_TRIP_VALUES))}; "
            f"pairs naming no field: {sorted(set(ROUND_TRIP_VALUES) - declared)}"
        )

    @pytest.mark.parametrize("field_name", sorted(ROUND_TRIP_VALUES))
    @pytest.mark.parametrize("which", (0, 1))
    def test_the_field_survives_the_store(self, field_name: str, which: int) -> None:
        """The written value is what the reader reads back, under both of the field's two values."""
        written = {name: pair[which] for name, pair in ROUND_TRIP_VALUES.items()}
        written[field_name] = ROUND_TRIP_VALUES[field_name][which]
        read_back = _round_trip(**written)
        assert getattr(read_back, field_name) == written[field_name], (
            f"write_report stored {field_name}={written[field_name]!r}; the reader read "
            f"{getattr(read_back, field_name)!r}"
        )

    def test_the_writer_accepts_every_field(self) -> None:
        """A field the writer cannot be told is one the store can never carry."""
        import inspect

        accepted = set(inspect.signature(write_report).parameters)
        missing = {field.name for field in fields(BranchReport)} - accepted
        assert missing == set(), f"BranchReport fields write_report takes no argument for: {sorted(missing)}"
