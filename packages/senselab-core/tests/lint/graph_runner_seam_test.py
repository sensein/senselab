"""The runner meets pydra's `GraphRunner` through a run-scoped store that pydra never reads results from.

`SenselabStore` defines none of the store methods through which `GraphRunner.ready` serves a job without sending it
(`read_succeeded`, `find`, `withdraw_success`, `work_dir`), and the runner builds its `GraphRunner` with
`lookup_before_dispatch=False`, so every leaf job reaches senselab's executor, which serves hits under the run's
reuse policy. Read from the source, so the check needs no pydra.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

RUNNER = Path(__file__).resolve().parents[2] / "src" / "senselab_core" / "runtime" / "execution" / "runner.py"
SERVING = frozenset({"read_succeeded", "find", "withdraw_success", "work_dir"})
"""The store attributes `GraphRunner` reads to settle a job from the store, or to place its work directory."""


def store_serving(source: str) -> set[str]:
    """Return the serving attributes the class `SenselabStore` of `source` defines."""
    [store] = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef) and n.name == "SenselabStore"]
    defined = {n.name for n in store.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    for node in store.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        defined |= {t.id for t in targets if isinstance(t, ast.Name)}
    return defined & SERVING


def lookup_settings(source: str) -> list[object]:
    """Return the `lookup_before_dispatch` value of each `GraphRunner(...)` call in `source`; `None` where unset."""
    found: list[object] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GraphRunner":
            given = {k.arg: k.value for k in node.keywords}
            value = given.get("lookup_before_dispatch")
            found.append(value.value if isinstance(value, ast.Constant) else None)
    return found


def test_the_runners_store_defines_no_serving_method() -> None:
    """`SenselabStore` has no `read_succeeded`, `find`, `withdraw_success` or `work_dir`."""
    assert store_serving(RUNNER.read_text(encoding="utf-8")) == set()


def test_the_runner_builds_its_graph_runner_without_lookup_before_dispatch() -> None:
    """Exactly one `GraphRunner(...)` is built, with `lookup_before_dispatch=False` written out."""
    assert lookup_settings(RUNNER.read_text(encoding="utf-8")) == [False]


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("class SenselabStore:\n    def find(self, job): ...\n", {"find"}),
        ("class SenselabStore:\n    work_dir = None\n", {"work_dir"}),
        ("class SenselabStore:\n    read_succeeded: object = None\n", {"read_succeeded"}),
        ("class SenselabStore:\n    def lookup(self): ...\n", set()),
    ],
    ids=["method", "attribute", "annotated", "clean"],
)
def test_a_planted_serving_attribute_is_found(source: str, found: set[str]) -> None:
    """A serving attribute planted as a method, an attribute or an annotation is reported; `lookup` is not."""
    assert store_serving(source) == found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("GraphRunner(t, s, on_error='raise_at_end', lookup_before_dispatch=False)", [False]),
        ("GraphRunner(t, s, on_error='raise_at_end', lookup_before_dispatch=True)", [True]),
        ("GraphRunner(t, s, on_error='raise_at_end')", [None]),
    ],
    ids=["false", "true", "unset"],
)
def test_a_planted_graph_runner_call_is_read(source: str, found: list[object]) -> None:
    """A `GraphRunner` call with the option false, true or unset is read as written."""
    assert lookup_settings(source) == found
