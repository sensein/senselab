"""Every digest of canonical JSON goes through ``senselab_worker.canonical.tagged_digest``."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
ROOTS = ("packages", "src")
HASHERS = {"sha256", "sha256_bytes"}
CANONICAL = "packages/senselab-worker/src/senselab_worker/canonical.py"
HELPER = "tagged_digest"


def _name(func: ast.expr) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def untagged_digests(source: str, module: str) -> list[str]:
    """``module:line`` of each ``sha256(...)`` / ``sha256_bytes(...)`` call that has ``jcs_dumps(...)`` inside it.

    The body of ``tagged_digest`` in ``canonical.py`` is the one place such a call is allowed.
    """
    found: list[str] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (*scope, child.name) if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.Call) and _name(child.func) in HASHERS:
                nested = any(
                    isinstance(n, ast.Call) and _name(n.func) == "jcs_dumps"
                    for arg in (*child.args, *(k.value for k in child.keywords))
                    for n in ast.walk(arg)
                )
                if nested and not (module == CANONICAL and HELPER in scope):
                    found.append(f"{module}:{child.lineno}")
            visit(child, inner)

    visit(ast.parse(source), ())
    return found


def _sources() -> list[Path]:
    """Every production source file: test trees are excluded, since they compare tagged against bare digests."""
    return [
        p
        for root in ROOTS
        for p in (REPO / root).rglob("*.py")
        if "tests" not in p.relative_to(REPO).parts and not p.name.endswith("_test.py") and ".venv" not in p.parts
    ]


@pytest.mark.logic
def test_no_digest_of_jcs_bytes_outside_the_tagged_helper() -> None:
    """No production file under ``packages/`` or ``src/`` hashes ``jcs_dumps(...)`` directly."""
    assert any(p.name == "canonical.py" for p in _sources())
    hits = [h for p in _sources() for h in untagged_digests(p.read_text(), p.relative_to(REPO).as_posix())]
    assert hits == []


@pytest.mark.logic
def test_the_helper_is_the_one_allowed_site() -> None:
    """``tagged_digest`` hashes JCS bytes in ``canonical.py``, and the same code anywhere else is reported."""
    text = (REPO / CANONICAL).read_text()
    assert "jcs_dumps" in text and untagged_digests(text, CANONICAL) == []
    assert len(untagged_digests(text, "other/canonical.py")) == 1


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        "hashlib.sha256(jcs_dumps(x)).hexdigest()",
        "sha256(jcs_dumps(x)).hexdigest()",
        "sha256_bytes(jcs_dumps(x))",
        "hashlib.sha256(b'tag' + jcs_dumps(x)).hexdigest()",
        "hashlib.sha256(data=canonical.jcs_dumps({'a': [x]})).hexdigest()",
        "h.hexdigest() if sha256(prefix + jcs_dumps(x)) else 0",
    ],
)
def test_a_planted_untagged_digest_is_found(line: str) -> None:
    """Every spelling of a digest taken over JCS bytes is reported, in a function or at module level."""
    wrapped = f"def f(x):\n    return {line}\n"
    assert untagged_digests(wrapped, "pkg/mod.py") == ["pkg/mod.py:2"]
    assert untagged_digests(f"{line}\n", "pkg/mod.py") == ["pkg/mod.py:1"]


@pytest.mark.logic
def test_hashing_other_bytes_and_serialising_without_hashing_pass() -> None:
    """A digest of non-JCS bytes, and ``jcs_dumps`` used for something other than a digest, are not reported."""
    src = "sha256(b'x')\nsha256_bytes(canonical_npy_bytes(a))\nwrite(jcs_dumps(x))\ntagged_digest('t', x)\n"
    assert untagged_digests(src, "pkg/mod.py") == []
