"""Run state is written only through `StorageDir`: no hand-rolled exclusive or atomic write outside `storage.py`."""

import ast
from pathlib import Path

import pytest
from senselab_core import runtime

ROOT = Path(runtime.__file__).parent
STORAGE = ROOT / "storage.py"
ALLOWED: dict[tuple[str, str], str] = {
    ("envs.py", "write_text"): "uv's input files in a throwaway work directory, read once by the resolver",
    ("envs.py", "write_bytes"): "a worker module copied into the site-packages of an env being built",
}
"""`(file relative to runtime/, form)` to the reason that file may use it; each is a write that is not run state."""

OS_CALLS = {"link", "replace", "rename"}
FILE_METHODS = {"write_text", "write_bytes"}


def violations(source: str) -> list[tuple[int, str]]:
    """Return `(line, form)` of every hand-rolled write in `source`: `write_text`/`write_bytes`, `os.link`, ..."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "O_EXCL":
            found.append((node.lineno, "O_EXCL"))
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        method, owner = node.func.attr, node.func.value
        if method in FILE_METHODS:
            found.append((node.lineno, method))
        elif method in OS_CALLS and isinstance(owner, ast.Name) and owner.id == "os":
            found.append((node.lineno, f"os.{method}"))
        elif method == "replace" and len(node.args) == 1 and not node.keywords and ast.unparse(owner) != "dataclasses":
            found.append((node.lineno, "Path.replace"))
    return found


@pytest.mark.logic
@pytest.mark.parametrize(
    ("source", "form"),
    [
        ("path.write_text('x')", "write_text"),
        ("p.parent.joinpath('a').write_bytes(b'x')", "write_bytes"),
        ("os.link(a, b)", "os.link"),
        ("os.replace(a, b)", "os.replace"),
        ("os.rename(a, b)", "os.rename"),
        ("os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL)", "O_EXCL"),
        ("staging.replace(path)", "Path.replace"),
        ("self.path.with_name('x').replace(path)", "Path.replace"),
    ],
)
def test_a_planted_hand_rolled_write_is_refused(source: str, form: str) -> None:
    """Each form of an unsynced atomic or exclusive write is found."""
    assert [f for _, f in violations(source)] == [form]


@pytest.mark.logic
def test_ordinary_calls_are_not_refused() -> None:
    """`str.replace` and `dataclasses.replace` are not writes."""
    source = "a.replace('x', 'y')\ndataclasses.replace(s, interrupted=True)\nname.replace('.', '')"
    assert violations(source) == []


@pytest.mark.logic
def test_run_state_is_written_only_through_storage_dir() -> None:
    """No file under `runtime/` but `storage.py` writes with `write_text`, `os.link`, `O_EXCL` or `os.replace`."""
    found: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*.py")):
        if path == STORAGE:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for line, form in violations(path.read_text(encoding="utf-8")):
            found[f"{relative}:{line}"] = form
    refused = {site: form for site, form in found.items() if (site.rpartition(":")[0], form) not in ALLOWED}
    assert refused == {}
    used = {(site.rpartition(":")[0], form) for site, form in found.items()}
    assert set(ALLOWED) <= used, "an allowed write was removed or moved; drop it from ALLOWED"
