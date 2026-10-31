"""No workflow string outside an ``env:`` map interpolates an expression that can carry untrusted text.

Every ``${{ }}`` in every string value of every workflow and composite action is parsed and its result classified.
An expression is clean when its value cannot carry text an outsider chose: a literal, a trusted context
(``TRUSTED`` and ``TRUSTED_FAMILIES``), a boolean (comparison, ``!``, ``contains`` and the status functions), or
``fromJSON``/``toJSON``/``format``/``join``/``&&``/``||`` over clean operands. ``matrix.*`` is trusted in a job only
when every expression in that job's matrix definition is clean. ``env.*`` is never trusted. The exempt strings are
the values of the ``env`` maps GitHub defines (``ENV_MAPS``), and nothing else named ``env``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
TRUSTED = frozenset(
    {
        "github.token",
        "github.workflow",
        "github.event_name",
        "github.ref",
        "github.repository",
        "github.run_id",
        "github.sha",
        "github.event.pull_request.number",
        "github.event.pull_request.draft",
        "github.event.issue.number",
        "github.event.release.prerelease",
        "runner.os",
        "runner.arch",
        "runner.temp",
        "job.status",
    }
)
"""Context values no outsider can choose: numbers, booleans, names fixed by the repository and its tokens."""
TRUSTED_FAMILIES = (
    re.compile(r"secrets\.[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"vars\.[A-Za-z_][A-Za-z0-9_]*"),
    re.compile(r"steps\.[A-Za-z0-9_-]+\.(outputs\.[A-Za-z0-9_-]+|outcome|conclusion)"),
    re.compile(r"needs\.[A-Za-z0-9_-]+\.(outputs\.[A-Za-z0-9_-]+|result)"),
)
"""Context families the repository sets: its secrets and variables, and the outputs its own steps and jobs write."""
BOOLEAN_FUNCTIONS = frozenset({"contains", "startswith", "endswith", "success", "always", "failure", "cancelled"})
PASSING_FUNCTIONS = frozenset({"fromjson", "tojson", "format", "join"})
"""Functions whose result is clean exactly when every argument is."""
HASH_FUNCTIONS = frozenset({"hashfiles"})

_TOKEN = re.compile(
    r"\s*(?:(?P<string>'(?:[^']|'')*')|(?P<number>-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?))"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_-]*)|(?P<op>==|!=|<=|>=|&&|\|\||[<>!()\[\].,*]))"
)


class ExpressionError(ValueError):
    """An expression the guard cannot parse, which is an offender."""


def _tokens(text: str) -> list[tuple[str, str]]:
    out, at = [], 0
    while at < len(text.rstrip()):
        match = _TOKEN.match(text, at)
        if not match or match.end() == at:
            raise ExpressionError(f"cannot read {text[at:]!r}")
        kind = match.lastgroup
        assert kind is not None
        out.append((kind, match.group(kind)))
        at = match.end()
    return out


class _Parser:
    """A recursive-descent reading of the GitHub expression grammar that returns whether the result is clean."""

    def __init__(self, text: str, matrix_clean: bool) -> None:
        self.tokens = _tokens(text)
        self.at = 0
        self.matrix_clean = matrix_clean

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        token = self.peek()
        if token is None or (value is not None and token[1] != value):
            raise ExpressionError(f"expected {value or 'a token'}, found {token}")
        self.at += 1
        return token

    def parse(self) -> bool:
        clean = self.disjunction()
        if self.peek() is not None:
            raise ExpressionError(f"unexpected {self.peek()}")
        return clean

    def disjunction(self) -> bool:
        clean = self.conjunction()
        while self.peek() == ("op", "||"):
            self.take()
            clean = self.conjunction() and clean
        return clean

    def conjunction(self) -> bool:
        clean = self.comparison()
        while self.peek() == ("op", "&&"):
            self.take()
            clean = self.comparison() and clean
        return clean

    def comparison(self) -> bool:
        clean = self.unary()
        while self.peek() in {("op", o) for o in ("==", "!=", "<", "<=", ">", ">=")}:
            self.take()
            self.unary()
            clean = True
        return clean

    def unary(self) -> bool:
        if self.peek() == ("op", "!"):
            self.take()
            self.unary()
            return True
        return self.postfix()

    def postfix(self) -> bool:
        token = self.take()
        if token[0] in ("string", "number") or (token[0] == "name" and token[1] in ("true", "false", "null")):
            return True
        if token == ("op", "("):
            clean = self.disjunction()
            self.take(")")
            return clean
        if token[0] != "name":
            raise ExpressionError(f"unexpected {token}")
        if self.peek() == ("op", "("):
            return self.call(token[1].lower())
        path, dynamic = [token[1]], False
        while self.peek() in (("op", "."), ("op", "[")):
            if self.take()[1] == ".":
                part = self.take()
                if part[0] != "name" and part != ("op", "*"):
                    raise ExpressionError(f"unexpected {part} after '.'")
                path.append(part[1])
            else:
                start, index = self.at, self.peek()
                self.disjunction()
                self.take("]")
                if index is not None and index[0] == "string" and self.at == start + 2:
                    path.append(index[1][1:-1].replace("''", "'"))
                else:
                    dynamic = True
        return not dynamic and self.trusted(".".join(path))

    def call(self, function: str) -> bool:
        self.take("(")
        arguments = []
        if self.peek() != ("op", ")"):
            arguments.append(self.disjunction())
            while self.peek() == ("op", ","):
                self.take()
                arguments.append(self.disjunction())
        self.take(")")
        if function in BOOLEAN_FUNCTIONS or function in HASH_FUNCTIONS:
            return True
        return function in PASSING_FUNCTIONS and all(arguments)

    def trusted(self, path: str) -> bool:
        if path.startswith("matrix."):
            return self.matrix_clean
        return path in TRUSTED or any(f.fullmatch(path) for f in TRUSTED_FAMILIES)


def clean(expression: str, *, matrix_clean: bool = False) -> bool:
    """Whether ``expression`` (the text between ``${{`` and ``}}``) cannot carry untrusted text.

    Raises:
        ExpressionError: If the expression does not parse.
    """
    return _Parser(expression, matrix_clean).parse()


ENV_MAPS = (
    ("env",),
    ("jobs", "*", "env"),
    ("jobs", "*", "steps", "*", "env"),
    ("jobs", "*", "container", "env"),
    ("jobs", "*", "services", "*", "env"),
    ("runs", "steps", "*", "env"),
)
"""Where an ``env:`` map sits: workflow, job, step, job container and service, composite-action step."""


def is_env_map(path: tuple[str, ...]) -> bool:
    """Whether the key path ``path`` is one of ``ENV_MAPS``, ``*`` standing for any one key."""
    return any(len(m) == len(path) and all(a in ("*", b) for a, b in zip(m, path, strict=True)) for m in ENV_MAPS)


def _strings(
    node: Any,  # noqa: ANN401
    path: tuple[str, ...] = (),
    *,
    exempt: bool = True,
) -> Iterator[tuple[tuple[str, ...], str]]:
    """Yield ``(path, string)`` for every string value under ``node``, skipping ``env`` maps when ``exempt``."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = (*path, str(key))
            if not (exempt and is_env_map(child)):
                yield from _strings(value, child, exempt=exempt)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _strings(item, (*path, str(i)), exempt=exempt)
    elif isinstance(node, str):
        yield path, node


def string_offenders(text: str, *, matrix_clean: bool) -> list[str]:
    """Each ``${{ }}`` in ``text`` that is unterminated, unparseable or not clean."""
    found = re.findall(r"\$\{\{(.*?)\}\}", text, flags=re.S)
    out = []
    for expression in found:
        try:
            if not clean(expression, matrix_clean=matrix_clean):
                out.append(" ".join(expression.split()))
        except ExpressionError as exc:
            out.append(f"{' '.join(expression.split())} ({exc})")
    if text.count("${{") != len(found):
        out.append("unterminated ${{")
    return out


def _matrix_clean(job: Any) -> bool:  # noqa: ANN401
    """Whether every expression in the job's ``strategy.matrix`` is clean; a literal matrix has none."""
    matrix = job.get("strategy", {}).get("matrix", {}) if isinstance(job, dict) else {}
    return not any(string_offenders(s, matrix_clean=False) for _, s in _strings(matrix, exempt=False))


def document_offenders(document: Any) -> list[str]:  # noqa: ANN401
    """Each offending expression in one parsed workflow or action, with its key path."""
    out = []
    jobs = document.get("jobs", {}) if isinstance(document, dict) else {}
    for path, text in _strings(document):
        job = jobs.get(path[1]) if len(path) > 1 and path[0] == "jobs" else None
        in_matrix = len(path) > 3 and path[2:4] == ("strategy", "matrix")
        matrix_clean = job is not None and not in_matrix and _matrix_clean(job)
        out += [f"{'.'.join(path)}: {o}" for o in string_offenders(text, matrix_clean=matrix_clean)]
    return out


def interpolation_offenders(directory: Path) -> list[str]:
    """Parse every workflow and composite action under ``directory``; list each offending ``${{ }}``."""
    out = []
    for path in sorted([*directory.rglob("*.yml"), *directory.rglob("*.yaml")]):
        out += [f"{path.name}: {o}" for o in document_offenders(yaml.safe_load(path.read_text()))]
    return out


@pytest.mark.logic
def test_n5_no_workflow_interpolates_an_untrusted_expression() -> None:
    """Untrusted contexts reach a workflow only through ``env:`` values."""
    assert interpolation_offenders(REPO / ".github") == []


@pytest.mark.logic
@pytest.mark.parametrize(
    ("script", "bad"),
    [
        ("run: echo ${{ github.head_ref }}", True),
        ("run: |\n        echo ${{\n          github.head_ref }}", True),
        ("run: |\n        echo hi\n        echo ${{ github.event.pull_request.title }}", True),
        ("run: >\n        echo\n        ${{ github.head_ref }}", True),
        ("run: >\n        echo ${{\n        github.head_ref\n        }}", True),
        ("run: echo ${{ github.head_ref", True),
        ("run: echo ${{ matrix.python-version }}", False),
        ("run: |\n        echo ${{\n          github.event.pull_request.number }}", False),
        ('run: echo "$HEAD"', False),
    ],
)
def test_m3_interpolation_guard_inspects_whole_scripts(script: str, bad: bool, tmp_path: Path) -> None:
    """The guard parses YAML and scans each script as a whole: multi-line, literal and folded forms are all caught."""
    (tmp_path / "w.yaml").write_text(f"jobs:\n  j:\n    steps:\n    - {script}\n")
    assert bool(interpolation_offenders(tmp_path)) is bad


@pytest.mark.logic
def test_m3_interpolation_guard_covers_composite_actions_and_github_script(tmp_path: Path) -> None:
    """A composite action's ``run`` and a github-script ``with.script`` are scripts too."""
    (tmp_path / "action.yml").write_text("runs:\n  steps:\n  - run: echo ${{ github.head_ref }}\n")
    (tmp_path / "w.yaml").write_text(
        "jobs:\n  j:\n    steps:\n    - uses: actions/github-script@v7\n"
        "      with:\n        script: x(${{ github.head_ref }})\n"
    )
    assert len(interpolation_offenders(tmp_path)) == 2


TITLE = "${{ github.event.pull_request.title }}"
PLACES = {
    "shell": f'jobs:\n  j:\n    steps:\n    - run: x\n      shell: bash -c "{TITLE}" {{0}}\n',
    "with.command": f"jobs:\n  j:\n    steps:\n    - uses: a/b@v1\n      with:\n        command: {TITLE}\n",
    "with.args": f"jobs:\n  j:\n    steps:\n    - uses: docker://alpine\n      with:\n        args: {TITLE}\n",
    "with.inlineScript": f"jobs:\n  j:\n    steps:\n    - uses: azure/cli@v2\n      with:\n"
    f"        inlineScript: {TITLE}\n",
    "working-directory": f"jobs:\n  j:\n    steps:\n    - run: ls\n      working-directory: {TITLE}\n",
    "uses": f"jobs:\n  j:\n    steps:\n    - uses: ./{TITLE}\n",
    "name": f"jobs:\n  j:\n    name: {TITLE}\n    steps:\n    - run: ls\n",
    "runs-on": f"jobs:\n  j:\n    runs-on: {TITLE}\n",
    "outputs": f"jobs:\n  j:\n    outputs:\n      x: {TITLE}\n",
    "env read back": "jobs:\n  j:\n    steps:\n    - run: echo ${{ env.TITLE }}\n",
    "fromJSON of untrusted": "jobs:\n  j:\n    steps:\n    - run: echo ${{ fromJSON(github.event.comment.body) }}\n",
    "format of untrusted": "jobs:\n  j:\n    steps:\n    - run: echo ${{ format('{0}', github.head_ref) }}\n",
    "or of untrusted": "jobs:\n  j:\n    steps:\n    - run: echo ${{ github.head_ref || 'main' }}\n",
    "dynamic index": "jobs:\n  j:\n    steps:\n    - run: echo ${{ secrets[github.head_ref] }}\n",
    "unknown function": "jobs:\n  j:\n    steps:\n    - run: echo ${{ evil(github.sha) }}\n",
    "unparseable": "jobs:\n  j:\n    steps:\n    - run: echo ${{ github.sha ; rm }}\n",
    "with.env": f"jobs:\n  j:\n    steps:\n    - uses: a/b@v1\n      with:\n        env: {TITLE}\n",
    "matrix key env": f"jobs:\n  j:\n    strategy:\n      matrix:\n        env: ['{TITLE}']\n"
    "    steps:\n    - run: ls\n",
    "env under with": f"jobs:\n  j:\n    steps:\n    - uses: a/b@v1\n      with:\n        env:\n          A: {TITLE}\n",
    "env of a job named env": f"jobs:\n  env:\n    name: {TITLE}\n",
}


@pytest.mark.logic
@pytest.mark.parametrize("place", sorted(PLACES))
def test_xg11_an_untrusted_expression_is_refused_in_every_string(place: str, tmp_path: Path) -> None:
    """``shell``, ``with`` values, ``working-directory``, ``uses``, names and outputs are inspected, as ``run`` is."""
    (tmp_path / "w.yaml").write_text(PLACES[place])
    assert interpolation_offenders(tmp_path), place


@pytest.mark.logic
def test_xg11_env_values_are_the_one_exemption(tmp_path: Path) -> None:
    """The same untrusted expression passes as an ``env:`` value at workflow, job and step level."""
    (tmp_path / "w.yaml").write_text(
        f"env:\n  A: {TITLE}\njobs:\n  j:\n    env:\n      B: {TITLE}\n"
        f'    steps:\n    - run: echo "$A $B $C"\n      env:\n        C: {TITLE}\n'
    )
    assert interpolation_offenders(tmp_path) == []


LABELLED = "contains(github.event.pull_request.labels.*.name, 'x') && fromJSON('[\"3.11\"]') || fromJSON('[]')"
MATRIX_RUN = "    steps:\n    - run: echo ${{ matrix.v }}\n      with:\n        python-version: ${{ matrix.v }}\n"


@pytest.mark.logic
@pytest.mark.parametrize(
    ("matrix", "bad"),
    [
        ("v: ['3.11', '3.12']", False),
        ("v: ${{ fromJSON(github.event.pull_request.title) }}", True),
        ("include:\n        - v: ${{ github.head_ref }}", True),
        (f"v: ${{{{ {LABELLED} }}}}", False),
        ("v: ${{ fromJSON(needs.setup.outputs.versions) }}", False),
        ("v: ${{ fromJSON(env.VERSIONS) }}", True),
    ],
)  # fmt: skip
def test_xg11_matrix_values_are_trusted_only_when_the_matrix_is_clean(matrix: str, bad: bool, tmp_path: Path) -> None:
    """``matrix.*`` is untrusted in a job whose matrix definition holds an expression that is not clean."""
    (tmp_path / "w.yaml").write_text(f"jobs:\n  j:\n    strategy:\n      matrix:\n        {matrix}\n{MATRIX_RUN}")
    offenders = interpolation_offenders(tmp_path)
    assert bool([o for o in offenders if "matrix.v" in o]) is bad


@pytest.mark.logic
@pytest.mark.parametrize(
    "expression",
    [
        "github.event.pull_request.number",
        "secrets.HF_TOKEN",
        "github.event_name == 'pull_request'",
        "!github.event.release.prerelease",
        "always() && needs.start-runner.result == 'success'",
        "github.event.issue.number || github.event.pull_request.number",
        "contains(github.event.pull_request.labels.*.name, 'full-matrix')",
        "steps.gpu-config.outputs.instance || vars.AWS_INSTANCE_TYPE",
        "hashFiles('**/uv.lock')",
        "toJSON(fromJSON('[1, 2]'))",
    ],
)
def test_clean_expressions_pass(expression: str) -> None:
    """Literals, trusted contexts, booleans and pass-through functions of clean operands are clean."""
    assert clean(expression)


@pytest.mark.logic
@pytest.mark.parametrize(
    "matrix",
    [
        "env: ['${{ github.event.pull_request.title }}']",
        "include:\n        - env: ${{ github.head_ref }}",
    ],
)
def test_xg11_a_matrix_variable_named_env_is_not_an_env_map(matrix: str, tmp_path: Path) -> None:
    """``matrix.env`` is a matrix value: an untrusted definition makes both it and its use offenders."""
    (tmp_path / "w.yaml").write_text(
        f"jobs:\n  j:\n    strategy:\n      matrix:\n        {matrix}\n"
        "    steps:\n    - run: echo ${{ matrix.env }}\n"
    )
    offenders = interpolation_offenders(tmp_path)
    assert [o for o in offenders if "matrix.env" in o] and [o for o in offenders if "strategy.matrix" in o]


@pytest.mark.logic
@pytest.mark.parametrize(
    "path",
    [
        ("env",),
        ("jobs", "j", "env"),
        ("jobs", "j", "steps", "0", "env"),
        ("jobs", "j", "container", "env"),
        ("jobs", "j", "services", "db", "env"),
        ("runs", "steps", "2", "env"),
    ],
)
def test_the_env_maps_are_exactly_githubs(path: tuple[str, ...]) -> None:
    """Each position GitHub defines an ``env`` map at is exempt; the same key one level off is not."""
    assert is_env_map(path)
    assert not is_env_map((*path[:-1], "with", "env"))
    assert not is_env_map((*path[:-1], "strategy", "matrix", "env"))
