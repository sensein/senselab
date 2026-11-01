"""Fail when a locked package a 2.x distribution ships has a license its policy does not allow, or none.

The packages are the runtime dependencies of ``senselab``, ``senselab-core`` and ``senselab-worker`` in ``uv.lock``,
transitive ones included. Each package's license is read from the metadata installed in the environment, in this
order: the PEP 639 ``License-Expression``, a one-line ``License`` field that parses, a trove classifier, a dated
manual record. A package not installed here (a platform-only dependency) has only the manual record. The native
libraries a wheel bundles are found in its ``RECORD`` and each needs a record of its own. The policy, with the
named license sets, the set each distribution allows and ``on_missing``, is ``scripts/data/license_policy.yaml``.

Usage::

    uv run python scripts/check_licenses.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).resolve().parent / "data" / "license_policy.yaml"
LOCK = ROOT / "uv.lock"
POLICY_FORMAT = "license-policy@1.0"
DISTRIBUTIONS = ("senselab", "senselab-core", "senselab-worker")
ON_MISSING = ("raise", "warn")

_SHARED = re.compile(r"\.(?:so(?:\.\d+)*|dylib|dll)$")
_EXTENSION = re.compile(r"\.(?:cpython-\d+\w*-[\w-]+|abi3|pypy\d+-\w+)\.(?:so|dylib)$|\.pyd$")
_TOKEN = re.compile(r"\(|\)|[^\s()]+")


class Record(NamedTuple):
    """One license: the SPDX expression, where it was read, and the source or file it came from."""

    expression: str
    how: str
    source: str


class Policy(NamedTuple):
    """The policy file, validated."""

    on_missing: str
    sets: dict[str, frozenset[str]]
    distributions: dict[str, frozenset[str]]
    aliases: dict[str, str]
    classifiers: dict[str, str]
    manual: dict[str, Record]
    bundled: dict[str, Record]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _record(entry: dict[str, Any], what: str) -> Record:
    verified = entry.get("verified") or {}
    if not (entry.get("expression") and entry.get("source") and verified.get("date") and verified.get("how")):
        raise ValueError(f"{what}: needs expression, source and verified.date and verified.how")
    return Record(str(entry["expression"]), f"{verified['how']} {verified['date']}", str(entry["source"]))


def load_policy(path: Path = POLICY) -> Policy:
    """Return the policy at `path`; an entry without its required fields, or a name that resolves to nothing, is refused."""
    raw = yaml.safe_load(path.read_text())
    if raw.get("format") != POLICY_FORMAT:
        raise ValueError(f"{path}: format is {raw.get('format')!r}, not {POLICY_FORMAT!r}")
    if raw.get("on_missing") not in ON_MISSING:
        raise ValueError(f"{path}: on_missing must be one of {ON_MISSING}")
    sets = {}
    for name, entry in raw["sets"].items():
        if not str(entry.get("derivation", "")).strip() or not entry.get("ids"):
            raise ValueError(f"{path}: set {name} needs ids and a derivation")
        sets[name] = frozenset(i.lower() for i in entry["ids"])
    distributions = {}
    for name, entry in raw["distributions"].items():
        unknown = set(entry["allow"]) - set(sets)
        if unknown:
            raise ValueError(f"{path}: {name} allows unknown sets {sorted(unknown)}")
        distributions[name] = frozenset().union(*(sets[s] for s in entry["allow"]))
    manual = {_norm(m["package"]): _record(m, f"manual {m['package']}") for m in raw.get("manual") or []}
    bundled = {name: _record(entry, f"bundled {name}") for name, entry in (raw.get("bundled_libraries") or {}).items()}
    return Policy(
        raw["on_missing"], sets, distributions, raw["aliases"], raw["classifiers"], manual, bundled
    )


def locked_closure(lock: Path = LOCK, roots: tuple[str, ...] = DISTRIBUTIONS) -> dict[str, set[str]]:
    """For each package in the runtime closure of `roots`, the roots that need it, from the lock."""
    packages = {p["name"]: p for p in tomllib.loads(lock.read_text())["package"]}
    needed_by: dict[str, set[str]] = {}
    for root in roots:
        stack = [root]
        while stack:
            name = stack.pop()
            if root in needed_by.get(name, set()):
                continue
            needed_by.setdefault(name, set()).add(root)
            stack += [d["name"] for d in packages[name].get("dependencies", [])]
    return needed_by


def parse_expression(text: str) -> Any:
    """Parse an SPDX expression into nested tuples: ``("or", a, b)``, ``("and", a, b)`` or an id string."""
    tokens = _TOKEN.findall(text)
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"expression ends early: {text!r}")
        pos += 1
        return tokens[pos - 1]

    def atom() -> Any:
        if peek() == "(":
            take()
            inner = either()
            if peek() != ")":
                raise ValueError(f"unbalanced parentheses in {text!r}")
            take()
            return inner
        if peek() is None or peek() in (")", "AND", "OR", "WITH"):
            raise ValueError(f"expected a license id in {text!r}")
        name = take()
        if peek() == "WITH":
            take()
            name = f"{name} WITH {take()}"
        return name

    def both() -> Any:
        node = atom()
        while peek() == "AND":
            take()
            node = ("and", node, atom())
        return node

    def either() -> Any:
        node = both()
        while peek() == "OR":
            take()
            node = ("or", node, both())
        return node

    tree = either()
    if peek() is not None:
        raise ValueError(f"trailing text in {text!r}")
    return tree


def ids_of(tree: Any) -> list[str]:
    """Every license id in a parsed expression."""
    return [tree] if isinstance(tree, str) else [i for part in tree[1:] for i in ids_of(part)]


def allows(tree: Any, allowed: frozenset[str]) -> bool:
    """Whether the expression is satisfied by the `allowed` ids: ``AND`` needs both sides, ``OR`` either."""
    if isinstance(tree, str):
        return tree.lower() in allowed
    left, right = allows(tree[1], allowed), allows(tree[2], allowed)
    return (left and right) if tree[0] == "and" else (left or right)


def _known(policy: Policy) -> frozenset[str]:
    return frozenset().union(*policy.sets.values())


def read_license(dist: metadata.Distribution, policy: Policy) -> Record | None:
    """The license of an installed distribution, from the first source that gives one, or ``None``."""
    meta = dist.metadata
    known = _known(policy)
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return Record(expression, "pep639", "License-Expression")
    field = " ".join((meta.get("License") or "").split())
    if field and "\n" not in (meta.get("License") or "").strip() and len(field) <= 100:
        candidate = policy.aliases.get(field, field)
        try:
            if all(i.lower() in known for i in ids_of(parse_expression(candidate))):
                return Record(candidate, "license-field", "License")
        except ValueError:
            pass
    for classifier in meta.get_all("Classifier") or []:
        if classifier in policy.classifiers:
            return Record(policy.classifiers[classifier], "classifier", classifier)
    return None


def library_name(path: str) -> str:
    """The name of the shared library at `path` in a wheel: no ``lib``, extension, version, hash or architecture."""
    name = _SHARED.sub("", path.rsplit("/", 1)[-1])
    name = re.sub(r"(?:-[0-9a-f]{8,})+$", "", name)
    name = re.sub(r"_(?:arm64|x86_64|aarch64|universal2|amd64)$", "", name)
    name = re.sub(r"-\d+(?:\.\d+)*$", "", name)
    return name.removeprefix("lib").strip("_")


def bundled_libraries(dist: metadata.Distribution) -> dict[str, str]:
    """The native libraries the wheel carries, as ``name -> file``: shared libraries that are not extension modules."""
    found = {}
    for f in dist.files or []:
        text = str(f)
        if _SHARED.search(text) and not _EXTENSION.search(text):
            found.setdefault(library_name(text), text)
    return found


class Finding(NamedTuple):
    """One license decision: the subject, its expression, where it was read, and the problem if any."""

    subject: str
    version: str
    expression: str
    how: str
    needed_by: tuple[str, ...]
    problem: str


def _judge(subject: str, record: Record | None, needed_by: set[str], policy: Policy, version: str = "") -> Finding:
    users = tuple(sorted(needed_by))
    if record is None:
        return Finding(subject, version, "", "", users, "no license record")
    try:
        tree = parse_expression(record.expression)
    except ValueError as exc:
        return Finding(subject, version, record.expression, record.how, users, f"unparseable: {exc}")
    for dist in users:
        if not allows(tree, policy.distributions[dist]):
            return Finding(subject, version, record.expression, record.how, users, f"not allowed for {dist}")
    return Finding(subject, version, record.expression, record.how, users, "")


def check(closure: dict[str, set[str]], policy: Policy, installed: dict[str, metadata.Distribution]) -> list[Finding]:
    """One finding per locked package and one per native library its wheel bundles."""
    out = []
    for name in sorted(closure):
        dist = installed.get(_norm(name))
        record = read_license(dist, policy) if dist else None
        record = record or policy.manual.get(_norm(name))
        out.append(_judge(name, record, closure[name], policy, dist.version if dist else "not installed"))
        for lib, file in sorted(bundled_libraries(dist).items() if dist else []):
            out.append(_judge(f"{name} bundles {lib} ({file})", policy.bundled.get(lib), closure[name], policy))
    return out


def failures(findings: list[Finding], policy: Policy) -> list[Finding]:
    """The findings that fail the gate: any problem, except a missing record when `on_missing` is ``warn``."""
    return [f for f in findings if f.problem and not (f.problem == "no license record" and policy.on_missing == "warn")]


def table(findings: list[Finding]) -> str:
    """The findings as aligned columns."""
    rows = [("package", "version", "license", "read from", "needed by", "problem")]
    rows += [(f.subject, f.version, f.expression, f.how, ",".join(f.needed_by), f.problem) for f in findings]
    widths = [max(len(r[i]) for r in rows) for i in range(6)]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


def main(argv: list[str]) -> int:
    """Check the locked runtime closure of the three distributions; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write the findings here")
    args = parser.parse_args(argv)
    policy = load_policy()
    installed = {_norm(d.metadata["Name"]): d for d in metadata.distributions() if d.metadata["Name"]}
    findings = check(locked_closure(), policy, installed)
    print(table(findings))
    if args.json:
        args.json.write_text(json.dumps([f._asdict() for f in findings], indent=2) + "\n")
    bad = failures(findings, policy)
    for f in bad:
        print(f"FAIL {f.subject}: {f.problem}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
