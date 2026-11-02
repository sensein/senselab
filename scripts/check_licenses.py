"""Fail when a locked package a 2.x distribution ships has a license its policy does not allow, or none.

The packages are the runtime dependencies of ``senselab``, ``senselab-core`` and ``senselab-worker`` in ``uv.lock``,
transitive ones included. Each package's license is read from the metadata installed in the environment, in this
order: the PEP 639 ``License-Expression``, a one-line ``License`` field that parses, a trove classifier, a dated
manual record. A package not installed here (a platform-only dependency) has only the manual record. The native
libraries a wheel bundles are found in its ``RECORD`` and each needs a record of its own, declared under that package. The policy, with the
named license sets, the set each distribution allows and ``on_missing``, is ``scripts/data/license_policy.yaml``.

Env locks (``--lock NAME``, ``--host``) are read the same way over the pinned packages of a lock in
``data/env_locks/``, or of the root lock for ``--host``. A package's metadata comes from ``--site`` (the
site-packages of a built env) or, without it, from the PyPI JSON of its pinned version. ``--write`` merges the records
into ``data/env_licenses.yaml``.

Usage::

    uv run python scripts/check_licenses.py [--json PATH]
    uv run python scripts/check_licenses.py (--lock NAME | --host) [--site DIR] [--write]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from email.message import Message
from importlib import metadata
from datetime import date
from pathlib import Path
from typing import Any, Callable, NamedTuple

import yaml
from senselab_core.runtime.envs import lock_pins
from senselab_core.runtime.licenses import allows, ids_of, parse_expression

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).resolve().parent / "data" / "license_policy.yaml"
LOCK = ROOT / "uv.lock"
ENV_LOCKS = ROOT / "packages/senselab/src/senselab/data/env_locks"
ENV_LICENSES = ROOT / "packages/senselab/src/senselab/data/env_licenses.yaml"
ENV_LICENSES_FORMAT = "env-licenses@1.0"
INSTALLER = "env-lock"
POLICY_FORMAT = "license-policy@1.2"
DISTRIBUTIONS = ("senselab", "senselab-core", "senselab-worker")
"""The shipped distributions; ``env-lock`` is the policy entry the pins of a backend env are judged against."""
ON_MISSING = ("raise", "warn")

_SHARED = re.compile(r"\.(?:so(?:\.\d+)*|dylib|dll)$")
_EXTENSION = re.compile(r"\.(?:cpython-\d+\w*-[\w-]+|abi3|pypy\d+-\w+)\.(?:so|dylib)$|\.pyd$")


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
    bundled_allow: dict[str, frozenset[str]]
    aliases: dict[str, str]
    classifiers: dict[str, str]
    manual: dict[str, Record]
    bundled: dict[str, dict[str, Record]]


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
    bundled_allow = {}
    for name, entry in raw["distributions"].items():
        for key in ("allow", "allow_bundled"):
            unknown = set(entry.get(key, [])) - set(sets)
            if unknown:
                raise ValueError(f"{path}: {name} {key} names unknown sets {sorted(unknown)}")
        distributions[name] = frozenset().union(*(sets[s] for s in entry["allow"]))
        bundled_allow[name] = frozenset().union(*(sets[s] for s in entry.get("allow_bundled", entry["allow"])))
    manual = {_norm(m["package"]): _record(m, f"manual {m['package']}") for m in raw.get("manual") or []}
    bundled = {
        _norm(package): {lib: _record(entry, f"bundled {package} {lib}") for lib, entry in libraries.items()}
        for package, libraries in (raw.get("bundled_libraries") or {}).items()
    }
    return Policy(
        raw["on_missing"], sets, distributions, bundled_allow, raw["aliases"], raw["classifiers"], manual, bundled
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


def _judge(
    subject: str, record: Record | None, needed_by: set[str], policy: Policy, version: str = "", bundled: bool = False
) -> Finding:
    users = tuple(sorted(needed_by))
    if record is None:
        return Finding(subject, version, "", "", users, "no license record")
    try:
        tree = parse_expression(record.expression)
    except ValueError as exc:
        return Finding(subject, version, record.expression, record.how, users, f"unparseable: {exc}")
    table = policy.bundled_allow if bundled else policy.distributions
    for dist in users:
        if not allows(tree, table[dist]):
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
            record = policy.bundled.get(_norm(name), {}).get(lib)
            out.append(_judge(f"{name} bundles {lib} ({file})", record, closure[name], policy, bundled=True))
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


class PypiDist:
    """The slice of `metadata.Distribution` the reader needs, built from a PyPI release's JSON."""

    def __init__(self, info: dict[str, Any]) -> None:
        """Map the release's ``license_expression``, ``license`` and ``classifiers`` to the core-metadata fields."""
        self.metadata = Message()
        self.metadata["Name"] = info["name"]
        for key, value in (("License-Expression", info.get("license_expression")), ("License", info.get("license"))):
            if value:
                self.metadata[key] = value
        for classifier in info.get("classifiers") or []:
            self.metadata["Classifier"] = classifier
        self.version = info["version"]
        self.files = None


def fetch_pypi(name: str, version: str) -> PypiDist | None:
    """The metadata of `name` at `version` from PyPI, or ``None`` when the release is not there."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:  # noqa: S310
            return PypiDist(json.load(response)["info"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def host_pins(lock: Path = LOCK) -> dict[str, str]:
    """The ``name -> version`` pins of the runtime closure of senselab in `uv.lock`, the repository's own packages apart."""
    packages = {_norm(p["name"]): p for p in tomllib.loads(lock.read_text())["package"]}
    closure = locked_closure(lock, DISTRIBUTIONS[:1])
    return {_norm(n): packages[_norm(n)].get("version", "") for n in closure if n not in DISTRIBUTIONS}


def pinned_dists(
    pins: dict[str, str],
    installed: dict[str, metadata.Distribution],
    fetch: Callable[[str, str], Any] = fetch_pypi,
) -> dict[str, Any]:
    """The metadata of each pin: the installed distribution when its version matches, else `fetch(name, version)`."""
    out = {}
    for name, version in sorted(pins.items()):
        dist = installed.get(name)
        dist = dist if dist is not None and dist.version == version else fetch(name, version)
        if dist is not None:
            out[name] = dist
    return out


def env_records(findings: list[Finding], date: str) -> dict[str, dict[str, Any]]:
    """The ``env-licenses`` entries of the package findings that hold a record, in name order."""
    out = {}
    for f in findings:
        if " bundles " in f.subject or not f.expression:
            continue
        how, _, when = f.how.partition(" ")
        pep639 = how == "pep639"
        out[_norm(f.subject)] = {
            "version": f.version,
            "expression": f.expression,
            "source": f"{f.subject} {f.version} {'License-Expression' if pep639 else how}",
            "verified": {"date": when or date, "how": "pep639" if pep639 else "manual"},
        }
    return out


def write_env_records(path: Path, env: str, records: dict[str, dict[str, Any]]) -> None:
    """Replace env `env`'s entries in the ``env-licenses`` file at `path`, keeping the other envs."""
    raw = yaml.safe_load(path.read_text()) if path.exists() else {"format": ENV_LICENSES_FORMAT, "envs": {}}
    raw["envs"][env] = records
    path.write_text(yaml.safe_dump(raw, sort_keys=True, width=120))


def main(argv: list[str]) -> int:
    """Check the locked runtime closure of the three distributions, or one env lock; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write the findings here")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--lock", metavar="NAME", help="check the pins of data/env_locks/NAME.txt")
    which.add_argument("--host", action="store_true", help="check the pins of the root lock's senselab closure")
    parser.add_argument("--site", type=Path, help="read metadata from this site-packages instead of PyPI")
    parser.add_argument("--write", action="store_true", help="merge the records into data/env_licenses.yaml")
    args = parser.parse_args(argv)
    policy = load_policy()
    if args.lock or args.host:
        pins = lock_pins((ENV_LOCKS / f"{args.lock}.txt").read_text()) if args.lock else host_pins()
        scan = (
            metadata.distributions(path=[str(args.site)])
            if args.site
            else (metadata.distributions() if args.host else [])
        )
        installed = {_norm(d.metadata["Name"]): d for d in scan if d.metadata["Name"]}
        dists = pinned_dists(pins, installed)
        findings = check({n: {INSTALLER} for n in pins}, policy, dists)
        if args.write and not failures(findings, policy):
            write_env_records(ENV_LICENSES, args.lock or "host", env_records(findings, date.today().isoformat()))
    else:
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
