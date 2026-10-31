"""Fail when a tracked object breaks the data-policy budget, which is bound to the 1.x freeze.

Every entry of the git index is an object of the budget, read from the index rather than the work tree, so a file
missing from the checkout or checked out differently is measured as it will be committed:

- a symlink (mode 120000), a submodule (mode 160000) or any other non-regular entry fails, without exception;
- a code file (`CODE_SUFFIXES`) over the code-file cap fails, and every other file over the data-file cap fails
  unless it is allowlisted or is a generated file named in the policy;
- code files, data files and the data files under the test trees are each capped in total across the whole tree;
  allowlisted files are not counted, generated files are;
- each cap is the policy's headroom times a quantity this script measures on the freeze branch, and the policy
  records that measurement, which must equal the measurement: the caps cannot move without the headroom moving;
- a generated-file exemption names one exact path that exists in the freeze, and the file must parse as the kind
  of lock the policy names;
- the allowlist lines are `<path>\t<sha256>\t<bytes>`. Each path must exist in the freeze branch with the same
  content hash, the recorded size may not exceed the frozen size, and the indexed file must still hash to the
  recorded value, so an entry can neither be added after the freeze nor be replaced by other bytes.

Usage::

    uv run python scripts/check_fixture_budget.py
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "fixture_budget_allowlist.txt"
POLICY = Path(__file__).resolve().parent / "data" / "fixture_budget.yaml"
FREEZE_BRANCH = "freeze-1x-2026-10-30"
TEST_TREES = ("src/tests/*", "packages/*/tests/*")
CODE_SUFFIXES = (".py", ".pyi", ".ipynb")
"""Code: capped per file and in total like data, under its own caps. Every other suffix is data."""
REGULAR_MODES = frozenset({"100644", "100755"})
SYMLINK_MODE = "120000"
GITLINK_MODE = "160000"
CAPS = (
    "code_file_max_bytes",
    "code_total_max_bytes",
    "data_file_max_bytes",
    "data_total_max_bytes",
    "test_data_total_max_bytes",
)
LOCK_KINDS = ("uv-lock", "venv-lock")
_FIX = "generate it with senselab_core.testing.synth, or pull it through a data manifest"
_SHA = re.compile(r"[0-9a-f]{64}")
_GLOB = re.compile(r"[*?\[\]]")


class Entry(NamedTuple):
    """One allowlist line: the content hash and byte size recorded for a grandfathered path."""

    sha256: str
    size: int


class TrackedObject(NamedTuple):
    """One entry of a git index or tree: its path, mode, object id and size in bytes (0 for a submodule)."""

    path: str
    mode: str
    oid: str
    size: int


class Cap(NamedTuple):
    """A cap as the policy states it: the quantity measured on the freeze and the headroom over it."""

    measured: int
    headroom: Fraction

    @property
    def limit(self) -> int:
        """The cap in bytes: the measurement times the headroom, rounded down."""
        return int(self.measured * self.headroom)


class Policy(NamedTuple):
    """The byte policy read from `data/fixture_budget.yaml`."""

    caps: dict[str, Cap]
    generated_files: dict[str, str]


def _cap(path: Path, name: str, cap: object) -> Cap:
    """Validate one cap entry of the policy.

    Raises:
        ValueError: if it is not `{measured: <int >= 0>, headroom: <number >= 1>}`.
    """
    if isinstance(cap, dict) and set(cap) == {"measured", "headroom"}:
        measured, headroom = cap["measured"], cap["headroom"]
        if type(measured) is int and measured >= 0 and type(headroom) in (int, float) and headroom >= 1:
            return Cap(measured, Fraction(str(headroom)))
    raise ValueError(f"{path}: {name} must be {{measured: <int >= 0>, headroom: <number >= 1>}}, got {cap!r}")


def load_policy(path: Path = POLICY) -> Policy:
    """Read the policy file.

    Raises:
        ValueError: if a cap is missing or malformed, a generated file is not an exact path mapped to a known lock
            kind, or a key is unknown.
    """
    data = yaml.safe_load(path.read_text())
    unknown = set(data) - {*CAPS, "generated_files", "derivation"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    caps = {name: _cap(path, name, data.get(name)) for name in CAPS}
    generated = data.get("generated_files", {})
    if not isinstance(generated, dict):
        raise ValueError(f"{path}: generated_files must map exact paths to a lock kind")
    for rel, kind in generated.items():
        if _GLOB.search(rel) or rel.startswith("/") or ".." in rel.split("/"):
            raise ValueError(f"{path}: generated file {rel!r} is not one exact repository path")
        if kind not in LOCK_KINDS:
            raise ValueError(f"{path}: generated file {rel!r} has kind {kind!r}, not one of {LOCK_KINDS}")
    return Policy(caps, dict(generated))


def in_test_tree(rel: str) -> bool:
    """Whether the repo-relative path is under a test tree."""
    return any(fnmatch.fnmatch(rel, p) for p in TEST_TREES)


def is_code(rel: str) -> bool:
    """Whether the repo-relative path is a code file; every other file is data."""
    return rel.endswith(CODE_SUFFIXES)


def parse_allowlist(text: str) -> dict[str, Entry]:
    """Parse `path<TAB>sha256<TAB>bytes` lines (blank lines and `#` comments skipped).

    Raises:
        ValueError: naming the line, if a line is not three tab-separated columns with a 64-hex hash and a size.
    """
    out: dict[str, Entry] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or not _SHA.fullmatch(parts[1]) or not parts[2].isdigit():
            raise ValueError(f"allowlist line {n}: expected '<path>\\t<sha256>\\t<bytes>', got {line!r}")
        out[parts[0]] = Entry(parts[1], int(parts[2]))
    return out


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True, input=stdin).stdout


def resolve_freeze_ref(root: Path = ROOT) -> str:
    """The ref of the freeze branch: its remote-tracking ref, and the local branch only when that is absent.

    Raises:
        LookupError: if neither exists, naming the fetch that creates it.
    """
    for ref in (f"origin/{FREEZE_BRANCH}", FREEZE_BRANCH):
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{ref}^{{commit}}"], cwd=root, capture_output=True
        )
        if probe.returncode == 0:
            return ref
    raise LookupError(
        f"the freeze branch {FREEZE_BRANCH} is not available; run: "
        f"git fetch origin {FREEZE_BRANCH}:refs/remotes/origin/{FREEZE_BRANCH}"
    )


def frozen_objects(root: Path, ref: str) -> list[TrackedObject]:
    """Every entry of the tree at `ref`, with its size (0 for a submodule)."""
    out = []
    for line in _git(root, "ls-tree", "-r", "-l", "-z", ref).decode().split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            mode, _kind, oid, size = meta.split()
            out.append(TrackedObject(path, mode, oid, int(size) if size.isdigit() else 0))
    return out


def indexed_objects(root: Path) -> list[TrackedObject]:
    """Every entry of the git index of `root`, with the size of its blob (0 for a submodule)."""
    rows = []
    for line in _git(root, "ls-files", "-s", "-z").decode().split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            mode, oid, _stage = meta.split()
            rows.append((path, mode, oid))
    blobs = [oid for _, mode, oid in rows if mode != GITLINK_MODE]
    sizes = _git(root, "cat-file", "--batch-check=%(objectsize)", stdin="".join(f"{o}\n" for o in blobs).encode())
    by_oid = dict(zip(blobs, (int(s) for s in sizes.split()), strict=True))
    return [TrackedObject(path, mode, oid, by_oid.get(oid, 0)) for path, mode, oid in rows]


def _blob(root: Path, oid: str) -> bytes:
    return _git(root, "cat-file", "blob", oid)


def measure(objects: Iterable[TrackedObject], allow: Mapping[str, object], generated: Iterable[str]) -> dict[str, int]:
    """The quantity each cap bounds, measured over the regular files among `objects`.

    The largest code file; the total of code; the largest data file neither allowlisted nor generated; the total of
    data that is not allowlisted; and that total under the test trees.
    """
    exempt = set(generated)
    regular = [o for o in objects if o.mode in REGULAR_MODES]
    code = [o for o in regular if is_code(o.path)]
    data = [o for o in regular if not is_code(o.path) and o.path not in allow]
    return {
        "code_file_max_bytes": max((o.size for o in code), default=0),
        "code_total_max_bytes": sum(o.size for o in code),
        "data_file_max_bytes": max((o.size for o in data if o.path not in exempt), default=0),
        "data_total_max_bytes": sum(o.size for o in data),
        "test_data_total_max_bytes": sum(o.size for o in data if in_test_tree(o.path)),
    }


def is_lock(data: bytes, kind: str) -> bool:
    """Whether `data` is a lock file of `kind`: a uv lock (TOML with a version and packages) or a venv lock."""
    if kind == "venv-lock":
        lines = data.decode(errors="replace").splitlines()
        return (
            len(lines) > 2
            and lines[0].startswith("# senselab subprocess-venv lock: ")
            and lines[1].startswith("# requirements-sha256: ")
        )
    try:
        parsed = tomllib.loads(data.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    return isinstance(parsed.get("version"), int) and isinstance(parsed.get("package"), list)


def check_policy(policy: Policy, root: Path, ref: str, allow: Mapping[str, object]) -> list[str]:
    """One message per cap whose recorded measurement is not the freeze's, and per exemption that is not a lock.

    A generated-file exemption must name a regular file of the freeze that, as indexed now, parses as its kind.
    """
    frozen = frozen_objects(root, ref)
    measured = measure(frozen, allow, policy.generated_files)
    out = [
        f"policy {name}: records {cap.measured} as measured at the freeze, {ref} measures {measured[name]}"
        for name, cap in policy.caps.items()
        if cap.measured != measured[name]
    ]
    frozen_paths = {o.path for o in frozen if o.mode in REGULAR_MODES}
    indexed = {o.path: o for o in indexed_objects(root)}
    for rel, kind in sorted(policy.generated_files.items()):
        if rel not in frozen_paths:
            out.append(f"policy generated file {rel}: not in {ref}; an exemption may only name a file of the freeze")
        elif rel in indexed and not is_lock(_blob(root, indexed[rel].oid), kind):
            out.append(f"policy generated file {rel}: is not a {kind}")
    return out


def check_allowlist(allow: Mapping[str, Entry], root: Path, ref: str) -> list[str]:
    """One message per allowlist entry that is not in the freeze with its hash, is larger, or no longer matches."""
    frozen = {o.path: o for o in frozen_objects(root, ref)}
    indexed = {o.path: o for o in indexed_objects(root)}
    out: list[str] = []
    for path, entry in sorted(allow.items()):
        if path not in frozen:
            out.append(f"{path}: not in {ref}; the allowlist may only grandfather files that existed at the freeze")
            continue
        if hashlib.sha256(_blob(root, frozen[path].oid)).hexdigest() != entry.sha256:
            out.append(f"{path}: recorded sha256 is not the content of the file in {ref}")
        elif entry.size > frozen[path].size:
            out.append(f"{path}: recorded {entry.size} bytes is larger than the {frozen[path].size} bytes in {ref}")
        current = indexed.get(path)
        if current is not None and current.mode in REGULAR_MODES:
            if current.size > entry.size:
                out.append(f"{path}: {current.size} bytes, grew past its grandfathered {entry.size} bytes; {_FIX}")
            elif hashlib.sha256(_blob(root, current.oid)).hexdigest() != entry.sha256:
                out.append(f"{path}: content differs from the grandfathered sha256; {_FIX}")
    return out


def check(
    objects: Sequence[TrackedObject],
    *,
    allow: Mapping[str, int],
    limits: Mapping[str, int],
    generated: Iterable[str] = (),
) -> list[str]:
    """Return one violation message per problem among `objects`, naming the path, its size and the limit.

    `allow` maps each grandfathered path to its recorded size; `limits` maps each name in `CAPS` to bytes;
    `generated` are the exact paths exempt from the data-file cap.
    """
    exempt = set(generated)
    out: list[str] = []
    for o in objects:
        if o.mode == SYMLINK_MODE:
            out.append(f"{o.path}: tracked symlink; commit the file itself")
        elif o.mode == GITLINK_MODE:
            out.append(f"{o.path}: tracked submodule; commit no other repository, pull its data through a manifest")
        elif o.mode not in REGULAR_MODES:
            out.append(f"{o.path}: tracked object of mode {o.mode}; only regular files may be committed")
        elif o.path in allow:
            if o.size > allow[o.path]:
                out.append(f"{o.path}: {o.size} bytes, grew past its grandfathered {allow[o.path]} bytes; {_FIX}")
        elif is_code(o.path):
            if o.size > limits["code_file_max_bytes"]:
                out.append(
                    f"{o.path}: {o.size} bytes exceeds the {limits['code_file_max_bytes']} byte code-file limit; "
                    "code is not a data carrier"
                )
        elif o.size > limits["data_file_max_bytes"] and o.path not in exempt:
            limit = limits["data_file_max_bytes"]
            out.append(f"{o.path}: {o.size} bytes exceeds the {limit} byte per-file limit; {_FIX}")
    totals = measure(objects, allow, exempt)
    for name, label in (
        ("code_total_max_bytes", "code files"),
        ("data_total_max_bytes", "non-allowlisted data files"),
        ("test_data_total_max_bytes", "non-allowlisted data files under the test trees"),
    ):
        if totals[name] > limits[name]:
            out.append(f"total of {label}: {totals[name]} bytes exceeds the {limits[name]} byte limit; {_FIX}")
    return out


def main(root: Path = ROOT, allowlist: Path = ALLOWLIST, policy: Path = POLICY) -> int:
    """Run the check over every indexed object; return the exit status."""
    try:
        allow = parse_allowlist(allowlist.read_text())
        loaded = load_policy(policy)
        ref = resolve_freeze_ref(root)
    except (ValueError, LookupError) as e:
        print(e)
        return 1
    violations = check_allowlist(allow, root, ref) + check_policy(loaded, root, ref, allow)
    violations += check(
        indexed_objects(root),
        allow={p: e.size for p, e in allow.items()},
        limits={name: cap.limit for name, cap in loaded.caps.items()},
        generated=loaded.generated_files,
    )
    for v in violations:
        print(v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
