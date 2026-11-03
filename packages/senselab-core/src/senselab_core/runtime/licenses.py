"""SPDX license expressions, the license records derived from them, and a user's license policy (P-23).

`parse_expression`, `ids_of` and `allows` read an expression; `derive_flags` reads its flags from the packaged
table `data/spdx_licenses.yaml`; `license_record` builds a `License` whose flags are those; `check_license` refuses
a record whose flags contradict its expression; `backend_license` derives a backend's code license from its env
lock's package records, which `lock_records_from` reads from an `env-licenses` document. `LicensePolicy` is a
run's own allowed and refused license ids.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any, Literal, NamedTuple

from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.license import License, require_license
from senselab_core.runtime.config import load_yaml

SPDX_TABLE_FORMAT = "spdx-licenses@1.0"
_TOKEN = re.compile(r"\(|\)|[^\s()]+")
_COMMERCIAL = ("no", "unknown", "yes")
_COPYLEFT = ("none", "weak", "strong", "network")


class Flags(NamedTuple):
    """The flags of a license record that follow from its expression."""

    commercial_use: Literal["yes", "no", "unknown"]
    copyleft: Literal["none", "weak", "strong", "network"]
    attribution: bool


class _Parser:
    def __init__(self, text: str) -> None:
        self.text, self.tokens, self.pos = text, _TOKEN.findall(text), 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        if self.pos >= len(self.tokens):
            raise ValueError(f"expression ends early: {self.text!r}")
        self.pos += 1
        return str(self.tokens[self.pos - 1])

    def atom(self) -> Any:  # noqa: ANN401
        if self.peek() == "(":
            self.take()
            inner = self.either()
            if self.peek() != ")":
                raise ValueError(f"unbalanced parentheses in {self.text!r}")
            self.take()
            return inner
        if self.peek() is None or self.peek() in (")", "AND", "OR", "WITH"):
            raise ValueError(f"expected a license id in {self.text!r}")
        name = self.take()
        if self.peek() == "WITH":
            self.take()
            name = f"{name} WITH {self.take()}"
        return name

    def both(self) -> Any:  # noqa: ANN401
        node = self.atom()
        while self.peek() == "AND":
            self.take()
            node = ("and", node, self.atom())
        return node

    def either(self) -> Any:  # noqa: ANN401
        node = self.both()
        while self.peek() == "OR":
            self.take()
            node = ("or", node, self.both())
        return node


def parse_expression(text: str) -> Any:  # noqa: ANN401
    """Parse an SPDX expression into nested tuples: `("or", a, b)`, `("and", a, b)` or an id string.

    Raises:
        ValueError: if the expression is empty, unbalanced, ends early or has trailing text.
    """
    parser = _Parser(text)
    tree = parser.either()
    if parser.peek() is not None:
        raise ValueError(f"trailing text in {text!r}")
    return tree


def ids_of(tree: Any) -> list[str]:  # noqa: ANN401
    """Every license id in a parsed expression."""
    return [tree] if isinstance(tree, str) else [i for part in tree[1:] for i in ids_of(part)]


def allows(tree: Any, allowed: frozenset[str]) -> bool:  # noqa: ANN401
    """Whether the expression is satisfied by the `allowed` lower-case ids: `AND` needs both sides, `OR` either."""
    if isinstance(tree, str):
        return tree.lower() in allowed
    left, right = allows(tree[1], allowed), allows(tree[2], allowed)
    return (left and right) if tree[0] == "and" else (left or right)


@cache
def spdx_table() -> Mapping[str, Flags]:
    """Return the packaged table, `lower-case id -> Flags`.

    Raises:
        ValueError: if the file lacks its format, a group lacks a derivation, or a group's flags are not valid.
    """
    raw = load_yaml(
        resources.files("senselab_core.runtime").joinpath("data", "spdx_licenses.yaml"), "spdx_licenses.yaml"
    )
    if raw.get("format") != SPDX_TABLE_FORMAT:
        raise ValueError(f"spdx_licenses.yaml must carry format {SPDX_TABLE_FORMAT!r}")
    table: dict[str, Flags] = {}
    for group in raw["groups"]:
        if not str(group.get("derivation", "")).strip():
            raise ValueError(f"a license group needs a derivation: {group.get('ids')}")
        commercial = {True: "yes", False: "no"}.get(group["commercial_use"], group["commercial_use"])
        if commercial not in _COMMERCIAL or group["copyleft"] not in _COPYLEFT:
            raise ValueError(f"a license group has an unknown flag value: {group['ids']}")
        flags = Flags(commercial, group["copyleft"], bool(group["attribution"]))
        for license_id in group["ids"]:
            if table.setdefault(license_id.lower(), flags) is not flags:
                raise ValueError(f"license id {license_id!r} is in two groups")
    return table


def _flags_of(tree: Any, table: Mapping[str, Flags], expression: str) -> Flags:  # noqa: ANN401
    if isinstance(tree, str):
        if tree.lower() not in table:
            raise ValueError(f"license id {tree!r} in {expression!r} is not in the license table")
        return table[tree.lower()]
    a, b = _flags_of(tree[1], table, expression), _flags_of(tree[2], table, expression)
    if tree[0] == "and":
        return Flags(
            min(a.commercial_use, b.commercial_use, key=_COMMERCIAL.index),
            max(a.copyleft, b.copyleft, key=_COPYLEFT.index),
            a.attribution or b.attribution,
        )
    return Flags(
        max(a.commercial_use, b.commercial_use, key=_COMMERCIAL.index),
        min(a.copyleft, b.copyleft, key=_COPYLEFT.index),
        a.attribution and b.attribution,
    )


def derive_flags(expression: str) -> Flags:
    """Return the flags of `expression`: `AND` takes the stricter side of each flag, `OR` the more permissive.

    Raises:
        ValueError: if the expression is malformed or names an id the table does not hold.
    """
    return _flags_of(parse_expression(expression), spdx_table(), expression)


def check_license(record: License) -> License:
    """Return `record`, or raise `ValueError` when its flags differ from those its expression derives."""
    derived = derive_flags(record.expression)
    given = Flags(record.commercial_use, record.copyleft, record.attribution)
    if given != derived:
        raise ValueError(f"license flags {tuple(given)} contradict {record.expression!r}, which gives {tuple(derived)}")
    return record


def license_record(
    expression: str, *, source: str, verified: Mapping[str, str], gated: bool, flags: Mapping[str, Any] | None = None
) -> License:
    """Return the `License` of `expression` with its flags derived; `flags`, when given, must equal them.

    Raises:
        ValueError: if the expression is malformed, names an unknown id, or `flags` contradict it.
    """
    derived = derive_flags(expression)
    record = License.model_validate(
        {
            **derived._asdict(),
            **(flags or {}),
            "expression": expression,
            "source": source,
            "verified": verified,
            "gated": gated,
        }
    )
    return check_license(record)


def backend_license(backend: BackendSpec, lock_records: Mapping[str, License | None]) -> License:
    """Return the code license of `backend`: every package of its env lock, `AND`-joined, flags derived.

    `lock_records` maps each package name of the lock (the root lock for `Env.host`) to its record. The
    record is `manual` verified unless every package record was read as `pep639`, and dated by the latest.

    Raises:
        MissingLicense: if a package has no record.
        ValueError: if `lock_records` is empty or a package record's flags contradict its expression.
    """
    if not lock_records:
        raise ValueError(f"{backend.task}/{backend.name}: no lock records to derive a license from")
    subject = f"{backend.task}/{backend.name}"
    records = {
        name: check_license(require_license(r, f"package {name} of {subject}")) for name, r in lock_records.items()
    }
    expressions = sorted({r.expression for r in records.values()})
    how = "pep639" if {r.verified["how"] for r in records.values()} == {"pep639"} else "manual"
    return license_record(
        " AND ".join(f"({e})" if len(expressions) > 1 else e for e in expressions),
        source=f"env lock of {subject}: {len(records)} packages",
        verified={"date": max(r.verified["date"] for r in records.values()), "how": how},
        gated=any(r.gated for r in records.values()),
    )


LockRecords = Callable[[BackendSpec], Mapping[str, License | None]]
"""A backend's env-lock package records, `package name -> license record` (`None` for a package without one)."""
ENV_LICENSES_FORMAT = "env-licenses@1.0"
HOST_ENV = "host"


def env_name(backend: BackendSpec) -> str:
    """Return the env whose records cover `backend`: its lock name, or ``host``."""
    return HOST_ENV if backend.env.kind == "host" else str(backend.env.name)


def lock_records_from(text: str) -> LockRecords:
    """Return the provider of ``package -> License`` over the ``env-licenses`` document `text`.

    The document is written by ``scripts/check_licenses.py --lock NAME`` (or ``--host``) from package metadata; its
    envs are the lock names, and ``host`` for the root lock.

    Raises:
        ValueError: if the format is not ``env-licenses@1.0``, or (from the provider) a backend's env has no records.
    """
    raw = load_yaml(text, "env licenses")
    if raw.get("format") != ENV_LICENSES_FORMAT:
        raise ValueError(f"env licenses must carry format {ENV_LICENSES_FORMAT!r}")
    envs = raw["envs"]

    def records(backend: BackendSpec) -> dict[str, License | None]:
        name = env_name(backend)
        if name not in envs:
            raise ValueError(f"no license records for env {name!r}; run scripts/check_licenses.py --lock {name}")
        return {
            package: license_record(
                entry["expression"], source=entry["source"], verified=entry["verified"], gated=False
            )
            for package, entry in envs[name].items()
        }

    return records


def packaged_lock_records(package: str) -> LockRecords:
    """Return the provider over ``data/env_licenses.yaml`` of `package`, read on its first use."""
    read: list[LockRecords] = []

    def records(backend: BackendSpec) -> dict[str, License | None]:
        if not read:
            read.append(
                lock_records_from(resources.files(package).joinpath("data", "env_licenses.yaml").read_text("utf-8"))
            )
        return dict(read[0](backend))

    return records


class LicenseRefused(ValueError):
    """A run's license policy refuses the license of a backend's env or of a model."""


@dataclass(frozen=True)
class LicensePolicy:
    """A run's license policy: the SPDX ids it allows and those it refuses, matched ignoring case.

    `allow` None allows every id `refuse` does not name. A license is accepted when its expression is satisfied by
    the allowed ids that are not refused (`allows`); a subject with no license record is refused.
    """

    allow: frozenset[str] | None = None
    refuse: frozenset[str] = frozenset()

    def refusal(self, subject: str, record: License | None) -> str | None:
        """Return why the policy refuses `record`, the license of `subject`, or None when it accepts it."""
        if record is None:
            return f"{subject} has no license record"
        tree = parse_expression(record.expression)
        allowed = {i.lower() for i in (ids_of(tree) if self.allow is None else self.allow)}
        if allows(tree, frozenset(allowed - {i.lower() for i in self.refuse})):
            return None
        return f"{subject} is licensed {record.expression!r}, which the run's license policy does not allow"

    def check(self, subject: str, record: License | None) -> None:
        """Raise `LicenseRefused` when the policy refuses `record`, the license of `subject`."""
        if (reason := self.refusal(subject, record)) is not None:
            raise LicenseRefused(reason)
