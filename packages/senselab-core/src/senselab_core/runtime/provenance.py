"""An append-only provenance store, modelled on W3C PROV.

Entities are what the graph believes exists, activities are node executions, agents are what acted.
Relations are PROV's own. A record holds the canonical form it was identified by and cannot change
after it is built, so merging two stores is a set union and is order-independent.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ClassVar, Literal, NoReturn, Sequence, TypeVar, cast, get_args

import numpy
from senselab_worker import digest as worker_digest
from senselab_worker.canonical import (
    MAX_SAFE_INT,
    NoCanonicalForm,
    Recurse,
    Tokeniser,
    Walk,
    canonical_npy_bytes,
    canonical_scalar,
    canonical_text,
    in_json_domain,
    jcs_dumps,
    jcs_loads,
)
from senselab_worker.digest import identity_digest

from senselab_core.runtime.storage import write_atomically

PROV_TYPE = Literal[
    "span",
    "word",
    "speaker",
    "interval",
    "measurement",
    "kind",
    "stream",
    "pii",
    "verdict",
    "branch_report",
    "assertion",
    "target_match",
    "branch_decision",
    "enrollment",
    "artifact",
]
"""Every entity type the store understands.

``artifact`` is a file set identified by its content alone: it may carry a checksum without a
``path``, for callers whose file names must not be recorded. See
``specs/20261007-reproduce-driver/design.md``.

``branch_report`` is what a node that reports rather than decides writes: the four triage branches
and QUALITY. It carries task conformance and deviations and no outcome, which is why it is not a
``verdict``: a reader folding verdicts must not pick up a decision no branch made.

``kind`` is **historical**: TAXONOMY's evidence fold wrote it until 2026-09-13 and nothing writes it
now. It stays here because this vocabulary is the *reader's*, and the store is append-only: a reader
that refuses a record it previously wrote makes finished runs unopenable. The writer's vocabulary may
shrink; this one may not. See ``specs/20260912-ruleset-in-pipeline/design.md``.
"""
AGENT_TYPE = Literal["model", "software"]
ENVIRONMENT_KIND = Literal["host", "venv"]
RELATION = Literal[
    "wasGeneratedBy", "used", "wasAssociatedWith", "wasAttributedTo", "wasDerivedFrom", "wasInvalidatedBy"
]

_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
CHECKSUM_ALGORITHM = "sha256"
CHECKSUM_KEY = "checksum_sha256"
CHECKSUM_UNRESOLVED_KEY = "checksum_unresolved_reason"
SIZE_KEY = "size_bytes"
MTIME_KEY = "mtime_ns"
PATH_KEY = "path"
_PROV_TYPES = frozenset(get_args(PROV_TYPE))
_AGENT_TYPES = frozenset(get_args(AGENT_TYPE))
_ENVIRONMENT_KINDS = frozenset(get_args(ENVIRONMENT_KIND))
_RELATIONS = frozenset(get_args(RELATION))


def file_digest(path: str | Path) -> tuple[str | None, str | None]:
    """Read a file and return `worker_digest.file_digest("file@1", file)`, or the reason it has none.

    Args:
        path: The file to digest.

    Returns:
        ``(digest, reason)`` — a 64-character lowercase hex digest and None, or None and a reason
        the digest could not be taken.
    """
    target = Path(path)
    try:
        with target.open("rb") as handle:
            digest = worker_digest.file_digest("file@1", handle)
    except FileNotFoundError:
        return None, "file not found"
    except IsADirectoryError:
        return None, "path is a directory"
    except PermissionError:
        return None, "permission denied"
    except OSError as err:
        return None, f"read failed: {type(err).__name__}"
    return digest, None


def file_attributes(path: str | Path, *, digest: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Entity attributes describing a file's bytes as they are now.

    Args:
        path: The file, used for ``size_bytes`` and ``mtime_ns`` and, unless a digest or a reason is
            supplied, read to compute the digest.
        digest: An already-computed SHA-256 to record instead of reading the file again.
        reason: Why no digest is available, recorded instead of reading the file again.

    Returns:
        ``checksum_sha256`` or ``checksum_unresolved_reason``, and ``size_bytes`` and ``mtime_ns``
        when the file can be stat'd.

    Raises:
        ValueError: If both ``digest`` and ``reason`` are supplied.
    """
    if digest is not None and reason is not None:
        raise ValueError("digest and reason contradict each other; supply at most one")
    if digest is None and reason is None:
        digest, reason = file_digest(path)
    out: dict[str, Any] = {CHECKSUM_KEY: digest} if digest is not None else {CHECKSUM_UNRESOLVED_KEY: reason}
    try:
        stat = Path(path).stat()
    except OSError:
        return out
    out[SIZE_KEY] = int(stat.st_size)
    out[MTIME_KEY] = int(stat.st_mtime_ns)
    return out


def _check_entity_attributes(prov_type: str, attributes: Mapping[str, Any]) -> None:
    """Refuse checksum attributes the store never accepts, at write time and at read-back alike.

    Args:
        prov_type: The entity's type; only an ``artifact`` may carry a checksum without a ``path``.
        attributes: The entity's attributes.

    Raises:
        ValueError: If a digest and a reason it is missing are supplied together, if the digest is
            not 64 lowercase hex characters, if the reason is empty, or if either is present without
            a ``path`` saying which file it describes on an entity other than an ``artifact``.
    """
    digest, reason = attributes.get(CHECKSUM_KEY), attributes.get(CHECKSUM_UNRESOLVED_KEY)
    if digest is not None and reason is not None:
        raise ValueError(f"{CHECKSUM_KEY} and {CHECKSUM_UNRESOLVED_KEY} contradict each other; supply exactly one")
    if digest is not None and not (isinstance(digest, str) and _DIGEST.fullmatch(digest)):
        raise ValueError(f"{CHECKSUM_KEY} must be 64 lowercase hex characters, got {digest!r}")
    if reason is not None and not (isinstance(reason, str) and reason.strip()):
        raise ValueError(f"{CHECKSUM_UNRESOLVED_KEY} must not be empty; an empty reason says nothing")
    if (digest is not None or reason is not None) and prov_type != "artifact" and not attributes.get(PATH_KEY):
        raise ValueError(f"a checksum needs a {PATH_KEY!r} naming the file it describes")


def _check_agent_fields(
    agent_type: str, model_id: str | None, commit_sha: str | None, unresolved_reason: str | None
) -> None:
    """Refuse agent field combinations the store never accepts, at write time and at read-back alike.

    Raises:
        ValueError: If ``commit_sha`` is not exactly 40 hex characters, if ``commit_sha`` and
            ``unresolved_reason`` are supplied together, if ``unresolved_reason`` is empty, or if a
            model agent is missing ``model_id`` or supplies neither a commit nor a reason it is
            missing.
    """
    if commit_sha is not None and not _SHA.fullmatch(commit_sha):
        raise ValueError(
            f"commit_sha must be a resolved 40-hex commit, got {commit_sha!r}. A ref recorded as a "
            "commit makes the provenance confidently wrong."
        )
    if commit_sha is not None and unresolved_reason is not None:
        raise ValueError("commit_sha and unresolved_reason contradict each other; supply exactly one")
    if unresolved_reason is not None and not unresolved_reason.strip():
        raise ValueError("unresolved_reason must not be empty; an empty reason says nothing")
    if agent_type == "model" and model_id is None:
        raise ValueError("a model agent needs model_id")
    if agent_type == "model" and commit_sha is None and unresolved_reason is None:
        raise ValueError("a model agent needs commit_sha or unresolved_reason; silence is not a third option")


PROVENANCE_FORMAT = "provenance@2.0"
"""The format `write_jsonl` writes: its first line is `{"format":"provenance@2.0","record":"format"}`.

A file whose first line is not a format record is `provenance@1`, the previous major, which
`read_jsonl` still reads.
"""

_ID_TAG = "provenance-id@4"
_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_INT_TOKEN_TEXT = re.compile(r"^-?[1-9][0-9]*$")
_PROV_TOKENS = frozenset({"$f", "$int", "$path", "$enum", "$nd", "$dec", "$bytes", "$set"})
_FLOAT_NAMES = frozenset({"NaN", "Infinity", "-Infinity"})


class UnrecordableValue(NoCanonicalForm):
    """A provenance attribute or parameter holds a value outside the declared provenance value type.

    Attributes:
        path: Where the value sits, from the attribute or parameter root, e.g. `attributes.evidence[2]`.
        type_name: The offending value's type, `module.qualname`.
    """

    refusal = "is not a provenance value"


def _nd_token(a: numpy.ndarray, recurse: Recurse, path: str) -> object:
    return {"$nd": base64.b64encode(canonical_npy_bytes(a)).decode("ascii")}


_TOKENISER = Tokeniser(array=_nd_token, error=UnrecordableValue)


def normalised(value: object, path: str = "value") -> object:
    """Return `value` as JSON the shared encoder accepts, every value outside the JSON domain as a tagged token.

    The provenance value type is what the shared `canonical.Tokeniser` writes, with a `numpy.ndarray`
    written as `{"$nd": <base64 of its canonical .npy bytes>}`.

    Raises:
        UnrecordableValue: naming `path` and the type, for a value of no family, an array of a dtype
            canonical `.npy` refuses, a non-`str` key, two keys equal after NFC, a container that contains
            itself, an integer too long to write, or nesting deeper than `MAX_DEPTH`.
    """
    return _TOKENISER(value, path)


class FrozenDict(dict[str, Any]):
    """A provenance object as a record holds it: a `dict` whose every mutating method raises `TypeError`."""

    __slots__ = ()

    def _refuse(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError("a provenance record is read-only")

    __setitem__ = __delitem__ = __ior__ = _refuse  # type: ignore[assignment]
    clear = pop = popitem = setdefault = update = _refuse  # type: ignore[assignment]

    def __reduce__(self) -> tuple[type[FrozenDict], tuple[dict[str, Any]]]:
        """Pickle as a new read-only mapping of the same items."""
        return type(self), (dict(self),)

    def __copy__(self) -> FrozenDict:
        """Return `self`, which cannot change."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        """Return `self`, which cannot change, nor can anything it holds."""
        return self


def _thawed(value: object, walk: Walk | None = None) -> Any:  # noqa: ANN401
    """Return a canonical form as plain `dict` and `list`, each shared container thawed once."""
    walk = Walk() if walk is None else walk
    t = type(value)
    if t is dict or t is FrozenDict:
        items = cast("dict[str, object]", value)
        return walk.enter(value, lambda: {k: _thawed(v, walk) for k, v in items.items()})
    if t is list or t is tuple:
        elements = cast("Sequence[object]", value)
        return walk.enter(value, lambda: [_thawed(v, walk) for v in elements])
    return value


def _refuse(path: str, value: object, reason: str) -> NoReturn:
    raise UnrecordableValue(path, value, reason)


def _check_token(key: str, payload: object, path: str, check: Callable[[object, str], None]) -> None:
    where = f"{path}.{key}"
    if key == "$f":
        if payload not in _FLOAT_NAMES:
            _refuse(where, payload, f"is not one of {sorted(_FLOAT_NAMES)}")
    elif key == "$int":
        if not (isinstance(payload, str) and _INT_TOKEN_TEXT.fullmatch(payload)):
            _refuse(where, payload, "is not a decimal integer")
        try:
            number = int(payload)
        except ValueError as err:
            _refuse(where, payload, str(err))
        if abs(number) <= MAX_SAFE_INT:
            _refuse(where, payload, "is an integer JSON writes as a number")
    elif key in ("$path", "$dec", "$bytes", "$nd"):
        if not isinstance(payload, str):
            _refuse(where, payload, "is not text")
        if key == "$dec":
            try:
                same = str(Decimal(payload)) == payload
            except InvalidOperation:
                same = False
            if not same:
                _refuse(where, payload, "is not a decimal in its own text form")
        elif key == "$bytes" and not re.fullmatch(r"(?:[0-9a-f]{2})*", payload):
            _refuse(where, payload, "is not lowercase hex of whole bytes")
        elif key == "$nd":
            _check_npy(payload, where)
    elif key == "$enum":
        if not (isinstance(payload, (list, tuple)) and len(payload) == 2 and isinstance(payload[0], str)):
            _refuse(where, payload, "is not [<type name>, <value>]")
        check(payload[1], f"{where}.value")
    else:
        if not isinstance(payload, (list, tuple)):
            _refuse(where, payload, "is not a list of elements")
        forms = []
        for i, item in enumerate(payload):
            check(item, f"{where}[{i}]")
            forms.append(jcs_dumps(_thawed(item)))
        if any(a >= b for a, b in zip(forms, forms[1:])):
            _refuse(where, payload, "is not its elements, one per JCS form, sorted by their JCS bytes")


def _check_npy(text: str, where: str) -> None:
    try:
        data = base64.b64decode(text, validate=True)
    except binascii.Error as err:
        _refuse(where, text, f"is not base64: {err}")
    if base64.b64encode(data).decode("ascii") != text:
        _refuse(where, text, "is not base64 in its canonical form")
    try:
        array = numpy.load(io.BytesIO(data), allow_pickle=False)
        canonical = canonical_npy_bytes(array)
    except (ValueError, TypeError, OSError) as err:
        _refuse(where, text, f"is not a .npy array: {err}")
    if canonical != data:
        _refuse(where, text, "is not an array's canonical .npy bytes")


def check_normal(value: object, path: str = "value") -> None:
    """Check that a JSON value is a provenance normal form: what `normalised` can write, after JCS.

    Every object whose one key starts with a single `$` is a token of a provenance family with a
    well-formed payload (`$set` elements sorted by their JCS bytes, one per form; `$nd` the base64 of
    canonical `.npy` bytes); every other key that starts with `$` starts with `$$`. A container shared
    by several parents is checked once per depth.

    Raises:
        UnrecordableValue: naming the path, for anything else, a cycle, or nesting deeper than `MAX_DEPTH`.
    """
    walk = Walk()

    def check(v: object, p: str) -> None:
        try:
            if isinstance(v, dict):
                walk.enter(v, lambda: check_object(cast("dict[str, object]", v), p))
            elif isinstance(v, (list, tuple)):
                walk.enter(v, lambda: check_items(cast("Sequence[object]", v), p))
        except UnrecordableValue:
            raise
        except ValueError as err:
            _refuse(p, v, str(err))

    def check_items(v: Sequence[object], p: str) -> None:
        for i, item in enumerate(v):
            check(item, f"{p}[{i}]")

    def check_object(v: dict[str, object], p: str) -> None:
        if len(v) == 1:
            (key,) = v
            if key.startswith("$") and not key.startswith("$$"):
                if key not in _PROV_TOKENS:
                    _refuse(f"{p}.{key}", v, "is not a provenance token")
                _check_token(key, v[key], p, check)
                return
        for key, item in v.items():
            if key.startswith("$") and not key.startswith("$$"):
                _refuse(f"{p}.{key}", v, "is a token key inside an object of several keys")
            check(item, f"{p}.{key}")

    check(value, path)


def _canonical(value: object, path: str) -> Any:  # noqa: ANN401
    """Return the read-only canonical form of a normalised value: the value JCS writes and reads back.

    Objects become `FrozenDict` and arrays tuples, text is in canonical form, and an integral float
    JSON writes as an integer is that integer; a container shared by several parents is built once.

    Raises:
        UnrecordableValue: if the value is not JSON (a non-`str` key, two keys with one canonical
            form, a number outside the JSON domain, any other type, a cycle, nesting deeper than
            `MAX_DEPTH`) or not a normal form (`check_normal`).
    """
    walk = Walk()

    def canon(v: object, p: str) -> object:
        t = type(v)
        if t is dict or t is FrozenDict:
            return walk.enter(v, lambda: canon_object(cast("dict[object, object]", v), p))
        if t is list or t is tuple:
            elements = cast("Sequence[object]", v)
            return walk.enter(v, lambda: tuple(canon(x, f"{p}[{i}]") for i, x in enumerate(elements)))
        try:
            scalar = canonical_scalar(v)
        except TypeError:
            _refuse(p, v, "is not JSON")
        if not in_json_domain(scalar):
            _refuse(p, v, "is a number outside the JSON domain")
        if type(scalar) is str:
            return canonical_text(scalar)
        if type(scalar) is float and scalar.is_integer() and abs(scalar) <= MAX_SAFE_INT:
            return int(scalar)
        return scalar

    def canon_object(v: dict[object, object], p: str) -> FrozenDict:
        out: dict[str, object] = {}
        for key, item in v.items():
            if type(key) is not str:
                _refuse(f"{p}[{key!r}]", key, "is not a str object key")
            text = canonical_text(key)
            if text in out:
                _refuse(f"{p}.{key}", key, "is a key equal after NFC to another key of its object")
            out[text] = canon(item, f"{p}.{key}")
        return FrozenDict(out)

    try:
        result = canon(value, path)
    except UnrecordableValue:
        raise
    except ValueError as err:
        raise UnrecordableValue(path, value, str(err)) from err
    check_normal(result, path)
    return result


def _text(value: object, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text, got {type(value).__name__}")


def _extent(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(f"extent must be (start, end) or None, got {value!r}")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"extent must hold two finite numbers of seconds, got {value!r}")
    start, end = float(value[0]), float(value[1])
    if start > end:
        raise ValueError(f"extent ends before it starts: {value!r}")
    return start, end


class _Record:
    """What every provenance record shares: it is canonical, read-only, and bound to its id.

    `_DIGESTED` names the fields the id covers, after `run_id`; `_CANONICAL` the fields held in
    canonical form. A record whose `run_id` is `None` was read from a `provenance@1` file and keeps the
    id that file gave it; any other record's id is checked against its content when it is built.
    """

    _DIGESTED: ClassVar[tuple[str, ...]]
    _CANONICAL: ClassVar[tuple[str, ...]]
    _KIND: ClassVar[str]
    id: str
    run_id: str | None

    def _prefix(self) -> str:
        raise NotImplementedError

    def _check(self) -> None:
        raise NotImplementedError

    def __post_init__(self) -> None:
        """Check the fields, hold them in canonical form, and check the id against them."""
        _text(self.id, "id")
        _text(self.run_id, "run_id", optional=True)
        self._check()
        for name in self._CANONICAL:
            object.__setattr__(self, name, _canonical(getattr(self, name), name))
        if self.run_id is not None and self.id != self.minted_id():
            raise ValueError(f"{self._KIND} id {self.id!r} is not the id of its content, {self.minted_id()!r}")

    def minted_id(self, run_id: str | None = None) -> str:
        """Return the id of this content in `run_id` (by default the record's own): `<prefix>-` and 16 hex.

        The hex is `identity_digest("provenance-id@4", [run_id, *<the digested fields, canonical>])`.

        Raises:
            ValueError: if neither `run_id` nor the record's has a value.
            UnrecordableValue: if the content, inside the id's envelope, nests deeper than `MAX_DEPTH` or
                exceeds `MAX_CANONICAL_CHARS`.
        """
        run = self.run_id if run_id is None else run_id
        if run is None:
            raise ValueError(f"{self._KIND} {self.id!r} was read from provenance@1 and has no run_id to mint from")
        parts = [run, *(_thawed(getattr(self, name)) for name in self._DIGESTED)]
        try:
            digest = identity_digest(_ID_TAG, parts)
        except ValueError as err:
            raise UnrecordableValue(".".join(("run_id", *self._DIGESTED)), parts, str(err)) from err
        return f"{self._prefix()}-{digest[:16]}"

    def as_record(self) -> dict[str, Any]:
        """Return the JSON object a file holds for this record."""
        return {"record": self._KIND, **{f.name: _thawed(getattr(self, f.name)) for f in fields(cast(Any, self))}}


@dataclass(frozen=True)
class Entity(_Record):
    """Something the graph believes exists.

    Attributes:
        id: `<prov_type>-<16 hex>`, the digest of `run_id`, `prov_type`, `extent` and `attributes`.
        prov_type: One of `PROV_TYPE`.
        extent: `(start, end)` in seconds, finite, `start <= end`, or None.
        attributes: The normalised, read-only attributes.
        run_id: The run the id was minted in, or None for a record read from `provenance@1`.
    """

    _DIGESTED: ClassVar[tuple[str, ...]] = ("prov_type", "extent", "attributes")
    _CANONICAL: ClassVar[tuple[str, ...]] = ("attributes",)
    _KIND: ClassVar[str] = "entity"

    id: str
    prov_type: PROV_TYPE
    extent: tuple[float, float] | None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def _prefix(self) -> str:
        return self.prov_type

    def _check(self) -> None:
        if self.prov_type not in _PROV_TYPES:
            raise ValueError(f"unknown prov_type {self.prov_type!r}; known: {sorted(_PROV_TYPES)}")
        object.__setattr__(self, "extent", _extent(self.extent))
        if not isinstance(self.attributes, Mapping):
            raise UnrecordableValue("attributes", self.attributes, "is not a mapping")
        _check_entity_attributes(self.prov_type, self.attributes)


@dataclass(frozen=True)
class Activity(_Record):
    """One node execution, or one step of one.

    Attributes:
        id: The activity's id. ``started`` and ``ended`` are not part of it.
        node: The node executing.
        step: Which step of it, when a node has several.
        started: When it began, ISO 8601, or None when unrecorded.
        ended: When it finished, ISO 8601, or None when unrecorded.
        parameters: The normalised, read-only values it ran with.
        run_id: The run the id was minted in, or None for a record read from `provenance@1`.
    """

    _DIGESTED: ClassVar[tuple[str, ...]] = ("node", "step", "parameters")
    _CANONICAL: ClassVar[tuple[str, ...]] = ("node", "step", "started", "ended", "parameters")
    _KIND: ClassVar[str] = "activity"

    id: str
    node: str
    step: str | None
    started: str | None = None
    ended: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def _prefix(self) -> str:
        return "act"

    def _check(self) -> None:
        _text(self.node, "node")
        for name in ("step", "started", "ended"):
            _text(getattr(self, name), name, optional=True)
        if not isinstance(self.parameters, Mapping):
            raise UnrecordableValue("parameters", self.parameters, "is not a mapping")


@dataclass(frozen=True)
class Agent(_Record):
    """What acted: a model at a resolved commit, or the software itself.

    Attributes:
        id: The agent's id.
        agent_type: ``"model"`` or ``"software"``.
        model_id: The model's identifier, for a model agent.
        commit_sha: A resolved 40-hex commit, when resolution succeeded.
        unresolved_reason: Why the commit is unknown, when it is. A provenance model that cannot say
            "I could not resolve this" forces either a lie or a crash.
        version: Software version, for a software agent.
        run_id: The run the id was minted in, or None for a record read from `provenance@1`.
    """

    _DIGESTED: ClassVar[tuple[str, ...]] = ("agent_type", "model_id", "commit_sha", "unresolved_reason", "version")
    _CANONICAL: ClassVar[tuple[str, ...]] = _DIGESTED
    _KIND: ClassVar[str] = "agent"

    id: str
    agent_type: AGENT_TYPE
    model_id: str | None = None
    commit_sha: str | None = None
    unresolved_reason: str | None = None
    version: str | None = None
    run_id: str | None = None

    def _prefix(self) -> str:
        return "agent"

    def _check(self) -> None:
        if self.agent_type not in _AGENT_TYPES:
            raise ValueError(f"unknown agent_type {self.agent_type!r}; known: {sorted(_AGENT_TYPES)}")
        for name in ("model_id", "commit_sha", "unresolved_reason", "version"):
            _text(getattr(self, name), name, optional=True)
        _check_agent_fields(self.agent_type, self.model_id, self.commit_sha, self.unresolved_reason)


@dataclass(frozen=True)
class Environment(_Record):
    """The runtime a run actually executed in: the host interpreter, or one subprocess venv.

    Attributes:
        id: The environment's id.
        kind: ``"host"`` for the interpreter running the pipeline, ``"venv"`` for one subprocess venv.
        label: The host's own name, or the venv's resolved directory name — which encodes its
            device key (e.g. ``"crisperwhisper-cpu"``).
        python_version: The interpreter's version.
        operating_system: ``platform.platform()``, for the host. None for a venv, which runs under
            the same operating system as the host that built it.
        dependencies: Package name to version, for whichever packages were recorded.
        dependencies_digest: SHA-256 over the venv's full installed-distribution listing, when
            ``dependencies`` holds a declared subset rather than that full listing.
        senselab_version: senselab's own installed version, for the host.
        run_id: The run the id was minted in, or None for a record read from `provenance@1`.
    """

    _DIGESTED: ClassVar[tuple[str, ...]] = (
        "kind",
        "label",
        "python_version",
        "operating_system",
        "dependencies",
        "dependencies_digest",
        "senselab_version",
    )
    _CANONICAL: ClassVar[tuple[str, ...]] = _DIGESTED
    _KIND: ClassVar[str] = "environment"

    id: str
    kind: ENVIRONMENT_KIND
    label: str
    python_version: str
    operating_system: str | None = None
    dependencies: Mapping[str, str] = field(default_factory=dict)
    dependencies_digest: str | None = None
    senselab_version: str | None = None
    run_id: str | None = None

    def _prefix(self) -> str:
        return "env"

    def _check(self) -> None:
        if self.kind not in _ENVIRONMENT_KINDS:
            raise ValueError(f"unknown environment kind {self.kind!r}; known: {sorted(_ENVIRONMENT_KINDS)}")
        _text(self.label, "label")
        _text(self.python_version, "python_version")
        for name in ("operating_system", "dependencies_digest", "senselab_version"):
            _text(getattr(self, name), name, optional=True)
        if not isinstance(self.dependencies, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.dependencies.items()
        ):
            raise ValueError("dependencies must map package names to version text")


_RECORD_TYPES: dict[str, type[_Record]] = {
    "entity": Entity,
    "activity": Activity,
    "agent": Agent,
    "environment": Environment,
}
_RECORD_KEYS: dict[str, frozenset[str]] = {
    **{kind: frozenset(f.name for f in fields(cast(Any, t))) for kind, t in _RECORD_TYPES.items()},
    "relation": frozenset({"relation", "source", "target"}),
}
_LEGACY_NORMALISED = {"entity": ("attributes",), "activity": ("parameters",), "environment": ("dependencies",)}


_R = TypeVar("_R", bound=_Record)


def _mint(cls: type[_R], run_id: str, raw: Mapping[str, object], **plain: object) -> _R:
    """Build a record of `cls` in `run_id`, `raw` fields normalised first and the id minted from the content.

    Raises:
        UnrecordableValue: if a field holds a value outside the provenance value type.
        ValueError: if the record's fields are refused.
    """
    normal = {name: normalised(value, name) for name, value in raw.items()}
    draft = cls(id="", run_id=None, **normal, **plain)  # type: ignore[call-arg]
    return cls(id=draft.minted_id(run_id), run_id=run_id, **normal, **plain)  # type: ignore[call-arg]


def _put(table: dict[str, _R], record: _R, where: str = "") -> None:
    """Hold `record` under its id, refusing a different record already held there.

    Raises:
        ValueError: if a different record holds the id.
    """
    held = table.get(record.id)
    if held is not None and held != record:
        raise ValueError(f"{where}two different records hold the id {record.id!r}: {held!r} and {record!r}")
    table[record.id] = record


class ProvStore:
    """An append-only PROV document.

    Args:
        run_id: Mixed into every id so two runs never collide.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty store."""
        self.run_id = run_id
        self._entities: dict[str, Entity] = {}
        self._activities: dict[str, Activity] = {}
        self._agents: dict[str, Agent] = {}
        self._environments: dict[str, Environment] = {}
        self._relations: list[tuple[RELATION, str, str]] = []

    def entity(self, *, prov_type: PROV_TYPE, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """Add an entity holding the normalised, read-only form of `attributes`.

        Args:
            prov_type: What sort of thing it is.
            extent: ``(start, end)`` in seconds, finite, ``start <= end``, or None for something without one.
            attributes: Whatever describes it; a mapping with ``str`` keys of provenance values.

        Returns:
            Its id.

        Raises:
            UnrecordableValue: If ``attributes`` is not a mapping of provenance values.
            ValueError: If ``prov_type`` is unknown, the extent is malformed, or the attributes carry
                both a checksum and a reason it is missing, a malformed digest, an empty reason, or
                either without a ``path`` on an entity other than an ``artifact``.
        """
        record = _mint(Entity, self.run_id, {"attributes": attributes}, prov_type=prov_type, extent=extent)
        _put(self._entities, record)
        return record.id

    def activity(
        self,
        *,
        node: str,
        step: str | None,
        parameters: dict[str, Any],
        started: str | None = None,
        ended: str | None = None,
    ) -> str:
        """Add an activity, replacing the timing of one already held with the same id.

        Args:
            node: The node executing.
            step: Which step of it, when a node has several.
            parameters: The values it ran with.
            started: When it began, ISO 8601. Excluded from the id digest.
            ended: When it finished, ISO 8601. Excluded from the id digest.

        Returns:
            Its id — the same for two executions that differ only in ``started``/``ended``.

        Raises:
            UnrecordableValue: If ``parameters`` is not a mapping of provenance values.
        """
        record = _mint(
            Activity, self.run_id, {"parameters": parameters}, node=node, step=step, started=started, ended=ended
        )
        self._activities[record.id] = record
        return record.id

    def agent(
        self,
        *,
        agent_type: AGENT_TYPE,
        model_id: str | None = None,
        commit_sha: str | None = None,
        unresolved_reason: str | None = None,
        version: str | None = None,
    ) -> str:
        """Add an agent.

        Args:
            agent_type: ``"model"`` or ``"software"``.
            model_id: Required for a model agent.
            commit_sha: A resolved 40-hex commit.
            unresolved_reason: Why the commit is unknown, if it is.
            version: Software version, for a software agent.

        Returns:
            Its id.

        Raises:
            ValueError: If ``agent_type`` is unknown, ``commit_sha`` is not exactly 40 hex characters,
                ``commit_sha`` and ``unresolved_reason`` are supplied together, ``unresolved_reason``
                is empty, or a model agent is missing ``model_id`` or supplies neither a commit nor a
                reason it is missing.
        """
        record = _mint(
            Agent,
            self.run_id,
            {},
            agent_type=agent_type,
            model_id=model_id,
            commit_sha=commit_sha,
            unresolved_reason=unresolved_reason,
            version=version,
        )
        _put(self._agents, record)
        return record.id

    def environment(
        self,
        *,
        kind: ENVIRONMENT_KIND,
        label: str,
        python_version: str,
        operating_system: str | None = None,
        dependencies: dict[str, str] | None = None,
        dependencies_digest: str | None = None,
        senselab_version: str | None = None,
    ) -> str:
        """Add an environment: the host interpreter, or one subprocess venv.

        Args:
            kind: ``"host"`` or ``"venv"``.
            label: The host's own name, or the venv's resolved directory name.
            python_version: The interpreter's version.
            operating_system: ``platform.platform()``, for the host.
            dependencies: Package name to version.
            dependencies_digest: SHA-256 over the venv's full installed-distribution listing, when
                ``dependencies`` is a declared subset of it.
            senselab_version: senselab's own installed version, for the host.

        Returns:
            Its id.

        Raises:
            ValueError: If ``kind`` is unknown or a field is not text.
        """
        record = _mint(
            Environment,
            self.run_id,
            {"dependencies": dict(dependencies or {})},
            kind=kind,
            label=label,
            python_version=python_version,
            operating_system=operating_system,
            dependencies_digest=dependencies_digest,
            senselab_version=senselab_version,
        )
        _put(self._environments, record)
        return record.id

    def _relate(self, relation: RELATION, source: str, target: str) -> None:
        triple = (relation, source, target)
        if triple not in self._relations:
            self._relations.append(triple)

    def was_generated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an activity produced an entity."""
        self._relate("wasGeneratedBy", entity_id, activity_id)

    def used(self, activity_id: str, entity_id: str) -> None:
        """Record that an activity read an entity."""
        self._relate("used", activity_id, entity_id)

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        """Record which agent ran an activity."""
        self._relate("wasAssociatedWith", activity_id, agent_id)

    def was_attributed_to(self, entity_id: str, agent_id: str) -> None:
        """Record which agent is answerable for an entity."""
        self._relate("wasAttributedTo", entity_id, agent_id)

    def was_derived_from(self, entity_id: str, source_entity_id: str) -> None:
        """Record that an entity refines or answers another, keeping both."""
        self._relate("wasDerivedFrom", entity_id, source_entity_id)

    def was_invalidated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an entity should no longer be read as what it was. It is not removed."""
        self._relate("wasInvalidatedBy", entity_id, activity_id)

    def get_entity(self, entity_id: str) -> Entity:
        """Return one entity."""
        return self._entities[entity_id]

    def get_activity(self, activity_id: str) -> Activity:
        """Return one activity."""
        return self._activities[activity_id]

    def get_agent(self, agent_id: str) -> Agent:
        """Return one agent."""
        return self._agents[agent_id]

    def get_environment(self, environment_id: str) -> Environment:
        """Return one environment."""
        return self._environments[environment_id]

    def entities(self, prov_type: PROV_TYPE | None = None) -> list[Entity]:
        """Return entities, optionally of one type."""
        return [e for e in self._entities.values() if prov_type is None or e.prov_type == prov_type]

    def activities(self, node: str | None = None) -> list[Activity]:
        """Return activities, optionally of one node."""
        return [a for a in self._activities.values() if node is None or a.node == node]

    def agents(self, agent_type: AGENT_TYPE | None = None) -> list[Agent]:
        """Return agents, optionally of one type.

        Args:
            agent_type: ``"model"`` or ``"software"``, or None for every agent.

        Returns:
            The agents, in write order.
        """
        return [g for g in self._agents.values() if agent_type is None or g.agent_type == agent_type]

    def environments(self, kind: ENVIRONMENT_KIND | None = None) -> list[Environment]:
        """Return environments, optionally of one kind.

        Args:
            kind: ``"host"`` or ``"venv"``, or None for every environment.

        Returns:
            The environments, in write order.
        """
        return [e for e in self._environments.values() if kind is None or e.kind == kind]

    def relations(self) -> list[tuple[RELATION, str, str]]:
        """Return every relation as ``(relation, source, target)``, in write order."""
        return list(self._relations)

    def associations_of(self, agent_id: str) -> list[str]:
        """Return the activities an agent was associated with.

        The inverse of :meth:`associated_with`, for a reader asking what one agent did rather than
        who ran one activity.

        Args:
            agent_id: The agent's id.

        Returns:
            The activity ids, in the order the relations were recorded.
        """
        return [s for r, s, t in self._relations if r == "wasAssociatedWith" and t == agent_id]

    def _targets(self, relation: RELATION, source: str) -> list[str]:
        return [t for r, s, t in self._relations if r == relation and s == source]

    def generated_by(self, entity_id: str) -> str | None:
        """Return the activity that generated an entity, or None."""
        found = self._targets("wasGeneratedBy", entity_id)
        return found[0] if found else None

    def uses_of(self, activity_id: str) -> list[str]:
        """Return the entities an activity read."""
        return self._targets("used", activity_id)

    def associated_with(self, activity_id: str) -> list[str]:
        """Return the agents associated with an activity."""
        return self._targets("wasAssociatedWith", activity_id)

    def derived_from(self, entity_id: str) -> list[str]:
        """Return the entities an entity was derived from."""
        return self._targets("wasDerivedFrom", entity_id)

    def invalidated_by(self, entity_id: str) -> list[str]:
        """Return the activities that invalidated an entity, in the order recorded.

        Args:
            entity_id: The entity's id.

        Returns:
            The activity ids, empty when the entity is still live.
        """
        return self._targets("wasInvalidatedBy", entity_id)

    def is_invalidated(self, entity_id: str) -> bool:
        """Whether an entity has been invalidated."""
        return bool(self._targets("wasInvalidatedBy", entity_id))

    def records(self) -> list[_Record]:
        """Return every entity, activity, agent and environment, in that order and in write order within each."""
        return [
            *self._entities.values(),
            *self._activities.values(),
            *self._agents.values(),
            *self._environments.values(),
        ]

    def _hold(self, record: _Record, where: str = "") -> None:
        tables: dict[str, dict[str, Any]] = {
            "entity": self._entities,
            "activity": self._activities,
            "agent": self._agents,
            "environment": self._environments,
        }
        _put(tables[record._KIND], record, where)

    def write_jsonl(self, path: str | Path) -> None:
        """Write the store as `provenance@2.0`, one canonical JSON (RFC 8785) record per line, atomically.

        The first line is the format record. Each entity, activity, agent and environment line holds
        every field of the record as the store holds it, `run_id` included, so `read_jsonl` rebuilds
        the same records and checks each id against its content. The records go to a temporary file in
        the target's directory, which is flushed, fsynced and renamed onto the target, so the file at
        ``path`` is always either the previous store or the new one whole. An existing target's
        permission bits are kept.
        """
        lines = [jcs_dumps({"format": PROVENANCE_FORMAT, "record": "format"})]
        lines += [jcs_dumps(record.as_record()) for record in self.records()]
        lines += [
            jcs_dumps({"record": "relation", "relation": r, "source": s, "target": t}) for r, s, t in self._relations
        ]
        write_atomically(Path(path), b"".join(line + b"\n" for line in lines))

    @classmethod
    def read_jsonl(cls, path: str | Path, run_id: str = "read") -> "ProvStore":
        """Read a store back through the same record validation the store's own methods use.

        A `provenance@2.0` file (`write_jsonl`) is read line by line as canonical JSON; every record is
        rebuilt as its type, which checks its fields and, for a record with a `run_id`, that its id is
        the id of its content. A file without a format line is `provenance@1`: its values are
        normalised as the writer normalises them, and its records keep the ids that file gave them,
        with no `run_id`. Relations pass through the same membership check as the relation methods, so
        a relation line repeated reads back as one triple; a record repeated is held once, and two
        different records with one id are refused.

        Args:
            path: The JSONL file to read.
            run_id: The run id of the returned store, under which it mints the records it adds.

        Returns:
            The reconstructed store; for a file `write_jsonl` wrote, records equal to the written store's.

        Raises:
            ValueError: If a line is not a record object, a `provenance@2.0` line is not canonical JSON,
                a record has an unrecognised kind, missing or unexpected keys, a field its type refuses,
                or an id that is not the id of its content, two records share an id, or the format
                is not one this reader reads. The error names the file, the line and the record.
        """
        store = cls(run_id=run_id)
        lines = [(n, line) for n, line in enumerate(Path(path).read_bytes().split(b"\n"), start=1) if line.strip()]
        legacy = True
        if lines:
            first = _parse(lines[0][1], f"{path}, line {lines[0][0]}", legacy=True)
            if first.get("record") == "format":
                if first != {"format": PROVENANCE_FORMAT, "record": "format"}:
                    raise ValueError(
                        f"{path}, line {lines[0][0]}: reads {PROVENANCE_FORMAT} and provenance@1, got {first!r}"
                    )
                legacy = False
                lines = lines[1:]
        for line_no, line in lines:
            where = f"{path}, line {line_no}"
            rec = _parse(line, where, legacy=legacy)
            kind = rec.pop("record", None)
            expected = _RECORD_KEYS.get(cast(str, kind))
            if expected is None:
                raise ValueError(f"{where}: unrecognised record kind {kind!r}")
            if legacy and kind != "relation":
                expected = expected - {"run_id"}
            missing, unexpected = sorted(expected - rec.keys()), sorted(rec.keys() - expected)
            if missing or unexpected:
                raise ValueError(
                    f"{where}: {kind} record missing keys {missing}, carrying unexpected keys {unexpected}"
                )
            if kind == "relation":
                if rec["relation"] not in _RELATIONS:
                    raise ValueError(f"{where}: unknown relation {rec['relation']!r}")
                if not (isinstance(rec["source"], str) and isinstance(rec["target"], str)):
                    raise ValueError(f"{where}: a relation joins two ids, got {rec!r}")
                store._relate(cast(RELATION, rec["relation"]), rec["source"], rec["target"])
                continue
            if legacy:
                for name in _LEGACY_NORMALISED.get(cast(str, kind), ()):
                    rec[name] = normalised(rec[name], name)
            try:
                record = _RECORD_TYPES[cast(str, kind)](**rec)
            except (ValueError, TypeError) as err:
                raise ValueError(f"{where}: {kind} {rec.get('id')!r}: {err}") from err
            store._hold(record, f"{where}: ")
        return store

    @classmethod
    def merge(cls, stores: Sequence["ProvStore"]) -> "ProvStore":
        """Union several stores; order-independent, since one id holds one record.

        Raises:
            ValueError: If two stores hold different records under one id.
        """
        out = cls(run_id="merged")
        for s in stores:
            for record in s.records():
                out._hold(record)
        seen: set[tuple[RELATION, str, str]] = set()
        for s in stores:
            for rel in s._relations:
                if rel not in seen:
                    seen.add(rel)
                    out._relations.append(rel)
        return out

    def fingerprint(self) -> str:
        """A content hash that ignores insertion order: 16 hex of `provenance-id@4` over the sorted ids."""
        listing = {
            "e": sorted(self._entities),
            "act": sorted(self._activities),
            "ag": sorted(self._agents),
            "env": sorted(self._environments),
            "r": sorted(f"{r}:{s}:{t}" for r, s, t in self._relations),
        }
        return identity_digest(_ID_TAG, [listing])[:16]


def _parse(line: bytes, where: str, *, legacy: bool) -> dict[str, Any]:
    """Parse a line: canonical JSON for `provenance@2.0`; any JSON, bare non-finite literals too, for `provenance@1`."""
    try:
        value = json.loads(line) if legacy else jcs_loads(line)
    except ValueError as err:
        raise ValueError(f"{where}: not {'JSON' if legacy else 'canonical JSON'}: {err}") from err
    if not isinstance(value, dict) or "record" not in value:
        raise ValueError(f"{where}: not an object carrying a 'record' key: {line[:200]!r}")
    return value
