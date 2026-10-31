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
import math
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, ClassVar, Literal, NoReturn, Sequence, TypeVar, cast, get_args, overload

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
    escaped_key,
    in_json_domain,
    jcs_dumps,
    jcs_loads,
)
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import FrozenDict, FrozenList, frozen_array, thawed
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
"""The format `write_jsonl` writes a store whose every record is verified: its first line is the format record.

Every record in it carries a `run_id`, and its id is checked against its content when it is read.
"""

UNVERIFIED_FORMAT = "provenance@1.1"
"""The format `write_jsonl` writes a store holding records read from `provenance@1`.

The lines are canonical like `provenance@2.0`'s; a record with `run_id` null is unverified and keeps the
id the `provenance@1` file gave it. A file whose first line is not a format record is `provenance@1`.
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


@dataclass(frozen=True)
class EnumMember:
    """An enum member as a record holds it: the qualified name of its type and its value, equal by both."""

    type_name: str
    value: Any


def _held_family(v: object) -> tuple[str | None, object] | None:
    t = type(v)
    if t is FrozenDict:
        return None, dict(cast("FrozenDict", v))
    if t is FrozenList:
        return None, list(cast("FrozenList", v))
    if t is EnumMember:
        member = cast(EnumMember, v)
        return "$enum", [member.type_name, member.value]
    return None


def _nd_token(a: numpy.ndarray, recurse: Recurse, path: str) -> object:
    return {"$nd": base64.b64encode(canonical_npy_bytes(a)).decode("ascii")}


_TOKENISER = Tokeniser(array=_nd_token, extra=_held_family, error=UnrecordableValue)


def normalised(value: object, path: str = "value") -> object:
    """Return `value` as JSON the shared encoder accepts, every value outside the JSON domain as a tagged token.

    The provenance value type is what the shared `canonical.Tokeniser` writes, with a `numpy.ndarray`
    written as `{"$nd": <base64 of its canonical .npy bytes>}`. A value read back from a record
    (`FrozenDict`, `FrozenList`, `EnumMember`) is written as the value it stands for, so recording a
    read-back records the same tokens again.

    Raises:
        UnrecordableValue: naming `path` and the type, for a value of no family, an array of a dtype
            canonical `.npy` refuses, a non-`str` key, two keys equal after NFC, a container that contains
            itself, an integer too long to write, or nesting deeper than `MAX_DEPTH`.
    """
    return _TOKENISER(value, path)


def _refuse(path: str, value: object, reason: str) -> NoReturn:
    raise UnrecordableValue(path, value, reason)


def _token_value(key: str, payload: object, path: str, walk: Walk) -> object:
    where = f"{path}.{key}"
    if key == "$f":
        if payload not in _FLOAT_NAMES:
            _refuse(where, payload, f"is not one of {sorted(_FLOAT_NAMES)}")
        return float(cast(str, payload))
    if key == "$int":
        if not (isinstance(payload, str) and _INT_TOKEN_TEXT.fullmatch(payload)):
            _refuse(where, payload, "is not a decimal integer")
        try:
            return int(payload)
        except ValueError as err:
            _refuse(where, payload, str(err))
    if key == "$enum":
        if not (isinstance(payload, list) and len(payload) == 2 and isinstance(payload[0], str)):
            _refuse(where, payload, "is not [<type name>, <value>]")
        return EnumMember(payload[0], _decoded(payload[1], f"{where}.value", walk, hashable=True))
    if key == "$set":
        if not isinstance(payload, list):
            _refuse(where, payload, "is not a list of elements")
        return frozenset(_decoded(item, f"{where}[{i}]", walk, hashable=True) for i, item in enumerate(payload))
    if not isinstance(payload, str):
        _refuse(where, payload, "is not text")
    if key == "$path":
        return Path(payload)
    if key == "$dec":
        try:
            return Decimal(payload)
        except InvalidOperation as err:
            _refuse(where, payload, f"is not a decimal: {err}")
    if key == "$bytes":
        try:
            return bytes.fromhex(payload)
        except ValueError as err:
            _refuse(where, payload, f"is not hex: {err}")
    try:
        return frozen_array(numpy.load(io.BytesIO(base64.b64decode(payload, validate=True)), allow_pickle=False))
    except (ValueError, TypeError, OSError, binascii.Error) as err:
        _refuse(where, payload, f"is not a .npy array in base64: {err}")


def _decoded(value: object, path: str, walk: Walk, *, hashable: bool = False) -> Any:  # noqa: ANN401
    """Return the value a token form stands for: objects as `FrozenDict` (keys unescaped), arrays as `FrozenList`.

    With `hashable`, an array is a `tuple` (an element of a set or an enum's value).

    Raises:
        UnrecordableValue: naming the path, for a malformed token, a cycle, or nesting deeper than `MAX_DEPTH`.
    """
    try:
        if isinstance(value, dict):
            items = cast("dict[str, object]", value)
            if len(items) == 1:
                ((key, payload),) = items.items()
                if key in _PROV_TOKENS:
                    return walk.enter(value, lambda: _token_value(key, payload, path, walk))
            return walk.enter(
                value,
                lambda: FrozenDict(
                    {(k[1:] if k.startswith("$$") else k): _decoded(v, f"{path}.{k}", walk) for k, v in items.items()}
                ),
            )
        if isinstance(value, (list, tuple)):
            elements = cast("Sequence[object]", value)
            kind: Callable[[Any], Any] = tuple if hashable else FrozenList
            return walk.enter(
                value,
                lambda: kind(_decoded(v, f"{path}[{i}]", walk, hashable=hashable) for i, v in enumerate(elements)),
            )
    except UnrecordableValue:
        raise
    except ValueError as err:
        _refuse(path, value, str(err))
    return value


def _frozen_json(value: object, walk: Walk) -> Any:  # noqa: ANN401
    if isinstance(value, dict):
        items = cast("dict[str, object]", value)
        return walk.enter(value, lambda: FrozenDict({k: _frozen_json(v, walk) for k, v in items.items()}))
    if isinstance(value, list):
        elements = cast("list[object]", value)
        return walk.enter(value, lambda: FrozenList(_frozen_json(v, walk) for v in elements))
    return value


def _line(record: Mapping[str, Any]) -> bytes:
    """Return the line a store file holds for a record: JSON with sorted keys, every number as Python writes it."""
    return json.dumps(thawed(record), sort_keys=True, allow_nan=False).encode("utf-8")


def recorded(value: object) -> Any:  # noqa: ANN401
    """Return what a caller reads back from a record that recorded `value`: every token decoded.

    Objects are `FrozenDict`, arrays `FrozenList` (equal to a `dict` and a `list` of the same items), a
    non-finite float or a large integer is itself, a path is a `Path`, an array a read-only `numpy.ndarray`,
    a set a `frozenset`, bytes `bytes`, a decimal a `Decimal`, and an enum member an `EnumMember`.

    Raises:
        UnrecordableValue: if `value` is outside the provenance value type.
    """
    return _decoded(normalised(value, "value"), "value", Walk())


def plain_values(value: object) -> Any:  # noqa: ANN401
    """Return a store file's JSON value as a caller reads it back: every token decoded, as `recorded` returns.

    For readers that parse a store file as JSON rather than through `ProvStore.read_jsonl`. A
    `provenance@1` value holds no tokens and is returned with its containers read-only.

    Raises:
        UnrecordableValue: for a malformed token.
    """
    return _decoded(value, "value", Walk())


def _held_value(value: object, name: str) -> tuple[Any, Any]:
    """Return a record field's tokens, read-only, and the decoded value a caller reads back.

    Raises:
        UnrecordableValue: if `value` is not a mapping of provenance values.
    """
    tokens = normalised(value, name)
    if not isinstance(tokens, dict):
        _refuse(name, value, "is not a mapping")
    return _frozen_json(tokens, Walk()), _decoded(tokens, name, Walk())


def _check_normal_form(tokens: object, name: str) -> Any:  # noqa: ANN401
    """Return the decoded value of a field read from a `provenance@2` file, refusing a form the writer cannot write.

    Raises:
        UnrecordableValue: if decoding and normalising again does not give the same JSON.
    """
    value = _decoded(tokens, name, Walk())
    if _line({"v": normalised(value, name)}) != _line({"v": tokens}):
        _refuse(name, tokens, "is not a form the store writes: decoding and recording it again differs")
    return value


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
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError(f"extent must hold two finite numbers of seconds, got {value!r}")
    if start > end:
        raise ValueError(f"extent ends before it starts: {value!r}")
    return start, end


class _Record:
    """What every provenance record shares: it is read-only and bound to its id.

    `_DIGESTED` names the fields the id covers, after `run_id`; `_VALUES` the fields of provenance values
    (attributes, parameters, dependencies). A value field holds what a caller reads back, every token
    decoded; the record also keeps the tokens it was identified by, which the file holds. A record with a
    `run_id` is verified: its id is checked against its content when it is built. A record whose `run_id`
    is `None` is unverified: one read from a `provenance@1` file, which keeps the id that file gave it, or
    a copy `revised` returns; a store holds an unverified record only when it read it from a file that may
    carry one. Two records are equal when their files' lines identify the same content (JCS).
    """

    _DIGESTED: ClassVar[tuple[str, ...]]
    _VALUES: ClassVar[tuple[str, ...]] = ()
    _KIND: ClassVar[str]
    __hash__ = None  # type: ignore[assignment]
    id: str
    run_id: str | None

    def _prefix(self) -> str:
        raise NotImplementedError

    def _check(self) -> None:
        raise NotImplementedError

    def __post_init__(self) -> None:
        """Check the fields, hold each value field decoded beside its tokens, and check the id against them."""
        _text(self.id, "id")
        _text(self.run_id, "run_id", optional=True)
        tokens: dict[str, Any] = {}
        for name in self._VALUES:
            tokens[name], held = _held_value(getattr(self, name), name)
            object.__setattr__(self, name, held)
        object.__setattr__(self, "_tokens", FrozenDict(tokens))
        self._check()
        if self.run_id is not None and self.id != self.minted_id():
            raise ValueError(f"{self._KIND} id {self.id!r} is not the id of its content, {self.minted_id()!r}")

    def _field(self, name: str) -> Any:  # noqa: ANN401
        return self._tokens[name] if name in self._VALUES else getattr(self, name)  # type: ignore[attr-defined]

    def minted_id(self, run_id: str | None = None) -> str:
        """Return the id of this content in `run_id` (by default the record's own): `<prefix>-` and 16 hex.

        The hex is `identity_digest("provenance-id@4", [run_id, *<the digested fields, value fields as tokens>])`.

        Raises:
            ValueError: if neither `run_id` nor the record's has a value.
            UnrecordableValue: if the content, inside the id's envelope, nests deeper than `MAX_DEPTH` or
                exceeds `MAX_CANONICAL_CHARS`.
        """
        run = self.run_id if run_id is None else run_id
        if run is None:
            raise ValueError(f"{self._KIND} {self.id!r} was read from provenance@1 and has no run_id to mint from")
        parts = [run, *(thawed(self._field(name)) for name in self._DIGESTED)]
        try:
            digest = identity_digest(_ID_TAG, parts)
        except ValueError as err:
            raise UnrecordableValue(".".join(("run_id", *self._DIGESTED)), parts, str(err)) from err
        return f"{self._prefix()}-{digest[:16]}"

    @property
    def verified(self) -> bool:
        """Whether this record's id was checked against its content: it has a `run_id`."""
        return self.run_id is not None

    def as_record(self) -> dict[str, Any]:
        """Return the JSON object a file holds for this record: value fields as their tokens."""
        return {"record": self._KIND, **{f.name: thawed(self._field(f.name)) for f in fields(cast(Any, self))}}

    def __eq__(self, other: object) -> bool:
        """Return whether `other` is a record of the same kind whose file line identifies the same content."""
        if not isinstance(other, _Record):
            return NotImplemented
        return type(self) is type(other) and jcs_dumps(self.as_record()) == jcs_dumps(other.as_record())


@dataclass(frozen=True, eq=False)
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
    _VALUES: ClassVar[tuple[str, ...]] = ("attributes",)
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
        _check_entity_attributes(self.prov_type, self.attributes)


@dataclass(frozen=True, eq=False)
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
    _VALUES: ClassVar[tuple[str, ...]] = ("parameters",)
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


@dataclass(frozen=True, eq=False)
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


@dataclass(frozen=True, eq=False)
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
    _VALUES: ClassVar[tuple[str, ...]] = ("dependencies",)
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
        if not all(isinstance(v, str) for v in self.dependencies.values()):
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


_R = TypeVar("_R", bound=_Record)
_UNTIMED: dict[str, Any] = {"started": None, "ended": None}


def _mint(cls: type[_R], run_id: str, raw: Mapping[str, object], **plain: object) -> _R:
    """Build a record of `cls` in `run_id` with the id minted from its content.

    Raises:
        UnrecordableValue: if a field holds a value outside the provenance value type.
        ValueError: if the record's fields are refused.
    """
    draft = cls(id="", run_id=None, **raw, **plain)  # type: ignore[call-arg]
    return cast(_R, replace(cast(Any, draft), id=draft.minted_id(run_id), run_id=run_id))


@overload
def revised(target: _R, /, **changes: object) -> _R: ...
@overload
def revised(target: Mapping[str, Any], /, **changes: object) -> FrozenDict: ...
def revised(target: _Record | Mapping[str, Any], /, **changes: object) -> _Record | FrozenDict:
    """Return a copy of a record, or of a mapping a record holds, with `changes`; the original is untouched.

    For a mapping a record holds (attributes, parameters, an object inside them): what a caller reads back
    from a record of its items with `changes` (`recorded`). For a record: a copy with the changed fields
    that keeps the record's id; the copy is unverified (`run_id` None), so a store refuses to hold it. It
    is a view, such as a word with a bounded extent.

    Raises:
        UnrecordableValue: if a change holds a value outside the provenance value type.
        ValueError: if a record's changed fields are refused, or a change names no field of the record.
    """
    if isinstance(target, _Record):
        names = {f.name for f in fields(cast(Any, target))}
        unknown = sorted(set(changes) - names)
        if unknown:
            raise ValueError(f"{type(target).__name__} has no fields {unknown}")
        return cast(_Record, replace(cast(Any, target), run_id=None, **changes))
    if not isinstance(target, Mapping):
        raise TypeError(f"revised takes a record or a mapping a record holds, not {type(target).__name__}")
    return cast(FrozenDict, recorded({**target, **changes}))


def _timing(record: _Record) -> tuple[str, str]:
    if isinstance(record, Activity):
        return record.started or "", record.ended or ""
    return "", ""


def _put(table: dict[str, _R], record: _R, where: str = "") -> None:
    """Hold `record` under its id, refusing a different record already held there.

    Two activities that differ only in `started` and `ended` are one activity: the held one is the
    later by `(started, ended)`, so holding them in either order holds the same record.

    Raises:
        ValueError: if a different record holds the id.
    """
    held = table.get(record.id)
    if held is not None and held != record:
        untimed = isinstance(record, Activity) and replace(cast(Any, held), **_UNTIMED) == replace(record, **_UNTIMED)
        if not untimed:
            raise ValueError(f"{where}two different records hold the id {record.id!r}: {held!r} and {record!r}")
        if _timing(held) >= _timing(record):
            return
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
        """Add an activity; of two with one id, differing only in timing, the store holds the later.

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
        _put(self._activities, record)
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

    @property
    def verified(self) -> bool:
        """Whether every record's id was checked against its content; False once it holds a `provenance@1` record."""
        return all(record.verified for record in self.records())

    def _hold(self, record: _Record, where: str = "", *, unverified: bool = False) -> None:
        if not (record.verified or unverified):
            raise ValueError(f"{where}{record._KIND} {record.id!r} has no run_id; a store holds it only from a file")
        tables: dict[str, dict[str, Any]] = {
            "entity": self._entities,
            "activity": self._activities,
            "agent": self._agents,
            "environment": self._environments,
        }
        _put(tables[record._KIND], record, where)

    def write_jsonl(self, path: str | Path) -> None:
        """Write the store as one JSON record per line, keys sorted and numbers as Python writes them, atomically.

        The first line is the format record: `provenance@2.0` when every record is verified, and
        `provenance@1.1` when the store holds records read from `provenance@1`, which are written with
        `run_id` null. Each entity, activity, agent and environment line holds every field of the record,
        value fields as the tokens the record was identified by, so `read_jsonl` rebuilds the same records
        (a float stays a float) and checks each verified id against its content. The records go to a
        temporary file in
        the target's directory, which is flushed, fsynced and renamed onto the target, so the file at
        ``path`` is always either the previous store or the new one whole. An existing target's
        permission bits are kept.
        """
        written = PROVENANCE_FORMAT if self.verified else UNVERIFIED_FORMAT
        lines = [_line({"format": written, "record": "format"})]
        lines += [_line(record.as_record()) for record in self.records()]
        lines += [_line({"record": "relation", "relation": r, "source": s, "target": t}) for r, s, t in self._relations]
        write_atomically(Path(path), b"".join(line + b"\n" for line in lines))

    @classmethod
    def read_jsonl(cls, path: str | Path, run_id: str = "read") -> "ProvStore":
        """Read a store back through the same record validation the store's own methods use.

        A `provenance@2.0` file is read line by line as JSON; each value field must be a form the writer
        writes (decoding and recording it again gives the same JSON), every record is rebuilt as its type,
        which checks its fields and that its id is the id of its content, and a record without a `run_id`
        is refused. Callers read every value decoded, as `recorded` returns it. A `provenance@1.1` file is
        read the same way, its records without a `run_id` held unverified. A file without a format line
        is `provenance@1`: its values are recorded as the store's methods record them, and its records keep
        the ids that file gave them, unverified.
        Relations pass through the same membership check as the relation methods, so a relation line
        repeated reads back as one triple; a record repeated is held once, and two different records
        with one id are refused.

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
        legacy, unverified = True, True
        if lines:
            first = _parse(lines[0][1], f"{path}, line {lines[0][0]}", legacy=True)
            if first.get("record") == "format":
                readable = {PROVENANCE_FORMAT, UNVERIFIED_FORMAT}
                if set(first) != {"format", "record"} or first["format"] not in readable:
                    raise ValueError(
                        f"{path}, line {lines[0][0]}: reads {sorted(readable)} and provenance@1, got {first!r}"
                    )
                legacy, unverified = False, first["format"] == UNVERIFIED_FORMAT
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
            try:
                if not legacy:
                    for name in _RECORD_TYPES[cast(str, kind)]._VALUES:
                        rec[name] = _check_normal_form(rec[name], name)
                record = _RECORD_TYPES[cast(str, kind)](**rec)
            except (ValueError, TypeError) as err:
                raise ValueError(f"{where}: {kind} {rec.get('id')!r}: {err}") from err
            store._hold(record, f"{where}: ", unverified=unverified)
        return store

    @classmethod
    def merge(cls, stores: Sequence["ProvStore"]) -> "ProvStore":
        """Union several stores; order-independent, since one id holds one record and two timings of an activity one.

        A record a store holds unverified is held unverified by the union.

        Raises:
            ValueError: If two stores hold different records under one id.
        """
        out = cls(run_id="merged")
        for s in stores:
            for record in s.records():
                out._hold(record, unverified=True)
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


def _refuse_constant(text: str) -> NoReturn:
    raise ValueError(f"non-finite number {text} is written as a token in this format")


def _parse(line: bytes, where: str, *, legacy: bool) -> dict[str, Any]:
    """Parse a line: JSON without bare non-finite literals, which `provenance@1` alone may hold."""
    try:
        value = json.loads(line) if legacy else json.loads(line, parse_constant=_refuse_constant)
    except ValueError as err:
        raise ValueError(f"{where}: not JSON: {err}") from err
    if not isinstance(value, dict) or "record" not in value:
        raise ValueError(f"{where}: not an object carrying a 'record' key: {line[:200]!r}")
    return value
