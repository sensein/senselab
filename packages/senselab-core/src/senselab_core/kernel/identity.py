"""Identity v2: model references, logical keys, execution context and reuse policy (ADR-002, P-38)."""

import json
import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BeforeValidator, Field, PlainSerializer, Strict, model_validator
from senselab_worker.canonical import jcs_dumps
from senselab_worker.hashing import sha256_bytes

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, StrMap
from senselab_core.kernel.license import License

_HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_HF_REF = re.compile(r"[^\s@]+")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_URL_HASH = "#sha256="


class ModelRef(Frozen):
    """A model reference: hub id and revision, a URL, or a local path.

    Text forms, read by `parse` and written by `str`:
    - `<id>@<ref>`: source `hf`. `<id>` is `name` or `org/name` (letters, digits, `.`, `_`, `-`);
      `<ref>` is a branch, tag or commit, non-empty, without whitespace or `@`.
    - `url:<url>#sha256=<64 hex>`: source `url`. `<url>` is the id and must be an absolute URL with a
      scheme and a host; `ref` holds the declared weight hash (pip-style), which `resolve` requires and
      which is checked against the downloaded bytes at load. `ref` is empty only for a url with no
      declared hash, which `resolve` refuses.
    - `local:<path>`: source `local`. The rest of the string is the id verbatim, any non-empty path;
      `ref` is empty.
    """

    source: Literal["hf", "url", "local"]
    id: NonEmptyStr
    ref: str

    @model_validator(mode="after")
    def _check_form(self) -> "ModelRef":
        if self.source == "hf":
            if not _HF_ID.fullmatch(self.id):
                raise ValueError(f"hub id must be 'name' or 'org/name', got {self.id!r}")
            if not _HF_REF.fullmatch(self.ref):
                raise ValueError(
                    f"hub reference {self.id!r} needs a revision without whitespace or '@', got {self.ref!r}"
                )
            return self
        if self.source == "local" and self.ref:
            raise ValueError(f"local references carry no revision, got {self.ref!r}")
        if self.source == "url":
            if self.ref and not _SHA256.fullmatch(self.ref):
                raise ValueError(f"a url reference's declared hash is 64 lowercase hex, got {self.ref!r}")
            parts = urlsplit(self.id)
            if not parts.scheme or not parts.netloc or re.search(r"\s", self.id):
                raise ValueError(f"url reference must be an absolute URL with a host, got {self.id!r}")
        return self

    @classmethod
    def parse(cls, text: str) -> "ModelRef":
        """Parse `org/name@rev`, `url:<url>` or `local:<path>`; raise `ValueError` on anything else."""
        if text.startswith("local:"):
            return cls(source="local", id=text[len("local:") :], ref="")
        if text.startswith("url:"):
            url, sep, digest = text[len("url:") :].partition(_URL_HASH)
            return cls(source="url", id=url, ref=digest if sep else "")
        model_id, sep, ref = text.partition("@")
        if not sep:
            raise ValueError(f"{text!r} needs an explicit revision: '{text}@<ref>', or a 'url:' or 'local:' prefix")
        return cls(source="hf", id=model_id, ref=ref)

    def __str__(self) -> str:
        """Return the text form that `parse` reads."""
        if self.source == "hf":
            return f"{self.id}@{self.ref}"
        return (
            f"url:{self.id}{_URL_HASH}{self.ref}" if self.source == "url" and self.ref else f"{self.source}:{self.id}"
        )


class ResolvedModel(Frozen):
    """A reference bound to the identity of its weights.

    A pinned (`unresolved_reason is None`) hub model's identity is its 40-hex commit; a pinned url or
    local model's is the sha256 of its weight set (for a url, the hash declared in the reference).
    A source that cannot be pinned carries a non-empty `unresolved_reason`, and its identity is
    unconstrained; none exists in Phase 1.
    """

    ref: ModelRef
    identity: NonEmptyStr
    unresolved_reason: NonEmptyStr | None = None
    license: License | None = None

    @model_validator(mode="after")
    def _check_identity(self) -> "ResolvedModel":
        if self.unresolved_reason is None:
            pattern, form = (_COMMIT, "a 40-hex commit") if self.ref.source == "hf" else (_SHA256, "a sha256")
            if not pattern.fullmatch(self.identity):
                raise ValueError(f"a pinned {self.ref.source} model's identity is {form}, got {self.identity!r}")
        return self

    @property
    def pinned(self) -> bool:
        """Whether the identity pins the weights."""
        return self.unresolved_reason is None


def _thaw(v: object) -> object:
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_thaw(x) for x in v]
    return v


def _freeze(v: object) -> object:
    if isinstance(v, Mapping):
        return FrozenMapping({k: _freeze(x) for k, x in v.items()})
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def _canonical_params(v: object) -> object:
    if not isinstance(v, Mapping):
        raise ValueError(f"params must be a JSON object, got {type(v).__name__}")
    try:
        return json.loads(jcs_dumps(_thaw(v)))
    except TypeError as e:
        raise ValueError(f"params must be canonical JSON: {e}") from e


Params = Annotated[
    Mapping[str, Any],
    BeforeValidator(_canonical_params),
    AfterValidator(_freeze),
    PlainSerializer(_thaw),
]
"""A JSON object held in its RFC 8785 canonical value, deeply read-only (objects as mappings, arrays as tuples)."""


class LogicalKey(Frozen):
    """The exact, portable identity of a task call (ADR-002).

    `model` is the weight identity; `params` is canonicalised (NFC strings, JCS numbers) at
    construction and refused if it is not JSON; `inputs` maps input names to content or upstream
    output identities. No execution context, code or environment is part of the key.
    """

    schema_version: Annotated[int, Strict(), Field(ge=1)] = 1
    task: NonEmptyStr
    task_version: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    model: NonEmptyStr | None
    params: Params
    inputs: StrMap
    profile_slices: StrMap = FrozenMapping({})
    batch_formation: NonEmptyStr | None = None
    chunk_grid: NonEmptyStr | None = None

    def digest(self) -> str:
        """Return the sha256 hex of the JCS of the key's fields."""
        return sha256_bytes(jcs_dumps(self.model_dump()))


class ExecutionContext(Frozen):
    """Where and how a result was computed; recorded with each result, never keyed (ADR-002).

    `protocol` is recorded but checked only at the worker handshake, so it is not fingerprinted.
    """

    env_lock: NonEmptyStr
    platform: NonEmptyStr
    wheel_set: NonEmptyStr
    numerics: StrMap
    decoder: NonEmptyStr | None
    driver: StrMap
    host: NonEmptyStr
    protocol: NonEmptyStr

    def fingerprint(self) -> str:
        """Return the sha256 hex of the JCS of every field except `host` and `protocol`."""
        return sha256_bytes(jcs_dumps(self.model_dump(exclude={"host", "protocol"})))


class ReusePolicy(str, Enum):
    """Whether a stored result from another execution context may serve the current one."""

    EXACT = "exact"
    EQUIVALENT = "equivalent"
    TOLERANT = "tolerant"


DEFAULT_REUSE_POLICY = ReusePolicy.EQUIVALENT

EQUIVALENCE: Mapping[str, Mapping[str, frozenset[str]]] = {}
"""Measured equivalence classes per backend: backend name to class name to the context fingerprints in it."""


def acceptable(stored: ExecutionContext, current: ExecutionContext, policy: ReusePolicy, *, backend: str) -> bool:
    """Return whether a result `backend` computed in `stored` may be reused in `current` under `policy`.

    `exact` needs equal fingerprints; `equivalent` also accepts two fingerprints in one of the
    backend's classes in `EQUIVALENCE`; `tolerant` accepts any context.
    """
    if policy is ReusePolicy.TOLERANT:
        return True
    a, b = stored.fingerprint(), current.fingerprint()
    if policy is ReusePolicy.EXACT or a == b:
        return a == b
    return any(a in members and b in members for members in EQUIVALENCE.get(backend, {}).values())
