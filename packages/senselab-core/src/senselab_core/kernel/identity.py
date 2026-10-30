"""Identity v2: model references, logical keys, execution context and reuse policy (ADR-002, P-38)."""

import json
import re
import warnings
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


class ModelRef(Frozen):
    """A model reference: hub id and revision, a URL, or a local path.

    Text forms, read by `parse` and written by `str`:
    - `<id>@<ref>`: source `hf`. `<id>` is `name` or `org/name` (letters, digits, `.`, `_`, `-`);
      `<ref>` is a branch, tag or commit, non-empty, without whitespace or `@`.
    - `url:<url>`: source `url`. The rest of the string is the id verbatim and must be an absolute
      URL with a scheme and a host; `ref` is empty.
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
        if self.ref:
            raise ValueError(f"{self.source} references carry no revision, got {self.ref!r}")
        if self.source == "url":
            parts = urlsplit(self.id)
            if not parts.scheme or not parts.netloc or re.search(r"\s", self.id):
                raise ValueError(f"url reference must be an absolute URL with a host, got {self.id!r}")
        return self

    @classmethod
    def parse(cls, text: str) -> "ModelRef":
        """Parse `org/name@rev`, `url:<url>` or `local:<path>`; raise `ValueError` on anything else."""
        for source in ("url", "local"):
            prefix = source + ":"
            if text.startswith(prefix):
                return cls(source=source, id=text[len(prefix) :], ref="")
        model_id, sep, ref = text.partition("@")
        if not sep:
            raise ValueError(f"{text!r} needs an explicit revision: '{text}@<ref>', or a 'url:' or 'local:' prefix")
        return cls(source="hf", id=model_id, ref=ref)

    def __str__(self) -> str:
        """Return the text form that `parse` reads."""
        return f"{self.id}@{self.ref}" if self.source == "hf" else f"{self.source}:{self.id}"


class ResolvedModel(Frozen):
    """A reference bound to the identity of its weights.

    A pinned (`unresolved_reason is None`) hub model's identity is its 40-hex commit; a pinned url or
    local model's is the sha256 of its weight set. An unpinnable source carries a non-empty
    `unresolved_reason`, and its identity is unconstrained.
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


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "schema"', category=UserWarning)

    class LogicalKey(Frozen):
        """The exact, portable identity of a task call (ADR-002).

        `model` is the weight identity; `params` is canonicalised (NFC strings, JCS numbers) at
        construction and refused if it is not JSON; `inputs` maps input names to content or upstream
        output identities. No execution context, code or environment is part of the key.
        """

        schema: Annotated[int, Strict(), Field(ge=1)] = 1  # type: ignore[assignment]
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
    """Where and how a result was computed; recorded with each result, never keyed (ADR-002)."""

    env_lock: NonEmptyStr
    platform: NonEmptyStr
    wheel_set: NonEmptyStr
    numerics: StrMap
    decoder: NonEmptyStr | None
    driver: StrMap
    host: NonEmptyStr
    protocol: NonEmptyStr

    def fingerprint(self) -> str:
        """Return the sha256 hex of the JCS of every field except `host`."""
        return sha256_bytes(jcs_dumps(self.model_dump(exclude={"host"})))


class ReusePolicy(str, Enum):
    """Whether a stored result from another execution context may serve the current one."""

    EXACT = "exact"
    EQUIVALENT = "equivalent"
    TOLERANT = "tolerant"


DEFAULT_REUSE_POLICY = ReusePolicy.EQUIVALENT

EQUIVALENCE: dict[str, frozenset[str]] = {}
"""Measured equivalence classes: class name to the context fingerprints in it."""


def acceptable(stored: ExecutionContext, current: ExecutionContext, policy: ReusePolicy = DEFAULT_REUSE_POLICY) -> bool:
    """Return whether a result computed in `stored` may be reused in `current` under `policy`.

    `exact` needs equal fingerprints; `equivalent` also accepts two fingerprints in one class of
    `EQUIVALENCE`; `tolerant` accepts any context.
    """
    if policy is ReusePolicy.TOLERANT:
        return True
    a, b = stored.fingerprint(), current.fingerprint()
    if a == b or policy is ReusePolicy.EXACT:
        return a == b
    return any(a in members and b in members for members in EQUIVALENCE.values())
