"""Identity v2: model references, logical keys, execution context and reuse policy (ADR-002, P-38)."""

import re
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal, assert_never
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PlainSerializer,
    Strict,
    StringConstraints,
    model_validator,
)
from senselab_worker.canonical import jcs_dumps, jcs_loads
from senselab_worker.digest import COMMIT_HEX, SHA256_HEX, identity_digest

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, StrMap, thawed
from senselab_core.kernel.license import License

_HF_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
_HF_REF = re.compile(r"(?!.*\.\.)[^\s@/\\.][^\s@/\\]*(/[^\s@/\\.][^\s@/\\]*)*")
_URL_HASH = "#sha256="

ContentIdentity = Annotated[str, StringConstraints(pattern=r"^content:[0-9a-f]{64}$")]
"""The identity of source bytes: `content:` and a 64-hex digest."""

OutputIdentity = Annotated[str, StringConstraints(pattern=r"^output:[0-9a-f]{64}$")]
"""The identity of a stored result's outputs: `output:` and a 64-hex digest."""

InputIdentity = ContentIdentity | OutputIdentity
"""What a logical key's input names: source content or an upstream output."""

ProfileSliceIdentity = Annotated[str, StringConstraints(pattern=r"^profile:[0-9a-f]{64}$")]
"""The identity of a profile slice a call reads: `profile:` and a 64-hex digest."""


class UnpinnedModel(ValueError):
    """A model whose identity does not pin its weights was asked for a key identity."""


def _pinned_identity_error(source: str, identity: str) -> str | None:
    pattern, form = (COMMIT_HEX, "a 40-hex commit") if source == "hf" else (SHA256_HEX, "a sha256")
    if pattern.fullmatch(identity):
        return None
    return f"a pinned {source} model's identity is {form}, got {identity!r}"


def _text_fields(text: str) -> tuple[Literal["hf", "url", "local"], str, str]:
    """Return the `(source, id, ref)` a reference's text form names; raise `ValueError` if it names none."""
    if text.startswith("local:"):
        return "local", text[len("local:") :], ""
    if text.startswith("url:"):
        url, sep, digest = text[len("url:") :].partition(_URL_HASH)
        return "url", url, digest if sep else ""
    model_id, sep, ref = text.partition("@")
    if not sep:
        raise ValueError(f"{text!r} needs an explicit revision: '{text}@<ref>', or a 'url:' or 'local:' prefix")
    return "hf", model_id, ref


class ModelRef(Frozen):
    """A model reference: hub id and revision, a URL, or a local path.

    Text forms, read by `parse` and written by `str`:
    - `<id>@<ref>`: source `hf`. `<id>` is `name` or `org/name` (letters, digits, `.`, `_`, `-`);
      `<ref>` is a branch, tag or commit: `/`-separated non-empty components without whitespace, `@` or a backslash,
      none starting with `.`, and no `..` anywhere.
    - `url:<url>#sha256=<64 hex>`: source `url`. `<url>` is the id and must be an absolute URL with a
      scheme and a host; `ref` holds the declared weight hash (pip-style), which `resolve` requires and
      which is checked against the downloaded bytes at load. `ref` is empty only for a url with no
      declared hash, which `resolve` refuses.
    - `local:<path>`: source `local`. The rest of the string is the id verbatim, any non-empty path;
      `ref` is empty.

    Every reference has a text form: one whose `str` does not parse back to it is refused.
    """

    source: Literal["hf", "url", "local"]
    id: NonEmptyStr
    ref: str

    @model_validator(mode="after")
    def _check_form(self) -> "ModelRef":
        self._check_source_form()
        if _text_fields(str(self)) != (self.source, self.id, self.ref):
            raise ValueError(f"{self.source} reference {self.id!r} has no text form that parses back to it")
        return self

    def _check_source_form(self) -> None:
        if self.source == "hf":
            if not _HF_ID.fullmatch(self.id):
                raise ValueError(f"hub id must be 'name' or 'org/name', got {self.id!r}")
            if not _HF_REF.fullmatch(self.ref):
                raise ValueError(
                    f"hub reference {self.id!r} needs a revision of '/'-separated components without whitespace, "
                    f"'@', '\\', a leading '.' or '..', got {self.ref!r}"
                )
            return
        if self.source == "local" and self.ref:
            raise ValueError(f"local references carry no revision, got {self.ref!r}")
        if self.source == "url":
            if self.ref and not SHA256_HEX.fullmatch(self.ref):
                raise ValueError(f"a url reference's declared hash is 64 lowercase hex, got {self.ref!r}")
            parts = urlsplit(self.id)
            if not parts.scheme or not parts.netloc or re.search(r"\s", self.id):
                raise ValueError(f"url reference must be an absolute URL with a host, got {self.id!r}")

    @classmethod
    def parse(cls, text: str) -> "ModelRef":
        """Parse `org/name@rev`, `url:<url>` or `local:<path>`; raise `ValueError` on anything else."""
        source, model_id, ref = _text_fields(text)
        return cls(source=source, id=model_id, ref=ref)

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
        if self.unresolved_reason is None and (error := _pinned_identity_error(self.ref.source, self.identity)):
            raise ValueError(error)
        return self

    @property
    def pinned(self) -> bool:
        """Whether the identity pins the weights."""
        return self.unresolved_reason is None

    def key_identity(self) -> "ModelIdentity":
        """Return the model's identity as a logical key holds it.

        Raises:
            UnpinnedModel: if the model is not pinned.
        """
        if not self.pinned:
            raise UnpinnedModel(f"{self.ref} is not pinned ({self.unresolved_reason}); it cannot enter a key")
        model_id = self.ref.id if self.ref.source == "hf" else None
        return ModelIdentity(id=model_id, identity=self.identity)


class ModelIdentity(Frozen):
    """A pinned model's identity in a logical key: a hub repo and commit, or the digest of the weights alone.

    Built by `ResolvedModel.key_identity`. With `id` a hub repo id, `identity` is its 40-hex commit;
    with `id` None, `identity` is the sha256 of the weights (`file@1` of a single file, `weights@3` of
    a set), whether they were reached by `url:` or `local:`. The source is recorded in the execution
    context (`ExecutionContext.model_source`), not here.
    """

    id: NonEmptyStr | None
    identity: NonEmptyStr

    @model_validator(mode="after")
    def _check_pinned(self) -> "ModelIdentity":
        if error := _pinned_identity_error("hf" if self.id is not None else "weights", self.identity):
            raise ValueError(error)
        return self


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
        return jcs_loads(jcs_dumps(thawed(v)))
    except TypeError as e:
        raise ValueError(f"params must be canonical JSON: {e}") from e


Params = Annotated[
    Mapping[str, Any],
    BeforeValidator(_canonical_params),
    AfterValidator(_freeze),
    PlainSerializer(thawed),
]
"""A JSON object held in its RFC 8785 canonical value, deeply read-only (objects as mappings, arrays as tuples)."""


class LogicalKey(Frozen):
    """The exact, portable identity of a task call (ADR-002).

    `model` is a pinned model's identity; `params` and `backend_params` are canonicalised (NFC strings,
    JCS numbers) at construction and refused if they are not JSON; `inputs` maps input names to `content:` or
    `output:` identities and `profile_slices` slice names to `profile:` identities. The digest is
    computed at construction, so a key that validates can be digested. No execution context, code
    or environment is part of the key.
    """

    schema_version: Annotated[int, Strict(), Field(ge=1)] = 1
    task: NonEmptyStr
    task_version: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    model: ModelIdentity | None
    params: Params
    backend_params: Params
    inputs: Annotated[Mapping[str, InputIdentity], AfterValidator(FrozenMapping), PlainSerializer(dict)]
    profile_slices: Annotated[
        Mapping[str, ProfileSliceIdentity], AfterValidator(FrozenMapping), PlainSerializer(dict)
    ] = FrozenMapping({})
    batch_formation: NonEmptyStr | None = None
    chunk_grid: NonEmptyStr | None = None

    def _derived(self) -> Mapping[str, object]:
        return {**super()._derived(), "_digest": identity_digest("logical-key@6", self.model_dump())}

    def digest(self) -> str:
        """Return `identity_digest("logical-key@6", fields)`, computed at construction."""
        return str(self.__pydantic_private__["_digest"])  # type: ignore[index]


class ExecutionContext(Frozen):
    """Where and how a result was computed; recorded with each result, never keyed (ADR-002).

    `protocol` is recorded but checked only at the worker handshake; `model_source` is how the model's
    weights were reached (`hf`, `url`, `local`, or None for a call without a model). Neither is fingerprinted.
    """

    env_lock: NonEmptyStr
    platform: NonEmptyStr
    wheel_set: NonEmptyStr
    numerics: StrMap
    decoder: NonEmptyStr | None
    driver: StrMap
    host: NonEmptyStr
    protocol: NonEmptyStr
    model_source: Literal["hf", "url", "local"] | None

    def fingerprint(self) -> str:
        """Return `identity_digest("context@3", fields)` over every field but `host`, `protocol` and `model_source`."""
        return identity_digest("context@3", self.model_dump(exclude={"host", "protocol", "model_source"}))


class ReusePolicy(str, Enum):
    """Whether a stored result from another execution context may serve the current one."""

    EXACT = "exact"
    EQUIVALENT = "equivalent"
    TOLERANT = "tolerant"


DEFAULT_REUSE_POLICY = ReusePolicy.EQUIVALENT

Equivalence = Mapping[str, Mapping[str, frozenset[str]]]
"""Measured equivalence classes: backend name to class name to the context fingerprints in it."""

EQUIVALENCE_FORMAT = "equivalence@1"


def _equivalence_class(where: str, raw: object) -> frozenset[str]:
    if not isinstance(raw, Mapping) or set(raw) != {"fingerprints", "derivation"}:
        raise ValueError(
            f"{EQUIVALENCE_FORMAT} class {where} must be a mapping of exactly 'fingerprints' and 'derivation'"
        )
    derivation, fingerprints = raw["derivation"], raw["fingerprints"]
    if not isinstance(derivation, str) or not derivation.strip():
        raise ValueError(f"{EQUIVALENCE_FORMAT} class {where} needs a non-empty derivation")
    if not isinstance(fingerprints, list) or not all(
        isinstance(f, str) and SHA256_HEX.fullmatch(f) for f in fingerprints
    ):
        raise ValueError(f"{EQUIVALENCE_FORMAT} class {where} must list context fingerprints, 64 lowercase hex each")
    members = frozenset(fingerprints)
    if len(members) < 2:  # noqa: PLR2004 - a class is two or more members by definition, not a tunable
        raise ValueError(f"{EQUIVALENCE_FORMAT} class {where} needs at least two distinct fingerprints")
    return members


def parse_equivalence(text: str) -> Equivalence:
    """Return the read-only equivalence table an `equivalence@1` YAML document declares.

    The document holds `format: equivalence@1` and `backends`, a mapping of backend name to class name
    to `{fingerprints: [...], derivation: ...}`; a class lists at least two distinct context
    fingerprints and a non-empty derivation.

    Raises:
        ValueError: if the document does not have that form.
    """
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or set(raw) != {"format", "backends"} or raw["format"] != EQUIVALENCE_FORMAT:
        raise ValueError(f"an equivalence table is a mapping of exactly format {EQUIVALENCE_FORMAT!r} and 'backends'")
    backends = raw["backends"]
    if not isinstance(backends, dict) or not all(isinstance(classes, dict) for classes in backends.values()):
        raise ValueError(f"{EQUIVALENCE_FORMAT} backends must map each backend to a mapping of classes")
    return FrozenMapping(
        {
            str(backend): FrozenMapping(
                {str(name): _equivalence_class(f"{backend}/{name}", cls) for name, cls in classes.items()}
            )
            for backend, classes in backends.items()
        }
    )


NO_EQUIVALENCE: Equivalence = FrozenMapping({})
"""The empty table: no measured classes, so `equivalent` reuse admits only equal fingerprints."""


def acceptable(
    stored: ExecutionContext,
    current: ExecutionContext,
    policy: ReusePolicy | str,
    *,
    backend: str,
    equivalence: Equivalence = NO_EQUIVALENCE,
) -> bool:
    """Return whether a result `backend` computed in `stored` may be reused in `current` under `policy`.

    `policy` is coerced with `ReusePolicy(policy)`. `exact` needs equal fingerprints; `equivalent`
    also accepts two fingerprints in one of the backend's classes in `equivalence`; `tolerant`
    accepts any context.

    Raises:
        ValueError: if `policy` is not a `ReusePolicy` value.
    """
    policy = ReusePolicy(policy)
    a, b = stored.fingerprint(), current.fingerprint()
    match policy:
        case ReusePolicy.TOLERANT:
            return True
        case ReusePolicy.EXACT:
            return a == b
        case ReusePolicy.EQUIVALENT:
            return a == b or any(a in members and b in members for members in equivalence.get(backend, {}).values())
        case _:
            assert_never(policy)
