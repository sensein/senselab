"""Declaration registry: task and backend specs, refusal of duplicates (P-37), the logical key builder."""

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model
from senselab_core.kernel._base import NonEmptyStr
from senselab_core.kernel.declare import (
    BackendSpec,
    DuplicateDeclaration,
    NonCanonicalParams,
    Registry,
    TaskParams,
    TaskSpec,
    UnknownTask,
    check_canonical_params,
    key_for,
)
from senselab_core.kernel.identity import ModelIdentity, ModelRef, ResolvedModel, UnpinnedModel
from senselab_core.kernel.license import License

COMMIT = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
AUDIO = "content:" + "a" * 64
SLICE = "profile:" + "b" * 64

LICENSE = License(
    expression="Apache-2.0",
    commercial_use="yes",
    copyleft="none",
    attribution=True,
    gated=False,
    source="test",
    verified={"date": "2026-10-30", "how": "manual"},
)


class _Params(TaskParams):
    window: int = 3


def _task(**overrides: object) -> TaskSpec:
    fields: dict[str, Any] = {
        "name": "t.measure",
        "version": "1",
        "kind": "measure",
        "inputs": {"audio": "Audio"},
        "outputs": {"pitch": "Track"},
        "params": _Params,
    }
    fields.update(overrides)
    return TaskSpec(**fields)


def _backend(**overrides: object) -> BackendSpec:
    fields: dict[str, Any] = {
        "task": "t.measure",
        "name": "b",
        "version": "1",
        "stack": "fake",
        "env": "host",
        "devices": ("cpu",),
        "license": LICENSE,
    }
    fields.update(overrides)
    return BackendSpec(**fields)


@pytest.mark.logic
def test_duplicate_signature_refused() -> None:
    """A second task with the same kind and port types is the same operation under another name (P-37)."""
    registry = Registry()
    first = _task()
    registry.add_task(first)
    with pytest.raises(DuplicateDeclaration, match="t.measure") as info:
        registry.add_task(_task(name="t.other"))
    assert info.value.existing is first
    assert [t.name for t in registry.tasks()] == ["t.measure"]


@pytest.mark.logic
@pytest.mark.parametrize(
    "change",
    [{"kind": "transform"}, {"inputs": {"audio": "Video"}}, {"outputs": {"pitch": "Track", "voicing": "Track"}}],
    ids=["other_kind", "other_input_type", "extra_output"],
)
def test_a_different_signature_is_admitted(change: dict[str, Any]) -> None:
    """Tasks differing in kind, an input type or an output port are distinct operations."""
    registry = Registry()
    registry.add_task(_task())
    registry.add_task(_task(name="t.other", **change))
    assert [t.name for t in registry.tasks()] == ["t.measure", "t.other"]


@pytest.mark.logic
def test_duplicate_task_name_refused() -> None:
    """A task name is declared once."""
    registry = Registry()
    first = _task()
    registry.add_task(first)
    with pytest.raises(DuplicateDeclaration) as info:
        registry.add_task(_task(outputs={"other": "Track"}))
    assert info.value.existing is first


@pytest.mark.logic
def test_duplicate_backend_refused_per_task() -> None:
    """A backend name is declared once per task; the same name under another task is a different backend."""
    registry = Registry()
    registry.add_task(_task())
    registry.add_task(_task(name="t.other", kind="transform"))
    first = _backend()
    registry.add_backend(first)
    registry.add_backend(_backend(task="t.other"))
    with pytest.raises(DuplicateDeclaration, match="'b'") as info:
        registry.add_backend(_backend(version="2"))
    assert info.value.existing is first
    assert registry.backends("t.measure") == [first]


@pytest.mark.logic
def test_backend_of_an_undeclared_task_refused() -> None:
    """A backend implements a declared task; listing backends of an unknown task is an error, not empty."""
    registry = Registry()
    with pytest.raises(UnknownTask, match="t.measure"):
        registry.add_backend(_backend())
    with pytest.raises(UnknownTask):
        registry.backends("t.measure")


@pytest.mark.logic
def test_registries_are_independent() -> None:
    """A fresh registry starts empty whatever another holds."""
    a, b = Registry(), Registry()
    a.add_task(_task())
    assert b.tasks() == []


@pytest.mark.logic
def test_params_refuse_unknown_keys_with_a_suggestion() -> None:
    """`TaskParams` forbids unknown keys and names the close match."""
    with pytest.raises(ValidationError, match="'windw'.*did you mean 'window'"):
        _Params.model_validate({"windw": 4})
    with pytest.raises(ValidationError, match="'zzz'"):
        _Params.model_validate({"zzz": 4})
    assert _Params.model_validate({"window": 4}).window == 4


@pytest.mark.logic
def test_task_params_must_be_a_model() -> None:
    """`params` is a pydantic model class."""
    with pytest.raises(ValidationError):
        _task(params=dict)


@pytest.mark.logic
def test_output_affecting_names_only_key_components() -> None:
    """A backend may declare only batch formation and chunk grid output-affecting (ADR-002)."""
    assert _backend(output_affecting=frozenset({"chunk_grid"})).output_affecting == {"chunk_grid"}
    with pytest.raises(ValidationError, match="batch_size"):
        _backend(output_affecting=frozenset({"batch_size"}))


@pytest.mark.logic
def test_key_for_builds_the_logical_key() -> None:
    """The key carries the declared task and backend identities, the params' JSON and the input identities."""
    key = key_for(_task(), _backend(), params=_Params(window=5), inputs={"audio": AUDIO}, profile_slices={"p": SLICE})
    assert (key.task, key.task_version, key.backend, key.backend_version, key.model) == (
        "t.measure",
        "1",
        "b",
        "1",
        None,
    )
    assert dict(key.params) == {"window": 5}
    assert dict(key.inputs) == {"audio": AUDIO}
    assert dict(key.profile_slices) == {"p": SLICE}
    assert (key.batch_formation, key.chunk_grid) == (None, None)


class _Other(BaseModel):
    window: int = 3


@pytest.mark.logic
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"params": _Other(), "inputs": {"audio": AUDIO}}, "_Params"),
        ({"params": _Params(), "inputs": {}}, "audio"),
        ({"params": _Params(), "inputs": {"audio": AUDIO, "video": AUDIO}}, "video"),
    ],
    ids=["params_of_another_model", "missing_input", "undeclared_input"],
)
def test_key_for_refuses_a_call_off_the_declaration(kwargs: dict[str, Any], match: str) -> None:
    """Params must be the task's model, and inputs exactly its declared ports."""
    with pytest.raises(ValueError, match=match):
        key_for(_task(), _backend(), **kwargs)


@pytest.mark.logic
def test_key_for_refuses_a_backend_of_another_task() -> None:
    """The backend must implement the task."""
    with pytest.raises(ValueError, match="t.other"):
        key_for(_task(), _backend(task="t.other"), params=_Params(), inputs={"audio": AUDIO})


@pytest.mark.logic
def test_key_for_puts_the_pinned_model_identity_into_the_key() -> None:
    """`key_for` takes a resolved model and keys its identity; an unpinned one is refused."""
    model = ResolvedModel(ref=ModelRef.parse("org/name@main"), identity=COMMIT)
    key = key_for(_task(), _backend(), params=_Params(), inputs={"audio": AUDIO}, model=model)
    assert key.model == ModelIdentity(source="hf", id="org/name", identity=COMMIT)
    unpinned = ResolvedModel(ref=ModelRef.parse("local:/m"), identity="/m", unresolved_reason="not hashed")
    with pytest.raises(UnpinnedModel):
        key_for(_task(), _backend(), params=_Params(), inputs={"audio": AUDIO}, model=unpinned)


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["/home/x/a.wav", "a" * 64, SLICE])
def test_a5_key_for_refuses_untyped_inputs(bad: str) -> None:
    """An input that is a path, bare hex or another kind of identity is refused."""
    with pytest.raises(ValidationError):
        key_for(_task(), _backend(), params=_Params(), inputs={"audio": bad})


class _Colour(str, Enum):
    RED = "red"


class _Mode(Enum):
    A = "a"


class _Level(Enum):
    LOW = 1


class _Inner(TaskParams):
    n: int = 1


class _Plain(BaseModel):
    n: int = 1


_CANONICAL: list[object] = [
    int,
    float,
    str,
    bool,
    type(None),
    Literal["a", 1, None, True],
    _Colour,
    _Mode,
    list[int],
    tuple[int, ...],
    tuple[int, str],
    dict[str, float],
    Mapping[str, list[str]],
    int | None,
    Optional[str],  # noqa: UP045
    Annotated[int, Field(ge=0)],
    NonEmptyStr,
    _Inner,
]

_NON_CANONICAL: list[object] = [
    set[str],
    frozenset[int],
    Decimal,
    datetime,
    date,
    time,
    timedelta,
    Path,
    bytes,
    complex,
    Any,
    object,
    list,
    dict,
    dict[int, str],
    Literal[b"x"],
    _Level,
    _Plain,
]


def _wrappers() -> list[tuple[str, Callable[[Any], Any]]]:
    def nested(t: Any) -> Any:  # noqa: ANN401
        return create_model("_Nested", __base__=TaskParams, v=(t, ...))

    return [
        ("bare", lambda t: t),
        ("list", lambda t: list[t]),  # type: ignore[valid-type]
        ("tuple", lambda t: tuple[t, ...]),  # type: ignore[valid-type]
        ("pair", lambda t: tuple[int, t]),  # type: ignore[valid-type]
        ("dict", lambda t: dict[str, t]),  # type: ignore[valid-type]
        ("mapping", lambda t: Mapping[str, t]),  # type: ignore[valid-type]
        ("optional", lambda t: t | None),
        ("annotated", lambda t: Annotated[t, "meta"]),
        ("nested", nested),
    ]


def _params_with(annotation: object) -> type[BaseModel]:
    return create_model("_Generated", __base__=TaskParams, field=(annotation, ...))


@pytest.mark.logic
@pytest.mark.parametrize("wrap", [w for _, w in _wrappers()], ids=[n for n, _ in _wrappers()])
@pytest.mark.parametrize("annotation", _CANONICAL, ids=repr)
def test_canonical_param_types_are_admitted(annotation: object, wrap: Callable[[Any], Any]) -> None:
    """Every canonical type, in every container position, passes the registry-time check."""
    check_canonical_params(_params_with(wrap(annotation)))


@pytest.mark.logic
@pytest.mark.parametrize("wrap", [w for _, w in _wrappers()], ids=[n for n, _ in _wrappers()])
@pytest.mark.parametrize("annotation", _NON_CANONICAL, ids=repr)
def test_a6_registry_refuses_non_canonical_param_types(annotation: object, wrap: Callable[[Any], Any]) -> None:
    """A type with no single canonical JSON form is refused wherever it sits, naming the field."""
    registry = Registry()
    with pytest.raises(NonCanonicalParams, match="_Generated.field"):
        registry.add_task(_task(params=_params_with(wrap(annotation))))
    assert registry.tasks() == []
    assert issubclass(NonCanonicalParams, TypeError)
