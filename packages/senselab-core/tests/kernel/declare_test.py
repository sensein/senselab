"""Declaration registry: task and backend specs, refusal of duplicates (P-37), the logical key builder."""

from collections.abc import Callable, Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Optional

import jsonschema
import pytest
from pydantic import BaseModel, Field, ValidationError, create_model
from senselab_core.kernel._base import NonEmptyStr
from senselab_core.kernel.declare import (
    SCHEMA_DIALECT,
    BackendParams,
    BackendSpec,
    DuplicateDeclaration,
    Env,
    InconsistentInputs,
    InputsAgree,
    NonCanonicalParams,
    ParamsOrdered,
    Registry,
    ServerConfig,
    TaskParams,
    TaskSpec,
    UnknownTask,
    backend,
    check_canonical_params,
    json_schema,
    kernel_types,
    key_for,
    task,
)
from senselab_core.kernel.identity import LogicalKey, ModelIdentity, ModelRef, ResolvedModel, UnpinnedModel
from senselab_core.kernel.results import Annotation, Embedding, Metric, TextSpans
from senselab_core.kernel.signals import Audio, RegularSeries, Text

COMMIT = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
AUDIO = "content:" + "a" * 64
SLICE = "profile:" + "b" * 64


class _Params(TaskParams):
    window: int = 3


class _Bounds(TaskParams):
    lo: int | None = None
    hi: int | None = None


def _task(**overrides: object) -> TaskSpec:
    fields: dict[str, Any] = {
        "name": "t.measure",
        "version": "1",
        "kind": "transform",
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
        "env": Env.host(),
        "devices": ("cpu",),
        "batch_invariant": True,
        "worker": "senselab_worker.fake",
        "load_stub": {},
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
    [{"kind": "fit"}, {"inputs": {"audio": "Video"}}, {"outputs": {"pitch": "Track", "voicing": "Track"}}],
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
    registry.add_task(_task(name="t.other", kind="fit"))
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
@pytest.mark.parametrize("model", [_Params, create_model("_Beam", __base__=BackendParams, window=(int, 3))])
def test_unknown_param_suggests(model: type[BaseModel]) -> None:
    """Task and backend params forbid unknown keys and name the close match, or list the known keys."""
    with pytest.raises(ValidationError, match="'windw'.*did you mean 'window'"):
        model.model_validate({"windw": 4})
    with pytest.raises(ValidationError, match=r"'zzz'.*known params: \['window'\]"):
        model.model_validate({"zzz": 4})
    assert model.model_validate({"window": 4}).window == 4  # type: ignore[attr-defined]


@pytest.mark.logic
@pytest.mark.parametrize("params", [dict, BaseModel, BackendParams])
def test_task_params_must_be_task_params(params: type) -> None:
    """A task's `params` is a `TaskParams` subclass."""
    with pytest.raises(ValidationError):
        _task(params=params)


@pytest.mark.logic
@pytest.mark.parametrize("params", [dict, BaseModel, TaskParams])
def test_backend_params_must_be_backend_params(params: type) -> None:
    """A backend's `params` is a `BackendParams` subclass; it defaults to `BackendParams` itself."""
    assert _backend().params is BackendParams
    with pytest.raises(ValidationError):
        _backend(params=params)


class _Beam(BackendParams):
    beam: int = 1


class _Chunk(BackendParams):
    chunk_s: float = 30.0


@pytest.mark.logic
def test_backend_params_validated_per_backend() -> None:
    """Each backend validates its own params; a knob of another backend of the task is refused."""
    registry = Registry()
    registry.add_task(_task())
    registry.add_backend(_backend(name="beam", params=_Beam))
    registry.add_backend(_backend(name="chunk", params=_Chunk))
    beam, chunk = registry.backends("t.measure")
    assert beam.params.model_validate({"beam": 4}) == _Beam(beam=4)
    with pytest.raises(ValidationError, match="'beam'.*known params: \\['chunk_s'\\]"):
        chunk.params.model_validate({"beam": 4})
    with pytest.raises(NonCanonicalParams, match="_Generated.field"):
        registry.add_backend(
            _backend(name="bad", params=create_model("_Generated", __base__=BackendParams, field=(set[str], ...)))
        )
    assert [b.name for b in registry.backends("t.measure")] == ["beam", "chunk"]


@pytest.mark.logic
def test_server_backend_declares_its_engine_configuration_and_no_worker_sources() -> None:
    """`server` is declared by an `Env.server` backend and by no other, and a server backend places no sources."""
    server = ServerConfig(settings={"max_model_len": 1}, startup_timeout_s=5)
    assert _backend(env=Env.server("e"), server=server).server == server
    with pytest.raises(ValidationError, match="declares `server`"):
        _backend(env=Env.server("e"))
    with pytest.raises(ValidationError, match="declares `server`"):
        _backend(env=Env.isolated("e"), server=server)
    with pytest.raises(ValidationError, match="no worker sources"):
        _backend(
            env=Env.server("e"),
            server=server,
            worker_sources={"senselab_worker.fake": "x.py"},
            worker="senselab_worker.fake",
        )
    with pytest.raises(ValidationError):
        ServerConfig(settings={}, startup_timeout_s=0)


@pytest.mark.logic
def test_env_kinds_replace_free_string() -> None:
    """`env` is an `Env` of kind host, isolated or server; a free string is refused."""
    assert (Env.host().kind, Env.host().name) == ("host", None)
    assert (Env.isolated("crisperwhisper").kind, Env.isolated("crisperwhisper").name) == ("isolated", "crisperwhisper")
    assert (Env.server("vllm").kind, Env.server("vllm").name) == ("server", "vllm")
    assert _backend(env=Env.isolated("e")).env == Env.isolated("e")
    for free in ("host", "isolated", "e"):
        with pytest.raises(ValidationError):
            _backend(env=free)
    for bad in (
        {"kind": "host", "name": "e"},
        {"kind": "isolated"},
        {"kind": "server", "name": None},
        {"kind": "venv", "name": "e"},
    ):
        with pytest.raises(ValidationError):
            Env.model_validate(bad)
    for name in ("", "../e", "a/b", "E", "-e"):
        with pytest.raises(ValidationError):
            Env.isolated(name)


def _declare_backend(registry: Registry, **overrides: object) -> type:
    fields: dict[str, Any] = {
        "name": "b",
        "version": "1",
        "task": "t.measure",
        "stack": "fake",
        "env": Env.host(),
        "devices": ["cpu"],
        "batch_invariant": True,
        "worker": "senselab_worker.fake",
        "load_stub": {},
        "registry": registry,
    }
    fields.update(overrides)
    fields = {k: v for k, v in fields.items() if v is not _ABSENT}

    @backend(**fields)
    class _B:
        pass

    return _B


_ABSENT = object()


@pytest.mark.logic
@pytest.mark.parametrize("missing", ["name", "version"])
def test_backend_requires_name_and_version(missing: str) -> None:
    """`@backend` and `BackendSpec` have no default name or version."""
    registry = Registry()
    registry.add_task(_task())
    with pytest.raises(TypeError, match=missing):
        _declare_backend(registry, **{missing: _ABSENT})
    with pytest.raises(ValidationError, match=missing):
        BackendSpec(**{k: v for k, v in _backend().model_dump().items() if k != missing})
    assert registry.backends("t.measure") == []


@pytest.mark.logic
def test_backend_decorator_declares_its_spec() -> None:
    """`@backend` registers the spec it builds and sets it on the class."""
    registry = Registry()
    registry.add_task(_task())
    cls = _declare_backend(registry, models=["org/m"], tunables={"dtype": ["float16", "int8"]}, params=_Beam)
    assert registry.backends("t.measure") == [cls.spec]  # type: ignore[attr-defined]
    assert cls.spec.models == ("org/m",)  # type: ignore[attr-defined]
    assert dict(cls.spec.tunables) == {"dtype": ("float16", "int8")}  # type: ignore[attr-defined]
    assert cls.spec.params is _Beam  # type: ignore[attr-defined]


@pytest.mark.logic
@pytest.mark.parametrize(
    "change,match",
    [
        ({"tunables": {"dtype": []}}, "dtype"),
        ({"worker": "senselab_worker/fake"}, "worker"),
        ({"output_affecting": ["batch_formation"]}, "batch-invariant"),
    ],
    ids=["empty_tunable", "worker_not_a_module", "invariant_yet_batch_affecting"],
)
def test_backend_declaration_refused(change: dict[str, Any], match: str) -> None:
    """A tunable with no values, a worker that is not a module name, and contradictory batching are refused."""
    registry = Registry()
    registry.add_task(_task())
    with pytest.raises(ValidationError, match=match):
        _declare_backend(registry, **change)
    assert registry.backends("t.measure") == []


@pytest.mark.logic
def test_task_decorator_declares_its_spec() -> None:
    """`@task` reads `Params`, `inputs` and `outputs` from the class, registers the spec and sets it on the class."""
    registry = Registry()

    @task(name="t.embed", version="2", kind="transform", registry=registry)
    class _Embed:
        Params = _Params
        inputs = {"audio": Audio}
        outputs = {"embedding": Embedding}

    @task(name="t.label", version="1", kind="transform", registry=registry)
    class _Label:
        inputs = {"audio": Audio}
        outputs = {"label": Annotation}

    assert registry.tasks() == [_Embed.spec, _Label.spec]  # type: ignore[attr-defined]
    assert _Embed.spec.signature() == ("transform", {("audio", "Audio")}, {("embedding", "Embedding")})  # type: ignore[attr-defined]
    assert (_Embed.spec.params, _Label.spec.params) == (_Params, TaskParams)  # type: ignore[attr-defined]


@pytest.mark.logic
def test_task_ports_may_name_protocols() -> None:
    """A port may name a port protocol, which `kernel_types` lists by name."""
    registry = Registry()

    @task(name="t.pii", version="1", kind="transform", registry=registry)
    class _Pii:
        inputs = {"text": Text, "series": RegularSeries}
        outputs: dict[str, type] = {"spans": TextSpans, "metric": Metric}

    assert _Pii.spec.inputs == {"text": "Text", "series": "RegularSeries"}  # type: ignore[attr-defined]
    assert {"Text", "RegularSeries"} <= kernel_types().keys()


@pytest.mark.logic
@pytest.mark.parametrize(
    "ports",
    [
        {"inputs": {"audio": "Audio"}},
        {"inputs": {"audio": int}},
        {"inputs": None},
        {"inputs": {"env": Env}},
        {"inputs": {"spec": BackendSpec}},
        {"inputs": {"spec": TaskSpec}},
        {"inputs": {"key": LogicalKey}},
    ],
    ids=["name", "plain", "missing", "env", "backend_spec", "task_spec", "logical_key"],
)
def test_task_ports_are_kernel_value_types(ports: dict[str, Any]) -> None:
    """Ports map to kernel value types; a type name, a plain type, a declaration or a key is refused."""
    registry = Registry()
    body = {"outputs": {"e": Embedding}, **{k: v for k, v in ports.items() if v is not None}}
    with pytest.raises(TypeError, match="inputs"):
        task(name="t", version="1", kind="transform", registry=registry)(type("_T", (), body))
    assert registry.tasks() == []


@pytest.mark.logic
def test_output_affecting_names_only_key_components() -> None:
    """A backend may declare only batch formation and chunk grid output-affecting (ADR-002)."""
    assert _backend(output_affecting=frozenset({"chunk_grid"})).output_affecting == ("chunk_grid",)
    both = ("chunk_grid", "batch_formation", "chunk_grid")
    assert _backend(output_affecting=both, batch_invariant=False).output_affecting == ("batch_formation", "chunk_grid")
    with pytest.raises(ValidationError, match="batch_size"):
        _backend(output_affecting=frozenset({"batch_size"}))


@pytest.mark.logic
def test_key_for_builds_the_logical_key() -> None:
    """The key carries the declared task and backend identities, the params' JSON and the input identities."""
    key = key_for(
        _task(),
        _backend(),
        backend_params=BackendParams(),
        params=_Params(window=5),
        inputs={"audio": AUDIO},
        profile_slices={"p": SLICE},
    )
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
        key_for(_task(), _backend(), backend_params=BackendParams(), **kwargs)


@pytest.mark.logic
def test_key_for_keys_backend_params() -> None:
    """Backend params enter the key; params of another backend's model are refused."""
    backend = _backend(params=_Beam)
    call: dict[str, Any] = {"params": _Params(), "inputs": {"audio": AUDIO}}
    default = key_for(_task(), backend, backend_params=_Beam(), **call)
    assert dict(default.backend_params) == {"beam": 1}
    assert key_for(_task(), backend, backend_params=_Beam(beam=2), **call).digest() != default.digest()
    with pytest.raises(ValueError, match="_Beam"):
        key_for(_task(), backend, backend_params=BackendParams(), **call)


@pytest.mark.logic
@pytest.mark.parametrize(
    "declared,given,match",
    [((), {"chunk_grid": "g"}, "does not declare chunk_grid"), (("chunk_grid",), {}, "declares chunk_grid")],
    ids=["undeclared_given", "declared_missing"],
)
def test_key_for_takes_an_output_affecting_component_exactly_when_declared(
    declared: tuple[str, ...], given: dict[str, Any], match: str
) -> None:
    """A batch formation or chunk grid is keyed only when declared, and must be given when it is."""
    with pytest.raises(ValueError, match=match):
        key_for(
            _task(),
            _backend(output_affecting=declared),
            backend_params=BackendParams(),
            params=_Params(),
            inputs={"audio": AUDIO},
            **given,
        )


@pytest.mark.logic
def test_key_for_refuses_a_backend_of_another_task() -> None:
    """The backend must implement the task."""
    with pytest.raises(ValueError, match="t.other"):
        key_for(
            _task(), _backend(task="t.other"), backend_params=BackendParams(), params=_Params(), inputs={"audio": AUDIO}
        )


@pytest.mark.logic
def test_key_for_puts_the_pinned_model_identity_into_the_key() -> None:
    """`key_for` takes a resolved model and keys its identity; an unpinned one is refused."""
    model = ResolvedModel(ref=ModelRef.parse("org/name@main"), identity=COMMIT)
    key = key_for(
        _task(), _backend(), backend_params=BackendParams(), params=_Params(), inputs={"audio": AUDIO}, model=model
    )
    assert key.model == ModelIdentity(id="org/name", identity=COMMIT)
    unpinned = ResolvedModel(ref=ModelRef.parse("local:/m"), identity="/m", unresolved_reason="not hashed")
    with pytest.raises(UnpinnedModel):
        key_for(
            _task(),
            _backend(),
            backend_params=BackendParams(),
            params=_Params(),
            inputs={"audio": AUDIO},
            model=unpinned,
        )


@pytest.mark.logic
@pytest.mark.parametrize("bad", ["/home/user/a.wav", "a" * 64, SLICE])
def test_key_for_refuses_untyped_inputs(bad: str) -> None:
    """An input that is a path, bare hex or another kind of identity is refused."""
    with pytest.raises(ValidationError):
        key_for(_task(), _backend(), backend_params=BackendParams(), params=_Params(), inputs={"audio": bad})


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
def test_registry_refuses_non_canonical_param_types(annotation: object, wrap: Callable[[Any], Any]) -> None:
    """A type with no single canonical JSON form is refused wherever it sits, naming the field."""
    registry = Registry()
    with pytest.raises(NonCanonicalParams, match="_Generated.field"):
        registry.add_task(_task(params=_params_with(wrap(annotation))))
    assert registry.tasks() == []
    assert issubclass(NonCanonicalParams, TypeError)


class _Rich(TaskParams):
    language: str | None = None
    timestamps: Literal["word", "segment", "none"] = "word"
    colour: _Colour = _Colour.RED
    window: Annotated[int, Field(ge=1)] = 3
    pair: tuple[int, str] = (1, "a")
    weights: dict[str, float] = {}
    inner: _Inner = _Inner()


def _schemas(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [document["inputs"], document["params"], document["outputs"], *document["backend_params"].values()]


@pytest.mark.contract
def test_schema_validates_2020_12() -> None:
    """Every generated schema names and validates under the 2020-12 metaschema, and checks instances as declared."""
    from senselab_core.runtime.fake_backend import ECHO  # noqa: PLC0415 - importing it registers fake.echo

    registry = Registry()
    spec = _task(name="t.rich", params=_Rich, inputs={"audio": "Audio", "other": "Audio"})
    registry.add_task(spec)
    registry.add_backend(_backend(task="t.rich", name="beam", params=_Beam))
    registry.add_backend(_backend(task="t.rich", name="plain"))
    documents = [json_schema(spec, registry), json_schema(ECHO)]
    assert set(documents[0]["backend_params"]) == {"beam", "plain"}
    for schema in [s for d in documents for s in _schemas(d)]:
        assert schema["$schema"] == SCHEMA_DIALECT == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(schema)
    params = jsonschema.Draft202012Validator(documents[0]["params"])
    assert params.is_valid(_Rich().model_dump(mode="json"))
    assert not params.is_valid({"timestamps": "char"})
    assert not params.is_valid({"windw": 3})
    beam = jsonschema.Draft202012Validator(documents[0]["backend_params"]["beam"])
    assert beam.is_valid({"beam": 2}) and not beam.is_valid({"chunk_s": 1.0})
    inputs = jsonschema.Draft202012Validator(documents[0]["inputs"])
    assert documents[0]["inputs"]["properties"] == {"audio": {"title": "Audio"}, "other": {"title": "Audio"}}
    assert not inputs.is_valid({"audio": "x"}) and not inputs.is_valid({"audio": "x", "other": "y", "z": "w"})


@pytest.mark.logic
def test_constraints_name_declared_ports_and_params_and_are_checked_as_data() -> None:
    """A constraint naming no declared port or param is refused; the declared ones refuse what contradicts them."""
    with pytest.raises(ValidationError, match="does not declare"):
        _task(constraints=(InputsAgree(ports=("audio", "other"), property="clock_id"),))
    with pytest.raises(ValidationError, match="does not declare"):
        _task(constraints=(ParamsOrdered(low="window", high="missing"),))
    spec = _task(
        inputs={"audio": "Audio", "other": "Audio"},
        params=_Bounds,
        constraints=(InputsAgree(ports=("audio", "other"), property="clock_id"), ParamsOrdered(low="lo", high="hi")),
    )
    spec.check_inputs({"audio": SimpleNamespace(clock_id="a"), "other": SimpleNamespace(clock_id="a")})
    with pytest.raises(InconsistentInputs, match="differ in clock_id"):
        spec.check_inputs({"audio": SimpleNamespace(clock_id="a"), "other": SimpleNamespace(clock_id="b")})
    spec.check_params(_Bounds(lo=1, hi=2))
    spec.check_params(_Bounds(lo=3))
    with pytest.raises(InconsistentInputs, match="exceeds"):
        spec.check_params(_Bounds(lo=3, hi=2))
    assert TaskSpec.model_validate(spec.model_dump()) == spec
