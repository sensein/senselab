"""The generated task API, `describe` and `models`: drift, calls run through `run_call`, schemas, discovery, errors."""

import importlib.util
import inspect
import json
import platform
import subprocess
import sys
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

import jsonschema
import numpy
import pytest
import senselab_core.runtime.fake_backend as fake
from pydantic import Field
from senselab._api.call import ApiError
from senselab._api.describe import JSON_FORM, PORT_FORMS, describe, models, tasks
from senselab._api.packaged import CATALOGUE
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, Registry, TaskParams, TaskSpec, kernel_types
from senselab_core.kernel.signals import Audio
from senselab_core.runtime.call import Packaged, Result, TaskCall, input_binding, packaged, run_call, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile

import senselab

_AUDIO = Audio.from_array(numpy.zeros((1, 8), dtype=numpy.float32), 16000)
ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "scripts" / "generate_task_api.py"
_spec = importlib.util.spec_from_file_location("generate_task_api", _SCRIPT)
assert _spec and _spec.loader
generator = importlib.util.module_from_spec(_spec)
sys.modules["generate_task_api"] = generator
_spec.loader.exec_module(generator)


class Colour(Enum):
    """A param enum."""

    RED = "red"
    BLUE = "blue"


class RichParams(TaskParams):
    """Params of every kind the generator renders."""

    language: str | None = None
    timestamps: Literal["word", "segment", "none"] = "word"
    colour: Colour = Colour.RED
    window: Annotated[int, Field(ge=1)] = 3
    pair: tuple[int, str] = (1, "a")
    weights: dict[str, float] = {}
    required: int


def _registry() -> Registry:
    registry = Registry()
    registry.add_task(
        TaskSpec(
            name="speech.embed",
            version="2",
            kind="transform",
            inputs={"audio": "Audio"},
            outputs={"embedding": "Embedding"},
            params=RichParams,
        )
    )
    registry.add_task(
        TaskSpec(
            name="speech.fit_space",
            version="1",
            kind="fit",
            inputs={"first": "Audio", "second": "Audio"},
            outputs={"events": "Events"},
            params=TaskParams,
        )
    )
    for name, stack, devices in (("a", "ct2", ("cuda", "cpu")), ("b", "onnx", ("cpu",))):
        registry.add_backend(
            BackendSpec(
                task="speech.embed",
                name=name,
                version="1",
                stack=stack,
                env=Env.host(),
                devices=devices,
                batch_invariant=True,
                worker="senselab_worker.fake",
                load_stub={},
            )
        )
    return registry


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """The module generated from `_registry()`, imported, its classes bound to that registry."""
    registry = _registry()
    path = tmp_path_factory.mktemp("generated") / "generated_api.py"
    path.write_text(generator.render(registry))
    spec = importlib.util.spec_from_file_location("generated_api", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["generated_api"] = module
    spec.loader.exec_module(module)
    for name in module.__all__:
        if inspect.isclass(getattr(module, name)):
            getattr(module, name).registry = registry
    return module


def _check(output: Path) -> int:
    """Run `--check` over `output` in a fresh interpreter, whose registry holds only the declared tasks."""
    done = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check", "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return done.returncode


@pytest.mark.contract
def test_generated_api_has_no_drift(tmp_path: Path) -> None:
    """The committed module is what the generator writes now; an edit to it is drift."""
    assert _check(generator.OUTPUT) == 0
    edited = tmp_path / "generated.py"
    edited.write_text(generator.OUTPUT.read_text() + "\nEXTRA = 1\n")
    assert _check(edited) == 1
    assert _check(tmp_path / "missing.py") == 1


@pytest.mark.contract
def test_generated_module_of_a_task_is_current(tmp_path: Path) -> None:
    """A module rendered from a registry with tasks is accepted for that registry, and refused once edited."""
    registry = _registry()
    output = tmp_path / "generated.py"
    output.write_text(generator.render(registry))
    assert generator.is_current(registry, output)
    output.write_text(output.read_text().replace('name = "speech.embed"', 'name = "speech.other"'))
    assert not generator.is_current(registry, output)
    assert not generator.is_current(Registry(), output)


@pytest.mark.contract
def test_generated_source_is_formatter_stable(generated: ModuleType) -> None:
    """The generated source is already in ruff's format, so a formatting hook never makes it drift."""
    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff is not installed")
    source = Path(str(generated.__file__)).read_text()
    done = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", "--stdin-filename", "generated.py", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    """Every env is the running interpreter: the declared test seam of the fake stack; nothing is built."""
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


TOY_CATALOGUE = Catalogue.load(
    """
format: catalogue@1.0
entries:
- id: toyorg/embedder
  revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  tasks: [speech.embed]
  backends: [a, b]
  stacks: [ct2, onnx]
  default_for: [speech.embed]
  license:
    expression: CC-BY-NC-4.0
    source: toy card
    verified: {date: 2026-11-01, how: hf-card}
    gated: true
"""
)


@pytest.fixture
def fake_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """`run_call` reads the toy catalogue, opens every env as this interpreter and stores under `tmp_path`."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path))
    previous = use_packaged(Packaged(TOY_CATALOGUE, _this_interpreter))
    yield
    use_packaged(previous)


def _echo_namespace() -> dict[str, Any]:
    registry = Registry()
    registry.add_task(fake.ECHO)
    namespace: dict[str, Any] = {}
    exec(compile(generator.render(registry), "generated_echo", "exec"), namespace)  # noqa: S102 - generated here
    return namespace


@pytest.mark.logic
def test_function_equals_configured_call(fake_stack: None) -> None:  # noqa: ARG001 - the fixture's effect
    """`echo(x, **params)` is exactly `Echo(**params)(x)`, which is `run_call` of `Echo(**params).call(x)`."""
    namespace = _echo_namespace()
    echo, cls = namespace["echo"], namespace["Echo"]
    configured = cls(salt="s")
    assert isinstance(configured.call({"x": 1}), TaskCall)
    result = echo({"x": 1}, salt="s")
    assert isinstance(result, Result)
    for again in (configured({"x": 1}), run_call(configured.call({"x": 1}))):
        assert (again.outputs, again.output_identity, again.key) == (result.outputs, result.output_identity, result.key)
    assert result.output_identity != echo({"x": 1}, salt="t").output_identity
    assert configured.call({"x": 1}).params.salt == "s"


@pytest.mark.logic
def test_call_from_describe_alone(fake_stack: None) -> None:  # noqa: ARG001 - the fixture's effect
    """A call written only from `describe()` JSON validates as a `TaskCall` and runs on the fake stack."""
    document = json.loads(json.dumps(describe(fake.ECHO.name)))
    params = {name: field["default"] for name, field in document["params"]["properties"].items()}
    call = {
        "task": document["task"],
        "inputs": {port: {"text": "hello"} for port in document["inputs"]["required"]},
        "params": params,
    }
    result = run_call(TaskCall.model_validate(json.loads(json.dumps(call))))
    assert set(result.outputs) == set(document["outputs"]["required"])
    assert result.placement.backend.stack in document["stacks"]


@pytest.mark.logic
def test_describe_lists_catalogue_models() -> None:
    """`describe` lists the models serving the task with their licenses, and the catalogue default."""
    document = describe("speech.embed", _registry(), TOY_CATALOGUE)
    [model] = document["models"]
    assert (model["id"], model["revision"], model["backends"]) == ("toyorg/embedder", "b" * 40, ["a", "b"])
    assert model["license"]["expression"] == "CC-BY-NC-4.0" and model["license"]["commercial_use"] == "no"
    assert document["default_model"] == "toyorg/embedder@" + "b" * 40
    fit = describe("speech.fit_space", _registry(), TOY_CATALOGUE)
    assert (fit["models"], fit["default_model"]) == ([], None)
    assert models(catalogue=TOY_CATALOGUE) == [model] and models("speech.fit_space", TOY_CATALOGUE) == []


@pytest.mark.logic
def test_packaged_catalogue_is_what_run_call_and_models_read() -> None:
    """Importing `senselab` hands `run_call` the packaged catalogue, which `models()` lists."""
    assert packaged().catalogue is CATALOGUE
    assert [m["id"] for m in models("speech_to_text")] == [
        e.id for e in CATALOGUE.entries if "speech_to_text" in e.tasks
    ]


@pytest.mark.logic
def test_classes_are_immutable_and_with_params_returns_a_new_one(generated: ModuleType) -> None:
    """Objects never mutate; `with_params` changes only the params it names."""
    embed = generated.Embed(model="org/name@main", required=1, language="en")
    other = embed.with_params(language="es")
    assert (embed.params.language, other.params.language) == ("en", "es")
    assert other.params.required == 1 and str(other.model) == "org/name@main"
    with pytest.raises(AttributeError):
        embed.language = "x"
    assert generated.Embed(required=1).call(_AUDIO).inputs["audio"] is _AUDIO


@pytest.mark.logic
def test_signatures_are_real(generated: ModuleType) -> None:
    """Class, `call` and function carry the declared params and ports as typed keywords."""
    init = inspect.signature(generated.Embed.__init__).parameters
    assert {"model", "language", "timestamps", "colour", "window", "pair", "required", "backend_params"} <= set(init)
    assert init["window"].default == 3 and init["required"].default is inspect.Parameter.empty
    assert init["colour"].default is Colour.RED and init["pair"].default == (1, "a")
    assert init["required"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="windw"):
        generated.Embed(required=1, windw=3)
    assert list(inspect.signature(generated.Embed.call).parameters) == ["self", "audio"]
    assert list(inspect.signature(generated.Embed.__call__).parameters) == ["self", "audio"]
    assert list(inspect.signature(generated.FitSpace.call).parameters) == ["self", "first", "second"]
    assert list(inspect.signature(generated.embed).parameters)[0] == "audio"


@pytest.mark.logic
def test_non_kernel_ports_take_a_json_value() -> None:
    """A port whose type is not a kernel type is described as a JSON value, and is not linked."""
    registry = Registry()
    registry.add_task(fake.ECHO)
    port = describe("fake.echo", registry)["inputs"]["properties"]["payload"]
    assert "$ref" not in port and port["description"] == "a JSON value"


@pytest.mark.logic
def test_describe_is_valid_schema() -> None:
    """Each schema of `describe` is valid 2020-12; a kernel-typed port links to its type's format schema."""
    document = describe("speech.embed", _registry())
    assert (document["task"], document["version"], document["kind"]) == ("speech.embed", "2", "transform")
    schemas = [document[k] for k in ("inputs", "params", "outputs")] + list(document["backend_params"].values())
    for schema in schemas:
        jsonschema.Draft202012Validator.check_schema(schema)
    assert document["inputs"]["properties"]["audio"]["$ref"] == "#/$defs/Audio"
    assert "groups" in document["inputs"]["$defs"]["Audio"]["properties"]
    embedding = document["outputs"]["$defs"]["Embedding"]
    assert embedding["x-senselab-format"] == "embedding@1.0" and embedding["properties"]["vector"]["type"] == "array"
    assert document["callable"] == {"class": "Embed", "function": "embed"}
    assert describe("speech.fit_space", _registry())["callable"] == {"class": "FitSpace", "function": "fit_space"}
    assert [b["name"] for b in document["backends"]] == ["a", "b"]
    assert document["stacks"] == ["ct2", "onnx"] and document["devices"] == ["cpu", "cuda"]
    params = jsonschema.Draft202012Validator(document["params"])
    assert params.is_valid({"required": 1}) and not params.is_valid({"required": 1, "windw": 3})


@pytest.mark.logic
def test_tasks_filters_by_accepted_type() -> None:
    """`tasks(accepts=T)` lists the tasks with an input port of type T, all of them without it."""
    registry = _registry()
    assert [t["name"] for t in tasks(registry=registry)] == ["speech.embed", "speech.fit_space"]
    assert [t["name"] for t in tasks("Audio", registry)] == ["speech.embed", "speech.fit_space"]
    assert tasks("Embedding", registry) == []
    assert tasks("Audio", registry)[1] == {
        "name": "speech.fit_space",
        "version": "1",
        "kind": "fit",
        "inputs": {"first": "Audio", "second": "Audio"},
        "outputs": {"events": "Events"},
    }


@pytest.mark.logic
def test_error_envelope_shape(generated: ModuleType) -> None:
    """Every refusal is one `{code, message, fix, field, suggestions}` envelope."""
    keys = {"code", "message", "fix", "field", "suggestions"}
    with pytest.raises(ApiError) as unknown_task:
        describe("speech.embd", _registry())
    assert set(unknown_task.value.to_dict()) == keys
    assert unknown_task.value.to_dict()["suggestions"] == ["speech.embed"]
    with pytest.raises(ApiError) as unknown_param:
        generated.Embed.from_mapping(params={"required": 1, "windw": 3})
    assert unknown_param.value.to_dict() == {
        **unknown_param.value.to_dict(),
        "code": "unknown_param",
        "field": "windw",
        "suggestions": ["window"],
    }
    with pytest.raises(ApiError) as invalid:
        generated.Embed.from_mapping(params={"required": 1, "window": 0})
    assert (invalid.value.envelope.code, invalid.value.envelope.field) == ("invalid_param", "window")
    with pytest.raises(ApiError) as unknown_type:
        tasks("Audi", _registry())
    assert unknown_type.value.envelope.suggestions == ("Audio",)
    with pytest.raises(ApiError) as bad_model:
        generated.Embed(model="no-revision", required=1)
    assert bad_model.value.envelope.field == "model"
    with pytest.raises(ApiError) as bad_input:
        generated.Embed(required=1).call(3)
    assert bad_input.value.envelope.code == "wrong_input_type"
    assert set(bad_input.value.to_dict()) == keys


@pytest.mark.logic
def test_facade_exposes_discovery_and_the_generated_names() -> None:
    """`sl.tasks`, `sl.describe` and `sl.models` are the discovery functions; `__all__` adds the generated names."""
    from senselab._api import generated  # noqa: PLC0415

    assert senselab.tasks is tasks and senselab.describe is describe and senselab.models is models
    assert senselab.__all__ == ["describe", "models", "tasks", *generated.__all__]


@pytest.mark.logic
def test_describe_of_every_task_is_enough_to_write_a_call() -> None:
    """Every registered task's describe states each input's form, the call envelope and a param example."""
    for spec in REGISTRY.tasks():
        document = describe(spec.name)
        for port, port_type in spec.inputs.items():
            form = document["inputs"]["properties"][port]["description"]
            expected = PORT_FORMS[input_binding(port_type)].format(type=port_type)
            assert form == (expected if port_type in kernel_types() else JSON_FORM)
            if input_binding(port_type) == "path":
                assert "a path to a local file" in form
        call = document["call"]
        assert set(call["keys"]) == {"task", "task_version", "inputs", "params", "model", "backend_params"}
        assert call["model"]["allowed"] == [f"{m['id']}@{m['revision']}" for m in document["models"]]
        assert call["backend"]["allowed"] == [b["name"] for b in document["backends"]]
        assert set(call["example"]) == {"task", "inputs", "params", "model", "backend_params"}
        for name in document["params"].get("required", ()):
            prop = document["params"]["properties"][name]
            assert prop.get("description") and prop.get("examples"), f"{spec.name}.{name}"
        if not document["params"].get("required"):
            TaskCall.model_validate(call["example"])


@pytest.mark.logic
def test_text_spans_text_says_how_to_get_its_identity() -> None:
    """The `text` of span inputs names the expression that yields it."""
    schema = describe("span_f1")["inputs"]["$defs"]["TextSpans"]["properties"]["text"]
    assert "PlainText(content=" in schema["description"] and schema["examples"]
