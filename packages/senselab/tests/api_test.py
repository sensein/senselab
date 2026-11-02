"""The generated task API, `describe` and `models`: drift, calls run through `run_call`, schemas, discovery, errors."""

import importlib.util
import inspect
import json
import os
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
from senselab_core.kernel.identity import TEXT_IDENTITY_FORMS
from senselab_core.kernel.results import TextSpans
from senselab_core.kernel.signals import Audio, PlainText
from senselab_core.runtime.call import (
    Packaged,
    Result,
    TaskCall,
    bind_input,
    input_binding,
    packaged,
    run_call,
    use_packaged,
)
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
    assert document["inputs"]["properties"]["audio"]["anyOf"][1] == {"$ref": "#/$defs/Audio"}
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
            if port_type in kernel_types():
                _assert_port_schema(document["inputs"], port, input_binding(port_type) != "value")
        for side in ("inputs", "outputs"):
            assert not _bare_arrays(document[side]), f"{spec.name} {side}"
        assert "anyOf" not in json.dumps(document["outputs"]["properties"])
        assert "oneOf" not in json.dumps(document["outputs"])
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


_SPANS = {"text": "x", "length": 5, "start": [0], "end": [5], "label": ["A"]}
_ID = PlainText(content="hello").identity


@pytest.mark.logic
def test_text_spans_text_states_both_forms_from_the_rule() -> None:
    """The `text` of span inputs is described as exactly the two forms the rule accepts."""
    prop = describe("span_f1")["inputs"]["$defs"]["TextSpans"]["properties"]["text"]
    forms = [f["description"] for f in prop["oneOf"]]
    assert tuple(forms) == TEXT_IDENTITY_FORMS and all(f["examples"] for f in prop["oneOf"])


@pytest.mark.logic
def test_text_identity_given_as_content_binds_to_the_identity_of_that_text() -> None:
    """`{"content": text}` and the identity of that text bind to one identity, as does the object built in code."""
    by_text = bind_input("reference", "TextSpans", {**_SPANS, "text": {"content": "hello"}})
    by_identity = bind_input("reference", "TextSpans", {**_SPANS, "text": _ID})
    built = bind_input("reference", "TextSpans", TextSpans.model_validate({**_SPANS, "text": _ID}))
    assert by_text[0] == by_identity[0] == built[0] and getattr(by_text[1], "text") == _ID


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad",
    [
        "hello",
        _ID[:-1],
        _ID.upper(),
        "content:" + "A" * 64,
        "local:/x",
        "url:http://a",
        "output:" + "0" * 64,
        " " + _ID,
        _ID + "\n",
        {"content": "a", "extra": 1},
    ],
)
def test_a_text_identity_in_neither_form_is_refused(bad: object) -> None:
    """A text identity in neither form, `content:<64 lowercase hex>` or `{"content": text}`, is refused."""
    with pytest.raises(ValueError, match=r"field 'text' takes .*content:.*\{\"content\""):
        bind_input("reference", "TextSpans", {**_SPANS, "text": bad})


def _bare_arrays(node: object, path: str = "") -> list[str]:
    """Return the paths of array schemas that give no `items`."""
    if isinstance(node, dict):
        bare = [path] if node.get("type") == "array" and "items" not in node else []
        return bare + [b for k, v in node.items() for b in _bare_arrays(v, f"{path}/{k}")]
    if isinstance(node, list):
        return [b for i, v in enumerate(node) for b in _bare_arrays(v, f"{path}[{i}]")]
    return []


def _unresolved(node: object, defs: dict[str, Any]) -> list[str]:
    if isinstance(node, dict):
        found = [r for r in [node.get("$ref")] if isinstance(r, str) and r.rsplit("/", 1)[-1] not in defs]
        return found + [u for v in node.values() for u in _unresolved(v, defs)]
    if isinstance(node, list):
        return [u for v in node for u in _unresolved(v, defs)]
    return []


def _assert_port_schema(inputs: dict[str, Any], port: str, takes_string: bool) -> None:
    """A kernel port's schema is complete, and accepts a string where `bind_input` does."""
    schema = inputs["properties"][port]
    assert "$ref" in schema or "anyOf" in schema
    defs = inputs.get("$defs", {})
    assert not _unresolved(schema, defs)
    if takes_string:
        jsonschema.validate("a string", {**schema, "$defs": defs})


@pytest.mark.logic
def test_array_fields_state_element_rank_and_unit() -> None:
    """An array field's schema is generated from its declaration: element type, rank and unit where it has one."""
    defs = describe("der")["inputs"]["$defs"]["Intervals"]["properties"]
    assert defs["start_ns"]["items"] == {"type": "integer"} and defs["start_ns"]["x-senselab-unit"] == "ns"
    trials = describe("eer")["inputs"]["$defs"]["Trials"]["properties"]["first"]
    assert trials["x-senselab-ndim"] == 2 and trials["items"]["items"] == {"type": "number"}
    spans = describe("span_f1")["inputs"]["$defs"]["TextSpans"]["properties"]
    assert spans["start"]["x-senselab-unit"] == "characters" and spans["length"]["description"]


@pytest.mark.logic
def test_params_omission_is_stated_from_the_declaration() -> None:
    """`params` may be omitted exactly when every param has a default; otherwise the required ones are named."""
    assert "may be omitted" in describe("pii_spans")["call"]["keys"]["params"]
    assert "['target_rate'] must be given" in describe("signal.resample")["call"]["keys"]["params"]


_RUN_CALLS = importlib.util.spec_from_file_location("run_calls", ROOT / "scripts" / "run_calls.py")
assert _RUN_CALLS and _RUN_CALLS.loader


@pytest.mark.logic
def test_run_calls_reports_one_line_per_call_and_fails_on_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A good call prints `ok` with its output types, an invalid one `FAILED` with its error type; exit is 1."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path))
    module = importlib.util.module_from_spec(_RUN_CALLS)  # type: ignore[arg-type]
    sys.modules["run_calls"] = module
    _RUN_CALLS.loader.exec_module(module)  # type: ignore[union-attr]
    transcript = {"segments": [{"text": "hello world"}]}
    good = {"task": "wer", "inputs": {"reference": transcript, "hypothesis": transcript}, "params": {}}
    bad = {"task": "wer", "inputs": {}, "params": {"nope": 1}}
    for calls, code in (([good], 0), ([good, bad], 1)):
        path = tmp_path / "calls.json"
        path.write_text(json.dumps(calls))
        assert module.main([str(path)]) == code
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("wer: ok ") and lines[-1].startswith("wer: FAILED ")


@pytest.mark.logic
def test_a_fresh_process_finds_the_packaged_tasks_without_importing_senselab() -> None:
    """`packaged()` loads the `senselab.packages` entry points itself, so `run_call` needs no `import senselab`."""
    code = "from senselab_core.runtime.call import packaged; print(sorted(packaged().registry._tasks))"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)  # noqa: S603
    assert "'wer'" in run.stdout and "'speech_to_text'" in run.stdout


def _child(
    tmp_path: Path, entries: dict[str, str], script: str, files: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `script` in a fresh interpreter where `entries` (module name to source) are entry points of the group."""
    info = tmp_path / "extra-0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: extra\nVersion: 0\n")
    (info / "entry_points.txt").write_text("[senselab.packages]\n" + "".join(f"{name} = {name}\n" for name in entries))
    for name, source in {**entries, **(files or {})}.items():
        path = tmp_path / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        (path.parent / "__init__.py").touch()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120, check=False
    )  # noqa: S603


_DECLARE = (
    "from senselab_core.kernel.declare import REGISTRY, TaskParams, TaskSpec\n"
    "REGISTRY.add_task(TaskSpec(name='dup.t', version='1', kind='transform', inputs={'a': 'json'},"
    " outputs={'b': 'sha256'}, params=TaskParams))\n"
)


@pytest.mark.logic
def test_concurrent_first_calls_all_see_the_full_registry() -> None:
    """Eight threads making the first `packaged()` call all find the packaged tasks, not an empty registry."""
    code = (
        "import threading\n"
        "from senselab_core.runtime.call import packaged\n"
        "gate = threading.Barrier(8)\n"
        "seen = []\n"
        "def first():\n"
        "    gate.wait()\n"
        "    seen.append('speech_to_text' in packaged().registry._tasks)\n"
        "threads = [threading.Thread(target=first) for _ in range(8)]\n"
        "[t.start() for t in threads]\n"
        "[t.join() for t in threads]\n"
        "print(seen)\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120)  # noqa: S603
    assert run.stdout.strip() == str([True] * 8)


@pytest.mark.logic
def test_a_failed_entry_point_is_named_and_retried(tmp_path: Path) -> None:
    """A load that fails raises naming the entry point, and the next call tries again rather than going quiet."""
    script = (
        "from senselab_core.runtime.call import packaged\n"
        "for _ in range(2):\n"
        "    try:\n        packaged()\n    except Exception as e:\n        print(type(e).__name__, e)\n"
    )
    run = _child(tmp_path, {"boom": "raise ImportError('no such thing')"}, script)
    lines = run.stdout.splitlines()
    assert len(lines) == 2 and all("boom = boom" in line and "PackageLoadError" in line for line in lines)


@pytest.mark.logic
def test_two_packages_declaring_one_task_refuse_naming_both(tmp_path: Path) -> None:
    """A second entry point redeclaring a task raises naming both entry points, never overriding."""
    script = (
        "from senselab_core.runtime.call import packaged\ntry:\n    packaged()\nexcept Exception as e:\n    print(e)\n"
    )
    run = _child(tmp_path, {"pkg_a": _DECLARE, "pkg_b": _DECLARE}, script)
    assert "pkg_b = pkg_b" in run.stdout and "pkg_a = pkg_a" in run.stdout and "'dup.t'" in run.stdout


@pytest.mark.logic
def test_signal_samples_admit_null() -> None:
    """Samples of a signal may be missing, so the items of its array schema admit null."""
    group = describe("signal.resample")["inputs"]["$defs"]["Audio"]["properties"]["groups"]
    assert "null" in json.dumps(group) and '"null"]' in json.dumps(group)


@pytest.mark.logic
def test_a_load_that_fails_after_declaring_is_rolled_back_and_retried_cleanly(tmp_path: Path) -> None:
    """The failed load's declaration is undone; the retry loads once, so the registry holds each task once."""
    marker = tmp_path / "failed-once"
    entry = (
        f"import halfway.decl\nimport os\nif not os.path.exists({str(marker)!r}):\n"
        f"    open({str(marker)!r}, 'w').close()\n    raise ImportError('late failure')\n"
    )
    script = (
        "from senselab_core.runtime.call import packaged\n"
        "try:\n    packaged()\nexcept Exception as e:\n    print('first', type(e).__name__)\n"
        "names = [t.name for t in packaged().registry.tasks()]\n"
        "print('second', names.count('dup.t'), len(names) == len(set(names)))\n"
    )
    run = _child(tmp_path, {"halfway.entry": entry}, script, {"halfway.decl": _DECLARE})
    assert run.stdout.splitlines() == ["first PackageLoadError", "second 1 True"], run.stderr
