"""`tasks`, `describe` and `models`: discovery over the declarations and the catalogue."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from senselab_core.kernel.declare import REGISTRY, Registry, TaskSpec, json_schema, kernel_types
from senselab_core.kernel.formats import Formatted
from senselab_core.runtime.call import input_binding, input_model, packaged
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry, UnknownModel

from senselab._api.call import ApiError, callable_names, closest, task_spec


class _KernelSchema(GenerateJsonSchema):
    """JSON Schema of a kernel type: an array field is an `array`."""

    def is_instance_schema(self, schema: Any) -> dict[str, Any]:  # noqa: ANN401, ARG002 - pydantic's hook signature
        return {"type": "array"}


def _format_schema(kernel_type: type[BaseModel]) -> dict[str, Any]:
    schema = kernel_type.model_json_schema(schema_generator=_KernelSchema, ref_template="#/$defs/{model}")
    if issubclass(kernel_type, Formatted):
        schema["x-senselab-format"] = kernel_type.model_fields["format"].default
    return schema


PORT_FORMS = {
    "path": "a path to a local file, read when the call runs, or the {type} object",
    "text": "the text itself as a string (not a path), or a PlainText",
    "value": "the {type} object as its format schema gives it; a string is not read as a path",
}
STRING_FORMS = {"path": "path to a local file", "text": "the text itself"}
JSON_FORM = "a JSON value"


def _link_ports(ports: dict[str, Any], declared: Mapping[str, str]) -> dict[str, Any]:
    """Return `ports` with each port saying how a call gives it, and each kernel-typed port linked to its schema.

    How a port is given is `input_binding` of its type, in `PORT_FORMS`.
    """
    defs: dict[str, Any] = {}
    properties = {}
    for port, prop in ports["properties"].items():
        model = input_model(declared[port])
        if model is None:
            properties[port] = {**prop, "description": JSON_FORM}
            continue
        schema = _format_schema(model)
        defs.update(schema.pop("$defs", {}))
        defs[model.__name__] = schema
        form = PORT_FORMS[input_binding(declared[port])].format(type=declared[port])
        by_str = STRING_FORMS.get(input_binding(declared[port]))
        ref = {"$ref": f"#/$defs/{model.__name__}"}
        properties[port] = (
            {**prop, "description": form, "anyOf": [{"type": "string", "description": by_str}, ref]}
            if by_str
            else {**prop, "description": form, **ref}
        )
    return {**ports, "properties": properties, **({"$defs": defs} if defs else {})}


def _example(spec: TaskSpec, params: dict[str, Any], default_model: str | None) -> dict[str, Any]:
    """Return a call of `spec` with a placeholder per input and an example per required param."""
    placeholders = {"path": "<path>", "text": "<text>", "value": "<object>"}
    required = {
        n: p.get("examples", ["<example>"])[0]
        for n, p in params["properties"].items()
        if n in params.get("required", ())
    }
    return {
        "task": spec.name,
        "inputs": {port: placeholders[input_binding(t)] for port, t in spec.inputs.items()},
        "params": required,
        "model": default_model,
        "backend_params": {},
    }


def _envelope(
    spec: TaskSpec, schema: dict[str, Any], backends: list[Any], model_ids: list[str], default_model: str | None
) -> dict[str, Any]:
    """Return the call envelope: its keys, how a call names a model and a backend, and what this task allows."""
    return {
        "keys": {
            "task": f"the task name, {spec.name!r}",
            "task_version": f"optional; the declared version, {spec.version!r}",
            "inputs": "one value per input port, as each port's description says",
            "params": "the `params` schema; omitted params take their defaults",
            "model": "optional; `org/name@revision`, `url:<url>#sha256=<hex>` or `local:<path>`; None or omitted takes "
            "`default_model`",
            "backend_params": "optional; the params of the chosen backend, per `backend_params`; default {}",
        },
        "model": {"allowed": model_ids, "default": default_model},
        "backend": {
            "named_by": "no call key: the backend is the first, in the execution profile's stack preference for the "
            "task, that serves the model",
            "allowed": [b.name for b in backends],
            "stacks": sorted({b.stack for b in backends}),
        },
        "example": _example(spec, schema["params"], default_model),
    }


def tasks(accepts: str | None = None, registry: Registry = REGISTRY) -> list[dict[str, Any]]:
    """Return the declared tasks as `{name, version, kind, inputs, outputs}`.

    With `accepts`, only the tasks with an input port of that type.

    Raises:
        ApiError: `unknown_type`, with the closest type names, if `accepts` names no kernel type or declared port type.
    """
    declared = registry.tasks()
    if accepts is not None:
        known = set(kernel_types()) | {t for s in declared for t in (*s.inputs.values(), *s.outputs.values())}
        if accepts not in known:
            raise ApiError(
                "unknown_type",
                f"no port type {accepts!r} is known",
                "name a kernel type such as `Audio`",
                field="accepts",
                suggestions=closest(accepts, known),
            )
        declared = [s for s in declared if accepts in s.inputs.values()]
    return [_summary(s) for s in declared]


def _summary(spec: TaskSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "version": spec.version,
        "kind": spec.kind,
        "inputs": dict(spec.inputs),
        "outputs": dict(spec.outputs),
    }


def _model(entry: CatalogueEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "revision": entry.revision,
        "tasks": list(entry.tasks),
        "backends": list(entry.backends),
        "stacks": list(entry.stacks),
        "license": entry.license.model_dump(mode="json"),
    }


def models(task: str | None = None, catalogue: Catalogue | None = None) -> list[dict[str, Any]]:
    """Return the catalogue's models as `{id, revision, tasks, backends, stacks, license}`, in catalogue order.

    With `task`, only the models serving it. `catalogue` is the one `run_call` reads when None.
    """
    entries = (catalogue or packaged().catalogue).entries
    return [_model(e) for e in entries if task is None or task in e.tasks]


def describe(name: str, registry: Registry = REGISTRY, catalogue: Catalogue | None = None) -> dict[str, Any]:
    """Return the JSON Schema of task `name` with its backends, stacks, devices and models.

    The result holds `task`, `version`, `kind`, `call` (the call envelope), the schemas `inputs`, `params`,
    `backend_params` and `outputs` (each port says how a call gives it and, of a kernel type, links to that type's
    format schema in its `$defs`), `backends` as `{name, version, stack, devices}`, the sorted `stacks` and `devices`
    they cover, `callable`, the generated `{class, function}` names, `models` (as `models(name)` gives them, licenses
    included) and `default_model`, the catalogue default as `id@revision` (None for a task with none). `catalogue` is
    the one `run_call` reads when None.

    The function and the class run the task on the inputs, given as positional data in the form each port's
    description states (an object, a file path or the text itself; a port that is not a kernel type takes a JSON
    value). The one call shape of
    the CLI and JSON, `{task, inputs, params, model, backend_params}`, validates as a `TaskCall` and runs with
    `run_call`.

    Raises:
        ApiError: `unknown_task`, with the closest declared names.
    """
    spec = task_spec(name, registry)
    schema = json_schema(spec, registry)
    backends = registry.backends(name)
    model_list = models(name, catalogue)
    default = _default_model(name, catalogue or packaged().catalogue)
    return {
        "task": spec.name,
        "version": spec.version,
        "kind": spec.kind,
        "call": _envelope(spec, schema, backends, [f"{m['id']}@{m['revision']}" for m in model_list], default),
        **schema,
        "inputs": _link_ports(schema["inputs"], spec.inputs),
        "outputs": _link_ports(schema["outputs"], spec.outputs),
        "backends": [
            {"name": b.name, "version": b.version, "stack": b.stack, "devices": list(b.devices)} for b in backends
        ],
        "callable": dict(zip(("class", "function"), callable_names(spec), strict=True)),
        "stacks": sorted({b.stack for b in backends}),
        "devices": sorted({d for b in backends for d in b.devices}),
        "models": model_list,
        "default_model": default,
    }


def _default_model(task: str, catalogue: Catalogue) -> str | None:
    try:
        entry = catalogue.default(task)
    except UnknownModel:
        return None
    return f"{entry.id}@{entry.revision}"
