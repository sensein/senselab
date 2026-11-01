"""`tasks`, `describe` and `models`: discovery over the declarations and the catalogue."""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from senselab_core.kernel.declare import REGISTRY, Registry, TaskSpec, json_schema, kernel_types
from senselab_core.kernel.formats import Formatted
from senselab_core.runtime.call import packaged
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


def _link_ports(ports: dict[str, Any], declared: Mapping[str, str]) -> dict[str, Any]:
    """Return `ports` with each port of a kernel type linked to that type's format schema under `$defs`.

    A port of another type takes a path or URI.
    """
    types = kernel_types()
    defs: dict[str, Any] = {}
    properties = {}
    for port, prop in ports["properties"].items():
        kernel_type = types.get(declared[port])
        if kernel_type is None:
            properties[port] = {**prop, "description": "a path or URI"}
            continue
        schema = _format_schema(kernel_type)
        defs.update(schema.pop("$defs", {}))
        defs[kernel_type.__name__] = schema
        properties[port] = {**prop, "$ref": f"#/$defs/{kernel_type.__name__}"}
    return {**ports, "properties": properties, **({"$defs": defs} if defs else {})}


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

    The result holds `task`, `version`, `kind`, the schemas `inputs`, `params`, `backend_params` and `outputs`
    (each port of a kernel type links to that type's format schema in its `$defs`), `backends` as
    `{name, version, stack, devices}`, the sorted `stacks` and `devices` they cover, `callable`, the generated
    `{class, function}` names, `models` (as `models(name)` gives them, licenses included) and `default_model`, the
    catalogue default as `id@revision` (None for a task with none). `catalogue` is the one `run_call` reads when None.

    The function and the class run the task on the inputs, given as positional data: an object, a path or a URI (a
    port whose type is not a kernel type takes a JSON value, such as a path or URI string). The one call shape of
    the CLI and JSON, `{task, inputs, params, model, backend_params}`, validates as a `TaskCall` and runs with
    `run_call`.

    Raises:
        ApiError: `unknown_task`, with the closest declared names.
    """
    spec = task_spec(name, registry)
    schema = json_schema(spec, registry)
    backends = registry.backends(name)
    return {
        "task": spec.name,
        "version": spec.version,
        "kind": spec.kind,
        **schema,
        "inputs": _link_ports(schema["inputs"], spec.inputs),
        "outputs": _link_ports(schema["outputs"], spec.outputs),
        "backends": [
            {"name": b.name, "version": b.version, "stack": b.stack, "devices": list(b.devices)} for b in backends
        ],
        "callable": dict(zip(("class", "function"), callable_names(spec), strict=True)),
        "stacks": sorted({b.stack for b in backends}),
        "devices": sorted({d for b in backends for d in b.devices}),
        "models": models(name, catalogue),
        "default_model": _default_model(name, catalogue or packaged().catalogue),
    }


def _default_model(task: str, catalogue: Catalogue) -> str | None:
    try:
        entry = catalogue.default(task)
    except UnknownModel:
        return None
    return f"{entry.id}@{entry.revision}"
