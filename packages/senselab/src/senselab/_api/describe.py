"""`tasks` and `describe`: discovery over the declarations."""

from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema
from senselab_core.kernel.declare import REGISTRY, Registry, TaskSpec, json_schema
from senselab_core.kernel.formats import Formatted

from senselab._api.call import ApiError, closest, kernel_types, task_spec


class _KernelSchema(GenerateJsonSchema):
    """JSON Schema of a kernel type: an array field is an `array`."""

    def is_instance_schema(self, schema: Any) -> dict[str, Any]:  # noqa: ANN401, ARG002 - pydantic's hook signature
        return {"type": "array"}


def _format_schema(kernel_type: type[BaseModel]) -> dict[str, Any]:
    schema = kernel_type.model_json_schema(schema_generator=_KernelSchema, ref_template="#/$defs/{model}")
    if issubclass(kernel_type, Formatted):
        schema["x-senselab-format"] = kernel_type.model_fields["format"].default
    return schema


def _link_ports(ports: dict[str, Any]) -> dict[str, Any]:
    """Return `ports` with each port of a kernel type linked to that type's format schema under `$defs`."""
    types = kernel_types()
    defs: dict[str, Any] = {}
    properties = {}
    for port, prop in ports["properties"].items():
        kernel_type = types.get(prop["title"])
        if kernel_type is None:
            properties[port] = prop
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


def describe(name: str, registry: Registry = REGISTRY) -> dict[str, Any]:
    """Return the JSON Schema of task `name` with its backends, stacks and devices.

    The result holds `task`, `version`, `kind`, the schemas `inputs`, `params`, `backend_params` and `outputs`
    (each port of a kernel type links to that type's format schema in its `$defs`), `backends` as
    `{name, version, stack, devices}`, and the sorted `stacks` and `devices` they cover.

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
        "inputs": _link_ports(schema["inputs"]),
        "outputs": _link_ports(schema["outputs"]),
        "backends": [
            {"name": b.name, "version": b.version, "stack": b.stack, "devices": list(b.devices)} for b in backends
        ],
        "stacks": sorted({b.stack for b in backends}),
        "devices": sorted({d for b in backends for d in b.devices}),
    }
