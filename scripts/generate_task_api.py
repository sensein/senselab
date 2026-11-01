"""Generate `senselab/_api/generated.py`, the task classes and functions, from the task declarations.

uv run python scripts/generate_task_api.py          # write the module
uv run python scripts/generate_task_api.py --check  # exit 1 if the committed module differs
"""

import argparse
import importlib
import json
import math
import sys
import types
import typing
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from senselab_core.kernel.declare import REGISTRY, Registry, TaskSpec

from senselab._api.call import kernel_types
from senselab._api.declarations import DECLARATION_MODULES

OUTPUT = Path(__file__).resolve().parents[1] / "packages" / "senselab" / "src" / "senselab" / "_api" / "generated.py"
RESERVED = frozenset({"self", "model", "backend_params"})
_SCALARS = (bool, int, float, str)


class _Imports:
    def __init__(self) -> None:
        self.names: dict[str, set[str]] = {}

    def add(self, module: str, name: str) -> None:
        self.names.setdefault(module, set()).add(name)

    def lines(self) -> list[str]:
        return [f"from {m} import {', '.join(sorted(n))}" for m, n in sorted(self.names.items())]


def _annotation(t: object, imports: _Imports) -> str:
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if t is None or t is type(None):
        return "None"
    if origin is Annotated:
        return _annotation(args[0], imports)
    if origin is Literal:
        imports.add("typing", "Literal")
        return f"Literal[{', '.join(_literal(a) for a in args)}]"
    if origin in (Union, types.UnionType):
        return " | ".join(_annotation(a, imports) for a in args)
    if origin is not None:
        inner = ", ".join("..." if a is Ellipsis else _annotation(a, imports) for a in args)
        if origin is Mapping or origin is typing.Mapping:
            imports.add("collections.abc", "Mapping")
            return f"Mapping[{inner}]"
        return f"{origin.__name__}[{inner}]"
    assert isinstance(t, type), f"unrenderable annotation {t!r}"
    if t.__module__ == "builtins":
        return t.__name__
    return _named(t, imports)


def _named(t: type, imports: _Imports) -> str:
    if "<locals>" in t.__qualname__:
        raise ValueError(f"{t.__module__}.{t.__qualname__} is not importable; declare it at module level")
    imports.add(t.__module__, t.__qualname__.split(".")[0])
    return t.__qualname__


def _literal(value: object) -> str:
    return json.dumps(value) if isinstance(value, str) else repr(value)


def _default(field: FieldInfo, imports: _Imports) -> str | None:
    """The source of a field's default: `None` if it has none, `UNSET` if it is not a literal."""
    d = field.default
    if d is PydanticUndefined:
        return None if field.is_required() else "UNSET"
    if isinstance(d, Enum):
        return f"{_named(type(d), imports)}.{d.name}"
    if d is None or (isinstance(d, _SCALARS) and not (isinstance(d, float) and not math.isfinite(d))):
        return _literal(d)
    if isinstance(d, tuple) and all(x is None or isinstance(x, _SCALARS) for x in d):
        return "(" + ", ".join(_literal(x) for x in d) + ("," if len(d) == 1 else "") + ")"
    return "UNSET"


def _port_annotation(type_name: str, imports: _Imports) -> str:
    kernel = kernel_types().get(type_name)
    if kernel is None:
        return "object"
    imports.add("os", "PathLike")
    return f"{_named(kernel, imports)} | str | PathLike[str]"


def _names(spec: TaskSpec) -> tuple[str, str]:
    segment = spec.name.rsplit(".", 1)[-1]
    return "".join(w.capitalize() for w in segment.split("_")), segment


def _signature(defs: str, head: list[str], keyword: list[str], returns: str, indent: str = "") -> list[str]:
    items = [*head, *(["*"] if keyword else []), *keyword]
    return [f"{indent}{defs}(", *(f"{indent}    {i}," for i in items), f"{indent}) -> {returns}:"]


def _call(callee: str, args: list[str], indent: str) -> list[str]:
    return [f"{indent}{callee}(", *(f"{indent}    {a}," for a in args), f"{indent})"]


def _mapping(names: list[str]) -> list[str]:
    """The lines of a dict literal `{"n": n, ...}`, exploded so its layout never depends on the formatter."""
    return ["{", *(f'    "{n}": {n},' for n in names), "}"] if names else ["{}"]


def _render_task(spec: TaskSpec, imports: _Imports) -> tuple[str, str, str]:
    cls, func = _names(spec)
    ports = list(spec.inputs)
    fields = spec.params.model_fields
    clash = RESERVED & (set(fields) | set(ports))
    if clash or not all(p.isidentifier() for p in [*ports, *fields]):
        raise ValueError(f"task {spec.name!r}: names {sorted(clash)} are reserved or not identifiers")
    for module, name in (
        ("senselab._api.call", "ConfiguredTask"),
        ("senselab._api.call", "TaskCall"),
        ("senselab._api.call", "UNSET"),
        ("senselab_core.kernel.identity", "ModelRef"),
        ("collections.abc", "Mapping"),
        ("typing", "Any"),
    ):
        imports.add(module, name)
    params, optional = [], []
    for name, field in fields.items():
        default = _default(field, imports)
        annotated = f"{name}: {_annotation(field.annotation, imports)}"
        params.append(annotated if default is None else f"{annotated} = {default}")
        optional.append(f"{annotated} = UNSET")
    data = [f"{p}: {_port_annotation(spec.inputs[p], imports)}" for p in ports]
    backend = "backend_params: Mapping[str, Any] | None = None"
    model = "model: ModelRef | str | None = None"
    summary = f"{spec.name} v{spec.version} ({spec.kind}): {', '.join(ports)} -> {', '.join(spec.outputs)}."
    names = list(fields)
    lines = [
        f"class {cls}(ConfiguredTask):",
        f'    """The configured, immutable task {summary}"""',
        "",
        f'    name = "{spec.name}"',
        "",
        *_signature("def __init__", ["self", model], [*params, backend], "None", "    "),
        '        """Configure the task; params are validated now."""',
        "        super().__init__(",
        "            model,",
        *_dict_arg(names, "            "),
        "            backend_params,",
        "        )",
        "",
        *_signature("def with_params", ["self"], optional, f'"{cls}"', "    "),
        '        """Return a new task with these params changed; this one is unchanged."""',
        "        return self._replace(",
        *_dict_arg(names, "            "),
        "        )",
        "",
        *_signature("def call", ["self", *data], [], "TaskCall", "    "),
        '        """Return the validated call of this task on the inputs."""',
        "        return self._call(",
        *_dict_arg(ports, "            "),
        "        )",
        "",
        "",
        *_signature(f"def {func}", data, [model, *params, backend], "TaskCall"),
        f'    """Return the call of `{cls}` on the inputs: `{cls}(model, **params).call(...)`."""',
        f"    return {cls}(",
        "        model,",
        *(f"        {n}={n}," for n in names),
        "        backend_params=backend_params,",
        f"    ).call({', '.join(ports)})",
    ]
    return "\n".join(lines), cls, func


def _dict_arg(names: list[str], indent: str) -> list[str]:
    body = _mapping(names)
    if len(body) == 1:
        return [f"{indent}{{}},"]
    return [f"{indent}{body[0]}", *(f"{indent}{b}" for b in body[1:-1]), f"{indent}}},"]


def render(registry: Registry = REGISTRY) -> str:
    """Return the source of the generated module for the tasks of `registry`."""
    imports = _Imports()
    bodies, exported = [], []
    for spec in registry.tasks():
        body, cls, func = _render_task(spec, imports)
        bodies.append(body)
        exported += [cls, func]
    if len(set(exported)) != len(exported):
        raise ValueError(f"two tasks share a class or function name: {sorted(exported)}")
    head = (
        '"""Task classes and functions generated from the task declarations; do not edit.\n\n'
        'Regenerate with `uv run python scripts/generate_task_api.py`.\n"""\n\n# ruff: noqa'
    )
    modules = [f"import {m}" for m in DECLARATION_MODULES]
    names = "".join(f'    "{n}",\n' for n in sorted(exported))
    all_ = f"__all__ = [\n{names}]" if exported else "__all__: list[str] = []"
    parts = [head, *(["\n".join(modules)] if modules else []), *(["\n".join(imports.lines())] if exported else [])]
    return "\n\n".join(parts) + "\n\n" + all_ + "".join(f"\n\n\n{b}" for b in bodies) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the module, or with `--check` compare it with the committed one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed module differs")
    args = parser.parse_args(argv)
    for module in DECLARATION_MODULES:
        importlib.import_module(module)
    source = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != source:
            print(f"{OUTPUT} is out of date; run scripts/generate_task_api.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
