"""Markdown of API entries read from the code: a signature and the docstring of a function, class or method."""

import inspect
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

_QUALIFIED = re.compile(r"\b(?:[a-z_][a-z_0-9]*\.)+(?=[A-Za-z_])")
_BLOCK = re.compile(r"(Raises|Attributes|Args|Returns):")
_ITEM = re.compile(r"    (\w[\w.]*(?:, \w[\w.]*)*): (.*)")


_LONG = 60


def _annotation(parameter: inspect.Parameter | inspect.Signature, empty: object) -> str:
    raw = parameter.annotation if isinstance(parameter, inspect.Parameter) else parameter.return_annotation
    if raw is empty:
        return ""
    text = raw if isinstance(raw, str) else inspect.formatannotation(raw)
    text = _QUALIFIED.sub("", text)
    return "..." if len(text) > _LONG or "StringConstraints" in text else text


def _default(value: object) -> str:
    text = repr(value)
    return "..." if re.search(r"<.* at 0x[0-9a-f]+>", text) else _QUALIFIED.sub("", text)


def signature(obj: Callable[..., Any]) -> str:
    """Return `obj`'s call signature, one parameter a line, without `self`, module prefixes or long annotations.

    An annotation longer than a line of the page is shown as `...`. Where Python cannot give a signature, `(...)`.
    """
    try:
        found = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    params = [p for p in found.parameters.values() if p.name not in ("self", "cls")]
    lines: list[str] = []
    keyword_only = False
    for p in params:
        if p.kind is p.KEYWORD_ONLY and not keyword_only:
            keyword_only = True
            if not any(q.kind is q.VAR_POSITIONAL for q in params):
                lines.append("    *,")
        note = _annotation(p, p.empty)
        stars = "*" if p.kind is p.VAR_POSITIONAL else "**" if p.kind is p.VAR_KEYWORD else ""
        default = "" if p.default is p.empty else f" = {_default(p.default)}"
        lines.append(f"    {stars}{p.name}{': ' + note if note else ''}{default},")
    result = "" if inspect.isclass(obj) else _annotation(found, found.empty)
    tail = f" -> {result}" if result else ""
    if not lines:
        return f"(){tail}"
    if len(lines) == 1 and len(lines[0]) < _LONG:
        return f"({lines[0].strip().rstrip(',')}){tail}"
    return "(\n" + "\n".join(lines) + f"\n){tail}"


def prose(text: str) -> str:
    """Return a Google-style docstring as markdown: its `Raises`, `Attributes` and `Args` blocks as bullet lists."""
    lines: list[str] = []
    block = False
    for line in text.splitlines():
        found = _BLOCK.fullmatch(line)
        if found:
            lines += ["", f"**{found[1]}**", ""]
            block = True
        elif block and line.startswith("    "):
            item = _ITEM.fullmatch(line)
            lines.append(f"- `{item[1]}`: {item[2]}" if item else f"  {line.strip()}")
        else:
            block = block and not line.strip()
            lines.append(line)
    return "\n".join(lines)


def doc(obj: object) -> str:
    """Return the docstring of `obj` as markdown, or the empty string."""
    return prose(inspect.getdoc(obj) or "")


def entry(name: str, obj: Callable[..., Any], *, level: int = 3) -> str:
    """Return a section headed `name`: the call with its signature in a code block, then the docstring."""
    return f"{'#' * level} `{name}`\n\n```python\n{name}{signature(obj)}\n```\n\n{doc(obj)}\n"


def members(owner: type, names: Iterable[str], *, level: int = 4, label: str | None = None) -> str:
    """Return an `entry` for each method of `owner` in `names`, called as `<label or owner name>.<method>`."""
    prefix = label or owner.__name__
    return "\n".join(entry(f"{prefix}.{n}", getattr(owner, n), level=level) for n in names)


def properties(owner: type, names: Iterable[str]) -> str:
    """Return a bullet list of each property of `owner` in `names` with the first line of its docstring."""
    rows = []
    for name in names:
        first = (inspect.getdoc(getattr(owner, name)) or "").split("\n\n")[0]
        rows.append(f"- `{owner.__name__}.{name}`: {' '.join(first.split())}")
    return "\n".join(rows)


def module_summary(module: object) -> str:
    """Return the first paragraph of `module`'s docstring on one line."""
    return " ".join((inspect.getdoc(module) or "").split("\n\n")[0].split())


def field_names(model: Any) -> str:  # noqa: ANN401 - a pydantic model class or a dataclass
    """Return the field names of a pydantic model or dataclass as a comma-separated list of code spans."""
    fields: Mapping[str, object] = getattr(model, "model_fields", None) or getattr(model, "__dataclass_fields__", {})
    return ", ".join(f"`{n}`" for n in fields)
