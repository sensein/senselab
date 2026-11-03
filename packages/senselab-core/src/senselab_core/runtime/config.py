"""The one YAML reader: dates stay text, a repeated key is refused, a non-mapping document is refused."""

from __future__ import annotations

from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A configuration document that cannot be read; names what it is, where it came from and why it failed."""

    def __init__(self, what: str, label: str, message: str) -> None:
        """Record `what` (the kind of document), `label` (its path or `<text>`) and `message`."""
        super().__init__(f"{what} {label}: {message}")
        self.what = what
        self.label = label
        self.message = message


class _Loader(yaml.SafeLoader):
    """A safe loader that leaves `2026-11-01` a string and refuses a mapping holding one key twice."""


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:timestamp"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _unique_mapping(loader: _Loader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
    repeated = sorted({str(k) for k in keys if keys.count(k) > 1})
    if repeated:
        raise yaml.YAMLError(f"duplicate keys {repeated}")
    return loader.construct_mapping(node, deep=deep)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_yaml(source: str | Path | Traversable, what: str) -> dict[str, Any]:
    """Return the mapping in the YAML `source`: the text itself, or the file when a `Path` or resource.

    Args:
        source: YAML text, a `Path`, or a packaged resource.
        what: the kind of document, for the error.

    Raises:
        ConfigError: the file cannot be read, the text is not YAML, a key repeats, or the document is not a mapping.
    """
    if isinstance(source, str):
        label, text = "<text>", source
    else:
        label = str(source)
        try:
            text = source.read_text("utf-8")
        except OSError as err:
            raise ConfigError(what, label, f"cannot be read: {err}") from err
    try:
        raw = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as err:
        raise ConfigError(what, label, f"is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(what, label, "must be a mapping")
    return raw


def dump_yaml(document: dict[str, Any]) -> str:
    """Return `document` as YAML text, keys in insertion order and no line wrapping."""
    return yaml.safe_dump(document, sort_keys=False, width=float("inf"))
