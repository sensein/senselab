"""The one YAML reader: dates stay text, a repeated key is refused, a non-mapping document is refused.

`merge_over` deep-merges a user's config mapping over a packaged one.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
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


class _DuplicateKeys(yaml.YAMLError):
    """A mapping that holds one key twice."""


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
        raise _DuplicateKeys(f"duplicate keys {repeated}")
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
        except (OSError, UnicodeDecodeError) as err:
            raise ConfigError(what, label, f"cannot be read: {err}") from err
    try:
        raw = yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass
    except _DuplicateKeys as err:
        raise ConfigError(what, label, str(err)) from err
    except yaml.YAMLError as err:
        raise ConfigError(what, label, f"is not valid YAML: {err}") from err
    if not isinstance(raw, dict):
        raise ConfigError(what, label, "must be a mapping")
    return raw


def dump_yaml(document: dict[str, Any]) -> str:
    """Return `document` as YAML text, keys in insertion order and no line wrapping."""
    return yaml.safe_dump(document, sort_keys=False, width=float("inf"))


def merge_over(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    """Return `top` deep-merged over `base`: a key both hold as mappings merges, any other key of `top` replaces."""
    return {
        k: merge_over(base[k], top[k])
        if isinstance(base.get(k), Mapping) and isinstance(top.get(k), Mapping)
        else top[k]
        if k in top
        else base[k]
        for k in {**base, **top}
    }


_DEFAULTED = re.compile(r"\$\{(\w+):-([^}]*)\}")


def expand_variables(raw: str) -> str:
    """Return `raw` with `${VAR:-fallback}`, then `~` and `$VAR`, expanded; a variable that is not set stays."""
    chosen = _DEFAULTED.sub(lambda m: os.environ.get(m[1]) or m[2], raw)
    return os.path.expanduser(os.path.expandvars(chosen))


def expand(raw: str) -> str:
    """Return `raw` with `${VAR:-fallback}`, then `~` and `$VAR`, expanded from the environment.

    Raises:
        ValueError: `raw` names a variable that is not set.
    """
    text = expand_variables(raw)
    if "$" in text:
        raise ValueError(f"{raw!r} names a variable that is not set")
    return text
