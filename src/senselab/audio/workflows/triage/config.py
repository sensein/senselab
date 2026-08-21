"""Loading the triage configuration.

Every number the triage workflow uses lives in ``data/config/default.yaml`` beside the measurement that
produced it. A value nobody has measured is ``null`` there, and reading it raises rather than returning a
number nobody chose.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_DEFAULT = Path(__file__).parent / "data" / "config" / "default.yaml"
_OPEN_QUESTIONS = "specs/20260817-triage-workflow-dag/benchmarks/open.md"
_ABSENT = object()


@dataclass(frozen=True)
class TriageConfig:
    """One resolved configuration.

    Attributes:
        name: The configuration's name.
        version: Schema version of the file.
        config_hash: Hash of the merged mapping, so a run's configuration can be named.
        values: The merged mapping.
    """

    name: str
    version: int
    config_hash: str
    values: dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:  # noqa: ANN401
        """Read a value, returning ``default`` when it is absent or null.

        Args:
            path: Dotted path, e.g. ``"spans.onset_drop_db"``.
            default: Returned when the value is missing or null.

        Returns:
            The value, or ``default``.
        """
        node = self._lookup(path)
        return default if node is _ABSENT or node is None else node

    def require(self, path: str) -> Any:  # noqa: ANN401
        """Read a value that must have been measured.

        Args:
            path: Dotted path.

        Returns:
            The value.

        Raises:
            ValueError: If the key does not exist (a typo), or if it is null because nobody has
                measured it.
        """
        found = self._lookup(path)
        if found is _ABSENT:
            raise ValueError(
                f"unknown configuration key {path!r} in {self.name}; check the spelling against "
                "data/config/default.yaml"
            )
        if found is None:
            raise ValueError(
                f"{path} has no value in {self.name}. It is null because nobody has measured it — see "
                f"{_OPEN_QUESTIONS} for what would settle it. Supply it with a config override rather "
                "than defaulting it here."
            )
        return found

    def _lookup(self, path: str) -> Any:  # noqa: ANN401
        node: Any = self.values
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return _ABSENT
            node = node[part]
        return node


def _merge(base: dict[str, Any], over: dict[str, Any], trail: str = "") -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in over.items():
        where = f"{trail}.{key}" if trail else key
        if key not in out:
            raise ValueError(f"unknown configuration key {where!r}; overrides may not introduce keys")
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def load_triage_config(override: str | Path | None = None) -> TriageConfig:
    """Load the packaged configuration, deep-merging one override over it.

    Args:
        override: Path to a partial YAML. Its keys must already exist in the packaged file — a typo
            is refused rather than silently ignored.

    Returns:
        The resolved configuration, carrying the hash of the merged mapping.

    Raises:
        ValueError: If the override introduces a key the packaged file does not have.
    """
    values = yaml.safe_load(_DEFAULT.read_text())
    if override is not None:
        values = _merge(values, yaml.safe_load(Path(override).read_text()) or {})
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return TriageConfig(name=values["name"], version=int(values["version"]), config_hash=digest, values=values)
