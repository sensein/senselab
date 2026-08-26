"""Loading the triage configuration.

Every number the triage workflow uses lives in ``data/config/default.yaml`` beside the measurement that
produced it. A value nobody has measured is ``null`` there, and reading it raises rather than returning a
number nobody chose.

An override may not introduce a key, because a key the packaged file does not have is a typo and
ignoring it would run the packaged value while the caller believed otherwise. Two kinds of mapping
live in that file, though, and the rule applies to one of them:

* A **schema** mapping's keys are names the code reads — ``spans.onset_drop_db``,
  ``taxonomy.voice_min_duration_s``. A key the code never reads does nothing, so a new one is refused.
* A **data** mapping's keys are values the caller supplies — a HeAR label, a hint tag, a vocal task,
  a kind. Refusing a new one refuses the configuration's whole purpose: a campaign screening for
  sneezes could not add ``airway.confirmation_map.Sneeze`` without editing the installed package.

The data mappings are named explicitly in :data:`DATA_MAP_PATHS` rather than detected by shape. A
structural rule — "a dict whose values are not dicts" — matches almost every leaf section in the
packaged file, so it would exempt the schema along with the data and refuse nothing at all.
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
MIN_AST_HOP_S = 8.0
"""Shortest supported AST hop, in seconds.

AST scores a 10.24-second context window. More frequent outputs would look like independent
time-local evidence in the report even though their acoustic context heavily overlaps, so this is a
configuration constraint rather than a presentation preference.
"""

DATA_MAP_PATHS = frozenset(
    {
        "airway.confirmation_map",
        "airway.k_db_by_task",
        "routing.hint_kind_map",
        "spans.k_db",
        "voice.f0_range_by_population",
        "voice.task_duration_ranges",
        "windows.ast.label_thresholds",
        "windows.hear.label_thresholds",
        "windows.yamnet.label_thresholds",
    }
)
"""Dotted paths whose mapping is keyed by data, so an override may add entries to it.

Every other mapping is schema and an override may only change keys it already has. Renaming one of
these paths without updating this set silently returns it to the schema rule; ``config_test`` pins
each path's existence against the packaged file.
"""


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
    """Deep-merge one override mapping over the packaged one.

    Args:
        base: The packaged mapping at this level.
        over: The override mapping at this level.
        trail: The dotted path of this level, for the message and for the data-map lookup.

    Returns:
        The merged mapping.

    Raises:
        ValueError: If the override names a key the packaged mapping does not have, outside a path
            in :data:`DATA_MAP_PATHS`.
    """
    out = deepcopy(base)
    for key, value in over.items():
        where = f"{trail}.{key}" if trail else key
        if key not in out:
            raise ValueError(f"unknown configuration key {where!r}; overrides may not introduce keys")
        if where in DATA_MAP_PATHS and isinstance(value, dict):
            out[key] = {**out[key], **deepcopy(value)} if isinstance(out[key], dict) else deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def _validate(values: dict[str, Any]) -> None:
    """Reject resolved settings that would misrepresent AST's temporal resolution."""
    try:
        hop_s = float(values["windows"]["ast"]["hop_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("windows.ast.hop_s must be a numeric number of seconds") from error
    if hop_s < MIN_AST_HOP_S:
        raise ValueError(f"windows.ast.hop_s must be at least {MIN_AST_HOP_S:g} s; received {hop_s:g} s")


def load_triage_config(override: str | Path | None = None) -> TriageConfig:
    """Load the packaged configuration, deep-merging one override over it.

    Args:
        override: Path to a partial YAML. Its keys must already exist in the packaged file — a typo
            is refused rather than silently ignored — except inside a mapping named in
            :data:`DATA_MAP_PATHS`, whose keys are data and where an override may add entries.

    Returns:
        The resolved configuration, carrying the hash of the merged mapping.

    Raises:
        ValueError: If the override introduces a key the packaged file does not have, outside a
            data mapping.
    """
    values = yaml.safe_load(_DEFAULT.read_text())
    if override is not None:
        values = _merge(values, yaml.safe_load(Path(override).read_text()) or {})
    _validate(values)
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return TriageConfig(name=values["name"], version=int(values["version"]), config_hash=digest, values=values)
