"""Loading the triage configuration.

Every number the triage workflow uses lives in ``data/config/default.yaml``. A value nobody has
measured is ``null`` there, and :meth:`TriageConfig.require` raises on it rather than returning a
number nobody chose.

An override may not introduce a key, with one exception: a **data** mapping, whose keys are values
the caller supplies — a HeAR label, a hint tag, a vocal task, a kind, a task's trailing index — may
gain entries. Those mappings are named in :data:`DATA_MAP_PATHS`; every other mapping is schema,
whose keys are names the code reads, and a key it does not already have is refused.

A dataset profile (:mod:`~senselab.audio.workflows.triage.dataset_profile`) sits between the two: its own
override is merged over the packaged file before a run's override, and the data files it supplies are read
instead of the packaged ones. A configuration loaded with a profile carries it, and its hash covers the profile's
name, version and every byte the profile lists; one loaded without a profile hashes exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.dataset_profile import (
    DatasetProfile,
    activate_profile,
    active_profile,
    check_run_data,
    load_profile,
    set_run_data,
)
from senselab.audio.workflows.triage.workflow import listed_stages

_DEFAULT = Path(__file__).parent / "data" / "config" / "default.yaml"
_OPEN_QUESTIONS = "specs/20260817-triage-workflow-dag/benchmarks/open.md"
_ABSENT = object()
MIN_AST_HOP_S = 8.0
"""Shortest supported AST hop, in seconds. AST scores a 10.24-second context window."""

DATA_MAP_PATHS = frozenset(
    {
        "branch.label_sets",
        "branch.phoneme_place_classes",
        "branch.phoneme_vowel_classes",
        "redaction.name_approvals",
        "routing.hint_branch_map",
        "verdict.conformance_flags_by_family",
        "verdict.flag_gate_exemptions",
        "verdict.gates.default",
        "verdict.gates.by_family",
        "verdict.gates.by_group.EFFORT",
        "verdict.gates.by_group.EVENT_ALTERNATION",
        "verdict.gates.by_group.EVENT_SERIES",
        "verdict.gates.by_group.FREE_RESPONSE",
        "verdict.gates.by_group.GLIDE",
        "verdict.gates.by_group.ITEM_LIST",
        "verdict.gates.by_group.ORDERED_TOKENS",
        "verdict.gates.by_group.PER_SENTENCE",
        "verdict.gates.by_group.SOUND_COVERAGE",
        "verdict.gates.by_group.SUSTAINED",
        "verdict.gates.by_group.SYLLABLE_SEQUENCE",
        "verdict.gates.by_group.SYLLABLE_TRAIN",
        "voice.f0_range_by_population",
        "voice.task_duration_ranges",
        "windows.ast.label_thresholds",
        "windows.hear.label_thresholds",
        "windows.yamnet.label_thresholds",
    }
)
"""Dotted paths whose mapping is keyed by data, so an override may add entries to it.

Every other mapping is schema and an override may only change keys it already has.
"""

FIXED_PATHS = frozenset({"resample.target_hz"})
"""Dotted paths no override may change: the 16 kHz working rate every model triage runs is native at."""

DATA_SECTION = "data"
"""The configuration's ``data`` section: per data file, the values a profile's or run's override changes, merged by
:func:`~senselab.audio.workflows.triage.dataset_profile.data_values` over the file's packaged values."""


class UnknownConfigKey(ValueError):
    """A path no packaged key spells. A typo in the calling code, never a missing measurement."""


class UnmeasuredConfigKey(ValueError):
    """A packaged key whose value is null because nobody has measured it.

    A caller entitled to proceed without the value catches this and only this.
    """


@dataclass(frozen=True)
class TriageConfig:
    """One resolved configuration.

    Attributes:
        name: The configuration's name.
        version: Schema version of the file.
        config_hash: Hash of the merged mapping, and of the profile when there is one, so a run's
            configuration can be named.
        values: The merged mapping.
        profile: The dataset profile it was loaded with, or None.
    """

    name: str
    version: int
    config_hash: str
    values: dict[str, Any]
    profile: DatasetProfile | None = None

    def get(self, path: str, default: Any = None) -> Any:  # noqa: ANN401
        """Read a value, returning ``default`` when it is absent or null.

        Args:
            path: Dotted path, e.g. ``"spans.k_db"``.
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
            UnknownConfigKey: If no packaged key spells this path. A ``ValueError`` subclass.
            UnmeasuredConfigKey: If the key exists and is null because nobody has measured it. Also
                a ``ValueError`` subclass, and the only one a caller entitled to proceed without a
                value may catch.
        """
        found = self._lookup(path)
        if found is _ABSENT:
            raise UnknownConfigKey(
                f"unknown configuration key {path!r} in {self.name}; check the spelling against "
                "data/config/default.yaml"
            )
        if found is None:
            raise UnmeasuredConfigKey(
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
        if where == DATA_SECTION:
            if not isinstance(value, dict):
                raise ValueError(f"configuration key {where!r} is a mapping of data file to the values it changes")
            out[key] = {**out[key]}
            for relative, values in value.items():
                out[key][relative] = _deep(out[key].get(relative), values)
            continue
        if key not in out:
            raise ValueError(f"unknown configuration key {where!r}; overrides may not introduce keys")
        if where in FIXED_PATHS and value != out[key]:
            raise ValueError(f"configuration key {where!r} is fixed at {out[key]!r}; an override may not change it")
        if where in DATA_MAP_PATHS and isinstance(value, dict):
            out[key] = {**out[key], **deepcopy(value)} if isinstance(out[key], dict) else deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def _deep(base: Any, over: Any) -> Any:  # noqa: ANN401 -- YAML values
    """Merge two data-section entries; the data file's own rules are applied when it is read."""
    if isinstance(base, dict) and isinstance(over, dict):
        return {**base, **{key: _deep(base.get(key), value) for key, value in over.items()}}
    return deepcopy(over)


RECOGNIZER_TIMESTAMPS = ("native", "bundled_aligner")
"""How a recognizer's word times are taken: by the recognizer itself, or by its ``timestamp_model``."""

_RECOGNIZER_KEYS = frozenset({"name", "model", "revision", "timestamps", "timestamp_model", "options"})

RECOGNIZER_PREFIX = "asr_"
"""What every recognizer's name starts with: the name is its measurement's, which no other block may write."""


def _default_aligner(model: str) -> str | None:
    """The aligner a bundled-aligner recognizer uses when its options name none; None where senselab knows none."""
    from senselab.audio.tasks.speech_to_text.api import _QWEN_ASR_PREFIXES  # noqa: PLC0415
    from senselab.audio.tasks.speech_to_text.qwen import _DEFAULT_FORCED_ALIGNER  # noqa: PLC0415

    return _DEFAULT_FORCED_ALIGNER if model.startswith(_QWEN_ASR_PREFIXES) else None


def _validate_recognizers(recognizers: Any) -> None:  # noqa: ANN401 -- parsed YAML
    """Reject an ``asr.recognizers`` list PREPROCESS could not run."""
    if not isinstance(recognizers, list) or not recognizers:
        raise ValueError("asr.recognizers must list at least one speech recognizer")
    names = []
    for index, entry in enumerate(recognizers):
        where = f"asr.recognizers[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(entry) - _RECOGNIZER_KEYS)
        if unknown:
            raise ValueError(f"{where} has unknown keys {unknown}; the keys are {sorted(_RECOGNIZER_KEYS)}")
        for key in ("name", "model", "revision"):
            if not entry.get(key):
                raise ValueError(f"{where} needs a {key}")
        if entry.get("timestamps") not in RECOGNIZER_TIMESTAMPS:
            raise ValueError(f"{where}.timestamps must be one of {RECOGNIZER_TIMESTAMPS}")
        if not str(entry["name"]).startswith(RECOGNIZER_PREFIX):
            raise ValueError(f"{where}.name must start with {RECOGNIZER_PREFIX!r}, so it names no other measurement")
        if (entry["timestamps"] == "bundled_aligner") != bool(entry.get("timestamp_model")):
            raise ValueError(f"{where}: timestamp_model is given exactly when timestamps is bundled_aligner")
        options = entry.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError(f"{where}.options must be a mapping")
        if entry.get("timestamp_model"):
            aligner = options.get("forced_aligner") or _default_aligner(str(entry["model"]))
            if aligner != entry["timestamp_model"]:
                raise ValueError(
                    f"{where}: timestamp_model {entry['timestamp_model']!r} is not the aligner the recognizer uses "
                    f"({aligner!r}); name it in options.forced_aligner"
                )
        names.append(str(entry["name"]))
    if len(set(names)) != len(names):
        raise ValueError(f"asr.recognizers names a recognizer twice: {names}")


def _validate(values: dict[str, Any]) -> None:
    """Reject resolved settings that would misrepresent AST's temporal resolution, or name no recognizer."""
    _validate_recognizers((values.get("asr") or {}).get("recognizers"))
    try:
        hop_s = float(values["windows"]["ast"]["hop_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("windows.ast.hop_s must be a numeric number of seconds") from error
    if hop_s < MIN_AST_HOP_S:
        raise ValueError(f"windows.ast.hop_s must be at least {MIN_AST_HOP_S:g} s; received {hop_s:g} s")


def profile_config_values(profile: DatasetProfile | None) -> dict[str, Any]:
    """The packaged configuration with each profile layer's override merged over it, the base first.

    Reads files only: no profile is activated and the run's ``data`` section is left as it is, so a loader may
    ask it what the active profile configures without resetting the run (:func:`load_triage_config` sets both).

    Args:
        profile: The profile, or None for the packaged configuration alone.

    Returns:
        The merged mapping, before a run's override and before validation.

    Raises:
        ValueError: If a layer introduces a key the packaged file does not have, outside a data mapping, or
            carries a ``data`` section.
    """
    values = yaml.safe_load(_DEFAULT.read_text())
    for layer in profile.chain() if profile is not None else ():
        if layer.config is not None:
            layer_values = yaml.safe_load(layer.config.read_text()) or {}
            if layer_values.get(DATA_SECTION):
                raise ValueError(
                    f"{layer.config}: a profile changes data files by supplying them, not through a `data:` section, "
                    "which is a run's (it would apply after every profile's files, out of the chain's order)"
                )
            values = _merge(values, layer_values)
    return values


PROFILE_ONLY_KEYS: tuple[tuple[str, str], ...] = (("taxonomy", "airway_ontology_roots"),)
"""Settings a dataset profile may set and a run's override may not: the routing labels are built from the packaged
configuration and the profile's layers (:func:`profile_config_values`), once per profile, so a run's value would be
recorded in its configuration and never applied."""


def load_triage_config(
    override: str | Path | None = None, *, profile: str | Path | DatasetProfile | None = None
) -> TriageConfig:
    """Load the packaged configuration, then a dataset profile's override, then a run's override.

    Args:
        override: Path to a partial YAML. Its keys must already exist in the packaged file,
            except inside a mapping named in :data:`DATA_MAP_PATHS`, which may gain entries.
        profile: A dataset profile, or its directory. When None and a profile is already active
            (:func:`~senselab.audio.workflows.triage.dataset_profile.active_profile`), that profile is used, so the
            configuration always names the profile its data files are read from. The profile's own override, if
            it has one, is merged under ``override`` by the same rules, and the profile is activated only once
            the merged configuration has validated.

    Returns:
        The resolved configuration, carrying the hash of the merged mapping and of the profile.

    Raises:
        ValueError: If an override introduces a key the packaged file does not have, outside a
            data mapping, or the profile is malformed.
        RuntimeError: If a different profile is already active in this process.
    """
    loaded = load_profile(profile) if isinstance(profile, (str, Path)) else profile
    if loaded is None:
        loaded = active_profile()
    values = profile_config_values(loaded)
    if override is not None:
        run_values = yaml.safe_load(Path(override).read_text()) or {}
        for section, key in PROFILE_ONLY_KEYS:
            if isinstance(run_values.get(section), Mapping) and key in run_values[section]:
                raise ValueError(
                    f"{override}: {section}.{key} is read when the routing labels are built from the dataset "
                    "profile, so a run's override cannot change it; set it in the dataset profile's config"
                )
        values = _merge(values, run_values)
    _validate(values)
    hashed = json.dumps(values, sort_keys=True, default=str)
    listed_stages(values)  # an unknown stage, or one that may not be off, is refused here
    run_data = check_run_data(values.get(DATA_SECTION))
    if loaded is not None:
        activate_profile(loaded)
        hashed += f"\0profile\0{loaded.name}\0{loaded.version}\0{loaded.digest}"
    set_run_data(run_data)
    digest = hashlib.sha256(hashed.encode()).hexdigest()[:16]
    return TriageConfig(
        name=values["name"], version=int(values["version"]), config_hash=digest, values=values, profile=loaded
    )
