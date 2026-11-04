"""A dataset profile: the data a dataset supplies for triage, and how triage finds it.

Triage reads its data files by their path under ``data/``. They come in four kinds:

- **Dataset** paths (:data:`DATASET_PATHS`) hold what only a dataset can say: its task list, cohort conditions,
  lexicons and corpus-fitted distributions. senselab ships none of them; a profile supplies them, and without one
  :func:`data_path` raises :class:`DatasetDataMissing`.
- **Value** files (:data:`MERGED_PATHS`) hold senselab's parameters. A profile's copy holds only the values it
  changes and is merged over the packaged file key by key (:func:`data_values`); a key the packaged file does not
  have is refused, except inside a mapping the packaged file leaves empty, and the keys of :data:`FIXED_KEYS` may
  not change.
- **Replaced** files (:data:`REPLACED_PATHS`) are word lists and model vocabularies: a profile's copy is read whole
  instead of the packaged one.
- **Fixed** files (every other packaged file) define senselab's outputs (the corpus-table dictionary, the evidence
  descriptions, the triage table's columns). They are always the packaged copy.

A run's configuration may carry a ``data`` section, ``{data path: values}``, applied over the profile by the same
rules (:func:`set_run_data`); ``load_triage_config`` sets it, and the configuration hash covers it.

A dataset profile is a directory holding ``profile.yaml`` and the files it supplies::

    name: bridge2ai-voice-adult      # required
    version: "1"                     # required
    provenance: {...}                # optional, free-form: what the values were fitted on
    config: config.yaml              # optional: a config override, merged between the packaged config and a run's
    files:                           # data path -> path relative to the profile directory
      cohort_conditions: cohort_conditions
      task_extent.yaml: task_extent.yaml

:func:`activate_profile` makes a profile the one :func:`data_path` reads from. Loaders cache what they read under
the active profile (:func:`profile_cache`), so a profile activated later is never served what an earlier one
filled.

See ``specs/20261022-triage-dataset-profile/plan.md`` and ``inventory.md``.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, TypeVar

import yaml  # type: ignore[import-untyped]

PACKAGED_DATA = Path(__file__).parent / "data"
"""The packaged data directory: the engine files and the defaulted files."""

MANIFEST = "profile.yaml"
"""The file that makes a directory a dataset profile."""

PACKAGED_PROFILES = Path(__file__).parent / "profiles"
"""Profiles senselab ships, opt-in and by name: a profile builds on one with ``extends: senselab:<name>``."""

SENSELAB_PROFILE = "senselab:"

DATASET_PATHS = frozenset(
    {
        "airway_event_tokens.yaml",
        "cohort_conditions",
        "detector_profile",
        "discard_contested.yaml",
        "evaluation",
        "task_lexicon",
        "tasks.yaml",
        "time_release.yaml",
    }
)
"""Data paths only a dataset profile supplies; senselab ships no copy."""

MERGED_PATHS = frozenset(
    {
        "asr_degenerate.yaml",
        "background_model.yaml",
        "background_speech.yaml",
        "breath_pattern.yaml",
        "cohort_stage.yaml",
        "cough_pattern.yaml",
        "decision_reasons.yaml",
        "disagreement_profile",
        "discard_replaced_task.yaml",
        "quality_join.yaml",
        "review_page.yaml",
        "session_background.yaml",
        "speaker_vector_profile",
        "task_content.yaml",
        "task_events.yaml",
        "task_extent.yaml",
        "task_minimum_duration.yaml",
        "task_speech.yaml",
        "task_squim.yaml",
        "voice_phonation.yaml",
        "word_timing.yaml",
    }
)
"""senselab's value files: a profile or run changes any of their values, one at a time (:func:`data_values`)."""

REPLACED_PATHS = frozenset(
    {"classifier_ontology", "function_words_es.yaml", "pii_category_families.yaml", "temporal_words.yaml"}
)
"""Word lists and model vocabularies a profile replaces whole (another language, other models)."""

DEFAULTED_PATHS = MERGED_PATHS | REPLACED_PATHS
"""Data paths senselab packages that a profile may change: merged or replaced."""

FIXED_KEYS: Mapping[str, frozenset[str]] = {
    "decision_reasons.yaml": frozenset({"precedence", "meaning", "keys", "annotations", "release_keys", "prefixes"}),
}
"""Keys of a value file that define senselab's output vocabulary, so no profile or run may change them. Every
file's ``version`` / ``schema_version`` may be restated but not changed."""

_VERSION_KEYS = frozenset({"version", "schema_version"})

SUPPLIABLE_PATHS = DATASET_PATHS | DEFAULTED_PATHS
"""Every path a profile may list. Anything else is an engine file, a typo, or a file no loader reads."""


class DatasetDataMissing(FileNotFoundError):
    """Triage needs a dataset path that no active profile supplies."""


@dataclass(frozen=True)
class DatasetProfile:
    """One loaded dataset profile.

    Attributes:
        name: The profile's name.
        version: The profile's version, as written.
        root: The profile directory.
        files: Data path to the file or directory the profile supplies for it.
        config: The profile's config override, or None.
        provenance: Whatever the profile says about where its values came from.
        digest: Hash of the manifest, of every byte of every file it supplies, and of the profile it extends.
        base: The profile it builds on (``extends``), or None.
    """

    name: str
    version: str
    root: Path
    files: Mapping[str, Path] = field(default_factory=dict)
    config: Path | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""
    base: "DatasetProfile | None" = None

    def supplies(self, relative: str) -> Path | None:
        """Where this profile itself, not the one it extends, supplies a data path, or None.

        Args:
            relative: A path under ``data/``, e.g. ``task_lexicon/categories/animals.yaml``.

        Returns:
            The supplied path (which need not exist inside a supplied directory), or None.
        """
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] not in self.files:
            return None
        return self.files[parts[0]].joinpath(*parts[1:])

    def resolve(self, relative: str) -> Path | None:
        """Where the profile, or the nearest profile it extends, supplies a data path; None when none does.

        Args:
            relative: A path under ``data/``.

        Returns:
            The supplied path, or None.
        """
        own = self.supplies(relative)
        if own is not None:
            return own
        return self.base.resolve(relative) if self.base is not None else None

    def chain(self) -> tuple["DatasetProfile", ...]:
        """This profile and those it extends, the outermost base first: the order their values are merged in."""
        return (*(self.base.chain() if self.base is not None else ()), self)


def _relative(name: object, where: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: a path must be a non-empty string, got {name!r}")
    path = PurePosixPath(name.strip())
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{where}: {name!r} must be a relative path that stays inside its directory")
    return str(path)


def _file_digests(path: Path) -> list[tuple[str, str]]:
    members = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return [
        (str(member.relative_to(path)) if path.is_dir() else "", hashlib.sha256(member.read_bytes()).hexdigest())
        for member in members
    ]


def profile_directory(reference: str, *, relative_to: Path | None = None) -> Path:
    """The directory a profile reference names: ``senselab:<name>`` for a profile senselab ships, else a path.

    Args:
        reference: ``senselab:<name>``, or a directory path.
        relative_to: The directory a relative path is read from (the extending profile's).

    Returns:
        The directory.

    Raises:
        ValueError: If ``senselab:<name>`` names no profile senselab ships.
    """
    if reference.startswith(SENSELAB_PROFILE):
        name = reference[len(SENSELAB_PROFILE) :]
        target = PACKAGED_PROFILES / name
        shipped = sorted(path.name for path in PACKAGED_PROFILES.iterdir() if (path / MANIFEST).is_file())
        if not name or "/" in name or not (target / MANIFEST).is_file():
            raise ValueError(f"senselab ships no profile {name!r}; it ships {shipped}")
        return target
    path = Path(reference)
    return path if path.is_absolute() or relative_to is None else relative_to / path


def load_profile(directory: str | Path, *, _seen: tuple[Path, ...] = ()) -> DatasetProfile:
    """Read and check a dataset profile, and the profile it extends.

    Args:
        directory: The profile directory, holding :data:`MANIFEST`, or ``senselab:<name>`` for a profile senselab
            ships.
        _seen: The profiles already being loaded in this chain, to refuse a cycle.

    Returns:
        The profile, with its digest.

    Raises:
        ValueError: If the manifest lacks a name or version, lists a path outside :data:`SUPPLIABLE_PATHS` (an
            engine file, a file inside a directory instead of the directory, or a typo), names a path that
            leaves its directory, supplies a file where a directory belongs or the reverse, lists something
            that does not exist, or extends a profile that extends it.
    """
    root = (profile_directory(directory) if isinstance(directory, str) else Path(directory)).resolve()
    if root in _seen:
        raise ValueError(f"profile {root} extends itself through {[str(path) for path in _seen]}")
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"{root} is not a dataset profile: it has no {MANIFEST}")
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must hold a mapping")
    for key in ("name", "version"):
        if manifest.get(key) in (None, ""):
            raise ValueError(f"{manifest_path} must give a {key}")
    files: dict[str, Path] = {}
    for data_name, source in (manifest.get("files") or {}).items():
        relative = _relative(data_name, f"{manifest_path} files")
        if relative not in SUPPLIABLE_PATHS:
            raise ValueError(
                f"{manifest_path}: {relative!r} is not a path a profile can supply; supply one of "
                f"{sorted(SUPPLIABLE_PATHS)} (a directory whole, not a file inside it)"
            )
        target = root / _relative(source, f"{manifest_path} files.{relative}")
        if not target.exists():
            raise ValueError(f"{manifest_path}: {relative!r} names {target}, which does not exist")
        expect_dir = not relative.endswith((".yaml", ".yml", ".json", ".parquet"))
        if target.is_dir() != expect_dir:
            raise ValueError(f"{manifest_path}: {relative!r} must be {'a directory' if expect_dir else 'a file'}")
        files[relative] = target
    config = None
    if manifest.get("config") is not None:
        config = root / _relative(manifest["config"], f"{manifest_path} config")
        if not config.is_file():
            raise ValueError(f"{manifest_path}: config names {config}, which does not exist")
    base = None
    if manifest.get("extends") not in (None, ""):
        if not isinstance(manifest["extends"], str):
            raise ValueError(f"{manifest_path}: extends names one profile, senselab:<name> or a path")
        base = load_profile(profile_directory(manifest["extends"], relative_to=root), _seen=(*_seen, root))
    hasher = hashlib.sha256(manifest_path.read_bytes())
    if base is not None:
        hasher.update(f"\0extends\0{base.name}\0{base.version}\0{base.digest}".encode())
    for relative in sorted(files):
        for member, digest in _file_digests(files[relative]):
            hasher.update(f"\0{relative}\0{member}\0{digest}".encode())
    if config is not None:
        hasher.update(b"\0config\0" + hashlib.sha256(config.read_bytes()).hexdigest().encode())
    return DatasetProfile(
        name=str(manifest["name"]),
        version=str(manifest["version"]),
        root=root,
        files=files,
        config=config,
        provenance=dict(manifest.get("provenance") or {}),
        digest=hasher.hexdigest()[:16],
        base=base,
    )


_ACTIVE: DatasetProfile | None = None


def activate_profile(profile: DatasetProfile) -> None:
    """Make a profile the one :func:`data_path` reads from.

    Activating the profile already active again does nothing.

    Args:
        profile: The profile.

    Raises:
        RuntimeError: If a different profile is already active. Call :func:`deactivate_profile` first; a run that
            needs two profiles runs them in separate processes.
    """
    global _ACTIVE
    if _ACTIVE is not None and (_ACTIVE.root, _ACTIVE.digest) != (profile.root, profile.digest):
        raise RuntimeError(
            f"dataset profile {_ACTIVE.name} {_ACTIVE.version} ({_ACTIVE.root}) is already active; deactivate it "
            f"before activating {profile.name} {profile.version} ({profile.root})"
        )
    _ACTIVE = profile


def deactivate_profile() -> None:
    """Return :func:`data_path` to the packaged files only."""
    global _ACTIVE
    _ACTIVE = None


def active_profile() -> DatasetProfile | None:
    """The profile :func:`data_path` reads from, or None."""
    return _ACTIVE


def data_path(relative: str) -> Path:
    """Where a data file or directory is read from.

    Args:
        relative: A path under ``data/``, e.g. ``task_extent.yaml`` or ``task_lexicon/cinderella-story.yaml``.

    Returns:
        The active profile's path where it supplies one, otherwise the packaged path.

    Raises:
        DatasetDataMissing: If the path is a dataset path (:data:`DATASET_PATHS`) and no active profile supplies
            it.
    """
    supplied = _ACTIVE.resolve(relative) if _ACTIVE is not None else None
    if supplied is not None:
        return supplied
    parts = PurePosixPath(relative).parts
    top = parts[0] if parts else ""
    if top in DATASET_PATHS:
        who = f"the active profile {_ACTIVE.name} {_ACTIVE.version} does not" if _ACTIVE is not None else "no profile"
        raise DatasetDataMissing(
            f"triage needs {relative!r}, which only a dataset profile supplies, and {who} supplies {top!r}. "
            "Run with a dataset profile: load_triage_config(profile=...), or --profile on the scripts."
        )
    return PACKAGED_DATA / relative


_RUN_DATA: dict[str, Any] = {}
_RUN_DATA_DIGEST = ""


RUN_DATA_DATASET_FILES = frozenset({"airway_event_tokens.yaml", "discard_contested.yaml", "time_release.yaml"})
"""The dataset files read through :func:`data_values`, so a run's ``data`` section can change their values. The
task list, cohort conditions, lexicons and detector profiles are read whole, and a run changes them only through its
profile."""


def check_run_data(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """A run's ``data`` section, checked: every file it names is one :func:`data_values` reads.

    Args:
        data: ``{data path: mapping}``, or None.

    Returns:
        The section, as ``{data path: mapping}``.

    Raises:
        ValueError: If a path names a file whose values no loader merges (a run's change to it would be dropped while
            the config hash claimed it), or its values are not a mapping.
    """
    held = {str(relative): values for relative, values in (data or {}).items()}
    for relative, values in held.items():
        parts = PurePosixPath(relative).parts
        top = parts[0] if parts else ""
        merged = top in MERGED_PATHS or top in REPLACED_PATHS
        if not relative.endswith((".yaml", ".yml")) or not (merged or relative in RUN_DATA_DATASET_FILES):
            raise ValueError(
                f"the run's data section names {relative!r}, whose values no loader merges; it may name the value "
                f"files of {sorted(MERGED_PATHS | REPLACED_PATHS)} and {sorted(RUN_DATA_DATASET_FILES)}"
            )
        if len(parts) > 1 and not (PACKAGED_DATA / relative).is_file():
            raise ValueError(f"the run's data section names {relative!r}, which senselab does not package")
        if not isinstance(values, Mapping):
            raise ValueError(f"the run's data section for {relative!r} is not a mapping")
    return held


def set_run_data(data: Mapping[str, Any] | None) -> None:
    """Set the run's ``data`` section: per data path, the values the run changes (:func:`data_values`).

    Args:
        data: ``{data path: mapping}``, e.g. ``{"breath_pattern.yaml": {"train": {"distance_s": 0.5}}}``; None or
            empty clears it.

    Raises:
        ValueError: As :func:`check_run_data`; nothing is changed then.
    """
    global _RUN_DATA, _RUN_DATA_DIGEST
    held = check_run_data(data)
    _RUN_DATA = held
    _RUN_DATA_DIGEST = hashlib.sha256(repr(sorted(held.items())).encode()).hexdigest() if held else ""


def run_data() -> dict[str, Any]:
    """The run's ``data`` section (:func:`set_run_data`)."""
    return dict(_RUN_DATA)


def merge_values(base: Any, over: Any, *, where: str, fixed: frozenset[str] = frozenset()) -> Any:  # noqa: ANN401
    """Merge a value file's override over its packaged values, key by key.

    Args:
        base: The packaged values at this level.
        over: The override at this level.
        where: ``<file>:<dotted key>`` of this level, for messages.
        fixed: Top-level keys that may not change (:data:`FIXED_KEYS`).

    Returns:
        The merged values. A list or a scalar is replaced whole; a mapping is merged.

    Raises:
        ValueError: On a key the packaged mapping does not have (unless it leaves that mapping empty), a fixed key
            or a version changed.
    """
    if not isinstance(over, Mapping) or not isinstance(base, Mapping):
        return over
    out = dict(base)
    top = ":" not in where or where.endswith(":")
    for key, value in over.items():
        here = f"{where}{key}" if where.endswith(":") else f"{where}.{key}"
        if top and (key in _VERSION_KEYS or key in fixed):
            if key in base and value != base[key]:
                raise ValueError(f"{here} is fixed: {base[key]!r} may not become {value!r}")
            out[key] = base.get(key, value)
            continue
        if key not in base:
            if base:
                raise ValueError(f"{here} is not a key senselab's file has; an override may not introduce keys")
            out[key] = value
            continue
        out[key] = merge_values(base[key], value, where=here)
    return out


def _load(path: Path) -> Any:  # noqa: ANN401 -- a YAML document
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def data_values(relative: str) -> Any:  # noqa: ANN401 -- a YAML document
    """A YAML data file's values: senselab's, the active profile's and the run's, by the file's kind.

    Args:
        relative: A path under ``data/``, e.g. ``breath_pattern.yaml`` or ``speaker_vector_profile/2026-09-22.yaml``.

    Returns:
        For a value file (:data:`MERGED_PATHS`), the packaged values with each profile's copy merged over them,
        the profile it extends first, then the run's ``data`` section (:func:`merge_values`); for a replaced file,
        the nearest profile's copy whole where one supplies it, else the packaged file, and the run's section whole
        where it names one; for a dataset path, the nearest profile's copy with the run's section merged over it;
        for any other file, the packaged
        file.

    Raises:
        DatasetDataMissing: For a dataset path no active profile supplies.
        ValueError: Where an override breaks :func:`merge_values`'s rules.
    """
    parts = PurePosixPath(relative).parts
    top = parts[0] if parts else ""
    run = _RUN_DATA.get(relative)
    if top in MERGED_PATHS:
        packaged = PACKAGED_DATA / relative
        values = _load(packaged) if packaged.is_file() else {}
        fixed = FIXED_KEYS.get(relative, frozenset())
        for layer in _ACTIVE.chain() if _ACTIVE is not None else ():
            supplied = layer.supplies(relative)
            if supplied is not None:
                values = merge_values(values, _load(supplied), where=f"{relative}:", fixed=fixed)
        return merge_values(values, run, where=f"{relative}:", fixed=fixed) if run is not None else values
    if top in REPLACED_PATHS:
        return run if run is not None else _load(data_path(relative))
    if top in DATASET_PATHS:
        values = _load(data_path(relative))
        return merge_values(values, run, where=f"{relative}:") if run is not None else values
    return _load(PACKAGED_DATA / relative)


_R = TypeVar("_R")


def profile_cache(func: Callable[..., _R]) -> Callable[..., _R]:
    """Cache a loader's result per active profile and run data section, so neither is served another's data.

    Args:
        func: A loader taking hashable positional arguments.

    Returns:
        The cached loader, with ``cache_clear``.
    """

    @functools.lru_cache(maxsize=None)
    def cached(profile_key: tuple[str, str, str], *args: Any) -> _R:  # noqa: ANN401 -- the loader's own arguments
        return func(*args)

    @functools.wraps(func)
    def wrapper(*args: Any) -> _R:  # noqa: ANN401 -- the loader's own arguments
        key = (
            (str(_ACTIVE.root), _ACTIVE.digest, _RUN_DATA_DIGEST) if _ACTIVE is not None else ("", "", _RUN_DATA_DIGEST)
        )
        return cached(key, *args)

    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    return wrapper


REQUIRED_DATASET_PATHS = (
    "airway_event_tokens.yaml",
    "discard_contested.yaml",
    "tasks.yaml",
    "time_release.yaml",
)
"""Dataset paths every run reads; a profile must supply them."""

PROFILE_VALUES_FILE = "profile_values.yaml"
"""The packaged list of every value senselab leaves to a dataset profile, its kind and whether a run needs it."""

CONFIG_FILE = "config/default.yaml"
"""How :data:`PROFILE_VALUES_FILE` names the configuration's values."""


def profile_values() -> dict[str, list[dict[str, Any]]]:
    """Every value senselab leaves to a profile: ``{data file: [{path, kind, required}, ...]}``.

    A ``path`` is the list of keys from the file's top (a key may itself hold a dot, as the routing gates'
    ``voice.glide``).

    Returns:
        The list, as :data:`PROFILE_VALUES_FILE` holds it.
    """
    return dict((yaml.safe_load((PACKAGED_DATA / PROFILE_VALUES_FILE).read_text()) or {}).get("values") or {})


def required_profile_values() -> dict[str, tuple[tuple[str, ...], ...]]:
    """The profile values a run needs, by data file, each as its key path.

    Returns:
        ``{data file: (key path, ...)}``; the configuration's under :data:`CONFIG_FILE`.
    """
    return {
        relative: tuple(tuple(str(part) for part in entry["path"]) for entry in entries if entry.get("required"))
        for relative, entries in profile_values().items()
    }


def _lookup(node: Any, path: tuple[str, ...]) -> Any:  # noqa: ANN401 -- a YAML value
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def missing_dataset_values(config_values: Mapping[str, Any]) -> list[str]:
    """Every dataset value a run needs that neither the configuration nor the active profile supplies.

    Args:
        config_values: The merged configuration (``TriageConfig.values``).

    Returns:
        One description per missing value; empty when the run has everything it needs.
    """
    missing = []
    for relative in REQUIRED_DATASET_PATHS:
        try:
            if not data_path(relative).exists():
                missing.append(f"{relative} (the profile lists it, but the file is absent)")
        except DatasetDataMissing:
            missing.append(relative)
    for relative, paths in required_profile_values().items():
        try:
            document = config_values if relative == CONFIG_FILE else data_values(relative)
        except DatasetDataMissing:
            continue  # the file itself is listed missing above
        where = "config" if relative == CONFIG_FILE else relative
        missing += [f"{where} {'.'.join(path)}" for path in paths if _lookup(document, path) is None]
    if "tasks.yaml" not in missing and not (data_values("tasks.yaml").get("tasks") or {}):
        missing.append("tasks.yaml tasks (the task list names no task)")
    return missing


def require_dataset_values(config_values: Mapping[str, Any]) -> None:
    """Refuse to run when a dataset value is missing, naming every one, before any node runs.

    Args:
        config_values: The merged configuration (``TriageConfig.values``).

    Raises:
        DatasetDataMissing: If anything :func:`missing_dataset_values` lists is missing.
    """
    missing = missing_dataset_values(config_values)
    if missing:
        who = f"profile {_ACTIVE.name} {_ACTIVE.version}" if _ACTIVE is not None else "no dataset profile"
        raise DatasetDataMissing(
            f"triage cannot run with {who}: it needs these dataset values, which senselab does not ship: "
            + "; ".join(missing)
            + ". Supply them in a dataset profile (load_triage_config(profile=...), or --profile on the scripts)."
        )
