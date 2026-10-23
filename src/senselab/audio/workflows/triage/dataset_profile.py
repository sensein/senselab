"""A dataset profile: the data a dataset supplies for triage, and how triage finds it.

Triage reads its data files by their path under ``data/``. They come in three kinds:

- **Engine** files are schemas, vocabularies and settings about no dataset. They are always the packaged copy.
- **Dataset** paths (:data:`DATASET_PATHS`) hold what only a dataset can say: its task requirements, cohort
  conditions, lexicons and corpus-fitted distributions. senselab ships none of them; a profile supplies them,
  and without one :func:`data_path` raises :class:`DatasetDataMissing`.
- **Defaulted** files (:data:`DEFAULTED_PATHS`) are packaged with values defensible for any dataset. A profile may
  supply its own complete copy, which is then read instead.

A file comes from the profile or from the package, never from both: a profile supplies a listed path whole (a
directory with everything in it), and a path inside a supplied directory resolves inside the profile.

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

DATASET_PATHS = frozenset(
    {
        "airway_event_requirements.yaml",
        "airway_event_tokens.yaml",
        "cohort_conditions",
        "detector_profile",
        "discard_contested.yaml",
        "evaluation",
        "task_lexicon",
        "time_release.yaml",
    }
)
"""Data paths only a dataset profile supplies; senselab ships no copy."""

DEFAULTED_PATHS = frozenset(
    {
        "asr_degenerate.yaml",
        "breath_pattern.yaml",
        "cohort_stage.yaml",
        "cough_pattern.yaml",
        "decision_reasons.yaml",
        "discard_replaced_task.yaml",
        "quality_join.yaml",
        "session_background.yaml",
        "speaker_vector_profile",
        "task_content.yaml",
        "task_events.yaml",
        "task_extent.yaml",
        "task_minimum_duration.yaml",
        "task_speech.yaml",
        "word_timing.yaml",
    }
)
"""Data paths packaged with dataset-neutral defaults, which a profile may supply its own copy of."""

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
        digest: Hash of the manifest and of every byte of every file it supplies.
    """

    name: str
    version: str
    root: Path
    files: Mapping[str, Path] = field(default_factory=dict)
    config: Path | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""

    def resolve(self, relative: str) -> Path | None:
        """Where the profile supplies a data path, or None when it does not supply it.

        Args:
            relative: A path under ``data/``, e.g. ``task_lexicon/categories/animals.yaml``.

        Returns:
            The supplied path (which need not exist inside a supplied directory), or None.
        """
        parts = PurePosixPath(relative).parts
        if not parts or parts[0] not in self.files:
            return None
        return self.files[parts[0]].joinpath(*parts[1:])


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


def load_profile(directory: str | Path) -> DatasetProfile:
    """Read and check a dataset profile.

    Args:
        directory: The profile directory, holding :data:`MANIFEST`.

    Returns:
        The profile, with its digest.

    Raises:
        ValueError: If the manifest lacks a name or version, lists a path outside :data:`SUPPLIABLE_PATHS` (an
            engine file, a file inside a directory instead of the directory, or a typo), names a path that
            leaves its directory, supplies a file where a directory belongs or the reverse, or lists something
            that does not exist.
    """
    root = Path(directory).resolve()
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
    hasher = hashlib.sha256(manifest_path.read_bytes())
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


_R = TypeVar("_R")


def profile_cache(func: Callable[..., _R]) -> Callable[..., _R]:
    """Cache a loader's result per active profile, so a later profile is never served an earlier one's data.

    Args:
        func: A loader taking hashable positional arguments.

    Returns:
        The cached loader, with ``cache_clear``.
    """

    @functools.lru_cache(maxsize=None)
    def cached(profile_key: tuple[str, str] | None, *args: Any) -> _R:  # noqa: ANN401 -- the loader's own arguments
        return func(*args)

    @functools.wraps(func)
    def wrapper(*args: Any) -> _R:  # noqa: ANN401 -- the loader's own arguments
        key = (str(_ACTIVE.root), _ACTIVE.digest) if _ACTIVE is not None else None
        return cached(key, *args)

    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    return wrapper


REQUIRED_DATASET_PATHS = (
    "airway_event_requirements.yaml",
    "airway_event_tokens.yaml",
    "discard_contested.yaml",
    "time_release.yaml",
)
"""Dataset paths every run reads; a profile must supply them."""

REQUIRED_CONFIG_VALUES: tuple[tuple[str, ...], ...] = tuple(
    ("taxonomy", "ruleset", "gates", gate, "threshold")
    for gate in ("voice.glide", "voice.chant", "airway.breath", "airway.cough", "airway.ppg_silent_fraction")
)
"""Config values senselab ships unset because only a corpus can give them: the classifier-score routing gates."""

REQUIRED_DATA_VALUES: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "breath_pattern.yaml": (("veto", "active_fraction_min"),),
    "cohort_stage.yaml": (
        ("participant", "consistency_cosine_min"),
        ("condition", "hot_rms_dbfs_min"),
        ("condition", "clipped_fraction_min"),
        ("condition", "shift_level_db_min"),
        ("condition", "shift_snr_db_min"),
        ("shared_nonmatch", "fraction_min"),
    ),
    "task_extent.yaml": (("count_in", "gap_max_s"), ("atypical", "cough_event_s_max"), ("atypical", "ddk_rate_hz")),
}
"""Values in defaulted files that senselab ships unset (corpus quantiles and fences); a profile's copy sets them."""


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
    missing += [f"config {'.'.join(path)}" for path in REQUIRED_CONFIG_VALUES if _lookup(config_values, path) is None]
    for relative, paths in REQUIRED_DATA_VALUES.items():
        document = yaml.safe_load(data_path(relative).read_text()) or {}
        missing += [f"{relative} {'.'.join(path)}" for path in paths if _lookup(document, path) is None]
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
