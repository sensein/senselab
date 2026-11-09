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

    name: my-dataset                 # required
    version: "1"                     # required
    provenance: {...}                # optional, free-form: what the values were fitted on
    config: config.yaml              # optional: a config override, merged between the packaged config and a run's
    files:                           # data path -> path relative to the profile directory
      recruited_conditions: recruited_conditions
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
from typing import Any, Callable, Collection, Mapping, Sequence, TypeVar

import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.workflow import STAGE_NAMES, WORKFLOW_SETTING, off_notes

PACKAGED_DATA = Path(__file__).parent / "data"
"""The packaged data directory: the engine files and the defaulted files."""

MANIFEST = "profile.yaml"
"""The file that makes a directory a dataset profile."""

FITTED_FILE = "fitted.yaml"
"""The file in a profile that ``scripts/fit_profile.py`` writes: the corpus statistics fitted on a pilot run and what
they were fitted on (plan step 10). Its values sit above the profile it extends and below the profile's own files."""

PILOT_KEY = "pilot"
"""The manifest key that marks a profile as a pilot: its corpus statistics (kind b) may be empty (plan step 10)."""

PACKAGED_PROFILES = Path(__file__).parent / "profiles"
"""Profiles senselab ships, opt-in and by name: a profile builds on one with ``extends: senselab:<name>``."""

SENSELAB_PROFILE = "senselab:"

DATASET_PATHS = frozenset(
    {
        "airway_event_tokens.yaml",
        "recruited_conditions",
        "detector_profile",
        "discard_contested.yaml",
        "evaluation",
        "redaction_policy.yaml",
        "review_policy.txt",
        "task_guidance.yaml",
        "task_lexicon",
        "tasks.yaml",
        "time_release.yaml",
    }
)
"""Data paths only a dataset profile supplies; senselab ships no copy (``redaction_policy.yaml``, the redaction word
lists, comes with the reference profile ``adult_clinical_speech``)."""

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

_AS_BASE = object()
""":func:`merge_values`' default for ``packaged``: judge a new key against the merged lower layers."""

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
        pilot: Whether the manifest marks it a pilot (:data:`PILOT_KEY`): its corpus statistics may be empty until
            ``scripts/fit_profile.py`` fits them.
        fitted: The values its :data:`FITTED_FILE` holds, by data file; empty where it has none.
    """

    name: str
    version: str
    root: Path
    files: Mapping[str, Path] = field(default_factory=dict)
    config: Path | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    digest: str = ""
    base: "DatasetProfile | None" = None
    pilot: bool = False
    fitted: Mapping[str, Any] = field(default_factory=dict)

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

        A file inside a supplied directory is the nearest layer's that holds it: two layers may each supply a
        directory, each with the files it changes. Where no layer holds it, the nearest layer's (absent) path.

        Args:
            relative: A path under ``data/``.

        Returns:
            The supplied path, or None.
        """
        nearest = None
        for layer in reversed(self.chain()):
            own = layer.supplies(relative)
            if own is None:
                continue
            if len(PurePosixPath(relative).parts) == 1 or own.exists():
                return own
            nearest = nearest or own
        return nearest

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
        expect_dir = not relative.endswith((".yaml", ".yml", ".json", ".parquet", ".txt"))
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
    pilot = manifest.get(PILOT_KEY, False)
    if not isinstance(pilot, bool):
        raise ValueError(f"{manifest_path}: {PILOT_KEY} is true or false, got {pilot!r}")
    fitted = load_fitted(root / FITTED_FILE)
    hasher = hashlib.sha256(manifest_path.read_bytes())
    if (root / FITTED_FILE).is_file():
        hasher.update(b"\0fitted\0" + hashlib.sha256((root / FITTED_FILE).read_bytes()).hexdigest().encode())
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
        pilot=pilot,
        fitted=fitted,
    )


UNLISTED_SKIPPED = ("__pycache__",)
"""Directories under a profile whose files :func:`unlisted_files` leaves out: the interpreter's byte-code caches."""


def unlisted_files(profile: DatasetProfile) -> dict[str, Any]:
    """The files in a profile's directories that its manifest does not list, and their digest.

    These are what a profile carries beside its data: its ``hints.py`` and whatever resources that reads (a task
    registry's copy). Triage reads none of them, so :attr:`DatasetProfile.digest` leaves them out and a change to one
    is no profile change; a run records this digest beside it, so a reader can tell which hint builder it ran with.

    Args:
        profile: The profile; every layer of its chain is read.

    Returns:
        ``{"digest": <16 hex>, "files": ["<layer name>/<path>", ...]}``, the outermost base's first; an empty list
        where every file is listed.
    """
    hasher = hashlib.sha256()
    names = []
    for layer in profile.chain():
        listed = {layer.root / MANIFEST, layer.root / FITTED_FILE, *layer.files.values()}
        if layer.config is not None:
            listed.add(layer.config)
        for path in sorted(layer.root.rglob("*")):
            inside = path.relative_to(layer.root)
            if not path.is_file() or path.suffix == ".pyc" or set(inside.parts) & set(UNLISTED_SKIPPED):
                continue
            if any(path == target or target in path.parents for target in listed):
                continue
            name = f"{layer.name}/{inside.as_posix()}"
            names.append(name)
            hasher.update(f"\0{name}\0{hashlib.sha256(path.read_bytes()).hexdigest()}".encode())
    return {"digest": hasher.hexdigest()[:16], "files": names}


def load_fitted(path: Path) -> dict[str, Any]:
    """The values a profile's :data:`FITTED_FILE` holds, by data file.

    Args:
        path: The file; it need not exist.

    Returns:
        ``{data file: values}``; empty where the file is absent.

    Raises:
        ValueError: If the file holds no ``values`` mapping, or names a data file that is not a value file
            (:data:`MERGED_PATHS`).
    """
    if not path.is_file():
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = document.get("values") if isinstance(document, Mapping) else None
    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must hold a values mapping, by data file")
    for relative, held in values.items():
        if relative not in MERGED_PATHS or not isinstance(held, Mapping):
            raise ValueError(f"{path}: {relative!r} is not a value file a fit writes ({sorted(MERGED_PATHS)})")
    return {str(relative): dict(held) for relative, held in values.items()}


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
        The active profile's path where it supplies one (:meth:`DatasetProfile.resolve`), unless the profile's copy is
        absent and senselab packages the file; otherwise the packaged path.

    Raises:
        DatasetDataMissing: If the path is a dataset path (:data:`DATASET_PATHS`) and no active profile supplies
            it.
    """
    supplied = _ACTIVE.resolve(relative) if _ACTIVE is not None else None
    if supplied is not None and (supplied.exists() or not (PACKAGED_DATA / relative).exists()):
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


def newest_member(directory: str) -> str:
    """The newest dated file of a value directory, across senselab's copy and every layer of the active profile.

    Args:
        directory: A supplied directory under ``data/``, e.g. ``speaker_vector_profile``, holding ``<date>.yaml``
            files.

    Returns:
        ``<directory>/<newest name>``, for :func:`data_values` and :func:`data_path`.

    Raises:
        FileNotFoundError: If neither senselab nor any layer holds one.
    """
    folders = [PACKAGED_DATA / directory]
    folders += [supplied for layer in (_ACTIVE.chain() if _ACTIVE else ()) if (supplied := layer.supplies(directory))]
    names = sorted({path.name for folder in folders if folder.is_dir() for path in folder.glob("*.yaml")})
    if not names:
        raise FileNotFoundError(f"no {directory} file in {[str(folder) for folder in folders]}")
    return f"{directory}/{names[-1]}"


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


def merge_values(
    base: Any,  # noqa: ANN401
    over: Any,  # noqa: ANN401
    *,
    where: str,
    fixed: frozenset[str] = frozenset(),
    open_mappings: frozenset[str] = frozenset(),
    packaged: Any = _AS_BASE,  # noqa: ANN401 -- a YAML value
) -> Any:  # noqa: ANN401
    """Merge a value file's override over its packaged values, key by key.

    Args:
        base: The values at this level so far: the packaged file's, with every lower layer merged over them.
        over: The override at this level.
        where: ``<file>:<dotted key>`` of this level, for messages.
        fixed: Top-level keys that may not change (:data:`FIXED_KEYS`).
        open_mappings: ``<file>:<dotted key>`` of mappings keyed by task family, which take keys the lower layers
            lack (:func:`open_mappings`).
        packaged: The packaged file's values at this level, which say whether senselab leaves a mapping empty: one
            it leaves empty takes new keys from every layer, though a lower layer has filled it. Omitted, ``base``
            stands in for them.

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
    left_empty = packaged is not _AS_BASE and isinstance(packaged, Mapping) and not packaged
    for key, value in over.items():
        here = f"{where}{key}" if where.endswith(":") else f"{where}.{key}"
        if top and (key in _VERSION_KEYS or key in fixed):
            if key in base and value != base[key]:
                raise ValueError(f"{here} is fixed: {base[key]!r} may not become {value!r}")
            out[key] = base.get(key, value)
            continue
        if key not in base:
            if base and where not in open_mappings and not left_empty:
                raise ValueError(f"{here} is not a key senselab's file has; an override may not introduce keys")
            out[key] = value
            continue
        below = _AS_BASE
        if packaged is not _AS_BASE:
            below = packaged.get(key, _AS_BASE) if isinstance(packaged, Mapping) else _AS_BASE
        out[key] = merge_values(base[key], value, where=here, open_mappings=open_mappings, packaged=below)
    return out


def _yield_to_own(fitted: Any, own: Any, *, where: str, open_mappings: frozenset[str]) -> Any:  # noqa: ANN401
    """A layer's fitted values without the per-family mappings its own file sets: there, its own mapping wins whole.

    Args:
        fitted: The layer's fitted values at this level (:data:`FITTED_FILE`).
        own: The layer's own file's values at this level.
        where: ``<file>:<dotted key>`` of this level, as :func:`merge_values` names it.
        open_mappings: :func:`open_mappings`.

    Returns:
        ``fitted``, less each mapping of :func:`open_mappings` the own file also sets.
    """
    if not isinstance(fitted, Mapping) or not isinstance(own, Mapping):
        return fitted
    out = {}
    for key, value in fitted.items():
        here = f"{where}{key}" if where.endswith(":") else f"{where}.{key}"
        if key in own and here in open_mappings:
            continue
        out[key] = _yield_to_own(value, own[key], where=here, open_mappings=open_mappings) if key in own else value
    return out


@functools.lru_cache(maxsize=None)
def open_mappings() -> frozenset[str]:
    """The value mappings keyed by task family (a fitted method's ``by``), which a profile may add families to.

    Returns:
        ``<file>:<dotted key>`` of each, e.g. ``discard_replaced_task.yaml:short_response.family_median_s``.
    """
    return frozenset(
        f"{relative}:{'.'.join(str(part) for part in entry['path'])}"
        for relative, entries in profile_values().items()
        for entry in entries
        if isinstance(entry.get("method"), Mapping) and entry["method"].get("by")
    )


def _load(path: Path) -> Any:  # noqa: ANN401 -- a YAML document
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def data_values(relative: str) -> Any:  # noqa: ANN401 -- a YAML document
    """A YAML data file's values: senselab's, the active profile's and the run's, by the file's kind.

    Args:
        relative: A path under ``data/``, e.g. ``breath_pattern.yaml`` or ``speaker_vector_profile/2026-09-22.yaml``.

    Returns:
        For a value file (:data:`MERGED_PATHS`), the packaged values with each profile's fitted values
        (:data:`FITTED_FILE`) and then its own copy merged over them, the profile it extends first, then the run's
        ``data`` section (:func:`merge_values`); for a replaced file,
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
        packaged_file = PACKAGED_DATA / relative
        packaged = _load(packaged_file) if packaged_file.is_file() else {}
        values = packaged
        rules: dict[str, Any] = {
            "where": f"{relative}:",
            "fixed": FIXED_KEYS.get(relative, frozenset()),
            "open_mappings": open_mappings(),
            "packaged": packaged if packaged_file.is_file() else _AS_BASE,
        }
        for layer in _ACTIVE.chain() if _ACTIVE is not None else ():
            supplied = layer.supplies(relative)
            own = _load(supplied) if supplied is not None and supplied.is_file() else None
            if relative in layer.fitted:
                fitted = layer.fitted[relative]
                if own is not None:
                    fitted = _yield_to_own(fitted, own, where=f"{relative}:", open_mappings=rules["open_mappings"])
                values = merge_values(values, fitted, **rules)
            if own is not None:
                values = merge_values(values, own, **rules)
        if run is None:
            return values
        return merge_values(values, run, **rules)
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
    "redaction_policy.yaml",
    "tasks.yaml",
    "time_release.yaml",
)
"""Dataset paths every run reads; a profile must supply them. The verdict folds the redaction mask plan whatever
runs, so the redaction word lists are among them."""

TEMPLATE_MARKER = b"TEMPLATE:"
"""What a template's file begins with until a dataset writes its own, after ``# `` in a YAML file; the start check
refuses it."""


def is_template(path: Path) -> bool:
    """Whether a file is still a template: its first line begins with :data:`TEMPLATE_MARKER`, as a comment or not.

    Args:
        path: The file.

    Returns:
        True while the dataset has not written its own.
    """
    first = path.read_bytes().split(b"\n", 1)[0]
    return first.startswith(TEMPLATE_MARKER) or first.lstrip(b"#").lstrip().startswith(TEMPLATE_MARKER)


STAGE_DATASET_PATHS: Mapping[str, tuple[str, ...]] = {
    "review_policy.txt": ("REVIEW", "SECOND_OPINION"),
    "task_guidance.yaml": ("REVIEW", "SECOND_OPINION"),
}
"""Dataset paths only some stages read, and which: a run needs one only when one of its stages runs (step 6b)."""


def stage_dataset_paths(running: Collection[str]) -> tuple[str, ...]:
    """The dataset paths of :data:`STAGE_DATASET_PATHS` a run driving ``running`` needs.

    Args:
        running: The stages the run drives.

    Returns:
        The paths one of whose stages runs, in the table's order.
    """
    return tuple(path for path, stages in STAGE_DATASET_PATHS.items() if any(stage in running for stage in stages))


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


FITTED_KIND = "b"
"""The kind of a profile value computed from the dataset's own recordings, which a pilot run leaves empty."""


def needed(entry: Mapping[str, Any], running: Collection[str] | None, *, pilot: bool = False) -> bool:
    """Whether a run that drives ``running`` needs a profile value.

    Args:
        entry: The value's entry in :data:`PROFILE_VALUES_FILE`.
        running: The stages the run drives (``workflow.running_stages``), or None for every stage.
        pilot: Whether the run's profile is a pilot, which needs no corpus statistic (:data:`FITTED_KIND`).

    Returns:
        True when the entry is required and names no stage, or names one that runs; for a pilot, never for a
        corpus statistic.
    """
    if not entry.get("required") or (pilot and entry.get("kind") == FITTED_KIND):
        return False
    stages = entry.get("stages") or ()
    return running is None or not stages or any(stage in running for stage in stages)


def required_profile_values(
    running: Collection[str] | None = None, *, pilot: bool = False
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """The profile values a run needs, by data file, each as its key path.

    Args:
        running: The stages the run drives; a value whose ``stages`` are all off is not needed. None: every stage.
        pilot: Whether the run's profile is a pilot (:func:`needed`).

    Returns:
        ``{data file: (key path, ...)}``; the configuration's under :data:`CONFIG_FILE`.
    """
    return {
        relative: tuple(
            tuple(str(part) for part in entry["path"]) for entry in entries if needed(entry, running, pilot=pilot)
        )
        for relative, entries in profile_values().items()
    }


def unfitted_values(running: Collection[str] | None = None) -> list[str]:
    """The corpus statistics a pilot run leaves empty: ``<data file> <dotted key path>`` each.

    Args:
        running: The stages the run drives, or None for every stage.

    Returns:
        Every required corpus statistic (:data:`FITTED_KIND`) a run would need that the active profile leaves
        empty; empty where the profile is no pilot or has them all.
    """
    if _ACTIVE is None or not _ACTIVE.pilot:
        return []
    out = []
    for relative, entries in profile_values().items():
        wanted = [entry for entry in entries if entry.get("kind") == FITTED_KIND and needed(entry, running)]
        if relative == CONFIG_FILE or not wanted:
            continue
        document = data_values(relative)
        for entry in wanted:
            path = tuple(str(part) for part in entry["path"])
            if lookup_value(document, path) is None:
                out.append(f"{relative} {'.'.join(path)}")
    return out


def lookup_value(node: Any, path: Sequence[str]) -> Any:  # noqa: ANN401 -- a YAML value
    """The value at a key path in a YAML document, or None where any key is missing.

    Args:
        node: The document.
        path: The keys from its top.

    Returns:
        The value, or None.
    """
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
    off = off_notes(config_values)
    running = frozenset(stage for stage in STAGE_NAMES if stage not in off)
    for relative in (*REQUIRED_DATASET_PATHS, *stage_dataset_paths(running)):
        try:
            path = data_path(relative)
        except DatasetDataMissing:
            missing.append(relative)
            continue
        if not path.exists():
            missing.append(f"{relative} (the profile lists it, but the file is absent)")
        elif path.is_file() and is_template(path):
            missing.append(f"{relative} (still the template: write your own)")
    pilot = _ACTIVE is not None and _ACTIVE.pilot
    for relative, paths in required_profile_values(running, pilot=pilot).items():
        try:
            document = config_values if relative == CONFIG_FILE else data_values(relative)
        except DatasetDataMissing:
            continue  # the file itself is listed missing above
        except ValueError:
            continue  # a file that does not merge is listed below (_value_file_problems)
        where = "config" if relative == CONFIG_FILE else relative
        missing += [f"{where} {'.'.join(path)}" for path in paths if lookup_value(document, path) is None]
    recruited = (config_values.get("verdict") or {}).get("recruited_conditions")
    if recruited:
        try:
            named = data_path("recruited_conditions") / f"{recruited}.yaml"
        except DatasetDataMissing:
            named = None
        if named is None or not named.is_file():
            missing.append(f"recruited_conditions/{recruited}.yaml (config verdict.recruited_conditions names it)")
    if not any(item == "tasks.yaml" or item.startswith("tasks.yaml (") for item in missing):
        tasks = data_values("tasks.yaml").get("tasks") or {}
        if not tasks:
            missing.append("tasks.yaml tasks (the task list names no task)")
        for name, entry in tasks.items():
            branch = entry.get("branch") if isinstance(entry, Mapping) else None
            if branch in off:
                missing.append(f"tasks.yaml task {name}: its branch {branch} is {off[branch]} ({WORKFLOW_SETTING})")
        missing += _task_list_problems(config_values)
    if not any(item.split()[0] == "redaction_policy.yaml" for item in missing):
        from senselab.text.tasks.pii_detection.redaction_policy import lexicon_problems  # noqa: PLC0415

        missing += [
            f"redaction_policy.yaml ({problem})" for problem in lexicon_problems(data_path("redaction_policy.yaml"))
        ]
    return missing + _value_file_problems()


def _declared_file(relative: str, key: str, expected: str, where: str) -> list[str]:
    """Why a task list's data file cannot be read for what names it, if it cannot.

    Args:
        relative: The file's data path, as the task list gives it.
        key: The key the file declares its owner under (``family``, ``category``).
        expected: The owner it must declare: the task, or the category.
        where: The task list entry, for the message.

    Returns:
        One description, or none where the file is there and declares ``expected``.
    """
    try:
        path = data_path(relative)
    except DatasetDataMissing:
        return [f"{where}: {relative} (no profile supplies it)"]
    if not path.is_file():
        return [f"{where}: {relative} (the file is absent)"]
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    declared = document.get(key) if isinstance(document, Mapping) else None
    if str(declared) != expected:
        return [f"{where}: {relative} declares {key} {declared!r}, not {expected!r}: each names its own file"]
    return []


def _task_list_problems(config_values: Mapping[str, Any]) -> list[str]:
    """What in the task list, or keyed by its tasks and kinds, would fail only once a recording reads it.

    Args:
        config_values: The merged configuration.

    Returns:
        One description per problem: a task list that does not parse; a task's lexicon or a list category's words
        file that is absent or declares another task or category (so two tasks sharing one lexicon); a task whose
        fixed ``item_category`` is not one of its categories; a configured per-task gate or flag naming no task, or a
        gate triage does not have; a task-content kind triage does not know.
    """
    from senselab.audio.workflows.triage.nodes.gates import GATE_SPECS  # noqa: PLC0415
    from senselab.audio.workflows.triage.task_catalogue import KINDS, task_list  # noqa: PLC0415

    try:
        tasks = task_list()
    except ValueError as error:
        return [f"tasks.yaml (it does not parse: {error})"]
    problems: list[str] = []
    for name, entry in tasks.items():
        where = f"tasks.yaml task {name}"
        if entry.lexicon is not None:
            problems += _declared_file(entry.lexicon, "family", name, f"{where} lexicon")
        for category in entry.categories.values():
            if category.words is not None:
                problems += _declared_file(category.words, "category", category.name, f"{where} category words")
        fixed = entry.expects.item_category
        if fixed is not None and fixed not in entry.categories:
            problems.append(f"{where}: expects.item_category {fixed!r} is not one of its categories")
    verdict = config_values.get("verdict") or {}
    by_family = (verdict.get("gates") or {}).get("by_family") or {}
    for path, keyed in (
        ("verdict.gates.by_family", by_family),
        ("verdict.conformance_flags_by_family", verdict.get("conformance_flags_by_family") or {}),
    ):
        unknown = sorted(str(key) for key in keyed if key not in tasks)
        if unknown:
            problems.append(f"config {path} names tasks the task list lacks: {unknown}")
    for task, gates in by_family.items():
        unknown = sorted(str(gate) for gate in (gates or {}) if gate not in GATE_SPECS)
        if unknown:
            problems.append(f"config verdict.gates.by_family.{task} names gates triage does not have: {unknown}")
    try:
        content = data_values("task_content.yaml")
    except ValueError:
        return problems  # a file that does not merge is listed by _value_file_problems
    for key in ("families", "any_word_families"):
        unknown = sorted(str(kind) for kind in content.get(key) or () if kind not in KINDS)
        if unknown:
            problems.append(f"task_content.yaml {key} names kinds triage does not know: {unknown} (of {list(KINDS)})")
    return problems


def _value_file_problems() -> list[str]:
    """Every value file the profile chain or the run's ``data`` section changes that does not merge.

    A key senselab's file lacks, or a fixed one changed, would otherwise raise only once a node reads the file.

    Returns:
        ``<data path> (<why>)`` per file that does not merge (:func:`data_values`).
    """
    paths = set(_RUN_DATA)
    for layer in _ACTIVE.chain() if _ACTIVE is not None else ():
        paths |= set(layer.fitted)
        for relative, supplied in layer.files.items():
            if relative not in MERGED_PATHS:
                continue
            if supplied.is_dir():
                paths |= {f"{relative}/{member.name}" for member in supplied.glob("*.yaml")}
            elif relative.endswith((".yaml", ".yml")):
                paths.add(relative)
    problems = []
    for relative in sorted(paths):
        try:
            data_values(relative)
        except (ValueError, OSError) as error:
            problems.append(f"{relative} ({error})")
    return problems


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
