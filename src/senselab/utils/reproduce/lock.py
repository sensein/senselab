"""Sources locks: one file per run listing the data, models, URLs, code and env locks its steps may use.

A manifest references exactly one lock (`spec.SourcesRef`), ``<base_dir>/sources/<name>.lock.yaml``, by
path and sha256; the lock's sha256 joins every step key. `load_lock` reads it, `step_findings` checks a
step's declared pins against it, `resolve_inputs` reads the registry entries of a step's ``lock_inputs``
and checks they are the ones the lock pinned, and `lock_findings` lints a lock alone.

Errors are a `LockError` whose message is a fixed code only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from senselab.utils.reproduce import inputs
from senselab.utils.reproduce.spec import (
    RESERVED_ROOT_NAMES,
    Commit,
    HFFile,
    HFRepo,
    Name,
    RelPath,
    Sha256,
    SourcesRef,
    StepSpec,
)

LOCK_SCHEMA = "reproduce-sources-lock-1"
SOURCES_DIR = "sources"
CODES = frozenset({"E_LOCK_SHA", "E_LOCK_INVALID", "E_LOCK_REF", "E_LOCK_MISMATCH", "E_LOCK_STALE"})


class LockError(Exception):
    """A sources lock could not be used; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: One of `CODES`, or ``E_INPUT_REGISTRY``.
        """
        super().__init__(code)
        self.code = code


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisteredEntry(_Model):
    """A registered input, pinned by the sha256 of its registry file and of its manifest."""

    id: Name
    kind: Literal["registered"]
    registry_sha256: Sha256
    manifest_sha256: Sha256


class PatternEntry(_Model):
    """Pattern inputs under a profile root; they stay content-digested at run time."""

    kind: Literal["pattern"]
    root: Name
    patterns: list[str] = Field(min_length=1)

    _grammar = field_validator("patterns")(inputs.check_patterns)


class ModelEntry(_Model):
    """Hugging Face weights: a repository at a 40-hex commit and the sha256 of each file."""

    id: Name
    repo: HFRepo
    repo_type: Literal["model", "dataset"] = "model"
    revision: Commit
    files: list[HFFile] = Field(min_length=1)


class UrlEntry(_Model):
    """A file fetched from a URL, pinned by its sha256."""

    file: RelPath
    sha256: Sha256


class CodeEntry(_Model):
    """A repository at a commit, with the sha256 of each patch in order and each submodule's commit."""

    repo: str
    commit: Commit
    patches: list[Sha256] = []
    submodules: dict[str, Commit] = {}


class EnvEntry(_Model):
    """An env lock file of the manifest repository and its sha256."""

    lock: RelPath
    sha256: Sha256


DataEntry = Annotated[RegisteredEntry | PatternEntry, Field(discriminator="kind")]


class SourcesLock(_Model):
    """What a run may use: data and external inputs, models, URL files, code and env locks."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lock_schema: Literal["reproduce-sources-lock-1"] = Field(alias="schema")
    data: list[DataEntry] = []
    external: list[RegisteredEntry] = []
    models: list[ModelEntry] = []
    urls: list[UrlEntry] = []
    code: list[CodeEntry] = []
    envs: list[EnvEntry] = []

    @model_validator(mode="after")
    def _unique(self) -> SourcesLock:
        ids = [e.id for e in self.registered()]
        if len(set(ids)) != len(ids):
            raise ValueError("registered input ids repeat")
        models = [m.id for m in self.models]
        if len(set(models)) != len(models):
            raise ValueError("model ids repeat")
        envs = [e.lock for e in self.envs]
        if len(set(envs)) != len(envs):
            raise ValueError("env locks repeat")
        return self

    def registered(self) -> list[RegisteredEntry]:
        """The registered entries of ``data`` and ``external``."""
        return [e for e in self.data if isinstance(e, RegisteredEntry)] + list(self.external)


def parse_lock(data: bytes) -> SourcesLock:
    """Validate a lock from its bytes.

    Raises:
        LockError: ``E_LOCK_INVALID`` when the bytes are not a valid lock.
    """
    try:
        return SourcesLock.model_validate(yaml.safe_load(data.decode()))
    except (UnicodeDecodeError, ValueError, yaml.YAMLError, pydantic.ValidationError):
        raise LockError("E_LOCK_INVALID") from None


def load_lock(base_dir: Path, ref: SourcesRef) -> tuple[SourcesLock, str]:
    """Read the lock a manifest references.

    Args:
        base_dir: The manifest repository.
        ref: The manifest's ``sources``.

    Returns:
        ``(lock, sha256)``.

    Raises:
        LockError: ``E_LOCK_SHA`` for a missing or unreadable file or bytes whose sha256 is not
            ``ref.sha256``; ``E_LOCK_INVALID`` as `parse_lock`.
    """
    try:
        data = (Path(base_dir) / ref.lock).read_bytes()
    except OSError:
        raise LockError("E_LOCK_SHA") from None
    sha = hashlib.sha256(data).hexdigest()
    if sha != ref.sha256:
        raise LockError("E_LOCK_SHA")
    return parse_lock(data), sha


def _env_refs(step: StepSpec) -> list[tuple[str, str]]:
    env = step.env
    refs = [env.torch, env.body, env.wheels, env.test, env.notebook_tools, *(s.build_env for s in env.sdist_allow)]
    return [(r.lock, r.lock_sha256) for r in refs if r is not None]


def step_findings(
    step: StepSpec, lock: SourcesLock, submodules: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """Check a step's declared pins against a lock; nothing is read from disk.

    Args:
        step: The step.
        lock: The manifest's lock.
        submodules: The checkout's submodule commits, when known; compared with the code entry's.

    Returns:
        ``(code, detail)`` pairs, the detail naming the step field. ``E_LOCK_REF``: a ``lock_inputs`` id
        that is no registered entry, a data-root input or ``stage`` source whose patterns no pattern entry of
        its root lists, or ``hf_weights`` with no models entry of the same repository, type and revision
        carrying each declared file with its sha256 (the step may declare a subset of the entry's files), or a
        ``url_weights`` file with no ``urls`` entry of the same file and sha256.
        ``E_LOCK_MISMATCH``: no code entry with the step's repository, commit and patch sha256s (and
        submodules, when given), or an env lock missing from ``envs`` or pinned there with another sha256.
    """
    found: list[tuple[str, str]] = []
    registered = {e.id for e in lock.registered()}
    found += [("E_LOCK_REF", f"lock_inputs {i}") for i in step.lock_inputs if i not in registered]
    listed = {(e.root, p) for e in lock.data if isinstance(e, PatternEntry) for p in e.patterns}
    sourced = [("inputs", i, s.root, s.patterns) for i, s in enumerate(step.inputs)]
    sourced += [("stage", i, s.root, s.patterns) for i, s in enumerate(step.stage)]
    for field, index, root, patterns in sourced:
        data_root = not root.startswith("up:") and root not in RESERVED_ROOT_NAMES
        if data_root and any((root, p) not in listed for p in patterns):
            found.append(("E_LOCK_REF", f"{field}[{index}]"))
    for w in step.hf_weights:
        pinned = [
            {f.path: f.sha256 for f in m.files}
            for m in lock.models
            if (m.repo, m.repo_type, m.revision) == (w.repo, w.repo_type, w.revision)
        ]
        if not any(all(files.get(f.path) == f.sha256 for f in w.files) for files in pinned):
            found.append(("E_LOCK_REF", f"hf_weights {w.repo}"))
    pinned_urls = {(u.file, u.sha256) for u in lock.urls}
    found += [
        ("E_LOCK_REF", f"url_weights {w.file}") for w in step.url_weights if (w.file, w.sha256) not in pinned_urls
    ]
    patches = [p.sha256 for p in step.patches]
    code = [c for c in lock.code if (c.repo, c.commit, c.patches) == (step.repo, step.commit, patches)]
    if not code or (submodules is not None and all(c.submodules != dict(submodules) for c in code)):
        found.append(("E_LOCK_MISMATCH", "code"))
    envs = {e.lock: e.sha256 for e in lock.envs}
    found += [("E_LOCK_MISMATCH", f"env {path}") for path, sha in _env_refs(step) if envs.get(path) != sha]
    return found


def resolve_inputs(base_dir: Path, lock: SourcesLock, ids: Iterable[str]) -> dict[str, inputs.RegisteredInput]:
    """Read the registry entries of ``ids`` and check they are the entries the lock pinned.

    Args:
        base_dir: The manifest repository.
        lock: The manifest's lock.
        ids: Registered input ids (a step's ``lock_inputs``).

    Returns:
        Each id's registry entry. No file of an input is opened.

    Raises:
        LockError: ``E_LOCK_REF`` for an id the lock does not register; ``E_LOCK_STALE`` when the registry
            file's sha256 or its manifest's sha256 is not the one the lock pinned; ``E_INPUT_REGISTRY`` when
            the entry cannot be read (`inputs.read_registry_entry`).
    """
    pinned = {e.id: e for e in lock.registered()}
    out: dict[str, inputs.RegisteredInput] = {}
    for input_id in ids:
        if input_id not in pinned:
            raise LockError("E_LOCK_REF")
        try:
            entry, sha = inputs.read_registry_entry(base_dir, input_id)
        except inputs.InputError as error:
            raise LockError(error.code) from None
        if sha != pinned[input_id].registry_sha256 or entry.manifest.sha256 != pinned[input_id].manifest_sha256:
            raise LockError("E_LOCK_STALE")
        out[input_id] = entry
    return out


def lock_findings(base_dir: Path, path: Path) -> list[tuple[str, str]]:
    """Lint one lock file alone: it validates, and every registered entry is current.

    Args:
        base_dir: The manifest repository.
        path: The lock file.

    Returns:
        ``(code, detail)`` pairs: ``E_LOCK_SHA`` (unreadable), ``E_LOCK_INVALID``, and ``E_LOCK_STALE`` or
        ``E_INPUT_REGISTRY`` with the input id.
    """
    try:
        lock = parse_lock(Path(path).read_bytes())
    except OSError:
        return [("E_LOCK_SHA", "unreadable")]
    except LockError as error:
        return [(error.code, "schema")]
    found = []
    for entry in lock.registered():
        try:
            resolve_inputs(base_dir, lock, [entry.id])
        except LockError as error:
            found.append((error.code, entry.id))
    return found
