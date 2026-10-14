"""Pinned weights: Hugging Face snapshots in the profile's hub cache, checked by sha256, and the per-step hub view.

A step's ``hf_weights`` name repositories at 40-hex revisions and the sha256 of every file the step reads.
The files live in the profile's read-only ``hf_hub_cache`` (Hugging Face layout:
``<type>s--<org>--<name>/snapshots/<revision>/<path>``, each a link into ``blobs/``).

`check_hf_present` is the cheap check the inputs gate makes (``E_WEIGHTS_MISSING``). `verify_hf` is the
prefetch check: each declared file once per (type, repository, revision, path), and each blob hashed at most
once. A file is accepted without hashing when its step names the registered model-cache input that holds it
(``HFWeights.input``), that input's manifest row for the blob carries the declared sha256, the input's latest
verification record is current and ``ok`` (`inputs.accepted`), and the blob's ``lstat`` size and
``mtime_ns`` equal the row (or the record's overlay). `build_hub_view` lays out the step's own hub cache
under its ``{tmp}``: ``snapshots`` and ``blobs`` link to the profile's copy, ``refs/main`` names the
declared revision.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from senselab.utils.dependencies import _point_ref_at, is_hf_model_cached
from senselab.utils.reproduce import inputs
from senselab.utils.reproduce.spec import HFWeights, Profile, StepSpec

WEIGHTS_MISSING = "E_WEIGHTS_MISSING"
WEIGHTS_MISMATCH = "E_WEIGHTS_MISMATCH"
WEIGHTS_VIEW = "E_WEIGHTS_VIEW"
CONFIG = "config.json"
_CHUNK = 8 << 20


class WeightsError(Exception):
    """A hub view that could not be built; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``."""
        super().__init__(code)
        self.code = code


def repo_dir_name(repo: str, repo_type: str = "model") -> str:
    """``<type>s--<org>--<name>``, a repository's directory in a hub cache."""
    org, name = repo.split("/")
    return f"{repo_type}s--{org}--{name}"


def hf_snapshot(hub: Path, w: HFWeights) -> Path:
    """``<hub>/<type>s--<org>--<name>/snapshots/<revision>``.

    Args:
        hub: A hub cache.
        w: The declared weights.

    Returns:
        The snapshot dir.
    """
    return hub / repo_dir_name(w.repo, w.repo_type) / "snapshots" / w.revision


def check_hf_present(step: StepSpec, profile: Profile) -> list[str]:
    """Whether every declared file of the step exists in its snapshot of the profile's hub cache; no hashing.

    Args:
        step: The step.
        profile: The site profile.

    Returns:
        ``["E_WEIGHTS_MISSING"]`` when a file is missing (a dangling link counts as missing), or the profile has
        no ``hf_hub_cache``; empty otherwise, and for a step with no ``hf_weights``.
    """
    if not step.hf_weights:
        return []
    hub = profile.hf_hub_cache
    if hub is None:
        return [WEIGHTS_MISSING]
    for w in step.hf_weights:
        snapshot = hf_snapshot(hub, w)
        if not all((snapshot / f.path).is_file() for f in w.files):
            return [WEIGHTS_MISSING]
    return []


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path: str, parent: str) -> bool:
    return path.startswith(parent.rstrip("/") + "/")


class _Registered:
    """What the accept rule needs of one registered input, read once per `verify_hf`."""

    def __init__(self, entry: inputs.RegisteredInput, profile: Profile, now: datetime.datetime) -> None:
        self.usable = False
        codes, overlay = inputs.accepted(entry, profile, now=now)
        if codes:
            return
        try:
            rows = inputs.manifest_rows(entry, profile)
            self.top = os.path.realpath(inputs.input_top(entry, profile))
        except inputs.InputError:
            return
        self.rows = {r.relpath: r for r in rows}
        self.overlay = overlay or {}
        self.usable = True

    def accepts(self, blob: str, sha256: str) -> bool:
        if not self.usable or not _under(blob, self.top):
            return False
        rel = os.path.relpath(blob, self.top)
        row = self.rows.get(rel)
        if row is None or row.sha256 != sha256:
            return False
        try:
            st = os.lstat(blob)
        except OSError:
            return False
        return st.st_size == row.size and st.st_mtime_ns in (row.mtime_ns, self.overlay.get(rel))


def verify_hf(
    steps: Iterable[StepSpec],
    profile: Profile,
    registry: Mapping[str, inputs.RegisteredInput],
    *,
    now: datetime.datetime | None = None,
) -> list[str]:
    """Check every declared Hugging Face file of ``steps`` against its sha256, once per file and blob.

    Files are taken once per (type, repository, revision, path); a blob two declarations share is hashed once.
    A file all of whose declarations name a registered input that accepts it (see the module docstring) is
    not hashed. Prints ``weights files=<n> hashed=<n> accepted=<n>``.

    Args:
        steps: Every step of the prefetch.
        profile: The site profile; its ``hf_hub_cache`` holds the snapshots.
        registry: The registry entries the steps' ``HFWeights.input`` name, as their sources locks pin them;
            an input missing here is not used to accept a file.
        now: The time the verification records are judged at; defaults to now.

    Returns:
        One ``<code> <repo>@<revision[:8]> <path>`` per failing declaration: ``E_WEIGHTS_MISSING`` (no hub
        cache, or no such file) or ``E_WEIGHTS_MISMATCH``; empty when every file checks.
    """
    when = now or datetime.datetime.now(datetime.UTC)
    declared: dict[tuple[str, str, str, str], list[tuple[str, str | None]]] = {}
    for step in steps:
        for w in step.hf_weights:
            for f in w.files:
                uses = declared.setdefault((w.repo_type, w.repo, w.revision, f.path), [])
                if (f.sha256, w.input) not in uses:
                    uses.append((f.sha256, w.input))
    hub = profile.hf_hub_cache
    registered: dict[str, _Registered] = {}
    digests: dict[str, str] = {}
    findings: list[str] = []
    accepted = 0
    for (repo_type, repo, revision, path), uses in sorted(declared.items()):
        where = f"{repo}@{revision[:8]} {path}"
        file = None if hub is None else hub / repo_dir_name(repo, repo_type) / "snapshots" / revision / path
        if file is None or not file.is_file():
            findings += [f"{WEIGHTS_MISSING} {where}"] * len({sha for sha, _ in uses})
            continue
        blob = os.path.realpath(file)
        pending: list[str] = []
        for sha, input_id in uses:
            if input_id is not None and input_id in registry:
                state = registered.setdefault(input_id, _Registered(registry[input_id], profile, when))
                if state.accepts(blob, sha):
                    continue
            pending.append(sha)
        if not pending:
            accepted += 1
            continue
        if blob not in digests:
            digests[blob] = _sha256(blob)
        findings += [f"{WEIGHTS_MISMATCH} {where}" for sha in sorted(set(pending)) if sha != digests[blob]]
    print(f"weights files={len(declared)} hashed={len(digests)} accepted={accepted}")
    return findings


def build_hub_view(view_dir: Path, step: StepSpec, profile: Profile) -> None:
    """Lay out the step's hub cache at ``view_dir``: per declared repository, links to the profile's copy.

    ``<view>/<type>s--<org>--<name>/`` holds ``snapshots`` and ``blobs`` as absolute links to the profile's
    dirs and ``refs/main`` naming the declared revision (`dependencies._point_ref_at`). A repository that
    declares ``config.json`` must also be found by `dependencies.is_hf_model_cached` in the profile's cache.

    Args:
        view_dir: The view's root (``{tmp}/hf-hub``); created.
        step: The step.
        profile: The site profile.

    Raises:
        WeightsError: ``E_WEIGHTS_MISSING`` when a declared file is missing (`check_hf_present`);
            ``E_WEIGHTS_VIEW`` when ``refs/main`` could not be written.
    """
    if check_hf_present(step, profile):
        raise WeightsError(WEIGHTS_MISSING)
    hub = profile.hf_hub_cache
    assert hub is not None
    view_dir.mkdir(parents=True, exist_ok=True)
    for w in step.hf_weights:
        if CONFIG in {f.path for f in w.files} and not is_hf_model_cached(
            w.repo, w.revision, w.repo_type, cache_dir=hub
        ):
            raise WeightsError(WEIGHTS_MISSING)
        source = Path(os.path.realpath(hub / repo_dir_name(w.repo, w.repo_type)))
        root = view_dir / repo_dir_name(w.repo, w.repo_type)
        root.mkdir()
        for sub in ("snapshots", "blobs"):
            (root / sub).symlink_to(source / sub)
        _point_ref_at(root, "main", w.revision)
        try:
            written = (root / "refs" / "main").read_text()
        except OSError:
            written = ""
        if written != w.revision:
            raise WeightsError(WEIGHTS_VIEW)
