"""Pinned weights: Hugging Face snapshots and URL files, checked by sha256, and where each step sees them.

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

``url_weights`` files live under the profile's ``url_weights_root``. `check_url_present` and `verify_url` are
their presence check and prefetch check (the same accept rule, against the step's registered inputs), and
`place_url_weights` links each where its loader looks under the step's ``{tmp}``.

Stamps: given a stamps dir (``<cache>/weights-stamps``), `verify_hf` and `verify_url` write a stamp for each file
they verify (by hash or by the accept rule): the blob's real path, the declared sha256, and the blob's ``lstat``
size, ``mtime_ns``, device and inode. `check_stamps` is the inputs gate's re-check: a declared file with no stamp
is ``E_WEIGHTS_UNVERIFIED``, one whose blob no longer has the stamped size, mtime or inode is ``E_WEIGHTS_CHANGED``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

from senselab.utils.dependencies import _point_ref_at, is_hf_model_cached
from senselab.utils.reproduce import inputs
from senselab.utils.reproduce.spec import HFWeights, Profile, StepSpec, UrlWeight

WEIGHTS_MISSING = "E_WEIGHTS_MISSING"
WEIGHTS_MISMATCH = "E_WEIGHTS_MISMATCH"
WEIGHTS_VIEW = "E_WEIGHTS_VIEW"
WEIGHTS_CHANGED = "E_WEIGHTS_CHANGED"
WEIGHTS_UNVERIFIED = "E_WEIGHTS_UNVERIFIED"
STAMPS_DIR = "weights-stamps"
"""The stamps dir under the driver cache: one ``<sha256 of blob path and declared sha256>.json`` per verified file."""
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
        self.entry, self.profile, self.now = entry, profile, now
        self.top: str | None = None
        self.state: tuple[dict[str, inputs.Row], dict[str, int]] | None = None
        self.read = False
        try:
            self.top = os.path.realpath(inputs.input_top(entry, profile))
        except inputs.InputError:
            pass

    def _rows(self) -> tuple[dict[str, inputs.Row], dict[str, int]] | None:
        """The record check and the manifest rows, read on the first file under the input's tree."""
        if not self.read:
            self.read = True
            codes, overlay = inputs.accepted(self.entry, self.profile, now=self.now)
            if not codes:
                try:
                    rows = inputs.manifest_rows(self.entry, self.profile)
                except inputs.InputError:
                    return None
                self.state = {r.relpath: r for r in rows}, overlay or {}
        return self.state

    def accepts(self, blob: str, sha256: str) -> bool:
        if self.top is None or not _under(blob, self.top):
            return False
        state = self._rows()
        if state is None:
            return False
        rows, overlay = state
        rel = os.path.relpath(blob, self.top)
        row = rows.get(rel)
        if row is None or row.sha256 != sha256:
            return False
        try:
            st = os.lstat(blob)
        except OSError:
            return False
        return st.st_size == row.size and st.st_mtime_ns in (row.mtime_ns, overlay.get(rel))


_STAMP_FIELDS = ("st_size", "st_mtime_ns", "st_dev", "st_ino")


def _stamp_path(stamps: Path, blob: str, sha256: str) -> Path:
    name = hashlib.sha256(f"{blob}\n{sha256}".encode()).hexdigest()
    return stamps / f"{name}.json"


def _stamp_of(st: os.stat_result) -> dict[str, int]:
    return {field: int(getattr(st, field)) for field in _STAMP_FIELDS}


def write_stamp(stamps: Path, file: Path, sha256: str, st: os.stat_result | None = None) -> None:
    """Record that ``file`` (its real path) was verified against ``sha256``: its ``lstat`` size, mtime, device, inode.

    ``st`` is the ``lstat`` taken before the file was hashed; with it, nothing is written when the file changed
    meanwhile. Written atomically.
    """
    blob = os.path.realpath(file)
    now = os.lstat(blob)
    if st is not None and _stamp_of(st) != _stamp_of(now):
        remove_stamp(stamps, file, sha256)
        return
    stamps.mkdir(parents=True, exist_ok=True)
    path = _stamp_path(stamps, blob, sha256)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps({"blob": blob, "sha256": sha256, **_stamp_of(now)}, sort_keys=True))
    os.replace(tmp, path)


def remove_stamp(stamps: Path, file: Path, sha256: str) -> None:
    """Remove the stamp of ``file`` against ``sha256``, if there is one."""
    _stamp_path(stamps, os.path.realpath(file), sha256).unlink(missing_ok=True)


def declared_files(step: StepSpec, profile: Profile) -> list[tuple[Path | None, str]]:
    """Every Hugging Face and URL weights file a step declares, with its declared sha256; None without a root."""
    hub = profile.hf_hub_cache
    out: list[tuple[Path | None, str]] = []
    for w in step.hf_weights:
        out += [(None if hub is None else hf_snapshot(hub, w) / f.path, f.sha256) for f in w.files]
    out += [(url_file(profile, w), w.sha256) for w in step.url_weights]
    return out


def check_stamps(step: StepSpec, profile: Profile, stamps: Path) -> list[str]:
    """The gate's re-check of a step's weights files against the stamps prefetch wrote; no hashing.

    A file that is absent is left to the presence checks (`check_hf_present`, `check_url_present`).

    Returns:
        Sorted codes: ``E_WEIGHTS_UNVERIFIED`` for a file with no readable stamp for its declared sha256 (prefetch
        did not verify it), ``E_WEIGHTS_CHANGED`` for one whose blob's size, ``mtime_ns``, device or inode differ
        from the stamp; empty when every file matches.
    """
    codes = set()
    for file, sha256 in declared_files(step, profile):
        if file is None or not file.is_file():
            continue
        blob = os.path.realpath(file)
        try:
            stamp = json.loads(_stamp_path(stamps, blob, sha256).read_text())
            recorded = {field: int(stamp[field]) for field in _STAMP_FIELDS}
            matches = stamp["blob"] == blob and stamp["sha256"] == sha256
        except (OSError, ValueError, KeyError, TypeError):
            codes.add(WEIGHTS_UNVERIFIED)
            continue
        try:
            current = _stamp_of(os.lstat(blob))
        except OSError:
            continue
        if not matches:
            codes.add(WEIGHTS_UNVERIFIED)
        elif current != recorded:
            codes.add(WEIGHTS_CHANGED)
    return sorted(codes)


_Use = tuple[str, tuple[str, ...]]
"""One declaration of a file: its sha256 and the registered inputs that may accept it."""
_Files = dict[tuple[str, ...], tuple[str, Path | None, list[_Use]]]
"""Declared files by identity: the label findings print, the file (None without a root) and its declarations."""


def _check(
    files: _Files,
    profile: Profile,
    registry: Mapping[str, inputs.RegisteredInput],
    now: datetime.datetime,
    stamps: Path | None = None,
) -> tuple[list[str], int, int]:
    """Findings, blobs hashed and files accepted, for each declared file; each verified file is stamped."""
    registered: dict[str, _Registered] = {}
    digests: dict[str, str] = {}
    befores: dict[str, os.stat_result] = {}
    findings: list[str] = []
    accepted = 0
    for _, (label, file, uses) in sorted(files.items()):
        if file is None or not file.is_file():
            findings.append(f"{WEIGHTS_MISSING} {label}")
            continue
        blob = os.path.realpath(file)
        before = befores.setdefault(blob, os.lstat(blob))
        pending: list[str] = []
        for sha, ids in uses:
            for input_id in (i for i in ids if i in registry):
                if input_id not in registered:
                    registered[input_id] = _Registered(registry[input_id], profile, now)
                if registered[input_id].accepts(blob, sha):
                    break
            else:
                pending.append(sha)
        if pending and blob not in digests:
            digests[blob] = _sha256(blob)
        # Once the blob is hashed (here, or for another file on it), the digest decides every declaration.
        bad = {sha for sha, _ in uses if sha != digests[blob]} if blob in digests else set()
        if not pending:
            accepted += 1
        findings += [f"{WEIGHTS_MISMATCH} {label} {sha[:8]}" for sha in sorted(bad)]
        if stamps is not None:
            for sha in {sha for sha, _ in uses}:
                if sha in bad:
                    remove_stamp(stamps, file, sha)
                else:
                    write_stamp(stamps, file, sha, before)
    return findings, len(digests), accepted


def _add(files: _Files, key: tuple[str, ...], label: str, file: Path | None, use: _Use) -> None:
    uses = files.setdefault(key, (label, file, []))[2]
    if use not in uses:
        uses.append(use)


def verify_hf(
    steps: Iterable[StepSpec],
    profile: Profile,
    registry: Mapping[str, inputs.RegisteredInput],
    *,
    now: datetime.datetime | None = None,
    stamps: Path | None = None,
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
        stamps: The stamps dir (`STAMPS_DIR` under the driver cache); each verified file is stamped there, and a
            mismatching one loses its stamp. None writes no stamp.

    Returns:
        ``E_WEIGHTS_MISSING <repo>@<revision[:8]> <path>`` per missing file (or every file, with no hub cache)
        and ``E_WEIGHTS_MISMATCH <repo>@<revision[:8]> <path> <declared sha256[:8]>`` per declared sha256 the
        blob does not have (a dataset repository's lines start ``dataset:<repo>``); empty when every file
        checks.
    """
    hub = profile.hf_hub_cache
    files: _Files = {}
    for step in steps:
        for w in step.hf_weights:
            snapshot = None if hub is None else hf_snapshot(hub, w)
            for f in w.files:
                label = f"{w.repo}@{w.revision[:8]} {f.path}"
                if w.repo_type != "model":
                    label = f"{w.repo_type}:{label}"
                file = None if snapshot is None else snapshot / f.path
                key = (w.repo_type, w.repo, w.revision, f.path)
                _add(files, key, label, file, (f.sha256, () if w.input is None else (w.input,)))
    findings, hashed, accepted = _check(files, profile, registry, now or datetime.datetime.now(datetime.UTC), stamps)
    print(f"weights files={len(files)} hashed={hashed} accepted={accepted}")
    return findings


def url_file(profile: Profile, w: UrlWeight) -> Path | None:
    """``<url_weights_root>/<file>``; None when the profile has no root, or the file resolves outside it."""
    root = profile.url_weights_root
    if root is None:
        return None
    file = root / w.file
    return file if _under(os.path.realpath(file), os.path.realpath(root)) else None


def check_url_present(step: StepSpec, profile: Profile) -> list[str]:
    """Whether every ``url_weights`` file of the step exists under the profile's root; no hashing.

    Returns:
        ``["E_WEIGHTS_MISSING"]`` when one is missing, resolves outside the root, or the profile has no
        ``url_weights_root``; else empty.
    """
    for w in step.url_weights:
        file = url_file(profile, w)
        if file is None or not file.is_file():
            return [WEIGHTS_MISSING]
    return []


def verify_url(
    steps: Iterable[StepSpec],
    profile: Profile,
    registry: Mapping[str, inputs.RegisteredInput],
    *,
    now: datetime.datetime | None = None,
    stamps: Path | None = None,
) -> list[str]:
    """Check every ``url_weights`` file of ``steps`` against its sha256, once per file.

    The accept rule is `verify_hf`'s, tried against each registered input of the step's ``lock_inputs`` (the
    model-cache input holding the URL files, when the step registers it). Prints
    ``url-weights files=<n> hashed=<n> accepted=<n>``.

    Args:
        steps: Every step of the prefetch.
        profile: The site profile; its ``url_weights_root`` holds the files.
        registry: The registry entries of the steps' ``lock_inputs``, as their sources locks pin them.
        now: The time the verification records are judged at; defaults to now.
        stamps: The stamps dir (`STAMPS_DIR` under the driver cache); each verified file is stamped there, and a
            mismatching one loses its stamp. None writes no stamp.

    Returns:
        ``E_WEIGHTS_MISSING url <file>`` and ``E_WEIGHTS_MISMATCH url <file> <declared sha256[:8]>`` lines;
        empty when every file checks.
    """
    files: _Files = {}
    for step in steps:
        for w in step.url_weights:
            _add(files, (w.file,), f"url {w.file}", url_file(profile, w), (w.sha256, tuple(step.lock_inputs)))
    findings, hashed, accepted = _check(files, profile, registry, now or datetime.datetime.now(datetime.UTC), stamps)
    print(f"url-weights files={len(files)} hashed={hashed} accepted={accepted}")
    return findings


def place_url_weights(tmp: Path, step: StepSpec, profile: Profile) -> None:
    """Link each ``url_weights`` file where its loader looks, under the step's ``tmp`` (`UrlWeight.target`).

    Args:
        tmp: The step's ``{tmp}`` on the host.
        step: The step.
        profile: The site profile.

    Raises:
        WeightsError: ``E_WEIGHTS_MISSING`` when a file is missing (`check_url_present`); ``E_WEIGHTS_VIEW``
            when a link cannot be made.
    """
    if check_url_present(step, profile):
        raise WeightsError(WEIGHTS_MISSING)
    for w in step.url_weights:
        file = url_file(profile, w)
        assert file is not None
        link = tmp / w.target()
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.realpath(file))
        except OSError:
            raise WeightsError(WEIGHTS_VIEW) from None


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
            ``E_WEIGHTS_VIEW`` when a dir, link or ``refs/main`` could not be written.
    """
    if check_hf_present(step, profile):
        raise WeightsError(WEIGHTS_MISSING)
    hub = profile.hf_hub_cache
    assert hub is not None
    try:
        view_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise WeightsError(WEIGHTS_VIEW) from None
    for w in step.hf_weights:
        if CONFIG in {f.path for f in w.files} and not is_hf_model_cached(
            w.repo, w.revision, w.repo_type, cache_dir=hub
        ):
            raise WeightsError(WEIGHTS_MISSING)
        source = Path(os.path.realpath(hub / repo_dir_name(w.repo, w.repo_type)))
        root = view_dir / repo_dir_name(w.repo, w.repo_type)
        try:
            root.mkdir()
            for sub in ("snapshots", "blobs"):
                (root / sub).symlink_to(source / sub)
        except OSError:
            raise WeightsError(WEIGHTS_VIEW) from None
        _point_ref_at(root, "main", w.revision)
        try:
            written = (root / "refs" / "main").read_text()
        except OSError:
            written = ""
        if written != w.revision:
            raise WeightsError(WEIGHTS_VIEW)
