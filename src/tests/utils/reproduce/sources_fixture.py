"""Test helpers: a sources lock that pins exactly what a set of steps declares, and registered-input fixtures."""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from senselab.utils.reproduce import inputs, lock
from senselab.utils.reproduce.spec import RESERVED_ROOT_NAMES, Profile, StepSpec


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def env_locks(step: StepSpec) -> list[dict[str, str]]:
    """Every hashed lock of a step's env, as sources-lock ``envs`` entries."""
    env = step.env
    refs = [env.torch, env.body, env.wheels, env.test, env.notebook_tools, *(s.build_env for s in env.sdist_allow)]
    return [{"lock": r.lock, "sha256": r.lock_sha256} for r in refs if r is not None]


def lock_data(steps: Iterable[StepSpec], registered: Mapping[str, Mapping[str, str]] = {}) -> dict[str, Any]:
    """A sources lock (as data) pinning every step's code, env locks, data-root patterns, weights and URL files."""
    data: list[dict[str, Any]] = [{"id": i, "kind": "registered", **dict(e)} for i, e in sorted(registered.items())]
    code: list[dict[str, Any]] = []
    envs: list[dict[str, str]] = []
    models: list[dict[str, Any]] = []
    urls: list[dict[str, str]] = []
    for step in steps:
        urls += [u for u in ({"file": w.file, "sha256": w.sha256} for w in step.url_weights) if u not in urls]
        entry = {"repo": step.repo, "commit": step.commit, "patches": [p.sha256 for p in step.patches]}
        if {**entry, "submodules": {}} not in code:
            code.append({**entry, "submodules": {}})
        envs += [e for e in env_locks(step) if e not in envs]
        for spec in step.inputs:
            if spec.root.startswith("up:") or spec.root in RESERVED_ROOT_NAMES:
                continue
            pattern = {"kind": "pattern", "root": spec.root, "patterns": sorted(set(spec.patterns))}
            if pattern not in data:
                data.append(pattern)
        for w in step.hf_weights:
            model = {
                "id": f"{w.repo.replace('/', '__')}-{w.revision[:8]}"
                + ("-dataset" if w.repo_type == "dataset" else ""),
                "repo": w.repo,
                "repo_type": w.repo_type,
                "revision": w.revision,
                "files": [{"path": f.path, "sha256": f.sha256} for f in w.files],
            }
            if model not in models:
                models.append(model)
    return {
        "schema": lock.LOCK_SCHEMA,
        "data": data,
        "external": [],
        "models": models,
        "urls": urls,
        "code": code,
        "envs": envs,
    }


def write_lock(base: Path, name: str, data: Mapping[str, Any]) -> dict[str, str]:
    """Write ``<base>/sources/<name>.lock.yaml`` and return the manifest's ``sources`` reference to it."""
    rel = f"{lock.SOURCES_DIR}/{name}.lock.yaml"
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(data), sort_keys=False).encode()
    path.write_bytes(text)
    return {"lock": rel, "sha256": _sha(text)}


def register_input(
    base: Path,
    roots: Mapping[str, Path],
    input_id: str,
    *,
    root: str,
    relpath: str,
    sha256: bool = False,
    **over: Any,  # noqa: ANN401
) -> dict[str, str]:
    """Freeze ``roots[root]/relpath`` into ``<root>/_manifests``, register it and return its lock entry.

    ``sha256`` also records each file's sha256 in the manifest.
    """
    manifest_rel = f"_manifests/{input_id}.xxh3.tsv"
    out = roots[root] / manifest_rel
    inputs.write_manifest(
        roots[root] / relpath,
        out,
        include=over.get("include", ()),
        exclude=over.get("exclude", ()),
        sha256=sha256,
        label=f"{{root:{root}}}/{relpath}",
    )
    entry: dict[str, Any] = {
        "id": input_id,
        "root": root,
        "relpath": relpath,
        "manifest": {"root": root, "relpath": manifest_rel, "sha256": _sha(out.read_bytes())},
        "rung": "a",
        "rows": "none",
        "restricted": False,
        "licence": "CC0-1.0",
        "category": "derivative",
        **over,
    }
    path = base / inputs.REGISTRY_DIR / f"{input_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(entry, sort_keys=False))
    return {"registry_sha256": _sha(path.read_bytes()), "manifest_sha256": entry["manifest"]["sha256"]}


def verify_input(
    base: Path, profile: Profile, input_id: str, *, now: datetime.datetime | None = None
) -> inputs.VerifyRecord:
    """Verify a registered input as ``verify-inputs`` does and append its record; return the record."""
    entry, _ = inputs.read_registry_entry(base, input_id)
    when = now or datetime.datetime.now(datetime.UTC)
    verification = inputs.verify(entry, profile, now=when)
    inputs.append_record(inputs.verify_dir(entry, profile), input_id, verification)
    return verification.record
