"""The provenance record of one step: ``store.jsonl`` plus BIDS BEP028 files, naming no file.

`record_step` writes, into a step's ``prov/`` directory, a `ProvStore` holding one activity (the step
run), one software agent (the step's code at its commit, ``Version`` the 40-hex commit), two
environments (the step's venv with its installed packages, and the host with the driver's
interpreter), and one ``artifact`` entity per declared input and output, then exports it with
`write_bep028_files` under the label ``reproduce``.

Nothing in the record is a path. Artifacts carry the input-set digest, the file count, the role, the
manifest position (``inputs[<i>]``, ``outputs[<i>]``) and, for an input, the data-root name. The
command is withheld (``Command`` is null) and recorded as ``argv_sha256``; the step environment as
its sorted key list and ``env_sha256``. The activity's parameters carry the lock sha256s, the overlay
sha256 and the venv's ``installed_sha256`` (the value in its marker), so the record can be checked
against the venv. Every value is checked against a closed form, and the graph against
`check_graph`, before anything is written; step stdout and logs are never read.

That is the confidential record. A profile whose policy sets ``confidential_provenance: false`` passes
each input's ``paths`` and the rendered ``argv`` as well; they are then recorded (the artifact's
``paths``, the activity's ``argv``) and the activity's description says so (`OPEN_DESCRIPTION`).
"""

from __future__ import annotations

import datetime
import os
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from senselab.utils.prov_bep028 import check_graph, to_bep028_graph, write_bep028_files
from senselab.utils.prov_store import CHECKSUM_KEY, ProvStore
from senselab.utils.reproduce.env import VenvResult, installed_sha256
from senselab.utils.reproduce.host import HOST_KEYS

LABEL = "reproduce"
NODE = "reproduce"
STORE_FILE = "store.jsonl"
DRIVER_DESCRIPTION = (
    "senselab reproduce step, run as a sandboxed subprocess; its command is withheld because its "
    "arguments can name data files, and is recorded as argv_sha256: {label}"
)
"""The activity ``Description``; ``{label}`` is replaced by ``reproduce / <step>``."""
OPEN_DESCRIPTION = (
    "senselab reproduce step, run as a sandboxed subprocess; its rendered argv and input paths are recorded "
    "in the activity parameters and the input artifacts: {label}"
)
"""The ``Description`` of a record that carries the argv and input paths."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STEP_KEY = re.compile(r"^[0-9a-f]{16,64}$")
_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
_VENV_NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}(\+[0-9a-f]{8})?$")
_ROOT = re.compile(r"^(up:)?[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}$")
_CODE = re.compile(r"^[EW]_[A-Z0-9_]{1,64}$")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PACKAGE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}$")
_WORDS = re.compile(r"^[A-Za-z0-9()@.,_+-]+( [A-Za-z0-9()@.,_+-]+)*$")
_NUMERIC = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_HOST_STRINGS: dict[str, re.Pattern[str]] = {
    "hostname": re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]{0,252}[A-Za-z0-9])?$"),
    "cpu_model": _WORDS,
    "gpu_name": _WORDS,
    "gpu_driver": _NUMERIC,
    "python": _VERSION,
    "torch": _VERSION,
    "torch_cuda": _NUMERIC,
    "slurm_job_id": re.compile(r"^[0-9]+(_[0-9]+)?$"),
}
_HOST_INTS = ("cpu_count", "cpus_allowed", "gpu_count")
_HOST_STRING_MAX = 128
_PREFIX = f"prov-{LABEL}_"


@dataclass(frozen=True)
class Artifact:
    """One declared input or output, as its digest.

    Attributes:
        digest: ``input_set_digest``'s hex digest of the file set.
        count: ``input_set_digest``'s file count.
        root: The data-root name the input's patterns are relative to (``InputSpec.root``); None for an
            output.
        paths: The input's patterns joined to its root dir, recorded only for a non-confidential policy;
            None otherwise and for an output.
    """

    digest: str
    count: int
    root: str | None = None
    paths: tuple[str, ...] | None = None


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"record_step: {what} is not in its closed form")


def _matches(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_name(value: object, pattern: re.Pattern[str] = _NAME) -> bool:
    return _matches(value, pattern) and str(value).removeprefix("up:") not in (".", "..")


def _check_time(value: object, what: str) -> None:
    try:
        parsed = datetime.datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    _check(parsed is not None and parsed.tzinfo is not None, what)


def _check_artifacts(artifacts: Sequence[Artifact], role: str) -> None:
    for artifact in artifacts:
        _check(_matches(artifact.digest, _HEX64), f"{role} digest")
        _check(_is_int(artifact.count) and artifact.count >= 0, f"{role} count")
        if role == "input":
            _check(_is_name(artifact.root, _ROOT), "input root")
            _check(artifact.paths is None or _strings(artifact.paths), "input paths")
        else:
            _check(artifact.root is None and artifact.paths is None, "output root")


def _strings(values: Sequence[object]) -> bool:
    return all(isinstance(v, str) and v and "\0" not in v for v in values)


def _check_host(host: Mapping[str, Any]) -> None:
    _check(set(host) == set(HOST_KEYS), "host record")
    for key in _HOST_INTS:
        _check(host[key] is None or (_is_int(host[key]) and host[key] >= 0), f"host {key}")
    for key, pattern in _HOST_STRINGS.items():
        value = host[key]
        _check(value is None or (_matches(value, pattern) and len(value) <= _HOST_STRING_MAX), f"host {key}")


def drop_unrecordable_host_values(host: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Set every host value `record_step` would refuse to None, so an odd host never costs a step its record.

    Args:
        host: `host.host_record`'s result.

    Returns:
        ``(host, dropped)``: a copy holding exactly `HOST_KEYS`, with each value outside its closed form
        (a string off its pattern or too long, a count that is not a non-negative integer) set to None,
        and the sorted names of the keys so dropped. The dropped values are never returned.
    """
    clean: dict[str, Any] = {key: host.get(key) for key in HOST_KEYS}
    dropped = sorted(set(host) - set(HOST_KEYS))
    for key in _HOST_INTS:
        value = clean[key]
        if value is not None and not (_is_int(value) and value >= 0):
            clean[key] = None
            dropped.append(key)
    for key, pattern in _HOST_STRINGS.items():
        value = clean[key]
        if value is not None and not (_matches(value, pattern) and len(value) <= _HOST_STRING_MAX):
            clean[key] = None
            dropped.append(key)
    return clean, sorted(set(dropped))


def _check_venv(venv: VenvResult) -> str:
    _check(_is_name(venv.path.name, _VENV_NAME), "venv name")
    _check(_matches(venv.env_key, _HEX64), "venv env_key")
    _check(venv.overlay_sha256 is None or _matches(venv.overlay_sha256, _HEX64), "venv overlay_sha256")
    _check(
        all(_matches(name, _PACKAGE) and _matches(ver, _VERSION) for name, ver in venv.packages.items()),
        "venv packages",
    )
    installed = installed_sha256(venv.path)
    _check(_matches(installed, _HEX64), "venv installed_sha256")
    return str(installed)


def _senselab_version() -> str | None:
    try:
        return version("senselab")
    except PackageNotFoundError:
        return None


def record_step(
    prov_dir: str | os.PathLike[str],
    *,
    step_key: str,
    step: str,
    commit_sha: str,
    venv: VenvResult,
    torch_lock_sha256: str | None,
    body_lock_sha256: str,
    host: Mapping[str, Any],
    inputs: Sequence[Artifact],
    outputs: Sequence[Artifact],
    started: str,
    ended: str,
    exit_code: int,
    codes: Sequence[str] = (),
    env_keys: Sequence[str] = (),
    env_sha256: str | None = None,
    argv_sha256: str | None = None,
    argv: Sequence[str] | None = None,
) -> list[Path]:
    """Write one step's provenance into ``prov_dir``.

    Args:
        prov_dir: The step's ``{out}/prov`` directory; created if absent.
        step_key: The step's cache key (hex); also the store's run id.
        step: The step's manifest name.
        commit_sha: The resolved 40-hex commit of the step's code.
        venv: The step's venv; its directory name, key, overlay sha256 and packages are recorded, with
            the ``installed_sha256`` its marker holds (`env.installed_sha256`).
        torch_lock_sha256: The env's torch lock sha256, None for an env without one.
        body_lock_sha256: The env's body lock sha256.
        host: `host.host_record`'s result.
        inputs: One artifact per declared input, in manifest order.
        outputs: One artifact per declared output pattern, in manifest order.
        started: When the step began, ISO 8601 with a timezone.
        ended: When it finished, ISO 8601 with a timezone.
        exit_code: The step's exit code.
        codes: Fixed error and warning codes (``E_*``, ``W_*``).
        env_keys: The step environment's variable names.
        env_sha256: The sha256 of the step environment's ``key=value`` pairs.
        argv_sha256: The sha256 of the step's argv.
        argv: The rendered argv, given only under a non-confidential policy; recorded as the activity's
            ``argv``.

    Returns:
        The files written: ``store.jsonl``, then the BEP028 files in `write_bep028_files` order. Any
        other ``prov-reproduce_*.json`` left in ``prov_dir`` by an earlier write is removed.

    Raises:
        ValueError: If any value is outside its closed form, if the venv's ``installed.json`` does not
            match its marker, if the graph fails `check_graph`, or if ``prov_dir`` is a symlink. Nothing is
            written then.
    """
    _check(_matches(step_key, _STEP_KEY), "step_key")
    _check(_is_name(step), "step")
    _check(_matches(commit_sha, _COMMIT), "commit_sha")
    _check(torch_lock_sha256 is None or _matches(torch_lock_sha256, _HEX64), "torch_lock_sha256")
    _check(_matches(body_lock_sha256, _HEX64), "body_lock_sha256")
    _check_artifacts(inputs, "input")
    _check_artifacts(outputs, "output")
    _check_host(host)
    _check_time(started, "started")
    _check_time(ended, "ended")
    _check(_is_int(exit_code), "exit_code")
    _check(all(_matches(code, _CODE) for code in codes), "code")
    _check(all(_matches(key, _ENV_KEY) for key in env_keys), "env key")
    for value, what in ((env_sha256, "env_sha256"), (argv_sha256, "argv_sha256")):
        _check(value is None or _matches(value, _HEX64), what)
    _check(argv is None or (not isinstance(argv, str) and _strings(argv)), "argv")
    installed = _check_venv(venv)

    store = ProvStore(run_id=step_key)
    agent = store.agent(agent_type="software", commit_sha=commit_sha, version=f"{step} {commit_sha}")
    parameters: dict[str, Any] = {
        "step_key": step_key,
        "env_key": venv.env_key,
        "torch_lock_sha256": torch_lock_sha256,
        "body_lock_sha256": body_lock_sha256,
        "overlay_sha256": venv.overlay_sha256,
        "installed_sha256": installed,
        "exit_code": exit_code,
        "codes": sorted(codes),
        "env_keys": sorted(env_keys),
        "env_sha256": env_sha256,
        "argv_sha256": argv_sha256,
        "host": {key: host[key] for key in HOST_KEYS},
    }
    if argv is not None:
        parameters["argv"] = list(argv)
    activity = store.activity(node=NODE, step=step, parameters=parameters, started=started, ended=ended)
    store.was_associated_with(activity, agent)
    for index, artifact in enumerate(inputs):
        attributes: dict[str, Any] = {
            "role": "input",
            "declared": f"inputs[{index}]",
            "root": artifact.root,
            CHECKSUM_KEY: artifact.digest,
            "file_count": artifact.count,
        }
        if artifact.paths is not None:
            attributes["paths"] = list(artifact.paths)
        store.used(activity, store.entity(prov_type="artifact", extent=None, attributes=attributes))
    for index, artifact in enumerate(outputs):
        attributes = {
            "role": "output",
            "declared": f"outputs[{index}]",
            CHECKSUM_KEY: artifact.digest,
            "file_count": artifact.count,
        }
        store.was_generated_by(store.entity(prov_type="artifact", extent=None, attributes=attributes), activity)
    store.environment(
        kind="venv",
        label=venv.path.name,
        python_version=str(host["python"] or "unknown"),
        dependencies=dict(sorted(venv.packages.items())),
    )
    store.environment(
        kind="host",
        label=str(host["hostname"] or "unknown"),
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        senselab_version=_senselab_version(),
    )
    opened = argv is not None or any(a.paths is not None for a in inputs)
    graph = to_bep028_graph(store, label=LABEL, description=OPEN_DESCRIPTION if opened else DRIVER_DESCRIPTION)
    problems = check_graph(graph)
    _check(not problems, "graph")

    target = Path(prov_dir)
    _check(not target.is_symlink(), "prov_dir")
    target.mkdir(parents=True, exist_ok=True)
    _check(not target.is_symlink() and target.is_dir(), "prov_dir")
    store_path = target / STORE_FILE
    store.write_jsonl(store_path)
    written = write_bep028_files(graph, target, label=LABEL)
    for stale in target.glob(f"{_PREFIX}*.json"):
        if stale not in written:
            stale.unlink()
    return [store_path, *written]
