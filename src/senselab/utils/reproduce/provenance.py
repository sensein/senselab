"""The provenance record of one step: ``store.jsonl`` plus BIDS BEP028 files, naming no file.

`record_step` writes, into a step's ``prov/`` directory, a `ProvStore` holding one activity (the step
run), one software agent (the step's code at its commit, ``Version`` the 40-hex commit), two
environments (the step's venv with its installed packages, and the host with the driver's
interpreter), and one ``artifact`` entity per declared input and output, then exports it with
`write_bep028_files` under the label ``reproduce``.

Nothing in the record is a path. Artifacts carry the input-set digest, the file count, the role, the
manifest position (``inputs[<i>]``, ``outputs[<i>]``) and, for an input, the data-root name. The
command is withheld (``Command`` is null) and recorded as ``argv_sha256``; the step environment as
its sorted key list and ``env_sha256``. Every value is checked against a closed form before anything
is written, and step stdout and logs are never read.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from senselab.utils.prov_bep028 import to_bep028_graph, write_bep028_files
from senselab.utils.prov_store import CHECKSUM_KEY, ProvStore
from senselab.utils.reproduce.env import VenvResult
from senselab.utils.reproduce.host import HOST_KEYS

LABEL = "reproduce"
NODE = "reproduce"
STORE_FILE = "store.jsonl"
DRIVER_DESCRIPTION = (
    "senselab reproduce step, run as a sandboxed subprocess; its command is withheld because its "
    "arguments can name data files, and is recorded as argv_sha256: {label}"
)
"""The activity ``Description``; ``{label}`` is replaced by ``reproduce / <step>``."""

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_STEP_KEY = re.compile(r"^[0-9a-f]{16,64}$")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_ROOT = re.compile(r"^(up:)?[A-Za-z0-9_.-]+$")
_CODE = re.compile(r"^[EW]_[A-Z0-9_]+$")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Artifact:
    """One declared input or output, as its digest.

    Attributes:
        digest: ``input_set_digest``'s hex digest of the file set.
        count: ``input_set_digest``'s file count.
        root: The data-root name the input's patterns are relative to (``InputSpec.root``); None for an
            output.
    """

    digest: str
    count: int
    root: str | None = None


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"record_step: {what} is not in its closed form")


def _check_time(value: str, what: str) -> None:
    try:
        datetime.datetime.fromisoformat(value)
    except (TypeError, ValueError):
        _check(False, what)


def _check_artifacts(artifacts: Sequence[Artifact], role: str) -> None:
    for artifact in artifacts:
        _check(isinstance(artifact.digest, str) and bool(_HEX64.fullmatch(artifact.digest)), f"{role} digest")
        _check(isinstance(artifact.count, int) and artifact.count >= 0, f"{role} count")
        if role == "input":
            _check(isinstance(artifact.root, str) and bool(_ROOT.fullmatch(artifact.root)), "input root")
        else:
            _check(artifact.root is None, "output root")


def _check_host(host: Mapping[str, Any]) -> None:
    _check(set(host) == set(HOST_KEYS), "host record")
    _check(
        all(value is None or isinstance(value, (str, int)) for value in host.values()),
        "host record",
    )


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
) -> list[Path]:
    """Write one step's provenance into ``prov_dir``.

    Args:
        prov_dir: The step's ``{out}/prov`` directory; created if absent.
        step_key: The step's cache key (hex); also the store's run id.
        step: The step's manifest name.
        commit_sha: The resolved 40-hex commit of the step's code.
        venv: The step's venv; only its directory name, key, packages and overlay are recorded.
        host: `host.host_record`'s result.
        inputs: One artifact per declared input, in manifest order.
        outputs: One artifact per declared output pattern, in manifest order.
        started: When the step began, ISO 8601.
        ended: When it finished, ISO 8601.
        exit_code: The step's exit code.
        codes: Fixed error and warning codes (``E_*``, ``W_*``).
        env_keys: The step environment's variable names.
        env_sha256: The sha256 of the step environment's ``key=value`` pairs.
        argv_sha256: The sha256 of the step's argv.

    Returns:
        The files written: ``store.jsonl``, then the BEP028 files in `write_bep028_files` order.

    Raises:
        ValueError: If any value is outside its closed form (a non-hex key or digest, a step name or
            root with a separator, an unknown host key, a malformed code, variable name or time).
            Nothing is written then.
    """
    _check(isinstance(step_key, str) and bool(_STEP_KEY.fullmatch(step_key)), "step_key")
    _check(isinstance(step, str) and bool(_NAME.fullmatch(step)) and step not in (".", ".."), "step")
    _check(isinstance(commit_sha, str) and bool(_COMMIT.fullmatch(commit_sha)), "commit_sha")
    _check_artifacts(inputs, "input")
    _check_artifacts(outputs, "output")
    _check_host(host)
    _check_time(started, "started")
    _check_time(ended, "ended")
    _check(isinstance(exit_code, int), "exit_code")
    _check(all(isinstance(code, str) and _CODE.fullmatch(code) for code in codes), "code")
    _check(all(isinstance(key, str) and _ENV_KEY.fullmatch(key) for key in env_keys), "env key")
    for value, what in ((env_sha256, "env_sha256"), (argv_sha256, "argv_sha256")):
        _check(value is None or bool(_HEX64.fullmatch(value)), what)
    _check(bool(_NAME.fullmatch(venv.path.name)), "venv name")
    _check(bool(_HEX64.fullmatch(venv.env_key)), "venv env_key")

    store = ProvStore(run_id=step_key)
    agent = store.agent(agent_type="software", commit_sha=commit_sha, version=f"{step} {commit_sha}")
    parameters: dict[str, Any] = {
        "step_key": step_key,
        "env_key": venv.env_key,
        "overlay_sha256": venv.overlay_sha256,
        "exit_code": exit_code,
        "codes": sorted(codes),
        "env_keys": sorted(env_keys),
        "env_sha256": env_sha256,
        "argv_sha256": argv_sha256,
        "host": {key: host[key] for key in HOST_KEYS},
    }
    activity = store.activity(node=NODE, step=step, parameters=parameters, started=started, ended=ended)
    store.was_associated_with(activity, agent)
    for index, artifact in enumerate(inputs):
        attributes = {
            "role": "input",
            "declared": f"inputs[{index}]",
            "root": artifact.root,
            CHECKSUM_KEY: artifact.digest,
            "file_count": artifact.count,
        }
        store.used(activity, store.entity(prov_type="artifact", extent=None, attributes=attributes))
    for index, artifact in enumerate(outputs):
        attributes = {
            "role": "output",
            "declared": f"outputs[{index}]",
            CHECKSUM_KEY: artifact.digest,
            "file_count": artifact.count,
        }
        store.was_generated_by(store.entity(prov_type="artifact", extent=None, attributes=attributes), activity)
    packages = dict(sorted(venv.packages.items()))
    store.environment(
        kind="venv",
        label=venv.path.name,
        python_version=str(host["python"] or "unknown"),
        dependencies=packages,
        dependencies_digest=hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest(),
    )
    store.environment(
        kind="host",
        label=str(host["hostname"] or "unknown"),
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        senselab_version=_senselab_version(),
    )

    target = Path(prov_dir)
    target.mkdir(parents=True, exist_ok=True)
    store_path = target / STORE_FILE
    store.write_jsonl(store_path)
    graph = to_bep028_graph(store, label=LABEL, description=DRIVER_DESCRIPTION)
    return [store_path, *write_bep028_files(graph, target, label=LABEL)]
