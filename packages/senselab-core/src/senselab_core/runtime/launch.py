"""The one launch environment of every process senselab starts: isolated, declared families, fixed cwd, digested."""

import functools
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from senselab_worker.digest import identity_digest

from senselab_core.runtime.execution.supervisor import Supervisor, current
from senselab_core.runtime.storage import load_policy_values

LAUNCH_TAG = "launch-env@1"
"""The `identity_digest` tag of a launch: its recorded ambient variables and its declared variables."""

LAUNCH_ENVIRONMENT_FORMAT = "launch-environment@2.0"
"""The format of `data/launch_environment.yaml`, the families of ambient variables a launched process inherits."""

ISOLATED = {"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"}
"""Set on every launch: a console script, which runs without `-I`, gets no user site and no script directory."""


@dataclass(frozen=True)
class Family:
    """Ambient variables passed together: by exact name or by prefix.

    `recorded` ones enter the launch digest; the `measured` names enter a measure unit (`Launch.measured`).
    """

    names: frozenset[str]
    prefixes: tuple[str, ...]
    recorded: bool
    measured: frozenset[str]

    def holds(self, name: str) -> bool:
        """Return whether `name` is one of this family's variables."""
        return name in self.names or name.startswith(self.prefixes)


@functools.cache
def launch_environment() -> tuple[dict[str, Family], Family]:
    """Return the packaged families by name, and the `never` family no launch passes.

    Raises:
        ValueError: the file lacks its format.
    """
    values = load_policy_values("senselab_core.runtime", "launch_environment.yaml", LAUNCH_ENVIRONMENT_FORMAT)
    entries = cast("dict[str, dict[str, Any]]", values)
    families = {
        name: Family(
            frozenset(v["names"]),
            tuple(v["prefixes"]),
            bool(v.get("recorded", False)),
            frozenset(v.get("measured", ())),
        )
        for name, v in entries.items()
    }
    return families, families.pop("never")


@dataclass(frozen=True)
class Launch:
    """How one process is started.

    Attributes:
        python: The interpreter.
        variables: The process's whole environment: the ambient variables of a family, `ISOLATED` and the declared.
        cwd: Its working directory, the interpreter's directory.
        digest: `identity_digest(LAUNCH_TAG, ...)` of its recorded ambient variables and its declared variables.
        measured: Its variables, ambient or declared, that a family names `measured`: what a measure unit holds.
    """

    python: str
    variables: Mapping[str, str]
    cwd: Path
    digest: str
    measured: Mapping[str, str]

    def module(self, module: str, *args: str) -> list[str]:
        """Return the argv running `module` as `__main__` under `-I`."""
        return [self.python, "-I", "-m", module, *args]

    @staticmethod
    def script(script: str | os.PathLike[str], *args: str) -> list[str]:
        """Return the argv running the env's console `script`, isolated by `ISOLATED` and `variables`."""
        return [os.fspath(script), *args]

    def start(
        self,
        argv: Sequence[str],
        *,
        supervisor: Supervisor | None = None,
        **popen: Any,  # noqa: ANN401
    ) -> "subprocess.Popen[bytes]":
        """Start `argv` with this launch's variables and directory through `supervisor` (`current()` when None).

        `popen` is passed to `Supervisor.popen`.
        """
        chosen = supervisor if supervisor is not None else current()
        return chosen.popen(argv, env=dict(self.variables), cwd=self.cwd, **popen)


def launch(python: str | os.PathLike[str], declared: Mapping[str, str] | None = None) -> Launch:
    """Return the launch of `python` with `declared` set over the ambient variables its families pass."""
    families, never = launch_environment()
    passed: dict[str, str] = {}
    ambient: dict[str, str] = {}
    for name, value in os.environ.items():
        family = next((f for f in families.values() if f.holds(name)), None)
        if family is None or never.holds(name):
            continue
        passed[name] = value
        if family.recorded:
            ambient[name] = value
    recorded = {**ambient, **(declared or {})}
    python = os.path.abspath(python)
    return Launch(
        python=python,
        variables={**passed, **ISOLATED, **recorded},
        cwd=Path(python).parent,
        digest=identity_digest(LAUNCH_TAG, sorted([k, v] for k, v in recorded.items())),
        measured={k: v for k, v in recorded.items() if any(k in f.measured for f in families.values())},
    )
