"""The one launch environment of every process senselab starts: isolated mode, an allowlist, a fixed cwd, a digest."""

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from senselab_worker.digest import identity_digest

LAUNCH_TAG = "launch-env@1"
"""The `identity_digest` tag of a launch: its numerics variables, its declared variables and its import paths."""

PASSED = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_TOKEN",
)
"""Ambient variables a launched process inherits without their reaching the digest: plumbing and credentials."""

NUMERICS_PREFIXES = ("CUDA_", "CUBLAS_", "CUDNN_", "NCCL_", "VLLM_", "TORCH_", "PYTORCH_", "OMP_", "MKL_", "OPENBLAS_")
"""Prefixes of ambient variables a launched process inherits and the digest records: they can change its numerics."""

ISOLATED = {"PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1"}
"""Set on every launch: a console script, which runs without `-I`, gets no user site and no script directory."""

_BOOT = (
    "import json, runpy, sys; sys.path[:0] = json.loads(sys.argv.pop(1));"
    " runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)"
)


@dataclass(frozen=True)
class Launch:
    """How one process is started.

    Attributes:
        python: The interpreter.
        variables: The process's whole environment: the allowlisted ambient variables, `ISOLATED` and the declared.
        cwd: Its working directory, the interpreter's directory.
        paths: Import roots prepended to its `sys.path`: this process's when `python` is this interpreter, else none.
        digest: `identity_digest(LAUNCH_TAG, ...)` of its numerics and declared variables and its paths.
    """

    python: str
    variables: Mapping[str, str]
    cwd: Path
    paths: tuple[str, ...]
    digest: str

    def module(self, module: str, *args: str) -> list[str]:
        """Return the argv running `module` as `__main__` under `-I`, with `paths` prepended."""
        if not self.paths:
            return [self.python, "-I", "-m", module, *args]
        return [self.python, "-I", "-c", _BOOT, json.dumps(list(self.paths)), module, *args]

    @staticmethod
    def script(script: str | os.PathLike[str], *args: str) -> list[str]:
        """Return the argv running the env's console `script`, isolated by `ISOLATED` and `variables`."""
        return [os.fspath(script), *args]


def launch(python: str | os.PathLike[str], declared: Mapping[str, str] | None = None) -> Launch:
    """Return the launch of `python` with `declared` set over the allowlisted ambient variables."""
    ambient = os.environ
    numerics = {k: v for k, v in ambient.items() if k.startswith(NUMERICS_PREFIXES)}
    recorded = {**numerics, **(declared or {})}
    python = os.path.abspath(python)
    paths = tuple(p for p in sys.path if p) if python == os.path.abspath(sys.executable) else ()
    return Launch(
        python=python,
        variables={**{k: ambient[k] for k in PASSED if k in ambient}, **ISOLATED, **recorded},
        cwd=Path(python).parent,
        paths=paths,
        digest=identity_digest(LAUNCH_TAG, [sorted([k, v] for k, v in recorded.items()), list(paths)]),
    )
