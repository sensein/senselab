"""The step sandbox: a user, network and mount namespace with read-only data, and an allowlisted env.

`sandbox_argv` wraps a step's argv so that it runs inside ``unshare -rnm --propagation private`` with
the data roots, the checkout and the venv mounted read-only, the writable directories bound read-write
and loopback raised. `build_env` builds the step's whole environment from the profile and the step;
nothing is inherited from the caller except ``SLURM_*`` and ``CUDA_VISIBLE_DEVICES``, which are
recorded but not keyed. `canary` is a Python snippet whose TCP connect must fail inside the sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from senselab.utils.reproduce.spec import Profile, StepSpec, render

IP = "/usr/sbin/ip"
UNSHARE = ["unshare", "-rnm", "--propagation", "private"]
PROXY = "http://127.0.0.1:9"
PROXY_VARS = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy")
NO_PROXY_VARS = ("no_proxy", "NO_PROXY")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
FIXED_KEYED = {"LANG": "C.UTF-8", "TZ": "UTC"}
SYSTEM_PATH = ("/usr/bin", "/bin")
INHERITED = ("CUDA_VISIBLE_DEVICES",)
INHERITED_PREFIXES = ("SLURM_",)
VOLATILE_TEMPLATES = {
    "TMPDIR": "{tmp}",
    "HOME": "{tmp}/home",
    "IPYTHONDIR": "{tmp}/ipython",
    "JUPYTER_RUNTIME_DIR": "{tmp}/jupyter/runtime",
}
RESERVED_PREFIXES = ("SLURM_", "REPRO_", "JUPYTER_")
FIXED_PATHS = ("out", "tmp", "checkout", "venv", "cache")


def _abs(path: str | os.PathLike[str]) -> str:
    if not Path(path).is_absolute():
        raise ValueError(f"sandbox paths must be absolute: {os.fspath(path)!r}")
    return os.path.realpath(path)


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def sandbox_script(
    ro_roots: Iterable[str | os.PathLike[str]],
    rw_dirs: Iterable[str | os.PathLike[str]],
    ro_extra: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Return the ``sh`` script `sandbox_argv` runs inside the namespace before ``exec "$@"``.

    Args:
        ro_roots: Data roots, bound to themselves and remounted read-only.
        rw_dirs: Directories that stay writable. Those under a data root are bound before the root is
            remounted read-only; the others are writable already.
        ro_extra: Further directories (the checkout, the venv) bound and remounted read-only last.

    Returns:
        The script. It starts with ``set -e``, so a failed mount ends it before the step runs.

    Raises:
        ValueError: If a path is relative, a read-only path lies under (or is) a writable directory, or a
            writable directory lies under a ``ro_extra`` directory.
    """
    roots, rw, extra = [_abs(p) for p in ro_roots], [_abs(p) for p in rw_dirs], [_abs(p) for p in ro_extra]
    for d in rw:
        hidden = [p for p in (*roots, *extra) if _under(p, d)]
        if hidden:
            raise ValueError(f"read-only paths under a writable directory would be hidden by its bind: {hidden}")
        if any(_under(d, e) for e in extra):
            raise ValueError(f"writable directory under a read-only checkout or venv: {d}")
    q = shlex.quote
    lines = ["set -e"]
    lines += [f"mount --bind {q(r)} {q(r)}" for r in roots]
    lines += [f"mount --bind {q(d)} {q(d)}" for d in rw if any(_under(d, r) for r in roots)]
    lines += [f"mount -o remount,bind,ro {q(r)}" for r in roots]
    for e in extra:
        lines += [f"mount --bind {q(e)} {q(e)}", f"mount -o remount,bind,ro {q(e)}"]
    lines += [f"{IP} link set lo up", 'exec "$@"']
    return "\n".join(lines) + "\n"


def sandbox_argv(
    argv: Sequence[str],
    ro_roots: Iterable[str | os.PathLike[str]],
    rw_dirs: Iterable[str | os.PathLike[str]],
    ro_extra: Iterable[str | os.PathLike[str]] = (),
) -> list[str]:
    """Wrap a step's argv to run in the sandbox.

    Args:
        argv: The expanded step argv; the child receives it unchanged.
        ro_roots: Data roots, mounted read-only.
        rw_dirs: Directories that stay writable, such as ``{out}`` and ``{tmp}``.
        ro_extra: The checkout and the venv, mounted read-only.

    Returns:
        ``["unshare", "-rnm", "--propagation", "private", "sh", "-c", SCRIPT, "sh", *argv]``.

    Raises:
        ValueError: If argv is empty, or as `sandbox_script`.
    """
    if not argv:
        raise ValueError("empty argv")
    return [*UNSHARE, "sh", "-c", sandbox_script(ro_roots, rw_dirs, ro_extra), "sh", *argv]


def userns_available() -> bool:
    """Return whether an unprivileged namespace with loopback can be created here.

    Returns:
        True if ``unshare -rnm`` plus raising loopback succeeds.
    """
    try:
        done = subprocess.run(
            sandbox_argv(["true"], [], []),
            env={"PATH": ":".join(SYSTEM_PATH)},
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _reserved(key: str, driver: Iterable[str]) -> bool:
    return (
        key in driver
        or key in INHERITED
        or key.startswith(RESERVED_PREFIXES)
        or key.lower() in PROXY_VARS
        or key in NO_PROXY_VARS
    )


def _sha(env: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(env), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _keyed(step: StepSpec, profile: Profile) -> tuple[dict[str, str], dict[str, str]]:
    templated = {
        "PATH": ":".join(["{venv}/bin", *SYSTEM_PATH, *(_escape(os.fspath(d)) for d in profile.native_dirs)]),
        "JUPYTER_PATH": "{venv}/share/jupyter",
        "JUPYTER_DATA_DIR": "{tmp}/jupyter/data",
        "JUPYTER_CONFIG_DIR": "{tmp}/jupyter/config",
        "HF_HOME": "{cache}/huggingface",
        "TORCH_HOME": "{cache}/torch",
        "XDG_CACHE_HOME": "{tmp}/cache",
        "CUDA_CACHE_PATH": "{out}/cuda-cache",
        "REPRO_OUT": "{out}",
        "REPRO_TMP": "{tmp}",
    }
    if step.pythonpath:
        templated["PYTHONPATH"] = ":".join(step.pythonpath)
    if step.env.r_lib is not None:
        templated["R_LIBS_USER"] = step.env.r_lib
    literal = {**FIXED_KEYED, **OFFLINE}
    literal |= {v: PROXY for v in PROXY_VARS} | {v.upper(): PROXY for v in PROXY_VARS} | {v: "" for v in NO_PROXY_VARS}
    driver = {*templated, *literal, *VOLATILE_TEMPLATES}
    bad = sorted(k for k in profile.env if _reserved(k, driver))
    if bad:
        raise ValueError(f"profile env may not set driver-set, inherited or proxy variables: {bad}")
    return {**profile.env, **literal}, templated


def keyed_templates(step: StepSpec, profile: Profile) -> dict[str, str]:
    """Return the keyed part of a step's env, with path variables in template form.

    Args:
        step: The step.
        profile: The caller's profile.

    Returns:
        Name to value or template, e.g. ``{"PATH": "{venv}/bin:/usr/bin:/bin", "TZ": "UTC", ...}``.

    Raises:
        ValueError: If the profile env map names a driver-set, inherited or proxy variable.
    """
    literal, templated = _keyed(step, profile)
    return literal | templated


def build_env(
    step: StepSpec,
    profile: Profile,
    paths: Mapping[str, str | os.PathLike[str]],
    parent: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], str, str]:
    """Build a step's whole environment.

    Args:
        step: The step.
        profile: The caller's profile; its ``roots`` fill the ``{root:<name>}`` placeholders.
        paths: Values for ``out``, ``tmp``, ``checkout``, ``venv`` and ``cache``, plus any ``up:<step>``.
        parent: Where ``SLURM_*`` and ``CUDA_VISIBLE_DEVICES`` are read from; defaults to ``os.environ``.
            Nothing else is read from it.

    Returns:
        ``(env, keyed_sha, recorded_sha)``: the expanded env to run with; the sha256 of the keyed
        variables in template form, for the step key; and the sha256 of every expanded variable, for
        provenance.

    Raises:
        ValueError: If a fixed path is missing or relative, or as `keyed_templates`.
    """
    missing = sorted(set(FIXED_PATHS) - set(paths))
    if missing:
        raise ValueError(f"missing sandbox paths: {missing}")
    values = {k: _abs(v) for k, v in paths.items()}
    values |= {f"root:{name}": _abs(root) for name, root in profile.roots.items()}
    literal, templated = _keyed(step, profile)
    env = literal | {k: render(v, values) for k, v in templated.items()}
    env |= {k: render(v, values) for k, v in VOLATILE_TEMPLATES.items()}
    source = os.environ if parent is None else parent
    env |= {k: v for k, v in source.items() if k in INHERITED or k.startswith(INHERITED_PREFIXES)}
    return env, _sha(literal | templated), _sha(env)


def canary(host: str, port: int = 443, timeout: float = 5.0) -> str:
    """Return a Python snippet that tries a TCP connect and prints ``{"canary_failed": <bool>}``.

    Args:
        host: The profile's canary host, a name or an address.
        port: The TCP port.
        timeout: Seconds before the connect counts as failed.

    Returns:
        Source for ``python -c``; it always exits 0. Inside the sandbox the connect must fail.
    """
    return (
        "import json, socket\n"
        "try:\n"
        f"    socket.create_connection(({host!r}, {int(port)}), timeout={float(timeout)}).close()\n"
        "    failed = False\n"
        "except OSError:\n"
        "    failed = True\n"
        'print(json.dumps({"canary_failed": failed}))\n'
    )
