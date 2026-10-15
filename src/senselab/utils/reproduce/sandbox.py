"""The step sandbox: user, network, mount and PID namespaces with read-only data, and an allowlisted env.

`sandbox_argv` wraps a step's argv so that it runs inside a user, network, mount and PID namespace in which
every mount is read-only except the writable directories it names, with loopback raised; the step itself
runs with every capability dropped, so it cannot undo the mounts.
`build_env` builds the step's whole environment from the profile and the step; nothing is inherited
from the caller except a few ``SLURM_*`` job ids and ``CUDA_VISIBLE_DEVICES``, which are recorded but
not keyed. A step's ``draw`` reaches it as ``REPRO_SEED``, ``REPRO_DATA_ORDER_SEED`` and ``REPRO_SAMPLING``.
Proxies point at a dead port; an engine step also gets ``REPRO_ENGINE_IDENTITY``, and a ``vllm`` step
``no_proxy=127.0.0.1,localhost`` and its compile caches under ``{tmp}`` (`engine`). `canary` is a Python snippet
whose TCP connect must fail inside the sandbox.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from senselab.utils.reproduce import engine
from senselab.utils.reproduce.spec import DrawSpec, Profile, StepSpec, check_step_profile, render

IP = "/usr/sbin/ip"
SH = "/bin/sh"
MOUNT = "/usr/bin/mount"
SETPRIV = "/usr/bin/setpriv"
UNSHARE = ["/usr/bin/unshare", "-rnm", "--pid", "--fork", "--kill-child", "--mount-proc", "--propagation", "private"]
UNSHARE_NO_NETNS = ["/usr/bin/unshare", "-rm", *UNSHARE[2:]]
DROP = [SETPRIV, "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs"]
PROXY = "http://127.0.0.1:9"
PROXY_VARS = ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy")
NO_PROXY_VARS = ("no_proxy", "NO_PROXY")
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
FIXED_KEYED = {"LANG": "C.UTF-8", "TZ": "UTC"}
SYSTEM_PATH = ("/usr/bin", "/bin")
INHERITED = ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")
VOLATILE_TEMPLATES = {
    "TMPDIR": "{tmp}",
    "HOME": "{tmp}/home",
    "IPYTHONDIR": "{tmp}/ipython",
    "JUPYTER_RUNTIME_DIR": "{tmp}/jupyter/runtime",
}
RESERVED_PREFIXES = ("SLURM_", "REPRO_", "JUPYTER_", "HF_", "XDG_", "R_LIBS", "R_ENVIRON", "R_PROFILE")
R_STARTUP_TEMPLATE = "{cache}/empty-r-startup"
"""``R_ENVIRON_USER`` and ``R_PROFILE_USER`` of an R step: an empty file the driver owns, so a ``.Renviron`` or
``.Rprofile`` in the step's working directory is never read."""
MASKED_HOST_DIRS = ("/run", "/var/run", "/var/lib/sss", "/var/spool/slurmd")
"""Host directories holding daemon sockets, hidden under an empty read-only tmpfs in every sandbox."""
RESERVED = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "TORCH_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "CUDA_CACHE_PATH",
        "TMPDIR",
        "HOME",
        "IPYTHONDIR",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "GCONV_PATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONNOUSERSITE",
        *FIXED_KEYED,
        *OFFLINE,
        *INHERITED,
        *NO_PROXY_VARS,
    }
)
FIXED_PATHS = ("out", "tmp", "checkout", "venv", "cache")


def _abs(path: str | os.PathLike[str]) -> str:
    if not Path(path).is_absolute():
        raise ValueError(f"sandbox paths must be absolute: {os.fspath(path)!r}")
    return os.path.realpath(path)


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def _shallow_first(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    return sorted({_abs(p) for p in paths}, key=lambda p: (p.rstrip("/").count("/"), p))


SWEEP = """\
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
KEEP = {"nosuid": 2, "nodev": 4, "noexec": 8, "noatime": 1024, "nodiratime": 2048, "relatime": 1 << 21,
        "strictatime": 1 << 24}
PSEUDO = set(sys.argv[1].split(","))
skip = {os.fsencode(p) for p in sys.argv[2:] if not p.startswith("@")}
masked = [os.fsencode(p[1:]).rstrip(b"/") + b"/" for p in sys.argv[2:] if p.startswith("@")]
top = {}
with open("/proc/self/mountinfo", "rb") as handle:
    for line in handle:
        left, _, right = line.partition(b" - ")
        fields = left.split()
        top[fields[4]] = (fields[5], right.split()[0].decode())
bad = []
for raw, (options, fstype) in top.items():
    path = raw.decode("unicode_escape").encode("latin-1") if b"\\\\" in raw else raw
    if path in skip or fstype == "proc" or any(path.startswith(m) for m in masked):
        continue
    flags = 32 | 4096 | 1
    for option in options.split(b","):
        flags |= KEEP.get(option.decode(), 0)
    if libc.mount(None, path, None, flags, None) != 0 and fstype not in PSEUDO:
        bad.append(fstype)
if bad:
    sys.stderr.write("reproduce sandbox: mounts left writable: " + ", ".join(sorted(set(bad))) + "\\n")
    sys.exit(1)
"""
"""Run with the driver's interpreter (``-I -S``) inside the namespace, before the capability drop: remounts the
topmost mount at every mount point read-only, keeping each mount's locked flags, except the paths named on
its command line after the comma-separated pseudo types (the writable binds), the fresh ``/proc``, and every
mount point under a masked dir (named with a leading ``@``), which the mask makes unreachable. A
mount that refuses is fatal unless its type is a kernel pseudo filesystem in `PSEUDO_FS`."""

PSEUDO_FS = frozenset(
    {
        "autofs", "binfmt_misc", "bpf", "cgroup", "cgroup2", "configfs", "debugfs", "devpts", "efivarfs",
        "fusectl", "nsfs", "proc", "pstore", "rpc_pipefs", "securityfs", "sysfs", "tracefs",
    }
)  # fmt: skip
SYSTEM_TMP = "/tmp"
SHORT_TMP = "/dev/shm/repro-tmp"
"""Where the step's tmp is bound when ``/tmp`` cannot be: a short path on the sandbox's private ``/dev/shm``, so an
AF_UNIX socket under ``TMPDIR`` fits the 108-byte ``sun_path`` limit."""


def _over_tmp(seen: Iterable[str]) -> bool:
    return not any(_under(p, SYSTEM_TMP) for p in seen)


def _seen(roots: list[str], rw: list[str], extra: list[str], visible: Iterable[str | os.PathLike[str]]) -> list[str]:
    driver = [os.path.realpath(sys.executable), os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)]
    return [*roots, *rw, *extra, *(_abs(v) for v in visible), *driver]


def inner_tmp(
    ro_roots: Iterable[str | os.PathLike[str]],
    rw_dirs: Iterable[str | os.PathLike[str]],
    ro_extra: Iterable[str | os.PathLike[str]],
    *,
    tmp: str | os.PathLike[str],
    visible: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """The path the step sees its tmp at: ``/tmp`` when the bind there hides nothing, else `SHORT_TMP`.

    Takes the same arguments as `sandbox_script`, so it gives the path the script binds.
    """
    roots, rw, extra = _shallow_first(ro_roots), _shallow_first([*rw_dirs, tmp]), _shallow_first(ro_extra)
    return SYSTEM_TMP if _over_tmp(_seen(roots, rw, extra, visible)) else SHORT_TMP


def sandbox_script(
    ro_roots: Iterable[str | os.PathLike[str]],
    rw_dirs: Iterable[str | os.PathLike[str]],
    ro_extra: Iterable[str | os.PathLike[str]] = (),
    *,
    netns: bool = True,
    tmp: str | os.PathLike[str] | None = None,
    visible: Iterable[str | os.PathLike[str]] = (),
    masked: Iterable[str | os.PathLike[str]] = (),
) -> str:
    """Return the ``sh`` script `sandbox_argv` runs inside the namespace; it ends by exec'ing the step.

    Everything is read-only by default: after the binds, every mount in the namespace is remounted
    read-only (`SWEEP`), so only ``rw_dirs`` (and ``tmp``) stay writable, and a fresh private tmpfs is
    mounted on ``/dev/shm``.

    Args:
        ro_roots: Data roots; each is bound to itself first, so a missing root aborts the script.
        rw_dirs: The only directories that stay writable, each bound to itself before the sweep.
        ro_extra: Further directories (the checkout, the venv) that must exist and stay read-only.
        netns: Whether the script runs in its own network namespace and raises loopback there; False
            only for the test-only mode that runs with the host network.
        tmp: A writable directory also bound on ``/tmp``; when any path given here (or in ``visible``, or
            the driver's own interpreter, which runs the sweep) lies under ``/tmp``, the bind would hide it,
            so ``/tmp`` stays read-only and the dir is bound on `SHORT_TMP` instead (`inner_tmp` names
            which).
        visible: Other paths the step needs to see, checked only against the ``/tmp`` bind.
        masked: Directories hidden under an empty tmpfs (made read-only by the sweep), when they exist; the
            host socket dirs of `MASKED_HOST_DIRS` are always masked unless a path the sandbox needs lies
            under one.

    Returns:
        The script. It starts with ``set -e``, so a failed mount ends it before the step runs, and it
        execs the step through ``setpriv`` with every capability dropped. Each list of binds is written
        shallow-first, so a nested path is bound after its parent.

    Raises:
        ValueError: If a path is relative, a read-only path lies under (or is) a writable directory, or a
            writable directory lies under a ``ro_extra`` directory.
    """
    roots, rw, extra = _shallow_first(ro_roots), _shallow_first(rw_dirs), _shallow_first(ro_extra)
    masks = [_abs(m) for m in masked]
    if tmp is not None:
        rw = _shallow_first([*rw, tmp])
    for d in rw:
        hidden = [p for p in (*roots, *extra) if _under(p, d)]
        if hidden:
            raise ValueError(f"read-only paths under a writable directory would be hidden by its bind: {hidden}")
        if any(_under(d, e) for e in extra):
            raise ValueError(f"writable directory under a read-only checkout or venv: {d}")
    seen = _seen(roots, rw, extra, visible)
    over_tmp = tmp is not None and _over_tmp(seen)
    q = shlex.quote
    skip = [*rw, *([SYSTEM_TMP] if over_tmp else [])]
    lines = ["set -e", f"{MOUNT} --make-rprivate /"]
    hidden = [h for h in MASKED_HOST_DIRS if not any(_under(p, h) for p in seen)]
    lines += [_mask(h) for h in hidden]
    lines += [f"{MOUNT} --bind {q(r)} {q(r)}" for r in roots]
    lines += [f"{MOUNT} --bind {q(d)} {q(d)}" for d in rw]
    lines += [f"{MOUNT} --bind {q(e)} {q(e)}" for e in extra]
    if over_tmp:
        lines.append(f"{MOUNT} --bind {q(_abs(tmp))} {SYSTEM_TMP}")  # type: ignore[arg-type]
    lines += [_mask(m) for m in masks]
    pseudo = ",".join(sorted(PSEUDO_FS))
    under_masks = [f"@{p}" for p in (*hidden, *masks)]
    sweep_args = [pseudo, *skip, *under_masks]
    lines.append(" ".join([q(sys.executable), "-I", "-S", "-c", q(SWEEP), *(q(p) for p in sweep_args)]))
    lines.append(f"if [ -d /dev/shm ]; then {MOUNT} -t tmpfs -o nosuid,nodev,mode=1777 tmpfs /dev/shm; fi")
    if tmp is not None and not over_tmp:
        lines += [f"/usr/bin/mkdir {SHORT_TMP}", f"{MOUNT} --bind {q(_abs(tmp))} {SHORT_TMP}"]
    lines += [f"{IP} link set lo up"] if netns else []
    lines += [f'exec {" ".join(DROP)} "$@"']
    return "\n".join(lines) + "\n"


def _mask(path: str) -> str:
    q = shlex.quote(path)
    return (
        f"if [ -d {q} ] && [ ! -L {q} ]; then {MOUNT} -t tmpfs -o nosuid,nodev,noexec,size=64k,mode=755 tmpfs {q}; fi"
    )


def sandbox_argv(
    argv: Sequence[str],
    ro_roots: Iterable[str | os.PathLike[str]],
    rw_dirs: Iterable[str | os.PathLike[str]],
    ro_extra: Iterable[str | os.PathLike[str]] = (),
    *,
    netns: bool = True,
    tmp: str | os.PathLike[str] | None = None,
    visible: Iterable[str | os.PathLike[str]] = (),
    masked: Iterable[str | os.PathLike[str]] = (),
) -> list[str]:
    """Wrap a step's argv to run in the read-only-by-default sandbox.

    Args:
        argv: The expanded step argv; the child receives it unchanged.
        ro_roots: Data roots; they must exist.
        rw_dirs: The only directories that stay writable.
        ro_extra: The checkout and the venv; they must exist.
        netns: False drops the network namespace (`UNSHARE_NO_NETNS`, no loopback step); test-only.
        tmp: The step's scratch dir, writable and also bound on ``/tmp`` (see `sandbox_script`).
        visible: Other paths the step must see, checked against the ``/tmp`` bind.
        masked: Directories hidden under an empty read-only tmpfs (see `sandbox_script`).

    Returns:
        ``[*UNSHARE, "/bin/sh", "-c", SCRIPT, "sh", *argv]``, where `UNSHARE` is ``/usr/bin/unshare -rnm
        --pid --fork --kill-child --mount-proc --propagation private``.

    Raises:
        ValueError: If argv is empty, or as `sandbox_script`.
    """
    if not argv:
        raise ValueError("empty argv")
    unshare = UNSHARE if netns else UNSHARE_NO_NETNS
    script = sandbox_script(ro_roots, rw_dirs, ro_extra, netns=netns, tmp=tmp, visible=visible, masked=masked)
    return [*unshare, SH, "-c", script, "sh", *argv]


_TOOL_MASKS: tuple[str, ...] | None = None


@contextlib.contextmanager
def tool_masks(dirs: Iterable[str | os.PathLike[str]]) -> Iterator[None]:
    """Mask ``dirs`` (with any already set) in every `tool_argv` sandbox while the block runs."""
    global _TOOL_MASKS  # noqa: PLW0603
    saved = _TOOL_MASKS
    _TOOL_MASKS = (*(saved or ()), *(os.fspath(d) for d in dirs))
    try:
        yield
    finally:
        _TOOL_MASKS = saved


def active_tool_masks() -> tuple[str, ...] | None:
    """The dirs `tool_argv` masks now; None outside every `tool_masks` block."""
    return _TOOL_MASKS


def tool_argv(
    argv: Sequence[str], rw_dirs: Iterable[str | os.PathLike[str]], ro_extra: Iterable[str | os.PathLike[str]] = ()
) -> list[str]:
    """`sandbox_argv` for build and env tools: no data root is bound, and the `tool_masks` dirs are masked.

    Raises:
        ValueError: ``E_SANDBOX_NO_MASKS`` outside every `tool_masks` block, so a caller with no profile
            cannot run third-party code with restricted dirs visible; or as `sandbox_argv`.
    """
    if _TOOL_MASKS is None:
        raise ValueError("E_SANDBOX_NO_MASKS: build and env sandboxes run only inside sandbox.tool_masks")
    return sandbox_argv(argv, [], rw_dirs, ro_extra, masked=_TOOL_MASKS)


SWEEP_REFUSED = "reproduce sandbox: mounts left writable"
"""The start of the sweep's message when a mount refuses the read-only remount (it names filesystem types only)."""


def sweep_refusal(text: str) -> str | None:
    """The sweep's refusal line in ``text`` (a sandbox's stderr), or None."""
    for line in text.splitlines():
        if line.startswith(SWEEP_REFUSED):
            return line.strip()
    return None


def sandbox_status() -> tuple[str, str]:
    """Whether a sandbox can be built here: ``("ok", "")``, ``("no-userns", detail)`` or ``("sweep-refused", line)``.

    ``sweep-refused`` means the namespaces work but a mount refused the read-only remount, so this node cannot
    host a sandbox; the line names the refusing filesystem types.
    """
    try:
        done = subprocess.run(
            sandbox_argv(["true"], [], []),
            env={"PATH": ":".join(SYSTEM_PATH)},
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return "no-userns", type(error).__name__
    if done.returncode == 0:
        return "ok", ""
    stderr = done.stderr.decode(errors="replace") if isinstance(done.stderr, bytes) else str(done.stderr or "")
    refusal = sweep_refusal(stderr)
    if refusal is not None:
        return "sweep-refused", refusal
    return "no-userns", stderr.strip()[-200:]


def userns_available() -> bool:
    """Return whether a sandbox can be built here (`sandbox_status` is ``ok``)."""
    return sandbox_status()[0] == "ok"


def _escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _reserved(key: str) -> bool:
    return (
        key in RESERVED or key in VOLATILE_TEMPLATES or key.startswith(RESERVED_PREFIXES) or key.lower() in PROXY_VARS
    )


_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
"""`spec.render`'s token grammar: ``{{``, ``}}``, one placeholder, or a stray brace."""


def _native(step: StepSpec, profile: Profile, kind: str) -> list[str]:
    """The ``{native:<name>:<i>}`` templates of the selected entries' ``bin_dirs`` or ``lib_dirs``.

    ``i`` counts an entry's ``bin_dirs`` then its ``lib_dirs``, as `cache.native_digests` does, so the keyed env
    names each dir by entry and position and its host path is not keyed (its content is, by digest).
    """
    out = []
    for name in step.resources.native:
        entry = profile.native[name]
        start = 0 if kind == "bin_dirs" else len(entry.bin_dirs)
        out += [f"{{native:{name}:{start + i}}}" for i in range(len(getattr(entry, kind)))]
    return out


def _native_values(step: StepSpec, profile: Profile) -> dict[str, str]:
    """Each ``native:<name>:<i>`` placeholder of `_native` mapped to its absolute dir."""
    values = {}
    for name in step.resources.native:
        entry = profile.native[name]
        for i, path in enumerate([*entry.bin_dirs, *entry.lib_dirs]):
            values[f"native:{name}:{i}"] = os.fspath(path)
    return values


def _render_native(template: str, values: Mapping[str, str]) -> str:
    """Expand the ``{native:<name>:<i>}`` placeholders only; every other token is kept for `render`."""

    def expand(match: re.Match[str]) -> str:
        name = match.group(1)
        if name is not None and name.startswith("native:"):
            return _escape(values[name])
        return match.group(0)

    return _TOKEN.sub(expand, template)


def _sha(env: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(env), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _keyed(step: StepSpec, profile: Profile) -> tuple[dict[str, str], dict[str, str]]:
    check_step_profile(step, profile)
    bad = sorted(k for k in profile.env if _reserved(k))
    if bad:
        raise ValueError(f"profile env may not set driver-owned variables: {bad}")
    colon = [p for p in step.pythonpath if ":" in p]
    if colon:
        raise ValueError(f"pythonpath entries may not contain ':': {colon}")
    templated = {
        "PATH": ":".join(["{venv}/bin", *SYSTEM_PATH, *_native(step, profile, "bin_dirs")]),
        "JUPYTER_PATH": "{venv}/share/jupyter",
        "JUPYTER_DATA_DIR": "{tmp}/jupyter/data",
        "JUPYTER_CONFIG_DIR": "{tmp}/jupyter/config",
        "HF_HOME": "{cache}/huggingface",
        "TORCH_HOME": "{cache}/torch",
        "XDG_CACHE_HOME": "{tmp}/cache",
        "CUDA_CACHE_PATH": "{tmp}/cuda-cache",
        "REPRO_OUT": "{out}",
        "REPRO_TMP": "{tmp}",
    }
    if step.hf_weights:
        templated["HF_HOME"] = "{tmp}/hf"
        templated["HF_HUB_CACHE"] = "{tmp}/hf-hub"
        templated["HUGGINGFACE_HUB_CACHE"] = "{tmp}/hf-hub"
    if step.url_weights:
        templated["TORCH_HOME"] = "{tmp}/torch"
    if step.pythonpath:
        templated["PYTHONPATH"] = ":".join(step.pythonpath)
    if _native(step, profile, "lib_dirs"):
        templated["LD_LIBRARY_PATH"] = ":".join(_native(step, profile, "lib_dirs"))
    literal = {**FIXED_KEYED, **OFFLINE}
    literal |= {v: PROXY for v in PROXY_VARS} | {v.upper(): PROXY for v in PROXY_VARS} | {v: "" for v in NO_PROXY_VARS}
    if step.env.r_lib is not None:
        templated["R_LIBS"] = step.env.r_lib
        templated["R_LIBS_USER"] = step.env.r_lib
        templated["R_ENVIRON_USER"] = R_STARTUP_TEMPLATE
        templated["R_PROFILE_USER"] = R_STARTUP_TEMPLATE
        literal["R_LIBS_SITE"] = ""
    if step.shards is not None:
        literal["REPRO_SHARDS"] = str(step.shards.count)
    if step.draw is not None:
        literal |= _draw_env(step.draw)
    if step.engine is not None:
        literal["REPRO_ENGINE_IDENTITY"] = engine.identity_json(step.engine)
        if step.engine.name == "vllm":
            literal |= {v: engine.LOOPBACK for v in NO_PROXY_VARS}
            templated |= engine.CACHE_TEMPLATES
    return {**profile.env, **literal}, templated


def _draw_env(draw: DrawSpec) -> dict[str, str]:
    """``REPRO_SEED``, ``REPRO_DATA_ORDER_SEED`` and ``REPRO_SAMPLING`` (compact sorted JSON), each only when set."""
    out: dict[str, str] = {}
    if draw.seed is not None:
        out["REPRO_SEED"] = str(draw.seed)
    if draw.data_order_seed is not None:
        out["REPRO_DATA_ORDER_SEED"] = str(draw.data_order_seed)
    if draw.sampling:
        out["REPRO_SAMPLING"] = json.dumps(draw.sampling, sort_keys=True, separators=(",", ":"))
    return out


def keyed_templates(step: StepSpec, profile: Profile) -> dict[str, str]:
    """Return the keyed part of a step's env, with path variables in template form.

    Args:
        step: The step.
        profile: The caller's profile.

    Returns:
        Name to value or template, e.g. ``{"PATH": "{venv}/bin:/usr/bin:/bin", "TZ": "UTC", ...}``.

    Raises:
        ValueError: If the profile env map names a driver-owned variable, a ``pythonpath`` entry
            contains ``:``, or as `spec.check_step_profile`.
    """
    literal, templated = _keyed(step, profile)
    return literal | templated


def build_env(
    step: StepSpec,
    profile: Profile,
    paths: Mapping[str, str | os.PathLike[str]],
    parent: Mapping[str, str] | None = None,
    shard: int | None = None,
) -> tuple[dict[str, str], str, str]:
    """Build a step's whole environment.

    Args:
        step: The step.
        profile: The caller's profile; its ``roots`` fill the ``{root:<name>}`` placeholders.
        paths: Values for ``out``, ``tmp``, ``checkout``, ``venv`` and ``cache``, plus any ``up:<step>``.
        parent: Where the recorded-only job variables (``SLURM_JOB_ID``, ``SLURM_ARRAY_JOB_ID``,
            ``SLURM_ARRAY_TASK_ID``, ``SLURM_CPUS_PER_TASK``, ``CUDA_VISIBLE_DEVICES``) are read from;
            defaults to ``os.environ``. Nothing else is read from it.
        shard: A sharded step's shard index, set as ``REPRO_SHARD`` (keyed by the step key's ``shard``, so it
            is not in the keyed sha).

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
    native = _native_values(step, profile)
    env = literal | {k: render(_render_native(v, native), values) for k, v in templated.items()}
    env |= {k: render(v, values) for k, v in VOLATILE_TEMPLATES.items()}
    source = os.environ if parent is None else parent
    env |= {k: v for k, v in source.items() if k in INHERITED}
    if shard is not None:
        env["REPRO_SHARD"] = str(shard)
    return env, _sha(literal | templated), _sha(env)


def canary(host: str, port: int = 443, timeout: float = 5.0) -> str:
    """Return a Python snippet that tries a TCP connect and prints ``{"canary_failed": <bool>}``.

    Args:
        host: The profile's canary host: a name or an address, optionally ``host:port``.
        port: The TCP port when ``host`` carries none.
        timeout: Seconds before the connect counts as failed.

    Returns:
        Source for ``python -c``; it always exits 0. Inside the sandbox the connect must fail.
    """
    name, sep, tail = host.rpartition(":")
    if sep and tail.isdigit() and ":" not in name:
        host, port = name, int(tail)
    return (
        "import json, socket\n"
        "try:\n"
        f"    socket.create_connection(({host!r}, {int(port)}), timeout={float(timeout)}).close()\n"
        "    failed = False\n"
        "except OSError:\n"
        "    failed = True\n"
        'print(json.dumps({"canary_failed": failed}))\n'
    )
