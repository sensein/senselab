"""Step keys, content-addressed result dirs, the completion marker, run claims and dated run links.

`step_key` is the sha256 of `canonical_json` over `step_key_payload`: everything that decides a
step's result, and nothing volatile (job ids, visible GPUs, ``TMPDIR``, ``HOME``, the date, where the
data roots or results live, the partition and limits, ``min_driver``, the step's own name or its
repository URL). `native_digests` and `driver_env_sha256` compute two of its run-time inputs, and
`reproduce_source_sha256` stands for the driver's own behaviour. A step's outputs live in
`result_dir` = ``<results_root>/<step_key>``. The dir is complete (`is_done`) once its
``summary.json`` exists with ``"exit": 0``; `write_summary` writes that file whole or not at all,
and is the last write of a run. `claim` holds a `SharedFileLock` on
``<results_root>/.claims/<step_key>`` so two jobs never compute one step at once. `link_run` points
``<runs_root>/<YYYY-MM-DD>/<analysis>/<step>`` at the step's result dir with a relative symlink.

Errors are a `CacheError` whose message is a fixed code only.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sysconfig
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import Distribution, distributions
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from senselab.utils.file_lock import SharedFileLock
from senselab.utils.reproduce import engine
from senselab.utils.reproduce.digest import input_set_digest
from senselab.utils.reproduce.env import env_key
from senselab.utils.reproduce.sandbox import keyed_templates
from senselab.utils.reproduce.source import patch_set_sha256
from senselab.utils.reproduce.spec import Profile, StepSpec

STEP_KEY_SCHEMA = "reproduce-step-key-4"
REPRODUCE_SOURCE_SCHEMA = "reproduce-source-1"
SHARDS_KEY_SCHEMA = "reproduce-shards-1"
DRIVER_PACKAGE = "senselab"
NATIVE_EXCLUDE = ("**/__pycache__/**", "**/*.pyc")
SUMMARY = "summary.json"
ATTEMPTS = "attempts.jsonl"
RESUME_KEEP = ".resume-keep"
CLAIMS = ".claims"
R_KINDS = frozenset({"r", "ir_notebook"})
CLAIM_TIMEOUT_S = 600.0
CODES = frozenset(
    {
        "E_KEY_BAD",
        "E_KEY_UPSTREAM",
        "E_KEY_INPUTS",
        "E_KEY_ADAPTER",
        "E_KEY_NATIVE",
        "E_KEY_SUBMODULES",
        "E_KEY_R_SESSION",
        "E_KEY_NO_WHEELS",
        "E_KEY_LOCK",
        "E_KEY_SHARD",
        "E_RESULT_DONE",
        "E_RESULT_UNREADABLE",
        "E_RESULT_NOT_DIR",
        "E_RESULT_RESET",
        "E_RESULT_MISSING",
        "E_CLAIM_HELD",
        "E_RUN_BAD_NAME",
        "E_LINK_NOT_SYMLINK",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class CacheError(Exception):
    """A step key, result dir, claim or run link failure; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: One of `CODES`.
        """
        super().__init__(code)
        self.code = code


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.match(value))


def canonical_json(obj: object) -> bytes:
    """The bytes a step key hashes: JSON with sorted keys, no whitespace, ASCII only, no NaN or infinity.

    Args:
        obj: A JSON-serialisable value.

    Returns:
        The UTF-8 (in fact ASCII) encoding.

    Raises:
        ValueError: For a NaN or infinite float.
        TypeError: For a value JSON cannot encode.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _native_dirs(profile: Profile, name: str) -> list[tuple[Path, list[str]]]:
    spec = profile.native[name]
    return [(d, spec.bin_patterns) for d in spec.bin_dirs] + [(d, spec.lib_patterns) for d in spec.lib_dirs]


def native_digests(step: StepSpec, profile: Profile) -> dict[str, list[str]]:
    """Digest the content of every native dir a step selects, for `step_key`'s ``native``.

    Args:
        step: The step.
        profile: The caller's profile.

    Returns:
        Each name in ``step.resources.native`` mapped to one `digest.input_set_digest` hex per dir
        (``bin_dirs`` then ``lib_dirs``), over the files the entry's ``bin_patterns`` or
        ``lib_patterns`` select, leaving out `NATIVE_EXCLUDE` (``__pycache__`` dirs and ``*.pyc``).

    Raises:
        CacheError: ``E_KEY_NATIVE`` if the step selects a native entry the profile lacks.
        DigestError: As `digest.input_set_digest` with ``require_nonempty``: ``E_DIGEST_EMPTY`` when a dir
            matches nothing, ``E_DIGEST_UNREADABLE`` for a dangling symlink or unreadable file among the
            matches.
    """
    if not set(step.resources.native) <= set(profile.native):
        raise CacheError("E_KEY_NATIVE")
    return {
        name: [
            input_set_digest(d, patterns, exclude_patterns=NATIVE_EXCLUDE, require_nonempty=True)[0]
            for d, patterns in _native_dirs(profile, name)
        ]
        for name in step.resources.native
    }


@functools.cache
def reproduce_source_sha256() -> str:
    """The identity of the driver's behaviour: a sha256 over the installed ``senselab.utils.reproduce`` sources.

    Computed once per process, over every ``*.py`` file under the package directory (this module's
    own directory, i.e. ``Path(senselab.utils.reproduce.__file__).parent``), as sorted pairs of
    relative POSIX path and content sha256.

    Returns:
        A hex sha256.
    """
    root = Path(__file__).resolve().parent
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    pairs = [[rel, hashlib.sha256((root / rel).read_bytes()).hexdigest()] for rel in files]
    return hashlib.sha256(canonical_json({"schema": REPRODUCE_SOURCE_SCHEMA, "files": pairs})).hexdigest()


def _site_dirs() -> list[str]:
    paths = sysconfig.get_paths()
    return sorted({paths["purelib"], paths["platlib"]})


def _runtime_closure(installed: Mapping[str, Distribution]) -> set[str]:
    """`DRIVER_PACKAGE`'s installed runtime dependency closure, transitively.

    `DRIVER_PACKAGE`'s own requirements are taken with no extra; a requirement that asks for extras
    (``dep[x]``) also brings in ``dep``'s requirements for those extras.
    """
    seen: set[tuple[str, str]] = set()
    names: set[str] = set()
    todo: list[tuple[str, frozenset[str]]] = [(DRIVER_PACKAGE, frozenset())]
    while todo:
        name, extras = todo.pop()
        wanted = {"", *extras} - {extra for seen_name, extra in seen if seen_name == name}
        if name not in installed or not wanted:
            continue
        seen.update((name, extra) for extra in wanted)
        names.add(name)
        for text in installed[name].requires or []:
            try:
                req = Requirement(text)
            except InvalidRequirement:
                continue
            if req.marker is None or any(req.marker.evaluate({"extra": extra}) for extra in wanted):
                todo.append((canonicalize_name(req.name), frozenset(canonicalize_name(e) for e in req.extras)))
    return names


def driver_env_sha256() -> str:
    """The identity of the driver's own interpreter env, for an adapter that runs in it.

    Only the interpreter's own ``purelib`` and ``platlib`` dirs are read (never ``sys.path``, the cwd or
    ``PYTHONPATH``), and only `DRIVER_PACKAGE`'s runtime dependency closure there: its requirements
    without an extra, followed transitively along with the extras each requirement asks for.
    `DRIVER_PACKAGE` itself is left out; its reproduce code is `reproduce_source_sha256`.

    Returns:
        A hex sha256 over the interpreter's implementation, version and free-threading build flag
        (``platform.python_implementation()``, ``platform.python_version()``, ``Py_GIL_DISABLED``; not
        ``sys.version``, whose build date and compiler differ between builds of one release) and the sorted
        ``name==version`` of that closure.
    """
    installed: dict[str, Distribution] = {}
    for dist in distributions(path=_site_dirs()):
        name = dist.metadata["Name"] if dist.read_text("METADATA") else None
        if name:
            installed.setdefault(canonicalize_name(name), dist)
    closure = _runtime_closure(installed) - {DRIVER_PACKAGE}
    names = sorted(f"{name}=={installed[name].version}" for name in closure)
    python = [
        platform.python_implementation(),
        platform.python_version(),
        bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    ]
    return hashlib.sha256(canonical_json({"python": python, "distributions": names})).hexdigest()


def step_key_payload(
    step: StepSpec,
    profile: Profile,
    *,
    upstream: Mapping[str, str],
    submodules: Mapping[str, str],
    adapter_sha256: str,
    adapter_env_sha256: str,
    input_digests: Sequence[str],
    native: Mapping[str, Sequence[str]],
    sources_lock_sha256: str,
    r_session_sha256: str | None = None,
    r_module_env_sha256: str | None = None,
    shard: int | None = None,
) -> dict[str, Any]:
    """The object `step_key` hashes.

    Args:
        step: The step.
        profile: The caller's profile; only the step's keyed env (`sandbox.keyed_templates`) and its
            ``policy`` (as `policy_sha256`) are taken from it.
        upstream: Each name in ``step.needs`` mapped to that step's key, and nothing else.
        submodules: Submodule path to 40-hex commit, as the checkout records it; empty when the step
            does not use submodules.
        adapter_sha256: The sha256 of the step's adapter script.
        adapter_env_sha256: The identity of the interpreter env that runs the adapter: the step venv's
            ``installed_sha256``, or `driver_env_sha256` when the adapter runs in the driver's env.
        input_digests: One `digest.input_set_digest` hex per ``step.inputs`` entry, in manifest order.
        native: Each name in ``step.resources.native`` mapped to `native_digests`' digests for it (one per
            ``bin_dirs`` then ``lib_dirs`` entry of the profile's `NativeSpec`), and nothing else.
        sources_lock_sha256: The sha256 of the manifest's sources lock (`lock.load_lock`); with the step's
            ``lock_inputs`` ids it stands for the registered inputs, whose files are not read for the key.
        r_session_sha256: The sha256 of the prefetch ``SESSION.json``; required for ``r`` and
            ``ir_notebook`` steps and refused for others.
        r_module_env_sha256: The sha256 of the captured R module env; required and refused likewise.
        shard: The shard index of a sharded step (``0 <= shard < shards.count``), keyed as ``[shard, count]``;
            required for a sharded step and refused for others.

    Returns:
        A JSON-serialisable dict, tagged with `STEP_KEY_SCHEMA`.

    Raises:
        CacheError: ``E_KEY_UPSTREAM`` if ``upstream`` is not exactly the needs with hex keys,
            ``E_KEY_INPUTS`` if the input digests do not match ``step.inputs`` one to one,
            ``E_KEY_ADAPTER`` for an adapter or adapter-env sha256 that is not hex, ``E_KEY_NATIVE`` if
            ``native`` does not give one hex digest per selected native dir, ``E_KEY_SUBMODULES`` for a commit
            that is not 40-hex or submodules on a step that disables them, ``E_KEY_R_SESSION`` for R
            digests missing on an R step or given on another, ``E_KEY_NO_WHEELS`` for an env with
            sdist builds and no recorded ``wheels.lock``, ``E_KEY_LOCK`` for a lock sha256 that is not hex,
            ``E_KEY_SHARD`` for a shard index missing, out of range, or given for a step with no shards.
        ValueError: As `sandbox.keyed_templates`.
    """
    if not _is_sha256(sources_lock_sha256):
        raise CacheError("E_KEY_LOCK")
    if set(upstream) != set(step.needs) or not all(_is_sha256(k) for k in upstream.values()):
        raise CacheError("E_KEY_UPSTREAM")
    if len(input_digests) != len(step.inputs) or not all(_is_sha256(d) for d in input_digests):
        raise CacheError("E_KEY_INPUTS")
    if not (_is_sha256(adapter_sha256) and _is_sha256(adapter_env_sha256)):
        raise CacheError("E_KEY_ADAPTER")
    selected = step.resources.native
    if set(native) != set(selected) or not all(
        name in profile.native
        and len(native[name]) == len(_native_dirs(profile, name))
        and all(_is_sha256(d) for d in native[name])
        for name in selected
    ):
        raise CacheError("E_KEY_NATIVE")
    if (submodules and not step.submodules) or not all(
        isinstance(p, str) and p and isinstance(c, str) and _COMMIT.match(c) for p, c in submodules.items()
    ):
        raise CacheError("E_KEY_SUBMODULES")
    r_digests = (r_session_sha256, r_module_env_sha256)
    if step.kind in R_KINDS:
        if not all(_is_sha256(d) for d in r_digests):
            raise CacheError("E_KEY_R_SESSION")
    elif any(d is not None for d in r_digests):
        raise CacheError("E_KEY_R_SESSION")
    count = None if step.shards is None else step.shards.count
    if (count is None) != (shard is None) or (count is not None and shard is not None and not 0 <= shard < count):
        raise CacheError("E_KEY_SHARD")
    env = step.env
    if env.sdist_allow and env.wheels is None:
        raise CacheError("E_KEY_NO_WHEELS")
    return {
        "schema": STEP_KEY_SCHEMA,
        "reproduce_source_sha256": reproduce_source_sha256(),
        "commit": step.commit,
        "submodules": dict(sorted(submodules.items())),
        "patch_set_sha256": patch_set_sha256(step.patches),
        "env_key": env_key(env),
        "wheels_lock_sha256": None if env.wheels is None else env.wheels.lock_sha256,
        "overlay_lock_sha256s": {
            "test": None if env.test is None else env.test.lock_sha256,
            "notebook_tools": None if env.notebook_tools is None else env.notebook_tools.lock_sha256,
        },
        "env": keyed_templates(step, profile),
        "kind": step.kind,
        "argv": list(step.argv),
        "cwd": step.cwd,
        "inputs": [{"root": i.root, "patterns": sorted(set(i.patterns))} for i in step.inputs],
        "input_digests": list(input_digests),
        "stage": [{"root": s.root, "patterns": sorted(set(s.patterns)), "dest": s.dest} for s in step.stage],
        "outputs": list(step.outputs),
        "native": {name: list(native[name]) for name in sorted(selected)},
        "gpu_arch": None if step.resources.gpu_arch is None else sorted(set(step.resources.gpu_arch)),
        "adapter_sha256": adapter_sha256,
        "adapter_env_sha256": adapter_env_sha256,
        "hf_weights": [
            {
                "repo": w.repo,
                "repo_type": w.repo_type,
                "revision": w.revision,
                "files": sorted(([f.path, f.sha256] for f in w.files)),
            }
            for w in sorted(step.hf_weights, key=lambda w: (w.repo, w.repo_type))
        ],
        "url_weights": sorted([w.file, w.sha256, w.place] for w in step.url_weights),
        "r_session_sha256": r_session_sha256,
        "r_module_env_sha256": r_module_env_sha256,
        "metrics_schema": step.metrics_schema.model_dump(mode="json"),
        "expect": [e.model_dump(mode="json") for e in step.expect],
        "extract": [e.model_dump(mode="json") for e in step.extract],
        "replicate": step.replicate,
        "upstream": dict(sorted(upstream.items())),
        "policy_sha256": policy_sha256(profile),
        "sources_lock_sha256": sources_lock_sha256,
        "lock_inputs": sorted(step.lock_inputs),
        "restricted": step.restricted,
        "shard": None if shard is None else [shard, count],
        "resumable": step.resumable,
        "resume_dir": step.resume_dir,
        "engine": _engine_key(step),
        "draw": None if step.draw is None or step.draw.is_empty() else step.draw.model_dump(mode="json"),
    }


def _engine_key(step: StepSpec) -> dict[str, Any] | None:
    """The keyed engine: its identity, the served model and, for ``vllm``, the server module's sha256."""
    spec = step.engine
    if spec is None:
        return None
    return {
        "identity": engine.engine_identity(spec),
        "model": [spec.model.repo, spec.model.revision],
        "server_module_sha256": engine.server_module_sha256() if spec.name == "vllm" else None,
    }


def policy_sha256(profile: Profile) -> str:
    """The sha256 of `canonical_json` over the profile's ``policy``, so a policy change re-keys every step."""
    return hashlib.sha256(canonical_json(profile.policy.model_dump(mode="json"))).hexdigest()


def step_key(
    step: StepSpec,
    profile: Profile,
    *,
    upstream: Mapping[str, str],
    submodules: Mapping[str, str],
    adapter_sha256: str,
    adapter_env_sha256: str,
    input_digests: Sequence[str],
    native: Mapping[str, Sequence[str]],
    sources_lock_sha256: str,
    r_session_sha256: str | None = None,
    r_module_env_sha256: str | None = None,
    shard: int | None = None,
) -> str:
    """A step's cache key: the sha256 of `canonical_json` over `step_key_payload`.

    Args:
        step: The step.
        profile: The caller's profile.
        upstream: Each need's step key.
        submodules: Submodule path to commit.
        adapter_sha256: The adapter script's sha256.
        adapter_env_sha256: The identity of the env that runs the adapter.
        input_digests: One input-set digest per ``step.inputs`` entry.
        native: Each selected native entry's dir digests.
        sources_lock_sha256: The sha256 of the manifest's sources lock.
        r_session_sha256: The prefetch ``SESSION.json`` sha256 (R steps only).
        r_module_env_sha256: The captured R module env sha256 (R steps only).
        shard: The shard index (sharded steps only).

    Returns:
        A 64-hex key.

    Raises:
        CacheError: As `step_key_payload`.
        ValueError: As `step_key_payload`.
    """
    payload = step_key_payload(
        step,
        profile,
        upstream=upstream,
        submodules=submodules,
        adapter_sha256=adapter_sha256,
        adapter_env_sha256=adapter_env_sha256,
        input_digests=input_digests,
        native=native,
        sources_lock_sha256=sources_lock_sha256,
        r_session_sha256=r_session_sha256,
        r_module_env_sha256=r_module_env_sha256,
        shard=shard,
    )
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def shards_key(keys: Sequence[str]) -> str:
    """The key a sharded step stands under for its dependants: a sha256 over its shard keys in shard order.

    Args:
        keys: Each shard's step key, by index.

    Returns:
        A 64-hex key; no result dir has it.

    Raises:
        CacheError: ``E_KEY_BAD`` if a key is not 64 lowercase hex characters.
    """
    if not all(_is_sha256(k) for k in keys):
        raise CacheError("E_KEY_BAD")
    return hashlib.sha256(canonical_json({"schema": SHARDS_KEY_SCHEMA, "shards": list(keys)})).hexdigest()


def result_dir(results_root: Path, key: str) -> Path:
    """Where a step's outputs live: ``<results_root>/<key>``.

    Args:
        results_root: The profile's results root.
        key: A step key.

    Returns:
        The result dir; it is not created.

    Raises:
        CacheError: ``E_KEY_BAD`` if ``key`` is not 64 lowercase hex characters.
    """
    if not _is_sha256(key):
        raise CacheError("E_KEY_BAD")
    return results_root / key


def is_done(out: Path) -> bool:
    """Whether a result dir is complete: its ``summary.json`` is a JSON object with integer ``exit`` 0.

    Args:
        out: A result dir.

    Returns:
        True only for a complete, successful run; a missing dir, missing or unreadable summary, a
        non-zero or non-integer exit is False.
    """
    try:
        summary = json.loads((out / SUMMARY).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(summary, dict) and type(summary.get("exit")) is int and summary["exit"] == 0


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_summary(out: Path, summary: Mapping[str, Any]) -> Path:
    """Write ``summary.json`` into a result dir atomically: a fsynced temporary file renamed into place.

    Args:
        out: An existing result dir.
        summary: The summary; ``"exit": 0`` marks the dir done.

    Returns:
        The summary path.

    Raises:
        ValueError: For a NaN or infinite value; nothing is written.
        TypeError: For a value JSON cannot encode; nothing is written.
        OSError: If the write or rename fails; the temporary file is removed and any earlier summary
            is left as it was.
    """
    text = json.dumps(dict(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = out / SUMMARY
    tmp = out / f".{SUMMARY}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660)
    try:
        try:
            os.write(fd, text.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(out)
    return target


def _real_dir(base: Path, rel: str) -> Path | None:
    """``base/rel`` when every component of ``rel`` is a real directory (no symlink), else None."""
    path = base
    for part in PurePosixPath(rel).parts:
        path = path / part
        try:
            if not stat.S_ISDIR(os.lstat(path).st_mode):
                return None
        except FileNotFoundError:
            return None
    return path


def _remove(path: Path) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def _wipe_keeping(out: Path, keep: str | None, keep_attempts: bool) -> None:
    """Empty ``out`` but for ``out/<keep>`` (moved to `RESUME_KEEP` and back) and, if asked, `ATTEMPTS`."""
    held = out / RESUME_KEEP
    store = _real_dir(out, keep) if keep is not None else None
    if store is not None:
        if os.path.lexists(held):
            _remove(held)
        os.rename(store, held)
    for entry in os.scandir(out):
        attempts = keep_attempts and entry.name == ATTEMPTS and entry.is_file(follow_symlinks=False)
        if entry.name != RESUME_KEEP and not attempts:
            _remove(Path(entry.path))
    if keep is not None and _real_dir(out, RESUME_KEEP) is not None:
        parent = out
        for part in PurePosixPath(keep).parts[:-1]:
            parent = parent / part
            parent.mkdir(mode=0o770, exist_ok=True)
        os.rename(held, out / keep)
    elif os.path.lexists(held):
        _remove(held)


def reset_result_dir(results_root: Path, key: str, *, keep: str | None = None, keep_attempts: bool = False) -> Path:
    """Give a step an empty result dir, removing what an incomplete or failed run left; call under `claim`.

    Args:
        results_root: The profile's results root.
        key: The step key.
        keep: A dir relative to the result dir (an attempt's ``work/<resume_dir>``) kept through the wipe: moved
            to `RESUME_KEEP` and back, only when every component is a real directory. A `RESUME_KEEP` left by a
            reset that died is restored the same way.
        keep_attempts: Keep `ATTEMPTS`, the per-attempt record.

    Returns:
        The step's result dir (`result_dir`), now empty but for what was kept.

    A dir is wiped only when its ``summary.json`` is absent or records a non-zero integer exit.

    Raises:
        CacheError: ``E_KEY_BAD`` for a malformed key, ``E_RESULT_NOT_DIR`` if a symlink or non-directory
            holds the path, ``E_RESULT_DONE`` if the dir is complete, ``E_RESULT_UNREADABLE`` if the
            summary cannot be read or decoded or is not an object with an integer exit (nothing is deleted then),
            ``E_RESULT_RESET`` if removing or creating it fails.
    """
    out = result_dir(results_root, key)
    if out.is_symlink() or (os.path.lexists(out) and not out.is_dir()):
        raise CacheError("E_RESULT_NOT_DIR")
    if out.is_dir():
        try:
            text = (out / SUMMARY).read_text()
        except FileNotFoundError:
            text = None
        except (OSError, ValueError):
            raise CacheError("E_RESULT_UNREADABLE") from None
        if text is not None:
            try:
                summary = json.loads(text)
            except ValueError:
                raise CacheError("E_RESULT_UNREADABLE") from None
            if not isinstance(summary, dict) or type(summary.get("exit")) is not int:
                raise CacheError("E_RESULT_UNREADABLE")
            if summary["exit"] == 0:
                raise CacheError("E_RESULT_DONE")
    try:
        if out.exists() and (keep is not None or keep_attempts):
            _wipe_keeping(out, keep, keep_attempts)
            return out
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    except OSError:
        raise CacheError("E_RESULT_RESET") from None
    return out


@contextmanager
def claim(results_root: Path, key: str, *, timeout: float = CLAIM_TIMEOUT_S) -> Iterator[tuple[Path, bool]]:
    """Hold the exclusive claim on one step key while it is checked, computed and marked done.

    The lock files are ``<results_root>/.claims/<key>.{lock,heartbeat,holder}``; the results root's own
    mode is never changed. ``leftover`` is the lock's own takeover verdict (`SharedFileLock.took_over`): an
    identity in ``.holder`` read before an uncontended acquire means a holder that never released, i.e. one that
    was killed, since a clean release clears ``.holder`` and a waiter's acquire is contended.

    Args:
        results_root: The profile's results root.
        key: The step key.
        timeout: Seconds to wait for a live holder before giving up.

    Yields:
        ``(out, leftover)``: the step's result dir (`result_dir`), and whether a killed holder's identity was
        left. Re-check `is_done` inside the claim, since another job may have completed the step while this
        one waited.

    Raises:
        CacheError: ``E_KEY_BAD`` for a malformed key, ``E_CLAIM_HELD`` if another live process held
            the claim for the whole timeout.
    """
    out = result_dir(results_root, key)
    claims = results_root / CLAIMS
    claims.mkdir(parents=True, exist_ok=True)
    lock = SharedFileLock(claims / key, timeout=timeout, manage_dir_mode=False)
    try:
        lock.__enter__()
    except TimeoutError:
        raise CacheError("E_CLAIM_HELD") from None
    try:
        yield out, lock.took_over
    finally:
        lock.__exit__(None, None, None)


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise CacheError("E_RUN_BAD_NAME")
    return value


def run_dir(runs_root: Path, date: datetime.date, analysis: str) -> Path:
    """The dated run dir of one analysis: ``<runs_root>/<YYYY-MM-DD>/<analysis>``.

    Args:
        runs_root: The profile's runs root.
        date: The run date.
        analysis: The manifest name.

    Returns:
        The run dir; it is not created.

    Raises:
        CacheError: ``E_RUN_BAD_NAME`` if ``analysis`` is not a plain name (letters, digits, ``_``,
            ``-``, ``.``, no leading dot).
    """
    return runs_root / date.strftime("%Y-%m-%d") / _name(analysis)


def link_run(run: Path, step: str, out: Path) -> Path:
    """Point ``<run>/<step>`` at a result dir with a relative symlink, replacing any older link atomically.

    Args:
        run: The run dir (`run_dir`); created if absent.
        step: The step name.
        out: The step's result dir.

    Returns:
        The link path.

    Raises:
        CacheError: ``E_RUN_BAD_NAME`` for a step name that is not a plain name, ``E_RESULT_MISSING`` if
            ``out`` is not a directory, ``E_LINK_NOT_SYMLINK`` if something other than a symlink holds
            the link path.
    """
    _name(step)
    if not out.is_dir():
        raise CacheError("E_RESULT_MISSING")
    run.mkdir(parents=True, exist_ok=True)
    link = run / step
    if os.path.lexists(link) and not link.is_symlink():
        raise CacheError("E_LINK_NOT_SYMLINK")
    target = os.path.relpath(os.path.realpath(out), os.path.realpath(run))
    if link.is_symlink() and os.readlink(link) == target:
        return link
    tmp = run / f".{step}.{uuid.uuid4().hex}.tmp"
    os.symlink(target, tmp)
    try:
        os.replace(tmp, link)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return link
