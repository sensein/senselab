"""Step keys, content-addressed result dirs, the completion marker, run claims and dated run links.

`step_key` is the sha256 of `canonical_json` over `step_key_payload`: everything that decides a
step's result, and nothing volatile (job ids, visible GPUs, ``TMPDIR``, ``HOME``, the date, where the
data roots or results live, the partition, the step's name or repository URL). A step's outputs live
in `result_dir` = ``<results_root>/<step_key>``. The dir is complete (`is_done`) once its
``summary.json`` exists with ``"exit": 0``; `write_summary` writes that file whole or not at all,
and is the last write of a run. `claim` holds a `SharedFileLock` on
``<results_root>/.claims/<step_key>`` so two jobs never compute one step at once. `link_run` points
``<runs_root>/<YYYY-MM-DD>/<analysis>/<step>`` at the step's result dir with a relative symlink.

Errors are a `CacheError` whose message is a fixed code only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from senselab.utils.file_lock import SharedFileLock
from senselab.utils.reproduce.env import env_key
from senselab.utils.reproduce.sandbox import keyed_templates
from senselab.utils.reproduce.source import patch_set_sha256
from senselab.utils.reproduce.spec import Profile, StepSpec

STEP_KEY_SCHEMA = "reproduce-step-key-1"
SUMMARY = "summary.json"
CLAIMS = ".claims"
R_KINDS = frozenset({"r", "ir_notebook"})
CLAIM_TIMEOUT_S = 600.0
CODES = frozenset(
    {
        "E_KEY_BAD",
        "E_KEY_UPSTREAM",
        "E_KEY_INPUTS",
        "E_KEY_ADAPTER",
        "E_KEY_SUBMODULES",
        "E_KEY_R_SESSION",
        "E_KEY_NO_WHEELS",
        "E_RESULT_DONE",
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


def step_key_payload(
    step: StepSpec,
    profile: Profile,
    *,
    upstream: Mapping[str, str],
    submodules: Mapping[str, str],
    adapter_sha256: str,
    input_digests: Sequence[str],
    r_session_sha256: str | None = None,
    r_module_env_sha256: str | None = None,
) -> dict[str, Any]:
    """The object `step_key` hashes.

    Args:
        step: The step.
        profile: The caller's profile; only the step's keyed env (`sandbox.keyed_templates`) is taken
            from it.
        upstream: Each name in ``step.needs`` mapped to that step's key, and nothing else.
        submodules: Submodule path to 40-hex commit, as the checkout records it; empty when the step
            does not use submodules.
        adapter_sha256: The sha256 of the step's adapter script.
        input_digests: One `digest.input_set_digest` hex per ``step.inputs`` entry, in manifest order.
        r_session_sha256: The sha256 of the prefetch ``SESSION.json``; required for ``r`` and
            ``ir_notebook`` steps and refused for others.
        r_module_env_sha256: The sha256 of the captured R module env; required and refused likewise.

    Returns:
        A JSON-serialisable dict, tagged with `STEP_KEY_SCHEMA`.

    Raises:
        CacheError: ``E_KEY_UPSTREAM`` if ``upstream`` is not exactly the needs with hex keys,
            ``E_KEY_INPUTS`` if the input digests do not match ``step.inputs`` one to one,
            ``E_KEY_ADAPTER`` for an adapter sha256 that is not hex, ``E_KEY_SUBMODULES`` for a commit
            that is not 40-hex or submodules on a step that disables them, ``E_KEY_R_SESSION`` for R
            digests missing on an R step or given on another, ``E_KEY_NO_WHEELS`` for an env with
            sdist builds and no recorded ``wheels.lock``.
        ValueError: As `sandbox.keyed_templates`.
    """
    if set(upstream) != set(step.needs) or not all(_is_sha256(k) for k in upstream.values()):
        raise CacheError("E_KEY_UPSTREAM")
    if len(input_digests) != len(step.inputs) or not all(_is_sha256(d) for d in input_digests):
        raise CacheError("E_KEY_INPUTS")
    if not _is_sha256(adapter_sha256):
        raise CacheError("E_KEY_ADAPTER")
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
    env = step.env
    if env.sdist_allow and env.wheels is None:
        raise CacheError("E_KEY_NO_WHEELS")
    return {
        "schema": STEP_KEY_SCHEMA,
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
        "inputs": [i.model_dump(mode="json") for i in step.inputs],
        "input_digests": list(input_digests),
        "stage": [s.model_dump(mode="json") for s in step.stage],
        "outputs": list(step.outputs),
        "adapter_sha256": adapter_sha256,
        "hf_weights": [w.model_dump(mode="json") for w in step.hf_weights],
        "r_session_sha256": r_session_sha256,
        "r_module_env_sha256": r_module_env_sha256,
        "metrics_schema": step.metrics_schema.model_dump(mode="json"),
        "expect": [e.model_dump(mode="json") for e in step.expect],
        "replicate": step.replicate,
        "upstream": dict(sorted(upstream.items())),
    }


def step_key(
    step: StepSpec,
    profile: Profile,
    *,
    upstream: Mapping[str, str],
    submodules: Mapping[str, str],
    adapter_sha256: str,
    input_digests: Sequence[str],
    r_session_sha256: str | None = None,
    r_module_env_sha256: str | None = None,
) -> str:
    """A step's cache key: the sha256 of `canonical_json` over `step_key_payload`.

    Args:
        step: The step.
        profile: The caller's profile.
        upstream: Each need's step key.
        submodules: Submodule path to commit.
        adapter_sha256: The adapter script's sha256.
        input_digests: One input-set digest per ``step.inputs`` entry.
        r_session_sha256: The prefetch ``SESSION.json`` sha256 (R steps only).
        r_module_env_sha256: The captured R module env sha256 (R steps only).

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
        input_digests=input_digests,
        r_session_sha256=r_session_sha256,
        r_module_env_sha256=r_module_env_sha256,
    )
    return hashlib.sha256(canonical_json(payload)).hexdigest()


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
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
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


def reset_result_dir(out: Path) -> Path:
    """Give a step an empty result dir, removing what an incomplete or failed run left; call under `claim`.

    Args:
        out: The step's result dir.

    Returns:
        ``out``, now an empty directory.

    Raises:
        CacheError: ``E_RESULT_DONE`` if ``out`` is complete; it is left untouched.
    """
    if is_done(out):
        raise CacheError("E_RESULT_DONE")
    if out.exists() or out.is_symlink():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


@contextmanager
def claim(results_root: Path, key: str, *, timeout: float = CLAIM_TIMEOUT_S) -> Iterator[Path]:
    """Hold the exclusive claim on one step key while it is checked, computed and marked done.

    The lock files are ``<results_root>/.claims/<key>.{lock,heartbeat,holder}``; the results root's own
    mode is never changed.

    Args:
        results_root: The profile's results root.
        key: The step key.
        timeout: Seconds to wait for a live holder before giving up.

    Yields:
        The step's result dir (`result_dir`); re-check `is_done` inside the claim, since another job may
        have completed the step while this one waited.

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
        yield out
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
    if link.exists(follow_symlinks=False) and not link.is_symlink():
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
