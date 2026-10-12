"""Integration tests for ``senselab.utils.subprocess_venv.ensure_venv`` CUDA routing.

All ``subprocess.run`` invocations are mocked. No real venv is created.
Covers: marker-mismatch rebuild paths, install-argv routing through the
chosen PyTorch index, ``SenselabCudaCompatibilityError`` wrapping on wheel
not-found errors, pass-through of unrelated failures, and the same
behavior across the three real subprocess-venv backends.
"""

import functools
import json
import os
import re
import stat
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import pytest

from senselab.utils import subprocess_venv
from senselab.utils.cuda_probe import HostCuda, SenselabCudaCompatibilityError, TorchIndex
from senselab.utils.subprocess_venv import (
    _classify_uv_failure,
    _normalize_package_name,
    _venv_dist_info,
    _venv_python_version,
    ensure_venv,
    record_venv_use,
    venv_environment,
)
from senselab.utils.venv_lock import (
    IPC_REQUIREMENTS,
    VenvLock,
    VenvLockError,
    load_lock,
    parse_lock,
    requirements_sha256,
)

# ── Fixtures + helpers ─────────────────────────────────────────────

_REAL_RESOLVE_LOCK = subprocess_venv._resolve_lock


@pytest.fixture
def fake_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``_cache_dir()`` to a tmp dir so we never touch ~/.cache."""
    monkeypatch.setattr(subprocess_venv, "_cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make ``_find_uv()`` return a deterministic fake path so it doesn't shell out to ``which uv``."""
    monkeypatch.setattr(subprocess_venv, "_find_uv", lambda: "/fake/uv")
    return "/fake/uv"


@pytest.fixture
def force_cu128(monkeypatch: pytest.MonkeyPatch) -> TorchIndex:
    """Pin the resolved index to cu128 so tests don't depend on the host's actual CUDA."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = TorchIndex(
        url="https://download.pytorch.org/whl/cu128",
        tag="cu128",
        cuda_version=(12, 8),
        source="static-map",
    )
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)
    monkeypatch.setattr(
        subprocess_venv, "pick_torch_index", lambda host_cuda, env_override=None, max_cuda_version=None: idx
    )
    return idx


_FAKE_TORCH = {"torch": "2.8.0", "torchaudio": "2.8.0"}


def _spec_name(spec: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9._-]+)", spec)
    return match.group(1).lower() if match else ""


def _write_fake_lock(
    lock_dir: Path, name: str, requirements: list[str], python_version: str, pins: Optional[list[str]] = None
) -> VenvLock:
    """Write a lock shaped like a compiled one: torch pins in the header, everything else in the body."""
    if pins is None:
        pins = [f"{p}=={v}" for p, v in _FAKE_TORCH.items() if any(_spec_name(r) == p for r in requirements)]
    body = [r for r in requirements if _spec_name(r) not in _FAKE_TORCH] + list(IPC_REQUIREMENTS)
    path = lock_dir / f"{name}.txt"
    staged = lock_dir / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    staged.write_text(
        "\n".join(
            [
                f"# senselab subprocess-venv lock: {name}",
                f"# requirements-sha256: {requirements_sha256(requirements, python_version)}",
                f"# python: {python_version}",
                f"# torch: {' '.join(pins) if pins else 'none'}",
                *body,
            ]
        )
        + "\n"
    )
    os.replace(staged, path)
    return parse_lock(path, name)


@pytest.fixture(autouse=True)
def fake_locks(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give every venv name a lock compiled from exactly the requirements it is called with."""
    lock_dir = tmp_path_factory.mktemp("locks")

    def resolve(
        name: str,
        requirements: list[str],
        python_version: Optional[str],
        max_cuda_version: Optional[tuple[int, int]],
        compile_lock: bool,
    ) -> VenvLock:
        return _write_fake_lock(lock_dir, name, requirements, python_version or "3.12")

    monkeypatch.setattr(subprocess_venv, "_resolve_lock", resolve)
    return lock_dir


def _installed_marker(venv_dir: Path, name: str, requirements: list[str], index: Optional[TorchIndex] = None) -> None:
    """Write the completion marker a finished build of this lock leaves behind."""
    lock = subprocess_venv._resolve_lock(name, requirements, "3.12", None, False)
    data: dict[str, object] = {"lock": lock.path.name, "lock_sha256": lock.sha256, "python_version": "3.12"}
    if index is not None:
        data["torch_index"] = {"tag": index.tag, "url": index.url, "source": index.source}
    venv_dir.mkdir(parents=True, exist_ok=True)
    (venv_dir / ".senselab-installed").write_text(json.dumps(data))


class _SubprocessRecorder:
    """Replacement for ``subprocess.run`` that records calls and replays canned results.

    ``uv venv`` is simulated by creating the target directory so the rest of
    ``ensure_venv`` (which writes the marker file inside it) doesn't trip on
    the missing parent.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Per-call hook — set to raise/return per recorded call index.
        self.hook: Optional[Callable[[list[list[str]]], subprocess.CompletedProcess]] = None

    def __call__(
        self,
        argv: list[str],
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        # Simulate ``uv venv --python X /path/to/venv`` by mkdir-ing the target.
        if len(argv) >= 5 and argv[1] == "venv" and argv[2] == "--python":
            Path(argv[4]).mkdir(parents=True, exist_ok=True)
        # If a hook is set, let it drive the response (raise / return).
        if self.hook is not None:
            return self.hook(self.calls)
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")


# ── Group-readable venv permissions ────────────────────────────────


def test_a_completed_venv_is_group_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second user must be able to run the interpreter the first user built.

    Group-writable lock files let user B take over a stale build; they do not let
    B execute A's venv. Venv trees are created under the default umask, so without
    this a shared cache is buildable but not usable.
    """
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    interpreter = bin_dir / "python3.12"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o700)  # owner rwx only -- typical default-umask result for an executable

    module = site_packages / "pkg.py"
    module.write_text("x = 1\n")
    module.chmod(0o600)  # owner rw only, no execute bit to mirror

    for d in (venv_dir, bin_dir, site_packages):
        d.chmod(0o700)

    subprocess_venv._make_group_readable(venv_dir)

    # An already-executable file gets group-execute mirrored alongside group-read.
    assert stat.S_IMODE(interpreter.stat().st_mode) & 0o070 == 0o050
    # A plain, non-executable file gets group-read only -- it must not become runnable
    # just because it lives in the same tree as bin/python.
    assert stat.S_IMODE(module.stat().st_mode) & 0o070 == 0o040
    # Directories always need group read+execute, or the group can't traverse them
    # regardless of what's inside.
    for d in (venv_dir, bin_dir, site_packages):
        assert stat.S_IMODE(d.stat().st_mode) & 0o070 == 0o050


def test_group_readable_ignores_chmod_failures_on_foreign_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry owned by a different user must not abort the walk for the rest.

    A shared cache directory can hold leftovers from a different user's earlier,
    unrelated build; this process cannot re-permission those, and raising on the
    first one would stop the group-readable pass before it reaches everything else
    in the tree that this process *does* own.
    """
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    owned = bin_dir / "mine.py"
    owned.write_text("x = 1\n")
    foreign = bin_dir / "not-mine.py"
    foreign.write_text("y = 2\n")

    real_chmod = os.chmod

    def _flaky_chmod(path: object, mode: int, *args: object, **kwargs: object) -> None:
        if str(path) == str(foreign):
            raise PermissionError("not the owner")
        real_chmod(path, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "chmod", _flaky_chmod)

    subprocess_venv._make_group_readable(venv_dir)  # must not raise

    assert stat.S_IMODE(owned.stat().st_mode) & 0o040 == 0o040


def test_group_readable_runs_before_the_marker_write(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker must not exist yet while the chmod pass is still running.

    A hard kill (OOM, CI timeout -- both have happened in this repo) between the chmod
    pass and the marker write must leave NO marker, so the next call's marker check fails,
    `shutil.rmtree` fires, and the rebuild reruns chmod to completion. Marker-first would
    instead let that same kill leave `.senselab-installed` present with the chmod pass
    incomplete: every later `ensure_venv` call takes the reuse fast path on seeing the
    marker and returns immediately, and since that path never calls
    `_make_group_readable`, the half-permissioned venv would never be repaired -- a second
    user hits a permission error deep in `site-packages` with no remedy short of deleting
    the venv by hand. A real kill mid-chmod needs a process signal to demonstrate for real;
    this asserts the cheaper, equivalent property -- that the marker is absent for the
    entire duration of the chmod pass -- via a real (non-mocked) `ensure_venv` call against
    torch-free requirements, so no 2.5 GB install is needed to exercise it.
    """
    name = "t-order"
    venv_dir = fake_cache_dir / name
    marker = venv_dir / ".senselab-installed"

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    real_make_group_readable = subprocess_venv._make_group_readable
    marker_seen_during_chmod: list[bool] = []

    def _spy(path: Path) -> None:
        marker_seen_during_chmod.append(marker.exists())
        real_make_group_readable(path)

    monkeypatch.setattr(subprocess_venv, "_make_group_readable", _spy)

    ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")

    assert marker_seen_during_chmod == [False], "marker existed while the chmod pass was still running"
    assert marker.is_file(), "marker must exist once the (successful) chmod pass has completed"


# ── Marker mismatch + rebuild paths ────────────────────────────────


def test_marker_from_a_requirements_list_build_triggers_rebuild(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker written before venvs were installed from locks (no ``lock_sha256``) must rebuild."""
    name = "t-no-index"
    venv_dir = fake_cache_dir / f"{name}-cu128"
    venv_dir.mkdir(parents=True)
    (venv_dir / ".senselab-installed").write_text(
        json.dumps({"requirements": ["torch>=2.8,<2.9"], "python_version": "3.12"})
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    # Three subprocess invocations: uv venv + Stage-1 torch install + Stage-2
    # backend install. If we'd hit the cache fast-path we'd see zero.
    assert len(recorder.calls) == 3
    # New marker now carries the resolved index.
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url
    assert written["torch_index"]["tag"] == "cu128"


def test_marker_with_matching_torch_index_is_cache_hit(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker whose lock digest + ``torch_index.url`` match → no install, no rebuild."""
    name = "t-cache-hit"
    venv_dir = fake_cache_dir / f"{name}-cu128"
    _installed_marker(venv_dir, name, ["torch>=2.8,<2.9"], force_cu128)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    # Zero subprocess invocations on the cache fast-path.
    assert recorder.calls == []


def test_marker_with_different_torch_index_triggers_rebuild(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker whose stored index disagrees with the directory's own key must still rebuild.

    With the directory itself keyed by ``torch_index.tag``, two different indexes can no
    longer collide on one ``venv_dir`` -- a resolved ``cu128`` lands in ``t-different-
    index-cu128``, never in a directory holding a ``cu121`` marker. This exercises the
    defensive fallback for a marker that disagrees with its own directory's key anyway
    (e.g. hand-edited, or a bug in the key computation): it must still rebuild rather than
    trust a stale marker.
    """
    name = "t-different-index"
    venv_dir = fake_cache_dir / f"{name}-cu128"
    cu121 = TorchIndex(
        url="https://download.pytorch.org/whl/cu121", tag="cu121", cuda_version=(12, 1), source="static-map"
    )
    _installed_marker(venv_dir, name, ["torch>=2.8,<2.9"], cu121)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    assert len(recorder.calls) == 3  # uv venv + Stage-1 torch + Stage-2 backend install
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url


def test_a_changed_lock_rebuilds_the_venv(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A venv built from an earlier lock (same name, same index) is rebuilt when the lock changes."""
    name = "t-changed-lock"
    venv_dir = fake_cache_dir / f"{name}-cu128"
    _installed_marker(venv_dir, name, ["torch>=2.8,<2.9"], force_cu128)
    stored = json.loads((venv_dir / ".senselab-installed").read_text())
    stored["lock_sha256"] = "0" * 64
    (venv_dir / ".senselab-installed").write_text(json.dumps(stored))

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert len(recorder.calls) == 3
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["lock_sha256"] != "0" * 64


def test_a_takeover_during_build_refuses_to_certify_the_venv(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If this build's lock was taken over mid-way, the marker must not be written.

    Otherwise a build that lost its lock to a concurrent takeover -- which may already
    be mutating the same ``venv_dir`` -- would still declare itself complete, exactly
    the "importable-looking venv missing a shared object" corruption reported in
    specs/20260817-triage-workflow-dag/benchmarks/orcd-scheduling-2026-09-08.md.
    """
    from senselab.utils.file_lock import SharedFileLock

    name = "t-lost-lock"
    venv_dir = fake_cache_dir / name

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    monkeypatch.setattr(SharedFileLock, "owns", lambda self: False)

    with pytest.raises(RuntimeError, match="lost its lock"):
        ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")

    assert not venv_dir.exists(), "a venv that lost its lock must be removed, not left half-certified"


def test_stampede_on_a_slow_build_lets_every_late_arrival_reuse_the_winners_venv(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several concurrent callers race one never-before-built backend: one builds, the rest reuse.

    Reproduces the shape of the ORCD stampede
    (specs/20260817-triage-workflow-dag/benchmarks/orcd-scheduling-2026-09-08.md, diagnosed in
    specs/20260907-shared-lock-heartbeat-inf/stampede-timeout-and-identity-file.md): several
    processes race ``ensure_venv`` for one backend while the (stubbed) build outlasts a lock
    timeout shorter than it -- the same shape as 600s against measured 590-800s
    ``crisperwhisper`` cold builds, at test scale. Threads, not processes: ``time.sleep``
    releases the GIL and ``fcntl.flock`` treats distinct open file descriptions (one per
    ``filelock`` poll attempt) independently even within one process, so this gets genuine
    OS-level lock contention without subprocess/env-var plumbing.

    Before the fix (holder identity stored inside the same ``.lock`` file ``filelock``
    truncates on every failed poll, verified by running this test against the code at
    ``d349b216``): the eventual winner's own ``owns()`` check reads back no identity -- wiped
    by the other threads' concurrent polling -- so it raises "lost its lock" and deletes the
    venv it just finished, even though it never actually lost the OS-level flock; this
    reproduces as one or more ``RuntimeError`` results below and more than one real "install"
    call. After the fix, ``owns()`` reads the separate ``.holder`` file that only a genuine new
    holder ever writes, so the winner certifies normally and every other thread's own eventual
    acquire hits the marker fast path: zero errors, one real build.
    """
    import threading
    import time as time_module

    name = "t-stampede"
    build_seconds = 1.0
    n_workers = 6
    install_calls: list[int] = []
    install_lock = threading.Lock()
    start_barrier = threading.Barrier(n_workers)

    def fake_run(
        argv: list[str],
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> subprocess.CompletedProcess:
        if len(argv) >= 5 and argv[1] == "venv" and argv[2] == "--python":
            Path(argv[4]).mkdir(parents=True, exist_ok=True)
        elif len(argv) >= 3 and argv[1] == "pip" and argv[2] == "install":
            with install_lock:
                install_calls.append(threading.get_ident())
            time_module.sleep(build_seconds)  # releases the GIL -- see the docstring above
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Shorter than build_seconds -- the ORCD ratio (timeout < build) at test scale.
    monkeypatch.setenv("SENSELAB_VENV_LOCK_TIMEOUT", "0.3")

    results: list[tuple[str, object]] = []
    results_lock = threading.Lock()

    def worker() -> None:
        start_barrier.wait()
        try:
            out = ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")
            with results_lock:
                results.append(("ok", out))
        except Exception as exc:  # noqa: BLE001 -- capture every failure shape for the assertion below
            with results_lock:
                results.append(("error", exc))

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "a worker thread did not finish within 30s"

    errors = [exc for kind, exc in results if kind == "error"]
    assert not errors, f"{len(errors)}/{n_workers} calls raised instead of reusing a completed build: {errors}"
    assert len(results) == n_workers
    outs = {out for kind, out in results if kind == "ok"}
    assert outs == {fake_cache_dir / name}
    assert len(install_calls) == 1, f"expected exactly one real build, got {len(install_calls)}: {install_calls}"


# ── Directory keyed by dependency (device-keyed venv directories) ─────


def test_torch_free_dir_is_stable_across_calls_and_carries_no_tag_suffix(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torch-free backend's directory is the bare ``name`` -- no tag, no host dependence.

    The probe never runs for a torch-free ``requirements`` list, so nothing about the
    host or an env override can affect which directory it resolves to: a CPU node and a
    GPU node build the identical venv and must share it, never rebuild over each other.
    """
    name = "t-torch-free-stable"
    monkeypatch.setattr(
        subprocess_venv,
        "detect_host_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("torch-free venv must not invoke the probe")),
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    out = ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")

    assert out == fake_cache_dir / name
    assert out.name == name, "a torch-free venv directory must carry no tag suffix"

    # A second call (cache hit -- zero subprocess invocations) resolves the same directory.
    recorder2 = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder2)
    out2 = ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")
    assert out2 == out
    assert recorder2.calls == []


def test_torch_bearing_dir_differs_by_resolved_index(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two different resolved indexes for the same backend name land in two different directories.

    Driven entirely through ``SENSELAB_TORCH_INDEX_URL`` so no GPU is needed: unset, a
    torch-bearing backend on a host with no CUDA resolves the ``cpu`` index; with the
    override set, it resolves ``override`` -- two different tags, two different
    directories, never one directory that both builds fight over.
    """
    name = "t-index-dir"
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: HostCuda(version=None, source="none", raw=""))

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)
    cpu_dir = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")
    assert cpu_dir == fake_cache_dir / f"{name}-cpu"

    monkeypatch.setenv("SENSELAB_TORCH_INDEX_URL", "https://pypi.internal.example.com/pytorch/cu128")
    recorder2 = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder2)
    override_dir = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")
    assert override_dir == fake_cache_dir / f"{name}-override"

    assert override_dir != cpu_dir
    assert cpu_dir.exists()
    assert override_dir.exists()


def test_dir_name_carries_every_tag_pick_torch_index_can_produce(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every tag ``pick_torch_index`` can return folds into a valid, single-path-component directory name.

    Enumerated from the source rather than hand-listed: the static CUDA-index map
    (``cu130``/``cu128``/``cu126``/``cu124``/``cu121``) plus the two tags ``pick_torch_index``
    returns outside that map (``cpu`` for no host CUDA, ``override`` for an operator's
    ``SENSELAB_TORCH_INDEX_URL``). Adding a future ``cuXXX`` entry to the map is picked
    up automatically -- this test does not hardcode the tag list.
    """
    from senselab.utils.cuda_probe import _PYTORCH_INDEX_MAP

    tags = {tag for tag, _ in _PYTORCH_INDEX_MAP} | {"cpu", "override"}
    assert tags == {"cu130", "cu128", "cu126", "cu124", "cu121", "cpu", "override"}, (
        "the static CUDA-index map grew or shrank -- update this test's expectations, not just the map"
    )

    for tag in sorted(tags):
        idx = TorchIndex(url=f"https://example.invalid/{tag}", tag=tag, cuda_version=None, source="static-map")
        monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: HostCuda(version=None, source="none", raw=""))
        monkeypatch.setattr(
            subprocess_venv,
            "pick_torch_index",
            lambda host_cuda, env_override=None, max_cuda_version=None, idx=idx: idx,
        )
        recorder = _SubprocessRecorder()
        monkeypatch.setattr(subprocess, "run", recorder)

        out = ensure_venv(f"t-tag-{tag}", ["torch>=2.8,<2.9"], python_version="3.12")

        assert out.name == f"t-tag-{tag}-{tag}"
        # A valid single path component: no separators, and not "." / "..".
        assert os.sep not in out.name
        assert out.name not in (".", "..")
        assert out.parent == fake_cache_dir


# ── Install argv routing ───────────────────────────────────────────


def test_stage_one_installs_the_locks_torch_pins_from_the_chosen_index_alone(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 1 names ONLY the chosen CUDA index and only the lock's exact torch pins.

    No ``--extra-index-url`` is allowed in Stage 1: uv ranks it above ``--index-url``, so a PyPI
    fallback would let PyPI's differently-tagged torch / torchaudio win for these two packages.
    """
    name = "t-stage-one"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"], python_version="3.12")

    # recorder.calls[0] = uv venv; [1] = Stage 1 install; [2] = Stage 2 install.
    stage_one = recorder.calls[1]
    assert stage_one[0:3] == [fake_uv, "pip", "install"]
    idx_pos = stage_one.index("--index-url")
    assert stage_one[idx_pos + 1] == force_cu128.url
    assert "--extra-index-url" not in stage_one
    assert stage_one[-2:] == ["torch==2.8.0", "torchaudio==2.8.0"]


def test_stage_one_takes_the_pins_from_the_lock_not_the_requirements(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    fake_locks: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever ``torch`` the lock fixed is what Stage 1 installs, whatever range the backend gave."""
    name = "t-lock-pins"
    monkeypatch.setattr(
        subprocess_venv,
        "_resolve_lock",
        lambda name, requirements, python_version, max_cuda_version, compile_lock: _write_fake_lock(
            fake_locks, name, requirements, "3.12", pins=["torch==2.7.1"]
        ),
    )
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.4"], python_version="3.12")

    assert recorder.calls[1][-1] == "torch==2.7.1"


def test_stage_two_installs_the_lock_body_with_no_deps_from_default_pypi(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 2 installs exactly the lock body: ``--no-deps``, no index flags, no torch spec."""
    name = "t-stage-two"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["coqui-tts~=0.27", "torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"], python_version="3.12")

    stage_two = recorder.calls[2]
    assert stage_two[0:3] == [fake_uv, "pip", "install"]
    assert "--index-url" not in stage_two
    assert "--extra-index-url" not in stage_two
    assert "--no-deps" in stage_two
    lock_file = Path(stage_two[stage_two.index("--requirement") + 1])
    assert lock_file.name == f"{name}.txt"
    assert not any(arg.startswith("torch") for arg in stage_two)
    body = lock_file.read_text()
    assert "coqui-tts~=0.27" in body and "safetensors" in body and "numpy" in body


def test_a_missing_lock_is_refused_before_anything_is_built(
    fake_cache_dir: Path,
    fake_uv: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no committed lock, ``ensure_venv`` raises and runs nothing."""
    monkeypatch.setattr(subprocess_venv, "_resolve_lock", _REAL_RESOLVE_LOCK)
    monkeypatch.setattr(subprocess_venv, "load_lock", functools.partial(load_lock, lock_dir=tmp_path))
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(VenvLockError, match="No lock"):
        ensure_venv("t-unlocked", ["some-pure-python-pkg==1.0"], python_version="3.12")
    assert recorder.calls == []


def test_a_lock_from_other_requirements_is_refused(
    fake_cache_dir: Path,
    fake_uv: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock compiled from a different requirement list is stale; the venv is not built from it."""
    _write_fake_lock(tmp_path, "t-stale", ["some-pure-python-pkg==1.0"], "3.12")
    monkeypatch.setattr(subprocess_venv, "_resolve_lock", _REAL_RESOLVE_LOCK)
    monkeypatch.setattr(subprocess_venv, "load_lock", functools.partial(load_lock, lock_dir=tmp_path))
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(VenvLockError, match="different requirements"):
        ensure_venv("t-stale", ["some-pure-python-pkg==2.0"], python_version="3.12")
    with pytest.raises(VenvLockError, match="Python 3.12"):
        ensure_venv("t-stale", ["some-pure-python-pkg==1.0"], python_version="3.11")
    assert recorder.calls == []


def test_compile_lock_builds_a_probe_venv_from_a_lock_compiled_now(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``compile_lock=True`` compiles into the cache and installs from that, for compatibility probes."""
    monkeypatch.setattr(subprocess_venv, "_resolve_lock", _REAL_RESOLVE_LOCK)
    seen: dict[str, object] = {}

    def fake_compile(name: str, requirements: list[str], python: str, cap: object, **kwargs: object) -> str:
        seen.update(name=name, kwargs=kwargs)
        return f"# requirements-sha256: x\n# python: {python}\n# torch: none\nsome-pure-python-pkg==1.0\n"

    monkeypatch.setattr(subprocess_venv.venv_lock, "compile_lock", fake_compile)
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv("t-probe", ["some-pure-python-pkg>=1"], python_version="3.12", compile_lock=True)

    assert seen == {"name": "t-probe", "kwargs": {"check_torch_index": False}}
    install = recorder.calls[1]
    assert Path(install[install.index("--requirement") + 1]) == fake_cache_dir / ".locks" / "t-probe.txt"


# ── Auto-detection: torch-free venvs skip the probe and Stage 1 ────


def test_no_torch_in_requirements_skips_probe_and_stage_one(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends without ``torch`` / ``torchaudio`` in ``requirements`` skip the probe entirely.

    ``ensure_venv`` decides the install shape from the caller's
    ``requirements`` itself — no separate flag. A venv that declares
    neither package (yamnet, continuous-ser, or a future pure-Python
    backend) gets the leanest possible install: one ``uv pip install``
    pass against default PyPI, no ``nvidia-smi`` shellout, no
    ``torchaudio`` force-appended.
    """
    name = "t-no-torch"
    # If the probe were invoked it would raise — that's how we confirm
    # auto-detection short-circuits before reaching it.
    monkeypatch.setattr(
        subprocess_venv,
        "detect_host_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("probe should not run when no torch in requirements")),
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")

    # Exactly two subprocess calls: uv venv + a single uv pip install.
    assert len(recorder.calls) == 2
    install_argv = recorder.calls[1]
    assert install_argv[0:3] == [fake_uv, "pip", "install"]
    assert "--index-url" not in install_argv
    assert "--extra-index-url" not in install_argv
    assert "--no-deps" in install_argv
    assert not any(arg.startswith("torch") for arg in install_argv)
    body = Path(install_argv[install_argv.index("--requirement") + 1]).read_text()
    assert "some-pure-python-pkg==1.0" in body and "safetensors" in body and "numpy" in body

    # Marker carries no ``torch_index`` field, so a later call whose
    # requirements grow a torch spec will correctly invalidate + rebuild.
    written = json.loads((fake_cache_dir / name / ".senselab-installed").read_text())
    assert "torch_index" not in written


def test_adding_torch_to_requirements_invalidates_torch_free_cache(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A venv built without torch is left alone when requirements later grow a torch spec.

    Directories are keyed by dependency, so a torch-free venv (bare ``name``) and a
    torch-bearing one (``f"{name}-{tag}"``) are simply two different directories -- the
    torch-requiring call never touches, invalidates, or rebuilds the old torch-free
    directory. It builds fresh under the tag-suffixed name instead, orphaning the old
    one (see the migration note in specs/20260907-venv-dir-keyed-by-index/).
    """
    name = "t-switch-to-torch"
    old_venv_dir = fake_cache_dir / name
    old_venv_dir.mkdir(parents=True)
    (old_venv_dir / ".senselab-installed").write_text(
        json.dumps({"requirements": ["some-pure-python-pkg==1.0"], "python_version": "3.12"})
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    # Same name, but requirements now include a torch pin → resolves a new, tag-suffixed directory.
    out = ensure_venv(name, ["some-pure-python-pkg==1.0", "torch>=2.8,<2.9"], python_version="3.12")

    new_venv_dir = fake_cache_dir / f"{name}-cu128"
    assert out == new_venv_dir
    assert out != old_venv_dir
    # uv venv + Stage 1 + Stage 2 — a full install into the new directory.
    assert len(recorder.calls) == 3
    written = json.loads((new_venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url
    # The old torch-free directory is untouched -- orphaned, not invalidated in place.
    old_written = json.loads((old_venv_dir / ".senselab-installed").read_text())
    assert "torch_index" not in old_written


def test_yamnet_style_requirements_omit_torchaudio_from_install(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yamnet / continuous-ser-style venvs no longer get ``torchaudio`` force-installed.

    Before this change, every subprocess venv paid for ``torchaudio``
    via an unconditional IPC append, even backends that read audio via
    ``soundfile`` and never imported torchaudio (~200 MB of wheels per
    venv for no functional benefit). Now that the append is gone,
    such a venv ends up with neither torch nor torchaudio in its
    install argv.
    """
    name = "t-yamnet-style"
    monkeypatch.setattr(
        subprocess_venv,
        "detect_host_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("torch-free venv must not invoke the probe")),
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    # Mirror yamnet's actual _REQUIREMENTS shape — TF-based, soundfile for audio I/O.
    ensure_venv(
        name,
        ["tensorflow", "tensorflow-hub", "setuptools<70", "numpy", "soundfile"],
        python_version="3.12",
    )

    install_argv = recorder.calls[1]
    assert len(recorder.calls) == 2
    assert not any(arg.startswith("torch") for arg in install_argv)
    body = Path(install_argv[install_argv.index("--requirement") + 1]).read_text()
    assert "tensorflow" in body and "tensorflow-hub" in body


# ── env override ───────────────────────────────────────────────────


def test_env_override_routes_through_override_url_and_still_probes_for_diagnostic(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SENSELAB_TORCH_INDEX_URL routes the install verbatim; probe still runs for diagnostic surface."""
    name = "t-override"
    override_url = "https://pypi.internal.example.com/pytorch/cu128"
    monkeypatch.setenv("SENSELAB_TORCH_INDEX_URL", override_url)

    # Probe still runs so the host-CUDA value is available for any
    # ``SenselabCudaCompatibilityError`` message produced on the override path.
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    install_argv = recorder.calls[1]
    idx_pos = install_argv.index("--index-url")
    assert install_argv[idx_pos + 1] == override_url
    # The venv directory itself carries the "override" tag.
    venv_dir = fake_cache_dir / f"{name}-override"
    assert venv_dir.is_dir()
    # Marker records the override source.
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["source"] == "env-override"
    assert written["torch_index"]["tag"] == "override"


def test_env_override_empty_string_does_not_short_circuit(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty SENSELAB_TORCH_INDEX_URL must be ignored (treated as unset)."""
    name = "t-override-empty"
    monkeypatch.setenv("SENSELAB_TORCH_INDEX_URL", "")

    host = HostCuda(version=(12, 4), source="nvidia-smi", raw="CUDA Version: 12.4")
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    written = json.loads((fake_cache_dir / f"{name}-cu124" / ".senselab-installed").read_text())
    assert written["torch_index"]["source"] == "static-map"
    assert written["torch_index"]["tag"] == "cu124"


# ── Failure handling ───────────────────────────────────────────────


def test_no_matching_distribution_failure_wraps_into_compatibility_error(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uv 'no matching distribution' → SenselabCudaCompatibilityError, venv removed."""
    name = "t-wheel-missing"

    def _hook(calls: list[list[str]]) -> subprocess.CompletedProcess:
        if calls[-1][1] == "venv":
            return subprocess.CompletedProcess(args=calls[-1], returncode=0, stdout="", stderr="")
        # pip install: fail with a "no matching distribution" stderr.
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=calls[-1],
            output="",
            stderr=(
                "error: No solution found when resolving dependencies:\n"
                "  No matching distribution for `torch>=2.8,<2.9`\n"
            ),
        )

    recorder = _SubprocessRecorder()
    recorder.hook = _hook
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(SenselabCudaCompatibilityError) as excinfo:
        ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    err = excinfo.value
    assert err.attempted_index.url == force_cu128.url
    assert any("torch>=2.8,<2.9" in p for p in err.failing_packages)
    # Half-built venv must be wiped -- at its resolved, tag-suffixed directory.
    assert not (fake_cache_dir / f"{name}-cu128").exists()


def test_unrelated_install_failure_passes_through_as_called_process_error(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network / permission / unrelated errors → original CalledProcessError, not wrapped."""
    name = "t-network-error"

    def _hook(calls: list[list[str]]) -> subprocess.CompletedProcess:
        if calls[-1][1] == "venv":
            return subprocess.CompletedProcess(args=calls[-1], returncode=0, stdout="", stderr="")
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=calls[-1],
            output="",
            stderr="error: connection timed out while reading https://pypi.org/simple/torch/\n",
        )

    recorder = _SubprocessRecorder()
    recorder.hook = _hook
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(subprocess.CalledProcessError):
        ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")
    # The half-built venv is also wiped for unrelated failures so next run starts clean.
    assert not (fake_cache_dir / f"{name}-cu128").exists()


# ── _classify_uv_failure unit tests ────────────────────────────────


def test_classify_uv_failure_no_matching_distribution() -> None:
    """Uv 'no matching distribution' stderr → returns the failing package list."""
    stderr = (
        "error: No solution found when resolving dependencies:\n  No matching distribution for `torch==2.8.1+cu128`\n"
    )
    failing = _classify_uv_failure(stderr)
    assert failing == ["torch==2.8.1+cu128"]


def test_classify_uv_failure_could_not_find_version() -> None:
    """Uv 'could not find a version that satisfies' stderr → returns the failing package list."""
    stderr = "error: Could not find a version that satisfies the requirement `torchaudio>=2.8,<2.9`\n"
    failing = _classify_uv_failure(stderr)
    assert failing == ["torchaudio>=2.8,<2.9"]


def test_classify_uv_failure_unrelated_returns_none() -> None:
    """Unrelated errors (network, syntax, permission) must return None."""
    assert _classify_uv_failure("error: connection timed out") is None
    assert _classify_uv_failure("error: permission denied while writing to /tmp/foo") is None
    assert _classify_uv_failure("") is None


def test_classify_uv_failure_dedupes_repeated_specs() -> None:
    """Multiple mentions of the same failing spec collapse to one entry."""
    stderr = (
        "error: No matching distribution for `torch>=2.8,<2.9`\nwarning: try setting an index for `torch>=2.8,<2.9`\n"
    )
    failing = _classify_uv_failure(stderr)
    assert failing == ["torch>=2.8,<2.9"]


# ── Cross-backend regression test (US3) ────────────────────────────


def test_all_three_subprocess_backends_route_through_same_torch_index(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three ASR backends, from their committed locks, each route Stage 1 through cu128.

    Stage 1 carries only exact ``torch``/``torchaudio`` pins and the CUDA index; Stage 2 is the lock
    body with no index flag and no torch spec.
    """
    from senselab.audio.tasks.speech_to_text import canary_qwen, nemo, qwen

    monkeypatch.setattr(subprocess_venv, "_resolve_lock", _REAL_RESOLVE_LOCK)
    for module, venv, reqs, python in (
        (canary_qwen, "_CANARY_VENV", "_CANARY_REQUIREMENTS", "_CANARY_PYTHON"),
        (nemo, "_NEMO_VENV", "_NEMO_REQUIREMENTS", "_NEMO_PYTHON"),
        (qwen, "_QWEN_VENV", "_QWEN_REQUIREMENTS", "_QWEN_PYTHON"),
    ):
        label = getattr(module, venv)
        recorder = _SubprocessRecorder()
        monkeypatch.setattr(subprocess, "run", recorder)
        ensure_venv(label, list(getattr(module, reqs)), python_version=getattr(module, python))
        stage_one = recorder.calls[1]
        assert stage_one[stage_one.index("--index-url") + 1] == force_cu128.url, label
        assert "--extra-index-url" not in stage_one, label
        assert any(arg.startswith("torch==") for arg in stage_one), label
        assert any(arg.startswith("torchaudio==") for arg in stage_one), label
        stage_two = recorder.calls[2]
        assert "--index-url" not in stage_two and "--extra-index-url" not in stage_two, label
        assert not any(arg.startswith("torch") for arg in stage_two), label


# ── TLS trust for the isolated venvs ────────────────────────────────────────


def test_subprocess_env_points_tls_at_a_bundle_that_exists() -> None:
    """The uv-managed interpreter these venvs run on has no usable system CA path.

    python-build-standalone is statically linked, so `ssl.create_default_context()` finds no trust
    store and every `urlopen` inside a worker fails with `CERTIFICATE_VERIFY_FAILED` — on a host whose
    network is fine and where `curl` to the same URL succeeds, because curl uses the system bundle and
    Python does not. Measured on MIT ORCD: the coqui venv's Python failed as-is and succeeded with
    `SSL_CERT_FILE` pointed at the certifi bundle already installed in that same venv.

    The failure surfaced two layers away, as a `RuntimeError` re-raised by
    `parse_subprocess_result` from the worker's structured error, which is why it read as a Coqui
    problem rather than a trust-store one.
    """
    import os

    from senselab.utils.subprocess_venv import _clean_subprocess_env

    env = _clean_subprocess_env()
    bundle = env.get("SSL_CERT_FILE")
    assert bundle, "a worker with no CA bundle cannot verify TLS on a standalone interpreter"
    assert os.path.exists(bundle), f"SSL_CERT_FILE points at a file that does not exist: {bundle}"
    assert env.get("REQUESTS_CA_BUNDLE") == bundle, "requests-based workers need the same bundle"


def test_an_operators_own_ca_bundle_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host behind a corporate CA has already answered this question.

    Overriding it would break precisely the setup that took the trouble to configure it, so the
    defaults are applied only when the variables are unset.
    """
    from senselab.utils.subprocess_venv import _clean_subprocess_env

    monkeypatch.setenv("SSL_CERT_FILE", "/etc/pki/corp/ca-bundle.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/pki/corp/ca-bundle.pem")
    env = _clean_subprocess_env()
    assert env["SSL_CERT_FILE"] == "/etc/pki/corp/ca-bundle.pem"
    assert env["REQUESTS_CA_BUNDLE"] == "/etc/pki/corp/ca-bundle.pem"


def test_cache_dir_path_does_not_create_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A test's skip gate must be able to ask *where* a venv would live without creating anything.

    At import time, on a read-only or sandboxed HOME, the mkdir in _cache_dir() would fail
    and take collection down with it.
    """
    target = tmp_path / "does-not-exist-yet"
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(target))

    assert subprocess_venv._cache_dir_path() == Path(str(target))
    assert not target.exists()


def test_cache_dir_creates_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_cache_dir() keeps its creating behaviour — callers that are about to build a venv rely on it."""
    target = tmp_path / "created-on-demand"
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(target))

    assert subprocess_venv._cache_dir() == Path(str(target))
    assert target.is_dir()


def test_cache_dir_path_honours_the_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate must match where the venv is actually built.

    Otherwise it skips on a host that has the venv and runs on a host that does not.
    """
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(tmp_path / "elsewhere"))
    assert str(tmp_path / "elsewhere") == str(subprocess_venv._cache_dir_path())


def test_every_worker_spawn_goes_through_the_shared_env_helper() -> None:
    """A helper only helps the call sites that call it.

    `_clean_subprocess_env` existed, and six of the twenty-one venv-python spawns did not use it:
    `voice_cloning/coqui.clone_voices` and `features_extraction/ppg` each hand-rolled their own copy
    of the MPLBACKEND filter, while `voice_cloning/sparc`, `features_extraction/sparc`,
    `compatibility_test_runner`, and `subprocess_venv`'s own shim passed no `env` at all and
    inherited `os.environ` wholesale. So the CA-bundle fix landed in one place and four backends kept
    failing — `voice_cloning_test` still raised CERTIFICATE_VERIFY_FAILED on ORCD after the helper
    was already correct, and `subprocess_venv_test` passed the whole time because it tested the
    helper rather than its callers.

    A duplicated env dict is the kind of thing that reads as harmless at every individual call site,
    so this asserts the invariant structurally instead: launching a venv interpreter means passing an
    `env`, and building one from `os.environ` happens in exactly one module.
    """
    import ast
    import pathlib

    import senselab

    # From the package itself, so moving this test file cannot silently point the scan at nothing.
    pkg = pathlib.Path(senselab.__file__).resolve().parent
    assert pkg.is_dir(), pkg

    missing_env: list[str] = []
    hand_rolled: list[str] = []
    for source in sorted(pkg.rglob("*.py")):
        text = source.read_text()
        rel = source.relative_to(pkg.parent)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            # Every env built from os.environ belongs to the one helper.
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("items", "copy")
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and source.name != "subprocess_venv.py"
            ):
                hand_rolled.append(f"{rel}:{node.lineno}")
            # A spawn of the venv interpreter must be handed an env.
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.args
                and isinstance(node.args[0], ast.List)
                and node.args[0].elts
                and isinstance(node.args[0].elts[0], ast.Name)
                and node.args[0].elts[0].id == "python"
            ):
                continue
            if not any(kw.arg == "env" for kw in node.keywords):
                missing_env.append(f"{rel}:{node.lineno}")

    assert not missing_env, "venv-python spawns with no env (they inherit os.environ): " + ", ".join(missing_env)
    assert not hand_rolled, "env built from os.environ outside subprocess_venv.py: " + ", ".join(hand_rolled)


def test_a_file_ref_lock_leaves_the_callers_directory_mode_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locking an input file must not widen the directory the caller keeps it in.

    `file_lock_test` covers the primitive: `SharedFileLock(..., manage_dir_mode=False)` leaves a
    directory's mode alone. It cannot cover the thing that actually broke, which is that
    `call_in_venv` is the only construction site of a locked `FileRef` in the repo and the
    primitive's default is still `True` — drop the keyword there and every test in that file
    still passes while a caller's `0700` directory goes back to being chmodded to `0o2775`
    (setgid, group-write, *and* other-read/other-execute) on every locked call.

    So this drives the real call path with the subprocess faked out, and asserts on the
    directory rather than on the argument: an assertion about a call signature would survive
    the argument moving into a helper or a default, and would say nothing about permissions.
    """
    data_dir = tmp_path / "private"
    data_dir.mkdir()
    os.chmod(data_dir, 0o700)  # pin an exact private mode regardless of the runner's umask
    audio = data_dir / "input.wav"
    audio.write_bytes(b"x")

    monkeypatch.setattr(subprocess_venv, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(subprocess_venv, "venv_python", lambda *a, **k: str(tmp_path / "venv" / "bin" / "python"))

    mode_while_held: list[int] = []
    lock_seen: list[bool] = []

    def fake_run(*args: object, **kwargs: object) -> "subprocess.CompletedProcess[str]":
        # Sampled while the lock is held. `__exit__` never restores a mode it changed, so a
        # post-hoc check would catch a regression too -- but the window the subprocess runs in
        # is the window a directory would be exposed for, so that is where this looks.
        mode_while_held.append(stat.S_IMODE(data_dir.stat().st_mode))
        lock_seen.append(Path(str(audio) + ".lock").exists())
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    subprocess_venv.call_in_venv(
        name="fake",
        requirements=[],
        module="fake_module",
        function="fake_function",
        args={"audio": subprocess_venv.FileRef(path=audio, lock=True)},
    )

    # The lock really was taken -- otherwise the mode assertion below would pass vacuously,
    # for the uninteresting reason that nothing ever tried to chmod anything.
    assert lock_seen == [True], "expected a .lock beside the input file; the FileRef lock path did not run"
    assert mode_while_held == [0o700], f"caller dir was widened while held: {[oct(m) for m in mode_while_held]}"
    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700


# ── Environment capture ───────────────────────────────────────────


def _make_fake_venv(root: Path, python_dir: str = "python3.12") -> Path:
    """Build a minimal on-disk venv tree: pyvenv.cfg plus a handful of dist-info directories."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = /fake\nimplementation = CPython\nversion_info = 3.12.11\n")
    site_packages = root / "lib" / python_dir / "site-packages"
    site_packages.mkdir(parents=True)
    for name, version in [
        ("torch", "2.14.0"),
        ("transformers", "5.16.1"),
        ("crisperwhisper", "2.0.1"),
        ("ctranslate2", "4.6.0"),
        ("tensorflow_hub", "0.16.1"),
    ]:
        (site_packages / f"{name}-{version}.dist-info").mkdir()
    return root


class TestRecordVenvUse:
    """``record_venv_use`` captures the venv directory each ``ensure_venv`` call resolves to."""

    def test_a_cache_hit_is_still_recorded(
        self, fake_cache_dir: Path, fake_uv: str, force_cu128: TorchIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fast cache-hit path (zero subprocess calls) still notes what it returned."""
        name = "t-record"
        venv_dir = fake_cache_dir / f"{name}-cu128"
        _installed_marker(venv_dir, name, ["torch>=2.8,<2.9"], force_cu128)
        monkeypatch.setattr(subprocess, "run", _SubprocessRecorder())

        with record_venv_use() as used:
            out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

        assert used == {name: venv_dir}
        assert out == venv_dir

    def test_outside_the_context_manager_nothing_is_recorded(
        self, fake_cache_dir: Path, fake_uv: str, force_cu128: TorchIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller that never opens ``record_venv_use`` pays nothing and gets no dict back."""
        name = "t-unrecorded"
        venv_dir = fake_cache_dir / f"{name}-cu128"
        _installed_marker(venv_dir, name, ["torch>=2.8,<2.9"], force_cu128)
        monkeypatch.setattr(subprocess, "run", _SubprocessRecorder())

        # No error, and the module-level recorder stays unset.
        ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")
        assert subprocess_venv._VENV_USE_RECORDER.get() is None

    def test_two_venvs_used_in_one_context_are_both_recorded(
        self, fake_cache_dir: Path, fake_uv: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that reaches several backends collects one entry per backend name."""
        for name in ("yamnet", "hear"):
            _installed_marker(fake_cache_dir / name, name, [])
        monkeypatch.setattr(subprocess, "run", _SubprocessRecorder())

        with record_venv_use() as used:
            ensure_venv("yamnet", [], python_version="3.12")
            ensure_venv("hear", [], python_version="3.12")

        assert set(used) == {"yamnet", "hear"}
        assert used["yamnet"] == fake_cache_dir / "yamnet"
        assert used["hear"] == fake_cache_dir / "hear"


class TestVenvEnvironment:
    """``venv_environment`` reads a venv's python version and installed distributions from disk."""

    def test_python_version_comes_from_pyvenv_cfg(self, tmp_path: Path) -> None:
        """The ``version_info`` line, not a directory-name guess, names the interpreter."""
        venv_dir = _make_fake_venv(tmp_path / "v")
        assert _venv_python_version(venv_dir) == "3.12.11"

    def test_missing_pyvenv_cfg_reports_unknown_rather_than_raising(self, tmp_path: Path) -> None:
        """A venv tree with no ``pyvenv.cfg`` degrades to ``"unknown"`` rather than an OSError."""
        venv_dir = tmp_path / "no-cfg"
        venv_dir.mkdir()
        assert _venv_python_version(venv_dir) == "unknown"

    def test_dist_info_names_are_parsed_into_name_and_version(self, tmp_path: Path) -> None:
        """Every ``*.dist-info`` directory becomes one name/version pair, nothing dropped."""
        venv_dir = _make_fake_venv(tmp_path / "v")
        found = _venv_dist_info(venv_dir)
        assert found["torch"] == "2.14.0"
        assert found["transformers"] == "5.16.1"
        assert found["tensorflow_hub"] == "0.16.1"
        assert len(found) == 5

    def test_underscore_and_hyphen_spellings_normalize_the_same(self) -> None:
        """``tensorflow_hub`` on disk must match a declared ``tensorflow-hub``."""
        assert _normalize_package_name("tensorflow_hub") == _normalize_package_name("tensorflow-hub")

    def test_declared_subset_excludes_unlisted_packages(self, tmp_path: Path) -> None:
        """ctranslate2 is installed but not a declared package, so it is not stored in full."""
        venv_dir = _make_fake_venv(tmp_path / "v")
        record = venv_environment("crisperwhisper", venv_dir)
        assert "ctranslate2" not in record["dependencies"]
        assert record["dependencies"]["torch"] == "2.14.0"
        assert record["dependencies"]["crisperwhisper"] == "2.0.1"

    def test_label_is_the_resolved_directory_name(self, tmp_path: Path) -> None:
        """The device key lives in the directory name, e.g. a ``-cpu``/``-cu128`` suffix."""
        venv_dir = _make_fake_venv(tmp_path / "crisperwhisper-cpu")
        record = venv_environment("crisperwhisper", venv_dir)
        assert record["label"] == "crisperwhisper-cpu"

    def test_the_digest_changes_when_the_full_listing_does(self, tmp_path: Path) -> None:
        """A package the declared subset excludes still moves the digest, so a mismatch is visible."""
        base = _make_fake_venv(tmp_path / "a")
        same = _make_fake_venv(tmp_path / "b")
        assert venv_environment("x", base)["dependencies_digest"] == venv_environment("x", same)["dependencies_digest"]
        (base / "lib" / "python3.12" / "site-packages" / "extra-1.0.dist-info").mkdir()
        assert venv_environment("x", base)["dependencies_digest"] != venv_environment("x", same)["dependencies_digest"]
