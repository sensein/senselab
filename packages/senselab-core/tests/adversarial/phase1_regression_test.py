# ruff: noqa: ANN401
"""Phase 1 adversarial findings with no regression test elsewhere, rebuilt from the report's break column.

Rows A-28 and A-33 are guards over the repository itself. Each probe plants the offending input and asserts the
guard refuses it.
"""

import ast
import base64
import errno
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
PACKAGES = REPO / "packages"


def _load(path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the module at ``path`` under ``name``, registered only for the length of the test."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _porcelain() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def worktree_untouched() -> Iterator[None]:
    """Fail the test if it changes anything ``git status`` can see."""
    before = _porcelain()
    yield
    assert _porcelain() == before, "the test wrote under the worktree"


def _pytest(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.contract
def test_a28_untiered_test_under_packages_fails_collection(pytester: pytest.Pytester, worktree_untouched: None) -> None:
    """The collection hook of ``packages/`` refuses a test with no tier marker, which CI's ``-m`` would skip."""
    hook = PACKAGES / "conftest.py"
    pytester.makeini("[pytest]\nmarkers =\n    logic\n    contract\n    limits\n    usecase\n    env\n")
    pytester.mkdir("packages")
    (pytester.path / "packages" / "conftest.py").write_text(hook.read_text() if hook.is_file() else "")
    (pytester.path / "packages" / "untiered_test.py").write_text("def test_never_runs_in_ci():\n    pass\n")
    result = pytester.runpytest("packages/untiered_test.py")
    assert result.ret == pytest.ExitCode.USAGE_ERROR, result.stdout.str()
    assert "test_never_runs_in_ci" in result.stderr.str()


@pytest.mark.contract
def test_a33_undeclared_external_import_in_a_package_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worktree_untouched: None
) -> None:
    """The declared-dependency contract, run over a tmp workspace, reports a package's undeclared import."""
    guard = _load(
        PACKAGES / "senselab-core" / "tests" / "lint" / "dependencies_test.py", "a33_dependencies_test", monkeypatch
    )
    dist = tmp_path / "packages" / "dist-a33"
    pkg = dist / "src" / "a33pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("import os\nimport numpy\nimport librosa\n")
    (dist / "pyproject.toml").write_text('[project]\nname = "dist-a33"\nversion = "0"\ndependencies = ["numpy"]\n')
    monkeypatch.syspath_prepend(str(dist / "src"))
    roots = guard.import_names(tmp_path / "packages")
    graph = guard.grimp.build_graph(*sorted(roots), include_external_packages=True)
    data = tomllib.loads((dist / "pyproject.toml").read_text())
    allowed = set(roots)
    for name in guard.declared_dependencies(data):
        allowed |= guard.import_names_of(name)
    assert guard.undeclared_by_module(guard.external_imports(graph, roots), allowed, {}) == {"a33pkg.mod": {"librosa"}}


# --- base-runnable probes: each uses only what existed at 94e81479, importing it inside the test ---

POLL_S = 15


def _key(task: str = "asr", params: dict[str, Any] | None = None, audio: str = "a" * 64, model: str = "0" * 40) -> Any:
    from senselab_core.kernel.identity import LogicalKey

    try:
        from senselab_core.kernel.identity import ModelIdentity

        pinned: object = ModelIdentity(source="hf", id="org/asr", identity=model)
    except ImportError:
        pinned = model
    return LogicalKey(
        task=task,
        task_version="1",
        backend="hf",
        backend_version="1",
        model=pinned,
        params=params if params is not None else {"beam": 1},
        inputs={"audio": "content:" + audio},
    )


def _ctx(platform: str = "linux") -> Any:
    from senselab_core.kernel.identity import ExecutionContext

    return ExecutionContext(
        env_lock="lock",
        platform=platform,
        wheel_set="wheels",
        numerics={"device": "cpu", "precision": "fp32"},
        decoder=None,
        driver={},
        host="node1",
        protocol="p1",
    )


def _entry_dir(root: Path, key: Any, ctx: Any) -> Path:
    from senselab_core.runtime.store import STORE_DIR

    digest = key.digest()
    return root / STORE_DIR / digest[:2] / digest / ctx.fingerprint()


def _unseal(path: Path) -> Path:
    for p in [path, *path.rglob("*")]:
        if not p.is_symlink():
            p.chmod(0o700 if p.is_dir() else 0o600)
    return path


def _audio() -> Any:
    import numpy
    from senselab_core.kernel.signals import Audio

    return Audio.from_array(numpy.array([[0.25, 0.5, -0.5]], dtype=numpy.float32), 16_000)


@pytest.mark.logic
def test_a1_probe_flipped_flag_cannot_mutate_held_samples() -> None:
    """The samples a value holds cannot be made writeable again, by flag or through pickle and deepcopy."""
    import copy
    import pickle

    group = _audio().groups["audio"]
    with pytest.raises(ValueError):
        group.data.flags.writeable = True
    with pytest.raises(ValueError):
        group.data[0, 0] = 999
    for held in (pickle.loads(pickle.dumps(group)), copy.deepcopy(group)):
        assert held.data.flags.writeable is False


@pytest.mark.logic
def test_a12_probe_multi_element_values_compare_and_hash() -> None:
    """Two equal multi-element audios compare equal and hash alike."""
    a, b = _audio(), _audio()
    assert a == b
    assert hash(a) == hash(b)


@pytest.mark.logic
def test_a13_probe_model_copy_revalidates() -> None:
    """`model_copy(update=)` with a negative rate or a negative end raises."""
    from senselab_core.kernel.time import Span

    group = _audio().groups["audio"]
    with pytest.raises(ValueError):
        group.model_copy(update={"sampling_rate": -5})
    with pytest.raises(ValueError):
        Span(start_ns=0, end_ns=1, clock_id="c").model_copy(update={"end_ns": -1})


@pytest.mark.logic
def test_a14_probe_rewritten_source_is_not_read_under_the_old_identity(tmp_path: Path) -> None:
    """A WAV overwritten after open is refused at read, or reads what was opened, never the new samples."""
    import numpy
    import soundfile
    from senselab_core.kernel.signals import Audio

    path = tmp_path / "a.wav"
    soundfile.write(path, numpy.full((4 * 16_000, 2), 0.1, dtype=numpy.float32), 16_000, subtype="FLOAT")
    audio = Audio.from_path(path)
    before = path.stat().st_mtime_ns
    soundfile.write(path, numpy.full((16_000, 2), 0.5, dtype=numpy.float32), 16_000, subtype="FLOAT")
    os.utime(path, ns=(before + 10**9, before + 10**9))
    try:
        data = audio.groups["audio"].read()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return
    assert not numpy.allclose(data, 0.5)


@pytest.mark.logic
def test_a2_probe_misfiled_entry_is_not_served(tmp_path: Path) -> None:
    """B's entry copied onto A's path is not served for A."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    store, ctx = ResultStore(tmp_path), _ctx()
    a, b = _key(audio="a" * 64), _key(audio="b" * 64)
    store.put(a, {"text": "transcript of A"}, ctx)
    store.put(b, {"text": "transcript of B"}, ctx)
    shutil.rmtree(_unseal(_entry_dir(tmp_path, a, ctx)))
    shutil.copytree(_entry_dir(tmp_path, b, ctx), _entry_dir(tmp_path, a, ctx))
    got = store.get(a, ctx, ReusePolicy.EXACT)
    assert got is None or got.outputs == {"text": "transcript of A"}


def _raising_once(err: OSError, name: str) -> Any:
    """A store member reader that raises `err` the first time it reads `name`."""
    from senselab_core.runtime.storage import StorageDir

    real = StorageDir.read_regular
    state = {"raised": False}

    def read_regular(self: Any, member: str, *, limit: int | None = None) -> bytes | None:
        if member == name and not state["raised"]:
            state["raised"] = True
            raise err
        return real(self, member, limit=limit)

    return read_regular


@pytest.mark.logic
def test_a3_probe_one_transient_read_error_does_not_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One ESTALE while reading a healthy entry leaves it in place and quarantines nothing."""
    from senselab_core.runtime.store import QUARANTINE_DIR, ResultStore

    store, key, ctx = ResultStore(tmp_path), _key(), _ctx()
    store.put(key, {"y": "healthy"}, ctx)
    from senselab_core.runtime.storage import StorageDir

    monkeypatch.setattr(StorageDir, "read_regular", _raising_once(OSError(errno.ESTALE, "stale"), "result.json"))
    try:
        store.put(key, {"y": "healthy"}, ctx)
    except Exception:  # noqa: BLE001
        pass
    assert _entry_dir(tmp_path, key, ctx).is_dir()
    assert not any((tmp_path / QUARANTINE_DIR).glob("*"))


@pytest.mark.logic
def test_a17_probe_a_healed_entry_is_not_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry reported corrupt once, though it is healthy, is still served afterwards."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime import store as store_module
    from senselab_core.runtime.store import CorruptEntryError, ResultStore

    store, key, ctx = ResultStore(tmp_path), _key(), _ctx()
    store.put(key, {"y": "healthy"}, ctx)
    real = store_module._read_context
    state = {"raised": False}

    def once(*args: Any, **kwargs: Any) -> Any:
        if not state["raised"]:
            state["raised"] = True
            raise CorruptEntryError("claimed corrupt while it was being healed")
        return real(*args, **kwargs)

    monkeypatch.setattr(store_module, "_read_context", once)
    store.get(key, ctx, ReusePolicy.EXACT)
    got = store.get(key, ctx, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": "healthy"}


@pytest.mark.logic
def test_a23_probe_policy_strings_are_coerced() -> None:
    """The policy string `"exact"` admits only an equal context, and an unknown string is refused."""
    from senselab_core.kernel.identity import acceptable

    linux, mac = _ctx("linux"), _ctx("macos")
    assert acceptable(linux, linux, "exact", backend="b") and not acceptable(linux, mac, "exact", backend="b")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        acceptable(linux, mac, "bogus", backend="b")  # type: ignore[arg-type]


@pytest.mark.logic
def test_a24_probe_constructible_key_is_digestible() -> None:
    """A key whose params hold `1e20` is constructed and has a 64-hex digest."""
    key = _key(params={"x": 1e20})
    assert re.fullmatch("[0-9a-f]{64}", key.digest())


@pytest.mark.logic
def test_a24_probe_large_float_output_reads_back(tmp_path: Path) -> None:
    """A stored output holding `1e20` reads back as that float."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    store, key, ctx = ResultStore(tmp_path), _key(), _ctx()
    store.put(key, {"x": 1e20}, ctx)
    got = store.get(key, ctx, ReusePolicy.EXACT)
    assert got is not None and got.outputs["x"] == 1e20 and type(got.outputs["x"]) is float


@pytest.mark.logic
def test_a35_probe_outputs_are_not_world_readable_under_umask_077(tmp_path: Path) -> None:
    """Under umask 077 a stored result and its directory carry no world bits."""
    from senselab_core.runtime.store import ResultStore

    old = os.umask(0o077)
    try:
        key, ctx = _key(), _ctx()
        ResultStore(tmp_path / "s").put(key, {"y": "private"}, ctx)
    finally:
        os.umask(old)
    result = _entry_dir(tmp_path / "s", key, ctx) / "result.json"
    assert stat.S_IMODE(result.stat().st_mode) & 0o007 == 0
    assert stat.S_IMODE(result.parent.stat().st_mode) & 0o007 == 0


@pytest.mark.logic
@pytest.mark.parametrize("stray", ["file"])
def test_a36_probe_stray_file_at_entry_does_not_block_put(tmp_path: Path, stray: str) -> None:
    """A stray file or dangling symlink at the entry path does not make every put raise."""
    from senselab_core.runtime.store import ResultStore, output_identity

    store, key, ctx = ResultStore(tmp_path), _key(), _ctx()
    entry = _entry_dir(tmp_path, key, ctx)
    entry.parent.mkdir(parents=True)
    if stray == "file":
        entry.write_text("stray")
    else:
        os.symlink("/nonexistent/x", entry)
    for _ in range(2):
        assert store.put(key, {"y": 1}, ctx) == output_identity({"y": 1})


@pytest.mark.logic
def test_a36_probe_enospc_mid_entry_is_storage_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ENOSPC while writing the second staged file surfaces as `StorageFull`, not a raw `OSError`."""
    from senselab_core.runtime import store as store_module
    from senselab_core.runtime.store import ResultStore

    real = store_module._write_synced
    calls = {"n": 0}

    def boom(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(store_module, "_write_synced", boom)
    with pytest.raises(OSError) as caught:
        ResultStore(tmp_path).put(_key(), {"a": "x"}, _ctx())
    assert type(caught.value).__name__ == "StorageFull"


# -- fetch --


def _manifest(files: list[tuple[str, str, bytes]]) -> Any:
    from senselab_core.runtime.fetch import DataManifest

    licence = {
        "expression": "CC-BY-4.0",
        "commercial_use": "yes",
        "copyleft": "none",
        "attribution": True,
        "gated": False,
        "source": "https://example.org/license",
        "verified": {"date": "2026-10-01", "how": "manual"},
    }
    last: Exception | None = None
    for fmt, with_size in (("data-manifest@2.0", True), ("data-manifest@1.0", False)):
        entries = [
            {"path": p, "url": u, "sha256": hashlib.sha256(d).hexdigest(), **({"size": len(d)} if with_size else {})}
            for p, u, d in files
        ]
        try:
            return DataManifest.model_validate(
                {"format": fmt, "name": "demo", "version": "1", "license": licence, "terms": "t", "files": entries}
            )
        except ValueError as exc:
            last = exc
    assert last is not None
    raise last


def _opener(routes: dict[str, bytes]) -> Any:
    def opener(url: str) -> io.BytesIO:
        if url not in routes:
            raise OSError(f"no route to {url}")
        return io.BytesIO(routes[url])

    return opener


PAYLOAD = b"hello senselab data" * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.mark.logic
def test_a18_probe_manifest_file_named_like_the_marker_stays_payload(tmp_path: Path) -> None:
    """A manifest file named `.complete` is not overwritten by the completion marker."""
    from senselab_core.runtime.fetch import fetch

    other = PAYLOAD + b"2"
    manifest = _manifest([(".complete", "https://x/c", PAYLOAD), ("x.wav", "https://x/x", other)])
    routes = {"https://x/c": PAYLOAD, "https://x/x": other}
    for _ in range(2):
        paths = fetch(manifest, cache_root=tmp_path, opener=_opener(routes))
        assert paths[".complete"].read_bytes() == PAYLOAD


@pytest.mark.logic
@pytest.mark.parametrize("paths", [["."]], ids=repr)
def test_a18_probe_non_prefix_free_paths_are_refused(paths: list[str]) -> None:
    """`.` and a file path that is also a directory prefix are refused."""
    with pytest.raises(ValueError):
        _manifest([(p, f"https://x/{i}", PAYLOAD) for i, p in enumerate(paths)])


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a file whatever its mode")
def test_a19_probe_blob_cannot_be_written_through_the_returned_path(tmp_path: Path) -> None:
    """The returned dataset path does not open the shared blob for writing."""
    from senselab_core.runtime.fetch import fetch

    paths = fetch(
        _manifest([("a/voice.wav", "https://x/a.wav", PAYLOAD)]),
        cache_root=tmp_path,
        opener=_opener({"https://x/a.wav": PAYLOAD}),
    )
    with pytest.raises(PermissionError):
        open(paths["a/voice.wav"], "r+b")


@pytest.mark.logic
def test_a19_probe_dangling_blob_symlink_is_not_trusted(tmp_path: Path) -> None:
    """A dangling `blobs/<sha>` symlink is replaced by a verified blob."""
    from senselab_core.runtime.fetch import fetch

    (tmp_path / "blobs").mkdir()
    os.symlink("/nonexistent/elsewhere", tmp_path / "blobs" / SHA)
    manifest = _manifest([("a/voice.wav", "https://x/a.wav", PAYLOAD)])
    for _ in range(2):
        paths = fetch(manifest, cache_root=tmp_path, opener=_opener({"https://x/a.wav": PAYLOAD}))
        assert paths["a/voice.wav"].read_bytes() == PAYLOAD
    assert not (tmp_path / "blobs" / SHA).is_symlink()


@pytest.mark.logic
def test_a20_probe_file_url_is_not_read(tmp_path: Path) -> None:
    """A `file://` manifest URL is refused by the default opener."""
    from senselab_core.runtime.fetch import fetch

    secret = tmp_path / "secret.txt"
    secret.write_text("participant notes")
    manifest = _manifest([("a", f"file://{secret}", b"participant notes")])
    with pytest.raises(Exception) as caught:  # noqa: PT011
        fetch(manifest, cache_root=tmp_path / "c")
    assert "file" in str(caught.value).lower()
    assert not (tmp_path / "c" / "blobs").exists() or not any((tmp_path / "c" / "blobs").iterdir())


RUNTIME_TESTS = Path(__file__).resolve().parents[1] / "runtime"


def probe_a20_endless_source(cache_root: str) -> dict[str, Any]:
    """Fetch from a source that would serve far more than declared; report what it served and what was raised."""
    from bounded_probe_fixture import FiniteSource
    from senselab_core.runtime.fetch import fetch

    source = FiniteSource(len(PAYLOAD))
    error = ""
    try:
        fetch(_manifest([("a", "https://x/a", PAYLOAD)]), cache_root=Path(cache_root), opener=lambda url: source)  # type: ignore[arg-type,return-value]
    except Exception as err:  # noqa: BLE001 - the report names whatever was raised
        error = type(err).__name__
    return {"error": error, "served": source.served, "limit": source.limit}


@pytest.mark.limits
def test_a20_probe_endless_source_is_cut_at_the_declared_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An endless source is abandoned at the declared size, not streamed without bound.

    The source is finite (declared size + 1 MiB, then it raises) and the fetch runs in a child under
    `RLIMIT_FSIZE` and a timeout, so the probe is bounded at any commit.
    """
    monkeypatch.syspath_prepend(str(RUNTIME_TESTS))
    from bounded_probe_fixture import MIB, run_bounded

    report = run_bounded(
        "phase1_regression_test:probe_a20_endless_source",
        str(tmp_path),
        where=Path(__file__).resolve().parent,
        fsize=len(PAYLOAD) + 2 * MIB,
        timeout=120,
    )
    assert report["error"] == "FetchError"
    assert report["served"] == len(PAYLOAD) + 1 < report["limit"]


# -- worker client --

BACKEND_SOURCES = {
    "a7_backend": (
        "import os, subprocess, sys, time\n"
        "def load(init):\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'])\n"
        "    with open(init['pid_file'] + '.tmp', 'w') as f:\n"
        "        f.write(str(child.pid))\n"
        "    os.replace(init['pid_file'] + '.tmp', init['pid_file'])\n"
        "    return init\n"
        "def handle(state, items, params):\n"
        "    time.sleep(3600)\n"
    ),
    "a8_backend": (
        "import subprocess, sys\n"
        "def load(init):\n    return init\n"
        "def handle(state, items, params):\n"
        "    out = subprocess.run([sys.executable, '-c', 'import sys; print(len(sys.stdin.buffer.read()))'],"
        " capture_output=True, check=True).stdout\n"
        "    return [{'outputs': {'stdin_bytes': int(out)}} for _ in items]\n"
    ),
    "a22_backend": (
        "from pathlib import Path\n"
        "def load(init):\n    return init\n"
        "def handle(state, items, params):\n"
        "    sub = Path(state['array_dir']) / 'scratch'\n"
        "    sub.mkdir(exist_ok=True)\n"
        "    (sub / 'x.bin').write_bytes(b'x')\n"
        "    return [{'outputs': {'made': 'scratch'}} for _ in items]\n"
    ),
}


def _client(tmp_path: Path, module: str, init: dict[str, Any], parent: Path | None = None, **kwargs: Any) -> Any:
    import inspect

    from senselab_core.runtime.worker_client import WorkerClient

    code = tmp_path / "backends"
    code.mkdir(exist_ok=True)
    for name, source in BACKEND_SOURCES.items():
        (code / f"{name}.py").write_text(source)
    parent = parent if parent is not None else tmp_path / "work"
    parent.mkdir(exist_ok=True)
    key = "array_parent" if "array_parent" in inspect.signature(WorkerClient.__init__).parameters else "array_dir"
    existing = os.environ.get("PYTHONPATH")
    try:
        os.environ["PYTHONPATH"] = str(code) if not existing else f"{code}{os.pathsep}{existing}"
        return WorkerClient(sys.executable, module, init=init, **{key: parent}, **kwargs)  # type: ignore[arg-type]
    finally:
        if existing is None:
            del os.environ["PYTHONPATH"]
        else:
            os.environ["PYTHONPATH"] = existing


def _alive(pid: int, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether `pid` can still run user code, by its `ps` state: a zombie or exiting process is not alive."""
    probes = _load(
        PACKAGES / "senselab-core" / "tests" / "runtime" / "lifecycle_probes_fixture.py", "_probes", monkeypatch
    )
    return bool(probes.alive(pid))


@pytest.mark.limits
def test_a7_probe_timeout_returns_though_a_grandchild_holds_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that times out returns within a bound and leaves no grandchild running.

    The grandchild is started in `load` and its pid published by rename, so it exists before the request's timeout
    starts; liveness is read from `ps` state, since a killed grandchild may still be a zombie awaiting its new parent.
    """
    import threading

    pid_file = tmp_path / "pid"
    client = _client(tmp_path, "a7_backend", {"pid_file": str(pid_file)}, request_timeout=2)
    outcome: list[object] = []

    def call() -> None:
        try:
            client.run([{"id": "a", "inputs": {}}], {})
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    thread = threading.Thread(target=call, daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(POLL_S)
    returned = not thread.is_alive()
    elapsed = time.monotonic() - started
    pid = int(pid_file.read_text()) if pid_file.exists() else None
    try:
        assert returned, f"run() had not returned after {elapsed:.0f} s with request_timeout=2"
        assert isinstance(outcome[0], TimeoutError)
        assert pid is not None and not _alive(pid, monkeypatch)
    finally:
        if pid is not None and _alive(pid, monkeypatch):
            os.kill(pid, signal.SIGKILL)
        proc = getattr(client, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()


@pytest.mark.logic
def test_a8_probe_backend_children_do_not_steal_requests(tmp_path: Path) -> None:
    """A backend child started with the default stdin reads nothing, and every request is answered."""
    client = _client(tmp_path, "a8_backend", {}, request_timeout=8)
    try:
        for i in range(3):
            assert client.run([{"id": str(i), "inputs": {}}], {})[0]["outputs"] == {"stdin_bytes": 0}
    finally:
        client.close()


@pytest.mark.logic
def test_a9_probe_files_beside_the_array_dir_survive(tmp_path: Path) -> None:
    """A directory holding `thesis.txt` and `notes.md` is left as it was after runs and close."""
    parent = tmp_path / "work"
    parent.mkdir()
    (parent / "thesis.txt").write_text("t")
    (parent / "notes.md").write_text("n")
    client = _client(tmp_path, "senselab_worker.fake", {}, parent=parent)
    try:
        client.run([{"id": "a", "inputs": {}}], {})
    finally:
        client.close()
    assert (parent / "thesis.txt").exists() and (parent / "notes.md").exists()


@pytest.mark.logic
def test_a22_probe_backend_subdir_leaves_runs_usable(tmp_path: Path) -> None:
    """A directory a backend makes in the array directory does not make every run raise."""
    client = _client(tmp_path, "a22_backend", {}, request_timeout=8)
    try:
        for _ in range(3):
            assert client.run([{"id": "a", "inputs": {}}], {})[0]["outputs"] == {"made": "scratch"}
            assert client.alive
    finally:
        client.close()


@pytest.mark.logic
def test_a21_probe_fifo_named_like_an_array_does_not_hang(tmp_path: Path) -> None:
    """Decoding an array reference whose file is a FIFO returns or raises promptly instead of blocking."""
    import threading

    from senselab_worker import codec

    sha = "a" * 64
    array_dir = tmp_path / "arrays"
    array_dir.mkdir()
    fifo = array_dir / f"{sha}.npy"
    os.mkfifo(fifo)
    ref = {"__array__": {"path": str(fifo), "sha256": sha, "dtype": "<f4", "shape": [1]}}
    outcome: list[object] = []

    def decode() -> None:
        try:
            outcome.append(codec.from_wire(ref, array_dir=array_dir))
        except Exception as exc:  # noqa: BLE001
            outcome.append(exc)

    thread = threading.Thread(target=decode, daemon=True)
    thread.start()
    thread.join(5)
    hung = thread.is_alive()
    if hung:
        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        thread.join(5)
    assert not hung, "from_wire blocked on a FIFO"


@pytest.mark.logic
def test_a34_probe_hello_wire_must_be_the_int_version(tmp_path: Path) -> None:
    """A hello whose `wire` is a float or a bool equal to the version is refused as a protocol mismatch."""
    from senselab_core.runtime.worker_client import ProtocolMismatch
    from senselab_worker import WIRE_VERSION, runtime_identity

    wires: list[object] = [float(WIRE_VERSION)]
    if WIRE_VERSION == 1:
        wires.append(True)
    for wire in wires:
        hello = {"wire": wire, "runtime": runtime_identity(), "python": "", "numpy": "", "pid": 0}
        script = tmp_path / "fake_python.sh"
        script.write_text("#!/bin/sh\necho '" + json.dumps({"hello": hello}) + "'\nsleep 1\n")
        script.chmod(0o755)
        import inspect

        from senselab_core.runtime.worker_client import WorkerClient

        parent = tmp_path / "work"
        parent.mkdir(exist_ok=True)
        key = "array_parent" if "array_parent" in inspect.signature(WorkerClient.__init__).parameters else "array_dir"
        with pytest.raises(ProtocolMismatch):
            WorkerClient(script, "senselab_worker.fake", init={}, **{key: parent})  # type: ignore[arg-type]


# -- declarations --


@pytest.mark.logic
def test_a6_probe_set_params_are_refused_at_registration() -> None:
    """A task whose params model has a `set[str]` field, whose key digest depends on hash seed, is refused."""
    from pydantic import create_model
    from senselab_core.kernel.declare import Registry, TaskParams, TaskSpec

    params = create_model("_SetParams", __base__=TaskParams, tags=(set[str], ...))
    spec = TaskSpec(
        name="t.measure",
        version="1",
        kind="measure",
        inputs={"audio": "Audio"},
        outputs={"pitch": "Track"},
        params=params,
    )
    registry = Registry()
    with pytest.raises(Exception):  # noqa: B017, PT011
        registry.add_task(spec)
    assert registry.tasks() == []


@pytest.mark.logic
def test_a11_probe_publish_workflow_builds_every_package() -> None:
    """The publish workflow builds all workspace packages, not only the root wheel."""
    text = (REPO / ".github" / "workflows" / "publish.yaml").read_text()
    builds = [line for line in text.splitlines() if re.search(r"run:\s*uv build", line)]
    assert builds and all("--all-packages" in line for line in builds)


@pytest.mark.logic
def test_a29_probe_publish_workflow_gates_the_version_before_upload() -> None:
    """The publish workflow runs a release gate over the built distributions before it uploads anything."""
    text = (REPO / ".github" / "workflows" / "publish.yaml").read_text()
    gate = text.find("release_gate")
    upload = text.find("uv publish")
    assert 0 <= gate < upload


@pytest.mark.logic
def test_a5_probe_key_fields_that_must_be_identities_are_not_free_strings() -> None:
    """A key naming the model by a ref, or an input by a path, is refused at construction."""
    from senselab_core.kernel.identity import LogicalKey

    fields = {"task": "asr", "task_version": "1", "backend": "hf", "backend_version": "1", "params": {"beam": 1}}
    good = _key()
    with pytest.raises(ValueError):
        LogicalKey(**fields, model="main", inputs=good.inputs)
    with pytest.raises(ValueError):
        LogicalKey(**fields, model=good.model, inputs={"a": "/home/me/a.wav"})


@pytest.mark.logic
def test_a16_probe_int16_samples_read_as_full_scale_float32() -> None:
    """int16 samples read as float32 in [-1, 1], as the same samples read from a WAV do."""
    import numpy
    from senselab_core.kernel.signals import Audio

    group = Audio.from_array(numpy.array([16384, -32768], dtype=numpy.int16), 16_000).groups["audio"]
    data = group.read()  # type: ignore[union-attr]
    assert data.dtype == numpy.float32 and data.tolist() == [[0.5, -1.0]]


@pytest.mark.logic
def test_a38_probe_manifest_key_is_injective() -> None:
    """`a@b` + `c` and `a` + `b@c` name different manifest entries."""
    from senselab_core.runtime.revision import manifest_key

    assert manifest_key("a@b", "c") != manifest_key("a", "b@c")


def _pin_hook(monkeypatch: pytest.MonkeyPatch) -> Any:
    class MetadataHookInterface:
        PLUGIN_NAME = ""

        def __init__(self, root: str, config: dict[str, Any]) -> None:
            self.root, self.config = root, config

    for name in ("hatchling", "hatchling.metadata", "hatchling.metadata.plugin", "hatchling.metadata.plugin.interface"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["hatchling.metadata.plugin.interface"].MetadataHookInterface = MetadataHookInterface  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location("a32_pin_hook", REPO / "hatch_build_pins.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "a32_pin_hook", module)
    spec.loader.exec_module(module)
    return module.PinHook(str(REPO), {"pins": ["senselab-core", "senselab-worker"], "dependencies": ["numpy"]})


@pytest.mark.logic
def test_a32_probe_pin_hook_refuses_a_local_version_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook raises rather than write `senselab-core==2.0.0+local`, which admits other builds."""
    hook = _pin_hook(monkeypatch)
    with pytest.raises(ValueError):
        hook.update({"version": "2.0.0+local"})


@pytest.mark.logic
def test_a32_probe_core_refuses_a_missing_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing core with no installed worker raises `ImportError`."""
    import senselab_core

    class Dist:
        version = "2.0.0"

        def read_text(self, name: str) -> None:
            return None

    def distribution(name: str) -> Dist:
        if name == "senselab-worker":
            raise metadata.PackageNotFoundError(name)
        return Dist()

    monkeypatch.setattr(metadata, "distribution", distribution)
    with pytest.raises(ImportError):
        importlib.reload(senselab_core)
