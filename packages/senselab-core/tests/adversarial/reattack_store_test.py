"""H8c re-attack findings on the store, fetch, locks and run manifests (XS-1..XS-7).

The findings and their probes are in reviews/phase1-reattack/store.md.

Each `test_xs<n>_*` rebuilds its probe; the generated tests are contracts over the class the finding belongs to.
Every probe runs in a child under `bounded_probe_fixture.run_bounded` (file-size limit, address-space limit
where the platform takes one, timeout) with finite in-memory sources, so it is bounded at any commit. Planted
symlinks point only at a victim directory inside the probe's own scratch directory.
"""

import errno
import functools
import hashlib
import io
import json
import os
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
RUNTIME_TESTS = HERE.parent / "runtime"
PAYLOAD = b"hello senselab data" * 1000
SHA = hashlib.sha256(PAYLOAD).hexdigest()
VICTIM_BYTES = b"belongs to someone else"
OLD_UUID = "f" * 32
LICENCE = {
    "expression": "CC-BY-4.0",
    "commercial_use": "yes",
    "copyleft": "none",
    "attribution": True,
    "gated": False,
    "source": "https://example.org/license",
    "verified": {"date": "2026-10-01", "how": "manual"},
}


@pytest.fixture(autouse=True)
def _limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(RUNTIME_TESTS))


def _bounded(probe: str, *args: str) -> Any:  # noqa: ANN401 - the probe's JSON report
    from bounded_probe_fixture import MIB, run_bounded

    return run_bounded(f"reattack_store_test:{probe}", *args, where=HERE, fsize=16 * MIB, timeout=120)


# -- helpers the probes share (run in the child) --


def _manifest(paths: list[str]) -> Any:  # noqa: ANN401
    from senselab_core.runtime.fetch import DataManifest

    files = [{"path": p, "url": f"https://x/{i}", "sha256": SHA, "size": len(PAYLOAD)} for i, p in enumerate(paths)]
    return DataManifest.model_validate(
        {
            "format": "data-manifest@2.0",
            "name": "demo",
            "version": "1",
            "license": LICENCE,
            "terms": "t",
            "files": files,
        }
    )


def _opener(url: str) -> io.BytesIO:
    return io.BytesIO(PAYLOAD)


def _key() -> Any:  # noqa: ANN401
    from senselab_core.kernel.identity import LogicalKey, ModelIdentity

    return LogicalKey(
        task="asr",
        task_version="1",
        backend="hf",
        backend_version="1",
        model=ModelIdentity(source="hf", id="org/asr", identity="0" * 40),
        params={"beam": 1},
        inputs={"audio": "content:" + "a" * 64},
    )


def _ctx(platform: str = "linux") -> Any:  # noqa: ANN401
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


def _entry(root: Path, platform: str = "linux") -> Path:
    from senselab_core.runtime.store import STORE_DIR

    d = _key().digest()
    return root / STORE_DIR / d[:2] / d / _ctx(platform).fingerprint()


def _unseal(path: Path) -> None:
    for current, dirs, files in os.walk(path):
        os.chmod(current, 0o770)
        for name in dirs:
            os.chmod(os.path.join(current, name), 0o770)
        for name in files:
            os.chmod(os.path.join(current, name), 0o660)


def _snapshot(directory: Path) -> dict[str, list[Any]]:
    """Every object under `directory`, never followed: its kind, bytes or target, and mode."""
    seen: dict[str, list[Any]] = {}
    for current, dirs, files in os.walk(directory):
        for name in [*dirs, *files]:
            path = Path(current) / name
            st = os.lstat(path)
            rel = str(path.relative_to(directory))
            if stat.S_ISLNK(st.st_mode):
                seen[rel] = ["symlink", os.readlink(path), 0]
            elif stat.S_ISDIR(st.st_mode):
                seen[rel] = ["directory", "", stat.S_IMODE(st.st_mode)]
            else:
                seen[rel] = ["file", path.read_bytes().decode("latin-1"), stat.S_IMODE(st.st_mode)]
    return seen


def _victim(scratch: Path) -> Path:
    """A directory outside every root, holding a file, an old uuid-named directory and an old uuid-named file."""
    victim = scratch / "victim"
    (victim / OLD_UUID).mkdir(parents=True)
    (victim / "f").write_bytes(VICTIM_BYTES)
    (victim / OLD_UUID / "kept").write_bytes(VICTIM_BYTES)
    (victim / ("e" * 32)).write_bytes(VICTIM_BYTES)
    for path in (victim / OLD_UUID, victim / ("e" * 32)):
        os.utime(path, (0, 0))
    return victim


def _outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "", call()
    except Exception as err:  # noqa: BLE001 - the report names whatever was raised
        return type(err).__name__, None


# -- XS-1: fetch writes through a planted directory symlink under files/ --


def probe_xs1(scratch: str) -> dict[str, Any]:
    """Plant `files/sub -> victim` and fetch `sub/f`; report the error and the victim."""
    from senselab_core.runtime.fetch import fetch

    base = Path(scratch)
    victim = _victim(base)
    cache = base / "cache"
    files = cache / "datasets" / "demo@1" / "files"
    files.mkdir(parents=True)
    os.symlink(victim, files / "sub")
    before = _snapshot(victim)
    error, _ = _outcome(lambda: fetch(_manifest(["sub/f"]), cache_root=cache, opener=_opener))
    return {"error": error, "victim_unchanged": _snapshot(victim) == before}


@pytest.mark.logic
def test_xs1_fetch_does_not_write_through_a_symlinked_directory_under_files(tmp_path: Path) -> None:
    """A symlink to a directory outside the cache, planted under `files/`, is refused; nothing outside is written."""
    report = _bounded("probe_xs1", str(tmp_path))
    assert report == {"error": "ManifestConflictError", "victim_unchanged": True}


# -- XS-2: a directory at blobs/<sha256> wedges the dataset --


def probe_xs2(scratch: str, sealed: str) -> dict[str, Any]:
    """Make a directory at `blobs/<sha256>` and fetch twice; report the errors, the blob and the bytes served."""
    from senselab_core.runtime.fetch import fetch

    cache = Path(scratch) / "cache"
    (cache / "blobs" / SHA / "inner").mkdir(parents=True)
    if sealed == "sealed":
        os.chmod(cache / "blobs", 0o555)
    outcomes = []
    for _ in range(2):
        error, paths = _outcome(lambda: fetch(_manifest(["a/voice.wav"]), cache_root=cache, opener=_opener))
        message = ""
        if error:
            try:
                fetch(_manifest(["a/voice.wav"]), cache_root=cache, opener=_opener)
            except Exception as err:  # noqa: BLE001
                message = str(err)
        outcomes.append(
            {
                "error": error,
                "message": message,
                "served": paths["a/voice.wav"].read_bytes() == PAYLOAD if paths else None,
            }
        )
    os.chmod(cache / "blobs", 0o770)
    blob = os.lstat(cache / "blobs" / SHA)
    return {"outcomes": outcomes, "blob_is_file": stat.S_ISREG(blob.st_mode), "blob_path": str(cache / "blobs" / SHA)}


@pytest.mark.logic
def test_xs2_a_directory_at_a_blob_is_set_aside_and_the_blob_fetched(tmp_path: Path) -> None:
    """A directory where a blob belongs is moved aside, never a retryable error that wedges the dataset."""
    report = _bounded("probe_xs2", str(tmp_path), "open")
    assert [o["error"] for o in report["outcomes"]] == ["", ""]
    assert [o["served"] for o in report["outcomes"]] == [True, True]
    assert report["blob_is_file"] is True


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root renames out of a read-only directory")
def test_xs2_an_unmovable_directory_at_a_blob_is_a_typed_error_naming_it(tmp_path: Path) -> None:
    """When the directory cannot be moved, the error is not retryable and names the blob path and its owner."""
    report = _bounded("probe_xs2", str(tmp_path), "sealed")
    first = report["outcomes"][0]
    assert first["error"] == "WrongKindError"
    assert report["blob_path"] in first["message"] and "owned by" in first["message"]


# -- XS-3: a file at a store directory wedges put --


def probe_xs3(scratch: str, level: str, sealed: str) -> dict[str, Any]:
    """Replace the store directory at `level` on the way to an entry with a regular file; put and get."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k, c = ResultStore(root), _key(), _ctx()
    store.put(k, {"y": 1}, c)
    entry = _entry(root)
    target = {"store": entry.parents[2], "prefix": entry.parents[1], "digest": entry.parent}[level]
    _unseal(target)
    for current, dirs, files in os.walk(target, topdown=False):
        for name in files:
            os.unlink(os.path.join(current, name))
        for name in dirs:
            os.rmdir(os.path.join(current, name))
    os.rmdir(target)
    target.write_bytes(b"not a directory")
    if sealed == "sealed":
        os.chmod(target.parent, 0o555)
    get_error, got = _outcome(lambda: store.get(k, c, ReusePolicy.EXACT))
    put_error, _ = _outcome(lambda: store.put(k, {"y": 2}, c))
    message = ""
    if put_error:
        try:
            store.put(k, {"y": 2}, c)
        except Exception as err:  # noqa: BLE001
            message = str(err)
    os.chmod(target.parent, 0o770)
    again = store.get(k, c, ReusePolicy.EXACT) if not put_error else None
    return {
        "get_error": get_error,
        "got": None if got is None else got.outputs,
        "put_error": put_error,
        "message": message,
        "target": str(target),
        "after": None if again is None else again.outputs,
    }


@pytest.mark.logic
@pytest.mark.parametrize("level", ["store", "prefix", "digest"])
def test_xs3_a_file_at_a_store_directory_is_set_aside_and_put_commits(tmp_path: Path, level: str) -> None:
    """A regular file where a store directory belongs is quarantined; put commits and get serves the new entry."""
    report = _bounded("probe_xs3", str(tmp_path), level, "open")
    assert (report["get_error"], report["got"], report["put_error"]) == ("", None, "")
    assert report["after"] == {"y": 2}


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root renames out of a read-only directory")
@pytest.mark.parametrize("level", ["prefix", "digest"])
def test_xs3_an_unmovable_file_at_a_store_directory_is_a_typed_error_naming_it(tmp_path: Path, level: str) -> None:
    """When the file cannot be moved, put raises a non-retryable error naming the path and its owner."""
    report = _bounded("probe_xs3", str(tmp_path), level, "sealed")
    assert report["put_error"] == "WrongKindError"
    assert report["target"] in report["message"] and "owned by" in report["message"]


# -- XS-4: result.json is not bound to context.json --

SWAPPABLE = ("result.json", "context.json", "arrays")


def _outputs_of(platform: str) -> dict[str, Any]:
    import numpy

    return {"y": 1 if platform == "linux" else 2, "a": numpy.full(3, 1.0 if platform == "linux" else 2.0, "<f8")}


def _plain(outputs: dict[str, Any] | None) -> Any:  # noqa: ANN401
    if outputs is None:
        return None
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in outputs.items()}


def probe_xs4(scratch: str, swapped: str, policy: str) -> dict[str, Any]:
    """Put one key under two contexts, swap the members named in `swapped` between the entries, and get both."""
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k = ResultStore(root), _key()
    for platform in ("linux", "macos"):
        store.put(k, _outputs_of(platform), _ctx(platform))
    linux, macos = _entry(root, "linux"), _entry(root, "macos")
    _unseal(linux)
    _unseal(macos)
    members = set(swapped.split("+"))
    names = {"result.json", "context.json"} & members
    if "arrays" in members:
        names |= {p.name for p in [*linux.glob("*.npy"), *macos.glob("*.npy")]}
    for name in sorted(names):
        a, b = linux / name, macos / name
        held = (a.read_bytes() if a.exists() else None, b.read_bytes() if b.exists() else None)
        for path, data in ((a, held[1]), (b, held[0])):
            if path.exists():
                path.unlink()
            if data is not None:
                path.write_bytes(data)
    for entry in (linux, macos):
        for member in entry.iterdir():
            os.chmod(member, 0o440)
        os.chmod(entry, 0o550)
    served = {}
    for platform in ("linux", "macos"):
        hit = store.get(k, _ctx(platform), policy)
        served[platform] = None if hit is None else [hit.context.platform, _plain(hit.outputs)]
    return {"served": served, "expected": {p: _plain(_outputs_of(p)) for p in ("linux", "macos")}}


@pytest.mark.logic
def test_xs4_swapped_result_files_are_not_served_under_the_other_context(tmp_path: Path) -> None:
    """Swapping two contexts' result.json files and arrays makes both entries misfiled; neither is served."""
    report = _bounded("probe_xs4", str(tmp_path), "result.json+arrays", "exact")
    assert report["served"] == {"linux": None, "macos": None}


def _subsets() -> Iterator[str]:
    for mask in range(1, 1 << len(SWAPPABLE)):
        yield "+".join(name for i, name in enumerate(SWAPPABLE) if mask >> i & 1)


@pytest.mark.contract
@pytest.mark.parametrize("policy", ["exact", "tolerant"])
@pytest.mark.parametrize("swapped", list(_subsets()))
def test_an_entry_serves_only_the_outputs_its_context_wrote_whatever_members_are_swapped(
    tmp_path: Path, swapped: str, policy: str
) -> None:
    """For every set of members swapped between two contexts' entries, a hit's outputs are its own context's."""
    report = _bounded("probe_xs4", str(tmp_path), swapped, policy)
    for hit in report["served"].values():
        assert hit is None or hit[1] == report["expected"][hit[0]], report


# -- XS-5: put trusts an entry whose array is corrupt --

ARRAY_CORRUPTIONS = ("npy_bytes", "npy_truncated", "npy_missing", "npy_directory", "npy_symlink")


def probe_xs5(scratch: str, corruption: str) -> dict[str, Any]:
    """Corrupt a committed entry's array, put the same outputs, and get at once."""
    import numpy
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k, c = ResultStore(root), _key(), _ctx()
    outputs: dict[str, object] = {"a": numpy.arange(4, dtype="<f8")}
    identity = store.put(k, outputs, c)
    entry = _entry(root)
    _unseal(entry)
    (npy,) = entry.glob("*.npy")
    data = npy.read_bytes()
    npy.unlink()
    if corruption == "npy_bytes":
        npy.write_bytes(data[:-8] + b"\1" * 8)
    elif corruption == "npy_truncated":
        npy.write_bytes(data[:-8])
    elif corruption == "npy_directory":
        npy.mkdir()
    elif corruption == "npy_symlink":
        (Path(scratch) / "elsewhere.npy").write_bytes(data)
        os.symlink(Path(scratch) / "elsewhere.npy", npy)
    again = store.put(k, outputs, c)
    hit = store.get(k, c, ReusePolicy.EXACT)
    return {"same_identity": again == identity, "served": None if hit is None else _plain(hit.outputs)}


@pytest.mark.logic
def test_xs5_put_over_an_entry_with_a_corrupt_array_rewrites_it(tmp_path: Path) -> None:
    """A put over an entry whose array no longer hashes replaces it, so the get right after it is served."""
    report = _bounded("probe_xs5", str(tmp_path), "npy_bytes")
    assert report == {"same_identity": True, "served": {"a": [0.0, 1.0, 2.0, 3.0]}}


@pytest.mark.contract
@pytest.mark.parametrize("corruption", ARRAY_CORRUPTIONS)
def test_a_put_that_reports_success_is_served_by_the_next_get(tmp_path: Path, corruption: str) -> None:
    """For every way an entry's array can be damaged, the put after it leaves an entry that get serves."""
    report = _bounded("probe_xs5", str(tmp_path), corruption)
    assert report == {"same_identity": True, "served": {"a": [0.0, 1.0, 2.0, 3.0]}}


# -- XS-6: took_over after a clean release --


def probe_xs6(scratch: str, hold: str) -> dict[str, Any]:
    """Hold the lock for `hold` seconds in one thread while another waits for it; report the waiter's took_over."""
    from senselab_core.runtime.locks import SharedFileLock

    resource = Path(scratch) / "thing"
    acquired, release = threading.Event(), threading.Event()

    def holder() -> None:
        with SharedFileLock(resource, timeout=30):
            acquired.set()
            time.sleep(float(hold))
        release.set()

    thread = threading.Thread(target=holder)
    thread.start()
    assert acquired.wait(30)
    with SharedFileLock(resource, timeout=30) as waiter:
        waited = waiter.took_over
    thread.join(30)
    leftover = Path(str(resource) + ".holder")
    leftover.write_text(json.dumps({"user": "u", "host": "h", "pid": 1, "taken_at": time.time()}))
    Path(str(resource) + ".heartbeat").touch()
    with SharedFileLock(resource, timeout=30) as taker:
        crashed = taker.took_over
    return {"after_clean_release": waited, "after_a_holder_that_never_released": crashed}


@pytest.mark.logic
def test_xs6_took_over_is_false_after_a_clean_release(tmp_path: Path) -> None:
    """A waiter that gets the lock after a clean 0.3 s hold did not take over; a leftover holder is taken over."""
    report = _bounded("probe_xs6", str(tmp_path), "0.3")
    assert report == {"after_clean_release": False, "after_a_holder_that_never_released": True}


@pytest.mark.contract
@pytest.mark.parametrize("hold", ["0", "0.05", "0.3", "0.6", "1.2"])
def test_took_over_names_only_a_holder_that_never_released_whatever_the_wait(tmp_path: Path, hold: str) -> None:
    """For every hold time, a clean release is not a takeover and a leftover identity is."""
    report = _bounded("probe_xs6", str(tmp_path), hold)
    assert report == {"after_clean_release": False, "after_a_holder_that_never_released": True}


# -- XS-7: run manifests: umask, run id, leftover temporaries --

BAD_RUN_IDS = ("../../escaped", "..", ".", "", "a/b", "a\\b", "a\x00b", "/abs")


def probe_xs7_run_id(scratch: str, run: str) -> dict[str, Any]:
    """Record a resolution under run id `run`; report the error and every file written outside `runs/`."""
    from senselab_core.runtime import revision

    cache = Path(scratch) / "cache"
    os.environ["SENSELAB_CACHE"] = str(cache)
    error, _ = _outcome(lambda: revision.record_resolution("org/m", "main", "a" * 40, run=run))
    runs = cache / "runs"
    outside = sorted(
        str(Path(current, name).relative_to(scratch))
        for current, _, files in os.walk(scratch)
        for name in files
        if not Path(current, name).is_relative_to(runs)
    )
    return {"error": error, "outside": outside}


@pytest.mark.logic
def test_xs7_a_run_id_with_a_parent_component_is_refused(tmp_path: Path) -> None:
    """`../../escaped` is not a run id: nothing is written outside `runs/`."""
    assert _bounded("probe_xs7_run_id", str(tmp_path), "../../escaped") == {"error": "ValueError", "outside": []}


@pytest.mark.logic
@pytest.mark.parametrize("run", BAD_RUN_IDS, ids=repr)
def test_every_run_id_that_is_not_one_component_is_refused(tmp_path: Path, run: str) -> None:
    """Each run id that is not a single path component is refused before anything is written."""
    assert _bounded("probe_xs7_run_id", str(tmp_path), run) == {"error": "ValueError", "outside": []}


def probe_xs7_env_run_id(scratch: str, run: str) -> str:
    """Return the error `run_id()` raises when SENSELAB_RUN_ID is `run`."""
    from senselab_core.runtime import revision

    os.environ["SENSELAB_CACHE"] = str(Path(scratch) / "cache")
    os.environ["SENSELAB_RUN_ID"] = run
    return _outcome(revision.run_id)[0]


@pytest.mark.logic
def test_xs7_a_run_id_from_the_environment_is_one_component(tmp_path: Path) -> None:
    """SENSELAB_RUN_ID naming a parent directory is refused."""
    assert _bounded("probe_xs7_env_run_id", str(tmp_path), "../../escaped") == "ValueError"
    assert _bounded("probe_xs7_env_run_id", str(tmp_path), "job-123") == ""


def probe_xs7_modes(scratch: str, umask: str) -> dict[str, Any]:
    """Record two resolutions under `umask`, a dead writer's temporary planted between; report modes and leftovers."""
    from senselab_core.runtime import revision
    from senselab_core.runtime.storage import StorePolicy, committed_modes, shared_dir_mode

    cache = Path(scratch) / "cache"
    os.environ["SENSELAB_CACHE"] = str(cache)
    os.umask(int(umask, 8))
    run_dir = cache / "runs" / "r1"
    leftover = run_dir / f".{revision.MANIFEST_NAME}.12.34.{'0' * 32}.tmp"
    revision.record_resolution("org/m", "main", "a" * 40, run="r1")
    leftover.write_bytes(b"{")
    revision.record_resolution("org/n", "main", "b" * 40, run="r1")
    policy = StorePolicy.load_default()

    def mode(path: Path) -> int:
        return stat.S_IMODE(os.lstat(path).st_mode) & 0o777

    return {
        "modes": {
            "runs": mode(cache / "runs"),
            "run": mode(run_dir),
            "manifest": mode(run_dir / revision.MANIFEST_NAME),
        },
        "expected": {
            "runs": shared_dir_mode(policy) & 0o777,
            "run": shared_dir_mode(policy) & 0o777,
            "manifest": committed_modes(policy)[0] & 0o777,
        },
        "leftover_removed": not leftover.exists(),
        "entries": sorted(json.loads((run_dir / revision.MANIFEST_NAME).read_text())),
    }


@pytest.mark.logic
def test_xs7_run_manifests_take_the_policy_modes_under_umask_0(tmp_path: Path) -> None:
    """Under umask 0, `runs/`, the run directory and the manifest have the policy's modes, not world-writable ones."""
    report = _bounded("probe_xs7_modes", str(tmp_path), "0")
    assert report["modes"] == report["expected"]
    assert report["leftover_removed"] is True and len(report["entries"]) == 2


@pytest.mark.contract
@pytest.mark.parametrize("umask", ["0", "022", "077", "007"])
def test_run_manifest_modes_do_not_depend_on_the_umask(tmp_path: Path, umask: str) -> None:
    """For every umask, the run manifest and its directories carry exactly the policy's modes."""
    report = _bounded("probe_xs7_modes", str(tmp_path), umask)
    assert report["modes"] == report["expected"]


# -- XS-1..XS-3 class: every directory an operation reaches, planted with a symlink outside or a file --

OPERATIONS = ("put", "get", "collect", "fetch", "record")
TYPED = {"", "WrongKindError", "ManifestConflictError"}


def _setup(op: str, root: Path) -> None:
    """Bring `root` to the state `op` starts from."""
    from senselab_core.runtime.store import ResultStore

    if op in ("get", "collect"):
        ResultStore(root).put(_key(), _outputs_of("linux"), _ctx())
    if op == "get":
        ResultStore(root).put(_key(), _outputs_of("macos"), _ctx("macos"))
        damaged = _entry(root, "macos")
        _unseal(damaged)
        (damaged / "context.json").write_bytes(b"{}")
    if op == "collect":
        for name in ("quarantine", "tmp"):
            old = root / name / OLD_UUID
            old.mkdir(parents=True, exist_ok=True)
            os.utime(old, (0, 0))


def _run(op: str, root: Path) -> bool:
    """Run `op` against `root`; return whether its result is right."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime import revision
    from senselab_core.runtime.fetch import fetch
    from senselab_core.runtime.store import ResultStore

    if op == "put":
        store = ResultStore(root)
        store.put(_key(), _outputs_of("linux"), _ctx())
        hit = store.get(_key(), _ctx(), ReusePolicy.EXACT)
        return hit is not None and _plain(hit.outputs) == _plain(_outputs_of("linux"))
    if op == "get":
        hit = ResultStore(root).get(_key(), _ctx(), ReusePolicy.EXACT)
        return hit is None or _plain(hit.outputs) == _plain(_outputs_of("linux"))
    if op == "collect":
        ResultStore(root).collect(now=time.time() + 10**9)
        return True
    if op == "fetch":
        paths = fetch(_manifest(["a/voice.wav"]), cache_root=root, opener=_opener)
        return paths["a/voice.wav"].read_bytes() == PAYLOAD
    os.environ["SENSELAB_CACHE"] = str(root)
    revision.record_resolution("org/m", "main", "a" * 40, run="r1")
    return revision.read_manifest("r1") == {revision.manifest_key("org/m", "main"): "a" * 40}


def _directories(op: str, scratch: Path) -> list[str]:
    root = scratch / "clean" / "root"
    _setup(op, root)
    _run(op, root)
    return sorted(str(Path(c).relative_to(root)) for c, _, _ in os.walk(root) if Path(c) != root)


def _remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    _unseal(path)
    for current, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.unlink(os.path.join(current, name))
        for name in dirs:
            target = os.path.join(current, name)
            os.unlink(target) if os.path.islink(target) else os.rmdir(target)
    os.rmdir(path)


def probe_planted(scratch: str, op: str, plant: str) -> list[dict[str, Any]]:
    """For each directory `op` reaches, plant a symlink to a victim (or a file) there and run `op` again."""
    base = Path(scratch)
    reports = []
    for i, rel in enumerate(_directories(op, base)):
        case = base / f"case{i}"
        root, victim = case / "root", _victim(case)
        _setup(op, root)
        target = root / rel
        if os.path.lexists(target):
            _remove_tree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if plant == "symlink":
            os.symlink(victim, target)
        else:
            target.write_bytes(VICTIM_BYTES)
        before = _snapshot(victim)
        error, right = _outcome(functools.partial(_run, op, root))
        reports.append({"at": rel, "error": error, "right": right, "victim_unchanged": _snapshot(victim) == before})
    return reports


@pytest.mark.contract
@pytest.mark.parametrize("plant", ["symlink", "file"])
@pytest.mark.parametrize("op", OPERATIONS)
def test_no_operation_writes_through_or_wedges_on_a_planted_object_at_any_directory(
    tmp_path: Path, op: str, plant: str
) -> None:
    """At every directory an operation reaches, a planted symlink or file is never written or deleted through.

    The operation either completes correctly or raises a typed, non-retryable error.
    """
    reports = _bounded("probe_planted", str(tmp_path), op, plant)
    assert reports, "the operation reached no directory"
    for report in reports:
        assert report["victim_unchanged"], report
        assert report["error"] in TYPED, report
        assert report["error"] or report["right"], report
    if op != "fetch" and plant == "symlink":
        assert all(r["error"] == "" for r in reports), reports


# -- the primitive: one planted test per operation --


def probe_primitive(scratch: str, operation: str) -> dict[str, Any]:
    """Run one `StorageDir` operation on a name or path that is a symlink to a victim; report the victim and error."""
    from senselab_core.runtime.storage import TEMPORARY_NAME, StorageDir
    from senselab_worker.ownership import UUID_HEX

    base = Path(scratch)
    victim = _victim(base)
    root = base / "root"
    (root / "real").mkdir(parents=True)
    os.symlink(victim, root / "link")
    os.symlink(victim / "f", root / "file_link")
    before = _snapshot(victim)

    def run() -> Any:  # noqa: ANN401
        with StorageDir.open_root(root) as top:
            if operation == "child":
                return top.child("link")
            if operation == "child_create":
                return top.child("link", create=True, mode=0o2770)
            if operation == "walk_intermediate":
                return top.walk("link", "below", create=True, mode=0o2770)
            if operation == "open_new":
                return top.open_new("file_link", 0o640)
            if operation == "write_new":
                return top.write_new("file_link", b"overwritten", 0o640)
            if operation == "write_atomically":
                top.write_atomically("file_link", b"overwritten", mode=0o640)
                return top.kind("file_link")
            if operation == "symlink":
                return top.symlink("elsewhere", "link")
            if operation == "link_source":
                top.link("file_link", top, "hard")
                return top.kind("hard")
            if operation == "link_target":
                (root / "real" / "x").write_bytes(b"x")
                with top.child("real") as real:
                    return real.link("x", top, "file_link")
            if operation == "rename_onto":
                (root / "real" / "x").write_bytes(b"x")
                with top.child("real") as real:
                    real.rename("x", top, "file_link", replace=True)
                return top.kind("file_link")
            if operation == "remove_owned":
                os.symlink(victim / "f", root / ("0" * 32))
                os.symlink(victim, root / ("1" * 32))
                return [top.remove_owned("0" * 32, pattern=UUID_HEX), top.remove_owned("1" * 32, pattern=UUID_HEX)]
            if operation == "remove_owned_temporary":
                name = f".x.1.2.{'0' * 32}.tmp"
                os.symlink(victim / "f", root / name)
                return top.remove_owned(name, pattern=TEMPORARY_NAME)
            if operation == "set_aside":
                with top.child("real") as real:
                    return str(top.set_aside("link", real))
            raise AssertionError(operation)

    error, value = _outcome(run)
    if isinstance(value, Path):
        value = str(value)
    return {
        "error": error,
        "value": value if isinstance(value, (str, list, bool, type(None))) else repr(value),
        "victim_unchanged": _snapshot(victim) == before,
    }


PRIMITIVE = {
    "child": "WrongKindError",
    "child_create": "WrongKindError",
    "walk_intermediate": "WrongKindError",
    "open_new": "FileExistsError",
    "write_new": "FileExistsError",
    "write_atomically": "",
    "symlink": "FileExistsError",
    "link_source": "",
    "link_target": "FileExistsError",
    "rename_onto": "",
    "remove_owned": "",
    "remove_owned_temporary": "",
    "set_aside": "",
}


@pytest.mark.contract
@pytest.mark.parametrize("operation", sorted(PRIMITIVE))
def test_every_storage_operation_on_a_planted_symlink_leaves_its_target_alone(tmp_path: Path, operation: str) -> None:
    """Every `StorageDir` operation given a symlink refuses it or acts on the symlink itself.

    The symlink is a directory on the way, the name operated on, or the target; what it points at is never
    written, moved or removed.
    """
    report = _bounded("probe_primitive", str(tmp_path), operation)
    assert report["victim_unchanged"], report
    assert report["error"] == PRIMITIVE[operation], report
    if operation in ("write_atomically", "rename_onto"):
        assert report["value"] == "file"
    if operation == "link_source":
        assert report["value"] == "symlink"
