"""Planted-object probes for the storage root, run in a child under `bounded_probe_fixture.run_bounded`.

Each `probe_*` plants a symlink, hard link, file, directory or corrupt member where a store, fetch, lease or run
manifest writes, runs the operation, and returns a JSON report of the outcome and of what outside the root changed.
The probes use finite in-memory sources and plant symlinks only at a victim directory inside their own scratch
directory, so they are bounded at any commit.
"""

import functools
import hashlib
import io
import json
import os
import re
import stat
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
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


def bounded(probe: str, *args: str) -> Any:  # noqa: ANN401 - the probe's JSON report
    """Run `probe` in a bounded child and return its JSON report."""
    from bounded_probe_fixture import MIB, run_bounded

    return run_bounded(f"planted_objects_fixture:{probe}", *args, where=HERE, fsize=16 * MIB, timeout=120)


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


def store_key(task: str = "asr", **inputs: str) -> Any:  # noqa: ANN401
    """Return the store tests' key of `task` over `inputs`, port to 64-hex content digest (`audio` by default)."""
    from senselab_core.kernel.identity import LogicalKey, ModelIdentity

    return LogicalKey(
        task=task,
        task_version="1",
        backend="hf",
        backend_version="1",
        model=ModelIdentity(id="org/asr", identity="0" * 40),
        params={"beam": 1},
        backend_params={},
        inputs={port: "content:" + digest for port, digest in (inputs or {"audio": "a" * 64}).items()},
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
        model_source="hf",
        env_license=None,
    )


def _entry(root: Path, platform: str = "linux") -> Path:
    from senselab_core.runtime.store import STORE_DIR

    d = store_key().digest()
    entry: Path = root / STORE_DIR / d[:2] / d / _ctx(platform).fingerprint()
    return entry


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
                seen[rel] = ["symlink", os.readlink(path), 0, st.st_mtime_ns]
            elif stat.S_ISDIR(st.st_mode):
                seen[rel] = ["directory", "", stat.S_IMODE(st.st_mode), st.st_mtime_ns]
            else:
                seen[rel] = ["file", path.read_bytes().decode("latin-1"), stat.S_IMODE(st.st_mode), st.st_mtime_ns]
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


# -- fetch writes through a planted directory symlink under files/ --


def probe_symlinked_directory_under_files(scratch: str) -> dict[str, Any]:
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


# -- a directory at blobs/<sha256> wedges the dataset --


def probe_directory_at_a_blob(scratch: str, sealed: str) -> dict[str, Any]:
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


# -- a file at a store directory wedges put --


def probe_file_at_a_store_directory(scratch: str, level: str, sealed: str) -> dict[str, Any]:
    """Replace the store directory at `level` on the way to an entry with a regular file; put and get."""
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k, c = ResultStore(root), store_key(), _ctx()
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


# -- result.json is not bound to context.json --

SWAPPABLE = ("result.json", "context.json", "arrays")


def _outputs_of(platform: str) -> dict[str, Any]:
    import numpy

    return {"y": 1 if platform == "linux" else 2, "a": numpy.full(3, 1.0 if platform == "linux" else 2.0, "<f8")}


def _plain(outputs: dict[str, Any] | None) -> Any:  # noqa: ANN401
    if outputs is None:
        return None
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in outputs.items()}


def probe_swapped_entry_members(scratch: str, swapped: str, policy: str) -> dict[str, Any]:
    """Put one key under two contexts, swap the members named in `swapped` between the entries, and get both."""
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k = ResultStore(root), store_key()
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


def swap_subsets() -> Iterator[str]:
    """Every non-empty set of swappable entry members, as `+`-joined names."""
    for mask in range(1, 1 << len(SWAPPABLE)):
        yield "+".join(name for i, name in enumerate(SWAPPABLE) if mask >> i & 1)


# -- a re-put over an entry whose array is damaged --

ARRAY_CORRUPTIONS = ("npy_bytes", "npy_truncated", "npy_missing", "npy_directory", "npy_symlink")


def probe_corrupt_array(scratch: str, corruption: str) -> dict[str, Any]:
    """Corrupt a committed entry's array, then twice put the same outputs and get at once; report each get."""
    import numpy
    from senselab_core.kernel.identity import ReusePolicy
    from senselab_core.runtime.store import ResultStore

    root = Path(scratch) / "root"
    store, k, c = ResultStore(root), store_key(), _ctx()
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
    served = []
    for _ in range(2):
        again = store.put(k, outputs, c)
        hit = store.get(k, c, ReusePolicy.EXACT)
        served.append(None if hit is None else _plain(hit.outputs))
    return {"same_identity": again == identity, "served": served}


# -- run manifests: umask, run id, leftover temporaries --

BAD_RUN_IDS = ("../../escaped", "..", ".", "", "a/b", "a\\b", "a\x00b", "/abs")


def probe_run_id(scratch: str, run: str) -> dict[str, Any]:
    """Record a resolution under run id `run`; report the error and every file written outside `runs/`."""
    from senselab_core.runtime import revision

    cache = Path(scratch) / "cache"
    os.environ["SENSELAB_CACHE"] = str(cache)
    error, _ = _outcome(lambda: revision.record_resolution("org/m", "main", "a" * 40, run=run))
    runs = cache / revision.NAMESPACE / "runs"
    outside = sorted(
        str(Path(current, name).relative_to(scratch))
        for current, _, files in os.walk(scratch)
        for name in files
        if not Path(current, name).is_relative_to(runs)
    )
    return {"error": error, "outside": outside}


def probe_environment_run_id(scratch: str, run: str) -> str:
    """Return the error `run_id()` raises when SENSELAB_RUN_ID is `run`."""
    from senselab_core.runtime import revision

    os.environ["SENSELAB_CACHE"] = str(Path(scratch) / "cache")
    os.environ["SENSELAB_RUN_ID"] = run
    return _outcome(revision.run_id)[0]


def probe_run_manifest_modes(scratch: str, umask: str) -> dict[str, Any]:
    """Record two resolutions under `umask`, a dead writer's temporary planted between; report modes and leftovers."""
    from senselab_core.runtime import revision
    from senselab_core.runtime.storage import StorePolicy, committed_modes, shared_dir_mode

    cache = Path(scratch) / "cache"
    os.environ["SENSELAB_CACHE"] = str(cache)
    os.umask(int(umask, 8))
    run_dir = cache / revision.NAMESPACE / "runs" / "r1"
    leftover = run_dir / f".{revision.MANIFEST_NAME}.12.34.{'0' * 32}.tmp"
    revision.record_resolution("org/m", "main", "a" * 40, run="r1")
    leftover.write_bytes(b"{")
    revision.record_resolution("org/n", "main", "b" * 40, run="r1")
    policy = StorePolicy.load_default()

    def mode(path: Path) -> int:
        return stat.S_IMODE(os.lstat(path).st_mode) & 0o777

    return {
        "modes": {
            "namespace": mode(cache / revision.NAMESPACE),
            "runs": mode(cache / revision.NAMESPACE / "runs"),
            "run": mode(run_dir),
            "manifest": mode(run_dir / revision.MANIFEST_NAME),
        },
        "expected": {
            "namespace": shared_dir_mode(policy) & 0o777,
            "runs": shared_dir_mode(policy) & 0o777,
            "run": shared_dir_mode(policy) & 0o777,
            "manifest": committed_modes(policy)[0] & 0o777,
        },
        "leftover_removed": not leftover.exists(),
        "entries": sorted(json.loads((run_dir / revision.MANIFEST_NAME).read_text())),
    }


# -- every directory an operation reaches, planted with a symlink outside or a file --

OPERATIONS = ("put", "get", "collect", "fetch", "record")
TYPED = {"", "WrongKindError", "ManifestConflictError"}


def _setup(op: str, root: Path) -> None:
    """Bring `root` to the state `op` starts from."""
    from senselab_core.runtime.store import ResultStore

    if op in ("get", "collect"):
        ResultStore(root).put(store_key(), _outputs_of("linux"), _ctx())
    if op == "get":
        ResultStore(root).put(store_key(), _outputs_of("macos"), _ctx("macos"))
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
        store.put(store_key(), _outputs_of("linux"), _ctx())
        hit = store.get(store_key(), _ctx(), ReusePolicy.EXACT)
        return hit is not None and _plain(hit.outputs) == _plain(_outputs_of("linux"))
    if op == "get":
        hit = ResultStore(root).get(store_key(), _ctx(), ReusePolicy.EXACT)
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


# -- the primitive: one planted test per operation --


def probe_primitive(  # noqa: C901, PLR0915 - the table of probed primitives, one branch each
    scratch: str, operation: str
) -> dict[str, Any]:
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

    def run() -> Any:  # noqa: ANN401, C901, PLR0911, PLR0912 - one return per probed operation
        with StorageDir.open_root(root) as top:
            if operation == "child":
                return top.child("link")
            if operation == "child_create":
                return top.child("link", create=True, mode=0o2770)
            if operation == "walk_intermediate":
                return top.walk("link", "below", create=True, mode=0o2770)
            if operation == "open_new":
                return top.open_new("file_link", 0o640, grammar=ANY)
            if operation == "write_new":
                return top.write_new("file_link", b"overwritten", 0o640, grammar=ANY)
            if operation == "write_atomically":
                top.write_atomically("file_link", b"overwritten", grammar=ANY, mode=0o640)
                return top.kind("file_link")
            if operation == "write_atomically_hard_link":
                os.link(victim / "f", root / "hard_link")
                top.write_atomically("hard_link", b"overwritten", grammar=ANY, mode=0o640)
                return top.kind("hard_link")
            if operation == "fs_now_symlink":
                os.symlink(victim / "f", root / ".clock")
                top.fs_now()
                return top.kind(".clock")
            if operation == "fs_now_hard_link":
                os.link(victim / "f", root / ".clock")
                top.fs_now()
                return top.kind(".clock")
            if operation == "read_regular":
                return top.read_regular("file_link")
            if operation == "outside_grammar":
                return top.write_new("not-owned", b"x", 0o640, grammar=re.compile("owned"))
            if operation == "symlink":
                return top.symlink("elsewhere", "link", grammar=ANY)
            if operation == "link_source":
                top.link("file_link", top, "hard", grammar=ANY)
                return top.kind("hard")
            if operation == "link_target":
                (root / "real" / "x").write_bytes(b"x")
                with top.child("real") as real:
                    return real.link("x", top, "file_link", grammar=ANY)
            if operation == "rename_onto":
                (root / "real" / "x").write_bytes(b"x")
                with top.child("real") as real:
                    real.rename("x", top, "file_link", grammar=ANY, replace=True)
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


ANY = re.compile(r".+")
PRIMITIVE = {
    "write_atomically_hard_link": "",
    "fs_now_symlink": "",
    "fs_now_hard_link": "",
    "read_regular": "WrongKindError",
    "outside_grammar": "ValueError",
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


# -- every name a writer writes, planted with a symlink, a hard link, a directory or corrupt content --

NAME_OPERATIONS = ("put", "get", "collect", "fetch", "record", "lease")
PLANTS = ("symlink", "hardlink", "directory", "corrupt")
NAME_TYPED = {"", "WrongKindError", "ManifestConflictError", "LeaseTimeout", "LeaseUnavailable"}
OLD = 10**9
NAME_BOUND_S = 30.0


def _run_named(op: str, root: Path) -> bool:
    """Run `op` (the directory contract's operations, plus taking and releasing a lease) against `root`."""
    if op != "lease":
        return _run(op, root)
    from senselab_core.runtime.locks import Lease

    root.mkdir(parents=True, exist_ok=True)
    with Lease(root / "thing", timeout=5, stale_after=0.5) as lease:
        return lease.owns()


def _fd_path(fd: int) -> Path:
    """The path of the directory open on `fd`."""
    if os.path.isdir("/proc/self/fd"):
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    import fcntl

    return Path(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0").decode())  # type: ignore[attr-defined]


def _at(name: Any, dir_fd: int | None) -> Path:  # noqa: ANN401
    path = Path(os.fsdecode(name))
    return path if dir_fd is None or path.is_absolute() else _fd_path(dir_fd) / path


def _recording() -> list[Path]:
    """Record in this child the path of every name created through `os` (open, link, symlink, rename); return it.

    `shutil` is imported first: it decides at import whether `os.open` takes `dir_fd`.
    """
    import shutil  # noqa: F401

    written: list[Path] = []
    real_open, real_link, real_symlink = os.open, os.link, os.symlink
    real_rename, real_replace = os.rename, os.replace

    def opened(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:  # noqa: ANN401
        if flags & os.O_CREAT:
            written.append(_at(path, dir_fd))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def linked(src: Any, dst: Any, *, src_dir_fd: int | None = None, dst_dir_fd: int | None = None, **kw: Any) -> None:  # noqa: ANN401
        written.append(_at(dst, dst_dir_fd))
        real_link(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, **kw)

    def symlinked(src: Any, dst: Any, target_is_directory: bool = False, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        written.append(_at(dst, dir_fd))
        real_symlink(src, dst, target_is_directory, dir_fd=dir_fd)

    def moving(real: Callable[..., None]) -> Callable[..., None]:
        def moved(src: Any, dst: Any, *, src_dir_fd: int | None = None, dst_dir_fd: int | None = None) -> None:  # noqa: ANN401
            written.append(_at(dst, dst_dir_fd))
            real(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        return moved

    os.open = opened
    os.link = linked
    os.symlink = symlinked
    os.rename = moving(real_rename)
    os.replace = moving(real_replace)
    return written


def _ephemeral(rel: Path) -> bool:
    from senselab_core.runtime.fetch import LINK_STAGING_NAME
    from senselab_core.runtime.storage import TEMPORARY_NAME
    from senselab_worker.ownership import UUID_HEX

    return any(UUID_HEX.fullmatch(p) for p in rel.parts) or bool(
        TEMPORARY_NAME.fullmatch(rel.name) or LINK_STAGING_NAME.fullmatch(rel.name)
    )


def _names(op: str, scratch: Path, written: list[Path]) -> list[str]:
    """Run `op` cleanly; return the names to plant at."""
    root = scratch / "clean" / "root"
    _setup(op, root)
    written.clear()
    _run_named(op, root)
    real_root = root.resolve()
    rels = {Path(os.path.realpath(p.parent)).relative_to(real_root) / p.name for p in written}
    final = {Path(c).relative_to(root) / n for c, _, files in os.walk(root) for n in files}
    clocks = {r.with_name(".clock") for r in rels if r.name.endswith(".lease")}
    return sorted(str(r) for r in rels | final | clocks if not _ephemeral(r))


def _plant(target: Path, plant: str, victim: Path) -> None:
    if os.path.lexists(target):
        os.chmod(target.parent, 0o770)
        _remove_tree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if plant == "symlink":
        os.symlink(victim / "f", target)
        os.utime(target, (OLD, OLD), follow_symlinks=False)
    elif plant == "hardlink":
        os.link(victim / "f", target)
    elif plant == "directory":
        target.mkdir()
        (target / "inside").write_bytes(b"inside")
        os.utime(target, (OLD, OLD))
    else:
        target.write_bytes(b"\x00{corrupt")
        os.utime(target, (OLD, OLD))


def probe_planted_names(scratch: str, op: str, plant: str) -> dict[str, Any]:
    """At each name `op` writes, plant `plant` and run `op`; report each outcome, the victim and what is outside."""
    written = _recording()
    base = Path(scratch)
    targets = _names(op, base, written)
    reports = []
    for i, rel in enumerate(targets):
        case = base / f"case{i}"
        root, victim = case / "root", case / "victim"
        victim.mkdir(parents=True)
        (victim / "f").write_bytes(VICTIM_BYTES)
        os.chmod(victim / "f", 0o600)
        os.utime(victim / "f", (OLD, OLD))
        _setup(op, root)
        _plant(root / rel, plant, victim)
        before = _snapshot(victim)
        started = time.monotonic()
        error, right = _outcome(lambda root=root: _run_named(op, root))  # type: ignore[misc]
        reports.append(
            {
                "at": rel,
                "error": error,
                "right": right,
                "seconds": time.monotonic() - started,
                "victim_unchanged": _snapshot(victim) == before,
                "outside": sorted(p.name for p in case.iterdir()),
            }
        )
    return {"reports": reports}


# -- run manifests, leases and fetch under planted objects --

LEASE = "resolutions@2.json.lease"


def probe_record_with_plant(  # noqa: C901 - one branch per plant
    scratch: str, name: str, plant: str
) -> dict[str, Any]:
    """Plant `plant` at `runs/r1/<name>` and record a resolution; report the error, the victim and what is outside."""
    from senselab_core.runtime import revision

    base = Path(scratch)
    cache, victim = base / "cache", base / "victim"
    victim.mkdir()
    (victim / "f").write_bytes(VICTIM_BYTES)
    os.chmod(victim / "f", 0o600)
    os.utime(victim / "f", (OLD, OLD))
    target = cache / revision.NAMESPACE / "runs" / "r1" / name
    target.parent.mkdir(parents=True)
    if plant == "symlink":
        os.symlink(victim / "f", target)
    elif plant == "dangling":
        os.symlink(victim / "created-outside", target)
    elif plant == "hardlink":
        os.link(victim / "f", target)
    elif plant == "directory":
        target.mkdir()
    elif plant == "fifo":
        os.mkfifo(target)
    elif plant in ("unopenable", "unmovable"):
        target.write_bytes(b"someone else's")
        os.chmod(target, 0o000)
    else:
        target.write_bytes(plant.encode())
    if plant != "hardlink":
        os.utime(target, (OLD, OLD), follow_symlinks=False)
    before = _snapshot(victim)
    os.environ["SENSELAB_CACHE"] = str(cache)
    if plant == "unmovable":
        os.chmod(target.parent, 0o555)
    started = time.monotonic()
    error, _ = _outcome(lambda: revision.record_resolution("org/m", "main", "a" * 40, run="r1"))
    message = ""
    if error:
        try:
            revision.record_resolution("org/m", "main", "a" * 40, run="r1")
        except Exception as err:  # noqa: BLE001
            message = str(err)
    os.chmod(target.parent, 0o770)
    return {
        "message": message,
        "error": error,
        "seconds": time.monotonic() - started,
        "recorded": revision.read_manifest("r1"),
        "victim_unchanged": _snapshot(victim) == before,
        "victim": sorted(p.name for p in victim.iterdir()),
    }


def probe_swap_after_link(scratch: str) -> dict[str, Any]:
    """Swap `blobs/<sha>` for other bytes right after fetch links it; report the marker's stamps and a re-fetch."""
    import json as json_module

    from senselab_core.runtime import fetch as fetch_module

    cache = Path(scratch) / "cache"
    real_link, real_replace = os.link, os.replace
    modes: list[int] = []

    def committing(real: Callable[..., None]) -> Callable[..., None]:
        def commit(
            src: Any,  # noqa: ANN401
            dst: Any,  # noqa: ANN401
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            **kw: Any,  # noqa: ANN401
        ) -> None:
            final = os.path.basename(os.fsdecode(dst))
            if final in (SHA, ".complete") and src_dir_fd is not None:
                modes.append(stat.S_IMODE(os.fstat(src_dir_fd).st_mode))
            real(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, **kw)
            if final == SHA:
                where = _at(dst, dst_dir_fd).parent
                (where / ".swap").write_bytes(b"x" * len(PAYLOAD))
                real_replace(where / ".swap", where / final)

        return commit

    os.link = committing(real_link)
    os.replace = committing(real_replace)
    fetch_module.fetch(_manifest(["a/voice.wav"]), cache_root=cache, opener=_opener)
    os.link, os.replace = real_link, real_replace
    marker = json_module.loads((cache / "datasets" / "demo@1" / ".complete").read_bytes())
    paths = fetch_module.fetch(_manifest(["a/voice.wav"]), cache_root=cache, opener=_opener, verify="size")
    return {
        "stamped": sorted(marker["stamps"]),
        "served": paths["a/voice.wav"].read_bytes() == PAYLOAD,
        "staging_modes": sorted(set(modes)),
    }


def probe_clock_hard_link(scratch: str, op: str) -> dict[str, Any]:
    """Hard-link a victim at the root's `.clock`, then collect on the store clock, or fetch; report the victim."""
    base = Path(scratch)
    root, victim = base / "root", base / "victim"
    victim.mkdir()
    (victim / "f").write_bytes(VICTIM_BYTES)
    os.chmod(victim / "f", 0o600)
    os.utime(victim / "f", (OLD, OLD))
    _setup("collect" if op == "collect" else "fetch", root)
    root.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(root / ".clock"):
        os.unlink(root / ".clock")
    os.link(victim / "f", root / ".clock")
    before = _snapshot(victim)
    if op == "collect":
        from senselab_core.runtime.store import ResultStore

        error, _ = _outcome(lambda: ResultStore(root).collect())
    else:
        error, _ = _outcome(lambda: _run(op, root))
    return {"error": error, "victim_unchanged": _snapshot(victim) == before}


UNOPENABLE_NAMES = (LEASE, "resolutions@2.json")
