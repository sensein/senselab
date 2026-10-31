"""The content-addressed result store: layout, reuse policy, concurrent writers and provenance."""

import errno
import hashlib
import json
import multiprocessing as mp
import os
import pwd
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Barrier as MpBarrier
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from typing import Any

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelIdentity, ReusePolicy
from senselab_core.runtime import storage as storage_module
from senselab_core.runtime import store as store_module
from senselab_core.runtime.provenance import CHECKSUM_KEY, ProvStore
from senselab_core.runtime.storage import StorageFull, StorePolicy, StoreUnavailable
from senselab_core.runtime.store import (
    QUARANTINE_DIR,
    STORE_DIR,
    CorruptEntryError,
    MisfiledEntryError,
    ResultStore,
    StoredResult,
    output_identity,
)
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.digest import identity_digest
from senselab_worker.ownership import UUID_HEX


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(task: str = "asr", **inputs: str) -> LogicalKey:
    return LogicalKey(
        task=task,
        task_version="1",
        backend="hf",
        backend_version="1",
        model=ModelIdentity(source="hf", id="org/asr", identity="0" * 40),
        params={"beam": 1},
        inputs={port: "content:" + digest for port, digest in (inputs or {"audio": "a" * 64}).items()},
    )


def _ctx(platform: str = "linux", host: str = "node1") -> ExecutionContext:
    return ExecutionContext(
        env_lock="lock",
        platform=platform,
        wheel_set="wheels",
        numerics={"device": "cpu", "precision": "fp32"},
        decoder=None,
        driver={},
        host=host,
        protocol="p1",
    )


def _entries(root: Path) -> list[Path]:
    return sorted((root / STORE_DIR).rglob("result.json"))


@pytest.mark.logic
def test_put_get_exact(tmp_path: Path) -> None:
    """A put reads back exactly: arrays as canonical .npy named by hash, the rest as JCS JSON."""
    s = ResultStore(tmp_path)
    k, c = _key(), _ctx()
    outputs: dict[str, object] = {"y": numpy.arange(6, dtype="<f4").reshape(2, 3), "text": "hi", "n": [1, 2.5, None]}
    ident = s.put(k, outputs, c)
    got = s.get(k, c, ReusePolicy.EXACT)
    assert isinstance(got, StoredResult)
    assert got.output_identity == ident
    assert got.context == c
    assert got.policy_used is ReusePolicy.EXACT
    assert got.outputs["text"] == "hi" and got.outputs["n"] == [1, 2.5, None]
    y = got.outputs["y"]
    assert isinstance(y, numpy.ndarray) and y.dtype == numpy.dtype("<f4")
    numpy.testing.assert_array_equal(y, outputs["y"])
    d = k.digest()
    entry = tmp_path / STORE_DIR / d[:2] / d / c.fingerprint()
    assert (entry / "result.json").is_file() and (entry / "context.json").is_file()
    result = json.loads((entry / "result.json").read_bytes())
    assert result["format"] == "result@2.0"
    assert result["context_fingerprint"] == c.fingerprint() and result["key_digest"] == d
    assert result["output_identity"] == ident
    sha = _sha(canonical_npy_bytes(outputs["y"]))  # type: ignore[arg-type]
    assert (entry / f"{sha}.npy").read_bytes() == canonical_npy_bytes(outputs["y"])  # type: ignore[arg-type]
    assert json.loads((entry / "context.json").read_bytes())["format"] == "execution-context@1.0"
    assert s.get(_key(task="other"), c, ReusePolicy.TOLERANT) is None


@pytest.mark.logic
def test_cross_context_follows_policy(tmp_path: Path) -> None:
    """Another context's result is served only under a policy that admits it; host is not fingerprinted."""
    s, k = ResultStore(tmp_path), _key()
    s.put(k, {"y": 1}, _ctx(platform="linux"))
    assert s.get(k, _ctx(platform="macos"), ReusePolicy.EXACT) is None
    assert s.get(k, _ctx(platform="macos"), ReusePolicy.EQUIVALENT) is None
    got = s.get(k, _ctx(platform="macos"), ReusePolicy.TOLERANT)
    assert got is not None and got.policy_used is ReusePolicy.TOLERANT
    assert got.context.platform == "linux"
    same = s.get(k, _ctx(platform="linux", host="node2"), ReusePolicy.EQUIVALENT)
    assert same is not None and same.policy_used is ReusePolicy.EXACT


@pytest.mark.logic
def test_equivalence_class_admits_under_equivalent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A measured equivalence class admits a cross-context result under `equivalent`, not `exact`."""
    s, k = ResultStore(tmp_path), _key()
    linux, mac = _ctx(platform="linux"), _ctx(platform="macos")
    s.put(k, {"y": 1}, linux)
    monkeypatch.setattr(
        "senselab_core.kernel.identity.EQUIVALENCE",
        {"hf": {"cpu-fp32": frozenset({linux.fingerprint(), mac.fingerprint()})}},
    )
    got = s.get(k, mac, ReusePolicy.EQUIVALENT)
    assert got is not None and got.policy_used is ReusePolicy.EQUIVALENT
    assert s.get(k, mac, ReusePolicy.EXACT) is None


@pytest.mark.logic
def test_exact_match_preferred_over_other_contexts(tmp_path: Path) -> None:
    """The current context's own entry wins over any other the policy admits."""
    s, k = ResultStore(tmp_path), _key()
    s.put(k, {"y": 1}, _ctx(platform="linux"))
    s.put(k, {"y": 2}, _ctx(platform="macos"))
    got = s.get(k, _ctx(platform="macos"), ReusePolicy.TOLERANT)
    assert got is not None and got.outputs == {"y": 2} and got.policy_used is ReusePolicy.EXACT


def _put_racing(root: str, value: int, barrier: MpBarrier, out: "MpQueue[Any]") -> None:
    """Put one key after every racer reaches the barrier; module-scoped so spawn can pickle it."""
    s = ResultStore(Path(root))
    k, c = _key(), _ctx()
    barrier.wait()
    out.put(s.put(k, {"y": numpy.full(1000, value, dtype="<f8")}, c))


@pytest.mark.logic
def test_two_writers_one_entry(tmp_path: Path) -> None:
    """Two processes putting one key leave one entry and both return the winner's identity."""
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=_put_racing, args=(str(tmp_path), v, barrier, out)) for v in (1, 2)]
    for p in procs:
        p.start()
    identities = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    assert len(_entries(tmp_path)) == 1
    assert identities[0] == identities[1]
    got = ResultStore(tmp_path).get(_key(), _ctx(), ReusePolicy.EXACT)
    assert got is not None and got.output_identity == identities[0]
    assert not any((tmp_path / "tmp").iterdir())


def _put_many(root: str, value: int, keys: int, barrier: MpBarrier, out: "MpQueue[Any]") -> None:
    """Put `keys` distinct keys after every racer reaches the barrier; module-scoped for spawn."""
    s = ResultStore(Path(root))
    barrier.wait()
    out.put([s.put(_key(audio=f"{i:064x}"), {"y": numpy.full(1000, value, dtype="<f8")}, _ctx()) for i in range(keys)])


@pytest.mark.logic
def test_four_writers_agree_on_every_key(tmp_path: Path) -> None:
    """Four processes racing over twenty keys leave one entry per key and agree on every identity."""
    ctx, writers, keys = mp.get_context("spawn"), 4, 20
    barrier, out = ctx.Barrier(writers), ctx.Queue()
    procs = [ctx.Process(target=_put_many, args=(str(tmp_path), v, keys, barrier, out)) for v in range(writers)]
    for p in procs:
        p.start()
    results = [out.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    assert all(r == results[0] for r in results)
    assert len(_entries(tmp_path)) == keys
    assert not any((tmp_path / "tmp").iterdir())


@pytest.mark.logic
def test_losing_rename_returns_winner_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer whose rename meets a committed entry discards its staging and returns the winner's identity."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    winner = s.put(k, {"y": 1}, c)
    assert [a.parameters["outcome"] for a in s.provenance.activities()] == ["committed"]
    real = store_module._committed_identity
    calls: list[Path] = []

    def miss_first(entry: Path, *, digest: str, fingerprint: str) -> str | None:
        calls.append(entry)
        return None if len(calls) == 1 else real(entry, digest=digest, fingerprint=fingerprint)

    monkeypatch.setattr(store_module, "_committed_identity", miss_first)
    loser = s.put(k, {"y": 2}, c)
    monkeypatch.undo()
    assert len(calls) == 2
    assert loser == winner
    assert [a.parameters["outcome"] for a in s.provenance.activities()] == ["committed", "existing"]
    assert len(_entries(tmp_path)) == 1
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 1}
    assert not any((tmp_path / "tmp").iterdir())


@pytest.mark.logic
def test_crash_leaves_no_entry(tmp_path: Path) -> None:
    """A staging dir never renamed is invisible to get; put removes it once stale and keeps a fresh one."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    stale = tmp_path / "tmp" / ("5" * 32)
    stale.mkdir(parents=True)
    (stale / "result.json").write_bytes(jcs_dumps({"format": "result@1.0", "outputs": {}, "output_identity": "x"}))
    old = time.time() - StorePolicy.load_default().staging_stale_s - 60
    os.utime(stale, (old, old))
    live = tmp_path / "tmp" / ("1" * 32)
    live.mkdir()
    assert s.get(k, c, ReusePolicy.TOLERANT) is None
    s.put(_key(task="other"), {"y": 1}, c)
    assert not stale.exists()
    os.utime(live, (old, old))
    ResultStore(tmp_path).put(_key(task="other"), {"y": 1}, c)
    assert not live.exists()
    live.mkdir()
    assert live.is_dir()
    assert s.get(k, c, ReusePolicy.TOLERANT) is None


@pytest.mark.logic
def test_output_identity_enables_early_cutoff(tmp_path: Path) -> None:
    """Different keys with identical outputs share one output identity; dtype and shape change it."""
    s, c = ResultStore(tmp_path), _ctx()
    k1, k2 = _key(audio="a" * 64), _key(audio="b" * 64)
    assert s.put(k1, {"y": numpy.ones(3, "<f4")}, c) == s.put(k2, {"y": numpy.ones(3, "<f4")}, c)
    assert output_identity({"y": numpy.ones(3, "<f4")}) != output_identity({"y": numpy.ones(3, "<f8")})
    assert output_identity({"y": numpy.ones(3, "<f4")}) != output_identity({"y": numpy.ones(4, "<f4")})


@pytest.mark.logic
def test_output_identity_is_tagged_over_array_digests() -> None:
    """The identity is the `output@3` digest of the outputs with each array replaced by its .npy digest."""
    a = numpy.arange(4, dtype="<i8")
    expected = "output:" + identity_digest("output@3", {"y": {"$array": _sha(canonical_npy_bytes(a))}, "z": 1})
    assert output_identity({"z": 1, "y": a}) == expected
    assert output_identity({"y": numpy.asarray(a, dtype=">i8")}) == output_identity({"y": a})


@pytest.mark.logic
def test_put_refuses_what_it_cannot_serialise(tmp_path: Path) -> None:
    """Unserialisable values, the reserved key and non-finite floats are refused before anything is written."""
    s = ResultStore(tmp_path)
    with pytest.raises(TypeError):
        s.put(_key(), {"y": object()}, _ctx())
    with pytest.raises(ValueError, match=r"\$array"):
        s.put(_key(), {"y": {"$array": "a" * 64}}, _ctx())
    with pytest.raises(ValueError):
        s.put(_key(), {"y": float("nan")}, _ctx())
    assert _entries(tmp_path) == []


def _entry_dir(root: Path, k: LogicalKey, c: ExecutionContext) -> Path:
    d = k.digest()
    return root / STORE_DIR / d[:2] / d / c.fingerprint()


def _unseal(entry: Path) -> Path:
    """Make a committed entry writable, as an operator with the owner's rights could."""
    entry.chmod(0o770)
    for member in entry.iterdir():
        member.chmod(0o660)
    return entry


def _rewrite_json(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    _unseal(path.parent)
    record = json.loads(path.read_bytes())
    change(record)
    path.write_bytes(jcs_dumps(record))


def _only_npy(entry: Path) -> Path:
    (npy,) = _unseal(entry).glob("*.npy")
    return npy


CORRUPTIONS: dict[str, Callable[[Path], object]] = {
    "npy_hash": lambda e: _only_npy(e).write_bytes(canonical_npy_bytes(numpy.zeros(3, "<f4"))),
    "npy_missing": lambda e: _only_npy(e).unlink(),
    "context_missing": lambda e: (_unseal(e) / "context.json").unlink(),
    "result_missing": lambda e: (_unseal(e) / "result.json").unlink(),
    "result_garbled": lambda e: (_unseal(e) / "result.json").write_bytes(b"{not json"),
    "context_garbled": lambda e: (_unseal(e) / "context.json").write_bytes(b"[1, 2"),
    "npy_fifo": lambda e: _npy_to_fifo(e),
    "array_ref_escapes": lambda e: _rewrite_json(
        e / "result.json", lambda r: _forge(r, {"y": {"$array": "../" * 3 + "x"}, "n": 1})
    ),
    "context_field_missing": lambda e: _rewrite_json(e / "context.json", lambda r: r.pop("platform")),
    "identity_missing": lambda e: _rewrite_json(e / "result.json", lambda r: r.pop("output_identity")),
    "result_format": lambda e: _rewrite_json(e / "result.json", lambda r: r.update(format="result@1.0")),
    "context_format": lambda e: _rewrite_json(e / "context.json", lambda r: r.update(format="nope@1.0")),
    "outputs_tampered": lambda e: _rewrite_json(e / "result.json", lambda r: r["outputs"].update(n=2)),
    "key_digest_rewritten": lambda e: _rewrite_json(e / "result.json", lambda r: r.update(key_digest="0" * 64)),
    "key_rewritten": lambda e: _rewrite_json(
        e / "result.json", lambda r: r.update(key=_key(task="other").model_dump())
    ),
    "entry_is_file": lambda e: (shutil.rmtree(_unseal(e)), e.write_bytes(b"stray")),
}


def _npy_to_fifo(entry: Path) -> None:
    _only_npy(entry).unlink()
    os.mkfifo(_only_npy_name(entry))


def _only_npy_name(entry: Path) -> Path:
    return entry / str(json.loads((entry / "result.json").read_bytes())["outputs"]["y"]["$array"] + ".npy")


def _forge(record: dict[str, Any], outputs: dict[str, object]) -> None:
    """Replace a record's outputs and recompute their output identity, as a forger with write access could."""
    record["outputs"] = outputs
    record["output_identity"] = "output:" + identity_digest("output@3", outputs)


@pytest.mark.logic
@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_corrupt_entry_is_quarantined_not_served(tmp_path: Path, kind: str) -> None:
    """Every corruption is quarantined by get, never served, and a later put recommits the key."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": numpy.ones(3, "<f4"), "n": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS[kind](entry)
    with pytest.raises(CorruptEntryError):
        store_module._read_result(entry, digest=k.digest(), fingerprint=c.fingerprint())
    assert s.get(k, c, ReusePolicy.TOLERANT) is None
    assert not entry.exists()
    assert len(list((tmp_path / QUARANTINE_DIR).iterdir())) == 1
    ident = s.put(k, {"y": numpy.ones(3, "<f4"), "n": 1}, c)
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == ident


@pytest.mark.logic
def test_put_replaces_corrupt_entry(tmp_path: Path) -> None:
    """A put whose target entry is corrupt quarantines it and commits its own outputs."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": numpy.ones(3, "<f4")}, c)
    _only_npy(_entry_dir(tmp_path, k, c)).unlink()
    ident = s.put(k, {"y": numpy.zeros(3, "<f4")}, c)
    assert ident == output_identity({"y": numpy.zeros(3, "<f4")})
    assert {a.parameters["outcome"] for a in s.provenance.activities(node="asr")} == {"committed"}
    assert [a.parameters["outcome"] for a in s.provenance.activities(node="store.quarantine")] == ["quarantined"]
    assert ident in {e.attributes.get("output_identity") for e in s.provenance.entities("artifact")}
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == ident
    assert len(list((tmp_path / QUARANTINE_DIR).iterdir())) == 1


@pytest.mark.logic
def test_get_falls_through_to_sibling_context(tmp_path: Path) -> None:
    """A corrupt current-context entry is quarantined and an admitted sibling context served instead."""
    s, k = ResultStore(tmp_path), _key()
    linux, mac = _ctx(platform="linux"), _ctx(platform="macos")
    s.put(k, {"y": 1}, linux)
    s.put(k, {"y": 2}, mac)
    (_unseal(_entry_dir(tmp_path, k, mac)) / "result.json").write_bytes(b"")
    got = s.get(k, mac, ReusePolicy.TOLERANT)
    assert got is not None and got.outputs == {"y": 1} and got.policy_used is ReusePolicy.TOLERANT
    assert not _entry_dir(tmp_path, k, mac).exists()
    assert _entry_dir(tmp_path, k, linux).is_dir()


@pytest.mark.logic
def test_quarantine_tolerates_concurrent_quarantine(tmp_path: Path) -> None:
    """Quarantining an entry another process already moved is not an error."""
    s = ResultStore(tmp_path)
    s._quarantine(tmp_path / STORE_DIR / "ab" / ("ab" * 32) / ("0" * 64), digest="ab" * 32, fingerprint="0" * 64)
    assert not (tmp_path / QUARANTINE_DIR).exists() or list((tmp_path / QUARANTINE_DIR).iterdir()) == []
    assert s.provenance.activities(node="store.quarantine") == []


@pytest.fixture
def restrictive_umask() -> Iterator[None]:
    """Run the test under umask 0o077."""
    old = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.logic
def test_relaxed_reuse_is_recorded_in_provenance(tmp_path: Path) -> None:
    """A hit admitted under a policy other than exact appends a reuse activity; an exact hit does not."""
    s, k = ResultStore(tmp_path), _key()
    linux, mac = _ctx(platform="linux"), _ctx(platform="macos")
    ident = s.put(k, {"y": 1}, linux)
    assert s.get(k, linux, ReusePolicy.TOLERANT) is not None
    assert s.provenance.activities(node="store.reuse") == []
    assert s.get(k, mac, ReusePolicy.TOLERANT) is not None
    (act,) = s.provenance.activities(node="store.reuse")
    assert act.parameters == {
        "key": k.digest(),
        "stored_context": linux.fingerprint(),
        "current_context": mac.fingerprint(),
        "policy_used": "tolerant",
    }
    (used,) = [s.provenance.get_entity(e) for e in s.provenance.uses_of(act.id)]
    assert used.attributes[CHECKSUM_KEY] == ident.removeprefix("output:")


@pytest.mark.logic
def test_put_records_provenance(tmp_path: Path) -> None:
    """Put appends one activity: the key's inputs used, the output identity generated, the context recorded."""
    prov = ProvStore(run_id="r")
    s, k, c = ResultStore(tmp_path, provenance=prov), _key(audio="a" * 64), _ctx()
    ident = s.put(k, {"y": 1}, c)
    (act,) = prov.activities(node="asr")
    assert act.parameters["key"] == k.digest()
    assert act.parameters["context"]["fingerprint"] == c.fingerprint()
    assert act.parameters["context"]["host"] == "node1"
    used = [prov.get_entity(e) for e in prov.uses_of(act.id)]
    assert [(e.attributes["input"], e.attributes["identity"]) for e in used] == [("audio", "content:" + "a" * 64)]
    (generated,) = [e for e in prov.entities("artifact") if prov.generated_by(e.id) == act.id]
    assert generated.attributes[CHECKSUM_KEY] == ident.removeprefix("output:")
    assert ResultStore(tmp_path).provenance.activities() == []


@pytest.mark.logic
def test_a31_output_identity_differs_from_key_digest_of_same_tree() -> None:
    """The probe: a key's fields stored as outputs hashed to the key's own digest."""
    k = _key()
    tree = k.model_dump()
    assert output_identity(tree).removeprefix("output:") != k.digest()
    assert output_identity(tree) == "output:" + identity_digest("output@3", tree)


@pytest.mark.logic
@pytest.mark.parametrize(
    "lossy",
    [
        numpy.datetime64("NaT"),
        numpy.timedelta64(1, "ns"),
        numpy.ma.array([1, 2, 3], mask=[0, 0, 1]),
        numpy.zeros(1, dtype=[("a", "<i4")])[0],
        ReusePolicy.EXACT,
    ],
    ids=["nat", "timedelta", "masked", "void", "str_enum"],
)
def test_a4_store_refuses_lossy_outputs(tmp_path: Path, lossy: object) -> None:
    """The probe: NaT stored as null and a masked array as its fill values collided with other outputs."""
    with pytest.raises(TypeError):
        output_identity({"a": lossy})
    with pytest.raises(TypeError):
        ResultStore(tmp_path).put(_key(), {"a": [lossy]}, _ctx())
    assert _entries(tmp_path) == []


@pytest.mark.logic
def test_store_scalars_go_through_the_one_canonical_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scalar the store encodes passes through `canonical_scalar`; there is no second encoder."""
    seen: list[object] = []
    real = store_module.canonical_scalar

    def spy(v: object) -> object:
        seen.append(v)
        return real(v)

    monkeypatch.setattr(store_module, "canonical_scalar", spy)
    scalars = [None, True, 1, 0.5, "s", numpy.int32(2), numpy.float32(0.25)]
    output_identity({"xs": scalars, "y": numpy.ones(2, "<f4")})
    assert [type(v) for v in seen] == [type(v) for v in scalars]


@pytest.mark.logic
def test_a24_large_float_output_reads_back(tmp_path: Path) -> None:
    """An integral float beyond 2**53 survives put and get with its identity intact."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    ident = s.put(k, {"x": 1e20, "key": k.model_dump()}, c)
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == ident and got.outputs["x"] == 1e20


@pytest.mark.logic
@pytest.mark.parametrize("policy", ["exact", "equivalent", "tolerant", "bogus", None])
def test_a23_get_coerces_policy(tmp_path: Path, policy: object) -> None:
    """`get` coerces its policy like `acceptable`: a member's value works, anything else raises."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    if policy in {p.value for p in ReusePolicy}:
        hit = s.get(k, c, policy)  # type: ignore[arg-type]
        assert hit is not None and hit.policy_used is ReusePolicy.EXACT
        return
    with pytest.raises(ValueError):
        s.get(k, c, policy)  # type: ignore[arg-type]


@pytest.mark.logic
def test_store_dir_is_format_4() -> None:
    """Entries that record their context fingerprint are a new key space."""
    assert STORE_DIR == "store@4"


def _cycles() -> list[object]:
    d: dict[str, object] = {"a": 1}
    d["self"] = d
    lst: list[object] = [1]
    lst.append(lst)
    deep: list[object] = [{"x": []}]
    deep[0]["x"].append(deep)  # type: ignore[index]
    return [d, lst, deep]


@pytest.mark.logic
@pytest.mark.parametrize("cyclic", _cycles(), ids=["self_dict", "self_list", "through_dict"])
def test_store_cyclic_output_raises_value_error(tmp_path: Path, cyclic: object) -> None:
    """A cyclic output raises the encoders' `ValueError`, not `RecursionError`, and nothing is written."""
    with pytest.raises(ValueError, match="contains itself"):
        output_identity({"y": cyclic})
    with pytest.raises(ValueError, match="contains itself"):
        ResultStore(tmp_path).put(_key(), {"y": cyclic}, _ctx())
    assert _entries(tmp_path) == []
    shared = [1, 2]
    assert output_identity({"a": shared, "b": shared}) == output_identity({"a": [1, 2], "b": [1, 2]})


@pytest.mark.logic
def test_integral_float_output_reads_back_as_int(tmp_path: Path) -> None:
    """Canonical JSON writes `3.0` as `3`, so it reads back as the int 3 with the same identity."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    ident = s.put(k, {"x": 3.0, "y": 2.5, "z": [1.0]}, c)
    assert ident == output_identity({"x": 3, "y": 2.5, "z": [1]})
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == ident
    assert got.outputs == {"x": 3, "y": 2.5, "z": [1]}
    assert type(got.outputs["x"]) is int and type(got.outputs["y"]) is float


def _modes(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): stat.S_IMODE(p.lstat().st_mode) for p in [root, *root.rglob("*")]}


@pytest.mark.logic
def test_committed_entry_is_read_only(tmp_path: Path) -> None:
    """Every file and the directory of a committed entry carry no write bit for anyone."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": numpy.ones(3, "<f4"), "n": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    members = sorted(entry.rglob("*"))
    assert len(members) == 3
    for p in [entry, *members]:
        assert not (p.stat().st_mode & 0o222), p
    assert {stat.S_IMODE(p.stat().st_mode) for p in members} == {0o440}
    assert stat.S_IMODE(entry.stat().st_mode) & 0o777 == 0o550


@pytest.mark.logic
@pytest.mark.parametrize("umask", [0o077, 0o022, 0o000])
def test_umask_077_gives_no_world_bits(tmp_path: Path, umask: int) -> None:
    """Whatever the umask, nothing the store creates grants anything to others unless the policy says so."""
    root = tmp_path / "s"
    old = os.umask(umask)
    try:
        s, k, c = ResultStore(root), _key(), _ctx()
        s.put(k, {"y": numpy.ones(3, "<f4")}, c)
        s.put(k, {"y": 1}, _ctx(platform="macos"))
    finally:
        os.umask(old)
    created = {name: mode for name, mode in _modes(root).items() if name != "."}
    assert created and all(mode & 0o007 == 0 for mode in created.values()), created
    dirs = [p for p in root.rglob("*") if p.is_dir() and p.parent.name != c.fingerprint()]
    entries = {_entry_dir(root, k, c), _entry_dir(root, k, _ctx(platform="macos"))}
    for d in dirs:
        expected = 0o2550 if d in entries else 0o2770
        assert stat.S_IMODE(d.stat().st_mode) & ~stat.S_ISGID == expected & ~stat.S_ISGID, d


@pytest.mark.logic
def test_world_readable_policy_adds_only_world_read(tmp_path: Path) -> None:
    """A world-readable policy adds read (and search on directories) for others, and no write."""
    policy = StorePolicy(world_readable=True, quarantine_retention_s=60, staging_stale_s=60, collect_interval_s=60)
    old = os.umask(0o077)
    try:
        s, k, c = ResultStore(tmp_path / "s", policy=policy), _key(), _ctx()
        s.put(k, {"y": numpy.ones(3, "<f4")}, c)
    finally:
        os.umask(old)
    entry = _entry_dir(tmp_path / "s", k, c)
    assert {stat.S_IMODE(p.stat().st_mode) for p in entry.iterdir()} == {0o444}
    assert stat.S_IMODE(entry.stat().st_mode) & 0o777 == 0o555
    assert all(stat.S_IMODE(p.stat().st_mode) & 0o002 == 0 for p in (tmp_path / "s").rglob("*"))


def _swap_entry(src: Path, dst: Path) -> None:
    """Replace the entry at `dst` with a copy of the one at `src` (an rsync or restore mix-up)."""
    shutil.rmtree(_unseal(dst))
    shutil.copytree(src, dst)


@pytest.mark.logic
def test_a2_misfiled_entry_is_not_served(tmp_path: Path) -> None:
    """The probe: B's entry copied onto A's path was served for A, and put(A) returned B's identity."""
    s, c = ResultStore(tmp_path), _ctx()
    a, b = _key(audio="a" * 64), _key(audio="b" * 64)
    s.put(a, {"text": "transcript of A"}, c)
    b_identity = s.put(b, {"text": "transcript of B"}, c)
    _swap_entry(_entry_dir(tmp_path, b, c), _entry_dir(tmp_path, a, c))
    with pytest.raises(MisfiledEntryError):
        store_module._read_result(_entry_dir(tmp_path, a, c), digest=a.digest(), fingerprint=c.fingerprint())
    assert s.get(a, c, ReusePolicy.TOLERANT) is None
    a_identity = s.put(a, {"text": "transcript of A"}, c)
    assert a_identity != b_identity
    got = s.get(a, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"text": "transcript of A"}
    (moved,) = (tmp_path / QUARANTINE_DIR).iterdir()
    assert json.loads((moved / "result.json").read_bytes())["outputs"] == {"text": "transcript of B"}


@pytest.mark.logic
def test_a2_misfiled_context_is_not_served(tmp_path: Path) -> None:
    """An entry moved to another context's fingerprint directory is not served under that context."""
    s, k = ResultStore(tmp_path), _key()
    linux, mac = _ctx(platform="linux"), _ctx(platform="macos")
    s.put(k, {"y": "linux"}, linux)
    s.put(k, {"y": "mac"}, mac)
    _swap_entry(_entry_dir(tmp_path, k, linux), _entry_dir(tmp_path, k, mac))
    got = s.get(k, mac, ReusePolicy.TOLERANT)
    assert got is not None and got.outputs == {"y": "linux"} and got.policy_used is ReusePolicy.TOLERANT
    assert not _entry_dir(tmp_path, k, mac).exists()


@pytest.mark.logic
def test_forged_result_with_recomputed_identity_is_not_served(tmp_path: Path) -> None:
    """A rewritten result.json whose outputs, identity, key and key digest agree with each other but not the path."""
    s, c = ResultStore(tmp_path), _ctx()
    a, other = _key(audio="a" * 64), _key(audio="b" * 64)
    s.put(a, {"text": "real"}, c)
    entry = _entry_dir(tmp_path, a, c)

    def forge(r: dict[str, Any]) -> None:
        _forge(r, {"text": "FORGED"})
        r["key"] = _jcs_value(other)
        r["key_digest"] = other.digest()

    _rewrite_json(entry / "result.json", forge)
    with pytest.raises(MisfiledEntryError):
        store_module._read_result(entry, digest=a.digest(), fingerprint=c.fingerprint())
    assert s.get(a, c, ReusePolicy.EXACT) is None
    assert s.put(a, {"text": "real"}, c) == output_identity({"text": "real"})


def _jcs_value(key: LogicalKey) -> object:
    return json.loads(jcs_dumps(key.model_dump()))


def _raising_once(err: OSError, name: str) -> Callable[[Path], bytes]:
    real = Path.read_bytes
    state = {"raised": False}

    def read_bytes(self: Path) -> bytes:
        if self.name == name and not state["raised"]:
            state["raised"] = True
            raise err
        return real(self)

    return read_bytes


@pytest.mark.logic
@pytest.mark.parametrize("member", ["context.json", "result.json", "npy"])
@pytest.mark.parametrize("err", [errno.EACCES, errno.ESTALE, errno.EIO, errno.EMFILE])
def test_transient_error_never_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int, member: str
) -> None:
    """An I/O error while reading any member raises StoreUnavailable and leaves the entry in place, servable."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    outputs = {"y": numpy.ones(3, "<f4")}
    s.put(k, outputs, c)
    entry = _entry_dir(tmp_path, k, c)
    name = _only_npy_name(entry).name if member == "npy" else member
    monkeypatch.setattr(Path, "read_bytes", _raising_once(OSError(err, os.strerror(err)), name))
    with pytest.raises(StoreUnavailable) as caught:
        s.get(k, c, ReusePolicy.EXACT)
    assert caught.value.errno == err
    assert entry.is_dir() and not any((tmp_path / QUARANTINE_DIR).glob("*"))
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None
    numpy.testing.assert_array_equal(got.outputs["y"], outputs["y"])  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize("err", [errno.ESTALE, errno.EIO])
def test_a3_transient_error_in_put_never_quarantines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int) -> None:
    """The probe: one ESTALE moved a healthy entry into quarantine; a put that cannot read raises instead."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    ident = s.put(k, {"y": "healthy"}, c)
    monkeypatch.setattr(Path, "read_bytes", _raising_once(OSError(err, os.strerror(err)), "result.json"))
    with pytest.raises(StoreUnavailable):
        s.put(k, {"y": "healthy"}, c)
    assert not any((tmp_path / QUARANTINE_DIR).glob("*"))
    assert s.put(k, {"y": "other"}, c) == ident


@pytest.mark.logic
def test_put_verifies_the_arrays_of_an_existing_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A put over a valid committed entry reads every member, its arrays included, before returning its identity."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    ident = s.put(k, {"y": numpy.ones(1000, "<f8"), "z": numpy.zeros(4, "<i4")}, c)
    read: list[str] = []
    real = Path.read_bytes

    def spy(self: Path) -> bytes:
        read.append(self.name)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    assert s.put(k, {"y": numpy.ones(1000, "<f8"), "z": numpy.zeros(4, "<i4")}, c) == ident
    assert sorted(read)[-2:] == ["context.json", "result.json"] and len([n for n in read if n.endswith(".npy")]) == 2


def _heal_during_quarantine(root: str, detected: MpEvent, go: MpEvent, out: "MpQueue[Any]") -> None:
    """Find the corrupt entry, then pause before moving it until the parent has healed the address."""
    real = store_module.ResultStore._quarantine

    def paused(self: ResultStore, entry: Path, *, digest: str, fingerprint: str) -> None:
        detected.set()
        assert go.wait(60)
        real(self, entry, digest=digest, fingerprint=fingerprint)

    store_module.ResultStore._quarantine = paused  # type: ignore[method-assign]
    s = ResultStore(Path(root))
    s.get(_key(), _ctx(), ReusePolicy.EXACT)
    out.put([a.parameters for a in s.provenance.activities(node="store.quarantine")])


@pytest.mark.logic
def test_quarantine_restores_an_entry_healed_in_between(tmp_path: Path) -> None:
    """A reader that found a corrupt entry, then lost the race to a writer that replaced it, restores the new one."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"v": "v1"}, c)
    entry = _entry_dir(tmp_path, k, c)
    (_unseal(entry) / "context.json").write_bytes(b"torn")
    ctx = mp.get_context("spawn")
    detected, go, out = ctx.Event(), ctx.Event(), ctx.Queue()
    reader = ctx.Process(target=_heal_during_quarantine, args=(str(tmp_path), detected, go, out))
    reader.start()
    try:
        assert detected.wait(120)
        healed = s.put(k, {"v": "v2"}, c)
    finally:
        go.set()
    (activity,) = out.get(timeout=120)
    reader.join(60)
    assert reader.exitcode == 0
    assert activity["verdict"] == "verified" and activity["outcome"] == "restored"
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == healed and got.outputs == {"v": "v2"}
    assert stat.S_IMODE(entry.stat().st_mode) & 0o222 == 0


@pytest.mark.logic
def test_quarantine_keeps_the_copy_when_the_address_was_retaken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified copy is not renamed over an entry committed after it was moved; it stays in quarantine."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"v": "v1"}, c)
    entry = _entry_dir(tmp_path, k, c)
    real = store_module.ResultStore._verdict

    def commit_meanwhile(moved: Path, is_dir: bool, *, digest: str, fingerprint: str) -> tuple[str, str]:
        ResultStore(tmp_path).put(k, {"v": "v2"}, c)
        return real(moved, is_dir, digest=digest, fingerprint=fingerprint)

    monkeypatch.setattr(store_module.ResultStore, "_verdict", staticmethod(commit_meanwhile))
    s._quarantine(entry, digest=k.digest(), fingerprint=c.fingerprint())
    (activity,) = s.provenance.activities(node="store.quarantine")
    assert activity.parameters["outcome"] == "superseded"
    got = ResultStore(tmp_path).get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"v": "v2"}
    assert len(list((tmp_path / QUARANTINE_DIR).iterdir())) == 1


@pytest.mark.logic
def test_quarantine_records_reason(tmp_path: Path) -> None:
    """Quarantining a corrupt entry records the verification failure in provenance."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    CORRUPTIONS["result_garbled"](_entry_dir(tmp_path, k, c))
    assert s.get(k, c, ReusePolicy.EXACT) is None
    (activity,) = s.provenance.activities(node="store.quarantine")
    assert activity.parameters["outcome"] == "quarantined" and activity.parameters["verdict"] == "corrupt"
    assert "result.json" in activity.parameters["reason"] or "JSON" in activity.parameters["reason"]


@pytest.mark.logic
def test_collect_removes_only_aged_quarantine(tmp_path: Path) -> None:
    """`collect` removes quarantined entries and staging past their ages, keeps younger ones and foreign names."""
    policy = StorePolicy(world_readable=False, quarantine_retention_s=1000, staging_stale_s=100, collect_interval_s=60)
    s, c = ResultStore(tmp_path, policy=policy), _ctx()
    for audio in ("a", "b"):
        k = _key(audio=audio * 64)
        s.put(k, {"y": 1}, c)
        CORRUPTIONS["result_garbled"](_entry_dir(tmp_path, k, c))
        assert s.get(k, c, ReusePolicy.EXACT) is None
    old_q, young_q = sorted((tmp_path / QUARANTINE_DIR).iterdir(), key=lambda p: p.stat().st_mtime)
    stale, live = tmp_path / "tmp" / ("5" * 32), tmp_path / "tmp" / ("6" * 32)
    stale.mkdir()
    live.mkdir()
    foreign = tmp_path / QUARANTINE_DIR / "keep-me"
    foreign.write_bytes(b"x")
    now = time.time()
    for path, age in ((old_q, 1001), (young_q, 999), (stale, 101), (live, 99), (foreign, 10**6)):
        os.utime(path, (now - age, now - age), follow_symlinks=False)
    assert s.collect(now=now) == 2
    assert {p.name for p in (tmp_path / QUARANTINE_DIR).iterdir()} == {young_q.name, "keep-me"}
    assert not stale.exists() and live.exists()


@pytest.mark.logic
def test_a17_quarantine_is_collected_by_put(tmp_path: Path) -> None:
    """The probe: 406 quarantine directories were never collected; put now collects the aged ones."""
    policy = StorePolicy(world_readable=False, quarantine_retention_s=60, staging_stale_s=60, collect_interval_s=60)
    s, k, c = ResultStore(tmp_path, policy=policy), _key(), _ctx()
    for _ in range(5):
        s.put(k, {"y": 1}, c)
        CORRUPTIONS["result_garbled"](_entry_dir(tmp_path, k, c))
        assert s.get(k, c, ReusePolicy.EXACT) is None
    quarantined = list((tmp_path / QUARANTINE_DIR).iterdir())
    assert len(quarantined) == 5
    for path in quarantined:
        os.utime(path, (0, 0))
    ResultStore(tmp_path, policy=policy).put(_key(task="other"), {"y": 1}, c)
    assert list((tmp_path / QUARANTINE_DIR).iterdir()) == []


@pytest.mark.logic
def test_staging_age_uses_the_storage_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local clock running a day ahead does not make a live staging directory look stale."""
    s, c = ResultStore(tmp_path), _ctx()
    s.put(_key(), {"y": 1}, c)
    live = tmp_path / "tmp" / ("1" * 32)
    live.mkdir()
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 2 * StorePolicy.load_default().staging_stale_s)
    ResultStore(tmp_path).put(_key(task="other"), {"y": 1}, c)
    assert live.is_dir()


@pytest.mark.logic
@pytest.mark.parametrize("stray", ["file", "dangling_symlink", "fifo"])
def test_stray_file_at_entry_is_corrupt_not_fatal(tmp_path: Path, stray: str) -> None:
    """A non-directory at an entry path is quarantined; put commits and get serves the real entry."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    entry = _entry_dir(tmp_path, k, c)
    entry.parent.mkdir(parents=True)
    if stray == "file":
        entry.write_text("stray")
    elif stray == "dangling_symlink":
        entry.symlink_to(tmp_path / "nowhere")
    else:
        os.mkfifo(entry)
    assert s.get(k, c, ReusePolicy.EXACT) is None
    ident = s.put(k, {"y": 1}, c)
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.output_identity == ident
    (moved,) = (tmp_path / QUARANTINE_DIR).iterdir()
    assert not moved.is_dir() or moved.is_symlink()


@pytest.mark.logic
@pytest.mark.parametrize("err", [errno.ENOSPC, errno.EDQUOT])
def test_enospc_is_storage_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, err: int) -> None:
    """Running out of space while staging is StorageFull; no entry and no staging directory remain."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    real = store_module._write_synced
    calls = {"n": 0}

    def full(*args: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(err, os.strerror(err))
        real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(store_module, "_write_synced", full)
    with pytest.raises(StorageFull) as caught:
        s.put(k, {"y": numpy.ones(3, "<f4")}, c)
    assert caught.value.errno == err and "space" in str(caught.value)
    assert _entries(tmp_path) == []
    assert list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.logic
def test_a_file_at_the_quarantine_is_set_aside_and_the_quarantine_rebuilt(tmp_path: Path) -> None:
    """A file where the quarantine belongs is moved aside under the root; the corrupt entry is quarantined."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS["result_garbled"](entry)
    (tmp_path / QUARANTINE_DIR).write_bytes(b"not a directory")
    assert s.get(k, c, ReusePolicy.EXACT) is None
    assert s.put(k, {"y": 2}, c) == output_identity({"y": 2})
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 2}
    assert [p for p in entry.parent.iterdir() if p != entry] == []
    assert len(list((tmp_path / QUARANTINE_DIR).iterdir())) == 1
    (stray,) = [p for p in tmp_path.iterdir() if UUID_HEX.fullmatch(p.name)]
    assert stray.read_bytes() == b"not a directory"


@pytest.mark.logic
def test_unwritable_quarantine_never_raises_a_raw_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the quarantine cannot be opened, the corrupt entry is set aside in place and the key stays usable."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS["result_garbled"](entry)
    real = storage_module.StorageDir.child

    def child(self: storage_module.StorageDir, name: str, **kwargs: object) -> storage_module.StorageDir:
        if name == QUARANTINE_DIR:
            raise PermissionError(errno.EACCES, "Permission denied", name)
        return real(self, name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(storage_module.StorageDir, "child", child)
    assert s.get(k, c, ReusePolicy.EXACT) is None
    assert s.put(k, {"y": 2}, c) == output_identity({"y": 2})
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 2}
    (aside,) = [p for p in entry.parent.iterdir() if p != entry]
    assert UUID_HEX.fullmatch(aside.name)


def _refuse_chmod_on(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Refuse chmod on `target` with EPERM, as for an entry another user owns."""
    real = storage_module.set_mode

    def set_mode(path: Path | str, mode: int, *, fd: int | None = None) -> None:
        if Path(path) == target:
            raise PermissionError(errno.EPERM, "Operation not permitted", str(path))
        real(path, mode, fd=fd)

    monkeypatch.setattr(storage_module, "set_mode", set_mode)


@pytest.mark.logic
def test_corrupt_entry_of_another_user_does_not_wedge_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt entry that cannot be chmod-ed is renamed aside within its parent; put then commits the key."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS["result_garbled"](entry)
    _seal_entry(entry)
    _refuse_chmod_on(monkeypatch, entry)
    assert s.put(k, {"y": 2}, c) == output_identity({"y": 2})
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 2}
    (activity,) = s.provenance.activities(node="store.quarantine")
    assert activity.parameters["outcome"] == "set aside in place"
    aside = tmp_path / activity.parameters["moved_to"]
    assert aside.parent == entry.parent and UUID_HEX.fullmatch(aside.name)
    assert stat.S_IMODE(aside.stat().st_mode) & 0o222 == 0
    assert not (tmp_path / QUARANTINE_DIR).exists() or list((tmp_path / QUARANTINE_DIR).iterdir()) == []


@pytest.mark.contract
@settings(max_examples=30, deadline=None)
@given(
    ages=st.lists(st.tuples(st.sampled_from(["uuid", "fingerprint", "other"]), st.integers(0, 2000)), max_size=6),
    sealed=st.booleans(),
)
def test_collect_removes_exactly_the_aged_entries_set_aside_in_place(ages: list[tuple[str, int]], sealed: bool) -> None:
    """Under every key, `collect` removes the uuid-named children past the quarantine age, sealed or not, only."""
    policy = StorePolicy(world_readable=False, quarantine_retention_s=1000, staging_stale_s=100, collect_interval_s=60)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        s, k, c = ResultStore(root, policy=policy), _key(), _ctx()
        s.put(k, {"y": 1}, c)
        entry = _entry_dir(root, k, c)
        base = entry.parent
        now = time.time()
        expected = {entry.name}
        for i, (kind, age) in enumerate(ages):
            name = {"uuid": f"{i:032x}", "fingerprint": f"{i:064x}", "other": f"keep-{i}"}[kind]
            child = base / name
            child.mkdir()
            (child / "result.json").write_bytes(b"{}")
            if sealed:
                child.chmod(0o550)
            os.utime(child, (now - age, now - age))
            if not (kind == "uuid" and age > 1000):
                expected.add(name)
        os.utime(entry, (now - 10**6, now - 10**6))
        s.collect(now=now)
        assert {p.name for p in base.iterdir()} == expected
        for p in base.iterdir():
            p.chmod(0o750)


@pytest.mark.logic
def test_entry_set_aside_in_place_is_collected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt entry renamed aside beside its key, read-only, is removed by `collect` once past the quarantine age."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS["result_garbled"](entry)
    _seal_entry(entry)
    _refuse_chmod_on(monkeypatch, entry)
    s.put(k, {"y": 2}, c)
    (aside,) = [p for p in entry.parent.iterdir() if p != entry]
    assert s.collect(now=time.time() + s.policy.quarantine_retention_s + 1) == 1
    assert [p.name for p in entry.parent.iterdir()] == [entry.name]


@pytest.mark.logic
def test_unmovable_corrupt_entry_names_owner_and_remedy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the corrupt entry cannot be moved at all, the error names the entry, its owner and the command to run."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    CORRUPTIONS["result_garbled"](entry)
    _refuse_chmod_on(monkeypatch, entry)
    real_rename = os.rename

    def rename(src: object, dst: object, **dir_fds: int) -> None:
        if str(src) == entry.name:
            raise PermissionError(errno.EACCES, "Permission denied", str(src))
        real_rename(src, dst, **dir_fds)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(StoreUnavailable) as caught:
        s.put(k, {"y": 2}, c)
    owner = pwd.getpwuid(os.getuid()).pw_name
    message = str(caught.value)
    assert str(entry) in message and f"owned by {owner}" in message and "rm -rf" in message
    assert entry.is_dir()


def _seal_entry(entry: Path) -> None:
    for member in entry.iterdir():
        member.chmod(0o440)
    entry.chmod(0o550)


@pytest.mark.logic
@pytest.mark.parametrize("reader", ["put", "get"])
def test_writable_committed_entry_is_resealed(tmp_path: Path, reader: str) -> None:
    """An entry left writable by a writer that died between rename and seal is sealed when next found valid."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": 1}, c)
    entry = _entry_dir(tmp_path, k, c)
    entry.chmod(0o2770)
    if reader == "put":
        s.put(k, {"y": 1}, c)
    else:
        assert s.get(k, c, ReusePolicy.EXACT) is not None
    assert stat.S_IMODE(entry.stat().st_mode) & 0o222 == 0


@pytest.mark.logic
def test_put_collects_at_most_once_per_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Puts within `collect_interval_s` of each other run one collection, not one each."""
    s, c = ResultStore(tmp_path), _ctx()
    calls: list[object] = []
    real = ResultStore.collect

    def spy(self: ResultStore, now: float | None = None) -> int:
        calls.append(now)
        return real(self, now)

    monkeypatch.setattr(ResultStore, "collect", spy)
    for i in range(5):
        s.put(_key(audio=f"{i:064x}"), {"y": i}, c)
    assert len(calls) == 1
    clock = time.monotonic() + s.policy.collect_interval_s
    monkeypatch.setattr(time, "monotonic", lambda: clock)
    s.put(_key(task="other"), {"y": 1}, c)
    assert len(calls) == 2


@pytest.mark.logic
def test_storage_fault_is_the_one_mapping() -> None:
    """Every OSError maps to StoreUnavailable with its errno; ENOSPC and EDQUOT to StorageFull; faults pass through."""
    from senselab_core.runtime.storage import storage_fault

    for code in (errno.EACCES, errno.EIO, errno.ESTALE, errno.EMFILE, errno.ENOENT, errno.EROFS):
        fault = storage_fault(OSError(code, os.strerror(code)), Path("/x"))
        assert type(fault) is StoreUnavailable and fault.errno == code and fault.filename == "/x"
    for code in (errno.ENOSPC, errno.EDQUOT):
        assert type(storage_fault(OSError(code, "full"), Path("/x"))) is StorageFull
    fault = StoreUnavailable(errno.EIO, "x")
    assert storage_fault(fault, Path("/y")) is fault


@pytest.mark.logic
def test_a35_participant_outputs_are_not_world_readable_under_umask_077(tmp_path: Path) -> None:
    """The probe: under umask 077 result.json came out 0664 in a 2775 directory, readable by every user."""
    old = os.umask(0o077)
    try:
        s, k, c = ResultStore(tmp_path / "s"), _key(), _ctx()
        s.put(k, {"y": "private"}, c)
    finally:
        os.umask(old)
    result = _entry_dir(tmp_path / "s", k, c) / "result.json"
    assert stat.S_IMODE(result.stat().st_mode) & 0o007 == 0
    assert stat.S_IMODE(result.parent.stat().st_mode) & 0o007 == 0


@pytest.mark.logic
@pytest.mark.parametrize("stray", ["file", "dangling_symlink"])
def test_a36_stray_file_at_entry_does_not_block_put(tmp_path: Path, stray: str) -> None:
    """The probe: a stray file or dangling symlink at the entry path made every put raise NotADirectoryError."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    entry = _entry_dir(tmp_path, k, c)
    entry.parent.mkdir(parents=True)
    if stray == "file":
        entry.write_text("stray")
    else:
        os.symlink("/nonexistent/x", entry)
    for _ in range(2):
        assert s.put(k, {"y": 1}, c) == output_identity({"y": 1})
    assert entry.is_dir() and not entry.is_symlink()


@pytest.mark.logic
def test_a36_enospc_mid_entry_is_storage_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe: ENOSPC on the second staged file surfaced as a raw OSError; it is StorageFull, nothing left behind."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    real = store_module._write_synced
    calls = {"n": 0}

    def boom(*args: object) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(store_module, "_write_synced", boom)
    with pytest.raises(StorageFull):
        s.put(k, {"a": "x"}, c)
    assert _entries(tmp_path) == [] and list((tmp_path / "tmp").iterdir()) == []


@pytest.mark.contract
@pytest.mark.parametrize("shape", [(), (1,), (2, 2)])
def test_stored_bool_arrays_keep_their_shape_and_truth(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """A bool array holding byte 2 reads back from the store at its shape, as 0/1, and 0-d is not (1,)."""
    count = int(numpy.prod(shape))
    a = numpy.frombuffer(bytes([2, 0, 255, 1][:count]), dtype=numpy.bool_).reshape(shape)
    s, c = ResultStore(tmp_path), _ctx()
    s.put(_key(), {"y": a}, c)
    got = s.get(_key(), c, ReusePolicy.EXACT)
    assert got is not None
    y = got.outputs["y"]
    assert isinstance(y, numpy.ndarray) and y.shape == shape and numpy.array_equal(y, a)
    assert y.view(numpy.uint8).max(initial=0) <= 1
    if shape == ():
        assert output_identity({"y": a}) != output_identity({"y": a.reshape(1)})
