"""The content-addressed result store: layout, reuse policy, concurrent writers and provenance."""

import json
import os
import time
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Barrier as MpBarrier
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ReusePolicy
from senselab_core.runtime import store as store_module
from senselab_core.runtime.provenance import CHECKSUM_KEY, ProvStore
from senselab_core.runtime.store import (
    STAGING_STALE_SECONDS,
    STORE_DIR,
    ResultStore,
    StoredResult,
    output_identity,
)
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.hashing import sha256_bytes


def _key(task: str = "asr", **inputs: str) -> LogicalKey:
    return LogicalKey(
        task=task,
        task_version="1",
        backend="hf",
        backend_version="1",
        model="0" * 40,
        params={"beam": 1},
        inputs=inputs or {"audio": "a" * 64},
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
    assert result["format"] == "result@1.0"
    assert result["output_identity"] == ident
    sha = sha256_bytes(canonical_npy_bytes(outputs["y"]))  # type: ignore[arg-type]
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
    import multiprocessing as mp

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
    import multiprocessing as mp

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

    def miss_first(entry: Path) -> str | None:
        calls.append(entry)
        return None if len(calls) == 1 else real(entry)

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
    stale = tmp_path / "tmp" / "crashed"
    stale.mkdir(parents=True)
    (stale / "result.json").write_bytes(jcs_dumps({"format": "result@1.0", "outputs": {}, "output_identity": "x"}))
    old = time.time() - STAGING_STALE_SECONDS - 60
    os.utime(stale, (old, old))
    live = tmp_path / "tmp" / "live"
    live.mkdir()
    assert s.get(k, c, ReusePolicy.TOLERANT) is None
    s.put(_key(task="other"), {"y": 1}, c)
    assert not stale.exists()
    os.utime(live, (old, old))
    s.put(_key(task="other"), {"y": 1}, c)
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
def test_output_identity_is_jcs_over_array_digests() -> None:
    """The identity is sha256 over the JCS of outputs with each array replaced by its .npy digest."""
    a = numpy.arange(4, dtype="<i8")
    expected = sha256_bytes(jcs_dumps({"y": {"$array": sha256_bytes(canonical_npy_bytes(a))}, "z": 1}))
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


@pytest.mark.logic
def test_corrupt_array_is_refused_not_served(tmp_path: Path) -> None:
    """An array file that no longer hashes to its name raises rather than being served."""
    s, k, c = ResultStore(tmp_path), _key(), _ctx()
    s.put(k, {"y": numpy.ones(3, "<f4")}, c)
    (npy,) = (tmp_path / STORE_DIR).rglob("*.npy")
    npy.write_bytes(canonical_npy_bytes(numpy.zeros(3, "<f4")))
    with pytest.raises(ValueError, match="corrupt"):
        s.get(k, c, ReusePolicy.EXACT)


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
    assert [(e.attributes["input"], e.attributes["identity"]) for e in used] == [("audio", "a" * 64)]
    (generated,) = [e for e in prov.entities("artifact") if prov.generated_by(e.id) == act.id]
    assert generated.attributes[CHECKSUM_KEY] == ident
    assert ResultStore(tmp_path).provenance.activities() == []
