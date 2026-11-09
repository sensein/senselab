"""`ResultStore.admits`: the one servable predicate, shared by `execute`'s serving and by probing."""

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from planted_objects_fixture import entry_dir, rewrite_json, store_context, store_key, unseal
from senselab_core.kernel.declare import BackendSpec, Env, TaskSpec
from senselab_core.kernel.identity import ReusePolicy, member_key
from senselab_core.kernel.license import License
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.job import Job, _serve
from senselab_core.runtime.fake_backend import EchoParams
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.licenses import LicensePolicy, license_record
from senselab_core.runtime.servable import Servable, StoredContext, choose
from senselab_core.runtime.store import ResultStore, StoreFormatRetired

TASK = TaskSpec(
    name="servable.t", version="1", kind="transform", inputs={"text": "json"}, outputs={"v": "json"}, params=EchoParams
)
BACKEND = BackendSpec(
    task=TASK.name,
    name="hf",
    version="1",
    stack="hf",
    env=Env.host(),
    devices=("cpu",),
    batch_invariant=True,
    worker="x",
    load_stub={},
)
PLACEMENT = Placement(backend=BACKEND, backend_params=BACKEND.params(), device="cpu")
JOB = Job(JobRef(plan_hash="0" * 64, node="n", element=()), Call(TASK, EchoParams(), {}), {})
MIT = license_record("MIT", source="lock", verified={"date": "2026-11-04", "how": "manual"}, gated=False)
NC = license_record("CC-BY-NC-4.0", source="lock", verified={"date": "2026-11-04", "how": "manual"}, gated=False)
POLICIES = (None, LicensePolicy(refuse=frozenset({"CC-BY-NC-4.0"})), LicensePolicy(allow_unlicensed_hits=True))


def _ctx(platform: str = "linux", device: str = "cpu", licence: Any = MIT) -> Any:  # noqa: ANN401
    return store_context(platform=platform, device=device).model_copy(update={"env_license": licence})


def _admits(store: ResultStore, context: Any, policy: ReusePolicy, run: RunConfig) -> Any:  # noqa: ANN401
    """`admits` for the default key under `policy`; `policy` replaces `run.reuse`."""
    key = store_key()
    return store.admits(key.digest(), context, replace(run, reuse=policy), backend=key.backend)


@pytest.mark.logic
def test_entries_under_another_context_are_probe_candidates(tmp_path: Path) -> None:
    """An entry under an old fingerprint that is no equivalence member is not admitted, though its directory exists."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx(platform="linux"))
    assert _admits(store, _ctx(platform="linux"), ReusePolicy.EQUIVALENT, RunConfig()) is not None
    assert _admits(store, _ctx(platform="macos"), ReusePolicy.EQUIVALENT, RunConfig()) is None


@pytest.mark.logic
def test_an_equivalence_member_is_admitted_under_equivalent(tmp_path: Path) -> None:
    """A measured class admits its other member, naming the policy used."""
    linux, mac = _ctx(device="cpu"), _ctx(device="mps")
    table = {
        "hf": {"c": frozenset({member_key(linux.numerics, linux.env_lock), member_key(mac.numerics, mac.env_lock)})}
    }
    store = ResultStore(tmp_path, equivalence=table)
    store.put(store_key(), {"v": 1}, linux)
    got = _admits(store, mac, ReusePolicy.EQUIVALENT, RunConfig())
    assert got is not None and got.policy_used is ReusePolicy.EQUIVALENT and got.fingerprint == linux.fingerprint()


@pytest.mark.logic
def test_an_unlicensed_entry_is_not_admitted_when_hits_require_a_license(tmp_path: Path) -> None:
    """A hit recording no env license is admitted only where the policy allows unlicensed hits."""
    store = ResultStore(tmp_path)
    bare = _ctx(licence=None)
    store.put(store_key(), {"v": 1}, bare)
    assert _admits(store, bare, ReusePolicy.EXACT, RunConfig(licenses=LicensePolicy())) is None
    assert _admits(store, bare, ReusePolicy.EXACT, RunConfig()) is None
    assert _admits(store, bare, ReusePolicy.EXACT, RunConfig(licenses=LicensePolicy(allow_unlicensed_hits=True)))
    other = store_key(task="other")
    store.put(other, {"v": 1}, _ctx(licence=NC))
    refusing = RunConfig(reuse=ReusePolicy.EXACT, licenses=LicensePolicy(refuse=frozenset({"CC-BY-NC-4.0"})))
    assert store.admits(other.digest(), _ctx(licence=NC), refusing, backend="hf") is None


@pytest.mark.logic
def test_a_verify_sampled_entry_is_not_admitted(tmp_path: Path) -> None:
    """A cross-context hit that verify sampling recomputes is not admitted; an exact hit never is sampled."""
    store = ResultStore(tmp_path)
    store.put(store_key(), {"v": 1}, _ctx(platform="linux"))
    sampled_run = RunConfig(verify=1.0)
    assert _admits(store, _ctx(platform="macos"), ReusePolicy.TOLERANT, sampled_run) is None
    assert _admits(store, _ctx(platform="macos"), ReusePolicy.TOLERANT, RunConfig(verify=0.0)) is not None
    assert _admits(store, _ctx(platform="linux"), ReusePolicy.TOLERANT, sampled_run) is not None


@pytest.mark.logic
def test_the_policy_is_the_runs_and_a_run_is_required(tmp_path: Path) -> None:
    """`admits` reads `run.reuse` and takes no second policy or absent run."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx(platform="linux"))
    assert store.admits(key.digest(), _ctx(platform="macos"), RunConfig(reuse=ReusePolicy.TOLERANT), backend="hf")
    assert store.admits(key.digest(), _ctx(platform="macos"), RunConfig(reuse=ReusePolicy.EXACT), backend="hf") is None
    with pytest.raises(TypeError):
        store.admits(key.digest(), _ctx(), "exact", None, backend="hf")  # type: ignore[misc, arg-type]


def _plant(tmp_path: Path, context: Any, fault: str) -> None:  # noqa: ANN401
    entry = entry_dir(tmp_path, store_key(), context)
    match fault:
        case "context_missing":
            unseal(entry)
            (entry / "context.json").unlink()
        case "result_corrupt":
            unseal(entry)
            (entry / "result.json").write_bytes(b"not json")
        case "retired_context":
            rewrite_json(entry / "context.json", lambda r: r.update(format="execution-context@1.0"))
        case "retired_result":
            rewrite_json(entry / "result.json", lambda r: r.update(format="result@1.0"))


@pytest.mark.logic
def test_a_corrupt_result_is_admitted_until_a_read_finds_it(tmp_path: Path) -> None:
    """A probe reads contexts only: it admits an entry whose result is corrupt; serving finds and sets it aside."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx())
    _plant(tmp_path, _ctx(), "result_corrupt")
    run = RunConfig()
    assert store.admits(key.digest(), _ctx(), run, backend="hf") is not None
    assert store.fetch(key, _ctx(), run) == (None, None)
    assert store.admits(key.digest(), _ctx(), run, backend="hf") is None


@pytest.mark.logic
def test_serving_falls_to_the_entry_admits_then_selects(tmp_path: Path) -> None:
    """A corrupt exact entry is set aside, and the next admitted entry is the one served and the one admitted."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx(platform="linux"))
    store.put(key, {"v": 2}, _ctx(platform="macos"))
    _plant(tmp_path, _ctx(platform="macos"), "result_corrupt")
    run = RunConfig(reuse=ReusePolicy.TOLERANT)
    hit, verdict = store.fetch(key, _ctx(platform="macos"), run)
    assert hit is not None and hit.context.platform == "linux" and verdict is not None and verdict.value == "serve"
    admitted = store.admits(key.digest(), _ctx(platform="macos"), run, backend="hf")
    assert admitted is not None and admitted.fingerprint == hit.context.fingerprint()


@pytest.mark.logic
def test_admits_raises_for_a_key_holding_only_a_retired_entry(tmp_path: Path) -> None:
    """A probe of a key whose only entry is retired raises what `get` raises."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx())
    _plant(tmp_path, _ctx(), "retired_context")
    with pytest.raises(StoreFormatRetired):
        store.get(key, _ctx(), ReusePolicy.EXACT)
    with pytest.raises(StoreFormatRetired):
        store.admits(key.digest(), _ctx(), RunConfig(), backend="hf")


@pytest.mark.logic
def test_choose_takes_the_verdict_and_no_run() -> None:
    """`choose` returns the first entry not excluded with the verdict it is given."""
    a, b = (StoredContext(f, _ctx(), ReusePolicy.EXACT) for f in ("a", "b"))
    assert choose([a, b], lambda _: Servable.REFUSE, {"a"}) == (b, Servable.REFUSE)
    assert choose([a, b], lambda _: Servable.SERVE, {"a", "b"}) is None


@pytest.mark.logic
def test_a_retired_result_beside_a_current_context_is_set_aside(tmp_path: Path) -> None:
    """A retired result member under a current context is an altered entry: set aside, then `admits` agrees."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx())
    _plant(tmp_path, _ctx(), "retired_result")
    run = RunConfig()
    assert store.fetch(key, _ctx(), run) == (None, None)
    assert store.admits(key.digest(), _ctx(), run, backend="hf") is None
    assert not entry_dir(tmp_path, key, _ctx()).exists()


@pytest.mark.logic
def test_a_skipped_entry_does_not_rerank_the_key(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Serving lists and reads the key's contexts once, however many entries a read sets aside."""
    store, key = ResultStore(tmp_path), store_key()
    for platform in ("linux", "macos", "windows"):
        store.put(key, {"v": 1}, _ctx(platform=platform))
    rewrite_json(
        entry_dir(tmp_path, key, _ctx(platform="windows")) / "context.json",
        lambda r: r.update(format="execution-context@5.0"),
    )
    _plant(tmp_path, _ctx(platform="macos"), "result_corrupt")
    with caplog.at_level("INFO", logger="senselab"):
        hit, _ = store.fetch(key, _ctx(platform="macos"), RunConfig(reuse=ReusePolicy.TOLERANT))
    assert hit is not None and hit.context.platform == "linux"
    assert sum("another version's" in r.getMessage() for r in caplog.records) == 1


def _outcome(call: Any, fingerprint: Any) -> Any:  # noqa: ANN401
    try:
        got = call()
    except StoreFormatRetired:
        return "retired"
    return None if got is None else fingerprint(got)


POOL = tuple((p, d) for p in ("linux", "macos") for d in ("cpu", "mps"))
LICENCES = st.sampled_from([None, MIT, NC])


@pytest.mark.logic
@settings(max_examples=60, deadline=None)
@given(
    stored=st.dictionaries(st.sampled_from(POOL), LICENCES, max_size=len(POOL)),
    current=st.tuples(st.sampled_from(POOL), LICENCES),
    members=st.sets(st.sampled_from(POOL)),
    fault=st.sampled_from(["none", "context_missing", "result_corrupt", "retired_context", "retired_result"]),
    victim=st.integers(min_value=0, max_value=len(POOL) - 1),
    policy=st.sampled_from(list(ReusePolicy)),
    verify=st.sampled_from([0.0, 0.5, 1.0]),
    licenses=st.sampled_from(POLICIES),
)
def test_admits_agrees_with_serve(  # noqa: PLR0913
    stored: dict[tuple[str, str], License | None],
    current: tuple[tuple[str, str], License | None],
    members: set[tuple[str, str]],
    fault: str,
    victim: int,
    policy: ReusePolicy,
    verify: float,
    licenses: Any,  # noqa: ANN401
) -> None:
    """`admits` selects the entry `_serve` serves, or both refuse, or both raise, over every store state.

    An integrity failure of a result member no read has found yet is the one exception: it is admitted until
    served once.
    """
    with tempfile.TemporaryDirectory() as root:
        contexts = {k: _ctx(*k, licence) for k, licence in stored.items()}
        numerics = [_ctx(*k) for k in members]
        table = {"hf": {"c": frozenset(member_key(c.numerics, c.env_lock) for c in numerics)}} if members else {}
        store, key = ResultStore(Path(root), equivalence=table), store_key()
        for c in contexts.values():
            store.put(key, {"v": 1}, c)
        if contexts and fault != "none":
            _plant(Path(root), list(contexts.values())[victim % len(contexts)], fault)
        run = RunConfig(reuse=policy, verify=verify, licenses=licenses)
        context = _ctx(*current[0], current[1])

        def probe() -> Any:  # noqa: ANN401
            return _outcome(lambda: store.admits(key.digest(), context, run, backend="hf"), lambda e: e.fingerprint)

        def serve() -> Any:  # noqa: ANN401
            return _outcome(
                lambda: _serve(store, key, context, run, JOB, PLACEMENT)[0], lambda r: r.context.fingerprint()
            )

        before, first = probe(), serve()
        if fault not in {"result_corrupt", "retired_result"}:
            assert before == first
        assert probe() == serve()
