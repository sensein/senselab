"""`ResultStore.admits`: the one servable predicate, shared by `execute`'s serving and by probing."""

import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from planted_objects_fixture import store_context, store_key
from senselab_core.kernel.declare import BackendSpec, Env, TaskSpec
from senselab_core.kernel.identity import ReusePolicy, member_key
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.execution.executor import JobRef
from senselab_core.runtime.execution.job import Job, _serve
from senselab_core.runtime.fake_backend import EchoParams
from senselab_core.runtime.keys import Call, Placement
from senselab_core.runtime.licenses import LicensePolicy, license_record
from senselab_core.runtime.store import ResultStore

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
    key = store_key()
    return store.admits(key.digest(), context, policy, run, backend=key.backend)


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
    store.put(store_key(task="other"), {"v": 1}, _ctx(licence=NC))
    refusing = RunConfig(licenses=LicensePolicy(refuse=frozenset({"CC-BY-NC-4.0"})))
    other = store_key(task="other")
    assert store.admits(other.digest(), _ctx(licence=NC), "exact", refusing, backend="hf") is None


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
def test_admits_reads_no_outputs(tmp_path: Path) -> None:
    """A planted unreadable result member does not affect `admits`, which reads `context.json` only."""
    store, key = ResultStore(tmp_path), store_key()
    store.put(key, {"v": 1}, _ctx())
    (member,) = tmp_path.rglob("result.json")
    member.chmod(0o600)
    member.write_bytes(b"not json")
    assert _admits(store, _ctx(), ReusePolicy.EXACT, RunConfig()) is not None


@pytest.mark.logic
@settings(max_examples=60, deadline=None)
@given(
    stored=st.sets(
        st.tuples(
            st.sampled_from(["linux", "macos"]), st.sampled_from(["cpu", "mps"]), st.sampled_from([None, MIT, NC])
        ),
        max_size=4,
    ),
    current=st.tuples(
        st.sampled_from(["linux", "macos"]), st.sampled_from(["cpu", "mps"]), st.sampled_from([None, MIT, NC])
    ),
    policy=st.sampled_from(list(ReusePolicy)),
    verify=st.sampled_from([0.0, 0.5, 1.0]),
    licenses=st.sampled_from(POLICIES),
)
def test_admits_agrees_with_serve(stored: Any, current: Any, policy: ReusePolicy, verify: float, licenses: Any) -> None:  # noqa: ANN401
    """For any store state and context, `admits` is not None exactly when `_serve` serves."""
    with tempfile.TemporaryDirectory() as root:
        store, key = ResultStore(Path(root)), store_key()
        for platform, device, licence in sorted(stored, key=repr):
            store.put(key, {"v": 1}, _ctx(platform, device, licence))
        run = RunConfig(reuse=policy, verify=verify, licenses=licenses)
        context = _ctx(*current)
        served, _ = _serve(store, key, context, run, JOB, PLACEMENT)
        assert (served is not None) == (_admits(store, context, policy, run) is not None)
