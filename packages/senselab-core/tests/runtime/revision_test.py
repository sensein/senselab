"""Tests for run-scoped HF revision resolution."""

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lease_clock_fixture import ManualClock
from lease_time_fixture import install, on_create
from planted_objects_fixture import BAD_ATTEMPT_IDS, LEASE, NAME_BOUND_S, UNOPENABLE_NAMES, bounded
from senselab_core.runtime.locks import LeaseUnavailable
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.retry import RetryPolicy
from senselab_core.runtime.revision import (
    MANIFEST_NAME,
    NAMESPACE,
    Resolver,
    RevisionResolutionError,
    choose_attempt_id,
    get_cached_commit_hash,
    manifest_key,
    manifest_path,
    read_manifest,
    record_resolution,
    resolve_revision,
)
from senselab_core.runtime.storage import NAME_MAX, StorePolicy
from senselab_core.testing.leases import packaged_lease
from syscall_faults_fixture import displace_after_take, displace_before_proof, pause_replaces

FIRST_BACKOFF_S = 2.0
"""A backoff on the test clock, which sleeps it without waiting."""
SHA_A = "a" * 40
SHA_B = "b" * 40

HubLookup = Callable[[str, str, "str | None"], object]


@pytest.fixture
def offline(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> Callable[[HubLookup], Resolver]:
    """Build a fresh `Resolver` over an empty local HF cache and this test's cache root, asking `hub` on a miss."""

    def make(hub: HubLookup) -> Resolver:
        return Resolver(
            cache_dir=tmp_path / "hf-hub",
            hub=hub,
            cache_root=tmp_path / "senselab",
            retry=RetryPolicy(attempts=3, first_backoff_s=FIRST_BACKOFF_S),
            clock=lease_clock(),
        )

    return make


def _slept(resolver: Resolver) -> list[float]:
    """The waits a resolver's retry slept on its test clock."""
    assert isinstance(resolver.clock, ManualClock)
    return resolver.clock.slept


def _answers(sha: str) -> HubLookup:
    return lambda repo_id, ref, token: sha


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point SENSELAB_CACHE at a temp dir."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "senselab"))


_ATTEMPT_ID_TWICE = """
import json, os
from senselab_core.runtime.revision import attempt_id
print(json.dumps([attempt_id(), attempt_id(), os.environ["SENSELAB_RUN_ID"]]))
"""


def _attempt_id_in_a_fresh_process(**environ: str) -> list[str]:
    """`attempt_id()` twice and the exported `SENSELAB_RUN_ID`, in a fresh interpreter with run variables `environ`."""
    base = {k: v for k, v in os.environ.items() if k not in ("SENSELAB_RUN_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID")}
    proc = subprocess.run(
        [sys.executable, "-c", _ATTEMPT_ID_TWICE],
        env={**base, **environ},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return list(json.loads(proc.stdout))


@pytest.mark.logic
def test_attempt_id_is_generated_once_and_exported() -> None:
    """A generated attempt id is stable within the process and exported to the environment."""
    first, second, exported = _attempt_id_in_a_fresh_process()
    assert first == second, "attempt_id must be stable within a process"
    assert exported == first, "attempt id must be exported so subprocesses inherit it"


@pytest.mark.logic
def test_attempt_id_is_inherited_when_already_set() -> None:
    """A pre-set SENSELAB_RUN_ID is reused rather than replaced with a fresh UUID."""
    assert choose_attempt_id({"SENSELAB_RUN_ID": "job-12345"}) == "job-12345"


@pytest.mark.logic
def test_attempt_id_falls_back_to_the_slurm_job_id() -> None:
    """Without SENSELAB_RUN_ID, a Slurm job shares one manifest via its job id, not one per process."""
    assert choose_attempt_id({"SLURM_JOB_ID": "778899"}) == "778899"
    assert _attempt_id_in_a_fresh_process(SLURM_JOB_ID="778899") == ["778899"] * 3, (
        "the chosen id must be exported for subprocesses"
    )


@pytest.mark.logic
def test_attempt_id_prefers_the_array_job_id_so_an_array_shares_one_manifest() -> None:
    """SLURM_ARRAY_JOB_ID (shared by the whole array) wins over per-task SLURM_JOB_ID."""
    assert choose_attempt_id({"SLURM_ARRAY_JOB_ID": "112233", "SLURM_JOB_ID": "112240"}) == "112233"


@pytest.mark.logic
def test_explicit_attempt_id_wins_over_slurm() -> None:
    """An explicit SENSELAB_RUN_ID still takes precedence over any Slurm identity."""
    assert choose_attempt_id({"SENSELAB_RUN_ID": "explicit", "SLURM_ARRAY_JOB_ID": "112233"}) == "explicit"


@pytest.mark.logic
def test_a_full_sha_short_circuits_without_touching_the_manifest() -> None:
    """A ref that is already a full commit SHA resolves with no filesystem I/O."""
    assert resolve_revision("org/model", SHA_A) == SHA_A
    assert not manifest_path().exists(), "a 40-hex ref must resolve with no I/O at all"


@pytest.mark.logic
def test_record_then_read_round_trips() -> None:
    """A recorded resolution is readable back from the manifest."""
    record_resolution("org/model", "main", SHA_A)
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


@pytest.mark.logic
def test_manifest_key_is_injective() -> None:
    """`a@b` + `c` and `a` + `b@c` name different manifest entries."""
    assert manifest_key("a@b", "c") != manifest_key("a", "b@c")
    assert manifest_key("org/model", "main") == '["org/model","main"]'


_PART = st.text(alphabet='ab@\\"[],', max_size=4)


@pytest.mark.logic
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.tuples(_PART, _PART), st.tuples(_PART, _PART))
def test_manifest_key_is_injective_over_pairs(x: tuple[str, str], y: tuple[str, str]) -> None:
    """Two different (repo, ref) pairs never share a key, whatever separators and escapes they hold."""
    assert (manifest_key(*x) == manifest_key(*y)) == (x == y)


@pytest.mark.logic
def test_manifest_file_carries_its_key_format() -> None:
    """The manifest's name carries the format of its keys, so manifests keyed the old way are not read."""
    assert manifest_path().name == MANIFEST_NAME == "resolutions@2.json"


@pytest.mark.logic
def test_a_1x_manifest_under_the_cache_root_is_neither_read_nor_written(tmp_path: Path) -> None:
    """2.x manifests live under `2x/runs/`: a 1.x manifest at `runs/<run>/` is never read or touched."""
    cache = tmp_path / "senselab"
    alpha = cache / "runs" / "shared" / MANIFEST_NAME
    alpha.parent.mkdir(parents=True)
    alpha.write_text(json.dumps({manifest_key("org/model", "main"): SHA_B}))
    before = alpha.read_bytes()
    assert manifest_path("shared") == cache / NAMESPACE / "runs" / "shared" / MANIFEST_NAME
    assert read_manifest("shared") == {}
    assert record_resolution("org/model", "main", SHA_A, run="shared") == SHA_A
    assert read_manifest("shared") == {manifest_key("org/model", "main"): SHA_A}
    assert alpha.read_bytes() == before and sorted(p.name for p in alpha.parent.iterdir()) == [MANIFEST_NAME]


@pytest.mark.logic
def test_manifest_entries_are_immutable_for_the_run() -> None:
    """A second recording for the same key adopts the first SHA instead of overwriting it."""
    record_resolution("org/model", "main", SHA_A)
    returned = record_resolution("org/model", "main", SHA_B)
    assert returned == SHA_A, "the loser of a race adopts the winner's SHA, never overwrites it"
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


@pytest.mark.logic
def test_the_manifest_pins_across_an_upstream_move(offline: Callable[[HubLookup], Resolver]) -> None:
    """Once recorded, a run keeps its SHA even after upstream moves the ref."""
    record_resolution("org/model", "main", SHA_A)
    asked: list[str] = []

    def _moved(repo_id: str, ref: str, token: object = None) -> str:
        asked.append(ref)
        return SHA_B

    assert resolve_revision("org/model", "main", resolver=offline(_moved)) == SHA_A
    assert asked == [], "resolution must not be attempted when the manifest already has an answer"


@pytest.mark.logic
def test_an_injected_resolver_pins_nothing_for_another(tmp_path: Path) -> None:
    """Two resolvers over different cache roots keep their own memo and manifest: neither answer reaches the other."""
    first = Resolver(cache_dir=tmp_path / "hf-hub", hub=_answers(SHA_A), cache_root=tmp_path / "one")
    second = Resolver(cache_dir=tmp_path / "hf-hub", hub=_answers(SHA_B), cache_root=tmp_path / "two")
    assert resolve_revision("org/model", "main", resolver=first) == SHA_A
    assert resolve_revision("org/model", "main", resolver=second) == SHA_B
    key = manifest_key("org/model", "main")
    assert read_manifest(root=tmp_path / "one") == {key: SHA_A} and read_manifest(root=tmp_path / "two") == {key: SHA_B}
    assert first.memo == {key: SHA_A} and second.memo == {key: SHA_B}
    assert read_manifest() == {}, "nothing is recorded under the process's cache root"


@pytest.mark.logic
def test_an_unresolvable_ref_raises_rather_than_falling_back(offline: Callable[[HubLookup], Resolver]) -> None:
    """A ref that cannot be resolved raises instead of silently returning the ref itself."""

    def _fails(repo_id: str, ref: str, token: object = None) -> str:
        raise ValueError("cold cache, hub unreachable")

    with pytest.raises(RevisionResolutionError):
        resolve_revision("org/model", "main", resolver=offline(_fails))


@pytest.mark.logic
def test_a_resolved_ref_is_recorded_so_the_rest_of_the_run_follows(
    offline: Callable[[HubLookup], Resolver],
) -> None:
    """A freshly resolved SHA is written to the manifest so later calls skip resolution."""
    calls = []

    def _once(repo_id: str, ref: str, token: object = None) -> str:
        calls.append((repo_id, ref))
        return SHA_A

    assert resolve_revision("org/model", "main", resolver=offline(_once)) == SHA_A
    assert resolve_revision("org/model", "main", resolver=offline(_once)) == SHA_A, "a fresh memo reads the manifest"
    assert len(calls) == 1, "the manifest must answer after the first resolution"


@pytest.mark.logic
def test_a_corrupt_manifest_does_not_crash_resolution(offline: Callable[[HubLookup], Resolver]) -> None:
    """An unparsable manifest file degrades to 'resolve again' rather than raising."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text("{not json")
    assert resolve_revision("org/model", "main", resolver=offline(_answers(SHA_A))) == SHA_A


def _record_resolution_worker(sha: str, cache_dir: str, out: "mp.Queue[str]") -> None:
    """Run ``record_resolution`` in a spawned process and report the winning SHA.

    Module-level, not a closure inside the test: the ``spawn`` start method
    pickles the target by qualified name, and a function nested inside the test
    is a local object that plain ``pickle`` cannot resolve by name, so the
    original nested-closure version of this test failed with
    ``AttributeError: Can't pickle local object`` before any locking logic ran.
    The cache directory is passed as an explicit argument rather than captured
    from an enclosing scope for the same reason.
    """
    os.environ["SENSELAB_CACHE"] = cache_dir
    os.environ["SENSELAB_RUN_ID"] = "shared-run"
    import importlib

    import senselab_core.runtime.revision as mr

    importlib.reload(mr)
    out.put(mr.record_resolution("org/model", "main", sha))


@pytest.mark.logic
def test_concurrent_first_resolution_agrees_on_one_sha(tmp_path: Path) -> None:
    """Two processes recording different SHAs must converge on one winner."""
    cache = tmp_path / "shared"

    ctx = mp.get_context("spawn")
    q: "mp.Queue[str]" = ctx.Queue()
    procs = [ctx.Process(target=_record_resolution_worker, args=(s, str(cache), q)) for s in (SHA_A, SHA_B)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = {q.get(timeout=5) for _ in procs}
    assert len(results) == 1, f"both processes must adopt one SHA, got {results}"


@pytest.mark.logic
def test_a_corrupt_manifest_entry_is_ignored_rather_than_trusted(offline: Callable[[HubLookup], Resolver]) -> None:
    """A non-SHA entry must be re-resolved, not propagated into keys and provenance."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps({manifest_key("org/model", "main"): "main"}))
    assert resolve_revision("org/model", "main", resolver=offline(_answers(SHA_A))) == SHA_A


@pytest.mark.logic
def test_recording_a_non_sha_is_refused() -> None:
    """One bad write would poison every later participant, since entries are immutable."""
    with pytest.raises(RevisionResolutionError):
        record_resolution("org/model", "main", "main")


@pytest.mark.logic
def test_recording_a_newline_suffixed_sha_is_refused() -> None:
    """Unstripped subprocess stdout must not enter the manifest looking like a resolved commit."""
    with pytest.raises(RevisionResolutionError):
        record_resolution("org/model", "main", SHA_A + "\n")


@pytest.mark.logic
def test_a_newline_suffixed_sha_does_not_short_circuit_resolution(offline: Callable[[HubLookup], Resolver]) -> None:
    """A newline-suffixed full SHA — unstripped ``git rev-parse`` output — is not returned unchanged.

    It is not a commit, so it goes through resolution like any other ref instead of riding the
    already-a-SHA short-circuit into provenance.
    """
    assert resolve_revision("org/model", SHA_A + "\n", resolver=offline(_answers(SHA_A))) == SHA_A


@pytest.mark.logic
def test_a_newline_suffixed_manifest_entry_is_re_resolved(offline: Callable[[HubLookup], Resolver]) -> None:
    """A manifest entry carrying a trailing newline reads as absent, not as the run's answer."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps({manifest_key("org/model", "main"): SHA_A + "\n"}))
    assert resolve_revision("org/model", "main", resolver=offline(_answers(SHA_B))) == SHA_B


@pytest.mark.logic
def test_a_newline_suffixed_winner_is_overwritten_not_adopted() -> None:
    """``record_resolution`` replaces a newline-suffixed manifest winner instead of handing it back."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps({manifest_key("org/model", "main"): SHA_A + "\n"}))
    assert record_resolution("org/model", "main", SHA_B) == SHA_B
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_B


@pytest.mark.logic
def test_a_newline_suffixed_local_cache_answer_falls_through_to_the_hub(
    tmp_path: Path, offline: Callable[[HubLookup], Resolver]
) -> None:
    """A malformed local-cache answer degrades to a Hub lookup instead of becoming the resolved commit."""
    resolver = offline(_answers(SHA_B))
    assert resolver.cache_dir is not None
    snapshot = resolver.cache_dir / "models--org--model" / "snapshots" / (SHA_A + "\n")
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    assert get_cached_commit_hash("org/model", SHA_A + "\n", cache_dir=resolver.cache_dir) == SHA_A + "\n"
    assert resolver.lookup("org/model", SHA_A + "\n") == SHA_B


@pytest.mark.logic
def test_a_newline_suffixed_hub_sha_is_refused(offline: Callable[[HubLookup], Resolver]) -> None:
    """A Hub answer carrying a trailing newline raises rather than leaving the resolver as a commit."""
    with pytest.raises(RevisionResolutionError, match="no usable commit SHA"):
        offline(_answers(SHA_A + "\n")).lookup("org/model", "main")


@pytest.mark.logic
def test_resolution_retries_a_transient_hub_error(
    offline: Callable[[HubLookup], Resolver], hub_error: Callable[..., Exception]
) -> None:
    """The mandatory model_info round-trip is retried on a transient error, not fatal."""
    calls = {"n": 0}

    def _flaky(repo_id: str, ref: str, token: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise hub_error(429)
        return SHA_A

    resolver = offline(_flaky)
    assert resolve_revision("org/model", "main", resolver=resolver) == SHA_A
    assert calls["n"] == 2, "the transient error must have been retried once"
    assert _slept(resolver) == [FIRST_BACKOFF_S]


@pytest.mark.logic
def test_the_resolver_retries_the_hub_under_its_own_policy(
    tmp_path: Path, hub_error: Callable[..., Exception], lease_clock: Callable[..., ManualClock]
) -> None:
    """The attempts come from the resolver's `retry`, and the packaged policy is the default."""
    calls = {"n": 0}

    def _down(repo_id: str, ref: str, token: object) -> object:
        calls["n"] += 1
        raise hub_error(503)

    resolver = Resolver(
        cache_dir=tmp_path / "hf-hub",
        hub=_down,
        cache_root=tmp_path / "s",
        retry=RetryPolicy(attempts=2, first_backoff_s=FIRST_BACKOFF_S),
        clock=lease_clock(),
    )
    with pytest.raises(RevisionResolutionError):
        resolver.lookup("org/model", "main")
    assert calls["n"] == 2 and _slept(resolver) == [FIRST_BACKOFF_S]
    assert Resolver().retry == ExecutionProfile.load().retry


@pytest.mark.logic
def test_a_missing_repo_fails_without_being_retried(
    offline: Callable[[HubLookup], Resolver], hub_error: Callable[..., Exception]
) -> None:
    """A genuine 404, raised as the real `RepositoryNotFoundError`, surfaces on the first attempt."""
    calls = {"n": 0}

    def _always(repo_id: str, ref: str, token: object) -> object:
        calls["n"] += 1
        raise hub_error(404, RepositoryNotFoundError)

    with pytest.raises(RevisionResolutionError):
        resolve_revision("org/model", "main", resolver=offline(_always))
    assert calls["n"] == 1, "a 404 is a verdict, not a blip -- it must not be retried"


def _fake_cache(root: Path, repo_id: str, revision: str, sha: str) -> Path:
    """Fabricate an HF cache with ``refs/<revision> -> <sha>`` and a snapshot dir."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / revision).write_text(sha)
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    return snap


@pytest.mark.logic
def test_get_cached_commit_hash_resolves_real_sha_from_refs(tmp_path: Path) -> None:
    """get_cached_commit_hash returns the commit SHA from refs/<ref>, not the mutable branch name."""
    sha = "a" * 40
    _fake_cache(tmp_path, "org/model", "main", sha)
    assert get_cached_commit_hash("org/model", "main", cache_dir=tmp_path) == sha


@pytest.mark.logic
def test_sha_revision_is_passed_through(tmp_path: Path) -> None:
    """A full-SHA revision is returned as-is (already immutable)."""
    sha = "e" * 40
    _fake_cache(tmp_path, "org/model", "main", "f" * 40)
    assert get_cached_commit_hash("org/model", sha, cache_dir=tmp_path) == sha


@pytest.mark.logic
def test_cached_commit_hash_raises_rather_than_returning_a_ref(tmp_path: Path) -> None:
    """A cold cache must raise, not hand back the mutable ref as if it were a SHA."""
    with pytest.raises(RevisionResolutionError):
        get_cached_commit_hash("org/never-downloaded", "main", cache_dir=tmp_path / "hub")


TRAVERSING_REFS = ["../../../victim", "..", ".", "refs/../../victim", "a/b", "a\\b", "..\\victim", "/abs/victim", ""]


@pytest.mark.logic
@pytest.mark.parametrize("ref", TRAVERSING_REFS)
def test_a_ref_that_is_not_one_refs_entry_reads_nothing_from_the_cache(tmp_path: Path, ref: str) -> None:
    """A ref with a separator or `..` is refused before the cache is read, so no planted file attests a commit."""
    hub = tmp_path / "hub"
    _fake_cache(hub, "org/model", "main", "a" * 40)
    for victim in (tmp_path / "victim", hub / "models--org--model" / "refs" / "a" / "b"):
        victim.parent.mkdir(parents=True, exist_ok=True)
        victim.write_text("c" * 40)
    with pytest.raises(RevisionResolutionError):
        get_cached_commit_hash("org/model", ref, cache_dir=hub)
    resolver = Resolver(cache_dir=hub, hub=_answers(SHA_B), cache_root=tmp_path / "senselab")
    assert resolver.lookup("org/model", ref) == SHA_B


@pytest.mark.logic
def test_a_newline_suffixed_sha_is_not_passed_through(tmp_path: Path) -> None:
    """Unstripped subprocess stdout must not ride the full-SHA shortcut into provenance."""
    with pytest.raises(RevisionResolutionError):
        get_cached_commit_hash("org/model", "e" * 40 + "\n", cache_dir=tmp_path / "hub")


@pytest.mark.logic
def test_an_attempt_id_with_a_parent_component_is_refused(tmp_path: Path) -> None:
    """`../../escaped` is not an attempt id: nothing is written outside `runs/`."""
    assert bounded("probe_attempt_id", str(tmp_path), "../../escaped") == {"error": "ValueError", "outside": []}


@pytest.mark.logic
@pytest.mark.parametrize("run", BAD_ATTEMPT_IDS, ids=repr)
def test_every_attempt_id_that_is_not_one_component_is_refused(tmp_path: Path, run: str) -> None:
    """Each attempt id that is not a single path component is refused before anything is written."""
    assert bounded("probe_attempt_id", str(tmp_path), run) == {"error": "ValueError", "outside": []}


SET_ASIDE_FORMS = ["0123456789abcdef0123456789abcdef", "0" * 32, "f" * 32, uuid.UUID(int=7).hex]


@pytest.mark.logic
@pytest.mark.parametrize("run", SET_ASIDE_FORMS)
def test_an_attempt_id_of_the_set_aside_form_is_refused(tmp_path: Path, run: str) -> None:
    """An attempt id that a collector would read as a set-aside object is refused before anything is read or written."""
    with pytest.raises(ValueError, match="set-aside"):
        choose_attempt_id({"SENSELAB_RUN_ID": run})
    for call in (
        lambda: record_resolution("org/m", "main", SHA_A, run=run, root=tmp_path),
        lambda: read_manifest(run, root=tmp_path),
        lambda: manifest_path(run, root=tmp_path),
    ):
        with pytest.raises(ValueError, match="set-aside"):
            call()
    assert not (tmp_path / NAMESPACE).exists()


@pytest.mark.logic
def test_attempt_id_over_name_max_refused(tmp_path: Path) -> None:
    """An attempt id longer than one path component can hold is a `ValueError` before any I/O, never a raw `OSError`."""
    run = "x" * (NAME_MAX + 1)
    with pytest.raises(ValueError, match=str(NAME_MAX)):
        choose_attempt_id({"SENSELAB_RUN_ID": run})
    for call in (
        lambda: record_resolution("org/m", "main", SHA_A, run=run, root=tmp_path),
        lambda: read_manifest(run, root=tmp_path),
    ):
        with pytest.raises(ValueError, match=str(NAME_MAX)):
            call()
    assert not (tmp_path / NAMESPACE).exists()
    record_resolution("org/m", "main", SHA_A, run="x" * NAME_MAX, root=tmp_path)


@pytest.mark.logic
def test_a_lease_displaced_on_every_take_is_a_typed_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder displaced on each of `lease.record_attempts` takes raises `LeaseUnavailable` and writes nothing."""
    displaced = displace_after_take(monkeypatch, f"{MANIFEST_NAME}.lease")
    with pytest.raises(LeaseUnavailable, match="displaced"):
        record_resolution("org/m", "main", SHA_A, run="r", root=tmp_path)
    assert len(displaced) == ExecutionProfile.packaged().lease.record_attempts
    assert read_manifest("r", root=tmp_path) == {}


@pytest.mark.logic
def test_displaced_lease_keeps_manifest_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A holder displaced while it writes re-reads and merges under the lease: both holders' entries are kept.

    A is held at its manifest write while its lease is removed, as a breaker removes it;
    B then takes the lease, removes A's temporary as a dead writer's, and is held at its own write.
    """
    (a_reached, a_go), (b_reached, b_go) = pause_replaces(monkeypatch, MANIFEST_NAME, 2)
    answers: dict[str, object] = {}

    def record(name: str, sha: str) -> None:
        try:
            answers[name] = record_resolution(f"org/{name}", "main", sha, run="r", root=tmp_path)
        except Exception as err:  # noqa: BLE001 - reported to the test thread
            answers[name] = err

    a = threading.Thread(target=record, args=("a", SHA_A))
    a.start()
    assert a_reached.wait(60)
    shutil.rmtree(manifest_path("r", root=tmp_path).parent / f"{MANIFEST_NAME}.lease")
    b = threading.Thread(target=record, args=("b", SHA_B))
    b.start()
    assert b_reached.wait(60)
    a_go.set()
    b_go.set()
    a.join(60)
    b.join(60)
    assert answers == {"a": SHA_A, "b": SHA_B}
    keys = {manifest_key("org/a", "main"): SHA_A, manifest_key("org/b", "main"): SHA_B}
    assert read_manifest("r", root=tmp_path) == keys


@pytest.mark.logic
@pytest.mark.parametrize("run", ["job-123", "4242", str(uuid.UUID(int=7)), "0123456789abcdef0123456789abcdeF"])
def test_an_aged_run_survives_another_runs_collection(
    tmp_path: Path, run: str, lease_clock: Callable[..., ManualClock]
) -> None:
    """Every accepted attempt id's manifest outlives the quarantine age when another run collects what was set aside."""
    clock = lease_clock(storage_root=tmp_path)
    record_resolution("org/m", "main", SHA_A, run=run, root=tmp_path, clock=clock)
    clock.advance(2 * StorePolicy.load_default().quarantine_retention_s)
    record_resolution("org/m", "main", SHA_B, run="otherrun", root=tmp_path, clock=clock)
    assert read_manifest(run, root=tmp_path) == {manifest_key("org/m", "main"): SHA_A}


@pytest.mark.logic
def test_an_attempt_id_from_the_environment_is_one_component(tmp_path: Path) -> None:
    """SENSELAB_RUN_ID naming a parent directory is refused."""
    assert bounded("probe_environment_attempt_id", str(tmp_path), "../../escaped") == "ValueError"
    assert bounded("probe_environment_attempt_id", str(tmp_path), "job-123") == ""


@pytest.mark.logic
def test_run_manifests_take_the_policy_modes_under_umask_0(tmp_path: Path) -> None:
    """Under umask 0, `runs/`, the run directory and the manifest have the policy's modes, not world-writable ones."""
    report = bounded("probe_run_manifest_modes", str(tmp_path), "0")
    assert report["modes"] == report["expected"]
    assert report["leftover_removed"] is True and len(report["entries"]) == 2


@pytest.mark.contract
@pytest.mark.parametrize("umask", ["0", "022", "077", "007"])
def test_run_manifest_modes_do_not_depend_on_the_umask(tmp_path: Path, umask: str) -> None:
    """For every umask, the run manifest and its directories carry exactly the policy's modes."""
    report = bounded("probe_run_manifest_modes", str(tmp_path), umask)
    assert report["modes"] == report["expected"]


@pytest.mark.logic
@pytest.mark.parametrize("plant", ["symlink", "dangling"])
def test_the_packaged_lease_a_run_manifest_never_follows_a_planted_symlink(tmp_path: Path, plant: str) -> None:
    """A symlink at a run manifest's lease is neither chmodded nor created through."""
    report = bounded("probe_record_with_plant", str(tmp_path), LEASE, plant)
    assert report["error"] == "" and report["recorded"]
    assert report["victim_unchanged"] and report["victim"] == ["f"], report


@pytest.mark.logic
def test_a_hard_link_at_a_lease_leaves_its_target_alone(tmp_path: Path) -> None:
    """A hard link at a lease: the file it shares is not chmodded, truncated or re-stamped."""
    report = bounded("probe_record_with_plant", str(tmp_path), LEASE, "hardlink")
    assert report["error"] == "" and report["recorded"]
    assert report["victim_unchanged"], report


@pytest.mark.logic
@pytest.mark.parametrize("content", ["{}", "garbage", " ", '{"user": "u", "host": "h", "pid": 1}'])
def test_a_stale_lease_whatever_its_content_never_hangs_record_resolution(tmp_path: Path, content: str) -> None:
    """A stale lease is broken whatever it holds, and the resolution is recorded within the bound."""
    report = bounded("probe_record_with_plant", str(tmp_path), LEASE, content)
    assert report["error"] == "" and report["recorded"] and report["seconds"] < NAME_BOUND_S, report


@pytest.mark.logic
@pytest.mark.parametrize("plant", ["directory", "fifo", "symlink"])
def test_a_wrong_kind_at_the_lease_name_is_set_aside(tmp_path: Path, plant: str) -> None:
    """A directory, fifo or symlink at the lease is moved aside; the resolution is recorded, no OSError."""
    report = bounded("probe_record_with_plant", str(tmp_path), LEASE, plant)
    assert report["error"] == "" and report["recorded"], report


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a 0o000 file")
@pytest.mark.parametrize("name", UNOPENABLE_NAMES)
def test_an_unopenable_file_at_a_lease_or_manifest_name_never_wedges_record_resolution(
    tmp_path: Path, name: str
) -> None:
    """A regular file this user cannot open, as another user's 0o600 file is, is moved aside or replaced."""
    report = bounded("probe_record_with_plant", str(tmp_path), name, "unopenable")
    assert report["error"] == "" and report["recorded"] and report["seconds"] < NAME_BOUND_S, report


@pytest.mark.logic
@pytest.mark.skipif(os.geteuid() == 0, reason="root renames in a read-only directory")
def test_an_unopenable_lease_that_cannot_be_moved_is_a_typed_error_naming_its_remedy(tmp_path: Path) -> None:
    """When the run directory refuses the move too, the error is `LeaseUnavailable` naming the lease and its owner."""
    report = bounded("probe_record_with_plant", str(tmp_path), LEASE, "unmovable")
    assert report["error"] == "LeaseUnavailable", report
    assert LEASE in report["message"] and "owned by" in report["message"], report


@pytest.mark.logic
def test_the_manifest_lease_outlasts_its_stale_bound_while_it_beats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the manifest write the storage clock passes the stale bound; the lease beats, is never broken, writes."""
    lease = packaged_lease(30, 0.01, 0)
    clock = install(monkeypatch)
    run_dir = manifest_path("r", root=tmp_path).parent
    on_create(monkeypatch, f".{MANIFEST_NAME}.", lambda: clock.outlast(run_dir, lease.stale_after_s))
    assert record_resolution("org/m", "main", SHA_A, run="r", root=tmp_path, lease=lease) == SHA_A
    assert clock.attempts == 4 and clock.breaks == 0
    assert read_manifest("r", root=tmp_path) == {manifest_key("org/m", "main"): SHA_A}


@pytest.mark.logic
def test_a_holder_displaced_just_before_its_commit_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is committed only after a fresh `mkdir` proves the lease; one displaced before it writes nothing."""
    displaced = displace_before_proof(monkeypatch, f"{MANIFEST_NAME}.lease")
    with pytest.raises(LeaseUnavailable, match="displaced"):
        record_resolution("org/m", "main", SHA_A, run="r", root=tmp_path)
    assert len(displaced) == ExecutionProfile.packaged().lease.record_attempts
    assert read_manifest("r", root=tmp_path) == {}
