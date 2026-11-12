"""The keys that were fixed parameters: each is in both packaged profiles, required, and obeyed (P-34)."""

import json
import os
import signal
import subprocess
import tomllib
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import senselab_core
import yaml
from lease_clock_fixture import ManualClock
from planted_objects_fixture import store_context, store_key
from pydantic import ValidationError
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime import envs, hardware, storage
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution import supervisor as supervisor_module
from senselab_core.runtime.execution.run import Run
from senselab_core.runtime.execution.substrates import Local
from senselab_core.runtime.execution_settings import JobLease, LeaseBoundError
from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import STORE_DIR, ResultStore
from senselab_core.runtime.vllm_server import VllmServer, VllmStartError
from senselab_core.testing.leases import packaged_lease

PACKAGED = Path(senselab_core.__file__).parent / "runtime" / "data" / "profiles"
PROFILE = ExecutionProfile.packaged()
KEYS = {
    "lease": ["missed_beats", "wait_s", "poll_s", "remove_attempts", "record_attempts", "dir_mode"],
    "pool": ["idle_polls", "cancel_check_s", "event_poll_s"],
    "substrate": ["stop_grace_s"],
    "supervisor": ["reap_poll_s", "exit_bound_s"],
    "timeouts": ["fetch_s"],
    "store": ["commit_attempts", "probe_bound_s", "probe_poll_s", "clock_tolerance_s"],
    "vllm": ["startup_s", "poll_s", "request_s", "stop_s", "health_s"],
    "worker": ["hello_s", "leader_probe_s", "group_poll_max_s"],
}
PAIRS = [(section, key) for section, keys in KEYS.items() for key in keys]


@pytest.mark.logic
@pytest.mark.parametrize("name", ["laptop", "slurm"])
def test_both_packaged_profiles_carry_every_key(name: str) -> None:
    """The generic profiles, the local one and the Slurm template, set each key; no value is a site's figure."""
    raw = yaml.safe_load((PACKAGED / f"{name}.yaml").read_text())
    assert [(s, k) for s, k in PAIRS if k not in raw[s]] == []


@pytest.mark.logic
@pytest.mark.parametrize(("section", "key"), PAIRS)
def test_a_profile_missing_a_key_is_refused(section: str, key: str) -> None:
    """A key has no default: a profile without it fails validation, naming it."""
    document = PROFILE.model_dump(mode="json")
    del document[section][key]
    with pytest.raises(ValidationError, match=key):
        ExecutionProfile.model_validate(document)


@pytest.mark.logic
def test_the_package_ships_no_equivalence_table_and_no_site_profile() -> None:
    """The wheel's tree holds no `equivalence.yaml`; its only profiles are the two generic ones."""
    package = Path(senselab_core.__file__).parent
    assert [p for p in package.rglob("*") if "equivalence" in p.name and p.suffix in {".yaml", ".yml"}] == []
    assert sorted(p.name for p in PACKAGED.iterdir() if p.suffix == ".yaml") == ["laptop.yaml", "slurm.yaml"]
    forbidden = tomllib.loads((package.parents[1] / "pyproject.toml").read_text())
    assert "benchmarks" not in str(forbidden.get("tool", {}).get("hatch", {}).get("build", {}))


@pytest.mark.logic
def test_the_beat_bound_follows_missed_beats() -> None:
    """`missed_beats × renew_fraction × stale_after_s + attribute_delay_s < stale_after_s` is the bound that loads."""
    lease = packaged_lease(300, 0.33, 60)
    lease.check_bound()
    with pytest.raises(LeaseBoundError, match="3 × lease.renew_fraction"):
        lease.model_copy(update={"missed_beats": 3}).check_bound()


@pytest.mark.logic
def test_a_wait_for_a_lease_is_bounded_by_wait_s_and_polled_every_poll_s(
    tmp_path: Path, lease_clock: Callable[..., ManualClock]
) -> None:
    """A second taker polls every `poll_s` and gives up after `wait_s`, on the injected clock."""
    settings = PROFILE.lease.model_copy(update={"wait_s": 3.0, "poll_s": 0.5})
    resource = tmp_path / "thing"
    clock = lease_clock()
    with Lease(resource, settings=settings, timeout=0, clock=clock):
        with pytest.raises(LeaseTimeout):
            with Lease(resource, settings=settings, clock=clock):
                pass
    assert clock.slept == [0.5] * 6


@pytest.mark.logic
def test_a_lease_directory_is_made_with_dir_mode(tmp_path: Path) -> None:
    """The lease directory's mode is the profile's `dir_mode`, not a constant."""
    settings = PROFILE.lease.model_copy(update={"dir_mode": "0700"})
    with Lease(tmp_path / "thing", settings=settings, timeout=0) as held:
        assert os.stat(held.path).st_mode & 0o7777 == 0o700
    assert settings.mode == 0o700 and PROFILE.lease.mode == 0o2770


def _stats_in_a_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempts: int) -> int:
    """The `os.lstat` calls `Lease._remove` makes on a moved-aside lease that cannot be removed: one per attempt."""
    settings = PROFILE.lease.model_copy(update={"remove_attempts": attempts})
    work = tmp_path / f"w{attempts}"
    work.mkdir()
    held = Lease(work / "thing", settings=settings, timeout=0)
    aside = "gone-" + "0" * 32
    (work / aside).mkdir()
    (work / aside / "foreign.txt").write_text("not a token")
    held._directory()
    stats: list[str] = []
    real = os.lstat

    def counting(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:  # noqa: ANN401
        stats.append(str(path))
        return real(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "lstat", counting)
        held._remove(aside)
    assert len(held.set_aside) == 1
    return len(stats)


@pytest.mark.logic
def test_a_lease_moved_aside_is_removed_remove_attempts_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A removal that never takes is tried `remove_attempts` times: 3 more attempts look at the lease 3 more times."""
    assert _stats_in_a_removal(tmp_path, monkeypatch, 4) - _stats_in_a_removal(tmp_path, monkeypatch, 1) == 3


@pytest.mark.logic
def test_run_wait_polls_every_event_poll_s(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """`Run.wait` reads the event log every `pool.event_poll_s` of the pool section it was given."""
    pool = PROFILE.pool.model_copy(update={"event_poll_s": 2.0})
    clock = lease_clock()
    assert not Run(tmp_path, profile=PROFILE.model_copy(update={"pool": pool}), clock=clock).wait(7.0).done
    assert clock.slept == [2.0, 2.0, 2.0, 1.0]


@pytest.mark.logic
def test_a_local_worker_is_given_stop_grace_s_after_sigterm() -> None:
    """`Local.stop` waits `stop_grace_s` for a terminated worker before it kills it."""
    waited: list[float] = []

    class Proc:
        pid = 1

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            waited.append(timeout or 0.0)
            return 0

    local = Local(1, command=["x"], stop_grace_s=7.5)
    local._procs["1"] = SimpleNamespace(proc=Proc(), supervisor=SimpleNamespace(released=lambda pid: None))  # type: ignore[assignment]
    local.stop([SimpleNamespace(id="1")])  # type: ignore[list-item]
    assert waited == [7.5]


@pytest.mark.logic
def test_the_keeper_polls_its_reaping_every_reap_poll_s(lease_clock: Callable[..., ManualClock]) -> None:
    """A keeper that outlasts its stop bound is polled every `reap_poll_s` on the clock, then killed."""
    settings = PROFILE.supervisor.model_copy(update={"reap_poll_s": 1.0})
    clock = lease_clock()
    keeper = supervisor_module.Keeper(settings, clock)
    child = subprocess.Popen(["sleep", "30"])  # noqa: S607 - a child that outlasts the bound
    keeper.pid = child.pid
    try:
        keeper.stop(3.0)
    except ChildProcessError:
        pass
    assert clock.slept == [1.0] * 4
    child.kill()
    child.wait()


@pytest.mark.logic
def test_a_supervisor_leaves_within_exit_bound_s_polling_reap_poll_s() -> None:
    """A supervisor takes its exit bound and its reaper's interval from its settings."""
    settings = PROFILE.supervisor.model_copy(update={"reap_poll_s": 0.3, "exit_bound_s": 9.0})
    sup = supervisor_module.Supervisor(subreaper=False, settings=settings)
    assert (sup._poll_s, sup._keeper.settings.exit_bound_s) == (0.3, 9.0)
    assert supervisor_module.Supervisor(subreaper=False, poll_s=0.7, settings=settings)._poll_s == 0.7


@pytest.mark.logic
def test_a_torch_index_request_waits_timeout_s(monkeypatch: pytest.MonkeyPatch) -> None:
    """`compile_lock`'s index reads are bounded by `timeout_s`."""
    seen: list[float] = []

    def urlopen(url: str, timeout: float) -> Any:  # noqa: ANN401
        seen.append(timeout)
        raise OSError("stop here")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError, match="stop here"):
        envs._wheel_on_index("torch", "1", "cpu", "3.12", ("x",), 11.0)
    assert seen == [11.0]


@pytest.mark.logic
def test_a_hardware_probe_is_bounded_by_its_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """`probe_hardware` runs `nvidia-smi` within `timeout_s`; the packaged `sizing.device_query_s` when None."""
    seen: list[float] = []

    def run(cmd: list[str], **kwargs: Any) -> Any:  # noqa: ANN401
        seen.append(kwargs["timeout"])
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, "run", run)
    hardware.probe_hardware(timeout_s=3.5)
    hardware.probe_hardware()
    assert set(seen) == {3.5, PROFILE.sizing.device_query_s}


@pytest.mark.logic
def test_the_clock_probe_waits_probe_bound_s_polling_probe_poll_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """An mtime that never moves is waited on every `probe_poll_s` until `probe_bound_s`, then judged skewed."""
    settings = PROFILE.store.model_copy(update={"probe_bound_s": 2.0, "probe_poll_s": 0.5})
    clock = lease_clock()
    work = tmp_path / "work"
    work.mkdir()
    fixed = work.stat()
    real = Path.stat
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: fixed if self == work else real(self, *a, **k))
    assert storage._mtime_after_entry(work, settings, clock) == (False, 2.0)
    assert clock.slept == [0.5] * 4


@pytest.mark.logic
def test_the_clock_tolerance_is_the_stores(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """A storage clock more than `clock_tolerance_s` from the host's is not the host's."""
    work = tmp_path / "work"
    work.mkdir()
    skewed = lease_clock()
    skewed.local_now = lambda: 1e12  # type: ignore[method-assign]
    assert storage._mtime_after_entry(work, PROFILE.store, skewed)[0] is False
    other = tmp_path / "other"
    other.mkdir()
    wide = PROFILE.store.model_copy(update={"clock_tolerance_s": 1e13})
    assert storage._mtime_after_entry(other, wide, skewed)[0] is True


class _Stubborn(ResultStore):
    """A store that leaves each corrupt entry where it is, counting the times it was asked to quarantine one."""

    quarantines = 0

    def _quarantine(self, entry: Path, **found: str) -> None:
        self.quarantines += 1


@pytest.mark.logic
@pytest.mark.parametrize("attempts", [1, 5])
def test_a_put_commits_commit_attempts_times_before_it_gives_up(tmp_path: Path, attempts: int) -> None:
    """A target that keeps turning out corrupt is committed `commit_attempts` times."""
    key, context = store_key(), store_context()
    ResultStore(tmp_path).put(key, {"y": 1}, context)
    (entry,) = (tmp_path / STORE_DIR).glob("*/*/*")
    (entry / "result.json").chmod(0o644)
    (entry / "result.json").write_text("not a result")
    store = _Stubborn(tmp_path, settings=PROFILE.store.model_copy(update={"commit_attempts": attempts}))
    with pytest.raises(RuntimeError, match=f"after {attempts} commits"):
        store.put(key, {"y": 2}, context)
    assert store.quarantines == attempts


@pytest.mark.logic
def test_a_group_poll_never_exceeds_group_poll_max_s(lease_clock: Callable[..., ManualClock]) -> None:
    """A stopping group is read at doubling intervals up to `group_poll_max_s`."""
    import threading

    from senselab_core.runtime import worker_client
    from senselab_worker.codec import Deadline

    clock = lease_clock()
    exited = threading.Event()
    exited.set()
    gone = worker_client._members_gone(os.getpid(), exited, Deadline(1.0, clock.now), lambda pgid: True, clock, 0.25)
    assert gone is False and max(clock.slept) == 0.25 and clock.slept[:3] == [0.001, 0.002, 0.004]


class _Popen:
    pid = 4242

    def __init__(self) -> None:
        self.waited: list[float | None] = []

    def poll(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout)
        return 0


class _Reply:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


@pytest.mark.logic
def test_the_vllm_keys_bound_startup_the_poll_the_health_read_the_stop_and_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_clock: Callable[..., ManualClock]
) -> None:
    """Each `vllm` key is obeyed: start-up wait, health poll, health read, stop grace and request bound."""
    sha = "a" * 40
    settings = {
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "gpu_memory_utilization": 0.9,
        "kv_cache_dtype": "auto",
        "generation_config": "vllm",
        "limit_mm_per_prompt": {"image": 0},
        "enable_prefix_caching": False,
        "flashinfer_sampler": False,
    }
    update = {"startup_s": 3.0, "poll_s": 1.0, "health_s": 6.0, "stop_s": 8.0, "request_s": 9.0}
    clock = lease_clock()
    env = EnvHandle(
        python=str(tmp_path / "bin" / "python"), python_version="3.12", lock_sha="l", wheel_set="w", torch_index=None
    )
    model = ResolvedModel(ref=ModelRef.parse(f"org/model@{sha}"), identity=sha)
    server = VllmServer(
        env,
        model,
        settings,
        start_lock_dir=tmp_path,
        start_margin_s=1,
        lease=PROFILE.lease,
        vllm=PROFILE.vllm.model_copy(update=update),
        clock=clock,
    )
    popen = _Popen()
    server.url = "http://127.0.0.1:1"
    server._server = popen  # type: ignore[assignment]
    asked: list[float] = []

    def refuse(self: urllib.request.OpenerDirector, request: Any, data: Any = None, timeout: float = 0.0) -> Any:  # noqa: ANN401
        asked.append(timeout)
        if "/health" in str(request):
            raise OSError("not up")
        body = json.dumps({"model": server.served_name}).encode()
        return _Reply(body)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", refuse)
    with pytest.raises(VllmStartError, match="within 3s"):
        server._await_health()
    assert clock.slept == [1.0] * 4 and set(asked) == {6.0}
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    server.attestation = SimpleNamespace(served_model=server.served_name)  # type: ignore[assignment]
    asked.clear()
    server.chat([{"role": "user", "content": "x"}], sampling={})
    assert asked == [9.0]
    server.stop()
    assert popen.waited == [8.0, None] and sent[0][1] == signal.SIGTERM


@pytest.mark.logic
def test_profile_sections_are_frozen() -> None:
    """A section is a value: a key is not set after load."""
    with pytest.raises(ValidationError):
        PROFILE.lease.wait_s = 1.0  # type: ignore[misc]
    assert isinstance(PROFILE.lease, JobLease) and isinstance(Mapping, type)
