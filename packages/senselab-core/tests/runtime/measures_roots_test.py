"""The layered measures roots: writes, reads under a trust filter, the hardware index, the probe lease, collection."""

import errno
import getpass
import importlib.util
import logging
import os
import socket
import stat
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from senselab_core.kernel.identity import ModelIdentity
from senselab_core.runtime.calibration import Estimate, HardwareClass, Observation
from senselab_core.runtime.execution_settings import ResourceClass, Trust
from senselab_core.runtime.locks import Lease, LeaseTimeout
from senselab_core.runtime.measures import (
    Build,
    MeasureRoots,
    MeasureSource,
    MeasureUnit,
    PeerProbing,
    Provenance,
    Unleased,
    class_digest,
    resolve_roots,
)
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import PackagedTable
from senselab_core.runtime.revision import cache_root
from senselab_core.runtime.storage import StorePolicy
from senselab_core.testing.calibration_samples import UNIT, observation
from syscall_faults_fixture import fail_at

POLICY = StorePolicy.load_default()
UID = os.geteuid()
STRANGER = UID + 1
STALE_S = ExecutionProfile.load("orcd").lease.stale_after_s
PROVENANCE = Provenance(
    job="1234", run="run-1", senselab=Build(version="2.0.0", commit="a" * 40), profile="d" * 64, hardware=UNIT.hardware
)
DECLARED = ResourceClass(device="cuda", gpu_type=None, max_workers=1)
needs_non_root = pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")


def _observation(item: str = "i") -> Observation:
    return observation(probe=None, item=item)


def _roots(*roots: Path, trust: Trust = "group", uid: int = UID) -> MeasureRoots:
    return MeasureRoots(roots, POLICY, trust, uid, stale_after_s=STALE_S)


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _age(path: Path, seconds: float) -> None:
    then = time.time() - seconds
    os.utime(path, (then, then))


@pytest.mark.contract
@needs_non_root
def test_read_only_shared_root_falls_back_to_own(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """B2: a read-only group root fails no write; the observation lands in the own root and a log line names it."""
    shared, own = tmp_path / "shared", tmp_path / "own"
    shared.mkdir()
    shared.chmod(0o500)
    try:
        roots = _roots(shared, own)
        with caplog.at_level(logging.WARNING, logger="senselab"):
            written = roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
        assert written is not None and own in written.parents and roots.write_root == own
        assert any(str(own) in r.getMessage() and "writ" in r.getMessage() for r in caplog.records)
        assert roots.observations(UNIT) == (_observation(),)
        assert not any(shared.iterdir())
    finally:
        shared.chmod(0o700)


@pytest.mark.contract
@pytest.mark.parametrize("err", [errno.ENOSPC, errno.ESTALE])
def test_an_oserror_on_a_root_is_logged_and_never_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, err: int
) -> None:
    """A write fault moves the write to the next root, else the measure is in memory only; a read fault is logged."""
    first, second = tmp_path / "first", tmp_path / "second"
    error = OSError(err, os.strerror(err))
    with caplog.at_level(logging.WARNING, logger="senselab"):
        fail_at(monkeypatch, "p1.json", "write", error)
        both = _roots(first, second)
        landed = both.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
        assert landed is not None and second in landed.parents
        assert sum(str(first) in r.getMessage() for r in caplog.records) == 1
        fail_at(monkeypatch, "p2.json", "write", error)
        assert _roots(tmp_path / "only").put_observations(UNIT, "p2", [_observation()], PROVENANCE) is None
        fail_at(monkeypatch, "p1.json", "read", error, creating=False)
        assert both.observations(UNIT) == ()
        assert both.observations(UNIT) == (_observation(),)
    assert sum("cannot be read" in r.getMessage() and str(second) in r.getMessage() for r in caplog.records) == 1


@pytest.mark.contract
def test_lookup_and_write_use_one_hardware_digest(tmp_path: Path) -> None:
    """B3: a class declared with `gpu_type` None is read under the digest of the H100 the worker reported."""
    roots = _roots(tmp_path)
    assert DECLARED.gpu_type is None and roots.seen_classes(DECLARED) == ()
    seen = HardwareClass(
        device="cuda", name="NVIDIA H100 80GB HBM3", memory_mib=81559, compute_capability="9.0", cpus=8
    )
    roots.note_seen(DECLARED, seen)
    roots.note_seen(DECLARED, seen)
    (resolved,) = roots.seen_classes(DECLARED)
    assert class_digest(resolved) == class_digest(seen)
    written = UNIT.model_copy(update={"hardware": resolved})
    roots.put_observations(written, "p1", [_observation()], PROVENANCE)
    looked_up = UNIT.model_copy(update={"hardware": roots.seen_classes(DECLARED)[0]})
    assert looked_up.digest() == written.digest()
    assert roots.observations(looked_up) == (_observation(),)
    assert roots.observations(UNIT.model_copy(update={"hardware": seen.model_copy(update={"cpus": 4})})) == ()
    other = seen.model_copy(update={"name": "NVIDIA A100-SXM4-80GB"})
    roots.note_seen(DECLARED, other)
    assert {c.name for c in roots.seen_classes(DECLARED)} == {seen.name, other.name}
    assert roots.seen_classes(ResourceClass(device="cuda", gpu_type="h100", max_workers=1)) == ()


@pytest.mark.contract
def test_the_hardware_index_obeys_the_trust_filter(tmp_path: Path) -> None:
    """Under `own`, an index entry another uid owns is not read."""
    _roots(tmp_path).note_seen(DECLARED, UNIT.hardware)
    assert _roots(tmp_path, trust="own", uid=STRANGER).seen_classes(DECLARED) == ()
    assert _roots(tmp_path, trust="own").seen_classes(DECLARED) == (UNIT.hardware,)
    assert _roots(tmp_path, trust="group", uid=STRANGER).seen_classes(DECLARED) == (UNIT.hardware,)


@pytest.mark.contract
def test_no_measures_record_carries_an_account_or_host_name(tmp_path: Path) -> None:
    """Every file of every record kind, the unit, the index and the lease holds neither the account nor the host."""
    roots = _roots(tmp_path)
    roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    for kind in ("drift", "supersede", "regime", "oom"):
        roots.put_record(UNIT, kind, {"reason": kind, "outdates": ["p1"]}, PROVENANCE)
    roots.note_seen(DECLARED, UNIT.hardware)
    lease = roots.probe_lease(UNIT)
    assert isinstance(lease, Lease)
    with lease:
        pass
    names = {getpass.getuser(), socket.gethostname(), socket.gethostname().split(".")[0], str(Path.home())}
    names -= {"", "unit", "obs", "run", "drift", "oom"}
    files = _files(tmp_path)
    assert len(files) == 7
    for path in files:
        text = path.read_text() + str(path.relative_to(tmp_path))
        assert not [n for n in names if n in text], path


@pytest.mark.contract
def test_another_users_drift_record_does_not_act_under_trust_own(tmp_path: Path) -> None:
    """I5: under `own` a record of another uid is not read, so it can neither supersede nor cap a measure."""
    writer = _roots(tmp_path)
    writer.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    for kind in ("drift", "supersede", "oom"):
        writer.put_record(UNIT, kind, {"n": 1, "outdates": ["p1"]}, PROVENANCE)
    assert {r.kind for r in _roots(tmp_path, trust="group", uid=STRANGER).records(UNIT)} == {
        "drift",
        "supersede",
        "oom",
    }
    assert _roots(tmp_path, trust="own", uid=STRANGER).records(UNIT) == ()
    assert {r.uid for r in _roots(tmp_path, trust="own").records(UNIT)} == {UID}


@pytest.mark.contract
def test_a_supersede_record_names_the_probes_it_outdates_whatever_their_mtime(tmp_path: Path) -> None:
    """Pooling excludes by probe id: a superseded file touched newer stays out, and a newer unnamed one pools."""
    roots = _roots(tmp_path)
    old = roots.put_observations(UNIT, "old", [_observation("old")], PROVENANCE)
    assert old is not None
    roots.put_record(UNIT, "supersede", {"outdates": ["old"]}, PROVENANCE)
    for age in (1000, -1000, 0):
        _age(old, age)
        roots.put_observations(UNIT, f"fresh{age}", [_observation(f"fresh{age}")], PROVENANCE)
        assert "old" not in {o.item for o in roots.observations(UNIT)}
    assert {o.item for o in roots.observations(UNIT)} == {"fresh1000", "fresh-1000", "fresh0"}
    assert _roots(tmp_path, trust="own", uid=STRANGER).observations(UNIT) == ()
    refused: list[dict[str, object]] = [{}, {"outdates": []}, {"outdates": [""]}, {"outdates": "old"}]
    for bad in refused:
        with pytest.raises(ValueError, match="outdates"):
            roots.put_record(UNIT, "supersede", bad, PROVENANCE)


@pytest.mark.contract
def test_collect_never_removes_another_users_file(tmp_path: Path) -> None:
    """A collector removes superseded observations, stale staging and quarantine its uid owns, and nothing else."""
    roots = _roots(tmp_path)
    paths = [roots.put_observations(UNIT, f"p{n}", [_observation()], PROVENANCE) for n in range(3)]
    for n, path in enumerate(paths):
        assert path is not None
        _age(path, 1000 - n)
    superseded = roots.put_record(UNIT, "supersede", {"outdates": ["p0", "p1", "p2"]}, PROVENANCE)
    assert superseded is not None
    measures = tmp_path / "measures@1"
    staging, quarantined = measures / "tmp" / uuid.uuid4().hex, measures / "quarantine" / uuid.uuid4().hex
    quarantined.parent.mkdir(exist_ok=True)
    for old in (staging, quarantined):
        old.mkdir()
        _age(old, 10 * POLICY.quarantine_retention_s)
    far = time.time() + 10 * POLICY.quarantine_retention_s
    assert _roots(tmp_path, uid=STRANGER).collect(kept=1, now=far) == 0
    assert all(p is not None and p.exists() for p in paths) and staging.exists() and quarantined.exists()
    assert roots.collect(kept=1, now=far) == 4
    assert [p.exists() for p in paths if p is not None] == [False, False, True]
    assert not staging.exists() and not quarantined.exists() and superseded.exists()


@pytest.mark.contract
def test_two_writers_of_one_unit_both_keep_their_observations(tmp_path: Path) -> None:
    """Writers never conflict: two coordinators of one unit leave both probes, and one `unit.json`."""
    a, b = _roots(tmp_path), _roots(tmp_path)
    a.put_observations(UNIT, "pa", [_observation("a")], PROVENANCE)
    b.put_observations(UNIT, "pb", [_observation("b")], PROVENANCE)
    assert b.put_observations(UNIT, "pb", [_observation("other")], PROVENANCE) is not None
    assert sorted(o.item for o in _roots(tmp_path).observations(UNIT)) == ["a", "b"]
    assert len(list(tmp_path.rglob("unit.json"))) == 1


@pytest.mark.contract
def test_a_misfiled_unit_directory_is_quarantined_only_by_its_owner(tmp_path: Path) -> None:
    """A directory whose `unit.json` digests to another name is skipped by another uid, moved by its owner."""
    roots = _roots(tmp_path)
    roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    other = UNIT.model_copy(update={"model": ModelIdentity(id="openai/whisper-base", identity="e" * 40)})
    units = tmp_path / "measures@1" / "units"
    home = units / UNIT.digest()[:2] / UNIT.digest()
    misfiled = units / other.digest()[:2] / other.digest()
    misfiled.parent.mkdir(exist_ok=True)
    home.rename(misfiled)
    assert _roots(tmp_path, uid=STRANGER).observations(other) == ()
    assert misfiled.is_dir() and not (tmp_path / "measures@1" / "quarantine").exists()
    assert roots.observations(other) == ()
    assert not misfiled.exists()
    (moved,) = (tmp_path / "measures@1" / "quarantine").iterdir()
    assert (moved / "obs" / "p1.json").is_file()


@pytest.mark.contract
def test_a_file_of_an_unknown_format_is_skipped_not_moved(tmp_path: Path) -> None:
    """A later release's file is left where it is by every reader; a malformed one is moved only by its owner."""
    roots = _roots(tmp_path)
    written = roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    malformed = roots.put_observations(UNIT, "p2", [_observation()], PROVENANCE)
    assert written is not None and malformed is not None
    for path, text in (
        (written, written.read_text().replace("measure-observations@1.0", "measure-observations@2.0")),
        (malformed, "{"),
    ):
        path.chmod(0o640)
        path.write_text(text)
    quarantine = tmp_path / "measures@1" / "quarantine"
    assert _roots(tmp_path, uid=STRANGER).observations(UNIT) == ()
    assert written.exists() and malformed.exists() and not quarantine.exists()
    assert roots.observations(UNIT) == ()
    assert written.exists() and not malformed.exists() and len(list(quarantine.iterdir())) == 1


@pytest.mark.contract
def test_a_live_peer_lease_is_found_in_any_root_and_taken_only_in_the_write_root(tmp_path: Path) -> None:
    """A live lease in a root this reader does not write to is `PeerProbing`; with none the lease is in its own."""
    shared, own = tmp_path / "shared", tmp_path / "own"
    peer, me = _roots(shared), _roots(shared, own, trust="group")
    held = peer.probe_lease(UNIT)
    assert isinstance(held, Lease)
    with held:
        assert me.peer_holds(UNIT) and me.probe_lease(UNIT) == PeerProbing()
        with pytest.raises(LeaseTimeout):
            with Lease(shared / "measures@1" / "units" / UNIT.digest()[:2] / UNIT.digest() / "probe", timeout=0):
                pass
        assert not _roots(shared, trust="own", uid=STRANGER).peer_holds(UNIT)
    assert not me.peer_holds(UNIT)
    taken = me.probe_lease(UNIT)
    assert isinstance(taken, Lease)
    assert taken.path.parent == shared / "measures@1" / "units" / UNIT.digest()[:2] / UNIT.digest()


@pytest.mark.contract
def test_under_trust_own_a_foreign_live_lease_means_probe_without_a_lease(tmp_path: Path) -> None:
    """The filter that ignores a foreign lease also stops a take that would only time out: no lease, no wait."""
    holder = _roots(tmp_path).probe_lease(UNIT)
    assert isinstance(holder, Lease)
    with holder:
        stranger = _roots(tmp_path, trust="own", uid=STRANGER)
        assert not stranger.peer_holds(UNIT)
        assert stranger.probe_lease(UNIT) == Unleased("foreign lease")
        assert _roots(tmp_path, trust="group", uid=STRANGER).probe_lease(UNIT) == PeerProbing()
        assert _roots(tmp_path, trust="own").probe_lease(UNIT) == PeerProbing()


@pytest.mark.contract
@needs_non_root
def test_the_lease_is_taken_in_the_first_writable_root_and_no_root_is_a_typed_outcome(tmp_path: Path) -> None:
    """With the group root read-only the lease lies in the own root; with no writable root the answer says so."""
    shared, own = tmp_path / "shared", tmp_path / "own"
    shared.mkdir()
    shared.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        lease = _roots(shared, own).probe_lease(UNIT)
        assert isinstance(lease, Lease) and own in lease.path.parents
        assert _roots(shared).probe_lease(UNIT) == Unleased("no writable root")
    finally:
        shared.chmod(0o700)


@pytest.mark.contract
@needs_non_root
def test_a_live_lease_is_seen_in_a_root_that_cannot_be_written(tmp_path: Path) -> None:
    """A read-only group root cannot give the storage clock, so liveness falls back to the local clock."""
    holder = _roots(tmp_path).probe_lease(UNIT)
    assert isinstance(holder, Lease)
    with holder:
        subprocess.run(["chmod", "-R", "a-w", str(tmp_path)], check=True)
        try:
            assert _roots(tmp_path).peer_holds(UNIT)
        finally:
            subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=True)


@pytest.mark.contract
def test_a_file_that_cannot_be_parsed_is_malformed_never_raised(tmp_path: Path) -> None:
    """Deeply nested JSON and an out-of-range mtime are read as malformed or harmless, not raised."""
    roots = _roots(tmp_path)
    written = roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    record = roots.put_record(UNIT, "drift", {"n": 1}, PROVENANCE)
    assert written is not None and record is not None
    written.chmod(0o640)
    written.write_text("[" * 200000)
    os.utime(record, (-5, -5))
    assert roots.observations(UNIT) == ()
    assert [r.written_s for r in roots.records(UNIT)] == [0.0]


@pytest.mark.contract
def test_the_roots_are_one_source_of_the_measure_source(tmp_path: Path) -> None:
    """`MeasureSource` pools the observations of the roots it is given and estimates them once."""
    roots = _roots(tmp_path)
    for n in range(2):
        roots.put_observations(UNIT, f"p{n}", [_observation(str(n))], PROVENANCE)
    seen: list[int] = []

    def estimate(unit: MeasureUnit, observations: Sequence[Observation]) -> Estimate:
        seen.append(len(observations))
        return Estimate(status="unmeasured", unit_kind="none", reasons=("no_usable_observations",))

    packaged = PackagedTable.load(tmp_path / "none")
    assert MeasureSource([roots], packaged, estimate=estimate).get(UNIT) is None
    assert seen == [2]


@pytest.mark.contract
def test_an_unset_group_root_variable_is_no_root_and_blank_or_repeated_roots_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset variable is left out; the cache-root token is `cache_root()`; blank and repeated roots are refused."""
    monkeypatch.delenv("SENSELAB_MEASURES_ROOT", raising=False)
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    profile = ExecutionProfile.load("orcd")
    assert resolve_roots(profile.calibration.measures_roots, cache_root()) == (tmp_path / "cache" / "2x" / "measures",)
    monkeypatch.setenv("SENSELAB_MEASURES_ROOT", str(tmp_path / "group"))
    roots = MeasureRoots.from_profile(
        profile.calibration, POLICY, stale_after_s=profile.lease.stale_after_s, cache=cache_root()
    )
    assert roots.roots == (tmp_path / "group", tmp_path / "cache" / "2x" / "measures") and roots.trust == "group"
    assert resolve_roots(["${SENSELAB_MEASURES_ROOT}", str(tmp_path / "group")], cache_root()) == (tmp_path / "group",)
    for bad in (["a", "a"], ["/a", "  "], ["relative/root"], ["./x"]):
        with pytest.raises(ValueError, match="measures_roots"):
            ExecutionProfile.load("laptop", {"calibration": {"measures_roots": bad}})


@pytest.mark.contract
def test_an_empty_variable_and_a_relative_root_are_no_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty-valued variable is unset, so `${X}` is never the current directory and `${X}/m` never `/m`."""
    monkeypatch.setenv("SENSELAB_MEASURES_ROOT", "")
    own = str(tmp_path / "own")
    assert resolve_roots(["${SENSELAB_MEASURES_ROOT}", "${SENSELAB_MEASURES_ROOT}/m", own], tmp_path) == (Path(own),)
    monkeypatch.setenv("SENSELAB_MEASURES_ROOT", "rel")
    assert resolve_roots(["${SENSELAB_MEASURES_ROOT}", "~/m"], tmp_path) == (Path.home() / "m",)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    probes=st.sets(st.sampled_from("abcdef"), min_size=1),
    outdated=st.sets(st.sampled_from("abcdef")),
    trust=st.sampled_from(["group", "own"]),
    reader=st.sampled_from([UID, STRANGER]),
    split=st.integers(0, 1),
)
@pytest.mark.logic
def test_pooling_is_the_admitted_probes_less_the_outdated_whatever_the_root_split(  # noqa: PLR0913 - the strategy's draws
    tmp_path_factory: pytest.TempPathFactory,
    probes: set[str],
    outdated: set[str],
    trust: Trust,
    reader: int,
    split: int,
) -> None:
    """Over any placement of probes in two roots: a reader pools the probes of owners it admits, minus the outdated."""
    base = tmp_path_factory.mktemp("prop")
    first, second = base / "first", base / "second"
    writer = _roots(first, second)
    for n, probe in enumerate(sorted(probes)):
        _roots(*((first, second) if (n + split) % 2 else (second, first))).put_observations(
            UNIT, probe, [_observation(probe)], PROVENANCE
        )
    if outdated:
        writer.put_record(UNIT, "supersede", {"outdates": sorted(outdated)}, PROVENANCE)
    pooled = {o.item for o in _roots(first, second, trust=trust, uid=reader).observations(UNIT)}
    admitted = trust == "group" or reader == UID
    assert pooled == (probes - outdated if admitted else set())


_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "probe_measures_cross_node.py"


@pytest.mark.logic
def test_the_cross_node_verdict_reports_each_rounds_barrier_release_spread(tmp_path: Path) -> None:
    """The verdict JSON carries `max - min` of the writers' start times per round, to attribute a failed overlap."""
    spec = importlib.util.spec_from_file_location("probe_measures_cross_node", _SCRIPT)
    assert spec and spec.loader
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    events = [
        {"event": "wrote", "role": w, "round": r, "landed": True, "start_t": 10.0 * r + d, "end_t": 10.0 * r + d + 0.5}
        for r in range(2)
        for w, d in (("A", 0.0), ("B", 0.25), ("C", 1.0))
    ]
    report = script._check_writes(tmp_path, events, 3, 2)
    assert report["release_spread_s"] == [1.0, 1.0]
    assert report["overlapping_rounds"] == 0


@pytest.mark.contract
def test_a_file_over_the_policy_bound_is_malformed_not_a_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One byte over `measures_file_bytes_max` is skipped and logged; a reader of another uid skips and leaves it."""
    roots = _roots(tmp_path)
    written = roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    assert written is not None
    limit = POLICY.measures_file_bytes_max
    written.chmod(0o640)
    body = written.read_bytes()
    written.write_bytes(body + b" " * (limit - len(body)))
    assert roots.observations(UNIT) == (_observation(),)
    written.write_bytes(body + b" " * (limit + 1 - len(body)))
    with caplog.at_level(logging.WARNING, logger="senselab"):
        assert _roots(tmp_path, uid=STRANGER).observations(UNIT) == ()
        assert written.exists() and "skipped" in caplog.text
        assert roots.observations(UNIT) == ()
    assert not written.exists()


@pytest.mark.contract
def test_a_wrong_kind_entry_in_obs_is_malformed_and_silences_no_later_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A symlink planted in `obs/` is quarantined by its owner, not logged as a root read fault."""
    roots = _roots(tmp_path)
    written = roots.put_observations(UNIT, "p1", [_observation()], PROVENANCE)
    assert written is not None
    planted = written.parent / "p2.json"
    planted.symlink_to(written)
    with caplog.at_level(logging.WARNING, logger="senselab"):
        assert roots.observations(UNIT) == (_observation(),)
    assert not planted.is_symlink() and "cannot be read" not in caplog.text
    fail_at(monkeypatch, "p1.json", "read", OSError(errno.ESTALE, "stale"), creating=False)
    with caplog.at_level(logging.WARNING, logger="senselab"):
        assert roots.observations(UNIT) == ()
    assert "cannot be read" in caplog.text


@pytest.mark.contract
@needs_non_root
def test_the_local_clock_fallback_is_logged_once_per_root(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Judging a lease on the local clock in a root that cannot be written is said once."""
    holder = _roots(tmp_path).probe_lease(UNIT)
    assert isinstance(holder, Lease)
    with holder:
        subprocess.run(["chmod", "-R", "a-w", str(tmp_path)], check=True)
        try:
            reader = _roots(tmp_path)
            with caplog.at_level(logging.WARNING, logger="senselab"):
                reader.peer_holds(UNIT)
                reader.peer_holds(UNIT)
            assert caplog.text.count("local clock") == 1
        finally:
            subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=True)
