"""The restricted purge and results a run may still own: in-flight runs, finished shards and live work pools.

The sequence these tests rebuild: a public sharded step runs through a work pool while ``verify-inputs`` of a
restricted id purges its closure (`restricted.purge_closure`). The purge looks at a shard result before the shard
has written its summary, waits for the shard's claim, and reads the summary again under the claim. On shared
storage a host that looked up a missing file keeps getting "no such file" for a while after another host creates
it, until it lists the directory (`StaleHost`). The tests also cover a held claim, a shard between its attempts in a
pool with a running controller, results with no recorded closure, and the verify line.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils import work_pool
from senselab.utils.reproduce import cache, cli, restricted, run
from tests.utils.reproduce.restricted_test import Site
from tests.utils.reproduce.run_test import needs_userns, write_manifest_file
from tests.utils.reproduce.slurm_test import CLASSES, POOL_SETTINGS
from tests.utils.reproduce.sources_fixture import verify_input


@pytest.fixture
def site(tmp_path: Path) -> Site:
    """A fresh synthetic site with a restricted dir."""
    return Site(tmp_path)


def _result(root: Path, name: str, closure: list[str] | None, *, store: bool = False) -> Path:
    """A public result dir; with ``closure`` a summary recording it, else no summary (in flight, or died)."""
    out = root / hashlib.sha256(name.encode()).hexdigest()
    (out / run.WORK).mkdir(parents=True)
    (out / run.WORK / "x.txt").write_text("1\n")
    if store:
        (out / run.WORK / "items").mkdir()
        (out / run.WORK / "items" / "0.done").write_text("")
    if closure is not None:
        _write_summary(out, closure)
    return out


def _write_summary(out: Path, closure: list[str]) -> None:
    cache.write_summary(out, {"exit": 0, "codes": [], "executed": True, "input_closure": closure})


class StaleHost:
    """This host's view of some result dirs: their summary lookups say "no such file" until the dir is listed.

    With ``listing_stale`` the listing itself also leaves the summary out, so nothing refreshes the view.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *outs: Path, listing_stale: bool = False) -> None:
        """Install the view on `restricted`'s summary reads and dir listings only."""
        self.stale = {str(o) for o in outs}
        self.listing_stale = listing_stale
        real_state = restricted._closure_state
        real_os = restricted.os

        def lookup(out: Path) -> list[str] | str:
            return restricted._NO_SUMMARY if str(out) in self.stale else real_state(out)

        def listdir(path: Any) -> list[str]:  # noqa: ANN401
            names = real_os.listdir(path)
            if str(path) in self.stale:
                if self.listing_stale:
                    return [n for n in names if n != cache.SUMMARY]
                self.stale.discard(str(path))
            return names

        class _Os:
            def __getattr__(self, name: str) -> Any:  # noqa: ANN401
                return listdir if name == "listdir" else getattr(real_os, name)

        monkeypatch.setattr(restricted, "_closure_state", lookup)
        monkeypatch.setattr(restricted, "os", _Os())


def _hold(root: Path, out: Path, *, write: list[str] | None, hold_s: float) -> threading.Thread:
    """A run holding ``out``'s claim for ``hold_s`` seconds, writing a summary with ``write`` before it releases."""
    started = threading.Event()

    def holder() -> None:
        with cache.claim(root, out.name):
            started.set()
            time.sleep(hold_s)
            if write is not None:
                _write_summary(out, write)

    thread = threading.Thread(target=holder)
    thread.start()
    assert started.wait(10)
    return thread


@pytest.fixture
def running_controller(site: Site) -> Iterator[Path]:
    """A work pool under the runs root whose controller lock is held by another process (a running controller)."""
    root = site.tmp / "runs" / "2026-01-02" / "demo" / "pools" / "parts"
    pool = work_pool.WorkPool.create(root, lease_s=30.0, heartbeat_s=5.0)
    pool.publish([work_pool.Item(id="sub-0", resource_class="small", command=("true",))])
    lock = root / "controller" / "lock"
    lock.parent.mkdir(exist_ok=True)
    code = (
        "import fcntl, sys; f = open(sys.argv[1], 'a'); fcntl.lockf(f, fcntl.LOCK_EX); "
        "print('held', flush=True); sys.stdin.read()"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdout is not None and proc.stdout.readline().strip() == b"held"
    try:
        yield root
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(10)


# --- a shard that finishes while the purge looks at it ------------------------------------------------------


def test_a_shard_that_finishes_while_the_purge_waits_is_kept(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """No summary at the first look; the shard writes one (closure without the id) and releases; the purge keeps it.

    This host's lookups are stale until it lists the dir, which the purge does under the claim.
    """
    profile = site.profile()
    root = profile.results_root
    shard = _result(root, "shard-a", None)
    holder = _hold(root, shard, write=["other"], hold_s=1.0)
    StaleHost(monkeypatch, shard)
    result = restricted.purge_closure(profile, "flip")
    holder.join()
    assert shard.is_dir() and (shard / run.WORK / "x.txt").is_file()
    assert result == restricted.Purge(0, 0)


def test_a_contended_claim_keeps_a_result_this_host_cannot_see(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """As above, but even the listing leaves the new summary out: the contended claim alone keeps the result."""
    profile = site.profile()
    root = profile.results_root
    shard = _result(root, "shard-b", None)
    holder = _hold(root, shard, write=["other"], hold_s=1.0)
    StaleHost(monkeypatch, shard, listing_stale=True)
    result = restricted.purge_closure(profile, "flip")
    holder.join()
    assert shard.is_dir()
    assert result == restricted.Purge(0, 1)


def test_a_finished_shard_seen_stale_after_its_release_is_kept(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finished and released before the purge came; this host's lookup is stale until the listing; kept."""
    profile = site.profile()
    shard = _result(profile.results_root, "shard-c", ["other"])
    StaleHost(monkeypatch, shard)
    result = restricted.purge_closure(profile, "flip")
    assert shard.is_dir()
    assert result == restricted.Purge(0, 0)


def test_a_shard_between_attempts_keeps_its_store_while_its_controller_runs(
    site: Site, running_controller: Path
) -> None:
    """A shard result with no summary and a resume store, while a pool controller runs: kept and counted."""
    profile = site.profile()
    shard = _result(profile.results_root, "interrupted", None, store=True)
    result = restricted.purge_closure(profile, "flip")
    assert (shard / run.WORK / "items" / "0.done").is_file()
    assert result == restricted.Purge(0, 1)
    assert restricted.live_pools(profile.runs_root) == 1


def test_a_held_claim_is_counted_in_flight_not_an_error(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A claim held past the timeout is skipped and counted; the other results are purged; no ``E_CLAIM_HELD``."""
    profile = site.profile()
    root = profile.results_root
    held = _result(root, "held", None)
    done = _result(root, "done", ["flip"])
    monkeypatch.setattr(restricted, "PURGE_CLAIM_TIMEOUT_S", 0.5)
    holder = _hold(root, held, write=None, hold_s=2.0)
    try:
        result = restricted.purge_closure(profile, "flip")
    finally:
        holder.join()
    assert held.is_dir() and not done.exists()
    assert result == restricted.Purge(1, 1)


# --- what the purge must still remove -----------------------------------------------------------------------


def test_results_without_a_closure_are_purged_while_a_controller_runs(site: Site, running_controller: Path) -> None:
    """A finished summary with no ``input_closure`` cannot say what it read: purged even while a pool is live."""
    profile = site.profile()
    old = _result(profile.results_root, "old", None)
    (old / cache.SUMMARY).write_text(json.dumps({"exit": 0}))
    assert restricted.purge_closure(profile, "flip") == restricted.Purge(1, 0) and not old.exists()


def test_a_closure_seen_with_the_id_is_purged_even_if_the_second_read_fails(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first read shows the id; the read under the claim is unreadable (e.g. a stale handle): purged."""
    profile = site.profile()
    out = _result(profile.results_root, "holds", ["flip"])
    real = restricted._fresh_closure_state
    monkeypatch.setattr(restricted, "_fresh_closure_state", lambda o: restricted._UNREADABLE if o == out else real(o))
    assert restricted.purge_closure(profile, "flip") == restricted.Purge(1, 0) and not out.exists()


def test_between_summary_and_release_the_fresh_summary_decides(site: Site) -> None:
    """The run writes its summary, then holds the claim; the purge reads it under the claim: kept, or purged."""
    profile = site.profile()
    root = profile.results_root
    kept = _result(root, "kept", None)
    gone = _result(root, "gone", None)
    first = _hold(root, kept, write=["other"], hold_s=0.5)
    second = _hold(root, gone, write=["flip"], hold_s=0.5)
    result = restricted.purge_closure(profile, "flip")
    first.join()
    second.join()
    assert kept.is_dir() and not gone.exists() and result == restricted.Purge(1, 0)


def test_finished_results_with_the_id_are_purged_once(site: Site) -> None:
    """Done results whose closure holds the id go, with their run links; a dead run's dir goes; a second call is 0."""
    profile = site.profile()
    root = profile.results_root
    holds = _result(root, "holds", ["flip", "other"])
    dead = _result(root, "dead", None)
    other = _result(root, "other", ["other"])
    analysis = profile.runs_root / "2026-01-02" / "demo"
    (analysis / "parts").mkdir(parents=True)
    (analysis / "parts" / "0").symlink_to(holds)
    assert restricted.purge_closure(profile, "flip") == restricted.Purge(2, 0)
    assert not holds.exists() and not dead.exists() and other.is_dir()
    assert not (analysis / "parts" / "0").is_symlink()
    assert restricted.purge_closure(profile, "flip") == restricted.Purge(0, 0)


def test_a_pool_without_a_running_controller_is_not_live(site: Site) -> None:
    """A pool left with queued items and no controller (cancelled) is not live, so a dead run's dir is purged."""
    profile = site.profile()
    root = site.tmp / "runs" / "2026-01-02" / "demo" / "pools" / "parts"
    pool = work_pool.WorkPool.create(root, lease_s=30.0, heartbeat_s=5.0)
    pool.publish([work_pool.Item(id="sub-0", resource_class="small", command=("true",))])
    (root / "controller").mkdir(exist_ok=True)
    (root / "controller" / "lock").touch()
    dead = _result(profile.results_root, "dead", None)
    assert restricted.live_pools(profile.runs_root) == 0
    assert restricted.purge_closure(profile, "flip") == restricted.Purge(1, 0) and not dead.exists()


# --- the verify line ----------------------------------------------------------------------------------------


def test_verify_inputs_reports_skipped_in_flight_results(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A restricted verify while a public run holds its claim: ``ok … purged=0 skipped_inflight=1``."""
    site.register_canary()
    verify_input(site.base, site.profile(), "canary")
    profile = site.profile()
    root = profile.results_root
    flying = _result(root, "flying", None)
    profile_file = site.tmp / "profile.yaml"
    profile_file.write_text(yaml.safe_dump(profile.model_dump(mode="json")))
    monkeypatch.setattr(restricted, "PURGE_CLAIM_TIMEOUT_S", 0.5)
    holder = _hold(root, flying, write=None, hold_s=2.0)
    capsys.readouterr()
    try:
        cli.main(["verify-inputs", "--profile", str(profile_file), "--base-dir", str(site.base), "canary"])
    finally:
        holder.join()
    line = capsys.readouterr().out.strip()
    assert line.startswith("canary\tok\t") and line.endswith(" purged=0 skipped_inflight=1"), line
    assert flying.is_dir()


# --- every shard summary carries its closure ----------------------------------------------------------------


@needs_userns
def test_every_shard_summary_records_its_input_closure(site: Site) -> None:
    """A public sharded step's shard summaries each carry ``input_closure``, so the purge can always decide."""
    site.profile_extra = {"resource_classes": CLASSES, "work_pool": POOL_SETTINGS}
    part = site.step("parts", "prep", "{root:data}/in/a.txt", shards={"count": 2, "resource_class": "small"})
    manifest, ids = write_manifest_file(site, site.manifest(part))
    site.prefetch(manifest)
    outcome = site.run(manifest, manifest_ids=ids)["parts"]
    assert outcome.ok and len(outcome.shards) == 2, outcome.codes
    for out in outcome.shards:
        assert json.loads((out / cache.SUMMARY).read_text())["input_closure"] == []
