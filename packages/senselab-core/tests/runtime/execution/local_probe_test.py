"""Probing end to end on a local substrate: real `senselab execute` worker processes, a built host env, a text task.

Run 1 probes and writes observations; run 2, another run over the same measures root, plans no probe and reads run 1's
probe in its estimate; run 3, after the env's `wheel_set` changed, is another unit and probes again over run 1's
results, computing them; a run whose env is unbuilt when planned resolves at its worker's mark and probes.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from pool_probe_test import HOST, JOBS, _env, _plan, _registry, _unit, stack  # noqa: F401 - the autouse stack
from senselab_core.runtime.call import Packaged, packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import EventLog
from senselab_core.runtime.execution.executor import follow
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.substrates import substrate_of
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.units import UnitResolver, measure_roots

pytestmark = pytest.mark.contract

STACK = '''"""Makes every interpreter with this directory on PYTHONPATH read the probe test's one-task stack."""

import os
import platform

from pool_stack_fixture import host_env, registry, scoring
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.testing.workers import mit_lock_records

TEXTY = scoring("pool_probe_test.texty").model_copy(update={"inputs": {"text": "Text"}})


def _opener(backend, profile):
    base = host_env(backend, profile)
    wheel_set = os.environ["PROBE_WHEEL_SET"]
    return EnvHandle(base.python, platform.python_version(), base.lock_sha, wheel_set, base.torch_index)


CATALOGUE = Catalogue.load("format: catalogue@2.0\\nentries: []\\n")
use_packaged(Packaged(CATALOGUE, _opener, registry(TEXTY), mit_lock_records))
'''


@pytest.fixture
def workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Worker processes read the one-task stack, its env's `wheel_set` from `PROBE_WHEEL_SET`."""
    directory = tmp_path / "stack"
    directory.mkdir()
    (directory / "sitecustomize.py").write_text(STACK)
    here = Path(__file__).parent
    inherited = filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))
    paths = [str(directory), str(here), str(here.parent), *inherited]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("PROBE_WHEEL_SET", "4" * 64)
    yield directory


def _profile(tmp_path: Path, roots: str = "measures") -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 1}},
        "pool": {"poll_s": 1.0, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / roots)], "warmup_requests": 1},
    }
    return ExecutionProfile.load("laptop", override)


def _run(tmp_path: Path, profile: ExecutionProfile, name: str, store: ResultStore, unbuilt: bool = False) -> Pool:
    """Run the 60-text plan to its end on real workers as run `name`; return the pool, closed, and its results."""
    placed, values = _plan(profile, range(1, JOBS + 1))
    run_dir = tmp_path / name
    epoch, attempt, ordinal = prepare(placed, run_dir, store)
    roots = measure_roots(profile)
    pool = Pool(
        run_dir,
        queues=QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current),
        store=store,
        events=EventLog(run_dir, attempt, ordinal, epoch=epoch),
        profile=profile,
        substrate=substrate_of(profile, run_dir, placed.hash),
        epoch=epoch,
        roots=roots,
        resolver=UnitResolver(profile, roots),
    )
    try:
        handle = pool.submit(node_fragment(placed, pool.admit(placed, values)))
        pool.results = list(follow(pool, handle))  # type: ignore[attr-defined]
    finally:
        pool.close()
    return pool


def _needs_a_hardware_class() -> None:
    from senselab_core.runtime.call import host_hardware  # noqa: PLC0415 - one use

    if host_hardware().cpu_model is None:
        pytest.skip("this host names no CPU model, so no hardware class exists to probe under")


def test_probing_runs_on_real_workers_and_a_second_run_reads_the_probe(tmp_path: Path, workers: Path) -> None:
    """Run 1 probes; run 2 plans none and its estimate names run 1's probe; run 3 after a rebuild probes anew."""
    _needs_a_hardware_class()
    profile = _profile(tmp_path)
    placed, values = _plan(profile, range(1, JOBS + 1))
    with _env("4" * 64, _registry()):
        first_store = ResultStore(tmp_path / "store1")
        first = _run(tmp_path, profile, "run1", first_store)
        [probed] = first.probes.values()
        unit = _unit(first, placed, values)
        assert first.measures.get(unit) is not None and probed.items
        assert len(first.results) == JOBS and all(r.failure is None for r in first.results)  # type: ignore[attr-defined]
        second = _run(tmp_path, profile, "run2", ResultStore(tmp_path / "store2"))
        estimate = second.measures.get(unit)
        assert not second.probes and estimate is not None and probed.probe in estimate.source.probe_ids
        assert len(second.results) == JOBS  # type: ignore[attr-defined]
    os.environ["PROBE_WHEEL_SET"] = "5" * 64
    with _env("5" * 64, _registry()):
        third = _run(tmp_path, profile, "run3", first_store)
        rebuilt = _unit(third, placed, values)
    assert rebuilt.digest() != unit.digest() and rebuilt.env.wheel_set != unit.env.wheel_set
    [again] = third.probes.values()
    assert again.probe != probed.probe and again.items
    assert len(third.results) == JOBS and all(r.served is None for r in third.results)  # type: ignore[attr-defined]


def test_an_env_unbuilt_when_planned_resolves_at_its_workers_mark_and_probes(tmp_path: Path, workers: Path) -> None:
    """The coordinator reads no env, so the unit waits; the worker builds it, its mark resolves it and it probes."""
    _needs_a_hardware_class()
    profile = _profile(tmp_path, "measures-unbuilt")
    mit = packaged().lock_records
    opener = packaged().open_env

    def unbuilt(backend: object, profile: object) -> None:  # noqa: ARG001 - an EnvRead
        return None

    previous = use_packaged(
        Packaged(Catalogue.load("format: catalogue@2.0\nentries: []\n"), opener, _registry(), mit, env_read=unbuilt)
    )
    try:
        pool = _run(tmp_path, profile, "run-unbuilt", ResultStore(tmp_path / "store-unbuilt"))
    finally:
        use_packaged(previous)
    [probed] = pool.probes.values()
    assert probed.items and len(pool.results) == JOBS  # type: ignore[attr-defined]
    assert sys.executable and HOST
