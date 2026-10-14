"""Tests for sharded steps: one key per shard, the work-pool hand-off, dead shards and the gather step.

``sbatch`` is faked and the pool is drained by hand, except in the sandboxed gather test, which runs a
sharded step and its gather step locally in a synthetic site.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from senselab.utils import work_pool
from senselab.utils.reproduce import cache, run, slurm
from senselab.utils.reproduce.cache import CacheError
from senselab.utils.reproduce.dag import DagError, toposort
from senselab.utils.reproduce.spec import Manifest, check_step_profile
from tests.utils.reproduce.run_test import World, needs_userns
from tests.utils.reproduce.slurm_test import (
    CLASSES,
    DATE,
    PATHS,
    POOL_SETTINGS,
    PROBE_OK,
    FakeSbatch,
    _config,
    _manifest,
    _profile,
    _rows,
    _sbatch_lines,
    _spec,
    _step,
)

SHA = "ab" * 32
SHARDS = {"count": 3, "resource_class": "small"}


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _sharded(tmp: Path) -> tuple[list[Manifest], slurm.SubmitConfig]:
    profile = _profile(tmp, resource_classes=CLASSES, work_pool=POOL_SETTINGS)
    steps = [_step("prep"), _step("parts", needs=["prep"], shards=SHARDS), _step("gather", needs=["parts"])]
    return [_manifest("alpha", *steps)], _config(tmp, PATHS[:1], profile=profile)


def _run_config(config: slurm.SubmitConfig, manifests: list[Manifest]) -> run.RunConfig:
    sources = {m.name: m.sources for m in manifests}
    return run.RunConfig(
        profile=config.profile, cache=config.cache, base_dir=config.base_dir, date=DATE, sources=sources
    )


def _commands(config: slurm.SubmitConfig) -> list[list[str]]:
    base = slurm.run_argv(config)
    return [slurm.shard_command(base, "parts", i, "/opt/py/bin/python") for i in range(3)]


def test_a_sharded_step_publishes_one_item_per_shard_to_its_class(tmp_path: Path) -> None:
    """Submit makes the pool and a held pool job afterok the needs; that job publishes one item per shard."""
    manifests, config = _sharded(tmp_path)
    fake = FakeSbatch((0, PROBE_OK))
    submission = slurm.submit(manifests, config, runner=fake)
    assert [(j.step, j.job_id) for j in submission.jobs] == [("prep", "101"), ("parts", "102"), ("gather", "103")]
    pool_script, gather_script = fake.submissions()[1:3]
    flags = _sbatch_lines(pool_script)
    assert "--job-name=reproduce-parts-pool" in flags and "--partition=quick" in flags
    assert "--time=02:00:00" in flags and "--dependency=afterok:101" in flags
    assert pool_script.splitlines()[-1].endswith("--submission sub1 --pool parts")
    assert "--dependency=afterok:102" in _sbatch_lines(gather_script)
    root = tmp_path / "runs" / DATE.isoformat() / "alpha" / "pools" / "parts"
    assert json.loads((root / "pool.json").read_text())["lease_s"] == 30.0
    assert {r["step"]: r["job"] for r in _rows(tmp_path, "alpha")}["parts"] == "102"

    seen: dict[str, Any] = {}

    def drain(pool: work_pool.WorkPool, classes: list[work_pool.ResourceClass]) -> None:
        seen["status"] = pool.status()["classes"]
        seen["items"] = sorted(os.listdir(pool.root / "queue"))
        seen["commands"] = [json.loads((pool.root / "queue" / n).read_text())["command"] for n in seen["items"]]
        seen["classes"] = classes

    commands = _commands(config)
    outcome = slurm.run_pool(manifests, _run_config(config, manifests), "parts", "sub1", commands, drain=drain)
    assert seen["status"] == {"small": {"queued": 3, "leased": 0, "done": 0, "dead": 0}}
    assert seen["items"] == ["sub1-0.json", "sub1-1.json", "sub1-2.json"]
    for i, command in enumerate(seen["commands"]):
        assert command[:4] == ["/opt/py/bin/python", "-P", "-m", "senselab.utils.reproduce"]
        assert command[-6:] == ["--submission", "sub1", "--step", "parts", "--shard", str(i)]
    [cls] = seen["classes"]
    assert cls.name == "small" and [p.partition for p in cls.partitions] == ["main", "spare"]
    assert cls.partitions[1].exclude == ("n[1-2]",) and cls.partitions[1].requeue and cls.partitions[0].max_workers == 2
    assert outcome.status == "failed" and outcome.codes == ("E_SHARD_UNFINISHED",)
    again = slurm.run_pool(manifests, _run_config(config, manifests), "parts", "sub1", commands, drain=drain)
    assert again.codes == ("E_SHARD_UNFINISHED",) and len(seen["items"]) == 3

    unknown = _profile(tmp_path, resource_classes=CLASSES, work_pool=POOL_SETTINGS)
    bad = _spec("parts", shards={"count": 2, "resource_class": "nowhere"})
    assert [f.code for f in slurm.step_findings(bad, unknown)] == ["resource-class-unknown"]
    with pytest.raises(ValueError, match="E_SPEC_RESOURCE_CLASS"):
        check_step_profile(bad, unknown)
    no_pool = _profile(tmp_path, resource_classes=CLASSES)
    assert [f.code for f in slurm.step_findings(_spec("parts", shards=SHARDS), no_pool)] == ["pool-unset"]
    with pytest.raises(ValueError, match="reserved"):
        _spec("pools")
    reading = _step("sum", needs=["parts"], inputs=[{"root": "up:parts", "patterns": ["*.json"]}])
    with pytest.raises(DagError, match="E_DAG_SHARD_NEED: sum"):
        toposort(_manifest("alpha", _step("parts", shards=SHARDS), reading).steps)


def test_each_shard_has_its_own_key(tmp_path: Path) -> None:
    """The shard index and count are keyed; a sharded step is keyed only per shard, a plain step never."""
    profile = _profile(tmp_path, resource_classes=CLASSES, work_pool=POOL_SETTINGS)
    step = _spec("parts", shards=SHARDS)
    kwargs: dict[str, Any] = {
        "upstream": {},
        "submodules": {},
        "adapter_sha256": SHA,
        "adapter_env_sha256": SHA,
        "input_digests": [SHA],
        "native": {},
        "sources_lock_sha256": SHA,
    }
    keys = [cache.step_key(step, profile, shard=i, **kwargs) for i in range(3)]
    assert len(set(keys)) == 3
    assert cache.step_key_payload(step, profile, shard=1, **kwargs)["shard"] == [1, 3]
    four = _spec("parts", shards={**SHARDS, "count": 4})
    assert cache.step_key(four, profile, shard=0, **kwargs) != keys[0]
    for bad_step, shard in ((step, None), (step, 3), (step, -1), (_spec("plain"), 0)):
        with pytest.raises(CacheError, match="E_KEY_SHARD"):
            cache.step_key(bad_step, profile, shard=shard, **kwargs)
    assert cache.step_key_payload(_spec("plain"), profile, **kwargs)["shard"] is None
    assert cache.shards_key(keys) != cache.shards_key(list(reversed(keys)))


def _fail_or_complete(dead: set[int]) -> Any:  # noqa: ANN401
    def drain(pool: work_pool.WorkPool, classes: list[work_pool.ResourceClass]) -> None:
        while (lease := pool.claim([c.name for c in classes], "w")) is not None:
            if int(lease.item.id.rsplit("-", 1)[1]) in dead:
                pool.fail(lease, 1)
            else:
                pool.complete(lease, {"exit_code": 0})

    return drain


def test_a_dead_shard_fails_the_step_and_its_dependants_never_run(tmp_path: Path) -> None:
    """A shard that spent its failure budget fails the step with ``E_SHARD_DEAD``; its dependants never run."""
    manifests, config = _sharded(tmp_path)
    slurm.submit(manifests, config, runner=FakeSbatch((0, PROBE_OK)))
    rconfig = _run_config(config, manifests)
    outcome = slurm.run_pool(manifests, rconfig, "parts", "sub1", _commands(config), drain=_fail_or_complete({1}))
    pool = work_pool.WorkPool(tmp_path / "runs" / DATE.isoformat() / "alpha" / "pools" / "parts")
    assert [pool.state(f"sub1-{i}") for i in range(3)] == ["done", "dead", "done"]
    assert outcome.status == "failed" and outcome.codes == ("E_SHARD_DEAD",) and not outcome.ok
    rows = {r["step"]: r for r in _rows(tmp_path, "alpha")}
    assert (rows["parts"]["status"], rows["parts"]["codes"], rows["parts"]["job"]) == (
        "failed",
        ["E_SHARD_DEAD"],
        "102",
    )
    assert rows["parts"]["exit"] != 0
    assert (rows["gather"]["status"], rows["gather"]["codes"]) == ("never-ran", ["E_SHARD_DEAD"])
    assert rows["prep"]["status"] == "pending"


GATHER = """\
import json, os, sys
up = sys.argv[2]
names = sorted(os.listdir(up))
values = [json.load(open(os.path.join(up, n, "result.json")))["value"] for n in names]
blocked = []
for target in (os.path.join(up, "0", "x.txt"), os.path.join(up, "new.txt")):
    try:
        open(target, "w").write("x")
        blocked.append(False)
    except OSError:
        blocked.append(True)
open(sys.argv[1] + "/seen.json", "w").write(json.dumps({"names": names, "blocked": blocked}))
open(sys.argv[1] + "/result.json", "w").write(json.dumps({"value": sum(values)}))
"""

PART = """\
import json, os, sys
value = 10 * int(os.environ["REPRO_SHARD"]) + int(os.environ["REPRO_SHARDS"])
open(sys.argv[1] + "/result.json", "w").write(json.dumps({"value": float(value)}))
"""


def _escaped(code: str) -> str:
    return code.replace("{", "{{").replace("}", "}}")


@needs_userns
def test_the_gather_step_sees_every_shard_read_only(world: World) -> None:
    """Each shard runs with its own key and index; the gather step reads every shard, read-only, in ``{up:<step>}``."""
    world.profile_extra = {"resource_classes": CLASSES, "work_pool": POOL_SETTINGS}
    part = world.step("parts", "prep", "", argv=["{venv}/bin/python", "-c", _escaped(PART), "{out}"], shards=SHARDS)
    gather = world.step(
        "gather", "prep", "", argv=["{venv}/bin/python", "-c", _escaped(GATHER), "{out}", "{up:parts}"], needs=["parts"]
    )
    manifest = world.manifest(part, gather)
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    parts = outcomes["parts"]
    assert parts.status == "executed" and parts.out is None and len(parts.shards) == 3
    keys = [p.name for p in parts.shards]
    assert len(set(keys)) == 3 and parts.key == cache.shards_key(keys)
    values = [json.loads((p / run.WORK / "result.json").read_text())["value"] for p in parts.shards]
    assert values == [3.0, 13.0, 23.0]
    assert outcomes["gather"].out is not None
    seen = json.loads((outcomes["gather"].out / run.WORK / "seen.json").read_text())
    assert seen == {"names": ["0", "1", "2"], "blocked": [True, True]}
    assert json.loads((outcomes["gather"].out / run.WORK / "result.json").read_text())["value"] == 39.0
    assert not (outcomes["gather"].out / run.TMP).exists()
    day = world.tmp / "runs" / world.config().date.strftime("%Y-%m-%d") / "demo"
    assert [os.path.realpath(day / "parts" / str(i)) for i in range(3)] == [os.path.realpath(p) for p in parts.shards]
    rows = {r["step"]: r for r in json.loads((day / run.RUN_JSON).read_text())["steps"]}
    assert rows["parts"]["key"] == parts.key and rows["parts"]["shards"] == keys
    again = world.run(manifest)
    assert [again[n].status for n in ("parts", "gather")] == ["skipped", "skipped"]
    assert again["parts"].key == parts.key
