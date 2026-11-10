"""The scale ladder's trees, flatness checks and results."""

import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import soundfile
from senselab_core.kernel.dataset import ManifestRow
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Registry
from senselab_core.runtime import fake_backend
from senselab_core.runtime.calibration import HardwareClass, Observation, ProbeRef
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution import pool_parts
from senselab_core.runtime.execution.executor import FaultRecord, JobRef
from senselab_core.runtime.execution.queue import probe_queue
from senselab_core.runtime.measures import Build, MeasureRoots, MeasureSource, MeasureUnit, Provenance
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.sizing import Sample
from senselab_core.runtime.storage import StorePolicy
from senselab_core.testing.calibration_samples import UNIT, observation
from senselab_core.testing.workers import mit_lock_records

ROOT = Path(__file__).resolve().parents[4]


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bids_tree = _module("scale_bids_tree_under_test", ROOT / "benchmarks" / "scale" / "bids_tree.py")

TREE_DIGEST = "d9fabd59a9907c1ffd7afea617a5327ba9deb56d5be2947e7a80d49a9e63ebb9"
"""The digest of `synthetic(root, 3, 0)`: its relative paths and bytes, the same on every host."""


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


@pytest.mark.logic
def test_synthetic_tree_is_byte_identical_for_a_seed(tmp_path: Path) -> None:
    """A seed gives the pinned bytes twice; another seed differs; every tone decodes at the tree's rate and lengths."""
    first, again = bids_tree.synthetic(tmp_path / "a", 3, 0), bids_tree.synthetic(tmp_path / "b", 3, 0)
    assert _digest(first) == _digest(again) == TREE_DIGEST
    assert _digest(bids_tree.synthetic(tmp_path / "c", 3, 1)) != TREE_DIGEST
    tones = sorted(first.rglob("*_audio.wav"))
    assert [t.relative_to(first).parts[0] for t in tones] == ["sub-0", "sub-1", "sub-2"]
    for tone in tones:
        info = soundfile.info(tone)
        assert info.samplerate == bids_tree.RATE and bids_tree.SECONDS[0] <= info.duration <= bids_tree.SECONDS[1]
    [long] = bids_tree.synthetic(tmp_path / "long", 1, 0, seconds=7).rglob("*_audio.wav")
    assert soundfile.info(long).duration == 7
    with pytest.raises(ValueError, match="at least one"):
        bids_tree.synthetic(tmp_path / "none", 0, 0)


@pytest.mark.logic
def test_no_two_synthetic_tones_share_content(tmp_path: Path) -> None:
    """A hundred tones are a hundred contents, so a point of n items runs n jobs, none sharing another's key."""
    tones = sorted(bids_tree.synthetic(tmp_path / "t", 100, 0).rglob("*_audio.wav"))
    assert len({hashlib.sha256(t.read_bytes()).hexdigest() for t in tones}) == len(tones) == 100


@pytest.mark.logic
def test_librispeech_utterances_are_laid_out_as_bids(tmp_path: Path) -> None:
    """The first n utterances, in archive order, go to `sub-<speaker>` with a run per speaker, bytes unchanged."""
    archive = tmp_path / "test-clean.tar.gz"
    names = [
        "61/70968/61-70968-0000",
        "61/70968/61-70968-0001",
        "121/121726/121-121726-0000",
        "121/121726/121-121726-0001",
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for index, name in enumerate(names):
            data = f"flac {index}".encode()
            member = tarfile.TarInfo(f"LibriSpeech/test-clean/{name}.flac")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    root = bids_tree.lay_librispeech(tmp_path / "bids", archive, 3)
    laid = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.flac")}
    assert laid == {
        "sub-61/ses-01/audio/sub-61_ses-01_task-read_run-1_audio.flac": b"flac 0",
        "sub-61/ses-01/audio/sub-61_ses-01_task-read_run-2_audio.flac": b"flac 1",
        "sub-121/ses-01/audio/sub-121_ses-01_task-read_run-1_audio.flac": b"flac 2",
    }
    with pytest.raises(ValueError, match="fewer than 5"):
        bids_tree.lay_librispeech(tmp_path / "short", archive, 5)


PROFILE = ExecutionProfile.load("laptop")
CONFIG = PROFILE.calibration
COMMIT = "a" * 40
PROVENANCE = Provenance(
    job="1234", run="run-1", senselab=Build(version="2.0.0", commit=COMMIT), profile="d" * 64, hardware=UNIT.hardware
)
DECLARED = {(UNIT.task, UNIT.backend)}


def _line(
    units: Sequence[float], probe: str = "p1", count: int | None = None, hardware: HardwareClass = UNIT.hardware
) -> list[Observation]:
    """`count` (default `min_requests`) warm batch-1 requests of each length, one stratum per length."""
    return [
        observation(
            probe=ProbeRef(id=probe, phase="SINGLE", stratum=i),
            item=f"{probe}-{i}-{n}",
            units=x,
            padded_units=x,
            wall_s=0.1 + 0.01 * x,
            since_load=CONFIG.warmup_requests,
            hardware=hardware,
        )
        for i, x in enumerate(units)
        for n in range(CONFIG.min_requests if count is None else count)
    ]


def _roots(path: Path) -> MeasureRoots:
    return MeasureRoots(
        [path], StorePolicy.load_default(), "group", os.geteuid(), stale_after_s=PROFILE.lease.stale_after_s
    )


def _write(
    path: Path, unit: MeasureUnit, observations: Sequence[Observation], provenance: Provenance = PROVENANCE
) -> None:
    probes = {o.probe.id for o in observations if o.probe is not None}
    for probe in sorted(probes):
        mine = [o for o in observations if o.probe is not None and o.probe.id == probe]
        assert _roots(path).put_observations(unit, probe, mine, provenance) is not None


merge = _module("merge_measures_under_test", ROOT / "scripts" / "merge_measures.py")


def _reads(root: Path) -> MeasureSource:
    roots = _roots(root)
    return MeasureSource(
        [roots],
        estimate=Probe.estimator(
            CONFIG, records_of=roots.regime_records, memory_headroom=PROFILE.sizing.memory_headroom
        ),
    )


@pytest.mark.logic
def test_merged_ladder_points_are_a_root_the_source_reads_on_an_identity_match(tmp_path: Path, capsys: Any) -> None:  # noqa: ANN401
    """The script merges a ladder run's points into one root: its unit is measured there, another unit is not."""
    run, dest = tmp_path / "run", tmp_path / "group"
    other = UNIT.model_copy(update={"env": UNIT.env.model_copy(update={"wheel_set": "f" * 64})})
    _write(run / "t~b@100" / "measures", UNIT, _line([2.0, 8.0], "p1"))
    _write(run / "t~b@1000" / "measures", UNIT, _line([3.0, 9.0], "p2"))
    oom = {"resource": "host", "batch": 4, "length": 8.0, "cotenancy": 1, "grant_bytes": 1024, "basis": ["p1"]}
    assert _roots(run / "t~b@100" / "measures").put_record(UNIT, "oom", oom, PROVENANCE) is not None
    assert merge.main([str(run), "--out", str(dest)]) == 0
    assert f"unit {UNIT.label()}: 2 probes, 1 records" in capsys.readouterr().out
    assert [r.kind for r in _roots(dest).records(UNIT)] == ["oom"]
    source = _reads(dest)
    found = source.get(UNIT)
    assert found is not None and found.source.probe_ids == ("p1", "p2")
    assert source.get(other) is None
    assert merge.main([str(run), "--out", str(dest)]) == 0
    assert f"unit {UNIT.label()}: 0 probes, 0 records" in capsys.readouterr().out
    assert _reads(dest).get(UNIT) == found
    assert merge.main([str(tmp_path / "empty"), "--out", str(dest)]) == 1


@pytest.mark.logic
def test_flatness_covers_time_memory_files_and_bytes_against_budgets() -> None:
    """Each measure's value at 10⁴ over at 10² is judged against `flat_ratio`, its value at 10⁴ against its ceiling.

    Without a 10⁴ point the ceiling reads 10², and without either the largest point; a point with no per-item
    seconds enters no ratio and its ceiling reads n/a.
    """
    ladder = _module("scale_ladder_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")

    def point(at: int, wall: float | None, peak: int, files: int, written: int) -> dict[str, Any]:
        keys = ("per_item_s", "peak_rss_bytes", "files_written", "bytes_written")
        return {"task": "t", "point": at, "items": at, **dict(zip(keys, (wall, peak, files, written), strict=True))}

    results = [point(100, 1.0, 1000, 200, 10_000), point(10_000, 1.2, 1000, 30_000, 1_000_000)]
    budgets = {"flat_ratio": 1.25, "per_item": {"t": {"per_item_s": 1.1, "peak_rss_bytes": None}}}
    assert ladder.flatness(results, budgets) == [
        "t per_item_s: at 10000 / at 100 = 1.200, flat ok, ceiling EXCEEDED (1.1) (at 10000)",
        "t peak_rss_bytes: at 10000 / at 100 = 1.000, flat ok, ceiling unmeasured (at 10000)",
        "t files_per_item: at 10000 / at 100 = 1.500, flat EXCEEDED (1.25), ceiling unmeasured (at 10000)",
        "t bytes_per_item: at 10000 / at 100 = 1.000, flat ok, ceiling unmeasured (at 10000)",
    ]
    assert all("unmeasured" in line for line in ladder.flatness(results, {})[:1])
    probed = [point(1, 9.0, 1000, 20, 100), point(100, None, 1000, 200, 10_000)]
    assert (
        ladder.flatness(probed, budgets)[0] == "t per_item_s: at 10000 / at 100 = n/a, flat n/a, ceiling n/a (at 100)"
    )
    assert ladder.flatness(probed[:1], budgets)[0] == (
        "t per_item_s: at 10000 / at 100 = n/a, flat n/a, ceiling EXCEEDED (1.1) (at 1)"
    )


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.mark.logic
def test_a_suite_with_entries_that_do_not_plan_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry is planned first: each bad entry is named, a catalogued model plans, and nothing is written."""
    ladder = _module("scale_ladder_refusal_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    known = CatalogueEntry.model_validate(
        {
            "id": "org/known",
            "revision": "c" * 40,
            "tasks": [fake_backend.ECHO.name],
            "backends": [fake_backend.FAKE.name],
            "stacks": [fake_backend.FAKE.stack],
            "license": {
                "expression": "MIT",
                "source": "card",
                "verified": {"date": "2026-11-07", "how": "hf-card"},
                "gated": False,
            },
        }
    )
    previous = use_packaged(Packaged(Catalogue(entries=(known,)), _this_interpreter, REGISTRY, mit_lock_records))
    assert fake_backend.FAKE in REGISTRY.backends(fake_backend.ECHO.name)
    suite = tmp_path / "bad.yaml"
    suite.write_text(
        "format: scale-suite@1.0\nname: bad\nseed: 0\nlong_s: 1\ntasks:\n"
        "- {task: fake.echo, port: payload, data: payloads, points: [1, 100]}\n"
        "- {task: fake.echo, stack: absent, port: payload, data: payloads, points: [1]}\n"
        "- {task: no.such.task, port: x, data: payloads, points: [1]}\n"
        "- {task: fake.echo, model: org/known, port: payload, data: payloads, points: [1]}\n"
        "- {task: fake.echo, model: org/unknown, port: payload, data: payloads, points: [1]}\n"
    )
    try:
        with pytest.raises(ladder.SuiteRefused, match="3 entries do not fetch or plan") as refused:
            ladder.run(suite, ExecutionProfile.load("laptop"))
        assert ladder.pinned("org/known") == "org/known@" + "c" * 40
    finally:
        use_packaged(previous)
    lines = str(refused.value).splitlines()[1:]
    named = ["entry 1 (fake.echo, stack absent)", "entry 2 (no.such.task)", "entry 4 (fake.echo)"]
    assert [line.split(":")[0] for line in lines] == named
    assert "NoBackend" in lines[0] and "UnknownTask" in lines[1] and "UnknownModel" in lines[2]
    assert not list((tmp_path / "cache").rglob(ladder.commit()))
    runs = tmp_path / "cache" / "2x" / "runs"
    assert not list((tmp_path / "cache").rglob("*@*.json")) and not (runs.is_dir() and any(runs.iterdir()))


@pytest.mark.logic
def test_labels_name_backend_and_model_only_as_needed() -> None:
    """A task's entries are told apart by backend, and by model when they share a backend; a true twin collides."""
    ladder = _module("scale_ladder_labels_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    placed = [("t", "a", "org/m"), ("t", "b", "org/m"), ("u", "a", "org/m"), ("u", "a", "org/n"), ("v", "a", None)]
    assert ladder.labels(placed) == ["t~a", "t~b", "u~a~org--m", "u~a~org--n", "v~a"]
    assert ladder.labels([("t", "a", None), ("t", "a", None)]) == ["t~a", "t~a"]


@pytest.mark.logic
def test_two_entries_of_one_task_write_two_results_and_two_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two backends of one task each get their own result file and store; twins sharing a label are refused."""
    ladder = _module("scale_ladder_paths_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    registry = Registry()
    registry.add_task(fake_backend.ECHO)
    registry.add_backend(fake_backend.FAKE)
    registry.add_backend(fake_backend.FAKE.model_copy(update={"name": "fake2", "stack": "fake2"}))

    def measured(entry: Any, point: Any, suite: Any, profile: Any, label: str, out: Path) -> dict[str, Any]:  # noqa: ANN401, ARG001, PLR0913 - the measure_point signature
        (out / f"{label}@{point}" / "store").mkdir(parents=True)
        row = {"per_item_s": 1.0, "peak_rss_bytes": 1, "files_written": 1, "bytes_written": 1, "items": 1, "units": []}
        return {"sha": "a", "jobs": ["1"], "task": entry.task, "label": label, "point": point, "done": True, **row}

    monkeypatch.setattr(ladder, "measure_point", measured)
    head = "format: scale-suite@1.0\nname: two\nseed: 0\nlong_s: 1\ntasks:\n"
    twins = tmp_path / "twins.yaml"
    twins.write_text(
        head + "- {task: fake.echo, port: payload, data: payloads, points: [1]}\n"
        "- {task: fake.echo, port: payload, data: payloads, params: {salt: y}, points: [1]}\n"
    )
    pair = tmp_path / "two.yaml"
    pair.write_text(
        head + "- {task: fake.echo, stack: fake, port: payload, data: payloads, points: [1]}\n"
        "- {task: fake.echo, stack: fake2, port: payload, data: payloads, points: [1]}\n"
    )
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, registry, mit_lock_records))
    try:
        with pytest.raises(ladder.SuiteRefused, match="label 'fake.echo~fake' is another entry's too"):
            ladder.run(twins, ExecutionProfile.load("laptop"))
        ladder.run(pair, ExecutionProfile.load("laptop"))
    finally:
        use_packaged(previous)
    out = tmp_path / "cache" / "2x" / "benchmarks" / "scale" / "two" / ladder.commit()
    assert sorted(p.name for p in out.glob("*.json")) == ["fake.echo~fake2@1.json", "fake.echo~fake@1.json"]
    assert sorted(p.parent.name for p in out.glob("*/store")) == ["fake.echo~fake2@1", "fake.echo~fake@1"]


@pytest.mark.logic
def test_a_points_units_hold_their_probe_observations_and_estimate(tmp_path: Path) -> None:
    """Per unit its measures root holds: the unit, its probe observations and the `Estimate`, measured or not."""
    ladder = _module("scale_ladder_units_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    thin = UNIT.model_copy(update={"backend": "thin"})
    _write(tmp_path, UNIT, _line([2.0, 8.0]))
    _write(tmp_path, thin, _line([2.0, 8.0], count=1))
    units = ladder._units(_roots(tmp_path), PROFILE)
    by_label = {u["label"]: u for u in units}
    assert set(by_label) == {UNIT.label(), thin.label()} and [u["label"] for u in units] == sorted(by_label)
    assert (
        by_label[UNIT.label()]["estimate"]["status"] == "measured" and len(by_label[UNIT.label()]["observations"]) == 6
    )
    assert (
        by_label[thin.label()]["estimate"]["status"] == "unmeasured" and by_label[thin.label()]["estimate"]["reasons"]
    )
    assert MeasureUnit.model_validate(by_label[UNIT.label()]["unit"]) == UNIT
    assert json.loads(json.dumps(units, sort_keys=True)) == units


def _request(item: str, jobs: int, wall: float, item_wall: float | None, **update: Any) -> Observation:  # noqa: ANN401
    fields: dict[str, Any] = {"probe": None, "item": item, "jobs": jobs, "batch": jobs, "wall_s": wall}
    return observation(**{**fields, "item_wall_s": item_wall, **update})


BASIS = "production span on the worker clock, first claim to last commit, per production job"


def _ref(i: int) -> JobRef:
    return JobRef(plan_hash="0" * 64, node="n", element=(i,))


def _sample(began_s: float, ended_s: float, claim_s: float = 0.0) -> dict[str, Any]:
    """A worker's sample of an item claimed at `began_s` and measured at `ended_s`, on the worker's clock."""
    sample = Sample(
        wall_s=ended_s - began_s - claim_s,
        claim_s=claim_s,
        jobs=1,
        batch_size=1,
        input_s=None,
        baseline_bytes=0,
        resident_bytes=None,
        peak_memory_bytes=1,
        cold=False,
        job="7",
        ended_s=ended_s,
    )
    return sample.model_dump(mode="json")


def _outcome(queue: str, jobs: range, began_s: float, ended_s: float, failed: Sequence[int] = ()) -> dict[str, Any]:
    refs = [{"job": _ref(i).model_dump(mode="json"), "failure": {"x": 1} if i in failed else None} for i in jobs]
    return {"queue": queue, "jobs": refs, "measure": _sample(began_s, ended_s, claim_s=1.0)}


def _span_run(production_end: float, probe: tuple[float, float]) -> list[dict[str, Any]]:
    """Return the outcomes of a run: production jobs 10 to 14 and a probe's two jobs.

    The production jobs (14 failed) are one item claimed at 20 s and measured at `production_end`; the probe's are an
    item of the probe's queue claimed and measured at `probe`'s times. A model load and a worker start before the
    first claim leave no sample, nor does the wait after the last commit.
    """
    return [
        _outcome("cpu", range(10, 15), 20.0, production_end, failed=[14]),
        _outcome(probe_queue("cpu", "p1"), range(100, 102), *probe),
    ]


@pytest.mark.logic
def test_flatness_reads_request_walls_not_end_times() -> None:
    """Per-item seconds are the production span, first claim to last commit, over the jobs that gave a result.

    Both ends are the workers' own clock (the samples' `ended_s` less their claim's start), never the coordinator's
    events, which are written when the follower next reads: that lag is not in the figure. A model load before the first
    claim, the wait after the last commit and a probe running beside production do not change it, and a later commit
    does. Worker seconds per job come from request walls and leave the loading request out.
    """
    ladder = _module("scale_ladder_walls_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    probe = ProbeRef(id="p1", phase="SINGLE", stratum=0)
    observations = [
        _request("a", 2, 0.1, None, request_index=0),
        _request("a", 2, 0.2, 0.5, request_index=1),
        _request("b", 4, 0.3, 1.0),
        _request("probe", 1, 9.0, 9.0, probe=probe),
        _request("cold", 4, 4.0, 4.0, since_load=0, load_s=3.0),
        _request("lost", 4, 0.1, 0.1, outcome="failed", since_load=None, fault="content"),
    ]
    cost = ladder.request_cost(observations, _span_run(30.0, (40.0, 41.0)))
    assert cost["per_item_s"] == pytest.approx(10.0 / 4) and cost["worker_s_per_item"] == pytest.approx(0.6 / 8)
    assert (
        cost["per_item_basis"] == "production span on the worker clock, first claim to last commit, per production job"
    )
    for beside in ((40.0, 41.0), (22.0, 28.0), (5.0, 500.0)):
        assert ladder.request_cost(observations, _span_run(30.0, beside))["per_item_s"] == cost["per_item_s"]
    assert ladder.request_cost(observations, _span_run(32.0, (40.0, 41.0)))["per_item_s"] == pytest.approx(12.0 / 4)
    none = ladder.request_cost(observations, _span_run(30.0, (40.0, 41.0))[1:])
    assert none["per_item_s"] is None and none["per_item_basis"] == "no production items"
    assert not hasattr(ladder, "_steady") and not hasattr(ladder, "_item_samples")


@pytest.mark.logic
def test_a_claim_of_several_items_and_a_request_spanning_them_count_each_job_once() -> None:
    """Items claimed together carry the claim's clock, and a batched request is in every item it spans.

    Three one-job items of one claim report item walls 0.1, 0.2 and 0.3 (from the claim's start), and one request of the
    three jobs, 0.3 s, is reported in each of them: the worker seconds per job are 0.1, not the 0.2 or 0.6 the sums of
    those would give. The jobs per-item seconds divide by are the outcome records' own, each counted once.
    """
    ladder = _module("scale_ladder_claims_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    spanning = [_request(f"i{k}", 3, 0.3, 0.1 * (k + 1), request_index=k) for k in range(3)]
    outcomes = [_outcome("cpu", range(3), 1.0, 2.0)]
    cost = ladder.request_cost(spanning, outcomes)
    assert cost["worker_s_per_item"] == pytest.approx(0.1) and cost["per_item_s"] == pytest.approx(1.0 / 3)
    walls = (0.05, 0.1, 0.15)
    singles = [_request(f"j{k}", 1, w, sum(walls[: k + 1])) for k, w in enumerate(walls)]
    assert ladder.request_cost(singles, outcomes)["worker_s_per_item"] == pytest.approx(0.1)
    twice = [*singles, singles[0].model_copy(update={"item": "k0"})]
    assert ladder.request_cost(twice, outcomes)["per_item_s"] == pytest.approx(1.0 / 3)


@pytest.mark.logic
def test_the_outcome_record_a_pool_worker_commits_is_what_the_ladder_reads(tmp_path: Path) -> None:
    """`pool_parts.report`'s outcome record is read by `_outcomes`, `_samples` and `_observations` of the ladder.

    The key the worker writes the observations under is the one the ladder reads, so a rename on either side fails here
    rather than turning every point's per-item seconds into "no production items".
    """
    ladder = _module("scale_ladder_join_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    sample = Sample.model_validate(_sample(0.0, 1.0))
    seen = [_request("a", 2, 0.4, 0.5)]
    written: dict[str, Any] = {}

    class Claim:
        def commit(self, digests: list[str], extra: dict[str, Any]) -> list[str]:
            written.update(extra)
            return digests

    assert pool_parts.report(Claim(), "d" * 8, [], sample, seen)  # type: ignore[arg-type]
    run_dir = tmp_path / "run"
    outcome = run_dir / ladder.RUN_QUEUES / "cpu" / "outcome"
    outcome.mkdir(parents=True)
    (outcome / ("d" * 8)).write_text(json.dumps(written))
    outcomes = ladder._outcomes(run_dir)
    assert ladder._observations(outcomes) == seen and ladder._samples(outcomes) == [sample]
    assert outcomes[0]["queue"] == "cpu"


@pytest.mark.logic
def test_a_result_names_each_failed_record_by_its_last_failure() -> None:
    """A job failed twice is named once, by its last failure; finished jobs are not named; the message is bounded."""
    ladder = _module("scale_ladder_faults_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    rows = [ManifestRow(id=f"r{i}", entities={}, inputs={}) for i in range(3)]

    def event(kind: str, element: int, message: str = "") -> Any:  # noqa: ANN401 - an Event stand-in
        failure = FaultRecord(fault="content", type="Bad", message=message, gap=False, charged=True)
        return SimpleNamespace(kind=kind, job=SimpleNamespace(element=(element,)), failure=failure)

    events = [event("failed", 2, "first"), event("finished", 0), event("failed", 2, "x" * 500)]
    [named] = ladder._faults(events, rows)
    assert named == {"record": "r2", "fault": "content", "type": "Bad", "message": "x" * ladder.FAULT_MESSAGE_CHARS}


@pytest.mark.logic
def test_a_points_measures_root_is_its_own_directory(tmp_path: Path) -> None:
    """A point's pool probes into the point's directory, not the profile's roots, so no point reads another's."""
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - the module is loaded by path above

    ladder = _module("scale_ladder_point_roots", ROOT / "benchmarks" / "scale" / "ladder.py")
    profile = ExecutionProfile.load("laptop")
    roots = ladder.point_roots(profile, tmp_path / "point" / "measures")
    assert roots.roots == (tmp_path / "point" / "measures",) and roots.write_root == tmp_path / "point" / "measures"
