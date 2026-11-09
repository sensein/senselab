"""The scale ladder's trees and the resource tables generated from its results."""

import getpass
import hashlib
import importlib.util
import io
import json
import os
import platform
import socket
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
from senselab_core.runtime.calibration import EstimateSource, HardwareClass, Observation, ProbeRef
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue, CatalogueEntry
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution import pool_parts
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.measures import Build, MeasureRoots, MeasureUnit, Provenance
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import Derivation, MeasureTable, PackagedTable
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
generate = _module("generate_resources_under_test", ROOT / "scripts" / "generate_resources.py")

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


@pytest.mark.logic
def test_generator_refuses_an_unmeasured_estimate(tmp_path: Path) -> None:
    """A declared unit whose estimate is unmeasured refuses the whole run, naming it and why; nothing is written.

    An undeclared unit (the fake stack) is skipped and listed, measured or not.
    """
    root, out = tmp_path / "measures", tmp_path / "out"
    thin = UNIT.model_copy(update={"backend": "thin"})
    _write(root, UNIT, _line([2.0, 8.0]))
    _write(root, thin, _line([2.0, 8.0], count=1))
    with pytest.raises(generate.Refused, match="unmeasured") as refused:
        generate.generate([root], out, {*DECLARED, (UNIT.task, "thin")}, PROFILE)
    assert thin.label() in str(refused.value) and "single_below_min_requests" in str(refused.value)
    assert UNIT.label() not in str(refused.value) and not out.exists()
    lines = generate.generate([root], out, DECLARED, PROFILE)
    assert lines[0] == f"{UNIT.task}~thin unit {thin.label()}: not a packaged backend; skipped"
    assert sorted(p.name for p in out.rglob("*.yaml")) == ["whisper.yaml"]


@pytest.mark.logic
def test_generated_rows_carry_probe_ids_not_users(tmp_path: Path) -> None:
    """A row is the full unit, its estimate with the commit and probe ids, and the jobs; no account, host or uid."""
    root, out = tmp_path / "measures", tmp_path / "out"
    _write(root, UNIT, _line([2.0, 8.0], "p1") + _line([3.0, 9.0], "p2"))
    generate.generate([root], out, DECLARED, PROFILE)
    path = out / UNIT.task / f"{UNIT.backend}.yaml"
    table = MeasureTable.model_validate(load_yaml(path, "table"))
    [row] = table.rows
    assert row.unit == UNIT
    assert row.derivation == Derivation(commit=COMMIT, jobs=("1234",), probe_ids=("p1", "p2"))
    assert row.estimate.source == EstimateSource(probe_ids=("p1", "p2"), observations=12, commit=COMMIT)
    assert PackagedTable.load(out).get(UNIT) == row.estimate
    text = path.read_text()
    for name in (getpass.getuser(), socket.gethostname(), str(tmp_path), str(os.geteuid())):
        assert name not in text.split() and name not in text

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {k for v in value.values() for k in keys(v)}
        return {k for v in value for k in keys(v)} if isinstance(value, list) else set()

    assert not keys(load_yaml(path, "table")) & {"uid", "user", "username", "hostname", "account", "run"}


@pytest.mark.logic
def test_a_partial_regeneration_keeps_the_rows_of_the_units_it_does_not_regenerate(tmp_path: Path) -> None:
    """Regenerating one unit of a table replaces its row and leaves the others; a row of another commit is refused."""
    out = tmp_path / "out"
    other = UNIT.model_copy(update={"hardware": UNIT.hardware.model_copy(update={"cpus": 4})})
    first, second = tmp_path / "first", tmp_path / "second"
    _write(first, UNIT, _line([2.0, 8.0]))
    _write(second, other, _line([3.0, 9.0], "p2", hardware=other.hardware), PROVENANCE)
    for unit in (UNIT, other):
        generate.generate([first if unit is UNIT else second], out, DECLARED, PROFILE)
    table = PackagedTable.load(out)
    assert table.get(UNIT) is not None and table.get(other) is not None
    again = tmp_path / "again"
    _write(again, UNIT, _line([2.0, 9.0], "p3"))
    generate.generate([again], out, DECLARED, PROFILE)
    kept = PackagedTable.load(out)
    assert kept.get(other) == table.get(other) and kept.get(UNIT) is not None and kept.get(UNIT) != table.get(UNIT)
    later = tmp_path / "later"
    _write(
        later,
        UNIT,
        _line([2.0, 8.0], "p4"),
        PROVENANCE.model_copy(update={"senselab": Build(version="2.0.0", commit="b" * 40)}),
    )
    with pytest.raises(generate.Refused, match="does not regenerate"):
        generate.generate([later], out, DECLARED, PROFILE)
    assert PackagedTable.load(out).get(UNIT) == kept.get(UNIT)


@pytest.mark.logic
def test_a_ladder_run_read_under_other_calibration_settings_is_refused(tmp_path: Path) -> None:
    """The estimate is made under the settings the points were probed under: a table-profiled run needs its suite."""
    run = tmp_path / "run"
    table = PROFILE.calibration.model_copy(update={"length_strata": 3})
    for point, ran in (("t~b@100", table), ("t~b@200", PROFILE.calibration)):
        (run / point / "measures" / "measures@1").mkdir(parents=True)
        (run / f"{point}.json").write_text(json.dumps({"calibration": ran.model_dump(mode="json")}))
    (run / "t~b@300" / "measures" / "measures@1").mkdir(parents=True)
    lines = generate.ladder_mismatches([run], PROFILE)
    assert [line.split(":")[0] for line in lines] == ["t~b@100", "t~b@300"]
    assert "length_strata" in lines[0] and "no result" in lines[1]
    assert generate.ladder_mismatches([run], PROFILE.model_copy(update={"calibration": table}))[:1] == [
        "t~b@200: probed under other calibration settings (length_strata); give --suite"
    ]
    assert generate.ladder_mismatches([run / "t~b@100" / "measures"], PROFILE) == []


@pytest.mark.logic
def test_generator_refuses_rows_it_cannot_derive_or_of_two_commits(tmp_path: Path) -> None:
    """A probe with no commit or no scheduler job, or units measured at two commits, refuse the run; none is written."""
    out = tmp_path / "out"
    other = UNIT.model_copy(update={"backend": "other"})
    declared = {*DECLARED, (UNIT.task, "other")}
    cases = {
        "commit": (PROVENANCE.model_copy(update={"senselab": Build(version="2.0.0")}), "no commit"),
        "job": (PROVENANCE.model_copy(update={"job": None}), "no scheduler job"),
    }
    for name, (provenance, why) in cases.items():
        root = tmp_path / name
        _write(root, UNIT, _line([2.0, 8.0]), provenance)
        with pytest.raises(generate.Refused, match=why):
            generate.generate([root], out, DECLARED, PROFILE)
    root = tmp_path / "two"
    _write(root, UNIT, _line([2.0, 8.0]))
    _write(
        root,
        other,
        _line([2.0, 8.0]),
        PROVENANCE.model_copy(update={"senselab": Build(version="2.0.0", commit="b" * 40)}),
    )
    with pytest.raises(generate.Refused, match="several commits"):
        generate.generate([root], out, declared, PROFILE)
    assert not out.exists()


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


@pytest.mark.logic
def test_flatness_reads_request_walls_not_end_times() -> None:
    """Per-item seconds are the run's wall less its probe items' walls over its production jobs, from reports alone.

    The jobs are those of the `ok` production requests, a request that loaded the model among them (its load is run
    time); a probe item's own wall is the probe's. Worker seconds per job leave the loading request out. No end time
    enters either, so neither poll waits nor clock skew do; with no production request there are no per-item seconds.
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
    assert ladder.request_cost(observations, 21.0) == {
        "per_item_s": (21.0 - 9.0) / 12,
        "per_item_basis": "run wall less probe items, per production job",
        "worker_s_per_item": pytest.approx((0.1 + 0.2 + 0.3) / 8),
    }
    assert ladder.request_cost([observations[3], observations[5]], 21.0)["per_item_s"] is None
    assert not hasattr(ladder, "_steady") and not hasattr(ladder, "_item_samples")


@pytest.mark.logic
def test_a_claim_of_several_items_and_a_request_spanning_them_count_each_job_once() -> None:
    """Items claimed together carry the claim's clock, and a batched request is in every item it spans.

    Three one-job items of one claim report item walls 0.1, 0.2 and 0.3 (from the claim's start), and one request of the
    three jobs, 0.3 s, is reported in each of them: the worker seconds per job are 0.1, not the 0.2 or 0.6 the sums of
    those would give.
    """
    ladder = _module("scale_ladder_claims_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    spanning = [_request(f"i{k}", 3, 0.3, 0.1 * (k + 1), request_index=k) for k in range(3)]
    cost = ladder.request_cost(spanning, 1.0)
    assert cost["worker_s_per_item"] == pytest.approx(0.1) and cost["per_item_s"] == pytest.approx(1.0 / 3)
    walls = (0.05, 0.1, 0.15)
    singles = [_request(f"j{k}", 1, w, sum(walls[: k + 1])) for k, w in enumerate(walls)]
    cost = ladder.request_cost(singles, 0.6)
    assert cost["worker_s_per_item"] == pytest.approx(0.1) and cost["per_item_s"] == pytest.approx(0.2)
    twice = [*singles, *[o.model_copy(update={"item": f"k{n}"}) for n, o in enumerate(singles[:1])]]
    assert ladder.request_cost(twice, 0.6)["per_item_s"] == pytest.approx(0.6 / 3)


@pytest.mark.logic
def test_the_outcome_record_a_pool_worker_commits_is_what_the_ladder_reads(tmp_path: Path) -> None:
    """`pool_parts.report`'s outcome record is read by `_outcomes`, `_samples` and `_observations` of the ladder.

    The key the worker writes the observations under is the one the ladder reads, so a rename on either side fails here
    rather than turning every point's per-item seconds into "no production items".
    """
    ladder = _module("scale_ladder_join_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    sample = Sample(
        wall_s=1.0,
        claim_s=0.0,
        jobs=2,
        batch_size=2,
        input_s=None,
        baseline_bytes=0,
        resident_bytes=None,
        peak_memory_bytes=5,
        cold=False,
        job="7",
        ended_s=1.0,
    )
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
    assert ladder.request_cost(ladder._observations(outcomes), 2.0)["per_item_s"] == 1.0


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
