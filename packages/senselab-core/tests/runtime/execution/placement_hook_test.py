"""The placement request, explicit placement values and the site hook, over a stub `sbatch` on `PATH`."""

import json
import logging
import os
import stat
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import Called, ManualClock
from pool_stack_fixture import packaged, registry, scoring
from senselab_core.cli import main
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.events import read_placement_events, run_dir_of
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.execution.substrates import Fleet, SiteRefused, Slurm, SubstrateError
from senselab_core.runtime.execution_settings import SlurmCommands
from senselab_core.runtime.planner import Plan, plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.site_placement import (
    ClassNeeds,
    CurrentValues,
    PlacementAnswer,
    PlacementAnswerRefused,
    PlacementRequest,
    PlacementValues,
    Placer,
    check_answer,
    load_overrides,
    resolve_placement,
)
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.measures import CEILED_SIZING

SCORE = scoring("placement_hook_test.score")
REGISTRY = registry(SCORE)
PARTITION = "mit_secret_gpu"
ACCOUNT = "acct_secret"
CLASSES: dict[str, Any] = {
    "cpu": {"device": "cpu", "cpus": 4, "mem_bytes": 2**33, "time_s": 21600, "partition": PARTITION, "max_workers": 8}
}
STUBS = {
    "sbatch": """#!/bin/sh
d="$FAKE_SLURM"
echo "sbatch $*" >> "$d/calls.log"
for a in "$@"; do
  case "$a" in
    --time=360) echo "sbatch: error: Requested time limit is invalid (missing or exceeds some limit)" >&2; exit 1;;
    --account-check) ;;
  esac
done
[ -f "$d/refuse-account" ] && { echo "sbatch: error: Invalid account or account/partition combination specified" \
"($SBATCH_ACCOUNT on $FAKE_PARTITION)" >&2; exit 1; }
n=$(cat "$d/next" 2>/dev/null || echo 100)
echo $((n + 1)) > "$d/next"
echo "$n"
""",
    "squeue": '#!/bin/sh\necho "squeue $*" >> "$FAKE_SLURM/calls.log"\n',
    "sinfo": '#!/bin/sh\necho "sinfo $*" >> "$FAKE_SLURM/calls.log"\n',
    "sacctmgr": '#!/bin/sh\necho "sacctmgr $*" >> "$FAKE_SLURM/calls.log"\n',
    "scontrol": '#!/bin/sh\necho "scontrol $*" >> "$FAKE_SLURM/calls.log"\n',
    "scancel": '#!/bin/sh\necho "scancel $*" >> "$FAKE_SLURM/calls.log"\n',
}


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry and cache under `tmp_path`, with stub Slurm commands that record every call."""
    bin_dir, state = tmp_path / "bin", tmp_path / "slurm"
    bin_dir.mkdir()
    state.mkdir()
    for name, body in STUBS.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(state))
    monkeypatch.setenv("FAKE_PARTITION", PARTITION)
    with packaged(REGISTRY, tmp_path / "cache"):
        yield


def _calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "slurm" / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _sbatches(tmp_path: Path) -> list[str]:
    return [c for c in _calls(tmp_path) if c.startswith("sbatch ")]


def _profile() -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": CLASSES,
        "substrate": {"kind": "slurm", "commands": {"retry_first_s": 0.001}},
        "pool": {"quantum_items": 1},
        "sizing": CEILED_SIZING,
    }
    return ExecutionProfile.load("orcd", override)


def _commands() -> SlurmCommands:
    commands = _profile().substrate.commands
    assert commands is not None
    return commands


def _plan(**kwargs: Any) -> Plan:  # noqa: ANN401 - `plan`'s own keyword arguments
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=SCORE.name, task_version="1"),),
        edges=(Edge(source="input.text", target="a.text", kind="map"),),
        outputs={"value": "a.value"},
    )
    rows = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate("abc")]
    return plan(graph, RecordSet(rows), config=RunConfig(), execution=_profile(), **kwargs)


def _slurm(tmp_path: Path, placed: Plan, clock: ManualClock, epoch: int = 0) -> Slurm:
    run_dir = run_dir_of(placed.hash)
    return Slurm(
        _profile().resource_classes,
        None,
        command=["senselab", "execute"],
        plan_hash=placed.hash,
        run_dir=run_dir,
        epoch=epoch,
        commands=_commands(),
        clock=clock,
        placer=placed.placer(run_dir, clock=clock),
    )


def _needs(**given: Any) -> ClassNeeds:  # noqa: ANN401 - the fields a test sets
    current = CurrentValues(partition=PARTITION, time_s=21600, max_workers=8, layers={"time_s": "profile"})
    fields: dict[str, Any] = {
        "resource_class": "cpu",
        "device": "cpu",
        "workers": 3,
        "workers_max": 8,
        "time_per_worker_s": 7200.0,
        "current": current,
    }
    return ClassNeeds(**{**fields, **given})


def _answer(**values: Any) -> PlacementAnswer:  # noqa: ANN401 - the values a test answers
    return PlacementAnswer(classes={"cpu": PlacementValues(**values)}, why=f"{PARTITION} has idle nodes")


@pytest.mark.logic
def test_a_partitions_maxtime_refusal_names_the_request_and_layer(
    tmp_path: Path, clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    """A class time of 21600 s is `--time=360`; the site refuses it, and the refusal names the line, reply and layer."""
    placed = _plan()
    slurm = _slurm(tmp_path, placed, clock)
    with caplog.at_level(logging.INFO, logger="senselab"), pytest.raises(SiteRefused) as raised:
        slurm.start("cpu", 1)
    err = raised.value
    assert isinstance(err, SubstrateError) and err.resource_class == "cpu"
    assert "--time=360" in str(err) and "Requested time limit is invalid" in str(err)
    assert err.layers["time_s"] == "profile" and "--placement" in str(err)
    fleet = Fleet(slurm, workers_dir=None, budget=1, first_s=1.0, max_s=1.0, clock=clock)
    for _ in range(2):
        fleet.fail_substrate(err)
    assert "Requested time limit is invalid" in fleet.broken["*"].reason
    assert PARTITION in caplog.text
    corrected = _plan(placement={"cpu": {"time_s": 14400}})
    assert len(_slurm(tmp_path, corrected, clock).start("cpu", 1)) == 1
    assert "--time=240" in _sbatches(tmp_path)[-1]


@pytest.mark.logic
def test_the_hooks_answer_is_used_and_recorded(tmp_path: Path, clock: ManualClock) -> None:
    """The answer sets the partition, time and worker cap; one event per class per epoch records it, rendered."""
    asked: list[PlacementRequest] = []

    def site_hook(request: PlacementRequest) -> PlacementAnswer:
        asked.append(request)
        return _answer(partition="gpuq2", time_s=14400, max_workers=2)

    placed = _plan(placement_hook=site_hook)
    slurm = _slurm(tmp_path, placed, clock)
    assert len(slurm.start("cpu", 5)) == 2
    assert all({"--partition=gpuq2", "--time=240"} <= set(c.split()) for c in _sbatches(tmp_path))
    assert slurm.start("cpu", 1) == [] and len(asked) == 1
    assert [c.resource_class for c in asked[0].classes] == ["cpu"]
    slurm.epoch = 1
    slurm.live()
    slurm.stop(slurm.live())
    assert len(slurm.start("cpu", 1)) == 1 and len(asked) == 2
    first, second = read_placement_events(run_dir_of(placed.hash))
    assert (first.epoch, second.epoch, first.verdict) == (0, 1, "accepted")
    assert first.hook is not None and first.hook.path.endswith("site_hook")
    assert first.request is not None and first.request.workers_max == 8
    assert {first.used.layers[f] for f in ("partition", "time_s", "max_workers")} == {"hook"}
    assert (first.used.time_s, first.used.max_workers) == (14400, 2)
    assert PARTITION not in first.model_dump_json() and first.answer is not None
    assert "<partition>" in first.answer.why


@pytest.mark.logic
def test_an_answer_below_time_per_worker_is_refused(tmp_path: Path, clock: ManualClock) -> None:
    """A time under the longest worker is refused naming the field and both values, before any submission."""
    request = PlacementRequest(classes=(_needs(),))
    with pytest.raises(PlacementAnswerRefused) as raised:
        check_answer(request, _answer(time_s=3600))
    assert (raised.value.field, raised.value.value, raised.value.limit) == ("time_s", 3600, 7200.0)
    assert check_answer(request, _answer(time_s=7200)) is None
    floor = PlacementRequest(classes=(_needs(time_per_worker_s=None, time_floor_s=900.0),))
    with pytest.raises(PlacementAnswerRefused, match="time_floor_s"):
        check_answer(floor, _answer(time_s=600))
    unmeasured = PlacementRequest(classes=(_needs(time_per_worker_s=None),))
    assert "unchecked" in str(check_answer(unmeasured, _answer(time_s=60)))
    placed = _plan(placement_hook=lambda _: _answer(max_workers=99))
    with pytest.raises(PlacementAnswerRefused):
        _slurm(tmp_path, placed, clock).start("cpu", 1)
    assert _sbatches(tmp_path) == []
    [event] = read_placement_events(run_dir_of(placed.hash))
    assert event.verdict == "refused" and event.answer is not None


@pytest.mark.logic
def test_an_answer_above_the_runs_workers_is_refused(tmp_path: Path, clock: ManualClock) -> None:
    """More workers than the class's `workers_max`, fewer than one, or a class the request lacks, is refused."""
    request = PlacementRequest(classes=(_needs(),))
    with pytest.raises(PlacementAnswerRefused) as raised:
        check_answer(request, _answer(max_workers=9))
    assert (raised.value.field, raised.value.value, raised.value.limit) == ("max_workers", 9, 8)
    with pytest.raises(PlacementAnswerRefused, match="max_workers"):
        check_answer(request, _answer(max_workers=0))
    with pytest.raises(PlacementAnswerRefused, match="class"):
        check_answer(request, PlacementAnswer(classes={"gpu": PlacementValues(max_workers=1)}))
    assert check_answer(request, _answer(max_workers=8, partition="other")) is None
    refused = Placer(_profile(), {}, lambda _: _answer(max_workers=9), request=lambda: request, clock=clock)
    with pytest.raises(PlacementAnswerRefused):
        refused.for_class("cpu", 0)
    assert _sbatches(tmp_path) == []


@pytest.mark.logic
def test_a_failing_hook_falls_back_and_says_so(
    tmp_path: Path, clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    """A hook that raises, answers wrongly or outlasts `placement_hook_s` leaves the explicit values and says so."""
    request = PlacementRequest(classes=(_needs(),))

    def failing(_: PlacementRequest) -> PlacementAnswer:
        raise RuntimeError("the site tool is down")

    with caplog.at_level(logging.WARNING, logger="senselab"):
        placed = Placer(_profile(), {}, failing, request=lambda: request, clock=clock).for_class("cpu", 0)
    assert placed.verdict == "failed" and "RuntimeError" in str(placed.note) and "site tool is down" in str(placed.note)
    assert placed.values.time_s == 21600 and set(placed.values.layers.values()) == {"profile"}
    assert "keeps its explicit and profile values" in caplog.text
    wrong: Any = lambda _: {"cpu": {}}  # noqa: E731 - a hook returning the wrong type
    assert Placer(_profile(), {}, wrong, request=lambda: request, clock=clock).for_class("cpu", 0).verdict == "failed"
    gate = threading.Event()

    def blocked(_: PlacementRequest) -> PlacementAnswer:
        gate.wait(clock.hang_s)
        return _answer(time_s=14400)

    slow = Placer(_profile(), {}, blocked, request=lambda: request, clock=clock)
    call = Called(lambda: slow.for_class("cpu", 0), hang_s=clock.hang_s)
    assert clock.parked(call.thread)
    clock.advance(_profile().substrate.placement_hook_s)
    result: Any = call.result()
    gate.set()
    assert result.verdict == "failed" and "did not answer within" in str(result.note)
    broken = _plan(placement={"cpu": {"time_s": 14400}}, placement_hook=failing)
    assert len(_slurm(tmp_path, broken, clock).start("cpu", 1)) == 1
    [event] = read_placement_events(run_dir_of(broken.hash))
    assert event.verdict == "failed" and "site tool is down" in str(event.note)


@pytest.mark.logic
def test_a_site_refusal_renders_under_the_audience_rule(
    tmp_path: Path, clock: ManualClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal renders accounts, partitions and paths as labels in its line and the site's message."""
    monkeypatch.setenv("SBATCH_ACCOUNT", ACCOUNT)
    (tmp_path / "slurm" / "refuse-account").write_text("")
    placed = _plan(placement={"cpu": {"time_s": 14400}})
    slurm = _slurm(tmp_path, placed, clock)
    with caplog.at_level(logging.INFO, logger="senselab"), pytest.raises(SiteRefused) as raised:
        slurm.start("cpu", 1)
    shown = str(raised.value) + (raised.value.site_message or "")
    assert "Invalid account or account/partition" in shown
    assert PARTITION not in shown and ACCOUNT not in shown and str(tmp_path) not in shown
    assert "<partition>" in shown and "<account>" in shown and "--time=240" in shown
    assert PARTITION in caplog.text and ACCOUNT in caplog.text


@pytest.mark.logic
def test_explicit_placement_overrides_the_profile_and_is_recorded(tmp_path: Path, clock: ManualClock) -> None:
    """Profile, then override, then hook: each value takes the last layer that set it, and the event says which."""
    file = tmp_path / "placement.yaml"
    file.write_text("cpu:\n  partition: otherq\n  time_s: 3600\n  max_workers: 4\n")
    placed = _plan(placement=file)
    slurm = _slurm(tmp_path, placed, clock)
    assert len(slurm.start("cpu", 9)) == 4
    assert {"--partition=otherq", "--time=60"} <= set(_sbatches(tmp_path)[0].split())
    [event] = read_placement_events(run_dir_of(placed.hash))
    assert event.verdict == "no-hook" and event.hook is None
    assert {k: event.used.layers[k] for k in ("partition", "time_s", "max_workers", "exclude")} == {
        "partition": "override",
        "time_s": "override",
        "max_workers": "override",
        "exclude": "profile",
    }
    resolved = resolve_placement(
        _profile(),
        load_overrides(file),
        lambda _: _answer(partition="hookq"),
        request=PlacementRequest(classes=(_needs(),)),
    )
    layers = resolved.classes["cpu"].values.layers
    assert (layers["partition"], layers["time_s"], layers["max_workers"]) == ("hook", "override", "override")
    assert resolved.classes["cpu"].values.time_s == 3600
    with pytest.raises(PlacementAnswerRefused, match="no such class"):
        _plan(placement={"nonexistent": {"max_workers": 1}})
    assert placed.placement.nodes["a"].partition == "otherq"
    with pytest.raises(ValueError, match="refused"):
        load_overrides({"cpu": {"queue": "x"}})


@pytest.mark.logic
def test_the_request_is_produced_without_computing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The request is read from measures that exist: no run directory, event, worker or scheduler call is made."""
    placed = _plan()
    request = placed.placement_request(values={})
    [needs] = request.classes
    assert needs.resource_class == "cpu" and needs.workers_max == 8 and needs.current.partition == PARTITION
    assert needs.time_per_worker_s is None and needs.wall_s is None
    assert {u.figure for u in needs.unmeasured} >= {"workers", "time_per_worker_s", "wall_s"}
    assert all(u.why for u in needs.unmeasured)
    document = request.document()
    assert document["format"] == "placement-request@1.0" and document["classes"][0]["class"] == "cpu"
    assert PlacementRequest.model_validate(json.loads(json.dumps(document))) == request
    assert not run_dir_of(placed.hash).exists() and _calls(tmp_path) == []
    store = ResultStore(tmp_path / "store")
    prepare(placed, run_dir_of(placed.hash), store)
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(_profile().model_dump_json())
    assert main(["plan", placed.hash, "--json", "--profile", str(profile_file)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["classes"][0]["class"] == "cpu" and printed["classes"][0]["workers_max"] == 8
    assert not (run_dir_of(placed.hash) / "events").exists() and _calls(tmp_path) == []


@pytest.mark.logic
def test_senselab_runs_no_scheduler_query_to_place(tmp_path: Path, clock: ManualClock) -> None:
    """Resolving and requesting call no scheduler; starting calls `sbatch` and lists its own jobs, nothing else."""
    request = PlacementRequest(classes=(_needs(),))
    resolve_placement(_profile(), {}, lambda _: _answer(partition="hookq"), request=request)
    _plan().placement_request(values={})
    assert _calls(tmp_path) == []
    placed = _plan(placement_hook=lambda _: _answer(time_s=14400))
    slurm = _slurm(tmp_path, placed, clock)
    slurm.start("cpu", 2)
    calls = _calls(tmp_path)
    assert len([c for c in calls if c.startswith("sbatch ")]) == 2
    assert all(c.startswith("sbatch ") or c.startswith("squeue --me") for c in calls)
    assert not any("--test-only" in c for c in calls)
    assert not any(c.split()[0] in {"sinfo", "sacctmgr", "scontrol"} for c in calls)
