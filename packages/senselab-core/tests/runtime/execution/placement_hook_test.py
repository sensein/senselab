"""The placement request, explicit placement values and the site hook, over a stub `sbatch` on `PATH`."""

import json
import logging
import os
import stat
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from lease_clock_fixture import Called, ManualClock
from pool_stack_fixture import packaged, registry, scoring
from senselab_core.cli import main
from senselab_core.kernel.dataset import Record, ValueInput
from senselab_core.kernel.graph import Edge, Graph, Node
from senselab_core.runtime.call import RunConfig, bind_input
from senselab_core.runtime.datasets import RecordSet
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.events import (
    Event,
    EventLog,
    PlacementRecord,
    placement_recorder,
    placements,
    read_events,
    run_dir_of,
    status_from,
)
from senselab_core.runtime.execution.in_process import InProcess, node_fragment
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.queue import RUN_QUEUES, QueueSet
from senselab_core.runtime.execution.run import _record_placements, prepare
from senselab_core.runtime.execution.substrates import (
    ExternalWorkers,
    Fleet,
    SiteRefused,
    Slurm,
    SubstrateError,
    WorkerHandle,
    scale,
)
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
from senselab_core.runtime.units import Unresolved
from senselab_core.testing.measures import CEILED_SIZING, FixedUnit

SCORE = scoring("placement_hook_test.score")
REGISTRY = registry(SCORE)
PARTITION = "mit_secret_gpu"
ACCOUNT = "acct_secret"
CLASSES: dict[str, Any] = {
    "cpu": {"device": "cpu", "cpus": 4, "mem_bytes": 2**33, "time_s": 21600, "partition": PARTITION, "max_workers": 8}
}
SLURM_CLIENTS = (
    "sinfo squeue scontrol sacct sacctmgr sshare sprio sstat sdiag sreport salloc srun strigger scancel sattach sbcast "
    "scrontab sh5util"
).split()
"""Every Slurm client a stub stands for: each logs its call, so the allowlist judges every call senselab makes."""
SBATCH = """#!/bin/sh
d="$FAKE_SLURM"
echo "sbatch $*" >> "$d/calls.log"
part=""
for a in "$@"; do
  case "$a" in
    --time=360) echo "sbatch: error: Requested time limit is invalid (missing or exceeds some limit)" >&2; exit 1;;
    --partition=*) part="${a#--partition=}";;
  esac
done
[ -f "$d/refuse-account" ] && { echo "sbatch: error: Invalid account or account/partition combination specified" \
"($SBATCH_ACCOUNT on $part, excluding $FAKE_EXCLUDED)" >&2; exit 1; }
n=$(cat "$d/next" 2>/dev/null || echo 100)
echo $((n + 1)) > "$d/next"
echo "$n"
"""


@pytest.fixture(autouse=True)
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This module's registry and cache under `tmp_path`, with stub Slurm commands that record every call."""
    bin_dir, state = tmp_path / "bin", tmp_path / "slurm"
    bin_dir.mkdir()
    state.mkdir()
    stubs = {"sbatch": SBATCH, **{c: f'#!/bin/sh\necho "{c} $*" >> "$FAKE_SLURM/calls.log"\n' for c in SLURM_CLIENTS}}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(stat.S_IRWXU)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(state))
    with packaged(REGISTRY, tmp_path / "cache"):
        yield


def _calls(tmp_path: Path) -> list[str]:
    log = tmp_path / "slurm" / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _sbatches(tmp_path: Path) -> list[str]:
    return [c for c in _calls(tmp_path) if c.startswith("sbatch ")]


def _profile(partition: str = PARTITION, exclude: Sequence[str] = ()) -> ExecutionProfile:
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {**CLASSES["cpu"], "partition": partition}},
        "substrate": {"kind": "slurm", "exclude_nodes": list(exclude)},
        "pool": {"quantum_items": 1},
        "sizing": CEILED_SIZING,
    }
    return ExecutionProfile.load("orcd", override)


def _commands() -> SlurmCommands:
    commands = _profile().substrate.commands
    assert commands is not None
    return commands


def _plan(profile: ExecutionProfile | None = None, **kwargs: Any) -> Plan:  # noqa: ANN401 - `plan`'s keywords
    graph = Graph(
        inputs={"text": "list[json]"},
        nodes=(Node(id="a", task=SCORE.name, task_version="1"),),
        edges=(Edge(source="input.text", target="a.text", kind="map"),),
        outputs={"value": "a.value"},
    )
    rows = [Record(id=f"r{i}", inputs={"text": ValueInput(value=t)}) for i, t in enumerate("abc")]
    return plan(graph, RecordSet(rows), config=RunConfig(), execution=profile or _profile(), **kwargs)


def _placer(placed: Plan, clock: ManualClock, epoch: int = 0) -> Placer:
    """The placer of `placed`'s run as the pool builds it, recording to the run's log."""
    assert placed.execution is not None
    run_dir = run_dir_of(placed.hash)
    return Placer(
        placed.execution,
        placed.overrides,
        placed.hook,
        request=lambda: placed.placement_request(values={}),
        clock=clock,
        record=placement_recorder(EventLog(run_dir, "a1", 1, epoch=epoch)),
    )


def _slurm(placed: Plan, clock: ManualClock, epoch: int = 0) -> Slurm:
    assert placed.execution is not None
    return Slurm(
        placed.execution.resource_classes,
        None,
        command=["senselab", "execute"],
        plan_hash=placed.hash,
        run_dir=run_dir_of(placed.hash),
        epoch=epoch,
        exclude_nodes=placed.execution.substrate.exclude_nodes,
        commands=_commands(),
        clock=clock,
        placer=_placer(placed, clock, epoch),
    )


def _placed(placed: Plan) -> list[Event]:
    return placements(read_events(run_dir_of(placed.hash)))


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
    slurm = _slurm(placed, clock)
    with caplog.at_level(logging.INFO, logger="senselab"), pytest.raises(SiteRefused) as raised:
        slurm.start("cpu", 1)
    err = raised.value
    assert isinstance(err, SubstrateError) and err.resource_class == "cpu"
    assert "--time=360" in str(err) and "Requested time limit is invalid" in str(err)
    assert err.layers["time_s"] == "profile" and "--placement" in str(err)
    assert PARTITION in caplog.text
    corrected = _plan(placement={"cpu": {"time_s": 14400}})
    assert len(_slurm(corrected, clock).start("cpu", 1)) == 1
    assert "--time=240" in _sbatches(tmp_path)[-1]


@pytest.mark.logic
def test_the_hooks_answer_is_used_and_recorded(tmp_path: Path, clock: ManualClock) -> None:
    """The answer sets the partition, time and worker cap; one event per class per epoch records it, rendered."""
    asked: list[PlacementRequest] = []

    def site_hook(request: PlacementRequest) -> PlacementAnswer:
        asked.append(request)
        return _answer(partition="gpuq2", time_s=14400, max_workers=2)

    placed = _plan(placement_hook=site_hook)
    slurm = _slurm(placed, clock)
    assert len(slurm.start("cpu", 2)) == 2
    assert all({"--partition=gpuq2", "--time=240"} <= set(c.split()) for c in _sbatches(tmp_path))
    assert len(slurm.start("cpu", 1)) == 1 and len(asked) == 1
    assert [c.resource_class for c in asked[0].classes] == ["cpu"]
    assert slurm.placer is not None
    slurm.epoch = 1
    slurm.placer.record = placement_recorder(EventLog(run_dir_of(placed.hash), "a2", 2, epoch=1))
    assert len(slurm.start("cpu", 1)) == 1 and len(asked) == 2
    first, second = (e.placement for e in _placed(placed))
    assert first is not None and second is not None and first.verdict == "accepted"
    assert [e.epoch for e in _placed(placed)] == [0, 1]
    assert first.hook is not None and first.hook.path.endswith("site_hook") and first.hook.importable is False
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
        _slurm(placed, clock).start("cpu", 1)
    assert _sbatches(tmp_path) == []
    [event] = _placed(placed)
    assert event.placement is not None and event.placement.verdict == "refused" and event.placement.answer is not None


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
    assert len(_slurm(broken, clock).start("cpu", 1)) == 1
    [event] = _placed(broken)
    assert event.placement is not None and event.placement.verdict == "failed"
    assert "site tool is down" in str(event.placement.note)


def _secret_hook(_: PlacementRequest) -> PlacementAnswer:
    return PlacementAnswer(
        classes={"cpu": PlacementValues(partition="hookpart_secret")}, why="hookpart_secret has idle nodes"
    )


SOURCES = {
    "variable": ("realpart_secret", "realpart_secret", {}, (), None),
    "override": (PARTITION, "overridepart_secret", {"partition": "overridepart_secret"}, (), None),
    "hook": (PARTITION, "hookpart_secret", {}, (), _secret_hook),
    "exclude": (PARTITION, PARTITION, {}, ("node-secret-7",), None),
}


@pytest.mark.parametrize("source", list(SOURCES))
@pytest.mark.logic
def test_a_site_refusal_renders_under_the_audience_rule(
    tmp_path: Path,
    clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    source: str,
) -> None:
    """The refusal and the placement record render the partition, account and nodes of the line, from any source."""
    real, secret, override, exclude, hook = SOURCES[source]
    monkeypatch.setenv("SBATCH_ACCOUNT", ACCOUNT)
    monkeypatch.setenv("REAL_PART", real)
    monkeypatch.setenv("FAKE_EXCLUDED", ",".join(exclude))
    (tmp_path / "slurm" / "refuse-account").write_text("")
    profile = _profile("${REAL_PART}" if source == "variable" else PARTITION, exclude)
    placed = _plan(profile, placement={"cpu": {"time_s": 14400, **override}}, placement_hook=hook)
    slurm = _slurm(placed, clock)
    with caplog.at_level(logging.INFO, logger="senselab"), pytest.raises(SiteRefused) as raised:
        slurm.start("cpu", 1)
    shown = str(raised.value) + (raised.value.site_message or "")
    secrets = [ACCOUNT, str(tmp_path), *exclude, real, secret]
    assert "Invalid account or account/partition" in shown and "--time=240" in shown
    assert not [x for x in secrets if x in shown]
    assert "<partition>" in shown and "<account>" in shown and bool(exclude) == ("<node>" in shown)
    assert secret in caplog.text and ACCOUNT in caplog.text
    [event] = _placed(placed)
    assert not [x for x in secrets[2:] if x in event.model_dump_json()]
    if hook is not None:
        assert event.placement is not None and event.placement.answer is not None
        assert "<partition>" in event.placement.answer.why


@pytest.mark.logic
def test_explicit_placement_overrides_the_profile_and_is_recorded(tmp_path: Path, clock: ManualClock) -> None:
    """Profile, then override, then hook: each value takes the last layer that set it, and the event says which."""
    file = tmp_path / "placement.yaml"
    file.write_text("cpu:\n  partition: otherq\n  time_s: 3600\n  max_workers: 4\n")
    placed = _plan(placement=file)
    slurm = _slurm(placed, clock)
    assert len(slurm.start("cpu", 2)) == 2
    assert {"--partition=otherq", "--time=60"} <= set(_sbatches(tmp_path)[0].split())
    [found] = _placed(placed)
    event = found.placement
    assert event is not None and event.verdict == "no-hook" and event.hook is None and event.used.max_workers == 4
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
    slurm = _slurm(placed, clock)
    slurm.start("cpu", 2)
    calls = _calls(tmp_path)
    assert len([c for c in calls if c.startswith("sbatch ")]) == 2
    assert all(c.startswith("sbatch ") or c.startswith("squeue --me") for c in calls)
    assert not any("--test-only" in c for c in calls)
    assert {c.split()[0] for c in calls} <= {"sbatch", "squeue"}


class _Holding:
    """A substrate that records each start and holds its workers."""

    def __init__(self) -> None:
        self.handles: list[WorkerHandle] = []

    def start(self, resource_class: str, count: int) -> list[WorkerHandle]:
        made = [WorkerHandle(resource_class, f"w{len(self.handles) + i}") for i in range(count)]
        self.handles.extend(made)
        return made

    def live(self) -> list[WorkerHandle]:
        return list(self.handles)

    def running(self, resource_class: str) -> int:
        return sum(1 for h in self.handles if h.queue == resource_class)

    def stop(self, handles: Sequence[WorkerHandle]) -> None:
        self.handles = [h for h in self.handles if h not in handles]


@pytest.mark.logic
def test_a_site_refusal_holds_its_class_alone(tmp_path: Path, clock: ManualClock) -> None:
    """A refused class is broken after the budget; another class and the substrate keep starting."""
    placed = _plan()
    slurm = _slurm(placed, clock)
    with pytest.raises(SiteRefused) as raised:
        slurm.start("cpu", 1)
    fleet = Fleet(slurm, workers_dir=None, budget=1, first_s=1.0, max_s=1.0, clock=clock)
    for _ in range(2):
        fleet.fail_substrate(raised.value)
    assert set(fleet.broken) == {"cpu"} and "Requested time limit is invalid" in fleet.broken["cpu"].reason
    assert fleet.broken_for("gpu") is None and fleet.substrate_due()
    assert fleet.start("cpu", 1) == []


def _pool(tmp_path: Path, placed: Plan, clock: ManualClock, substrate: _Holding) -> Pool:
    assert placed.execution is not None
    profile, run_dir = placed.execution, tmp_path / "run"
    epoch = Epoch(run_dir).take()
    values = {bind_input("x", "json", t)[0]: t for t in "abc"}
    queues = QueueSet(run_dir / RUN_QUEUES, profile, epoch=Epoch(run_dir).current)

    def over(workers: Any) -> Pool:  # noqa: ANN401 - a substrate
        return Pool(
            run_dir,
            queues=queues,
            store=ResultStore(tmp_path / "store"),
            events=EventLog(run_dir, "a1", epoch, epoch=epoch),
            profile=profile,
            substrate=workers,
            epoch=epoch,
            clock=clock,
            resolver=FixedUnit(profile, Unresolved("a final cause")),
        )

    feeder, pool = over(ExternalWorkers()), over(substrate)
    admission = feeder.admit(placed, values)
    for element in range(3):
        feeder.submit(node_fragment(placed, admission, elements=[(element,)]))
    feeder.close()
    pool.admit(replace(placed, sources={}, live={}), values)
    return pool


@pytest.mark.logic
def test_one_worker_cap_is_the_placements_on_every_substrate(tmp_path: Path, clock: ManualClock) -> None:
    """The pool reads a class's cap from the placer: an explicit cap holds on a non-Slurm substrate and is recorded."""
    placed = _plan(placement={"cpu": {"max_workers": 1}})
    holding = _Holding()
    pool = _pool(tmp_path, placed, clock, holding)
    scale(pool.queues, holding, pool.profile, workers=lambda _q, _w: 3, granted=pool.granted)
    assert len(holding.handles) == 1 and pool.granted("cpu").max_workers == 1
    found = [e.placement for e in placements(read_events(tmp_path / "run")) if e.placement is not None]
    assert found and all(r.used.max_workers == 1 and r.used.layers["max_workers"] == "override" for r in found)
    assert all("recorded, not applied" in str(r.note) for r in found)


@pytest.mark.logic
def test_an_executor_without_workers_records_the_explicit_placement(tmp_path: Path) -> None:
    """A plan run in process records its explicit values as `placement` events, saying they are not applied."""
    placed = _plan(placement={"cpu": {"max_workers": 1}})
    log = EventLog(tmp_path / "run", "a1", 1, epoch=1)
    executor = InProcess(store=ResultStore(tmp_path / "store"), profile=_profile(), run=RunConfig())
    _record_placements(placed, executor, {}, log, 1)
    [found] = placements(read_events(tmp_path / "run"))
    assert found.placement is not None and found.placement.used.max_workers == 1
    assert "recorded, not applied" in str(found.placement.note)


@pytest.mark.logic
def test_the_request_is_built_once_per_epoch(clock: ManualClock) -> None:
    """A new epoch asks the request again, from the values bound for it; the same epoch asks once."""
    built = []

    def request() -> PlacementRequest:
        built.append(1)
        return PlacementRequest(classes=(_needs(),))

    placer = Placer(_profile(), {}, lambda _: _answer(partition="hookq"), request=request, clock=clock)
    placer.for_class("cpu", 0)
    placer.for_class("cpu", 0)
    assert len(built) == 1
    placer.for_class("cpu", 1)
    assert len(built) == 2


@pytest.mark.parametrize(
    ("override", "hook_value", "layer", "expected"),
    [
        (None, None, "profile", ("n-bad",)),
        ((), None, "override", ()),
        (("n-x",), None, "override", ("n-x",)),
        (None, ("n-y",), "hook", ("n-y",)),
    ],
    ids=["profile", "cleared", "override", "hook"],
)
@pytest.mark.logic
def test_excluded_nodes_come_from_one_list_with_its_layer(  # noqa: PLR0913 - the parametrised case, then fixtures
    tmp_path: Path,
    clock: ManualClock,
    override: tuple[str, ...] | None,
    hook_value: tuple[str, ...] | None,
    layer: str,
    expected: tuple[str, ...],
) -> None:
    """`substrate.exclude_nodes` is the base; an override or the hook replaces it; the list submitted is recorded."""
    given = {"time_s": 14400, **({} if override is None else {"exclude": list(override)})}
    hook = None if hook_value is None else (lambda _: _answer(exclude=hook_value))
    placed = _plan(_profile(exclude=("n-bad",)), placement={"cpu": given}, placement_hook=hook)
    slurm = _slurm(placed, clock)
    slurm.start("cpu", 1)
    line = _sbatches(tmp_path)[0]
    assert (f"--exclude={','.join(expected)}" in line) == bool(expected) and ("--exclude" in line) == bool(expected)
    [found] = _placed(placed)
    assert found.placement is not None and found.placement.used.layers["exclude"] == layer
    assert len(found.placement.exclude) == len(expected)


@pytest.mark.logic
def test_placement_is_an_event_of_the_one_run_log(tmp_path: Path) -> None:
    """A placement is a run-scoped event of the epoch's log: no node, ignored by status; a job event needs its node."""
    log = EventLog(tmp_path / "run", "a1", 1, epoch=1)
    record = PlacementRecord(
        resource_class="cpu", verdict="no-hook", used=CurrentValues(max_workers=1, layers={"max_workers": "profile"})
    )
    written = log.record("placement", None, placement=record)
    log.record("queued", "a")
    [found] = placements(read_events(tmp_path / "run"))
    assert found == written and found.node is None and found.format == "run-event@2.0"
    status = status_from(read_events(tmp_path / "run"))
    assert status.counts["queued"] == 1
    assert status_from([written]).last_event_ns is None
    with pytest.raises(ValueError, match="names its node"):
        Event(kind="queued", attempt="a", ordinal=1, writer="w", seq=0, t_ns=1)
    with pytest.raises(ValueError, match="carries its record"):
        Event(kind="placement", attempt="a", ordinal=1, writer="w", seq=0, t_ns=1)
