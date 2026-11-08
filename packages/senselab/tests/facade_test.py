"""The run facade: `Dataset`, `use`, `plan`, `run`, `resume` and the `.map` of a configured task, on the fake stack."""

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import soundfile
from api_test import _echo_namespace, fake_stack  # noqa: F401 - `fake_stack` is the fixture of the fake stack
from senselab._api.call import ApiError
from senselab._api.facade import PlanKwargs, session
from senselab_core.kernel._base import FrozenMapping
from senselab_core.kernel.dataset import Record
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.run import InputsUnavailable, NotRun, Output, ResultSet, Run
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore

import senselab as sl

needs_runner = pytest.mark.skipif(importlib.util.find_spec("pydra") is None, reason="the pydra extra is not installed")


SPEECH_TO_TEXT_PLAN = "7bf52ad0f8e735cba424cf45ba77d71f5346b336dea10de155782696f00da8c6"
"""The plan hash of `SpeechToText` over the one-recording BIDS tree of the BIDS binding test."""


def _records(count: int = 3) -> list[dict[str, object]]:
    return [{"id": f"r{i}", "entities": {"n": str(i)}, "inputs": {"payload": {"v": i}}} for i in range(count)]


@pytest.fixture(autouse=True)
def session_restored() -> Iterator[None]:
    """Leave the session defaults as they were."""
    before = session()
    yield
    sl.use(execution=before.execution, config=before.config, executor=before.executor, store=before.store)


@pytest.mark.contract
@needs_runner
@pytest.mark.usefixtures("fake_stack")
def test_map_equals_per_record_calls() -> None:
    """`Task.map(dataset)` gives, per record, the keys, output identities and outputs `Task(...)(payload)` gives."""
    echo = _echo_namespace()["Echo"](salt="s")
    mapped = echo.map(sl.Dataset.from_records(_records()))
    assert isinstance(mapped, ResultSet) and [r.id for r, _ in mapped] == ["r0", "r1", "r2"]
    for i, (record, output) in enumerate(mapped):
        single = echo({"v": i})
        assert isinstance(output, Output) and record.entities["n"] == str(i)
        assert output.outputs == single.outputs
        assert output.keys == {"value": single.key.digest()}
        assert output.identities == {"value": single.output_identity}
    assert mapped.outputs("value") == [echo({"v": i}).outputs["value"] for i in range(3)]


@pytest.mark.contract
@needs_runner
@pytest.mark.usefixtures("fake_stack")
def test_map_binds_a_port_from_the_record_input_bind_names() -> None:
    """`bind={"payload": "other"}` reads each record's `other` input for port `payload`."""
    echo = _echo_namespace()["Echo"]()
    rows = [{"id": "r0", "inputs": {"other": 5}}, {"id": "r1", "inputs": {"other": 6}}]
    mapped = echo.map(sl.Dataset.from_records(rows), bind={"payload": "other"})
    assert mapped.outputs("value") == [echo(5).outputs["value"], echo(6).outputs["value"]]


@pytest.mark.contract
@needs_runner
@pytest.mark.usefixtures("fake_stack")
def test_facade_plan_run_and_resume_agree_with_map() -> None:
    """`sl.plan` gives the plan, `sl.run` and `sl.resume` its results; all equal `.map`, which is the same run."""
    echo = _echo_namespace()["Echo"](salt="s")
    dataset = sl.Dataset.from_records(_records(2))
    placed = sl.plan(echo, dataset)
    assert isinstance(placed, Plan) and len(placed.manifest.rows) == 2
    ran, resumed = sl.run(echo, dataset), sl.resume(placed)
    assert ran.outputs("value") == resumed.outputs("value") == echo.map(dataset).outputs("value")
    run = placed.run()
    run.wait(60)
    assert run.run_dir.name == placed.hash and sl.resume(run).outputs("value") == ran.outputs("value")


@pytest.mark.contract
@needs_runner
@pytest.mark.usefixtures("fake_stack")
def test_a_session_executor_runs_over_its_own_store(tmp_path: Path) -> None:
    """`sl.run` and `sl.resume` read the results from the store the session's executor writes, not the default one."""
    echo = _echo_namespace()["Echo"](salt="s")
    dataset = sl.Dataset.from_records(_records(2))
    elsewhere = ResultStore(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="store"):
        sl.use(executor=InProcess(store=elsewhere))
    with sl.use(executor=InProcess(store=elsewhere), store=elsewhere):
        ran = sl.run(echo, dataset)
        resumed = sl.resume(sl.plan(echo, dataset))
    expected = [echo({"v": i}).outputs["value"] for i in range(2)]
    assert all(isinstance(o, Output) for _, o in ran) and ran.outputs("value") == expected
    assert resumed.outputs("value") == expected and any((tmp_path / "elsewhere").iterdir())


@pytest.mark.contract
@needs_runner
@pytest.mark.usefixtures("fake_stack")
def test_a_call_given_an_object_runs_live_and_a_resume_from_disk_names_the_input() -> None:
    """An object input is admitted live and its job runs; resuming the `Run` from the directory raises, naming it."""
    echo = _echo_namespace()["Echo"](salt="s")
    held = FrozenMapping({"v": 7})
    placed = sl.plan(echo.call(held))
    assert placed.sources["rows"] == [{"payload": {"memory": placed.manifest.rows[0].inputs["payload"]}}]
    ran = placed.run()
    status = ran.wait(60)
    assert status.done and status.counts["done"] + status.counts["failed"] == 1
    with pytest.raises(InputsUnavailable, match="payload"):
        sl.resume(Run(ran.run_dir))


@pytest.mark.logic
@pytest.mark.usefixtures("fake_stack")
def test_use_sets_the_session_default_and_restores_it() -> None:
    """`with sl.use(execution=...)` changes the default profile `sl.plan` places under, and restores it on exit."""
    other = ExecutionProfile.load("laptop", {"devices": ["cpu"]})
    assert session().execution is None
    with sl.use(execution=other) as inside:
        assert session() is inside and inside.execution is other
        placed = sl.plan(_echo_namespace()["Echo"](), sl.Dataset.from_records(_records(1)))
        assert placed.execution is other
    assert session().execution is None
    assert PlanKwargs.__optional_keys__ >= {"execution", "bind", "inputs", "config"}


@pytest.mark.logic
def test_sl_dataset_bids_builds_bids_dataset(tmp_path: Path) -> None:
    """`sl.Dataset.bids(root)` is the BIDS dataset of `bids_dataset(root)`, and `sl.Dataset` is not instantiable."""
    from senselab_core.runtime.bids import BidsDataset, bids_dataset  # noqa: PLC0415

    path = tmp_path / "bids" / "sub-01" / "audio" / "sub-01_task-read_audio.wav"
    path.parent.mkdir(parents=True)
    soundfile.write(path, numpy.zeros(160, dtype=numpy.float32), 16000)
    built = sl.Dataset.bids(tmp_path / "bids")
    assert isinstance(built, BidsDataset)
    assert [(r.id, dict(r.entities)) for r in built.records()] == [
        (r.id, dict(r.entities)) for r in bids_dataset(tmp_path / "bids").records()
    ]
    assert [r.entities["task"] for r in built.records()] == ["read"]
    with pytest.raises(TypeError, match="namespace"):
        sl.Dataset()


@pytest.mark.logic
def test_a_task_over_bids_binds_its_single_input(tmp_path: Path) -> None:
    """`Resample` plans over a BIDS tree as with `bind={'series': 'audio'}`; `SpeechToText`'s `audio` binds by name."""
    path = tmp_path / "bids" / "sub-01" / "audio" / "sub-01_task-read_audio.wav"
    path.parent.mkdir(parents=True)
    soundfile.write(path, numpy.zeros(160, dtype=numpy.float32), 16000)
    dataset, resample = sl.Dataset.bids(tmp_path / "bids"), sl.Resample(target_rate=8000)
    auto = sl.plan(resample, dataset)
    assert dict(auto.logical.bind) == {"series": "audio"}
    assert auto.hash == sl.plan(resample, dataset, bind={"series": "audio"}).hash
    by_name = sl.plan(sl.SpeechToText(), dataset)
    assert dict(by_name.logical.bind) == {} and by_name.hash == SPEECH_TO_TEXT_PLAN


@pytest.mark.logic
def test_a_second_port_never_binds_to_the_input_the_first_reads() -> None:
    """`Wer` over records holding only `reference` refuses; `hypothesis` does not read `reference` too."""
    records = sl.Dataset.from_records([{"id": "r", "inputs": {"reference": "a b"}}])
    with pytest.raises(ApiError, match=r"no input 'hypothesis'.*another port reads \['reference'\]"):
        sl.Wer().map(records)


@pytest.mark.logic
def test_to_frame_indexed_by_entities() -> None:
    """The frame is indexed by the records' entities, one column per output, and a failed row holds its fault class."""
    import pandas  # noqa: PLC0415 - a test dependency of this package, imported so its absence fails the test

    records = [Record(id=f"sub-0{i}/a.wav", entities={"subject": f"0{i}", "task": "x"}) for i in (1, 2, 3)]
    fault = FaultRecord(fault="content", type="ContentFault", message="bad", gap=False, charged=True)
    results = ResultSet(
        [
            (records[0], Output({"score": 0.5, "label": "a"})),
            (records[1], fault),
            (records[2], Output({"score": 0.25, "label": "b"})),
        ]
    )
    frame = results.to_frame()
    assert list(frame.index.names) == ["subject", "task"]
    assert list(frame.index) == [("01", "x"), ("02", "x"), ("03", "x")]
    assert list(frame.columns) == ["id", "label", "score", "fault", "not_run"]
    assert frame.loc[("01", "x"), "score"] == 0.5 and frame.loc[("03", "x"), "label"] == "b"
    assert frame.loc[("02", "x"), "fault"] == "content" and pandas.isna(frame.loc[("02", "x"), "score"])
    plain = ResultSet([(Record(id="r0"), Output({"v": 1})), (Record(id="r1"), NotRun("cancelled"))]).to_frame()
    assert list(plain.index) == ["r0", "r1"] and plain.index.name == "id" and plain.loc["r1", "not_run"] == "cancelled"
