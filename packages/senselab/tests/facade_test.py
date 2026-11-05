"""The run facade: `Dataset`, `use`, `plan`, `run`, `resume` and the `.map` of a configured task, on the fake stack."""

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import numpy
import pytest
import soundfile
from api_test import _echo_namespace, fake_stack  # noqa: F401 - `fake_stack` is the fixture of the fake stack
from senselab._api.facade import PlanKwargs, session
from senselab_core.runtime.execution.run import Output, ResultSet
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.profiles import ExecutionProfile

import senselab as sl

needs_runner = pytest.mark.skipif(importlib.util.find_spec("pydra") is None, reason="the pydra extra is not installed")


def _records(count: int = 3) -> list[dict[str, object]]:
    return [{"id": f"r{i}", "entities": {"n": str(i)}, "inputs": {"payload": {"v": i}}} for i in range(count)]


@pytest.fixture(autouse=True)
def session_restored() -> Iterator[None]:
    """Leave the session defaults as they were."""
    before = session()
    yield
    sl.use(execution=before.execution, config=before.config, executor=before.executor)


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
