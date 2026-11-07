"""The batch invariance probe on the fake stack: held-out items at batch 1 and batch N, one line per backend."""

import importlib.util
import platform
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import Field
from senselab_core.kernel.declare import REGISTRY, BackendParams, BackendSpec, Env, Registry, TaskParams
from senselab_core.runtime import fake_backend
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import mit_lock_records

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


def _probe_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("batch_invariance_under_test", BENCHMARKS / "batch_invariance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _probe_module()


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.fixture(autouse=True)
def packaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The probe reads the registry declaring the fake stack, opens every env as this interpreter, caches in tmp."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    assert fake_backend.FAKE in REGISTRY.backends(fake_backend.ECHO.name)
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    yield
    use_packaged(previous)


@pytest.mark.logic
def test_probe_runs_plumbing_suite_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The plumbing suite's held-out echoes agree at batch 1 and at batch N, N their one formation's population."""
    suite = BENCHMARKS / "suites" / "plumbing.yaml"
    [bench] = probe.harness().Suite.load(suite).benchmarks
    held = sum(probe.harness().split_of(f"echo-{n:02d}") == probe.HELDOUT for n in range(bench.items))
    code = probe.main([str(suite), "--out", str(tmp_path / "out"), "--sha", "abc", "--job", "7"])
    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"abc 7 fake.echo/fake: agree {held}/{held}, formation mixed, batch {held}, max diff none; "
        f"repeat agree {held}/{held}, max diff none"
    ]
    assert {p.name for p in (tmp_path / "out" / "echo").iterdir()} == {"batch-1", f"batch-{held}", "batch-1-repeat"}


@pytest.mark.logic
def test_probe_reports_a_task_no_device_here_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A suite task whose backend runs on no device the profile lists is one `not placeable` line, nothing run."""
    override = tmp_path / "mps.yaml"
    override.write_text("devices: [mps]\n")
    suite = str(BENCHMARKS / "suites" / "plumbing.yaml")
    args = [suite, "--override", str(override), "--out", str(tmp_path / "out"), "--sha", "abc", "--job", "7"]
    assert probe.main(args) == 0
    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("abc 7 fake.echo: not placeable: ") and not (tmp_path / "out" / "echo").exists()


SIZED_TASK = fake_backend.ECHO.model_copy(update={"name": "probe_test.sized"})
SIZED = BackendSpec(
    task=SIZED_TASK.name,
    name="sized",
    version="1",
    stack="sized",
    env=Env.host(),
    devices=("cpu",),
    batch_invariant=True,
    worker="batch_sized_backend_fixture",
    load_stub={},
)
UNKEYED = SIZED.model_copy(update={"name": "unkeyed", "stack": "unkeyed", "batch_invariant": False})


class StepParams(TaskParams):
    """Params of the scored task: the confidence per item of the forward's batch size."""

    step: float


class ForwardParams(BackendParams):
    """Backend params of the scored backend: items per forward."""

    batch_size: int = Field(default=1, ge=1)


SCORED_TASK = SIZED_TASK.model_copy(
    update={"name": "probe_test.scored", "outputs": {"transcript": "Transcript"}, "params": StepParams}
)
SCORED = SIZED.model_copy(
    update={
        "task": SCORED_TASK.name,
        "name": "scored",
        "stack": "scored",
        "worker": "scored_backend_fixture",
        "params": ForwardParams,
    }
)


def _sized_registry(backend: BackendSpec) -> Registry:
    registry = Registry()
    registry.add_task(SCORED_TASK if backend is SCORED else SIZED_TASK)
    registry.add_backend(backend)
    return registry


@pytest.mark.logic
def test_probe_reports_disagreement_of_a_batch_dependent_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend whose outputs depend on its request size agrees on no item, and the line names the differing field."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _sized_registry(SIZED), mit_lock_records))
    run, execution = probe.RunConfig(), ExecutionProfile.load()
    inputs = [{"payload": {"n": n}} for n in range(3)]
    jobs, placement = probe.node_jobs(SIZED_TASK.name, inputs, {}, None, run=run, execution=execution)
    found = probe.probe(jobs, placement, run=run, execution=execution, work=tmp_path)
    assert (
        found.line("abc", "7")
        == f"abc 7 {SIZED_TASK.name}/sized: agree 0/3, formation mixed, batch 3, max diff value=n/a; "
        "repeat agree 3/3, max diff none"
    )


@pytest.mark.logic
def test_probe_marks_a_single_formation_as_no_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend sent one item per request runs nothing and reports no evidence, however many items there are."""
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _sized_registry(UNKEYED), mit_lock_records))
    run, execution = probe.RunConfig(), ExecutionProfile.load()
    inputs = [{"payload": {"n": n}} for n in range(3)]
    jobs, placement = probe.node_jobs(SIZED_TASK.name, inputs, {}, None, run=run, execution=execution)
    found = probe.probe(jobs, placement, run=run, execution=execution, work=tmp_path)
    assert (
        found.line("abc", "7")
        == f"abc 7 {SIZED_TASK.name}/unkeyed: not batched — no evidence (formation single, batch 3)"
    )
    assert not tmp_path.joinpath("batch-1").exists()


@pytest.mark.logic
def test_probe_compares_wire_forms_under_their_rules_with_a_batched_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transcript whose confidence moves with the forward's batch size, within its threshold, agrees on every item.

    The gap is `step` times N - 1, so the backend's own `batch_size` param was N in the batched run and 1 at batch 1.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _sized_registry(SCORED), mit_lock_records))
    run, execution = probe.RunConfig(), ExecutionProfile.load()
    inputs = [{"payload": {"n": n}} for n in range(3)]
    step = run.thresholds.by_name["probability"].atol / (2 * len(inputs))
    jobs, placement = probe.node_jobs(SCORED_TASK.name, inputs, {"step": step}, None, run=run, execution=execution)
    found = probe.probe(jobs, placement, run=run, execution=execution, work=tmp_path)
    assert (found.agree, found.repeat, found.batch, found.repeat_largest) == (3, 3, 3, {})
    assert found.largest == {"transcript.segments[*].confidence": pytest.approx(step * (found.batch - 1))}


FAILING = SIZED.model_copy(update={"name": "failing", "stack": "failing", "worker": "failing_backend_fixture"})


def _failing_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: Sequence[bool]) -> str:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, _sized_registry(FAILING), mit_lock_records))
    run, execution = probe.RunConfig(), ExecutionProfile.load()
    inputs = [{"payload": {"n": n, "fail": fail}} for n, fail in enumerate(fails)]
    jobs, placement = probe.node_jobs(SIZED_TASK.name, inputs, {}, None, run=run, execution=execution)
    return str(probe.probe(jobs, placement, run=run, execution=execution, work=tmp_path).line("abc", "7"))


@pytest.mark.logic
def test_probe_reports_items_whose_job_failed_as_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend failing every item agrees on none and is not evidence; its error type and count are on the line."""
    assert _failing_line(tmp_path, monkeypatch, [True] * 3) == (
        f"abc 7 {SIZED_TASK.name}/failing: agree 0/0, unavailable 3/3 (ValueError: 3), formation mixed, batch 3, "
        "max diff none; repeat agree 0/0, unavailable 3/3 (ValueError: 3), max diff none — no evidence"
    )


@pytest.mark.logic
def test_probe_counts_agreement_over_the_items_available_in_both_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With one of three items failing, the other two agree and the failure is counted apart."""
    assert _failing_line(tmp_path, monkeypatch, [False, True, False]) == (
        f"abc 7 {SIZED_TASK.name}/failing: agree 2/2, unavailable 1/3 (ValueError: 1), formation mixed, batch 3, "
        "max diff none; repeat agree 2/2, unavailable 1/3 (ValueError: 1), max diff none — no evidence"
    )
