"""The batch invariance probe on the fake stack: held-out items at batch 1 and batch N, one line per backend."""

import importlib.util
import platform
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from senselab_core.kernel.declare import REGISTRY, BackendSpec, Env, Registry
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
        f"abc 7 fake.echo/fake: agree {held}/{held}, formation mixed, batch {held}, max diff none"
    ]
    assert (tmp_path / "out" / "echo" / "batch-1").is_dir() and (tmp_path / "out" / "echo" / f"batch-{held}").is_dir()


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


def _sized_registry(backend: BackendSpec) -> Registry:
    registry = Registry()
    registry.add_task(SIZED_TASK)
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
        == f"abc 7 {SIZED_TASK.name}/sized: agree 0/3, formation mixed, batch 3, max diff value=n/a"
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
