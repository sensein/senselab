"""Shared fixtures for the runtime tests."""

import importlib
import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from huggingface_hub import errors
from huggingface_hub.errors import HfHubHTTPError
from lease_clock_fixture import ManualClock


def _hub_client() -> ModuleType:
    """Return the HTTP client module `huggingface_hub` builds its errors on (`httpx` in 1.x, `httpx2` in 2.x)."""
    return importlib.import_module(vars(errors)["HTTPError"].__module__.partition(".")[0])


@pytest.fixture
def hub_client() -> ModuleType:
    """The HTTP client module `huggingface_hub` builds its errors on (`httpx` in 1.x, `httpx2` in 2.x)."""
    return _hub_client()


@pytest.fixture
def hub_error() -> Callable[..., Exception]:
    """Build a real ``huggingface_hub`` HTTP error over a response of the Hub's own HTTP client."""

    def _make(status: int, cls: type = HfHubHTTPError) -> Exception:
        client = _hub_client()
        response = client.Response(status, request=client.Request("GET", "https://huggingface.co/org/model"))
        error: Exception = cls(f"{status} from the Hub", response=response)
        return error

    return _make


@pytest.fixture
def backends_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let worker processes import the fixture backends as `backends.<name>`."""
    here = str(Path(__file__).parent)
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", here if not existing else f"{here}{os.pathsep}{existing}")


@pytest.fixture
def lease_clock(slack_tests: dict[str, Any], tier_bound: float) -> Callable[..., ManualClock]:
    """Build a `ManualClock`: polls and step from the test slack file, hang bound a fraction of the tier's."""
    section = slack_tests["lease_clock"]

    def make(**kwargs: Any) -> ManualClock:  # noqa: ANN401 - `ManualClock`'s own keyword arguments
        return ManualClock(
            poll_s=section["poll_s"], step_s=section["step_s"], hang_s=section["hang_fraction"] * tier_bound, **kwargs
        )

    return make


@pytest.fixture
def no_leaked_processes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[subprocess.Popen[bytes]]]:
    """Fail a test that leaves a worker process running or unreaped.

    Records every ``Popen`` the client creates and asserts each has been waited on
    (``returncode`` is set only by ``poll``/``wait``), so a zombie counts as a leak and
    children spawned by other tests, such as multiprocessing helpers, are ignored.
    """
    spawned: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    class _Recording(real):  # type: ignore[valid-type, misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(subprocess, "Popen", _Recording)
    yield spawned
    monkeypatch.undo()
    leaked = [p.pid for p in spawned if p.returncode is None]
    for p in spawned:
        if p.returncode is None:
            p.kill()
            p.wait()
    assert leaked == []


@pytest.fixture
def array_parent(tmp_path: Path) -> Iterator[Path]:
    """A parent for the client's array directory, which must hold nothing after the test."""
    path = tmp_path / "arrays"
    path.mkdir()
    yield path
    assert list(path.iterdir()) == []
