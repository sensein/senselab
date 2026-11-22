"""A worker granted a device its env's torch cannot use fails at load with a typed `DeviceUnusable`, never loading."""

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from senselab_worker.codec import to_wire
from senselab_worker.device_check import DeviceUnusable, require_device
from senselab_worker.faults import FaultClass, classification
from senselab_worker.serve import EXIT_FATAL, serve

pytestmark = pytest.mark.logic

FAKE_TORCH = """
__version__ = {version!r}
class version:
    cuda = {cuda!r}
class cuda:
    @staticmethod
    def is_available():
        return {available!r}
class backends:
    class mps:
        @staticmethod
        def is_available():
            return {available!r}
"""


def _torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cuda: str | None, available: bool) -> None:
    """Put a stand-in `torch` first on the path: a build with CUDA `cuda` (None: a CPU build) seeing a GPU or not."""
    root = tmp_path / "site"
    (root / "torch").mkdir(parents=True)
    version = "2.11.0+cpu" if cuda is None else "2.11.0+cu128"
    (root / "torch" / "__init__.py").write_text(FAKE_TORCH.format(version=version, cuda=cuda, available=available))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.syspath_prepend(str(root))


@pytest.mark.parametrize(
    ("cuda", "available", "says"),
    [(None, False, "a build without CUDA"), ("12.8", False, "sees no usable GPU")],
)
def test_a_cuda_grant_the_envs_torch_cannot_use_is_a_typed_resource_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cuda: str | None, available: bool, says: str
) -> None:
    """A CPU torch build, or a CUDA build that sees no GPU, on a cuda grant raises `DeviceUnusable`, naming the fix."""
    _torch(tmp_path, monkeypatch, cuda=cuda, available=available)
    with pytest.raises(DeviceUnusable, match=says) as caught:
        require_device("cuda")
    assert "envs.torch_index" in str(caught.value)
    assert classification(caught.value) == (FaultClass.RESOURCE, False)


def test_a_usable_device_and_a_cpu_grant_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA build that sees its GPU passes; a cpu grant, or no device, is never checked and imports no torch."""
    _torch(tmp_path, monkeypatch, cuda="12.8", available=True)
    require_device("cuda")
    monkeypatch.delitem(sys.modules, "torch")
    for device in ("cpu", None):
        require_device(device)
        assert "torch" not in sys.modules


def test_the_worker_reports_the_mismatch_as_its_fatal_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker given `device: cuda` over a CPU torch build answers `fatal` with a resource `DeviceUnusable`."""
    _torch(tmp_path, monkeypatch, cuda=None, available=False)
    loads: list[dict[str, Any]] = []
    init = {"init": {"array_dir": str(tmp_path), "device": "cuda"}}
    rx = io.BytesIO(json.dumps(to_wire(init, array_dir=tmp_path, at="line")).encode() + b"\n")
    tx = io.BytesIO()
    code = serve(loads.append, lambda *_: [], rx, tx)
    error = json.loads(tx.getvalue().splitlines()[-1])["error"]
    assert code == EXIT_FATAL and loads == []
    assert (error["type"], error["fault"], error["gap"]) == ("DeviceUnusable", "resource", False)
