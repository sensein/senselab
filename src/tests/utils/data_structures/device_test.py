"""Device selection helpers.

The concrete-device-string tests exist because a bare ``"cuda"`` is not a device string
some libraries can parse, and the failure is silent: SpeechBrain logs a warning and picks
device 0 on its own. On a cluster that has to respect the allocation rather than guess.
"""

import pytest

# ── Concrete device strings for library run_opts ─────────────────────


def test_cuda_resolves_to_an_indexed_device_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """SpeechBrain parses its ``run_opts["device"]`` with ``device.split(":")``.

    Handing it a bare ``"cuda"`` makes that unpack fail, and it logs "Could not parse CUDA
    device string" and silently calls ``torch.cuda.set_device(0)``. On a single-GPU
    allocation that lands on the right card by luck; on a node where the caller selected a
    different one it moves the model somewhere the caller did not ask for.
    """
    import torch

    from senselab.utils.data_structures.device import DeviceType, device_run_opt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert device_run_opt(DeviceType.CUDA) == "cuda:3"


def test_cuda_respects_the_currently_selected_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """The index must come from ``torch.cuda.current_device()``, not a hardcoded 0.

    That is what makes this correct under Slurm: ``CUDA_VISIBLE_DEVICES`` masks the
    allocation so the granted GPU is index 0 inside the process, while a caller who called
    ``set_device`` on a multi-GPU box gets the card they chose.
    """
    import torch

    from senselab.utils.data_structures.device import DeviceType, device_run_opt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert device_run_opt(DeviceType.CUDA) == "cuda:0"


def test_cpu_and_mps_are_passed_through_unindexed() -> None:
    """Only CUDA is indexed. ``cpu:0``/``mps:0`` are not valid device strings."""
    from senselab.utils.data_structures.device import DeviceType, device_run_opt

    assert device_run_opt(DeviceType.CPU) == "cpu"
    assert device_run_opt(DeviceType.MPS) == "mps"


def test_cuda_without_a_visible_gpu_does_not_invent_an_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking for an index when CUDA is unavailable would raise inside torch.

    Returning the bare name lets the downstream library produce its own, clearer error
    rather than this helper failing first with a less useful one.
    """
    import torch

    from senselab.utils.data_structures.device import DeviceType, device_run_opt

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_run_opt(DeviceType.CUDA) == "cuda"
