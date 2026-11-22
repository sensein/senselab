"""The check a worker makes before it loads: that the env's torch can use the device the worker was granted.

`require_device` imports torch only for a `cuda` or `mps` device, and only when the env has torch; a torch-free env,
and a `cpu` device, pass unchecked.
"""

import importlib
import importlib.util
import os

from senselab_worker.faults import ResourceFault


class DeviceUnusable(ResourceFault):
    """The env's torch cannot use the device the worker was granted; the message says which build and what to do."""


def require_device(device: object) -> None:
    """Raise `DeviceUnusable` when `device` is `cuda` or `mps` and the env's torch cannot use it.

    Raises:
        DeviceUnusable: torch was built without the device's backend, or sees no usable device.
    """
    if device not in ("cuda", "mps") or importlib.util.find_spec("torch") is None:
        return
    torch = importlib.import_module("torch")
    version = getattr(torch, "__version__", "unknown")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceUnusable(f"device mps was granted, and torch {version} in this env cannot use it")
        return
    if torch.version.cuda is None:
        raise DeviceUnusable(
            f"device cuda was granted, and this env's torch {version} is a build without CUDA; the env was built "
            "from a CPU wheel index: rebuild it for a cuda resource class, or set the execution profile's "
            "envs.torch_index to a CUDA index"
        )
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise DeviceUnusable(
            f"device cuda was granted, and torch {version} (CUDA {torch.version.cuda}) sees no usable GPU "
            f"(CUDA_VISIBLE_DEVICES={visible}): the node's driver may be older than the build's CUDA; set the "
            "execution profile's envs.torch_index to an older CUDA index, or check the grant"
        )
