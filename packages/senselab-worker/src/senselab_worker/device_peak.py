"""The device memory a request allocated: its peak and what stays allocated after it, read from torch's CUDA stats.

`torch_meter` measures a call when the process has imported torch and initialised CUDA (a backend running on a
GPU), and gives None otherwise; torch is never imported here. It reads the current device, the one card a worker
is given (`CUDA_VISIBLE_DEVICES`).
"""

import sys
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

Meter = Callable[[Callable[[], T]], tuple[T, tuple[int, int] | None]]
"""Run a call and return its result with `(peak, allocated after)` device bytes, or None where unmeasured."""


def torch_meter(call: Callable[[], T]) -> tuple[T, tuple[int, int] | None]:
    """Run `call` between a reset and a read of the current CUDA device's allocation peak; None without CUDA in use."""
    cuda = getattr(sys.modules.get("torch"), "cuda", None)
    if cuda is None or not cuda.is_available() or not cuda.is_initialized():
        return call(), None
    cuda.reset_peak_memory_stats()
    result = call()
    return result, (int(cuda.max_memory_allocated()), int(cuda.memory_allocated()))
