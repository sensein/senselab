""".. include:: ../../README.md"""  # noqa: D415

import asyncio
import platform
from multiprocessing import set_start_method

import nest_asyncio

# Raise error on incompatible macOS architecture
if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError(
        "Error: This package requires an ARM64 architecture on macOS "
        "since PyTorch 2.2.2+ does not support x86-64 on macOS."
    )


# Conditionally apply nest_asyncio to avoid uvloop conflict
def safe_apply_nest_asyncio() -> None:
    """Apply nest_asyncio to avoid uvloop conflict."""
    try:
        loop = asyncio.get_event_loop()
        if "uvloop" not in str(type(loop)):
            nest_asyncio.apply()
    except Exception as e:
        print(f"nest_asyncio not applied: {e}")


safe_apply_nest_asyncio()

# Ensure multiprocessing start method is 'spawn'
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Method already set
