""".. include:: ../../README.md"""  # noqa: D415

import platform
from multiprocessing import set_start_method

if platform.system() == "Darwin" and platform.machine() != "arm64":
    raise RuntimeError(
        "Error: This package requires an ARM64 architecture on macOS "
        "since PyTorch 2.2.2+ does not support x86-64 on macOS."
    )


import nest_asyncio

nest_asyncio.apply()

from senselab.utils.data_structures.pydra_helpers import *  # NOQA

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    # Already set
    pass
