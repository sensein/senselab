""".. include:: ../../README.md"""  # noqa: D415

from multiprocessing import set_start_method

import nest_asyncio

nest_asyncio.apply()

from senselab.utils.data_structures.pydra_helpers import *  # NOQA

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    # Already set
    pass
