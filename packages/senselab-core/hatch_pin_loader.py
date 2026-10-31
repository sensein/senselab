"""Hatchling metadata hook of senselab-core: loads the one shared pin hook, without a symlink.

In the repository the shared hook is ``hatch_build_pins.py`` at the workspace root. The sdist carries a copy
next to this file as ``_shared_hatch_build_pins.py``, so a build from the sdist needs nothing outside it.
"""

import importlib.util  # noqa: TID251
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SHARED = next(
    p for p in (_HERE / "_shared_hatch_build_pins.py", _HERE.parents[1] / "hatch_build_pins.py") if p.is_file()
)
_spec = importlib.util.spec_from_file_location("_senselab_shared_pin_hook", _SHARED)  # noqa: TID251
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)  # noqa: TID251
_spec.loader.exec_module(_module)

PinHook = _module.PinHook
