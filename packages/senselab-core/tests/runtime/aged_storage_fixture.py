"""Storage time a test advances: a `ManualClock` pinned to the filesystem's own time when it starts.

Files made after that carry stamps at or just past the pinned origin, so advancing the clock ages them, and
`plant` stamps a file as if it were made `ago` seconds before the clock's present.
"""

import os
from collections.abc import Callable
from pathlib import Path

from lease_clock_fixture import ManualClock
from senselab_core.runtime.clock import SystemClock
from senselab_core.runtime.storage import StorageDir


class StorageTime:
    """A `ManualClock` whose storage time starts at `root`'s filesystem time.

    Attributes:
        clock: The clock to inject.
        origin: The filesystem time the clock started at.
    """

    def __init__(self, make_clock: Callable[..., ManualClock], root: Path) -> None:
        """Pin a clock from `make_clock` to the time `root` (made if missing) reports now."""
        with StorageDir.open_root(root, create=True) as top:
            self.origin = SystemClock().storage_now(top)
        self.clock = make_clock(storage_origin=self.origin)

    def now(self) -> float:
        """Return the clock's storage time."""
        return self.origin + self.clock.now()

    def plant(self, path: Path, ago: float = 0.0) -> None:
        """Stamp `path` as made `ago` seconds before the clock's present, without following a symlink."""
        then = self.now() - ago
        os.utime(path, (then, then), follow_symlinks=False)
