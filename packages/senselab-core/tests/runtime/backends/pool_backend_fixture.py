"""A host backend for pool tests whose loads and faults any process can count and plant, through files.

Each load appends its pid to the file `SENSELAB_TEST_LOADS` names. A load raises `MemoryError` while the file
`SENSELAB_TEST_OOM_LOADS` names holds a positive count, taking one from it, and `RuntimeError` while
`SENSELAB_TEST_BROKEN` names an existing file. An item sleeps `SENSELAB_TEST_SLEEP_S` seconds, then returns its
input as `value`.
"""

import os
import time
from pathlib import Path
from typing import Any


def _take(variable: str) -> bool:
    """Take one from the count in the file `variable` names; whether there was one to take."""
    named = os.environ.get(variable)
    if not named or not Path(named).is_file():
        return False
    left = int(Path(named).read_text() or 0)
    if left <= 0:
        return False
    Path(named).write_text(str(left - 1))
    return True


def load(init: dict[str, Any]) -> object:
    """Record the load; raise a planted out-of-memory or broken load."""
    loads = os.environ.get("SENSELAB_TEST_LOADS")
    if loads:
        with Path(loads).open("a") as written:
            written.write(f"{os.getpid()}\n")
    if _take("SENSELAB_TEST_OOM_LOADS"):
        raise MemoryError("planted: the load ran out of memory")
    broken = os.environ.get("SENSELAB_TEST_BROKEN")
    if broken and Path(broken).exists():
        raise RuntimeError("planted: the load is broken")
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Answer each item with its input, after the planted sleep."""
    time.sleep(float(os.environ.get("SENSELAB_TEST_SLEEP_S", "0")))
    return [{"outputs": {"value": str(next(iter(item["inputs"].values())))}} for item in items]
