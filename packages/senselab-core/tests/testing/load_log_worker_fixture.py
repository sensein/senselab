"""The fake worker, logging each load to the file `init["load_log"]` names."""

from typing import Any

from senselab_worker import fake

handle = fake.handle


def load(init: dict[str, Any]) -> object:
    """Append one line to the load log and return the fake's state."""
    with open(init["load_log"], "a") as log:
        log.write("load\n")
    return fake.load({k: v for k, v in init.items() if k != "load_log"})
