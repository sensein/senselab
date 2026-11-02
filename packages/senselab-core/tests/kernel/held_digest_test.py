"""The held-array digest cache under garbage collection."""

import subprocess
import sys

import pytest

_COLLECTING = """
import gc, numpy
from senselab_core.kernel._base import frozen_array, held_npy_digest
gc.set_threshold(1, 1, 1)
for i in range(3000):
    a = frozen_array(numpy.arange(4.0) + i)
    cycle = [a]
    cycle.append(cycle)
    held_npy_digest(a)
    del a, cycle
print("done")
"""


@pytest.mark.logic
def test_digests_survive_a_collection_that_forgets_another_array() -> None:
    """A collection inside `held_npy_digest` that frees a cached array forgets it without blocking the caller."""
    done = subprocess.run([sys.executable, "-c", _COLLECTING], capture_output=True, text=True, timeout=60, check=True)
    assert done.stdout.strip() == "done"
