"""A `Grant` is what a unit's workers are granted: device or host memory, quantum ceilings."""

import pytest
from senselab_core.runtime.execution_settings import PoolSettings, ResourceClass
from senselab_core.runtime.sizing import Grant


@pytest.mark.logic
def test_a_grant_takes_device_memory_on_a_device_and_host_memory_otherwise() -> None:
    """On a device the class's device cap; else its memory, the host's last; the class's time bounds the quantum."""
    pool = PoolSettings(
        quantum_items=100,
        quantum_s=300,
        poll_s=30,
        first_poll_s=1,
        presence_refresh_s=30,
        segment_items=512,
        admit_threads=8,
    )
    capped = ResourceClass(device="cuda", max_workers=4, mem_bytes=2**36, memory_cap_bytes=2**33, time_s=60)
    assert Grant.of(capped, pool, on_device=True).model_dump() == {
        "max_workers": 4,
        "memory_bytes": 2**33,
        "quantum_items": 100,
        "quantum_s": 60,
        "devices": None,
        "on_device": True,
        "shared_host": False,
    }
    assert Grant.of(capped, pool, devices=2).devices == 2
    assert Grant.of(capped, pool).memory_bytes == 2**36 and not Grant.of(capped, pool, 2**40).shared_host
    bare = ResourceClass(device="cpu", max_workers=1)
    assert Grant.of(bare, pool, devices=2).devices is None
    assert Grant.of(bare, pool).memory_bytes is None and Grant.of(bare, pool, 2**30).memory_bytes == 2**30
    assert Grant.of(bare, pool, 2**30).shared_host and not Grant.of(bare, pool).shared_host
