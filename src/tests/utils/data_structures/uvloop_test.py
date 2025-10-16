"""Test senselab init with uvloop."""

import asyncio
import sys

import pytest

try:
    import uvloop

    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False


@pytest.mark.skipif(not UVLOOP_AVAILABLE, reason="uvloop not available")
def test_senselab_init_with_uvloop() -> None:
    """Test senselab init with uvloop."""
    # Force the use of uvloop in the test
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    try:
        # Force fresh import of senselab to trigger __init__.py
        sys.modules.pop("senselab", None)
        import senselab  # noqa: F401

        # If we reach here, the init did not crash due to uvloop conflict
        async def dummy() -> int:
            """Dummy async function."""
            return 42

        result = asyncio.run(dummy())
        assert result == 42

    except Exception as e:
        pytest.fail(f"senselab __init__ raised an unexpected error with uvloop: {e}")
