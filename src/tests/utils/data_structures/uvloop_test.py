"""Test senselab init with uvloop."""

import asyncio
import sys

import pytest

try:
    import uvloop

    UVLOOP_AVAILABLE = True
except Exception:
    UVLOOP_AVAILABLE = False


@pytest.mark.skipif(not UVLOOP_AVAILABLE, reason="uvloop not available")
def test_senselab_init_with_uvloop() -> None:
    """Test senselab init with uvloop."""
    # Force uvloop policy
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    try:
        # Force fresh import of senselab to trigger __init__.py
        sys.modules.pop("senselab", None)
        import senselab  # noqa: F401

        async def dummy() -> int:
            return 42

        # Create and set the loop explicitly (works with uvloop + 3.11)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(dummy())
            assert result == 42
        finally:
            # Clean up
            if not loop.is_closed():
                loop.close()
            asyncio.set_event_loop(None)

    except Exception as e:
        pytest.fail(f"senselab __init__ raised an unexpected error with uvloop: {e}")
