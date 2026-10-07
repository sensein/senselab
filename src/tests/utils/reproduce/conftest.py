"""Markers for the reproduce tests.

``slow`` tests build real venvs and need network; ``network`` tests need a compute node with
network outside the sandbox. Neither runs unless selected with ``-m``.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the two markers so ``--strict-markers`` accepts them."""
    config.addinivalue_line("markers", "slow: builds real environments and needs network")
    config.addinivalue_line("markers", "network: needs a compute node with network, outside the sandbox")
