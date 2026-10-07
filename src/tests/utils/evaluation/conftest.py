"""Markers for the evaluation tests.

``slow`` tests take more than a few seconds, ``network`` tests need to reach the internet and ``gpu``
tests need CUDA. The fast suite deselects all three with ``-m "not slow and not network and not gpu"``.
A ``gpu`` test also carries ``pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")``,
so selecting it on a CPU-only host skips it rather than failing.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the three markers so ``--strict-markers`` accepts them."""
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "network: needs to reach the internet")
    config.addinivalue_line("markers", "gpu: needs a CUDA device")
