"""The empty tool-mask set every reproduce test runs under, as a fixture a test module can import.

`sandbox.tool_argv` fails closed (``E_SANDBOX_NO_MASKS``) outside `sandbox.tool_masks`, which an entry point with
a profile enters. ``conftest.py`` imports `_tool_masks` so it is autouse everywhere; a module whose tests build
or check environments imports it too, so the tests keep their masks when the matrix drops the conftest
(``--noconftest``).
"""

from collections.abc import Iterator

import pytest

from senselab.utils.reproduce import sandbox


@pytest.fixture(autouse=True)
def _tool_masks() -> Iterator[None]:
    """Give every test an empty mask set for build and env sandboxes, as an entry point with a profile would."""
    with sandbox.tool_masks(()):
        yield
