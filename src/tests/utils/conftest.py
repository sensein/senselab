"""This script includes some fixtures for pytest unit testing."""

import os

import httpx

# Submodule import, not `huggingface_hub.errors` attribute access: the package's lazy __getattr__
# resolves names it re-exports, not the `errors` submodule itself, so the attribute form raises
# AttributeError in any interpreter that has not already imported it -- e.g. a freshly spawned
# multiprocessing child re-importing a test module.
from huggingface_hub.errors import HfHubHTTPError

# Global variables for file paths
CHA_TALK_BANK_PATH = os.path.abspath(r"src/tests/data_for_testing/childes_clinical_eng_poler_match_166.cha")


def hub_error(status: int, cls: type = HfHubHTTPError) -> Exception:
    """Build a real ``huggingface_hub`` HTTP error carrying *status*.

    Real, not a double, because a double is what hid the bug these callers test: ``_is_transient``
    read ``.code`` / ``.status_code``, which no huggingface_hub error defines, and any stand-in
    given a ``.status_code`` attribute would have classified correctly while the real error --
    which carries its status only on ``.response.status_code`` -- did not.

    Args:
        status: The HTTP status the error should carry.
        cls: The huggingface_hub error class to instantiate. All of them take the same
            ``(message, *, response)`` signature.

    Returns:
        An instance of *cls* whose ``.response.status_code`` is *status*.
    """
    response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/org/model"))
    return cls(f"{status} from the Hub", response=response)
