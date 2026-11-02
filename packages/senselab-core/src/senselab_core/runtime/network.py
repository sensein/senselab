"""The network gate every component that could reach the network consults: the profile's `network` policy."""

from typing import Literal

Network = Literal["online", "offline"]
"""An execution profile's network policy."""

OFFLINE_VARIABLES = {"HF_HUB_OFFLINE": "1"}
"""The variables every process started under an offline profile is given."""


class Offline(RuntimeError):
    """An operation that needs the network, refused because the profile is offline."""


def require_online(network: Network, what: str) -> None:
    """Return when `network` is `online`.

    Raises:
        Offline: `network` is `offline`; the message names `what`.
    """
    if network == "offline":
        raise offline(what)


def offline(what: str) -> Offline:
    """Return the `Offline` refusing `what`."""
    return Offline(f"{what} needs the network, and the execution profile is offline")


def network_variables(network: Network) -> dict[str, str]:
    """Return the variables a process started under `network` is given."""
    return dict(OFFLINE_VARIABLES) if network == "offline" else {}
