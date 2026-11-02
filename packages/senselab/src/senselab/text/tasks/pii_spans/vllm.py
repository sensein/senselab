"""Worker of backend `vllm` of `pii_spans`: the engine's prompt answered by a running `VllmServer`.

Runs in the calling process. `load` takes the running server in `init["server"]` (or `"stub": True` for the stub
model); `handle` is the engine's. Each item is one chat request, greedy and bounded by `MAX_NEW_TOKENS`.
"""

from collections.abc import Mapping
from typing import Any

from senselab.item_faults import init_model
from senselab.text.tasks.pii_spans.engine import MAX_NEW_TOKENS, State, StubModel, handle

__all__ = ["WORKER_MODULE", "ServerModel", "handle", "load"]

WORKER_MODULE = "senselab.text.tasks.pii_spans.vllm"
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir", "server"})
SAMPLING = {"temperature": 0, "max_tokens": MAX_NEW_TOKENS}


class ServerModel:
    """The served model: answers a prompt with one greedy chat completion of the attested server."""

    def __init__(self, server: Any) -> None:  # noqa: ANN401 - a `VllmServer`, which a worker module does not import
        """Hold the running `server`."""
        self.server = server

    def generate(self, prompt: str) -> str:
        """Return the server's answer to `prompt`.

        Raises:
            ValueError: if the reply has no completion text.
        """
        reply = self.server.chat([{"role": "user", "content": prompt}], sampling=SAMPLING)
        try:
            return str(reply.body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"the server's reply holds no completion ({type(exc).__name__})") from exc


def load(init: Mapping[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`) and `server`, or `"stub": True`; it may hold `device`,
    `faults`, the fields of `FakeFaults`, and `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, both `stub` and `model`, or a model
            without a `server`.
    """
    faults, model = init_model(init, _INIT_KEYS, "vllm")
    if model is None:
        return State(faults, StubModel())
    if init.get("server") is None:
        raise ValueError("vllm init takes the running `server` beside `model`")
    return State(faults, ServerModel(init["server"]))
