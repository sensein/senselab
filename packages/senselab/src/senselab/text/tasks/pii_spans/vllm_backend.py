"""Backend `vllm` of `pii_spans`: the catalogue's model served by `VllmServer` in the `vllm-server` env."""

from importlib import resources

from senselab_core.kernel.declare import Env, ServerConfig, backend
from senselab_core.runtime.config import load_yaml

from senselab.text.tasks.pii_spans.engine import MODEL_ID
from senselab.text.tasks.pii_spans.vllm import WORKER_MODULE

SETTINGS_FILE = "pii_spans_vllm.yaml"
SETTINGS_FORMAT = "server-settings@1.0"


def server_config() -> ServerConfig:
    """Return the packaged server configuration.

    Raises:
        ValueError: if the file is not a `server-settings@1.0`.
    """
    raw = load_yaml(resources.files("senselab").joinpath("data", "server_settings", SETTINGS_FILE), SETTINGS_FILE)
    if raw.get("format") != SETTINGS_FORMAT:
        raise ValueError(f"{SETTINGS_FILE} is not a {SETTINGS_FORMAT}: {raw.get('format')!r}")
    return ServerConfig(settings=raw["settings"], startup_timeout_s=raw["startup_timeout_s"])


@backend(
    task="pii_spans",
    name="vllm",
    version="1",
    stack="vllm",
    env=Env.server("vllm-server"),
    models=(MODEL_ID,),
    worker=WORKER_MODULE,
    devices=("cuda",),
    load_stub={"stub": True, "device": "cpu"},
    server=server_config(),
    batch_invariant=True,
)
class Vllm:
    """The catalogue's language model answering the profile's prompt with greedy decoding; each item alone."""
