"""Backend `vllm` of `pii_spans`: the catalogue's model served by `VllmServer` in the `vllm-server` env."""

from importlib import resources
from pathlib import Path

from senselab_core.kernel.declare import Env, ServerConfig, backend
from senselab_core.runtime.config import load_yaml

from senselab.text.tasks.pii_spans.engine import MODEL_ID
from senselab.text.tasks.pii_spans.vllm import WORKER_MODULE

SETTINGS_FILE = "pii_spans_vllm.yaml"
SETTINGS_FORMAT = "server-settings@2.0"


def server_config(source: Path | None = None) -> ServerConfig:
    """Return the server configuration in `source`, by default the packaged one.

    Raises:
        ValueError: if the file is not a `server-settings@2.0`; the message names the tag it has.
    """
    where = (
        source if source is not None else resources.files("senselab").joinpath("data", "server_settings", SETTINGS_FILE)
    )
    raw = load_yaml(where, SETTINGS_FILE)
    if raw.get("format") != SETTINGS_FORMAT:
        raise ValueError(f"{SETTINGS_FILE} is tagged {raw.get('format')!r}, not the {SETTINGS_FORMAT} this build reads")
    return ServerConfig(settings=raw["settings"])


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
