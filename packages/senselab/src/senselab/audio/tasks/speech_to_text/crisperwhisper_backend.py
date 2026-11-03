"""Backend `crisperwhisper` of `speech_to_text`: the CrisperWhisper library in its locked isolated env."""

from pathlib import Path

from pydantic import Field
from senselab_core.kernel.declare import BackendParams, Env, backend

from senselab.audio.tasks.speech_to_text.crisperwhisper import (
    DEFAULT_ENGINE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    WORKER_MODULE,
    Engine,
)

_HERE = Path(__file__).parent
_WORKER_SOURCES = {
    f"{__package__}.{name}": str(_HERE / f"{name}.py")
    for name in ("crisperwhisper", "token_confidence", "window_confidence")
} | {"senselab.item_faults": str(_HERE.parents[2] / "item_faults.py")}


class CrisperWhisperParams(BackendParams):
    """Backend params of `crisperwhisper`: the `engine` that decodes and the tokens a window may generate."""

    engine: Engine = DEFAULT_ENGINE
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1)


@backend(
    task="speech_to_text",
    name="crisperwhisper",
    version="1",
    stack="crisperwhisper",
    models=(MODEL_ID,),
    worker=WORKER_MODULE,
    worker_sources=_WORKER_SOURCES,
    load_stub={"stub": True, "device": "cpu"},
    batch_invariant=True,
    devices=("cuda", "mps", "cpu"),
    env=Env.isolated("crisperwhisper"),
    params=CrisperWhisperParams,
)
class CrisperWhisper:
    """The CrisperWhisper library, verbatim with word timestamps, over the one model it accepts."""
