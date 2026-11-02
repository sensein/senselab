"""Backend `transformers` of `pii_spans`: a pinned instruction-tuned language model in its locked isolated env."""

from pathlib import Path

from senselab_core.kernel.declare import REGISTRY, Env, backend

from senselab.text.tasks.pii_spans.transformers import MODEL_ID, WORKER_MODULE

_HERE = Path(__file__).parent
_WORKER_SOURCES = {
    WORKER_MODULE: str(_HERE / "transformers.py"),
    "senselab.text.tasks.pii_spans.engine": str(_HERE / "engine.py"),
    "senselab.item_faults": str(_HERE.parents[2] / "item_faults.py"),
}


@backend(
    task="pii_spans",
    name="transformers",
    version="1",
    stack="transformers",
    env=Env.isolated("pii-spans"),
    models=(MODEL_ID,),
    worker=WORKER_MODULE,
    worker_sources=_WORKER_SOURCES,
    devices=("cuda", "cpu"),
    registry=REGISTRY,
    load_stub={"stub": True, "device": "cpu"},
    batch_invariant=True,
)
class Transformers:
    """The catalogue's language model answering the profile's prompt with greedy decoding; each item alone."""
