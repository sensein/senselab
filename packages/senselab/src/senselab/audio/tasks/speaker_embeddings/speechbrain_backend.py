"""Backend `speechbrain` of `speaker_embeddings`: the SpeechBrain ECAPA encoder in its locked isolated env."""

from pathlib import Path

from senselab_core.kernel.declare import REGISTRY, Env, backend

from senselab.audio.tasks.speaker_embeddings.speechbrain import MODEL_ID, WORKER_MODULE

_HERE = Path(__file__).parent
_WORKER_SOURCES = {
    WORKER_MODULE: str(_HERE / "speechbrain.py"),
    "senselab.audio.item_faults": str(_HERE.parents[1] / "item_faults.py"),
}


@backend(
    task="speaker_embeddings",
    name="speechbrain",
    version="1",
    stack="speechbrain",
    env=Env.isolated("speechbrain"),
    models=(MODEL_ID,),
    worker=WORKER_MODULE,
    worker_sources=_WORKER_SOURCES,
    devices=("cuda", "mps", "cpu"),
    registry=REGISTRY,
    load_stub={"stub": True, "device": "cpu"},
    batch_invariant=True,
)
class SpeechBrain:
    """The SpeechBrain speaker encoder over the pinned catalogue model it is called with; each item is encoded alone."""
