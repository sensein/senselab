"""Backend `speechbrain` of `speaker_embeddings`: the SpeechBrain ECAPA encoder in its locked isolated env."""

from pathlib import Path

from senselab_core.kernel.declare import Env, backend

from senselab.audio.tasks.speaker_embeddings.speechbrain import MIN_INPUT_NS, MODEL_ID, WORKER_MODULE

_HERE = Path(__file__).parent
_WORKER_SOURCES = {
    WORKER_MODULE: str(_HERE / "speechbrain.py"),
    "senselab.item_faults": str(_HERE.parents[2] / "item_faults.py"),
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
    load_stub={"stub": True, "device": "cpu"},
    batch_invariant=True,
    min_input_ns=MIN_INPUT_NS,
)
class SpeechBrain:
    """The SpeechBrain speaker encoder over the pinned catalogue model it is called with; each item is encoded alone."""
