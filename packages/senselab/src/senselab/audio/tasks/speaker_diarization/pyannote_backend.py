"""Backend `pyannote` of `diarize`: the pyannote.audio pipeline in its locked isolated env."""

from pathlib import Path

from senselab_core.kernel.declare import REGISTRY, BackendParams, Env, backend

from senselab.audio.tasks.speaker_diarization.pyannote import MODEL_ID, WORKER_MODULE

_HERE = Path(__file__).parent
_WORKER_SOURCES = {
    WORKER_MODULE: str(_HERE / "pyannote.py"),
    "senselab.audio.item_faults": str(_HERE.parents[1] / "item_faults.py"),
}


class PyannoteParams(BackendParams):
    """Backend params of `pyannote`: `exclusive` returns the partition (one speaker per instant), else the overlap."""

    exclusive: bool = True


@backend(
    task="diarize",
    name="pyannote",
    version="1",
    stack="pyannote",
    env=Env.isolated("pyannote"),
    models=(MODEL_ID,),
    params=PyannoteParams,
    devices=("cuda", "mps", "cpu"),
    batch_invariant=True,
    worker=WORKER_MODULE,
    load_stub={"stub": True, "device": "cpu"},
    worker_sources=_WORKER_SOURCES,
    registry=REGISTRY,
)
class Pyannote:
    """The pyannote.audio diarization pipeline over the pinned catalogue model it is called with."""
