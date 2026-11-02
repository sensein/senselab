"""Worker of backend `speechbrain` of `speaker_embeddings`: a SpeechBrain `EncoderClassifier`, one vector per item.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model); `handle` answers one
`embedding@1.0` wire form, or one typed fault, per item. Each item is encoded alone: no item is padded to another's
length. Imports are stdlib, numpy and `senselab_worker`; `torch` and the library load when the model is built.
"""

import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy
from senselab_worker.fake import FakeFaults

from senselab.item_faults import init_model, mono_samples

WORKER_MODULE = "senselab.audio.tasks.speaker_embeddings.speechbrain"
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
EXPECTED_RATE = 16_000
STUB_DIMENSION = 192
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir"})


class StubModel:
    """The fake stack's model: a vector derived from the samples, so different audio gives different vectors."""

    def embed(self, samples: numpy.ndarray) -> numpy.ndarray:
        """Return a finite `STUB_DIMENSION` vector seeded by the digest of `samples`."""
        seed = int.from_bytes(hashlib.sha256(samples.tobytes()).digest()[:8], "big")
        return numpy.random.default_rng(seed).standard_normal(STUB_DIMENSION).astype(numpy.float32)


class Encoder:
    """The SpeechBrain speaker encoder, loaded from the pinned commit of the model onto one device."""

    def __init__(self, model: Mapping[str, Any], device: str) -> None:
        """Download the snapshot at the model's commit and build the classifier from that local copy."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - loads when the encoder is built
        from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415 - as above

        snapshot = snapshot_download(str(model["id"]), revision=str(model["identity"]))
        self.device = device
        self.classifier = EncoderClassifier.from_hparams(
            source=snapshot, savedir=tempfile.mkdtemp(prefix="speechbrain-"), run_opts={"device": device}
        )

    def embed(self, samples: numpy.ndarray) -> numpy.ndarray:
        """Return the 1-D embedding of `samples` as float32."""
        import torch  # noqa: PLC0415 - loads when the encoder is built

        with torch.inference_mode():
            vector = self.classifier.encode_batch(torch.from_numpy(samples[None, :]).to(self.device))
        return numpy.asarray(vector.reshape(-1).detach().cpu().numpy(), dtype=numpy.float32)


@dataclass(frozen=True)
class State:
    """A loaded backend: the model (a stub or the encoder), the space its vectors live in, and the faults to inject."""

    model: Any  # noqa: ANN401 - a StubModel or an Encoder
    space: str
    faults: FakeFaults


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True`; it may hold `faults`, the fields of `FakeFaults`, and `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    faults, model = init_model(init, _INIT_KEYS, "speechbrain")
    if model is None:
        return State(StubModel(), f"stub:{MODEL_ID}", faults)
    return State(Encoder(model, str(init.get("device", "cpu"))), f"{model['id']}@{model['identity']}", faults)


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"embedding": <embedding@1.0 wire form>}}` per item, or its typed fault.

    Raises:
        ValueError: on a backend param; this backend takes none.
    """
    if params.get("backend_params"):
        raise ValueError(f"unknown backend params {sorted(params['backend_params'])}; known: []")
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            samples, _ = mono_samples(item, state.faults, EXPECTED_RATE, "speechbrain")
            vector = state.model.embed(samples)
            results.append(
                {"outputs": {"embedding": {"format": "embedding@1.0", "vector": vector, "space": state.space}}}
            )
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": exc})
    return results
