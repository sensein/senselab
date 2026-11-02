"""Worker of backend `transformers` of `pii_spans`: a causal language model answering the engine's prompt.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model); `handle` is the engine's.
Each item is generated alone. `torch` and `transformers` load when the model is built.
"""

from collections.abc import Mapping
from typing import Any

from senselab.item_faults import init_model
from senselab.text.tasks.pii_spans.engine import MAX_NEW_TOKENS, MODEL_ID, State, StubModel, handle

__all__ = ["MODEL_ID", "WORKER_MODULE", "Generator", "handle", "load"]

WORKER_MODULE = "senselab.text.tasks.pii_spans.transformers"
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir"})


class Generator:
    """A causal language model and its tokenizer, loaded from the pinned commit of the model."""

    def __init__(self, model: Mapping[str, Any], device: str) -> None:
        """Download the snapshot at the model's commit and load the tokenizer and weights from that local copy."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - loads when the generator is built
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415 - as above

        snapshot = snapshot_download(str(model["id"]), revision=str(model["identity"]))
        self.tokenizer: Any = AutoTokenizer.from_pretrained(snapshot)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            snapshot, dtype="auto", device_map="auto" if device == "cuda" else device
        )
        self.model.train(False)

    def generate(self, prompt: str) -> str:
        """Return the model's greedy answer to `prompt`, at most `MAX_NEW_TOKENS` tokens."""
        import torch  # noqa: PLC0415 - loads when the generator is built

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return str(self.tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True`; it may hold `faults`, the fields of `FakeFaults`, and `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    faults, model = init_model(init, _INIT_KEYS, "transformers")
    if model is None:
        return State(faults, StubModel())
    return State(faults, Generator(model, str(init.get("device", "cpu"))))
