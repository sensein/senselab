"""Worker of backend `transformers` of `pii_spans`: a causal language model answering the engine's prompt.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model); `handle` answers one
`text-spans@1.0` wire form, or one typed fault, per item. Each item is generated alone. Imports are stdlib, numpy
and `senselab_worker`; `torch` and `transformers` load when the model is built.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from senselab_worker.fake import FakeFaults

from senselab.audio.item_faults import init_model, raise_injected, typed
from senselab.text.tasks.pii_spans.engine import (
    MAX_NEW_TOKENS,
    build_prompt,
    parse_completion,
    spans_wire_form,
    text_of,
)

WORKER_MODULE = "senselab.text.tasks.pii_spans.transformers"
MODEL_ID = "google/gemma-4-31B-it-qat-w4a16-ct"
STUB_CATEGORY = "ID"
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir"})
_DIGITS = re.compile(r"\d+")


class StubModel:
    """The fake stack's model: answers with every digit run of the text as one `ID` span."""

    def generate(self, prompt: str) -> str:
        """Return the JSON answer marking each distinct digit run of the prompt's text."""
        runs = sorted(set(_DIGITS.findall(text_of(prompt))))
        return (
            '{"spans": ['
            + ", ".join(f'{{"quote": "{r}", "category": "{STUB_CATEGORY}", "score": 0.5}}' for r in runs)
            + "]}"
        )


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


@dataclass(frozen=True)
class State:
    """A loaded backend: the model (a stub or the generator) and the faults to inject."""

    faults: FakeFaults
    model: Any  # noqa: ANN401 - a StubModel or a Generator


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


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"spans": <text-spans@1.0 wire form>}}` per item, or its typed fault.

    `params` holds the task's `profile` and `family` and the (empty) `backend_params`.

    Raises:
        ValueError: on a backend param; this backend takes none.
    """
    if params.get("backend_params"):
        raise ValueError(f"unknown backend params {sorted(params['backend_params'])}; known: []")
    profile = params["profile"]
    names = [c["name"] for c in profile["categories"]]
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            raise_injected(item["id"], state.faults)
            text = item["inputs"]["text"]
            completion = state.model.generate(build_prompt(profile, text["content"], params.get("family")))
            spans = parse_completion(completion, text["content"], names)
            results.append({"outputs": {"spans": spans_wire_form(text, spans)}})
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": typed(exc)})
    return results
