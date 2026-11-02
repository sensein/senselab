"""Worker of backend `pyannote` of `diarize`: the pyannote.audio diarization pipeline, one turn table per item.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model); `handle` answers one
`intervals@1.0` wire form of speaker turns, or one typed fault, per item. Imports are stdlib, numpy and
`senselab_worker`; `torch` and the library load when the pipeline is built. A gated model reads `HF_TOKEN` from the
environment.
"""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy
from senselab_worker.fake import FakeFaults

from senselab.item_faults import init_model, mono_samples

WORKER_MODULE = "senselab.audio.tasks.speaker_diarization.pyannote"
MODEL_ID = "pyannote/speaker-diarization-community-1"
EXPECTED_RATE = 16_000
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir"})
_HINTS = ("num_speakers", "min_speakers", "max_speakers")
_BACKEND_PARAMS = ("exclusive",)
_NS_PER_S = 1_000_000_000

Turns = list[tuple[str, float, float]]


class StubModel:
    """The fake stack's model: one speaker per equal share of the audio, derived from its samples."""

    def diarize(self, samples: numpy.ndarray, hints: Mapping[str, int | None], exclusive: bool) -> Turns:  # noqa: ARG002
        """Return `hints["num_speakers"]` (default 2) consecutive turns over `samples`, none for empty audio."""
        count = hints["num_speakers"] or 2
        seconds = samples.size / EXPECTED_RATE
        offset = hashlib.sha256(samples.astype(numpy.float32).tobytes()).digest()[0] % count
        step = seconds / count
        return [(f"SPEAKER_{(i + offset) % count:02d}", i * step, (i + 1) * step) for i in range(count) if step > 0]


class Pipeline:
    """The pyannote.audio pipeline, loaded from the pinned commit of the model onto one device."""

    def __init__(self, model: Mapping[str, Any], device: str) -> None:
        """Build the pipeline; the token for a gated model is `HF_TOKEN`.

        Raises:
            ValueError: when the hub returns no pipeline for the model.
        """
        import torch  # noqa: PLC0415 - loads when the pipeline is built
        from pyannote.audio import Pipeline as Loaded  # noqa: PLC0415 - as above

        pipeline = Loaded.from_pretrained(
            str(model["id"]), revision=str(model["identity"]), token=os.environ.get("HF_TOKEN")
        )
        if pipeline is None:
            raise ValueError(f"no pipeline for model {model['id']}")
        self.pipeline = pipeline.to(torch.device(device))

    def diarize(self, samples: numpy.ndarray, hints: Mapping[str, int | None], exclusive: bool) -> Turns:
        """Return the turns of `samples`: the exclusive partition, or the overlapping view when `exclusive` is false."""
        import torch  # noqa: PLC0415 - loads when the pipeline is built

        given: dict[str, Any] = {key: value for key, value in hints.items() if value is not None}
        output = self.pipeline({"waveform": torch.from_numpy(samples[None, :]), "sample_rate": EXPECTED_RATE}, **given)
        annotation = output.exclusive_speaker_diarization if exclusive else output.speaker_diarization
        return [(str(label), float(s.start), float(s.end)) for s, _, label in annotation.itertracks(yield_label=True)]


@dataclass(frozen=True)
class State:
    """A loaded backend: the model (a stub or the pipeline) and the faults to inject."""

    model: Any  # noqa: ANN401 - a StubModel or a Pipeline
    faults: FakeFaults


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True`; it may hold `faults`, the fields of `FakeFaults`, and `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    faults, model = init_model(init, _INIT_KEYS, "pyannote")
    if model is None:
        return State(StubModel(), faults)
    return State(Pipeline(model, str(init.get("device", "cpu"))), faults)


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"turns": <intervals@1.0 wire form>}}` per item, or its typed fault.

    `params` holds the task's `num_speakers`, `min_speakers` and `max_speakers` and `backend_params` (`exclusive`).

    Raises:
        ValueError: on an unknown backend param.
    """
    unknown = sorted(set(params.get("backend_params", {})) - set(_BACKEND_PARAMS))
    if unknown:
        raise ValueError(f"unknown backend params {unknown}; known: {list(_BACKEND_PARAMS)}")
    exclusive = bool(params.get("backend_params", {}).get("exclusive", True))
    hints = {key: params.get(key) for key in _HINTS}
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            samples, clock_id = mono_samples(item, state.faults, EXPECTED_RATE, "pyannote")
            turns = state.model.diarize(samples, hints, exclusive)
            results.append(
                {"outputs": {"turns": intervals(turns, clock_id, round(samples.size / EXPECTED_RATE * _NS_PER_S))}}
            )
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": exc})
    return results


def intervals(turns: Turns, clock_id: str, duration_ns: int) -> dict[str, Any]:
    """Return the `intervals@1.0` wire form of `turns`, by start; times are integer ns clamped to `duration_ns`."""
    ordered = sorted(turns, key=lambda turn: (turn[1], turn[2], turn[0]))
    start = [min(duration_ns, max(0, round(s * _NS_PER_S))) for _, s, _ in ordered]
    end = [min(duration_ns, max(b, round(e * _NS_PER_S))) for b, (_, _, e) in zip(start, ordered)]
    return {
        "format": "intervals@1.0",
        "start_ns": numpy.array(start, dtype=numpy.int64),
        "end_ns": numpy.array(end, dtype=numpy.int64),
        "label": numpy.array([label for label, _, _ in ordered], dtype=str),
        "clock_id": clock_id,
    }
