"""Worker of backend `crisperwhisper` of `speech_to_text`: the CrisperWhisper library, verbatim and word-timed.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model) and builds nothing; `handle`
builds the engine `backend_params["engine"]` names (`ct2` or `transformers`) on first use and answers one
`transcript@1.0` wire form, or one typed fault, per item. Imports are stdlib, numpy and `senselab_worker`; the library,
`torch`, `soundfile` and `huggingface_hub` load inside the engine.
"""

import hashlib
import importlib.metadata
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy
from senselab_worker.fake import FakeFaults

from senselab.item_faults import NS_PER_S, clamp_ns, init_model, mono_samples

WORKER_MODULE = "senselab.audio.tasks.speech_to_text.crisperwhisper"
MODEL_ID = "nyralabs/CrisperWhisper2.0_turbo"
Engine = Literal["ct2", "transformers"]
ENGINES: tuple[Engine, ...] = get_args(Engine)
DEFAULT_ENGINE: Engine = "transformers"
MAX_NEW_TOKENS = 256
EXPECTED_RATE = 16_000
LONGFORM_STRATEGY = "continuation"
DEFAULT_LANGUAGE = "en"
COMPUTE_TYPES = {"cuda": "float16", "mps": "float16", "cpu": "float32"}
_INIT_KEYS = frozenset({"stub", "model", "device", "faults", "array_dir"})


@dataclass(frozen=True)
class Decoded:
    """One decode: `words` as `(text, start_s, end_s)`, a confidence per word (or none), and the mean confidence."""

    words: list[tuple[str, float, float]]
    confidences: list[float | None]
    mean: float | None


class StubModel:
    """The fake stack's model: deterministic words, evenly spaced over the audio, derived from its samples."""

    def transcribe(self, samples: numpy.ndarray, language: str, max_new_tokens: int) -> Decoded:
        """Return three words spaced over `samples`; `language` and `max_new_tokens` are unused."""
        digest = hashlib.sha256(samples.astype(numpy.float32).tobytes()).hexdigest()
        step = samples.size / EXPECTED_RATE / 3
        words = [(f"w{digest[4 * i : 4 * i + 4]}", i * step, (i + 1) * step) for i in range(3)]
        return Decoded(words, [None] * 3, None)


@dataclass
class State:
    """A loaded backend: the pinned model, the device, the faults to inject and the engines built so far."""

    model: Mapping[str, Any] | None
    device: str
    faults: FakeFaults
    engines: dict[str, Any] = field(default_factory=dict)


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True`; it may hold `faults`, the fields of `FakeFaults`, and `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    device = str(init.get("device", "cpu"))
    faults, model = init_model(init, _INIT_KEYS, "crisperwhisper")
    if model is None:
        return State(None, device, faults, {engine: StubModel() for engine in ENGINES})
    return State(model, device, faults)


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"transcript": <transcript@1.0 wire form>}}` per item, or its typed fault.

    `params` holds the task's `language` and `timestamps` and `backend_params` (`engine`, `max_new_tokens`).

    Raises:
        ValueError: on an unknown engine or backend param, or an unknown `timestamps` level.
    """
    tuning = _backend_params(params.get("backend_params", {}))
    timestamps = str(params.get("timestamps", "word"))
    if timestamps not in ("word", "segment", "none"):
        raise ValueError(f"timestamps is word, segment or none, got {timestamps!r}")
    language = str(params.get("language") or DEFAULT_LANGUAGE)
    engine = _engine(state, tuning["engine"])
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            samples, clock_id = mono_samples(item, state.faults, EXPECTED_RATE, "crisperwhisper")
            decoded = engine.transcribe(samples, language, tuning["max_new_tokens"])
            duration_ns = round(samples.size / EXPECTED_RATE * NS_PER_S)
            results.append({"outputs": {"transcript": transcript(decoded, timestamps, clock_id, duration_ns)}})
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": exc})
    return results


def _backend_params(given: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(given) - {"engine", "max_new_tokens"})
    if unknown:
        raise ValueError(f"unknown backend params {unknown}; known: ['engine', 'max_new_tokens']")
    tuning = {"engine": DEFAULT_ENGINE, "max_new_tokens": MAX_NEW_TOKENS} | dict(given)
    if tuning["engine"] not in ENGINES:
        raise ValueError(f"engine is one of {list(ENGINES)}, got {tuning['engine']!r}")
    return tuning


def transcript(decoded: Decoded, timestamps: str, clock_id: str, duration_ns: int) -> dict[str, Any]:
    """Return the `transcript@1.0` wire form of `decoded` at the `timestamps` level.

    `word` gives one segment dividing into words, `segment` the same segment without words, `none` one untimed
    segment; no words give no segment. A word missing a time starts where the previous ended; spans are integer ns
    clamped to `duration_ns`.
    """
    text = " ".join(w for w, _, _ in decoded.words).strip()
    if not text:
        return _doc([])
    mean = None if decoded.mean is None else min(1.0, decoded.mean)
    if timestamps == "none":
        return _doc([_node(text, None, mean, [])])
    floor = 0
    children = []
    for (word, start, end), confidence in zip(decoded.words, decoded.confidences):
        begin, stop = clamp_ns(start, end, floor, duration_ns)
        children.append(_node(word, _span(begin, stop, clock_id), confidence, []))
        floor = stop
    whole = _span(children[0]["span"]["start_ns"], children[-1]["span"]["end_ns"], clock_id)
    return _doc([_node(text, whole, mean, children if timestamps == "word" else [])])


def _doc(segments: list[dict[str, Any]]) -> dict[str, Any]:
    return {"format": "transcript@1.0", "segments": segments}


def _span(start_ns: int, end_ns: int, clock_id: str) -> dict[str, Any]:
    return {"start_ns": start_ns, "end_ns": end_ns, "clock_id": clock_id}


def _node(
    text: str, span: dict[str, Any] | None, confidence: float | None, children: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"text": text, "span": span, "confidence": confidence, "children": children}


def _engine(state: State, name: str) -> Any:  # noqa: ANN401 - a StubModel or a Library
    if name not in state.engines:
        assert state.model is not None  # noqa: S101 - a stubbed state holds every engine from `load`
        state.engines[name] = Library(state.model, state.device, name)
    return state.engines[name]


class Library:
    """The CrisperWhisper library on one engine, loaded from the pinned snapshot of the model."""

    def __init__(self, model: Mapping[str, Any], device: str, engine: str) -> None:
        """Build the library model; `ct2` converts the snapshot once per model commit and compute type.

        Raises:
            ValueError: for `ct2` on `mps`.
        """
        from crisperwhisper import CrisperWhisperModel  # noqa: PLC0415 - the library loads in the engine only
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - as above

        if engine == "ct2" and device == "mps":
            raise ValueError("the ct2 engine runs on cuda or cpu, not mps")
        compute = COMPUTE_TYPES[device]
        path = snapshot_download(str(model["id"]), revision=str(model["identity"]))
        if engine == "ct2":
            path = _converted(path, str(model["identity"]), compute)
        if device == "mps":
            _disable_deferred_stop_check()
        self.engine = engine
        self.model = CrisperWhisperModel(path, backend=engine, device=device, compute_type=compute)

    def transcribe(self, samples: numpy.ndarray, language: str, max_new_tokens: int) -> Decoded:
        """Return the decode of `samples`, with its mean and word confidences.

        Both engines teacher-force each window's final ids once and score them by raw-logit log-probability,
        excluding the engine's `all_special_ids`.
        """
        import soundfile  # noqa: PLC0415 - loads in the engine only

        from senselab.audio.tasks.speech_to_text.token_confidence import (  # noqa: PLC0415 - as above
            merge_confidence_blocks,
            word_confidences,
        )
        from senselab.audio.tasks.speech_to_text.window_confidence import (  # noqa: PLC0415 - as above
            capture_windows,
            ct2_teacher_logits,
            transformers_teacher_logits,
            window_confidence,
        )

        engine = self.model._engine
        special = {int(i) for i in engine.all_special_ids} or None
        with tempfile.TemporaryDirectory(prefix="senselab-crisperwhisper-") as scratch:
            wav = Path(scratch) / "audio.wav"
            soundfile.write(str(wav), samples, EXPECTED_RATE, subtype="FLOAT")
            with capture_windows(engine) as windows:
                result = self.model.transcribe(
                    str(wav),
                    language=language,
                    word_timestamps=True,
                    longform_strategy=LONGFORM_STRATEGY,
                    max_new_tokens=max_new_tokens,
                )
        teacher = ct2_teacher_logits if self.engine == "ct2" else transformers_teacher_logits
        blocks = window_confidence(engine, windows, teacher, special)
        words = [(str(w.word or w.text), float(w.start), float(w.end)) for w in getattr(result, "words", None) or []]
        confidences: list[float | None] = [None] * len(words)
        mean = None
        if blocks:
            merged = merge_confidence_blocks(blocks)
            mean = None if merged["avg_logprob"] is None else math.exp(merged["avg_logprob"])
            if merged["tokens"]:
                scored = word_confidences(
                    merged["tokens"],
                    merged["token_logprob"],
                    lambda token: str(engine.tokenizer.decode([token])),
                    special,
                )
                if len(scored) == len(words):
                    confidences = list(scored)
        return Decoded(words, confidences, mean)


def _converted(snapshot: str, identity: str, compute: str) -> str:
    """Return `snapshot` if it is already CT2, else its conversion, made privately and published by one rename.

    The cache entry is named by the model commit, compute type and converter version.
    """
    from crisperwhisper.converter import ensure_ct2_model  # noqa: PLC0415 - the library loads in the engine only

    if (Path(snapshot) / "model.bin").exists():
        return snapshot
    root = Path(os.environ.get("SENSELAB_CACHE") or Path.home() / ".cache" / "senselab") / "crisperwhisper-ct2"
    entry = root / f"{identity}-{compute}-{importlib.metadata.version('crisperwhisper')}"
    if (entry / "model.bin").exists():
        return str(entry)
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".staging-{uuid.uuid4().hex}"
    try:
        converted = Path(ensure_ct2_model(snapshot, quantization=compute, cache_dir=str(staging)))
        try:
            os.rename(converted, entry)
        except OSError:
            if not (entry / "model.bin").exists():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(entry)


def _disable_deferred_stop_check() -> None:
    from transformers.generation import utils  # noqa: PLC0415 - loads in the engine only

    deferred = getattr(utils, "DeferredStopCheck", None)
    if deferred is not None:
        deferred.is_supported = staticmethod(lambda *args, **kwargs: False)
