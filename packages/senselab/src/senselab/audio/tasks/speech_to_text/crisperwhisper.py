"""Worker of backend `crisperwhisper` of `speech_to_text`: the CrisperWhisper library, verbatim and word-timed.

`load` takes the pinned model in `init["model"]` (or `"stub": True` for the stub model) and builds nothing; `handle`
builds the engine `backend_params["engine"]` names (`ct2` or `transformers`) on first use and answers one
`transcript@2.0` wire form, or one typed fault, per item. Imports are stdlib, numpy and `senselab_worker`; the library,
`torch`, `soundfile` and `huggingface_hub` load inside the engine.
"""

import contextlib
import functools
import hashlib
import importlib.metadata
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import numpy
from senselab_worker.fake import injected
from senselab_worker.units import NS_PER_S

from senselab.audio.tasks.speech_to_text.whisper_frontend import MIN_INPUT_NS
from senselab.item_faults import clamp_ns, init_model, mono_samples

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
_INIT_KEYS = frozenset({"stub", "model", "device", "array_dir"})
DIAGNOSTICS_KEY = "_diagnostics"
"""The request param that adds the `diagnostics` output: windows, chunks, the worker's seconds per phase, and how
many scored windows reused the decode's encoder pass."""


@dataclass(frozen=True)
class Decoded:
    """One decode: `words` as `(text, start_s, end_s)`, a confidence per word (or none), and the mean confidence."""

    words: list[tuple[str, float, float]]
    confidences: list[float | None]
    mean: float | None
    windows: int = 0
    chunks: int | None = None
    decode_s: float = 0.0
    confidence_s: float = 0.0
    encode_reused: int = 0


class StubModel:
    """The fake stack's model: deterministic words, evenly spaced over the audio, derived from its samples."""

    def transcribe(self, samples: numpy.ndarray, language: str, max_new_tokens: int, confidence: bool) -> Decoded:
        """Return three words spaced over `samples`; `language`, `max_new_tokens` and `confidence` are unused."""
        digest = hashlib.sha256(samples.astype(numpy.float32).tobytes()).hexdigest()
        step = samples.size / EXPECTED_RATE / 3
        words = [(f"w{digest[4 * i : 4 * i + 4]}", i * step, (i + 1) * step) for i in range(3)]
        return Decoded(words, [None] * 3, None)


@dataclass
class State:
    """A loaded backend: the pinned model, the device and the engines built so far."""

    model: Mapping[str, Any] | None
    device: str
    engines: dict[str, Any] = field(default_factory=dict)


def load(init: dict[str, Any]) -> State:
    """Return the worker's state from `init`.

    `init` holds `model` (`{"source": "hf", "id", "identity"}`, `identity` a 40-hex commit) and `device`, or
    `"stub": True`; it may hold `array_dir`.

    Raises:
        ValueError: on an unknown key, a model that is not a pinned hub commit, or both `stub` and `model`.
    """
    device = str(init.get("device", "cpu"))
    model = init_model(init, _INIT_KEYS, "crisperwhisper")
    if model is None:
        return State(None, device, {engine: StubModel() for engine in ENGINES})
    return State(model, device)


def handle(state: State, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"transcript": <transcript@2.0 wire form>}}` per item, or its typed fault.

    `params` holds the task's `language` and `timestamps` and `backend_params` (`engine`, `max_new_tokens`,
    `confidence`).

    Raises:
        ValueError: on an unknown engine or backend param, or an unknown `timestamps` level.
    """
    tuning = _backend_params(params.get("backend_params", {}))
    timestamps = str(params.get("timestamps", "word"))
    if timestamps not in ("word", "segment", "none"):
        raise ValueError(f"timestamps is word, segment or none, got {timestamps!r}")
    language = str(params.get("language") or DEFAULT_LANGUAGE)
    built = time.perf_counter()
    engine = _engine(state, tuning["engine"])
    engine_build_s = time.perf_counter() - built
    faults = injected(params)
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            samples, clock_id = mono_samples(item, faults, EXPECTED_RATE, "crisperwhisper", MIN_INPUT_NS)
            decoded = engine.transcribe(samples, language, tuning["max_new_tokens"], tuning["confidence"])
            duration_ns = round(samples.size / EXPECTED_RATE * NS_PER_S)
            outputs: dict[str, Any] = {"transcript": transcript(decoded, timestamps, clock_id, duration_ns)}
            if params.get(DIAGNOSTICS_KEY):
                outputs["diagnostics"] = {
                    "windows": decoded.windows,
                    "chunks": decoded.chunks,
                    "decode_s": decoded.decode_s,
                    "confidence_s": decoded.confidence_s,
                    "engine_build_s": engine_build_s,
                    "encode_reused": decoded.encode_reused,
                }
            results.append({"outputs": outputs})
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": exc})
    return results


def _backend_params(given: Mapping[str, Any]) -> dict[str, Any]:
    known = ["confidence", "engine", "max_new_tokens"]
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ValueError(f"unknown backend params {unknown}; known: {known}")
    tuning = {"engine": DEFAULT_ENGINE, "max_new_tokens": MAX_NEW_TOKENS, "confidence": True} | dict(given)
    if tuning["engine"] not in ENGINES:
        raise ValueError(f"engine is one of {list(ENGINES)}, got {tuning['engine']!r}")
    return tuning


def transcript(decoded: Decoded, timestamps: str, clock_id: str, duration_ns: int) -> dict[str, Any]:
    """Return the `transcript@2.0` wire form of `decoded` at the `timestamps` level.

    `word` gives one segment dividing into words, `segment` the same segment without words, `none` one untimed
    segment; no words give no segment. A word missing a time starts where the previous ended; spans are integer ns
    clamped to `duration_ns`.
    """
    doc = functools.partial(_doc, extent=_span(0, duration_ns, clock_id))
    text = " ".join(w for w, _, _ in decoded.words).strip()
    if not text:
        return doc([])
    mean = None if decoded.mean is None else min(1.0, decoded.mean)
    if timestamps == "none":
        return doc([_node(text, None, mean, [])])
    floor = 0
    children = []
    for (word, start, end), confidence in zip(decoded.words, decoded.confidences):
        begin, stop = clamp_ns(start, end, floor, duration_ns)
        children.append(_node(word, _span(begin, stop, clock_id), confidence, []))
        floor = stop
    whole = _span(children[0]["span"]["start_ns"], children[-1]["span"]["end_ns"], clock_id)
    return doc([_node(text, whole, mean, children if timestamps == "word" else [])])


def _doc(segments: list[dict[str, Any]], *, extent: dict[str, Any]) -> dict[str, Any]:
    return {"format": "transcript@2.0", "segments": segments, "extent": extent}


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

    def transcribe(self, samples: numpy.ndarray, language: str, max_new_tokens: int, confidence: bool) -> Decoded:
        """Return the decode of `samples`, with its mean and word confidences unless `confidence` is false.

        Each window's final ids are scored by raw-logit log-probability, excluding the engine's `all_special_ids`:
        on `transformers` from the decode's own logits, on `ct2` from one decoder pass over the decode's own
        encoder output. With `confidence` false no logits are recorded, nothing is scored and every confidence is none.
        """
        import soundfile  # noqa: PLC0415 - loads in the engine only

        from senselab.audio.tasks.speech_to_text.token_confidence import (  # noqa: PLC0415 - as above
            logprob_block,
            merge_confidence_blocks,
            word_confidences,
        )
        from senselab.audio.tasks.speech_to_text.window_confidence import (  # noqa: PLC0415 - as above
            Capture,
            Window,
            capture_decodes,
            capture_windows,
            ct2_window_logits,
            spliced_block,
            surviving_windows,
            window_confidence,
        )

        engine = self.model._engine
        special = {int(i) for i in engine.all_special_ids} or None
        logits_of_decode = confidence and self.engine == "transformers"
        capture = Capture()

        def score(window: Window) -> list[float]:
            began = time.perf_counter()
            [block] = window_confidence(engine, [window], ct2_window_logits, special)
            capture.scoring_s += time.perf_counter() - began
            return list(block["token_logprob"])

        score_now = score if confidence and not logits_of_decode else None
        started = time.perf_counter()
        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory(prefix="senselab-crisperwhisper-") as scratch:
            if logits_of_decode:
                capture = stack.enter_context(capture_decodes(engine.model))
            windows = stack.enter_context(capture_windows(engine, score_now))
            wav = Path(scratch) / "audio.wav"
            soundfile.write(str(wav), samples, EXPECTED_RATE, subtype="FLOAT")
            result = self.model.transcribe(
                str(wav),
                language=language,
                word_timestamps=True,
                longform_strategy=LONGFORM_STRATEGY,
                max_new_tokens=max_new_tokens,
            )
        decode_s = time.perf_counter() - started - capture.scoring_s
        scoring = time.perf_counter()

        def decode(token: int) -> str:
            return str(engine.tokenizer.decode([token]))

        captured = len(windows)
        chunks = getattr(result, "chunks", None) or []
        kept = [len(chunk.text.split()) for chunk in chunks]
        if len(windows) > 1 or kept:
            windows = surviving_windows(windows, kept, decode, special) if len(kept) == len(windows) else []
        scored_windows = [w for w in windows if w.ids]
        blocks: list[dict[str, Any]] = []
        encode_reused = 0
        if logits_of_decode:
            spliced = [spliced_block(w, capture.decodes, special) for w in scored_windows]
            blocks = [b for b in spliced if b is not None] if all(b is not None for b in spliced) else []
            encode_reused = len(blocks)
        elif confidence:
            blocks = [logprob_block(w.ids, w.token_logprob or [], special) for w in scored_windows]
            encode_reused = sum(w.reused for w in scored_windows)
        words = [(str(w.word or w.text), float(w.start), float(w.end)) for w in getattr(result, "words", None) or []]
        confidences: list[float | None] = [None] * len(words)
        mean = None
        if blocks:
            merged = merge_confidence_blocks(blocks)
            scored = [
                c
                for block in blocks
                for c in word_confidences(block["tokens"], block["token_logprob"], decode, special)
            ]
            if len(scored) == len(words) and (not kept or sum(kept) == len(words)):
                confidences = list(scored)
                mean = None if merged["avg_logprob"] is None else math.exp(merged["avg_logprob"])
        return Decoded(
            words,
            confidences,
            mean,
            captured,
            len(chunks) if chunks else None,
            decode_s,
            time.perf_counter() - scoring + capture.scoring_s if confidence else 0.0,
            encode_reused if blocks else 0,
        )


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
