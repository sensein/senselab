"""CrisperWhisper 2.0 (verbatim, word-timed) via an isolated subprocess venv.

CrisperWhisper 2.0 (nyralabs) is a Whisper-derived model tuned for **verbatim**
transcription and **word-level timestamps** (~30-40 ms boundary error). It ships
as the ``crisperwhisper`` pip package (2.x) with a CTranslate2 backend
(``crisperwhisper[ct2]``) rather than plain ``transformers`` weights, so it runs
in its own venv (same pattern as the Qwen / Canary / Brouhaha backends) — the
CT2 fork (``ctranslate2-crisperwhisper``) must not leak into the senselab core.

The worker returns per-word timestamps and native per-word confidence (when the
library exposes it) so the asr axis can consume a native uncertainty
signal via ``ScriptLine.score`` (line-level) and each word chunk's ``score``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env, resolve_model
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python
from senselab.utils.venv_worker import serve_in_venv

_CRISPER_VENV = "crisperwhisper"
# The CT2 backend on Linux x86_64, the transformers backend elsewhere. One requirement list with
# environment markers, so the venv lock and its digest are the same on every host; see
# specs/20261002-subprocess-venv-locks/design.md.
_IS_LINUX_X86 = sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
_LINUX_X86_MARKER = "sys_platform == 'linux' and platform_machine == 'x86_64'"
_CRISPER_REQUIREMENTS = [
    "crisperwhisper[transformers]==2.0.1",
    f"crisperwhisper[ct2]==2.0.1; {_LINUX_X86_MARKER}",
    "ctranslate2>=4.0; sys_platform != 'linux' or platform_machine != 'x86_64'",
    "torch>=2.4",
    "torchaudio>=2.4",
]
_CRISPER_PYTHON = "3.12"

# Backend token passed to CrisperWhisperModel(..., backend=...). "auto" would try
# ct2 first (and fail off Linux x86_64), so we pin it explicitly per platform.
_CRISPER_BACKEND = "ct2" if _IS_LINUX_X86 else "transformers"

# The library's HF->CT2 conversion cache: one directory per (model, quantization),
# holding ``model.bin`` and stamped with ``.conversion_complete`` when written.
_CT2_WEIGHTS = "model.bin"
_CT2_MARKER = ".conversion_complete"

# CTranslate2's C++ message when a decode step indexes past Whisper's 448 position
# encodings; see specs/20260910-crisperwhisper-decoder-positions/.
_CT2_POSITION_LIMIT = "No position encodings are defined for positions >="

LONGFORM_STRATEGY = "continuation"
"""The long-form strategy: 30 s windows, each prompted with the last words of the transcript so far."""

CONTEXT_CAPPED = "continuation_context_capped"
"""The ``decode_strategy`` of a recording decoded again after a position overrun, its context prompts capped."""

DECODER_POSITIONS = 448
"""Whisper's decoder position encodings, which a window's prompt and its generated tokens share."""

MAX_NEW_TOKENS = 256
"""The tokens a window may generate; the library's own default, passed explicitly."""


class CrisperWhisperDecoderPositionsExceeded(ValueError):
    """A chunk's prompt plus its token budget overran Whisper's 448 decoder positions; record it as an absence."""


def _ct2_cache_root() -> Path:
    """Return the conversion-cache root ``crisperwhisper.converter`` reads."""
    env = os.environ.get("CRISPERWHISPER_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "crisperwhisper"


def _ct2_cache_key(model_id: str, quantization: str) -> str:
    """Return the conversion-cache directory name for one model and quantization.

    Args:
        model_id: The path or repo id handed to ``CrisperWhisperModel``.
        quantization: The CT2 compute type (``float16``, ``float32``, ...).

    Returns:
        The directory name ``crisperwhisper.converter._cache_key`` would build.
    """
    slug = model_id.replace("/", "--").replace("\\", "--")
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return f"{slug}_{quantization}_{digest}"


def _ct2_entry_is_torn(entry: Path) -> bool:
    """Return whether a conversion-cache entry is stamped complete but has no weights.

    Args:
        entry: A conversion-cache directory.

    Returns:
        True when ``.conversion_complete`` exists and ``model.bin`` does not.
    """
    return (entry / _CT2_MARKER).exists() and not (entry / _CT2_WEIGHTS).exists()


def _discard_torn_ct2_entry(entry: Path) -> bool:
    """Detach and delete a conversion-cache entry that carries no weights.

    Args:
        entry: A conversion-cache directory.

    Returns:
        True when a torn entry was detached and deleted, False otherwise.
    """
    if not _ct2_entry_is_torn(entry):
        return False
    detached = entry.with_name(f"{entry.name}.torn-{uuid.uuid4().hex}")
    try:
        os.rename(entry, detached)
    except OSError:
        return False
    shutil.rmtree(detached, ignore_errors=True)
    return True


# Worker — runs inside the isolated venv, served by senselab.utils.venv_worker.
_CRISPER_WORKER_SCRIPT = r"""
import os
import shutil
import uuid
from pathlib import Path


def load(init):
    from crisperwhisper import CrisperWhisperModel

    model_id = init["model_id"]
    backend = init["backend"]
    compute_type = init["compute_type"]

    # The CT2 backend converts the HF snapshot into a shared cache directory whose
    # writer is neither atomic nor locked. Convert into a private staging directory
    # and publish it with one rename, so a concurrent converter can neither be read
    # half-written nor delete what this process just wrote.
    if backend == "ct2":
        entry = Path(init["ct2_entry"])
        if not (entry / "model.bin").exists():
            from crisperwhisper.converter import ensure_ct2_model

            staging = Path(init["ct2_staging_root"]) / (".staging-" + uuid.uuid4().hex)
            converted = Path(ensure_ct2_model(model_id, quantization=compute_type, cache_dir=str(staging)))
            entry.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(str(converted), str(entry))
            except OSError:
                if not (entry / "model.bin").exists():
                    shutil.rmtree(str(entry), ignore_errors=True)
                    os.rename(str(converted), str(entry))
            shutil.rmtree(str(staging), ignore_errors=True)
        model_id = str(entry)

    # CrisperWhisperModel takes no revision anywhere in its call chain; `model_id` is the local
    # snapshot directory the parent resolved and staged for the run-agreed commit.
    return CrisperWhisperModel(model_id, backend=backend, device=init["device"], compute_type=compute_type)


def _first_attr(obj, names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def handle(model, args):
    language = args.get("language") or "en"
    strategy = args["longform_strategy"]
    max_new_tokens = int(args["max_new_tokens"])
    prompt_budget = args.get("prompt_token_budget")
    position_limit = args["position_limit"]

    def _decode(path):
        return model.transcribe(
            path,
            language=language,
            word_timestamps=True,
            longform_strategy=strategy,
            max_new_tokens=max_new_tokens,
        )

    def _decode_capped(path):
        from crisperwhisper.prompt import PromptBuilder

        original = PromptBuilder._build

        def _build(self, mode, hotwords=None, context=None):
            ids = original(self, mode, hotwords=hotwords, context=context)
            words = (context or "").split()
            while words and len(ids) > prompt_budget:
                words = words[1:]
                ids = original(self, mode, hotwords=hotwords, context=" ".join(words) or None)
            return ids

        PromptBuilder._build = _build
        try:
            return _decode(path)
        finally:
            PromptBuilder._build = original

    results = []
    for path in args["audio_paths"]:
        used = strategy
        try:
            r = _decode(path)
        except Exception as overrun:
            if prompt_budget is None or position_limit not in str(overrun):
                raise
            used = args["capped_strategy"]
            r = _decode_capped(path)
        words = []
        for w in (getattr(r, "words", None) or []):
            conf = _first_attr(w, ("probability", "confidence", "score", "prob"))
            words.append({
                "text": _first_attr(w, ("word", "text")) or "",
                "start": float(w.start),
                "end": float(w.end),
                "score": (float(conf) if conf is not None else None),
            })
        line_conf = _first_attr(r, ("confidence", "avg_logprob", "score"))
        if line_conf is None:
            cs = [w["score"] for w in words if w["score"] is not None]
            line_conf = (sum(cs) / len(cs)) if cs else None
        results.append({
            "text": getattr(r, "text", "") or "",
            "language": getattr(r, "language", language),
            "words": words,
            "score": (float(line_conf) if line_conf is not None else None),
            "decode_strategy": used,
        })
    return {"results": results}
"""

_LOAD_TIMEOUT_S = 1800
_REQUEST_TIMEOUT_S = 1800


class CrisperWhisperASR:
    """CrisperWhisper 2.0 transcription via its isolated CT2 subprocess venv.

    Routed automatically by ``speech_to_text.api`` when the model id matches the
    ``nyralabs/CrisperWhisper2.0`` prefix. Returns one ``ScriptLine`` per audio
    with verbatim ``text``, per-word ``chunks`` (with ``score`` = native word
    confidence when available), and a line-level ``score``.
    """

    @classmethod
    def transcribe_with_crisperwhisper(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        language: Optional[str] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with CrisperWhisper 2.0 via the dedicated subprocess venv.

        Args:
            audios: Audio clips (mono, 16 kHz expected).
            model: HF model id (default ``nyralabs/CrisperWhisper2.0_turbo``).
            device: CPU or CUDA (CT2 auto-uses the GPU when available).
            language: Force a language (default ``en``); CrisperWhisper is en/de.

        Returns:
            One ``ScriptLine`` per input with verbatim ``text``, word-level
            ``chunks`` carrying timestamps + ``score`` (native word confidence
            when exposed), a line-level ``score``, and the ``decode_strategy``
            used: :data:`LONGFORM_STRATEGY`, or :data:`CONTEXT_CAPPED` for an input
            whose first decode overran the decoder's positions and was decoded
            again with each window's context prompt cut, leading words first,
            to :data:`DECODER_POSITIONS` less :data:`MAX_NEW_TOKENS`.

        Raises:
            CrisperWhisperDecoderPositionsExceeded: The capped decode also
                overran Whisper's 448 position encodings.
        """
        if model is None:
            model = HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo")
        model_id = str(model.path_or_uri)
        device_type, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        # float16 only on CUDA; CPU (e.g. macOS transformers backend) needs float32.
        device_str = "cuda" if device_type == DeviceType.CUDA else "cpu"
        compute_type = "float16" if device_str == "cuda" else "float32"

        venv_dir = ensure_venv(_CRISPER_VENV, _CRISPER_REQUIREMENTS, python_version=_CRISPER_PYTHON)
        python = venv_python(venv_dir)

        # CrisperWhisperModel has no revision parameter anywhere in its call chain (see
        # the worker-script comment), so resolve the ref to the run-agreed commit SHA
        # (download-once via the cross-process heartbeat lock) and point the worker at
        # that commit's already-staged local snapshot directory instead of the mutable
        # repo id -- both backends treat an existing local directory as already pinned.
        revision, snapshot_path = resolve_model(model_id, model.revision or "main")

        with tempfile.TemporaryDirectory(prefix="senselab-crisperwhisper-") as tmpdir:
            tmp = Path(tmpdir)
            audio_paths: List[str] = []
            for i, audio in enumerate(audios):
                path = str(tmp / f"audio_{i}.wav")
                audio.save_to_file(path)
                audio_paths.append(path)

            cache_root = _ct2_cache_root()
            ct2_entry = cache_root / _ct2_cache_key(str(snapshot_path), compute_type)
            if _CRISPER_BACKEND == "ct2":
                _discard_torn_ct2_entry(ct2_entry)
            # Stage the model once (cross-process heartbeat lock) and run the worker offline so
            # its weight fetch makes no Hub version check.
            env = hf_subprocess_env(model_id, revision, base_env=_clean_subprocess_env())
            try:
                output = serve_in_venv(
                    (Path(venv_dir).name, str(snapshot_path), revision, device_str, compute_type),
                    python=python,
                    script=_CRISPER_WORKER_SCRIPT,
                    init={
                        "model_id": str(snapshot_path),
                        "backend": _CRISPER_BACKEND,
                        "device": device_str,
                        "compute_type": compute_type,
                        "ct2_entry": str(ct2_entry),
                        "ct2_staging_root": str(cache_root),
                    },
                    request={
                        "audio_paths": audio_paths,
                        "language": language or "en",
                        "longform_strategy": LONGFORM_STRATEGY,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "prompt_token_budget": DECODER_POSITIONS - MAX_NEW_TOKENS - 1,
                        "capped_strategy": CONTEXT_CAPPED,
                        "position_limit": _CT2_POSITION_LIMIT,
                    },
                    env=env,
                    label="CrisperWhisper 2.0",
                    load_timeout_s=_LOAD_TIMEOUT_S,
                    request_timeout_s=_REQUEST_TIMEOUT_S,
                )
            except RuntimeError as err:
                if _CT2_POSITION_LIMIT in str(err):
                    raise CrisperWhisperDecoderPositionsExceeded(str(err)) from err
                raise

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                words = entry.get("words") or []
                chunks: Optional[List[ScriptLine]] = None
                line_start: Optional[float] = None
                line_end: Optional[float] = None
                if words:
                    chunks = [
                        ScriptLine(
                            text=w["text"],
                            start=float(w["start"]),
                            end=float(w["end"]),
                            score=(float(w["score"]) if w.get("score") is not None else None),
                        )
                        for w in words
                    ]
                    chunks.sort(key=lambda c: c.start if c.start is not None else 0.0)
                    starts = [c.start for c in chunks if c.start is not None]
                    ends = [c.end for c in chunks if c.end is not None]
                    line_start = min(starts) if starts else None
                    line_end = max(ends) if ends else None
                results.append(
                    ScriptLine(
                        text=entry.get("text", ""),
                        start=line_start,
                        end=line_end,
                        chunks=chunks,
                        score=(float(entry["score"]) if entry.get("score") is not None else None),
                        decode_strategy=entry.get("decode_strategy"),
                    )
                )
            return results
